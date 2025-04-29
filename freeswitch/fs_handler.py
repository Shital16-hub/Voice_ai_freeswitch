"""
Main FreeSWITCH handler for Voice AI Agent.
"""
import os
import logging
import asyncio
import time
from typing import Dict, Any, Optional, List, AsyncIterator, Callable, Awaitable

from freeswitch.esl_client import ESLClient
from freeswitch.call_manager import CallManager
from freeswitch.audio_processor import AudioProcessor
from freeswitch.utils.audio_utils import create_wav_file
from config.freeswitch_config import (
    GREETING_MESSAGE, SAMPLE_RATE_FS, SAMPLE_RATE_AI,
    AUDIO_BUFFER_SIZE, MAX_BUFFER_SIZE
)

logger = logging.getLogger(__name__)

class FreeSwitchHandler:
    """
    Main handler for FreeSWITCH integration with Voice AI Agent.
    """
    
    def __init__(
        self,
        pipeline,
        fs_client: Optional[ESLClient] = None
    ):
        """
        Initialize FreeSWITCH handler.
        
        Args:
            pipeline: Voice AI Agent pipeline
            fs_client: Optional ESL client (will create if not provided)
        """
        self.pipeline = pipeline
        self.fs_client = fs_client or ESLClient()
        self.call_manager = CallManager()
        self.audio_processor = AudioProcessor()
        
        # Track active calls and audio buffers
        self.active_calls = {}
        self.audio_buffers = {}
        self.processing_tasks = {}
        
        # Track state to avoid duplicate processing
        self.is_initialized = False
    
    async def init(self) -> None:
        """Initialize FreeSWITCH handler."""
        if self.is_initialized:
            return
            
        logger.info("Initializing FreeSWITCH handler")
        
        # Connect to FreeSWITCH
        connected = await self.fs_client.connect()
        if not connected:
            raise RuntimeError("Failed to connect to FreeSWITCH")
        
        # Register event handlers
        self.fs_client.register_event_handler("CHANNEL_CREATE", self._handle_channel_create)
        self.fs_client.register_event_handler("CHANNEL_ANSWER", self._handle_channel_answer)
        self.fs_client.register_event_handler("CHANNEL_HANGUP", self._handle_channel_hangup)
        self.fs_client.register_event_handler("DTMF", self._handle_dtmf)
        self.fs_client.register_event_handler("DETECTED_SPEECH", self._handle_speech_detected)
        
        # Initialize call manager
        await self.call_manager.init()
        
        self.is_initialized = True
        logger.info("FreeSWITCH handler initialized")
    
    async def shutdown(self) -> None:
        """Shutdown FreeSWITCH handler."""
        logger.info("Shutting down FreeSWITCH handler")
        
        # Stop all processing tasks
        for uuid, task in list(self.processing_tasks.items()):
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # Clear all buffers
        self.audio_buffers.clear()
        
        # Hang up all calls
        for uuid in list(self.active_calls.keys()):
            try:
                await self.fs_client.hangup_call(uuid)
            except:
                pass
        
        # Shutdown call manager
        await self.call_manager.stop()
        
        # Disconnect from FreeSWITCH
        await self.fs_client.disconnect()
        
        self.is_initialized = False
        logger.info("FreeSWITCH handler shutdown complete")
    
    async def _handle_channel_create(self, event) -> None:
        """
        Handle channel create event.
        
        Args:
            event: FreeSWITCH event
        """
        # Get call UUID
        call_uuid = event.get_header("Unique-ID")
        if not call_uuid:
            return
            
        # Get call details
        caller_id_number = event.get_header("Caller-Caller-ID-Number")
        destination_number = event.get_header("Caller-Destination-Number")
        
        logger.info(f"Call created - UUID: {call_uuid}, From: {caller_id_number}, To: {destination_number}")
        
        # Add to call manager
        self.call_manager.add_call(
            call_uuid=call_uuid,
            from_number=caller_id_number,
            to_number=destination_number
        )
        
        # Initialize audio buffer
        self.audio_buffers[call_uuid] = bytearray()
        
        # Add to active calls
        self.active_calls[call_uuid] = {
            "uuid": call_uuid,
            "from": caller_id_number,
            "to": destination_number,
            "state": "created",
            "created_at": time.time()
        }
    
    async def _handle_channel_answer(self, event) -> None:
        """
        Handle channel answer event.
        
        Args:
            event: FreeSWITCH event
        """
        # Get call UUID
        call_uuid = event.get_header("Unique-ID")
        if not call_uuid or call_uuid not in self.active_calls:
            return
            
        logger.info(f"Call answered - UUID: {call_uuid}")
        
        # Update call state
        self.active_calls[call_uuid]["state"] = "answered"
        self.call_manager.update_call_status(call_uuid, "active")
        
        # Start a task to handle this call
        self.processing_tasks[call_uuid] = asyncio.create_task(
            self._handle_call(call_uuid)
        )
    
    async def _handle_channel_hangup(self, event) -> None:
        """
        Handle channel hangup event.
        
        Args:
            event: FreeSWITCH event
        """
        # Get call UUID
        call_uuid = event.get_header("Unique-ID")
        if not call_uuid or call_uuid not in self.active_calls:
            return
            
        hangup_cause = event.get_header("Hangup-Cause")
        logger.info(f"Call hangup - UUID: {call_uuid}, Cause: {hangup_cause}")
        
        # Update call state
        self.active_calls[call_uuid]["state"] = "hangup"
        self.call_manager.update_call_status(call_uuid, "completed")
        
        # Cancel processing task if running
        if call_uuid in self.processing_tasks:
            task = self.processing_tasks[call_uuid]
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            del self.processing_tasks[call_uuid]
        
        # Clean up audio buffer
        if call_uuid in self.audio_buffers:
            del self.audio_buffers[call_uuid]
        
        # Remove from active calls
        del self.active_calls[call_uuid]
    
    async def _handle_dtmf(self, event) -> None:
        """
        Handle DTMF event.
        
        Args:
            event: FreeSWITCH event
        """
        # Get call UUID
        call_uuid = event.get_header("Unique-ID")
        if not call_uuid or call_uuid not in self.active_calls:
            return
            
        dtmf_digit = event.get_header("DTMF-Digit")
        logger.info(f"DTMF received - UUID: {call_uuid}, Digit: {dtmf_digit}")
        
        # Add to call manager
        self.call_manager.add_dtmf(call_uuid, dtmf_digit)
    
    async def _handle_speech_detected(self, event) -> None:
        """
        Handle speech detected event.
        
        Args:
            event: FreeSWITCH event
        """
        # Get call UUID
        call_uuid = event.get_header("Unique-ID")
        if not call_uuid or call_uuid not in self.active_calls:
            return
            
        logger.info(f"Speech detected - UUID: {call_uuid}")
        
        # We'll handle speech through audio processing
    
    async def _handle_call(self, call_uuid: str) -> None:
        """
        Handle an active call.
        
        Args:
            call_uuid: Call UUID
        """
        logger.info(f"Starting call handling for {call_uuid}")
        
        try:
            # Play greeting
            await self.fs_client.play_tts(call_uuid, GREETING_MESSAGE)
            
            # Start recording for audio capture
            await self.fs_client.start_recording(call_uuid)
            
            # Start audio processing loop
            await self._process_audio_loop(call_uuid)
            
        except asyncio.CancelledError:
            logger.info(f"Call handling cancelled for {call_uuid}")
        except Exception as e:
            logger.error(f"Error handling call {call_uuid}: {e}")
        finally:
            # Stop recording
            try:
                await self.fs_client.stop_recording(call_uuid)
            except:
                pass
                
            # Clean up
            if call_uuid in self.active_calls:
                try:
                    await self.fs_client.hangup_call(call_uuid)
                except:
                    pass
    
    async def _process_audio_loop(self, call_uuid: str) -> None:
        """
        Process audio in a loop for a call.
        
        Args:
            call_uuid: Call UUID
        """
        logger.info(f"Starting audio processing loop for {call_uuid}")
        
        # Create a listener for audio data
        # In a real implementation, you would access audio data from FreeSWITCH
        # using appropriate mechanisms (e.g., mod_shout, ESL events, etc.)
        # For this example, we'll use a simulated approach
        
        # Track silence for end of utterance detection
        silence_start = None
        is_speaking = False
        
        while call_uuid in self.active_calls:
            try:
                # In a real implementation, you would get audio data from FreeSWITCH
                # For this example, we'll simulate by reading from a recording file
                temp_recording_file = f"/tmp/voice_ai_recording_{call_uuid}_temp.wav"
                
                # Check if the file exists and has data
                if os.path.exists(temp_recording_file) and os.path.getsize(temp_recording_file) > 0:
                    import wave
                    try:
                        with wave.open(temp_recording_file, 'rb') as wf:
                            # Read audio frames
                            frames = wf.readframes(wf.getnframes())
                            
                            # Process audio data
                            if frames:
                                # Convert to the format expected by the pipeline
                                audio_data = self.audio_processor.convert_to_pipeline_format(
                                    frames,
                                    sample_rate_in=SAMPLE_RATE_FS,
                                    sample_rate_out=SAMPLE_RATE_AI
                                )
                                
                                # Add to buffer
                                buffer = self.audio_buffers.get(call_uuid, bytearray())
                                buffer.extend(audio_data)
                                
                                # Limit buffer size
                                if len(buffer) > MAX_BUFFER_SIZE:
                                    buffer = buffer[-MAX_BUFFER_SIZE:]
                                
                                self.audio_buffers[call_uuid] = buffer
                                
                                # Check for speech
                                is_silence = self.audio_processor.is_silence(audio_data)
                                
                                if not is_silence:
                                    # Reset silence timer if we detect speech
                                    silence_start = None
                                    is_speaking = True
                                elif is_speaking:
                                    # Start silence timer if we were speaking
                                    if silence_start is None:
                                        silence_start = time.time()
                                    
                                    # Check if we've been silent long enough to end utterance
                                    if time.time() - silence_start > 1.0:  # 1 second of silence
                                        # Process the audio buffer
                                        if len(buffer) >= AUDIO_BUFFER_SIZE:
                                            await self._process_buffer(call_uuid, bytes(buffer))
                                            buffer.clear()
                                            self.audio_buffers[call_uuid] = buffer
                                        
                                        is_speaking = False
                                        silence_start = None
                    except Exception as wave_error:
                        logger.error(f"Error reading audio file: {wave_error}")
                
                # Brief pause to avoid tight loop
                await asyncio.sleep(0.1)
                
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"Error in audio processing loop for {call_uuid}: {e}")
                await asyncio.sleep(1)  # Longer pause on error
    
    async def _process_buffer(self, call_uuid: str, audio_data: bytes) -> None:
        """
        Process an audio buffer through the AI pipeline.
        
        Args:
            call_uuid: Call UUID
            audio_data: Audio data as bytes
        """
        logger.info(f"Processing audio buffer for {call_uuid} ({len(audio_data)} bytes)")
        
        try:
            # Process through pipeline
            result = await self.pipeline.process_audio_data(
                audio_data=audio_data,
                speech_output_path=None  # We'll handle the audio directly
            )
            
            if "error" in result:
                logger.error(f"Error processing audio: {result['error']}")
                return
                
            # Get transcription and response
            transcription = result.get("transcription", "")
            response = result.get("response", "")
            
            if not transcription or not response:
                logger.warning(f"No valid transcription or response: {transcription}")
                return
                
            # Add to call manager
            self.call_manager.add_conversation_turn(call_uuid, "user", transcription)
            self.call_manager.add_conversation_turn(call_uuid, "assistant", response)
            
            # Get speech audio
            speech_audio = result.get("speech_audio")
            
            if speech_audio:
                # Convert to format expected by FreeSWITCH
                fs_audio = self.audio_processor.convert_to_freeswitch_format(
                    speech_audio,
                    sample_rate_in=SAMPLE_RATE_AI,
                    sample_rate_out=SAMPLE_RATE_FS
                )
                
                # Play to the call
                await self.fs_client.play_audio(call_uuid, fs_audio, sample_rate=SAMPLE_RATE_FS)
            else:
                # Fallback to TTS if speech audio not available
                await self.fs_client.play_tts(call_uuid, response)
            
        except Exception as e:
            logger.error(f"Error processing buffer: {e}")
    
    async def originate_call(
        self,
        destination: str,
        caller_id: str = "Voice AI <1000>"
    ) -> Dict[str, Any]:
        """
        Originate an outbound call.
        
        Args:
            destination: Destination number or SIP URI
            caller_id: Caller ID to use
            
        Returns:
            Call information
        """
        logger.info(f"Originating call to {destination}")
        
        if not self.is_initialized:
            raise RuntimeError("FreeSWITCH handler not initialized")
            
        # Originate the call
        result = await self.fs_client.originate_call(destination, caller_id)
        
        if result["status"] == "success" and result.get("uuid"):
            # Add call to manager
            self.call_manager.add_call(
                call_uuid=result["uuid"],
                from_number=caller_id.split('<')[1].split('>')[0],
                to_number=destination,
                direction="outbound"
            )
            
            # Initialize audio buffer
            self.audio_buffers[result["uuid"]] = bytearray()
            
            # Add to active calls
            self.active_calls[result["uuid"]] = {
                "uuid": result["uuid"],
                "from": caller_id,
                "to": destination,
                "state": "created",
                "created_at": time.time(),
                "direction": "outbound"
            }
            
        return result
    
    async def hangup_call(self, call_uuid: str) -> bool:
        """
        Hang up a call.
        
        Args:
            call_uuid: Call UUID
            
        Returns:
            True if successful
        """
        logger.info(f"Hanging up call {call_uuid}")
        
        if not self.is_initialized:
            raise RuntimeError("FreeSWITCH handler not initialized")
            
        # Check if call exists
        if call_uuid not in self.active_calls:
            logger.warning(f"Call {call_uuid} not found for hangup")
            return False
            
        # Hang up the call
        result = await self.fs_client.hangup_call(call_uuid)
        
        return result