"""
FreeSWITCH event handler implementation.
"""
import logging
import asyncio
import time
from typing import Dict, Any, Optional, Callable, List, Awaitable

from freeswitch.esl_client import ESLClient
from freeswitch.utils.esl_utils import parse_event_headers, extract_channel_variables
from config.freeswitch_config import (
   GREETING_MESSAGE,
   MAX_CALL_DURATION,
   SILENCE_THRESHOLD,
   SILENCE_DURATION
)

logger = logging.getLogger(__name__)

class EventHandler:
   """
   Handles FreeSWITCH events for the Voice AI Agent.
   
   This class registers handlers for FreeSWITCH events and processes
   them to coordinate the Voice AI functionality.
   """
   
   def __init__(
       self,
       esl_client: ESLClient,
       pipeline,
       call_manager
   ):
       """
       Initialize event handler.
       
       Args:
           esl_client: ESL client for FreeSWITCH communication
           pipeline: Voice AI pipeline
           call_manager: Call manager instance
       """
       self.esl_client = esl_client
       self.pipeline = pipeline
       self.call_manager = call_manager
       
       # Track active processing tasks
       self.processing_tasks = {}
       self.audio_buffers = {}
       
       # Function to handle different events
       self.event_handlers = {
           "CHANNEL_CREATE": self.handle_channel_create,
           "CHANNEL_ANSWER": self.handle_channel_answer,
           "CHANNEL_HANGUP": self.handle_channel_hangup,
           "CHANNEL_HANGUP_COMPLETE": self.handle_channel_hangup_complete,
           "DTMF": self.handle_dtmf,
           "DETECTED_SPEECH": self.handle_speech_detected,
           "RECORD_STOP": self.handle_record_stop
       }
       
       # Handler registration IDs for cleanup
       self.handler_ids = []
   
   async def register_handlers(self) -> None:
       """Register all event handlers with the ESL client."""
       logger.info("Registering FreeSWITCH event handlers")
       
       for event_name, handler_func in self.event_handlers.items():
           handler_id = self.esl_client.register_event_handler(event_name, handler_func)
           self.handler_ids.append(handler_id)
           
       logger.info(f"Registered {len(self.handler_ids)} event handlers")
   
   async def unregister_handlers(self) -> None:
       """Unregister all event handlers."""
       logger.info("Unregistering FreeSWITCH event handlers")
       
       for handler_id in self.handler_ids:
           self.esl_client.unregister_event_handler(handler_id)
           
       self.handler_ids.clear()
       logger.info("Event handlers unregistered")
   
   async def handle_channel_create(self, event) -> None:
       """
       Handle channel create event.
       
       Args:
           event: FreeSWITCH event
       """
       # Extract event headers
       headers = parse_event_headers(event)
       
       # Get call details
       call_uuid = headers.get("Unique-ID")
       caller_id_number = headers.get("Caller-Caller-ID-Number")
       destination_number = headers.get("Caller-Destination-Number")
       
       if not call_uuid:
           logger.warning("Received channel create event without UUID")
           return
           
       logger.info(f"Channel created - UUID: {call_uuid}, From: {caller_id_number}, To: {destination_number}")
       
       # Extract channel variables
       variables = extract_channel_variables(event)
       
       # Add to call manager
       self.call_manager.add_call(
           call_uuid=call_uuid,
           from_number=caller_id_number,
           to_number=destination_number
       )
       
       # Initialize audio buffer
       self.audio_buffers[call_uuid] = bytearray()
       
       # Store variables
       for name, value in variables.items():
           self.call_manager.set_variable(call_uuid, name, value)
   
   async def handle_channel_answer(self, event) -> None:
       """
       Handle channel answer event.
       
       Args:
           event: FreeSWITCH event
       """
       # Extract event headers
       headers = parse_event_headers(event)
       
       # Get call UUID
       call_uuid = headers.get("Unique-ID")
       
       if not call_uuid:
           logger.warning("Received channel answer event without UUID")
           return
           
       logger.info(f"Channel answered - UUID: {call_uuid}")
       
       # Update call status
       self.call_manager.update_call_status(call_uuid, "active")
       
       # Play greeting
       try:
           await self.esl_client.play_tts(call_uuid, GREETING_MESSAGE)
           
           # Start a task to handle the call
           self.processing_tasks[call_uuid] = asyncio.create_task(
               self._handle_call(call_uuid)
           )
       except Exception as e:
           logger.error(f"Error handling channel answer: {e}")
   
   async def handle_channel_hangup(self, event) -> None:
       """
       Handle channel hangup event.
       
       Args:
           event: FreeSWITCH event
       """
       # Extract event headers
       headers = parse_event_headers(event)
       
       # Get call UUID
       call_uuid = headers.get("Unique-ID")
       
       if not call_uuid:
           logger.warning("Received channel hangup event without UUID")
           return
           
       # Get hangup cause
       hangup_cause = headers.get("Hangup-Cause")
           
       logger.info(f"Channel hangup - UUID: {call_uuid}, Cause: {hangup_cause}")
       
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
   
   async def handle_channel_hangup_complete(self, event) -> None:
       """
       Handle channel hangup complete event.
       
       Args:
           event: FreeSWITCH event
       """
       # Extract event headers
       headers = parse_event_headers(event)
       
       # Get call UUID
       call_uuid = headers.get("Unique-ID")
       
       if not call_uuid:
           logger.warning("Received channel hangup complete event without UUID")
           return
           
       logger.info(f"Channel hangup complete - UUID: {call_uuid}")
       
       # Update call status
       self.call_manager.update_call_status(call_uuid, "completed")
       
       # Clean up audio buffer
       if call_uuid in self.audio_buffers:
           del self.audio_buffers[call_uuid]
   
   async def handle_dtmf(self, event) -> None:
       """
       Handle DTMF event.
       
       Args:
           event: FreeSWITCH event
       """
       # Extract event headers
       headers = parse_event_headers(event)
       
       # Get call UUID
       call_uuid = headers.get("Unique-ID")
       
       if not call_uuid:
           logger.warning("Received DTMF event without UUID")
           return
           
       # Get DTMF digit
       dtmf_digit = headers.get("DTMF-Digit")
           
       logger.info(f"DTMF received - UUID: {call_uuid}, Digit: {dtmf_digit}")
       
       # Add to call manager
       self.call_manager.add_dtmf(call_uuid, dtmf_digit)
   
   async def handle_speech_detected(self, event) -> None:
       """
       Handle speech detected event.
       
       Args:
           event: FreeSWITCH event
       """
       # Extract event headers
       headers = parse_event_headers(event)
       
       # Get call UUID
       call_uuid = headers.get("Unique-ID")
       
       if not call_uuid:
           logger.warning("Received speech detected event without UUID")
           return
           
       logger.info(f"Speech detected - UUID: {call_uuid}")
       
       # The actual audio processing will be handled in the _handle_call method
   
   async def handle_record_stop(self, event) -> None:
       """
       Handle record stop event.
       
       Args:
           event: FreeSWITCH event
       """
       # Extract event headers
       headers = parse_event_headers(event)
       
       # Get call UUID
       call_uuid = headers.get("Unique-ID")
       
       if not call_uuid:
           logger.warning("Received record stop event without UUID")
           return
           
       # Get recording file
       recording_file = headers.get("Record-File-Path")
           
       logger.info(f"Recording stopped - UUID: {call_uuid}, File: {recording_file}")
       
       # Process the recording if needed
       if recording_file and call_uuid in self.processing_tasks:
           # The recording will be processed in the _handle_call method
           pass
   
   async def _handle_call(self, call_uuid: str) -> None:
       """
       Handle an active call.
       
       Args:
           call_uuid: Call UUID
       """
       logger.info(f"Starting call handling for {call_uuid}")
       
       try:
           # Start recording for audio capture
           recording_info = await self.esl_client.start_recording(call_uuid)
           
           if recording_info.get("status") != "success":
               logger.error(f"Failed to start recording: {recording_info}")
               return
               
           # Get the recording filename
           recording_file = recording_info.get("filename")
           
           # Set starting time
           start_time = time.time()
           
           # Main call processing loop
           while time.time() - start_time < MAX_CALL_DURATION:
               # Check if the call is still active
               call_info = self.call_manager.get_call(call_uuid)
               if not call_info or call_info.get("status") != "active":
                   logger.info(f"Call {call_uuid} is no longer active")
                   break
               
               # Process audio in chunks
               # In a real implementation, we would analyze the recording file
               # periodically to extract new audio and process it
               
               # For this example, we'll just use a simple delay
               await asyncio.sleep(1.0)
               
               # Add more sophisticated processing here:
               # - Check for new audio in the recording file
               # - Detect speech vs. silence
               # - Process speech through the AI pipeline
               # - Play responses back to the caller
           
           # Maximum call duration reached
           if time.time() - start_time >= MAX_CALL_DURATION:
               logger.info(f"Maximum call duration reached for {call_uuid}")
               
               # Play goodbye message
               await self.esl_client.play_tts(
                   call_uuid, 
                   "Thank you for calling. We've reached the maximum call duration. Goodbye."
               )
               
               # Hang up the call
               await self.esl_client.hangup_call(call_uuid)
           
       except asyncio.CancelledError:
           logger.info(f"Call handling cancelled for {call_uuid}")
       except Exception as e:
           logger.error(f"Error handling call {call_uuid}: {e}")
       finally:
           # Stop recording
           try:
               await self.esl_client.stop_recording(call_uuid)
           except:
               pass
               
           # Clean up
           if call_uuid in self.audio_buffers:
               self.audio_buffers[call_uuid].clear()
               
           logger.info(f"Call handling completed for {call_uuid}")
   
   async def process_audio(self, call_uuid: str, audio_data: bytes) -> None:
       """
       Process audio data through the AI pipeline.
       
       Args:
           call_uuid: Call UUID
           audio_data: Audio data
       """
       logger.info(f"Processing audio for call {call_uuid} ({len(audio_data)} bytes)")
       
       try:
           # Process through the pipeline
           result = await self.pipeline.process_audio_data(audio_data)
           
           if "error" in result:
               logger.error(f"Error processing audio: {result['error']}")
               return
               
           # Get transcription and response
           transcription = result.get("transcription", "")
           response = result.get("response", "")
           
           # Add to call manager
           if transcription:
               self.call_manager.add_conversation_turn(call_uuid, "user", transcription)
               
           if response:
               self.call_manager.add_conversation_turn(call_uuid, "assistant", response)
               
               # Play response to caller
               speech_audio = result.get("speech_audio")
               if speech_audio:
                   await self.esl_client.play_audio(call_uuid, speech_audio)
               else:
                   await self.esl_client.play_tts(call_uuid, response)
                   
       except Exception as e:
           logger.error(f"Error processing audio: {e}")