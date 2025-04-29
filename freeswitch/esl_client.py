"""
FreeSWITCH ESL client implementation.
"""
import time
import logging
import asyncio
import uuid
from typing import Dict, Any, Optional, Callable, List, Union, AsyncIterator

# Use greenswitch for ESL communication
import greenswitch

from config.freeswitch_config import (
    FS_HOST, FS_PORT, FS_PASSWORD, FS_TIMEOUT,
    INBOUND_SOCKET_HOST, INBOUND_SOCKET_PORT
)

logger = logging.getLogger(__name__)

class ESLClient:
    """
    Event Socket Layer client for FreeSWITCH communication.
    
    Supports both inbound (we connect to FreeSWITCH) and outbound
    (FreeSWITCH connects to us) connection modes.
    """
    
    def __init__(
        self,
        host: str = FS_HOST,
        port: int = FS_PORT,
        password: str = FS_PASSWORD,
        timeout: int = FS_TIMEOUT
    ):
        """
        Initialize ESL client.
        
        Args:
            host: FreeSWITCH ESL host
            port: FreeSWITCH ESL port
            password: FreeSWITCH ESL password
            timeout: Connection timeout in seconds
        """
        self.host = host
        self.port = port
        self.password = password
        self.timeout = timeout
        self.connection = None
        self.connected = False
        self.event_handlers = {}
        self.handler_ids = {}  # Store handler IDs for removal
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 10
        self.reconnect_delay = 2  # seconds
        
        # Event processing
        self._event_loop = None
        self._event_task = None
        self._running = False
    
    async def connect(self) -> bool:
        """
        Connect to FreeSWITCH ESL.
        
        Returns:
            True if connection successful
        """
        logger.info(f"Connecting to FreeSWITCH ESL at {self.host}:{self.port}")
        
        try:
            # Create connection using greenswitch
            self.connection = greenswitch.InboundESL(
                host=self.host,
                port=self.port,
                password=self.password,
                timeout=self.timeout
            )
            self.connection.connect()
            
            # Check if connected
            if self.connection and self.connection.connected():
                logger.info("Connected to FreeSWITCH ESL")
                self.connected = True
                self.reconnect_attempts = 0
                
                # Subscribe to events
                await self._subscribe_to_events()
                
                # Start event processing
                self._start_event_processing()
                
                return True
            else:
                logger.error("Failed to connect to FreeSWITCH ESL")
                return False
                
        except Exception as e:
            logger.error(f"Error connecting to FreeSWITCH ESL: {e}")
            
            # Try to reconnect
            if self.reconnect_attempts < self.max_reconnect_attempts:
                self.reconnect_attempts += 1
                await asyncio.sleep(self.reconnect_delay * self.reconnect_attempts)
                return await self.connect()
            
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from FreeSWITCH ESL."""
        logger.info("Disconnecting from FreeSWITCH ESL")
        
        # Stop event processing
        self._stop_event_processing()
        
        # Disconnect
        if self.connection:
            try:
                self.connection.disconnect()
                logger.info("Disconnected from FreeSWITCH ESL")
            except Exception as e:
                logger.error(f"Error disconnecting from FreeSWITCH ESL: {e}")
        
        self.connected = False
        self.connection = None
    
    async def _subscribe_to_events(self) -> None:
        """Subscribe to FreeSWITCH events."""
        if not self.connected or not self.connection:
            logger.error("Cannot subscribe to events: not connected")
            return
        
        try:
            # Subscribe to basic call events
            events = [
                "CHANNEL_CREATE",
                "CHANNEL_ANSWER",
                "CHANNEL_HANGUP",
                "CHANNEL_HANGUP_COMPLETE",
                "DTMF",
                "DETECTED_SPEECH",
                "DETECTED_TONE",
                "RECORD_STOP",
                "CUSTOM"
            ]
            
            event_str = " ".join(events)
            self.connection.send(f"event plain {event_str}")
            
            logger.info(f"Subscribed to FreeSWITCH events: {event_str}")
        except Exception as e:
            logger.error(f"Error subscribing to events: {e}")
    
    def _start_event_processing(self) -> None:
        """Start event processing loop."""
        if self._event_task and not self._event_task.done():
            return
        
        # Get or create event loop
        try:
            self._event_loop = asyncio.get_event_loop()
        except RuntimeError:
            self._event_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._event_loop)
        
        # Start event processing task
        self._running = True
        self._event_task = asyncio.create_task(self._event_processing_loop())
        logger.info("Started FreeSWITCH event processing")
    
    def _stop_event_processing(self) -> None:
        """Stop event processing loop."""
        self._running = False
        
        if self._event_task and not self._event_task.done():
            self._event_task.cancel()
            logger.info("Stopped FreeSWITCH event processing")
        
        self._event_task = None
    
    async def _event_processing_loop(self) -> None:
        """
        Event processing loop.
        
        This loop runs in the background and processes events from FreeSWITCH.
        """
        logger.info("Starting FreeSWITCH event processing loop")
        
        while self._running and self.connected and self.connection:
            try:
                # Get event (non-blocking with timeout)
                event = self.connection.get_event(timeout=0.1)
                
                if event:
                    # Process event
                    await self._process_event(event)
            except asyncio.CancelledError:
                logger.info("Event processing loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in event processing loop: {e}")
                
                # Check if still connected
                if self.connection and not self.connection.connected():
                    logger.error("FreeSWITCH connection lost")
                    self.connected = False
                    
                    # Try to reconnect
                    await self._try_reconnect()
                    
                # Avoid tight loop on errors
                await asyncio.sleep(0.5)
    
    async def _try_reconnect(self) -> None:
        """Try to reconnect to FreeSWITCH."""
        if not self._running:
            return
            
        logger.info("Attempting to reconnect to FreeSWITCH")
        
        # Disconnect first
        await self.disconnect()
        
        # Try to reconnect
        await self.connect()
    
    async def _process_event(self, event) -> None:
        """
        Process a FreeSWITCH event.
        
        Args:
            event: FreeSWITCH event object
        """
        # Get event type
        event_name = event.get_header("Event-Name")
        
        if not event_name:
            return
            
        logger.debug(f"Processing FreeSWITCH event: {event_name}")
        
        # Call registered handlers for this event
        handlers = list(self.event_handlers.get(event_name, {}).values())
        handlers.extend(list(self.event_handlers.get("ALL", {}).values()))
        
        for handler in handlers:
            try:
                await handler(event)
            except Exception as e:
                logger.error(f"Error in event handler for {event_name}: {e}")
    
    def register_event_handler(
        self,
        event_name: str,
        handler: Callable
    ) -> str:
        """
        Register an event handler.
        
        Args:
            event_name: Event name to handle
            handler: Async function to call when event occurs
            
        Returns:
            Handler ID for later unregistration
        """
        if event_name not in self.event_handlers:
            self.event_handlers[event_name] = {}
        
        # Generate a unique ID for this handler
        handler_id = str(uuid.uuid4())
        
        # Store the handler with its ID
        self.event_handlers[event_name][handler_id] = handler
        self.handler_ids[handler_id] = event_name
        
        logger.info(f"Registered handler for event: {event_name} (ID: {handler_id})")
        return handler_id
    
    def unregister_event_handler(self, handler_id: str) -> bool:
        """
        Unregister an event handler.
        
        Args:
            handler_id: Handler ID returned from register_event_handler
            
        Returns:
            True if handler was found and removed
        """
        if handler_id not in self.handler_ids:
            logger.warning(f"Handler ID {handler_id} not found")
            return False
        
        # Get the event name for this handler
        event_name = self.handler_ids[handler_id]
        
        # Remove the handler
        if event_name in self.event_handlers and handler_id in self.event_handlers[event_name]:
            del self.event_handlers[event_name][handler_id]
            del self.handler_ids[handler_id]
            
            # Clean up empty event name dicts
            if not self.event_handlers[event_name]:
                del self.event_handlers[event_name]
                
            logger.info(f"Unregistered handler for event: {event_name} (ID: {handler_id})")
            return True
        
        return False
    
    async def send_command(self, command: str) -> str:
        """
        Send a command to FreeSWITCH.
        
        Args:
            command: Command to send
            
        Returns:
            Command response
        """
        if not self.connected or not self.connection:
            logger.error(f"Cannot send command: not connected - {command}")
            raise RuntimeError("Not connected to FreeSWITCH")
        
        logger.debug(f"Sending command: {command}")
        
        try:
            # Send command
            self.connection.send(command)
            
            # Get response
            response = self.connection.get_event().get_body()
            logger.debug(f"Command response: {response[:100]}...")
            
            return response
        except Exception as e:
            logger.error(f"Error sending command: {e}")
            
            # Check if still connected
            if self.connection and not self.connection.connected():
                logger.error("FreeSWITCH connection lost during command")
                self.connected = False
                await self._try_reconnect()
                
            raise
    
    async def originate_call(
        self,
        destination: str,
        caller_id: str = "Voice AI <1000>",
        context: str = "default",
        extension: str = "5000"
    ) -> Dict[str, Any]:
        """
        Originate an outbound call.
        
        Args:
            destination: Destination number/URI
            caller_id: Caller ID to use
            context: Dialplan context
            extension: Dialplan extension
            
        Returns:
            Call information
        """
        if not self.connected:
            logger.error("Cannot originate call: not connected")
            raise RuntimeError("Not connected to FreeSWITCH")
        
        logger.info(f"Originating call to {destination}")
        
        try:
            # Parse caller ID parts safely
            caller_id_name = caller_id.split('<')[0].strip()
            caller_id_number = "1000"  # Default
            
            # Extract number from caller ID if possible
            if '<' in caller_id and '>' in caller_id:
                caller_id_number = caller_id.split('<')[1].split('>')[0]
            
            # Build originate command
            originate_cmd = (
                f"originate {{caller_id_name='{caller_id_name}',"
                f"caller_id_number='{caller_id_number}',"
                f"origination_caller_id_name='{caller_id_name}',"
                f"origination_caller_id_number='{caller_id_number}'}}user/{destination} "
                f"{extension} XML {context}"
            )
            
            # Send command
            response = await self.send_command(originate_cmd)
            
            # Parse UUID from response
            call_uuid = None
            if "+OK " in response:
                call_uuid = response.split("+OK ")[1].strip()
                
            logger.info(f"Originated call to {destination}, UUID: {call_uuid}")
            
            return {
                "uuid": call_uuid,
                "destination": destination,
                "caller_id": caller_id,
                "status": "success" if call_uuid else "error",
                "response": response
            }
            
        except Exception as e:
            logger.error(f"Error originating call: {e}")
            
            return {
                "status": "error",
                "error": str(e),
                "destination": destination
            }
    
    async def hangup_call(self, call_uuid: str, cause: str = "NORMAL_CLEARING") -> bool:
        """
        Hang up a call.
        
        Args:
            call_uuid: Call UUID
            cause: Hangup cause
            
        Returns:
            True if successful
        """
        if not self.connected:
            logger.error("Cannot hangup call: not connected")
            return False
        
        logger.info(f"Hanging up call {call_uuid}")
        
        try:
            # Send hangup command
            hangup_cmd = f"uuid_kill {call_uuid} {cause}"
            response = await self.send_command(hangup_cmd)
            
            success = "+OK" in response
            
            if success:
                logger.info(f"Successfully hung up call {call_uuid}")
            else:
                logger.error(f"Failed to hang up call {call_uuid}: {response}")
                
            return success
            
        except Exception as e:
            logger.error(f"Error hanging up call: {e}")
            return False
    
    async def play_audio(
        self,
        call_uuid: str,
        audio_data: bytes,
        sample_rate: int = 8000,
        channels: int = 1
    ) -> bool:
        """
        Play audio to a call.
        
        Args:
            call_uuid: Call UUID
            audio_data: Audio data as bytes
            sample_rate: Sample rate of audio
            channels: Number of audio channels
            
        Returns:
            True if successful
        """
        if not self.connected:
            logger.error("Cannot play audio: not connected")
            return False
        
        logger.info(f"Playing audio to call {call_uuid} ({len(audio_data)} bytes)")
        
        try:
            # We'll use displace_session to play audio
            # This requires writing the audio to a temporary file
            import tempfile
            import os
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
                # Create WAV file
                import wave
                with wave.open(temp_file.name, 'wb') as wf:
                    wf.setnchannels(channels)
                    wf.setsampwidth(2)  # 16-bit
                    wf.setframerate(sample_rate)
                    wf.writeframes(audio_data)
                
                # Get file path
                file_path = temp_file.name
            
            # Now play the file
            play_cmd = f"uuid_displace {call_uuid} start {file_path} 0 mux"
            response = await self.send_command(play_cmd)
            
            success = "+OK" in response
            
            # Delete the temporary file
            try:
                os.unlink(file_path)
            except Exception as file_error:
                logger.warning(f"Could not delete temporary file {file_path}: {file_error}")
                
            if success:
                logger.info(f"Successfully playing audio to call {call_uuid}")
            else:
                logger.error(f"Failed to play audio to call {call_uuid}: {response}")
                
            return success
            
        except Exception as e:
            logger.error(f"Error playing audio: {e}")
            return False
    
    async def play_tts(self, call_uuid: str, text: str, voice: str = "slt") -> bool:
        """
        Play TTS audio to a call.
        
        Args:
            call_uuid: Call UUID
            text: Text to speak
            voice: TTS voice to use
            
        Returns:
            True if successful
        """
        if not self.connected:
            logger.error("Cannot play TTS: not connected")
            return False
        
        logger.info(f"Playing TTS to call {call_uuid}: {text[:50]}...")
        
        try:
            # Escape single quotes in text
            safe_text = text.replace("'", "\\'")
            
            # Use speak command
            speak_cmd = f'uuid_speak {call_uuid} tts://{voice}|{safe_text}'
            response = await self.send_command(speak_cmd)
            
            success = "+OK" in response
            
            if success:
                logger.info(f"Successfully playing TTS to call {call_uuid}")
            else:
                logger.error(f"Failed to play TTS to call {call_uuid}: {response}")
                
            return success
            
        except Exception as e:
            logger.error(f"Error playing TTS: {e}")
            return False
    
    async def start_recording(
        self, 
        call_uuid: str,
        channels: str = "mono"
    ) -> Dict[str, Any]:
        """
        Start recording a call.
        
        Args:
            call_uuid: Call UUID
            channels: "mono" or "stereo"
            
        Returns:
            Recording information
        """
        if not self.connected:
            logger.error("Cannot start recording: not connected")
            raise RuntimeError("Not connected to FreeSWITCH")
        
        logger.info(f"Starting recording for call {call_uuid}")
        
        try:
            # Generate a filename
            timestamp = int(time.time())
            filename = f"/tmp/voice_ai_recording_{call_uuid}_{timestamp}.wav"
            
            # Start recording
            record_cmd = f"uuid_record {call_uuid} start {filename}"
            response = await self.send_command(record_cmd)
            
            success = "+OK" in response
            
            if success:
                logger.info(f"Started recording for call {call_uuid}")
                return {
                    "status": "success",
                    "filename": filename,
                    "call_uuid": call_uuid
                }
            else:
                logger.error(f"Failed to start recording: {response}")
                return {
                    "status": "error",
                    "message": response,
                    "call_uuid": call_uuid
                }
                
        except Exception as e:
            logger.error(f"Error starting recording: {e}")
            return {
                "status": "error",
                "error": str(e),
                "call_uuid": call_uuid
            }
    
    async def stop_recording(self, call_uuid: str) -> bool:
        """
        Stop recording a call.
        
        Args:
            call_uuid: Call UUID
            
        Returns:
            True if successful
        """
        if not self.connected:
            logger.error("Cannot stop recording: not connected")
            return False
        
        logger.info(f"Stopping recording for call {call_uuid}")
        
        try:
            # Stop recording
            record_cmd = f"uuid_record {call_uuid} stop all"
            response = await self.send_command(record_cmd)
            
            success = "+OK" in response
            
            if success:
                logger.info(f"Stopped recording for call {call_uuid}")
            else:
                logger.error(f"Failed to stop recording: {response}")
                
            return success
            
        except Exception as e:
            logger.error(f"Error stopping recording: {e}")
            return False
    
    async def get_channel_variable(
        self,
        call_uuid: str,
        variable: str
    ) -> Optional[str]:
        """
        Get a channel variable.
        
        Args:
            call_uuid: Call UUID
            variable: Variable name
            
        Returns:
            Variable value or None
        """
        if not self.connected:
            logger.error("Cannot get channel variable: not connected")
            return None
        
        logger.debug(f"Getting channel variable {variable} for call {call_uuid}")
        
        try:
            # Get variable
            cmd = f"uuid_getvar {call_uuid} {variable}"
            response = await self.send_command(cmd)
            
            # Parse response
            if "+OK" in response:
                value = response.split("+OK")[1].strip()
                return value
            else:
                logger.warning(f"Failed to get variable {variable}: {response}")
                return None
                
        except Exception as e:
            logger.error(f"Error getting channel variable: {e}")
            return None
    
    async def set_channel_variable(
        self,
        call_uuid: str,
        variable: str,
        value: str
    ) -> bool:
        """
        Set a channel variable.
        
        Args:
            call_uuid: Call UUID
            variable: Variable name
            value: Variable value
            
        Returns:
            True if successful
        """
        if not self.connected:
            logger.error("Cannot set channel variable: not connected")
            return False
        
        logger.debug(f"Setting channel variable {variable}={value} for call {call_uuid}")
        
        try:
            # Set variable
            cmd = f"uuid_setvar {call_uuid} {variable} {value}"
            response = await self.send_command(cmd)
            
            success = "+OK" in response
            
            if not success:
                logger.warning(f"Failed to set variable {variable}: {response}")
                
            return success
                
        except Exception as e:
            logger.error(f"Error setting channel variable: {e}")
            return False