"""
Call management for FreeSWITCH integration.
"""
import logging
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime
import time

from config.freeswitch_config import MAX_CALL_DURATION

logger = logging.getLogger(__name__)

class CallManager:
    """
    Manages active calls and their states for FreeSWITCH integration.
    """
    
    def __init__(self):
        """Initialize call manager."""
        self.active_calls: Dict[str, Dict[str, Any]] = {}
        self._cleanup_task = None
    
    async def init(self):
        """Start the call manager."""
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("Call manager started")
    
    async def stop(self):
        """Stop the call manager."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        logger.info("Call manager stopped")
    
    def add_call(
        self,
        call_uuid: str,
        from_number: str,
        to_number: str,
        direction: str = "inbound"
    ) -> None:
        """
        Add a new call to tracking.
        
        Args:
            call_uuid: FreeSWITCH call UUID
            from_number: Caller phone number
            to_number: Called phone number
            direction: Call direction ("inbound" or "outbound")
        """
        self.active_calls[call_uuid] = {
            'call_uuid': call_uuid,
            'from_number': from_number,
            'to_number': to_number,
            'start_time': datetime.now(),
            'status': 'created',
            'direction': direction,
            'transcription': '',
            'response': '',
            'conversation_history': [],
            'dtmf_digits': '',
            'uuid_variables': {}
        }
        logger.info(f"Added {direction} call {call_uuid} from {from_number} to {to_number}")
    
    def update_call_status(self, call_uuid: str, status: str) -> None:
        """
        Update call status.
        
        Args:
            call_uuid: FreeSWITCH call UUID
            status: New status
        """
        if call_uuid in self.active_calls:
            self.active_calls[call_uuid]['status'] = status
            logger.info(f"Updated call {call_uuid} status to {status}")
    
    def add_conversation_turn(
        self,
        call_uuid: str,
        speaker: str,
        text: str
    ) -> None:
        """
        Add a conversation turn to call history.
        
        Args:
            call_uuid: FreeSWITCH call UUID
            speaker: 'user' or 'assistant'
            text: Transcription or response text
        """
        if call_uuid in self.active_calls:
            turn = {
                'speaker': speaker,
                'text': text,
                'timestamp': datetime.now().isoformat()
            }
            self.active_calls[call_uuid]['conversation_history'].append(turn)
            
            # Update transcription or response
            if speaker == 'user':
                self.active_calls[call_uuid]['transcription'] = text
            else:
                self.active_calls[call_uuid]['response'] = text
                
            logger.info(f"Added conversation turn for {call_uuid}: {speaker}: {text[:50]}...")
    
    def add_dtmf(self, call_uuid: str, digit: str) -> None:
        """
        Add DTMF digit to call.
        
        Args:
            call_uuid: FreeSWITCH call UUID
            digit: DTMF digit
        """
        if call_uuid in self.active_calls:
            self.active_calls[call_uuid]['dtmf_digits'] += digit
            logger.info(f"Added DTMF digit {digit} for call {call_uuid}")
    
    def set_variable(
        self,
        call_uuid: str,
        name: str,
        value: str
    ) -> None:
        """
        Set a call variable.
        
        Args:
            call_uuid: FreeSWITCH call UUID
            name: Variable name
            value: Variable value
        """
        if call_uuid in self.active_calls:
            self.active_calls[call_uuid]['uuid_variables'][name] = value
            logger.debug(f"Set variable {name}={value} for call {call_uuid}")
    
    def get_variable(
        self,
        call_uuid: str,
        name: str,
        default: Optional[str] = None
    ) -> Optional[str]:
        """
        Get a call variable.
        
        Args:
            call_uuid: FreeSWITCH call UUID
            name: Variable name
            default: Default value if not found
            
        Returns:
            Variable value or default
        """
        if call_uuid in self.active_calls:
            return self.active_calls[call_uuid]['uuid_variables'].get(name, default)
        return default
    
    def get_call(self, call_uuid: str) -> Optional[Dict[str, Any]]:
        """
        Get call information.
        
        Args:
            call_uuid: FreeSWITCH call UUID
            
        Returns:
            Call information or None if not found
        """
        return self.active_calls.get(call_uuid)
    
    def remove_call(self, call_uuid: str) -> None:
        """
        Remove a call from tracking.
        
        Args:
            call_uuid: FreeSWITCH call UUID
        """
        if call_uuid in self.active_calls:
            call_info = self.active_calls[call_uuid]
            duration = (datetime.now() - call_info['start_time']).total_seconds()
            logger.info(f"Removing call {call_uuid} after {duration:.1f}s")
            
            # Log conversation history
            self._log_conversation(call_info)
            
            del self.active_calls[call_uuid]
    
    def _log_conversation(self, call_info: Dict[str, Any]) -> None:
        """Log conversation history for a call."""
        logger.info(f"Conversation history for call {call_info['call_uuid']}:")
        for turn in call_info['conversation_history']:
            logger.info(f"  {turn['speaker']}: {turn['text'][:100]}")
    
    async def _cleanup_loop(self) -> None:
        """Periodically clean up old calls."""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                current_time = datetime.now()
                
                # Find calls to remove
                calls_to_remove = []
                for call_uuid, call_info in self.active_calls.items():
                    duration = (current_time - call_info['start_time']).total_seconds()
                    
                    # Remove if call is too old or completed
                    if duration > MAX_CALL_DURATION or call_info['status'] in ['completed', 'failed']:
                        calls_to_remove.append(call_uuid)
                
                # Remove old calls
                for call_uuid in calls_to_remove:
                    self.remove_call(call_uuid)
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
    
    def get_active_call_count(self) -> int:
        """Get number of active calls."""
        return len(self.active_calls)
    
    def get_call_stats(self) -> Dict[str, Any]:
        """Get statistics about calls."""
        total_calls = len(self.active_calls)
        active_calls = sum(1 for call in self.active_calls.values() if call['status'] == 'active')
        inbound_calls = sum(1 for call in self.active_calls.values() if call['direction'] == 'inbound')
        outbound_calls = sum(1 for call in self.active_calls.values() if call['direction'] == 'outbound')
        
        return {
            'total_calls': total_calls,
            'active_calls': active_calls,
            'inbound_calls': inbound_calls,
            'outbound_calls': outbound_calls,
            'calls_by_status': self._count_by_status()
        }
    
    def _count_by_status(self) -> Dict[str, int]:
        """Count calls by status."""
        status_count = {}
        for call in self.active_calls.values():
            status = call['status']
            status_count[status] = status_count.get(status, 0) + 1
        return status_count