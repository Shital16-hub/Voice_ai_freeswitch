"""
Event Socket Layer utility functions for FreeSWITCH integration.
"""
import logging
import asyncio
from typing import Dict, Any, Optional, List, Callable, Awaitable

logger = logging.getLogger(__name__)

def parse_event_headers(event) -> Dict[str, str]:
    """
    Parse FreeSWITCH event headers.
    
    Args:
        event: FreeSWITCH event object
        
    Returns:
        Dictionary of headers
    """
    headers = {}
    
    # Handle different event types (greenswitch or ESL module)
    if hasattr(event, 'get_header'):
        # greenswitch style
        header_names = event.get_headers()
        for name in header_names:
            headers[name] = event.get_header(name)
    elif hasattr(event, 'getHeader'):
        # ESL module style
        header_names = event.getHeaders()
        for name in header_names:
            headers[name] = event.getHeader(name)
    elif isinstance(event, dict):
        # Dictionary style
        headers = event
    else:
        logger.warning(f"Unknown event type: {type(event)}")
    
    return headers

def format_api_command(command: str, args: str) -> str:
    """
    Format an API command for FreeSWITCH.
    
    Args:
        command: Command name
        args: Command arguments
        
    Returns:
        Formatted command string
    """
    return f"api {command} {args}"

def format_bgapi_command(command: str, args: str, job_uuid: Optional[str] = None) -> str:
    """
    Format a background API command for FreeSWITCH.
    
    Args:
        command: Command name
        args: Command arguments
        job_uuid: Optional job UUID
        
    Returns:
        Formatted command string
    """
    if job_uuid:
        return f"bgapi {command} {args} {job_uuid}"
    else:
        return f"bgapi {command} {args}"

def extract_channel_variables(event) -> Dict[str, str]:
    """
    Extract channel variables from an event.
    
    Args:
        event: FreeSWITCH event
        
    Returns:
        Dictionary of channel variables
    """
    variables = {}
    headers = parse_event_headers(event)
    
    # Look for variable_ prefixed headers
    for name, value in headers.items():
        if name.startswith('variable_'):
            var_name = name[len('variable_'):]
            variables[var_name] = value
    
    return variables

def format_origination_variables(variables: Dict[str, str]) -> str:
    """
    Format variables for an origination command.
    
    Args:
        variables: Dictionary of variables
        
    Returns:
        Formatted variables string
    """
    vars_list = [f"{name}='{value}'" for name, value in variables.items()]
    return '{' + ','.join(vars_list) + '}'

async def wait_for_event(
    esl_client,
    event_name: str,
    filters: Dict[str, str],
    timeout: float = 30.0
) -> Optional[Dict[str, str]]:
    """
    Wait for a specific event to occur.
    
    Args:
        esl_client: ESL client
        event_name: Event name to wait for
        filters: Event filters (headers to match)
        timeout: Maximum time to wait in seconds
        
    Returns:
        Event headers or None if timeout
    """
    # Create future to be resolved when event is received
    future = asyncio.Future()
    
    # Event handler
    async def event_handler(event):
        headers = parse_event_headers(event)
        
        # Check if all filters match
        match = True
        for key, value in filters.items():
            if headers.get(key) != value:
                match = False
                break
        
        if match and not future.done():
            future.set_result(headers)
    
    # Register temporary handler
    handler_id = esl_client.register_event_handler(event_name, event_handler)
    
    try:
        # Wait for event with timeout
        return await asyncio.wait_for(future, timeout=timeout)
    except asyncio.TimeoutError:
        logger.warning(f"Timeout waiting for event: {event_name} with filters {filters}")
        return None
    finally:
        # Remove the temporary handler
        esl_client.unregister_event_handler(handler_id)