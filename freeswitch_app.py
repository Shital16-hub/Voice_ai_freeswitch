#!/usr/bin/env python3
"""
FreeSWITCH application for Voice AI Agent.
"""
import os
import sys
import asyncio
import logging
import time
import signal
from typing import Dict, Any, Optional

from dotenv import load_dotenv

# Import our modules
from voice_ai_agent import VoiceAIAgent
from freeswitch.fs_handler import FreeSwitchHandler
from freeswitch.esl_client import ESLClient
from config.freeswitch_config import (
    LOG_LEVEL, LOG_FORMAT, LOG_FILE,
    FS_HOST, FS_PORT, FS_PASSWORD,
    INBOUND_SOCKET_HOST, INBOUND_SOCKET_PORT
)

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format=LOG_FORMAT,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE)
    ]
)

logger = logging.getLogger(__name__)

# Global variables
voice_ai_agent = None
fs_handler = None
shutdown_event = asyncio.Event()
server = None

async def initialize_system(
    storage_dir: str = './storage',
    model_name: str = 'mistral:7b-instruct-v0.2-q4_0'
) -> Dict[str, Any]:
    """
    Initialize the Voice AI Agent with FreeSWITCH integration.
    
    Args:
        storage_dir: Directory for persistent storage
        model_name: LLM model name
        
    Returns:
        Initialized components
    """
    global voice_ai_agent, fs_handler
    
    logger.info("Initializing Voice AI Agent with FreeSWITCH integration")
    
    try:
        # Initialize Voice AI Agent
        voice_ai_agent = VoiceAIAgent(
            storage_dir=storage_dir,
            model_name=model_name,
            llm_temperature=0.7
        )
        await voice_ai_agent.init()
        logger.info("Voice AI Agent initialized")
        
        # Create ESL client
        esl_client = ESLClient(
            host=FS_HOST,
            port=FS_PORT,
            password=FS_PASSWORD
        )
        
        # Initialize FreeSWITCH handler
        fs_handler = FreeSwitchHandler(
            pipeline=voice_ai_agent,
            fs_client=esl_client
        )
        await fs_handler.init()
        logger.info("FreeSWITCH handler initialized")
        
        return {
            'voice_ai_agent': voice_ai_agent,
            'fs_handler': fs_handler
        }
        
    except Exception as e:
        logger.error(f"Error initializing system: {e}", exc_info=True)
        
        # Clean up if initialization fails
        if voice_ai_agent:
            await voice_ai_agent.shutdown()
        
        if fs_handler:
            await fs_handler.shutdown()
            
        raise

async def start_inbound_socket_server():
    """
    Start inbound socket server for FreeSWITCH to connect to.
    This would be used when FreeSWITCH sends calls to us using socket application.
    """
    global server
    
    import socket
    
    logger.info(f"Starting inbound socket server on {INBOUND_SOCKET_HOST}:{INBOUND_SOCKET_PORT}")
    
    # Create a socket server
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    # Set socket options
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    # Bind to address and port
    server.bind((INBOUND_SOCKET_HOST, INBOUND_SOCKET_PORT))
    
    # Listen for connections
    server.listen(5)
    
    # Set to non-blocking mode
    server.setblocking(False)
    
    # Create a loop task to accept connections
    asyncio.create_task(accept_connections(server))
    
    logger.info(f"Inbound socket server running on {INBOUND_SOCKET_HOST}:{INBOUND_SOCKET_PORT}")

async def accept_connections(server_socket):
    """
    Accept incoming socket connections from FreeSWITCH.
    
    Args:
        server_socket: Server socket object
    """
    loop = asyncio.get_event_loop()
    
    while not shutdown_event.is_set():
        try:
            # Accept connection (wrapped in executor to make it async)
            client, address = await loop.sock_accept(server_socket)
            logger.info(f"Accepted connection from {address}")
            
            # Handle client in a separate task
            asyncio.create_task(handle_client(client, address))
            
        except asyncio.CancelledError:
            logger.info("Accept connections task cancelled")
            break
        except Exception as e:
            logger.error(f"Error accepting connection: {e}")
            await asyncio.sleep(1)  # Avoid tight loop on errors

async def handle_client(client_socket, address):
    """
    Handle a client connection from FreeSWITCH.
    
    Args:
        client_socket: Client socket
        address: Client address
    """
    logger.info(f"Handling client connection from {address}")
    
    loop = asyncio.get_event_loop()
    
    try:
        # Set up connection with FreeSWITCH
        # This would involve implementing the ESL protocol
        # For a complete implementation, refer to FreeSWITCH ESL documentation
        
        # Example of initial interaction
        # Send welcome message
        await loop.sock_sendall(client_socket, b"Content-Type: auth/request\n\n")
        
        # Receive auth
        data = await loop.sock_recv(client_socket, 1024)
        
        # Process data and implement the ESL protocol
        # ...
        
        # Example of processing a call
        await process_freeswitch_call(client_socket, data)
        
    except Exception as e:
        logger.error(f"Error handling client: {e}", exc_info=True)
    finally:
        # Close client socket
        client_socket.close()
        logger.info(f"Closed connection from {address}")

async def process_freeswitch_call(client_socket, initial_data):
    """
    Process a FreeSWITCH call via socket connection.
    
    Args:
        client_socket: Client socket
        initial_data: Initial data received
    """
    # This is a placeholder for the actual implementation
    # In a real implementation, you would:
    # 1. Parse the channel data from FreeSWITCH
    # 2. Set up audio streaming
    # 3. Process audio through your AI pipeline
    # 4. Send responses back to FreeSWITCH
    
    # For now, we'll just log and send a simple response
    logger.info(f"Processing FreeSWITCH call: {initial_data}")
    
    loop = asyncio.get_event_loop()
    
    try:
        # Send a simple response
        response = "Content-Type: command/reply\nReply-Text: +OK\n\n"
        await loop.sock_sendall(client_socket, response.encode())
        
        # Example of playing a greeting
        play_cmd = "sendmsg\nCall-Command: execute\nExecute-App-Name: playback\nExecute-App-Arg: ivr/ivr-welcome_to_freeswitch.wav\n\n"
        await loop.sock_sendall(client_socket, play_cmd.encode())
        
        # Wait a bit for audio to play
        await asyncio.sleep(3)
        
        # Example of using the Voice AI pipeline would go here
        # ...
        
    except Exception as e:
        logger.error(f"Error processing call: {e}")
        
async def shutdown_system():
    """Shut down all components gracefully."""
    logger.info("Shutting down system...")
    
    # Stop accepting new connections
    global server
    if server:
        server.close()
        logger.info("Inbound socket server stopped")
    
    # Shut down components in reverse order
    if fs_handler:
        await fs_handler.shutdown()
        
    if voice_ai_agent:
        await voice_ai_agent.shutdown()
        
    logger.info("System shutdown complete")

def signal_handler():
    """Handle termination signals."""
    logger.info("Received termination signal")
    shutdown_event.set()

async def main():
    """Main function."""
    import argparse
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Voice AI Agent with FreeSWITCH socket integration")
    parser.add_argument('--storage-dir', type=str, default='./storage',
                        help='Directory for persistent storage')
    parser.add_argument('--model', type=str, default='mistral:7b-instruct-v0.2-q4_0',
                        help='LLM model name')
    parser.add_argument('--inbound-host', type=str, default=INBOUND_SOCKET_HOST,
                        help='Inbound socket host')
    parser.add_argument('--inbound-port', type=int, default=INBOUND_SOCKET_PORT,
                        help='Inbound socket port')
    parser.add_argument('--log-level', type=str, default=LOG_LEVEL,
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Logging level')
    
    args = parser.parse_args()
    
    # Update logging level if specified
    if args.log_level != LOG_LEVEL:
        logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Set up signal handlers
    for sig in (signal.SIGINT, signal.SIGTERM):
        asyncio.get_event_loop().add_signal_handler(sig, signal_handler)
    
    try:
        # Initialize the system
        await initialize_system(
            storage_dir=args.storage_dir,
            model_name=args.model
        )
        
        # Start inbound socket server
        await start_inbound_socket_server()
        
        logger.info("System initialized and running. Press Ctrl+C to exit.")
        
        # Run until shutdown event
        await shutdown_event.wait()
        
    except Exception as e:
        logger.error(f"Error in main: {e}", exc_info=True)
    finally:
        # Always shutdown properly
        await shutdown_system()

if __name__ == "__main__":
    asyncio.run(main())