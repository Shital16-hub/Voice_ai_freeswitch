#!/usr/bin/env python3
"""
Main entry point for the Voice AI Agent with FreeSWITCH integration.
"""
import os
import sys
import argparse
import asyncio
import logging
import signal
from typing import Dict, Any, Optional

from dotenv import load_dotenv

# Import our modules
from voice_ai_agent import VoiceAIAgent
from freeswitch.fs_handler import FreeSwitchHandler
from freeswitch.esl_client import ESLClient
from config.freeswitch_config import (
    LOG_LEVEL, LOG_FORMAT, LOG_FILE,
    FS_HOST, FS_PORT, FS_PASSWORD
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
        
async def shutdown_system():
    """Shut down all components gracefully."""
    logger.info("Shutting down system...")
    
    # Shut down in reverse order
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
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Voice AI Agent with FreeSWITCH integration")
    parser.add_argument('--storage-dir', type=str, default='./storage',
                        help='Directory for persistent storage')
    parser.add_argument('--model', type=str, default='mistral:7b-instruct-v0.2-q4_0',
                        help='LLM model name')
    parser.add_argument('--fs-host', type=str, default=FS_HOST,
                        help='FreeSWITCH ESL host')
    parser.add_argument('--fs-port', type=int, default=FS_PORT,
                        help='FreeSWITCH ESL port')
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
        
        logger.info("System initialized. Running...")
        
        # Run until shutdown event
        await shutdown_event.wait()
        
    except Exception as e:
        logger.error(f"Error in main: {e}", exc_info=True)
    finally:
        # Always shutdown properly
        await shutdown_system()

if __name__ == "__main__":
    asyncio.run(main())