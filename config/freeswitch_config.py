"""
FreeSWITCH configuration settings for Voice AI Agent.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# FreeSWITCH ESL Connection Settings
FS_HOST = os.getenv('FS_HOST', '127.0.0.1')
FS_PORT = int(os.getenv('FS_PORT', '8021'))
FS_PASSWORD = os.getenv('FS_PASSWORD', 'ClueCon')
FS_TIMEOUT = int(os.getenv('FS_TIMEOUT', '30'))

# Inbound socket settings for FreeSWITCH to connect to us
INBOUND_SOCKET_HOST = os.getenv('INBOUND_SOCKET_HOST', '127.0.0.1')
INBOUND_SOCKET_PORT = int(os.getenv('INBOUND_SOCKET_PORT', '8040'))

# Audio Settings
SAMPLE_RATE_FS = 8000  # FreeSWITCH default sample rate
SAMPLE_RATE_AI = 16000  # AI system sample rate
AUDIO_CHANNELS = 1  # Mono audio
AUDIO_FORMAT = 'L16'  # 16-bit linear PCM

# Buffer Settings
AUDIO_BUFFER_SIZE = 16000  # 2 seconds buffer at 8kHz
MAX_BUFFER_SIZE = 32000  # Maximum buffer size
CHUNK_SIZE = 320  # 20ms at 8kHz

# Performance Settings
MAX_CONCURRENT_CALLS = int(os.getenv('MAX_CONCURRENT_CALLS', '10'))
MAX_CALL_DURATION = 3600  # 1 hour maximum call duration
PROCESSING_TIMEOUT = 5.0  # Maximum time to process audio (seconds)

# Voice AI Settings
GREETING_MESSAGE = "Welcome to the Voice AI Agent. How can I help you today?"
SILENCE_THRESHOLD = 0.01  # Threshold for detecting silence
SILENCE_DURATION = 1.0  # Seconds of silence to consider end of speech

# FreeSWITCH Dialplan Settings
DEFAULT_EXTENSION = '5000'  # Extension to dial for the AI Agent
DEFAULT_CONTEXT = 'default'  # FreeSWITCH context

# Logging Settings
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_FILE = os.getenv('LOG_FILE', '/var/log/voice_ai_agent.log')
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'