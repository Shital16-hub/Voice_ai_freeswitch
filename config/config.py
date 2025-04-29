"""
Central configuration settings for Voice AI Agent.
"""
import os
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# AI Model Settings
MODEL_NAME = os.getenv('MODEL_NAME', 'mistral:7b-instruct-v0.2-q4_0')
LLM_TEMPERATURE = float(os.getenv('LLM_TEMPERATURE', '0.7'))
STT_MODEL_PATH = os.getenv('STT_MODEL_PATH', 'models/base.en')
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'sentence-transformers/paraphrase-MiniLM-L3-v2')

# Voice AI Pipeline Settings
MAX_TOKENS = int(os.getenv('MAX_TOKENS', '1024'))
MAX_CONVERSATION_HISTORY = int(os.getenv('MAX_CONVERSATION_HISTORY', '5'))
CONTEXT_WINDOW_SIZE = int(os.getenv('CONTEXT_WINDOW_SIZE', '4096'))
USE_GPU = os.getenv('USE_GPU', 'False').lower() == 'true'
ENABLE_CACHING = os.getenv('ENABLE_CACHING', 'True').lower() == 'true'
PROMPT_TEMPLATE = os.getenv('PROMPT_TEMPLATE', 
   "You are a helpful voice assistant. Answer the question based on the provided context. "
   "If you don't know the answer, say so clearly. Keep responses concise and conversational.")

# Storage Settings
PERSIST_DIR = os.getenv('PERSIST_DIR', './storage')
CACHE_DIR = os.getenv('CACHE_DIR', './cache')
DB_COLLECTION_NAME = os.getenv('DB_COLLECTION_NAME', 'voice_ai_knowledge')

# Document Processing Settings
CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', '512'))
CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', '50'))
MAX_DOCUMENT_SIZE_MB = int(os.getenv('MAX_DOCUMENT_SIZE_MB', '10'))

# Retrieval Settings
DEFAULT_RETRIEVE_COUNT = int(os.getenv('DEFAULT_RETRIEVE_COUNT', '3'))
MINIMUM_RELEVANCE_SCORE = float(os.getenv('MINIMUM_RELEVANCE_SCORE', '0.6'))
RERANKING_ENABLED = os.getenv('RERANKING_ENABLED', 'False').lower() == 'true'

# Speech-to-Text Settings
WHISPER_INITIAL_PROMPT = os.getenv('WHISPER_INITIAL_PROMPT', 
   "This is a clear business phone conversation. Transcribe the exact words spoken, "
   "ignoring background noise.")
WHISPER_TEMPERATURE = float(os.getenv('WHISPER_TEMPERATURE', '0.0'))
WHISPER_NO_CONTEXT = os.getenv('WHISPER_NO_CONTEXT', 'True').lower() == 'true'
STT_PRESET = os.getenv('STT_PRESET', 'default')
SPEECH_RECOGNITION_TIMEOUT = float(os.getenv('SPEECH_RECOGNITION_TIMEOUT', '15.0'))

# Text-to-Speech Settings
TTS_VOICE = os.getenv('TTS_VOICE', 'en-US-Neural2-F')
TTS_MODEL = os.getenv('TTS_MODEL', 'aura-asteria-en')
TTS_SAMPLE_RATE = int(os.getenv('TTS_SAMPLE_RATE', '24000'))
TTS_CONTAINER_FORMAT = os.getenv('TTS_CONTAINER_FORMAT', 'wav')

# API Settings
DEEPGRAM_API_KEY = os.getenv('DEEPGRAM_API_KEY', '')

# Logging Settings
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_FILE = os.getenv('LOG_FILE', '/var/log/voice_ai_agent.log')

# Application Settings
HOST = os.getenv('HOST', '0.0.0.0')
PORT = int(os.getenv('PORT', '5000'))
DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
BASE_URL = os.getenv('BASE_URL', 'http://localhost:5000')

# Server Settings
KEEP_ALIVE_INTERVAL = int(os.getenv('KEEP_ALIVE_INTERVAL', '30'))
RECONNECT_ATTEMPTS = int(os.getenv('RECONNECT_ATTEMPTS', '3'))
REQUEST_TIMEOUT = int(os.getenv('REQUEST_TIMEOUT', '60'))

# Performance Settings
MAX_CONCURRENT_CALLS = int(os.getenv('MAX_CONCURRENT_CALLS', '10'))
MAX_CALL_DURATION = int(os.getenv('MAX_CALL_DURATION', '3600'))  # 1 hour in seconds
AUDIO_BUFFER_SIZE = int(os.getenv('AUDIO_BUFFER_SIZE', '16000'))  # 1 second at 16kHz

# Function to get logging configuration
def get_logging_config() -> dict:
   """Get logging configuration dictionary."""
   return {
       'level': getattr(logging, LOG_LEVEL),
       'format': LOG_FORMAT,
       'handlers': [
           {
               'type': 'stream',
               'stream': 'ext://sys.stdout',
               'level': LOG_LEVEL
           },
           {
               'type': 'file',
               'filename': LOG_FILE,
               'level': LOG_LEVEL,
               'maxBytes': 10485760,  # 10MB
               'backupCount': 5
           }
       ]
   }

# Function to get model configuration
def get_model_config() -> dict:
   """Get model configuration dictionary."""
   return {
       'model_name': MODEL_NAME,
       'llm_temperature': LLM_TEMPERATURE,
       'stt_model_path': STT_MODEL_PATH,
       'embedding_model': EMBEDDING_MODEL,
       'max_tokens': MAX_TOKENS,
       'use_gpu': USE_GPU,
       'whisper_initial_prompt': WHISPER_INITIAL_PROMPT,
       'whisper_temperature': WHISPER_TEMPERATURE,
       'whisper_no_context': WHISPER_NO_CONTEXT,
       'stt_preset': STT_PRESET
   }

# Function to get storage configuration
def get_storage_config() -> dict:
   """Get storage configuration dictionary."""
   return {
       'persist_dir': PERSIST_DIR,
       'cache_dir': CACHE_DIR,
       'enable_caching': ENABLE_CACHING,
       'db_collection_name': DB_COLLECTION_NAME
   }

# Function to get document processing configuration
def get_document_config() -> dict:
   """Get document processing configuration dictionary."""
   return {
       'chunk_size': CHUNK_SIZE,
       'chunk_overlap': CHUNK_OVERLAP,
       'max_document_size_mb': MAX_DOCUMENT_SIZE_MB,
       'default_retrieve_count': DEFAULT_RETRIEVE_COUNT,
       'minimum_relevance_score': MINIMUM_RELEVANCE_SCORE,
       'reranking_enabled': RERANKING_ENABLED
   }

# Function to get TTS configuration
def get_tts_config() -> dict:
   """Get text-to-speech configuration dictionary."""
   return {
       'voice': TTS_VOICE,
       'model': TTS_MODEL,
       'sample_rate': TTS_SAMPLE_RATE,
       'container_format': TTS_CONTAINER_FORMAT
   }

# Function to get server configuration
def get_server_config() -> dict:
   """Get server configuration dictionary."""
   return {
       'host': HOST,
       'port': PORT,
       'debug': DEBUG,
       'base_url': BASE_URL,
       'keep_alive_interval': KEEP_ALIVE_INTERVAL,
       'reconnect_attempts': RECONNECT_ATTEMPTS,
       'request_timeout': REQUEST_TIMEOUT,
       'max_concurrent_calls': MAX_CONCURRENT_CALLS
   }