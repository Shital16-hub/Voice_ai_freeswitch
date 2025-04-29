"""
Audio utilities for FreeSWITCH integration.
"""
import os
import io
import wave
import numpy as np
import logging
from typing import Tuple, Optional, Dict, Any, Union

logger = logging.getLogger(__name__)

def create_wav_file(
    audio_data: Union[bytes, np.ndarray],
    file_path: str,
    sample_rate: int = 8000,
    channels: int = 1
) -> str:
    """
    Create a WAV file from audio data.
    
    Args:
        audio_data: Audio data as bytes or numpy array
        file_path: Output file path
        sample_rate: Sample rate in Hz
        channels: Number of audio channels
        
    Returns:
        Path to created WAV file
    """
    try:
        # Convert to bytes if needed
        if isinstance(audio_data, np.ndarray):
            # Convert float32 to int16 if needed
            if audio_data.dtype == np.float32:
                audio_data = (audio_data * 32767).astype(np.int16)
            audio_bytes = audio_data.tobytes()
        else:
            audio_bytes = audio_data
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        
        # Create WAV file
        with wave.open(file_path, 'wb') as wav_file:
            wav_file.setnchannels(channels)
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_bytes)
        
        logger.debug(f"Created WAV file: {file_path} ({len(audio_bytes)} bytes)")
        return file_path
    
    except Exception as e:
        logger.error(f"Error creating WAV file: {e}")
        raise

def read_wav_file(file_path: str) -> Tuple[np.ndarray, int]:
    """
    Read audio data from a WAV file.
    
    Args:
        file_path: Path to WAV file
        
    Returns:
        Tuple of (audio_data, sample_rate)
    """
    try:
        with wave.open(file_path, 'rb') as wav_file:
            # Get file properties
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            sample_rate = wav_file.getframerate()
            frames = wav_file.readframes(wav_file.getnframes())
            
            # Convert to numpy array
            if sample_width == 2:  # 16-bit
                audio_data = np.frombuffer(frames, dtype=np.int16)
            elif sample_width == 1:  # 8-bit
                audio_data = np.frombuffer(frames, dtype=np.uint8)
                audio_data = (audio_data.astype(np.int16) - 128) * 256
            elif sample_width == 4:  # 32-bit
                audio_data = np.frombuffer(frames, dtype=np.int32)
                audio_data = audio_data.astype(np.int16)
            else:
                raise ValueError(f"Unsupported sample width: {sample_width}")
            
            # Convert to float32
            audio_float = audio_data.astype(np.float32) / 32768.0
            
            logger.debug(f"Read WAV file: {file_path} ({len(frames)} bytes, {sample_rate}Hz)")
            return audio_float, sample_rate
    
    except Exception as e:
        logger.error(f"Error reading WAV file: {e}")
        raise

def convert_audio_format(
    audio_data: bytes,
    from_format: str,
    to_format: str,
    sample_rate_in: int = 8000,
    sample_rate_out: int = 16000,
    channels_in: int = 1,
    channels_out: int = 1
) -> bytes:
    """
    Convert audio between different formats.
    
    Args:
        audio_data: Audio data as bytes
        from_format: Input format ('wav', 'raw', etc.)
        to_format: Output format ('wav', 'raw', etc.)
        sample_rate_in: Input sample rate
        sample_rate_out: Output sample rate
        channels_in: Input channels
        channels_out: Output channels
        
    Returns:
        Converted audio data as bytes
    """
    try:
        # Convert input to numpy array
        if from_format == 'wav':
            # Extract WAV data
            with io.BytesIO(audio_data) as wav_buffer:
                with wave.open(wav_buffer, 'rb') as wav_file:
                    frames = wav_file.readframes(wav_file.getnframes())
                    audio_array = np.frombuffer(frames, dtype=np.int16)
                    audio_float = audio_array.astype(np.float32) / 32768.0
                    actual_sample_rate = wav_file.getframerate()
                    actual_channels = wav_file.getnchannels()
        elif from_format == 'raw':
            # Assuming 16-bit PCM
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            audio_float = audio_array.astype(np.float32) / 32768.0
            actual_sample_rate = sample_rate_in
            actual_channels = channels_in
        else:
            raise ValueError(f"Unsupported input format: {from_format}")
        
        # Resample if needed
        if actual_sample_rate != sample_rate_out:
            from scipy import signal
            audio_float = signal.resample(
                audio_float,
                int(len(audio_float) * sample_rate_out / actual_sample_rate)
            )
        
        # Convert channels if needed (only mono/stereo supported)
        if actual_channels != channels_out:
            if actual_channels == 1 and channels_out == 2:
                # Mono to stereo
                audio_float = np.column_stack((audio_float, audio_float))
            elif actual_channels == 2 and channels_out == 1:
                # Stereo to mono (average channels)
                audio_float = np.mean(audio_float.reshape(-1, 2), axis=1)
        
        # Convert to output format
        if to_format == 'wav':
            # Create WAV file in memory
            with io.BytesIO() as wav_buffer:
                with wave.open(wav_buffer, 'wb') as wav_file:
                    wav_file.setnchannels(channels_out)
                    wav_file.setsampwidth(2)  # 16-bit
                    wav_file.setframerate(sample_rate_out)
                    audio_int16 = (audio_float * 32767).astype(np.int16)
                    wav_file.writeframes(audio_int16.tobytes())
                return wav_buffer.getvalue()
        elif to_format == 'raw':
            # Convert to 16-bit PCM
            audio_int16 = (audio_float * 32767).astype(np.int16)
            return audio_int16.tobytes()
        else:
            raise ValueError(f"Unsupported output format: {to_format}")
    
    except Exception as e:
        logger.error(f"Error converting audio format: {e}")
        raise

def create_silent_audio(
    duration_ms: int,
    sample_rate: int = 8000,
    channels: int = 1
) -> bytes:
    """
    Create silent audio data.
    
    Args:
        duration_ms: Duration in milliseconds
        sample_rate: Sample rate in Hz
        channels: Number of channels
        
    Returns:
        Silent audio data as WAV bytes
    """
    try:
        # Calculate number of samples
        num_samples = int(sample_rate * duration_ms / 1000)
        
        # Create silent audio (all zeros)
        silent_audio = np.zeros(num_samples, dtype=np.int16)
        
        # Create WAV file in memory
        with io.BytesIO() as wav_buffer:
            with wave.open(wav_buffer, 'wb') as wav_file:
                wav_file.setnchannels(channels)
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(silent_audio.tobytes())
            return wav_buffer.getvalue()
    
    except Exception as e:
        logger.error(f"Error creating silent audio: {e}")
        raise

def detect_speech_activity(
    audio_data: Union[bytes, np.ndarray],
    threshold: float = 0.01,
    window_size: int = 160  # 10ms at 16kHz
) -> bool:
    """
    Detect if audio contains speech.
    
    Args:
        audio_data: Audio data as bytes or numpy array
        threshold: Energy threshold
        window_size: Window size in samples
        
    Returns:
        True if speech is detected
    """
    try:
        # Convert to numpy array if needed
        if isinstance(audio_data, bytes):
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            audio_array = audio_array.astype(np.float32) / 32768.0
        else:
            audio_array = audio_data
        
        # Calculate energy in windows
        num_windows = len(audio_array) // window_size
        if num_windows == 0:
            return False
        
        windows = np.array_split(audio_array[:num_windows * window_size], num_windows)
        energies = [np.mean(np.square(window)) for window in windows]
        
        # Consider it speech if any window has energy above threshold
        has_speech = any(energy > threshold for energy in energies)
        
        return has_speech
    
    except Exception as e:
        logger.error(f"Error detecting speech activity: {e}")
        return False