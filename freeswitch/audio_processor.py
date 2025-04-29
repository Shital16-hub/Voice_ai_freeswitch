"""
Enhanced audio processing utilities for FreeSWITCH integration.
"""
import numpy as np
import logging
from typing import Tuple, Dict, Any, Union
from scipy import signal
import io
import wave

from config.freeswitch_config import (
    SAMPLE_RATE_FS, SAMPLE_RATE_AI,
    AUDIO_CHANNELS, SILENCE_THRESHOLD
)

logger = logging.getLogger(__name__)

class AudioProcessor:
    """
    Handles audio conversion between FreeSWITCH and Voice AI formats.
    """
    
    def convert_to_pipeline_format(
        self,
        audio_data: Union[bytes, np.ndarray],
        sample_rate_in: int = SAMPLE_RATE_FS,
        sample_rate_out: int = SAMPLE_RATE_AI,
        enhance_audio: bool = True
    ) -> bytes:
        """
        Convert FreeSWITCH audio to pipeline format.
        
        Args:
            audio_data: Audio data from FreeSWITCH
            sample_rate_in: Input sample rate
            sample_rate_out: Output sample rate
            enhance_audio: Whether to apply audio enhancement
            
        Returns:
            Audio data as bytes in pipeline format
        """
        try:
            # Convert to numpy array if needed
            if isinstance(audio_data, bytes):
                # Assuming 16-bit PCM
                audio_array = np.frombuffer(audio_data, dtype=np.int16)
                audio_array = audio_array.astype(np.float32) / 32768.0
            else:
                audio_array = audio_data
            
            # Apply enhancement if requested
            if enhance_audio:
                audio_array = self.enhance_audio(audio_array)
            
            # Resample if needed
            if sample_rate_in != sample_rate_out:
                audio_array = self._resample(
                    audio_array,
                    sample_rate_in,
                    sample_rate_out
                )
            
            # Convert to bytes (16-bit PCM)
            audio_int16 = (audio_array * 32767).astype(np.int16)
            return audio_int16.tobytes()
            
        except Exception as e:
            logger.error(f"Error converting to pipeline format: {e}")
            # Return empty array instead of raising exception
            return b''
    
    def convert_to_freeswitch_format(
        self,
        audio_data: Union[bytes, np.ndarray],
        sample_rate_in: int = SAMPLE_RATE_AI,
        sample_rate_out: int = SAMPLE_RATE_FS
    ) -> bytes:
        """
        Convert pipeline audio to FreeSWITCH format.
        
        Args:
            audio_data: Audio data from pipeline
            sample_rate_in: Input sample rate
            sample_rate_out: Output sample rate
            
        Returns:
            Audio data as bytes in FreeSWITCH format
        """
        try:
            # Convert to numpy array if needed
            if isinstance(audio_data, bytes):
                # Assuming 16-bit PCM
                audio_array = np.frombuffer(audio_data, dtype=np.int16)
                audio_array = audio_array.astype(np.float32) / 32768.0
            else:
                audio_array = audio_data
            
            # Ensure audio is in [-1, 1] range
            audio_array = np.clip(audio_array, -1.0, 1.0)
            
            # Resample if needed
            if sample_rate_in != sample_rate_out:
                audio_array = self._resample(
                    audio_array,
                    sample_rate_in,
                    sample_rate_out
                )
            
            # Convert to bytes (16-bit PCM)
            audio_int16 = (audio_array * 32767).astype(np.int16)
            return audio_int16.tobytes()
            
        except Exception as e:
            logger.error(f"Error converting to FreeSWITCH format: {e}")
            # Return empty array instead of raising exception
            return b''
    
    def _resample(
        self,
        audio_array: np.ndarray,
        orig_sr: int,
        target_sr: int
    ) -> np.ndarray:
        """
        Resample audio array to new sample rate.
        
        Args:
            audio_array: Audio data as numpy array
            orig_sr: Original sample rate
            target_sr: Target sample rate
            
        Returns:
            Resampled audio array
        """
        try:
            # Use scipy's resample function
            return signal.resample(
                audio_array,
                int(len(audio_array) * target_sr / orig_sr)
            )
        except Exception as e:
            logger.error(f"Error resampling audio: {e}")
            return audio_array
    
    def enhance_audio(self, audio_array: np.ndarray) -> np.ndarray:
        """
        Enhance audio quality for better speech recognition.
        
        Args:
            audio_array: Audio data as numpy array
            
        Returns:
            Enhanced audio array
        """
        try:
            # Apply high-pass filter to remove low-frequency noise
            b, a = signal.butter(4, 80/(SAMPLE_RATE_AI/2), 'highpass')
            filtered_audio = signal.filtfilt(b, a, audio_array)
            
            # Apply band-pass filter for telephony frequency range (300-3400 Hz)
            b, a = signal.butter(4, [300/(SAMPLE_RATE_AI/2), 3400/(SAMPLE_RATE_AI/2)], 'band')
            filtered_audio = signal.filtfilt(b, a, filtered_audio)
            
            # Apply pre-emphasis filter to boost high frequencies
            pre_emphasis = 0.97
            emphasized_audio = np.append(
                filtered_audio[0],
                filtered_audio[1:] - pre_emphasis * filtered_audio[:-1]
            )
            
            # Apply noise reduction (simple noise gate)
            noise_gate = np.where(
                np.abs(emphasized_audio) < SILENCE_THRESHOLD,
                0,
                emphasized_audio
            )
            
            # Normalize audio level
            max_val = np.max(np.abs(noise_gate))
            if max_val > 0:
                normalized_audio = noise_gate * (0.9 / max_val)
            else:
                normalized_audio = noise_gate
                
            return normalized_audio
            
        except Exception as e:
            logger.error(f"Error enhancing audio: {e}")
            return audio_array
    
    def is_silence(
        self,
        audio_data: Union[bytes, np.ndarray],
        threshold: float = SILENCE_THRESHOLD
    ) -> bool:
        """
        Enhanced silence detection with frequency analysis.
        
        Args:
            audio_data: Audio data
            threshold: Silence threshold
            
        Returns:
            True if audio is considered silence
        """
        try:
            # Convert to numpy array if needed
            if isinstance(audio_data, bytes):
                # Assuming 16-bit PCM
                audio_array = np.frombuffer(audio_data, dtype=np.int16)
                audio_array = audio_array.astype(np.float32) / 32768.0
            else:
                audio_array = audio_data
                
            # Empty array is silence
            if len(audio_array) == 0:
                return True
                
            # Calculate RMS energy
            energy = np.sqrt(np.mean(np.square(audio_array)))
            
            # Calculate zero-crossing rate
            zero_crossings = np.sum(np.abs(np.diff(np.signbit(audio_array)))) / len(audio_array)
            
            # Combined silence detection
            is_silence = energy < threshold and zero_crossings < 0.1
            
            return is_silence
            
        except Exception as e:
            logger.error(f"Error in silence detection: {e}")
            return True
    
    def create_wav_file(
        self,
        audio_data: Union[bytes, np.ndarray],
        output_path: str,
        sample_rate: int = SAMPLE_RATE_FS,
        channels: int = AUDIO_CHANNELS
    ) -> bool:
        """
        Create a WAV file from audio data.
        
        Args:
            audio_data: Audio data
            output_path: Output file path
            sample_rate: Sample rate
            channels: Number of channels
            
        Returns:
            True if successful
        """
        try:
            # Convert to numpy array if needed
            if isinstance(audio_data, bytes):
                # Check if it's already a WAV file
                if audio_data[:4] == b'RIFF' and audio_data[8:12] == b'WAVE':
                    # Already a WAV file, just write it
                    with open(output_path, 'wb') as f:
                        f.write(audio_data)
                    return True
                    
                # Assuming 16-bit PCM
                audio_array = np.frombuffer(audio_data, dtype=np.int16)
            elif isinstance(audio_data, np.ndarray):
                # Convert float32 to int16 if needed
                if audio_data.dtype == np.float32:
                    audio_array = (audio_data * 32767).astype(np.int16)
                else:
                    audio_array = audio_data
            else:
                logger.error(f"Unsupported audio data type: {type(audio_data)}")
                return False
            
            # Create WAV file
            with wave.open(output_path, 'wb') as wf:
                wf.setnchannels(channels)
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(sample_rate)
                wf.writeframes(audio_array.tobytes())
                
            logger.debug(f"Created WAV file: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating WAV file: {e}")
            return False