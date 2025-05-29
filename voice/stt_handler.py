import io
import logging
import tempfile
import os
from typing import Optional
from pydub import AudioSegment
import speech_recognition as sr
from config import Config

logger = logging.getLogger(__name__)

class SpeechToTextHandler:
    """Handles speech-to-text conversion for phone calls"""
    
    def __init__(self):
        self.config = Config()
        self.recognizer = None
        self.microphone = None
    
    def initialize_stt_engine(self):
        """Initialize the speech-to-text engine (SpeechRecognition only)"""
        try:
            self.recognizer = sr.Recognizer()
            # Optimize for phone audio
            self.recognizer.energy_threshold = 300
            self.recognizer.dynamic_energy_threshold = True
            self.recognizer.pause_threshold = 0.8
            self.recognizer.phrase_threshold = 0.3
            logger.info("Speech-to-text engine initialized successfully (SpeechRecognition only)")
        except Exception as e:
            logger.error(f"Failed to initialize STT engine: {e}")
            self.recognizer = None
    
    def transcribe_audio(self, audio_data: bytes, audio_format: str = "wav") -> Optional[str]:
        """Transcribe audio data to text using SpeechRecognition"""
        try:
            # Create temporary file for audio processing
            with tempfile.NamedTemporaryFile(suffix=f".{audio_format}", delete=False) as temp_file:
                temp_file.write(audio_data)
                temp_path = temp_file.name
            try:
                # Process audio for better transcription
                processed_audio_path = self._preprocess_audio(temp_path)
                # Use SpeechRecognition
                if self.recognizer:
                    result = self._transcribe_with_sr(processed_audio_path)
                    if result:
                        return result
                logger.warning("All transcription methods failed")
                return None
            finally:
                # Clean up temporary files
                for path in [temp_path, processed_audio_path]:
                    if path and os.path.exists(path):
                        os.unlink(path)
        except Exception as e:
            logger.error(f"Error in transcribe_audio: {e}")
            return None
    
    def _preprocess_audio(self, audio_path: str) -> str:
        """Preprocess audio for better transcription"""
        try:
            # Load audio
            audio = AudioSegment.from_file(audio_path)
            # Convert to mono if stereo
            if audio.channels > 1:
                audio = audio.set_channels(1)
            # Set sample rate to 16kHz (optional, for phone audio)
            audio = audio.set_frame_rate(16000)
            # Normalize audio levels
            audio = audio.normalize()
            # Apply noise reduction (simple high-pass filter)
            audio = audio.high_pass_filter(80)
            # Apply compression to even out volume levels
            audio = audio.compress_dynamic_range(threshold=-20.0, ratio=4.0, attack=5.0, release=50.0)
            # Create processed file
            processed_path = audio_path.replace('.', '_processed.')
            audio.export(processed_path, format="wav")
            return processed_path
        except Exception as e:
            logger.error(f"Error preprocessing audio: {e}")
            return audio_path  # Return original if preprocessing fails
    
    def _transcribe_with_sr(self, audio_path: str) -> Optional[str]:
        """Transcribe using SpeechRecognition library"""
        try:
            with sr.AudioFile(audio_path) as source:
                # Adjust for ambient noise
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                # Record the audio
                audio = self.recognizer.record(source)
            # Try Google Speech Recognition (free tier)
            try:
                text = self.recognizer.recognize_google(audio)
                if text and len(text) > 0:
                    logger.info(f"Google STT transcription: {text}")
                    return text
            except sr.UnknownValueError:
                logger.warning("Google STT could not understand audio")
            except sr.RequestError as e:
                logger.error(f"Google STT request failed: {e}")
            # Try Sphinx as offline fallback
            try:
                text = self.recognizer.recognize_sphinx(audio)
                if text and len(text) > 0:
                    logger.info(f"Sphinx transcription: {text}")
                    return text
            except sr.UnknownValueError:
                logger.warning("Sphinx could not understand audio")
            except sr.RequestError as e:
                logger.error(f"Sphinx request failed: {e}")
            return None
        except Exception as e:
            logger.error(f"SpeechRecognition transcription failed: {e}")
            return None
    
    def transcribe_stream(self, audio_stream: io.BytesIO) -> Optional[str]:
        """Transcribe audio from a stream"""
        try:
            audio_data = audio_stream.read()
            return self.transcribe_audio(audio_data)
        except Exception as e:
            logger.error(f"Error transcribing stream: {e}")
            return None
    
    def is_speech_detected(self, audio_data: bytes) -> bool:
        """Detect if audio contains speech"""
        try:
            # Simple voice activity detection
            audio = AudioSegment.from_file(io.BytesIO(audio_data))
            # Check if audio is loud enough
            if audio.dBFS < -40:  # Very quiet
                return False
            # Check duration
            if len(audio) < 500:  # Less than 0.5 seconds
                return False
            return True
        except Exception as e:
            logger.error(f"Error in speech detection: {e}")
            return True  # Assume speech if detection fails
    
    def enhance_audio_for_transcription(self, audio_data: bytes) -> bytes:
        """Enhance audio quality for better transcription"""
        try:
            audio = AudioSegment.from_file(io.BytesIO(audio_data))
            # Apply enhancements
            audio = audio.normalize()
            audio = audio.high_pass_filter(100)
            audio = audio.low_pass_filter(8000)
            # Export enhanced audio
            output_buffer = io.BytesIO()
            audio.export(output_buffer, format="wav")
            return output_buffer.getvalue()
        except Exception as e:
            logger.error(f"Error enhancing audio: {e}")
            return audio_data  # Return original if enhancement fails 