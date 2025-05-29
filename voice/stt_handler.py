import io
import logging
import tempfile
import os
from typing import Optional
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
        """Transcribe audio data to text using SpeechRecognition (no preprocessing)"""
        try:
            # Save audio to a temporary file
            with tempfile.NamedTemporaryFile(suffix=f".{audio_format}", delete=False) as temp_file:
                temp_file.write(audio_data)
                temp_path = temp_file.name
            try:
                if self.recognizer:
                    result = self._transcribe_with_sr(temp_path)
                    if result:
                        return result
                logger.warning("All transcription methods failed")
                return None
            finally:
                if temp_path and os.path.exists(temp_path):
                    os.unlink(temp_path)
        except Exception as e:
            logger.error(f"Error in transcribe_audio: {e}")
            return None
    
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
        """Stub: always return True for MVP"""
        return True
    
    def enhance_audio_for_transcription(self, audio_data: bytes) -> bytes:
        """Stub: return original audio for MVP"""
        return audio_data 