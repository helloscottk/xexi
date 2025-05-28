import io
import logging
import tempfile
import os
from typing import Optional
import whisper
import torch
import torchaudio
from pydub import AudioSegment
import speech_recognition as sr
from config import Config

logger = logging.getLogger(__name__)

class SpeechToTextHandler:
    """Handles speech-to-text conversion for phone calls"""
    
    def __init__(self):
        self.config = Config()
        self.whisper_model = None
        self.recognizer = None
        self.microphone = None
        
    def initialize_stt_engine(self):
        """Initialize the speech-to-text engine"""
        try:
            # Initialize Whisper for high-quality transcription
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # Load Whisper model (base is good balance of speed/accuracy)
            self.whisper_model = whisper.load_model("base", device=device)
            
            # Also initialize SpeechRecognition as fallback
            self.recognizer = sr.Recognizer()
            
            # Optimize for phone audio
            self.recognizer.energy_threshold = 300
            self.recognizer.dynamic_energy_threshold = True
            self.recognizer.pause_threshold = 0.8
            self.recognizer.phrase_threshold = 0.3
            
            logger.info("Speech-to-text engine initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize STT engine: {e}")
            self._setup_fallback_stt()
    
    def _setup_fallback_stt(self):
        """Setup fallback STT if main engine fails"""
        try:
            self.recognizer = sr.Recognizer()
            logger.info("Fallback STT engine initialized")
        except Exception as e:
            logger.error(f"Fallback STT also failed: {e}")
            self.recognizer = None
    
    def transcribe_audio(self, audio_data: bytes, audio_format: str = "wav") -> Optional[str]:
        """Transcribe audio data to text"""
        try:
            # Create temporary file for audio processing
            with tempfile.NamedTemporaryFile(suffix=f".{audio_format}", delete=False) as temp_file:
                temp_file.write(audio_data)
                temp_path = temp_file.name
            
            try:
                # Process audio for better transcription
                processed_audio_path = self._preprocess_audio(temp_path)
                
                # Try Whisper first (more accurate)
                if self.whisper_model:
                    result = self._transcribe_with_whisper(processed_audio_path)
                    if result:
                        return result
                
                # Fallback to SpeechRecognition
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
            
            # Set sample rate to 16kHz (optimal for Whisper)
            audio = audio.set_frame_rate(16000)
            
            # Normalize audio levels
            audio = audio.normalize()
            
            # Apply noise reduction (simple high-pass filter)
            # Remove frequencies below 80Hz (typical phone noise)
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
    
    def _transcribe_with_whisper(self, audio_path: str) -> Optional[str]:
        """Transcribe using Whisper model"""
        try:
            # Transcribe with Whisper
            result = self.whisper_model.transcribe(
                audio_path,
                language="en",
                task="transcribe",
                fp16=torch.cuda.is_available()
            )
            
            text = result["text"].strip()
            
            if text and len(text) > 0:
                logger.info(f"Whisper transcription: {text}")
                return text
            
            return None
            
        except Exception as e:
            logger.error(f"Whisper transcription failed: {e}")
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
            
            # Simple frequency analysis
            # Speech typically has energy in 300-3400 Hz range
            # This is a simplified check
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