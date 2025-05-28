import os
import io
import logging
import tempfile
from typing import Optional, Dict, Any
import torch
import torchaudio
from TTS.api import TTS
import requests
from pydub import AudioSegment
from config import Config

logger = logging.getLogger(__name__)

class VoiceHandler:
    """Handles text-to-speech conversion with multiple voice options"""
    
    def __init__(self):
        self.config = Config()
        self.tts_engine = None
        self.voice_models = {}
        self.current_voice = "seductive_female"
        
    def initialize_voice_engine(self):
        """Initialize the TTS engine"""
        try:
            # Initialize Coqui TTS for high-quality voices
            if torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
                
            # Load a high-quality female voice model
            model_name = "tts_models/en/ljspeech/tacotron2-DDC"
            self.tts_engine = TTS(model_name=model_name, progress_bar=False).to(device)
            
            # Load additional voice models
            self._load_voice_models()
            
            logger.info("Voice engine initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize voice engine: {e}")
            self._setup_fallback_tts()
    
    def _load_voice_models(self):
        """Load different voice models for variety"""
        try:
            # Define available voice models
            self.voice_models = {
                "seductive_female": {
                    "model": "tts_models/en/ljspeech/tacotron2-DDC",
                    "speed": 0.9,
                    "pitch": 1.1
                },
                "sultry_female": {
                    "model": "tts_models/en/ljspeech/glow-tts",
                    "speed": 0.8,
                    "pitch": 1.0
                },
                "playful_female": {
                    "model": "tts_models/en/ljspeech/tacotron2-DDC",
                    "speed": 1.1,
                    "pitch": 1.2
                },
                "dominant_female": {
                    "model": "tts_models/en/ljspeech/tacotron2-DDC",
                    "speed": 0.95,
                    "pitch": 0.9
                }
            }
            
            logger.info(f"Loaded {len(self.voice_models)} voice models")
            
        except Exception as e:
            logger.error(f"Error loading voice models: {e}")
    
    def _setup_fallback_tts(self):
        """Setup fallback TTS if main engine fails"""
        try:
            # Use a simpler TTS model as fallback
            self.tts_engine = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False)
            logger.info("Fallback TTS engine initialized")
        except Exception as e:
            logger.error(f"Fallback TTS also failed: {e}")
            self.tts_engine = None
    
    def set_voice(self, voice_name: str):
        """Set the current voice"""
        if voice_name in self.voice_models:
            self.current_voice = voice_name
            logger.info(f"Voice set to: {voice_name}")
        else:
            logger.warning(f"Unknown voice: {voice_name}")
    
    def text_to_speech(self, text: str, output_format: str = "wav") -> Optional[bytes]:
        """Convert text to speech and return audio bytes"""
        try:
            if not self.tts_engine:
                logger.error("TTS engine not initialized")
                return None
            
            # Get voice settings
            voice_config = self.voice_models.get(self.current_voice, self.voice_models["seductive_female"])
            
            # Create temporary file for audio output
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_path = temp_file.name
            
            try:
                # Generate speech
                self.tts_engine.tts_to_file(
                    text=text,
                    file_path=temp_path
                )
                
                # Load and process audio
                audio = AudioSegment.from_wav(temp_path)
                
                # Apply voice modifications
                speed = voice_config.get("speed", 1.0)
                pitch = voice_config.get("pitch", 1.0)
                
                if speed != 1.0:
                    # Change speed
                    audio = audio.speedup(playback_speed=speed)
                
                if pitch != 1.0:
                    # Change pitch (simple implementation)
                    new_sample_rate = int(audio.frame_rate * pitch)
                    audio = audio._spawn(audio.raw_data, overrides={"frame_rate": new_sample_rate})
                    audio = audio.set_frame_rate(self.config.SAMPLE_RATE)
                
                # Convert to desired format
                if output_format.lower() == "mp3":
                    output_buffer = io.BytesIO()
                    audio.export(output_buffer, format="mp3", bitrate="128k")
                    audio_bytes = output_buffer.getvalue()
                else:
                    # Default to WAV
                    output_buffer = io.BytesIO()
                    audio.export(output_buffer, format="wav")
                    audio_bytes = output_buffer.getvalue()
                
                return audio_bytes
                
            finally:
                # Clean up temporary file
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                    
        except Exception as e:
            logger.error(f"Error in text_to_speech: {e}")
            return None
    
    def text_to_speech_stream(self, text: str) -> Optional[io.BytesIO]:
        """Convert text to speech and return as stream for real-time playback"""
        try:
            audio_bytes = self.text_to_speech(text, "wav")
            if audio_bytes:
                return io.BytesIO(audio_bytes)
            return None
        except Exception as e:
            logger.error(f"Error in text_to_speech_stream: {e}")
            return None
    
    def get_available_voices(self) -> Dict[str, Dict[str, Any]]:
        """Get list of available voices"""
        return self.voice_models
    
    def enhance_voice_for_nsfw(self, text: str) -> str:
        """Enhance text for more sensual speech delivery"""
        try:
            # Add pauses and emphasis for more sensual delivery
            enhanced_text = text
            
            # Add slight pauses after certain words for effect
            sensual_words = ["mmm", "oh", "yes", "baby", "sexy", "hot", "want", "need"]
            for word in sensual_words:
                enhanced_text = enhanced_text.replace(word, f"{word}...")
            
            # Add breathing sounds occasionally
            if len(text) > 50:
                enhanced_text = enhanced_text.replace(". ", "... *soft breath* ")
            
            return enhanced_text
            
        except Exception as e:
            logger.error(f"Error enhancing voice: {e}")
            return text

class ElevenLabsVoiceHandler:
    """Handler for ElevenLabs premium voices (if API key available)"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or Config().ELEVENLABS_API_KEY
        self.base_url = "https://api.elevenlabs.io/v1"
        self.voice_id = "21m00Tcm4TlvDq8ikWAM"  # Default female voice
        
    def is_available(self) -> bool:
        """Check if ElevenLabs API is available"""
        return bool(self.api_key and self.api_key != "")
    
    def text_to_speech(self, text: str) -> Optional[bytes]:
        """Convert text to speech using ElevenLabs"""
        if not self.is_available():
            return None
            
        try:
            url = f"{self.base_url}/text-to-speech/{self.voice_id}"
            headers = {
                "Accept": "audio/mpeg",
                "Content-Type": "application/json",
                "xi-api-key": self.api_key
            }
            
            data = {
                "text": text,
                "model_id": "eleven_monolingual_v1",
                "voice_settings": {
                    "stability": 0.5,
                    "similarity_boost": 0.5,
                    "style": 0.8,
                    "use_speaker_boost": True
                }
            }
            
            response = requests.post(url, json=data, headers=headers)
            
            if response.status_code == 200:
                return response.content
            else:
                logger.error(f"ElevenLabs API error: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"ElevenLabs TTS error: {e}")
            return None 