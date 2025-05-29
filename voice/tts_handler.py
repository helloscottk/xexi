import os
import io
import logging
import requests
from typing import Optional
from config import Config

logger = logging.getLogger(__name__)

class ElevenLabsVoiceHandler:
    """Handler for ElevenLabs premium voices"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or Config().ELEVENLABS_API_KEY
        self.base_url = "https://api.elevenlabs.io/v1"
        self.voice_id = "21m00Tcm4TlvDq8ikWAM"  # Default female voice
        
    def is_available(self) -> bool:
        """Check if ElevenLabs API key is available"""
        return bool(self.api_key)
        
    def text_to_speech(self, text: str) -> Optional[bytes]:
        """Convert text to speech using ElevenLabs API"""
        if not self.is_available():
            logger.warning("ElevenLabs API key not available")
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
                    "similarity_boost": 0.75
                }
            }
            
            response = requests.post(url, json=data, headers=headers)
            response.raise_for_status()
            
            return response.content
            
        except Exception as e:
            logger.error(f"Error in ElevenLabs text_to_speech: {e}")
            return None 