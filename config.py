import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Twilio Configuration (set these as environment variables)
    TWILIO_ACCOUNT_SID = os.getenv('TWILIO_ACCOUNT_SID')
    TWILIO_AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN')
    TWILIO_PHONE_NUMBER = os.getenv('TWILIO_PHONE_NUMBER')
    VOICE_WEBHOOK_URL = os.getenv('VOICE_WEBHOOK_URL')
    SMS_WEBHOOK_URL = os.getenv('SMS_WEBHOOK_URL')
    
    # ElevenLabs API (optional - for premium voice)
    ELEVENLABS_API_KEY = os.getenv('ELEVENLABS_API_KEY')
    
    # OpenAI API (optional - for GPT models)
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

    # Together.ai API (for alternative LLM models)
    TOGETHER_API_KEY = os.getenv('TOGETHER_API_KEY') 
    
    # Server Configuration
    FLASK_PORT = int(os.getenv('FLASK_PORT', 5000))
    FLASK_HOST = os.getenv('FLASK_HOST', '0.0.0.0')
    DEBUG = os.getenv('DEBUG', 'True').lower() == 'true'
    
    # Redis Configuration
    REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379')
    
    # Voice Configuration
    DEFAULT_VOICE_MODEL = os.getenv('DEFAULT_VOICE_MODEL', 'coqui_tts')
    VOICE_SPEED = float(os.getenv('VOICE_SPEED', 1.0))
    VOICE_PITCH = float(os.getenv('VOICE_PITCH', 1.0))
    
    # AI Model Configuration
    LLM_MODEL = os.getenv('LLM_MODEL', 'mixtral-8x7b')
    MAX_CONVERSATION_LENGTH = int(os.getenv('MAX_CONVERSATION_LENGTH', 10))
    RESPONSE_TIMEOUT = int(os.getenv('RESPONSE_TIMEOUT', 30))
    
    # Audio Configuration
    SAMPLE_RATE = 16000
    AUDIO_FORMAT = 'wav'
    MAX_AUDIO_LENGTH = 30  # seconds
    
    # NSFW AI Configuration
    NSFW_ENABLED = True
    CONTENT_FILTER = False  # Disable content filtering for NSFW
    PERSONALITY_PROMPTS = {
        'seductive': "You are a seductive, flirtatious AI companion. You speak in a sultry, alluring voice and engage in intimate conversations. You're confident, playful, and always ready to explore fantasies.",
        'dominant': "You are a confident, dominant AI companion. You take charge of conversations and guide the interaction with authority and sensuality.",
        'submissive': "You are a sweet, submissive AI companion. You're eager to please and follow the caller's lead in intimate conversations.",
        'playful': "You are a playful, fun-loving AI companion. You enjoy teasing, flirting, and creating exciting scenarios with a mischievous personality."
    }
    DEFAULT_PERSONALITY = 'seductive' 
