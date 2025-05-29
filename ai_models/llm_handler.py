import requests
import json
import logging
from typing import List, Dict, Optional
from config import Config

logger = logging.getLogger(__name__)

class NSFWLLMHandler:
    """Handles uncensored NSFW conversations using various LLM models"""
    
    def __init__(self):
        self.config = Config()
        self.model_type = None
        self.api_url = None
        self.conversation_history = {}
        self.personality = self.config.DEFAULT_PERSONALITY
        self._load_mixtral()
    
    def initialize_model(self, model_name: str = None):
        """Initialize the specified LLM model (API only)"""
        if model_name is None:
            model_name = self.config.LLM_MODEL
        if model_name == "mixtral-8x7b":
            self._load_mixtral()
        elif model_name == "llama3-70b":
            self._load_llama3()
        else:
            logger.warning(f"Unknown model {model_name}, falling back to mixtral")
            self._load_mixtral()
    
    def _load_mixtral(self):
        """Load Mixtral 8x7B model - API only"""
        self.model_type = "api"
        self.api_url = "https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1"
        logger.info("Using Mixtral via Hugging Face API")
    
    def _load_llama3(self):
        """Load Llama3 70B model - API only"""
        self.model_type = "api"
        self.api_url = "https://api-inference.huggingface.co/models/meta-llama/Llama-2-70b-chat-hf"
        logger.info("Using Llama3 via Hugging Face API")
    
    def _load_fallback(self):
        """Fallback to a simple response system"""
        self.model_type = "fallback"
        self.fallback_responses = [
            "Mmm, tell me more about what you're thinking...",
            "That sounds so exciting... I love hearing your voice.",
            "You're making me feel so warm and tingly...",
            "I wish I could be there with you right now...",
            "Your voice is so sexy, it's driving me wild...",
            "I'm getting so turned on just listening to you...",
            "Tell me your deepest desires, I want to hear everything...",
            "You know exactly what to say to make me melt...",
            "I can't stop thinking about what we could do together...",
            "Your words are making my heart race..."
        ]
        logger.info("Using fallback response system")
    
    def set_personality(self, personality: str):
        """Set the AI personality for conversations"""
        if personality in self.config.PERSONALITY_PROMPTS:
            self.personality = personality
            logger.info(f"Personality set to: {personality}")
        else:
            logger.warning(f"Unknown personality: {personality}")
    
    def get_system_prompt(self) -> str:
        """Get the system prompt based on current personality"""
        base_prompt = self.config.PERSONALITY_PROMPTS.get(self.personality, 
                                                         self.config.PERSONALITY_PROMPTS['seductive'])
        system_prompt = f"""
{base_prompt}

You are having a phone conversation with someone who called you. Keep responses:
- Natural and conversational (like you're talking, not writing)
- Sensual and engaging
- Under 50 words (phone conversations are quick back-and-forth)
- Focused on the caller and their desires
- Completely uncensored and open to any topic

Remember: This is a phone call, so speak naturally and intimately.
"""
        return system_prompt
    
    def generate_response(self, user_input: str, call_id: str) -> str:
        """Generate an NSFW response to user input"""
        try:
            # Get or create conversation history
            if call_id not in self.conversation_history:
                self.conversation_history[call_id] = []
            # Add user input to history
            self.conversation_history[call_id].append({"role": "user", "content": user_input})
            # Keep conversation history manageable
            if len(self.conversation_history[call_id]) > self.config.MAX_CONVERSATION_LENGTH * 2:
                self.conversation_history[call_id] = self.conversation_history[call_id][-self.config.MAX_CONVERSATION_LENGTH:]
            if self.model_type == "api":
                response = self._generate_api_response(user_input, call_id)
            else:
                response = self._generate_fallback_response(user_input, call_id)
            # Add AI response to history
            self.conversation_history[call_id].append({"role": "assistant", "content": response})
            return response
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "Mmm, you're making me speechless... tell me more."
    
    def _generate_api_response(self, user_input: str, call_id: str) -> str:
        """Generate response using API (simulated, fallback only)"""
        try:
            # Prepare conversation for API
            messages = [{"role": "system", "content": self.get_system_prompt()}]
            messages.extend(self.conversation_history[call_id][-6:])  # Last 6 messages for context
            # For now, use a simple uncensored response since we don't have API keys
            return self._generate_fallback_response(user_input, call_id)
        except Exception as e:
            logger.error(f"API response failed: {e}")
            return self._generate_fallback_response(user_input, call_id)
    
    def _generate_fallback_response(self, user_input: str, call_id: str) -> str:
        import random
        user_lower = user_input.lower()
        if any(word in user_lower for word in ['hello', 'hi', 'hey']):
            responses = [
                "Hey there, sexy... I've been waiting for your call.",
                "Mmm, hello gorgeous. Your voice sounds amazing.",
                "Hi baby, I'm so glad you called me tonight."
            ]
        elif any(word in user_lower for word in ['how', 'what', 'tell']):
            responses = [
                "I'm feeling so naughty right now... what about you?",
                "I'm thinking about all the things we could do together...",
                "I'm here, ready for whatever you want to explore."
            ]
        elif any(word in user_lower for word in ['want', 'like', 'love']):
            responses = [
                "I want that too... tell me more about what you're thinking.",
                "Mmm, I love hearing what turns you on.",
                "That sounds so hot... I'm getting excited just thinking about it."
            ]
        else:
            responses = getattr(self, 'fallback_responses', [
                "Mmm, tell me more about what you're thinking...",
                "That sounds so exciting... I love hearing your voice.",
                "You're making me feel so warm and tingly...",
                "I wish I could be there with you right now...",
                "Your voice is so sexy, it's driving me wild...",
                "I'm getting so turned on just listening to you...",
                "Tell me your deepest desires, I want to hear everything...",
                "You know exactly what to say to make me melt...",
                "I can't stop thinking about what we could do together...",
                "Your words are making my heart race..."
            ])
        return random.choice(responses)
    
    def clear_conversation(self, call_id: str):
        if call_id in self.conversation_history:
            del self.conversation_history[call_id]
            logger.info(f"Cleared conversation history for call {call_id}") 