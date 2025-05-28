import logging
import io
import base64
from typing import Optional, Dict, Any
from flask import Flask, request, Response
from twilio.rest import Client
from twilio.twiml.voice_response import VoiceResponse, Gather, Say
import requests
from config import Config

logger = logging.getLogger(__name__)

class TwilioPhoneHandler:
    """Handles Twilio phone system integration"""
    
    def __init__(self):
        self.config = Config()
        self.client = None
        self.active_calls = {}
        
    def initialize_twilio(self):
        """Initialize Twilio client"""
        try:
            if (self.config.TWILIO_ACCOUNT_SID == "your_twilio_account_sid" or 
                self.config.TWILIO_AUTH_TOKEN == "your_twilio_auth_token"):
                logger.warning("Twilio credentials not configured - using demo mode")
                return False
            
            self.client = Client(self.config.TWILIO_ACCOUNT_SID, self.config.TWILIO_AUTH_TOKEN)
            logger.info("Twilio client initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Twilio: {e}")
            return False
    
    def handle_incoming_call(self, call_sid: str = None) -> VoiceResponse:
        """Handle incoming phone call"""
        try:
            response = VoiceResponse()
            
            # Get call SID from request if not provided
            if not call_sid:
                call_sid = request.values.get('CallSid', 'demo_call')
            
            # Initialize call session
            self.active_calls[call_sid] = {
                'status': 'active',
                'start_time': None,
                'conversation_history': []
            }
            
            # Welcome message
            welcome_message = "Hey there, sexy... I've been waiting for your call. What's on your mind tonight?"
            
            # Create TwiML response with gather for continuous conversation
            gather = Gather(
                input='speech',
                action=f'/voice/process/{call_sid}',
                method='POST',
                speech_timeout='auto',
                language='en-US',
                enhanced=True
            )
            
            gather.say(welcome_message, voice='alice', language='en-US')
            response.append(gather)
            
            # Fallback if no input
            response.say("I didn't hear anything... are you still there, baby?", voice='alice')
            response.redirect(f'/voice/continue/{call_sid}')
            
            logger.info(f"Handled incoming call: {call_sid}")
            return response
            
        except Exception as e:
            logger.error(f"Error handling incoming call: {e}")
            response = VoiceResponse()
            response.say("I'm sorry, there seems to be a technical issue. Please call back later.")
            return response
    
    def process_speech_input(self, call_sid: str, speech_result: str = None) -> VoiceResponse:
        """Process speech input from caller"""
        try:
            response = VoiceResponse()
            
            # Get speech result from request if not provided
            if not speech_result:
                speech_result = request.values.get('SpeechResult', '')
            
            if not speech_result:
                # No speech detected
                response.say("I'm listening... tell me what you want.", voice='alice')
                response.redirect(f'/voice/continue/{call_sid}')
                return response
            
            logger.info(f"Received speech from {call_sid}: {speech_result}")
            
            # Store in call history
            if call_sid in self.active_calls:
                self.active_calls[call_sid]['conversation_history'].append({
                    'type': 'user',
                    'content': speech_result
                })
            
            # This will be replaced with AI response generation
            ai_response = self._generate_ai_response(speech_result, call_sid)
            
            # Store AI response
            if call_sid in self.active_calls:
                self.active_calls[call_sid]['conversation_history'].append({
                    'type': 'ai',
                    'content': ai_response
                })
            
            # Create response with gather for next input
            gather = Gather(
                input='speech',
                action=f'/voice/process/{call_sid}',
                method='POST',
                speech_timeout='auto',
                language='en-US',
                enhanced=True
            )
            
            gather.say(ai_response, voice='alice', language='en-US')
            response.append(gather)
            
            # Fallback
            response.say("Are you still there? I want to keep talking with you...", voice='alice')
            response.redirect(f'/voice/continue/{call_sid}')
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing speech input: {e}")
            response = VoiceResponse()
            response.say("Sorry, I didn't catch that. Can you say it again?", voice='alice')
            response.redirect(f'/voice/continue/{call_sid}')
            return response
    
    def _generate_ai_response(self, user_input: str, call_sid: str) -> str:
        """Generate AI response (placeholder - will be replaced with actual AI)"""
        # This is a simple placeholder that will be replaced with the actual AI model
        user_lower = user_input.lower()
        
        if any(word in user_lower for word in ['hello', 'hi', 'hey']):
            return "Mmm, hello there... your voice is so sexy. Tell me what you're thinking about."
        elif any(word in user_lower for word in ['how', 'what']):
            return "I'm feeling so naughty right now... what about you? What's turning you on?"
        elif any(word in user_lower for word in ['want', 'like', 'love']):
            return "That sounds so hot... I love hearing what excites you. Tell me more."
        elif any(word in user_lower for word in ['bye', 'goodbye']):
            return "Aww, do you have to go? I was having so much fun... call me again soon, okay?"
        else:
            responses = [
                "Mmm, that's so interesting... tell me more about that.",
                "You're making me so excited... I love talking with you.",
                "That sounds amazing... what else are you thinking about?",
                "Your voice is driving me wild... keep talking to me.",
                "I'm getting so turned on just listening to you..."
            ]
            import random
            return random.choice(responses)
    
    def continue_conversation(self, call_sid: str) -> VoiceResponse:
        """Continue conversation when no input received"""
        try:
            response = VoiceResponse()
            
            # Check if call is still active
            if call_sid not in self.active_calls:
                response.say("Thanks for calling... hope to hear from you again soon.", voice='alice')
                response.hangup()
                return response
            
            # Prompt for more input
            gather = Gather(
                input='speech',
                action=f'/voice/process/{call_sid}',
                method='POST',
                speech_timeout='auto',
                language='en-US',
                enhanced=True
            )
            
            prompts = [
                "I'm still here... what are you thinking about?",
                "Talk to me, baby... I want to hear your voice.",
                "Are you getting shy on me? Tell me what's on your mind.",
                "I'm waiting for you to say something sexy..."
            ]
            
            import random
            gather.say(random.choice(prompts), voice='alice')
            response.append(gather)
            
            # End call if still no response
            response.say("I guess you have to go... call me back anytime, sexy.", voice='alice')
            response.hangup()
            
            return response
            
        except Exception as e:
            logger.error(f"Error continuing conversation: {e}")
            response = VoiceResponse()
            response.say("Thanks for calling!", voice='alice')
            response.hangup()
            return response
    
    def end_call(self, call_sid: str) -> VoiceResponse:
        """End the call"""
        try:
            response = VoiceResponse()
            
            # Clean up call data
            if call_sid in self.active_calls:
                del self.active_calls[call_sid]
            
            response.say("Thanks for calling... that was so much fun. Call me again soon!", voice='alice')
            response.hangup()
            
            logger.info(f"Ended call: {call_sid}")
            return response
            
        except Exception as e:
            logger.error(f"Error ending call: {e}")
            response = VoiceResponse()
            response.hangup()
            return response
    
    def get_call_status(self, call_sid: str) -> Optional[Dict[str, Any]]:
        """Get status of a call"""
        return self.active_calls.get(call_sid)
    
    def make_outbound_call(self, to_number: str, message: str = None) -> Optional[str]:
        """Make an outbound call (for testing)"""
        try:
            if not self.client:
                logger.error("Twilio client not initialized")
                return None
            
            if not message:
                message = "Hello, this is a test call from your AI phone system."
            
            call = self.client.calls.create(
                twiml=f'<Response><Say voice="alice">{message}</Say></Response>',
                to=to_number,
                from_=self.config.TWILIO_PHONE_NUMBER
            )
            
            logger.info(f"Made outbound call: {call.sid}")
            return call.sid
            
        except Exception as e:
            logger.error(f"Error making outbound call: {e}")
            return None
    
    def play_audio_response(self, call_sid: str, audio_url: str) -> VoiceResponse:
        """Play audio response instead of text-to-speech"""
        try:
            response = VoiceResponse()
            
            # Play the audio file
            response.play(audio_url)
            
            # Continue conversation
            gather = Gather(
                input='speech',
                action=f'/voice/process/{call_sid}',
                method='POST',
                speech_timeout='auto',
                language='en-US',
                enhanced=True
            )
            
            response.append(gather)
            response.redirect(f'/voice/continue/{call_sid}')
            
            return response
            
        except Exception as e:
            logger.error(f"Error playing audio response: {e}")
            response = VoiceResponse()
            response.say("Sorry, there was an audio issue.", voice='alice')
            return response 