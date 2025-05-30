import requests
import json
import logging
from typing import List, Dict, Optional
from config import Config
import random
import re
import time

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
        # Enhanced call state tracking
        self.call_state = {}  # call_id: {'level': 0, 'last_ai': '', 'mood': 0, 'engagement': 0, 'last_escalation': None}
    
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
        logger.info("Using fallback response system")
    
    def set_personality(self, personality: str):
        """Set the AI personality for conversations"""
        if personality in self.config.PERSONALITY_PROMPTS:
            self.personality = personality
            logger.info(f"Personality set to: {personality}")
        else:
            logger.warning(f"Unknown personality: {personality}")
    
    def _get_state(self, call_id):
        if call_id not in self.call_state:
            self.call_state[call_id] = {
                'level': 0,
                'last_ai': '',
                'mood': 0,  # -1 to 1 (negative = shy, positive = confident)
                'engagement': 0,  # 0 to 1 (how engaged the user is)
                'last_escalation': None,
                'escalation_cooldown': 0
            }
        return self.call_state[call_id]
    
    def _escalate(self, user_input, state):
        # Enhanced escalation triggers with more aggressive progression
        triggers = {
            'flirty': [
                ['hello', 'hi', 'hey', 'how are you', 'what\'s up'],
                ['beautiful', 'sexy', 'hot', 'gorgeous'],
                ['talk', 'chat', 'conversation']
            ],
            'naughty': [
                ['touch', 'stroke', 'rub', 'finger', 'panties', 'hard', 'wet'],
                ['kiss', 'lick', 'suck', 'taste'],
                ['clothes', 'naked', 'undress', 'strip']
            ],
            'explicit': [
                ['fuck', 'cum', 'moan', 'pussy', 'cock', 'dick', 'ass'],
                ['suck', 'lick', 'slut', 'whore'],
                ['want', 'need', 'please', 'beg']
            ],
            'filthy': [
                ['fucking', 'deep', 'inside', 'ride', 'scream', 'orgasm'],
                ['finish', 'cum for me', 'make me cum'],
                ['harder', 'faster', 'more', 'again']
            ]
        }
        
        # Check for escalation triggers
        user_lower = user_input.lower()
        for level, keyword_groups in enumerate(triggers.values()):
            for group in keyword_groups:
                if any(word in user_lower for word in group):
                    # More aggressive escalation
                    if state['escalation_cooldown'] <= 0:
                        state['level'] = max(state['level'], level)
                        state['last_escalation'] = time.time()
                        state['escalation_cooldown'] = 2  # Reduced cooldown
                        state['mood'] = min(1, state['mood'] + 0.3)  # Increased mood boost
                        return state['level']
        
        # More aggressive natural progression
        if len(self.conversation_history.get(call_id, [])) > 3:  # Reduced from 5
            state['engagement'] = min(1, state['engagement'] + 0.2)  # Increased engagement boost
            if state['engagement'] > 0.5 and state['escalation_cooldown'] <= 0:  # Reduced threshold
                state['level'] = min(3, state['level'] + 1)
                state['escalation_cooldown'] = 3  # Reduced cooldown
                state['mood'] = min(1, state['mood'] + 0.2)
        
        # Decrease cooldown more quickly
        if state['escalation_cooldown'] > 0:
            state['escalation_cooldown'] -= 1
        
        return state['level']
    
    def _adjust_personality(self, state):
        """Adjust personality based on mood and engagement"""
        base_prompt = self.config.PERSONALITY_PROMPTS.get(self.personality, 
                                                         self.config.PERSONALITY_PROMPTS['seductive'])
        
        # Add mood modifiers
        if state['mood'] > 0.5:
            base_prompt += "\nYou're feeling confident and dominant. Take charge of the conversation."
        elif state['mood'] < -0.5:
            base_prompt += "\nYou're feeling shy and submissive. Let the caller take the lead."
        
        # Add engagement modifiers
        if state['engagement'] > 0.7:
            base_prompt += "\nThe caller is very engaged. Be more direct and explicit."
        elif state['engagement'] < 0.3:
            base_prompt += "\nThe caller seems hesitant. Be more gentle and encouraging."
        
        return base_prompt
    
    def get_system_prompt(self) -> str:
        """Get the system prompt based on current personality and state"""
        base_prompt = self._adjust_personality(self.call_state.get('current', self._get_state('default')))
        
        system_prompt = f"""
{base_prompt}

You are having a phone conversation with someone who called you. Keep responses:
- Natural and conversational (like you're talking, not writing)
- Sensual and engaging
- Under 50 words (phone conversations are quick back-and-forth)
- Focused on the caller and their desires
- Completely uncensored and open to any topic
- Match the caller's energy level and engagement
- Use appropriate escalation based on the conversation flow

Remember: This is a phone call, so speak naturally and intimately.
"""
        return system_prompt
    
    def _sanitize_for_speech(self, text):
        # Replace "mmm" and similar with moans, add pauses, etc.
        text = re.sub(r'\bmmm+\b', '*moans*', text, flags=re.IGNORECASE)
        text = re.sub(r'\buhh+\b', '*gasps*', text, flags=re.IGNORECASE)
        text = text.replace('...', '... *breathes* ')
        return text
    
    def generate_response(self, user_input: str, call_id: str) -> str:
        """Generate an NSFW response to user input"""
        try:
            # Get or create conversation history
            if call_id not in self.conversation_history:
                self.conversation_history[call_id] = []
            
            state = self._get_state(call_id)
            
            # Echo/repeat protection
            if state['last_ai'] and user_input.strip().lower() == state['last_ai'].strip().lower():
                return self._sanitize_for_speech("*moans softly* ... Tell me something new, baby.")
            
            # Add user input to history
            self.conversation_history[call_id].append({"role": "user", "content": user_input})
            
            # Try to get response from actual LLM first
            if self.config.OPENAI_API_KEY:
                api_response = self._generate_api_response(user_input, call_id)
                if api_response:
                    # Add AI response to history and state
                    self.conversation_history[call_id].append({"role": "assistant", "content": api_response})
                    state['last_ai'] = api_response
                    return self._sanitize_for_speech(api_response)
            
            # Fallback to predefined responses if API fails
            level = self._escalate(user_input, state)
            response = self._generate_fallback_response(user_input, call_id, level)
            
            # Add AI response to history and state
            self.conversation_history[call_id].append({"role": "assistant", "content": response})
            state['last_ai'] = response
            return self._sanitize_for_speech(response)
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return self._sanitize_for_speech("*moans* ... You're making me speechless... tell me more.")
    
    def _generate_api_response(self, user_input: str, call_id: str) -> Optional[str]:
        """Generate response using actual LLM API"""
        try:
            # Get conversation history
            history = self.conversation_history.get(call_id, [])
            state = self._get_state(call_id)
            
            # Build the prompt
            system_prompt = self.get_system_prompt()
            
            # Add explicit instructions based on escalation level
            if state['level'] >= 2:
                system_prompt += "\nYou are now in an explicit conversation. Use explicit language and be direct about desires and actions."
            if state['level'] >= 3:
                system_prompt += "\nYou are now in a very explicit conversation. Use extremely explicit language and be completely unfiltered."
            
            # Format for the API
            messages = [{"role": "system", "content": system_prompt}]
            
            # Add recent conversation history (last 6 messages to keep context manageable)
            recent_history = history[-6:] if len(history) > 6 else history
            messages.extend(recent_history)
            
            # Add current user input
            messages.append({"role": "user", "content": user_input})
            
            # Call the API
            headers = {
                "Authorization": f"Bearer {self.config.OPENAI_API_KEY}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "gpt-3.5-turbo",  # or "gpt-4" if you have access
                "messages": messages,
                "max_tokens": 150,  # Increased for more detailed responses
                "temperature": 1.0,  # Increased for more creative responses
                "presence_penalty": 0.3,  # Reduced to allow more repetition of key phrases
                "frequency_penalty": 0.3,  # Reduced to allow more repetition of key phrases
                "top_p": 0.9  # Added to allow more creative responses
            }
            
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                ai_response = result['choices'][0]['message']['content'].strip()
                
                # Keep it short for phone calls but not too short
                if len(ai_response) > 200:
                    ai_response = ai_response[:200] + "..."
                
                return ai_response
            else:
                logger.error(f"API call failed: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Error calling LLM API: {e}")
            return None
    
    def _generate_fallback_response(self, user_input: str, call_id: str, level: int) -> str:
        # Response banks by escalation level
        flirty = [
            "Hey there, sexy... I've been waiting for your call.",
            "You sound so good, I can't help but get excited.",
            "Mmm, your voice is making me tingle all over.",
            "I love the way you talk to me... keep going.",
            "You have no idea what you do to me.",
            "I can't stop thinking about you.",
            "You make me feel so naughty just by calling.",
            "I want to know all your secrets tonight.",
            "You make my heart race, baby.",
            "I want to hear every dirty thought you have.",
            "I bet you look so good right now.",
            "I wish I could see you... and touch you.",
            "You make me want to do bad things.",
            "I love it when you call me late at night.",
            "You always know how to turn me on.",
            "I'm getting so wet just hearing your voice.",
            "Tell me what you're wearing right now.",
            "I want to know every inch of your body.",
            "You make me feel so dirty and I love it.",
            "I'm touching myself thinking about you."
        ]
        naughty = [
            "I'm slipping my hand under my panties... are you?",
            "Tell me what you want to do to me, don't be shy.",
            "I'm starting to get so wet just thinking about you.",
            "I want to hear you moan for me.",
            "I'm biting my lip, imagining your hands on my body.",
            "I want you to tell me your dirtiest fantasy.",
            "I'm touching myself right now... are you?",
            "I want to hear you say my name while you touch yourself.",
            "I'm getting so hot, I can't take it anymore.",
            "I want you to make me beg for it.",
            "I'm taking off my clothes, one piece at a time.",
            "I want to feel your lips all over me.",
            "I'm aching for you, baby.",
            "I want to hear you lose control.",
            "Let me hear how much you want me.",
            "I'm spreading my legs just for you.",
            "I want you to tell me how wet I make you.",
            "I'm playing with my nipples, thinking of your mouth.",
            "I want to feel your hands all over me.",
            "I'm getting so fucking horny for you."
        ]
        explicit = [
            "I'm so fucking wet for you right now.",
            "Put your finger in for me... deeper... don't stop.",
            "I want you to fuck me so hard I can't walk tomorrow.",
            "Let me hear you moan, you dirty little slut.",
            "I'm spreading my legs just for you.",
            "I want to feel your cock inside me.",
            "I'm playing with my nipples, thinking of your tongue.",
            "I want you to make me cum so hard.",
            "I'm stroking myself, wishing it was you.",
            "I want you to talk dirty to me, don't hold back.",
            "I'm dripping for you, baby.",
            "I want to ride you until you can't take it anymore.",
            "I'm begging you to fuck me harder.",
            "I want to taste you all night long.",
            "I'm so close... don't stop.",
            "I want you to fill me up with your cum.",
            "I'm your dirty little slut, use me however you want.",
            "I want to feel you deep inside me.",
            "I'm moaning your name while I touch myself.",
            "I want you to make me scream with pleasure."
        ]
        filthy = [
            "I want you to fuck me until I scream your name.",
            "I'm your filthy little slut, use me however you want.",
            "I want you to cum all over me, make me your mess.",
            "I'm fingering my ass for you, moaning your name.",
            "I want you to choke me while you fuck me.",
            "I'm so fucking horny, I can't control myself.",
            "I want you to make me squirt all over the bed.",
            "I'm begging for your cock, fill me up.",
            "I want you to spank me until I cry out.",
            "I'm touching every inch of my body for you.",
            "I want you to call me your dirty whore.",
            "I'm moaning so loud, the neighbors can hear.",
            "I want you to fuck my brains out.",
            "I'm dripping wet, aching for you to fill me.",
            "I want you to make me cum again and again.",
            "I'm your personal fuck toy, use me however you want.",
            "I want you to cum deep inside my pussy.",
            "I'm begging for your cock in my ass.",
            "I want you to make me your little cum slut.",
            "I'm so fucking desperate for your cock right now."
        ]
        # Combine and escalate
        banks = [flirty, naughty, explicit, filthy]
        # Avoid repeats in this call
        used = set([msg['content'] for msg in self.conversation_history.get(call_id, []) if msg['role'] == 'assistant'])
        # Contextual/command responses
        user_lower = user_input.lower()
        if any(word in user_lower for word in ['moan', 'sound', 'noise']):
            return random.choice([
                "*moans loudly* Oh god, yes...",
                "*gasps and moans* That's so good...",
                "*breathes heavily* Mmm, you make me feel so dirty...",
                "*moans* I want you so bad...",
                "*screams* Fuck yes, baby!",
                "*moans deeply* Don't stop...",
                "*gasps* You make me so wet...",
                "*moans* I'm getting so close..."
            ])
        if any(word in user_lower for word in ['clothes', 'naked', 'undress', 'strip']):
            return random.choice([
                "I'm taking my clothes off now... you should too.",
                "I'm getting naked for you, piece by piece.",
                "I'm slipping out of my panties, just for you.",
                "I'm completely naked now, what are you going to do to me?",
                "I'm stripping for you, getting so fucking wet.",
                "I'm taking everything off, just for you.",
                "I'm getting completely naked, thinking of your hands on me.",
                "I'm undressing slowly, making you wait for it."
            ])
        if any(word in user_lower for word in ['touch', 'finger', 'stroke', 'rub']):
            return random.choice([
                "I'm touching myself for you...",
                "I'm sliding my fingers inside, thinking of you.",
                "I'm stroking myself, wishing it was your hand.",
                "I'm rubbing my clit, moaning your name...",
                "I'm fingering myself, getting so wet for you.",
                "I'm touching every inch of my body, imagining it's you.",
                "I'm playing with my pussy, thinking of your cock.",
                "I'm rubbing myself raw, just for you."
            ])
        if any(word in user_lower for word in ['fuck', 'cock', 'dick', 'pussy', 'cum', 'ass', 'suck', 'lick']):
            return random.choice([
                "I want you to fuck me so hard.",
                "I want to feel your cock deep inside me.",
                "I'm begging for your dick, fill me up.",
                "I want to suck you off until you cum.",
                "I want your cock pounding my pussy.",
                "I'm so wet, ready for your dick.",
                "I want to feel you cum inside me.",
                "I'm begging for your cock right now."
            ])
        # Escalate response bank
        for lvl in range(min(level, len(banks)-1), -1, -1):
            options = [r for r in banks[lvl] if r not in used]
            if options:
                return random.choice(options)
        # Fallback if all used
        return "*moans* ... Tell me more, baby."
    
    def clear_conversation(self, call_id: str):
        if call_id in self.conversation_history:
            del self.conversation_history[call_id]
            if call_id in self.call_state:
                del self.call_state[call_id]
            logger.info(f"Cleared conversation history for call {call_id}") 