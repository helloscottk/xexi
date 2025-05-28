from flask import Flask, request, Response
from config import Config
from phone.twilio_handler import TwilioPhoneHandler
from ai_models.llm_handler import NSFWLLMHandler
from voice.tts_handler import VoiceHandler, ElevenLabsVoiceHandler
from voice.stt_handler import SpeechToTextHandler
import logging

app = Flask(__name__)
config = Config()
logging.basicConfig(level=logging.INFO)

twilio_handler = TwilioPhoneHandler()
twilio_handler.initialize_twilio()
llm_handler = NSFWLLMHandler()
llm_handler.initialize_model()
tts_handler = VoiceHandler()
tts_handler.initialize_voice_engine()
stt_handler = SpeechToTextHandler()
stt_handler.initialize_stt_engine()
elevenlabs_handler = ElevenLabsVoiceHandler()

@app.route('/voice/incoming', methods=['POST'])
def voice_incoming():
    call_sid = request.values.get('CallSid', 'demo_call')
    return Response(str(twilio_handler.handle_incoming_call(call_sid)), mimetype='text/xml')

@app.route('/voice/process/<call_sid>', methods=['POST'])
def voice_process(call_sid):
    speech_result = request.values.get('SpeechResult', '')
    if not speech_result:
        # Try to get audio and transcribe
        if 'RecordingUrl' in request.values:
            audio_url = request.values['RecordingUrl']
            audio_data = requests.get(audio_url + '.wav').content
            speech_result = stt_handler.transcribe_audio(audio_data)
    if not speech_result:
        speech_result = ''
    # Generate AI response
    ai_response = llm_handler.generate_response(speech_result, call_sid)
    # Synthesize voice
    audio_bytes = None
    if elevenlabs_handler.is_available():
        audio_bytes = elevenlabs_handler.text_to_speech(ai_response)
    if not audio_bytes:
        audio_bytes = tts_handler.text_to_speech(ai_response)
    # Save audio to a file or serve as a URL (for Twilio <Play>), but for now use <Say>
    # For now, fallback to <Say> for demo
    response = twilio_handler.process_speech_input(call_sid, ai_response)
    return Response(str(response), mimetype='text/xml')

@app.route('/voice/continue/<call_sid>', methods=['POST', 'GET'])
def voice_continue(call_sid):
    return Response(str(twilio_handler.continue_conversation(call_sid)), mimetype='text/xml')

@app.route('/voice/end/<call_sid>', methods=['POST', 'GET'])
def voice_end(call_sid):
    return Response(str(twilio_handler.end_call(call_sid)), mimetype='text/xml')

if __name__ == '__main__':
    app.run(host=config.FLASK_HOST, port=config.FLASK_PORT, debug=config.DEBUG) 