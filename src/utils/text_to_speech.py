import requests
import json
from config import Config

class TextToSpeechHandler:
    def __init__(self):
        self.api_key = Config.ELEVENLABS_API_KEY
        self.voice_id = Config.ELEVENLABS_VOICE_ID
        self.base_url = "https://api.elevenlabs.io/v1"
    
    def convert_text_to_speech(self, text: str) -> bytes:
        """
        Convert text to speech using ElevenLabs
        Equivalent to n8n's HTTP Request node for TTS
        """
        url = f"{self.base_url}/text-to-speech/{self.voice_id}"
        
        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": self.api_key
        }
        
        # Clean the text to ensure valid JSON
        cleaned_text = self._clean_text_for_json(text)
        
        data = {
            "text": cleaned_text,
            "model_id": "eleven_monolingual_v1",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.5
            }
        }
        
        response = requests.post(url, json=data, headers=headers)
        
        if response.status_code == 200:
            return response.content
        else:
            raise Exception(f"TTS API error: {response.status_code} - {response.text}")
    
    def _clean_text_for_json(self, text: str) -> str:
        """
        Clean text to prevent JSON serialization errors
        Similar to the json.stringify logic in n8n
        """
        # Remove problematic characters
        text = text.replace('\n', ' ')
        text = text.replace('\r', '')
        text = text.replace('\t', ' ')
        text = text.replace('*', '')
        # Remove multiple spaces
        text = ' '.join(text.split())
        return text

