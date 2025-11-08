import speech_recognition as sr
from pydub import AudioSegment
import io
import os

class VoiceHandler:
    def __init__(self):
        self.recognizer = sr.Recognizer()
    
    def transcribe_audio(self, audio_file_path: str) -> str:
        """
        Transcribe audio file to text
        Similar to n8n's Transcribe node
        """
        try:
            # Convert audio to WAV format if needed
            audio = AudioSegment.from_file(audio_file_path)
            wav_io = io.BytesIO()
            audio.export(wav_io, format="wav")
            wav_io.seek(0)
            
            # Transcribe
            with sr.AudioFile(wav_io) as source:
                audio_data = self.recognizer.record(source)
                text = self.recognizer.recognize_google(audio_data)
                return text
        except Exception as e:
            print(f"Transcription error: {e}")
            return ""

