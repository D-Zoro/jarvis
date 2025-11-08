
from telegram import Update
from telegram.ext import Application, MessageHandler, filters, ContextTypes
import os
from config import Config
from agents.assistant_agent import AssistantAgent
from agents.jarvis_personality import JarvisPersonality
from utils.voice_handler import VoiceHandler
from utils.text_to_speech import TextToSpeechHandler

class TelegramHandler:
    def __init__(self):
        self.assistant = AssistantAgent()
        self.jarvis_personality = JarvisPersonality()
        self.voice_handler = VoiceHandler()
        self.tts_handler = TextToSpeechHandler()
        
        self.app = Application.builder().token(Config.TELEGRAM_BOT_TOKEN).build()
        
        # Add handlers
        self.app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_text))
        self.app.add_handler(MessageHandler(filters.VOICE, self.handle_voice))
    
    async def handle_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle text messages"""
        user_message = update.message.text
        
        # Process through assistant agent
        agent_response = self.assistant.run(user_message)
        
        # Add JARVIS personality
        jarvis_response = self.jarvis_personality.generate_response(agent_response)
        
        # Send text response
        await update.message.reply_text(jarvis_response)
        
        # Convert to speech and send audio
        try:
            audio_bytes = self.tts_handler.convert_text_to_speech(jarvis_response)
            
            # Save temporarily
            audio_path = f"temp_audio_{update.message.message_id}.mp3"
            with open(audio_path, "wb") as f:
                f.write(audio_bytes)
            
            # Send audio file
            await update.message.reply_voice(voice=open(audio_path, "rb"))
            
            # Clean up
            os.remove(audio_path)
        except Exception as e:
            print(f"TTS Error: {e}")
    
    async def handle_voice(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle voice messages"""
        # Download voice file
        voice_file = await update.message.voice.get_file()
        voice_path = f"temp_voice_{update.message.message_id}.ogg"
        await voice_file.download_to_drive(voice_path)
        
        # Transcribe
        transcribed_text = self.voice_handler.transcribe_audio(voice_path)
        
        # Clean up voice file
        os.remove(voice_path)
        
        if not transcribed_text:
            await update.message.reply_text("Sorry, I couldn't understand the audio.")
            return
        
        # Process through assistant agent
        agent_response = self.assistant.run(transcribed_text)
        
        # Add JARVIS personality
        jarvis_response = self.jarvis_personality.generate_response(agent_response)
        
        # Send text response
        await update.message.reply_text(jarvis_response)
        
        # Convert to speech and send audio
        try:
            audio_bytes = self.tts_handler.convert_text_to_speech(jarvis_response)
            
            audio_path = f"temp_audio_{update.message.message_id}.mp3"
            with open(audio_path, "wb") as f:
                f.write(audio_bytes)
            
            await update.message.reply_voice(voice=open(audio_path, "rb"))
            
            os.remove(audio_path)
        except Exception as e:
            print(f"TTS Error: {e}")
    
    def run(self):
        """Start the bot"""
        print("JARVIS is online...")
        self.app.run_polling()
