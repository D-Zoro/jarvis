import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # API Keys
    TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
    ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "your-jarvis-voice-id")
    
    # Google Calendar
    GOOGLE_CALENDAR_CREDENTIALS = os.getenv("GOOGLE_CALENDAR_CREDENTIALS")
    
    # Airtable (for contacts)
    AIRTABLE_API_KEY = os.getenv("AIRTABLE_API_KEY")
    AIRTABLE_BASE_ID = os.getenv("AIRTABLE_BASE_ID")
    AIRTABLE_TABLE_NAME = os.getenv("AIRTABLE_TABLE_NAME", "Contacts")
    
    # Pinecone (for expense vector DB)
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "expenses")
    
    # Model settings
    ANTHROPIC_MODEL = "claude-3-5-sonnet-20241022"

