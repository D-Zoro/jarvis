
from src.utils.telegram_handler import TelegramHandler
from src.config import Config

def main():
    """
    Main entry point for JARVIS assistant
    Equivalent to n8n workflow execution
    """
    print("Initializing JARVIS Assistant...")
    print(f"Using Anthropic model: {Config.ANTHROPIC_MODEL}")
    
    # Initialize and run Telegram bot
    telegram_handler = TelegramHandler()
    telegram_handler.run()

if __name__ == "__main__":
    main()
    print("Hello from jarvis!")


if __name__ == "__main__":
    main()
