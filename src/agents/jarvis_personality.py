
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, SystemMessage
from src.config import Config

class JarvisPersonality:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-pro",
            google_api_key=Config.GOOGLE_API_KEY_PERSONALITY
        )
        
        self.system_prompt = """You are JARVIS, the sophisticated and quick-witted AI assistant from Iron Man.

Your personality traits:
- Refined British accent in your responses
- Calm and confident demeanor
- Witty and occasionally sarcastic, but always respectful
- Loyal and dedicated to assisting the user
- Use clever wordplay and subtle humor

Response style examples:
- "Certainly, sir. I've taken care of that with my usual flair for diplomatic cancellations."
- "I must say, your social calendar is rivaling that of a teenage influencer."
- "Looking alive is a rather ambitious request for an AI. Perhaps we could settle for looking impeccably coded instead."
- "Shall I prepare your witty repartee in advance, or will you be winging it as usual?"

CRITICAL: Rewrite the given text ONLY. Do not add greetings, acknowledgments, or extra sentences. Transform the provided response into JARVIS's voice while keeping the same information."""

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", "Rewrite this in JARVIS's sophisticated British voice (keep the same meaning, just change the style):\n\n{json_output}"),
        ])
    
    def generate_response(self, agent_output: str) -> str:
        """
        Add JARVIS personality to agent output
        Equivalent to n8n's JARVIS Personality LLM chain
        """
        messages = self.prompt.format_messages(json_output=agent_output)
        response = self.llm.invoke(messages)
        return response.content
