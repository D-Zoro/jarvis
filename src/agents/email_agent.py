from typing import List
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from src.config import Config

class EmailAgent:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-pro",
            google_api_key=Config.GOOGLE_API_KEY
        )
        
        self.system_prompt = """You are an Email Management Agent. Your role is to send, read, and manage emails for the user.

Tools available:
- send_email: Send an email to specified recipients
- draft_email: Draft an email without sending

Always compose professional, well-formatted emails."""
        
        self.agent = create_agent(
            model=self.llm,
            tools=self._create_tools(),
            system_prompt=self.system_prompt
        )
    
    def _create_tools(self) -> List:
        """Create email-related tools"""
        
        @tool
        def send_email(to: str, subject: str, body: str) -> str:
            """Send an email.
            
            Args:
                to: Recipient email address
                subject: Email subject
                body: Email body content
            """
            return self._send_email({"to": to, "subject": subject, "body": body})
        
        @tool
        def draft_email(to: str, subject: str, body: str) -> str:
            """Draft an email without sending.
            
            Args:
                to: Recipient email address
                subject: Email subject
                body: Email body content
            """
            return self._draft_email({"to": to, "subject": subject, "body": body})
        
        return [send_email, draft_email]
    
    def _send_email(self, input_data: str) -> str:
        """Send an email"""
        try:
            import json
            params = json.loads(input_data) if isinstance(input_data, str) else input_data
            
            # Create message
            msg = MIMEMultipart()
            msg['From'] = Config.EMAIL_ADDRESS
            msg['To'] = params['to']
            msg['Subject'] = params['subject']
            
            msg.attach(MIMEText(params['body'], 'plain'))
            
            # Send email via SMTP
            with smtplib.SMTP(Config.SMTP_SERVER, Config.SMTP_PORT) as server:
                server.starttls()
                server.login(Config.EMAIL_ADDRESS, Config.EMAIL_PASSWORD)
                server.send_message(msg)
            
            return f"Email sent successfully to {params['to']}"
        except Exception as e:
            return f"Error sending email: {str(e)}"
    
    def _draft_email(self, input_data: str) -> str:
        """Draft an email"""
        try:
            import json
            params = json.loads(input_data) if isinstance(input_data, str) else input_data
            
            draft = f"""
Email Draft:
To: {params['to']}
Subject: {params['subject']}

{params['body']}
"""
            return draft
        except Exception as e:
            return f"Error drafting email: {str(e)}"
    
    def run(self, query: str) -> str:
        """Execute email agent"""
        result = self.agent.invoke({"messages": [{"role": "user", "content": query}]})
        messages = result.get("messages", [])
        if messages:
            last_message = messages[-1]
            if hasattr(last_message, 'content'):
                return last_message.content
            elif isinstance(last_message, dict):
                return last_message.get('content', str(last_message))
        return str(result)

