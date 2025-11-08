from typing import List
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import Tool
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from config import Config

class EmailAgent:
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4-turbo-preview",
            openai_api_key=Config.OPENAI_API_KEY
        )
        
        self.system_prompt = """You are an Email Management Agent. Your role is to send, read, and manage emails for the user.

Tools available:
- send_email: Send an email to specified recipients
- draft_email: Draft an email without sending

Always compose professional, well-formatted emails."""
        
        self.tools = self._create_tools()
        self.agent = self._create_agent()
    
    def _create_tools(self) -> List[Tool]:
        """Create email-related tools"""
        return [
            Tool(
                name="send_email",
                func=self._send_email,
                description="Send an email. Input should be a dict with 'to', 'subject', and 'body'."
            ),
            Tool(
                name="draft_email",
                func=self._draft_email,
                description="Draft an email without sending. Input should be a dict with 'to', 'subject', and 'body'."
            )
        ]
    
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
    
    def _create_agent(self) -> AgentExecutor:
        """Create the email agent"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        agent = create_openai_tools_agent(self.llm, self.tools, prompt)
        return AgentExecutor(agent=agent, tools=self.tools, verbose=True)
    
    def run(self, query: str) -> str:
        """Execute email agent"""
        result = self.agent.invoke({"input": query})
        return result["output"]

