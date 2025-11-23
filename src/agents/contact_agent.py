
from typing import List
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from pyairtable import Table
from src.config import Config

class ContactAgent:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-pro",
            google_api_key=Config.GOOGLE_API_KEY
        )
        
        self.table = Table(
            Config.AIRTABLE_API_KEY,
            Config.AIRTABLE_BASE_ID,
            Config.AIRTABLE_TABLE_NAME
        )
        
        self.system_prompt = """You are a Contact Database Agent. Your role is to retrieve and manage contact information.

Tools available:
- get_contact: Retrieve contact information by name
- search_contacts: Search for contacts
- add_contact: Add a new contact

Always provide accurate contact information."""
        
        self.agent = create_agent(
            model=self.llm,
            tools=self._create_tools(),
            system_prompt=self.system_prompt
        )
    
    def _create_tools(self) -> List:
        """Create contact-related tools"""
        
        @tool
        def get_contact(name: str) -> str:
            """Get contact information by name.
            
            Args:
                name: The name of the contact to retrieve
            """
            return self._get_contact(name)
        
        @tool
        def search_contacts(query: str) -> str:
            """Search for contacts.
            
            Args:
                query: Search query to find contacts
            """
            return self._search_contacts(query)
        
        @tool
        def add_contact(name: str, email: str, phone: str = "") -> str:
            """Add a new contact.
            
            Args:
                name: Contact name
                email: Contact email address
                phone: Optional phone number
            """
            return self._add_contact({"name": name, "email": email, "phone": phone})
        
        return [get_contact, search_contacts, add_contact]
    
    def _get_contact(self, name: str) -> str:
        """Get contact by name"""
        try:
            formula = f"{{Name}} = '{name}'"
            records = self.table.all(formula=formula)
            
            if not records:
                return f"No contact found with name: {name}"
            
            record = records[0]['fields']
            contact_info = f"Name: {record.get('Name', 'N/A')}\n"
            contact_info += f"Email: {record.get('Email', 'N/A')}\n"
            contact_info += f"Phone: {record.get('Phone', 'N/A')}"
            
            return contact_info
        except Exception as e:
            return f"Error retrieving contact: {str(e)}"
    
    def _search_contacts(self, query: str) -> str:
        """Search contacts"""
        try:
            all_records = self.table.all()
            matches = []
            
            for record in all_records:
                fields = record['fields']
                name = fields.get('Name', '').lower()
                email = fields.get('Email', '').lower()
                
                if query.lower() in name or query.lower() in email:
                    matches.append(f"{fields.get('Name')} ({fields.get('Email')})")
            
            if not matches:
                return f"No contacts found matching: {query}"
            
            return "Matching contacts:\n" + "\n".join(matches)
        except Exception as e:
            return f"Error searching contacts: {str(e)}"
    
    def _add_contact(self, input_data: str) -> str:
        """Add new contact"""
        try:
            import json
            params = json.loads(input_data) if isinstance(input_data, str) else input_data
            
            self.table.create({
                'Name': params['name'],
                'Email': params['email'],
                'Phone': params.get('phone', '')
            })
            
            return f"Contact added: {params['name']}"
        except Exception as e:
            return f"Error adding contact: {str(e)}"
    
    def run(self, query: str) -> str:
        """Execute contact agent"""
        result = self.agent.invoke({"messages": [{"role": "user", "content": query}]})
        messages = result.get("messages", [])
        if messages:
            last_message = messages[-1]
            if hasattr(last_message, 'content'):
                return last_message.content
            elif isinstance(last_message, dict):
                return last_message.get('content', str(last_message))
        return str(result)
