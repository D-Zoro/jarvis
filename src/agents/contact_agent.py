
from typing import List
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import Tool
from pyairtable import Table
from config import Config

class ContactAgent:
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4-turbo-preview",
            openai_api_key=Config.OPENAI_API_KEY
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
- update_contact: Update existing contact information

Always provide accurate contact information."""
        
        self.tools = self._create_tools()
        self.agent = self._create_agent()
    
    def _create_tools(self) -> List[Tool]:
        """Create contact-related tools"""
        return [
            Tool(
                name="get_contact",
                func=self._get_contact,
                description="Get contact information by name. Input should be the contact name as a string."
            ),
            Tool(
                name="search_contacts",
                func=self._search_contacts,
                description="Search for contacts. Input should be a search query as a string."
            ),
            Tool(
                name="add_contact",
                func=self._add_contact,
                description="Add a new contact. Input should be a dict with 'name', 'email', and optional 'phone'."
            )
        ]
    
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
    
    def _create_agent(self) -> AgentExecutor:
        """Create the contact agent"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        agent = create_openai_tools_agent(self.llm, self.tools, prompt)
        return AgentExecutor(agent=agent, tools=self.tools, verbose=True)
    
    def run(self, query: str) -> str:
        """Execute contact agent"""
        result = self.agent.invoke({"input": query})
        return result["output"]
