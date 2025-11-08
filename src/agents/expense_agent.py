
from typing import List
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import Tool
from pinecone import Pinecone
from config import Config
import gspread
from google.oauth2.service_account import Credentials

class ExpenseAgent:
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4-turbo-preview",
            openai_api_key=Config.OPENAI_API_KEY
        )
        
        # Initialize Pinecone for vector search
        self.pc = Pinecone(api_key=Config.PINECONE_API_KEY)
        self.index = self.pc.Index(Config.PINECONE_INDEX_NAME)
        self.embeddings = OpenAIEmbeddings()
        
        self.system_prompt = """You are a Personal Expense Agent. Your role is to provide accurate and relevant information about the user's expenses.

Tools available:
- query_expenses: Search expense history using semantic search
- get_credit_card_transactions: Get recent credit card transactions from Google Sheets
- calculate_spending: Calculate total spending for a category or time period

When to use each tool:
- Use query_expenses for historical expense searches and patterns
- Use get_credit_card_transactions for recent transaction data
- Use calculate_spending for aggregating expense data

Always provide clear, actionable insights about expenses."""
        
        self.tools = self._create_tools()
        self.agent = self._create_agent()
    
    def _create_tools(self) -> List[Tool]:
        """Create expense-related tools"""
        return [
            Tool(
                name="query_expenses",
                func=self._query_expenses,
                description="Search expense history using semantic search. Input should be a natural language query about expenses."
            ),
            Tool(
                name="get_credit_card_transactions",
                func=self._get_credit_card_transactions,
                description="Get credit card transactions. Input should be a dict with optional 'start_date' and 'end_date'."
            ),
            Tool(
                name="calculate_spending",
                func=self._calculate_spending,
                description="Calculate total spending. Input should be a dict with 'category' and/or 'time_period'."
            )
        ]
    
    def _query_expenses(self, query: str) -> str:
        """Query expenses using vector search"""
        try:
            # Create embedding for query
            query_embedding = self.embeddings.embed_query(query)
            
            # Search Pinecone
            results = self.index.query(
                vector=query_embedding,
                top_k=5,
                include_metadata=True
            )
            
            if not results['matches']:
                return "No matching expenses found."
            
            expense_summary = []
            for match in results['matches']:
                metadata = match['metadata']
                expense_summary.append(
                    f"- {metadata.get('description', 'N/A')}: ${metadata.get('amount', 0)} "
                    f"on {metadata.get('date', 'N/A')} (Category: {metadata.get('category', 'N/A')})"
                )
            
            return "Matching expenses:\n" + "\n".join(expense_summary)
        except Exception as e:
            return f"Error querying expenses: {str(e)}"
    
    def _get_credit_card_transactions(self, input_data: str) -> str:
        """Get credit card transactions from Google Sheets"""
        try:
            import json
            from datetime import datetime, timedelta
            
            params = json.loads(input_data) if isinstance(input_data, str) else {}
            
            # Initialize Google Sheets
            scope = ['https://spreadsheets.google.com/feeds',
                    'https://www.googleapis.com/auth/drive']
            creds = Credentials.from_service_account_file(
                Config.GOOGLE_SHEETS_CREDENTIALS,
                scopes=scope
            )
            client = gspread.authorize(creds)
            
            # Open the sheet
            sheet = client.open(Config.EXPENSE_SHEET_NAME).sheet1
            records = sheet.get_all_records()
            
            # Filter by date if provided
            if 'start_date' in params or 'end_date' in params:
                # Filter logic here
                pass
            
            transactions = []
            for record in records[:10]:  # Last 10 transactions
                transactions.append(
                    f"- {record.get('Description')}: ${record.get('Amount')} "
                    f"on {record.get('Date')} (Category: {record.get('Category')})"
                )
            
            return "Recent transactions:\n" + "\n".join(transactions)
        except Exception as e:
            return f"Error retrieving transactions: {str(e)}"
    
    def _calculate_spending(self, input_data: str) -> str:
        """Calculate total spending"""
        try:
            import json
            params = json.loads(input_data) if isinstance(input_data, str) else input_data
            
            category = params.get('category')
            time_period = params.get('time_period')
            
            # This would query your expense database
            # For demo purposes, returning a placeholder
            total = 1250.75  # This would be calculated from actual data
            
            result = f"Total spending"
            if category:
                result += f" on {category}"
            if time_period:
                result += f" for {time_period}"
            result += f": ${total}"
            
            return result
        except Exception as e:
            return f"Error calculating spending: {str(e)}"
    
    def _create_agent(self) -> AgentExecutor:
        """Create the expense agent"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        agent = create_openai_tools_agent(self.llm, self.tools, prompt)
        return AgentExecutor(agent=agent, tools=self.tools, verbose=True)
    
    def run(self, query: str) -> str:
        """Execute expense agent"""
        result = self.agent.invoke({"input": query})
        return result["output"]
