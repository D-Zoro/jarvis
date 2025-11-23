
from typing import List
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.tools import tool
from pinecone import Pinecone
from src.config import Config
import gspread
from google.oauth2.service_account import Credentials

class ExpenseAgent:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-pro",
            google_api_key=Config.GOOGLE_API_KEY
        )
        
        # Initialize Pinecone for vector search
        self.pc = Pinecone(api_key=Config.PINECONE_API_KEY)
        self.index = self.pc.Index(Config.PINECONE_INDEX_NAME)
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=Config.GOOGLE_API_KEY
        )
        
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
        
        self.agent = create_agent(
            model=self.llm,
            tools=self._create_tools(),
            system_prompt=self.system_prompt
        )
    
    def _create_tools(self) -> List:
        """Create expense-related tools"""
        
        @tool
        def query_expenses(query: str) -> str:
            """Search expense history using semantic search.
            
            Args:
                query: Natural language query about expenses
            """
            return self._query_expenses(query)
        
        @tool
        def get_credit_card_transactions(start_date: str = "", end_date: str = "") -> str:
            """Get credit card transactions.
            
            Args:
                start_date: Optional start date filter
                end_date: Optional end date filter
            """
            return self._get_credit_card_transactions({"start_date": start_date, "end_date": end_date} if start_date or end_date else {})
        
        @tool
        def calculate_spending(category: str = "", time_period: str = "") -> str:
            """Calculate total spending.
            
            Args:
                category: Optional category filter
                time_period: Optional time period filter
            """
            return self._calculate_spending({"category": category, "time_period": time_period})
        
        return [query_expenses, get_credit_card_transactions, calculate_spending]
    
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
    
    def run(self, query: str) -> str:
        """Execute expense agent"""
        result = self.agent.invoke({"messages": [{"role": "user", "content": query}]})
        messages = result.get("messages", [])
        if messages:
            last_message = messages[-1]
            if hasattr(last_message, 'content'):
                return last_message.content
            elif isinstance(last_message, dict):
                return last_message.get('content', str(last_message))
        return str(result)
