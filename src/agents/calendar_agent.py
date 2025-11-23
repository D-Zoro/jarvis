from typing import List
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from googleapiclient.discovery import build
from google.oauth2.service_account import Credentials
from datetime import datetime
from src.config import Config

class CalendarAgent:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-pro",
            google_api_key=Config.GOOGLE_API_KEY
        )
        
        # Initialize Google Calendar API
        creds = Credentials.from_service_account_file(
            Config.GOOGLE_CALENDAR_CREDENTIALS,
            scopes=['https://www.googleapis.com/auth/calendar']
        )
        self.calendar_service = build('calendar', 'v3', credentials=creds)
        
        self.system_prompt = """You are a Calendar Management Agent. Your role is to manage calendar events for the user.

Tools available:
- create_event: Create a new calendar event
- get_events: Retrieve calendar events
- delete_event: Delete a calendar event

Always provide clear confirmations and handle time zones appropriately."""
        
        self.agent = create_agent(
            model=self.llm,
            tools=self._create_tools(),
            system_prompt=self.system_prompt
        )
    
    def _create_tools(self) -> List:
        """Create calendar-related tools"""
        
        @tool
        def create_event(summary: str, start_time: str, end_time: str, description: str = "") -> str:
            """Create a calendar event.
            
            Args:
                summary: The title of the event
                start_time: Start time in ISO format
                end_time: End time in ISO format
                description: Optional event description
            """
            return self._create_event({"summary": summary, "start_time": start_time, "end_time": end_time, "description": description})
        
        @tool
        def get_events(start_date: str = "", end_date: str = "") -> str:
            """Get calendar events.
            
            Args:
                start_date: Optional start date filter
                end_date: Optional end date filter
            """
            return self._get_events({"start_date": start_date, "end_date": end_date} if start_date or end_date else {})
        
        @tool
        def delete_event(event_id: str) -> str:
            """Delete a calendar event.
            
            Args:
                event_id: The ID of the event to delete
            """
            return self._delete_event(event_id)
        
        return [create_event, get_events, delete_event]
    
    def _create_event(self, input_data: str) -> str:
        """Create a calendar event"""
        try:
            import json
            params = json.loads(input_data) if isinstance(input_data, str) else input_data
            
            event = {
                'summary': params['summary'],
                'description': params.get('description', ''),
                'start': {
                    'dateTime': params['start_time'],
                    'timeZone': 'America/Los_Angeles',
                },
                'end': {
                    'dateTime': params['end_time'],
                    'timeZone': 'America/Los_Angeles',
                },
            }
            
            created_event = self.calendar_service.events().insert(
                calendarId='primary',
                body=event
            ).execute()
            
            return f"Event created: {created_event['summary']} on {params['start_time']}"
        except Exception as e:
            return f"Error creating event: {str(e)}"
    
    def _get_events(self, input_data: str) -> str:
        """Get calendar events"""
        try:
            import json
            from datetime import datetime, timedelta
            
            params = json.loads(input_data) if isinstance(input_data, str) else {}
            
            # Default to next 7 days
            time_min = params.get('start_date', datetime.utcnow().isoformat() + 'Z')
            time_max = params.get('end_date', (datetime.utcnow() + timedelta(days=7)).isoformat() + 'Z')
            
            events_result = self.calendar_service.events().list(
                calendarId='primary',
                timeMin=time_min,
                timeMax=time_max,
                maxResults=10,
                singleEvents=True,
                orderBy='startTime'
            ).execute()
            
            events = events_result.get('items', [])
            
            if not events:
                return "No upcoming events found."
            
            event_list = []
            for event in events:
                start = event['start'].get('dateTime', event['start'].get('date'))
                event_list.append(f"- {event['summary']} at {start}")
            
            return "Upcoming events:\n" + "\n".join(event_list)
        except Exception as e:
            return f"Error retrieving events: {str(e)}"
    
    def _delete_event(self, event_id: str) -> str:
        """Delete a calendar event"""
        try:
            self.calendar_service.events().delete(
                calendarId='primary',
                eventId=event_id
            ).execute()
            
            return f"Event deleted successfully"
        except Exception as e:
            return f"Error deleting event: {str(e)}"
    
    def run(self, query: str) -> str:
        """Execute calendar agent"""
        result = self.agent.invoke({"messages": [{"role": "user", "content": query}]})
        # Extract the last message content
        messages = result.get("messages", [])
        if messages:
            last_message = messages[-1]
            if hasattr(last_message, 'content'):
                return last_message.content
            elif isinstance(last_message, dict):
                return last_message.get('content', str(last_message))
        return str(result)

