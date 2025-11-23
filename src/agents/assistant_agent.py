
from typing import TypedDict, Annotated, Sequence
import operator
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from src.config import Config

from src.agents.calendar_agent import CalendarAgent
from src.agents.email_agent import EmailAgent
from src.agents.contact_agent import ContactAgent
from src.agents.expense_agent import ExpenseAgent

class AgentState(TypedDict):
    """State passed between agents in the graph"""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    sender: str
    next_agent: str
    final_response: str

class AssistantAgent:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-pro",
            google_api_key=Config.GOOGLE_API_KEY,
            temperature=0
        )
        
        # Initialize child agents
        self.calendar_agent = CalendarAgent()
        self.email_agent = EmailAgent()
        self.contact_agent = ContactAgent()
        self.expense_agent = ExpenseAgent()
        
        self.system_prompt = """You are a Personal Assistant AI. Your role is to efficiently delegate user queries to appropriate tools/agents.

Available agents:
1. calendar_agent - For calendar-related actions (creating, retrieving, updating, or deleting events)
2. email_agent - For email-related actions (sending, drafting emails)
3. contact_agent - For contact management (retrieving contact information, searching contacts)
4. expense_agent - For expense tracking and financial queries (tracking spending, analyzing expenses)

Your responsibilities:
- Analyze user requests carefully
- Determine which agent(s) can best handle the request
- Delegate to the appropriate agent
- Synthesize responses from multiple agents if needed
- Provide clear, concise responses

When the user asks about calendar events, meetings, or scheduling -> use calendar_agent
When the user asks to send emails or compose messages -> use email_agent + contact_agent (to get email addresses)
When the user asks about contacts or contact information -> use contact_agent
When the user asks about expenses, spending, or finances -> use expense_agent

Always think step-by-step about what information you need and which agents to call."""
        
        self.graph = self._create_graph()
    
    def _create_graph(self) -> StateGraph:
        """Create LangGraph workflow"""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("router", self._router_node)
        workflow.add_node("calendar", self._calendar_node)
        workflow.add_node("email", self._email_node)
        workflow.add_node("contact", self._contact_node)
        workflow.add_node("expense", self._expense_node)
        workflow.add_node("synthesizer", self._synthesizer_node)
        
        # Set entry point
        workflow.set_entry_point("router")
        
        # Add conditional edges from router
        workflow.add_conditional_edges(
            "router",
            self._route_decision,
            {
                "calendar": "calendar",
                "email": "email",
                "contact": "contact",
                "expense": "expense",
                "end": "synthesizer"
            }
        )
        
        # All agents route to synthesizer
        workflow.add_edge("calendar", "synthesizer")
        workflow.add_edge("email", "synthesizer")
        workflow.add_edge("contact", "synthesizer")
        workflow.add_edge("expense", "synthesizer")
        workflow.add_edge("synthesizer", END)
        
        return workflow.compile()
    
    def _router_node(self, state: AgentState) -> AgentState:
        """Route to appropriate agent based on query"""
        last_message = state["messages"][-1].content
        
        # Use LLM to determine routing
        routing_prompt = ChatPromptTemplate.from_messages([
            ("system", """Analyze the user query and determine which agent should handle it.
            
Available agents:
- calendar: for calendar, meetings, events, schedules
- email: for sending emails, composing messages
- contact: for contact information, phone numbers, email addresses
- expense: for financial queries, spending, expenses
- end: if the query is a simple greeting or doesn't need an agent

Respond with ONLY the agent name, nothing else."""),
            ("human", "{query}")
        ])
        
        response = self.llm.invoke(routing_prompt.format_messages(query=last_message))
        next_agent = response.content.strip().lower()
        
        state["next_agent"] = next_agent
        return state
    
    def _route_decision(self, state: AgentState) -> str:
        """Determine next node based on routing decision"""
        return state.get("next_agent", "end")
    
    def _calendar_node(self, state: AgentState) -> AgentState:
        """Execute calendar agent"""
        query = state["messages"][-1].content
        result = self.calendar_agent.run(query)
        
        state["messages"].append(AIMessage(content=result, name="calendar_agent"))
        state["sender"] = "calendar_agent"
        return state
    
    def _email_node(self, state: AgentState) -> AgentState:
        """Execute email agent - may need contact info first"""
        query = state["messages"][-1].content
        
        # Check if we need contact info
        if "send" in query.lower() and "@" not in query:
            # Extract contact name and get email
            # This is simplified - you'd want more robust name extraction
            words = query.split()
            for i, word in enumerate(words):
                if word.lower() in ["to", "email"]:
                    if i + 1 < len(words):
                        contact_name = words[i + 1]
                        contact_info = self.contact_agent.run(f"Get contact information for {contact_name}")
                        state["messages"].append(AIMessage(content=contact_info, name="contact_agent"))
        
        result = self.email_agent.run(query)
        state["messages"].append(AIMessage(content=result, name="email_agent"))
        state["sender"] = "email_agent"
        return state
    
    def _contact_node(self, state: AgentState) -> AgentState:
        """Execute contact agent"""
        query = state["messages"][-1].content
        result = self.contact_agent.run(query)
        
        state["messages"].append(AIMessage(content=result, name="contact_agent"))
        state["sender"] = "contact_agent"
        return state
    
    def _expense_node(self, state: AgentState) -> AgentState:
        """Execute expense agent"""
        query = state["messages"][-1].content
        result = self.expense_agent.run(query)
        
        state["messages"].append(AIMessage(content=result, name="expense_agent"))
        state["sender"] = "expense_agent"
        return state
    
    def _synthesizer_node(self, state: AgentState) -> AgentState:
        """Synthesize final response"""
        # Get all agent responses
        agent_responses = [
            msg.content for msg in state["messages"]
            if isinstance(msg, AIMessage) and hasattr(msg, 'name')
        ]
        
        if agent_responses:
            # Combine responses if multiple agents were called
            final_response = "\n\n".join(agent_responses)
        else:
            # Direct response
            final_response = "I'm ready to assist you. What would you like help with?"
        
        state["final_response"] = final_response
        return state
    
    def run(self, query: str) -> str:
        """Execute the assistant agent workflow"""
        initial_state = {
            "messages": [HumanMessage(content=query)],
            "sender": "user",
            "next_agent": "",
            "final_response": ""
        }
        
        result = self.graph.invoke(initial_state)
        return result["final_response"]
