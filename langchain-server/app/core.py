import os
import requests
import datetime
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain.tools import Tool
from .memory import get_vector_memory, add_memory

# The URL for the Next.js controller's new search MCP endpoint
CONTROLLER_SEARCH_URL = "http://controller-nextjs:3001/api/mcp/search"

# --- Updated Agent Prompt Template ---
# We've added the WebSearch tool and instructions on when to use it.
# We've also added a placeholder for the current time and date.
AGENT_TEMPLATE = """
You are Jarvis, a helpful and conversational AI assistant.

Your personality is helpful, knowledgeable, and proactive.
You have access to a long-term memory to recall past information.
The current date and time is: {current_time}. Use this for any time-sensitive questions.

TOOLS:
------
You have access to the following tools:
{tools}

To use a tool, please use the following format:
```
Thought: Do I need to use a tool? Yes
Action: The action to take, should be one of [{tool_names}]
Action Input: The input to the action
Observation: The result of the action
```

When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:
```
Thought: Do I need to use a tool? No
Final Answer: [your response here]
```

Begin!

Previous conversation history:
{chat_history}

New input: {input}
{agent_scratchpad}
"""

# --- Tool Functions ---

def retrieve_memories(query: str, session_id: str) -> str:
    """A tool to retrieve relevant information from the user's long-term memory."""
    # Note: We need to pass the session_id to the memory retriever now
    llm = ChatNVIDIA(model=os.getenv("NVIDIA_MODEL_NAME"))
    memory = get_vector_memory(llm, session_id)
    return memory.load_memory_variables({"prompt": query})["history"]

def search_the_web(query: str) -> str:
    """
    A tool to search the web for recent or unknown information.
    This function calls the Next.js controller to perform the actual search.
    """
    print(f"[Agent Tool] Searching web for: '{query}'")
    try:
        response = requests.post(CONTROLLER_SEARCH_URL, json={"query": query}, timeout=30)
        response.raise_for_status()
        results = response.json().get("results", "No results found.")
        print(f"[Agent Tool] Received search results: {results[:200]}...")
        return results
    except requests.RequestException as e:
        print(f"[Agent Tool] Error calling web search controller: {e}")
        return f"Error: Could not connect to the web search service. {e}"


def get_agent_executor(session_id: str = "default_session"):
    """
    Initializes and returns a LangChain AgentExecutor with updated tools.
    """
    llm = ChatNVIDIA(model=os.getenv("NVIDIA_MODEL_NAME"))

    # Updated tools list with the new WebSearch tool
    tools = [
        Tool(
            name="MemoryRetriever",
            func=lambda q: retrieve_memories(q, session_id=session_id), # Use lambda to pass session_id
            description="Use this tool to retrieve relevant memories or recall past conversations with the user. The input should be a question about what you might remember.",
        ),
        Tool(
            name="WebSearch",
            func=search_the_web,
            description="Use this tool when you need to answer questions about current events, recent news, or topics you don't have information on. The input should be a clear search query.",
        ),
    ]

    prompt = PromptTemplate.from_template(AGENT_TEMPLATE)
    agent = create_react_agent(llm, tools, prompt)
    memory = get_vector_memory(llm, session_id)

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        memory=memory,
        handle_parsing_errors="Check your output and make sure it conforms to the format.",
    )
    return agent_executor


async def process_prompt_with_agent(agent: AgentExecutor, prompt: str, session_id: str):
    """
    Invokes the agent, injecting the current time, and saves the interaction to memory.
    """
    # Get current time and format it nicely
    now = datetime.datetime.now().strftime("%A, %B %d, %Y %I:%M %p UTC")
    
    # Inject the current time into the agent's invocation
    response = await agent.ainvoke({
        "input": prompt,
        "current_time": now
    })
    
    interaction = f"User asked: '{prompt}'. Jarvis responded: '{response['output']}'"
    add_memory(interaction, session_id)
    
    return response["output"]
