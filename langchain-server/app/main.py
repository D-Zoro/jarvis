import os
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from dotenv import load_dotenv

from .core import get_agent_executor, process_prompt_with_agent
from .memory import add_memory, get_all_memories, clear_all_memories
from .dependencies import get_agent

# Load environment variables from .env file
load_dotenv()

app = FastAPI(
    title="Jarvis AI Core - FastAPI",
    description="Handles AI processing, memory, and agentic tasks.",
    version="1.0.0"
)

# --- Pydantic Models for API Requests ---
class PromptRequest(BaseModel):
    prompt: str
    session_id: str = "default_session"

class MemoryRequest(BaseModel):
    text: str
    session_id: str = "default_session"

class SessionRequest(BaseModel):
    session_id: str = "default_session"


# --- API Endpoints ---

@app.on_event("startup")
async def startup_event():
    """
    On startup, we can initialize any required services.
    For now, we just print a message. The agent is initialized on-demand.
    """
    print("🚀 Jarvis AI Core (FastAPI) is starting up...")
    # Validate that the NVIDIA API key is set
    if not os.getenv("NVIDIA_API_KEY"):
        raise RuntimeError("NVIDIA_API_KEY environment variable not set. Please get a key from build.nvidia.com and add it to the .env file.")


@app.post("/prompt", summary="Process a user prompt")
async def handle_prompt(request: PromptRequest, agent=Depends(get_agent)):
    """
    The main endpoint to interact with the AI.
    It takes a user prompt, processes it using the LangChain agent,
    and returns the AI's response.
    """
    try:
        response = await process_prompt_with_agent(agent, request.prompt, request.session_id)
        return {"response": response}
    except Exception as e:
        print(f"Error processing prompt: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/memory/add", summary="Add a new memory to the vector store")
async def add_memory_endpoint(request: MemoryRequest):
    """
    Explicitly adds a piece of text to the long-term vector memory.
    """
    try:
        add_memory(request.text, request.session_id)
        return {"status": "success", "message": "Memory added."}
    except Exception as e:
        print(f"Error adding memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/memory/list", summary="List all memories for a session")
async def list_memories_endpoint(request: SessionRequest):
    """
    Retrieves and returns all memories stored for a given session.
    """
    try:
        memories = get_all_memories(request.session_id)
        return {"status": "success", "memories": memories}
    except Exception as e:
        print(f"Error listing memories: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/memory/clear", summary="Clear all memories for a session")
async def clear_memories_endpoint(request: SessionRequest):
    """
    Wipes all memories for a given session from the vector store.
    """
    try:
        clear_all_memories(request.session_id)
        return {"status": "success", "message": "All memories for the session have been cleared."}
    except Exception as e:
        print(f"Error clearing memories: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/", summary="Health check")
def read_root():
    """A simple health check endpoint."""
    return {"status": "Jarvis AI Core is running"}
