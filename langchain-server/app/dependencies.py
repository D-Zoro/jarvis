from fastapi import Request
from .core import get_agent_executor

# This file is for managing dependencies, making the app more modular.
# For now, it just provides the agent executor.

def get_agent(request: Request):
    """
    A dependency function to get the agent executor.
    This could be expanded to handle session management more robustly.
    """
    # For now, we use a single agent instance.
    # In a multi-user scenario, you'd create/cache agents per session_id.
    return get_agent_executor()
