import os
import chromadb
from langchain.memory import VectorStoreRetrieverMemory
from langchain_community.vectorstores import Chroma
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

# --- ChromaDB Client Initialization ---
# This sets up the connection to our persistent vector database.
# It runs in a separate Docker container.
client = chromadb.HttpClient(
    host=os.getenv("CHROMA_DB_HOST", "localhost"),
    port=int(os.getenv("CHROMA_DB_PORT", 8000))
)

# --- Embedding Model ---
# We use NVIDIA's embedding model to convert text into numerical vectors for storage.
embeddings = NVIDIAEmbeddings()

def get_vector_store(session_id: str = "default_session"):
    """
    Gets or creates a Chroma vector store instance for a given session.
    Each session has its own collection in ChromaDB, ensuring data isolation.
    """
    collection_name = f"jarvis_memory_{session_id}"
    vector_store = Chroma(
        client=client,
        collection_name=collection_name,
        embedding_function=embeddings,
    )
    return vector_store

def get_vector_memory(llm, session_id: str = "default_session"):
    """
    Creates a LangChain memory object backed by our Chroma vector store.
    This is what the agent interacts with.
    """
    vector_store = get_vector_store(session_id)
    retriever = vector_store.as_retriever(search_kwargs=dict(k=5)) # Retrieve top 5 relevant docs
    
    # VectorStoreRetrieverMemory links the vector store to the agent's memory
    memory = VectorStoreRetrieverMemory(
        retriever=retriever,
        memory_key="chat_history", # The key the agent prompt expects
        input_key="input"
    )
    return memory

def add_memory(text: str, session_id: str = "default_session"):
    """
    Adds a new piece of text to the long-term memory for a specific session.
    """
    vector_store = get_vector_store(session_id)
    vector_store.add_texts([text])
    print(f"Added memory for session '{session_id}': '{text}'")

def get_all_memories(session_id: str = "default_session"):
    """
    Retrieves all stored documents (memories) for a given session.
    """
    vector_store = get_vector_store(session_id)
    # The .get() method retrieves documents and their metadata.
    results = vector_store.get()
    return results.get('documents', [])

def clear_all_memories(session_id: str = "default_session"):
    """
    Deletes an entire collection from ChromaDB, effectively clearing memory.
    """
    collection_name = f"jarvis_memory_{session_id}"
    try:
        client.delete_collection(name=collection_name)
        print(f"Cleared all memories for session '{session_id}'.")
    except Exception as e:
        # ChromaDB might raise an error if the collection doesn't exist, which is fine.
        print(f"Could not clear memories for session '{session_id}': {e}")

