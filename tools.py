"""
tools.py — Defines the tools (abilities) available to the AI Research Agent.

Currently provides:
- Web search via Tavily Search API with timeout and retry logic
- Local document search via ChromaDB vector database
"""

import os
import json
import concurrent.futures
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type

# Load environment variables BEFORE initializing any tools,
# so that TAVILY_API_KEY is available when TavilySearch is created.
load_dotenv(override=True)

# Initialize the underlying Tavily search client
# Note: TavilySearch does not support a request_timeout parameter directly,
# so we enforce timeouts at the execution level using ThreadPoolExecutor.
_tavily_search = TavilySearch(max_results=5)

# Timeout for each individual Tavily search call (seconds)
SEARCH_TIMEOUT = 25

# ── ChromaDB Configuration ───────────────────────────────────────────────────
CHROMA_DB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "chroma_db")
COLLECTION_NAME = "local_documents"


@tool
def web_search(query: str) -> str:
    """Search the web for the given query to find real-time, recent, or factual information."""

    def _do_search():
        """Execute the Tavily search with retry for transient errors."""
        @retry(
            retry=retry_if_exception_type((TimeoutError, ConnectionError, OSError)),
            stop=stop_after_attempt(3),
            wait=wait_fixed(3),
            reraise=True,
        )
        def _search_with_retry():
            return _tavily_search.invoke(query)

        return _search_with_retry()

    # Run in a thread with a strict timeout to prevent hanging
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_do_search)
        try:
            res = future.result(timeout=SEARCH_TIMEOUT)
            if isinstance(res, str):
                return res
            return json.dumps(res)
        except concurrent.futures.TimeoutError:
            return (
                "Error: The web search timed out after "
                f"{SEARCH_TIMEOUT} seconds. Please try a more specific query."
            )
        except ConnectionError:
            return "Error: Could not connect to the search service. Please check your internet connection."
        except Exception as e:
            return f"Error executing search: {str(e)}"


@tool
def search_local_documents(query: str) -> str:
    """Search through locally indexed documents (PDFs, text files, markdown) stored in the ChromaDB vector database.
    Use this tool when the user asks about content from their own files, internal documents, or custom knowledge base."""
    try:
        # Lazy imports to avoid loading ChromaDB if the tool is never called
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        from langchain_chroma import Chroma

        # Check if the database directory exists
        if not os.path.exists(CHROMA_DB_DIR):
            return (
                "No local document index found. "
                "Please run 'python index_docs.py' to index your documents first."
            )

        # Connect to the persistent ChromaDB store
        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
        vector_store = Chroma(
            persist_directory=CHROMA_DB_DIR,
            embedding_function=embeddings,
            collection_name=COLLECTION_NAME,
        )

        # Perform similarity search — return top 4 most relevant chunks
        results = vector_store.similarity_search_with_relevance_scores(query, k=4)

        if not results:
            return "No relevant information found in the local documents for this query."

        # Format results with source citations
        formatted_results = []
        for i, (doc, score) in enumerate(results, 1):
            source = doc.metadata.get("source", "unknown")
            formatted_results.append(
                f"--- Result {i} (relevance: {score:.2f}) ---\n"
                f"Source: {source}\n"
                f"Content:\n{doc.page_content}\n"
            )

        return "\n".join(formatted_results)

    except Exception as e:
        return f"Error searching local documents: {str(e)}"


# Bundle all tools into a list for the agent to use.
tools = [web_search, search_local_documents]

