"""
tools.py — Defines the tools (abilities) available to the AI Research Agent.

Currently provides:
- Web search via Tavily Search API with timeout and retry logic
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


# Bundle all tools into a list for the agent to use.
tools = [web_search]
