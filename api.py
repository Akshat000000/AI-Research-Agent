"""
api.py — FastAPI backend for the AI Research Agent.

Exposes REST API endpoints:
  - POST /research  → Runs the LangGraph agent and returns the response
  - GET  /health    → Health check
"""

import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agent import run_agent

# ── FastAPI App ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="AI Research Agent API",
    description="An autonomous research agent powered by Gemini and LangGraph",
    version="1.0.0",
)

# ── CORS Middleware ───────────────────────────────────────────────────────────
# Allow Streamlit (port 8501) to call this backend (port 8000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response Models ─────────────────────────────────────────────────
class ResearchRequest(BaseModel):
    query: str


class StepInfo(BaseModel):
    type: str
    tool: str | None = None
    query: dict | None = None
    content: str | None = None


class ResearchResponse(BaseModel):
    response: str
    steps: list[StepInfo]


# ── Configuration ─────────────────────────────────────────────────────────────
API_TIMEOUT_SECONDS = 180  # Max time for the entire request


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
def health_check():
    """Simple health check to verify the API is running."""
    return {"status": "healthy", "service": "AI Research Agent"}


@app.post("/research", response_model=ResearchResponse)
async def research(request: ResearchRequest):
    """
    Run the AI Research Agent on a given query.

    The agent will:
    1. Search the web using Tavily
    2. Analyze the results with Gemini
    3. Optionally search again if more info is needed
    4. Return a comprehensive, well-cited answer
    """
    try:
        # Run the blocking agent call in a thread pool with an async timeout
        result = await asyncio.wait_for(
            asyncio.to_thread(run_agent, request.query),
            timeout=API_TIMEOUT_SECONDS,
        )

        return ResearchResponse(
            response=result["response"],
            steps=[StepInfo(**step) for step in result["steps"]],
        )
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=504,
            detail=(
                "The research request timed out after "
                f"{API_TIMEOUT_SECONDS} seconds. "
                "Please try a more specific query or try again later."
            ),
        )
    except TimeoutError as e:
        # Raised by run_agent's internal timeout
        raise HTTPException(status_code=504, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Agent error: {str(e)}"
        )
