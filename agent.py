"""
agent.py — The brain of the AI Research Agent.

Builds a LangGraph state machine that loops between:
  1. Agent node: Gemini reads the conversation and decides what to do.
  2. Tools node: Executes the requested tool (e.g., Tavily web search).

The loop continues until Gemini decides it has enough information
to provide a final answer, at which point the graph ends.
"""

import os
import datetime
import concurrent.futures
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception

from tools import tools

# Load environment variables from .env file
load_dotenv(override=True)

# ── LLM Setup ────────────────────────────────────────────────────────────────
# Initialize Gemini with tool-calling capabilities.
# temperature=0 makes responses deterministic and focused (ideal for research).
# max_retries=3 handles rate limit (429) errors by retrying with backoff.
# gemini-2.5-flash: 15 RPM, 1500 RPD, 1M TPM on free tier. Excellent tool-calling.
# Note: gemini-2.0-flash and 2.0-flash-lite were discontinued on June 1, 2026.
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    max_retries=3,
    timeout=60,
)

# Bind our tools to the LLM so Gemini knows what tools are available
# and can generate structured tool-call requests.
llm_with_tools = llm.bind_tools(tools)


# ── Custom Retry Logic for Transient Errors (429, 503, 504, Timeouts) ──────────
def is_transient_error(exception):
    err_msg = str(exception).lower()
    transient_indicators = [
        "429", "resource_exhausted", "quota",
        "504", "deadline_exceeded", "deadline expired",
        "503", "unavailable", "timeout", "connection"
    ]
    return any(indicator in err_msg for indicator in transient_indicators)

def log_retry(retry_state):
    print(f"⚠️ Gemini API transient error hit. Retrying in {retry_state.next_action.sleep:.2f} seconds... (Attempt {retry_state.attempt_number})")

@retry(
    retry=retry_if_exception(is_transient_error),
    stop=stop_after_attempt(4),
    wait=wait_random_exponential(multiplier=1.5, min=3, max=15),
    before_sleep=log_retry,
    reraise=True
)
def invoke_llm_with_retry(messages):
    return llm_with_tools.invoke(messages)


# ── System Prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert AI Research Agent. Your job is to thoroughly 
research any topic the user asks about.

CRITICAL INSTRUCTIONS:
- You have TWO search tools available:
  1. **web_search**: Use this to find real-time, recent, or publicly available information from the internet.
  2. **search_local_documents**: Use this to search through the user's locally indexed documents (PDFs, text files, markdown files stored in their knowledge base).

TOOL SELECTION GUIDELINES:
- If the user asks about their own files, internal documents, company policies, or custom knowledge base, use **search_local_documents** first.
- If the user asks about public facts, recent news, general knowledge, or real-time data, use **web_search**.
- If the query could benefit from both local context AND web information, use BOTH tools.
- For general knowledge, programming concepts, or historical facts you are confident about, use your own internal knowledge to answer quickly without any tools.

RESPONSE GUIDELINES:
- If you perform a web search, always cite your sources by including the URLs.
- If you retrieve information from local documents, cite the source file name.
- Use markdown formatting (headers, bullet points, bold text) to make your response easy to read.
- If the topic is broad, break it down into sub-topics and research each one.
- IMPORTANT: Be efficient. Aim to complete your research in 2-3 tool calls maximum. Do not search more than necessary.
"""


# ── Graph Nodes ───────────────────────────────────────────────────────────────
def agent_node(state: MessagesState):
    """
    The 'thinking' node. Sends the full conversation history to Gemini
    and receives either a tool-call request or a final text response.
    """
    # Get current date from system clock for temporal grounding
    current_date = datetime.date.today().strftime("%B %d, %Y")
    system_prompt_with_date = f"{SYSTEM_PROMPT}\n\nCURRENT CONTEXT:\n- Today's date is {current_date}."
    
    # Prepend the system prompt to guide Gemini's behavior
    messages = [{"role": "system", "content": system_prompt_with_date}] + state["messages"]
    response = invoke_llm_with_retry(messages)
    return {"messages": [response]}


# The 'doing' node. Automatically executes any tool calls requested by Gemini.
tools_node = ToolNode(tools=tools)


# ── Build the Graph ───────────────────────────────────────────────────────────
# Create the state graph with MessagesState (tracks conversation messages).
graph_builder = StateGraph(MessagesState)

# Add nodes
graph_builder.add_node("agent", agent_node)
graph_builder.add_node("tools", tools_node)

# Add edges
# START → agent: The graph always begins at the agent node.
graph_builder.add_edge(START, "agent")

# agent → tools OR END: After the agent runs, check if it requested a tool.
#   - If yes → go to "tools" node to execute the search.
#   - If no  → go to END (Gemini has provided its final answer).
graph_builder.add_conditional_edges("agent", tools_condition)

# tools → agent: After the tool executes, return to the agent so Gemini
# can read the results and decide what to do next.
graph_builder.add_edge("tools", "agent")

# Compile the graph into a runnable object.
graph = graph_builder.compile()

# ── Global Timeout for Agent Execution ────────────────────────────────────────
AGENT_TIMEOUT_SECONDS = 120  # Maximum time for the entire agent run


# ── Public Interface ──────────────────────────────────────────────────────────
def run_agent(user_message: str, chat_history: list = None):
    """
    Run the research agent with a user message and optional chat history.

    Args:
        user_message: The user's research query.
        chat_history: Previous messages for multi-turn conversation.

    Returns:
        A dict with:
          - "response": The final text answer from the agent.
          - "steps": A list of intermediate steps (tool calls, results)
                     for displaying the agent's thought process.
    """
    # Build the input messages
    if chat_history:
        messages = chat_history + [{"role": "user", "content": user_message}]
    else:
        messages = [{"role": "user", "content": user_message}]

    # Run the agent graph with a global timeout to prevent hanging
    def _execute_graph():
        steps = []
        final_response = ""

        # Stream through the graph execution, limiting loops to prevent timeouts
        for event in graph.stream({"messages": messages}, config={"recursion_limit": 6}):
            for node_name, node_output in event.items():
                if node_name == "agent":
                    msg = node_output["messages"][-1]
                    # Check if the agent decided to call a tool
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        for tool_call in msg.tool_calls:
                            steps.append({
                                "type": "tool_call",
                                "tool": tool_call["name"],
                                "query": tool_call["args"],
                            })
                    else:
                        # No tool calls means this is the final answer
                        content = msg.content
                        if isinstance(content, str):
                            final_response = content
                        elif isinstance(content, list):
                            parts = []
                            for part in content:
                                if isinstance(part, str):
                                    parts.append(part)
                                elif isinstance(part, dict) and "text" in part:
                                    parts.append(part["text"])
                            final_response = "".join(parts)
                        else:
                            final_response = str(content)

                elif node_name == "tools":
                    # Tool execution results
                    for tool_msg in node_output["messages"]:
                        steps.append({
                            "type": "tool_result",
                            "content": tool_msg.content[:500],  # Truncate for display
                        })

        return {"response": final_response, "steps": steps}

    # Execute with a global timeout
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_execute_graph)
        try:
            return future.result(timeout=AGENT_TIMEOUT_SECONDS)
        except concurrent.futures.TimeoutError:
            raise TimeoutError(
                f"Research took longer than {AGENT_TIMEOUT_SECONDS} seconds. "
                "Try a more specific query or try again later."
            )
