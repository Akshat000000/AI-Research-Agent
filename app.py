"""
app.py — Streamlit frontend for the AI Research Agent.

Provides a ChatGPT-style interface where users can type research topics.
Communicates with the FastAPI backend (http://localhost:8000) to run
the agent and display the results.
"""

import os
import time
import streamlit as st
import requests

# ── Configuration ─────────────────────────────────────────────────────────────
API_URL = "http://127.0.0.1:8000"
LOG_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(LOG_DIR, "debug_streamlit.log")

# ── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Research Agent",
    page_icon="🔍",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Custom Styling ────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main container styling */
    .stApp {
        background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
    }
    
    /* Header styling */
    .main-header {
        text-align: center;
        padding: 2rem 0 1rem 0;
    }
    .main-header h1 {
        background: linear-gradient(120deg, #a78bfa, #60a5fa, #34d399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0.3rem;
    }
    .main-header p {
        color: #94a3b8;
        font-size: 1.05rem;
    }
    
    /* Chat message styling */
    .stChatMessage {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 0.75rem;
    }
    
    /* Status container styling */
    .stStatus {
        background: rgba(99, 102, 241, 0.08);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 10px;
    }
    
    /* Input field styling */
    .stChatInput > div {
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 12px;
    }
    .stChatInput > div:focus-within {
        border-color: #a78bfa;
        box-shadow: 0 0 0 2px rgba(167, 139, 250, 0.2);
    }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🔍 AI Research Agent</h1>
    <p>Ask any research question — the agent will search the web, analyze sources, and deliver a comprehensive answer.</p>
</div>
""", unsafe_allow_html=True)

# ── Session State Initialization ──────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Display Chat History ──────────────────────────────────────────────────────
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar="🧑‍💻" if message["role"] == "user" else "🤖"):
        st.markdown(message["content"])

# ── Chat Input Handler ────────────────────────────────────────────────────────
if user_input := st.chat_input("Enter your research topic..."):
    # Display the user's message immediately
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(user_input)

    # Save to session state
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Run the agent via FastAPI backend
    with st.chat_message("assistant", avatar="🤖"):
        with st.status("🧠 Researching...", expanded=True) as status:
            st.write("🚀 Starting research on your topic...")

            start_time = time.time()

            try:
                response = requests.post(
                    f"{API_URL}/research",
                    json={"query": user_input},
                    timeout=300,  # Allow up to 5 minutes for deep research
                    proxies={"http": None, "https": None},  # Bypass system proxies for localhost
                )
                elapsed = time.time() - start_time

                # Log success
                with open(LOG_FILE, "a", encoding="utf-8") as lf:
                    lf.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] SUCCESS query: {user_input!r} in {elapsed:.2f}s\n")

                response.raise_for_status()
                result = response.json()

                # Display intermediate steps
                for step in result["steps"]:
                    if step["type"] == "tool_call":
                        if step["tool"] == "search_local_documents":
                            st.write(f"📂 **Searching local documents:** `{step['query']}`")
                        else:
                            st.write(f"🔎 **Searching the web:** `{step['query']}`")
                    elif step["type"] == "tool_result":
                        st.write(f"📄 **Found results** — analyzing content...")

                # Update status when done
                num_web = sum(1 for s in result["steps"] if s["type"] == "tool_call" and s.get("tool") == "web_search")
                num_local = sum(1 for s in result["steps"] if s["type"] == "tool_call" and s.get("tool") == "search_local_documents")
                parts = []
                if num_web:
                    parts.append(f"{num_web} web search(es)")
                if num_local:
                    parts.append(f"{num_local} local doc search(es)")
                summary = " and ".join(parts) if parts else "no searches needed"
                status.update(
                    label=f"✅ Research complete — {summary}",
                    state="complete",
                    expanded=False,
                )

                # Display the final response
                st.markdown(result["response"])

                # Save to session state
                st.session_state.messages.append({"role": "assistant", "content": result["response"]})

            except requests.exceptions.ConnectionError:
                elapsed = time.time() - start_time
                status.update(label="❌ Connection Error", state="error", expanded=True)
                st.error(
                    "**Could not connect to the backend API.**\n\n"
                    "Make sure the FastAPI server is running:\n"
                    "```bash\nuvicorn api:app --reload\n```"
                )
                with open(LOG_FILE, "a", encoding="utf-8") as lf:
                    lf.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] CONNECTION_ERROR after {elapsed:.2f}s for: {user_input!r}\n")

            except requests.exceptions.HTTPError as e:
                elapsed = time.time() - start_time
                # Extract the actual error detail from FastAPI's JSON response
                try:
                    error_detail = e.response.json().get("detail", str(e))
                except Exception:
                    error_detail = str(e)

                if e.response.status_code == 504:
                    status.update(label="⏰ Research Timed Out", state="error", expanded=True)
                    st.error(
                        f"**The research request timed out** ({elapsed:.0f}s).\n\n"
                        "**Suggestions:**\n"
                        "- Try a more specific or simpler query\n"
                        "- Check that your API keys are valid in the `.env` file\n"
                        "- Try again — the AI services may have been temporarily slow"
                    )
                else:
                    status.update(label="❌ API Error", state="error", expanded=True)
                    st.error(f"**Backend Error:** {error_detail}")

                with open(LOG_FILE, "a", encoding="utf-8") as lf:
                    lf.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] HTTP_{e.response.status_code} after {elapsed:.2f}s for: {user_input!r} — {error_detail}\n")

            except requests.exceptions.Timeout:
                elapsed = time.time() - start_time
                status.update(label="⏰ Request Timed Out", state="error", expanded=True)
                st.error(
                    f"**The request to the backend timed out** ({elapsed:.0f}s).\n\n"
                    "**Suggestions:**\n"
                    "- Try a more specific or simpler query\n"
                    "- Check that your API keys are valid in the `.env` file\n"
                    "- Try again — the AI services may have been temporarily slow"
                )
                with open(LOG_FILE, "a", encoding="utf-8") as lf:
                    lf.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] TIMEOUT after {elapsed:.2f}s for: {user_input!r}\n")

            except Exception as e:
                elapsed = time.time() - start_time
                status.update(label="❌ Error", state="error", expanded=True)
                st.error(f"An unexpected error occurred: {str(e)}")
                with open(LOG_FILE, "a", encoding="utf-8") as lf:
                    lf.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ERROR after {elapsed:.2f}s for: {user_input!r} — {str(e)}\n")
