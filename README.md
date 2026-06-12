# 🔍 AI Research Agent

An autonomous AI Research Agent that searches the web, analyzes multiple sources, and delivers comprehensive, well-cited answers to any research question.

Built with **Google Gemini**, **LangGraph**, **Tavily Search**, **FastAPI**, and **Streamlit**.

---

## 🌟 What is this project?

This project is a powerful example of both **Generative AI (GenAI)** and **Agentic AI**:
- **Generative AI:** It uses Large Language Models (LLMs) like Google Gemini to understand complex questions, reason through them, and generate human-readable text.
- **Agentic AI:** Unlike a standard chatbot that just answers in a single turn, this is an **Agent**. It acts autonomously, breaks down tasks, and uses external tools to achieve a goal. It operates in a loop: thinking, acting, observing the results, and deciding what to do next.

## 🛠️ The Brain and The Hands

To understand how this Agent works, think of the **LLM (Gemini) as a Manager**, and **Tavily Search as a Researcher**:

1. **The LLM (Gemini) does NOT browse the web directly.** It acts as the brain. When you ask a question, the LLM decides *if* a search is needed and formulates the perfect search query.
2. **Tavily acts as the Researcher.** It is a specialized Search Engine and API built for AI Agents. It's not just a web browser; it intelligently scrapes, filters, and extracts factual text from the web.
3. **The Synthesis:** Tavily brings back the raw data to the LLM. The LLM reads all the sources, cross-checks the facts, and synthesizes a polished, final report with citations for you.

---

## ⚙️ How It Works (The Agent Loop)

The core magic of this project is the **Agent Loop** powered by LangGraph. Gemini doesn't just search once. It can search multiple times, refining its queries based on what it learns, until it has enough information to give a thorough answer.

```text
User asks a question (Streamlit UI)
        ↓
   Streamlit sends HTTP request to FastAPI backend
        ↓
   FastAPI calls the LangGraph agent
        ↓
   Agent (Gemini) reads the question and decides what to search
        ↓
   Tavily searches the web and returns results
        ↓
   Agent reads the results — needs more info? → searches again
                           — has enough?      → writes final answer
        ↓
   FastAPI returns the response as JSON
        ↓
   Streamlit displays the answer with citations
```

---

## 🏗️ Architecture & Project Structure

```text
┌─────────────────────┐       HTTP        ┌─────────────────────┐
│   Streamlit (UI)    │  ◄──────────────► │   FastAPI (API)     │
│   Port 8501         │   POST /research  │   Port 8000         │
└─────────────────────┘                   └────────┬────────────┘
                                                   │
                                          ┌────────▼────────────┐
                                          │   LangGraph Agent   │
                                          │   (agent.py)        │
                                          │                     │
                                          │  ┌───────┐  ┌─────┐ │
                                          │  │Gemini │◄►│Tools│ │
                                          │  │ Node  │  │Node │ │
                                          │  └───────┘  └─────┘ │
                                          └─────────────────────┘
```

**Project Files:**
*   `tools.py`: Defines the Agent's abilities (specifically the Tavily web search tool).
*   `agent.py`: Contains the LangGraph state machine, defining how Gemini interacts with tools in a loop.
*   `api.py`: The FastAPI backend that exposes the `/research` endpoint.
*   `app.py`: The Streamlit frontend that provides a clean UI and calls the FastAPI backend via HTTP.

---

## 🚀 Setup & Installation

### 1. Prerequisites
- **Python 3.10+** installed on your system.
- A **Google AI Studio API key** (for Gemini): Get it at [aistudio.google.com](https://aistudio.google.com/)
- A **Tavily API key** (for web search): Get it at [tavily.com](https://tavily.com/)

### 2. Install Dependencies
Clone the repository and install the required Python packages:
```bash
pip install -r requirements.txt
```

### 3. Configure API Keys
Open the `.env` file in the root directory and replace the placeholder values with your actual keys:
```env
GOOGLE_API_KEY=your_actual_gemini_key
TAVILY_API_KEY=your_actual_tavily_key
```

### 4. Run the Application
You will need to run the backend and frontend in two separate terminal windows.

**Terminal 1 — Start the Backend:**
```bash
uvicorn api:app --reload
```
*The FastAPI backend will run at `http://localhost:8000`*

**Terminal 2 — Start the Frontend:**
```bash
streamlit run app.py
```
*The Streamlit UI will open automatically in your browser at `http://localhost:8501`*

---

## 🔌 API Endpoints

The FastAPI backend provides the following endpoints:

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/research` | Run the research agent. Body: `{"query": "your question"}` |
| `GET` | `/health` | Health check to verify the API is running. Returns `{"status": "healthy"}` |
| `GET` | `/docs` | Auto-generated Swagger UI for API documentation and testing |

---

## 💻 Technologies Used

| Technology | Purpose |
|---|---|
| **Google Gemini 2.0 Flash** | The core LLM for reasoning, planning, tool usage, and answer generation. |
| **LangGraph** | The state machine framework that manages the autonomous agent loop and memory. |
| **Tavily Search API** | An AI-optimized search engine that scrapes and extracts relevant web data. |
| **FastAPI** | A modern, fast web framework for building the backend REST API. |
| **Streamlit** | A rapid-development framework for building the interactive chat UI. |
| **LangChain** | The integration layer connecting Gemini with tools and LangGraph. |

---

## 💡 Example Usage

**User:** *"What are the latest breakthroughs in solid-state batteries in 2025?"*

**Behind the scenes, the Agent will:**
1. Realize it needs current information and format a query: `"solid state battery breakthroughs 2025"`.
2. Use the Tavily tool to search the web.
3. Read the extracted web content. If it finds a specific company mentioned but needs more details, it might autonomously run a *second* search for that company.
4. Compile a comprehensive, factual answer, synthesizing all the sources, and present it in the UI with proper citations.

---

## 📄 License
This project is for educational and portfolio purposes.
