<div align="center">

<img src="https://img.shields.io/badge/NeuralOps-AI%20Incident%20Analysis-FF6B35?style=for-the-badge&logo=lightning&logoColor=white" alt="NeuralOps"/>

# 🧠 NeuralOps — AI Incident Analysis Platform

### Production-grade SRE incident analysis powered by LangGraph, Groq, BM25 RAG, and LangSmith

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-Workflow-FF6B35?style=flat-square&logo=chainlink&logoColor=white)](https://github.com/langchain-ai/langgraph)
[![Groq](https://img.shields.io/badge/Groq-llama--3.3--70b-F55036?style=flat-square&logo=meta&logoColor=white)](https://groq.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![LangSmith](https://img.shields.io/badge/LangSmith-LLMOps-1C3C3C?style=flat-square&logo=langchain&logoColor=white)](https://smith.langchain.com)
[![SQLite](https://img.shields.io/badge/SQLite-Knowledge%20Base-003B57?style=flat-square&logo=sqlite&logoColor=white)](https://sqlite.org)
[![License](https://img.shields.io/badge/License-MIT-22C55E?style=flat-square)](LICENSE)

<br/>

> **NeuralOps** is an internal AI tool for SRE and DevOps teams that analyzes production logs, identifies root causes using evidence-based AI reasoning, retrieves similar past incidents via BM25 RAG, and generates specific, actionable fix recommendations — all in under 5 seconds.

<br/>

**[Live Demo](https://neuralops-ratna.streamlit.app/) · [API Docs](http://localhost:8000/docs) · [Report Bug](issues) · [Request Feature](issues)**

</div>

---

## 📸 Screenshots

| Analyze Page | Dashboard | Observability |
|:---:|:---:|:---:|
| Evidence-based root cause | Incident history & stats | LLMOps metrics |

---

## 📋 Table of Contents

- [About the Project](#about-the-project)
- [System Architecture](#system-architecture)
- [Tech Stack](#tech-stack)
- [Features](#features)
- [Issues Fixed](#issues-fixed--engineering-challenges-solved)
- [What Built](#what-I-developed)
- [Getting Started](#getting-started)
- [API Reference](#api-reference)
- [Project Structure](#project-structure)
- [Testing](#testing)
- [LLMOps](#llmops--observability)
- [Author](#author)

---

## 🎯 About the Project

NeuralOps is a **production-grade AI incident analysis platform** designed to feel like a real internal DevOps tool — similar in spirit to PagerDuty or Datadog's AI assistant, but fully open and self-hosted.

**The problem it solves:**

When a production incident hits at 2AM, on-call engineers face:
- Hundreds of log lines with no clear signal
- No easy way to find similar past incidents
- Time-consuming manual root cause analysis
- Generic fix suggestions that don't apply to the specific failure

**NeuralOps automates this entire workflow:**

```
Raw Log Input
    → Structured Parsing
    → Similar Incident Retrieval (BM25 RAG)
    → Evidence-Based Root Cause Analysis
    → Confidence-Aware Fix Recommendations
    → Result in < 5 seconds
```

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER INPUT                              │
│                   (Raw Log / Stack Trace)                       │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FASTAPI BACKEND                              │
│              POST /api/analyze  (port 8000)                     │
│         Input Validation → min 10 chars enforced               │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                  LANGGRAPH WORKFLOW                             │
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────┐     │
│  │ parse_logs  │───▶│  retrieve   │───▶│    analyze      │     │
│  │             │    │             │    │                 │     │
│  │ Log Analyzer│    │BM25+PageIdx │    │ Root Cause Agent│     │
│  │ Agent       │    │ RAG Search  │    │ (Groq LLM)      │     │
│  └─────────────┘    └─────────────┘    └────────┬────────┘     │
│                                                 │               │
│                          ┌──────────────────────┤               │
│                          │   CONFIDENCE ROUTER  │               │
│                          │  conf ≥ 0.65 → FIX   │               │
│                          │  retries < 2 → RETRY  │               │
│                          │  else → ESCALATE     │               │
│                          └──────────────────────┘               │
│                          │            │           │              │
│                          ▼            ▼           ▼              │
│                   ┌──────────┐ ┌──────────┐ ┌──────────┐        │
│                   │generate  │ │  retry   │ │ escalate │        │
│                   │  _fix    │ │(enriched │ │ (human   │        │
│                   │Fix Agent │ │ prompt)  │ │ review)  │        │
│                   └──────────┘ └──────────┘ └──────────┘        │
└─────────────────────────┬───────────────────────────────────────┘
                          │
              ┌───────────┼───────────┐
              ▼           ▼           ▼
        ┌──────────┐ ┌──────────┐ ┌──────────────┐
        │  SQLite  │ │LangSmith │ │  JSON Result │
        │Knowledge │ │ Tracing  │ │  to Streamlit│
        │   Base   │ │ LLMOps   │ │     UI       │
        └──────────┘ └──────────┘ └──────────────┘
```

### Confidence Routing Logic

```
analyze node
    │
    ├── confidence ≥ 0.65  ──────────────▶  generate_fix  ──▶  END
    │
    ├── confidence < 0.65 AND retries < 2 ─▶  retry (enriched prompt)
    │                                              │
    │                                              └──▶  analyze (loop)
    │
    └── confidence < 0.65 AND retries ≥ 2 ─▶  escalate  ──▶  END
```

---

## 🛠️ Tech Stack

| Layer | Technology | Version | Purpose |
|---|---|---|---|
| **API** | FastAPI + Uvicorn | 0.111.0 | Async REST backend |
| **Orchestration** | LangGraph | Latest | Stateful AI workflow with routing |
| **LLM** | Groq (llama-3.3-70b-versatile) | Latest | Fast inference |
| **LLM Framework** | LangChain + LangChain-Core | 0.2.x | Prompt templates + chains |
| **RAG** | rank-bm25 + PageIndex | 0.2.2 | Vectorless keyword retrieval |
| **Knowledge Base** | SQLite | Built-in | Incident history storage |
| **LLMOps** | LangSmith | 0.1.x | Tracing, latency, confidence tracking |
| **Frontend** | Streamlit | 1.35.0 | Interactive dashboard |
| **Optional Agents** | CrewAI + AutoGen | Latest | Multi-agent extensions |
| **Runtime** | Python | 3.11 | Core language |
| **Containerization** | Docker + Compose | Latest | Deployment |

---

## ✨ Features

### Core Pipeline
- 🔍 **Log Parsing Agent** — Extracts `error_type`, `service_name`, `severity`, `summary` from any log format
- 🧠 **Evidence-Based Root Cause** — 3-sentence structured analysis with direct log evidence quoted
- 📚 **BM25 RAG Retrieval** — Vectorless semantic search with PageIndex severity boosting
- 🔧 **Confidence-Aware Fix Agent** — Specific tool-named fixes (e.g. HikariCP, EXPLAIN ANALYZE, cert-manager)
- 🔄 **Retry with Enriched Prompts** — Retry attempts use previous confidence as context hint
- 📊 **LangSmith Observability** — Every LLM call traced with latency and confidence

### UI Pages
- **Analyze** — Log input, pipeline steps, root cause, evidence block, similar incidents, fix recommendation
- **Dashboard** — Total incidents, avg confidence, avg latency, incident history table
- **Observability** — LLMOps config, severity breakdown chart
- **Architecture** — System flow, tech stack table, confidence routing table

### Safety & Reliability
- ✅ Confidence never returns 0 (minimum enforced at 0.15)
- ✅ Root cause never returns null or None
- ✅ Every field has null-safety fallback
- ✅ Full try/except wrapping at every pipeline node
- ✅ Input validation before processing (min 10 chars)
- ✅ Real latency tracking (not hardcoded 0.0)
- ✅ Reasoning field included in all responses

---

## 🐛 Issues Fixed — Engineering Challenges Solved

This project went through significant debugging and iteration. Here are the real engineering problems we solved:

### 1. 🔴 Confidence Always Returning 0%
**Root cause:** LLM returned confidence as integer (e.g. `75`) but code treated it as float (`0.75`), then when LLM returned `0` the minimum enforcement wasn't triggered.

**Fix:**
```python
# Detect both 0-100 and 0.0-1.0 scale automatically
if val > 1.0:
    val = val / 100.0  # convert 75 → 0.75
if val <= 0:
    val = CONF_MIN     # enforce minimum 0.15
conf = max(0.15, min(0.95, val))
```

### 2. 🔴 Root Cause Returning None / null
**Root cause:** LLM occasionally returned `null` for the root_cause field, and the code had no null guard.

**Fix:** Added safety defaults at every layer:
```python
result["root_cause"] = result.get("root_cause") or \
    "Insufficient data to determine root cause. Manual investigation required."
```

### 3. 🔴 LLM Hallucinating Root Cause (Overfitting)
**Problem:** Timeout error on `/predict` was incorrectly diagnosed as a DB connection pool issue because RAG context contained DB-related past incidents.

**Fix:** Added explicit anti-hallucination blocks in the system prompt:
```
NEVER assume DB issues unless log explicitly contains:
SQL, database, query, pool, JDBC, HikariCP, connection pool
RAG context is secondary — the log is the primary truth.
```

### 4. 🔴 Generic Fix Suggestions
**Problem:** Fix agent returned the same generic advice regardless of error type:
`"Increase pool size. Implement retries. Optimize queries."`

**Fix:** Added error→fix mapping with specific tool names:
```
pool exhausted → HikariCP maximumPoolSize, EXPLAIN ANALYZE, Redis caching
timed out URL  → kubectl top pods, Jaeger tracing, HPA autoscaling
OOM / heap     → jmap -dump, Eclipse MAT, Caffeine eviction policy
```

### 5. 🔴 Package Version Conflicts
**Problem:** `langchain==0.2.1` conflicted with `langchain-groq==0.1.3` causing `ResolutionImpossible`.

**Fix:** Removed strict version pins, installed with flexible versions so pip resolves compatibility automatically.

### 6. 🔴 Wrong LangChain Import Paths
**Problem:** `ModuleNotFoundError: No module named 'langchain.prompts'`

**Fix:** Updated all imports to LangChain-Core:
```python
# Old (broken)
from langchain.prompts import ChatPromptTemplate
# New (correct)
from langchain_core.prompts import ChatPromptTemplate
```

### 7. 🔴 Groq Model Decommissioned
**Problem:** `Error 400 — llama3-70b-8192 has been decommissioned`

**Fix:** Updated all agent files to `llama-3.3-70b-versatile`.

### 8. 🔴 Latency Always Showing 0.0
**Problem:** Latency was hardcoded as `0.0` in the pipeline output.

**Fix:** Real measurement using `time.time()`:
```python
pipeline_start = time.time()
# ... pipeline runs ...
latency = round(time.time() - pipeline_start, 3)
```

### 9. 🔴 Retry Not Changing LLM Behavior
**Problem:** Retry attempts sent the exact same prompt — LLM returned the same low-confidence result.

**Fix:** Added enriched retry prompt with previous confidence and a dynamic hint:
```python
RETRY_USER_PROMPT = """
RETRY #{retry_count} — Previous confidence was {prev_confidence_pct:.0f}%.
Retry hint: {retry_hint}
Re-evaluate more carefully...
"""
```

### 10. 🔴 RAG Overfitting (Double-Biasing)
**Problem:** RAG query was built as `raw_input + LLM_summary` — the LLM summary repeated the same keywords, biasing BM25 toward one incident type.

**Fix:**
```python
# Before (double-biased)
query = state["raw_input"] + " " + state["parsed_data"].get("summary", "")
# After (clean signal)
query = state["raw_input"]
```

### 11. 🔴 UI Dark Mode Override
**Problem:** Streamlit's dark mode overrode the custom peach theme — text was invisible.

**Fix:** Added explicit CSS targeting all Streamlit data-testid selectors to force background and text colors.

### 12. 🔴 Service Name Showing "unknown"
**Problem:** Service name wasn't extracted from URL paths like `http://localhost:8000/predict`.

**Fix:** Added URL path extraction rule:
```
URL path /predict → predict-service
URL path /order   → order-service
```

---

## 🚀 What Developed

### Phase 1 — Core Architecture
Built the full multi-agent pipeline from scratch:
- FastAPI backend with lifespan DB initialization
- LangGraph state machine with 6 nodes (parse → retrieve → analyze → fix/retry/escalate)
- SQLite knowledge base with 8 pre-seeded incident types
- BM25 vectorless RAG with PageIndex severity boosting

### Phase 2 — Agent Engineering
Designed and iterated all 3 agents:
- **Log Analyzer** — strict fact extraction, no inference, verbatim summary copying
- **Root Cause Agent** — evidence table with 12 signal→cause mappings, 3-sentence mandatory format, anti-hallucination blocks
- **Fix Agent** — confidence-aware routing, error→fix mapping with specific tool names (HikariCP, jmap, cert-manager, kubectl, Jaeger)

### Phase 3 — Production Hardening
Applied production-grade robustness:
- Confidence clamping system (0.15 minimum, 0.95 maximum)
- Null-safety at every node and final output
- Input validation before pipeline execution
- Real latency measurement
- LangSmith tracing integration
- Retry with enriched prompt strategy

### Phase 4 — UI & Observability
Built complete Streamlit interface:
- 4 pages: Analyze, Dashboard, Observability, Architecture
- Evidence block showing exact log quote
- Similar incidents from RAG displayed with similarity scores
- Reasoning field with expandable details
- Retry count metric in output
- Feedback loop (1-5 star rating per incident)

### Optional Extensions (Implemented, Not Active)
- **CrewAI** multi-agent crew (`agents/crewai_solver.py`)
- **AutoGen** collaborative agents (`agents/autogen_crew.py`)
- Both are wired to use `llama-3.3-70b-versatile` via Groq's OpenAI-compatible endpoint

---

## 🏁 Getting Started

### Prerequisites
- Python 3.11
- Groq API key → [console.groq.com](https://console.groq.com)
- LangSmith API key → [smith.langchain.com](https://smith.langchain.com)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/neuralops.git
cd neuralops

# 2. Create virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
cp .env.example .env
# Edit .env and add your API keys
```

### Environment Variables

```env
GROQ_API_KEY=your_groq_api_key_here
LANGCHAIN_API_KEY=your_langsmith_api_key_here
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=NeuralOps
```

### Running the Application

```bash
# Terminal 1 — Start FastAPI backend
python main.py
# API available at http://localhost:8000
# Interactive docs at http://localhost:8000/docs

# Terminal 2 — Start Streamlit frontend
streamlit run ui/streamlit_app.py
# UI available at http://localhost:8501
```

### Docker

```bash
cd deployment
docker-compose up --build
# API: http://localhost:8000
# UI:  http://localhost:8501
```

---

## 📡 API Reference

### POST /api/analyze

Runs the full LangGraph incident analysis pipeline.

**Request:**
```json
{
  "log_input": "SQLTimeoutException: connection pool exhausted after 30s\n  Service: payment-service"
}
```

**Response:**
```json
{
  "incident_id": "INC-20260430-A3F1B2C4",
  "root_cause": "The database connection pool is exhausted, preventing new requests from acquiring a connection. This is evidenced by 'connection pool exhausted after 30s' in the error output. This occurs when concurrent traffic exceeds pool capacity or slow queries hold connections too long.",
  "evidence": "connection pool exhausted after 30s",
  "reasoning": "Signal 'pool exhausted' matched → DB connection pool full → confidence 0.91",
  "fix_suggestion": "Immediate: Increase HikariCP maximumPoolSize (10 → 20), restart service\nShort-term: EXPLAIN ANALYZE slowest queries, set connectionTimeout=30000ms\nLong-term: Add Prometheus pool metrics, implement read replicas, add Redis caching",
  "fix_summary": "Increase connection pool size and optimize slow queries",
  "confidence": 0.91,
  "severity": "Critical",
  "service_name": "payment-service",
  "evaluation": "High",
  "similar_incidents": [
    {
      "service": "payment-service",
      "root_cause": "Database connection pool exhausted under sustained load spike.",
      "score": 1.274
    }
  ],
  "retry_count": 0,
  "latency": 2.847
}
```

### GET /api/incidents
Returns all past incidents from the knowledge base.

### GET /api/stats
Returns aggregate stats: total incidents, avg confidence, avg latency, severity breakdown.

### POST /api/feedback
```json
{
  "incident_id": "INC-20260430-A3F1B2C4",
  "rating": 5,
  "comment": "Accurate root cause"
}
```

### GET /health
```json
{"status": "healthy", "service": "NeuralOps"}
```

---

## 📁 Project Structure

```
neuralops/
│
├── agents/
│   ├── log_analyzer.py       # Extracts error_type, service, severity, summary
│   ├── root_cause.py         # Evidence-based root cause + confidence scoring
│   ├── fix_agent.py          # Confidence-aware fix recommendations
│   ├── tools.py              # Shared utilities (incident ID, confidence eval, sanitize)
│   ├── autogen_crew.py       # Optional: AutoGen multi-agent (not in main pipeline)
│   └── crewai_solver.py      # Optional: CrewAI multi-agent (not in main pipeline)
│
├── api/
│   ├── main.py               # FastAPI app with all routes
│   └── mock_engine.py        # Disabled stub (raises NotImplementedError)
│
├── chains/
│   ├── log_parsing_chain.py  # Standalone log parsing chain (for testing)
│   ├── enrichment_chain.py   # Standalone enrichment chain (for testing)
│   ├── solution_chain.py     # Standalone fix chain (for testing)
│   └── pipeline.py           # Runs all chains in sequence (for testing)
│
├── graph/
│   └── incident_graph.py     # LangGraph workflow — main orchestration engine
│
├── rag/
│   ├── vectorless_rag.py     # BM25Okapi search over incident corpus
│   ├── pageindex.py          # Severity + service metadata boosting
│   └── hybrid_retriever.py   # Combines BM25 + PageIndex, cached singleton
│
├── data/
│   ├── seed_db.py            # SQLite init, seed data, CRUD operations
│   └── incidents.db          # Auto-generated SQLite database
│
├── tests/
│   ├── test_all.py           # 13 unit tests (tools, BM25, PageIndex)
│   └── run_tests.py          # Test runner script
│
├── ui/
│   ├── streamlit_app.py      # 4-page Streamlit dashboard
│   └── demo.html             # Standalone HTML demo (optional)
│
├── deployment/
│   ├── Dockerfile            # Python 3.11 slim image
│   └── docker-compose.yml    # API + UI services
│
├── main.py                   # Root entrypoint → uvicorn api.main:app
├── requirements.txt          # All dependencies
├── .env.example              # Environment variable template
└── README.md                 # This file
```

---

## 🧪 Testing

```bash
# Run all unit tests
python tests/run_tests.py

# Verbose output
python tests/run_tests.py -v
```

**Test coverage:**

| Module | Tests | What's tested |
|---|---|---|
| `agents/tools.py` | 6 | Incident ID format, confidence evaluation, log sanitization |
| `rag/vectorless_rag.py` | 5 | BM25 retrieval accuracy, empty query, no match, empty docs |
| `rag/pageindex.py` | 3 | Severity boosting, service match boosting, field preservation |

No API keys required for unit tests.

---

## 📊 LLMOps & Observability

NeuralOps tracks every request with:

| Metric | How |
|---|---|
| **Latency** | `time.time()` at pipeline start and end |
| **Confidence** | LLM output, clamped 0.15–0.95 |
| **Retry count** | Number of retry attempts before fix/escalate |
| **LangSmith traces** | Full agent flow visible in Smith dashboard |
| **Evidence field** | Exact log quote that triggered the analysis |

Enable LangSmith tracing in `.env`:
```env
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_key
LANGCHAIN_PROJECT=NeuralOps
```

---

## 🤔 Why LangGraph over CrewAI / AutoGen?

| | LangGraph | CrewAI | AutoGen |
|---|---|---|---|
| **Latency** | 2–4s | 15–30s | 10–25s |
| **Routing control** | ✅ Deterministic | ❌ Agent-decided | ❌ Conversational |
| **Retry strategy** | ✅ Explicit nodes | ❌ Limited | ❌ Limited |
| **Production-ready** | ✅ Yes | ⚠️ Partial | ⚠️ Partial |
| **Best for** | Fixed pipelines | Open-ended tasks | Conversational tasks |

CrewAI and AutoGen are implemented as optional extensions (`agents/crewai_solver.py`, `agents/autogen_crew.py`) and can be connected for deeper multi-agent analysis on escalated incidents.

---

## 🎯 Sample Inputs to Test

```bash
# SQL Connection Pool
SQLTimeoutException: connection pool exhausted after 30s
  at com.zaxxer.hikari.pool.HikariPool.getConnection(HikariPool.java:213)
  Service: payment-service | Env: production

# Kubernetes Scheduling
0/3 nodes are available: 3 Insufficient cpu
  Warning FailedScheduling pod/api-gateway-6c9d4f
  Event: pod evicted due to resource pressure

# Out of Memory
java.lang.OutOfMemoryError: Java heap space
  at java.util.Arrays.copyOf(Arrays.java:3210)
  Service: recommendation-engine | Heap: 4GB/4GB

# API Timeout
[ERROR] TimeoutError: Request to http://localhost:8000/predict timed out after 5s
  Service: inference-api | Env: production

# Redis Eviction
WARN Redis eviction maxmemory-policy=allkeys-lru triggered
  ERROR Cache miss rate: 94%
  Service: user-session-service | Redis memory: 4096/4096 MB
```

---

## 👤 Author

**Ratnaprava Mohapatra**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077B5?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/ratnapravamohapatra/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=flat-square&logo=github)](https://github.com/RatnapravaMohapatra)

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

---

<div align="center">

**Built by Ratnaprava Mohapatra with ❤️ for production SRE teams**

*NeuralOps — AI that reasons from evidence, not assumptions*

</div>
