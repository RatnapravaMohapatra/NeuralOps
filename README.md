cat > /home/claude/README.md << 'READMEEOF'
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
[![Render](https://img.shields.io/badge/Deployed-Render-46E3B7?style=flat-square&logo=render&logoColor=white)](https://render.com)
[![Status](https://img.shields.io/badge/Status-Work%20In%20Progress-FFB800?style=flat-square&logo=githubactions&logoColor=white)]()
[![License](https://img.shields.io/badge/License-MIT-22C55E?style=flat-square)](LICENSE)

<br/>

> **NeuralOps** is an internal AI tool built for SRE and DevOps teams that analyzes production logs,
> identifies root causes using evidence-based AI reasoning, retrieves similar past incidents via BM25 RAG,
> and generates specific, actionable fix recommendations — all in under 5 seconds.

<br/>

> ⚠️ **This project is actively under development. Features, prompts, and architecture are continuously being improved.**

<br/>

**[Live API](https://neuralops-api.onrender.com/health) · [API Docs](https://neuralops-api.onrender.com/docs) · [Live UI](https://neuralops-ui.onrender.com) · [Report Bug](../../issues) · [Request Feature](../../issues)**

</div>

---

## 📋 Table of Contents

- [About the Project](#-about-the-project)
- [System Architecture](#-system-architecture)
- [Tech Stack](#-tech-stack)
- [Features](#-features)
- [Issues Fixed](#-issues-fixed--engineering-challenges-solved)
- [What I Built](#-what-i-built)
- [Getting Started](#-getting-started)
- [Deployment on Render](#-deployment-on-render)
- [API Reference](#-api-reference)
- [Project Structure](#-project-structure)
- [Testing](#-testing)
- [LLMOps](#-llmops--observability)
- [Work In Progress](#-work-in-progress)
- [Author](#-author)

---

## 🎯 About the Project

NeuralOps is a **production-grade AI incident analysis platform** I built to feel like a real internal DevOps tool — similar in spirit to PagerDuty or Datadog's AI assistant, but fully open and self-hosted.

**The problem it solves:**

When a production incident hits at 2AM, on-call engineers face:
- Hundreds of log lines with no clear signal
- No easy way to find similar past incidents
- Time-consuming manual root cause analysis
- Generic fix suggestions that don't apply to the specific failure

**NeuralOps automates this entire workflow:**

```
Raw Log Input
    → Input Validation
    → Structured Log Parsing (LLM)
    → Similar Incident Retrieval (BM25 RAG)
    → Evidence-Based Root Cause Analysis (LLM)
    → Confidence-Aware Fix Recommendations (LLM)
    → Structured JSON Result in < 5 seconds
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
│         Input Validation → minimum 10 characters enforced      │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                  LANGGRAPH WORKFLOW                             │
│                                                                 │
│  ┌─────────────┐   ┌──────────────┐   ┌──────────────────┐     │
│  │ parse_logs  │──▶│   retrieve   │──▶│     analyze      │     │
│  │             │   │              │   │                  │     │
│  │ Log Analyzer│   │ BM25+PageIdx │   │ Root Cause Agent │     │
│  │   Agent     │   │  RAG Search  │   │   (Groq LLM)     │     │
│  └─────────────┘   └──────────────┘   └────────┬─────────┘     │
│                                                │               │
│                         ┌──────────────────────┤               │
│                         │   CONFIDENCE ROUTER  │               │
│                         │  conf ≥ 0.65 → FIX   │               │
│                         │  retries < 2 → RETRY │               │
│                         │  else → ESCALATE     │               │
│                         └──────────────────────┘               │
│                         │            │           │              │
│                         ▼            ▼           ▼              │
│                  ┌──────────┐ ┌──────────┐ ┌──────────┐        │
│                  │generate  │ │  retry   │ │ escalate │        │
│                  │  _fix    │ │(enriched │ │ (human   │        │
│                  │Fix Agent │ │ prompt)  │ │ review)  │        │
│                  └──────────┘ └──────────┘ └──────────┘        │
└─────────────────────────┬───────────────────────────────────────┘
                          │
              ┌───────────┼───────────┐
              ▼           ▼           ▼
        ┌──────────┐ ┌──────────┐ ┌────────────────┐
        │  SQLite  │ │LangSmith │ │  JSON Result   │
        │Knowledge │ │ Tracing  │ │  to Streamlit  │
        │   Base   │ │ LLMOps   │ │      UI        │
        └──────────┘ └──────────┘ └────────────────┘
```

### Confidence Routing Logic

```
analyze node
    │
    ├── confidence ≥ 0.65  ──────────────────▶  generate_fix  ──▶  END
    │
    ├── confidence < 0.65 AND retries < 2  ──▶  retry (enriched prompt)
    │                                                │
    │                                                └──▶  analyze (loop)
    │
    └── confidence < 0.65 AND retries ≥ 2  ──▶  escalate  ──▶  END
```

---

## 🛠️ Tech Stack

| Layer | Technology | Version | Purpose |
|---|---|---|---|
| **API** | FastAPI + Uvicorn | 0.111.0 | Async REST backend |
| **Orchestration** | LangGraph | Latest | Stateful AI workflow with routing |
| **LLM** | Groq (llama-3.3-70b-versatile) | Latest | Fast inference engine |
| **LLM Framework** | LangChain + LangChain-Core | 0.2.x | Prompt templates and chains |
| **RAG** | rank-bm25 + PageIndex | 0.2.2 | Vectorless keyword retrieval |
| **Knowledge Base** | SQLite | Built-in | Incident history storage |
| **LLMOps** | LangSmith | 0.1.x | Tracing, latency, confidence tracking |
| **Frontend** | Streamlit | 1.35.0 | Interactive dashboard |
| **Optional Agents** | CrewAI + AutoGen | Latest | Multi-agent extensions |
| **Deployment** | Render | — | Cloud hosting |
| **Runtime** | Python | 3.11 | Core language |
| **Containerization** | Docker + Compose | Latest | Local deployment |

---

## ✨ Features

### Core Pipeline
- 🔍 **Log Parsing Agent** — Extracts `error_type`, `service_name`, `severity`, `summary` from any log format
- 🧠 **Evidence-Based Root Cause** — 3-sentence structured analysis with direct log evidence quoted
- 📚 **BM25 RAG Retrieval** — Vectorless semantic search with PageIndex severity boosting
- 🔧 **Confidence-Aware Fix Agent** — Specific tool-named fixes (HikariCP, EXPLAIN ANALYZE, cert-manager, kubectl)
- 🔄 **Retry with Enriched Prompts** — Retry attempts use previous confidence score as context hint
- 📊 **LangSmith Observability** — Every LLM call traced with latency and confidence

### UI Pages
- **Analyze** — Log input, pipeline steps, root cause, evidence block, similar incidents, fix recommendation
- **Dashboard** — Total incidents, avg confidence, avg latency, incident history table
- **Observability** — LLMOps config, severity breakdown chart
- **Architecture** — System flow diagram, tech stack table, confidence routing table

### Safety and Reliability
- ✅ Confidence never returns 0 (minimum enforced at 0.15)
- ✅ Root cause never returns null or None
- ✅ Every field has null-safety fallback at every pipeline node
- ✅ Full try/except wrapping at every node
- ✅ Input validation before processing (min 10 chars)
- ✅ Real latency tracking using `time.time()` (not hardcoded 0.0)
- ✅ `reasoning` field included in every response
- ✅ `evidence` field shows exact log quote

---

## 🐛 Issues Fixed — Engineering Challenges Solved

This project went through significant debugging and iteration. Here are the real engineering problems I solved:

### 1. 🔴 Confidence Always Returning 0%

**Root cause:** LLM returned confidence as integer (e.g. `75`) but code treated it as float (`0.75`). When LLM returned `0`, minimum enforcement wasn't triggered.

**Fix:**
```python
# Detect both 0-100 and 0.0-1.0 scale automatically
if val > 1.0:
    val = val / 100.0  # convert 75 → 0.75
if val <= 0:
    val = CONF_MIN     # enforce minimum 0.15
conf = max(0.15, min(0.95, val))
```

---

### 2. 🔴 Root Cause Returning None / null

**Root cause:** LLM occasionally returned `null` for the root_cause field, and the code had no null guard downstream.

**Fix:** Added safety defaults at every layer:
```python
result["root_cause"] = result.get("root_cause") or \
    "Insufficient data to determine root cause. Manual investigation required."
```

---

### 3. 🔴 LLM Hallucinating Root Cause (Overfitting to RAG)

**Problem:** Timeout error on `/predict` was incorrectly diagnosed as a DB connection pool issue because RAG context contained DB-related past incidents.

**Fix:** Added explicit anti-hallucination blocks in the system prompt:
```
NEVER assume DB issues unless log explicitly contains:
SQL, database, query, pool, JDBC, HikariCP, connection pool
RAG context is SECONDARY — the log is the PRIMARY truth.
```

---

### 4. 🔴 Generic Fix Suggestions

**Problem:** Fix agent returned identical boilerplate regardless of error type:
`"Increase pool size. Implement retries. Optimize queries."`

**Fix:** Added error→fix mapping with specific tool names per error type:
```
pool exhausted → HikariCP maximumPoolSize, EXPLAIN ANALYZE, Redis caching layer
timed out URL  → kubectl top pods, Jaeger tracing, HPA autoscaling
OOM / heap     → jmap -dump, Eclipse MAT, Caffeine eviction policy
disk full      → find /var/log -mtime +7 -delete, logrotate config
```

---

### 5. 🔴 Package Version Conflicts on Install

**Problem:** `langchain==0.2.1` conflicted with `langchain-groq==0.1.3` causing `ResolutionImpossible` error.

**Fix:** Removed strict version pins and installed with flexible versions so pip resolves compatibility automatically.

---

### 6. 🔴 Wrong LangChain Import Paths

**Problem:** `ModuleNotFoundError: No module named 'langchain.prompts'` after installing newer LangChain.

**Fix:**
```python
# Old (broken in newer versions)
from langchain.prompts import ChatPromptTemplate
# Fixed
from langchain_core.prompts import ChatPromptTemplate
```

---

### 7. 🔴 Groq Model Decommissioned Mid-Development

**Problem:** `Error 400 — llama3-70b-8192 has been decommissioned and is no longer supported.`

**Fix:** Updated all agent files to use `llama-3.3-70b-versatile`.

---

### 8. 🔴 Latency Always Showing 0.0

**Problem:** Latency was hardcoded as `0.0` in the final pipeline output.

**Fix:** Real measurement using `time.time()`:
```python
pipeline_start = time.time()
# ... pipeline executes ...
latency = round(time.time() - pipeline_start, 3)
result["latency"] = latency
```

---

### 9. 🔴 Retry Logic Not Changing LLM Behavior

**Problem:** Retry attempts sent the exact same prompt — LLM returned the same low-confidence result every time.

**Fix:** Added enriched retry prompt with previous confidence score and a dynamic hint:
```python
RETRY_USER_PROMPT = """
RETRY #{retry_count} — Previous confidence was {prev_confidence_pct:.0f}%.
Re-evaluate more carefully. Retry hint: {retry_hint}
"""
```

---

### 10. 🔴 RAG Overfitting (Double-Biasing Query)

**Problem:** RAG query was built as `raw_input + LLM_summary`. Since the LLM summary repeated the same keywords from raw input, BM25 was biased toward the same incident type repeatedly.

**Fix:**
```python
# Before — double-biased
query = state["raw_input"] + " " + state["parsed_data"].get("summary", "")
# After — clean, single signal
query = state["raw_input"]
```

---

### 11. 🔴 Streamlit Dark Mode Overriding Custom Theme

**Problem:** Streamlit's default dark mode overrode the custom theme — text was invisible, layout broken.

**Fix:** Explicitly targeted all Streamlit `data-testid` selectors in CSS to force background and text colors regardless of browser theme.

---

### 12. 🔴 Service Name Always Showing "unknown"

**Problem:** When logs contained URLs like `http://localhost:8000/predict`, the service name was not being extracted.

**Fix:** Added URL path extraction rule in the log analyzer system prompt:
```
URL path /predict → predict-service
URL path /order   → order-service
Class name OrderService → order-service
```

---

### 13. 🔴 Confidence Routing Escalating Too Aggressively

**Problem:** Threshold was set to `0.8` — most analyses were being retried or escalated instead of generating fixes, even when confidence was reasonable.

**Fix:** Lowered threshold to `0.65` after calibrating against real outputs:
```python
CONFIDENCE_THRESHOLD = 0.65  # was 0.80
```

---

## 🚀 What I Built

### Phase 1 — Core Architecture
I built the full multi-agent pipeline from scratch including the FastAPI backend with lifespan DB initialization, LangGraph state machine with 6 nodes (parse → retrieve → analyze → fix/retry/escalate), SQLite knowledge base pre-seeded with 8 real incident types, and BM25 vectorless RAG with PageIndex severity boosting.

### Phase 2 — Agent Engineering
I designed and iterated all 3 core agents through multiple rounds of prompt engineering. The **Log Analyzer** uses strict fact extraction rules with no inference allowed and copies summary lines verbatim. The **Root Cause Agent** uses an evidence table with 12 signal→cause mappings, enforces a mandatory 3-sentence output format, and has explicit anti-hallucination blocks preventing DB/memory assumptions without log evidence. The **Fix Agent** is confidence-aware and maps each error type to specific tool names (HikariCP, jmap, cert-manager, kubectl, Jaeger).

### Phase 3 — Production Hardening
I applied production-grade robustness by building a confidence clamping system (0.15 minimum, 0.95 maximum), adding null-safety fallbacks at every node and in the final output, implementing input validation before pipeline execution, measuring real latency with `time.time()`, integrating LangSmith tracing, and designing an enriched retry prompt strategy that passes previous confidence as context.

### Phase 4 — UI and Observability
I built a complete 4-page Streamlit interface with an evidence block that shows the exact log quote that triggered the analysis, similar incidents from RAG displayed with similarity scores, a reasoning field with expandable details, a retry count metric, and a feedback loop (1–5 star rating per incident).

### Phase 5 — Deployment
I deployed the full stack on Render with the FastAPI backend and Streamlit UI as separate web services, environment variables managed through the Render dashboard, and LangSmith tracing active in production.

### Optional Extensions
I implemented **CrewAI** multi-agent crew (`agents/crewai_solver.py`) and **AutoGen** collaborative agents (`agents/autogen_crew.py`) as optional extensions. Both use `llama-3.3-70b-versatile` via Groq's OpenAI-compatible endpoint. They are built and available but not connected to the main pipeline — LangGraph is faster and more deterministic for this use case.

---

## 🏁 Getting Started

### Prerequisites
- Python 3.11
- Groq API key → [console.groq.com](https://console.groq.com)
- LangSmith API key → [smith.langchain.com](https://smith.langchain.com)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/ratnaprava/neuralops.git
cd neuralops

# 2. Create virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
cp .env.example .env
# Open .env and fill in your API keys
```

### Environment Variables

```env
GROQ_API_KEY=your_groq_api_key_here
LANGCHAIN_API_KEY=your_langsmith_api_key_here
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=NeuralOps
```

### Running Locally

```bash
# Terminal 1 — FastAPI backend
python main.py
# API: http://localhost:8000
# Docs: http://localhost:8000/docs

# Terminal 2 — Streamlit frontend
streamlit run ui/streamlit_app.py
# UI: http://localhost:8501
```

### Docker

```bash
cd deployment
docker-compose up --build
```

---

## ☁️ Deployment on Render

Both services are deployed on Render's free tier.

| Service | URL |
|---|---|
| **FastAPI API** | https://neuralops-api.onrender.com |
| **Streamlit UI** | https://neuralops-ui.onrender.com |
| **API Health** | https://neuralops-api.onrender.com/health |
| **API Docs** | https://neuralops-api.onrender.com/docs |

> ⚠️ **Free tier note:** Services spin down after 15 minutes of inactivity. The first request after sleep takes ~30 seconds to wake up. This is expected behavior on Render's free plan.

### Render Configuration

**API Service:**
```
Build Command: pip install -r requirements.txt
Start Command: uvicorn api.main:app --host 0.0.0.0 --port $PORT
```

**UI Service:**
```
Build Command: pip install -r requirements.txt
Start Command: streamlit run ui/streamlit_app.py --server.port $PORT --server.address 0.0.0.0
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
  "root_cause": "The database connection pool is exhausted, preventing new requests from acquiring a connection. This is evidenced by 'connection pool exhausted after 30s' in the error output. This occurs when concurrent traffic exceeds pool capacity or slow queries hold connections for too long.",
  "evidence": "connection pool exhausted after 30s",
  "reasoning": "Signal 'pool exhausted' matched DB connection pool full pattern — confidence 0.91",
  "fix_suggestion": "Immediate: Increase HikariCP maximumPoolSize (10 → 20), restart service\nShort-term: EXPLAIN ANALYZE slowest queries, set connectionTimeout=30000ms\nLong-term: Add Prometheus pool metrics, read replicas, Redis caching layer",
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
Returns all past incidents from the SQLite knowledge base (latest 100).

### GET /api/stats
Returns aggregate metrics: total incidents, average confidence, average latency, severity breakdown.

### POST /api/feedback
```json
{
  "incident_id": "INC-20260430-A3F1B2C4",
  "rating": 5,
  "comment": "Accurate root cause and specific fix"
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
│   ├── tools.py              # Shared utilities: incident ID, confidence eval, sanitize
│   ├── autogen_crew.py       # Optional: AutoGen multi-agent (not in main pipeline)
│   └── crewai_solver.py      # Optional: CrewAI multi-agent (not in main pipeline)
│
├── api/
│   ├── main.py               # FastAPI app with all routes and lifespan handler
│   └── mock_engine.py        # Disabled stub — raises NotImplementedError
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
│   ├── pageindex.py          # Severity and service metadata boosting
│   └── hybrid_retriever.py   # Combines BM25 + PageIndex, cached singleton
│
├── data/
│   ├── seed_db.py            # SQLite init, seed data (8 incidents), CRUD operations
│   └── incidents.db          # Auto-generated SQLite database
│
├── tests/
│   ├── test_all.py           # 13 unit tests covering tools, BM25, PageIndex
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
# Run all unit tests (no API keys needed)
python tests/run_tests.py

# Verbose output
python tests/run_tests.py -v
```

| Module | Tests | What Is Tested |
|---|---|---|
| `agents/tools.py` | 6 | Incident ID format, determinism, uniqueness, confidence evaluation, log sanitization |
| `rag/vectorless_rag.py` | 5 | BM25 retrieval accuracy, empty query handling, no match handling, empty docs error |
| `rag/pageindex.py` | 3 | Severity boosting order, service match boosting, field preservation |

---

## 📊 LLMOps & Observability

Every request is tracked with the following metrics:

| Metric | How It Is Tracked |
|---|---|
| **Latency** | `time.time()` measured at pipeline start and end |
| **Confidence** | LLM output clamped to 0.15–0.95 range |
| **Retry count** | Number of retry attempts before fix or escalation |
| **LangSmith traces** | Full agent flow visible in LangSmith dashboard |
| **Evidence** | Exact log quote that triggered the analysis |
| **Evaluation** | High / Medium / Low label based on confidence range |

Enable LangSmith in `.env`:
```env
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_key
LANGCHAIN_PROJECT=NeuralOps
```

---

## 🤔 Why LangGraph Over CrewAI / AutoGen?

| | LangGraph | CrewAI | AutoGen |
|---|---|---|---|
| **Latency** | 2–4s | 15–30s | 10–25s |
| **Routing control** | ✅ Deterministic | ❌ Agent-decided | ❌ Conversational |
| **Retry strategy** | ✅ Explicit nodes | ❌ Limited | ❌ Limited |
| **Production ready** | ✅ Yes | ⚠️ Partial | ⚠️ Partial |
| **Best for** | Fixed pipelines | Open-ended tasks | Conversational AI |

CrewAI and AutoGen are implemented as optional extensions and can be connected for deeper multi-agent analysis on escalated low-confidence incidents.

---

## 🧪 Sample Inputs to Test

```
# SQL Connection Pool Exhaustion
SQLTimeoutException: connection pool exhausted after 30s
  at com.zaxxer.hikari.pool.HikariPool.getConnection(HikariPool.java:213)
  Service: payment-service | Env: production

# Kubernetes Pod Scheduling Failure
0/3 nodes are available: 3 Insufficient cpu
  Warning FailedScheduling pod/api-gateway-6c9d4f
  Event: pod evicted due to resource pressure on node ip-10-0-1-42

# JVM Out of Memory
java.lang.OutOfMemoryError: Java heap space
  at java.util.Arrays.copyOf(Arrays.java:3210)
  Service: recommendation-engine | Heap: 4GB/4GB

# API Request Timeout
[ERROR] TimeoutError: Request to http://localhost:8000/predict timed out after 5s
  Service: inference-api | Env: production

# Redis Memory Eviction
WARN Redis eviction maxmemory-policy=allkeys-lru triggered
  ERROR Cache miss rate: 94%
  Service: user-session-service | Redis memory: 4096/4096 MB
```

---

## 🔧 Work In Progress

This project is actively being developed. Planned improvements include:

- [ ] Persistent storage on Render using PostgreSQL instead of SQLite
- [ ] Vector-based RAG using embeddings alongside BM25 for better retrieval
- [ ] Streaming response support in the Streamlit UI
- [ ] Multi-log batch analysis endpoint
- [ ] Webhook integration for PagerDuty and Slack alerts
- [ ] Evaluation dataset with ground-truth root causes for automated prompt testing
- [ ] Authentication layer for multi-user access
- [ ] Dashboard charts for confidence trends over time
- [ ] Export incidents to CSV / PDF report

---

## 👤 Author

<div align="center">

**Ratnaprava Mohapatra**

*AI Engineer | Building production-grade AI systems*

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077B5?style=flat-square&logo=linkedin&logoColor=white)](https://linkedin.com/in/ratnaprava-mohapatra)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/ratnaprava)

</div>

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

---

<div align="center">

**⭐ If you found this project useful, please consider giving it a star.**

*NeuralOps — AI that reasons from evidence, not assumptions.*

*Built and maintained by Ratnaprava Mohapatra*

</div>
READMEEOF
echo "Done: $(wc -l < /home/claude/README.md) lines"
Output

Done: 717 lines
Done
