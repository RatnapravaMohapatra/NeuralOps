# ─────────────────────────────
# 🔥 LangSmith INIT (MUST BE FIRST)
# ─────────────────────────────
import os
from utils.tracing import setup_langsmith

setup_langsmith()

# Optional debug (remove later if you want)
print("✅ LangSmith tracing enabled:", os.getenv("LANGCHAIN_TRACING_V2"))
print("📦 LangSmith project:", os.getenv("LANGCHAIN_PROJECT"))


# ─────────────────────────────
# STANDARD IMPORTS
# ─────────────────────────────
import time
import uuid
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# ─────────────────────────────
# SAFE IMPORTS (prevent crash)
# ─────────────────────────────
try:
    from graph.incident_graph import run_incident_pipeline
    PIPELINE_AVAILABLE = True
except Exception as e:
    print("⚠️ Pipeline import failed:", e)
    PIPELINE_AVAILABLE = False

try:
    from data.seed_db import init_db, get_all_incidents, get_stats, save_feedback
    DB_AVAILABLE = True
except Exception as e:
    print("⚠️ DB import failed:", e)
    DB_AVAILABLE = False


# ─────────────────────────────
# LOGGING
# ─────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)


# ─────────────────────────────
# APP LIFECYCLE
# ─────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting NeuralOps API...")

    if DB_AVAILABLE:
        try:
            init_db()
            logger.info("DB initialized")
        except Exception as e:
            logger.warning(f"DB init failed: {e}")

    yield

    logger.info("Shutting down NeuralOps API...")


# ─────────────────────────────
# APP INIT
# ─────────────────────────────
app = FastAPI(
    title="NeuralOps API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────
# SCHEMAS
# ─────────────────────────────
class AnalyzeRequest(BaseModel):
    log_input: str = Field(..., min_length=5, max_length=5000)


class FeedbackRequest(BaseModel):
    incident_id: str
    rating: int = Field(..., ge=1, le=5)
    comment: str = ""


# ─────────────────────────────
# MIDDLEWARE (REQUEST TRACKING)
# ─────────────────────────────
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = str(uuid.uuid4())
    start_time = time.time()

    logger.info(f"[{request_id}] {request.method} {request.url.path}")

    try:
        response = await call_next(request)
    except Exception as e:
        logger.exception(f"[{request_id}] Request failed: {e}")
        raise

    latency = round(time.time() - start_time, 3)
    response.headers["X-Request-ID"] = request_id

    logger.info(f"[{request_id}] Completed in {latency}s")
    return response


# ─────────────────────────────
# ROUTES
# ─────────────────────────────
@app.get("/")
async def root():
    return {
        "message": "NeuralOps API running 🚀",
        "pipeline": "available" if PIPELINE_AVAILABLE else "fallback",
        "db": "connected" if DB_AVAILABLE else "not_available",
        "tracing": os.getenv("LANGCHAIN_TRACING_V2"),
    }


@app.get("/health")
async def health():
    return {"status": "healthy"}


# ─────────────────────────────
# ANALYZE ENDPOINT
# ─────────────────────────────
@app.post("/api/analyze")
async def analyze_incident(request: AnalyzeRequest):
    start = time.perf_counter()

    if not PIPELINE_AVAILABLE:
        return {
            "incident_id": "INC-FALLBACK",
            "root_cause": "Pipeline not available",
            "confidence": 0.0,
            "severity": "Unknown",
            "service_name": "core-platform-service",
            "fix_suggestion": "Check deployment configuration",
            "evaluation": "Low",
            "api_latency": 0,
        }

    try:
        # 🔥 NOTE: tracing happens inside incident_graph.py (recommended)
        result = run_incident_pipeline(request.log_input)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        logger.exception("Pipeline failed")
        raise HTTPException(
            status_code=500,
            detail="Internal server error during analysis"
        )

    latency = round(time.perf_counter() - start, 3)

    return {
        **result,
        "api_latency": latency,
    }


# ─────────────────────────────
# INCIDENTS
# ─────────────────────────────
@app.get("/api/incidents")
async def list_incidents():
    if not DB_AVAILABLE:
        return {"incidents": []}

    try:
        return {"incidents": get_all_incidents()}
    except Exception:
        logger.exception("Failed to fetch incidents")
        raise HTTPException(status_code=500, detail="Failed to fetch incidents")


@app.get("/api/stats")
async def stats():
    if not DB_AVAILABLE:
        return {"total": 0}

    try:
        return get_stats()
    except Exception:
        logger.exception("Failed to fetch stats")
        raise HTTPException(status_code=500, detail="Failed to fetch stats")


# ─────────────────────────────
# FEEDBACK
# ─────────────────────────────
@app.post("/api/feedback")
async def feedback(request: FeedbackRequest):
    if not DB_AVAILABLE:
        return {"status": "db not available"}

    try:
        save_feedback(request.incident_id, request.rating, request.comment)
        return {"status": "ok"}
    except Exception:
        logger.exception("Failed to save feedback")
        raise HTTPException(status_code=500, detail="Failed to save feedback")
