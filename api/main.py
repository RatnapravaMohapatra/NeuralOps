import time
import uuid
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from graph.incident_graph import run_incident_pipeline
from data.seed_db import init_db, get_all_incidents, get_stats, save_feedback

# ─────────────────────────────
# Logging
# ─────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)

# ─────────────────────────────
# App Lifecycle
# ─────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing DB...")
    init_db()
    yield


app = FastAPI(
    title="NeuralOps API",
    version="1.0.0",
    lifespan=lifespan,
)

# ⚠️ In production, replace "*" with your frontend domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────
# Schemas
# ─────────────────────────────
class AnalyzeRequest(BaseModel):
    log_input: str = Field(..., min_length=10, max_length=5000)


class FeedbackRequest(BaseModel):
    incident_id: str
    rating: int = Field(..., ge=1, le=5)
    comment: str = ""


# ─────────────────────────────
# Middleware (Request ID + Logging)
# ─────────────────────────────
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = str(uuid.uuid4())
    start_time = time.time()

    logger.info(f"[{request_id}] Incoming request: {request.url.path}")

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
# Routes
# ─────────────────────────────
@app.get("/")
async def root():
    return {"message": "NeuralOps API running"}


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/api/analyze")
async def analyze_incident(request: AnalyzeRequest):
    start = time.perf_counter()

    try:
        # ⚠️ FIX: pipeline is sync → do NOT await
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


@app.get("/api/incidents")
async def list_incidents():
    try:
        return {"incidents": get_all_incidents()}
    except Exception as e:
        logger.exception("Failed to fetch incidents")
        raise HTTPException(status_code=500, detail="Failed to fetch incidents")


@app.get("/api/stats")
async def stats():
    try:
        return get_stats()
    except Exception as e:
        logger.exception("Failed to fetch stats")
        raise HTTPException(status_code=500, detail="Failed to fetch stats")


@app.post("/api/feedback")
async def feedback(request: FeedbackRequest):
    try:
        save_feedback(request.incident_id, request.rating, request.comment)
        return {"status": "ok"}
    except Exception as e:
        logger.exception("Failed to save feedback")
        raise HTTPException(status_code=500, detail="Failed to save feedback")
