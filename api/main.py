import time
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from graph.incident_graph import run_incident_pipeline
from data.seed_db import init_db, get_all_incidents, get_stats, save_feedback

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ================= ROOT =================
@app.get("/")
async def root():
    return {"message": "NeuralOps API running"}


@app.get("/health")
async def health():
    return {"status": "healthy"}


# ================= SCHEMA =================
class AnalyzeRequest(BaseModel):
    log_input: str


class FeedbackRequest(BaseModel):
    incident_id: str
    rating: int
    comment: str = ""


# ================= ROUTES =================
@app.post("/api/analyze")
async def analyze_incident(request: AnalyzeRequest):
    t0 = time.perf_counter()

    try:
        result = await run_incident_pipeline(request.log_input)
    except Exception as e:
        logger.exception(e)
        raise HTTPException(status_code=500, detail=str(e))

    result["latency"] = round(time.perf_counter() - t0, 3)
    return result


@app.get("/api/incidents")
async def list_incidents():
    return {"incidents": get_all_incidents()}


@app.get("/api/stats")
async def stats():
    return get_stats()


@app.post("/api/feedback")
async def feedback(request: FeedbackRequest):
    save_feedback(request.incident_id, request.rating, request.comment)
    return {"status": "ok"}
