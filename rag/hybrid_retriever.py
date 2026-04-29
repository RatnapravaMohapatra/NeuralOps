import logging
import threading
import re
from typing import List, Dict

from rag.vectorless_rag import VectorlessRAG
from rag.pageindex import PageIndex
from data.seed_db import get_all_error_texts

logger = logging.getLogger(__name__)

# ── Global cache ─────────────────────────────
_retriever: VectorlessRAG | None = None
_index: PageIndex | None = None
_cache: dict[str, List[Dict]] = {}

_lock = threading.Lock()
_last_loaded_count = 0


# ─────────────────────────────
# 🔥 RELEVANCE FILTER (CRITICAL FIX)
# ─────────────────────────────
TECH_KEYWORDS = [
    "error", "fail", "failed", "exception",
    "timeout", "memory", "cpu", "disk",
    "kubernetes", "pod", "node",
    "service", "connection", "refused",
    "oom", "latency", "network", "crash"
]


def is_relevant(text: str) -> bool:
    if not text:
        return False

    text = text.lower()

    # must contain at least one technical keyword
    if not any(k in text for k in TECH_KEYWORDS):
        return False

    # 🚫 filter garbage domains (like GST, finance, etc.)
    if any(x in text for x in ["gst", "tax", "invoice", "council"]):
        return False

    return True


# ─────────────────────────────
# 🔥 CLEAN RESULT FORMAT
# ─────────────────────────────
def normalize_result(item: Dict) -> Dict:
    return {
        "root_cause": item.get("root_cause") or item.get("content", "")[:300],
        "content": item.get("content", ""),
        "score": item.get("score", 0),
    }


# ─────────────────────────────
# Load / Reload Retriever
# ─────────────────────────────
def get_retriever() -> tuple[VectorlessRAG, PageIndex]:
    global _retriever, _index, _last_loaded_count

    with _lock:
        docs = get_all_error_texts()

        # 🔥 FILTER BAD DATA AT SOURCE
        docs = [d for d in docs if is_relevant(d)]

        if not docs:
            logger.warning("Knowledge base empty after filtering.")
            return VectorlessRAG([]), PageIndex([])

        if _retriever is None or len(docs) != _last_loaded_count:
            _retriever = VectorlessRAG(docs)
            _index = PageIndex(docs)
            _last_loaded_count = len(docs)

            logger.info("HybridRetriever loaded with %d clean documents.", len(docs))

    return _retriever, _index


# ─────────────────────────────
# Retrieve Similar
# ─────────────────────────────
def retrieve_similar(query: str, top_k: int = 3) -> List[Dict]:
    if not query or len(query.strip()) < 5:
        logger.warning("Weak query passed to retriever.")
        return []

    query = query.strip().lower()

    # 🔥 cache hit
    if query in _cache:
        return _cache[query]

    try:
        retriever, index = get_retriever()

        raw_results = retriever.retrieve(query, top_k=top_k * 2)  # get more → filter later

        if not raw_results:
            _cache[query] = []
            return []

        boosted = index.boost(raw_results, query)

        # 🔥 FILTER + NORMALIZE
        filtered = []
        for r in boosted:
            text = r.get("content", "") or r.get("root_cause", "")

            if is_relevant(text):
                filtered.append(normalize_result(r))

        # sort by score (best first)
        filtered = sorted(filtered, key=lambda x: x["score"], reverse=True)

        final_results = filtered[:top_k]

    except Exception as e:
        logger.error("Retrieval failed: %s", e)
        return []

    # 🔥 cache control
    if len(_cache) > 100:
        _cache.clear()

    _cache[query] = final_results
    return final_results
