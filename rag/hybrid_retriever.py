import logging
import threading
from typing import List, Dict

from rag.vectorless_rag import VectorlessRAG
from rag.pageindex import PageIndex
from data.seed_db import get_all_error_texts

logger = logging.getLogger(__name__)

# ── Global cache ─────────────────────────────
_retriever: VectorlessRAG | None = None
_index: PageIndex | None = None
_cache: dict[str, List[Dict]] = {}

# Thread safety
_lock = threading.Lock()

# Refresh control
_last_loaded_count = 0


# ─────────────────────────────
# Load / Reload Retriever
# ─────────────────────────────
def get_retriever() -> tuple[VectorlessRAG, PageIndex]:
    global _retriever, _index, _last_loaded_count

    with _lock:
        docs = get_all_error_texts()

        if not docs:
            logger.warning("Knowledge base empty — returning fallback retriever.")
            return VectorlessRAG([]), PageIndex([])

        # 🔥 Reload if new data added
        if _retriever is None or len(docs) != _last_loaded_count:
            _retriever = VectorlessRAG(docs)
            _index = PageIndex(docs)
            _last_loaded_count = len(docs)

            logger.info("HybridRetriever loaded with %d documents.", len(docs))

    return _retriever, _index


# ─────────────────────────────
# Retrieve Similar
# ─────────────────────────────
def retrieve_similar(query: str, top_k: int = 3) -> List[Dict]:
    if not query or len(query.strip()) < 5:
        logger.warning("Empty or weak query passed to retriever.")
        return []

    query = query.strip()

    # 🔥 Cache hit
    if query in _cache:
        return _cache[query]

    try:
        retriever, index = get_retriever()

        raw_results = retriever.retrieve(query, top_k=top_k)

        if not raw_results:
            logger.info("No results from retriever.")
            _cache[query] = []
            return []

        boosted = index.boost(raw_results, query)

    except Exception as e:
        logger.error("Retrieval failed: %s", e)
        return []

    # 🔥 Limit cache size (prevent memory leak)
    if len(_cache) > 100:
        _cache.clear()

    _cache[query] = boosted
    return boosted
