import logging
from typing import List, Dict

logger = logging.getLogger(__name__)


class PageIndex:
    """
    Lightweight metadata index that boosts BM25 results based on:
    - severity
    - service match
    - keyword overlap
    """

    SEVERITY_WEIGHT = {
        "Critical": 1.4,
        "High": 1.2,
        "Medium": 1.0,
        "Low": 0.8,
    }

    def __init__(self, documents: List[Dict]):
        self.documents = documents or []

    def boost(self, results: List[Dict], query: str) -> List[Dict]:
        if not results:
            return []

        query_lower = (query or "").lower()
        query_tokens = set(query_lower.split())

        boosted = []

        for doc in results:
            base_score = float(doc.get("bm25_score", 0.0))
            if base_score <= 0:
                continue

            # ── Severity boost ─────────────────
            severity = doc.get("severity", "Medium")
            score = base_score * self.SEVERITY_WEIGHT.get(severity, 1.0)

            # ── Service match boost ────────────
            service = doc.get("service_name", "")
            if service and service.lower() in query_lower:
                score *= 1.25

            # ── Keyword overlap boost ──────────
            text = doc.get("error_text", "").lower()
            doc_tokens = set(text.split())
            overlap = len(query_tokens & doc_tokens)

            if overlap > 2:
                score *= 1.2

            # ── Clamp to prevent explosion ─────
            score = min(score, base_score * 2.0)

            boosted.append({
                **doc,
                "boosted_score": round(score, 4)
            })

        boosted.sort(key=lambda x: x["boosted_score"], reverse=True)

        logger.info("PageIndex boosted %d results", len(boosted))
        return boosted
