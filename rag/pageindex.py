import logging

logger = logging.getLogger(__name__)


class PageIndex:
    """
    Lightweight metadata index that boosts BM25 results based on
    service name, severity, and recency signals.
    """

    SEVERITY_WEIGHT = {"Critical": 1.4, "High": 1.2, "Medium": 1.0, "Low": 0.8}

    def __init__(self, documents: list[dict]):
        self.documents = documents

    def boost(self, results: list[dict], query: str) -> list[dict]:
        query_lower = query.lower()
        boosted = []
        for doc in results:
            score = doc.get("bm25_score", 0.0)
            sev = doc.get("severity", "Medium")
            score *= self.SEVERITY_WEIGHT.get(sev, 1.0)
            service = doc.get("service_name", "")
            if service and service.lower() in query_lower:
                score *= 1.3
            boosted.append({**doc, "boosted_score": round(score, 4)})

        boosted.sort(key=lambda x: x["boosted_score"], reverse=True)
        logger.info("PageIndex boosting applied to %d results.", len(boosted))
        return boosted
