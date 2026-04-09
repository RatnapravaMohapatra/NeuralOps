import logging

logger = logging.getLogger(__name__)


class PageIndex:
    """
    Lightweight metadata index that boosts BM25 results based on
    service name, severity, and keyword overlap.
    """

    SEVERITY_WEIGHT = {
        "Critical": 1.4,
        "High": 1.2,
        "Medium": 1.0,
        "Low": 0.8
    }

    def __init__(self, documents: list[dict]):
        self.documents = documents or []

    def boost(self, results: list[dict], query: str) -> list[dict]:
        if not results:
            return []

        query_lower = (query or "").lower()
        query_tokens = set(query_lower.split())

        boosted = []

        for doc in results:
            score = float(doc.get("bm25_score", 0.0))

            # ✅ severity boost
            severity = doc.get("severity", "Medium")
            score *= self.SEVERITY_WEIGHT.get(severity, 1.0)

            # ✅ service match boost
            service = doc.get("service_name", "")
            if service and service.lower() in query_lower:
                score *= 1.3

            # ✅ keyword overlap boost (NEW but safe)
            error_text = doc.get("error_text", "").lower()
            overlap = sum(1 for token in query_tokens if token in error_text)

            if overlap >= 2:
                score *= 1.15

            boosted.append({
                **doc,
                "boosted_score": round(score, 4)
            })

        boosted.sort(key=lambda x: x["boosted_score"], reverse=True)

        logger.info(
            "PageIndex boost complete | results=%d query='%s'",
            len(boosted),
            query[:50]
        )

        return boosted
