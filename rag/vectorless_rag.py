import logging
import re
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


def tokenize(text: str):
    if not text:
        return []

    text = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    return [t for t in text.split() if len(t) > 2]


class VectorlessRAG:
    def __init__(self, documents):
        if not documents:
            raise ValueError("Empty docs")

        # ✅ filter valid documents
        self.documents = [
            doc for doc in documents if doc.get("error_text")
        ]

        if not self.documents:
            raise ValueError("No valid documents with error_text")

        corpus = [tokenize(doc["error_text"]) for doc in self.documents]

        self.bm25 = BM25Okapi(corpus)

        logger.info("VectorlessRAG initialized with %d docs", len(self.documents))

    def retrieve(self, query, top_k=3):
        tokens = tokenize(query)

        # ✅ required for tests
        if not tokens:
            return []

        scores = self.bm25.get_scores(tokens)

        ranked = sorted(
            enumerate(scores),
            key=lambda x: x[1],
            reverse=True
        )

        results = []

        # ✅ take more candidates, then filter
        for idx, score in ranked[: top_k * 2]:
            if score <= 0:
                continue

            results.append({
                **self.documents[idx],
                "bm25_score": round(float(score), 4)
            })

            if len(results) >= top_k:
                break

        logger.info(
            "BM25 retrieval | query_tokens=%d results=%d",
            len(tokens),
            len(results)
        )

        return results
