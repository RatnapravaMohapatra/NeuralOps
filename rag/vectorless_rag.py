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

        self.documents = documents
        corpus = [tokenize(doc["error_text"]) for doc in documents]
        self.bm25 = BM25Okapi(corpus)

    def retrieve(self, query, top_k=3):
        tokens = tokenize(query)
        if not tokens:
            return []

        scores = self.bm25.get_scores(tokens)

        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

        results = []
        for idx, score in ranked[:top_k]:
            if score <= 0:
                continue
            results.append({
                **self.documents[idx],
                "bm25_score": round(float(score), 4)
            })

        return results
