import logging
import re
from typing import List, Dict

from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


# ─────────────────────────────
# Tokenizer (Improved)
# ─────────────────────────────
def tokenize(text: str) -> List[str]:
    if not text:
        return []

    # Remove symbols but keep technical tokens
    text = re.sub(r"[^a-zA-Z0-9_/.-]", " ", text)

    tokens = [
        t.lower()
        for t in text.split()
        if len(t) > 2
    ]

    return tokens


# ─────────────────────────────
# RAG Class
# ─────────────────────────────
class VectorlessRAG:
    def __init__(self, documents: List[Dict]):
        if not documents:
            logger.warning("Initializing RAG with empty documents.")
            documents = []

        self.documents = documents

        corpus = []
        for doc in documents:
            text = doc.get("error_text", "")
            corpus.append(tokenize(text))

        self.bm25 = BM25Okapi(corpus)
        logger.info("VectorlessRAG initialized with %d documents.", len(documents))

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        tokens = tokenize(query)

        if not tokens:
            logger.warning("Empty tokens after tokenization.")
            return []

        try:
            scores = self.bm25.get_scores(tokens)
        except Exception as e:
            logger.error("BM25 scoring failed: %s", e)
            return []

        ranked = sorted(
            enumerate(scores),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]

        results = []

        for idx, score in ranked:
            if score <= 0:
                continue

            doc = self.documents[idx]

            results.append({
                **doc,
                "bm25_score": round(float(score), 4),
            })

        logger.info(
            "BM25 retrieval: tokens=%d results=%d",
            len(tokens),
            len(results)
        )

        return results
