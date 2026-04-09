from rank_bm25 import BM25Okapi
import logging

logger = logging.getLogger(__name__)


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in text.split() if len(t) > 2]


class VectorlessRAG:
    def __init__(self, documents: list[dict]):
        """
        documents: list of dicts with at least:
            - 'error_text' (str): the searchable text
            - any other fields returned as metadata
        """
        if not documents:
            raise ValueError("Cannot initialize RAG with empty document list.")

        self.documents = documents
        corpus = [tokenize(doc["error_text"]) for doc in documents]
        self.bm25 = BM25Okapi(corpus)
        logger.info("VectorlessRAG initialized with %d documents.", len(documents))

    def retrieve(self, query: str, top_k: int = 3) -> list[dict]:
        tokens = tokenize(query)
        if not tokens:
            logger.warning("Empty token list after tokenization. Returning no results.")
            return []

        scores = self.bm25.get_scores(tokens)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]

        results = []
        for idx, score in ranked:
            if score > 0:
                results.append({
                    **self.documents[idx],
                    "bm25_score": round(float(score), 4),
                })

        logger.info("BM25 retrieval: query_tokens=%d results=%d", len(tokens), len(results))
        return results
