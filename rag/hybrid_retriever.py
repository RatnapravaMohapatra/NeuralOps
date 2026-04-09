import logging
from rag.vectorless_rag import VectorlessRAG
from rag.pageindex import PageIndex
from data.seed_db import get_all_error_texts

logger = logging.getLogger(__name__)

_retriever: VectorlessRAG | None = None
_index: PageIndex | None = None


def get_retriever() -> tuple[VectorlessRAG, PageIndex]:
    global _retriever, _index
    if _retriever is None:
        docs = get_all_error_texts()
        if not docs:
            raise RuntimeError("Knowledge base is empty. Run seed_db.init_db() first.")
        _retriever = VectorlessRAG(docs)
        _index = PageIndex(docs)
        logger.info("HybridRetriever loaded with %d documents.", len(docs))
    return _retriever, _index


def retrieve_similar(query: str, top_k: int = 3) -> list[dict]:
    retriever, index = get_retriever()
    raw_results = retriever.retrieve(query, top_k=top_k)
    boosted = index.boost(raw_results, query)
    return boosted
