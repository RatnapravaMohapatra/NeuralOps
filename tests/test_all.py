# --------------------------------------------------------------------------- #
# Additional Edge Cases (Production Quality)
# --------------------------------------------------------------------------- #

def test_pageindex_empty_results():
    index = PageIndex([])
    boosted = index.boost([], "test query")
    assert boosted == []


def test_pageindex_missing_fields():
    docs = [
        {"error_text": "test", "bm25_score": 1.0},  # missing service + severity
    ]
    index = PageIndex(docs)
    boosted = index.boost(docs, "test")
    assert len(boosted) == 1
    assert "boosted_score" in boosted[0]


def test_tokenizer_handles_special_chars():
    rag = VectorlessRAG(DOCS)
    tokens = rag.retrieve("SQLTimeoutException: connection-pool-exhausted", top_k=3)
    assert isinstance(tokens, list)


def test_rag_with_partial_match():
    rag = VectorlessRAG(DOCS)
    results = rag.retrieve("redis memory", top_k=2)
    assert len(results) >= 1
    assert results[0]["service_name"] == "session-service"


def test_rag_top_k_limit():
    rag = VectorlessRAG(DOCS)
    results = rag.retrieve("memory", top_k=1)
    assert len(results) <= 1


def test_boosting_does_not_remove_docs():
    docs = [
        {"error_text": "a", "service_name": "svc1", "severity": "Low", "bm25_score": 0.5},
        {"error_text": "b", "service_name": "svc2", "severity": "High", "bm25_score": 0.4},
    ]
    index = PageIndex(docs)
    boosted = index.boost(docs, "a b")
    assert len(boosted) == len(docs)


def test_pipeline_like_flow():
    rag = VectorlessRAG(DOCS)
    index = PageIndex(DOCS)

    results = rag.retrieve("sql connection", top_k=3)
    boosted = index.boost(results, "sql connection")

    assert len(boosted) >= 1
    assert "boosted_score" in boosted[0]
