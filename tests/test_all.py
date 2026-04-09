import pytest
from agents.tools import generate_incident_id, evaluate_confidence, sanitize_log
from rag.vectorless_rag import VectorlessRAG
from rag.pageindex import PageIndex


# --------------------------------------------------------------------------- #
# agents/tools
# --------------------------------------------------------------------------- #

def test_generate_incident_id_deterministic():
    id1 = generate_incident_id("some log")
    id2 = generate_incident_id("some log")
    assert id1 == id2


def test_generate_incident_id_format():
    inc_id = generate_incident_id("some log")
    assert inc_id.startswith("INC-")
    parts = inc_id.split("-")
    assert len(parts) == 3
    assert len(parts[1]) == 8   # YYYYMMDD
    assert len(parts[2]) == 8   # SHA hex


def test_generate_incident_id_unique():
    id1 = generate_incident_id("log A")
    id2 = generate_incident_id("log B")
    assert id1 != id2


def test_evaluate_confidence_high():
    assert evaluate_confidence(0.9) == "High"
    assert evaluate_confidence(0.8) == "High"


def test_evaluate_confidence_medium():
    assert evaluate_confidence(0.7) == "Medium"
    assert evaluate_confidence(0.6) == "Medium"


def test_evaluate_confidence_low():
    assert evaluate_confidence(0.59) == "Low"
    assert evaluate_confidence(0.0) == "Low"


def test_sanitize_log_truncates():
    long_log = "x" * 5000
    result = sanitize_log(long_log)
    assert len(result) <= 4020
    assert "[truncated]" in result


def test_sanitize_log_strips():
    result = sanitize_log("  hello  ")
    assert result == "hello"


def test_sanitize_log_within_limit():
    short_log = "short log entry"
    result = sanitize_log(short_log)
    assert result == short_log
    assert "[truncated]" not in result


# --------------------------------------------------------------------------- #
# rag/vectorless_rag
# --------------------------------------------------------------------------- #

DOCS = [
    {
        "error_text": "connection pool exhausted timeout sql hikari",
        "service_name": "payment-service",
        "root_cause": "Pool full under load",
        "fix_suggestion": "Increase pool size",
        "severity": "Critical",
    },
    {
        "error_text": "out of memory java heap space arrays",
        "service_name": "recommendation-engine",
        "root_cause": "Heap exhausted by unbounded cache",
        "fix_suggestion": "Tune JVM heap and add eviction",
        "severity": "High",
    },
    {
        "error_text": "redis eviction maxmemory cache miss session",
        "service_name": "session-service",
        "root_cause": "Redis memory limit hit",
        "fix_suggestion": "Scale Redis or tune eviction policy",
        "severity": "High",
    },
]


def test_bm25_retrieval_top_result():
    rag = VectorlessRAG(DOCS)
    results = rag.retrieve("sql connection pool exhausted", top_k=3)
    assert len(results) >= 1
    assert results[0]["service_name"] == "payment-service"


def test_bm25_retrieval_returns_score():
    rag = VectorlessRAG(DOCS)
    results = rag.retrieve("java heap memory", top_k=2)
    assert all("bm25_score" in r for r in results)
    assert all(r["bm25_score"] > 0 for r in results)


def test_bm25_empty_query_returns_empty():
    rag = VectorlessRAG(DOCS)
    results = rag.retrieve("", top_k=3)
    assert results == []


def test_bm25_no_match_returns_empty():
    rag = VectorlessRAG(DOCS)
    results = rag.retrieve("xyzzy frobnicator quux", top_k=3)
    assert results == []


def test_bm25_raises_on_empty_docs():
    with pytest.raises(ValueError):
        VectorlessRAG([])


# --------------------------------------------------------------------------- #
# rag/pageindex
# --------------------------------------------------------------------------- #

def test_pageindex_boosts_critical_over_high():
    docs = [
        {"error_text": "redis eviction", "service_name": "cache", "severity": "High", "bm25_score": 1.0},
        {"error_text": "sql timeout", "service_name": "db", "severity": "Critical", "bm25_score": 1.0},
    ]
    index = PageIndex(docs)
    boosted = index.boost(docs, "redis sql")
    assert boosted[0]["severity"] == "Critical"


def test_pageindex_boosts_service_match():
    docs = [
        {"error_text": "timeout error", "service_name": "api-gateway", "severity": "Medium", "bm25_score": 1.0},
        {"error_text": "disk full", "service_name": "logging-agent", "severity": "Medium", "bm25_score": 1.0},
    ]
    index = PageIndex(docs)
    boosted = index.boost(docs, "api-gateway timeout")
    assert boosted[0]["service_name"] == "api-gateway"


def test_pageindex_preserves_all_fields():
    docs = [
        {"error_text": "test error", "service_name": "svc", "severity": "Low", "bm25_score": 0.5, "root_cause": "test cause"},
    ]
    index = PageIndex(docs)
    boosted = index.boost(docs, "test")
    assert boosted[0]["root_cause"] == "test cause"
    assert "boosted_score" in boosted[0]
