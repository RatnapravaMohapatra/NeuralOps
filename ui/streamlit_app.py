import os
import time
import httpx
import streamlit as st
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

API_BASE = os.getenv("API_BASE_URL") or st.secrets.get("API_BASE_URL")
if not API_BASE:
    st.error("❌ API_BASE_URL not set. Please configure secrets.")
    st.stop()

st.set_page_config(
    page_title="NeuralOps | AI Incident Analysis", 
    layout="wide", 
    page_icon="🤖"
)

# ============================================================================
# CUSTOM CSS - CLEAN BACKGROUND WITH BOLD WHITE TEXT
# ============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:opsz,wght@14..32,300;14..32,400;14..32,500;14..32,600;14..32,700;14..32,800&display=swap');
    
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .main .block-container {
        padding: 2rem 2.5rem;
        max-width: 1400px;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em !important;
    }
    
    h1 {
        font-size: 2.2rem !important;
        background: linear-gradient(135deg, #ffffff, #e2e8f0);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem !important;
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a0f1a 0%, #0f172a 100%);
        border-right: 1px solid #334155;
    }
    
    [data-testid="stSidebar"] * {
        color: #ffffff !important;
    }
    
    [data-testid="stSidebar"] .stMarkdown p {
        color: #ffffff !important;
    }
    
    div[data-testid="stRadio"] > div {
        gap: 0.5rem;
    }
    
    div[data-testid="stRadio"] label {
        background: transparent;
        padding: 0.75rem 1rem;
        border-radius: 12px;
        transition: all 0.2s ease;
        cursor: pointer;
    }
    
    div[data-testid="stRadio"] label:hover {
        background: #334155;
    }
    
    div[data-testid="stRadio"] [data-testid="stMarkdown"] p {
        font-weight: 600;
        color: #ffffff !important;
        font-size: 1rem;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
        font-size: 0.9rem;
        transition: all 0.2s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.4);
    }
    
    .stTextArea textarea {
        border-radius: 16px;
        border: 1px solid #334155;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
        line-height: 1.5;
        background: #1e293b;
        color: #ffffff;
        transition: all 0.2s ease;
    }
    
    .stTextArea textarea:focus {
        border-color: #6366f1;
        box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.2);
    }
    
    .stTextArea textarea::placeholder {
        color: #94a3b8;
    }
    
    [data-testid="stMetric"] {
        background: #1e293b;
        padding: 1rem;
        border-radius: 16px;
        border: 1px solid #334155;
    }
    
    [data-testid="stMetric"] label {
        color: #94a3b8 !important;
        font-weight: 500;
    }
    
    [data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-size: 1.8rem;
        font-weight: 700;
    }
    
    .stAlert {
        border-radius: 12px;
        border-left-width: 4px;
    }
    
    .stAlert [data-testid="stMarkdown"] p {
        color: #ffffff !important;
    }
    
    .stDataFrame {
        border-radius: 16px;
        overflow: hidden;
        border: 1px solid #334155;
    }
    
    .stDataFrame table {
        background: #1e293b;
        color: #ffffff;
    }
    
    .stDataFrame th {
        background: #0f172a;
        color: #ffffff;
        font-weight: 600;
    }
    
    .stDataFrame td {
        color: #e2e8f0;
    }
    
    hr {
        margin: 1.5rem 0;
        border-color: #334155;
    }
    
    .stCaption {
        color: #94a3b8 !important;
    }
    
    .stProgress > div > div {
        background: linear-gradient(90deg, #6366f1, #8b5cf6);
        border-radius: 20px;
    }
    
    .stSpinner > div {
        color: #ffffff !important;
    }
    
    footer {
        display: none;
    }
    
    code {
        background: #0f172a;
        color: #a5b4fc;
        padding: 0.2rem 0.4rem;
        border-radius: 6px;
    }
    
    .stMarkdown p {
        color: #e2e8f0;
    }
    
    .stSelectbox div[data-baseweb="select"] {
        background-color: #1e293b;
        border-color: #334155;
        color: #ffffff;
    }
    
    .stSelectbox svg {
        fill: #ffffff;
    }
    
    .stSlider [data-baseweb="slider"] {
        background-color: #334155;
    }
    
    .stSlider [data-testid="stThumbValue"] {
        color: #ffffff;
    }
    
    @media (max-width: 768px) {
        .main .block-container {
            padding: 1rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SAMPLE DATA
# ============================================================================
SAMPLES = {
    "SQL Timeout": (
        "SQLTimeoutException: connection pool exhausted after 30s\n"
        "  at com.zaxxer.hikari.pool.HikariPool.getConnection(HikariPool.java:213)\n"
        "  Service: payment-service | Env: production | Region: us-east-1"
    ),
    "OOM Crash": (
        "java.lang.OutOfMemoryError: Java heap space\n"
        "  at java.util.Arrays.copyOf(Arrays.java:3210)\n"
        "  at com.app.cache.InMemoryStore.put(InMemoryStore.java:88)\n"
        "  Service: recommendation-engine | Heap: 4GB/4GB"
    ),
    "K8s Failure": (
        "0/3 nodes are available: 3 Insufficient cpu\n"
        "  Warning FailedScheduling pod/api-gateway-6c9d4f\n"
        "  Event: pod evicted due to resource pressure\n"
        "  Namespace: production | Cluster: prod-eks"
    ),
    "Redis Eviction": (
        "WARN Redis eviction policy maxmemory-policy=allkeys-lru triggered\n"
        "  ERROR Cache miss rate: 94%\n"
        "  Service: user-session-service | Redis memory: 4096/4096 MB"
    ),
}

PAGES = ["Analyze", "Dashboard", "Observability", "Architecture"]

with st.sidebar:
    st.markdown("### NeuralOps")
    st.caption("AI Incident Analysis Platform")
    st.markdown("---")
    page = st.radio("Navigation", PAGES, label_visibility="collapsed")
    st.markdown("---")
    
    st.markdown("### System Status")
    st.markdown(
        '<div style="background: #1e293b; border-radius: 12px; padding: 0.75rem;">'
        '<div style="display: flex; align-items: center; gap: 0.5rem;">'
        '<div style="width: 8px; height: 8px; background: #22c55e; border-radius: 50%;"></div>'
        '<span style="color: #ffffff; font-weight: 500;">API Operational</span>'
        '</div>'
        '<div style="margin-top: 0.75rem;">'
        '<span style="color: #94a3b8; font-size: 0.75rem;">Model: </span>'
        '<span style="color: #ffffff; font-size: 0.75rem; font-weight: 500;">Llama 3 70B</span><br>'
        '<span style="color: #94a3b8; font-size: 0.75rem;">RAG: </span>'
        '<span style="color: #ffffff; font-size: 0.75rem; font-weight: 500;">BM25 + PageIndex</span>'
        '</div>'
        '</div>',
        unsafe_allow_html=True
    )
    
    st.markdown("---")
    st.markdown(
        '<div style="color: #64748b; font-size: 0.7rem; text-align: center;">'
        'Developed by<br><span style="color: #a5b4fc; font-weight: 600;">Ratnaprava Mohapatra</span>'
        '</div>',
        unsafe_allow_html=True
    )

# ============================================================================
# ANALYZE PAGE
# ============================================================================
if page == "Analyze":
    st.markdown("# NeuralOps")
    st.caption("AI-Powered Incident Analysis — Understand, Diagnose, and Fix Issues Instantly")
    st.markdown("---")
    
    col1, col2 = st.columns([2.5, 1.2])
    
    with col1:
        log_input = st.text_area(
            "Log or Error Message",
            height=200,
            placeholder="Paste your log, error, or stack trace here...",
            label_visibility="collapsed",
            key="log_text"
        )
    
    with col2:
        st.markdown("### Quick Samples")
        for label, sample_text in SAMPLES.items():
            if st.button(label, use_container_width=True, key=f"sample_{label[:10]}"):
                st.session_state["prefill"] = sample_text
                st.rerun()
    
    if "prefill" in st.session_state:
        log_input = st.session_state.pop("prefill")
    
    st.markdown("---")
    
    run = st.button("Run Analysis", type="primary", use_container_width=False)
    
    if run:
        if not log_input or not log_input.strip():
            st.error("Please enter a log or error message to analyze.")
        else:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            steps = ["Parsing logs", "Retrieving context", "Analyzing root cause", "Generating fix"]
            for i, step in enumerate(steps):
                status_text.markdown(f"**{step}...**")
                progress_bar.progress((i + 1) / len(steps))
                time.sleep(0.2)
            
            status_text.empty()
            progress_bar.empty()
            
            with st.spinner("Analyzing with AI agents..."):
                try:
                    resp = httpx.post(
                        f"{API_BASE}/api/analyze",
                        json={"log_input": log_input},
                        timeout=60.0
                    )
                    resp.raise_for_status()
                    result = resp.json()
                except httpx.TimeoutException:
                    st.error("⏳ Request timed out. Backend may be slow or sleeping.")
                    st.stop()
                except httpx.HTTPStatusError as e:
                    st.error(f"⚠️ API Error {e.response.status_code}")
                    st.code(e.response.text[:300])
                    st.stop()
                except Exception as e:
                    st.error("🚫 Backend not reachable")
                    st.caption(str(e))
                    st.stop()
            
            st.balloons()
            st.success(f"Analysis Complete — Incident ID: `{result['incident_id']}`")
            st.markdown("---")
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Confidence", f"{result['confidence'] * 100:.0f}%")
            col2.metric("Severity", result['severity'])
            col3.metric("Service", result['service_name'])
            col4.metric("Evaluation", result['evaluation'])
            
            # LOW CONFIDENCE WARNING
            if result["confidence"] < 0.6:
                st.warning("⚠️ Low confidence result — manual verification recommended")
            
            st.markdown("### Root Cause Analysis")
            st.info(result['root_cause'])
            
            st.markdown("### Fix Recommendation")
            st.success(result['fix_suggestion'])
            
            st.markdown("### Performance")
            st.caption(f"End-to-end latency: **{result['latency']}s**")
            
            st.markdown("---")
            st.markdown("### Submit Feedback")
            with st.form("feedback_form"):
                col1, col2 = st.columns([1, 2])
                with col1:
                    rating = st.slider("Rating", 1, 5, 4)
                with col2:
                    comment = st.text_input("Comment (optional)")
                submitted = st.form_submit_button("Submit Feedback")
                
                if submitted:
                    try:
                        fb_resp = httpx.post(
                            f"{API_BASE}/api/feedback",
                            json={
                                "incident_id": result["incident_id"],
                                "rating": rating,
                                "comment": comment
                            },
                            timeout=10.0
                        )
                        fb_resp.raise_for_status()
                        st.success("Thank you for your feedback!")
                    except Exception as e:
                        st.error(f"Could not submit feedback: {e}")

# ============================================================================
# DASHBOARD PAGE
# ============================================================================
elif page == "Dashboard":
    st.markdown("# Analytics Dashboard")
    st.caption("Real-time incident metrics and historical analysis")
    st.markdown("---")
    
    try:
        stats = httpx.get(f"{API_BASE}/api/stats", timeout=10.0).json()
        incidents = httpx.get(f"{API_BASE}/api/incidents", timeout=10.0).json()["incidents"]
    except Exception as e:
        st.error("🚫 Backend not reachable")
        st.caption(str(e))
        st.stop()
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Incidents", stats["total_incidents"])
    col2.metric("Avg Confidence", f"{stats['avg_confidence'] * 100:.0f}%")
    col3.metric("Avg Latency", f"{stats['avg_latency']:.2f}s")
    
    st.markdown("---")
    st.markdown("### Recent Incidents")
    
    if incidents:
        df = pd.DataFrame(incidents)
        df["confidence"] = df["confidence"].apply(lambda x: f"{x * 100:.0f}%")
        df["created_at"] = pd.to_datetime(df["created_at"]).dt.strftime("%Y-%m-%d %H:%M")
        df = df[["id", "service_name", "severity", "confidence", "created_at"]]
        df.columns = ["Incident ID", "Service", "Severity", "Confidence", "Created At"]
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No incidents analyzed yet. Run an analysis from the Analyze page.")
    
    if stats.get("by_severity"):
        st.markdown("---")
        st.markdown("### Severity Distribution")
        sev_df = pd.DataFrame(list(stats["by_severity"].items()), columns=["Severity", "Count"])
        st.bar_chart(sev_df.set_index("Severity"))

# ============================================================================
# OBSERVABILITY PAGE
# ============================================================================
elif page == "Observability":
    st.markdown("# Observability")
    st.caption("System health, LLMOps metrics, and performance tracking")
    st.markdown("---")
    
    try:
        stats = httpx.get(f"{API_BASE}/api/stats", timeout=10.0).json()
    except Exception as e:
        st.error("🚫 Backend not reachable")
        st.caption(str(e))
        st.stop()
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Requests Processed", stats["total_incidents"])
    col2.metric("Avg Confidence", f"{stats['avg_confidence'] * 100:.0f}%")
    col3.metric("Avg Latency", f"{stats['avg_latency']:.2f}s")
    
    st.markdown("---")
    st.markdown("### LLMOps Configuration")
    
    config_data = {
        "Setting": [
            "LLM Model", "Provider", "RAG Method", "Vector Store",
            "Routing Logic", "LangSmith Tracing", "Max Retries"
        ],
        "Value": [
            "Llama 3 70B", "Groq", "BM25 + PageIndex", "None (Vectorless)",
            "confidence >= 0.8 -> fix | retry < 2 | else escalate", "Enabled", "2"
        ]
    }
    config_df = pd.DataFrame(config_data)
    st.dataframe(config_df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    st.markdown("### Performance Metrics")
    
    if stats.get("by_severity"):
        sev_df = pd.DataFrame(list(stats["by_severity"].items()), columns=["Severity", "Count"])
        st.bar_chart(sev_df.set_index("Severity"))

# ============================================================================
# ARCHITECTURE PAGE
# ============================================================================
elif page == "Architecture":
    st.markdown("# System Architecture")
    st.caption("How NeuralOps works — from log ingestion to fix generation")
    st.markdown("---")
    
    st.markdown("### System Flow")
    st.code("""
Input Logs
    ↓
FastAPI (POST /api/analyze)
    ↓
LangGraph Workflow
    ├── parse_logs     (Log Analyzer Agent + Groq LLM)
    ├── retrieve       (BM25 + PageIndex RAG)
    ├── analyze        (Root Cause Agent + Groq LLM)
    ├── route          (confidence >= 0.8 -> fix | retry | escalate)
    └── generate_fix   (Fix Agent + Groq LLM)
    ↓
SQLite (incident stored)
    ↓
JSON Response
    ↓
Streamlit UI
    """)
    
    st.markdown("---")
    st.markdown("### Technology Stack")
    
    tech_data = {
        "Layer": ["API", "Orchestration", "LLM", "RAG", "Knowledge Base", "LLMOps", "Frontend"],
        "Technology": ["FastAPI + Uvicorn", "LangGraph", "Groq (Llama 3 70B)", "BM25 + PageIndex", "SQLite", "LangSmith", "Streamlit"],
        "Purpose": [
            "High-performance async endpoints",
            "Workflow state management",
            "Fast inference, high accuracy",
            "Keyword-based retrieval without vectors",
            "Lightweight incident storage",
            "Tracing, monitoring, evaluation",
            "Interactive dashboard UI"
        ]
    }
    tech_df = pd.DataFrame(tech_data)
    st.dataframe(tech_df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    st.markdown("### Confidence Routing Logic")
    
    routing_data = {
        "Confidence Score": [">= 0.8", "0.6 - 0.8", "< 0.6"],
        "Action": ["Generate fix immediately", "Retry (max 2 attempts)", "Escalate to human review"]
    }
    routing_df = pd.DataFrame(routing_data)
    st.dataframe(routing_df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    st.markdown("### Evaluation Scoring")
    
    eval_data = {
        "Score Range": [">= 0.8", "0.6 - 0.79", "< 0.6"],
        "Label": ["High Confidence", "Medium Confidence", "Low Confidence"]
    }
    eval_df = pd.DataFrame(eval_data)
    st.dataframe(eval_df, use_container_width=True, hide_index=True)

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown(
    '<div style="text-align: center; color: #64748b; font-size: 0.75rem; padding: 1rem 0;">'
    '© 2026 NeuralOps | AI Incident Analysis Platform | Engineered with ❤️ by Ratnaprava Mohapatra'
    '</div>',
    unsafe_allow_html=True
)
