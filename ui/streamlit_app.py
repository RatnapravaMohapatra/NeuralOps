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
# DAY/NIGHT MODE TOGGLE
# ============================================================================
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

# Theme toggle button in top right
col_title, col_theme = st.columns([6, 1])
with col_theme:
    if st.button("🌙" if st.session_state.theme == "light" else "☀️", help="Toggle theme"):
        st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
        st.rerun()

# ============================================================================
# DYNAMIC CSS BASED ON THEME
# ============================================================================
if st.session_state.theme == "dark":
    css = """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:opsz,wght@14..32,300;14..32,400;14..32,500;14..32,600;14..32,700;14..32,800&display=swap');
        
        * { font-family: 'Inter', -apple-system, sans-serif; }
        
        .main .block-container { padding: 2rem 2.5rem; max-width: 1400px; }
        
        .stApp { background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); }
        
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
        
        [data-testid="stSidebar"] * { color: #ffffff !important; }
        [data-testid="stSidebar"] .stMarkdown p { color: #ffffff !important; }
        
        div[data-testid="stRadio"] > div { gap: 0.5rem; }
        
        div[data-testid="stRadio"] label {
            background: transparent;
            padding: 0.75rem 1rem;
            border-radius: 12px;
            transition: all 0.2s ease;
            cursor: pointer;
        }
        
        div[data-testid="stRadio"] label:hover { background: #334155; }
        
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
        }
        
        .stTextArea textarea:focus {
            border-color: #6366f1;
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.2);
        }
        
        .stTextArea textarea::placeholder { color: #94a3b8; }
        
        [data-testid="stMetric"] {
            background: #1e293b;
            padding: 1rem;
            border-radius: 16px;
            border: 1px solid #334155;
        }
        
        [data-testid="stMetric"] label { color: #94a3b8 !important; font-weight: 500; }
        
        [data-testid="stMetric"] [data-testid="stMetricValue"] {
            color: #ffffff !important;
            font-size: 1.8rem;
            font-weight: 700;
        }
        
        .stAlert { border-radius: 12px; border-left-width: 4px; }
        .stAlert [data-testid="stMarkdown"] p { color: #ffffff !important; }
        
        .stDataFrame {
            border-radius: 16px;
            overflow: hidden;
            border: 1px solid #334155;
        }
        
        .stDataFrame table { background: #1e293b; color: #ffffff; }
        .stDataFrame th { background: #0f172a; color: #ffffff; font-weight: 600; }
        .stDataFrame td { color: #e2e8f0; }
        
        hr { margin: 1.5rem 0; border-color: #334155; }
        .stCaption { color: #94a3b8 !important; }
        
        .stProgress > div > div { background: linear-gradient(90deg, #6366f1, #8b5cf6); border-radius: 20px; }
        .stSpinner > div { color: #ffffff !important; }
        footer { display: none; }
        
        code { background: #0f172a; color: #a5b4fc; padding: 0.2rem 0.4rem; border-radius: 6px; }
        .stMarkdown p { color: #e2e8f0; }
        
        .stSelectbox div[data-baseweb="select"] {
            background-color: #1e293b;
            border-color: #334155;
            color: #ffffff;
        }
        
        .stSelectbox svg { fill: #ffffff; }
        .stSlider [data-baseweb="slider"] { background-color: #334155; }
        .stSlider [data-testid="stThumbValue"] { color: #ffffff; }
        
        @media (max-width: 768px) { .main .block-container { padding: 1rem; } }
    </style>
    """
else:
    css = """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:opsz,wght@14..32,300;14..32,400;14..32,500;14..32,600;14..32,700;14..32,800&display=swap');
        
        * { font-family: 'Inter', -apple-system, sans-serif; }
        
        .main .block-container { padding: 2rem 2.5rem; max-width: 1400px; }
        
        .stApp { background: linear-gradient(135deg, #f8fafc 0%, #eef2ff 100%); }
        
        h1, h2, h3, h4, h5, h6 {
            color: #1e293b !important;
            font-weight: 700 !important;
            letter-spacing: -0.02em !important;
        }
        
        h1 {
            font-size: 2.2rem !important;
            background: linear-gradient(135deg, #1e293b, #475569);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem !important;
        }
        
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
            border-right: 1px solid #e2e8f0;
        }
        
        [data-testid="stSidebar"] * { color: #1e293b !important; }
        [data-testid="stSidebar"] .stMarkdown p { color: #1e293b !important; }
        
        div[data-testid="stRadio"] > div { gap: 0.5rem; }
        
        div[data-testid="stRadio"] label {
            background: transparent;
            padding: 0.75rem 1rem;
            border-radius: 12px;
            transition: all 0.2s ease;
            cursor: pointer;
        }
        
        div[data-testid="stRadio"] label:hover { background: #e2e8f0; }
        
        div[data-testid="stRadio"] [data-testid="stMarkdown"] p {
            font-weight: 600;
            color: #1e293b !important;
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
            box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3);
        }
        
        .stTextArea textarea {
            border-radius: 16px;
            border: 1px solid #e2e8f0;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.85rem;
            line-height: 1.5;
            background: #ffffff;
            color: #1e293b;
        }
        
        .stTextArea textarea:focus {
            border-color: #6366f1;
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1);
        }
        
        .stTextArea textarea::placeholder { color: #94a3b8; }
        
        [data-testid="stMetric"] {
            background: #ffffff;
            padding: 1rem;
            border-radius: 16px;
            border: 1px solid #e2e8f0;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }
        
        [data-testid="stMetric"] label { color: #64748b !important; font-weight: 500; }
        
        [data-testid="stMetric"] [data-testid="stMetricValue"] {
            color: #1e293b !important;
            font-size: 1.8rem;
            font-weight: 700;
        }
        
        .stAlert { border-radius: 12px; border-left-width: 4px; }
        .stDataFrame {
            border-radius: 16px;
            overflow: hidden;
            border: 1px solid #e2e8f0;
        }
        
        .stDataFrame table { background: #ffffff; color: #1e293b; }
        .stDataFrame th { background: #f8fafc; color: #1e293b; font-weight: 600; }
        .stDataFrame td { color: #475569; }
        
        hr { margin: 1.5rem 0; border-color: #e2e8f0; }
        .stCaption { color: #64748b !important; }
        
        .stProgress > div > div { background: linear-gradient(90deg, #6366f1, #8b5cf6); border-radius: 20px; }
        footer { display: none; }
        
        code { background: #f1f5f9; color: #6366f1; padding: 0.2rem 0.4rem; border-radius: 6px; }
        .stMarkdown p { color: #475569; }
        
        .stSelectbox div[data-baseweb="select"] {
            background-color: #ffffff;
            border-color: #e2e8f0;
            color: #1e293b;
        }
        
        @media (max-width: 768px) { .main .block-container { padding: 1rem; } }
    </style>
    """

st.markdown(css, unsafe_allow_html=True)

# ============================================================================
# SAMPLE DATA
# ============================================================================
SAMPLES = {
    "🐘 SQL Timeout": (
        "SQLTimeoutException: connection pool exhausted after 30s\n"
        "  at com.zaxxer.hikari.pool.HikariPool.getConnection(HikariPool.java:213)\n"
        "  Service: payment-service | Env: production | Region: us-east-1"
    ),
    "💥 OOM Crash": (
        "java.lang.OutOfMemoryError: Java heap space\n"
        "  at java.util.Arrays.copyOf(Arrays.java:3210)\n"
        "  at com.app.cache.InMemoryStore.put(InMemoryStore.java:88)\n"
        "  Service: recommendation-engine | Heap: 4GB/4GB"
    ),
    "☸️ K8s Failure": (
        "0/3 nodes are available: 3 Insufficient cpu\n"
        "  Warning FailedScheduling pod/api-gateway-6c9d4f\n"
        "  Event: pod evicted due to resource pressure"
    ),
    "⚡ Redis Eviction": (
        "WARN Redis eviction policy maxmemory-policy=allkeys-lru triggered\n"
        "  ERROR Cache miss rate: 94%\n"
        "  Service: user-session-service | Redis memory: 4096/4096 MB"
    ),
}

PAGES = ["🔍 Analyze", "📊 Dashboard", "📈 Observability", "🏗️ Architecture"]

# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    st.markdown("### 🧠 NeuralOps")
    st.caption("AI Incident Analysis Platform")
    st.markdown("---")
    page = st.radio("Navigation", PAGES, label_visibility="collapsed")
    st.markdown("---")
    
    # API Health Check
    try:
        health = httpx.get(f"{API_BASE}/health", timeout=3)
        status_text = "🟢 Online"
        status_color = "#22c55e"
    except:
        status_text = "🔴 Offline"
        status_color = "#ef4444"
    
    st.markdown("### System Status")
    st.markdown(
        f'<div style="background: {"#1e293b" if st.session_state.theme == "dark" else "#f1f5f9"}; border-radius: 12px; padding: 0.75rem;">'
        f'<div style="display: flex; align-items: center; gap: 0.5rem;">'
        f'<div style="width: 8px; height: 8px; background: {status_color}; border-radius: 50%;"></div>'
        f'<span style="color: {"#ffffff" if st.session_state.theme == "dark" else "#1e293b"}; font-weight: 500;">{status_text}</span>'
        '</div>'
        '<div style="margin-top: 0.75rem;">'
        '<span style="color: #94a3b8; font-size: 0.75rem;">Model: </span>'
        '<span style="color: {"#ffffff" if st.session_state.theme == "dark" else "#1e293b"}; font-size: 0.75rem; font-weight: 500;">Llama 3 70B</span><br>'
        '<span style="color: #94a3b8; font-size: 0.75rem;">RAG: </span>'
        '<span style="color: {"#ffffff" if st.session_state.theme == "dark" else "#1e293b"}; font-size: 0.75rem; font-weight: 500;">BM25 + PageIndex</span>'
        '</div>'
        '</div>',
        unsafe_allow_html=True
    )
    
    st.markdown("---")
    st.markdown(
        '<div style="color: #64748b; font-size: 0.7rem; text-align: center;">'
        'Developed by<br><span style="color: #6366f1; font-weight: 700;">Ratnaprava Mohapatra</span>'
        '</div>',
        unsafe_allow_html=True
    )

# ============================================================================
# ANALYZE PAGE
# ============================================================================
if page == "🔍 Analyze":
    st.markdown("# 🧠 NeuralOps")
    st.caption("AI-Powered Incident Analysis — Understand, Diagnose, and Fix Issues Instantly")
    st.markdown("---")
    
    col1, col2 = st.columns([2.5, 1.2])
    
    with col1:
        log_input = st.text_area(
            "**Log or Error Message**",
            height=200,
            placeholder="Paste your log, error, or stack trace here...",
            label_visibility="collapsed",
            key="log_text"
        )
    
    with col2:
        st.markdown("**📋 Quick Samples**")
        for label, sample_text in SAMPLES.items():
            if st.button(label, use_container_width=True, key=f"sample_{label[:10]}"):
                st.session_state["prefill"] = sample_text
                st.rerun()
    
    if "prefill" in st.session_state:
        log_input = st.session_state.pop("prefill")
    
    st.markdown("---")
    
    run = st.button("🚀 Run Analysis", type="primary", use_container_width=False)
    
    if run:
        if not log_input or not log_input.strip():
            st.error("❌ Please enter a log or error message to analyze.")
        else:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            steps = ["📝 Parsing logs", "🔍 Retrieving context", "🧠 Analyzing root cause", "🔧 Generating fix"]
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
                    st.error("⏳ Request timed out. Backend may be slow.")
                    st.stop()
                except httpx.HTTPStatusError as e:
                    st.error(f"⚠️ API Error {e.response.status_code}")
                    st.stop()
                except Exception as e:
                    st.error("🚫 Backend not reachable")
                    st.stop()
            
            st.balloons()
            st.success(f"✅ Analysis Complete — Incident ID: `{result['incident_id']}`")
            st.markdown("---")
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("🎯 Confidence", f"{result['confidence'] * 100:.0f}%")
            col2.metric("⚠️ Severity", result['severity'])
            col3.metric("🛠️ Service", result['service_name'])
            col4.metric("📊 Evaluation", result['evaluation'])
            
            if result["confidence"] < 0.6:
                st.warning("⚠️ **Low confidence result** — manual verification recommended")
            
            st.markdown("### 🔍 Root Cause Analysis")
            st.info(result['root_cause'])
            
            st.markdown("### 🛠️ Fix Recommendation")
            st.success(result['fix_suggestion'])
            
            st.markdown("### ⏱️ Performance")
            st.caption(f"End-to-end latency: **{result['latency']}s**")

# ============================================================================
# DASHBOARD PAGE
# ============================================================================
elif page == "📊 Dashboard":
    st.markdown("# 📊 Analytics Dashboard")
    st.caption("Real-time incident metrics and historical analysis")
    st.markdown("---")
    
    try:
        stats = httpx.get(f"{API_BASE}/api/stats", timeout=10.0).json()
        data = httpx.get(f"{API_BASE}/api/incidents", timeout=10.0).json()
        incidents = data.get("incidents", [])
    except Exception as e:
        st.error("🚫 Backend not reachable")
        st.stop()
    
    col1, col2, col3 = st.columns(3)
    col1.metric("📋 Total Incidents", stats.get("total_incidents", 0))
    col2.metric("🎯 Avg Confidence", f"{stats.get('avg_confidence', 0) * 100:.0f}%")
    col3.metric("⚡ Avg Latency", f"{stats.get('avg_latency', 0):.2f}s")
    
    st.markdown("---")
    st.markdown("### 📋 Recent Incidents")
    
    if incidents:
        df = pd.DataFrame(incidents)
        df["confidence"] = df["confidence"].apply(lambda x: f"{x * 100:.0f}%")
        df["created_at"] = pd.to_datetime(df["created_at"]).dt.strftime("%Y-%m-%d %H:%M")
        df = df[["id", "service_name", "severity", "confidence", "created_at"]]
        df.columns = ["Incident ID", "Service", "Severity", "Confidence", "Created At"]
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No incidents analyzed yet. Run an analysis from the Analyze page.")

# ============================================================================
# OBSERVABILITY PAGE
# ============================================================================
elif page == "📈 Observability":
    st.markdown("# 📈 Observability")
    st.caption("System health, LLMOps metrics, and performance tracking")
    st.markdown("---")
    
    try:
        stats = httpx.get(f"{API_BASE}/api/stats", timeout=10.0).json()
    except Exception as e:
        st.error("🚫 Backend not reachable")
        st.stop()
    
    col1, col2, col3 = st.columns(3)
    col1.metric("📊 Requests Processed", stats.get("total_incidents", 0))
    col2.metric("🎯 Avg Confidence", f"{stats.get('avg_confidence', 0) * 100:.0f}%")
    col3.metric("⚡ Avg Latency", f"{stats.get('avg_latency', 0):.2f}s")
    
    st.markdown("---")
    st.markdown("### 🔧 LLMOps Configuration")
    
    config_data = {
        "Setting": ["LLM Model", "Provider", "RAG Method", "Routing Logic", "Max Retries"],
        "Value": ["Llama 3 70B", "Groq", "BM25 + PageIndex", "confidence >= 0.8 -> fix", "2"]
    }
    config_df = pd.DataFrame(config_data)
    st.dataframe(config_df, use_container_width=True, hide_index=True)
    
    if stats.get("by_severity"):
        st.markdown("---")
        st.markdown("### 📊 Severity Distribution")
        sev_df = pd.DataFrame(list(stats["by_severity"].items()), columns=["Severity", "Count"])
        st.bar_chart(sev_df.set_index("Severity"))

# ============================================================================
# ARCHITECTURE PAGE
# ============================================================================
elif page == "🏗️ Architecture":
    st.markdown("# 🏗️ System Architecture")
    st.caption("How NeuralOps works — from log ingestion to fix generation")
    st.markdown("---")
    
    st.markdown("### 🔄 System Flow")
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
    st.markdown("### 🛠️ Technology Stack")
    
    tech_data = {
        "Layer": ["API", "Orchestration", "LLM", "RAG", "Frontend"],
        "Technology": ["FastAPI + Uvicorn", "LangGraph", "Groq (Llama 3 70B)", "BM25 + PageIndex", "Streamlit"]
    }
    tech_df = pd.DataFrame(tech_data)
    st.dataframe(tech_df, use_container_width=True, hide_index=True)

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown(
    f'<div style="text-align: center; color: #64748b; font-size: 0.75rem; padding: 1rem 0;">'
    f'© 2026 NeuralOps | AI Incident Analysis Platform | Built with ❤️ by Ratnaprava Mohapatra'
    f'</div>',
    unsafe_allow_html=True
)
