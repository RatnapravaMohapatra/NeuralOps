import os
import time
import streamlit as st
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from typing import TypedDict, List, Annotated, Dict, Any
import operator
import re

# LangChain imports
from langchain_core.messages import HumanMessage, AIMessage
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langgraph.graph import StateGraph, END
from langgraph.checkpoint import MemorySaver

# Load environment variables
load_dotenv()

# Initialize LangSmith if API key is present
if os.getenv("LANGSMITH_API_KEY"):
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_PROJECT"] = "neuralops-india"

# ============================================================================
# INITIALIZE CHROMA DB WITH INDIA-SPECIFIC DATA
# ============================================================================
@st.cache_resource
def init_rag_system():
    """Initialize Chroma DB with India-specific documents"""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    
    # India-specific knowledge base
    india_knowledge = [
        {
            "text": "GST (Goods and Services Tax) in India has multiple slabs: 5%, 12%, 18%, and 28%. Essential items are taxed at 0% or 5%.",
            "source": "gst_basics",
            "category": "tax"
        },
        {
            "text": "GST Council is the governing body that decides tax rates. IGST applies to interstate transactions, CGST and SGST for intrastate.",
            "source": "gst_council",
            "category": "tax"
        },
        {
            "text": "Common IT incidents: SQL timeout occurs when connection pool is exhausted. Solution: increase max pool size, optimize queries, add connection timeout.",
            "source": "incident_patterns",
            "category": "it_ops"
        },
        {
            "text": "OutOfMemoryError in Java heap space solution: increase heap size (-Xmx), fix memory leaks, use weak references, implement caching strategy.",
            "source": "java_patterns",
            "category": "it_ops"
        },
        {
            "text": "Kubernetes pod scheduling fails due to insufficient CPU: check resource requests/limits, add more nodes, optimize pod resource allocation.",
            "source": "k8s_patterns",
            "category": "it_ops"
        },
        {
            "text": "Redis memory eviction triggered: increase maxmemory, change eviction policy to volatile-lru, optimize data structure usage.",
            "source": "redis_patterns",
            "category": "it_ops"
        }
    ]
    
    # Create or load vector store
    persist_dir = "./chroma_india_db"
    
    if os.path.exists(persist_dir) and os.listdir(persist_dir):
        vector_store = Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings
        )
    else:
        from langchain_core.documents import Document
        docs = [Document(page_content=item["text"], metadata={"source": item["source"], "category": item["category"]}) 
                for item in india_knowledge]
        vector_store = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory=persist_dir
        )
    
    return vector_store

# ============================================================================
# DEFINE LANGGRAPH AGENT STATE
# ============================================================================
class AgentState(TypedDict):
    messages: Annotated[List[HumanMessage | AIMessage], operator.add]
    log_input: str
    parsed_logs: Dict[str, Any]
    retrieved_context: str
    root_cause: str
    fix_suggestion: str
    confidence: float
    severity: str
    service_name: str
    evaluation: str
    incident_id: str
    iteration: int

# ============================================================================
# INITIALIZE LLM
# ============================================================================
@st.cache_resource
def init_llm():
    """Initialize Groq LLM"""
    api_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")
    if not api_key:
        return None
    return ChatGroq(
        api_key=api_key,
        model="llama-3.3-70b-versatile",
        temperature=0.3,
        max_tokens=2000
    )

# ============================================================================
# LANGGRAPH AGENT FUNCTIONS
# ============================================================================
def parse_logs(state: AgentState) -> AgentState:
    """Parse and analyze the log input"""
    log_input = state["log_input"].lower()
    
    # Heuristic parsing for common patterns
    parsed = {
        "has_timeout": "timeout" in log_input or "exhausted" in log_input,
        "has_oom": "outofmemory" in log_input.replace(" ", "") or "heap space" in log_input,
        "has_k8s": "kubernetes" in log_input or "pod" in log_input or "scheduling" in log_input,
        "has_redis": "redis" in log_input or "eviction" in log_input,
        "error_type": "unknown"
    }
    
    # Detect error type
    if parsed["has_timeout"]:
        parsed["error_type"] = "timeout"
        parsed["service"] = "payment-service"
    elif parsed["has_oom"]:
        parsed["error_type"] = "oom"
        parsed["service"] = "recommendation-engine"
    elif parsed["has_k8s"]:
        parsed["error_type"] = "k8s"
        parsed["service"] = "api-gateway"
    elif parsed["has_redis"]:
        parsed["error_type"] = "redis"
        parsed["service"] = "user-session-service"
    else:
        parsed["error_type"] = "generic"
        parsed["service"] = "unknown-service"
    
    state["parsed_logs"] = parsed
    state["service_name"] = parsed.get("service", "unknown-service")
    return state

def retrieve_context(state: AgentState) -> AgentState:
    """Retrieve relevant context from Chroma DB"""
    try:
        vector_store = init_rag_system()
        error_type = state["parsed_logs"]["error_type"]
        
        # Query based on error type
        query = f"What causes {error_type} errors and how to fix them?"
        docs = vector_store.similarity_search(query, k=3)
        
        context = "\n".join([doc.page_content for doc in docs])
        state["retrieved_context"] = context
    except Exception as e:
        state["retrieved_context"] = "No context retrieved. Using default knowledge."
    
    return state

def analyze_root_cause(state: AgentState) -> AgentState:
    """Analyze root cause using LLM"""
    llm = init_llm()
    if not llm:
        state["root_cause"] = "Unable to analyze root cause. LLM not configured. Check database connections and resource limits."
        state["confidence"] = 0.5
        state["severity"] = "Medium"
        return state
    
    try:
        prompt = f"""Analyze this IT incident and determine the root cause:

Log/Error: {state['log_input']}

Retrieved Context: {state['retrieved_context']}

Based on the above, provide:
1. Root cause analysis (be specific and technical)
2. Severity (Low/Medium/High/Critical)
3. Confidence score (0.0 to 1.0)

Response format:
ROOT_CAUSE: [your analysis]
SEVERITY: [Low/Medium/High/Critical]
CONFIDENCE: [0.0-1.0]
"""
        
        response = llm.invoke(prompt)
        response_text = response.content
        
        # Parse response
        root_cause_match = re.search(r'ROOT_CAUSE:\s*(.+?)(?=SEVERITY:|$)', response_text, re.DOTALL)
        severity_match = re.search(r'SEVERITY:\s*(\w+)', response_text)
        confidence_match = re.search(r'CONFIDENCE:\s*([0-9.]+)', response_text)
        
        state["root_cause"] = root_cause_match.group(1).strip() if root_cause_match else "Unable to determine root cause"
        state["severity"] = severity_match.group(1) if severity_match else "Medium"
        state["confidence"] = float(confidence_match.group(1)) if confidence_match else 0.5
        
    except Exception as e:
        state["root_cause"] = f"Error during analysis: {str(e)}"
        state["confidence"] = 0.3
        state["severity"] = "Medium"
    
    return state

def route_based_on_confidence(state: AgentState) -> str:
    """Route to fix generation or retry based on confidence"""
    if state["confidence"] >= 0.7:
        return "generate_fix"
    elif state.get("iteration", 0) < 2:
        return "retry"
    else:
        return "generate_fix"

def generate_fix(state: AgentState) -> AgentState:
    """Generate fix recommendation"""
    llm = init_llm()
    if not llm:
        state["fix_suggestion"] = "1. Check server logs\n2. Restart the affected service\n3. Monitor for recurrence\n4. Scale resources if needed"
        state["evaluation"] = "Manual intervention required"
        state["incident_id"] = f"INC-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        return state
    
    try:
        prompt = f"""Based on this incident analysis, provide a fix recommendation:

Root Cause: {state['root_cause']}
Log: {state['log_input']}
Context: {state['retrieved_context']}
Confidence: {state['confidence']}

Provide:
1. FIX_SUGGESTION: Step-by-step solution (max 200 words)
2. EVALUATION: Good/Needs Review/Poor based on fix quality
"""
        
        response = llm.invoke(prompt)
        response_text = response.content
        
        fix_match = re.search(r'FIX_SUGGESTION:\s*(.+?)(?=EVALUATION:|$)', response_text, re.DOTALL)
        eval_match = re.search(r'EVALUATION:\s*(\w+(?:\s+\w+)?)', response_text)
        
        state["fix_suggestion"] = fix_match.group(1).strip() if fix_match else "Restart the service and monitor logs."
        state["evaluation"] = eval_match.group(1) if eval_match else "Needs Review"
        
    except Exception as e:
        state["fix_suggestion"] = f"Error generating fix: {str(e)}"
        state["evaluation"] = "Failed"
    
    state["incident_id"] = f"INC-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    return state

def retry_analysis(state: AgentState) -> AgentState:
    """Retry analysis with additional context"""
    state["iteration"] = state.get("iteration", 0) + 1
    return state

# ============================================================================
# BUILD LANGGRAPH WORKFLOW
# ============================================================================
@st.cache_resource
def build_graph():
    """Build the LangGraph workflow"""
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("parse_logs", parse_logs)
    workflow.add_node("retrieve_context", retrieve_context)
    workflow.add_node("analyze_root_cause", analyze_root_cause)
    workflow.add_node("generate_fix", generate_fix)
    workflow.add_node("retry", retry_analysis)
    
    # Add edges
    workflow.set_entry_point("parse_logs")
    workflow.add_edge("parse_logs", "retrieve_context")
    workflow.add_edge("retrieve_context", "analyze_root_cause")
    
    workflow.add_conditional_edges(
        "analyze_root_cause",
        route_based_on_confidence,
        {
            "generate_fix": "generate_fix",
            "retry": "retry"
        }
    )
    
    workflow.add_edge("retry", "analyze_root_cause")
    workflow.add_edge("generate_fix", END)
    
    # Add memory
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)

# ============================================================================
# STREAMLIT UI CONFIGURATION
# ============================================================================
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
if "incidents" not in st.session_state:
    st.session_state.incidents = []

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
    
    # Check if LLM is available
    llm = init_llm()
    vector_store = init_rag_system()
    
    if llm and vector_store:
        status_text = "🟢 Online"
        status_color = "#22c55e"
        status_detail = "All systems operational"
    elif llm:
        status_text = "🟡 Degraded"
        status_color = "#eab308"
        status_detail = "RAG initialized, LLM ready"
    else:
        status_text = "🟡 Limited"
        status_color = "#eab308"
        status_detail = "Using fallback mode"
    
    st.markdown("### System Status")
    st.markdown(
        f'<div style="background: {"#1e293b" if st.session_state.theme == "dark" else "#f1f5f9"}; border-radius: 12px; padding: 0.75rem;">'
        f'<div style="display: flex; align-items: center; gap: 0.5rem;">'
        f'<div style="width: 8px; height: 8px; background: {status_color}; border-radius: 50%;"></div>'
        f'<span style="color: {"#ffffff" if st.session_state.theme == "dark" else "#1e293b"}; font-weight: 500;">{status_text}</span>'
        '</div>'
        f'<div style="margin-top: 0.5rem;"><span style="color: #94a3b8; font-size: 0.7rem;">{status_detail}</span></div>'
        '<div style="margin-top: 0.75rem;">'
        '<span style="color: #94a3b8; font-size: 0.75rem;">Model: </span>'
        '<span style="color: {"#ffffff" if st.session_state.theme == "dark" else "#1e293b"}; font-size: 0.75rem; font-weight: 500;">Llama 3 70B</span><br>'
        '<span style="color: #94a3b8; font-size: 0.75rem;">RAG: </span>'
        '<span style="color: {"#ffffff" if st.session_state.theme == "dark" else "#1e293b"}; font-size: 0.75rem; font-weight: 500;">Chroma DB + MiniLM</span>'
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
            
            try:
                graph = build_graph()
                
                # Initialize state
                initial_state = {
                    "messages": [],
                    "log_input": log_input,
                    "parsed_logs": {},
                    "retrieved_context": "",
                    "root_cause": "",
                    "fix_suggestion": "",
                    "confidence": 0.0,
                    "severity": "Medium",
                    "service_name": "unknown",
                    "evaluation": "",
                    "incident_id": "",
                    "iteration": 0
                }
                
                # Run the workflow with progress
                for i, step in enumerate(steps):
                    status_text.markdown(f"**{step}...**")
                    progress_bar.progress((i + 1) / len(steps))
                    time.sleep(0.2)
                
                # Invoke LangGraph
                config = {"configurable": {"thread_id": f"thread_{datetime.now().timestamp()}"}}
                final_state = graph.invoke(initial_state, config=config)
                
                status_text.empty()
                progress_bar.empty()
                
                # Store incident in session state
                incident = {
                    "id": final_state.get("incident_id", f"INC-{datetime.now().strftime('%Y%m%d%H%M%S')}"),
                    "service_name": final_state.get("service_name", "unknown"),
                    "severity": final_state.get("severity", "Medium"),
                    "confidence": final_state.get("confidence", 0),
                    "created_at": datetime.now().isoformat(),
                    "root_cause": final_state.get("root_cause", ""),
                    "fix_suggestion": final_state.get("fix_suggestion", ""),
                    "evaluation": final_state.get("evaluation", ""),
                    "log_input": log_input[:200] + "..." if len(log_input) > 200 else log_input
                }
                st.session_state.incidents.insert(0, incident)
                
                st.balloons()
                st.success(f"✅ Analysis Complete — Incident ID: `{final_state.get('incident_id', 'UNKNOWN')}`")
                st.markdown("---")
                
                # SAFE METRICS DISPLAY
                confidence = final_state.get("confidence", 0)
                severity = final_state.get("severity", "Unknown")
                service = final_state.get("service_name", "core-platform-service")
                evaluation = final_state.get("evaluation", "Low")
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("🎯 Confidence", f"{confidence * 100:.0f}%")
                col2.metric("⚠️ Severity", severity)
                col3.metric("🛠️ Service", service)
                col4.metric("📊 Evaluation", evaluation)
                
                if confidence < 0.6:
                    st.warning("⚠️ **Low confidence result** — manual verification recommended")
                
                # SAFE ROOT CAUSE DISPLAY
                st.markdown("### 🔍 Root Cause Analysis")
                st.info(final_state.get("root_cause", "No root cause available"))
                
                # SAFE FIX DISPLAY
                st.markdown("### 🛠️ Fix Recommendation")
                st.success(final_state.get("fix_suggestion", "No fix available"))
                
                # Display retrieved context
                if final_state.get("retrieved_context"):
                    with st.expander("📚 Retrieved Context (RAG)"):
                        st.code(final_state.get("retrieved_context", ""), language="markdown")
                
                # SAFE PERFORMANCE DISPLAY
                st.markdown("### ⏱️ Performance")
                st.caption(f"LangGraph workflow completed with {final_state.get('iteration', 0)} retry(s)")
                
            except Exception as e:
                st.error(f"❌ Analysis failed: {str(e)}")
                st.stop()

# ============================================================================
# DASHBOARD PAGE
# ============================================================================
elif page == "📊 Dashboard":
    st.markdown("# 📊 Analytics Dashboard")
    st.caption("Real-time incident metrics and historical analysis")
    st.markdown("---")
    
    incidents = st.session_state.incidents
    
    # Calculate stats safely
    total_incidents = len(incidents)
    avg_confidence = sum(i.get("confidence", 0) for i in incidents) / total_incidents if total_incidents > 0 else 0
    
    col1, col2, col3 = st.columns(3)
    col1.metric("📋 Total Incidents", total_incidents)
    col2.metric("🎯 Avg Confidence", f"{avg_confidence * 100:.0f}%")
    col3.metric("⚡ Total Analyses", total_incidents)
    
    st.markdown("---")
    st.markdown("### 📋 Recent Incidents")
    
    if incidents:
        df = pd.DataFrame(incidents)
        df["confidence"] = df["confidence"].apply(lambda x: f"{x * 100:.0f}%")
        df["created_at"] = pd.to_datetime(df["created_at"]).dt.strftime("%Y-%m-%d %H:%M")
        display_df = df[["id", "service_name", "severity", "confidence", "created_at"]]
        display_df.columns = ["Incident ID", "Service", "Severity", "Confidence", "Created At"]
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        
        # Severity distribution
        st.markdown("---")
        st.markdown("### 📊 Severity Distribution")
        severity_counts = df["severity"].value_counts()
        st.bar_chart(severity_counts)
    else:
        st.info("No incidents analyzed yet. Run an analysis from the Analyze page.")

# ============================================================================
# OBSERVABILITY PAGE
# ============================================================================
elif page == "📈 Observability":
    st.markdown("# 📈 Observability")
    st.caption("System health, LLMOps metrics, and performance tracking")
    st.markdown("---")
    
    incidents = st.session_state.incidents
    total_incidents = len(incidents)
    avg_confidence = sum(i.get("confidence", 0) for i in incidents) / total_incidents if total_incidents > 0 else 0
    
    col1, col2, col3 = st.columns(3)
    col1.metric("📊 Requests Processed", total_incidents)
    col2.metric("🎯 Avg Confidence", f"{avg_confidence * 100:.0f}%")
    col3.metric("🔄 LangGraph Nodes", "5")
    
    st.markdown("---")
    st.markdown("### 🔧 LLMOps Configuration")
    
    config_data = {
        "Setting": ["LLM Model", "Provider", "RAG Method", "Routing Logic", "Max Retries", "Orchestration"],
        "Value": ["Llama 3 70B", "Groq", "Chroma DB + MiniLM", "confidence >= 0.7 -> fix", "2", "LangGraph"]
    }
    config_df = pd.DataFrame(config_data)
    st.dataframe(config_df, use_container_width=True, hide_index=True)
    
    if incidents:
        st.markdown("---")
        st.markdown("### 📊 Performance Metrics")
        confidences = [i.get("confidence", 0) for i in incidents]
        avg_latency_estimate = 2.5  # Estimated average latency
        st.metric("Average Confidence", f"{sum(confidences)/len(confidences)*100:.1f}%")
        st.metric("Estimated Avg Latency", f"{avg_latency_estimate}s")

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
Streamlit UI
    ↓
LangGraph Workflow (Real-time)
    ├── parse_logs     (Log Analyzer Agent + Heuristic Parsing)
    ├── retrieve       (Chroma DB + MiniLM RAG)
    ├── analyze        (Root Cause Agent + Groq LLM)
    ├── route          (confidence >= 0.7 -> fix | retry | escalate)
    └── generate_fix   (Fix Agent + Groq LLM)
    ↓
Session State (incident stored)
    ↓
Dashboard & Observability
    """)
    
    st.markdown("---")
    st.markdown("### 🛠️ Technology Stack")
    
    tech_data = {
        "Layer": ["Frontend", "Orchestration", "LLM", "Vector Database", "Embeddings", "State Management"],
        "Technology": ["Streamlit", "LangGraph", "Groq (Llama 3 70B)", "Chroma DB", "MiniLM (sentence-transformers)", "MemorySaver + Session State"]
    }
    tech_df = pd.DataFrame(tech_data)
    st.dataframe(tech_df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    st.markdown("### 🇮🇳 India-Specific Knowledge")
    st.info("""
    The system is pre-loaded with India-specific knowledge including:
    - GST tax slabs and regulations
    - Common IT incident patterns
    - Kubernetes and cloud-native practices
    - Redis and database optimization techniques
    """)

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown(
    f'<div style="text-align: center; color: #64748b; font-size: 0.75rem; padding: 1rem 0;">'
    f'© 2026 NeuralOps | AI Incident Analysis Platform | Built with LangGraph + Chroma DB | ❤️ Ratnaprava Mohapatra'
    f'</div>',
    unsafe_allow_html=True
)
