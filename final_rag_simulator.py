"""
FINAL RAG Agent Simulator (FIXED)
==================================
✅ FIX 1: Proper Server Queue - Each server processes ONE task at a time
✅ FIX 2: RAG Agent is PRIMARY decision maker, ML Model only VERIFIES

Infrastructure:
- Edge: Servers 1-4 (4 parallel tasks max)
- Fog: Servers 5-6 (2 parallel tasks max)
- Cloud: Server 7 ONLY (1 task at a time - others MUST QUEUE)

Timing:
- Task arrives every 4 seconds
- Edge completion: 3 seconds
- Fog completion: 6 seconds
- Cloud completion: 10 seconds

Decision Flow:
1. Network condition arrives from CSV
2. RAG Agent (Groq LLM + Vector Search) → PRIMARY DECISION
3. ML Model (Gradient Boosting) → VERIFICATION ONLY
4. Agent checks available servers and executes

Run: streamlit run final_rag_simulator.py
"""

import os
import sys
import re
import pickle
import time
import random
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum
from collections import deque

# Page config
st.set_page_config(
    page_title="RAG Agent Simulator (FIXED)",
    page_icon="🤖",
    layout="wide"
)

# Paths
BASE_DIR = Path(__file__).parent
STORE_DIR = BASE_DIR / "network_faiss_store"
DATA_DIR = BASE_DIR / "data"
CSV_PATH = DATA_DIR / "simulation_data.csv"

if not CSV_PATH.exists():
    CSV_PATH = Path(r"c:\Users\Sahil Padole\Videos\AI_agent_ml_threshold\data\edgesimpy_failure_ml_+_thresh_(gb)_no_failure_20251223_075347_results.csv")

# Get API key
try:
    DEFAULT_GROQ_KEY = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY", ""))
except:
    DEFAULT_GROQ_KEY = os.environ.get("GROQ_API_KEY", "")

# =============================================================================
# CONSTANTS
# =============================================================================
TASK_ARRIVAL_INTERVAL = 4  # seconds

COMPLETION_TIMES = {
    "Edge": 3,   # seconds
    "Fog": 6,    # seconds
    "Cloud": 10  # seconds
}

# Server configuration - EACH SERVER PROCESSES ONE TASK AT A TIME
SERVERS = {
    1: {"layer": "Edge", "latency_ms": 5},
    2: {"layer": "Edge", "latency_ms": 5},
    3: {"layer": "Edge", "latency_ms": 5},
    4: {"layer": "Edge", "latency_ms": 5},
    5: {"layer": "Fog", "latency_ms": 25},
    6: {"layer": "Fog", "latency_ms": 25},
    7: {"layer": "Cloud", "latency_ms": 100},  # ONLY 1 SERVER!
}

MAX_PARALLEL_BY_LAYER = {
    "Edge": 4,   # 4 servers
    "Fog": 2,    # 2 servers
    "Cloud": 1   # 1 server ONLY - tasks MUST queue!
}


class TaskStatus(Enum):
    WAITING = "⏳ Waiting"
    RUNNING = "🔄 Running"
    COMPLETED = "✅ Done"


class DecisionMaker(Enum):
    RAG_PRIMARY = "🧠 RAG (Primary)"
    RAG_FALLBACK = "⚠️ RAG + Fallback"
    RAG_QUEUED = "⏳ RAG + Queued"


@dataclass
class ServerState:
    """Track server state - ONE TASK AT A TIME."""
    server_id: int
    layer: str
    latency_ms: int
    current_task: Optional[int] = None
    busy_until: Optional[datetime] = None
    tasks_completed: int = 0
    queue: List[int] = field(default_factory=list)  # Tasks waiting
    
    def is_busy(self, current_time: datetime) -> bool:
        if self.busy_until is None:
            return False
        return current_time < self.busy_until
    
    def is_available(self, current_time: datetime) -> bool:
        return not self.is_busy(current_time)
    
    def assign_task(self, task_id: int, duration: float, current_time: datetime):
        self.current_task = task_id
        self.busy_until = current_time + timedelta(seconds=duration)
    
    def release(self):
        self.tasks_completed += 1
        self.current_task = None
        self.busy_until = None
    
    def get_wait_time(self, current_time: datetime) -> float:
        """Calculate wait time if task is added now."""
        if not self.is_busy(current_time):
            return 0.0
        return (self.busy_until - current_time).total_seconds()


@dataclass
class Task:
    """Task with RAG prediction (primary) and ML verification (secondary)."""
    task_id: int
    arrival_time: datetime
    network: Dict
    
    # RAG Agent Decision (PRIMARY)
    rag_layer: str
    rag_server: int
    rag_reasoning: str
    
    # ML Model Verification (SECONDARY)
    ml_layer: str
    ml_confidence: float
    ml_matches_rag: bool
    
    # Agent Execution
    final_layer: str
    final_server: int
    execution_note: str
    decision_maker: DecisionMaker
    
    # Status tracking
    status: TaskStatus = TaskStatus.WAITING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    wait_time: float = 0.0


# =============================================================================
# LOAD MODELS
# =============================================================================
@st.cache_resource
def load_models():
    """Load RAG components and ML model."""
    models = {}
    
    # ML model
    model_path = STORE_DIR / "gradient_boosting_model.pkl"
    encoder_path = STORE_DIR / "label_encoder.pkl"
    scaler_path = STORE_DIR / "scaler.pkl"
    
    if model_path.exists() and encoder_path.exists():
        with open(model_path, 'rb') as f:
            models['ml_model'] = pickle.load(f)
        with open(encoder_path, 'rb') as f:
            models['label_encoder'] = pickle.load(f)
    
    if scaler_path.exists():
        with open(scaler_path, 'rb') as f:
            models['scaler'] = pickle.load(f)
    
    # TF-IDF embedder
    embedder_path = STORE_DIR / "tfidf_embedder.pkl"
    if embedder_path.exists():
        with open(embedder_path, 'rb') as f:
            data = pickle.load(f)
            models['vectorizer'] = data['vectorizer']
            models['svd'] = data['svd']
    
    # FAISS index
    faiss_path = STORE_DIR / "faiss.index"
    meta_path = STORE_DIR / "metadata.pkl"
    
    if faiss_path.exists() and meta_path.exists():
        import faiss
        models['faiss_index'] = faiss.read_index(str(faiss_path))
        with open(meta_path, 'rb') as f:
            models['metadata'] = pickle.load(f)
    
    return models


# =============================================================================
# RAG AGENT (PRIMARY DECISION MAKER)
# =============================================================================
class RAGAgent:
    """RAG Agent - PRIMARY decision maker using Groq LLM + Vector Search."""
    
    def __init__(self, models: dict, api_key: str):
        self.models = models
        self.api_key = api_key
    
    def search_similar(self, network: Dict, top_k: int = 5) -> List[str]:
        """Search FAISS for similar network conditions."""
        if 'vectorizer' not in self.models or 'faiss_index' not in self.models:
            return []
        
        query = f"Datarate {network['datarate_mbps']:.1f}Mbps SINR {network['sinr']:.1f}dB Latency {network['latency_ms']:.1f}ms"
        
        tfidf = self.models['vectorizer'].transform([query])
        query_emb = self.models['svd'].transform(tfidf).astype('float32')
        distances, indices = self.models['faiss_index'].search(query_emb, top_k)
        
        docs = self.models.get('metadata', {}).get('documents', [])
        return [docs[idx] for idx in indices[0] if 0 <= idx < len(docs)]
    
    def decide(self, network: Dict) -> Tuple[str, int, str]:
        """
        PRIMARY DECISION: RAG Agent (Groq LLM + Vector Search)
        Returns: (layer, server_id, reasoning)
        """
        # Step 1: Get similar scenarios
        similar = self.search_similar(network)
        context = "\n".join(similar[:3]) if similar else "No similar scenarios found."
        
        # Step 2: Query Groq LLM
        if not self.api_key:
            return self._rule_based_fallback(network)
        
        try:
            from groq import Groq
            client = Groq(api_key=self.api_key)
            
            prompt = f"""You are an Edge-Fog-Cloud deployment agent.

NETWORK CONDITIONS:
- Data Rate: {network['datarate_mbps']:.2f} Mbps
- SINR: {network['sinr']:.2f} dB
- Latency: {network['latency_ms']:.2f} ms
- RSRP: {network['rsrp_dbm']:.2f} dBm

INFRASTRUCTURE:
- Edge (Servers 1-4): Low latency (<10ms), 4 parallel tasks max
- Fog (Servers 5-6): Medium latency (~25ms), 2 parallel tasks max
- Cloud (Server 7): High latency (~100ms), 1 task at a time (queue!)

RULES (same as EdgeSimPy ML+Thresh):
- Latency < 20ms AND Datarate < 16.6 Mbps → Edge
- Datarate 9.6-16.6 Mbps AND SINR > 10 dB → Fog
- Datarate >= 16.6 Mbps OR SINR <= 10 dB → Cloud

SIMILAR SCENARIOS:
{context}

RESPOND IN THIS EXACT FORMAT:
LAYER: [Edge/Fog/Cloud] | SERVER: [1-7] | REASON: [brief explanation]"""

            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=100
            )
            
            result = response.choices[0].message.content.strip()
            
            # Parse
            layer_match = re.search(r'LAYER:\s*(Edge|Fog|Cloud)', result, re.IGNORECASE)
            server_match = re.search(r'SERVER:\s*(\d+)', result)
            reason_match = re.search(r'REASON:\s*(.+)', result, re.IGNORECASE)
            
            if layer_match:
                layer = layer_match.group(1).capitalize()
                server = int(server_match.group(1)) if server_match else self._default_server(layer)
                reason = reason_match.group(1) if reason_match else result
                return layer, server, f"🧠 RAG: {reason}"
            else:
                return self._rule_based_fallback(network)
                
        except Exception as e:
            return self._rule_based_fallback(network)
    
    def _rule_based_fallback(self, network: Dict) -> Tuple[str, int, str]:
        """Fallback when LLM unavailable - SAME LOGIC AS EdgeSimPy ML+Thresh."""
        latency = network.get('latency_ms', 100)
        datarate = network.get('datarate_mbps', 10)
        sinr = network.get('sinr', 10)
        
        # EdgeSimPy ML+Thresh (GB) Rules:
        # Rule 1: latency < 20ms AND datarate < 16.6 Mbps → Edge
        # Rule 2: 9.6 <= datarate < 16.6 AND SINR > 10 → Fog  
        # Rule 3: datarate >= 16.6 OR SINR <= 10 → Cloud
        
        if latency < 20 and datarate < 16.6:
            server = random.choice([1, 2, 3, 4])
            return "Edge", server, f"⚡ Low latency ({latency:.1f}ms) + datarate < 16.6 → Edge"
        elif 9.6 <= datarate < 16.6 and sinr > 10:
            server = random.choice([5, 6])
            return "Fog", server, f"📊 Datarate 9.6-16.6 + SINR > 10 → Fog"
        else:
            return "Cloud", 7, f"☁️ Datarate >= 16.6 OR SINR <= 10 → Cloud"
    
    def _default_server(self, layer: str) -> int:
        servers = {"Edge": [1, 2, 3, 4], "Fog": [5, 6], "Cloud": [7]}
        return random.choice(servers.get(layer, [7]))


# =============================================================================
# ML MODEL (VERIFICATION ONLY)
# =============================================================================
class MLVerifier:
    """ML Model - SECONDARY verification only."""
    
    def __init__(self, models: dict):
        self.models = models
        self.feature_cols = ['datarate', 'sinr', 'latency_ms', 'rsrp_dbm', 'cpu_demand', 'memory_demand']
    
    def verify(self, network: Dict) -> Tuple[str, float]:
        """
        VERIFICATION: ML Model predicts layer for comparison.
        Returns: (layer, confidence)
        """
        if 'ml_model' not in self.models:
            return "Unknown", 0.0
        
        features = [
            network.get('datarate_mbps', 10) * 1e6,  # Convert to bps
            network.get('sinr', 10),
            network.get('latency_ms', 50),
            network.get('rsrp_dbm', -100),
            network.get('cpu_demand', 30),
            network.get('memory_demand', 100)
        ]
        
        X = np.array([features])
        
        if 'scaler' in self.models:
            X = self.models['scaler'].transform(X)
        
        pred = self.models['ml_model'].predict(X)[0]
        prob = self.models['ml_model'].predict_proba(X)[0].max()
        layer = self.models['label_encoder'].inverse_transform([pred])[0]
        
        return layer, prob


# =============================================================================
# AGENT EXECUTOR (CHECKS AVAILABLE SERVERS)
# =============================================================================
def execute_decision(
    rag_layer: str,
    rag_server: int,
    servers: Dict[int, ServerState],
    network: Dict,
    current_time: datetime
) -> Tuple[str, int, str, DecisionMaker]:
    """
    AGENT EXECUTION: Check available servers and execute.
    - If RAG server available → execute immediately
    - If RAG server busy → try another server in same layer
    - If ALL servers in layer busy → fallback or QUEUE
    """
    
    # Check if RAG's recommended server is available
    if rag_server in servers:
        server = servers[rag_server]
        if server.is_available(current_time):
            return (
                rag_layer,
                rag_server,
                f"✅ Execute on Server {rag_server} (available)",
                DecisionMaker.RAG_PRIMARY
            )
    
    # RAG server busy - find another in same layer
    layer_servers = [s for s in servers.values() 
                     if s.layer == rag_layer and s.is_available(current_time)]
    
    if layer_servers:
        server = layer_servers[0]
        return (
            rag_layer,
            server.server_id,
            f"✅ Server {rag_server} busy → Using Server {server.server_id}",
            DecisionMaker.RAG_PRIMARY
        )
    
    # ALL servers in RAG layer are BUSY - need to decide
    # Option 1: Try fallback to another layer
    # Option 2: Queue on the RAG-recommended layer
    
    latency_req = network.get('latency_ms', 100)
    
    # Define fallback order based on original recommendation
    if rag_layer == "Cloud":
        fallback_order = ["Fog", "Edge"]
    elif rag_layer == "Edge":
        fallback_order = ["Fog"] if latency_req < 50 else ["Fog", "Cloud"]
    else:  # Fog
        fallback_order = ["Edge"] if latency_req < 30 else ["Edge", "Cloud"]
    
    # Try fallback layers
    for fallback in fallback_order:
        fallback_servers = [s for s in servers.values()
                          if s.layer == fallback and s.is_available(current_time)]
        if fallback_servers:
            server = fallback_servers[0]
            return (
                fallback,
                server.server_id,
                f"⚠️ {rag_layer} FULL → Fallback to {fallback} Server {server.server_id}",
                DecisionMaker.RAG_FALLBACK
            )
    
    # ALL SERVERS BUSY - Must QUEUE on RAG's recommended layer
    # Find server that will be free earliest
    rag_layer_servers = [s for s in servers.values() if s.layer == rag_layer]
    earliest_free = min(rag_layer_servers, key=lambda s: s.busy_until or datetime.min)
    wait_time = earliest_free.get_wait_time(current_time)
    
    return (
        rag_layer,
        earliest_free.server_id,
        f"⏳ ALL BUSY! Queued on {rag_layer} Server {earliest_free.server_id} (wait {wait_time:.1f}s)",
        DecisionMaker.RAG_QUEUED
    )


# =============================================================================
# MAIN APPLICATION
# =============================================================================
def load_csv():
    if CSV_PATH.exists():
        return pd.read_csv(CSV_PATH)
    return None


def main():
    st.title("🤖 RAG Agent Simulator (FIXED)")
    
    st.markdown("""
    ## ✅ FIXES Applied:
    
    ### Fix 1: Proper Server Queue
    | Layer | Servers | Max Parallel | Queue Behavior |
    |-------|---------|--------------|----------------|
    | Edge | 1, 2, 3, 4 | 4 tasks | Each server: 1 task |
    | Fog | 5, 6 | 2 tasks | Each server: 1 task |
    | **Cloud** | **7 only** | **1 task** | **Others MUST WAIT** |
    
    ### Fix 2: RAG is PRIMARY Decision Maker
    | Step | Component | Role |
    |------|-----------|------|
    | 1️⃣ | **RAG Agent** | 🧠 **PRIMARY** - Makes deployment decision |
    | 2️⃣ | **ML Model** | 📊 **VERIFY** - Just compares with RAG |
    | 3️⃣ | **Agent** | ✅ **EXECUTE** - Check availability & run |
    """)
    
    # Load models
    models = load_models()
    
    if not models:
        st.error("❌ Models not loaded! Run train_paper_gb_model.py first.")
        return
    
    # Sidebar - API Key
    st.sidebar.header("🔑 Groq API Key")
    groq_key = st.sidebar.text_input(
        "Enter API Key",
        value=DEFAULT_GROQ_KEY,
        type="password",
        help="Get free key at https://console.groq.com"
    )
    
    GROQ_API_KEY = groq_key if groq_key else DEFAULT_GROQ_KEY
    
    if not GROQ_API_KEY:
        st.sidebar.warning("⚠️ No API key - using rule-based fallback")
    else:
        st.sidebar.success("✅ Groq API configured")
    
    # Load CSV
    df = load_csv()
    if df is None:
        st.error("CSV not found!")
        return
    
    st.success(f"✅ Loaded {len(df)} network conditions | FAISS: {models.get('faiss_index').ntotal if 'faiss_index' in models else 0} vectors")
    
    # Initialize agents
    rag_agent = RAGAgent(models, GROQ_API_KEY)
    ml_verifier = MLVerifier(models)
    
    # Sidebar controls
    st.sidebar.header("🎛️ Simulation Controls")
    num_tasks = st.sidebar.slider("Tasks to Simulate", 5, 30, 12)
    speed = st.sidebar.slider("Speed Multiplier", 1, 10, 5)
    
    actual_completion = {k: v / speed for k, v in COMPLETION_TIMES.items()}
    actual_arrival = TASK_ARRIVAL_INTERVAL / speed
    
    st.sidebar.markdown(f"""
    **Timing ({speed}x speed):**
    - Task arrival: every {actual_arrival:.1f}s
    - Edge completion: {actual_completion['Edge']:.1f}s
    - Fog completion: {actual_completion['Fog']:.1f}s
    - Cloud completion: {actual_completion['Cloud']:.1f}s
    """)
    
    # Session state
    if 'running' not in st.session_state:
        st.session_state.running = False
    if 'tasks' not in st.session_state:
        st.session_state.tasks = []
    if 'servers' not in st.session_state:
        st.session_state.servers = {
            sid: ServerState(sid, cfg["layer"], cfg["latency_ms"])
            for sid, cfg in SERVERS.items()
        }
    if 'task_data' not in st.session_state:
        st.session_state.task_data = []
    if 'idx' not in st.session_state:
        st.session_state.idx = 0
    
    # Control buttons
    col1, col2, col3 = st.sidebar.columns(3)
    if col1.button("▶️ Start", type="primary"):
        st.session_state.running = True
        if not st.session_state.task_data:
            indices = random.sample(range(len(df)), min(num_tasks, len(df)))
            st.session_state.task_data = [
                {
                    "datarate_mbps": df.iloc[i]['datarate'] / 1e6,
                    "sinr": df.iloc[i]['sinr'],
                    "latency_ms": df.iloc[i]['latency_ms'],
                    "rsrp_dbm": df.iloc[i]['rsrp_dbm'],
                    "cpu_demand": int(df.iloc[i].get('cpu_demand', 30)),
                    "memory_demand": int(df.iloc[i].get('memory_demand', 100)),
                    "csv_layer": df.iloc[i]['assigned_layer'],
                    "csv_server": int(df.iloc[i]['server_id'])
                }
                for i in indices
            ]
    if col2.button("⏹️ Stop"):
        st.session_state.running = False
    if col3.button("🔄 Reset"):
        st.session_state.tasks = []
        st.session_state.task_data = []
        st.session_state.idx = 0
        st.session_state.running = False
        st.session_state.servers = {
            sid: ServerState(sid, cfg["layer"], cfg["latency_ms"])
            for sid, cfg in SERVERS.items()
        }
        st.rerun()
    
    st.divider()
    
    # Metrics row
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    completed = len([t for t in st.session_state.tasks if t.status == TaskStatus.COMPLETED])
    running = len([t for t in st.session_state.tasks if t.status == TaskStatus.RUNNING])
    waiting = len([t for t in st.session_state.tasks if t.status == TaskStatus.WAITING])
    rag_primary = len([t for t in st.session_state.tasks if t.decision_maker == DecisionMaker.RAG_PRIMARY])
    rag_ml_match = len([t for t in st.session_state.tasks if t.ml_matches_rag])
    
    m1.metric("✅ Done", completed)
    m2.metric("🔄 Running", running)
    m3.metric("⏳ Waiting", waiting)
    m4.metric("🧠 RAG Primary", rag_primary)
    m5.metric("🎯 RAG=ML", rag_ml_match)
    m6.metric("Total", len(st.session_state.tasks))
    
    if st.session_state.task_data:
        st.progress(st.session_state.idx / len(st.session_state.task_data))
    
    st.divider()
    
    # Main display
    left_col, right_col = st.columns([3, 2])
    
    with left_col:
        st.subheader("🎯 Decision Flow")
        decision_placeholder = st.empty()
    
    with right_col:
        st.subheader("🖥️ Server Status")
        server_placeholder = st.empty()
    
    st.divider()
    
    st.subheader("📋 Task Queue & Progress")
    queue_placeholder = st.empty()
    
    st.subheader("📊 Decision Log: RAG (Primary) vs ML (Verify)")
    log_placeholder = st.empty()
    
    # ==========================================================================
    # SIMULATION LOOP
    # ==========================================================================
    if st.session_state.running and st.session_state.task_data:
        
        while st.session_state.idx < len(st.session_state.task_data):
            current_time = datetime.now()
            
            # 1. Check for completed tasks
            for task in st.session_state.tasks:
                if task.status == TaskStatus.RUNNING and task.end_time:
                    if current_time >= task.end_time:
                        task.status = TaskStatus.COMPLETED
                        st.session_state.servers[task.final_server].release()
            
            # 2. Start waiting tasks if server available
            for task in st.session_state.tasks:
                if task.status == TaskStatus.WAITING:
                    server = st.session_state.servers[task.final_server]
                    if server.is_available(current_time):
                        task.start_time = current_time
                        task.end_time = current_time + timedelta(
                            seconds=actual_completion[task.final_layer]
                        )
                        task.wait_time = (current_time - task.arrival_time).total_seconds()
                        task.status = TaskStatus.RUNNING
                        server.assign_task(
                            task.task_id,
                            actual_completion[task.final_layer],
                            current_time
                        )
            
            # 3. New task arrives
            network = st.session_state.task_data[st.session_state.idx]
            task_id = st.session_state.idx + 1
            
            # STEP 1: RAG Agent Decision (PRIMARY)
            rag_layer, rag_server, rag_reasoning = rag_agent.decide(network)
            
            # STEP 2: ML Model Verification (SECONDARY)
            ml_layer, ml_confidence = ml_verifier.verify(network)
            ml_matches = (rag_layer == ml_layer)
            
            # STEP 3: Agent Execution (check availability)
            final_layer, final_server, exec_note, decision_maker = execute_decision(
                rag_layer, rag_server,
                st.session_state.servers,
                network,
                current_time
            )
            
            # Display decision flow
            with decision_placeholder.container():
                st.markdown(f"### 🎯 Task {task_id}")
                
                # Network condition
                n1, n2, n3, n4 = st.columns(4)
                n1.metric("📶 Data Rate", f"{network['datarate_mbps']:.1f} Mbps")
                n2.metric("📡 SINR", f"{network['sinr']:.1f} dB")
                n3.metric("⏱️ Latency", f"{network['latency_ms']:.1f} ms")
                n4.metric("📊 RSRP", f"{network['rsrp_dbm']:.1f} dBm")
                
                st.markdown("---")
                
                # RAG vs ML
                d1, d2, d3 = st.columns(3)
                
                with d1:
                    st.markdown("#### 🧠 RAG Agent (PRIMARY)")
                    color = {"Edge": "success", "Fog": "warning", "Cloud": "info"}
                    getattr(st, color.get(rag_layer, "info"))(f"**{rag_layer}** → Server {rag_server}")
                    st.caption(rag_reasoning[:80] + "..." if len(rag_reasoning) > 80 else rag_reasoning)
                
                with d2:
                    st.markdown("#### 📊 ML Model (VERIFY)")
                    getattr(st, color.get(ml_layer, "info"))(f"**{ml_layer}** ({ml_confidence:.0%})")
                    if ml_matches:
                        st.success("✅ Matches RAG")
                    else:
                        st.warning(f"⚠️ Different from RAG")
                
                with d3:
                    st.markdown("#### ✅ Final Execution")
                    getattr(st, color.get(final_layer, "info"))(f"**{final_layer}** → Server {final_server}")
                    st.caption(exec_note)
                
                st.markdown("---")
                st.caption(f"📁 CSV Original: {network.get('csv_layer', 'N/A')} → Server {network.get('csv_server', 'N/A')}")
            
            # Server status
            with server_placeholder.container():
                for layer in ["Edge", "Fog", "Cloud"]:
                    layer_servers = [s for s in st.session_state.servers.values() if s.layer == layer]
                    icon = {"Edge": "🟢", "Fog": "🟡", "Cloud": "🔵"}[layer]
                    
                    st.markdown(f"**{icon} {layer}** (max {MAX_PARALLEL_BY_LAYER[layer]} parallel)")
                    
                    cols = st.columns(len(layer_servers))
                    for i, server in enumerate(layer_servers):
                        with cols[i]:
                            if server.is_available(current_time):
                                st.markdown(f"**S{server.server_id}**: 🟢 FREE")
                            else:
                                remaining = server.get_wait_time(current_time)
                                st.markdown(f"**S{server.server_id}**: 🔴 BUSY")
                                st.caption(f"Task {server.current_task} ({remaining:.1f}s)")
                            st.caption(f"Done: {server.tasks_completed}")
            
            # Create and track task
            server_obj = st.session_state.servers[final_server]
            
            if server_obj.is_available(current_time):
                status = TaskStatus.RUNNING
                start_time = current_time
                end_time = current_time + timedelta(seconds=actual_completion[final_layer])
                server_obj.assign_task(task_id, actual_completion[final_layer], current_time)
            else:
                status = TaskStatus.WAITING
                start_time = None
                end_time = None
            
            task = Task(
                task_id=task_id,
                arrival_time=current_time,
                network=network,
                rag_layer=rag_layer,
                rag_server=rag_server,
                rag_reasoning=rag_reasoning,
                ml_layer=ml_layer,
                ml_confidence=ml_confidence,
                ml_matches_rag=ml_matches,
                final_layer=final_layer,
                final_server=final_server,
                execution_note=exec_note,
                decision_maker=decision_maker,
                status=status,
                start_time=start_time,
                end_time=end_time
            )
            
            st.session_state.tasks.append(task)
            st.session_state.idx += 1
            
            # Queue display
            with queue_placeholder.container():
                waiting_tasks = [t for t in st.session_state.tasks if t.status == TaskStatus.WAITING]
                running_tasks = [t for t in st.session_state.tasks if t.status == TaskStatus.RUNNING]
                
                if waiting_tasks:
                    st.markdown("**⏳ WAITING (Server Busy):**")
                    for t in waiting_tasks:
                        wait = (current_time - t.arrival_time).total_seconds()
                        st.warning(f"Task {t.task_id}: Queued for {t.final_layer} Server {t.final_server} ({wait:.1f}s waiting)")
                
                if running_tasks:
                    st.markdown("**🔄 RUNNING:**")
                    for t in running_tasks:
                        elapsed = (current_time - t.start_time).total_seconds()
                        total = (t.end_time - t.start_time).total_seconds()
                        progress = min(elapsed / total, 1.0)
                        remaining = max(0, total - elapsed)
                        
                        c1, c2, c3 = st.columns([2, 5, 1])
                        c1.write(f"Task {t.task_id} ({t.final_layer} S{t.final_server})")
                        c2.progress(progress)
                        c3.write(f"{remaining:.1f}s")
            
            # Log table
            with log_placeholder.container():
                log_data = []
                for t in st.session_state.tasks[-10:]:
                    log_data.append({
                        "Task": t.task_id,
                        "🧠 RAG Decision": f"{t.rag_layer} (S{t.rag_server})",
                        "📊 ML Verify": f"{t.ml_layer} ({t.ml_confidence:.0%})",
                        "Match?": "✅" if t.ml_matches_rag else "❌",
                        "Final": f"{t.final_layer} (S{t.final_server})",
                        "Type": t.decision_maker.value,
                        "Status": t.status.value
                    })
                if log_data:
                    st.dataframe(pd.DataFrame(log_data), use_container_width=True, hide_index=True)
            
            time.sleep(actual_arrival)
        
        # Wait for remaining tasks
        while any(t.status != TaskStatus.COMPLETED for t in st.session_state.tasks):
            current_time = datetime.now()
            
            for task in st.session_state.tasks:
                if task.status == TaskStatus.RUNNING and task.end_time and current_time >= task.end_time:
                    task.status = TaskStatus.COMPLETED
                    st.session_state.servers[task.final_server].release()
                
                if task.status == TaskStatus.WAITING:
                    server = st.session_state.servers[task.final_server]
                    if server.is_available(current_time):
                        task.start_time = current_time
                        task.end_time = current_time + timedelta(seconds=actual_completion[task.final_layer])
                        task.wait_time = (current_time - task.arrival_time).total_seconds()
                        task.status = TaskStatus.RUNNING
                        server.assign_task(task.task_id, actual_completion[task.final_layer], current_time)
            
            time.sleep(0.3)
        
        st.session_state.running = False
        st.balloons()
        
        # Final summary
        st.header("📊 Final Summary")
        
        f1, f2, f3, f4 = st.columns(4)
        
        total = len(st.session_state.tasks)
        rag_ml_match = sum(1 for t in st.session_state.tasks if t.ml_matches_rag)
        rag_direct = sum(1 for t in st.session_state.tasks if t.decision_maker == DecisionMaker.RAG_PRIMARY)
        queued = sum(1 for t in st.session_state.tasks if t.decision_maker == DecisionMaker.RAG_QUEUED)
        
        with f1:
            st.metric("🎯 RAG = ML Match", f"{rag_ml_match}/{total} ({rag_ml_match/total*100:.0f}%)")
        
        with f2:
            st.metric("🧠 RAG Direct Execute", rag_direct)
        
        with f3:
            fallbacks = sum(1 for t in st.session_state.tasks if t.decision_maker == DecisionMaker.RAG_FALLBACK)
            st.metric("⚠️ Fallbacks", fallbacks)
        
        with f4:
            avg_wait = sum(t.wait_time for t in st.session_state.tasks if t.wait_time > 0)
            avg_wait = avg_wait / queued if queued else 0
            st.metric("⏳ Queued Tasks", f"{queued} (avg {avg_wait:.1f}s wait)")
        
        # Layer distribution
        st.subheader("Layer Distribution")
        dist = pd.DataFrame({
            "Layer": ["Edge", "Fog", "Cloud"],
            "RAG Decision": [
                sum(1 for t in st.session_state.tasks if t.rag_layer == "Edge"),
                sum(1 for t in st.session_state.tasks if t.rag_layer == "Fog"),
                sum(1 for t in st.session_state.tasks if t.rag_layer == "Cloud"),
            ],
            "ML Verify": [
                sum(1 for t in st.session_state.tasks if t.ml_layer == "Edge"),
                sum(1 for t in st.session_state.tasks if t.ml_layer == "Fog"),
                sum(1 for t in st.session_state.tasks if t.ml_layer == "Cloud"),
            ],
            "Final Execute": [
                sum(1 for t in st.session_state.tasks if t.final_layer == "Edge"),
                sum(1 for t in st.session_state.tasks if t.final_layer == "Fog"),
                sum(1 for t in st.session_state.tasks if t.final_layer == "Cloud"),
            ]
        })
        st.dataframe(dist, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
