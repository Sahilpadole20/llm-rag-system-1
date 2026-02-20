"""
RAG-Based Agent Real-Time Deployment Simulator
===============================================
Primary Decision: RAG AGENT (Groq LLM + Vector Search)
Verification: ML Model (just to compare)

Flow:
1. Network condition arrives (from CSV)
2. RAG Agent queries knowledge base + LLM for decision
3. ML Model runs in parallel for verification
4. Agent checks available servers and executes

Run: streamlit run rag_agent_simulator.py
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
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from enum import Enum

# Page config
st.set_page_config(
    page_title="RAG Agent Simulator",
    page_icon="🤖",
    layout="wide"
)

# Paths
BASE_DIR = Path(__file__).parent
STORE_DIR = BASE_DIR / "network_faiss_store"
DATA_DIR = BASE_DIR / "data"
CSV_PATH = DATA_DIR / "simulation_data.csv"

# Fallback CSV path
if not CSV_PATH.exists():
    CSV_PATH = Path(r"c:\Users\Sahil Padole\Videos\AI_agent_ml_threshold\data\edgesimpy_failure_ml_+_thresh_(gb)_no_failure_20251223_075347_results.csv")

# Get API key from secrets/environment (will be overridden by sidebar input if provided)
try:
    DEFAULT_GROQ_KEY = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY", ""))
except:
    DEFAULT_GROQ_KEY = os.environ.get("GROQ_API_KEY", "")

# Constants
TASK_ARRIVAL_INTERVAL = 4  # seconds
COMPLETION_TIMES = {"Edge": 3, "Fog": 6, "Cloud": 10}

# Server configuration
SERVERS = {
    1: {"layer": "Edge", "latency": 5},
    2: {"layer": "Edge", "latency": 5},
    3: {"layer": "Edge", "latency": 5},
    4: {"layer": "Edge", "latency": 5},
    5: {"layer": "Fog", "latency": 25},
    6: {"layer": "Fog", "latency": 25},
    7: {"layer": "Cloud", "latency": 100},
}


class TaskStatus(Enum):
    WAITING = "waiting"
    RUNNING = "running"
    COMPLETED = "completed"


class DecisionSource(Enum):
    RAG_AGENT = "RAG Agent"
    RAG_FALLBACK = "RAG + Fallback"
    RAG_QUEUED = "RAG + Queued"


@dataclass
class ServerState:
    """Track server state."""
    server_id: int
    layer: str
    latency: int
    current_task: Optional[int] = None
    busy_until: Optional[datetime] = None
    tasks_completed: int = 0
    
    def is_available(self, current_time: datetime) -> bool:
        if self.busy_until is None:
            return True
        return current_time >= self.busy_until
    
    def assign_task(self, task_id: int, duration: float, current_time: datetime):
        self.current_task = task_id
        self.busy_until = current_time + timedelta(seconds=duration)
    
    def release(self):
        self.current_task = None
        self.busy_until = None
        self.tasks_completed += 1


@dataclass
class Task:
    """Task with RAG prediction and ML verification."""
    task_id: int
    arrival_time: datetime
    network_condition: Dict
    
    # RAG Agent Decision (primary)
    rag_layer: str
    rag_server: int
    rag_reasoning: str
    
    # ML Verification (secondary)
    ml_layer: str
    ml_confidence: float
    
    # Final execution
    final_layer: str
    final_server: int
    decision_source: DecisionSource
    
    status: TaskStatus = TaskStatus.WAITING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    wait_time: float = 0.0


# ============================================================================
# LOAD MODELS
# ============================================================================
@st.cache_resource
def load_models():
    """Load RAG components and ML model."""
    models = {}
    
    # Load ML model
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
    
    # Load TF-IDF embedder
    embedder_path = STORE_DIR / "tfidf_embedder.pkl"
    if embedder_path.exists():
        with open(embedder_path, 'rb') as f:
            data = pickle.load(f)
            models['vectorizer'] = data['vectorizer']
            models['svd'] = data['svd']
    
    # Load FAISS index
    faiss_path = STORE_DIR / "faiss.index"
    meta_path = STORE_DIR / "metadata.pkl"
    
    if faiss_path.exists() and meta_path.exists():
        import faiss
        models['faiss_index'] = faiss.read_index(str(faiss_path))
        with open(meta_path, 'rb') as f:
            models['metadata'] = pickle.load(f)
    
    return models


# ============================================================================
# RAG AGENT
# ============================================================================
class RAGDeploymentAgent:
    """RAG-based agent for deployment decisions."""
    
    def __init__(self, models: dict, api_key: str):
        self.models = models
        self.api_key = api_key
        self.feature_cols = ['datarate', 'sinr', 'latency_ms', 'rsrp_dbm', 'cpu_demand', 'memory_demand']
    
    def search_similar_scenarios(self, network: Dict, top_k: int = 5) -> List[str]:
        """Search FAISS for similar network conditions."""
        if 'vectorizer' not in self.models or 'faiss_index' not in self.models:
            return []
        
        # Create query from network conditions
        query = f"""Network condition: 
        Datarate={network['datarate_mbps']:.1f}Mbps, 
        SINR={network['sinr']:.1f}dB, 
        Latency={network['latency_ms']:.1f}ms, 
        RSRP={network['rsrp_dbm']:.1f}dBm"""
        
        # Embed and search
        tfidf = self.models['vectorizer'].transform([query])
        query_emb = self.models['svd'].transform(tfidf).astype('float32')
        distances, indices = self.models['faiss_index'].search(query_emb, top_k)
        
        docs = self.models.get('metadata', {}).get('documents', [])
        results = []
        for idx in indices[0]:
            if 0 <= idx < len(docs):
                results.append(docs[idx])
        
        return results
    
    def get_ml_prediction(self, network: Dict) -> Tuple[str, float]:
        """ML model prediction for verification."""
        if 'ml_model' not in self.models:
            return "Unknown", 0.0
        
        features = [
            network.get('datarate', 0) * 1e6,  # Convert to bps
            network.get('sinr', 0),
            network.get('latency_ms', 0),
            network.get('rsrp_dbm', 0),
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
    
    def query_rag_agent(self, network: Dict) -> Tuple[str, int, str]:
        """
        PRIMARY DECISION: RAG Agent (Groq LLM + Vector Search)
        Returns: (layer, server_id, reasoning)
        """
        # Step 1: Get similar scenarios from vector store
        similar_scenarios = self.search_similar_scenarios(network)
        context = "\n".join(similar_scenarios[:3]) if similar_scenarios else "No similar scenarios found."
        
        # Step 2: Query Groq LLM
        if not self.api_key:
            # Fallback to rule-based if no API key
            return self._rule_based_decision(network)
        
        try:
            from groq import Groq
            client = Groq(api_key=self.api_key)
            
            prompt = f"""You are an Edge-Fog-Cloud deployment agent for autonomous vehicles.

CURRENT NETWORK CONDITIONS:
- Data Rate: {network['datarate_mbps']:.2f} Mbps
- SINR: {network['sinr']:.2f} dB
- Latency: {network['latency_ms']:.2f} ms
- RSRP: {network['rsrp_dbm']:.2f} dBm
- CPU Demand: {network.get('cpu_demand', 30)}%
- Memory Demand: {network.get('memory_demand', 100)} MB

INFRASTRUCTURE:
- Edge (Servers 1-4): latency <10ms, limited CPU, best for real-time
- Fog (Servers 5-6): latency ~25ms, moderate resources
- Cloud (Server 7): latency ~100ms, unlimited resources

DEPLOYMENT RULES FROM KNOWLEDGE BASE:
- If latency < 20ms AND datarate < 16.6 Mbps → Edge
- If datarate 9.6-16.6 Mbps AND SINR > 10 dB → Fog  
- If datarate >= 16.6 Mbps OR SINR <= 10 dB → Cloud
- If RSRP < -120 dBm (poor signal) → Cloud

SIMILAR PAST SCENARIOS:
{context}

Based on the network conditions and rules, decide:
1. Which layer to deploy to (Edge, Fog, or Cloud)
2. Which specific server (1-7)
3. Why this choice

RESPOND IN EXACTLY THIS FORMAT (one line):
LAYER: [Edge/Fog/Cloud] | SERVER: [1-7] | REASON: [your reasoning]"""

            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=150
            )
            
            result = response.choices[0].message.content.strip()
            
            # Parse response
            layer_match = re.search(r'LAYER:\s*(Edge|Fog|Cloud)', result, re.IGNORECASE)
            server_match = re.search(r'SERVER:\s*(\d+)', result)
            reason_match = re.search(r'REASON:\s*(.+)', result, re.IGNORECASE)
            
            if layer_match:
                layer = layer_match.group(1).capitalize()
                server = int(server_match.group(1)) if server_match else self._get_default_server(layer)
                reason = reason_match.group(1) if reason_match else result
                return layer, server, f"🧠 RAG: {reason}"
            else:
                # Parse failed, use rule-based
                return self._rule_based_decision(network)
                
        except Exception as e:
            st.warning(f"RAG query failed: {e}. Using rule-based fallback.")
            return self._rule_based_decision(network)
    
    def _rule_based_decision(self, network: Dict) -> Tuple[str, int, str]:
        """Fallback rule-based decision."""
        latency = network.get('latency_ms', 100)
        datarate = network.get('datarate_mbps', 10)
        sinr = network.get('sinr', 10)
        rsrp = network.get('rsrp_dbm', -100)
        
        if rsrp < -120:
            return "Cloud", 7, "📡 Poor signal (RSRP < -120) → Cloud"
        elif latency < 20 and datarate < 16.6:
            server = random.choice([1, 2, 3, 4])
            return "Edge", server, f"⚡ Low latency ({latency:.1f}ms) → Edge"
        elif 9.6 <= datarate < 16.6 and sinr > 10:
            server = random.choice([5, 6])
            return "Fog", server, f"📊 Mid datarate + good SINR → Fog"
        else:
            return "Cloud", 7, f"☁️ High datarate or low SINR → Cloud"
    
    def _get_default_server(self, layer: str) -> int:
        """Get default server for layer."""
        servers_by_layer = {
            "Edge": [1, 2, 3, 4],
            "Fog": [5, 6],
            "Cloud": [7]
        }
        return random.choice(servers_by_layer.get(layer, [7]))


def agent_execute(
    rag_layer: str,
    rag_server: int,
    servers: Dict[int, ServerState],
    network: Dict,
    current_time: datetime
) -> Tuple[str, int, str, DecisionSource]:
    """
    AGENT EXECUTION: Check available nodes and execute.
    Returns: (final_layer, final_server, reason, decision_source)
    """
    
    # Check if RAG-recommended server is available
    if rag_server in servers and servers[rag_server].is_available(current_time):
        return (
            rag_layer,
            rag_server,
            f"✅ RAG recommended {rag_layer} Server {rag_server} - Available",
            DecisionSource.RAG_AGENT
        )
    
    # RAG server busy - find alternative in same layer
    layer_servers = [s for s in servers.values() 
                     if s.layer == rag_layer and s.is_available(current_time)]
    
    if layer_servers:
        server = layer_servers[0]
        return (
            rag_layer,
            server.server_id,
            f"✅ RAG layer {rag_layer}, Server {rag_server} busy → Using Server {server.server_id}",
            DecisionSource.RAG_AGENT
        )
    
    # RAG layer is FULL - intelligent fallback
    latency_req = network.get('latency_ms', 100)
    
    if rag_layer == "Cloud":
        fallback_order = ["Fog", "Edge"]
    elif rag_layer == "Edge":
        fallback_order = ["Fog"] if latency_req < 50 else ["Fog", "Cloud"]
    else:
        fallback_order = ["Edge"] if latency_req < 30 else ["Edge", "Cloud"]
    
    for fallback_layer in fallback_order:
        fallback_servers = [s for s in servers.values()
                          if s.layer == fallback_layer and s.is_available(current_time)]
        if fallback_servers:
            server = fallback_servers[0]
            return (
                fallback_layer,
                server.server_id,
                f"⚠️ RAG: {rag_layer} (BUSY) → Fallback to {fallback_layer} Server {server.server_id}",
                DecisionSource.RAG_FALLBACK
            )
    
    # All busy - queue on RAG layer
    rag_servers = [s for s in servers.values() if s.layer == rag_layer]
    earliest = min(rag_servers, key=lambda s: s.busy_until or datetime.min)
    wait_time = (earliest.busy_until - current_time).total_seconds() if earliest.busy_until else 0
    
    return (
        rag_layer,
        earliest.server_id,
        f"⏳ All busy! Queued on {rag_layer} Server {earliest.server_id} (wait ~{wait_time:.1f}s)",
        DecisionSource.RAG_QUEUED
    )


def load_csv():
    """Load CSV data."""
    if CSV_PATH.exists():
        return pd.read_csv(CSV_PATH)
    return None


def main():
    st.title("🤖 RAG Agent Real-Time Deployment")
    
    st.markdown("""
    ### Decision Pipeline:
    | Step | Component | Role |
    |------|-----------|------|
    | 1️⃣ | **RAG Agent** | Primary decision (Groq LLM + Vector Search) |
    | 2️⃣ | **ML Model** | Verification only (compare with RAG) |
    | 3️⃣ | **Agent** | Check available servers and execute |
    
    **RAG Agent uses:**
    - 📚 FAISS Vector Store (similar network conditions)
    - 🧠 Groq LLM (llama-3.3-70b-versatile)
    - 📏 Knowledge base rules for deployment
    """)
    
    # Load models
    models = load_models()
    
    if not models:
        st.error("❌ Models not loaded! Run train_paper_gb_model.py first.")
        return
    
    # Sidebar - API Key Input
    st.sidebar.header("🔑 API Configuration")
    
    groq_api_key = st.sidebar.text_input(
        "Groq API Key",
        value=DEFAULT_GROQ_KEY,
        type="password",
        help="Enter your Groq API key. Get one free at https://console.groq.com"
    )
    
    # Use the key from input (or default)
    GROQ_API_KEY = groq_api_key if groq_api_key else DEFAULT_GROQ_KEY
    
    # Check for API key
    if not GROQ_API_KEY:
        st.warning("⚠️ Enter Groq API Key in sidebar for RAG Agent. Get free key at https://console.groq.com")
    else:
        st.success("✅ Groq API key configured")
    
    # Load data
    df = load_csv()
    if df is None:
        st.error("CSV not found!")
        return
    
    st.success(f"✅ Loaded {len(df)} network conditions | FAISS: {models.get('faiss_index').ntotal if 'faiss_index' in models else 0} vectors")
    
    # Initialize RAG agent
    rag_agent = RAGDeploymentAgent(models, GROQ_API_KEY)
    
    # Sidebar
    st.sidebar.header("🎛️ Controls")
    num_tasks = st.sidebar.slider("Tasks to Simulate", 3, 20, 8)
    speed = st.sidebar.slider("Speed Multiplier", 1, 10, 5)
    
    actual_completion = {k: v / speed for k, v in COMPLETION_TIMES.items()}
    actual_arrival = TASK_ARRIVAL_INTERVAL / speed
    
    st.sidebar.markdown(f"""
    **Timing ({speed}x):**
    - Arrival: {actual_arrival:.1f}s
    - Edge: {actual_completion['Edge']:.1f}s
    - Fog: {actual_completion['Fog']:.1f}s
    - Cloud: {actual_completion['Cloud']:.1f}s
    """)
    
    # Session state
    if 'running' not in st.session_state:
        st.session_state.running = False
    if 'tasks' not in st.session_state:
        st.session_state.tasks = []
    if 'servers' not in st.session_state:
        st.session_state.servers = {
            sid: ServerState(sid, cfg["layer"], cfg["latency"])
            for sid, cfg in SERVERS.items()
        }
    if 'task_data' not in st.session_state:
        st.session_state.task_data = []
    if 'idx' not in st.session_state:
        st.session_state.idx = 0
    
    # Buttons
    c1, c2, c3 = st.sidebar.columns(3)
    if c1.button("▶️ Start", type="primary"):
        st.session_state.running = True
        if not st.session_state.task_data:
            indices = random.sample(range(len(df)), num_tasks)
            st.session_state.task_data = [
                {
                    "datarate_mbps": df.iloc[i]['datarate'] / 1e6,
                    "sinr": df.iloc[i]['sinr'],
                    "latency_ms": df.iloc[i]['latency_ms'],
                    "rsrp_dbm": df.iloc[i]['rsrp_dbm'],
                    "cpu_demand": int(df.iloc[i].get('cpu_demand', 30)),
                    "memory_demand": int(df.iloc[i].get('memory_demand', 100)),
                    "original_layer": df.iloc[i]['assigned_layer'],
                    "original_server": int(df.iloc[i]['server_id'])
                }
                for i in indices
            ]
    if c2.button("⏹️ Stop"):
        st.session_state.running = False
    if c3.button("🔄 Reset"):
        st.session_state.tasks = []
        st.session_state.task_data = []
        st.session_state.idx = 0
        st.session_state.running = False
        st.session_state.servers = {
            sid: ServerState(sid, cfg["layer"], cfg["latency"])
            for sid, cfg in SERVERS.items()
        }
        st.rerun()
    
    st.divider()
    
    # Metrics
    mc1, mc2, mc3, mc4, mc5 = st.columns(5)
    completed = len([t for t in st.session_state.tasks if t.status == TaskStatus.COMPLETED])
    running = len([t for t in st.session_state.tasks if t.status == TaskStatus.RUNNING])
    waiting = len([t for t in st.session_state.tasks if t.status == TaskStatus.WAITING])
    rag_direct = len([t for t in st.session_state.tasks if t.decision_source == DecisionSource.RAG_AGENT])
    rag_ml_match = len([t for t in st.session_state.tasks if t.rag_layer == t.ml_layer])
    
    mc1.metric("✅ Done", completed)
    mc2.metric("🔄 Running", running)
    mc3.metric("⏳ Waiting", waiting)
    mc4.metric("🧠 RAG Direct", rag_direct)
    mc5.metric("🎯 RAG=ML", rag_ml_match)
    
    if st.session_state.task_data:
        st.progress(st.session_state.idx / len(st.session_state.task_data))
    
    st.divider()
    
    # Main visualization
    col_left, col_right = st.columns([3, 2])
    
    with col_left:
        st.subheader("🤖 RAG Agent Decision Flow")
        flow_placeholder = st.empty()
    
    with col_right:
        st.subheader("🖥️ Server Status")
        server_placeholder = st.empty()
    
    st.divider()
    
    st.subheader("📋 Task Processing")
    queue_placeholder = st.empty()
    
    st.subheader("📜 Decision Comparison: RAG vs ML")
    log_placeholder = st.empty()
    
    # Simulation loop
    if st.session_state.running and st.session_state.task_data:
        
        while st.session_state.idx < len(st.session_state.task_data):
            current_time = datetime.now()
            
            # Update completed tasks
            for task in st.session_state.tasks:
                if task.status == TaskStatus.RUNNING and task.end_time:
                    if current_time >= task.end_time:
                        task.status = TaskStatus.COMPLETED
                        st.session_state.servers[task.final_server].release()
            
            # Start waiting tasks
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
            
            # New task arrives
            network = st.session_state.task_data[st.session_state.idx]
            task_id = st.session_state.idx + 1
            
            # STEP 1: RAG Agent Decision (PRIMARY)
            rag_layer, rag_server, rag_reasoning = rag_agent.query_rag_agent(network)
            
            # STEP 2: ML Model Verification (SECONDARY)
            ml_layer, ml_confidence = rag_agent.get_ml_prediction(network)
            
            # STEP 3: Agent Execution (check availability)
            final_layer, final_server, exec_reason, decision_source = agent_execute(
                rag_layer, rag_server,
                st.session_state.servers,
                network,
                current_time
            )
            
            # Display decision flow
            with flow_placeholder.container():
                st.markdown(f"## 🎯 Task {task_id}")
                
                # Network metrics
                st.markdown("### 📡 Network Condition")
                nc1, nc2, nc3, nc4 = st.columns(4)
                nc1.metric("Data Rate", f"{network['datarate_mbps']:.1f} Mbps")
                nc2.metric("SINR", f"{network['sinr']:.1f} dB")
                nc3.metric("Latency", f"{network['latency_ms']:.1f} ms")
                nc4.metric("RSRP", f"{network['rsrp_dbm']:.1f} dBm")
                
                st.markdown("---")
                
                # RAG vs ML comparison
                dc1, dc2, dc3 = st.columns(3)
                
                with dc1:
                    st.markdown("### 🧠 RAG Agent (Primary)")
                    color_map = {"Edge": "success", "Fog": "warning", "Cloud": "info"}
                    getattr(st, color_map.get(rag_layer, "info"))(f"**{rag_layer}** → Server {rag_server}")
                    st.caption(rag_reasoning[:100] + "..." if len(rag_reasoning) > 100 else rag_reasoning)
                
                with dc2:
                    st.markdown("### 📊 ML Model (Verify)")
                    getattr(st, color_map.get(ml_layer, "info"))(f"**{ml_layer}** ({ml_confidence:.0%})")
                    if rag_layer == ml_layer:
                        st.caption("✅ Matches RAG decision")
                    else:
                        st.caption(f"⚠️ Differs from RAG ({rag_layer})")
                
                with dc3:
                    st.markdown("### ✅ Final Execution")
                    getattr(st, color_map.get(final_layer, "info"))(f"**{final_layer}** → Server {final_server}")
                    st.caption(exec_reason)
                
                st.markdown("---")
                
                # Show original assignment from CSV
                st.caption(f"📂 CSV Original: {network.get('original_layer', 'N/A')} → Server {network.get('original_server', 'N/A')}")
            
            # Update server display
            with server_placeholder.container():
                for layer in ["Edge", "Fog", "Cloud"]:
                    layer_servers = [s for s in st.session_state.servers.values() if s.layer == layer]
                    
                    icon = {"Edge": "🟢", "Fog": "🟡", "Cloud": "🔵"}[layer]
                    st.markdown(f"**{icon} {layer}**")
                    
                    cols = st.columns(len(layer_servers))
                    for i, server in enumerate(layer_servers):
                        with cols[i]:
                            if server.is_available(current_time):
                                st.markdown(f"S{server.server_id}: 🟢 FREE")
                            else:
                                st.markdown(f"S{server.server_id}: 🔴 BUSY")
                                st.caption(f"Task {server.current_task}")
            
            # Create task
            server = st.session_state.servers[final_server]
            
            if server.is_available(current_time):
                status = TaskStatus.RUNNING
                start_time = current_time
                end_time = current_time + timedelta(seconds=actual_completion[final_layer])
                server.assign_task(task_id, actual_completion[final_layer], current_time)
            else:
                status = TaskStatus.WAITING
                start_time = None
                end_time = None
            
            task = Task(
                task_id=task_id,
                arrival_time=current_time,
                network_condition=network,
                rag_layer=rag_layer,
                rag_server=rag_server,
                rag_reasoning=rag_reasoning,
                ml_layer=ml_layer,
                ml_confidence=ml_confidence,
                final_layer=final_layer,
                final_server=final_server,
                decision_source=decision_source,
                status=status,
                start_time=start_time,
                end_time=end_time
            )
            
            st.session_state.tasks.append(task)
            st.session_state.idx += 1
            
            # Update queue display
            with queue_placeholder.container():
                waiting_tasks = [t for t in st.session_state.tasks if t.status == TaskStatus.WAITING]
                running_tasks = [t for t in st.session_state.tasks if t.status == TaskStatus.RUNNING]
                
                if waiting_tasks:
                    st.markdown("**⏳ Waiting:**")
                    for t in waiting_tasks:
                        wait = (current_time - t.arrival_time).total_seconds()
                        st.warning(f"Task {t.task_id}: Waiting for {t.final_layer} Server {t.final_server} ({wait:.1f}s)")
                
                if running_tasks:
                    st.markdown("**🔄 Running:**")
                    for t in running_tasks:
                        elapsed = (current_time - t.start_time).total_seconds()
                        total = (t.end_time - t.start_time).total_seconds()
                        progress = min(elapsed / total, 1.0)
                        remaining = max(0, total - elapsed)
                        
                        rc1, rc2, rc3 = st.columns([2, 4, 2])
                        rc1.write(f"Task {t.task_id} ({t.final_layer})")
                        rc2.progress(progress)
                        rc3.write(f"{remaining:.1f}s")
            
            # Update log
            with log_placeholder.container():
                log_data = []
                for t in st.session_state.tasks[-8:]:
                    log_data.append({
                        "Task": t.task_id,
                        "RAG Decision": f"{t.rag_layer} (S{t.rag_server})",
                        "ML Verify": f"{t.ml_layer} ({t.ml_confidence:.0%})",
                        "Match?": "✅" if t.rag_layer == t.ml_layer else "❌",
                        "Final": f"{t.final_layer} (S{t.final_server})",
                        "Source": t.decision_source.value,
                        "Status": t.status.value
                    })
                if log_data:
                    st.dataframe(pd.DataFrame(log_data), use_container_width=True, hide_index=True)
            
            # Wait for next task
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
        
        # Final analysis
        st.header("📊 Final Analysis: RAG vs ML")
        
        fc1, fc2, fc3, fc4 = st.columns(4)
        
        rag_ml_match = sum(1 for t in st.session_state.tasks if t.rag_layer == t.ml_layer)
        total = len(st.session_state.tasks)
        
        with fc1:
            st.metric("RAG = ML Match", f"{rag_ml_match}/{total}")
            st.metric("Match Rate", f"{rag_ml_match/total*100:.1f}%")
        
        with fc2:
            rag_direct = sum(1 for t in st.session_state.tasks 
                           if t.decision_source == DecisionSource.RAG_AGENT)
            st.metric("RAG Direct Execute", rag_direct)
        
        with fc3:
            fallbacks = sum(1 for t in st.session_state.tasks 
                          if t.decision_source == DecisionSource.RAG_FALLBACK)
            st.metric("Fallbacks Used", fallbacks)
        
        with fc4:
            queued = sum(1 for t in st.session_state.tasks 
                        if t.decision_source == DecisionSource.RAG_QUEUED)
            avg_wait = sum(t.wait_time for t in st.session_state.tasks if t.wait_time > 0)
            avg_wait = avg_wait / queued if queued else 0
            st.metric("Tasks Queued", queued)
            st.metric("Avg Wait", f"{avg_wait:.1f}s")
        
        # Distribution
        st.subheader("Layer Distribution Comparison")
        dist_data = {
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
        }
        st.dataframe(pd.DataFrame(dist_data), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
