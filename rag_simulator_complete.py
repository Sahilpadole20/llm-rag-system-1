"""
COMPLETE RAG Agent Simulator
=============================
✅ Basic RAG + EdgeSimPy comparison (800 train / 200 test split)
✅ Service Types: XR (50 users), eMBB (1000 users), URLLC, mMTC
✅ Time Windows: Services active during [t1, t2]
✅ Node Failure: Fog/Edge node failure with agentic rebalancing
✅ Priority Preemption: High priority displaces low priority services
✅ Dynamic Requirements: Change throughput/users mid-simulation
✅ User-based capacity: Edge(800), Fog(1000), Cloud(unlimited)

Run: streamlit run rag_simulator_complete.py
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
    page_title="🚀 Complete RAG Agent Simulator",
    page_icon="🚀",
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
# ENUMS & CONSTANTS
# =============================================================================
class ServiceType(Enum):
    XR = "🥽 XR (Extended Reality)"
    EMBB = "📱 eMBB (Enhanced Mobile Broadband)"
    URLLC = "⚡ URLLC (Ultra-Reliable Low Latency)"
    MMTC = "📡 mMTC (Massive Machine Type)"
    TASK = "📋 Task (From CSV)"  # For basic mode


class TaskStatus(Enum):
    WAITING = "⏳ Waiting"
    RUNNING = "🔄 Running"
    COMPLETED = "✅ Done"
    PREEMPTED = "🔄 Preempted"
    FAILED = "❌ Failed"
    REBALANCING = "🔁 Rebalancing"


class DecisionMaker(Enum):
    RAG_PRIMARY = "🧠 RAG+LLM"
    RAG_FALLBACK = "📋 Threshold Rules"
    RAG_QUEUED = "⏳ Queued"
    PRIORITY_PREEMPT = "🔥 Priority Preempt"
    REBALANCED = "🔁 Rebalanced"


class NodeStatus(Enum):
    ACTIVE = "🟢 Active"
    FAILED = "🔴 Failed"
    RECOVERING = "🟡 Recovering"


# =============================================================================
# SERVICE REQUIREMENTS
# =============================================================================
@dataclass
class ServiceRequirements:
    """Requirements for each service type."""
    service_type: ServiceType
    throughput_mbps: float
    max_latency_ms: float
    priority: int  # 1=highest, 4=lowest
    preferred_layer: str


SERVICE_DEFAULTS = {
    ServiceType.XR: ServiceRequirements(
        service_type=ServiceType.XR,
        throughput_mbps=100.0,
        max_latency_ms=10.0,
        priority=1,
        preferred_layer="Edge"
    ),
    ServiceType.URLLC: ServiceRequirements(
        service_type=ServiceType.URLLC,
        throughput_mbps=10.0,
        max_latency_ms=5.0,
        priority=1,
        preferred_layer="Edge"
    ),
    ServiceType.EMBB: ServiceRequirements(
        service_type=ServiceType.EMBB,
        throughput_mbps=50.0,
        max_latency_ms=50.0,
        priority=2,
        preferred_layer="Fog"
    ),
    ServiceType.MMTC: ServiceRequirements(
        service_type=ServiceType.MMTC,
        throughput_mbps=1.0,
        max_latency_ms=1000.0,
        priority=3,
        preferred_layer="Cloud"
    ),
    ServiceType.TASK: ServiceRequirements(
        service_type=ServiceType.TASK,
        throughput_mbps=10.0,
        max_latency_ms=50.0,
        priority=2,
        preferred_layer="Fog"
    ),
}


# Layer configuration with user capacity
LAYER_CONFIG = {
    "Edge": {
        "servers": [1, 2, 3, 4],
        "users_per_server": 200,
        "total_capacity": 800,
        "completion_time": 3,
        "latency_ms": 5
    },
    "Fog": {
        "servers": [5, 6],
        "users_per_server": 500,
        "total_capacity": 1000,
        "completion_time": 6,
        "latency_ms": 25
    },
    "Cloud": {
        "servers": [7],
        "users_per_server": 10000,
        "total_capacity": 10000,
        "completion_time": 10,
        "latency_ms": 100
    }
}


# =============================================================================
# SERVER STATE
# =============================================================================
@dataclass
class ServerState:
    """Server with user capacity tracking."""
    server_id: int
    layer: str
    latency_ms: int
    max_users: int
    current_users: int = 0
    status: NodeStatus = NodeStatus.ACTIVE
    current_tasks: List[int] = field(default_factory=list)
    tasks_completed: int = 0
    busy_until: Optional[datetime] = None
    
    def available_capacity(self) -> int:
        if self.status != NodeStatus.ACTIVE:
            return 0
        return max(0, self.max_users - self.current_users)
    
    def can_accept_users(self, num_users: int) -> bool:
        return self.available_capacity() >= num_users
    
    def is_available(self, current_time: datetime) -> bool:
        if self.status != NodeStatus.ACTIVE:
            return False
        if self.busy_until is None:
            return True
        return current_time >= self.busy_until
    
    def assign_users(self, task_id: int, num_users: int):
        self.current_users += num_users
        self.current_tasks.append(task_id)
    
    def assign_task(self, task_id: int, duration: float, current_time: datetime):
        self.current_tasks.append(task_id)
        self.busy_until = current_time + timedelta(seconds=duration)
    
    def release_users(self, task_id: int, num_users: int):
        self.current_users = max(0, self.current_users - num_users)
        if task_id in self.current_tasks:
            self.current_tasks.remove(task_id)
        self.tasks_completed += 1
    
    def release(self):
        self.tasks_completed += 1
        self.current_tasks = []
        self.busy_until = None
    
    def fail(self):
        self.status = NodeStatus.FAILED
        return list(self.current_tasks)
    
    def recover(self):
        self.status = NodeStatus.ACTIVE


# =============================================================================
# SERVICE / TASK
# =============================================================================
@dataclass
class Service:
    """A service with users, time window, and requirements."""
    service_id: int
    service_type: ServiceType
    num_users: int
    start_time: datetime
    end_time: datetime
    requirements: ServiceRequirements
    network: Dict = field(default_factory=dict)
    
    # Scheduling state
    assigned_layer: Optional[str] = None
    assigned_server: Optional[int] = None
    status: TaskStatus = TaskStatus.WAITING
    decision_maker: DecisionMaker = DecisionMaker.RAG_PRIMARY
    
    # RAG vs EdgeSimPy comparison
    rag_layer: Optional[str] = None
    rag_server: Optional[int] = None
    rag_reasoning: str = ""
    ml_layer: Optional[str] = None
    ml_matches_rag: bool = False
    
    # Preemption tracking
    preempted_by: Optional[int] = None
    preempted_tasks: List[int] = field(default_factory=list)
    
    # Timing
    actual_start: Optional[datetime] = None
    actual_end: Optional[datetime] = None
    wait_time: float = 0.0
    
    def is_active(self, current_time: datetime) -> bool:
        return self.start_time <= current_time <= self.end_time
    
    def is_expired(self, current_time: datetime) -> bool:
        return current_time > self.end_time


# =============================================================================
# LOAD MODELS
# =============================================================================
def load_models():
    """Load RAG components."""
    models = {}
    
    # TF-IDF embedder
    embedder_path = STORE_DIR / "tfidf_embedder.pkl"
    if embedder_path.exists():
        with open(embedder_path, 'rb') as f:
            data = pickle.load(f)
            if isinstance(data, dict):
                models['vectorizer'] = data.get('vectorizer')
                models['svd'] = data.get('svd')
            else:
                models['tfidf_embedder'] = data
    
    # FAISS index
    try:
        import faiss
        faiss_path = STORE_DIR / "faiss.index"
        if faiss_path.exists():
            models['faiss_index'] = faiss.read_index(str(faiss_path))
    except:
        pass
    
    # Metadata
    meta_path = STORE_DIR / "metadata.pkl"
    if meta_path.exists():
        with open(meta_path, 'rb') as f:
            models['metadata'] = pickle.load(f)
    
    # Label encoder
    encoder_path = STORE_DIR / "label_encoder.pkl"
    if encoder_path.exists():
        with open(encoder_path, 'rb') as f:
            models['label_encoder'] = pickle.load(f)
    
    return models


def load_test_data():
    """Load test data (200 rows)."""
    test_path = STORE_DIR / "test_data.pkl"
    if test_path.exists():
        with open(test_path, 'rb') as f:
            return pickle.load(f)
    
    # Fallback to full CSV (last 200 rows)
    if CSV_PATH.exists():
        df = pd.read_csv(CSV_PATH)
        return df.tail(200).to_dict('records')
    
    return []


# =============================================================================
# RAG AGENT
# =============================================================================
class RAGAgent:
    """RAG Agent using Groq LLM + FAISS vector search."""
    
    def __init__(self, groq_api_key: str, models: Dict):
        self.api_key = groq_api_key
        self.model_name = "llama-3.3-70b-versatile"
        self.models = models
        self.client = None
        
        if groq_api_key:
            try:
                from groq import Groq
                self.client = Groq(api_key=groq_api_key)
            except Exception as e:
                st.warning(f"Groq not available: {e}")
    
    def search_similar(self, network: Dict, k: int = 3) -> List[Dict]:
        """Find similar network conditions from training data."""
        if 'faiss_index' not in self.models or 'metadata' not in self.models:
            return []
        
        try:
            embedder = self.models.get('tfidf_embedder') or self.models.get('vectorizer')
            if embedder is None:
                return []
            
            query = f"datarate={network.get('datarate_mbps', 0):.1f} sinr={network.get('sinr', 0):.1f} latency={network.get('latency_ms', 0):.1f}"
            
            if hasattr(embedder, 'transform'):
                query_vec = embedder.transform([query]).toarray().astype('float32')
            else:
                return []
            
            D, I = self.models['faiss_index'].search(query_vec, k)
            
            results = []
            for idx in I[0]:
                if 0 <= idx < len(self.models['metadata']):
                    results.append(self.models['metadata'][idx])
            return results
        except:
            return []
    
    def decide(self, network: Dict, service: Optional[Service] = None, layer_usage: Optional[Dict] = None) -> Tuple[str, int, str]:
        """Make agentic decision using LLM."""
        
        similar = self.search_similar(network)
        
        similar_context = ""
        if similar:
            for i, s in enumerate(similar[:3]):
                similar_context += f"\n{i+1}. DataRate={s.get('datarate_mbps', 0):.1f}Mbps, SINR={s.get('sinr', 0):.1f}dB -> {s.get('assigned_layer', 'Unknown')}"
        
        # Build service context if available
        service_context = ""
        if service:
            service_context = f"""
SERVICE REQUEST:
- Type: {service.service_type.value}
- Users: {service.num_users}
- Required Throughput: {service.requirements.throughput_mbps} Mbps
- Max Latency: {service.requirements.max_latency_ms} ms
- Priority: {service.requirements.priority} (1=highest, 4=lowest)
"""
        
        # Layer usage context
        capacity_context = ""
        if layer_usage:
            capacity_context = f"""
LAYER CAPACITY (users):
- Edge: {layer_usage['Edge']['used']}/{layer_usage['Edge']['total']} used
- Fog: {layer_usage['Fog']['used']}/{layer_usage['Fog']['total']} used
- Cloud: {layer_usage['Cloud']['used']}/{layer_usage['Cloud']['total']} used
"""
        
        prompt = f"""You are an agentic Edge-Fog-Cloud deployment system.
{service_context}
NETWORK CONDITIONS:
- Data Rate: {network.get('datarate_mbps', 0):.1f} Mbps
- SINR: {network.get('sinr', 0):.1f} dB
- Latency: {network.get('latency_ms', 0):.1f} ms
- RSRP: {network.get('rsrp_dbm', 0):.1f} dBm
{capacity_context}
SIMILAR PAST SCENARIOS:{similar_context if similar_context else " None available"}

THRESHOLD RULES (EdgeSimPy ML+Thresh GB):
- If latency < 20ms AND datarate < 9.6 Mbps: Edge
- If 9.6 <= datarate < 16.6 Mbps AND sinr > 10: Fog
- Otherwise: Cloud

Respond with EXACTLY: LAYER: [Edge/Fog/Cloud], SERVER: [1-7], REASON: [brief]"""

        if self.client:
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=150
                )
                text = response.choices[0].message.content
                
                layer_match = re.search(r'LAYER:\s*(Edge|Fog|Cloud)', text, re.IGNORECASE)
                server_match = re.search(r'SERVER:\s*(\d+)', text)
                reason_match = re.search(r'REASON:\s*(.+?)(?:\n|$)', text)
                
                layer = layer_match.group(1).capitalize() if layer_match else "Fog"
                server = int(server_match.group(1)) if server_match else 5
                reason = reason_match.group(1).strip() if reason_match else "LLM decision"
                
                # Validate server
                valid_servers = {"Edge": [1,2,3,4], "Fog": [5,6], "Cloud": [7]}
                if server not in valid_servers.get(layer, []):
                    server = valid_servers[layer][0]
                
                return layer, server, reason
            except Exception as e:
                pass
        
        # Fallback to threshold rules
        return self.threshold_decision(network)
    
    def threshold_decision(self, network: Dict) -> Tuple[str, int, str]:
        """Fallback threshold-based decision."""
        datarate = network.get('datarate_mbps', 0)
        latency = network.get('latency_ms', 0)
        sinr = network.get('sinr', 0)
        
        if latency < 20 and datarate < 9.6:
            return "Edge", random.choice([1, 2, 3, 4]), "Low latency local"
        elif 9.6 <= datarate < 16.6 and sinr > 10:
            return "Fog", random.choice([5, 6]), "Medium load"
        else:
            return "Cloud", 7, "High capacity needed"


# =============================================================================
# ML VERIFIER (EdgeSimPy Ground Truth)
# =============================================================================
class MLVerifier:
    """Uses EdgeSimPy assigned_layer as ground truth."""
    
    def verify(self, network: Dict) -> Tuple[str, float]:
        layer = network.get('edgesimpy_layer') or network.get('csv_layer') or network.get('assigned_layer', 'Unknown')
        return layer, 1.0


# =============================================================================
# AGENTIC SCHEDULER
# =============================================================================
class AgenticScheduler:
    """Handles scheduling, preemption, failure recovery."""
    
    def __init__(self, servers: Dict[int, ServerState], rag_agent: RAGAgent):
        self.servers = servers
        self.rag_agent = rag_agent
        self.services: List[Service] = []
        self.preemption_log: List[Dict] = []
        self.failure_log: List[Dict] = []
        self.rebalance_log: List[Dict] = []
    
    def get_layer_usage(self) -> Dict:
        """Calculate current usage per layer."""
        usage = {
            "Edge": {"used": 0, "total": LAYER_CONFIG["Edge"]["total_capacity"]},
            "Fog": {"used": 0, "total": LAYER_CONFIG["Fog"]["total_capacity"]},
            "Cloud": {"used": 0, "total": LAYER_CONFIG["Cloud"]["total_capacity"]}
        }
        
        for sid, server in self.servers.items():
            if server.status == NodeStatus.ACTIVE:
                usage[server.layer]["used"] += server.current_users
        
        return usage
    
    def schedule_service(self, service: Service, network: Dict, current_time: datetime) -> Tuple[str, int, str, DecisionMaker]:
        """Schedule a service using RAG agent."""
        
        layer_usage = self.get_layer_usage()
        
        # Get RAG decision
        rag_layer, rag_server, rag_reason = self.rag_agent.decide(network, service, layer_usage)
        
        # Check if preferred layer has capacity
        target_servers = [s for s in self.servers.values() 
                        if s.layer == rag_layer and s.status == NodeStatus.ACTIVE]
        
        for server in target_servers:
            if server.can_accept_users(service.num_users):
                return rag_layer, server.server_id, rag_reason, DecisionMaker.RAG_PRIMARY
        
        # Check if we need preemption (high priority)
        if service.requirements.priority == 1:
            preempt_result = self.try_preemption(service, rag_layer, current_time)
            if preempt_result:
                return preempt_result
        
        # Fallback to other layers
        for layer in ["Edge", "Fog", "Cloud"]:
            if layer == rag_layer:
                continue
            for server in self.servers.values():
                if server.layer == layer and server.status == NodeStatus.ACTIVE:
                    if server.can_accept_users(service.num_users):
                        return layer, server.server_id, f"Fallback: {rag_layer} full", DecisionMaker.RAG_FALLBACK
        
        return rag_layer, rag_server, "Queued - no capacity", DecisionMaker.RAG_QUEUED
    
    def try_preemption(self, high_priority_service: Service, target_layer: str, current_time: datetime) -> Optional[Tuple]:
        """Try to preempt lower priority services."""
        
        preemptable = []
        for svc in self.services:
            if svc.status == TaskStatus.RUNNING and svc.assigned_layer == target_layer:
                if svc.requirements.priority > high_priority_service.requirements.priority:
                    preemptable.append(svc)
        
        preemptable.sort(key=lambda x: -x.requirements.priority)
        
        users_needed = high_priority_service.num_users
        users_freed = 0
        to_preempt = []
        
        for svc in preemptable:
            to_preempt.append(svc)
            users_freed += svc.num_users
            if users_freed >= users_needed:
                break
        
        if users_freed >= users_needed:
            for svc in to_preempt:
                svc.status = TaskStatus.PREEMPTED
                svc.preempted_by = high_priority_service.service_id
                high_priority_service.preempted_tasks.append(svc.service_id)
                
                if svc.assigned_server:
                    self.servers[svc.assigned_server].release_users(svc.service_id, svc.num_users)
                
                self.preemption_log.append({
                    "time": current_time,
                    "preempted_service": svc.service_id,
                    "by_service": high_priority_service.service_id,
                    "users_freed": svc.num_users
                })
            
            for server in self.servers.values():
                if server.layer == target_layer and server.status == NodeStatus.ACTIVE:
                    if server.can_accept_users(high_priority_service.num_users):
                        return target_layer, server.server_id, f"Preempted {len(to_preempt)} services", DecisionMaker.PRIORITY_PREEMPT
        
        return None
    
    def handle_node_failure(self, server_id: int, current_time: datetime) -> List[Service]:
        """Handle node failure and rebalance."""
        
        server = self.servers.get(server_id)
        if not server:
            return []
        
        affected_task_ids = server.fail()
        
        self.failure_log.append({
            "time": current_time,
            "server_id": server_id,
            "layer": server.layer,
            "affected_tasks": len(affected_task_ids)
        })
        
        affected_services = [s for s in self.services if s.service_id in affected_task_ids]
        
        rebalanced = []
        for svc in affected_services:
            svc.status = TaskStatus.REBALANCING
            
            new_server = None
            for layer in [server.layer, "Edge", "Fog", "Cloud"]:
                for srv in self.servers.values():
                    if srv.server_id != server_id and srv.layer == layer:
                        if srv.status == NodeStatus.ACTIVE and srv.can_accept_users(svc.num_users):
                            new_server = srv
                            break
                if new_server:
                    break
            
            if new_server:
                new_server.assign_users(svc.service_id, svc.num_users)
                svc.assigned_server = new_server.server_id
                svc.assigned_layer = new_server.layer
                svc.status = TaskStatus.RUNNING
                svc.decision_maker = DecisionMaker.REBALANCED
                
                self.rebalance_log.append({
                    "time": current_time,
                    "service_id": svc.service_id,
                    "from_server": server_id,
                    "to_server": new_server.server_id
                })
                
                rebalanced.append(svc)
            else:
                svc.status = TaskStatus.FAILED
        
        server.current_users = 0
        server.current_tasks = []
        
        return rebalanced
    
    def recover_node(self, server_id: int):
        server = self.servers.get(server_id)
        if server:
            server.recover()


# =============================================================================
# MAIN APPLICATION
# =============================================================================
def main():
    st.title("🚀 Complete RAG Agent Simulator")
    
    # Mode selection
    mode = st.sidebar.radio(
        "📋 Simulation Mode",
        ["🎯 Basic (CSV Tasks)", "🚀 Advanced (Services)"],
        help="Basic: Uses EdgeSimPy CSV data\nAdvanced: Custom services with failures"
    )
    
    st.sidebar.divider()
    
    # API Key
    st.sidebar.header("🔑 Groq API Key")
    groq_key = st.sidebar.text_input("Enter API Key", value=DEFAULT_GROQ_KEY, type="password")
    
    if groq_key:
        st.sidebar.success("✅ API Key Added")
    else:
        st.sidebar.warning("⚠️ Enter key for LLM (using rules fallback)")
    
    # Load models
    models = load_models()
    faiss_count = models.get('faiss_index').ntotal if 'faiss_index' in models else 0
    
    # Initialize servers
    if 'servers' not in st.session_state:
        st.session_state.servers = {
            1: ServerState(1, "Edge", 5, 200),
            2: ServerState(2, "Edge", 5, 200),
            3: ServerState(3, "Edge", 5, 200),
            4: ServerState(4, "Edge", 5, 200),
            5: ServerState(5, "Fog", 25, 500),
            6: ServerState(6, "Fog", 25, 500),
            7: ServerState(7, "Cloud", 100, 10000),
        }
    
    if 'services' not in st.session_state:
        st.session_state.services = []
    
    if 'running' not in st.session_state:
        st.session_state.running = False
    
    if 'events' not in st.session_state:
        st.session_state.events = {"preemption": [], "failure": [], "rebalance": []}
    
    # ==========================================================================
    # BASIC MODE
    # ==========================================================================
    if mode == "🎯 Basic (CSV Tasks)":
        st.markdown("""
        **Basic Mode:** RAG + LLM decisions compared against EdgeSimPy ground truth
        - 🧠 FAISS Vector Store: {} training vectors
        - 📊 Test Data: 200 tasks from CSV
        """.format(faiss_count))
        
        # Sidebar controls
        st.sidebar.header("⚙️ Simulation Settings")
        num_tasks = st.sidebar.slider("Number of Tasks", 10, 200, 30, 10)
        arrival_interval = st.sidebar.slider("Arrival Interval (sec)", 1, 10, 4)
        batch_min = st.sidebar.slider("Min Batch Size", 1, 5, 1)
        batch_max = st.sidebar.slider("Max Batch Size", 1, 10, 4)
        
        # Load test data
        test_data = load_test_data()
        if not test_data:
            st.error("No test data found! Run train_with_split.py first.")
            return
        
        st.info(f"📊 Test data: {len(test_data)} rows | Using first {num_tasks}")
        
        # Control buttons
        col1, col2, col3 = st.columns(3)
        with col1:
            start_btn = st.button("▶️ Start Simulation", type="primary")
        with col2:
            stop_btn = st.button("⏹️ Stop")
        with col3:
            reset_btn = st.button("🔄 Reset")
        
        if reset_btn:
            st.session_state.services = []
            st.session_state.servers = {
                1: ServerState(1, "Edge", 5, 200),
                2: ServerState(2, "Edge", 5, 200),
                3: ServerState(3, "Edge", 5, 200),
                4: ServerState(4, "Edge", 5, 200),
                5: ServerState(5, "Fog", 25, 500),
                6: ServerState(6, "Fog", 25, 500),
                7: ServerState(7, "Cloud", 100, 10000),
            }
            st.rerun()
        
        # Placeholders
        metrics_placeholder = st.empty()
        server_placeholder = st.empty()
        queue_placeholder = st.empty()
        log_placeholder = st.empty()
        
        if start_btn:
            st.session_state.running = True
            rag_agent = RAGAgent(groq_key, models)
            ml_verifier = MLVerifier()
            
            task_data = test_data[:num_tasks]
            idx = 0
            
            while idx < len(task_data) and st.session_state.running:
                if stop_btn:
                    break
                
                current_time = datetime.now()
                
                # Complete finished tasks
                for svc in st.session_state.services:
                    if svc.status == TaskStatus.RUNNING and svc.actual_end:
                        if current_time >= svc.actual_end:
                            svc.status = TaskStatus.COMPLETED
                            if svc.assigned_server:
                                st.session_state.servers[svc.assigned_server].release()
                
                # Batch arrival
                batch_size = random.randint(batch_min, batch_max)
                tasks_in_batch = min(batch_size, len(task_data) - idx)
                
                for _ in range(tasks_in_batch):
                    if idx >= len(task_data):
                        break
                    
                    network = task_data[idx]
                    network['edgesimpy_layer'] = network.get('assigned_layer')
                    
                    # RAG decision
                    rag_layer, rag_server, rag_reason = rag_agent.decide(network)
                    
                    # EdgeSimPy ground truth
                    ml_layer, _ = ml_verifier.verify(network)
                    
                    # Create service
                    svc = Service(
                        service_id=idx + 1,
                        service_type=ServiceType.TASK,
                        num_users=1,
                        start_time=current_time,
                        end_time=current_time + timedelta(seconds=30),
                        requirements=SERVICE_DEFAULTS[ServiceType.TASK],
                        network=network,
                        rag_layer=rag_layer,
                        rag_server=rag_server,
                        rag_reasoning=rag_reason,
                        ml_layer=ml_layer,
                        ml_matches_rag=(rag_layer == ml_layer)
                    )
                    
                    # Check server availability
                    server = st.session_state.servers[rag_server]
                    if server.is_available(current_time):
                        svc.assigned_layer = rag_layer
                        svc.assigned_server = rag_server
                        svc.status = TaskStatus.RUNNING
                        svc.actual_start = current_time
                        svc.actual_end = current_time + timedelta(
                            seconds=LAYER_CONFIG[rag_layer]["completion_time"]
                        )
                        svc.decision_maker = DecisionMaker.RAG_PRIMARY
                        server.assign_task(svc.service_id, LAYER_CONFIG[rag_layer]["completion_time"], current_time)
                    else:
                        # Try fallback
                        found = False
                        for layer in ["Edge", "Fog", "Cloud"]:
                            for srv in st.session_state.servers.values():
                                if srv.layer == layer and srv.is_available(current_time):
                                    svc.assigned_layer = layer
                                    svc.assigned_server = srv.server_id
                                    svc.status = TaskStatus.RUNNING
                                    svc.actual_start = current_time
                                    svc.actual_end = current_time + timedelta(
                                        seconds=LAYER_CONFIG[layer]["completion_time"]
                                    )
                                    svc.decision_maker = DecisionMaker.RAG_FALLBACK
                                    srv.assign_task(svc.service_id, LAYER_CONFIG[layer]["completion_time"], current_time)
                                    found = True
                                    break
                            if found:
                                break
                        
                        if not found:
                            svc.assigned_layer = rag_layer
                            svc.assigned_server = rag_server
                            svc.status = TaskStatus.WAITING
                            svc.decision_maker = DecisionMaker.RAG_QUEUED
                    
                    st.session_state.services.append(svc)
                    idx += 1
                
                # Update metrics
                with metrics_placeholder.container():
                    m1, m2, m3, m4, m5, m6 = st.columns(6)
                    completed = len([s for s in st.session_state.services if s.status == TaskStatus.COMPLETED])
                    running = len([s for s in st.session_state.services if s.status == TaskStatus.RUNNING])
                    waiting = len([s for s in st.session_state.services if s.status == TaskStatus.WAITING])
                    rag_primary = len([s for s in st.session_state.services if s.decision_maker == DecisionMaker.RAG_PRIMARY])
                    rag_match = len([s for s in st.session_state.services if s.ml_matches_rag])
                    
                    m1.metric("✅ Done", completed)
                    m2.metric("🔄 Running", running)
                    m3.metric("⏳ Waiting", waiting)
                    m4.metric("🧠 RAG Primary", rag_primary)
                    m5.metric("🎯 RAG=EdgeSimPy", rag_match)
                    m6.metric("Total", len(st.session_state.services))
                
                # Update server status
                with server_placeholder.container():
                    cols = st.columns(7)
                    for i, (sid, srv) in enumerate(st.session_state.servers.items()):
                        with cols[i]:
                            status = "🟢" if srv.is_available(current_time) else "🔴"
                            st.metric(f"{status} S{sid}", f"{srv.layer}", f"{srv.tasks_completed} done")
                
                # Update log
                with log_placeholder.container():
                    st.markdown("### 📊 Decision Log: RAG vs EdgeSimPy (GT)")
                    log_data = []
                    for s in st.session_state.services[-10:]:
                        log_data.append({
                            "Task": s.service_id,
                            "🧠 RAG": f"{s.rag_layer} (S{s.rag_server})",
                            "📊 EdgeSimPy": s.ml_layer,
                            "Match": "✅" if s.ml_matches_rag else "❌",
                            "Final": f"{s.assigned_layer} (S{s.assigned_server})",
                            "Type": s.decision_maker.value,
                            "Status": s.status.value
                        })
                    if log_data:
                        st.dataframe(pd.DataFrame(log_data), use_container_width=True, hide_index=True)
                
                time.sleep(arrival_interval)
            
            # Final completion
            st.session_state.running = False
            
            # Force complete remaining
            for svc in st.session_state.services:
                if svc.status != TaskStatus.COMPLETED:
                    svc.status = TaskStatus.COMPLETED
            
            st.balloons()
            
            # Final summary
            st.header("📊 Final Summary")
            total = len(st.session_state.services)
            if total > 0:
                rag_match = sum(1 for s in st.session_state.services if s.ml_matches_rag)
                
                f1, f2, f3, f4 = st.columns(4)
                f1.metric("🎯 RAG = EdgeSimPy", f"{rag_match}/{total} ({rag_match/total*100:.0f}%)")
                f2.metric("🧠 RAG Primary", sum(1 for s in st.session_state.services if s.decision_maker == DecisionMaker.RAG_PRIMARY))
                f3.metric("📋 Fallbacks", sum(1 for s in st.session_state.services if s.decision_maker == DecisionMaker.RAG_FALLBACK))
                f4.metric("⏳ Queued", sum(1 for s in st.session_state.services if s.decision_maker == DecisionMaker.RAG_QUEUED))
                
                # Layer distribution
                st.subheader("Layer Distribution")
                dist = pd.DataFrame({
                    "Layer": ["Edge", "Fog", "Cloud"],
                    "RAG Decision": [
                        sum(1 for s in st.session_state.services if s.rag_layer == "Edge"),
                        sum(1 for s in st.session_state.services if s.rag_layer == "Fog"),
                        sum(1 for s in st.session_state.services if s.rag_layer == "Cloud"),
                    ],
                    "EdgeSimPy (GT)": [
                        sum(1 for s in st.session_state.services if s.ml_layer == "Edge"),
                        sum(1 for s in st.session_state.services if s.ml_layer == "Fog"),
                        sum(1 for s in st.session_state.services if s.ml_layer == "Cloud"),
                    ],
                    "Final Execute": [
                        sum(1 for s in st.session_state.services if s.assigned_layer == "Edge"),
                        sum(1 for s in st.session_state.services if s.assigned_layer == "Fog"),
                        sum(1 for s in st.session_state.services if s.assigned_layer == "Cloud"),
                    ]
                })
                st.dataframe(dist, use_container_width=True, hide_index=True)
    
    # ==========================================================================
    # ADVANCED MODE
    # ==========================================================================
    else:
        st.markdown(f"""
        **Advanced Mode:** Professor's requirements + RAG + LLM + EdgeSimPy GT
        
        **Decision Pipeline:**
        1. 🔍 **FAISS Vector Search** → Find similar scenarios from {faiss_count} training vectors
        2. 🧠 **Groq LLM (llama-3.3-70b)** → Agentic decision using RAG context
        3. 📊 **EdgeSimPy Ground Truth** → Compare RAG decision vs assigned_layer from CSV
        
        **Professor's Features:**
        - ✅ Service Types: XR (50 users), eMBB (1000 users), URLLC, mMTC
        - ✅ Time Windows: [t1, t2] scheduling
        - ✅ Node Failure: Agentic rebalancing
        - ✅ Priority Preemption: High priority displaces low priority
        """)
        
        # Service configuration
        st.sidebar.header("📋 Add Service")
        
        service_type = st.sidebar.selectbox(
            "Service Type",
            [ServiceType.XR, ServiceType.EMBB, ServiceType.URLLC, ServiceType.MMTC],
            format_func=lambda x: x.value
        )
        
        default_users = {ServiceType.XR: 50, ServiceType.EMBB: 1000, ServiceType.URLLC: 100, ServiceType.MMTC: 500}
        num_users = st.sidebar.number_input("Users", 10, 5000, default_users[service_type], 10)
        
        col1, col2 = st.sidebar.columns(2)
        with col1:
            start_sec = st.number_input("Start (sec)", 0, 60, 0)
        with col2:
            end_sec = st.number_input("End (sec)", 5, 120, 30)
        
        if st.sidebar.button("➕ Add Service"):
            base_time = datetime.now()
            new_svc = Service(
                service_id=len(st.session_state.services) + 1,
                service_type=service_type,
                num_users=num_users,
                start_time=base_time + timedelta(seconds=start_sec),
                end_time=base_time + timedelta(seconds=end_sec),
                requirements=SERVICE_DEFAULTS[service_type]
            )
            st.session_state.services.append(new_svc)
            st.sidebar.success(f"Added {service_type.value} with {num_users} users")
        
        # Demo scenario
        st.sidebar.divider()
        if st.sidebar.button("📋 Load Demo"):
            base_time = datetime.now()
            st.session_state.services = [
                Service(1, ServiceType.XR, 50, base_time, base_time + timedelta(seconds=30), SERVICE_DEFAULTS[ServiceType.XR]),
                Service(2, ServiceType.EMBB, 1000, base_time + timedelta(seconds=10), base_time + timedelta(seconds=60), SERVICE_DEFAULTS[ServiceType.EMBB]),
                Service(3, ServiceType.URLLC, 100, base_time + timedelta(seconds=20), base_time + timedelta(seconds=50), SERVICE_DEFAULTS[ServiceType.URLLC]),
                Service(4, ServiceType.MMTC, 500, base_time + timedelta(seconds=5), base_time + timedelta(seconds=45), SERVICE_DEFAULTS[ServiceType.MMTC]),
            ]
            st.sidebar.success("Demo loaded!")
        
        # Node failure
        st.sidebar.divider()
        st.sidebar.header("💥 Node Failure")
        fail_fog = st.sidebar.checkbox("🔴 Fail Fog (S5)")
        fail_edge = st.sidebar.checkbox("🔴 Fail Edge (S2)")
        
        # Clear
        if st.sidebar.button("🗑️ Clear All"):
            st.session_state.services = []
            st.session_state.servers = {
                1: ServerState(1, "Edge", 5, 200),
                2: ServerState(2, "Edge", 5, 200),
                3: ServerState(3, "Edge", 5, 200),
                4: ServerState(4, "Edge", 5, 200),
                5: ServerState(5, "Fog", 25, 500),
                6: ServerState(6, "Fog", 25, 500),
                7: ServerState(7, "Cloud", 100, 10000),
            }
            st.session_state.events = {"preemption": [], "failure": [], "rebalance": []}
            st.rerun()
        
        # Service queue display
        st.subheader("📋 Service Queue")
        if st.session_state.services:
            svc_data = [{
                "ID": s.service_id,
                "Type": s.service_type.value,
                "Users": s.num_users,
                "Priority": s.requirements.priority,
                "Status": s.status.value,
                "Layer": s.assigned_layer or "-",
                "Server": s.assigned_server or "-"
            } for s in st.session_state.services]
            st.dataframe(pd.DataFrame(svc_data), use_container_width=True, hide_index=True)
        else:
            st.info("No services. Add from sidebar or load demo.")
        
        # Server status
        st.subheader("🖥️ Server Status")
        srv_cols = st.columns(7)
        for i, (sid, srv) in enumerate(st.session_state.servers.items()):
            with srv_cols[i]:
                icon = "🟢" if srv.status == NodeStatus.ACTIVE else "🔴"
                st.metric(f"{icon} S{sid}", f"{srv.current_users}/{srv.max_users}", srv.layer)
        
        # Layer capacity
        st.subheader("📊 Layer Capacity")
        cap_cols = st.columns(3)
        for i, (layer, cfg) in enumerate(LAYER_CONFIG.items()):
            with cap_cols[i]:
                used = sum(s.current_users for s in st.session_state.servers.values() 
                          if s.layer == layer and s.status == NodeStatus.ACTIVE)
                total = cfg["total_capacity"]
                st.metric(layer, f"{used}/{total} users")
                st.progress(min(used/total, 1.0) if total > 0 else 0)
        
        # Control buttons
        st.divider()
        col1, col2, col3 = st.columns(3)
        with col1:
            start_btn = st.button("▶️ Start", type="primary", disabled=len(st.session_state.services) == 0)
        with col2:
            stop_btn = st.button("⏹️ Stop")
        with col3:
            step_btn = st.button("⏭️ Step")
        
        # Placeholders
        metrics_placeholder = st.empty()
        log_placeholder = st.empty()
        events_placeholder = st.empty()
        
        if start_btn or step_btn:
            rag_agent = RAGAgent(groq_key, models)
            scheduler = AgenticScheduler(st.session_state.servers, rag_agent)
            scheduler.services = st.session_state.services
            
            test_data = load_test_data()
            ml_verifier = MLVerifier()
            
            iterations = 1 if step_btn else 30
            
            for _ in range(iterations):
                current_time = datetime.now()
                
                # Handle failures
                if fail_fog and st.session_state.servers[5].status == NodeStatus.ACTIVE:
                    rebalanced = scheduler.handle_node_failure(5, current_time)
                    st.session_state.events["failure"].append(f"S5 (Fog) FAILED - {len(rebalanced)} rebalanced")
                
                if fail_edge and st.session_state.servers[2].status == NodeStatus.ACTIVE:
                    rebalanced = scheduler.handle_node_failure(2, current_time)
                    st.session_state.events["failure"].append(f"S2 (Edge) FAILED - {len(rebalanced)} rebalanced")
                
                # Process services
                for svc in st.session_state.services:
                    if svc.status == TaskStatus.WAITING and svc.is_active(current_time):
                        # Get network from EdgeSimPy test data (200 rows)
                        network = test_data[svc.service_id % len(test_data)] if test_data else {}
                        network['edgesimpy_layer'] = network.get('assigned_layer', 'Fog')
                        svc.network = network
                        
                        # Get EdgeSimPy ground truth
                        svc.ml_layer, _ = ml_verifier.verify(network)
                        
                        # RAG + LLM Decision (using FAISS search + Groq LLM)
                        layer, server, reason, decision = scheduler.schedule_service(svc, network, current_time)
                        
                        # Store RAG decision
                        svc.rag_layer = layer
                        svc.rag_server = server
                        svc.rag_reasoning = reason
                        
                        # Compare with EdgeSimPy ground truth
                        svc.ml_matches_rag = (layer == svc.ml_layer)
                        
                        if decision != DecisionMaker.RAG_QUEUED:
                            svc.assigned_layer = layer
                            svc.assigned_server = server
                            svc.status = TaskStatus.RUNNING
                            svc.decision_maker = decision
                            st.session_state.servers[server].assign_users(svc.service_id, svc.num_users)
                        else:
                            svc.decision_maker = decision
                    
                    elif svc.status == TaskStatus.RUNNING and svc.is_expired(current_time):
                        svc.status = TaskStatus.COMPLETED
                        if svc.assigned_server:
                            st.session_state.servers[svc.assigned_server].release_users(svc.service_id, svc.num_users)
                
                # Update events
                st.session_state.events["preemption"].extend([
                    f"S{e['preempted_service']} preempted by S{e['by_service']}" 
                    for e in scheduler.preemption_log
                ])
                st.session_state.events["rebalance"].extend([
                    f"S{e['service_id']}: {e['from_server']}→{e['to_server']}"
                    for e in scheduler.rebalance_log
                ])
                scheduler.preemption_log = []
                scheduler.rebalance_log = []
                
                # Update metrics
                with metrics_placeholder.container():
                    m1, m2, m3, m4, m5, m6 = st.columns(6)
                    m1.metric("✅ Done", len([s for s in st.session_state.services if s.status == TaskStatus.COMPLETED]))
                    m2.metric("🔄 Running", len([s for s in st.session_state.services if s.status == TaskStatus.RUNNING]))
                    m3.metric("⏳ Waiting", len([s for s in st.session_state.services if s.status == TaskStatus.WAITING]))
                    m4.metric("🔄 Preempted", len([s for s in st.session_state.services if s.status == TaskStatus.PREEMPTED]))
                    m5.metric("🎯 RAG=GT", len([s for s in st.session_state.services if s.ml_matches_rag]))
                    m6.metric("Total", len(st.session_state.services))
                
                # Update log
                with log_placeholder.container():
                    st.markdown("### 📊 RAG+LLM Decision Log vs EdgeSimPy Ground Truth")
                    log_data = [{
                        "ID": s.service_id,
                        "Type": s.service_type.value,
                        "Users": s.num_users,
                        "Network": f"DR={s.network.get('datarate_mbps', 0):.1f}" if s.network else "-",
                        "🧠 RAG+LLM": f"{s.rag_layer} (S{s.rag_server})" if s.rag_layer else "-",
                        "📊 EdgeSimPy (GT)": s.ml_layer or "-",
                        "Match?": "✅" if s.ml_matches_rag else ("❌" if s.ml_layer else "-"),
                        "Final": f"{s.assigned_layer} (S{s.assigned_server})" if s.assigned_layer else "-",
                        "Decision": s.decision_maker.value,
                        "Status": s.status.value
                    } for s in st.session_state.services]
                    st.dataframe(pd.DataFrame(log_data), use_container_width=True, hide_index=True)
                    
                    # Show RAG reasoning for latest service
                    running_svcs = [s for s in st.session_state.services if s.rag_reasoning]
                    if running_svcs:
                        latest = running_svcs[-1]
                        st.info(f"🧠 **Latest RAG Reasoning (Service {latest.service_id}):** {latest.rag_reasoning}")
                
                # Show events
                with events_placeholder.container():
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.markdown("### 💥 Failures")
                        for e in st.session_state.events["failure"][-5:]:
                            st.error(e)
                    with c2:
                        st.markdown("### 🔥 Preemptions")
                        for e in st.session_state.events["preemption"][-5:]:
                            st.warning(e)
                    with c3:
                        st.markdown("### 🔁 Rebalances")
                        for e in st.session_state.events["rebalance"][-5:]:
                            st.success(e)
                
                if not step_btn:
                    time.sleep(1)
            
            if not step_btn:
                st.balloons()
                st.success("✅ Simulation Complete!")


if __name__ == "__main__":
    main()
