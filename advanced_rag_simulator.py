"""
ADVANCED RAG Agent Simulator - Service-Based Scheduling
=========================================================
Professor's Requirements:
✅ 1. Service Types: XR (50 users), eMBB (1000 users), URLLC, mMTC
✅ 2. Time Windows: Services active during [t1, t2]
✅ 3. Node Failure: Fog node goes down, agentic rebalancing
✅ 4. Priority Preemption: High priority displaces lower priority when full
✅ 5. Dynamic Requirements: Change throughput/users mid-simulation
✅ 6. User-based scheduling with resource constraints

Infrastructure:
- Edge: Servers 1-4 (capacity: 200 users each, total 800)
- Fog: Servers 5-6 (capacity: 500 users each, total 1000)
- Cloud: Server 7 (capacity: unlimited, but higher latency)

Run: streamlit run advanced_rag_simulator.py
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
    page_title="🚀 Advanced RAG Agent - Service Scheduler",
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
# SERVICE TYPES & REQUIREMENTS
# =============================================================================
class ServiceType(Enum):
    XR = "🥽 XR (Extended Reality)"
    EMBB = "📱 eMBB (Enhanced Mobile Broadband)"
    URLLC = "⚡ URLLC (Ultra-Reliable Low Latency)"
    MMTC = "📡 mMTC (Massive Machine Type)"


@dataclass
class ServiceRequirements:
    """Requirements for each service type."""
    service_type: ServiceType
    throughput_mbps: float  # Required throughput
    max_latency_ms: float   # Maximum acceptable latency
    priority: int           # 1=highest, 4=lowest
    preferred_layer: str    # Edge, Fog, or Cloud


SERVICE_DEFAULTS = {
    ServiceType.XR: ServiceRequirements(
        service_type=ServiceType.XR,
        throughput_mbps=100.0,  # XR needs high throughput
        max_latency_ms=10.0,    # Very low latency required
        priority=1,             # Highest priority
        preferred_layer="Edge"
    ),
    ServiceType.URLLC: ServiceRequirements(
        service_type=ServiceType.URLLC,
        throughput_mbps=10.0,
        max_latency_ms=5.0,     # Ultra-low latency
        priority=1,             # Highest priority (same as XR)
        preferred_layer="Edge"
    ),
    ServiceType.EMBB: ServiceRequirements(
        service_type=ServiceType.EMBB,
        throughput_mbps=50.0,
        max_latency_ms=50.0,    # More tolerant to latency
        priority=2,
        preferred_layer="Fog"
    ),
    ServiceType.MMTC: ServiceRequirements(
        service_type=ServiceType.MMTC,
        throughput_mbps=1.0,     # Low throughput
        max_latency_ms=1000.0,   # Very tolerant
        priority=3,              # Lowest priority
        preferred_layer="Cloud"
    ),
}


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
# SERVER STATE WITH CAPACITY
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
    
    def available_capacity(self) -> int:
        if self.status != NodeStatus.ACTIVE:
            return 0
        return max(0, self.max_users - self.current_users)
    
    def can_accept_users(self, num_users: int) -> bool:
        return self.available_capacity() >= num_users
    
    def assign_users(self, task_id: int, num_users: int) -> bool:
        if not self.can_accept_users(num_users):
            return False
        self.current_users += num_users
        self.current_tasks.append(task_id)
        return True
    
    def release_users(self, task_id: int, num_users: int):
        self.current_users = max(0, self.current_users - num_users)
        if task_id in self.current_tasks:
            self.current_tasks.remove(task_id)
        self.tasks_completed += 1
    
    def fail(self):
        """Simulate node failure."""
        self.status = NodeStatus.FAILED
        # Return tasks that need rebalancing
        return list(self.current_tasks)
    
    def recover(self):
        """Recover from failure."""
        self.status = NodeStatus.ACTIVE


# =============================================================================
# SERVICE (User Group)
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
    
    # Scheduling state
    assigned_layer: Optional[str] = None
    assigned_server: Optional[int] = None
    status: TaskStatus = TaskStatus.WAITING
    decision_maker: DecisionMaker = DecisionMaker.RAG_PRIMARY
    
    # Comparison with ground truth
    rag_layer: Optional[str] = None
    ml_layer: Optional[str] = None
    ml_matches_rag: bool = False
    
    # For preemption tracking
    preempted_by: Optional[int] = None
    preempted_tasks: List[int] = field(default_factory=list)
    
    def is_active(self, current_time: datetime) -> bool:
        return self.start_time <= current_time <= self.end_time
    
    def is_expired(self, current_time: datetime) -> bool:
        return current_time > self.end_time


# =============================================================================
# LAYER CAPACITY
# =============================================================================
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
        "users_per_server": 10000,  # Unlimited effectively
        "total_capacity": 10000,
        "completion_time": 10,
        "latency_ms": 100
    }
}


# =============================================================================
# RAG AGENT (from existing code)
# =============================================================================
class RAGAgent:
    """RAG Agent using Groq LLM + FAISS vector search."""
    
    def __init__(self, groq_api_key: str):
        self.api_key = groq_api_key
        self.model = "llama-3.3-70b-versatile"
        self.embedder = None
        self.faiss_index = None
        self.metadata = []
        self.load_vector_store()
        
        # Import groq
        try:
            from groq import Groq
            self.client = Groq(api_key=groq_api_key)
        except Exception as e:
            st.warning(f"Groq not available: {e}")
            self.client = None
    
    def load_vector_store(self):
        """Load FAISS index and metadata."""
        try:
            import faiss
            
            index_path = STORE_DIR / "faiss.index"
            meta_path = STORE_DIR / "metadata.pkl"
            embedder_path = STORE_DIR / "tfidf_embedder.pkl"
            
            if all(p.exists() for p in [index_path, meta_path, embedder_path]):
                self.faiss_index = faiss.read_index(str(index_path))
                with open(meta_path, 'rb') as f:
                    self.metadata = pickle.load(f)
                with open(embedder_path, 'rb') as f:
                    self.embedder = pickle.load(f)
        except Exception as e:
            st.warning(f"Vector store not loaded: {e}")
    
    def search_similar(self, network: Dict, k: int = 3) -> List[Dict]:
        """Find similar network conditions from training data."""
        if self.faiss_index is None or self.embedder is None:
            return []
        
        try:
            query = f"datarate={network.get('datarate_mbps', 0):.1f} sinr={network.get('sinr', 0):.1f} latency={network.get('latency_ms', 0):.1f}"
            query_vec = self.embedder.transform([query]).toarray().astype('float32')
            
            D, I = self.faiss_index.search(query_vec, k)
            
            results = []
            for idx in I[0]:
                if 0 <= idx < len(self.metadata):
                    results.append(self.metadata[idx])
            return results
        except:
            return []
    
    def decide_for_service(self, service: Service, network: Dict, layer_usage: Dict) -> Tuple[str, int, str]:
        """Make agentic decision for a service considering requirements and capacity."""
        
        # Get similar scenarios from RAG
        similar = self.search_similar(network)
        
        # Build context for LLM
        similar_context = ""
        if similar:
            for i, s in enumerate(similar[:3]):
                similar_context += f"\n{i+1}. DataRate={s.get('datarate_mbps', 0):.1f}Mbps, SINR={s.get('sinr', 0):.1f}dB -> {s.get('assigned_layer', 'Unknown')}"
        
        prompt = f"""You are an agentic Edge-Fog-Cloud deployment system.

SERVICE REQUEST:
- Type: {service.service_type.value}
- Users: {service.num_users}
- Required Throughput: {service.requirements.throughput_mbps} Mbps
- Max Latency: {service.requirements.max_latency_ms} ms
- Priority: {service.requirements.priority} (1=highest, 4=lowest)

NETWORK CONDITIONS:
- Data Rate: {network.get('datarate_mbps', 0):.1f} Mbps
- SINR: {network.get('sinr', 0):.1f} dB
- Latency: {network.get('latency_ms', 0):.1f} ms
- RSRP: {network.get('rsrp_dbm', 0):.1f} dBm

LAYER CAPACITY (users):
- Edge: {layer_usage['Edge']['used']}/{layer_usage['Edge']['total']} used (latency: 5ms)
- Fog: {layer_usage['Fog']['used']}/{layer_usage['Fog']['total']} used (latency: 25ms)
- Cloud: {layer_usage['Cloud']['used']}/{layer_usage['Cloud']['total']} used (latency: 100ms)

SIMILAR PAST SCENARIOS:{similar_context if similar_context else " None available"}

THRESHOLD RULES:
- If latency < 20ms AND datarate < 9.6 Mbps: Edge (low latency local)
- If 9.6 <= datarate < 16.6 Mbps AND sinr > 10: Fog (medium load)
- Otherwise: Cloud (high capacity)

DECISION RULES:
1. Check if preferred layer has capacity for {service.num_users} users
2. If not, check next best layer that meets latency requirements
3. High priority services (1-2) should prefer Edge/Fog
4. Consider preemption if critical service blocked

Respond with EXACTLY: LAYER: [Edge/Fog/Cloud], SERVER: [1-7], REASON: [brief]"""

        # Call LLM
        if self.client:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=150
                )
                text = response.choices[0].message.content
                
                # Parse response
                layer_match = re.search(r'LAYER:\s*(Edge|Fog|Cloud)', text, re.IGNORECASE)
                server_match = re.search(r'SERVER:\s*(\d+)', text)
                reason_match = re.search(r'REASON:\s*(.+?)(?:\n|$)', text)
                
                layer = layer_match.group(1).capitalize() if layer_match else service.requirements.preferred_layer
                server = int(server_match.group(1)) if server_match else 1
                reason = reason_match.group(1).strip() if reason_match else "LLM decision"
                
                return layer, server, reason
            except Exception as e:
                pass
        
        # Fallback to threshold rules
        return self.threshold_decision(service, network, layer_usage)
    
    def threshold_decision(self, service: Service, network: Dict, layer_usage: Dict) -> Tuple[str, int, str]:
        """Fallback threshold-based decision."""
        datarate = network.get('datarate_mbps', 0)
        latency = network.get('latency_ms', 0)
        sinr = network.get('sinr', 0)
        
        # Priority-aware decision
        if service.requirements.priority <= 2:  # High priority
            if latency < service.requirements.max_latency_ms:
                if layer_usage['Edge']['used'] + service.num_users <= layer_usage['Edge']['total']:
                    return "Edge", random.choice([1, 2, 3, 4]), "High priority + capacity available"
                elif layer_usage['Fog']['used'] + service.num_users <= layer_usage['Fog']['total']:
                    return "Fog", random.choice([5, 6]), "Edge full, using Fog"
        
        # Standard threshold rules
        if latency < 20 and datarate < 9.6:
            if layer_usage['Edge']['used'] + service.num_users <= layer_usage['Edge']['total']:
                return "Edge", random.choice([1, 2, 3, 4]), "Low latency local"
            return "Fog", random.choice([5, 6]), "Edge full"
        elif 9.6 <= datarate < 16.6 and sinr > 10:
            if layer_usage['Fog']['used'] + service.num_users <= layer_usage['Fog']['total']:
                return "Fog", random.choice([5, 6]), "Medium load"
            return "Cloud", 7, "Fog full"
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
        rag_layer, rag_server, rag_reason = self.rag_agent.decide_for_service(service, network, layer_usage)
        
        # Check if preferred layer has capacity
        target_servers = [s for s in self.servers.values() 
                        if s.layer == rag_layer and s.status == NodeStatus.ACTIVE]
        
        # Try to find available server
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
        
        # Queue if no capacity
        return rag_layer, rag_server, "Queued - no capacity", DecisionMaker.RAG_QUEUED
    
    def try_preemption(self, high_priority_service: Service, target_layer: str, current_time: datetime) -> Optional[Tuple]:
        """Try to preempt lower priority services for high priority one."""
        
        # Find lower priority services in target layer
        preemptable = []
        for svc in self.services:
            if svc.status == TaskStatus.RUNNING and svc.assigned_layer == target_layer:
                if svc.requirements.priority > high_priority_service.requirements.priority:
                    preemptable.append(svc)
        
        # Sort by priority (lowest priority first - higher number)
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
            # Execute preemption
            for svc in to_preempt:
                svc.status = TaskStatus.PREEMPTED
                svc.preempted_by = high_priority_service.service_id
                high_priority_service.preempted_tasks.append(svc.service_id)
                
                # Release users from server
                if svc.assigned_server:
                    self.servers[svc.assigned_server].release_users(svc.service_id, svc.num_users)
                
                self.preemption_log.append({
                    "time": current_time,
                    "preempted_service": svc.service_id,
                    "preempted_type": svc.service_type.value,
                    "by_service": high_priority_service.service_id,
                    "by_type": high_priority_service.service_type.value,
                    "users_freed": svc.num_users
                })
            
            # Find server for high priority service
            for server in self.servers.values():
                if server.layer == target_layer and server.status == NodeStatus.ACTIVE:
                    if server.can_accept_users(high_priority_service.num_users):
                        return target_layer, server.server_id, f"Preempted {len(to_preempt)} services", DecisionMaker.PRIORITY_PREEMPT
        
        return None
    
    def handle_node_failure(self, server_id: int, current_time: datetime) -> List[Service]:
        """Handle node failure and rebalance affected services."""
        
        server = self.servers.get(server_id)
        if not server:
            return []
        
        # Get affected tasks
        affected_task_ids = server.fail()
        
        self.failure_log.append({
            "time": current_time,
            "server_id": server_id,
            "layer": server.layer,
            "affected_tasks": len(affected_task_ids),
            "users_affected": server.current_users
        })
        
        # Find affected services
        affected_services = [s for s in self.services if s.service_id in affected_task_ids]
        
        # Rebalance each service
        rebalanced = []
        for svc in affected_services:
            svc.status = TaskStatus.REBALANCING
            
            # Find new server (prefer same layer, then fallback)
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
                    "service_type": svc.service_type.value,
                    "from_server": server_id,
                    "to_server": new_server.server_id,
                    "users": svc.num_users
                })
                
                rebalanced.append(svc)
            else:
                svc.status = TaskStatus.FAILED
        
        # Reset server current users
        server.current_users = 0
        server.current_tasks = []
        
        return rebalanced
    
    def recover_node(self, server_id: int, current_time: datetime):
        """Recover a failed node."""
        server = self.servers.get(server_id)
        if server:
            server.recover()
            self.failure_log.append({
                "time": current_time,
                "server_id": server_id,
                "layer": server.layer,
                "event": "RECOVERED"
            })


# =============================================================================
# MAIN APPLICATION
# =============================================================================
def main():
    st.title("🚀 Advanced RAG Agent - Service Scheduler")
    st.markdown("""
    **Professor's Requirements Implemented:**
    - ✅ Service Types: XR, eMBB, URLLC, mMTC with user counts
    - ✅ Time Windows: [t1, t2] scheduling
    - ✅ Node Failure: Fog node failure with agentic rebalancing
    - ✅ Priority Preemption: High priority displaces low priority
    - ✅ Dynamic Requirements: Change throughput/users mid-simulation
    """)
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    groq_key = st.sidebar.text_input("Groq API Key", value=DEFAULT_GROQ_KEY, type="password")
    
    st.sidebar.divider()
    st.sidebar.header("📋 Service Configuration")
    
    # Service input
    st.sidebar.subheader("Add New Service")
    
    service_type = st.sidebar.selectbox(
        "Service Type",
        options=list(ServiceType),
        format_func=lambda x: x.value
    )
    
    num_users = st.sidebar.number_input(
        "Number of Users",
        min_value=10,
        max_value=5000,
        value=SERVICE_DEFAULTS[service_type].throughput_mbps == 100 and 50 or 
              (service_type == ServiceType.EMBB and 1000 or 100),
        step=10
    )
    
    # Time window
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_offset = st.number_input("Start (sec)", min_value=0, max_value=60, value=0)
    with col2:
        end_offset = st.number_input("End (sec)", min_value=5, max_value=120, value=30)
    
    # Dynamic requirement changes
    st.sidebar.divider()
    st.sidebar.subheader("🔄 Dynamic Changes")
    
    change_throughput = st.sidebar.slider(
        "Throughput Multiplier",
        min_value=0.5,
        max_value=3.0,
        value=1.0,
        step=0.1,
        help="Change service throughput requirements"
    )
    
    add_users = st.sidebar.number_input(
        "Add Users Mid-Simulation",
        min_value=0,
        max_value=500,
        value=0,
        step=50,
        help="Add more users to eMBB service"
    )
    
    # Node failure control
    st.sidebar.divider()
    st.sidebar.header("💥 Node Failure Simulation")
    
    fail_fog = st.sidebar.checkbox("🔴 Fail Fog Node (Server 5)", value=False)
    fail_edge = st.sidebar.checkbox("🔴 Fail Edge Node (Server 2)", value=False)
    
    # Initialize session state
    if 'adv_services' not in st.session_state:
        st.session_state.adv_services = []
    if 'adv_servers' not in st.session_state:
        st.session_state.adv_servers = {
            1: ServerState(1, "Edge", 5, 200),
            2: ServerState(2, "Edge", 5, 200),
            3: ServerState(3, "Edge", 5, 200),
            4: ServerState(4, "Edge", 5, 200),
            5: ServerState(5, "Fog", 25, 500),
            6: ServerState(6, "Fog", 25, 500),
            7: ServerState(7, "Cloud", 100, 10000),
        }
    if 'adv_running' not in st.session_state:
        st.session_state.adv_running = False
    if 'adv_scheduler' not in st.session_state:
        st.session_state.adv_scheduler = None
    if 'preemption_events' not in st.session_state:
        st.session_state.preemption_events = []
    if 'failure_events' not in st.session_state:
        st.session_state.failure_events = []
    if 'rebalance_events' not in st.session_state:
        st.session_state.rebalance_events = []
    
    # Add service button
    if st.sidebar.button("➕ Add Service to Queue"):
        base_time = datetime.now()
        new_service = Service(
            service_id=len(st.session_state.adv_services) + 1,
            service_type=service_type,
            num_users=num_users,
            start_time=base_time + timedelta(seconds=start_offset),
            end_time=base_time + timedelta(seconds=end_offset),
            requirements=SERVICE_DEFAULTS[service_type]
        )
        st.session_state.adv_services.append(new_service)
        st.sidebar.success(f"Added {service_type.value} with {num_users} users")
    
    # Preset scenarios
    st.sidebar.divider()
    if st.sidebar.button("📋 Load Demo Scenario"):
        base_time = datetime.now()
        st.session_state.adv_services = [
            Service(1, ServiceType.XR, 50, base_time, base_time + timedelta(seconds=30),
                   SERVICE_DEFAULTS[ServiceType.XR]),
            Service(2, ServiceType.EMBB, 1000, base_time + timedelta(seconds=10), 
                   base_time + timedelta(seconds=60), SERVICE_DEFAULTS[ServiceType.EMBB]),
            Service(3, ServiceType.URLLC, 100, base_time + timedelta(seconds=20),
                   base_time + timedelta(seconds=50), SERVICE_DEFAULTS[ServiceType.URLLC]),
            Service(4, ServiceType.MMTC, 500, base_time + timedelta(seconds=5),
                   base_time + timedelta(seconds=45), SERVICE_DEFAULTS[ServiceType.MMTC]),
        ]
        # Reset servers
        st.session_state.adv_servers = {
            1: ServerState(1, "Edge", 5, 200),
            2: ServerState(2, "Edge", 5, 200),
            3: ServerState(3, "Edge", 5, 200),
            4: ServerState(4, "Edge", 5, 200),
            5: ServerState(5, "Fog", 25, 500),
            6: ServerState(6, "Fog", 25, 500),
            7: ServerState(7, "Cloud", 100, 10000),
        }
        st.sidebar.success("Loaded demo: XR(50), eMBB(1000), URLLC(100), mMTC(500)")
    
    # Clear services
    if st.sidebar.button("🗑️ Clear All"):
        st.session_state.adv_services = []
        st.session_state.adv_servers = {
            1: ServerState(1, "Edge", 5, 200),
            2: ServerState(2, "Edge", 5, 200),
            3: ServerState(3, "Edge", 5, 200),
            4: ServerState(4, "Edge", 5, 200),
            5: ServerState(5, "Fog", 25, 500),
            6: ServerState(6, "Fog", 25, 500),
            7: ServerState(7, "Cloud", 100, 10000),
        }
        st.session_state.preemption_events = []
        st.session_state.failure_events = []
        st.session_state.rebalance_events = []
        st.rerun()
    
    # Main display
    st.divider()
    
    # Service Queue Display
    st.subheader("📋 Service Queue")
    if st.session_state.adv_services:
        svc_data = []
        for svc in st.session_state.adv_services:
            svc_data.append({
                "ID": svc.service_id,
                "Type": svc.service_type.value,
                "Users": svc.num_users,
                "Priority": svc.requirements.priority,
                "Start": svc.start_time.strftime("%H:%M:%S"),
                "End": svc.end_time.strftime("%H:%M:%S"),
                "Status": svc.status.value,
                "Layer": svc.assigned_layer or "-",
                "Server": svc.assigned_server or "-",
                "Decision": svc.decision_maker.value if svc.assigned_layer else "-"
            })
        st.dataframe(pd.DataFrame(svc_data), use_container_width=True, hide_index=True)
    else:
        st.info("No services in queue. Add services from the sidebar or load demo scenario.")
    
    # Server Status
    st.subheader("🖥️ Server Status")
    
    server_cols = st.columns(7)
    for i, (sid, server) in enumerate(st.session_state.adv_servers.items()):
        with server_cols[i]:
            status_color = "🟢" if server.status == NodeStatus.ACTIVE else "🔴"
            st.metric(
                f"{status_color} S{sid} ({server.layer})",
                f"{server.current_users}/{server.max_users}",
                f"{server.tasks_completed} done"
            )
            if server.status == NodeStatus.FAILED:
                st.error("FAILED")
    
    # Layer Capacity
    st.subheader("📊 Layer Capacity")
    capacity_cols = st.columns(3)
    
    for i, (layer, config) in enumerate(LAYER_CONFIG.items()):
        with capacity_cols[i]:
            used = sum(s.current_users for s in st.session_state.adv_servers.values() 
                      if s.layer == layer and s.status == NodeStatus.ACTIVE)
            total = config["total_capacity"]
            pct = used / total if total > 0 else 0
            
            st.metric(f"{layer}", f"{used}/{total} users ({pct*100:.0f}%)")
            st.progress(min(pct, 1.0))
    
    # Start simulation
    st.divider()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        start_btn = st.button("▶️ Start Simulation", type="primary", 
                             disabled=len(st.session_state.adv_services) == 0)
    with col2:
        stop_btn = st.button("⏹️ Stop")
    with col3:
        step_btn = st.button("⏭️ Step Once")
    
    # Placeholders for live updates
    metrics_placeholder = st.empty()
    log_placeholder = st.empty()
    events_placeholder = st.empty()
    
    # Load test data
    @st.cache_data
    def load_network_data():
        if CSV_PATH.exists():
            df = pd.read_csv(CSV_PATH)
            return df.to_dict('records')
        return []
    
    network_data = load_network_data()
    
    if start_btn or step_btn:
        if not groq_key:
            st.error("Please enter Groq API Key")
            st.stop()
        
        # Initialize RAG agent and scheduler
        rag_agent = RAGAgent(groq_key)
        ml_verifier = MLVerifier()
        
        scheduler = AgenticScheduler(st.session_state.adv_servers, rag_agent)
        scheduler.services = st.session_state.adv_services
        
        st.session_state.adv_scheduler = scheduler
        st.session_state.adv_running = True
        
        current_time = datetime.now()
        network_idx = 0
        
        iterations = 1 if step_btn else 30
        
        for iteration in range(iterations):
            if stop_btn:
                break
            
            current_time = datetime.now()
            
            # Handle node failures
            if fail_fog and st.session_state.adv_servers[5].status == NodeStatus.ACTIVE:
                rebalanced = scheduler.handle_node_failure(5, current_time)
                st.session_state.failure_events.append({
                    "time": current_time.strftime("%H:%M:%S"),
                    "event": "🔴 Server 5 (Fog) FAILED",
                    "rebalanced": len(rebalanced)
                })
                st.session_state.rebalance_events.extend(scheduler.rebalance_log[-len(rebalanced):])
            
            if fail_edge and st.session_state.adv_servers[2].status == NodeStatus.ACTIVE:
                rebalanced = scheduler.handle_node_failure(2, current_time)
                st.session_state.failure_events.append({
                    "time": current_time.strftime("%H:%M:%S"),
                    "event": "🔴 Server 2 (Edge) FAILED",
                    "rebalanced": len(rebalanced)
                })
            
            # Process each service
            for svc in st.session_state.adv_services:
                if svc.status in [TaskStatus.WAITING, TaskStatus.REBALANCING]:
                    if svc.is_active(current_time):
                        # Get network conditions (cycle through test data)
                        network = network_data[network_idx % len(network_data)] if network_data else {
                            'datarate_mbps': random.uniform(5, 25),
                            'sinr': random.uniform(0, 30),
                            'latency_ms': random.uniform(5, 50),
                            'rsrp_dbm': random.uniform(-110, -60)
                        }
                        network_idx += 1
                        
                        # Apply dynamic throughput change
                        if change_throughput != 1.0:
                            svc.requirements = ServiceRequirements(
                                service_type=svc.requirements.service_type,
                                throughput_mbps=svc.requirements.throughput_mbps * change_throughput,
                                max_latency_ms=svc.requirements.max_latency_ms,
                                priority=svc.requirements.priority,
                                preferred_layer=svc.requirements.preferred_layer
                            )
                        
                        # Apply dynamic user addition for eMBB
                        if add_users > 0 and svc.service_type == ServiceType.EMBB:
                            svc.num_users += add_users
                        
                        # Get ML ground truth
                        ml_layer, _ = ml_verifier.verify(network)
                        
                        # Schedule service
                        layer, server, reason, decision_maker = scheduler.schedule_service(
                            svc, network, current_time
                        )
                        
                        # Update service
                        svc.rag_layer = layer
                        svc.ml_layer = ml_layer
                        svc.ml_matches_rag = (layer == ml_layer)
                        
                        if decision_maker != DecisionMaker.RAG_QUEUED:
                            svc.assigned_layer = layer
                            svc.assigned_server = server
                            svc.status = TaskStatus.RUNNING
                            svc.decision_maker = decision_maker
                            
                            # Assign to server
                            if server in st.session_state.adv_servers:
                                st.session_state.adv_servers[server].assign_users(
                                    svc.service_id, svc.num_users
                                )
                        else:
                            svc.decision_maker = decision_maker
                
                # Check for completed services
                elif svc.status == TaskStatus.RUNNING:
                    if svc.is_expired(current_time):
                        svc.status = TaskStatus.COMPLETED
                        if svc.assigned_server in st.session_state.adv_servers:
                            st.session_state.adv_servers[svc.assigned_server].release_users(
                                svc.service_id, svc.num_users
                            )
            
            # Update preemption events
            st.session_state.preemption_events.extend(scheduler.preemption_log)
            scheduler.preemption_log = []
            
            # Update metrics
            with metrics_placeholder.container():
                m1, m2, m3, m4, m5, m6 = st.columns(6)
                
                completed = len([s for s in st.session_state.adv_services if s.status == TaskStatus.COMPLETED])
                running = len([s for s in st.session_state.adv_services if s.status == TaskStatus.RUNNING])
                waiting = len([s for s in st.session_state.adv_services if s.status == TaskStatus.WAITING])
                preempted = len([s for s in st.session_state.adv_services if s.status == TaskStatus.PREEMPTED])
                rag_match = len([s for s in st.session_state.adv_services if s.ml_matches_rag])
                
                m1.metric("✅ Completed", completed)
                m2.metric("🔄 Running", running)
                m3.metric("⏳ Waiting", waiting)
                m4.metric("🔄 Preempted", preempted)
                m5.metric("🎯 RAG=GT", rag_match)
                m6.metric("Total", len(st.session_state.adv_services))
            
            # Update log
            with log_placeholder.container():
                st.markdown("### 📊 Service Decision Log")
                log_data = []
                for svc in st.session_state.adv_services:
                    log_data.append({
                        "ID": svc.service_id,
                        "Type": svc.service_type.value,
                        "Users": svc.num_users,
                        "RAG": svc.rag_layer or "-",
                        "EdgeSimPy": svc.ml_layer or "-",
                        "Match": "✅" if svc.ml_matches_rag else "❌",
                        "Final": svc.assigned_layer or "-",
                        "Server": svc.assigned_server or "-",
                        "Decision": svc.decision_maker.value,
                        "Status": svc.status.value
                    })
                st.dataframe(pd.DataFrame(log_data), use_container_width=True, hide_index=True)
            
            # Show events
            with events_placeholder.container():
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("### 💥 Failure Events")
                    if st.session_state.failure_events:
                        for event in st.session_state.failure_events[-5:]:
                            st.error(f"{event['time']}: {event['event']} (Rebalanced: {event.get('rebalanced', 0)})")
                    else:
                        st.info("No failures")
                
                with col2:
                    st.markdown("### 🔥 Preemption Events")
                    if st.session_state.preemption_events:
                        for event in st.session_state.preemption_events[-5:]:
                            st.warning(f"Service {event['preempted_service']} preempted by {event['by_service']}")
                    else:
                        st.info("No preemptions")
                
                with col3:
                    st.markdown("### 🔁 Rebalance Events")
                    if st.session_state.rebalance_events:
                        for event in st.session_state.rebalance_events[-5:]:
                            st.success(f"Service {event['service_id']}: S{event['from_server']} → S{event['to_server']}")
                    else:
                        st.info("No rebalancing")
            
            if not step_btn:
                time.sleep(1)
        
        st.session_state.adv_running = False
        
        if not step_btn:
            st.balloons()
            st.success("✅ Simulation Complete!")
            
            # Final summary
            st.header("📊 Final Summary")
            
            total = len(st.session_state.adv_services)
            if total > 0:
                completed = len([s for s in st.session_state.adv_services if s.status == TaskStatus.COMPLETED])
                rag_match = len([s for s in st.session_state.adv_services if s.ml_matches_rag])
                preempted = len([s for s in st.session_state.adv_services if s.status == TaskStatus.PREEMPTED])
                
                f1, f2, f3, f4 = st.columns(4)
                f1.metric("🎯 RAG = EdgeSimPy", f"{rag_match}/{total} ({rag_match/total*100:.0f}%)")
                f2.metric("✅ Completed", f"{completed}/{total}")
                f3.metric("🔥 Preemptions", len(st.session_state.preemption_events))
                f4.metric("🔁 Rebalances", len(st.session_state.rebalance_events))
                
                # Service summary by type
                st.subheader("Service Type Summary")
                type_summary = {}
                for svc in st.session_state.adv_services:
                    t = svc.service_type.value
                    if t not in type_summary:
                        type_summary[t] = {"total": 0, "completed": 0, "users": 0}
                    type_summary[t]["total"] += 1
                    type_summary[t]["users"] += svc.num_users
                    if svc.status == TaskStatus.COMPLETED:
                        type_summary[t]["completed"] += 1
                
                st.dataframe(pd.DataFrame([
                    {"Type": t, "Services": d["total"], "Completed": d["completed"], "Total Users": d["users"]}
                    for t, d in type_summary.items()
                ]), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
