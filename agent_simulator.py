"""
Agent-Based Real-Time Deployment Simulator
==========================================
Clear separation between:
1. ML MODEL: Predicts layer based on network conditions
2. AGENT: Makes REAL decision based on ML + Available Nodes

Agent Logic:
- Gets ML prediction (Edge/Fog/Cloud)
- Checks if predicted layer has available servers
- If YES: Deploy to ML-recommended layer
- If NO: Make intelligent fallback decision

Run: streamlit run agent_simulator.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
import random
from datetime import datetime, timedelta
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum

# Page config
st.set_page_config(
    page_title="Agent-Based Deployment Simulator",
    page_icon="🤖",
    layout="wide"
)

# Constants
TASK_ARRIVAL_INTERVAL = 4  # seconds
COMPLETION_TIMES = {
    "Edge": 3,   # seconds
    "Fog": 6,    # seconds  
    "Cloud": 10  # seconds
}

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

# CSV path
BASE_DIR = Path(__file__).parent
CSV_PATH = BASE_DIR / "data" / "simulation_data.csv"
if not CSV_PATH.exists():
    CSV_PATH = Path(r"c:\Users\Sahil Padole\Videos\AI_agent_ml_threshold\data\edgesimpy_failure_ml_+_thresh_(gb)_no_failure_20251223_075347_results.csv")


class TaskStatus(Enum):
    WAITING = "waiting"
    RUNNING = "running"
    COMPLETED = "completed"


class DecisionType(Enum):
    ML_FOLLOWED = "ML Followed"           # Agent followed ML recommendation
    FALLBACK_USED = "Fallback Used"       # ML layer busy, used alternative
    QUEUED = "Queued"                     # Had to wait for server


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
    """Task with ML prediction and Agent decision."""
    task_id: int
    arrival_time: datetime
    network_condition: Dict
    
    # ML Model Output
    ml_prediction: str
    ml_server: int
    
    # Agent Decision (may differ from ML)
    agent_decision: str
    agent_server: int
    agent_reason: str
    decision_type: DecisionType
    
    # Execution
    status: TaskStatus = TaskStatus.WAITING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    wait_time: float = 0.0


def load_csv():
    """Load CSV data."""
    if CSV_PATH.exists():
        return pd.read_csv(CSV_PATH)
    return None


def get_ml_prediction(network: Dict) -> Tuple[str, str]:
    """
    ML MODEL: Predicts deployment layer based on network conditions.
    This simulates the Gradient Boosting model's prediction.
    """
    datarate = network['datarate_mbps']
    sinr = network['sinr']
    latency = network['latency_ms']
    rsrp = network['rsrp_dbm']
    
    # Paper's threshold algorithm (simulating ML model)
    if latency < 20 and datarate < 9.64:
        return "Edge", "Low latency & low bandwidth → Edge optimal"
    elif 9.64 <= datarate <= 16.6 and sinr > 10:
        return "Fog", "Medium bandwidth & good SINR → Fog optimal"
    else:
        return "Cloud", "High bandwidth or poor SINR → Cloud optimal"


def agent_decide(
    ml_prediction: str,
    servers: Dict[int, ServerState],
    network: Dict,
    current_time: datetime
) -> Tuple[str, int, str, DecisionType]:
    """
    AGENT: Makes real-time decision based on ML prediction + Available nodes.
    
    Logic:
    1. Try ML-recommended layer first
    2. If all servers busy → check fallback layers
    3. If all fallbacks busy → queue on ML-preferred layer
    """
    
    # Get available servers for ML-predicted layer
    ml_layer_servers = [s for s in servers.values() 
                        if s.layer == ml_prediction and s.is_available(current_time)]
    
    if ml_layer_servers:
        # ML layer has available server - follow ML recommendation
        server = ml_layer_servers[0]
        return (
            ml_prediction,
            server.server_id,
            f"✅ ML recommended {ml_prediction}, Server {server.server_id} available",
            DecisionType.ML_FOLLOWED
        )
    
    # ML layer is FULL - Agent needs to make smart decision
    # Fallback strategy based on requirements
    
    latency_req = network.get('latency_ms', 100)
    
    if ml_prediction == "Cloud":
        # Cloud busy → try Fog first (moderate latency), then Edge
        fallback_order = ["Fog", "Edge"]
    elif ml_prediction == "Edge":
        # Edge busy → try Fog (if latency allows), avoid Cloud
        if latency_req < 50:
            fallback_order = ["Fog"]  # Can't go to Cloud due to latency
        else:
            fallback_order = ["Fog", "Cloud"]
    else:  # Fog
        # Fog busy → check latency requirement
        if latency_req < 30:
            fallback_order = ["Edge"]
        else:
            fallback_order = ["Edge", "Cloud"]
    
    # Try fallback layers
    for fallback_layer in fallback_order:
        fallback_servers = [s for s in servers.values()
                          if s.layer == fallback_layer and s.is_available(current_time)]
        if fallback_servers:
            server = fallback_servers[0]
            return (
                fallback_layer,
                server.server_id,
                f"⚠️ ML: {ml_prediction} (BUSY) → Agent fallback to {fallback_layer} (Server {server.server_id})",
                DecisionType.FALLBACK_USED
            )
    
    # ALL servers busy - must queue on preferred layer
    # Find the server that will be free earliest
    ml_servers = [s for s in servers.values() if s.layer == ml_prediction]
    earliest = min(ml_servers, key=lambda s: s.busy_until or datetime.min)
    wait_time = (earliest.busy_until - current_time).total_seconds() if earliest.busy_until else 0
    
    return (
        ml_prediction,
        earliest.server_id,
        f"⏳ All servers BUSY! Queued on {ml_prediction} Server {earliest.server_id} (wait ~{wait_time:.1f}s)",
        DecisionType.QUEUED
    )


def main():
    st.title("🤖 Agent-Based Real-Time Deployment")
    
    st.markdown("""
    ### How It Works:
    | Component | Role |
    |-----------|------|
    | **📊 ML Model** | Predicts optimal layer based on network conditions |
    | **🤖 Agent** | Makes REAL decision considering **available servers** |
    
    **Agent Logic:**
    - IF ML layer has available server → Follow ML
    - IF ML layer BUSY → Find intelligent fallback
    - IF all BUSY → Queue on ML-preferred layer
    """)
    
    # Load data
    df = load_csv()
    if df is None:
        st.error("CSV not found!")
        return
    
    st.success(f"✅ Loaded {len(df)} network conditions")
    
    # Sidebar
    st.sidebar.header("🎛️ Controls")
    num_tasks = st.sidebar.slider("Tasks to Simulate", 5, 30, 10)
    speed = st.sidebar.slider("Speed Multiplier", 1, 10, 4)
    
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
                    "original_ml": df.iloc[i]['assigned_layer'],
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
    ml_followed = len([t for t in st.session_state.tasks if t.decision_type == DecisionType.ML_FOLLOWED])
    fallback = len([t for t in st.session_state.tasks if t.decision_type == DecisionType.FALLBACK_USED])
    
    mc1.metric("✅ Done", completed)
    mc2.metric("🔄 Running", running)
    mc3.metric("⏳ Waiting", waiting)
    mc4.metric("📊 ML Followed", ml_followed)
    mc5.metric("⚠️ Fallback", fallback)
    
    if st.session_state.task_data:
        st.progress(st.session_state.idx / len(st.session_state.task_data))
    
    st.divider()
    
    # Main visualization
    col_left, col_right = st.columns([3, 2])
    
    with col_left:
        st.subheader("📡 Network Condition → Decision Flow")
        flow_placeholder = st.empty()
    
    with col_right:
        st.subheader("🖥️ Server Status (Real-Time)")
        server_placeholder = st.empty()
    
    st.divider()
    
    # Task queue
    st.subheader("📋 Task Processing")
    queue_placeholder = st.empty()
    
    # Decision log
    st.subheader("📜 Decision History")
    log_placeholder = st.empty()
    
    # Simulation
    if st.session_state.running and st.session_state.task_data:
        
        while st.session_state.idx < len(st.session_state.task_data):
            current_time = datetime.now()
            
            # 1. Update completed tasks
            for task in st.session_state.tasks:
                if task.status == TaskStatus.RUNNING and task.end_time:
                    if current_time >= task.end_time:
                        task.status = TaskStatus.COMPLETED
                        st.session_state.servers[task.agent_server].release()
            
            # 2. Start waiting tasks if server available
            for task in st.session_state.tasks:
                if task.status == TaskStatus.WAITING:
                    server = st.session_state.servers[task.agent_server]
                    if server.is_available(current_time):
                        task.start_time = current_time
                        task.end_time = current_time + timedelta(
                            seconds=actual_completion[task.agent_decision]
                        )
                        task.wait_time = (current_time - task.arrival_time).total_seconds()
                        task.status = TaskStatus.RUNNING
                        server.assign_task(
                            task.task_id,
                            actual_completion[task.agent_decision],
                            current_time
                        )
            
            # 3. New task arrives
            network = st.session_state.task_data[st.session_state.idx]
            task_id = st.session_state.idx + 1
            
            # ML Model prediction
            ml_layer, ml_reason = get_ml_prediction(network)
            
            # Agent decision (considering availability)
            agent_layer, agent_server, agent_reason, decision_type = agent_decide(
                ml_layer,
                st.session_state.servers,
                network,
                current_time
            )
            
            # Show decision flow
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
                
                # ML vs Agent
                dc1, dc2 = st.columns(2)
                
                with dc1:
                    st.markdown("### 📊 ML Model Prediction")
                    if ml_layer == "Edge":
                        st.success(f"**{ml_layer}**")
                    elif ml_layer == "Fog":
                        st.warning(f"**{ml_layer}**")
                    else:
                        st.info(f"**{ml_layer}**")
                    st.caption(ml_reason)
                
                with dc2:
                    st.markdown("### 🤖 Agent Decision")
                    if decision_type == DecisionType.ML_FOLLOWED:
                        st.success(f"**{agent_layer}** → Server {agent_server}")
                        st.caption("✅ Following ML recommendation")
                    elif decision_type == DecisionType.FALLBACK_USED:
                        st.warning(f"**{agent_layer}** → Server {agent_server}")
                        st.caption(f"⚠️ {ml_layer} busy, using fallback")
                    else:
                        st.error(f"**{agent_layer}** → Server {agent_server}")
                        st.caption("⏳ Queued - all servers busy")
                
                st.markdown("---")
                st.markdown(f"**Agent Reasoning:** {agent_reason}")
            
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
                            st.caption(f"Done: {server.tasks_completed}")
            
            # Create task
            server = st.session_state.servers[agent_server]
            
            if server.is_available(current_time):
                status = TaskStatus.RUNNING
                start_time = current_time
                end_time = current_time + timedelta(seconds=actual_completion[agent_layer])
                server.assign_task(task_id, actual_completion[agent_layer], current_time)
            else:
                status = TaskStatus.WAITING
                start_time = None
                end_time = None
            
            task = Task(
                task_id=task_id,
                arrival_time=current_time,
                network_condition=network,
                ml_prediction=ml_layer,
                ml_server=network.get('original_server', 0),
                agent_decision=agent_layer,
                agent_server=agent_server,
                agent_reason=agent_reason,
                decision_type=decision_type,
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
                    st.markdown("**⏳ Waiting (Server Busy):**")
                    for t in waiting_tasks:
                        wait = (current_time - t.arrival_time).total_seconds()
                        st.warning(f"Task {t.task_id}: Waiting for {t.agent_decision} Server {t.agent_server} ({wait:.1f}s)")
                
                if running_tasks:
                    st.markdown("**🔄 Currently Running:**")
                    for t in running_tasks:
                        elapsed = (current_time - t.start_time).total_seconds()
                        total = (t.end_time - t.start_time).total_seconds()
                        progress = min(elapsed / total, 1.0)
                        remaining = max(0, total - elapsed)
                        
                        rc1, rc2, rc3 = st.columns([2, 4, 2])
                        rc1.write(f"Task {t.task_id} ({t.agent_decision})")
                        rc2.progress(progress)
                        rc3.write(f"{remaining:.1f}s")
            
            # Update log
            with log_placeholder.container():
                log_data = []
                for t in st.session_state.tasks[-10:]:
                    log_data.append({
                        "Task": t.task_id,
                        "ML Prediction": t.ml_prediction,
                        "Agent Decision": t.agent_decision,
                        "Server": t.agent_server,
                        "Type": t.decision_type.value,
                        "Status": t.status.value
                    })
                if log_data:
                    st.dataframe(pd.DataFrame(log_data), use_container_width=True, hide_index=True)
            
            # Wait for next task
            time.sleep(actual_arrival)
        
        # Wait for remaining tasks to complete
        while any(t.status != TaskStatus.COMPLETED for t in st.session_state.tasks):
            current_time = datetime.now()
            
            for task in st.session_state.tasks:
                if task.status == TaskStatus.RUNNING and task.end_time:
                    if current_time >= task.end_time:
                        task.status = TaskStatus.COMPLETED
                        st.session_state.servers[task.agent_server].release()
                
                if task.status == TaskStatus.WAITING:
                    server = st.session_state.servers[task.agent_server]
                    if server.is_available(current_time):
                        task.start_time = current_time
                        task.end_time = current_time + timedelta(
                            seconds=actual_completion[task.agent_decision]
                        )
                        task.wait_time = (current_time - task.arrival_time).total_seconds()
                        task.status = TaskStatus.RUNNING
                        server.assign_task(
                            task.task_id,
                            actual_completion[task.agent_decision],
                            current_time
                        )
            
            time.sleep(0.3)
        
        st.session_state.running = False
        st.balloons()
        
        # Final stats
        st.header("📊 Final Analysis")
        
        fc1, fc2, fc3 = st.columns(3)
        
        with fc1:
            ml_same = sum(1 for t in st.session_state.tasks 
                         if t.ml_prediction == t.agent_decision)
            st.metric("Agent=ML Decisions", f"{ml_same}/{len(st.session_state.tasks)}")
            st.metric("Accuracy", f"{ml_same/len(st.session_state.tasks)*100:.1f}%")
        
        with fc2:
            fallbacks = [t for t in st.session_state.tasks 
                        if t.decision_type == DecisionType.FALLBACK_USED]
            st.metric("Fallback Decisions", len(fallbacks))
            if fallbacks:
                st.caption("Agent chose different layer due to busy servers")
        
        with fc3:
            queued = [t for t in st.session_state.tasks
                     if t.decision_type == DecisionType.QUEUED]
            avg_wait = sum(t.wait_time for t in queued) / len(queued) if queued else 0
            st.metric("Queued Tasks", len(queued))
            st.metric("Avg Wait Time", f"{avg_wait:.1f}s")


if __name__ == "__main__":
    main()
