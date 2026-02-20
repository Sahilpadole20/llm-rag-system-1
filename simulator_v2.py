"""
Real-Time Task Simulator v2
============================
Realistic simulation with proper server queuing:
- Each server processes ONE task at a time
- Cloud (1 server) = tasks wait in queue
- Fog (2 servers) = can process 2 tasks simultaneously  
- Edge (4 servers) = can process 4 tasks simultaneously
- Tasks arrive every 4 seconds
- Completion times: Edge=3s, Fog=6s, Cloud=10s

Run: streamlit run simulator_v2.py
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
from collections import deque
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum

# Page config
st.set_page_config(
    page_title="Real-Time Task Simulator v2",
    page_icon="🎮",
    layout="wide"
)

# Constants
TASK_ARRIVAL_INTERVAL = 4  # seconds
COMPLETION_TIMES = {
    "Edge": 3,   # seconds
    "Fog": 6,    # seconds  
    "Cloud": 10  # seconds
}

# Server configuration - each server processes 1 task at a time
SERVER_CONFIG = {
    1: {"layer": "Edge", "capacity": 1},
    2: {"layer": "Edge", "capacity": 1},
    3: {"layer": "Edge", "capacity": 1},
    4: {"layer": "Edge", "capacity": 1},
    5: {"layer": "Fog", "capacity": 1},
    6: {"layer": "Fog", "capacity": 1},
    7: {"layer": "Cloud", "capacity": 1},
}

# CSV path
BASE_DIR = Path(__file__).parent
CSV_PATH = BASE_DIR / "data" / "simulation_data.csv"

if not CSV_PATH.exists():
    CSV_PATH = Path(r"c:\Users\Sahil Padole\Videos\AI_agent_ml_threshold\data\edgesimpy_failure_ml_+_thresh_(gb)_no_failure_20251223_075347_results.csv")


class TaskStatus(Enum):
    WAITING = "waiting"      # In queue, waiting for server
    RUNNING = "running"      # Currently processing
    COMPLETED = "completed"  # Done


@dataclass
class ServerState:
    """Track server state."""
    server_id: int
    layer: str
    current_task: Optional[int] = None  # Task ID
    busy_until: Optional[datetime] = None
    tasks_completed: int = 0
    
    def is_busy(self, current_time: datetime) -> bool:
        if self.busy_until is None:
            return False
        return current_time < self.busy_until
    
    def assign_task(self, task_id: int, duration_seconds: float, current_time: datetime):
        self.current_task = task_id
        self.busy_until = current_time + timedelta(seconds=duration_seconds)
    
    def complete_task(self):
        self.current_task = None
        self.busy_until = None
        self.tasks_completed += 1


@dataclass
class SimulatedTask:
    """A simulated task."""
    task_id: int
    arrival_time: datetime
    network_condition: Dict
    ml_prediction: str
    agent_decision: str
    assigned_server: int
    completion_time_seconds: float
    status: TaskStatus = TaskStatus.WAITING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    wait_time_seconds: float = 0.0


def load_csv_data():
    """Load CSV data."""
    if CSV_PATH.exists():
        return pd.read_csv(CSV_PATH)
    return None


def get_random_tasks(df: pd.DataFrame, num_tasks: int) -> List[Dict]:
    """Get random tasks from CSV."""
    indices = random.sample(range(len(df)), min(num_tasks, len(df)))
    tasks = []
    for idx in indices:
        row = df.iloc[idx]
        tasks.append({
            "step": int(row.get('step', idx)),
            "datarate_mbps": row['datarate'] / 1e6,
            "sinr": row['sinr'],
            "latency_ms": row['latency_ms'],
            "rsrp_dbm": row['rsrp_dbm'],
            "cpu_demand": row['cpu_demand'],
            "memory_demand": row['memory_demand'],
            "ml_prediction": row['assigned_layer'],
            "server_id": int(row['server_id'])
        })
    return tasks


def make_agent_decision(network: Dict) -> tuple:
    """Agent makes decision based on network conditions."""
    datarate = network['datarate_mbps']
    sinr = network['sinr']
    latency = network['latency_ms']
    
    if latency < 20 and datarate < 16.6:
        layer = "Edge"
        reason = f"Low latency ({latency:.1f}ms) requires Edge"
    elif 9.64 <= datarate <= 16.6 and sinr > 10:
        layer = "Fog"
        reason = f"Medium bandwidth ({datarate:.1f}Mbps) + Good SINR"
    else:
        layer = "Cloud"
        reason = f"High bandwidth ({datarate:.1f}Mbps) or Poor SINR"
    
    return layer, reason


def find_available_server(layer: str, servers: Dict[int, ServerState], current_time: datetime) -> Optional[int]:
    """Find an available server for the given layer."""
    layer_servers = [s for s in servers.values() if s.layer == layer]
    
    # Find first available server
    for server in layer_servers:
        if not server.is_busy(current_time):
            return server.server_id
    
    return None  # All servers busy


def find_earliest_available_server(layer: str, servers: Dict[int, ServerState]) -> tuple:
    """Find server that will be available earliest."""
    layer_servers = [s for s in servers.values() if s.layer == layer]
    
    earliest_server = min(layer_servers, key=lambda s: s.busy_until or datetime.min)
    return earliest_server.server_id, earliest_server.busy_until


def create_server_status_chart(servers: Dict[int, ServerState], current_time: datetime):
    """Create server status visualization."""
    data = []
    for server_id, server in servers.items():
        status = "🔴 Busy" if server.is_busy(current_time) else "🟢 Free"
        task_info = f"Task {server.current_task}" if server.current_task else "None"
        data.append({
            "Server": f"Server {server_id}",
            "Layer": server.layer,
            "Status": status,
            "Current Task": task_info,
            "Completed": server.tasks_completed
        })
    return pd.DataFrame(data)


def main():
    st.title("🎮 Real-Time Task Simulator v2")
    st.markdown("""
    **Realistic server queuing:**
    - Each server processes **ONE task at a time**
    - Cloud (Server 7): **1 server** → Tasks wait in queue
    - Fog (Server 5-6): **2 servers** → Max 2 parallel tasks
    - Edge (Server 1-4): **4 servers** → Max 4 parallel tasks
    """)
    
    # Load data
    df = load_csv_data()
    if df is None:
        st.error("CSV file not found!")
        return
    
    st.success(f"✅ Loaded {len(df)} network conditions")
    
    # Sidebar controls
    st.sidebar.header("🎛️ Simulation Controls")
    
    num_tasks = st.sidebar.slider("Number of Tasks", 5, 50, 15, 5)
    speed_multiplier = st.sidebar.slider("Speed Multiplier", 1, 10, 3)
    
    # Actual timing
    actual_arrival = TASK_ARRIVAL_INTERVAL / speed_multiplier
    actual_completion = {k: v / speed_multiplier for k, v in COMPLETION_TIMES.items()}
    
    st.sidebar.markdown(f"""
    **Timing ({speed_multiplier}x speed):**
    - Task Arrival: {actual_arrival:.1f}s
    - Edge: {actual_completion['Edge']:.1f}s
    - Fog: {actual_completion['Fog']:.1f}s
    - Cloud: {actual_completion['Cloud']:.1f}s
    """)
    
    st.sidebar.markdown("""
    **Server Capacity:**
    - Edge: 4 servers (4 parallel)
    - Fog: 2 servers (2 parallel)
    - Cloud: 1 server (**queue!**)
    """)
    
    # Initialize session state
    if 'sim_running' not in st.session_state:
        st.session_state.sim_running = False
    if 'tasks' not in st.session_state:
        st.session_state.tasks = []
    if 'task_queue' not in st.session_state:
        st.session_state.task_queue = []
    if 'servers' not in st.session_state:
        st.session_state.servers = {
            sid: ServerState(server_id=sid, layer=cfg["layer"])
            for sid, cfg in SERVER_CONFIG.items()
        }
    if 'current_idx' not in st.session_state:
        st.session_state.current_idx = 0
    if 'waiting_queue' not in st.session_state:
        st.session_state.waiting_queue = []  # Tasks waiting for servers
    
    # Control buttons
    col1, col2, col3 = st.sidebar.columns(3)
    with col1:
        start = st.button("▶️ Start", type="primary")
    with col2:
        stop = st.button("⏹️ Stop")
    with col3:
        reset = st.button("🔄 Reset")
    
    if reset:
        st.session_state.tasks = []
        st.session_state.task_queue = get_random_tasks(df, num_tasks)
        st.session_state.current_idx = 0
        st.session_state.sim_running = False
        st.session_state.waiting_queue = []
        st.session_state.servers = {
            sid: ServerState(server_id=sid, layer=cfg["layer"])
            for sid, cfg in SERVER_CONFIG.items()
        }
        st.rerun()
    
    if stop:
        st.session_state.sim_running = False
    
    if start:
        st.session_state.sim_running = True
        if not st.session_state.task_queue:
            st.session_state.task_queue = get_random_tasks(df, num_tasks)
    
    # Main display
    st.divider()
    
    # Progress metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        completed = len([t for t in st.session_state.tasks if t.status == TaskStatus.COMPLETED])
        st.metric("✅ Completed", completed)
    with col2:
        running = len([t for t in st.session_state.tasks if t.status == TaskStatus.RUNNING])
        st.metric("🔄 Running", running)
    with col3:
        waiting = len([t for t in st.session_state.tasks if t.status == TaskStatus.WAITING])
        st.metric("⏳ Waiting", waiting)
    with col4:
        pending = len(st.session_state.task_queue) - st.session_state.current_idx
        st.metric("📋 Pending", pending)
    with col5:
        total = len(st.session_state.task_queue) if st.session_state.task_queue else num_tasks
        st.metric("📊 Progress", f"{completed}/{total}")
    
    if st.session_state.task_queue:
        progress = st.session_state.current_idx / len(st.session_state.task_queue)
        st.progress(progress)
    
    st.divider()
    
    # Two columns: Network + Decision
    col_net, col_dec = st.columns(2)
    
    with col_net:
        st.subheader("📡 Step 1: Network Condition")
        network_placeholder = st.empty()
    
    with col_dec:
        st.subheader("🤖 Step 2: Agent Decision")
        decision_placeholder = st.empty()
    
    st.divider()
    
    # Server status
    st.subheader("🖥️ Server Status (Real-Time)")
    server_placeholder = st.empty()
    
    # Queue visualization
    st.subheader("📋 Task Queue & Processing")
    queue_placeholder = st.empty()
    
    # Completed tasks
    st.subheader("✅ Completed Tasks")
    completed_placeholder = st.empty()
    
    # Simulation loop
    if st.session_state.sim_running and st.session_state.task_queue:
        
        while (st.session_state.current_idx < len(st.session_state.task_queue) or 
               any(t.status != TaskStatus.COMPLETED for t in st.session_state.tasks)):
            
            current_time = datetime.now()
            
            # 1. Check for completed tasks
            for task in st.session_state.tasks:
                if task.status == TaskStatus.RUNNING and task.end_time:
                    if current_time >= task.end_time:
                        task.status = TaskStatus.COMPLETED
                        # Free the server
                        st.session_state.servers[task.assigned_server].complete_task()
            
            # 2. Process waiting queue - assign to available servers
            new_waiting_queue = []
            for task in st.session_state.tasks:
                if task.status == TaskStatus.WAITING:
                    server_id = find_available_server(
                        task.agent_decision, 
                        st.session_state.servers, 
                        current_time
                    )
                    if server_id:
                        # Server available - start task
                        task.assigned_server = server_id
                        task.start_time = current_time
                        task.end_time = current_time + timedelta(seconds=actual_completion[task.agent_decision])
                        task.wait_time_seconds = (current_time - task.arrival_time).total_seconds()
                        task.status = TaskStatus.RUNNING
                        
                        # Mark server busy
                        st.session_state.servers[server_id].assign_task(
                            task.task_id,
                            actual_completion[task.agent_decision],
                            current_time
                        )
            
            # 3. Bring in new task if it's time
            if st.session_state.current_idx < len(st.session_state.task_queue):
                task_data = st.session_state.task_queue[st.session_state.current_idx]
                task_id = st.session_state.current_idx + 1
                
                # Show network condition
                with network_placeholder.container():
                    st.markdown(f"### 🎯 Task {task_id}")
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("📶 Data Rate", f"{task_data['datarate_mbps']:.1f} Mbps")
                    c2.metric("📊 SINR", f"{task_data['sinr']:.1f} dB")
                    c3.metric("⏱️ Latency", f"{task_data['latency_ms']:.1f} ms")
                    c4.metric("📡 RSRP", f"{task_data['rsrp_dbm']:.1f} dBm")
                
                # Agent decision
                agent_layer, reason = make_agent_decision(task_data)
                ml_layer = task_data['ml_prediction']
                
                # Check server availability
                available_server = find_available_server(
                    agent_layer, 
                    st.session_state.servers, 
                    current_time
                )
                
                with decision_placeholder.container():
                    st.markdown(f"### 🎯 Task {task_id} Assignment")
                    
                    dc1, dc2 = st.columns(2)
                    with dc1:
                        st.markdown("**🤖 Agent Decision**")
                        if agent_layer == "Edge":
                            st.success(f"Layer: **{agent_layer}**")
                        elif agent_layer == "Fog":
                            st.warning(f"Layer: **{agent_layer}**")
                        else:
                            st.info(f"Layer: **{agent_layer}**")
                        st.caption(reason)
                    
                    with dc2:
                        st.markdown("**🧠 ML Prediction**")
                        if ml_layer == "Edge":
                            st.success(f"Layer: **{ml_layer}**")
                        elif ml_layer == "Fog":
                            st.warning(f"Layer: **{ml_layer}**")
                        else:
                            st.info(f"Layer: **{ml_layer}**")
                    
                    if available_server:
                        st.success(f"✅ Server {available_server} available - Starting immediately!")
                    else:
                        server_id, available_at = find_earliest_available_server(
                            agent_layer, st.session_state.servers
                        )
                        wait_seconds = (available_at - current_time).total_seconds() if available_at else 0
                        st.warning(f"⏳ All {agent_layer} servers busy! Task queued. Wait ~{wait_seconds:.1f}s for Server {server_id}")
                
                # Create task
                task = SimulatedTask(
                    task_id=task_id,
                    arrival_time=current_time,
                    network_condition=task_data,
                    ml_prediction=ml_layer,
                    agent_decision=agent_layer,
                    assigned_server=available_server or 0,
                    completion_time_seconds=COMPLETION_TIMES[agent_layer],
                    status=TaskStatus.WAITING
                )
                
                # If server available, start immediately
                if available_server:
                    task.assigned_server = available_server
                    task.start_time = current_time
                    task.end_time = current_time + timedelta(seconds=actual_completion[agent_layer])
                    task.status = TaskStatus.RUNNING
                    
                    st.session_state.servers[available_server].assign_task(
                        task_id,
                        actual_completion[agent_layer],
                        current_time
                    )
                
                st.session_state.tasks.append(task)
                st.session_state.current_idx += 1
            
            # 4. Update server status display
            with server_placeholder.container():
                server_df = create_server_status_chart(st.session_state.servers, current_time)
                
                # Group by layer
                for layer in ["Edge", "Fog", "Cloud"]:
                    layer_servers = server_df[server_df["Layer"] == layer]
                    if layer == "Edge":
                        icon = "🟢"
                    elif layer == "Fog":
                        icon = "🟡"
                    else:
                        icon = "🔵"
                    
                    st.markdown(f"**{icon} {layer} Layer**")
                    cols = st.columns(len(layer_servers))
                    for i, (_, row) in enumerate(layer_servers.iterrows()):
                        with cols[i]:
                            st.markdown(f"**{row['Server']}**")
                            st.markdown(row['Status'])
                            st.caption(f"Task: {row['Current Task']}")
                            st.caption(f"Done: {row['Completed']}")
            
            # 5. Update queue display
            with queue_placeholder.container():
                running_tasks = [t for t in st.session_state.tasks if t.status == TaskStatus.RUNNING]
                waiting_tasks = [t for t in st.session_state.tasks if t.status == TaskStatus.WAITING]
                
                if waiting_tasks:
                    st.markdown("**⏳ Waiting in Queue:**")
                    for t in waiting_tasks:
                        wait_time = (current_time - t.arrival_time).total_seconds()
                        st.warning(f"Task {t.task_id} ({t.agent_decision}) - Waiting {wait_time:.1f}s")
                
                if running_tasks:
                    st.markdown("**🔄 Currently Running:**")
                    for t in running_tasks:
                        elapsed = (current_time - t.start_time).total_seconds()
                        total = (t.end_time - t.start_time).total_seconds()
                        progress = min(elapsed / total, 1.0) if total > 0 else 0
                        remaining = max(0, total - elapsed)
                        
                        col1, col2, col3 = st.columns([2, 4, 2])
                        with col1:
                            st.write(f"Task {t.task_id} ({t.agent_decision}/S{t.assigned_server})")
                        with col2:
                            st.progress(progress)
                        with col3:
                            st.write(f"{remaining:.1f}s left")
            
            # 6. Update completed tasks
            with completed_placeholder.container():
                completed_tasks = [t for t in st.session_state.tasks if t.status == TaskStatus.COMPLETED]
                if completed_tasks:
                    data = []
                    for t in completed_tasks[-15:]:  # Last 15
                        data.append({
                            "Task": t.task_id,
                            "Layer": t.agent_decision,
                            "Server": t.assigned_server,
                            "Process Time": f"{t.completion_time_seconds}s",
                            "Wait Time": f"{t.wait_time_seconds:.1f}s",
                            "Status": "✅"
                        })
                    st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)
            
            # Check if all done
            if (st.session_state.current_idx >= len(st.session_state.task_queue) and
                all(t.status == TaskStatus.COMPLETED for t in st.session_state.tasks)):
                break
            
            # Wait for next cycle
            time.sleep(0.5)
        
        # Simulation complete
        st.session_state.sim_running = False
        st.balloons()
        st.success("🎉 Simulation Complete!")
        
        # Final statistics
        st.header("📊 Final Statistics")
        
        col1, col2, col3 = st.columns(3)
        
        completed_tasks = [t for t in st.session_state.tasks if t.status == TaskStatus.COMPLETED]
        
        with col1:
            cloud_tasks = [t for t in completed_tasks if t.agent_decision == "Cloud"]
            cloud_wait = sum(t.wait_time_seconds for t in cloud_tasks) / len(cloud_tasks) if cloud_tasks else 0
            st.metric("🔵 Cloud Tasks", len(cloud_tasks))
            st.metric("Avg Wait Time", f"{cloud_wait:.1f}s")
        
        with col2:
            fog_tasks = [t for t in completed_tasks if t.agent_decision == "Fog"]
            fog_wait = sum(t.wait_time_seconds for t in fog_tasks) / len(fog_tasks) if fog_tasks else 0
            st.metric("🟡 Fog Tasks", len(fog_tasks))
            st.metric("Avg Wait Time", f"{fog_wait:.1f}s")
        
        with col3:
            edge_tasks = [t for t in completed_tasks if t.agent_decision == "Edge"]
            edge_wait = sum(t.wait_time_seconds for t in edge_tasks) / len(edge_tasks) if edge_tasks else 0
            st.metric("🟢 Edge Tasks", len(edge_tasks))
            st.metric("Avg Wait Time", f"{edge_wait:.1f}s")


if __name__ == "__main__":
    main()
