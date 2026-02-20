"""
Real-Time Task Simulator
========================
Realistic simulation with:
- Tasks arriving every 4 seconds
- Completion times: Edge=3s, Fog=6s, Cloud=10s
- Real-time visualization of network conditions and decisions
- Manual selection of number of tasks

Run: streamlit run simulator_app.py
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
from plotly.subplots import make_subplots
from collections import deque
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum

# Page config
st.set_page_config(
    page_title="Real-Time Task Simulator",
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

CSV_PATH = Path(r"c:\Users\Sahil Padole\Videos\AI_agent_ml_threshold\data\edgesimpy_failure_ml_+_thresh_(gb)_no_failure_20251223_075347_results.csv")


class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"


@dataclass
class SimulatedTask:
    """A simulated task with network conditions."""
    task_id: int
    arrival_time: datetime
    network_condition: Dict
    ml_prediction: str
    agent_decision: str
    assigned_server: int
    completion_time_seconds: int
    status: TaskStatus = TaskStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    progress: float = 0.0


def load_csv_data():
    """Load CSV data."""
    if CSV_PATH.exists():
        df = pd.read_csv(CSV_PATH)
        return df
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
    
    # Decision rules
    if latency < 20 and datarate < 16.6:
        layer = "Edge"
        server = random.choice([1, 2, 3, 4])
        reason = f"Low latency ({latency:.1f}ms) requires Edge"
    elif 9.64 <= datarate <= 16.6 and sinr > 10:
        layer = "Fog"
        server = random.choice([5, 6])
        reason = f"Medium bandwidth ({datarate:.1f}Mbps) + Good SINR"
    else:
        layer = "Cloud"
        server = 7
        reason = f"High bandwidth ({datarate:.1f}Mbps) or Poor SINR"
    
    return layer, server, reason


def create_network_gauge(value: float, title: str, min_val: float, max_val: float, color: str):
    """Create a gauge chart for network metric."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title, 'font': {'size': 14}},
        gauge={
            'axis': {'range': [min_val, max_val]},
            'bar': {'color': color},
            'steps': [
                {'range': [min_val, (max_val-min_val)*0.33+min_val], 'color': "lightgray"},
                {'range': [(max_val-min_val)*0.33+min_val, (max_val-min_val)*0.66+min_val], 'color': "gray"},
            ],
        }
    ))
    fig.update_layout(height=150, margin=dict(l=20, r=20, t=40, b=20))
    return fig


def create_task_timeline(tasks: List[SimulatedTask]):
    """Create timeline visualization of tasks."""
    if not tasks:
        return None
    
    data = []
    for task in tasks:
        if task.start_time:
            data.append({
                'Task': f'Task {task.task_id}',
                'Start': task.start_time,
                'End': task.end_time or datetime.now(),
                'Layer': task.agent_decision,
                'Status': task.status.value,
                'Server': f'Server {task.assigned_server}'
            })
    
    if not data:
        return None
    
    df = pd.DataFrame(data)
    
    color_map = {
        'Edge': '#00CC00',
        'Fog': '#FFD700', 
        'Cloud': '#1E90FF'
    }
    
    fig = px.timeline(
        df, x_start='Start', x_end='End', y='Task',
        color='Layer', color_discrete_map=color_map,
        hover_data=['Server', 'Status']
    )
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=30, b=20))
    return fig


def main():
    st.title("🎮 Real-Time Task Simulator")
    st.markdown("""
    **Realistic simulation showing:**
    - Network conditions from CSV data
    - Agent & ML model decisions in real-time
    - Task completion: Edge=3s, Fog=6s, Cloud=10s
    - Tasks arrive every 4 seconds
    """)
    
    # Load data
    df = load_csv_data()
    if df is None:
        st.error("CSV file not found!")
        return
    
    st.success(f"✅ Loaded {len(df)} network conditions from CSV")
    
    # Sidebar controls
    st.sidebar.header("🎛️ Simulation Controls")
    
    num_tasks = st.sidebar.slider(
        "Number of Tasks to Simulate",
        min_value=5,
        max_value=100,
        value=20,
        step=5
    )
    
    tasks_per_batch = st.sidebar.slider(
        "Random Tasks per Batch (1-4)",
        min_value=1,
        max_value=4,
        value=1
    )
    
    speed_multiplier = st.sidebar.slider(
        "Speed Multiplier",
        min_value=1,
        max_value=10,
        value=2,
        help="Increase to speed up simulation"
    )
    
    # Calculate actual times
    actual_arrival = TASK_ARRIVAL_INTERVAL / speed_multiplier
    actual_completion = {k: v / speed_multiplier for k, v in COMPLETION_TIMES.items()}
    
    st.sidebar.markdown(f"""
    **Timing (with {speed_multiplier}x speed):**
    - Task Arrival: {actual_arrival:.1f}s
    - Edge Completion: {actual_completion['Edge']:.1f}s
    - Fog Completion: {actual_completion['Fog']:.1f}s
    - Cloud Completion: {actual_completion['Cloud']:.1f}s
    """)
    
    # Initialize session state
    if 'simulation_running' not in st.session_state:
        st.session_state.simulation_running = False
    if 'tasks' not in st.session_state:
        st.session_state.tasks = []
    if 'current_task_idx' not in st.session_state:
        st.session_state.current_task_idx = 0
    if 'task_queue' not in st.session_state:
        st.session_state.task_queue = []
    
    # Control buttons
    col_btn1, col_btn2, col_btn3 = st.sidebar.columns(3)
    
    with col_btn1:
        start_btn = st.button("▶️ Start", type="primary", use_container_width=True)
    with col_btn2:
        stop_btn = st.button("⏹️ Stop", use_container_width=True)
    with col_btn3:
        reset_btn = st.button("🔄 Reset", use_container_width=True)
    
    if reset_btn:
        st.session_state.tasks = []
        st.session_state.current_task_idx = 0
        st.session_state.simulation_running = False
        st.session_state.task_queue = get_random_tasks(df, num_tasks)
        st.rerun()
    
    if stop_btn:
        st.session_state.simulation_running = False
    
    if start_btn:
        st.session_state.simulation_running = True
        if not st.session_state.task_queue:
            st.session_state.task_queue = get_random_tasks(df, num_tasks)
    
    # Main visualization area
    st.divider()
    
    # Progress
    progress_col1, progress_col2, progress_col3, progress_col4 = st.columns(4)
    with progress_col1:
        st.metric("Tasks Completed", 
                  len([t for t in st.session_state.tasks if t.status == TaskStatus.COMPLETED]))
    with progress_col2:
        st.metric("Tasks Running", 
                  len([t for t in st.session_state.tasks if t.status == TaskStatus.RUNNING]))
    with progress_col3:
        st.metric("Tasks Pending", 
                  len(st.session_state.task_queue) - st.session_state.current_task_idx)
    with progress_col4:
        total = len(st.session_state.task_queue) if st.session_state.task_queue else num_tasks
        completed = len([t for t in st.session_state.tasks if t.status == TaskStatus.COMPLETED])
        st.metric("Progress", f"{completed}/{total}")
    
    # Overall progress bar
    if st.session_state.task_queue:
        progress = st.session_state.current_task_idx / len(st.session_state.task_queue)
        st.progress(progress, text=f"Simulation Progress: {progress*100:.0f}%")
    
    st.divider()
    
    # Current task visualization
    col_network, col_decision = st.columns(2)
    
    with col_network:
        st.subheader("📡 Step 1: Network Condition")
        network_placeholder = st.empty()
    
    with col_decision:
        st.subheader("🤖 Step 2: Agent Decision")
        decision_placeholder = st.empty()
    
    st.divider()
    
    # Timeline
    st.subheader("📊 Task Timeline")
    timeline_placeholder = st.empty()
    
    # Statistics
    st.subheader("📈 Layer Distribution")
    stats_placeholder = st.empty()
    
    # Running tasks display
    st.subheader("🔄 Currently Running Tasks")
    running_placeholder = st.empty()
    
    # Completed tasks log
    st.subheader("✅ Completed Tasks Log")
    log_placeholder = st.empty()
    
    # Simulation loop
    if st.session_state.simulation_running and st.session_state.task_queue:
        
        while (st.session_state.current_task_idx < len(st.session_state.task_queue) and 
               st.session_state.simulation_running):
            
            # Get next task from queue
            task_data = st.session_state.task_queue[st.session_state.current_task_idx]
            task_id = st.session_state.current_task_idx + 1
            
            # Step 1: Show network condition
            with network_placeholder.container():
                st.markdown(f"### 🎯 Task {task_id} Network Metrics")
                
                metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                
                with metric_col1:
                    st.metric("📶 Data Rate", f"{task_data['datarate_mbps']:.1f} Mbps")
                with metric_col2:
                    st.metric("📊 SINR", f"{task_data['sinr']:.1f} dB")
                with metric_col3:
                    st.metric("⏱️ Latency", f"{task_data['latency_ms']:.1f} ms")
                with metric_col4:
                    st.metric("📡 RSRP", f"{task_data['rsrp_dbm']:.1f} dBm")
                
                # Visual gauges
                gauge_col1, gauge_col2, gauge_col3 = st.columns(3)
                with gauge_col1:
                    fig = create_network_gauge(task_data['datarate_mbps'], "Data Rate (Mbps)", 0, 50, "blue")
                    st.plotly_chart(fig, use_container_width=True)
                with gauge_col2:
                    fig = create_network_gauge(task_data['sinr'], "SINR (dB)", 0, 25, "green")
                    st.plotly_chart(fig, use_container_width=True)
                with gauge_col3:
                    fig = create_network_gauge(task_data['latency_ms'], "Latency (ms)", 0, 300, "orange")
                    st.plotly_chart(fig, use_container_width=True)
            
            time.sleep(1 / speed_multiplier)  # Pause to show network condition
            
            # Step 2: Agent makes decision
            agent_layer, agent_server, reason = make_agent_decision(task_data)
            ml_layer = task_data['ml_prediction']
            ml_server = task_data['server_id']
            
            with decision_placeholder.container():
                st.markdown(f"### 🎯 Task {task_id} Deployment Decision")
                
                dec_col1, dec_col2 = st.columns(2)
                
                with dec_col1:
                    st.markdown("#### 🤖 Agent Decision")
                    if agent_layer == "Edge":
                        st.success(f"**Layer: {agent_layer}** (Server {agent_server})")
                    elif agent_layer == "Fog":
                        st.warning(f"**Layer: {agent_layer}** (Server {agent_server})")
                    else:
                        st.info(f"**Layer: {agent_layer}** (Server {agent_server})")
                    st.caption(f"Reason: {reason}")
                    st.metric("Completion Time", f"{COMPLETION_TIMES[agent_layer]}s")
                
                with dec_col2:
                    st.markdown("#### 🧠 ML Model Prediction")
                    if ml_layer == "Edge":
                        st.success(f"**Layer: {ml_layer}** (Server {ml_server})")
                    elif ml_layer == "Fog":
                        st.warning(f"**Layer: {ml_layer}** (Server {ml_server})")
                    else:
                        st.info(f"**Layer: {ml_layer}** (Server {ml_server})")
                    st.caption("Based on Gradient Boosting model")
                    st.metric("Completion Time", f"{COMPLETION_TIMES[ml_layer]}s")
                
                # Agreement indicator
                if agent_layer == ml_layer:
                    st.success("✅ Agent and ML model AGREE!")
                else:
                    st.warning(f"⚠️ Different decisions: Agent={agent_layer}, ML={ml_layer}")
            
            # Create task object
            now = datetime.now()
            task = SimulatedTask(
                task_id=task_id,
                arrival_time=now,
                network_condition=task_data,
                ml_prediction=ml_layer,
                agent_decision=agent_layer,
                assigned_server=agent_server,
                completion_time_seconds=COMPLETION_TIMES[agent_layer],
                status=TaskStatus.RUNNING,
                start_time=now,
                end_time=now + timedelta(seconds=actual_completion[agent_layer])
            )
            
            st.session_state.tasks.append(task)
            st.session_state.current_task_idx += 1
            
            # Update running tasks
            with running_placeholder.container():
                running_tasks = [t for t in st.session_state.tasks if t.status == TaskStatus.RUNNING]
                if running_tasks:
                    for t in running_tasks:
                        elapsed = (datetime.now() - t.start_time).total_seconds()
                        progress = min(elapsed / (t.completion_time_seconds / speed_multiplier), 1.0)
                        
                        col1, col2, col3 = st.columns([2, 4, 2])
                        with col1:
                            st.write(f"Task {t.task_id} ({t.agent_decision})")
                        with col2:
                            st.progress(progress)
                        with col3:
                            remaining = max(0, (t.completion_time_seconds / speed_multiplier) - elapsed)
                            st.write(f"{remaining:.1f}s left")
            
            # Update timeline
            fig = create_task_timeline(st.session_state.tasks)
            if fig:
                timeline_placeholder.plotly_chart(fig, use_container_width=True)
            
            # Update statistics
            with stats_placeholder.container():
                completed_tasks = [t for t in st.session_state.tasks]
                if completed_tasks:
                    layer_counts = pd.DataFrame([{'Layer': t.agent_decision} for t in completed_tasks])
                    fig = px.pie(layer_counts, names='Layer', 
                                 color='Layer',
                                 color_discrete_map={'Edge': '#00CC00', 'Fog': '#FFD700', 'Cloud': '#1E90FF'})
                    fig.update_layout(height=250)
                    st.plotly_chart(fig, use_container_width=True)
            
            # Check for completed tasks
            for t in st.session_state.tasks:
                if t.status == TaskStatus.RUNNING:
                    elapsed = (datetime.now() - t.start_time).total_seconds()
                    if elapsed >= (t.completion_time_seconds / speed_multiplier):
                        t.status = TaskStatus.COMPLETED
                        t.end_time = datetime.now()
            
            # Update completed log
            with log_placeholder.container():
                completed = [t for t in st.session_state.tasks if t.status == TaskStatus.COMPLETED]
                if completed:
                    log_data = []
                    for t in completed[-10:]:  # Last 10
                        log_data.append({
                            'Task': t.task_id,
                            'Layer': t.agent_decision,
                            'Server': t.assigned_server,
                            'Time': f"{t.completion_time_seconds}s",
                            'Status': '✅'
                        })
                    st.dataframe(pd.DataFrame(log_data), use_container_width=True, hide_index=True)
            
            # Wait for next task arrival
            time.sleep(actual_arrival)
        
        # Simulation complete
        st.session_state.simulation_running = False
        st.balloons()
        st.success("🎉 Simulation Complete!")
    
    # Show final statistics if tasks exist
    if st.session_state.tasks and not st.session_state.simulation_running:
        st.divider()
        st.header("📊 Final Statistics")
        
        col1, col2, col3 = st.columns(3)
        
        completed = [t for t in st.session_state.tasks if t.status == TaskStatus.COMPLETED]
        
        with col1:
            edge_count = len([t for t in completed if t.agent_decision == "Edge"])
            st.metric("🟢 Edge Tasks", edge_count)
        
        with col2:
            fog_count = len([t for t in completed if t.agent_decision == "Fog"])
            st.metric("🟡 Fog Tasks", fog_count)
        
        with col3:
            cloud_count = len([t for t in completed if t.agent_decision == "Cloud"])
            st.metric("🔵 Cloud Tasks", cloud_count)
        
        # Agreement rate
        agreement = len([t for t in st.session_state.tasks if t.agent_decision == t.ml_prediction])
        total = len(st.session_state.tasks)
        st.metric("Agent-ML Agreement Rate", f"{agreement/total*100:.1f}%")


if __name__ == "__main__":
    main()
