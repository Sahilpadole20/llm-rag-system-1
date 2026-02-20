"""
Real-Time Network Simulator
============================
Simulates real-world network conditions using CSV data or live API.
Provides continuous monitoring and automatic decision making.

Features:
- Read network metrics from CSV (ai4mobile dataset format)
- Simulate real-time API data stream
- Auto-detect node failures based on thresholds
- Automatic migration decisions
- Real-time visualization data
"""

import pandas as pd
import numpy as np
import time
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Generator
from dataclasses import dataclass, field
from enum import Enum
import threading


class NodeStatus(Enum):
    """Node health status."""
    ACTIVE = "active"
    DEGRADED = "degraded"
    FAILED = "failed"
    RECOVERING = "recovering"


@dataclass
class NetworkCondition:
    """Real-time network condition from sensor/API."""
    timestamp: datetime
    step: int
    datarate: float  # bps
    sinr: float  # dB
    latency_ms: float
    rsrp_dbm: float
    cpu_demand: int
    memory_demand: int
    
    # Optional fields from CSV data
    assigned_layer: Optional[str] = None
    server_id: Optional[int] = None
    model: Optional[str] = None
    
    # Derived metrics
    datarate_mbps: float = field(init=False)
    signal_quality: str = field(init=False)
    bandwidth_category: str = field(init=False)
    
    def __post_init__(self):
        self.datarate_mbps = self.datarate / 1_000_000
        
        # Signal quality assessment
        if self.rsrp_dbm >= -80:
            self.signal_quality = "Excellent"
        elif self.rsrp_dbm >= -90:
            self.signal_quality = "Good"
        elif self.rsrp_dbm >= -100:
            self.signal_quality = "Fair"
        elif self.rsrp_dbm >= -110:
            self.signal_quality = "Poor"
        else:
            self.signal_quality = "Very Poor"
        
        # Bandwidth category
        if self.datarate_mbps < 9.64:
            self.bandwidth_category = "Low"
        elif self.datarate_mbps < 16.60:
            self.bandwidth_category = "Medium"
        else:
            self.bandwidth_category = "High"
    
    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "step": self.step,
            "datarate_mbps": round(self.datarate_mbps, 2),
            "sinr_db": round(self.sinr, 2),
            "latency_ms": round(self.latency_ms, 2),
            "rsrp_dbm": round(self.rsrp_dbm, 2),
            "cpu_demand": self.cpu_demand,
            "memory_demand": self.memory_demand,
            "signal_quality": self.signal_quality,
            "bandwidth_category": self.bandwidth_category
        }


@dataclass
class NodeHealth:
    """Real-time node health metrics."""
    node_id: int
    layer: str
    status: NodeStatus
    cpu_utilization: float  # %
    memory_utilization: float  # %
    network_latency: float  # ms
    packet_loss: float  # %
    last_heartbeat: datetime
    consecutive_failures: int = 0
    
    def to_dict(self) -> Dict:
        return {
            "node_id": self.node_id,
            "layer": self.layer,
            "status": self.status.value,
            "cpu_utilization": round(self.cpu_utilization, 2),
            "memory_utilization": round(self.memory_utilization, 2),
            "network_latency": round(self.network_latency, 2),
            "packet_loss": round(self.packet_loss, 2),
            "last_heartbeat": self.last_heartbeat.isoformat(),
            "consecutive_failures": self.consecutive_failures
        }


class RealTimeSimulator:
    """
    Real-time network simulator using CSV data or simulated API.
    
    Provides:
    - Streaming network conditions
    - Node health monitoring
    - Automatic failure detection
    - Layer recommendation based on live metrics
    """
    
    # Thresholds for automatic decisions
    LOW_PING_THRESHOLD = 20  # ms
    DATARATE_33RD = 9.64e6  # 9.64 Mbps
    DATARATE_66TH = 16.60e6  # 16.60 Mbps
    SINR_THRESHOLD = 10  # dB
    
    # Node failure thresholds
    CPU_CRITICAL = 95.0  # %
    MEMORY_CRITICAL = 95.0  # %
    LATENCY_CRITICAL = 500  # ms
    PACKET_LOSS_CRITICAL = 10  # %
    HEARTBEAT_TIMEOUT = 30  # seconds
    
    def __init__(self, csv_path: Optional[str] = None):
        """Initialize simulator with optional CSV data source."""
        self.csv_path = csv_path
        self.df = None
        self.current_step = 0
        self.is_running = False
        
        # Node health tracking
        self.node_health: Dict[int, NodeHealth] = {}
        self._initialize_nodes()
        
        # Network conditions from CSV
        self.conditions: List[NetworkCondition] = []
        
        # History for visualization
        self.condition_history: List[NetworkCondition] = []
        self.decision_history: List[Dict] = []
        self.max_history = 100
        
        # Load CSV if provided
        if csv_path:
            self.load_csv(csv_path)
    
    def _initialize_nodes(self):
        """Initialize all nodes with healthy status."""
        node_config = [
            (1, "Edge"), (2, "Edge"), (3, "Edge"), (4, "Edge"),
            (5, "Fog"), (6, "Fog"),
            (7, "Cloud")
        ]
        
        for node_id, layer in node_config:
            self.node_health[node_id] = NodeHealth(
                node_id=node_id,
                layer=layer,
                status=NodeStatus.ACTIVE,
                cpu_utilization=random.uniform(20, 50),
                memory_utilization=random.uniform(30, 60),
                network_latency=5 if layer == "Edge" else (25 if layer == "Fog" else 100),
                packet_loss=random.uniform(0, 1),
                last_heartbeat=datetime.now()
            )
    
    def load_csv(self, csv_path: str) -> bool:
        """Load CSV data for simulation."""
        try:
            self.df = pd.read_csv(csv_path)
            self.csv_path = csv_path
            
            # Standardize column names for edgesimpy format
            column_mapping = {
                'data_rate': 'datarate',
                'SINR': 'sinr',
                'latency': 'latency_ms',
                'RSRP': 'rsrp_dbm',
                'cpu': 'cpu_demand',
                'memory': 'memory_demand',
                'ping_ms': 'ping',  # Keep ping separate from latency_ms
            }
            
            for old, new in column_mapping.items():
                if old in self.df.columns and new not in self.df.columns:
                    self.df.rename(columns={old: new}, inplace=True)
            
            # Convert datarate from bps to Mbps if values are large
            if 'datarate' in self.df.columns and self.df['datarate'].mean() > 1e5:
                self.df['datarate_mbps'] = self.df['datarate'] / 1e6
            elif 'datarate' in self.df.columns:
                self.df['datarate_mbps'] = self.df['datarate']
            
            # Parse existing network conditions into list
            self.conditions = []
            for idx, row in self.df.iterrows():
                condition = NetworkCondition(
                    timestamp=datetime.now(),
                    step=int(row.get('step', idx)),
                    datarate=row.get('datarate', 0),
                    sinr=row.get('sinr', 0),
                    latency_ms=row.get('latency_ms', 50),
                    rsrp_dbm=row.get('rsrp_dbm', -100),
                    cpu_demand=int(row.get('cpu_demand', 30)),
                    memory_demand=int(row.get('memory_demand', 500)),
                    assigned_layer=row.get('assigned_layer', None),
                    server_id=int(row.get('server_id', 0)) if pd.notna(row.get('server_id')) else None,
                    model=row.get('model', 'gradient_boosting')
                )
                self.conditions.append(condition)
            
            self.current_step = 0
            return True
        except Exception as e:
            print(f"Error loading CSV: {e}")
            return False
    
    def get_next_condition(self) -> Optional[NetworkCondition]:
        """Get next network condition from CSV or generate simulated data."""
        if self.conditions and self.current_step < len(self.conditions):
            # Use pre-parsed conditions
            condition = self.conditions[self.current_step]
            self.current_step += 1
        elif self.df is not None and self.current_step < len(self.df):
            row = self.df.iloc[self.current_step]
            condition = NetworkCondition(
                timestamp=datetime.now(),
                step=self.current_step,
                datarate=row.get('datarate', random.uniform(5e6, 30e6)),
                sinr=row.get('sinr', random.uniform(5, 20)),
                latency_ms=row.get('latency_ms', random.uniform(5, 200)),
                rsrp_dbm=row.get('rsrp_dbm', random.uniform(-120, -80)),
                cpu_demand=int(row.get('cpu_demand', random.randint(10, 100))),
                memory_demand=int(row.get('memory_demand', random.randint(100, 1000))),
                assigned_layer=row.get('assigned_layer', None),
                server_id=int(row.get('server_id', 0)) if pd.notna(row.get('server_id')) else None,
                model=row.get('model', None)
            )
            self.current_step += 1
        else:
            # Generate simulated real-time data
            condition = self._generate_simulated_condition()
        
        # Store in history
        self.condition_history.append(condition)
        if len(self.condition_history) > self.max_history:
            self.condition_history.pop(0)
        
        return condition
    
    def _generate_simulated_condition(self) -> NetworkCondition:
        """Generate realistic simulated network condition."""
        # Add some correlation to previous values for realism
        if self.condition_history:
            prev = self.condition_history[-1]
            datarate = prev.datarate + random.gauss(0, 2e6)
            datarate = max(1e6, min(50e6, datarate))
            sinr = prev.sinr + random.gauss(0, 2)
            sinr = max(0, min(30, sinr))
            latency = prev.latency_ms + random.gauss(0, 10)
            latency = max(1, min(500, latency))
            rsrp = prev.rsrp_dbm + random.gauss(0, 3)
            rsrp = max(-130, min(-70, rsrp))
        else:
            datarate = random.uniform(5e6, 30e6)
            sinr = random.uniform(5, 20)
            latency = random.uniform(5, 100)
            rsrp = random.uniform(-120, -85)
        
        self.current_step += 1
        
        return NetworkCondition(
            timestamp=datetime.now(),
            step=self.current_step,
            datarate=datarate,
            sinr=sinr,
            latency_ms=latency,
            rsrp_dbm=rsrp,
            cpu_demand=random.randint(10, 100),
            memory_demand=random.randint(100, 1000)
        )
    
    def update_node_health(self, node_id: int, metrics: Dict = None):
        """Update node health based on real/simulated metrics."""
        if node_id not in self.node_health:
            return
        
        node = self.node_health[node_id]
        
        if metrics:
            node.cpu_utilization = metrics.get('cpu', node.cpu_utilization)
            node.memory_utilization = metrics.get('memory', node.memory_utilization)
            node.network_latency = metrics.get('latency', node.network_latency)
            node.packet_loss = metrics.get('packet_loss', node.packet_loss)
        else:
            # Simulate metric changes
            node.cpu_utilization += random.gauss(0, 5)
            node.cpu_utilization = max(0, min(100, node.cpu_utilization))
            node.memory_utilization += random.gauss(0, 3)
            node.memory_utilization = max(0, min(100, node.memory_utilization))
            node.packet_loss = max(0, min(100, random.gauss(1, 2)))
        
        node.last_heartbeat = datetime.now()
        
        # Auto-detect failure conditions
        self._check_node_failure(node_id)
    
    def _check_node_failure(self, node_id: int):
        """Automatically detect if node should be marked as failed."""
        node = self.node_health[node_id]
        
        failure_conditions = [
            node.cpu_utilization >= self.CPU_CRITICAL,
            node.memory_utilization >= self.MEMORY_CRITICAL,
            node.network_latency >= self.LATENCY_CRITICAL,
            node.packet_loss >= self.PACKET_LOSS_CRITICAL
        ]
        
        if any(failure_conditions):
            node.consecutive_failures += 1
            if node.consecutive_failures >= 3:
                node.status = NodeStatus.FAILED
            else:
                node.status = NodeStatus.DEGRADED
        else:
            node.consecutive_failures = 0
            if node.status in [NodeStatus.DEGRADED, NodeStatus.RECOVERING]:
                node.status = NodeStatus.ACTIVE
    
    def simulate_node_failure(self, node_id: int):
        """Manually trigger node failure for testing."""
        if node_id in self.node_health:
            self.node_health[node_id].status = NodeStatus.FAILED
            self.node_health[node_id].consecutive_failures = 10
    
    def recover_node(self, node_id: int):
        """Recover a failed node."""
        if node_id in self.node_health:
            node = self.node_health[node_id]
            node.status = NodeStatus.RECOVERING
            node.consecutive_failures = 0
            node.cpu_utilization = 30
            node.memory_utilization = 40
            node.last_heartbeat = datetime.now()
            # After a moment, set to active
            node.status = NodeStatus.ACTIVE
    
    def get_recommended_layer(self, condition: NetworkCondition) -> Tuple[str, str, List[int]]:
        """
        Determine recommended layer based on network condition and node availability.
        
        Returns: (layer, reason, available_servers)
        """
        # Get available nodes per layer
        edge_nodes = [n for n, h in self.node_health.items() 
                      if h.layer == "Edge" and h.status == NodeStatus.ACTIVE]
        fog_nodes = [n for n, h in self.node_health.items() 
                     if h.layer == "Fog" and h.status == NodeStatus.ACTIVE]
        cloud_nodes = [n for n, h in self.node_health.items() 
                       if h.layer == "Cloud" and h.status == NodeStatus.ACTIVE]
        
        # Determine ideal layer based on network metrics
        ideal_layer = self._calculate_ideal_layer(condition)
        
        # Check availability and fallback
        if ideal_layer == "Edge":
            if edge_nodes:
                return "Edge", f"Low latency ({condition.latency_ms:.1f}ms < 20ms)", edge_nodes
            elif fog_nodes:
                return "Fog", "Edge unavailable - Fallback to Fog", fog_nodes
            elif cloud_nodes:
                return "Cloud", "Edge & Fog unavailable - Fallback to Cloud", cloud_nodes
            else:
                return "None", "ALL NODES FAILED - CRITICAL", []
        
        elif ideal_layer == "Fog":
            if fog_nodes:
                return "Fog", f"Medium bandwidth ({condition.datarate_mbps:.1f} Mbps), Good SINR ({condition.sinr:.1f} dB)", fog_nodes
            elif edge_nodes:
                return "Edge", "Fog unavailable - Fallback to Edge", edge_nodes
            elif cloud_nodes:
                return "Cloud", "Fog & Edge unavailable - Fallback to Cloud", cloud_nodes
            else:
                return "None", "ALL NODES FAILED - CRITICAL", []
        
        else:  # Cloud
            if cloud_nodes:
                return "Cloud", f"High bandwidth ({condition.datarate_mbps:.1f} Mbps) or Poor SINR ({condition.sinr:.1f} dB)", cloud_nodes
            elif fog_nodes:
                return "Fog", "Cloud unavailable - Fallback to Fog", fog_nodes
            elif edge_nodes:
                return "Edge", "Cloud & Fog unavailable - Fallback to Edge", edge_nodes
            else:
                return "None", "ALL NODES FAILED - CRITICAL", []
    
    def _calculate_ideal_layer(self, condition: NetworkCondition) -> str:
        """Calculate ideal layer based on paper algorithm."""
        if condition.latency_ms < self.LOW_PING_THRESHOLD:
            if condition.datarate < self.DATARATE_66TH:
                return "Edge"
        
        if (self.DATARATE_33RD <= condition.datarate < self.DATARATE_66TH and 
            condition.sinr > self.SINR_THRESHOLD):
            return "Fog"
        
        if condition.datarate >= self.DATARATE_66TH or condition.sinr <= self.SINR_THRESHOLD:
            return "Cloud"
        
        if condition.rsrp_dbm < -120:
            return "Cloud"
        
        return "Edge"  # Default
    
    def get_layer_status(self) -> Dict[str, Dict]:
        """Get aggregated status for each layer."""
        layers = {"Edge": [], "Fog": [], "Cloud": []}
        
        for node_id, health in self.node_health.items():
            layers[health.layer].append(health)
        
        result = {}
        for layer, nodes in layers.items():
            active = sum(1 for n in nodes if n.status == NodeStatus.ACTIVE)
            degraded = sum(1 for n in nodes if n.status == NodeStatus.DEGRADED)
            failed = sum(1 for n in nodes if n.status == NodeStatus.FAILED)
            
            avg_cpu = np.mean([n.cpu_utilization for n in nodes]) if nodes else 0
            avg_mem = np.mean([n.memory_utilization for n in nodes]) if nodes else 0
            
            if failed == len(nodes):
                status = "OFFLINE"
            elif failed > 0 or degraded > 0:
                status = "DEGRADED"
            else:
                status = "HEALTHY"
            
            result[layer] = {
                "status": status,
                "total_nodes": len(nodes),
                "active": active,
                "degraded": degraded,
                "failed": failed,
                "avg_cpu": round(avg_cpu, 1),
                "avg_memory": round(avg_mem, 1)
            }
        
        return result
    
    def stream_conditions(self, interval_ms: int = 1000) -> Generator[NetworkCondition, None, None]:
        """Generator for streaming network conditions."""
        self.is_running = True
        while self.is_running:
            condition = self.get_next_condition()
            if condition:
                # Update all node health
                for node_id in self.node_health:
                    self.update_node_health(node_id)
                
                yield condition
            
            time.sleep(interval_ms / 1000)
    
    def stop(self):
        """Stop the streaming."""
        self.is_running = False
    
    def reset(self):
        """Reset simulator to initial state."""
        self.current_step = 0
        self.condition_history.clear()
        self.decision_history.clear()
        self._initialize_nodes()
    
    def get_visualization_data(self) -> Dict:
        """Get data formatted for Streamlit visualization."""
        # Recent conditions for charts
        recent = self.condition_history[-20:] if self.condition_history else []
        
        return {
            "network_metrics": {
                "timestamps": [c.timestamp.strftime("%H:%M:%S") for c in recent],
                "datarate": [c.datarate_mbps for c in recent],
                "sinr": [c.sinr for c in recent],
                "latency": [c.latency_ms for c in recent],
                "rsrp": [c.rsrp_dbm for c in recent]
            },
            "node_health": {
                node_id: health.to_dict() 
                for node_id, health in self.node_health.items()
            },
            "layer_status": self.get_layer_status(),
            "current_step": self.current_step,
            "total_steps": len(self.df) if self.df is not None else "∞"
        }
