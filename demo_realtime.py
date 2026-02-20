"""
Real-Time Simulator Demo
========================
Demonstrates the real-time simulation using your actual CSV data.

This shows:
1. Loading CSV data from edgesimpy
2. Real-time network condition streaming
3. Automatic layer recommendations
4. Node health monitoring
5. Automatic failover when nodes fail

Run: python demo_realtime.py
"""

import sys
import time
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.realtime_simulator import RealTimeSimulator, NodeStatus, NetworkCondition
from src.service_manager import ServiceManager, ServiceType, Priority
from src.task_scheduler import TaskScheduler
from src.auto_failover import AutoFailoverSystem

# Path to your CSV data
CSV_PATH = Path(r"c:\Users\Sahil Padole\Videos\AI_agent_ml_threshold\data\edgesimpy_failure_ml_+_thresh_(gb)_no_failure_20251223_075347_results.csv")


def print_header(text: str):
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)


def print_node_status(simulator: RealTimeSimulator):
    """Print current node health status."""
    print("\n🖥️  NODE HEALTH DASHBOARD")
    print("-" * 60)
    
    for layer in ["Edge", "Fog", "Cloud"]:
        nodes = [n for n in simulator.node_health.values() if n.layer == layer]
        layer_icon = "🟢" if layer == "Edge" else ("🟡" if layer == "Fog" else "🔵")
        print(f"\n{layer_icon} {layer} Layer:")
        
        for node in nodes:
            status_icon = {
                NodeStatus.ACTIVE: "✅",
                NodeStatus.DEGRADED: "⚠️",
                NodeStatus.FAILED: "❌",
                NodeStatus.RECOVERING: "🔄"
            }.get(node.status, "❓")
            
            print(f"   Server {node.node_id}: {status_icon} {node.status.value.upper():<10} "
                  f"| CPU: {node.cpu_utilization:5.1f}% | MEM: {node.memory_utilization:5.1f}%")


def print_condition(condition: NetworkCondition, step: int):
    """Print network condition details."""
    print(f"\n📡 STEP {step}: Network Condition")
    print("-" * 40)
    print(f"   Data Rate:     {condition.datarate_mbps:>8.2f} Mbps ({condition.bandwidth_category})")
    print(f"   SINR:          {condition.sinr:>8.2f} dB")
    print(f"   Latency:       {condition.latency_ms:>8.2f} ms")
    print(f"   RSRP:          {condition.rsrp_dbm:>8.2f} dBm ({condition.signal_quality})")
    print(f"   CPU Demand:    {condition.cpu_demand:>8d} %")
    print(f"   Memory Demand: {condition.memory_demand:>8d} MB")
    
    if condition.assigned_layer:
        print(f"\n   📋 ORIGINAL DECISION:")
        print(f"      Model: {condition.model}")
        print(f"      Layer: {condition.assigned_layer} → Server {condition.server_id}")


def demo_csv_loading():
    """Demo 1: Loading CSV data."""
    print_header("DEMO 1: Loading CSV Data")
    
    simulator = RealTimeSimulator()
    
    if CSV_PATH.exists():
        print(f"\n📂 Loading: {CSV_PATH.name}")
        success = simulator.load_csv(str(CSV_PATH))
        
        if success:
            print(f"✅ Loaded {len(simulator.conditions)} network conditions")
            print(f"\n📊 Data Summary:")
            print(f"   Total records: {len(simulator.df)}")
            print(f"   Columns: {list(simulator.df.columns[:8])}...")
            
            # Show data distribution
            if 'assigned_layer' in simulator.df.columns:
                layer_counts = simulator.df['assigned_layer'].value_counts()
                print(f"\n   Layer Distribution:")
                for layer, count in layer_counts.items():
                    print(f"      {layer}: {count} ({count/len(simulator.df)*100:.1f}%)")
            
            return simulator
        else:
            print("❌ Failed to load CSV")
    else:
        print(f"⚠️ CSV not found: {CSV_PATH}")
        print("   Using simulated data instead...")
        return simulator
    
    return None


def demo_realtime_streaming(simulator: RealTimeSimulator, num_steps: int = 10):
    """Demo 2: Real-time streaming of network conditions."""
    print_header("DEMO 2: Real-Time Network Condition Streaming")
    
    print(f"\n🔄 Streaming {num_steps} network conditions...")
    print("   (Press Ctrl+C to stop)")
    
    for i in range(num_steps):
        condition = simulator.get_next_condition()
        
        if condition:
            print_condition(condition, i + 1)
            
            # Get recommendation
            layer, reason, available = simulator.get_recommended_layer(condition)
            
            print(f"\n   🎯 RECOMMENDED: {layer}")
            print(f"      Reason: {reason}")
            print(f"      Available Servers: {available}")
        
        if i < num_steps - 1:
            time.sleep(0.5)  # Simulate real-time delay


def demo_node_failure(simulator: RealTimeSimulator):
    """Demo 3: Node failure and automatic recommendations."""
    print_header("DEMO 3: Node Failure Simulation")
    
    print_node_status(simulator)
    
    # Simulate Edge node failure
    print("\n\n💥 SIMULATING: Server 1 (Edge) FAILURE...")
    simulator.node_health[1].status = NodeStatus.FAILED
    simulator.node_health[1].cpu_utilization = 100.0
    
    print_node_status(simulator)
    
    # Get a condition and see how recommendation changes
    condition = simulator.get_next_condition()
    if condition:
        print_condition(condition, simulator.current_step)
        
        layer, reason, available = simulator.get_recommended_layer(condition)
        print(f"\n   🎯 RECOMMENDED (after failure): {layer}")
        print(f"      Reason: {reason}")
        print(f"      Available Servers: {available}")
    
    # Recover the node
    print("\n\n🔧 RECOVERING: Server 1...")
    simulator.node_health[1].status = NodeStatus.ACTIVE
    simulator.node_health[1].cpu_utilization = 30.0
    
    print_node_status(simulator)


def demo_auto_failover():
    """Demo 4: Full automatic failover with services."""
    print_header("DEMO 4: Automatic Failover with Services")
    
    # Initialize components
    service_manager = ServiceManager()
    scheduler = TaskScheduler(service_manager)
    simulator = RealTimeSimulator()
    
    # Load CSV
    if CSV_PATH.exists():
        simulator.load_csv(str(CSV_PATH))
    
    failover = AutoFailoverSystem(service_manager, scheduler, simulator)
    
    # Create some services
    print("\n📦 Creating services...")
    
    # XR service (needs Edge)
    xr_service = service_manager.create_service(
        service_type=ServiceType.XR,
        cpu_demand=1000,
        memory_demand=2000,
        latency_req=20.0,
        throughput_mbps=50.0,
        priority=Priority.CRITICAL
    )
    result = scheduler.schedule_service(xr_service)
    print(f"   XR Service: {xr_service.service_id} → {result}")
    
    # eMBB service (needs Cloud)
    embb_service = service_manager.create_service(
        service_type=ServiceType.EMBB,
        cpu_demand=5000,
        memory_demand=10000,
        latency_req=200.0,
        throughput_mbps=100.0,
        priority=Priority.HIGH
    )
    result = scheduler.schedule_service(embb_service)
    print(f"   eMBB Service: {embb_service.service_id} → {result}")
    
    # Show initial state
    print("\n📊 Initial Service Assignment:")
    for service in service_manager.get_running_services():
        print(f"   {service.service_id}: {service.service_type.value} → "
              f"{service.assigned_layer}/Server{service.assigned_server}")
    
    print_node_status(simulator)
    
    # Simulate failure of the server hosting XR service
    xr_server = xr_service.assigned_server
    print(f"\n\n💥 TRIGGERING FAILURE: Server {xr_server} ({simulator.node_health[xr_server].layer})...")
    simulator.node_health[xr_server].status = NodeStatus.FAILED
    
    # Run failover check
    print("\n🔄 Running automatic failover check...")
    decision = failover.check_and_migrate()
    
    print(f"\n📋 FAILOVER DECISION:")
    print(f"   Trigger: {decision.trigger}")
    print(f"   Affected Services: {decision.affected_services}")
    print(f"   Migrations Attempted: {decision.migrations_attempted}")
    print(f"   Migrations Successful: {decision.migrations_successful}")
    print(f"   Message: {decision.message}")
    
    # Show new state
    print("\n📊 Service Assignment AFTER Failover:")
    for service in service_manager.get_running_services():
        print(f"   {service.service_id}: {service.service_type.value} → "
              f"{service.assigned_layer}/Server{service.assigned_server}")
    
    # Show migration history
    print("\n📜 Migration History:")
    for event in failover.migration_history:
        status = "✅" if event.success else "❌"
        print(f"   {status} {event.service_id}: Server{event.source_server} → Server{event.target_server} "
              f"({event.reason.value})")
    
    # Statistics
    stats = failover.get_statistics()
    print(f"\n📈 Statistics:")
    print(f"   Total Migrations: {stats['total_migrations']}")
    print(f"   Success Rate: {stats['success_rate']:.1f}%")


def main():
    """Run all demos."""
    print("\n" + "=" * 60)
    print("  🎮 REAL-TIME SIMULATOR DEMONSTRATION")
    print("  Using your actual EdgeSimPy CSV data")
    print("=" * 60)
    
    try:
        # Demo 1: Load CSV
        simulator = demo_csv_loading()
        
        if simulator:
            input("\n\n▶️  Press Enter to continue to Demo 2 (Real-time Streaming)...")
            
            # Demo 2: Real-time streaming
            demo_realtime_streaming(simulator, num_steps=5)
            
            input("\n\n▶️  Press Enter to continue to Demo 3 (Node Failure)...")
            
            # Demo 3: Node failure
            demo_node_failure(simulator)
            
            input("\n\n▶️  Press Enter to continue to Demo 4 (Auto Failover)...")
            
            # Demo 4: Full failover
            demo_auto_failover()
        
        print_header("DEMO COMPLETE")
        print("\n✅ All professor's features demonstrated in real-time!")
        print("\n   Features shown:")
        print("   1. ✅ CSV data loading (EdgeSimPy format)")
        print("   2. ✅ Real-time network condition streaming")
        print("   3. ✅ Automatic layer recommendations")
        print("   4. ✅ Node health monitoring")
        print("   5. ✅ Node failure detection")
        print("   6. ✅ Automatic failover (Edge→Fog→Cloud)")
        print("   7. ✅ Migration history tracking")
        print("   8. ✅ Statistics reporting")
        
        print("\n🚀 To see the full Streamlit UI: streamlit run app.py")
        
    except KeyboardInterrupt:
        print("\n\n⏹️ Demo stopped by user")


if __name__ == "__main__":
    main()
