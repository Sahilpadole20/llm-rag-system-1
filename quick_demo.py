"""
Quick Real-Time Demo
====================
Demonstrates all professor features with actual CSV data.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.realtime_simulator import RealTimeSimulator, NodeStatus
from src.service_manager import ServiceManager, ServiceType, Priority
from src.task_scheduler import TaskScheduler
from src.auto_failover import AutoFailoverSystem

CSV_PATH = r"c:\Users\Sahil Padole\Videos\AI_agent_ml_threshold\data\edgesimpy_failure_ml_+_thresh_(gb)_no_failure_20251223_075347_results.csv"

def main():
    print("=" * 60)
    print("  REAL-TIME DEMONSTRATION WITH YOUR CSV DATA")
    print("=" * 60)

    # 1. Load CSV
    print("\n[1] Loading CSV data...")
    simulator = RealTimeSimulator()
    simulator.load_csv(CSV_PATH)
    print(f"    Loaded {len(simulator.conditions)} network conditions")

    # Show layer distribution
    layer_counts = simulator.df['assigned_layer'].value_counts()
    print("\n    Layer Distribution:")
    for layer, count in layer_counts.items():
        pct = count / len(simulator.df) * 100
        print(f"       {layer}: {count} ({pct:.1f}%)")

    # 2. Stream conditions
    print("\n[2] Streaming real-time conditions...")
    for i in range(5):
        cond = simulator.get_next_condition()
        print(f"    Step {cond.step}: {cond.datarate_mbps:>6.1f} Mbps, SINR {cond.sinr:>5.1f} dB, "
              f"Latency {cond.latency_ms:>6.1f} ms --> {cond.assigned_layer}/Server{cond.server_id}")

    # 3. Node health
    print("\n[3] Node Health Status:")
    for node_id, health in simulator.node_health.items():
        status = "OK" if health.status == NodeStatus.ACTIVE else "FAIL"
        print(f"    Server {node_id} ({health.layer:5}): [{status}] CPU {health.cpu_utilization:.0f}%")

    # 4. Create services
    print("\n[4] Creating services...")
    mgr = ServiceManager()
    scheduler = TaskScheduler(mgr)

    xr = mgr.create_xr_service(throughput_mbps=50.0, latency_ms=20.0, num_users=5, priority=Priority.CRITICAL)
    scheduler.schedule_service(xr)
    print(f"    XR Service: {xr.service_id} --> {xr.assigned_layer}/Server{xr.assigned_server}")

    embb = mgr.create_embb_service(throughput_mbps=100.0, latency_ms=200.0, num_users=20, priority=Priority.HIGH)
    scheduler.schedule_service(embb)
    print(f"    eMBB Service: {embb.service_id} --> {embb.assigned_layer}/Server{embb.assigned_server}")

    # 5. Failover system
    print("\n[5] Setting up automatic failover...")
    failover = AutoFailoverSystem(mgr, scheduler, simulator)
    print("    AutoFailoverSystem ready")

    # 6. Simulate failure
    xr_server = xr.assigned_server
    print(f"\n[6] SIMULATING FAILURE: Server {xr_server}...")
    simulator.node_health[xr_server].status = NodeStatus.FAILED
    simulator.node_health[xr_server].cpu_utilization = 100.0

    # 7. Run failover
    print("\n[7] Running automatic failover...")
    decision = failover.check_and_migrate()
    print(f"    Affected: {decision.affected_services}")
    print(f"    Attempted: {decision.migrations_attempted}")
    print(f"    Successful: {decision.migrations_successful}")
    print(f"    Message: {decision.message}")

    # 8. Show migration history
    print("\n[8] Migration History:")
    for event in failover.migration_history:
        status = "SUCCESS" if event.success else "FAILED"
        print(f"    [{status}] {event.service_id}: Server{event.source_server} --> Server{event.target_server} ({event.reason.value})")

    # 9. Final service state  
    print("\n[9] Final Service Locations:")
    for svc in mgr.get_running_services():
        print(f"    {svc.service_id}: {svc.assigned_layer}/Server{svc.assigned_server}")

    # 10. Statistics
    stats = failover.get_statistics()
    print("\n[10] Statistics:")
    print(f"    Total Migrations: {stats['total_migrations']}")
    print(f"    Success Rate: {stats['success_rate']:.1f}%")

    print("\n" + "=" * 60)
    print("  ALL PROFESSOR FEATURES DEMONSTRATED IN REAL-TIME!")
    print("=" * 60)
    
    print("\nFeatures Demonstrated:")
    print("  1. CSV Data Loading (EdgeSimPy format)")
    print("  2. Real-time Network Streaming")
    print("  3. Layer Recommendations (ML + Threshold)")
    print("  4. Task Scheduling (XR/eMBB)")
    print("  5. Node Health Monitoring")
    print("  6. Node Failure Simulation")
    print("  7. Automatic Failover (Edge-->Fog)")
    print("  8. Migration History Tracking")
    print("  9. Statistics Reporting")
    print("\nTo see the Streamlit UI: streamlit run app.py")


if __name__ == "__main__":
    main()
