"""
Task Scheduler Module
=====================
Handles task scheduling for XR and eMBB services across Edge-Fog-Cloud layers.

Professor's Requirement #1: Task Scheduling for Different Services
- XR Services: Edge preferred (low latency)
- eMBB Services: Cloud/Fog preferred (high throughput)
- Load balancing across servers
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import random

from src.service_manager import (
    Service, Server, ServiceManager, ServiceType, Priority
)


@dataclass
class SchedulingResult:
    """Result of a scheduling operation."""
    success: bool
    service_id: str
    assigned_layer: Optional[str] = None
    assigned_server: Optional[int] = None
    message: str = ""
    latency_ms: float = 0.0
    cost_per_hour: float = 0.0


@dataclass
class SchedulingMetrics:
    """Metrics for scheduling performance."""
    total_scheduled: int = 0
    total_failed: int = 0
    xr_scheduled: int = 0
    embb_scheduled: int = 0
    edge_assignments: int = 0
    fog_assignments: int = 0
    cloud_assignments: int = 0
    total_latency_ms: float = 0.0
    total_cost_per_hour: float = 0.0
    
    def add_result(self, result: SchedulingResult, service_type: ServiceType):
        """Add a scheduling result to metrics."""
        if result.success:
            self.total_scheduled += 1
            self.total_latency_ms += result.latency_ms
            self.total_cost_per_hour += result.cost_per_hour
            
            if service_type == ServiceType.XR:
                self.xr_scheduled += 1
            else:
                self.embb_scheduled += 1
            
            if result.assigned_layer == "Edge":
                self.edge_assignments += 1
            elif result.assigned_layer == "Fog":
                self.fog_assignments += 1
            else:
                self.cloud_assignments += 1
        else:
            self.total_failed += 1
    
    @property
    def average_latency(self) -> float:
        """Calculate average latency."""
        return self.total_latency_ms / self.total_scheduled if self.total_scheduled > 0 else 0
    
    def to_dict(self) -> Dict:
        """Convert metrics to dictionary."""
        return {
            "total_scheduled": self.total_scheduled,
            "total_failed": self.total_failed,
            "success_rate": (self.total_scheduled / (self.total_scheduled + self.total_failed) * 100 
                           if (self.total_scheduled + self.total_failed) > 0 else 0),
            "xr_scheduled": self.xr_scheduled,
            "embb_scheduled": self.embb_scheduled,
            "edge_assignments": self.edge_assignments,
            "fog_assignments": self.fog_assignments,
            "cloud_assignments": self.cloud_assignments,
            "average_latency_ms": round(self.average_latency, 2),
            "total_cost_per_hour": round(self.total_cost_per_hour, 2)
        }


class TaskScheduler:
    """
    Task Scheduler for XR and eMBB services.
    
    Scheduling Strategy:
    - XR Services (low latency): Prefer Edge -> Fog -> Cloud
    - eMBB Services (high throughput): Prefer Cloud -> Fog -> Edge
    - Consider server utilization and load balancing
    - Respect latency constraints
    """
    
    # Utilization threshold for load balancing (avoid overloading servers)
    UTILIZATION_THRESHOLD = 80.0  # 80% as per professor's requirement
    
    def __init__(self, service_manager: ServiceManager):
        """Initialize scheduler with service manager."""
        self.service_manager = service_manager
        self.metrics = SchedulingMetrics()
        self.scheduling_history: List[Dict] = []
    
    def _get_layer_priority(self, service: Service) -> List[str]:
        """
        Get preferred layer order based on service type.
        
        XR: Edge -> Fog -> Cloud (latency-sensitive)
        eMBB: Cloud -> Fog -> Edge (throughput-focused)
        """
        if service.service_type == ServiceType.XR:
            return ["Edge", "Fog", "Cloud"]
        else:  # eMBB
            return ["Cloud", "Fog", "Edge"]
    
    def _check_latency_constraint(self, service: Service, server: Server) -> bool:
        """Check if server meets service latency requirements."""
        return server.base_latency_ms <= service.latency_ms
    
    def _find_best_server(self, service: Service, layer: str) -> Optional[Server]:
        """
        Find best server in a layer for a service.
        
        Selection criteria:
        1. Server is active
        2. Has sufficient resources
        3. Meets latency requirements
        4. Below utilization threshold (for load balancing)
        5. Prefer least utilized server
        """
        servers = self.service_manager.get_servers_by_layer(layer)
        candidates = []
        
        for server in servers:
            # Check basic requirements
            if not server.is_active:
                continue
            
            if not server.can_accommodate(service.cpu_demand, service.memory_demand):
                continue
            
            if not self._check_latency_constraint(service, server):
                continue
            
            # Prefer servers below threshold for load balancing
            avg_util = (server.cpu_utilization + server.memory_utilization) / 2
            if avg_util < self.UTILIZATION_THRESHOLD:
                candidates.append((server, avg_util))
        
        if not candidates:
            # If no server below threshold, try any available server
            for server in servers:
                if (server.is_active and 
                    server.can_accommodate(service.cpu_demand, service.memory_demand) and
                    self._check_latency_constraint(service, server)):
                    avg_util = (server.cpu_utilization + server.memory_utilization) / 2
                    candidates.append((server, avg_util))
        
        if not candidates:
            return None
        
        # Select least utilized server
        candidates.sort(key=lambda x: x[1])
        return candidates[0][0]
    
    def schedule_service(self, service: Service) -> SchedulingResult:
        """
        Schedule a service to an appropriate server.
        
        Algorithm:
        1. Get layer priority based on service type
        2. Try each layer in priority order
        3. Find best available server in each layer
        4. Allocate resources if found
        """
        layer_priority = self._get_layer_priority(service)
        
        for layer in layer_priority:
            server = self._find_best_server(service, layer)
            
            if server:
                # Allocate resources
                success = server.allocate(
                    service.cpu_demand, 
                    service.memory_demand, 
                    service.service_id
                )
                
                if success:
                    # Update service status
                    service.assigned_layer = layer
                    service.assigned_server = server.server_id
                    service.status = "running"
                    
                    result = SchedulingResult(
                        success=True,
                        service_id=service.service_id,
                        assigned_layer=layer,
                        assigned_server=server.server_id,
                        message=f"Scheduled {service.service_type.value} service to Server {server.server_id} ({layer})",
                        latency_ms=server.base_latency_ms,
                        cost_per_hour=server.cost_per_hour
                    )
                    
                    # Record metrics and history
                    self.metrics.add_result(result, service.service_type)
                    self._record_scheduling(service, server, result)
                    
                    return result
        
        # No suitable server found
        service.status = "failed"
        result = SchedulingResult(
            success=False,
            service_id=service.service_id,
            message=f"No suitable server found for {service.service_type.value} service (CPU: {service.cpu_demand}, MEM: {service.memory_demand})"
        )
        self.metrics.add_result(result, service.service_type)
        return result
    
    def schedule_all_pending(self) -> List[SchedulingResult]:
        """Schedule all pending services by priority."""
        pending = self.service_manager.get_pending_services()
        results = []
        
        for service in pending:
            result = self.schedule_service(service)
            results.append(result)
        
        return results
    
    def _record_scheduling(self, service: Service, server: Server, result: SchedulingResult):
        """Record scheduling event in history."""
        self.scheduling_history.append({
            "timestamp": datetime.now().isoformat(),
            "service_id": service.service_id,
            "service_type": service.service_type.value,
            "priority": service.priority.value,
            "server_id": server.server_id,
            "layer": server.layer,
            "cpu_allocated": service.cpu_demand,
            "memory_allocated": service.memory_demand,
            "latency_ms": result.latency_ms,
            "cost_per_hour": result.cost_per_hour
        })
    
    def get_server_load_report(self) -> List[Dict]:
        """Get load report for all servers."""
        report = []
        for server_id, server in self.service_manager.servers.items():
            report.append({
                "server_id": server_id,
                "layer": server.layer,
                "is_active": server.is_active,
                "cpu_used": server.used_cpu,
                "cpu_total": server.total_cpu,
                "cpu_util_%": round(server.cpu_utilization, 2),
                "mem_used": server.used_memory,
                "mem_total": server.total_memory,
                "mem_util_%": round(server.memory_utilization, 2),
                "services_count": len(server.running_services),
                "base_latency_ms": server.base_latency_ms,
                "cost_per_hour": server.cost_per_hour
            })
        return report
    
    def simulate_workload(self, 
                          num_xr_services: int = 5, 
                          num_embb_services: int = 10) -> Dict:
        """
        Simulate a workload with mixed XR and eMBB services.
        
        Returns scheduling results and metrics.
        """
        results = {
            "xr_results": [],
            "embb_results": [],
            "metrics": None,
            "server_load": None
        }
        
        # Create and schedule XR services
        for _ in range(num_xr_services):
            service = self.service_manager.create_xr_service()
            result = self.schedule_service(service)
            results["xr_results"].append(result)
        
        # Create and schedule eMBB services
        for _ in range(num_embb_services):
            service = self.service_manager.create_embb_service()
            result = self.schedule_service(service)
            results["embb_results"].append(result)
        
        results["metrics"] = self.metrics.to_dict()
        results["server_load"] = self.get_server_load_report()
        
        return results
    
    def reset_metrics(self):
        """Reset scheduling metrics."""
        self.metrics = SchedulingMetrics()
        self.scheduling_history.clear()


def run_scheduling_demo():
    """Run a demonstration of the task scheduler."""
    print("=" * 60)
    print("Task Scheduler Demo - XR and eMBB Services")
    print("=" * 60)
    
    # Initialize
    manager = ServiceManager()
    scheduler = TaskScheduler(manager)
    
    # Simulate workload
    print("\nSimulating workload: 5 XR + 10 eMBB services...")
    results = scheduler.simulate_workload(num_xr_services=5, num_embb_services=10)
    
    # Print results
    print("\n--- XR Service Results ---")
    for r in results["xr_results"]:
        status = "✓" if r.success else "✗"
        print(f"  {status} {r.message}")
    
    print("\n--- eMBB Service Results ---")
    for r in results["embb_results"]:
        status = "✓" if r.success else "✗"
        print(f"  {status} {r.message}")
    
    print("\n--- Metrics ---")
    for key, value in results["metrics"].items():
        print(f"  {key}: {value}")
    
    print("\n--- Server Load ---")
    for load in results["server_load"]:
        print(f"  Server {load['server_id']} ({load['layer']}): "
              f"CPU {load['cpu_util_%']}%, MEM {load['mem_util_%']}%, "
              f"Services: {load['services_count']}")
    
    return results


if __name__ == "__main__":
    run_scheduling_demo()
