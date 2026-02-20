"""
Failure Handler Module
======================
Handles node failures and service migration/rebalancing.

Professor's Requirement #2: Node Failure Handling
- Consider Fog server goes down
- With active XR and eMBB services
- Migrate services to other servers
- Rebalance load across available servers
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import copy

from src.service_manager import (
    Service, Server, ServiceManager, ServiceType, Priority
)
from src.task_scheduler import TaskScheduler, SchedulingResult


@dataclass
class FailureEvent:
    """Records a failure event."""
    timestamp: datetime
    server_id: int
    layer: str
    affected_services: List[str]
    migrated_services: List[str]
    failed_migrations: List[str]
    message: str


@dataclass
class MigrationResult:
    """Result of a service migration."""
    success: bool
    service_id: str
    source_server: int
    target_server: Optional[int]
    target_layer: Optional[str]
    message: str


class FailureHandler:
    """
    Handles node failures and service recovery.
    
    When a node fails:
    1. Mark node as inactive
    2. Identify affected services
    3. Attempt to migrate services to other available servers
    4. Rebalance load if needed
    5. Log failure event
    """
    
    def __init__(self, service_manager: ServiceManager, scheduler: TaskScheduler):
        """Initialize failure handler."""
        self.service_manager = service_manager
        self.scheduler = scheduler
        self.failure_history: List[FailureEvent] = []
    
    def simulate_server_failure(self, server_id: int) -> FailureEvent:
        """
        Simulate a server failure and handle recovery.
        
        Args:
            server_id: ID of the server that failed
        
        Returns:
            FailureEvent with details of the failure and recovery
        """
        server = self.service_manager.get_server(server_id)
        if not server:
            return FailureEvent(
                timestamp=datetime.now(),
                server_id=server_id,
                layer="Unknown",
                affected_services=[],
                migrated_services=[],
                failed_migrations=[],
                message=f"Server {server_id} not found"
            )
        
        if not server.is_active:
            return FailureEvent(
                timestamp=datetime.now(),
                server_id=server_id,
                layer=server.layer,
                affected_services=[],
                migrated_services=[],
                failed_migrations=[],
                message=f"Server {server_id} is already inactive"
            )
        
        # Mark server as failed
        server.is_active = False
        affected_services = server.running_services.copy()
        
        print(f"\n⚠️ SERVER FAILURE: Server {server_id} ({server.layer}) has failed!")
        print(f"   Affected services: {len(affected_services)}")
        
        # Attempt to migrate each affected service
        migrated = []
        failed = []
        
        for service_id in affected_services:
            service = self.service_manager.get_service(service_id)
            if service:
                result = self._migrate_service(service, server)
                if result.success:
                    migrated.append(service_id)
                else:
                    failed.append(service_id)
        
        # Clear the failed server's resources
        server.used_cpu = 0
        server.used_memory = 0
        server.running_services.clear()
        
        # Create failure event
        event = FailureEvent(
            timestamp=datetime.now(),
            server_id=server_id,
            layer=server.layer,
            affected_services=affected_services,
            migrated_services=migrated,
            failed_migrations=failed,
            message=f"Server {server_id} failed. Migrated {len(migrated)}/{len(affected_services)} services."
        )
        
        self.failure_history.append(event)
        
        print(f"   ✓ Migrated: {len(migrated)}")
        print(f"   ✗ Failed: {len(failed)}")
        
        return event
    
    def _migrate_service(self, service: Service, failed_server: Server) -> MigrationResult:
        """
        Migrate a service from a failed server to another available server.
        
        Migration strategy:
        1. First try same layer (for latency consistency)
        2. Then try layer according to service type preference
        3. XR: Edge -> Fog -> Cloud
        4. eMBB: Cloud -> Fog -> Edge
        """
        original_layer = failed_server.layer
        original_status = service.status
        
        # Get layer priority for migration
        if service.service_type == ServiceType.XR:
            # XR prefers low latency
            layer_priority = ["Edge", "Fog", "Cloud"]
        else:
            # eMBB prefers throughput
            layer_priority = ["Cloud", "Fog", "Edge"]
        
        # Try same layer first (if not the only server in layer)
        if original_layer in layer_priority:
            layer_priority.remove(original_layer)
            layer_priority.insert(0, original_layer)
        
        # Reset service for rescheduling
        service.assigned_layer = None
        service.assigned_server = None
        service.status = "pending"
        
        # Try to schedule to each layer
        for layer in layer_priority:
            servers = self.service_manager.get_servers_by_layer(layer)
            
            for server in servers:
                # Skip the failed server and inactive servers
                if server.server_id == failed_server.server_id or not server.is_active:
                    continue
                
                # Check if server can accommodate the service
                if server.can_accommodate(service.cpu_demand, service.memory_demand):
                    # Check latency constraint
                    if server.base_latency_ms <= service.latency_ms:
                        # Allocate resources
                        success = server.allocate(
                            service.cpu_demand,
                            service.memory_demand,
                            service.service_id
                        )
                        
                        if success:
                            service.assigned_layer = layer
                            service.assigned_server = server.server_id
                            service.status = "running"
                            
                            print(f"   → Migrated {service.service_type.value} service {service.service_id} "
                                  f"to Server {server.server_id} ({layer})")
                            
                            return MigrationResult(
                                success=True,
                                service_id=service.service_id,
                                source_server=failed_server.server_id,
                                target_server=server.server_id,
                                target_layer=layer,
                                message=f"Migrated to Server {server.server_id} ({layer})"
                            )
        
        # Migration failed - no suitable server found
        service.status = "failed"
        print(f"   ✗ Failed to migrate {service.service_type.value} service {service.service_id}")
        
        return MigrationResult(
            success=False,
            service_id=service.service_id,
            source_server=failed_server.server_id,
            target_server=None,
            target_layer=None,
            message="No suitable server found for migration"
        )
    
    def recover_server(self, server_id: int) -> Dict:
        """
        Recover a failed server and make it available again.
        
        Args:
            server_id: ID of the server to recover
        
        Returns:
            Recovery status
        """
        server = self.service_manager.get_server(server_id)
        if not server:
            return {"success": False, "message": f"Server {server_id} not found"}
        
        if server.is_active:
            return {"success": False, "message": f"Server {server_id} is already active"}
        
        # Reactivate server
        server.is_active = True
        server.used_cpu = 0
        server.used_memory = 0
        server.running_services.clear()
        
        print(f"\n✓ Server {server_id} ({server.layer}) recovered and available")
        
        return {
            "success": True,
            "message": f"Server {server_id} recovered",
            "server_id": server_id,
            "layer": server.layer,
            "available_cpu": server.available_cpu,
            "available_memory": server.available_memory
        }
    
    def rebalance_load(self) -> Dict:
        """
        Rebalance load across all active servers.
        
        This migrates services from overloaded servers to underutilized ones.
        """
        OVERLOAD_THRESHOLD = 85.0  # Consider server overloaded above 85%
        TARGET_THRESHOLD = 60.0    # Target utilization after rebalancing
        
        migrations_performed = 0
        servers_rebalanced = []
        
        # Find overloaded servers
        for server_id, server in self.service_manager.servers.items():
            if not server.is_active:
                continue
            
            avg_util = (server.cpu_utilization + server.memory_utilization) / 2
            
            if avg_util > OVERLOAD_THRESHOLD and server.running_services:
                print(f"\n🔄 Server {server_id} is overloaded ({avg_util:.1f}%), rebalancing...")
                
                # Try to migrate some services
                services_to_migrate = []
                for service_id in server.running_services:
                    service = self.service_manager.get_service(service_id)
                    if service and service.priority.value >= Priority.MEDIUM.value:
                        services_to_migrate.append(service)
                
                # Sort by priority (lower priority first for migration)
                services_to_migrate.sort(key=lambda x: -x.priority.value)
                
                for service in services_to_migrate[:len(services_to_migrate)//2]:  # Migrate up to half
                    # Deallocate from current server
                    server.deallocate(
                        service.cpu_demand,
                        service.memory_demand,
                        service.service_id
                    )
                    
                    # Try to migrate
                    result = self._migrate_service(service, server)
                    if result.success:
                        migrations_performed += 1
                    else:
                        # Rollback - reallocate to original server
                        server.allocate(
                            service.cpu_demand,
                            service.memory_demand,
                            service.service_id
                        )
                        service.assigned_server = server_id
                        service.assigned_layer = server.layer
                        service.status = "running"
                    
                    # Check if we've reduced load enough
                    new_avg = (server.cpu_utilization + server.memory_utilization) / 2
                    if new_avg <= TARGET_THRESHOLD:
                        break
                
                servers_rebalanced.append(server_id)
        
        return {
            "servers_rebalanced": servers_rebalanced,
            "migrations_performed": migrations_performed,
            "message": f"Rebalanced {len(servers_rebalanced)} servers with {migrations_performed} migrations"
        }
    
    def get_failure_report(self) -> List[Dict]:
        """Get report of all failure events."""
        return [
            {
                "timestamp": event.timestamp.isoformat(),
                "server_id": event.server_id,
                "layer": event.layer,
                "affected_services": len(event.affected_services),
                "migrated_services": len(event.migrated_services),
                "failed_migrations": len(event.failed_migrations),
                "message": event.message
            }
            for event in self.failure_history
        ]
    
    def get_system_health(self) -> Dict:
        """Get overall system health status."""
        total_servers = len(self.service_manager.servers)
        active_servers = len(self.service_manager.get_active_servers())
        failed_servers = total_servers - active_servers
        
        # Calculate average utilization
        active = self.service_manager.get_active_servers()
        if active:
            avg_cpu = sum(s.cpu_utilization for s in active) / len(active)
            avg_mem = sum(s.memory_utilization for s in active) / len(active)
        else:
            avg_cpu = avg_mem = 0
        
        health_status = "healthy"
        if failed_servers > 0:
            health_status = "degraded"
        if failed_servers >= total_servers // 2:
            health_status = "critical"
        
        return {
            "status": health_status,
            "total_servers": total_servers,
            "active_servers": active_servers,
            "failed_servers": failed_servers,
            "avg_cpu_utilization": round(avg_cpu, 2),
            "avg_memory_utilization": round(avg_mem, 2),
            "total_failure_events": len(self.failure_history)
        }


def run_failure_demo():
    """Run a demonstration of failure handling."""
    print("=" * 60)
    print("Failure Handler Demo - Fog Server Failure Scenario")
    print("=" * 60)
    
    # Initialize
    manager = ServiceManager()
    scheduler = TaskScheduler(manager)
    handler = FailureHandler(manager, scheduler)
    
    # Create some services
    print("\n1. Creating services...")
    for _ in range(3):
        service = manager.create_xr_service()
        scheduler.schedule_service(service)
    
    for _ in range(5):
        service = manager.create_embb_service()
        scheduler.schedule_service(service)
    
    # Print initial state
    print("\n2. Initial server load:")
    for load in scheduler.get_server_load_report():
        if load["services_count"] > 0:
            print(f"   Server {load['server_id']} ({load['layer']}): "
                  f"{load['services_count']} services")
    
    # Simulate Fog server failure (server 5)
    print("\n3. Simulating Fog server failure...")
    event = handler.simulate_server_failure(5)
    
    # Print system health
    print("\n4. System health after failure:")
    health = handler.get_system_health()
    for key, value in health.items():
        print(f"   {key}: {value}")
    
    # Recover server
    print("\n5. Recovering failed server...")
    recovery = handler.recover_server(5)
    print(f"   {recovery['message']}")
    
    # Final health check
    print("\n6. Final system health:")
    health = handler.get_system_health()
    for key, value in health.items():
        print(f"   {key}: {value}")
    
    return handler.get_failure_report()


if __name__ == "__main__":
    run_failure_demo()
