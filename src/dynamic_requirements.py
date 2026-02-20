"""
Dynamic Requirements Module
===========================
Handles dynamic changes in service requirements during runtime.

Professor's Requirement #4: Change in Service Requirements
- Increase in throughput requirements
- Increase in number of users
- Re-evaluate resource allocation
- Migrate to more suitable server if needed
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from src.service_manager import (
    Service, Server, ServiceManager, ServiceType, Priority
)
from src.task_scheduler import TaskScheduler, SchedulingResult


class ChangeType(Enum):
    """Types of requirement changes."""
    THROUGHPUT_INCREASE = "throughput_increase"
    THROUGHPUT_DECREASE = "throughput_decrease"
    USER_INCREASE = "user_increase"
    USER_DECREASE = "user_decrease"
    LATENCY_TIGHTER = "latency_tighter"
    LATENCY_RELAXED = "latency_relaxed"


@dataclass
class RequirementChange:
    """Records a requirement change event."""
    timestamp: datetime
    service_id: str
    change_type: ChangeType
    old_value: float
    new_value: float
    old_cpu_demand: int
    new_cpu_demand: int
    old_memory_demand: int
    new_memory_demand: int
    migration_required: bool
    migration_success: Optional[bool]
    new_server_id: Optional[int]
    message: str


@dataclass 
class DynamicUpdateResult:
    """Result of a dynamic requirement update."""
    success: bool
    service_id: str
    change_type: str
    old_server: Optional[int]
    new_server: Optional[int]
    migration_performed: bool
    additional_resources: Dict
    message: str


class DynamicRequirementsHandler:
    """
    Handles dynamic changes in service requirements.
    
    When requirements change:
    1. Recalculate resource demands
    2. Check if current server can accommodate new demands
    3. If not, find a better server and migrate
    4. Update service parameters and allocation
    """
    
    # Thresholds for determining need for migration
    SCALE_UP_THRESHOLD = 0.3  # 30% increase triggers evaluation
    
    def __init__(self, service_manager: ServiceManager, scheduler: TaskScheduler):
        """Initialize dynamic requirements handler."""
        self.service_manager = service_manager
        self.scheduler = scheduler
        self.change_history: List[RequirementChange] = []
    
    def _calculate_new_demands(self, service: Service) -> Tuple[int, int]:
        """
        Recalculate CPU and memory demands based on current parameters.
        
        XR: Higher CPU/memory per Mbps per user
        eMBB: Lower CPU/memory per Mbps per user
        """
        if service.service_type == ServiceType.XR:
            cpu = int(service.throughput_mbps * service.num_users * 2)
            memory = int(service.throughput_mbps * service.num_users * 10)
        else:  # eMBB
            cpu = int(service.throughput_mbps * service.num_users * 0.5)
            memory = int(service.throughput_mbps * service.num_users * 5)
        
        return cpu, memory
    
    def update_throughput(self, 
                          service_id: str, 
                          new_throughput_mbps: float) -> DynamicUpdateResult:
        """
        Update throughput requirement for a service.
        
        Args:
            service_id: Service to update
            new_throughput_mbps: New throughput requirement
        
        Returns:
            DynamicUpdateResult with details of the update
        """
        service = self.service_manager.get_service(service_id)
        if not service:
            return DynamicUpdateResult(
                success=False,
                service_id=service_id,
                change_type="throughput",
                old_server=None,
                new_server=None,
                migration_performed=False,
                additional_resources={},
                message=f"Service {service_id} not found"
            )
        
        old_throughput = service.throughput_mbps
        old_cpu = service.cpu_demand
        old_memory = service.memory_demand
        old_server = service.assigned_server
        
        # Determine change type
        if new_throughput_mbps > old_throughput:
            change_type = ChangeType.THROUGHPUT_INCREASE
        else:
            change_type = ChangeType.THROUGHPUT_DECREASE
        
        # Update throughput and recalculate demands
        service.throughput_mbps = new_throughput_mbps
        new_cpu, new_memory = self._calculate_new_demands(service)
        
        cpu_increase = new_cpu - old_cpu
        memory_increase = new_memory - old_memory
        
        print(f"\n📊 REQUIREMENT CHANGE: Service {service_id}")
        print(f"   Throughput: {old_throughput:.1f} → {new_throughput_mbps:.1f} Mbps")
        print(f"   CPU demand: {old_cpu} → {new_cpu} ({cpu_increase:+d})")
        print(f"   Memory demand: {old_memory} → {new_memory} ({memory_increase:+d})")
        
        # Handle the change
        result = self._handle_resource_change(
            service, old_cpu, old_memory, new_cpu, new_memory, change_type
        )
        
        return result
    
    def update_users(self, 
                     service_id: str, 
                     new_num_users: int) -> DynamicUpdateResult:
        """
        Update number of users for a service.
        
        Args:
            service_id: Service to update
            new_num_users: New number of users
        
        Returns:
            DynamicUpdateResult with details of the update
        """
        service = self.service_manager.get_service(service_id)
        if not service:
            return DynamicUpdateResult(
                success=False,
                service_id=service_id,
                change_type="users",
                old_server=None,
                new_server=None,
                migration_performed=False,
                additional_resources={},
                message=f"Service {service_id} not found"
            )
        
        old_users = service.num_users
        old_cpu = service.cpu_demand
        old_memory = service.memory_demand
        
        # Determine change type
        if new_num_users > old_users:
            change_type = ChangeType.USER_INCREASE
        else:
            change_type = ChangeType.USER_DECREASE
        
        # Update users and recalculate demands
        service.num_users = new_num_users
        new_cpu, new_memory = self._calculate_new_demands(service)
        
        print(f"\n👥 USER CHANGE: Service {service_id}")
        print(f"   Users: {old_users} → {new_num_users}")
        print(f"   CPU demand: {old_cpu} → {new_cpu}")
        print(f"   Memory demand: {old_memory} → {new_memory}")
        
        # Handle the change
        result = self._handle_resource_change(
            service, old_cpu, old_memory, new_cpu, new_memory, change_type
        )
        
        return result
    
    def update_latency_requirement(self, 
                                    service_id: str, 
                                    new_latency_ms: float) -> DynamicUpdateResult:
        """
        Update latency requirement for a service.
        
        Args:
            service_id: Service to update
            new_latency_ms: New maximum latency requirement
        
        Returns:
            DynamicUpdateResult with details of the update
        """
        service = self.service_manager.get_service(service_id)
        if not service:
            return DynamicUpdateResult(
                success=False,
                service_id=service_id,
                change_type="latency",
                old_server=None,
                new_server=None,
                migration_performed=False,
                additional_resources={},
                message=f"Service {service_id} not found"
            )
        
        old_latency = service.latency_ms
        old_server_id = service.assigned_server
        
        # Determine change type
        if new_latency_ms < old_latency:
            change_type = ChangeType.LATENCY_TIGHTER
        else:
            change_type = ChangeType.LATENCY_RELAXED
        
        # Update latency
        service.latency_ms = new_latency_ms
        
        print(f"\n⏱️ LATENCY CHANGE: Service {service_id}")
        print(f"   Max Latency: {old_latency:.1f} → {new_latency_ms:.1f} ms")
        
        # Check if current server still meets latency requirement
        if old_server_id:
            server = self.service_manager.get_server(old_server_id)
            if server and server.base_latency_ms > new_latency_ms:
                print(f"   Current server latency ({server.base_latency_ms}ms) exceeds new requirement")
                
                # Need to migrate to lower latency server
                result = self._migrate_for_latency(service, server)
                
                # Record change
                self._record_change(
                    service, change_type, old_latency, new_latency_ms,
                    service.cpu_demand, service.cpu_demand,
                    service.memory_demand, service.memory_demand,
                    result.migration_performed, result.success, result.new_server
                )
                
                return result
        
        # No migration needed
        self._record_change(
            service, change_type, old_latency, new_latency_ms,
            service.cpu_demand, service.cpu_demand,
            service.memory_demand, service.memory_demand,
            False, True, old_server_id
        )
        
        return DynamicUpdateResult(
            success=True,
            service_id=service_id,
            change_type=change_type.value,
            old_server=old_server_id,
            new_server=old_server_id,
            migration_performed=False,
            additional_resources={},
            message="Latency requirement updated, no migration needed"
        )
    
    def _handle_resource_change(self,
                                 service: Service,
                                 old_cpu: int,
                                 old_memory: int,
                                 new_cpu: int,
                                 new_memory: int,
                                 change_type: ChangeType) -> DynamicUpdateResult:
        """
        Handle a resource demand change.
        
        If demands increased:
        1. Check if current server can accommodate
        2. If not, migrate to a server with more capacity
        
        If demands decreased:
        1. Simply update the allocation
        """
        cpu_diff = new_cpu - old_cpu
        memory_diff = new_memory - old_memory
        
        old_server_id = service.assigned_server
        
        if old_server_id is None:
            # Service not running, just update demands
            service.cpu_demand = new_cpu
            service.memory_demand = new_memory
            
            return DynamicUpdateResult(
                success=True,
                service_id=service.service_id,
                change_type=change_type.value,
                old_server=None,
                new_server=None,
                migration_performed=False,
                additional_resources={"cpu": cpu_diff, "memory": memory_diff},
                message="Service not running, demands updated"
            )
        
        server = self.service_manager.get_server(old_server_id)
        
        if cpu_diff <= 0 and memory_diff <= 0:
            # Demands decreased - just update
            server.used_cpu += cpu_diff  # This will decrease
            server.used_memory += memory_diff
            service.cpu_demand = new_cpu
            service.memory_demand = new_memory
            
            self._record_change(
                service, change_type, 
                service.throughput_mbps, service.throughput_mbps,
                old_cpu, new_cpu, old_memory, new_memory,
                False, True, old_server_id
            )
            
            print(f"   ✓ Resources released on Server {old_server_id}")
            
            return DynamicUpdateResult(
                success=True,
                service_id=service.service_id,
                change_type=change_type.value,
                old_server=old_server_id,
                new_server=old_server_id,
                migration_performed=False,
                additional_resources={"cpu": cpu_diff, "memory": memory_diff},
                message="Resources decreased, updated in place"
            )
        
        # Demands increased - check if server can accommodate
        additional_cpu_needed = max(0, cpu_diff)
        additional_memory_needed = max(0, memory_diff)
        
        if (server.available_cpu >= additional_cpu_needed and 
            server.available_memory >= additional_memory_needed):
            # Current server can accommodate
            server.used_cpu += additional_cpu_needed
            server.used_memory += additional_memory_needed
            service.cpu_demand = new_cpu
            service.memory_demand = new_memory
            
            self._record_change(
                service, change_type,
                service.throughput_mbps, service.throughput_mbps,
                old_cpu, new_cpu, old_memory, new_memory,
                False, True, old_server_id
            )
            
            print(f"   ✓ Additional resources allocated on Server {old_server_id}")
            
            return DynamicUpdateResult(
                success=True,
                service_id=service.service_id,
                change_type=change_type.value,
                old_server=old_server_id,
                new_server=old_server_id,
                migration_performed=False,
                additional_resources={"cpu": additional_cpu_needed, "memory": additional_memory_needed},
                message="Additional resources allocated on current server"
            )
        
        # Need to migrate to a server with more capacity
        print(f"   → Server {old_server_id} cannot accommodate, migrating...")
        
        service.cpu_demand = new_cpu
        service.memory_demand = new_memory
        
        result = self._migrate_for_resources(service, server, new_cpu, new_memory)
        
        self._record_change(
            service, change_type,
            service.throughput_mbps, service.throughput_mbps,
            old_cpu, new_cpu, old_memory, new_memory,
            True, result.success, result.new_server
        )
        
        return result
    
    def _migrate_for_resources(self,
                                service: Service,
                                old_server: Server,
                                new_cpu: int,
                                new_memory: int) -> DynamicUpdateResult:
        """Migrate service to a server with more resources."""
        # Deallocate from old server (using old demands)
        old_server.deallocate(
            service.cpu_demand,
            service.memory_demand,
            service.service_id
        )
        
        # Update demands
        service.cpu_demand = new_cpu
        service.memory_demand = new_memory
        service.assigned_server = None
        service.assigned_layer = None
        service.status = "pending"
        
        # Try to schedule
        result = self.scheduler.schedule_service(service)
        
        if result.success:
            print(f"   ✓ Migrated to Server {result.assigned_server} ({result.assigned_layer})")
            return DynamicUpdateResult(
                success=True,
                service_id=service.service_id,
                change_type="resource_increase",
                old_server=old_server.server_id,
                new_server=result.assigned_server,
                migration_performed=True,
                additional_resources={"cpu": new_cpu, "memory": new_memory},
                message=f"Migrated to Server {result.assigned_server}"
            )
        else:
            # Rollback
            service.status = "running"
            service.assigned_server = old_server.server_id
            service.assigned_layer = old_server.layer
            old_server.allocate(new_cpu, new_memory, service.service_id)
            
            print(f"   ✗ Migration failed, staying on Server {old_server.server_id}")
            return DynamicUpdateResult(
                success=False,
                service_id=service.service_id,
                change_type="resource_increase",
                old_server=old_server.server_id,
                new_server=old_server.server_id,
                migration_performed=False,
                additional_resources={},
                message="Migration failed, no suitable server found"
            )
    
    def _migrate_for_latency(self,
                              service: Service,
                              old_server: Server) -> DynamicUpdateResult:
        """Migrate service to a lower latency server."""
        # Deallocate from old server
        old_server.deallocate(
            service.cpu_demand,
            service.memory_demand,
            service.service_id
        )
        
        service.assigned_server = None
        service.assigned_layer = None
        service.status = "pending"
        
        # Try to schedule (scheduler will respect latency constraint)
        result = self.scheduler.schedule_service(service)
        
        if result.success:
            print(f"   ✓ Migrated to Server {result.assigned_server} ({result.assigned_layer}) "
                  f"with {result.latency_ms}ms latency")
            return DynamicUpdateResult(
                success=True,
                service_id=service.service_id,
                change_type="latency_tighter",
                old_server=old_server.server_id,
                new_server=result.assigned_server,
                migration_performed=True,
                additional_resources={},
                message=f"Migrated to lower latency server {result.assigned_server}"
            )
        else:
            # Rollback
            service.status = "running"
            service.assigned_server = old_server.server_id
            service.assigned_layer = old_server.layer
            old_server.allocate(service.cpu_demand, service.memory_demand, service.service_id)
            
            print(f"   ✗ Migration failed, no server meets latency requirement")
            return DynamicUpdateResult(
                success=False,
                service_id=service.service_id,
                change_type="latency_tighter",
                old_server=old_server.server_id,
                new_server=old_server.server_id,
                migration_performed=False,
                additional_resources={},
                message="No server meets new latency requirement"
            )
    
    def _record_change(self,
                       service: Service,
                       change_type: ChangeType,
                       old_value: float,
                       new_value: float,
                       old_cpu: int,
                       new_cpu: int,
                       old_memory: int,
                       new_memory: int,
                       migration_required: bool,
                       migration_success: bool,
                       new_server_id: Optional[int]):
        """Record a requirement change event."""
        change = RequirementChange(
            timestamp=datetime.now(),
            service_id=service.service_id,
            change_type=change_type,
            old_value=old_value,
            new_value=new_value,
            old_cpu_demand=old_cpu,
            new_cpu_demand=new_cpu,
            old_memory_demand=old_memory,
            new_memory_demand=new_memory,
            migration_required=migration_required,
            migration_success=migration_success if migration_required else None,
            new_server_id=new_server_id,
            message=f"{change_type.value}: {old_value} → {new_value}"
        )
        self.change_history.append(change)
    
    def get_change_report(self) -> List[Dict]:
        """Get report of all requirement changes."""
        return [
            {
                "timestamp": change.timestamp.isoformat(),
                "service_id": change.service_id,
                "change_type": change.change_type.value,
                "old_value": change.old_value,
                "new_value": change.new_value,
                "old_cpu": change.old_cpu_demand,
                "new_cpu": change.new_cpu_demand,
                "old_memory": change.old_memory_demand,
                "new_memory": change.new_memory_demand,
                "migration_required": change.migration_required,
                "migration_success": change.migration_success,
                "new_server_id": change.new_server_id,
                "message": change.message
            }
            for change in self.change_history
        ]
    
    def get_change_statistics(self) -> Dict:
        """Get statistics on requirement changes."""
        total_changes = len(self.change_history)
        migrations_required = sum(1 for c in self.change_history if c.migration_required)
        migrations_successful = sum(1 for c in self.change_history 
                                    if c.migration_required and c.migration_success)
        
        by_type = {}
        for change in self.change_history:
            key = change.change_type.value
            by_type[key] = by_type.get(key, 0) + 1
        
        return {
            "total_changes": total_changes,
            "migrations_required": migrations_required,
            "migrations_successful": migrations_successful,
            "migration_success_rate": (migrations_successful / migrations_required * 100 
                                       if migrations_required > 0 else 100),
            "changes_by_type": by_type
        }


def run_dynamic_requirements_demo():
    """Run a demonstration of dynamic requirements handling."""
    print("=" * 60)
    print("Dynamic Requirements Demo - Throughput & User Changes")
    print("=" * 60)
    
    # Initialize
    manager = ServiceManager()
    scheduler = TaskScheduler(manager)
    dynamic_handler = DynamicRequirementsHandler(manager, scheduler)
    
    # Create and schedule an XR service
    print("\n1. Creating and scheduling XR service...")
    xr_service = manager.create_xr_service(
        throughput_mbps=15,
        num_users=3
    )
    result = scheduler.schedule_service(xr_service)
    print(f"   Scheduled to Server {result.assigned_server} ({result.assigned_layer})")
    print(f"   CPU: {xr_service.cpu_demand}, Memory: {xr_service.memory_demand}")
    
    # Simulate throughput increase
    print("\n2. Simulating throughput increase (15 → 25 Mbps)...")
    update_result = dynamic_handler.update_throughput(xr_service.service_id, 25)
    print(f"   Result: {update_result.message}")
    print(f"   Migration performed: {update_result.migration_performed}")
    
    # Create and schedule an eMBB service
    print("\n3. Creating and scheduling eMBB service...")
    embb_service = manager.create_embb_service(
        throughput_mbps=60,
        num_users=20
    )
    result = scheduler.schedule_service(embb_service)
    print(f"   Scheduled to Server {result.assigned_server} ({result.assigned_layer})")
    print(f"   CPU: {embb_service.cpu_demand}, Memory: {embb_service.memory_demand}")
    
    # Simulate user count increase
    print("\n4. Simulating user count increase (20 → 40 users)...")
    update_result = dynamic_handler.update_users(embb_service.service_id, 40)
    print(f"   Result: {update_result.message}")
    print(f"   New server: {update_result.new_server}")
    
    # Simulate latency requirement tightening
    print("\n5. Simulating tighter latency requirement for XR service...")
    update_result = dynamic_handler.update_latency_requirement(
        xr_service.service_id, 10  # Must be under 10ms
    )
    print(f"   Result: {update_result.message}")
    
    # Print change statistics
    print("\n6. Change statistics:")
    stats = dynamic_handler.get_change_statistics()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    return dynamic_handler.get_change_report()


if __name__ == "__main__":
    run_dynamic_requirements_demo()
