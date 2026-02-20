"""
Priority Preemption Module
==========================
Handles higher priority service preemption when servers are overloaded.

Professor's Requirement #3: Higher Priority Services (Preemption)
- When higher priority service arrives
- Server utilization exceeds 80%
- Lower priority services get preempted
- Higher priority service gets scheduled
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from src.service_manager import (
    Service, Server, ServiceManager, ServiceType, Priority
)
from src.task_scheduler import TaskScheduler, SchedulingResult


@dataclass
class PreemptionEvent:
    """Records a preemption event."""
    timestamp: datetime
    preempted_service_id: str
    preempted_service_type: str
    preempted_priority: int
    new_service_id: str
    new_service_type: str
    new_priority: int
    server_id: int
    layer: str
    message: str


@dataclass
class PreemptionResult:
    """Result of a preemption operation."""
    success: bool
    new_service_scheduled: bool
    preempted_services: List[str]
    rescheduled_services: List[str]
    failed_reschedules: List[str]
    message: str


class PriorityPreemptionHandler:
    """
    Handles priority-based preemption of services.
    
    When a high-priority service arrives and servers are overloaded (>80%):
    1. Identify lower priority services that can be preempted
    2. Preempt enough services to make room
    3. Schedule the high-priority service
    4. Attempt to reschedule preempted services on other servers
    """
    
    UTILIZATION_THRESHOLD = 80.0  # 80% as per professor's requirement
    
    def __init__(self, service_manager: ServiceManager, scheduler: TaskScheduler):
        """Initialize preemption handler."""
        self.service_manager = service_manager
        self.scheduler = scheduler
        self.preemption_history: List[PreemptionEvent] = []
    
    def _get_preemptable_services(self, 
                                   server: Server, 
                                   new_priority: Priority) -> List[Service]:
        """
        Get services that can be preempted by a higher priority service.
        
        Returns services sorted by priority (lowest first).
        """
        preemptable = []
        
        for service_id in server.running_services:
            service = self.service_manager.get_service(service_id)
            if service and service.priority.value > new_priority.value:
                preemptable.append(service)
        
        # Sort by priority (lowest priority first = highest value)
        preemptable.sort(key=lambda x: -x.priority.value)
        
        return preemptable
    
    def _can_preempt_for_resources(self,
                                    server: Server,
                                    cpu_needed: int,
                                    memory_needed: int,
                                    new_priority: Priority) -> Tuple[bool, List[Service]]:
        """
        Check if we can free enough resources through preemption.
        
        Returns (can_preempt, list_of_services_to_preempt)
        """
        preemptable = self._get_preemptable_services(server, new_priority)
        
        if not preemptable:
            return False, []
        
        # Calculate how much we need to free
        cpu_deficit = cpu_needed - server.available_cpu
        memory_deficit = memory_needed - server.available_memory
        
        if cpu_deficit <= 0 and memory_deficit <= 0:
            return True, []  # No preemption needed
        
        # Select services to preempt
        services_to_preempt = []
        freed_cpu = 0
        freed_memory = 0
        
        for service in preemptable:
            services_to_preempt.append(service)
            freed_cpu += service.cpu_demand
            freed_memory += service.memory_demand
            
            if freed_cpu >= cpu_deficit and freed_memory >= memory_deficit:
                return True, services_to_preempt
        
        # Not enough preemptable services
        return False, []
    
    def try_schedule_with_preemption(self, service: Service) -> PreemptionResult:
        """
        Try to schedule a service, using preemption if necessary.
        
        Algorithm:
        1. First try normal scheduling
        2. If fails and service is high priority, try preemption
        3. Find server where preemption can free enough resources
        4. Preempt lower priority services
        5. Schedule the new service
        6. Try to reschedule preempted services elsewhere
        """
        # First try normal scheduling
        result = self.scheduler.schedule_service(service)
        
        if result.success:
            return PreemptionResult(
                success=True,
                new_service_scheduled=True,
                preempted_services=[],
                rescheduled_services=[],
                failed_reschedules=[],
                message="Scheduled without preemption"
            )
        
        # Check if preemption is warranted (only for HIGH or CRITICAL priority)
        if service.priority.value > Priority.HIGH.value:
            return PreemptionResult(
                success=False,
                new_service_scheduled=False,
                preempted_services=[],
                rescheduled_services=[],
                failed_reschedules=[],
                message="Service priority too low for preemption"
            )
        
        # Get layer priority based on service type
        if service.service_type == ServiceType.XR:
            layer_priority = ["Edge", "Fog", "Cloud"]
        else:
            layer_priority = ["Cloud", "Fog", "Edge"]
        
        # Try each layer
        for layer in layer_priority:
            servers = self.service_manager.get_servers_by_layer(layer)
            
            for server in servers:
                if not server.is_active:
                    continue
                
                # Check latency constraint
                if server.base_latency_ms > service.latency_ms:
                    continue
                
                # Check if utilization is above threshold
                avg_util = (server.cpu_utilization + server.memory_utilization) / 2
                
                if avg_util >= self.UTILIZATION_THRESHOLD:
                    # Check if preemption can help
                    can_preempt, to_preempt = self._can_preempt_for_resources(
                        server,
                        service.cpu_demand,
                        service.memory_demand,
                        service.priority
                    )
                    
                    if can_preempt and to_preempt:
                        # Perform preemption
                        return self._perform_preemption(
                            service, server, to_preempt
                        )
        
        # Could not schedule even with preemption
        service.status = "failed"
        return PreemptionResult(
            success=False,
            new_service_scheduled=False,
            preempted_services=[],
            rescheduled_services=[],
            failed_reschedules=[],
            message="Could not schedule service (no preemption candidates)"
        )
    
    def _perform_preemption(self,
                            new_service: Service,
                            server: Server,
                            services_to_preempt: List[Service]) -> PreemptionResult:
        """
        Perform the actual preemption.
        
        1. Deallocate preempted services
        2. Allocate new service
        3. Try to reschedule preempted services
        """
        preempted_ids = []
        rescheduled_ids = []
        failed_reschedule_ids = []
        
        print(f"\n⚡ PREEMPTION: Server {server.server_id} ({server.layer})")
        print(f"   New service: {new_service.service_type.value} (Priority {new_service.priority.value})")
        
        # Preempt services
        for service in services_to_preempt:
            # Deallocate resources
            server.deallocate(
                service.cpu_demand,
                service.memory_demand,
                service.service_id
            )
            service.status = "preempted"
            service.assigned_server = None
            service.assigned_layer = None
            preempted_ids.append(service.service_id)
            
            # Record preemption event
            event = PreemptionEvent(
                timestamp=datetime.now(),
                preempted_service_id=service.service_id,
                preempted_service_type=service.service_type.value,
                preempted_priority=service.priority.value,
                new_service_id=new_service.service_id,
                new_service_type=new_service.service_type.value,
                new_priority=new_service.priority.value,
                server_id=server.server_id,
                layer=server.layer,
                message=f"Preempted for higher priority {new_service.service_type.value} service"
            )
            self.preemption_history.append(event)
            
            print(f"   → Preempted: {service.service_type.value} service {service.service_id} "
                  f"(Priority {service.priority.value})")
        
        # Allocate new service
        success = server.allocate(
            new_service.cpu_demand,
            new_service.memory_demand,
            new_service.service_id
        )
        
        if success:
            new_service.assigned_server = server.server_id
            new_service.assigned_layer = server.layer
            new_service.status = "running"
            print(f"   ✓ Scheduled: {new_service.service_type.value} service {new_service.service_id}")
        else:
            # Rollback - shouldn't happen but handle gracefully
            new_service.status = "failed"
            print(f"   ✗ Failed to schedule new service (unexpected)")
        
        # Try to reschedule preempted services
        for service in services_to_preempt:
            service.status = "pending"
            result = self.scheduler.schedule_service(service)
            
            if result.success:
                rescheduled_ids.append(service.service_id)
                print(f"   ↻ Rescheduled: {service.service_id} to Server {result.assigned_server}")
            else:
                failed_reschedule_ids.append(service.service_id)
                service.status = "preempted"
                print(f"   ✗ Failed to reschedule: {service.service_id}")
        
        return PreemptionResult(
            success=success,
            new_service_scheduled=success,
            preempted_services=preempted_ids,
            rescheduled_services=rescheduled_ids,
            failed_reschedules=failed_reschedule_ids,
            message=f"Preempted {len(preempted_ids)} services, rescheduled {len(rescheduled_ids)}"
        )
    
    def get_preemption_report(self) -> List[Dict]:
        """Get report of all preemption events."""
        return [
            {
                "timestamp": event.timestamp.isoformat(),
                "preempted_service_id": event.preempted_service_id,
                "preempted_type": event.preempted_service_type,
                "preempted_priority": event.preempted_priority,
                "new_service_id": event.new_service_id,
                "new_type": event.new_service_type,
                "new_priority": event.new_priority,
                "server_id": event.server_id,
                "layer": event.layer,
                "message": event.message
            }
            for event in self.preemption_history
        ]
    
    def get_preemption_statistics(self) -> Dict:
        """Get preemption statistics."""
        total_preemptions = len(self.preemption_history)
        
        by_layer = {"Edge": 0, "Fog": 0, "Cloud": 0}
        by_type = {"XR": 0, "eMBB": 0}
        
        for event in self.preemption_history:
            by_layer[event.layer] = by_layer.get(event.layer, 0) + 1
            by_type[event.preempted_service_type] = by_type.get(event.preempted_service_type, 0) + 1
        
        return {
            "total_preemptions": total_preemptions,
            "preemptions_by_layer": by_layer,
            "preempted_by_type": by_type
        }


def run_preemption_demo():
    """Run a demonstration of preemption handling."""
    print("=" * 60)
    print("Priority Preemption Demo - 80% Utilization Threshold")
    print("=" * 60)
    
    # Initialize
    manager = ServiceManager()
    scheduler = TaskScheduler(manager)
    preemption_handler = PriorityPreemptionHandler(manager, scheduler)
    
    # Fill up servers with low priority eMBB services
    print("\n1. Filling servers with low priority eMBB services...")
    for _ in range(8):
        service = manager.create_embb_service(priority=Priority.LOW)
        scheduler.schedule_service(service)
    
    # Print server load
    print("\n2. Current server utilization:")
    for load in scheduler.get_server_load_report():
        if load["services_count"] > 0:
            print(f"   Server {load['server_id']} ({load['layer']}): "
                  f"CPU {load['cpu_util_%']}%, "
                  f"Services: {load['services_count']}")
    
    # Create a high priority XR service
    print("\n3. Creating HIGH priority XR service...")
    critical_xr = manager.create_xr_service(priority=Priority.CRITICAL)
    critical_xr.cpu_demand = 10000  # Large demand
    critical_xr.memory_demand = 20000
    
    # Try to schedule with preemption
    print("\n4. Attempting to schedule with preemption...")
    result = preemption_handler.try_schedule_with_preemption(critical_xr)
    
    print(f"\n5. Result: {result.message}")
    print(f"   Service scheduled: {result.new_service_scheduled}")
    print(f"   Preempted: {len(result.preempted_services)}")
    print(f"   Rescheduled: {len(result.rescheduled_services)}")
    print(f"   Failed reschedules: {len(result.failed_reschedules)}")
    
    # Print preemption statistics
    print("\n6. Preemption statistics:")
    stats = preemption_handler.get_preemption_statistics()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    return preemption_handler.get_preemption_report()


if __name__ == "__main__":
    run_preemption_demo()
