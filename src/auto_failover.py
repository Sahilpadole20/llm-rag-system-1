"""
Automatic Failover System
=========================
Handles automatic migration decisions when nodes fail.

Features:
- Continuous health monitoring
- Automatic failure detection
- Intelligent migration: Edge→Fog, Fog→Cloud, etc.
- Load-aware server selection
- Recovery detection and rebalancing
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from src.service_manager import Service, Server, ServiceManager, ServiceType, Priority
from src.task_scheduler import TaskScheduler
from src.realtime_simulator import RealTimeSimulator, NodeStatus, NetworkCondition


class MigrationReason(Enum):
    """Reason for migration."""
    NODE_FAILURE = "node_failure"
    NODE_DEGRADED = "node_degraded"
    NETWORK_CHANGE = "network_change"
    LOAD_BALANCING = "load_balancing"
    LATENCY_OPTIMIZATION = "latency_optimization"
    MANUAL = "manual"


@dataclass
class MigrationEvent:
    """Records an automatic migration event."""
    timestamp: datetime
    service_id: str
    service_type: str
    source_layer: str
    source_server: int
    target_layer: str
    target_server: int
    reason: MigrationReason
    success: bool
    message: str
    network_condition: Optional[Dict] = None


@dataclass 
class FailoverDecision:
    """Decision made by the failover system."""
    timestamp: datetime
    trigger: str  # What triggered this decision
    affected_services: List[str]
    migrations_attempted: int
    migrations_successful: int
    new_assignments: Dict[str, int]  # service_id -> new_server
    message: str


class AutoFailoverSystem:
    """
    Automatic failover and migration system.
    
    Monitors node health and automatically migrates services when:
    - Node fails completely
    - Node becomes degraded
    - Network conditions change (suggesting better layer)
    - Load imbalance detected
    """
    
    # Migration priorities
    LAYER_FALLBACK = {
        "Edge": ["Fog", "Cloud"],
        "Fog": ["Edge", "Cloud"],
        "Cloud": ["Fog", "Edge"]
    }
    
    def __init__(self, 
                 service_manager: ServiceManager,
                 scheduler: TaskScheduler,
                 simulator: RealTimeSimulator):
        """Initialize the failover system."""
        self.service_manager = service_manager
        self.scheduler = scheduler
        self.simulator = simulator
        
        self.migration_history: List[MigrationEvent] = []
        self.decision_history: List[FailoverDecision] = []
        
        # Sync simulator nodes with service manager
        self._sync_node_status()
    
    def _sync_node_status(self):
        """Sync node status between simulator and service manager."""
        for node_id, health in self.simulator.node_health.items():
            server = self.service_manager.get_server(node_id)
            if server:
                server.is_active = (health.status == NodeStatus.ACTIVE)
    
    def check_and_migrate(self, condition: Optional[NetworkCondition] = None) -> FailoverDecision:
        """
        Main failover check - called periodically.
        
        1. Sync node status
        2. Find services on failed/degraded nodes
        3. Migrate them automatically
        4. Return decision record
        """
        self._sync_node_status()
        
        affected_services = []
        migrations_attempted = 0
        migrations_successful = 0
        new_assignments = {}
        messages = []
        
        # Check each server
        for server_id, server in self.service_manager.servers.items():
            node_health = self.simulator.node_health.get(server_id)
            
            if not node_health:
                continue
            
            # If node is failed, migrate all services
            if node_health.status == NodeStatus.FAILED:
                services_on_node = self._get_services_on_server(server_id)
                
                for service in services_on_node:
                    affected_services.append(service.service_id)
                    migrations_attempted += 1
                    
                    # Find fallback layer
                    result = self._migrate_service(
                        service, 
                        server,
                        MigrationReason.NODE_FAILURE,
                        condition
                    )
                    
                    if result.success:
                        migrations_successful += 1
                        new_assignments[service.service_id] = result.target_server
                        messages.append(f"✓ {service.service_id}: Server {server_id} → {result.target_server}")
                    else:
                        messages.append(f"✗ {service.service_id}: Migration failed - {result.message}")
            
            # If node is degraded, consider migration for critical services
            elif node_health.status == NodeStatus.DEGRADED:
                services_on_node = self._get_services_on_server(server_id)
                critical_services = [s for s in services_on_node 
                                     if s.priority.value <= Priority.HIGH.value]
                
                for service in critical_services:
                    affected_services.append(service.service_id)
                    migrations_attempted += 1
                    
                    result = self._migrate_service(
                        service,
                        server,
                        MigrationReason.NODE_DEGRADED,
                        condition
                    )
                    
                    if result.success:
                        migrations_successful += 1
                        new_assignments[service.service_id] = result.target_server
        
        # Create decision record
        decision = FailoverDecision(
            timestamp=datetime.now(),
            trigger="health_check",
            affected_services=affected_services,
            migrations_attempted=migrations_attempted,
            migrations_successful=migrations_successful,
            new_assignments=new_assignments,
            message=" | ".join(messages) if messages else "No migrations needed"
        )
        
        self.decision_history.append(decision)
        return decision
    
    def _get_services_on_server(self, server_id: int) -> List[Service]:
        """Get all services running on a specific server."""
        return [s for s in self.service_manager.get_running_services()
                if s.assigned_server == server_id]
    
    def _migrate_service(self,
                         service: Service,
                         source_server: Server,
                         reason: MigrationReason,
                         condition: Optional[NetworkCondition] = None) -> MigrationEvent:
        """
        Migrate a service from failed/degraded server to a healthy one.
        
        Migration strategy:
        1. Try same layer first (other servers)
        2. Then try fallback layers based on service type
        3. Respect latency constraints
        """
        source_layer = source_server.layer
        fallback_order = self.LAYER_FALLBACK.get(source_layer, ["Cloud", "Fog", "Edge"])
        
        # Build layer priority: same layer first, then fallbacks
        layer_priority = [source_layer] + fallback_order
        
        # Adjust based on service type
        if service.service_type == ServiceType.XR:
            # XR prefers low latency: Edge > Fog > Cloud
            layer_priority = ["Edge", "Fog", "Cloud"]
        elif service.service_type == ServiceType.EMBB:
            # eMBB prefers capacity: Cloud > Fog > Edge
            layer_priority = ["Cloud", "Fog", "Edge"]
        
        # Deallocate from source
        source_server.deallocate(
            service.cpu_demand,
            service.memory_demand,
            service.service_id
        )
        
        # Try each layer
        for target_layer in layer_priority:
            servers = self.service_manager.get_servers_by_layer(target_layer)
            
            # Sort by utilization (prefer least loaded)
            servers = sorted(servers, 
                           key=lambda s: (s.cpu_utilization + s.memory_utilization) / 2)
            
            for target_server in servers:
                # Skip source server or inactive servers
                if target_server.server_id == source_server.server_id:
                    continue
                
                if not target_server.is_active:
                    continue
                
                # Check capacity
                if not target_server.can_accommodate(service.cpu_demand, service.memory_demand):
                    continue
                
                # Check latency constraint
                if target_server.base_latency_ms > service.latency_ms:
                    continue
                
                # Allocate to new server
                success = target_server.allocate(
                    service.cpu_demand,
                    service.memory_demand,
                    service.service_id
                )
                
                if success:
                    service.assigned_layer = target_layer
                    service.assigned_server = target_server.server_id
                    service.status = "running"
                    
                    event = MigrationEvent(
                        timestamp=datetime.now(),
                        service_id=service.service_id,
                        service_type=service.service_type.value,
                        source_layer=source_layer,
                        source_server=source_server.server_id,
                        target_layer=target_layer,
                        target_server=target_server.server_id,
                        reason=reason,
                        success=True,
                        message=f"Migrated from Server {source_server.server_id} to {target_server.server_id}",
                        network_condition=condition.to_dict() if condition else None
                    )
                    
                    self.migration_history.append(event)
                    return event
        
        # Migration failed - rollback
        source_server.allocate(
            service.cpu_demand,
            service.memory_demand,
            service.service_id
        )
        service.status = "running"
        
        event = MigrationEvent(
            timestamp=datetime.now(),
            service_id=service.service_id,
            service_type=service.service_type.value,
            source_layer=source_layer,
            source_server=source_server.server_id,
            target_layer=None,
            target_server=None,
            reason=reason,
            success=False,
            message="No suitable server found",
            network_condition=condition.to_dict() if condition else None
        )
        
        self.migration_history.append(event)
        return event
    
    def handle_network_change(self, condition: NetworkCondition) -> FailoverDecision:
        """
        React to network condition changes.
        
        If network suggests a different layer is better, migrate services.
        """
        recommended_layer, reason, available_servers = self.simulator.get_recommended_layer(condition)
        
        affected_services = []
        migrations_attempted = 0
        migrations_successful = 0
        new_assignments = {}
        
        # Check if any running service should move
        for service in self.service_manager.get_running_services():
            current_layer = service.assigned_layer
            
            # Only migrate if recommended layer is different and better
            if current_layer != recommended_layer and recommended_layer != "None":
                # Check if migration would actually help
                if self._should_migrate_for_network(service, current_layer, recommended_layer, condition):
                    affected_services.append(service.service_id)
                    migrations_attempted += 1
                    
                    current_server = self.service_manager.get_server(service.assigned_server)
                    if current_server:
                        result = self._migrate_service(
                            service,
                            current_server,
                            MigrationReason.NETWORK_CHANGE,
                            condition
                        )
                        
                        if result.success:
                            migrations_successful += 1
                            new_assignments[service.service_id] = result.target_server
        
        decision = FailoverDecision(
            timestamp=datetime.now(),
            trigger=f"network_change: {reason}",
            affected_services=affected_services,
            migrations_attempted=migrations_attempted,
            migrations_successful=migrations_successful,
            new_assignments=new_assignments,
            message=f"Network suggests {recommended_layer}: {migrations_successful}/{migrations_attempted} migrated"
        )
        
        self.decision_history.append(decision)
        return decision
    
    def _should_migrate_for_network(self,
                                     service: Service,
                                     current_layer: str,
                                     recommended_layer: str,
                                     condition: NetworkCondition) -> bool:
        """Determine if migration is worth it based on network change."""
        # XR services should chase low latency
        if service.service_type == ServiceType.XR:
            if recommended_layer == "Edge" and current_layer != "Edge":
                return condition.latency_ms < 15  # Only if latency really low
        
        # eMBB services should chase bandwidth
        if service.service_type == ServiceType.EMBB:
            if recommended_layer == "Cloud" and current_layer != "Cloud":
                return condition.datarate_mbps > 20  # Only if bandwidth really high
        
        return False
    
    def get_migration_report(self) -> List[Dict]:
        """Get migration history as list of dicts."""
        return [
            {
                "timestamp": e.timestamp.isoformat(),
                "service_id": e.service_id,
                "service_type": e.service_type,
                "source": f"{e.source_layer}/Server{e.source_server}",
                "target": f"{e.target_layer}/Server{e.target_server}" if e.success else "FAILED",
                "reason": e.reason.value,
                "success": e.success,
                "message": e.message
            }
            for e in self.migration_history[-20:]  # Last 20 migrations
        ]
    
    def get_decision_report(self) -> List[Dict]:
        """Get decision history as list of dicts."""
        return [
            {
                "timestamp": d.timestamp.isoformat(),
                "trigger": d.trigger,
                "affected": len(d.affected_services),
                "attempted": d.migrations_attempted,
                "successful": d.migrations_successful,
                "message": d.message
            }
            for d in self.decision_history[-20:]
        ]
    
    def get_statistics(self) -> Dict:
        """Get failover statistics."""
        total_migrations = len(self.migration_history)
        successful = sum(1 for m in self.migration_history if m.success)
        
        by_reason = {}
        for m in self.migration_history:
            key = m.reason.value
            by_reason[key] = by_reason.get(key, 0) + 1
        
        by_layer = {"Edge": 0, "Fog": 0, "Cloud": 0}
        for m in self.migration_history:
            if m.success and m.target_layer:
                by_layer[m.target_layer] = by_layer.get(m.target_layer, 0) + 1
        
        return {
            "total_migrations": total_migrations,
            "successful_migrations": successful,
            "success_rate": (successful / total_migrations * 100) if total_migrations > 0 else 100,
            "migrations_by_reason": by_reason,
            "migrations_to_layer": by_layer,
            "total_decisions": len(self.decision_history)
        }
