"""
Service Manager Module
======================
Manages XR (Extended Reality) and eMBB (enhanced Mobile Broadband) services
for Edge-Fog-Cloud deployment system.

Professor's Requirements:
- XR Services: 15-20 Mbps throughput, 5-20ms latency, 2-5 users
- eMBB Services: 50-100 Mbps throughput, 50-200ms latency, 10-50 users
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import random
import uuid
from datetime import datetime


class ServiceType(Enum):
    """Service types supported by the system."""
    XR = "XR"      # Extended Reality - high priority, low latency
    EMBB = "eMBB"  # Enhanced Mobile Broadband - high throughput


class Priority(Enum):
    """Priority levels for services."""
    CRITICAL = 1    # Must be served, preempt others
    HIGH = 2        # High priority XR services
    MEDIUM = 3      # Standard eMBB services
    LOW = 4         # Best-effort services


@dataclass
class Service:
    """
    Represents a service (XR or eMBB) with its requirements.
    
    Attributes:
        service_id: Unique identifier
        service_type: XR or eMBB
        priority: Priority level (1=highest, 4=lowest)
        throughput_mbps: Required throughput in Mbps
        latency_ms: Maximum acceptable latency in ms
        num_users: Number of concurrent users
        cpu_demand: CPU units required
        memory_demand: Memory units required
        created_at: Service creation timestamp
        assigned_layer: Assigned deployment layer (Edge/Fog/Cloud)
        assigned_server: Assigned server ID
    """
    service_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    service_type: ServiceType = ServiceType.EMBB
    priority: Priority = Priority.MEDIUM
    throughput_mbps: float = 50.0
    latency_ms: float = 100.0
    num_users: int = 10
    cpu_demand: int = 50
    memory_demand: int = 200
    created_at: datetime = field(default_factory=datetime.now)
    assigned_layer: Optional[str] = None
    assigned_server: Optional[int] = None
    status: str = "pending"  # pending, running, preempted, completed, failed
    
    def __post_init__(self):
        """Validate service parameters after initialization."""
        if self.service_type == ServiceType.XR:
            # XR services: 15-20 Mbps, 5-20ms latency, 2-5 users
            if not (15 <= self.throughput_mbps <= 25):
                self.throughput_mbps = random.uniform(15, 20)
            if not (5 <= self.latency_ms <= 20):
                self.latency_ms = random.uniform(5, 20)
            if not (2 <= self.num_users <= 5):
                self.num_users = random.randint(2, 5)
            # XR typically has higher priority
            if self.priority == Priority.MEDIUM:
                self.priority = Priority.HIGH
        else:
            # eMBB services: 50-100 Mbps, 50-200ms latency, 10-50 users
            if not (50 <= self.throughput_mbps <= 100):
                self.throughput_mbps = random.uniform(50, 100)
            if not (50 <= self.latency_ms <= 200):
                self.latency_ms = random.uniform(50, 200)
            if not (10 <= self.num_users <= 50):
                self.num_users = random.randint(10, 50)
    
    def get_resource_requirements(self) -> Dict:
        """Calculate resource requirements based on service parameters."""
        base_cpu = self.throughput_mbps * 2  # 2 CPU units per Mbps
        base_memory = self.throughput_mbps * 10  # 10 MB per Mbps
        
        # Scale by number of users
        user_factor = self.num_users / 10
        
        return {
            "cpu": int(base_cpu * user_factor),
            "memory": int(base_memory * user_factor),
            "throughput_mbps": self.throughput_mbps,
            "max_latency_ms": self.latency_ms
        }
    
    def to_dict(self) -> Dict:
        """Convert service to dictionary representation."""
        return {
            "service_id": self.service_id,
            "service_type": self.service_type.value,
            "priority": self.priority.value,
            "throughput_mbps": self.throughput_mbps,
            "latency_ms": self.latency_ms,
            "num_users": self.num_users,
            "cpu_demand": self.cpu_demand,
            "memory_demand": self.memory_demand,
            "assigned_layer": self.assigned_layer,
            "assigned_server": self.assigned_server,
            "status": self.status,
            "created_at": self.created_at.isoformat()
        }


@dataclass
class Server:
    """
    Represents a server in the Edge-Fog-Cloud infrastructure.
    
    Attributes:
        server_id: Unique server identifier
        layer: Edge, Fog, or Cloud
        total_cpu: Total CPU capacity
        total_memory: Total memory capacity
        used_cpu: Currently used CPU
        used_memory: Currently used memory
        base_latency_ms: Network latency to this server
        cost_per_hour: Operational cost
        running_services: List of running service IDs
    """
    server_id: int
    layer: str
    total_cpu: int
    total_memory: int
    used_cpu: int = 0
    used_memory: int = 0
    base_latency_ms: float = 50.0
    cost_per_hour: float = 1.0
    running_services: List[str] = field(default_factory=list)
    is_active: bool = True
    
    @property
    def available_cpu(self) -> int:
        """Return available CPU capacity."""
        return self.total_cpu - self.used_cpu
    
    @property
    def available_memory(self) -> int:
        """Return available memory capacity."""
        return self.total_memory - self.used_memory
    
    @property
    def cpu_utilization(self) -> float:
        """Return CPU utilization percentage."""
        return (self.used_cpu / self.total_cpu) * 100 if self.total_cpu > 0 else 0
    
    @property
    def memory_utilization(self) -> float:
        """Return memory utilization percentage."""
        return (self.used_memory / self.total_memory) * 100 if self.total_memory > 0 else 0
    
    def can_accommodate(self, cpu_required: int, memory_required: int) -> bool:
        """Check if server can accommodate resource requirements."""
        return (self.is_active and 
                self.available_cpu >= cpu_required and 
                self.available_memory >= memory_required)
    
    def allocate(self, cpu: int, memory: int, service_id: str) -> bool:
        """Allocate resources to a service."""
        if not self.can_accommodate(cpu, memory):
            return False
        self.used_cpu += cpu
        self.used_memory += memory
        self.running_services.append(service_id)
        return True
    
    def deallocate(self, cpu: int, memory: int, service_id: str) -> bool:
        """Deallocate resources from a service."""
        self.used_cpu = max(0, self.used_cpu - cpu)
        self.used_memory = max(0, self.used_memory - memory)
        if service_id in self.running_services:
            self.running_services.remove(service_id)
        return True
    
    def to_dict(self) -> Dict:
        """Convert server to dictionary representation."""
        return {
            "server_id": self.server_id,
            "layer": self.layer,
            "total_cpu": self.total_cpu,
            "total_memory": self.total_memory,
            "used_cpu": self.used_cpu,
            "used_memory": self.used_memory,
            "available_cpu": self.available_cpu,
            "available_memory": self.available_memory,
            "cpu_utilization": round(self.cpu_utilization, 2),
            "memory_utilization": round(self.memory_utilization, 2),
            "base_latency_ms": self.base_latency_ms,
            "cost_per_hour": self.cost_per_hour,
            "running_services": self.running_services.copy(),
            "is_active": self.is_active
        }


class ServiceManager:
    """
    Manages services and server infrastructure.
    
    Handles service creation, server management, and resource tracking.
    """
    
    def __init__(self):
        """Initialize the service manager with default infrastructure."""
        self.services: Dict[str, Service] = {}
        self.servers: Dict[int, Server] = {}
        self._initialize_infrastructure()
    
    def _initialize_infrastructure(self):
        """Initialize servers based on paper's infrastructure."""
        # Edge servers (1-4): 16K CPU, 32K Memory
        for i in range(1, 5):
            self.servers[i] = Server(
                server_id=i,
                layer="Edge",
                total_cpu=16000,
                total_memory=32000,
                base_latency_ms=5.0,
                cost_per_hour=0.50
            )
        
        # Fog servers (5-6): 64K CPU, 128K Memory
        for i in range(5, 7):
            self.servers[i] = Server(
                server_id=i,
                layer="Fog",
                total_cpu=64000,
                total_memory=128000,
                base_latency_ms=25.0,
                cost_per_hour=1.00
            )
        
        # Cloud server (7): 200K CPU, 512K Memory
        self.servers[7] = Server(
            server_id=7,
            layer="Cloud",
            total_cpu=200000,
            total_memory=512000,
            base_latency_ms=100.0,
            cost_per_hour=5.00
        )
    
    def create_xr_service(self, 
                          throughput_mbps: float = None,
                          latency_ms: float = None,
                          num_users: int = None,
                          priority: Priority = Priority.HIGH) -> Service:
        """Create a new XR service with given or random parameters."""
        service = Service(
            service_type=ServiceType.XR,
            priority=priority,
            throughput_mbps=throughput_mbps or random.uniform(15, 20),
            latency_ms=latency_ms or random.uniform(5, 20),
            num_users=num_users or random.randint(2, 5)
        )
        service.cpu_demand = int(service.throughput_mbps * service.num_users * 2)
        service.memory_demand = int(service.throughput_mbps * service.num_users * 10)
        self.services[service.service_id] = service
        return service
    
    def create_embb_service(self,
                            throughput_mbps: float = None,
                            latency_ms: float = None,
                            num_users: int = None,
                            priority: Priority = Priority.MEDIUM) -> Service:
        """Create a new eMBB service with given or random parameters."""
        service = Service(
            service_type=ServiceType.EMBB,
            priority=priority,
            throughput_mbps=throughput_mbps or random.uniform(50, 100),
            latency_ms=latency_ms or random.uniform(50, 200),
            num_users=num_users or random.randint(10, 50)
        )
        service.cpu_demand = int(service.throughput_mbps * service.num_users * 0.5)
        service.memory_demand = int(service.throughput_mbps * service.num_users * 5)
        self.services[service.service_id] = service
        return service
    
    def get_service(self, service_id: str) -> Optional[Service]:
        """Get service by ID."""
        return self.services.get(service_id)
    
    def get_server(self, server_id: int) -> Optional[Server]:
        """Get server by ID."""
        return self.servers.get(server_id)
    
    def get_servers_by_layer(self, layer: str) -> List[Server]:
        """Get all servers in a specific layer."""
        return [s for s in self.servers.values() if s.layer == layer]
    
    def get_active_servers(self) -> List[Server]:
        """Get all active servers."""
        return [s for s in self.servers.values() if s.is_active]
    
    def get_pending_services(self) -> List[Service]:
        """Get all pending services sorted by priority."""
        pending = [s for s in self.services.values() if s.status == "pending"]
        return sorted(pending, key=lambda x: (x.priority.value, x.created_at))
    
    def get_running_services(self) -> List[Service]:
        """Get all running services."""
        return [s for s in self.services.values() if s.status == "running"]
    
    def get_layer_utilization(self, layer: str) -> Dict:
        """Get aggregate utilization for a layer."""
        servers = self.get_servers_by_layer(layer)
        if not servers:
            return {"cpu": 0, "memory": 0}
        
        total_cpu = sum(s.total_cpu for s in servers)
        used_cpu = sum(s.used_cpu for s in servers)
        total_memory = sum(s.total_memory for s in servers)
        used_memory = sum(s.used_memory for s in servers)
        
        return {
            "cpu_percent": (used_cpu / total_cpu * 100) if total_cpu > 0 else 0,
            "memory_percent": (used_memory / total_memory * 100) if total_memory > 0 else 0,
            "total_cpu": total_cpu,
            "used_cpu": used_cpu,
            "total_memory": total_memory,
            "used_memory": used_memory
        }
    
    def reset_all(self):
        """Reset all services and server allocations."""
        self.services.clear()
        for server in self.servers.values():
            server.used_cpu = 0
            server.used_memory = 0
            server.running_services.clear()
            server.is_active = True
    
    def get_statistics(self) -> Dict:
        """Get overall system statistics."""
        running = self.get_running_services()
        pending = self.get_pending_services()
        
        xr_count = len([s for s in running if s.service_type == ServiceType.XR])
        embb_count = len([s for s in running if s.service_type == ServiceType.EMBB])
        
        return {
            "total_services": len(self.services),
            "running_services": len(running),
            "pending_services": len(pending),
            "xr_services_running": xr_count,
            "embb_services_running": embb_count,
            "edge_utilization": self.get_layer_utilization("Edge"),
            "fog_utilization": self.get_layer_utilization("Fog"),
            "cloud_utilization": self.get_layer_utilization("Cloud"),
            "active_servers": len(self.get_active_servers()),
            "total_servers": len(self.servers)
        }
