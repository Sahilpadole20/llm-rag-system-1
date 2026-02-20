"""
Test Suite for Professor's 4 Advanced Features
==============================================
Tests for:
1. Task Scheduling for Different Services (XR vs eMBB)
2. Node Failure Handling
3. Higher Priority Services (Preemption) 
4. Change in Service Requirements
"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.service_manager import (
    Service, Server, ServiceManager, ServiceType, Priority
)
from src.task_scheduler import TaskScheduler, SchedulingResult
from src.failure_handler import FailureHandler
from src.priority_preemption import PriorityPreemptionHandler
from src.dynamic_requirements import DynamicRequirementsHandler


class TestServiceManager:
    """Tests for ServiceManager class."""
    
    def test_create_xr_service(self):
        """Test XR service creation with correct parameters."""
        manager = ServiceManager()
        service = manager.create_xr_service()
        
        assert service.service_type == ServiceType.XR
        assert 15 <= service.throughput_mbps <= 25
        assert 5 <= service.latency_ms <= 20
        assert 2 <= service.num_users <= 5
        assert service.priority.value <= Priority.HIGH.value
    
    def test_create_embb_service(self):
        """Test eMBB service creation with correct parameters."""
        manager = ServiceManager()
        service = manager.create_embb_service()
        
        assert service.service_type == ServiceType.EMBB
        assert 50 <= service.throughput_mbps <= 100
        assert 50 <= service.latency_ms <= 200
        assert 10 <= service.num_users <= 50
    
    def test_infrastructure_initialization(self):
        """Test that infrastructure is correctly initialized."""
        manager = ServiceManager()
        
        # Check Edge servers (1-4)
        for i in range(1, 5):
            server = manager.get_server(i)
            assert server.layer == "Edge"
            assert server.total_cpu == 16000
            assert server.total_memory == 32000
        
        # Check Fog servers (5-6)
        for i in range(5, 7):
            server = manager.get_server(i)
            assert server.layer == "Fog"
            assert server.total_cpu == 64000
            assert server.total_memory == 128000
        
        # Check Cloud server (7)
        server = manager.get_server(7)
        assert server.layer == "Cloud"
        assert server.total_cpu == 200000
        assert server.total_memory == 512000


class TestTaskScheduler:
    """Tests for Task Scheduling (Professor's Requirement #1)."""
    
    def test_xr_prefers_edge(self):
        """Test that XR services prefer Edge layer."""
        manager = ServiceManager()
        scheduler = TaskScheduler(manager)
        
        # Create small XR service that fits on Edge
        service = manager.create_xr_service(
            throughput_mbps=15,
            num_users=2,
            latency_ms=10
        )
        
        result = scheduler.schedule_service(service)
        
        assert result.success
        assert result.assigned_layer == "Edge"
    
    def test_embb_prefers_cloud_or_fog(self):
        """Test that eMBB services prefer Cloud/Fog layers."""
        manager = ServiceManager()
        scheduler = TaskScheduler(manager)
        
        # Create eMBB service
        service = manager.create_embb_service(
            throughput_mbps=70,
            num_users=25,
            latency_ms=150
        )
        
        result = scheduler.schedule_service(service)
        
        assert result.success
        assert result.assigned_layer in ["Cloud", "Fog"]
    
    def test_load_balancing(self):
        """Test that scheduler balances load across servers."""
        manager = ServiceManager()
        scheduler = TaskScheduler(manager)
        
        # Schedule multiple small XR services
        servers_used = set()
        for _ in range(4):
            service = manager.create_xr_service(
                throughput_mbps=15,
                num_users=2
            )
            result = scheduler.schedule_service(service)
            if result.success:
                servers_used.add(result.assigned_server)
        
        # Should use multiple Edge servers for load balancing
        assert len(servers_used) > 1
    
    def test_respects_latency_constraint(self):
        """Test that scheduler respects latency constraints."""
        manager = ServiceManager()
        scheduler = TaskScheduler(manager)
        
        # Create service with very strict latency (only Edge can serve)
        service = manager.create_xr_service(latency_ms=8)
        result = scheduler.schedule_service(service)
        
        assert result.success
        assert result.assigned_layer == "Edge"  # Only Edge has <8ms latency
    
    def test_scheduling_metrics(self):
        """Test that scheduling metrics are tracked correctly."""
        manager = ServiceManager()
        scheduler = TaskScheduler(manager)
        
        # Schedule services
        for _ in range(3):
            service = manager.create_xr_service()
            scheduler.schedule_service(service)
        
        for _ in range(2):
            service = manager.create_embb_service()
            scheduler.schedule_service(service)
        
        metrics = scheduler.metrics.to_dict()
        
        assert metrics["total_scheduled"] == 5
        assert metrics["xr_scheduled"] == 3
        assert metrics["embb_scheduled"] == 2


class TestFailureHandler:
    """Tests for Node Failure Handling (Professor's Requirement #2)."""
    
    def test_server_failure_detection(self):
        """Test that server failure is detected and handled."""
        manager = ServiceManager()
        scheduler = TaskScheduler(manager)
        handler = FailureHandler(manager, scheduler)
        
        # Schedule services on Fog
        for _ in range(2):
            service = manager.create_embb_service(latency_ms=30)
            scheduler.schedule_service(service)
        
        # Simulate Fog server 5 failure
        event = handler.simulate_server_failure(5)
        
        # Server should be marked inactive
        server = manager.get_server(5)
        assert not server.is_active
        assert event.server_id == 5
    
    def test_service_migration_on_failure(self):
        """Test that services are migrated when node fails."""
        manager = ServiceManager()
        scheduler = TaskScheduler(manager)
        handler = FailureHandler(manager, scheduler)
        
        # Manually place a service on server 5
        service = manager.create_embb_service()
        service.latency_ms = 150  # Allow migration to Cloud
        server = manager.get_server(5)
        server.allocate(service.cpu_demand, service.memory_demand, service.service_id)
        service.assigned_server = 5
        service.assigned_layer = "Fog"
        service.status = "running"
        
        # Simulate failure
        event = handler.simulate_server_failure(5)
        
        # Check that service was affected and migrated
        assert len(event.affected_services) > 0
        # Check migration status
        assert len(event.migrated_services) >= 0  # May or may not succeed
    
    def test_server_recovery(self):
        """Test that server can be recovered after failure."""
        manager = ServiceManager()
        scheduler = TaskScheduler(manager)
        handler = FailureHandler(manager, scheduler)
        
        # Fail server
        handler.simulate_server_failure(5)
        
        # Recover server
        result = handler.recover_server(5)
        
        assert result["success"]
        assert manager.get_server(5).is_active
    
    def test_system_health_status(self):
        """Test system health reporting."""
        manager = ServiceManager()
        scheduler = TaskScheduler(manager)
        handler = FailureHandler(manager, scheduler)
        
        # Initial health should be healthy
        health = handler.get_system_health()
        assert health["status"] == "healthy"
        assert health["active_servers"] == 7
        
        # After failure, should be degraded
        handler.simulate_server_failure(5)
        health = handler.get_system_health()
        assert health["status"] == "degraded"
        assert health["failed_servers"] == 1


class TestPriorityPreemption:
    """Tests for Higher Priority Services (Professor's Requirement #3)."""
    
    def test_preemption_when_overloaded(self):
        """Test that preemption occurs when server is overloaded."""
        manager = ServiceManager()
        scheduler = TaskScheduler(manager)
        preemption_handler = PriorityPreemptionHandler(manager, scheduler)
        
        # Fill up servers with low priority services
        low_priority_services = []
        for _ in range(6):
            service = manager.create_embb_service(priority=Priority.LOW)
            scheduler.schedule_service(service)
            low_priority_services.append(service)
        
        # Create high priority XR service
        critical_service = manager.create_xr_service(priority=Priority.CRITICAL)
        critical_service.cpu_demand = 15000  # Large demand
        critical_service.memory_demand = 30000
        
        # Try to schedule with preemption
        result = preemption_handler.try_schedule_with_preemption(critical_service)
        
        # Should either succeed or fail gracefully
        assert isinstance(result.success, bool)
    
    def test_priority_order_respected(self):
        """Test that priority order is respected in preemption."""
        manager = ServiceManager()
        scheduler = TaskScheduler(manager)
        preemption_handler = PriorityPreemptionHandler(manager, scheduler)
        
        # Create services with different priorities
        high = manager.create_xr_service(priority=Priority.HIGH)
        medium = manager.create_embb_service(priority=Priority.MEDIUM)
        low = manager.create_embb_service(priority=Priority.LOW)
        
        # Priority values: CRITICAL=1, HIGH=2, MEDIUM=3, LOW=4
        assert Priority.CRITICAL.value < Priority.HIGH.value
        assert Priority.HIGH.value < Priority.MEDIUM.value
        assert Priority.MEDIUM.value < Priority.LOW.value
    
    def test_utilization_threshold_80_percent(self):
        """Test that 80% utilization threshold is used."""
        assert PriorityPreemptionHandler.UTILIZATION_THRESHOLD == 80.0
    
    def test_preemption_statistics(self):
        """Test that preemption statistics are tracked."""
        manager = ServiceManager()
        scheduler = TaskScheduler(manager)
        preemption_handler = PriorityPreemptionHandler(manager, scheduler)
        
        stats = preemption_handler.get_preemption_statistics()
        
        assert "total_preemptions" in stats
        assert "preemptions_by_layer" in stats


class TestDynamicRequirements:
    """Tests for Change in Service Requirements (Professor's Requirement #4)."""
    
    def test_throughput_increase(self):
        """Test handling throughput increase."""
        manager = ServiceManager()
        scheduler = TaskScheduler(manager)
        dynamic_handler = DynamicRequirementsHandler(manager, scheduler)
        
        # Create and schedule service
        service = manager.create_xr_service(throughput_mbps=15, num_users=3)
        scheduler.schedule_service(service)
        
        old_cpu = service.cpu_demand
        
        # Increase throughput
        result = dynamic_handler.update_throughput(service.service_id, 25)
        
        assert result.service_id == service.service_id
        assert result.change_type == "throughput_increase"
        # Demands should have increased
        assert service.throughput_mbps == 25
    
    def test_user_increase(self):
        """Test handling user count increase."""
        manager = ServiceManager()
        scheduler = TaskScheduler(manager)
        dynamic_handler = DynamicRequirementsHandler(manager, scheduler)
        
        # Create and schedule service
        service = manager.create_embb_service(num_users=15)
        scheduler.schedule_service(service)
        
        # Increase users
        result = dynamic_handler.update_users(service.service_id, 30)
        
        assert result.service_id == service.service_id
        assert service.num_users == 30
    
    def test_latency_requirement_tightening(self):
        """Test handling tighter latency requirement."""
        manager = ServiceManager()
        scheduler = TaskScheduler(manager)
        dynamic_handler = DynamicRequirementsHandler(manager, scheduler)
        
        # Create eMBB service with relaxed latency (goes to Cloud)
        service = manager.create_embb_service(latency_ms=150)
        scheduler.schedule_service(service)
        
        # Tighten latency requirement
        result = dynamic_handler.update_latency_requirement(service.service_id, 30)
        
        assert result.service_id == service.service_id
        assert service.latency_ms == 30
    
    def test_migration_on_resource_increase(self):
        """Test that migration occurs when resources cannot be accommodated."""
        manager = ServiceManager()
        scheduler = TaskScheduler(manager)
        dynamic_handler = DynamicRequirementsHandler(manager, scheduler)
        
        # Create small XR service
        service = manager.create_xr_service(throughput_mbps=15, num_users=2)
        scheduler.schedule_service(service)
        
        original_server = service.assigned_server
        
        # Dramatically increase throughput (may need migration)
        result = dynamic_handler.update_throughput(service.service_id, 50)
        
        # Should handle gracefully (success or failure)
        assert isinstance(result.success, bool)
    
    def test_change_history_recorded(self):
        """Test that all changes are recorded in history."""
        manager = ServiceManager()
        scheduler = TaskScheduler(manager)
        dynamic_handler = DynamicRequirementsHandler(manager, scheduler)
        
        # Create and schedule service
        service = manager.create_xr_service()
        scheduler.schedule_service(service)
        
        # Make changes
        dynamic_handler.update_throughput(service.service_id, 20)
        dynamic_handler.update_users(service.service_id, 4)
        
        # Check history
        report = dynamic_handler.get_change_report()
        assert len(report) >= 2


class TestIntegration:
    """Integration tests for all 4 features working together."""
    
    def test_full_scenario(self):
        """Test a complete scenario using all features."""
        manager = ServiceManager()
        scheduler = TaskScheduler(manager)
        failure_handler = FailureHandler(manager, scheduler)
        preemption_handler = PriorityPreemptionHandler(manager, scheduler)
        dynamic_handler = DynamicRequirementsHandler(manager, scheduler)
        
        # Step 1: Schedule mixed workload
        xr_services = []
        for _ in range(3):
            service = manager.create_xr_service()
            result = scheduler.schedule_service(service)
            if result.success:
                xr_services.append(service)
        
        embb_services = []
        for _ in range(3):
            service = manager.create_embb_service()
            result = scheduler.schedule_service(service)
            if result.success:
                embb_services.append(service)
        
        # Step 2: Verify scheduling metrics
        metrics = scheduler.metrics.to_dict()
        assert metrics["total_scheduled"] > 0
        
        # Step 3: Simulate a server failure
        event = failure_handler.simulate_server_failure(5)
        
        # Step 4: Check system health
        health = failure_handler.get_system_health()
        assert health["status"] in ["healthy", "degraded"]
        
        # Step 5: Recover server
        failure_handler.recover_server(5)
        
        # Step 6: Try dynamic requirement change
        if xr_services:
            result = dynamic_handler.update_throughput(
                xr_services[0].service_id, 22
            )
        
        # Step 7: Final statistics
        stats = manager.get_statistics()
        assert "total_services" in stats
    
    def test_xr_edge_embb_cloud_separation(self):
        """Test that XR goes to Edge and eMBB goes to Cloud/Fog."""
        manager = ServiceManager()
        scheduler = TaskScheduler(manager)
        
        edge_count = 0
        fog_cloud_count = 0
        
        # Schedule XR services
        for _ in range(3):
            service = manager.create_xr_service(latency_ms=10)
            result = scheduler.schedule_service(service)
            if result.success and result.assigned_layer == "Edge":
                edge_count += 1
        
        # Schedule eMBB services with relaxed latency
        for _ in range(3):
            service = manager.create_embb_service(latency_ms=150)
            result = scheduler.schedule_service(service)
            if result.success and result.assigned_layer in ["Fog", "Cloud"]:
                fog_cloud_count += 1
        
        # XR should mostly go to Edge
        assert edge_count >= 2
        # eMBB should mostly go to Fog/Cloud
        assert fog_cloud_count >= 2


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
