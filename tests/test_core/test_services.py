# tests/test_core/test_services.py
"""Tests for service management."""

import pytest
from aws_genai_architecture.core.services import ServiceManager
from aws_genai_architecture.core.models import ServiceType, ServiceConfig, Position


class TestServiceManager:
    """Test cases for ServiceManager class."""

    @pytest.fixture
    def service_manager(self):
        """Create a service manager for testing."""
        return ServiceManager()

    def test_service_creation(self, service_manager):
        """Test creating services."""
        config = ServiceConfig(
            instance_type="test.instance",
            vcpus=4,
            memory=16
        )

        service = service_manager.create_service(
            service_type=ServiceType.EC2,
            name="Test Service",
            position=Position(100, 100),
            config=config
        )

        assert service is not None
        assert service.name == "Test Service"
        assert service.service_type == ServiceType.EC2
        assert service.config.instance_type == "test.instance"
        assert service.id in service_manager.services

    def test_service_metrics_update(self, service_manager):
        """Test updating service metrics."""
        # Create a service
        config = ServiceConfig(instance_type="test", vcpus=2, memory=8)
        service = service_manager.create_service(
            ServiceType.LAMBDA, "Test", Position(0, 0), config
        )

        # Update metrics
        success = service_manager.update_service_metrics(
            service.id, utilization=80.0, latency=25.0
        )

        assert success
        assert service.metrics.utilization == 80.0
        assert service.metrics.latency == 25.0

    def test_service_removal(self, service_manager):
        """Test removing services."""
        # Create and remove service
        config = ServiceConfig(instance_type="test", vcpus=1, memory=2)
        service = service_manager.create_service(
            ServiceType.S3, "Test", Position(0, 0), config
        )

        service_id = service.id
        assert service_id in service_manager.services

        success = service_manager.remove_service(service_id)
        assert success
        assert service_id not in service_manager.services

    def test_service_failure_simulation(self, service_manager):
        """Test service failure simulation."""
        # Create service
        config = ServiceConfig(instance_type="test", vcpus=2, memory=4)
        service = service_manager.create_service(
            ServiceType.EC2, "Test", Position(0, 0), config
        )

        # Simulate failure
        success = service_manager.simulate_failure(service.id)
        assert success
        assert service.health_status.value == "Critical"
        assert service.metrics.utilization == 0