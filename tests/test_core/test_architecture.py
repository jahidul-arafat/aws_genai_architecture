# tests/test_core/test_architecture.py
"""Tests for the main architecture class."""

import pytest
import time
from datetime import datetime
from aws_genai_architecture.core.architecture import AWSGenAIArchitecture
from aws_genai_architecture.core.models import ServiceType, CommunicationType


class TestAWSGenAIArchitecture:
    """Test cases for AWSGenAIArchitecture class."""

    def test_initialization(self, architecture):
        """Test architecture initialization."""
        assert architecture is not None
        assert not architecture._running
        assert architecture.current_training_session is None
        assert architecture.training_progress == 0.0

    def test_start_stop(self, architecture):
        """Test starting and stopping the architecture."""
        # Test start
        architecture.start()
        assert architecture._running

        # Test stop
        architecture.stop()
        assert not architecture._running

    def test_service_creation(self, running_architecture):
        """Test creating AWS services."""
        initial_count = len(running_architecture.get_services())

        # Test creating a service through initialization
        running_architecture._initialize_default_services()

        services = running_architecture.get_services()
        assert len(services) > initial_count

        # Check service types
        service_types = [s.service_type for s in services]
        assert ServiceType.S3 in service_types
        assert ServiceType.SAGEMAKER in service_types

    def test_training_session(self, running_architecture):
        """Test training session management."""
        # Start training
        session = running_architecture.start_training(
            model_name="Test-Model",
            model_size="7B",
            total_epochs=5
        )

        assert session is not None
        assert session.model_name == "Test-Model"
        assert session.status == "running"
        assert running_architecture.current_training_session == session

        # Stop training
        running_architecture.stop_training()
        assert running_architecture.current_training_session is None

    def test_scaling_operations(self, running_architecture):
        """Test scaling infrastructure."""
        # Initialize services first
        running_architecture._initialize_default_services()
        initial_count = len(running_architecture.get_services())

        # Scale out
        new_services = running_architecture.scale_out(count=2)
        assert len(new_services) == 2
        assert len(running_architecture.get_services()) == initial_count + 2

        # Scale in
        removed_count = running_architecture.scale_in(count=1)
        assert removed_count == 1
        assert len(running_architecture.get_services()) == initial_count + 1

    def test_communication_management(self, running_architecture):
        """Test communication management."""
        # Initialize services
        running_architecture._initialize_default_services()

        # Get initial state
        initial_total = running_architecture.communication_manager.metrics.total_communications

        # Add communication
        running_architecture.add_communication("data-transfer")

        # Allow time for processing
        import time
        time.sleep(0.2)

        # Check that communication was created
        current_total = running_architecture.communication_manager.metrics.total_communications
        assert current_total > initial_total

    def test_metrics_collection(self, running_architecture):
        """Test metrics collection."""
        metrics = running_architecture.get_current_metrics()

        assert isinstance(metrics, dict)
        assert "active_services" in metrics
        assert "active_communications" in metrics
        assert "total_cost_per_hour" in metrics
        assert "reliability_score" in metrics

    def test_configuration_handling(self):
        """Test configuration handling."""
        config = {
            "services": {"auto_initialize": True},
            "communication": {"auto_generate": True, "frequency": 2.0},
            "monitoring": {"log_level": "DEBUG"}
        }

        arch = AWSGenAIArchitecture(config=config)
        assert arch.config == config

        # Check that configuration affects behavior
        assert arch.config["services"]["auto_initialize"]