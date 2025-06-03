# tests/test_monitoring/test_console_logger.py
"""Tests for console logging."""

import pytest
import tempfile
import os
from aws_genai_architecture.monitoring.console_logger import ConsoleLogger
from aws_genai_architecture.core.models import CommunicationType


class TestConsoleLogger:
    """Test cases for ConsoleLogger class."""

    @pytest.fixture
    def logger(self):
        """Create a logger for testing."""
        return ConsoleLogger(level="DEBUG", enable_colors=False)

    def test_logging_levels(self, logger):
        """Test different logging levels."""
        logger.log_debug("Test", "Debug message")
        logger.log_info("Test", "Info message")
        logger.log_warning("Test", "Warning message")
        logger.log_error("Test", "Error message")
        logger.log_critical("Test", "Critical message")

        # Check that messages were queued
        assert not logger.log_queue.empty()

    def test_communication_logging(self, logger):
        """Test communication-specific logging."""
        logger.log_communication_started(
            comm_id="test_comm",
            from_service="Service A",
            to_service="Service B",
            comm_type=CommunicationType.DATA_TRANSFER,
            data_size="1.5 GB",
            throughput="100 MB/s",
            protocol="TCP",
            estimated_duration=15.0
        )

        assert "test_comm" in logger.active_communications

        # Test progress logging
        logger.log_communication_progress("test_comm", 0.5, "120 MB/s", "750 MB")

        # Test completion
        logger.log_communication_completed("test_comm")
        assert "test_comm" not in logger.active_communications

    def test_file_output(self):
        """Test logging to file."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
            temp_filename = temp_file.name

        try:
            logger = ConsoleLogger(output_file=temp_filename, enable_colors=False)
            logger.start()

            logger.log_info("Test", "File output test")

            # Give time for logging thread to process
            import time
            time.sleep(0.1)

            logger.stop()

            # Check file contents
            with open(temp_filename, 'r') as f:
                content = f.read()
                assert "File output test" in content

        finally:
            if os.path.exists(temp_filename):
                os.unlink(temp_filename)

    def test_metrics_logging(self, logger):
        """Test metrics logging."""
        test_metrics = {
            "utilization": 75.0,
            "cost_per_hour": 100,
            "throughput": "500 MB/s",
            "latency": 25.0
        }

        logger.log_service_metrics("test_service", test_metrics)
        assert "test_service" in logger.service_stats
        assert logger.service_stats["test_service"]["utilization"] == 75.0
