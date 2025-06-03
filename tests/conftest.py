
# 🚨 CORRECT FILE STRUCTURE

## 📁 Keep conftest.py CLEAN - Only pytest fixtures:

# tests/conftest.py (KEEP THIS CLEAN)
"""Pytest configuration and fixtures."""

import pytest
import sys
import os
from unittest.mock import Mock

# Add the package to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from aws_genai_architecture import AWSGenAIArchitecture
from aws_genai_architecture.monitoring.console_logger import ConsoleLogger

@pytest.fixture(autouse=True)
def setup_test_environment():
    """Set up test environment variables."""
    os.environ['FLASK_ENV'] = 'testing'
    os.environ['TESTING'] = 'true'

@pytest.fixture
def mock_logger():
    """Create a mock logger for testing."""
    logger = Mock(spec=ConsoleLogger)
    logger.log_info = Mock()
    logger.log_error = Mock()
    logger.log_warning = Mock()
    logger.log_debug = Mock()
    logger.start = Mock()
    logger.stop = Mock()
    logger.log_communication_started = Mock()
    logger.log_communication_completed = Mock()
    logger.log_communication_failed = Mock()
    logger.log_service_metrics = Mock()
    logger.log_system_metrics = Mock()
    return logger

@pytest.fixture
def test_config():
    """Create test configuration."""
    return {
        "services": {"auto_initialize": False},
        "communication": {"auto_generate": False},
        "monitoring": {"console_output": False, "log_level": "DEBUG"}
    }

@pytest.fixture
def architecture(mock_logger, test_config):
    """Create a test architecture instance."""
    return AWSGenAIArchitecture(config=test_config, logger=mock_logger)

@pytest.fixture
def running_architecture(architecture):
    """Create a running architecture instance."""
    architecture.start()
    yield architecture
    architecture.stop()
