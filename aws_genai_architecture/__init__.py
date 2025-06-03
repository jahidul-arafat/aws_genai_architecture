# aws_genai_architecture/__init__.py
"""
AWS GenAI Architecture Visualizer

An interactive Python package for visualizing AWS GenAI LLM training architectures
with real-time communication monitoring and detailed console logging.
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# Import main classes with proper aliases to avoid conflicts
from .core.architecture import AWSGenAIArchitecture
from .core.services import ServiceManager, ServiceInstance as Service
from .core.communications import CommunicationManager, CommunicationInstance as Communication
from .core.metrics import MetricsCollector, MetricsSnapshot
from .monitoring.console_logger import ConsoleLogger
from .monitoring.communication_monitor import CommunicationMonitor
from .web.app import create_app

__all__ = [
    "AWSGenAIArchitecture",
    "Service",  # Alias for ServiceInstance
    "ServiceManager",
    "Communication",  # Alias for CommunicationInstance
    "CommunicationManager",
    "MetricsCollector",
    "MetricsSnapshot",
    "ConsoleLogger",
    "CommunicationMonitor",
    "create_app",
]