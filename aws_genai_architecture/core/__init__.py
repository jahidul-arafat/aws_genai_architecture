# aws_genai_architecture/core/__init__.py
"""Core components for AWS GenAI Architecture simulation."""

from .architecture import AWSGenAIArchitecture
from .services import ServiceManager, ServiceInstance
from .communications import CommunicationManager, CommunicationInstance
from .metrics import MetricsCollector, MetricsSnapshot
from .models import (
    ServiceType, ServiceConfig, ServiceMetrics, Position,
    CommunicationType, CommunicationStatus, TrainingSession,
    SystemMetrics, HealthStatus
)

# Create aliases for backward compatibility
Service = ServiceInstance
Communication = CommunicationInstance

__all__ = [
    "AWSGenAIArchitecture",
    "ServiceManager",
    "ServiceInstance",
    "Service",  # Alias
    "CommunicationManager",
    "CommunicationInstance",
    "Communication",  # Alias
    "MetricsCollector",
    "MetricsSnapshot",
    "ServiceType",
    "ServiceConfig",
    "ServiceMetrics",
    "Position",
    "CommunicationType",
    "CommunicationStatus",
    "TrainingSession",
    "SystemMetrics",
    "HealthStatus",
]