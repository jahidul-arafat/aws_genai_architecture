# aws_genai_architecture/core/models.py
"""Data models for AWS GenAI Architecture components."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
from datetime import datetime
import uuid

class ServiceType(Enum):
    """AWS Service types supported in the architecture."""
    EC2 = "EC2"
    SAGEMAKER = "SageMaker"
    S3 = "S3"
    BEDROCK = "Bedrock"
    LAMBDA = "Lambda"
    ECS = "ECS"
    CLOUDWATCH = "CloudWatch"

class HealthStatus(Enum):
    """Service health status."""
    HEALTHY = "Healthy"
    WARNING = "Warning"
    CRITICAL = "Critical"
    UNKNOWN = "Unknown"

class CommunicationType(Enum):
    """Types of communications between services."""
    DATA_TRANSFER = "data-transfer"
    MODEL_SYNC = "model-sync"
    CONTROL_SIGNAL = "control-signal"
    CHECKPOINT = "checkpoint"
    MONITORING = "monitoring"
    INFERENCE = "inference"
    PREPROCESSING = "preprocessing"

class CommunicationStatus(Enum):
    """Communication status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class Position:
    """2D position for service placement."""
    x: float
    y: float

@dataclass
class ServiceConfig:
    """Configuration for AWS service instances."""
    instance_type: str
    vcpus: int
    memory: int  # GB
    storage: Optional[int] = None  # GB
    gpu_count: Optional[int] = None
    gpu_type: Optional[str] = None
    network_performance: str = "High"
    spot_instance: bool = False

@dataclass
class ServiceMetrics:
    """Real-time metrics for a service."""
    utilization: float  # Percentage
    throughput: str
    latency: float  # Milliseconds
    cost_per_hour: float
    requests_per_second: Optional[float] = None
    error_rate: float = 0.0
    uptime: float = 99.9  # Percentage

@dataclass
class Organelle:
    """Internal component within a service (microservice/process)."""
    id: str
    x: float
    y: float
    radius: float
    speed: float
    angle: float
    color: str

@dataclass
class ServiceInstance:
    """Complete service instance with all properties."""
    id: str
    name: str
    service_type: ServiceType
    position: Position
    radius: float
    config: ServiceConfig
    metrics: ServiceMetrics
    health_status: HealthStatus = HealthStatus.HEALTHY
    availability_zone: str = "us-east-1a"
    launch_time: datetime = field(default_factory=datetime.now)
    sla_target: float = 99.5
    reliability_score: float = 99.0
    organelles: List[Organelle] = field(default_factory=list)
    tags: Dict[str, str] = field(default_factory=dict)
    activity_level: float = 0.0  # 0.0 to 1.0
    pulse: float = 0.0

@dataclass
class CommunicationPacket:
    """Individual data packet in communication."""
    id: str
    size: str
    timestamp: datetime
    position: float  # 0.0 to 1.0 along communication path

@dataclass
class CommunicationInstance:
    """Service-to-service communication instance."""
    id: str
    from_service_id: str
    to_service_id: str
    communication_type: CommunicationType
    status: CommunicationStatus
    progress: float  # 0.0 to 1.0
    start_time: datetime
    data_size: str
    throughput: str
    latency: float
    protocol: str
    packets: List[CommunicationPacket] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    estimated_completion: Optional[datetime] = None
    actual_completion: Optional[datetime] = None

@dataclass
class TrainingSession:
    """ML training session configuration and state."""
    id: str
    model_name: str
    model_size: str  # e.g., "70B"
    batch_size: int
    learning_rate: float
    current_epoch: int
    total_epochs: int
    current_loss: float
    best_loss: float
    start_time: datetime
    estimated_completion: Optional[datetime] = None
    status: str = "running"
    checkpoints: List[str] = field(default_factory=list)

@dataclass
class SystemMetrics:
    """Overall system metrics."""
    active_services: int
    active_communications: int
    total_cost_per_hour: float
    average_utilization: float
    total_throughput: str
    reliability_score: float
    training_progress: float
    data_processed: str
    network_io: str
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class LogEntry:
    """Structured log entry for console output."""
    timestamp: datetime
    level: str
    service_id: Optional[str]
    communication_id: Optional[str]
    message: str
    details: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        timestamp_str = self.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        prefix = f"[{timestamp_str}] {self.level.upper()}"

        if self.service_id:
            prefix += f" [{self.service_id}]"
        if self.communication_id:
            prefix += f" [{self.communication_id}]"

        return f"{prefix}: {self.message}"