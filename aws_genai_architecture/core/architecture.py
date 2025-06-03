# aws_genai_architecture/core/architecture.py
"""Main AWS GenAI Architecture orchestrator."""
import json
import random
import threading
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable

from .models import (
    ServiceInstance, ServiceType, ServiceConfig, ServiceMetrics,
    Position, CommunicationType, TrainingSession, SystemMetrics,
    HealthStatus
)
from .services import ServiceManager
from .communications import CommunicationManager
from .metrics import MetricsCollector
from ..monitoring.console_logger import ConsoleLogger


class AWSGenAIArchitecture:
    """Main orchestrator for AWS GenAI training architecture simulation."""

    def __init__(self, config: Optional[Dict[str, Any]] = None, logger: Optional[ConsoleLogger] = None):
        self.config = config or {}
        self.logger = logger or ConsoleLogger()

        # Core managers
        self.service_manager = ServiceManager()
        self.communication_manager = CommunicationManager()
        self.metrics_collector = MetricsCollector()

        # State
        self._running = False
        self._threads: List[threading.Thread] = []

        # Training state
        self.current_training_session: Optional[TrainingSession] = None
        self.training_progress = 0.0

        # Event callbacks
        self.event_callbacks: List[Callable] = []

        # Register event handlers
        self.service_manager.register_event_callback(self._handle_service_event)
        self.communication_manager.register_event_callback(self._handle_communication_event)

        # Initialize default services if configured
        if self.config.get("services", {}).get("auto_initialize", True):
            self._initialize_default_services()

    def register_event_callback(self, callback: Callable):
        """Register callback for architecture events."""
        self.event_callbacks.append(callback)

    def _emit_event(self, event_type: str, details: Dict[str, Any]):
        """Emit architecture event."""
        event = {
            'type': event_type,
            'timestamp': datetime.now(),
            'details': details
        }

        for callback in self.event_callbacks:
            try:
                callback(event)
            except Exception as e:
                print(f"Error in event callback: {e}")

    def _handle_service_event(self, event: Dict[str, Any]):
        """Handle service manager events."""
        event_type = event.get('type')
        service_id = event.get('service_id')
        details = event.get('details', {})

        if event_type == 'service_created':
            self.logger.log_info(
                "Service Manager",
                f"🏗️ Service created: {service_id}",
                service_id=service_id,
                details=details
            )
        elif event_type == 'service_failure':
            self.logger.log_error(
                "Service Manager",
                f"🚨 Service failure: {service_id}",
                service_id=service_id,
                details=details
            )
        elif event_type == 'service_recovered':
            self.logger.log_info(
                "Service Manager",
                f"✅ Service recovered: {service_id}",
                service_id=service_id,
                details=details
            )
        elif event_type == 'service_scaled':
            self.logger.log_info(
                "Service Manager",
                f"📈 Service scaled: {service_id}",
                service_id=service_id,
                details=details
            )
        elif event_type == 'metrics_updated':
            # Log significant metric changes
            old_util = details.get('old_utilization', 0)
            new_util = details.get('new_utilization', 0)
            if abs(new_util - old_util) > 15:  # Significant change
                self.logger.log_debug(
                    "Metrics",
                    f"📊 Significant utilization change: {service_id} ({old_util:.1f}% → {new_util:.1f}%)",
                    service_id=service_id,
                    details=details
                )

    def _handle_communication_event(self, event: Dict[str, Any]):
        """Handle communication manager events."""
        event_type = event.get('type')
        comm_id = event.get('communication_id')
        details = event.get('details', {})

        if event_type == 'communication_started':
            self.logger.log_communication_started(
                comm_id=comm_id,
                from_service=details['from_service'],
                to_service=details['to_service'],
                comm_type=CommunicationType(details['type']),
                data_size=details['data_size'],
                throughput=details['throughput'],
                protocol=details['protocol'],
                estimated_duration=details['estimated_duration']
            )
        elif event_type == 'communication_completed':
            self.logger.log_communication_completed(comm_id)
        elif event_type == 'communication_cancelled':
            self.logger.log_communication_failed(comm_id, "Communication cancelled")

    def _initialize_default_services(self):
        """Initialize default AWS services."""
        center_x, center_y = 400, 300  # Canvas center

        # S3 Data Lake
        s3_service = self.service_manager.create_service(
            service_type=ServiceType.S3,
            name="S3 Data Lake",
            position=Position(center_x, center_y - 200),
            config=ServiceConfig(
                instance_type="S3 Standard",
                vcpus=0,
                memory=0,
                storage=10000,  # 10TB
                network_performance="High"
            ),
            availability_zone="us-east-1a",
            sla_target=99.9
        )

        # SageMaker Training
        sagemaker_config = self.config.get("services", {}).get("sagemaker", {})
        instance_type = sagemaker_config.get("instance_type", "ml.p4d.24xlarge")

        sagemaker_service = self.service_manager.create_service(
            service_type=ServiceType.SAGEMAKER,
            name="SageMaker Training",
            position=Position(center_x - 300, center_y),
            config=ServiceConfig(
                instance_type=instance_type,
                vcpus=96,
                memory=1152,
                gpu_count=8,
                gpu_type="A100",
                network_performance="100 Gbps",
                spot_instance=sagemaker_config.get("managed_spot", False)
            ),
            availability_zone="us-east-1a",
            sla_target=99.5
        )

        # EC2 Training Cluster
        ec2_config = self.config.get("services", {}).get("ec2_training", {})
        ec2_instance_type = ec2_config.get("instance_type", "p4d.24xlarge")
        ec2_count = ec2_config.get("count", 2)

        for i in range(ec2_count):
            ec2_service = self.service_manager.create_service(
                service_type=ServiceType.EC2,
                name=f"EC2 Training Node {i+1}",
                position=Position(center_x + 300 + i * 50, center_y + i * 30),
                config=ServiceConfig(
                    instance_type=ec2_instance_type,
                    vcpus=96,
                    memory=1152,
                    gpu_count=8,
                    gpu_type="A100",
                    network_performance="100 Gbps",
                    spot_instance=ec2_config.get("spot_instances", False)
                ),
                availability_zone=f"us-east-1{'abc'[i % 3]}",
                sla_target=99.0
            )

        # Bedrock Foundation Models
        bedrock_service = self.service_manager.create_service(
            service_type=ServiceType.BEDROCK,
            name="Bedrock Foundation Models",
            position=Position(center_x, center_y + 250),
            config=ServiceConfig(
                instance_type="Managed Service",
                vcpus=0,
                memory=0,
                network_performance="Elastic"
            ),
            availability_zone="us-east-1c",
            sla_target=99.9
        )

        # Lambda Data Processing
        lambda_service = self.service_manager.create_service(
            service_type=ServiceType.LAMBDA,
            name="Lambda Data Pipeline",
            position=Position(center_x - 150, center_y - 300),
            config=ServiceConfig(
                instance_type="15GB Memory",
                vcpus=6,
                memory=15,
                network_performance="Up to 25 Gbps"
            ),
            availability_zone="Multi-AZ",
            sla_target=99.95
        )

        # ECS Model Serving
        ecs_service = self.service_manager.create_service(
            service_type=ServiceType.ECS,
            name="ECS Model Serving",
            position=Position(center_x + 150, center_y + 150),
            config=ServiceConfig(
                instance_type="c5.4xlarge",
                vcpus=16,
                memory=32,
                network_performance="Up to 10 Gbps"
            ),
            availability_zone="us-east-1a",
            sla_target=99.5
        )

        # CloudWatch Monitoring
        cloudwatch_service = self.service_manager.create_service(
            service_type=ServiceType.CLOUDWATCH,
            name="CloudWatch Monitoring",
            position=Position(center_x - 200, center_y + 200),
            config=ServiceConfig(
                instance_type="Managed Service",
                vcpus=0,
                memory=0,
                network_performance="High"
            ),
            availability_zone="Multi-AZ",
            sla_target=99.9
        )

        self.logger.log_info("Architecture", "🏗️ Default AWS services initialized")

    def start(self):
        """Start the architecture simulation."""
        if self._running:
            return

        self._running = True
        self.logger.start()

        # Start managers
        self.service_manager.start_monitoring()
        self.communication_manager.start_processing()

        # Start background threads
        self._start_auto_communication_generator()
        self._start_metrics_collector()

        self.logger.log_info("Architecture", "🚀 AWS GenAI Architecture started")
        self._emit_event('architecture_started', {})

    def stop(self):
        """Stop the architecture simulation."""
        if not self._running:
            return

        self._running = False

        # Stop training if active
        if self.current_training_session:
            self.stop_training()

        # Stop managers
        self.service_manager.stop_monitoring()
        self.communication_manager.stop_processing()

        # Wait for threads to finish
        for thread in self._threads:
            thread.join(timeout=2)

        self.logger.log_info("Architecture", "🛑 AWS GenAI Architecture stopped")
        self.logger.stop()

        self._emit_event('architecture_stopped', {})

    def start_training(
            self,
            model_name: str = "LLaMA-70B",
            model_size: str = "70B",
            batch_size: int = 32,
            learning_rate: float = 0.0001,
            total_epochs: int = 100
    ) -> TrainingSession:
        """Start ML training simulation."""
        if self.current_training_session and self.current_training_session.status == "running":
            self.logger.log_warning("Training", "Training session already active")
            return self.current_training_session

        # Create training session
        session_id = f"training_{uuid.uuid4().hex[:8]}"
        self.current_training_session = TrainingSession(
            id=session_id,
            model_name=model_name,
            model_size=model_size,
            batch_size=batch_size,
            learning_rate=learning_rate,
            current_epoch=0,
            total_epochs=total_epochs,
            current_loss=4.5,  # Starting loss
            best_loss=float('inf'),
            start_time=datetime.now(),
            estimated_completion=datetime.now() + timedelta(hours=12)
        )

        # Trigger initial data transfers
        s3_services = self.service_manager.get_services_by_type(ServiceType.S3)
        training_services = (
                self.service_manager.get_services_by_type(ServiceType.SAGEMAKER) +
                self.service_manager.get_services_by_type(ServiceType.EC2)
        )

        if s3_services and training_services:
            s3_service = s3_services[0]

            for training_service in training_services:
                # Data transfer from S3 to training services
                self.communication_manager.create_communication(
                    from_service=s3_service,
                    to_service=training_service,
                    comm_type=CommunicationType.DATA_TRANSFER,
                    data_size="2.3 GB",
                    metadata={"training_session": session_id}
                )

                # Activate training service
                training_service.activity_level = 1.0
                self.service_manager.update_service_metrics(
                    training_service.id,
                    utilization=min(95, training_service.metrics.utilization + 30)
                )

        # Start training progress thread
        training_thread = threading.Thread(target=self._training_loop, daemon=True)
        training_thread.start()
        self._threads.append(training_thread)

        self.logger.log_info(
            "Training",
            f"🎯 Training started: {model_name} ({model_size})",
            details={
                "session_id": session_id,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "total_epochs": total_epochs
            }
        )

        self._emit_event('training_started', {
            'session_id': session_id,
            'model_name': model_name,
            'model_size': model_size
        })

        return self.current_training_session

    def stop_training(self):
        """Stop current training session."""
        if not self.current_training_session:
            self.logger.log_warning("Training", "No active training session")
            return

        # Save final checkpoint
        self.save_checkpoint()

        session = self.current_training_session
        session.status = "stopped"

        self.logger.log_info(
            "Training",
            f"⏹️ Training stopped: {session.model_name}",
            details={
                "session_id": session.id,
                "final_epoch": session.current_epoch,
                "final_loss": session.current_loss,
                "duration_hours": (datetime.now() - session.start_time).total_seconds() / 3600
            }
        )

        self._emit_event('training_stopped', {
            'session_id': session.id,
            'final_epoch': session.current_epoch,
            'final_loss': session.current_loss
        })

        self.current_training_session = None

    def pause_training(self):
        """Pause current training session."""
        if not self.current_training_session:
            return

        self.current_training_session.status = "paused"
        self.save_checkpoint()

        self.logger.log_info("Training", "⏸️ Training paused")
        self._emit_event('training_paused', {'session_id': self.current_training_session.id})

    def resume_training(self):
        """Resume paused training session."""
        if not self.current_training_session or self.current_training_session.status != "paused":
            return

        self.current_training_session.status = "running"

        self.logger.log_info("Training", "▶️ Training resumed")
        self._emit_event('training_resumed', {'session_id': self.current_training_session.id})

    def save_checkpoint(self) -> Optional[str]:
        """Save model checkpoint."""
        if not self.current_training_session:
            return None

        session = self.current_training_session
        checkpoint_id = f"ckpt_epoch_{session.current_epoch}_step_{session.current_epoch * 100}"

        # Trigger checkpoint communications
        training_services = (
                self.service_manager.get_services_by_type(ServiceType.SAGEMAKER) +
                self.service_manager.get_services_by_type(ServiceType.EC2)
        )
        s3_services = self.service_manager.get_services_by_type(ServiceType.S3)

        if training_services and s3_services:
            s3_service = s3_services[0]

            for training_service in training_services:
                self.communication_manager.create_communication(
                    from_service=training_service,
                    to_service=s3_service,
                    comm_type=CommunicationType.CHECKPOINT,
                    data_size="1.2 GB",
                    metadata={
                        "checkpoint_id": checkpoint_id,
                        "epoch": session.current_epoch,
                        "loss": session.current_loss
                    }
                )

        session.checkpoints.append(checkpoint_id)

        self.logger.log_info(
            "Training",
            f"💾 Checkpoint saved: {checkpoint_id}",
            details={
                "epoch": session.current_epoch,
                "loss": session.current_loss,
                "model_size": session.model_size
            }
        )

        return checkpoint_id

    def scale_out(self, instance_type: str = "p4d.24xlarge", count: int = 1) -> List[ServiceInstance]:
        """Scale out training infrastructure."""
        new_services = []

        for i in range(count):
            # Create new EC2 training instance
            new_service = self.service_manager.create_service(
                service_type=ServiceType.EC2,
                name=f"EC2 Scale Instance {i+1}",
                position=Position(200 + i * 100, 200 + i * 50),
                config=ServiceConfig(
                    instance_type=instance_type,
                    vcpus=96,
                    memory=1152,
                    gpu_count=8,
                    gpu_type="A100",
                    network_performance="100 Gbps",
                    spot_instance=True  # Use spot instances for scaling
                ),
                availability_zone="us-east-1c",
                sla_target=95.0  # Lower SLA for spot instances
            )

            new_services.append(new_service)

            # If training is active, join the new instance to training
            if self.current_training_session:
                # Control signal to join training
                cloudwatch_services = self.service_manager.get_services_by_type(ServiceType.CLOUDWATCH)
                if cloudwatch_services:
                    self.communication_manager.create_communication(
                        from_service=cloudwatch_services[0],
                        to_service=new_service,
                        comm_type=CommunicationType.CONTROL_SIGNAL,
                        metadata={"action": "join_training", "session_id": self.current_training_session.id}
                    )

        self.logger.log_info(
            "Scaling",
            f"📈 Scaled out: Added {count} {instance_type} instances",
            details={"instance_type": instance_type, "count": count}
        )

        self._emit_event('scaled_out', {
            'instance_type': instance_type,
            'count': count,
            'new_service_ids': [s.id for s in new_services]
        })

        return new_services

    def scale_in(self, count: int = 1) -> int:
        """Scale in training infrastructure."""
        # Remove spot instances first, then on-demand
        ec2_services = self.service_manager.get_services_by_type(ServiceType.EC2)
        spot_instances = [s for s in ec2_services if s.config.spot_instance]

        services_to_remove = spot_instances[:count]
        if len(services_to_remove) < count:
            # Need to remove some on-demand instances
            on_demand = [s for s in ec2_services if not s.config.spot_instance]
            services_to_remove.extend(on_demand[:count - len(services_to_remove)])

        removed_count = 0
        for service in services_to_remove:
            if self.service_manager.remove_service(service.id):
                removed_count += 1

        self.logger.log_info(
            "Scaling",
            f"📉 Scaled in: Removed {removed_count} instances",
            details={"requested_count": count, "actual_count": removed_count}
        )

        self._emit_event('scaled_in', {
            'requested_count': count,
            'actual_count': removed_count
        })

        return removed_count

    def simulate_failure(self, service_id: Optional[str] = None) -> bool:
        """Simulate service failure."""
        if service_id:
            return self.service_manager.simulate_failure(service_id)
        else:
            # Random service failure
            services = self.service_manager.get_all_services()
            if services:
                import random
                target_service = random.choice(services)
                return self.service_manager.simulate_failure(target_service.id)
        return False

    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current system metrics."""
        services = self.service_manager.get_all_services()
        active_communications = self.communication_manager.get_active_communications()

        total_cost = self.service_manager.get_total_cost()
        avg_utilization = self.service_manager.get_average_utilization()

        # Calculate reliability score
        health_summary = self.service_manager.get_service_health_summary()
        healthy_services = health_summary.get('Healthy', 0)
        total_services = sum(health_summary.values())
        reliability = (healthy_services / max(1, total_services)) * 100

        metrics = {
            "active_services": len(services),
            "active_communications": len(active_communications),
            "total_cost_per_hour": f"${total_cost:.0f}",
            "average_utilization": avg_utilization,
            "reliability_score": reliability,
            "data_processed": "2.3TB",  # Simulated
            "network_io": "450 MB/s",  # Simulated
            "timestamp": datetime.now().isoformat()
        }

        if self.current_training_session:
            metrics.update({
                "training_active": True,
                "training_progress": self.training_progress,
                "current_epoch": self.current_training_session.current_epoch,
                "current_loss": self.current_training_session.current_loss,
                "model_size": self.current_training_session.model_size
            })
        else:
            metrics["training_active"] = False

        return metrics

    def get_services(self) -> List[ServiceInstance]:
        """Get all services."""
        return self.service_manager.get_all_services()

    def get_communications(self) -> List[Any]:  # CommunicationInstance
        """Get all active communications."""
        return self.communication_manager.get_active_communications()

    def add_communication(
            self,
            comm_type: str,
            from_service_id: Optional[str] = None,
            to_service_id: Optional[str] = None
    ):
        """Add a communication between services."""
        try:
            communication_type = CommunicationType(comm_type)
        except ValueError:
            self.logger.log_error("Communication", f"Invalid communication type: {comm_type}")
            return

        services = self.service_manager.get_all_services()
        if len(services) < 2:
            return

        # Select services based on communication patterns or random
        if from_service_id and to_service_id:
            from_service = self.service_manager.get_service(from_service_id)
            to_service = self.service_manager.get_service(to_service_id)
        else:
            # Use intelligent service selection based on communication type
            from_service, to_service = self._select_services_for_communication(communication_type)

        if from_service and to_service and from_service != to_service:
            self.communication_manager.create_communication(
                from_service=from_service,
                to_service=to_service,
                comm_type=communication_type
            )

    def _select_services_for_communication(self, comm_type: CommunicationType) -> tuple:
        """Intelligently select services for communication based on type."""
        # Communication patterns matching the original JavaScript logic
        patterns = self.communication_manager.communication_patterns.get(comm_type, {})
        from_patterns = patterns.get('from', [])
        to_patterns = patterns.get('to', [])

        services = self.service_manager.get_all_services()

        # Find matching services by type or ID pattern
        from_candidates = []
        to_candidates = []

        for service in services:
            service_type_str = service.service_type.value

            # Check if service matches from pattern
            for pattern in from_patterns:
                if pattern in service.id or service_type_str in pattern:
                    from_candidates.append(service)
                    break

            # Check if service matches to pattern
            for pattern in to_patterns:
                if pattern in service.id or service_type_str in pattern:
                    to_candidates.append(service)
                    break

        # Fallback to service type matching
        if not from_candidates:
            from_candidates = services
        if not to_candidates:
            to_candidates = services

        # Select random candidates
        import random
        from_service = random.choice(from_candidates) if from_candidates else None
        to_service = random.choice(to_candidates) if to_candidates else None

        # Ensure different services
        if from_service == to_service and len(services) > 1:
            remaining = [s for s in services if s != from_service]
            to_service = random.choice(remaining)

        return from_service, to_service

    def _start_auto_communication_generator(self):
        """Start automatic communication generation."""
        comm_config = self.config.get("communication", {})
        if not comm_config.get("auto_generate", True):
            return

        def communication_generator():
            frequency = comm_config.get("frequency", 1.8)
            comm_types = comm_config.get("types", [
                "data-transfer", "model-sync", "monitoring",
                "control-signal", "inference", "preprocessing"
            ])

            # Probability weights for different communication types
            type_weights = {
                "data-transfer": 0.25,
                "model-sync": 0.20,
                "monitoring": 0.20,
                "control-signal": 0.15,
                "inference": 0.10,
                "preprocessing": 0.10
            }

            while self._running:
                try:
                    import random

                    # Generate communication based on probability
                    if random.random() < 0.6:  # 60% chance per cycle
                        # Select communication type based on weights
                        rand_val = random.random()
                        cumulative = 0

                        for comm_type in comm_types:
                            weight = type_weights.get(comm_type, 0.1)
                            cumulative += weight
                            if rand_val <= cumulative:
                                self.add_communication(comm_type)
                                break

                    # Special logic for training-related communications
                    if self.current_training_session and self.current_training_session.status == "running":
                        # Generate more training-related communications
                        if random.random() < 0.3:  # 30% chance for checkpoint
                            if self.current_training_session.current_epoch % 5 == 0:
                                self.add_communication("checkpoint")

                        if random.random() < 0.4:  # 40% chance for model sync
                            self.add_communication("model-sync")

                    time.sleep(frequency)

                except Exception as e:
                    self.logger.log_error("AutoComm", f"Error in communication generator: {e}")
                    time.sleep(5)  # Wait before retrying

        comm_thread = threading.Thread(target=communication_generator, daemon=True)
        comm_thread.start()
        self._threads.append(comm_thread)

    def _start_metrics_collector(self):
        """Start metrics collection thread."""
        def metrics_collector():
            interval = self.config.get("monitoring", {}).get("metrics_interval", 2.0)

            while self._running:
                try:
                    # Collect and log system metrics
                    metrics = self.get_current_metrics()
                    self.logger.log_system_metrics(metrics)

                    # Collect individual service metrics
                    for service in self.service_manager.get_all_services():
                        service_metrics = {
                            "utilization": service.metrics.utilization,
                            "cost_per_hour": service.metrics.cost_per_hour,
                            "throughput": service.metrics.throughput,
                            "latency": service.metrics.latency,
                            "error_rate": service.metrics.error_rate,
                            "uptime": service.metrics.uptime
                        }
                        self.logger.log_service_metrics(service.id, service_metrics)

                    time.sleep(interval)

                except Exception as e:
                    self.logger.log_error("MetricsCollector", f"Error collecting metrics: {e}")
                    time.sleep(interval)

        metrics_thread = threading.Thread(target=metrics_collector, daemon=True)
        metrics_thread.start()
        self._threads.append(metrics_thread)

    def _training_loop(self):
        """Main training simulation loop."""
        if not self.current_training_session:
            return

        session = self.current_training_session

        while self._running and session.status == "running":
            try:
                # Simulate training progress
                time.sleep(2)  # 2 seconds per step

                if session.status != "running":
                    break

                # Update progress (simplified simulation)
                steps_per_epoch = 100
                current_step = (session.current_epoch * steps_per_epoch) + 1
                total_steps = session.total_epochs * steps_per_epoch

                self.training_progress = (current_step / total_steps) * 100

                # Simulate loss decrease with some noise
                if session.current_epoch == 0:
                    session.current_loss = 4.5
                else:
                    # Gradual loss decrease with some random variation
                    base_decrease = 0.001 * (1 + session.current_epoch * 0.1)
                    noise = (random.random() - 0.5) * 0.01
                    session.current_loss = max(1.2, session.current_loss - base_decrease + noise)

                # Update best loss
                if session.current_loss < session.best_loss:
                    session.best_loss = session.current_loss

                # Progress to next epoch every 100 steps
                if current_step % steps_per_epoch == 0:
                    session.current_epoch += 1

                    self.logger.log_info(
                        "Training",
                        f"📈 Epoch {session.current_epoch}/{session.total_epochs} completed",
                        details={
                            "loss": f"{session.current_loss:.4f}",
                            "best_loss": f"{session.best_loss:.4f}",
                            "progress": f"{self.training_progress:.1f}%"
                        }
                    )

                    # Auto-checkpoint every 5 epochs
                    if session.current_epoch % 5 == 0:
                        self.save_checkpoint()

                    # Training completion
                    if session.current_epoch >= session.total_epochs:
                        session.status = "completed"

                        self.logger.log_info(
                            "Training",
                            f"🎉 Training completed: {session.model_name}",
                            details={
                                "total_epochs": session.total_epochs,
                                "final_loss": f"{session.current_loss:.4f}",
                                "best_loss": f"{session.best_loss:.4f}",
                                "duration_hours": (datetime.now() - session.start_time).total_seconds() / 3600
                            }
                        )

                        self._emit_event('training_completed', {
                            'session_id': session.id,
                            'final_loss': session.current_loss,
                            'best_loss': session.best_loss,
                            'total_epochs': session.total_epochs
                        })
                        break

            except Exception as e:
                self.logger.log_error("Training", f"Error in training loop: {e}")
                session.status = "failed"
                break

    def export_configuration(self, filename: str):
        """Export current configuration to file."""
        current_config = {
            "services": {
                "auto_initialize": True,
                "services_count": len(self.service_manager.get_all_services())
            },
            "communication": self.config.get("communication", {}),
            "monitoring": self.config.get("monitoring", {}),
            "web": self.config.get("web", {}),
            "exported_at": datetime.now().isoformat()
        }

        # Add service details
        current_config["services"]["details"] = []
        for service in self.service_manager.get_all_services():
            service_info = {
                "id": service.id,
                "name": service.name,
                "type": service.service_type.value,
                "instance_type": service.config.instance_type,
                "availability_zone": service.availability_zone,
                "cost_per_hour": service.metrics.cost_per_hour,
                "utilization": service.metrics.utilization
            }
            current_config["services"]["details"].append(service_info)

        with open(filename, 'w') as f:
            json.dump(current_config, f, indent=2)

        self.logger.log_info("Configuration", f"📄 Configuration exported to {filename}")

    def get_training_status(self) -> Optional[Dict[str, Any]]:
        """Get current training status."""
        if not self.current_training_session:
            return None

        session = self.current_training_session
        duration = datetime.now() - session.start_time

        return {
            "session_id": session.id,
            "model_name": session.model_name,
            "model_size": session.model_size,
            "status": session.status,
            "current_epoch": session.current_epoch,
            "total_epochs": session.total_epochs,
            "current_loss": session.current_loss,
            "best_loss": session.best_loss,
            "progress_percent": self.training_progress,
            "duration_hours": duration.total_seconds() / 3600,
            "checkpoints_saved": len(session.checkpoints),
            "estimated_completion": session.estimated_completion.isoformat() if session.estimated_completion else None
        }

    def get_communication_statistics(self) -> Dict[str, Any]:
        """Get communication statistics."""
        return self.communication_manager.get_communication_statistics()

    def get_service_health_summary(self) -> Dict[str, Any]:
        """Get service health summary."""
        health_summary = self.service_manager.get_service_health_summary()
        total_services = sum(health_summary.values())

        return {
            "total_services": total_services,
            "health_breakdown": health_summary,
            "healthy_percentage": (health_summary.get('Healthy', 0) / max(1, total_services)) * 100,
            "total_cost_per_hour": self.service_manager.get_total_cost(),
            "average_utilization": self.service_manager.get_average_utilization()
        }

    def trigger_data_ingestion(self):
        """Trigger data ingestion process."""
        s3_services = self.service_manager.get_services_by_type(ServiceType.S3)
        lambda_services = self.service_manager.get_services_by_type(ServiceType.LAMBDA)

        if s3_services and lambda_services:
            self.communication_manager.create_communication(
                from_service=s3_services[0],
                to_service=lambda_services[0],
                comm_type=CommunicationType.DATA_TRANSFER,
                data_size="3.2 GB",
                metadata={"process": "data_ingestion"}
            )

            self.logger.log_info("DataPipeline", "📥 Data ingestion triggered")

    def trigger_preprocessing(self):
        """Trigger data preprocessing."""
        lambda_services = self.service_manager.get_services_by_type(ServiceType.LAMBDA)
        training_services = (
                self.service_manager.get_services_by_type(ServiceType.SAGEMAKER) +
                self.service_manager.get_services_by_type(ServiceType.EC2)
        )

        if lambda_services and training_services:
            lambda_service = lambda_services[0]

            for training_service in training_services:
                self.communication_manager.create_communication(
                    from_service=lambda_service,
                    to_service=training_service,
                    comm_type=CommunicationType.PREPROCESSING,
                    data_size="1.8 GB",
                    metadata={"process": "preprocessing"}
                )

            # Update Lambda service activity
            lambda_service.activity_level = 1.0
            self.service_manager.update_service_metrics(
                lambda_service.id,
                utilization=min(90, lambda_service.metrics.utilization + 20)
            )

            self.logger.log_info("DataPipeline", "⚙️ Data preprocessing triggered")

    def trigger_model_validation(self):
        """Trigger model validation process."""
        serving_services = self.service_manager.get_services_by_type(ServiceType.ECS)
        bedrock_services = self.service_manager.get_services_by_type(ServiceType.BEDROCK)

        if serving_services or bedrock_services:
            # Validation inference requests
            if serving_services:
                service = serving_services[0]
                self.add_communication("inference")

            # Generate validation metrics
            import random
            accuracy = 92.5 + random.uniform(-2, 5)
            f1_score = 0.89 + random.uniform(-0.05, 0.08)
            perplexity = 15.2 + random.uniform(-3, 5)

            self.logger.log_info(
                "Validation",
                "🎯 Model validation completed",
                details={
                    "accuracy": f"{accuracy:.2f}%",
                    "f1_score": f"{f1_score:.3f}",
                    "perplexity": f"{perplexity:.1f}"
                }
            )

            return {
                "accuracy": accuracy,
                "f1_score": f1_score,
                "perplexity": perplexity,
                "validation_time": datetime.now().isoformat()
            }

        return None

    def run_health_check(self):
        """Run comprehensive health check on all services."""
        services = self.service_manager.get_all_services()

        # Send monitoring signals to all services
        cloudwatch_services = self.service_manager.get_services_by_type(ServiceType.CLOUDWATCH)

        if cloudwatch_services:
            cloudwatch = cloudwatch_services[0]

            for service in services:
                if service != cloudwatch:
                    self.communication_manager.create_communication(
                        from_service=service,
                        to_service=cloudwatch,
                        comm_type=CommunicationType.MONITORING,
                        metadata={"check_type": "health_check"}
                    )

        # Generate health report
        health_summary = self.get_service_health_summary()

        self.logger.log_info(
            "HealthCheck",
            "❤️ Health check completed",
            details={
                "total_services": health_summary["total_services"],
                "healthy_services": health_summary["health_breakdown"].get("Healthy", 0),
                "overall_health": f"{health_summary['healthy_percentage']:.1f}%",
                "total_cost": f"${health_summary['total_cost_per_hour']:.0f}/hour"
            }
        )

        return health_summary

    def clear_communications(self):
        """Clear all completed communications."""
        self.communication_manager.clear_completed_communications()
        self.logger.log_info("System", "🧹 Cleared completed communications")

    def reset_architecture(self):
        """Reset the entire architecture to initial state."""
        # Stop current training
        if self.current_training_session:
            self.stop_training()

        # Clear all communications
        self.communication_manager.communications.clear()
        self.communication_manager.active_communications.clear()

        # Reset all services
        services_to_remove = list(self.service_manager.services.keys())
        for service_id in services_to_remove:
            self.service_manager.remove_service(service_id)

        # Reinitialize default services
        if self.config.get("services", {}).get("auto_initialize", True):
            self._initialize_default_services()

        # Reset metrics
        self.training_progress = 0.0

        self.logger.log_info("System", "🔄 Architecture reset completed")
        self._emit_event('architecture_reset', {})