# aws_genai_architecture/core/services.py (UPDATED - Fix missing imports)
"""Service management for AWS GenAI Architecture."""

import time
import threading
from typing import Dict, List, Optional, Callable
from datetime import datetime, timedelta
import uuid
import math
import random

from .models import (
    ServiceInstance, ServiceType, ServiceConfig, ServiceMetrics,
    HealthStatus, Position, Organelle, LogEntry
)

# Add missing Service alias
Service = ServiceInstance

class ServiceManager:
    """Manages AWS services in the architecture."""

    def __init__(self):
        self.services: Dict[str, ServiceInstance] = {}
        self.event_callbacks: List[Callable] = []
        self._running = False
        self._update_thread: Optional[threading.Thread] = None

    def register_event_callback(self, callback: Callable):
        """Register callback for service events."""
        self.event_callbacks.append(callback)

    def _emit_event(self, event_type: str, service_id: str, details: Dict = None):
        """Emit service event to registered callbacks."""
        for callback in self.event_callbacks:
            try:
                callback({
                    'type': event_type,
                    'service_id': service_id,
                    'details': details or {},
                    'timestamp': datetime.now()
                })
            except Exception as e:
                print(f"Error in event callback: {e}")

    def create_service(
            self,
            service_type: ServiceType,
            name: str,
            position: Position,
            config: ServiceConfig,
            **kwargs
    ) -> ServiceInstance:
        """Create a new service instance."""

        service_id = f"{service_type.value}_{uuid.uuid4().hex[:8]}"

        # Generate organelles (internal components)
        organelles = self._create_organelles(config.vcpus // 2 + 2)

        # Calculate initial metrics
        metrics = self._calculate_initial_metrics(service_type, config)

        service = ServiceInstance(
            id=service_id,
            name=name,
            service_type=service_type,
            position=position,
            radius=self._calculate_radius(service_type, config),
            config=config,
            metrics=metrics,
            organelles=organelles,
            launch_time=datetime.now(),
            **kwargs
        )

        self.services[service_id] = service

        self._emit_event('service_created', service_id, {
            'service_type': service_type.value,
            'name': name,
            'instance_type': config.instance_type,
            'cost_per_hour': metrics.cost_per_hour
        })

        return service

    def _create_organelles(self, count: int) -> List[Organelle]:
        """Create internal organelles for a service."""
        organelles = []

        for i in range(count):
            organelle = Organelle(
                id=f"org_{i}",
                x=(random.random() - 0.5) * 80,
                y=(random.random() - 0.5) * 80,
                radius=3 + random.random() * 6,
                speed=0.3 + random.random() * 0.8,
                angle=random.random() * math.pi * 2,
                color=f"hsl({200 + random.random() * 80}, 70%, 65%)"
            )
            organelles.append(organelle)

        return organelles

    def _calculate_radius(self, service_type: ServiceType, config: ServiceConfig) -> float:
        """Calculate service radius based on type and configuration."""
        base_radii = {
            ServiceType.S3: 65,
            ServiceType.SAGEMAKER: 75,
            ServiceType.EC2: 70,
            ServiceType.BEDROCK: 60,
            ServiceType.LAMBDA: 45,
            ServiceType.ECS: 50,
            ServiceType.CLOUDWATCH: 40,
        }

        base_radius = base_radii.get(service_type, 50)

        # Adjust based on instance size
        if hasattr(config, 'vcpus') and config.vcpus > 0:
            scaling_factor = 1 + (config.vcpus / 100) * 0.3
            base_radius *= scaling_factor

        return base_radius

    def _calculate_initial_metrics(
            self,
            service_type: ServiceType,
            config: ServiceConfig
    ) -> ServiceMetrics:
        """Calculate initial metrics for a service."""

        # Base costs per hour by service type
        base_costs = {
            ServiceType.EC2: {
                'p4d.24xlarge': 434,
                'p3dn.24xlarge': 318,
                'trn1.32xlarge': 245,
                'inf2.48xlarge': 156,
                'c5.4xlarge': 95,
            },
            ServiceType.SAGEMAKER: {
                'ml.p4d.24xlarge': 458,
                'ml.p3dn.24xlarge': 350,
                'ml.trn1.32xlarge': 275,
            },
            ServiceType.S3: 150,
            ServiceType.BEDROCK: 125,
            ServiceType.LAMBDA: 45,
            ServiceType.ECS: 95,
            ServiceType.CLOUDWATCH: 25,
        }

        # Calculate cost
        if service_type in [ServiceType.EC2, ServiceType.SAGEMAKER]:
            cost = base_costs[service_type].get(config.instance_type, 200)
            if config.spot_instance:
                cost *= 0.3  # 70% savings for spot instances
        else:
            cost = base_costs.get(service_type, 100)

        # Calculate throughput based on service type
        throughput_map = {
            ServiceType.EC2: f"{980 + random.randint(-100, 200)} tokens/sec",
            ServiceType.SAGEMAKER: f"{1200 + random.randint(-200, 300)} tokens/sec",
            ServiceType.S3: f"{2.1 + random.uniform(-0.3, 0.5):.1f} GB/s",
            ServiceType.BEDROCK: f"{500 + random.randint(-50, 100)} tokens/sec",
            ServiceType.LAMBDA: f"{2500 + random.randint(-500, 800)} requests/sec",
            ServiceType.ECS: f"{850 + random.randint(-100, 200)} requests/sec",
            ServiceType.CLOUDWATCH: f"{10000 + random.randint(-1000, 2000)} metrics/sec",
        }

        # Calculate latency
        latency_map = {
            ServiceType.EC2: 38 + random.uniform(-5, 10),
            ServiceType.SAGEMAKER: 45 + random.uniform(-8, 15),
            ServiceType.S3: 12 + random.uniform(-3, 8),
            ServiceType.BEDROCK: 120 + random.uniform(-20, 40),
            ServiceType.LAMBDA: 25 + random.uniform(-5, 15),
            ServiceType.ECS: 15 + random.uniform(-3, 8),
            ServiceType.CLOUDWATCH: 8 + random.uniform(-2, 5),
        }

        return ServiceMetrics(
            utilization=45 + random.uniform(0, 30),
            throughput=throughput_map.get(service_type, "100 ops/sec"),
            latency=latency_map.get(service_type, 50),
            cost_per_hour=cost,
            uptime=99.0 + random.uniform(0, 0.9)
        )

    def get_service(self, service_id: str) -> Optional[ServiceInstance]:
        """Get service by ID."""
        return self.services.get(service_id)

    def get_services_by_type(self, service_type: ServiceType) -> List[ServiceInstance]:
        """Get all services of a specific type."""
        return [service for service in self.services.values()
                if service.service_type == service_type]

    def get_all_services(self) -> List[ServiceInstance]:
        """Get all services."""
        return list(self.services.values())

    def remove_service(self, service_id: str) -> bool:
        """Remove a service."""
        if service_id in self.services:
            service = self.services[service_id]
            del self.services[service_id]

            self._emit_event('service_removed', service_id, {
                'service_type': service.service_type.value,
                'name': service.name,
                'uptime_hours': (datetime.now() - service.launch_time).total_seconds() / 3600
            })
            return True
        return False

    def update_service_metrics(self, service_id: str, **kwargs) -> bool:
        """Update service metrics."""
        if service_id not in self.services:
            return False

        service = self.services[service_id]
        old_metrics = service.metrics

        # Update metrics with provided values
        for key, value in kwargs.items():
            if hasattr(service.metrics, key):
                setattr(service.metrics, key, value)

        # Emit metrics update event
        self._emit_event('metrics_updated', service_id, {
            'old_utilization': old_metrics.utilization,
            'new_utilization': service.metrics.utilization,
            'throughput': service.metrics.throughput,
            'cost_per_hour': service.metrics.cost_per_hour
        })

        return True

    def simulate_failure(self, service_id: str) -> bool:
        """Simulate service failure."""
        if service_id not in self.services:
            return False

        service = self.services[service_id]
        service.health_status = HealthStatus.CRITICAL
        service.reliability_score = max(80, service.reliability_score - 15)
        service.metrics.utilization = 0
        service.metrics.error_rate = 95.0

        self._emit_event('service_failure', service_id, {
            'service_type': service.service_type.value,
            'name': service.name,
            'failure_time': datetime.now().isoformat(),
            'previous_reliability': service.reliability_score + 15
        })

        # Auto-recovery after delay
        def recover():
            time.sleep(10)  # 10 second recovery time
            if service_id in self.services:
                service.health_status = HealthStatus.HEALTHY
                service.reliability_score = min(99.5, service.reliability_score + 10)
                service.metrics.utilization = 45 + random.uniform(0, 30)
                service.metrics.error_rate = 0.1

                self._emit_event('service_recovered', service_id, {
                    'recovery_time': datetime.now().isoformat(),
                    'new_reliability': service.reliability_score
                })

        threading.Thread(target=recover, daemon=True).start()
        return True

    def scale_service(self, service_id: str, scaling_factor: float) -> bool:
        """Scale service resources."""
        if service_id not in self.services:
            return False

        service = self.services[service_id]

        # Update configuration
        service.config.vcpus = int(service.config.vcpus * scaling_factor)
        service.config.memory = int(service.config.memory * scaling_factor)

        # Update metrics
        service.metrics.cost_per_hour *= scaling_factor
        service.metrics.throughput = self._recalculate_throughput(service, scaling_factor)

        # Update visual representation
        service.radius = self._calculate_radius(service.service_type, service.config)

        self._emit_event('service_scaled', service_id, {
            'scaling_factor': scaling_factor,
            'new_vcpus': service.config.vcpus,
            'new_memory': service.config.memory,
            'new_cost': service.metrics.cost_per_hour
        })

        return True

    def _recalculate_throughput(self, service: ServiceInstance, scaling_factor: float) -> str:
        """Recalculate throughput after scaling."""
        current_throughput = service.metrics.throughput

        # Extract numeric value and unit
        parts = current_throughput.split()
        if len(parts) >= 2:
            try:
                value = float(parts[0].replace(',', ''))
                unit = ' '.join(parts[1:])
                new_value = value * scaling_factor

                if new_value >= 1000:
                    return f"{new_value/1000:.1f}k {unit}"
                else:
                    return f"{new_value:.0f} {unit}"
            except ValueError:
                pass

        return current_throughput

    def get_service_health_summary(self) -> Dict[str, int]:
        """Get summary of service health status."""
        summary = {status.value: 0 for status in HealthStatus}

        for service in self.services.values():
            summary[service.health_status.value] += 1

        return summary

    def get_total_cost(self) -> float:
        """Calculate total cost per hour for all services."""
        return sum(service.metrics.cost_per_hour for service in self.services.values())

    def get_average_utilization(self) -> float:
        """Calculate average utilization across all services."""
        if not self.services:
            return 0.0

        total_utilization = sum(service.metrics.utilization for service in self.services.values())
        return total_utilization / len(self.services)

    def get_services_by_availability_zone(self, az: str) -> List[ServiceInstance]:
        """Get services in specific availability zone."""
        return [service for service in self.services.values()
                if service.availability_zone == az]

    def start_monitoring(self):
        """Start continuous service monitoring."""
        if self._running:
            return

        self._running = True
        self._update_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._update_thread.start()

    def stop_monitoring(self):
        """Stop service monitoring."""
        self._running = False
        if self._update_thread:
            self._update_thread.join(timeout=1)

    def _monitoring_loop(self):
        """Continuous monitoring loop."""
        while self._running:
            try:
                self._update_all_services()
                time.sleep(2)  # Update every 2 seconds
            except Exception as e:
                print(f"Error in monitoring loop: {e}")

    def _update_all_services(self):
        """Update all service states."""
        for service in self.services.values():
            self._update_service_state(service)

    def _update_service_state(self, service: ServiceInstance):
        """Update individual service state."""
        # Update activity pulse
        service.pulse += 0.04

        # Decay activity level
        service.activity_level = max(0, service.activity_level - 0.006)

        # Update organelles
        for organelle in service.organelles:
            organelle.angle += organelle.speed * 0.008

        # Simulate realistic metric fluctuations
        if service.health_status == HealthStatus.HEALTHY:
            # Utilization fluctuation
            utilization_change = random.uniform(-2, 2)
            service.metrics.utilization = max(10, min(95,
                                                      service.metrics.utilization + utilization_change))

            # Latency fluctuation
            latency_change = random.uniform(-5, 5)
            service.metrics.latency = max(5, service.metrics.latency + latency_change)

            # Error rate
            service.metrics.error_rate = max(0, random.uniform(0, 0.5))

            # Uptime calculation
            uptime_hours = (datetime.now() - service.launch_time).total_seconds() / 3600
            expected_uptime = service.sla_target
            actual_uptime = min(expected_uptime,
                                expected_uptime - service.metrics.error_rate * 0.1)
            service.metrics.uptime = actual_uptime