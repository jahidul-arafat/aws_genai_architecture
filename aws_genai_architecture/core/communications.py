# aws_genai_architecture/core/communications.py (UPDATED - Fix missing imports)
"""Communication management between AWS services."""

import time
import threading
import uuid
import random
from typing import Dict, List, Optional, Callable, Tuple
from datetime import datetime, timedelta
from queue import Queue

from .models import (
    CommunicationInstance, CommunicationType, CommunicationStatus,
    CommunicationPacket, ServiceInstance, LogEntry
)

# Add missing Communication alias
Communication = CommunicationInstance

class CommunicationManager:
    """Manages communications between AWS services."""

    def __init__(self):
        self.communications: Dict[str, CommunicationInstance] = {}
        self.active_communications: List[str] = []
        self.communication_queue = Queue()
        self.event_callbacks: List[Callable] = []
        self._running = False
        self._processor_thread: Optional[threading.Thread] = None

        # ADD THIS - Missing metrics tracking
        self.metrics = type('CommunicationMetrics', (), {
            'total_communications': 0,
            'successful_communications': 0,
            'failed_communications': 0,
            'average_duration': 0.0,
            'total_data_transferred': 0.0,
            'average_throughput': 0.0,
            'peak_concurrent_communications': 0,
            'communication_types': {},
            'service_pairs': {},
            'hourly_statistics': {}
        })()

        # Communication patterns for different AWS services
        self.communication_patterns = {
            CommunicationType.DATA_TRANSFER: {
                'from': ['S3_DATALAKE'],
                'to': ['SAGEMAKER_TRAINING', 'EC2_TRAINING_NODES', 'LAMBDA_PREPROCESSING'],
                'speed': 0.008,
                'particle_count': 6,
                'protocol': 'AWS S3 Transfer Acceleration'
            },
            CommunicationType.MODEL_SYNC: {
                'from': ['SAGEMAKER_TRAINING', 'EC2_TRAINING_NODES'],
                'to': ['ECS_SERVING', 'BEDROCK_MODELS'],
                'speed': 0.006,
                'particle_count': 8,
                'protocol': 'AWS PrivateLink'
            },
            CommunicationType.CONTROL_SIGNAL: {
                'from': ['CLOUDWATCH_MONITORING'],
                'to': ['SAGEMAKER_TRAINING', 'EC2_TRAINING_NODES'],
                'speed': 0.02,
                'particle_count': 2,
                'protocol': 'AWS Systems Manager'
            },
            CommunicationType.CHECKPOINT: {
                'from': ['SAGEMAKER_TRAINING', 'EC2_TRAINING_NODES'],
                'to': ['S3_DATALAKE'],
                'speed': 0.004,
                'particle_count': 4,
                'protocol': 'AWS S3 Multipart Upload'
            },
            CommunicationType.MONITORING: {
                'from': ['SAGEMAKER_TRAINING', 'EC2_TRAINING_NODES', 'ECS_SERVING'],
                'to': ['CLOUDWATCH_MONITORING'],
                'speed': 0.015,
                'particle_count': 3,
                'protocol': 'CloudWatch Metrics API'
            },
            CommunicationType.INFERENCE: {
                'from': ['ECS_SERVING', 'BEDROCK_MODELS'],
                'to': ['LAMBDA_PREPROCESSING'],
                'speed': 0.018,
                'particle_count': 4,
                'protocol': 'Amazon API Gateway'
            },
            CommunicationType.PREPROCESSING: {
                'from': ['LAMBDA_PREPROCESSING'],
                'to': ['SAGEMAKER_TRAINING', 'EC2_TRAINING_NODES'],
                'speed': 0.012,
                'particle_count': 5,
                'protocol': 'AWS Step Functions'
            }
        }

    def register_event_callback(self, callback: Callable):
        """Register callback for communication events."""
        self.event_callbacks.append(callback)

    def _emit_event(self, event_type: str, comm_id: str, details: Dict = None):
        """Emit communication event to registered callbacks."""
        for callback in self.event_callbacks:
            try:
                callback({
                    'type': event_type,
                    'communication_id': comm_id,
                    'details': details or {},
                    'timestamp': datetime.now()
                })
            except Exception as e:
                print(f"Error in communication event callback: {e}")

    def create_communication(
            self,
            from_service: ServiceInstance,
            to_service: ServiceInstance,
            comm_type: CommunicationType,
            data_size: Optional[str] = None,
            metadata: Optional[Dict] = None
    ) -> CommunicationInstance:
        """Create a new communication between services."""

        comm_id = f"comm_{uuid.uuid4().hex[:8]}"

        # Get communication pattern
        pattern = self.communication_patterns.get(comm_type, {})

        # Generate realistic data size if not provided
        if not data_size:
            data_size = self._generate_data_size(comm_type)

        # Calculate throughput and estimated completion
        throughput = self._calculate_throughput(comm_type, from_service, to_service)
        estimated_duration = self._estimate_duration(data_size, throughput)

        communication = CommunicationInstance(
            id=comm_id,
            from_service_id=from_service.id,
            to_service_id=to_service.id,
            communication_type=comm_type,
            status=CommunicationStatus.PENDING,
            progress=0.0,
            start_time=datetime.now(),
            data_size=data_size,
            throughput=throughput,
            latency=self._calculate_latency(from_service, to_service),
            protocol=pattern.get('protocol', 'TCP/IP'),
            packets=self._generate_packets(pattern.get('particle_count', 3), data_size),
            metadata=metadata or {},
            estimated_completion=datetime.now() + estimated_duration
        )

        self.communications[comm_id] = communication
        self.active_communications.append(comm_id)

        # Update service activity
        from_service.activity_level = min(1.0, from_service.activity_level + 0.5)
        to_service.activity_level = min(1.0, to_service.activity_level + 0.4)

        self._emit_event('communication_started', comm_id, {
            'from_service': from_service.name,
            'to_service': to_service.name,
            'type': comm_type.value,
            'data_size': data_size,
            'throughput': throughput,
            'protocol': communication.protocol,
            'estimated_duration': estimated_duration.total_seconds()
        })

        return communication

    def _generate_data_size(self, comm_type: CommunicationType) -> str:
        """Generate realistic data size for communication type."""
        size_ranges = {
            CommunicationType.DATA_TRANSFER: (1.5, 5.0, "GB"),
            CommunicationType.MODEL_SYNC: (200, 800, "MB"),
            CommunicationType.CONTROL_SIGNAL: (10, 50, "KB"),
            CommunicationType.CHECKPOINT: (800, 1500, "MB"),
            CommunicationType.MONITORING: (100, 800, "KB"),
            CommunicationType.INFERENCE: (50, 200, "MB"),
            CommunicationType.PREPROCESSING: (100, 600, "MB")
        }

        min_val, max_val, unit = size_ranges.get(comm_type, (1, 100, "MB"))
        size = random.uniform(min_val, max_val)

        return f"{size:.1f} {unit}"

    def _calculate_throughput(
            self,
            comm_type: CommunicationType,
            from_service: ServiceInstance,
            to_service: ServiceInstance
    ) -> str:
        """Calculate realistic throughput for communication."""

        # Base throughput by communication type
        base_throughputs = {
            CommunicationType.DATA_TRANSFER: (1.8, 2.5, "GB/s"),
            CommunicationType.MODEL_SYNC: (600, 1200, "MB/s"),
            CommunicationType.CONTROL_SIGNAL: (10, 30, "KB/s"),
            CommunicationType.CHECKPOINT: (800, 1800, "MB/s"),
            CommunicationType.MONITORING: (300, 800, "metrics/s"),
            CommunicationType.INFERENCE: (200, 600, "req/s"),
            CommunicationType.PREPROCESSING: (1.5, 4.0, "MB/s")
        }

        min_val, max_val, unit = base_throughputs.get(comm_type, (100, 200, "MB/s"))

        # Factor in service performance
        from_utilization_factor = 1.0 - (from_service.metrics.utilization / 200)
        to_utilization_factor = 1.0 - (to_service.metrics.utilization / 200)

        performance_factor = (from_utilization_factor + to_utilization_factor) / 2
        performance_factor = max(0.5, min(1.2, performance_factor))

        throughput = random.uniform(min_val, max_val) * performance_factor

        return f"{throughput:.1f} {unit}"

    def _calculate_latency(
            self,
            from_service: ServiceInstance,
            to_service: ServiceInstance
    ) -> float:
        """Calculate communication latency between services."""

        # Base latency calculation
        base_latency = 5.0  # 5ms base

        # Distance factor
        distance = ((to_service.position.x - from_service.position.x) ** 2 +
                    (to_service.position.y - from_service.position.y) ** 2) ** 0.5
        distance_latency = distance * 0.1

        # Service load factor
        from_load_factor = from_service.metrics.utilization / 100 * 10
        to_load_factor = to_service.metrics.utilization / 100 * 8

        # Availability zone factor
        az_latency = 0.0
        if from_service.availability_zone != to_service.availability_zone:
            az_latency = random.uniform(2, 8)

        total_latency = base_latency + distance_latency + from_load_factor + to_load_factor + az_latency
        return round(total_latency + random.uniform(-2, 3), 1)

    def _estimate_duration(self, data_size: str, throughput: str) -> timedelta:
        """Estimate communication duration."""
        try:
            # Parse data size
            size_parts = data_size.split()
            size_value = float(size_parts[0])
            size_unit = size_parts[1] if len(size_parts) > 1 else "MB"

            # Parse throughput
            throughput_parts = throughput.split()
            throughput_value = float(throughput_parts[0])
            throughput_unit = throughput_parts[1] if len(throughput_parts) > 1 else "MB/s"

            # Convert to common units (MB)
            size_mb = self._convert_to_mb(size_value, size_unit)
            throughput_mbs = self._convert_throughput_to_mbs(throughput_value, throughput_unit)

            if throughput_mbs > 0:
                duration_seconds = size_mb / throughput_mbs
                return timedelta(seconds=max(1, duration_seconds))
            else:
                return timedelta(seconds=30)

        except (ValueError, IndexError):
            return timedelta(seconds=30)

    def _convert_to_mb(self, value: float, unit: str) -> float:
        """Convert data size to MB."""
        conversions = {
            "B": value / (1024 * 1024),
            "KB": value / 1024,
            "MB": value,
            "GB": value * 1024,
            "TB": value * 1024 * 1024
        }
        return conversions.get(unit, value)

    def _convert_throughput_to_mbs(self, value: float, unit: str) -> float:
        """Convert throughput to MB/s."""
        if "GB/s" in unit:
            return value * 1024
        elif "KB/s" in unit:
            return value / 1024
        elif "MB/s" in unit:
            return value
        else:
            # For non-bandwidth units, use a rough estimate
            return 10.0  # Default 10 MB/s equivalent

    def _generate_packets(self, count: int, data_size: str) -> List[CommunicationPacket]:
        """Generate communication packets."""
        packets = []

        for i in range(count):
            packet = CommunicationPacket(
                id=f"packet_{i}",
                size=f"{random.uniform(0.1, 2.0):.1f} MB",
                timestamp=datetime.now(),
                position=i / count  # Evenly spaced along path initially
            )
            packets.append(packet)

        return packets

    def get_communication(self, comm_id: str) -> Optional[CommunicationInstance]:
        """Get communication by ID."""
        return self.communications.get(comm_id)

    def get_active_communications(self) -> List[CommunicationInstance]:
        """Get all active communications."""
        return [self.communications[comm_id] for comm_id in self.active_communications
                if comm_id in self.communications]

    def get_communications_for_service(self, service_id: str) -> List[CommunicationInstance]:
        """Get all communications involving a specific service."""
        return [comm for comm in self.communications.values()
                if comm.from_service_id == service_id or comm.to_service_id == service_id]

    def update_communication_progress(self, comm_id: str, progress_delta: float = None):
        """Update communication progress."""
        if comm_id not in self.communications:
            return False

        communication = self.communications[comm_id]

        if communication.status != CommunicationStatus.IN_PROGRESS:
            communication.status = CommunicationStatus.IN_PROGRESS

        # Calculate progress delta based on communication speed
        if progress_delta is None:
            pattern = self.communication_patterns.get(communication.communication_type, {})
            speed = pattern.get('speed', 0.01)
            progress_delta = speed

        communication.progress = min(1.0, communication.progress + progress_delta)

        # Update packet positions
        for i, packet in enumerate(communication.packets):
            packet.position = (communication.progress + i * 0.12) % 1.0

        # Check if communication is complete
        if communication.progress >= 1.0:
            self._complete_communication(comm_id)

        return True

    def _complete_communication(self, comm_id: str):
        """Complete a communication."""
        if comm_id not in self.communications:
            return

        communication = self.communications[comm_id]
        communication.status = CommunicationStatus.COMPLETED
        communication.actual_completion = datetime.now()

        # Remove from active list
        if comm_id in self.active_communications:
            self.active_communications.remove(comm_id)

        # Calculate actual duration
        duration = communication.actual_completion - communication.start_time

        self._emit_event('communication_completed', comm_id, {
            'from_service': communication.from_service_id,
            'to_service': communication.to_service_id,
            'type': communication.communication_type.value,
            'duration_seconds': duration.total_seconds(),
            'data_transferred': communication.data_size,
            'average_throughput': communication.throughput
        })

        # Auto-cleanup after some time
        def cleanup():
            time.sleep(30)  # Keep for 30 seconds for logging
            if comm_id in self.communications:
                del self.communications[comm_id]

        threading.Thread(target=cleanup, daemon=True).start()

    def cancel_communication(self, comm_id: str) -> bool:
        """Cancel an active communication."""
        if comm_id not in self.communications:
            return False

        communication = self.communications[comm_id]
        communication.status = CommunicationStatus.FAILED

        if comm_id in self.active_communications:
            self.active_communications.remove(comm_id)

        self._emit_event('communication_cancelled', comm_id, {
            'from_service': communication.from_service_id,
            'to_service': communication.to_service_id,
            'progress_at_cancel': communication.progress
        })

        return True

    def start_processing(self):
        """Start communication processing."""
        if self._running:
            return

        self._running = True
        self._processor_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self._processor_thread.start()

    def stop_processing(self):
        """Stop communication processing."""
        self._running = False
        if self._processor_thread:
            self._processor_thread.join(timeout=1)

    def _processing_loop(self):
        """Main communication processing loop."""
        while self._running:
            try:
                # Update all active communications
                active_comms = self.get_active_communications()
                for communication in active_comms:
                    self.update_communication_progress(communication.id)

                time.sleep(0.1)  # Update 10 times per second

            except Exception as e:
                print(f"Error in communication processing loop: {e}")

    def get_communication_statistics(self) -> Dict:
        """Get communication statistics."""
        total_comms = len(self.communications)
        active_comms = len(self.active_communications)
        completed_comms = len([c for c in self.communications.values()
                               if c.status == CommunicationStatus.COMPLETED])
        failed_comms = len([c for c in self.communications.values()
                            if c.status == CommunicationStatus.FAILED])

        # Calculate average completion time
        completed_communications = [c for c in self.communications.values()
                                    if c.status == CommunicationStatus.COMPLETED and c.actual_completion]

        avg_duration = 0
        if completed_communications:
            total_duration = sum((c.actual_completion - c.start_time).total_seconds()
                                 for c in completed_communications)
            avg_duration = total_duration / len(completed_communications)

        return {
            'total_communications': total_comms,
            'active_communications': active_comms,
            'completed_communications': completed_comms,
            'failed_communications': failed_comms,
            'success_rate': (completed_comms / max(1, total_comms)) * 100,
            'average_duration_seconds': round(avg_duration, 2)
        }

    def clear_completed_communications(self):
        """Clear all completed communications."""
        completed_ids = [comm_id for comm_id, comm in self.communications.items()
                         if comm.status in [CommunicationStatus.COMPLETED, CommunicationStatus.FAILED]]

        for comm_id in completed_ids:
            del self.communications[comm_id]

        self._emit_event('communications_cleared', '', {
            'cleared_count': len(completed_ids)
        })