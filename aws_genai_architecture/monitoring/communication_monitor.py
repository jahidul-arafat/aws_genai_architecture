# aws_genai_architecture/communication_monitor.py
"""Enhanced communication monitoring with detailed tracking."""

import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Callable
from collections import defaultdict, deque
from dataclasses import dataclass, field

from aws_genai_architecture.core.models import CommunicationType, CommunicationStatus, ServiceInstance
from aws_genai_architecture.monitoring.console_logger import ConsoleLogger


@dataclass
class CommunicationMetrics:
    """Detailed metrics for communication monitoring."""
    total_communications: int = 0
    successful_communications: int = 0
    failed_communications: int = 0
    average_duration: float = 0.0
    total_data_transferred: float = 0.0  # In MB
    average_throughput: float = 0.0  # MB/s
    peak_concurrent_communications: int = 0
    communication_types: Dict[str, int] = field(default_factory=dict)
    service_pairs: Dict[str, int] = field(default_factory=dict)
    hourly_statistics: Dict[str, Dict] = field(default_factory=dict)


class CommunicationMonitor:
    """Enhanced communication monitoring with detailed analytics."""

    def __init__(self, logger: Optional[ConsoleLogger] = None):
        self.logger = logger or ConsoleLogger()

        # Tracking data
        self.tracked_types: Set[CommunicationType] = set()
        self.tracked_service_pairs: Set[tuple] = set()
        self.communication_history: deque = deque(maxlen=10000)
        self.active_communications: Dict[str, Dict] = {}

        # Metrics
        self.metrics = CommunicationMetrics()
        self.real_time_metrics: Dict[str, float] = {}

        # Event callbacks
        self.event_callbacks: List[Callable] = []

        # Threading
        self._running = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._analysis_thread: Optional[threading.Thread] = None

        # Configuration
        self.detailed_logging = True
        self.alert_thresholds = {
            'max_concurrent_communications': 50,
            'min_success_rate': 95.0,
            'max_average_latency': 1000.0,  # ms
            'max_failed_communications_per_hour': 10
        }

    def register_event_callback(self, callback: Callable):
        """Register callback for communication events."""
        self.event_callbacks.append(callback)

    def track_communication_type(self, comm_type: CommunicationType):
        """Track specific communication type."""
        self.tracked_types.add(comm_type)
        self.logger.log_info(
            "CommunicationMonitor",
            f"📡 Now tracking communication type: {comm_type.value}"
        )

    def track_service_pair(self, from_service_id: str, to_service_id: str):
        """Track communications between specific service pair."""
        pair = (from_service_id, to_service_id)
        self.tracked_service_pairs.add(pair)
        self.logger.log_info(
            "CommunicationMonitor",
            f"🔗 Now tracking service pair: {from_service_id} → {to_service_id}"
        )

    def start_monitoring(self):
        """Start communication monitoring."""
        if self._running:
            return

        self._running = True

        # Start monitoring thread
        self._monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._monitor_thread.start()

        # Start analysis thread
        self._analysis_thread = threading.Thread(target=self._analysis_loop, daemon=True)
        self._analysis_thread.start()

        self.logger.log_info("CommunicationMonitor", "🚀 Communication monitoring started")

    def stop_monitoring(self):
        """Stop communication monitoring."""
        if not self._running:
            return

        self._running = False

        if self._monitor_thread:
            self._monitor_thread.join(timeout=2)
        if self._analysis_thread:
            self._analysis_thread.join(timeout=2)

        self.logger.log_info("CommunicationMonitor", "⏹️ Communication monitoring stopped")

    def record_communication_start(
            self,
            comm_id: str,
            from_service: ServiceInstance,
            to_service: ServiceInstance,
            comm_type: CommunicationType,
            data_size: str,
            estimated_duration: float
    ):
        """Record the start of a communication."""
        start_time = datetime.now()

        comm_record = {
            'id': comm_id,
            'from_service_id': from_service.id,
            'to_service_id': to_service.id,
            'from_service_name': from_service.name,
            'to_service_name': to_service.name,
            'type': comm_type,
            'data_size': data_size,
            'start_time': start_time,
            'estimated_duration': estimated_duration,
            'actual_duration': None,
            'status': CommunicationStatus.IN_PROGRESS,
            'progress': 0.0,
            'throughput_samples': [],
            'latency_samples': [],
            'bytes_transferred': 0,
            'peak_throughput': 0.0,
            'from_service_type': from_service.service_type.value,
            'to_service_type': to_service.service_type.value,
            'availability_zones': {
                'from': from_service.availability_zone,
                'to': to_service.availability_zone
            }
        }

        self.active_communications[comm_id] = comm_record

        # Update metrics
        self.metrics.total_communications += 1
        type_key = comm_type.value
        self.metrics.communication_types[type_key] = self.metrics.communication_types.get(type_key, 0) + 1

        # Track service pair
        pair_key = f"{from_service.name} → {to_service.name}"
        self.metrics.service_pairs[pair_key] = self.metrics.service_pairs.get(pair_key, 0) + 1

        # Update peak concurrent communications
        current_active = len(self.active_communications)
        if current_active > self.metrics.peak_concurrent_communications:
            self.metrics.peak_concurrent_communications = current_active

        # Check if this communication should be tracked
        should_track = (
                not self.tracked_types or comm_type in self.tracked_types or
                not self.tracked_service_pairs or
                (from_service.id, to_service.id) in self.tracked_service_pairs
        )

        if should_track and self.detailed_logging:
            self.logger.log_communication_started(
                comm_id=comm_id,
                from_service=from_service.name,
                to_service=to_service.name,
                comm_type=comm_type,
                data_size=data_size,
                throughput="Calculating...",
                protocol="AWS",
                estimated_duration=estimated_duration
            )

        # Emit event
        self._emit_event('communication_started', {
            'communication_id': comm_id,
            'from_service': from_service.name,
            'to_service': to_service.name,
            'type': comm_type.value,
            'data_size': data_size,
            'concurrent_count': current_active
        })

    def record_communication_progress(
            self,
            comm_id: str,
            progress: float,
            current_throughput: Optional[float] = None,
            current_latency: Optional[float] = None
    ):
        """Record communication progress update."""
        if comm_id not in self.active_communications:
            return

        comm_record = self.active_communications[comm_id]
        comm_record['progress'] = progress

        # Record throughput sample
        if current_throughput is not None:
            comm_record['throughput_samples'].append({
                'timestamp': datetime.now(),
                'value': current_throughput
            })

            if current_throughput > comm_record['peak_throughput']:
                comm_record['peak_throughput'] = current_throughput

        # Record latency sample
        if current_latency is not None:
            comm_record['latency_samples'].append({
                'timestamp': datetime.now(),
                'value': current_latency
            })

        # Calculate bytes transferred
        data_size_mb = self._parse_data_size_to_mb(comm_record['data_size'])
        comm_record['bytes_transferred'] = data_size_mb * progress

        # Log progress for tracked communications
        should_track = self._should_track_communication(comm_record)

        if should_track and progress > 0.1 and progress < 0.9:
            elapsed = (datetime.now() - comm_record['start_time']).total_seconds()

            # Only log progress every 5 seconds for long-running communications
            if elapsed > 5 and not hasattr(comm_record, 'last_progress_log'):
                comm_record['last_progress_log'] = datetime.now()

                self.logger.log_communication_progress(
                    comm_id=comm_id,
                    progress=progress,
                    current_throughput=f"{current_throughput:.1f} MB/s" if current_throughput else None,
                    bytes_transferred=f"{comm_record['bytes_transferred']:.1f} MB"
                )
            elif hasattr(comm_record, 'last_progress_log'):
                time_since_log = (datetime.now() - comm_record['last_progress_log']).total_seconds()
                if time_since_log > 10:  # Log every 10 seconds
                    comm_record['last_progress_log'] = datetime.now()

                    self.logger.log_communication_progress(
                        comm_id=comm_id,
                        progress=progress,
                        current_throughput=f"{current_throughput:.1f} MB/s" if current_throughput else None,
                        bytes_transferred=f"{comm_record['bytes_transferred']:.1f} MB"
                    )

    def record_communication_completion(self, comm_id: str, success: bool = True, error_message: str = None):
        """Record communication completion."""
        if comm_id not in self.active_communications:
            return

        comm_record = self.active_communications[comm_id]
        end_time = datetime.now()
        duration = (end_time - comm_record['start_time']).total_seconds()

        comm_record['actual_duration'] = duration
        comm_record['status'] = CommunicationStatus.COMPLETED if success else CommunicationStatus.FAILED
        comm_record['end_time'] = end_time

        if not success and error_message:
            comm_record['error_message'] = error_message

        # Calculate final metrics
        data_size_mb = self._parse_data_size_to_mb(comm_record['data_size'])
        average_throughput = data_size_mb / duration if duration > 0 else 0

        comm_record['final_throughput'] = average_throughput
        comm_record['final_data_transferred'] = data_size_mb

        # Update global metrics
        if success:
            self.metrics.successful_communications += 1
            self.metrics.total_data_transferred += data_size_mb
        else:
            self.metrics.failed_communications += 1

        # Update average duration
        total_completed = self.metrics.successful_communications + self.metrics.failed_communications
        if total_completed > 0:
            self.metrics.average_duration = (
                    (self.metrics.average_duration * (total_completed - 1) + duration) / total_completed
            )

        # Update average throughput
        if self.metrics.successful_communications > 0:
            self.metrics.average_throughput = (
                    self.metrics.total_data_transferred /
                    sum(record['actual_duration'] for record in self.communication_history
                        if record.get('status') == CommunicationStatus.COMPLETED and record.get('actual_duration'))
            )

        # Add to history and remove from active
        self.communication_history.append(comm_record.copy())
        del self.active_communications[comm_id]

        # Log completion
        should_track = self._should_track_communication(comm_record)

        if should_track and self.detailed_logging:
            if success:
                self.logger.log_communication_completed(comm_id)
            else:
                self.logger.log_communication_failed(comm_id, error_message or "Unknown error")

        # Emit event
        self._emit_event('communication_completed', {
            'communication_id': comm_id,
            'success': success,
            'duration': duration,
            'data_transferred': data_size_mb,
            'average_throughput': average_throughput,
            'type': comm_record['type'].value,
            'from_service': comm_record['from_service_name'],
            'to_service': comm_record['to_service_name']
        })

        # Check for alerts
        self._check_alerts()

    def _should_track_communication(self, comm_record: Dict) -> bool:
        """Check if communication should be tracked based on filters."""
        comm_type = comm_record['type']
        from_id = comm_record['from_service_id']
        to_id = comm_record['to_service_id']

        type_match = not self.tracked_types or comm_type in self.tracked_types
        pair_match = not self.tracked_service_pairs or (from_id, to_id) in self.tracked_service_pairs

        return type_match and pair_match

    def _parse_data_size_to_mb(self, data_size: str) -> float:
        """Parse data size string to MB."""
        try:
            parts = data_size.split()
            if len(parts) != 2:
                return 0.0

            value = float(parts[0])
            unit = parts[1].upper()

            conversions = {
                'B': value / (1024 * 1024),
                'KB': value / 1024,
                'MB': value,
                'GB': value * 1024,
                'TB': value * 1024 * 1024
            }

            return conversions.get(unit, 0.0)
        except (ValueError, IndexError):
            return 0.0

    def _emit_event(self, event_type: str, details: Dict):
        """Emit event to registered callbacks."""
        event = {
            'type': event_type,
            'timestamp': datetime.now(),
            'details': details
        }

        for callback in self.event_callbacks:
            try:
                callback(event)
            except Exception as e:
                self.logger.log_error("CommunicationMonitor", f"Error in event callback: {e}")

    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self._running:
            try:
                # Update real-time metrics
                self._update_real_time_metrics()

                # Clean up old communications (those that might have been missed)
                self._cleanup_stale_communications()

                # Update hourly statistics
                self._update_hourly_statistics()

                time.sleep(1)  # Monitor every second

            except Exception as e:
                self.logger.log_error("CommunicationMonitor", f"Error in monitoring loop: {e}")
                time.sleep(5)

    def _analysis_loop(self):
        """Analysis loop for generating insights."""
        while self._running:
            try:
                # Generate analysis every 30 seconds
                self._analyze_communication_patterns()
                self._detect_anomalies()

                time.sleep(30)

            except Exception as e:
                self.logger.log_error("CommunicationMonitor", f"Error in analysis loop: {e}")
                time.sleep(60)

    def _update_real_time_metrics(self):
        """Update real-time metrics."""
        current_time = datetime.now()

        # Current active communications
        self.real_time_metrics['active_communications'] = len(self.active_communications)

        # Communications in last minute
        minute_ago = current_time - timedelta(minutes=1)
        recent_communications = [
            comm for comm in self.communication_history
            if comm.get('start_time', datetime.min) >= minute_ago
        ]
        self.real_time_metrics['communications_per_minute'] = len(recent_communications)

        # Average throughput of active communications
        active_throughputs = []
        for comm in self.active_communications.values():
            if comm['throughput_samples']:
                latest_sample = comm['throughput_samples'][-1]
                if (current_time - latest_sample['timestamp']).total_seconds() < 5:
                    active_throughputs.append(latest_sample['value'])

        self.real_time_metrics['current_avg_throughput'] = (
            sum(active_throughputs) / len(active_throughputs) if active_throughputs else 0.0
        )

        # Success rate in last hour
        hour_ago = current_time - timedelta(hours=1)
        recent_completed = [
            comm for comm in self.communication_history
            if comm.get('end_time', datetime.min) >= hour_ago and
               comm.get('status') in [CommunicationStatus.COMPLETED, CommunicationStatus.FAILED]
        ]

        if recent_completed:
            successful_recent = [
                comm for comm in recent_completed
                if comm.get('status') == CommunicationStatus.COMPLETED
            ]
            self.real_time_metrics['success_rate_1h'] = (
                    len(successful_recent) / len(recent_completed) * 100
            )
        else:
            self.real_time_metrics['success_rate_1h'] = 100.0

    def _cleanup_stale_communications(self):
        """Clean up communications that have been active too long."""
        current_time = datetime.now()
        stale_threshold = timedelta(hours=1)  # Consider stale after 1 hour

        stale_communications = []
        for comm_id, comm_record in self.active_communications.items():
            if current_time - comm_record['start_time'] > stale_threshold:
                stale_communications.append(comm_id)

        for comm_id in stale_communications:
            self.logger.log_warning(
                "CommunicationMonitor",
                f"⚠️ Cleaning up stale communication: {comm_id}",
                communication_id=comm_id
            )
            self.record_communication_completion(comm_id, success=False, error_message="Timeout")

    def _update_hourly_statistics(self):
        """Update hourly statistics."""
        current_hour = datetime.now().strftime("%Y-%m-%d %H:00")

        if current_hour not in self.metrics.hourly_statistics:
            self.metrics.hourly_statistics[current_hour] = {
                'total_communications': 0,
                'successful_communications': 0,
                'failed_communications': 0,
                'total_data_transferred': 0.0,
                'average_duration': 0.0,
                'peak_concurrent': 0,
                'communication_types': defaultdict(int),
                'service_pairs': defaultdict(int)
            }

        hour_stats = self.metrics.hourly_statistics[current_hour]

        # Update from recent communications
        hour_start = datetime.strptime(current_hour, "%Y-%m-%d %H:%M")
        hour_end = hour_start + timedelta(hours=1)

        hour_communications = [
            comm for comm in self.communication_history
            if hour_start <= comm.get('start_time', datetime.min) < hour_end
        ]

        hour_stats['total_communications'] = len(hour_communications)
        hour_stats['successful_communications'] = len([
            comm for comm in hour_communications
            if comm.get('status') == CommunicationStatus.COMPLETED
        ])
        hour_stats['failed_communications'] = len([
            comm for comm in hour_communications
            if comm.get('status') == CommunicationStatus.FAILED
        ])

        if hour_communications:
            completed_communications = [
                comm for comm in hour_communications
                if comm.get('actual_duration') is not None
            ]

            if completed_communications:
                total_duration = sum(comm['actual_duration'] for comm in completed_communications)
                hour_stats['average_duration'] = total_duration / len(completed_communications)

                hour_stats['total_data_transferred'] = sum(
                    comm.get('final_data_transferred', 0) for comm in completed_communications
                )

        # Clean up old hourly statistics (keep last 7 days)
        cutoff_time = datetime.now() - timedelta(days=7)
        cutoff_hour = cutoff_time.strftime("%Y-%m-%d %H:00")

        old_hours = [hour for hour in self.metrics.hourly_statistics.keys() if hour < cutoff_hour]
        for old_hour in old_hours:
            del self.metrics.hourly_statistics[old_hour]

    def _analyze_communication_patterns(self):
        """Analyze communication patterns for insights."""
        if len(self.communication_history) < 10:
            return

        # Analyze most common communication patterns
        recent_communications = list(self.communication_history)[-100:]  # Last 100 communications

        # Service pair analysis
        service_pair_frequency = defaultdict(int)
        type_frequency = defaultdict(int)

        for comm in recent_communications:
            pair = f"{comm['from_service_name']} → {comm['to_service_name']}"
            service_pair_frequency[pair] += 1
            type_frequency[comm['type'].value] += 1

        # Find top patterns
        top_pairs = sorted(service_pair_frequency.items(), key=lambda x: x[1], reverse=True)[:5]
        top_types = sorted(type_frequency.items(), key=lambda x: x[1], reverse=True)[:3]

        # Log insights
        if top_pairs:
            self.logger.log_info(
                "CommunicationAnalysis",
                f"📊 Top communication pairs: {', '.join([f'{pair}({count})' for pair, count in top_pairs[:3]])}"
            )

        if top_types:
            self.logger.log_info(
                "CommunicationAnalysis",
                f"📈 Most frequent types: {', '.join([f'{type_}({count})' for type_, count in top_types])}"
            )

    def _detect_anomalies(self):
        """Detect communication anomalies."""
        # Check for unusually long communications
        long_running_threshold = timedelta(minutes=30)
        current_time = datetime.now()

        long_running_comms = [
            comm_id for comm_id, comm in self.active_communications.items()
            if current_time - comm['start_time'] > long_running_threshold
        ]

        for comm_id in long_running_comms:
            comm = self.active_communications[comm_id]
            self.logger.log_warning(
                "CommunicationAnalysis",
                f"⏰ Long-running communication detected: {comm['from_service_name']} → {comm['to_service_name']} ({comm['type'].value})",
                communication_id=comm_id,
                details={
                    'duration_minutes': (current_time - comm['start_time']).total_seconds() / 60,
                    'progress': comm['progress']
                }
            )

        # Check for high failure rate
        recent_failures = [
            comm for comm in self.communication_history
            if comm.get('end_time', datetime.min) >= current_time - timedelta(minutes=10) and
               comm.get('status') == CommunicationStatus.FAILED
        ]

        if len(recent_failures) > 3:  # More than 3 failures in 10 minutes
            self.logger.log_warning(
                "CommunicationAnalysis",
                f"🚨 High failure rate detected: {len(recent_failures)} failures in last 10 minutes"
            )

    def _check_alerts(self):
        """Check for alert conditions."""
        current_active = len(self.active_communications)

        # Too many concurrent communications
        if current_active > self.alert_thresholds['max_concurrent_communications']:
            self.logger.log_warning(
                "CommunicationMonitor",
                f"⚠️ High concurrent communications: {current_active} active",
                details={'threshold': self.alert_thresholds['max_concurrent_communications']}
            )

        # Low success rate
        success_rate = self.real_time_metrics.get('success_rate_1h', 100.0)
        if success_rate < self.alert_thresholds['min_success_rate']:
            self.logger.log_warning(
                "CommunicationMonitor",
                f"⚠️ Low success rate: {success_rate:.1f}% in last hour",
                details={'threshold': self.alert_thresholds['min_success_rate']}
            )

    def get_statistics(self) -> Dict:
        """Get comprehensive communication statistics."""
        current_time = datetime.now()

        # Calculate success rate
        total_completed = self.metrics.successful_communications + self.metrics.failed_communications
        success_rate = (self.metrics.successful_communications / max(1, total_completed)) * 100

        # Recent statistics (last hour)
        hour_ago = current_time - timedelta(hours=1)
        recent_communications = [
            comm for comm in self.communication_history
            if comm.get('start_time', datetime.min) >= hour_ago
        ]

        recent_completed = [
            comm for comm in recent_communications
            if comm.get('status') in [CommunicationStatus.COMPLETED, CommunicationStatus.FAILED]
        ]

        recent_success_rate = 100.0
        if recent_completed:
            recent_successful = [
                comm for comm in recent_completed
                if comm.get('status') == CommunicationStatus.COMPLETED
            ]
            recent_success_rate = (len(recent_successful) / len(recent_completed)) * 100

        return {
            'total_communications': self.metrics.total_communications,
            'successful_communications': self.metrics.successful_communications,
            'failed_communications': self.metrics.failed_communications,
            'success_rate': success_rate,
            'recent_success_rate_1h': recent_success_rate,
            'average_duration': self.metrics.average_duration,
            'total_data_transferred_mb': self.metrics.total_data_transferred,
            'average_throughput_mbs': self.metrics.average_throughput,
            'peak_concurrent_communications': self.metrics.peak_concurrent_communications,
            'current_active_communications': len(self.active_communications),
            'communication_types': dict(self.metrics.communication_types),
            'top_service_pairs': dict(sorted(self.metrics.service_pairs.items(), key=lambda x: x[1], reverse=True)[:10]),
            'real_time_metrics': dict(self.real_time_metrics),
            'hourly_statistics': dict(self.metrics.hourly_statistics),
            'alert_status': {
                'high_concurrent': len(self.active_communications) > self.alert_thresholds['max_concurrent_communications'],
                'low_success_rate': recent_success_rate < self.alert_thresholds['min_success_rate']
            }
        }

    def export_detailed_report(self, filename: str, hours: int = 24):
        """Export detailed communication report."""
        import json

        current_time = datetime.now()
        cutoff_time = current_time - timedelta(hours=hours)

        # Filter communications from specified time period
        period_communications = [
            comm for comm in self.communication_history
            if comm.get('start_time', datetime.min) >= cutoff_time
        ]

        # Prepare report data
        report_data = {
            'report_metadata': {
                'generated_at': current_time.isoformat(),
                'period_hours': hours,
                'total_communications_in_period': len(period_communications)
            },
            'summary_statistics': self.get_statistics(),
            'detailed_communications': []
        }

        # Add detailed communication data
        for comm in period_communications:
            comm_data = {
                'id': comm['id'],
                'from_service': comm['from_service_name'],
                'to_service': comm['to_service_name'],
                'type': comm['type'].value,
                'start_time': comm['start_time'].isoformat(),
                'end_time': comm.get('end_time', {}).isoformat() if comm.get('end_time') else None,
                'duration_seconds': comm.get('actual_duration'),
                'data_size': comm['data_size'],
                'status': comm['status'].value,
                'progress': comm.get('progress', 0.0),
                'throughput_samples': len(comm.get('throughput_samples', [])),
                'peak_throughput': comm.get('peak_throughput', 0.0),
                'availability_zones': comm.get('availability_zones', {}),
                'service_types': {
                    'from': comm.get('from_service_type'),
                    'to': comm.get('to_service_type')
                }
            }

            if 'error_message' in comm:
                comm_data['error_message'] = comm['error_message']

            report_data['detailed_communications'].append(comm_data)

        # Write report
        with open(filename, 'w') as f:
            json.dump(report_data, f, indent=2)

        self.logger.log_info(
            "CommunicationMonitor",
            f"📄 Detailed report exported: {filename}",
            details={
                'period_hours': hours,
                'communications_included': len(period_communications)
            }
        )

        return filename

    def clear_history(self):
        """Clear communication history."""
        history_count = len(self.communication_history)
        self.communication_history.clear()

        # Reset metrics
        self.metrics = CommunicationMetrics()
        self.real_time_metrics.clear()

        self.logger.log_info(
            "CommunicationMonitor",
            f"🧹 Communication history cleared: {history_count} records removed"
        )