# aws_genai_architecture/core/metrics.py (UPDATED - Fix missing imports)
"""Metrics collection and analysis for AWS GenAI Architecture."""

import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from queue import Queue

from .models import SystemMetrics, ServiceInstance, CommunicationInstance


@dataclass
class MetricsSnapshot:
    """Point-in-time metrics snapshot."""
    timestamp: datetime
    system_metrics: SystemMetrics
    service_metrics: Dict[str, Dict[str, Any]]
    communication_metrics: Dict[str, Any]
    training_metrics: Optional[Dict[str, Any]] = None


class MetricsCollector:
    """Collects and analyzes system metrics."""

    def __init__(self, history_size: int = 1000):
        self.history_size = history_size
        self.metrics_history: List[MetricsSnapshot] = []
        self.metrics_queue = Queue()

        # Aggregated metrics
        self.hourly_metrics: Dict[str, List[float]] = {}
        self.daily_metrics: Dict[str, List[float]] = {}

        # Threading
        self._running = False
        self._collector_thread: Optional[threading.Thread] = None

    def start_collection(self):
        """Start metrics collection."""
        if self._running:
            return

        self._running = True
        self._collector_thread = threading.Thread(target=self._collection_loop, daemon=True)
        self._collector_thread.start()

    def stop_collection(self):
        """Stop metrics collection."""
        self._running = False
        if self._collector_thread:
            self._collector_thread.join(timeout=1)

    def record_snapshot(
            self,
            services: List[ServiceInstance],
            communications: List[CommunicationInstance],
            training_session: Optional[Any] = None
    ) -> MetricsSnapshot:
        """Record a metrics snapshot."""

        # Calculate system metrics
        total_cost = sum(service.metrics.cost_per_hour for service in services)
        avg_utilization = sum(service.metrics.utilization for service in services) / max(1, len(services))

        # Calculate reliability
        healthy_services = len([s for s in services if s.health_status.value == "Healthy"])
        reliability = (healthy_services / max(1, len(services))) * 100

        system_metrics = SystemMetrics(
            active_services=len(services),
            active_communications=len(communications),
            total_cost_per_hour=total_cost,
            average_utilization=avg_utilization,
            total_throughput="Calculated",  # Would calculate from services
            reliability_score=reliability,
            training_progress=0.0,  # Updated if training active
            data_processed="2.3TB",  # Simulated
            network_io="450 MB/s"  # Simulated
        )

        # Collect service metrics
        service_metrics = {}
        for service in services:
            service_metrics[service.id] = {
                "utilization": service.metrics.utilization,
                "cost_per_hour": service.metrics.cost_per_hour,
                "throughput": service.metrics.throughput,
                "latency": service.metrics.latency,
                "uptime": service.metrics.uptime,
                "error_rate": service.metrics.error_rate,
                "health_status": service.health_status.value,
                "instance_type": service.config.instance_type,
                "availability_zone": service.availability_zone
            }

        # Collect communication metrics
        comm_metrics = {
            "total_communications": len(communications),
            "types_breakdown": {},
            "average_latency": 0.0,
            "total_data_transferred": "0 MB"
        }

        if communications:
            # Group by type
            type_counts = {}
            total_latency = 0

            for comm in communications:
                comm_type = comm.communication_type.value
                type_counts[comm_type] = type_counts.get(comm_type, 0) + 1
                total_latency += comm.latency

            comm_metrics["types_breakdown"] = type_counts
            comm_metrics["average_latency"] = total_latency / len(communications)

        # Training metrics
        training_metrics = None
        if training_session:
            training_metrics = {
                "session_id": training_session.id,
                "model_name": training_session.model_name,
                "current_epoch": training_session.current_epoch,
                "current_loss": training_session.current_loss,
                "progress": getattr(training_session, 'progress', 0),
                "status": training_session.status
            }
            system_metrics.training_progress = training_metrics["progress"]

        snapshot = MetricsSnapshot(
            timestamp=datetime.now(),
            system_metrics=system_metrics,
            service_metrics=service_metrics,
            communication_metrics=comm_metrics,
            training_metrics=training_metrics
        )

        # Store snapshot
        self.metrics_history.append(snapshot)

        # Trim history if too large
        if len(self.metrics_history) > self.history_size:
            self.metrics_history = self.metrics_history[-self.history_size:]

        # Update aggregated metrics
        self._update_aggregated_metrics(snapshot)

        return snapshot

    def _update_aggregated_metrics(self, snapshot: MetricsSnapshot):
        """Update hourly and daily aggregated metrics."""
        timestamp = snapshot.timestamp
        hour_key = timestamp.strftime("%Y-%m-%d %H:00")
        day_key = timestamp.strftime("%Y-%m-%d")

        # Hourly metrics
        if hour_key not in self.hourly_metrics:
            self.hourly_metrics[hour_key] = []

        hourly_data = {
            "total_cost": snapshot.system_metrics.total_cost_per_hour,
            "avg_utilization": snapshot.system_metrics.average_utilization,
            "reliability": snapshot.system_metrics.reliability_score,
            "active_services": snapshot.system_metrics.active_services,
            "active_communications": snapshot.system_metrics.active_communications
        }
        self.hourly_metrics[hour_key].append(hourly_data)

        # Daily metrics (similar structure)
        if day_key not in self.daily_metrics:
            self.daily_metrics[day_key] = []

        self.daily_metrics[day_key].append(hourly_data)

        # Clean up old metrics (keep last 7 days)
        cutoff_date = datetime.now() - timedelta(days=7)
        cutoff_hour = cutoff_date.strftime("%Y-%m-%d %H:00")
        cutoff_day = cutoff_date.strftime("%Y-%m-%d")

        # Remove old entries
        self.hourly_metrics = {k: v for k, v in self.hourly_metrics.items() if k >= cutoff_hour}
        self.daily_metrics = {k: v for k, v in self.daily_metrics.items() if k >= cutoff_day}

    def get_latest_snapshot(self) -> Optional[MetricsSnapshot]:
        """Get the most recent metrics snapshot."""
        return self.metrics_history[-1] if self.metrics_history else None

    def get_metrics_history(self, hours: int = 24) -> List[MetricsSnapshot]:
        """Get metrics history for specified hours."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [
            snapshot for snapshot in self.metrics_history
            if snapshot.timestamp >= cutoff_time
        ]

    def get_service_trend(self, service_id: str, metric: str, hours: int = 24) -> List[tuple]:
        """Get trend data for a specific service metric."""
        history = self.get_metrics_history(hours)
        trend_data = []

        for snapshot in history:
            if service_id in snapshot.service_metrics:
                service_data = snapshot.service_metrics[service_id]
                if metric in service_data:
                    trend_data.append((snapshot.timestamp, service_data[metric]))

        return trend_data

    def get_cost_analysis(self, hours: int = 24) -> Dict[str, Any]:
        """Get cost analysis for specified period."""
        history = self.get_metrics_history(hours)

        if not history:
            return {"error": "No metrics data available"}

        costs = [snapshot.system_metrics.total_cost_per_hour for snapshot in history]

        total_cost = sum(costs) * (len(costs) / 60)  # Assuming snapshots every minute
        avg_cost = sum(costs) / len(costs)
        max_cost = max(costs)
        min_cost = min(costs)

        # Cost breakdown by service type
        service_type_costs = {}
        latest_snapshot = history[-1]

        for service_id, metrics in latest_snapshot.service_metrics.items():
            instance_type = metrics.get("instance_type", "Unknown")
            cost = metrics.get("cost_per_hour", 0)

            if instance_type not in service_type_costs:
                service_type_costs[instance_type] = 0
            service_type_costs[instance_type] += cost

        return {
            "period_hours": hours,
            "total_cost": total_cost,
            "average_cost_per_hour": avg_cost,
            "max_cost_per_hour": max_cost,
            "min_cost_per_hour": min_cost,
            "cost_trend": [(s.timestamp, s.system_metrics.total_cost_per_hour) for s in history[-20:]],
            "service_type_breakdown": service_type_costs,
            "projected_monthly_cost": avg_cost * 24 * 30
        }

    def get_performance_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get performance summary for specified period."""
        history = self.get_metrics_history(hours)

        if not history:
            return {"error": "No metrics data available"}

        # Utilization statistics
        utilizations = [s.system_metrics.average_utilization for s in history]
        avg_utilization = sum(utilizations) / len(utilizations)
        max_utilization = max(utilizations)
        min_utilization = min(utilizations)

        # Reliability statistics
        reliability_scores = [s.system_metrics.reliability_score for s in history]
        avg_reliability = sum(reliability_scores) / len(reliability_scores)
        min_reliability = min(reliability_scores)

        # Communication statistics
        total_communications = sum(s.system_metrics.active_communications for s in history)
        avg_communications = total_communications / len(history)

        return {
            "period_hours": hours,
            "utilization": {
                "average": avg_utilization,
                "maximum": max_utilization,
                "minimum": min_utilization,
                "trend": [(s.timestamp, s.system_metrics.average_utilization) for s in history[-20:]]
            },
            "reliability": {
                "average": avg_reliability,
                "minimum": min_reliability,
                "uptime_percentage": avg_reliability
            },
            "communications": {
                "total_processed": total_communications,
                "average_active": avg_communications
            },
            "service_count": {
                "average": sum(s.system_metrics.active_services for s in history) / len(history),
                "maximum": max(s.system_metrics.active_services for s in history)
            }
        }

    def _collection_loop(self):
        """Background metrics collection loop."""
        while self._running:
            try:
                # Process any queued metrics
                while not self.metrics_queue.empty():
                    metric_data = self.metrics_queue.get_nowait()
                    # Process metric data if needed

                time.sleep(1)  # Collection interval

            except Exception as e:
                print(f"Error in metrics collection loop: {e}")