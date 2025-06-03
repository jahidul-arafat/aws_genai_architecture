# aws_genai_architecture/monitoring/console_logger.py
"""Enhanced console logging with detailed communication tracking."""

import sys
import threading
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, TextIO
from queue import Queue, Empty
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
from rich.live import Live
from rich.layout import Layout
from rich.box import ROUNDED
from colorama import init, Fore, Back, Style

from ..core.models import LogEntry, CommunicationType, CommunicationStatus

# Initialize colorama for Windows compatibility
init(autoreset=True)

class ConsoleLogger:
    """Enhanced console logger with real-time communication monitoring."""

    def __init__(
            self,
            level: str = "INFO",
            show_metrics: bool = True,
            show_communications: bool = True,
            show_services: bool = True,
            output_file: Optional[str] = None,
            max_log_entries: int = 1000,
            enable_colors: bool = True
    ):
        self.level = level.upper()
        self.show_metrics = show_metrics
        self.show_communications = show_communications
        self.show_services = show_services
        self.max_log_entries = max_log_entries
        self.enable_colors = enable_colors

        # Rich console for enhanced output
        self.console = Console(color_system="auto" if enable_colors else None)

        # Log storage
        self.log_queue = Queue()
        self.log_entries: List[LogEntry] = []

        # Threading
        self._running = False
        self._logger_thread: Optional[threading.Thread] = None

        # File output
        self.output_file: Optional[TextIO] = None
        if output_file:
            self.output_file = open(output_file, 'a', encoding='utf-8')

        # Metrics tracking
        self.current_metrics: Dict[str, Any] = {}
        self.communication_stats: Dict[str, int] = {}
        self.service_stats: Dict[str, Dict[str, Any]] = {}

        # Communication monitoring
        self.active_communications: Dict[str, Dict[str, Any]] = {}
        self.communication_history: List[Dict[str, Any]] = []

        # Level priorities
        self.level_priorities = {
            "DEBUG": 0,
            "INFO": 1,
            "WARNING": 2,
            "ERROR": 3,
            "CRITICAL": 4
        }

        # Emoji mappings for different communication types
        self.comm_type_emojis = {
            CommunicationType.DATA_TRANSFER: "🔄",
            CommunicationType.MODEL_SYNC: "🧠",
            CommunicationType.CONTROL_SIGNAL: "📡",
            CommunicationType.CHECKPOINT: "💾",
            CommunicationType.MONITORING: "👁️",
            CommunicationType.INFERENCE: "🎯",
            CommunicationType.PREPROCESSING: "⚙️"
        }

        # Status emojis
        self.status_emojis = {
            CommunicationStatus.PENDING: "⏳",
            CommunicationStatus.IN_PROGRESS: "🔄",
            CommunicationStatus.COMPLETED: "✅",
            CommunicationStatus.FAILED: "❌"
        }

    def start(self):
        """Start the console logger."""
        if self._running:
            return

        self._running = True
        self._logger_thread = threading.Thread(target=self._logging_loop, daemon=True)
        self._logger_thread.start()

        self.log_info("Console Logger", "🚀 AWS GenAI Architecture Console Logger Started")

    def stop(self):
        """Stop the console logger."""
        if not self._running:
            return

        self._running = False

        if self._logger_thread:
            self._logger_thread.join(timeout=2)

        if self.output_file:
            self.output_file.close()

    def log(
            self,
            level: str,
            source: str,
            message: str,
            service_id: Optional[str] = None,
            communication_id: Optional[str] = None,
            details: Optional[Dict[str, Any]] = None
    ):
        """Log a message with enhanced formatting."""
        level = level.upper()

        # Check if message should be logged based on level
        if self.level_priorities.get(level, 0) < self.level_priorities.get(self.level, 1):
            return

        log_entry = LogEntry(
            timestamp=datetime.now(),
            level=level,
            service_id=service_id,
            communication_id=communication_id,
            message=f"[{source}] {message}",
            details=details or {}
        )

        self.log_queue.put(log_entry)

    def log_debug(self, source: str, message: str, **kwargs):
        """Log debug message."""
        self.log("DEBUG", source, message, **kwargs)

    def log_info(self, source: str, message: str, **kwargs):
        """Log info message."""
        self.log("INFO", source, message, **kwargs)

    def log_warning(self, source: str, message: str, **kwargs):
        """Log warning message."""
        self.log("WARNING", source, message, **kwargs)

    def log_error(self, source: str, message: str, **kwargs):
        """Log error message."""
        self.log("ERROR", source, message, **kwargs)

    def log_critical(self, source: str, message: str, **kwargs):
        """Log critical message."""
        self.log("CRITICAL", source, message, **kwargs)

    def log_communication_started(
            self,
            comm_id: str,
            from_service: str,
            to_service: str,
            comm_type: CommunicationType,
            data_size: str,
            throughput: str,
            protocol: str,
            estimated_duration: float
    ):
        """Log communication start with detailed information."""
        emoji = self.comm_type_emojis.get(comm_type, "📡")

        message = f"{emoji} {comm_type.value.upper()}: {from_service} → {to_service}"

        details = {
            "data_size": data_size,
            "throughput": throughput,
            "protocol": protocol,
            "estimated_duration": f"{estimated_duration:.1f}s",
            "from_service": from_service,
            "to_service": to_service
        }

        # Store active communication
        self.active_communications[comm_id] = {
            "start_time": datetime.now(),
            "from_service": from_service,
            "to_service": to_service,
            "type": comm_type,
            "data_size": data_size,
            "throughput": throughput,
            "protocol": protocol,
            "progress": 0.0,
            "status": CommunicationStatus.IN_PROGRESS
        }

        self.log_info("Communication", message, communication_id=comm_id, details=details)

        # Print detailed breakdown
        self._print_communication_details(comm_id, details, "STARTED")

    def log_communication_progress(
            self,
            comm_id: str,
            progress: float,
            current_throughput: Optional[str] = None,
            bytes_transferred: Optional[str] = None
    ):
        """Log communication progress update."""
        if comm_id in self.active_communications:
            comm_info = self.active_communications[comm_id]
            comm_info["progress"] = progress

            if progress >= 1.0:
                self.log_communication_completed(comm_id)
            else:
                # Only log progress for long-running communications
                elapsed = (datetime.now() - comm_info["start_time"]).total_seconds()
                if elapsed > 5:  # Only show progress for communications > 5 seconds
                    emoji = self.comm_type_emojis.get(comm_info["type"], "📡")
                    message = f"{emoji} Progress: {comm_info['from_service']} → {comm_info['to_service']} ({progress*100:.1f}%)"

                    details = {
                        "progress_percent": f"{progress*100:.1f}%",
                        "elapsed_seconds": f"{elapsed:.1f}s"
                    }

                    if current_throughput:
                        details["current_throughput"] = current_throughput
                    if bytes_transferred:
                        details["bytes_transferred"] = bytes_transferred

                    self.log_debug("Communication", message, communication_id=comm_id, details=details)

    def log_communication_completed(self, comm_id: str):
        """Log communication completion."""
        if comm_id not in self.active_communications:
            return

        comm_info = self.active_communications[comm_id]
        duration = (datetime.now() - comm_info["start_time"]).total_seconds()

        emoji = self.comm_type_emojis.get(comm_info["type"], "📡")
        status_emoji = self.status_emojis[CommunicationStatus.COMPLETED]

        message = f"{emoji} {status_emoji} COMPLETED: {comm_info['from_service']} → {comm_info['to_service']}"

        details = {
            "duration_seconds": f"{duration:.1f}s",
            "data_size": comm_info["data_size"],
            "average_throughput": comm_info["throughput"],
            "protocol": comm_info["protocol"]
        }

        self.log_info("Communication", message, communication_id=comm_id, details=details)

        # Move to history and remove from active
        comm_info["status"] = CommunicationStatus.COMPLETED
        comm_info["end_time"] = datetime.now()
        comm_info["duration"] = duration

        self.communication_history.append(comm_info.copy())
        del self.active_communications[comm_id]

        # Update stats
        comm_type_str = comm_info["type"].value
        if comm_type_str not in self.communication_stats:
            self.communication_stats[comm_type_str] = 0
        self.communication_stats[comm_type_str] += 1

        # Print completion details
        self._print_communication_details(comm_id, details, "COMPLETED")

    def log_communication_failed(self, comm_id: str, error_message: str):
        """Log communication failure."""
        if comm_id not in self.active_communications:
            return

        comm_info = self.active_communications[comm_id]
        duration = (datetime.now() - comm_info["start_time"]).total_seconds()

        emoji = self.comm_type_emojis.get(comm_info["type"], "📡")
        status_emoji = self.status_emojis[CommunicationStatus.FAILED]

        message = f"{emoji} {status_emoji} FAILED: {comm_info['from_service']} → {comm_info['to_service']}"

        details = {
            "error": error_message,
            "duration_seconds": f"{duration:.1f}s",
            "progress_at_failure": f"{comm_info.get('progress', 0)*100:.1f}%"
        }

        self.log_error("Communication", message, communication_id=comm_id, details=details)

        # Move to history and remove from active
        comm_info["status"] = CommunicationStatus.FAILED
        comm_info["end_time"] = datetime.now()
        comm_info["duration"] = duration
        comm_info["error"] = error_message

        self.communication_history.append(comm_info.copy())
        del self.active_communications[comm_id]

    def log_service_metrics(self, service_id: str, metrics: Dict[str, Any]):
        """Log service metrics update."""
        self.service_stats[service_id] = {
            **metrics,
            "last_updated": datetime.now()
        }

        # Only log significant metric changes
        if self._is_significant_metric_change(service_id, metrics):
            utilization = metrics.get("utilization", 0)
            cost = metrics.get("cost_per_hour", 0)

            message = f"📊 Metrics Update: {service_id}"
            details = {
                "utilization": f"{utilization:.1f}%",
                "cost_per_hour": f"${cost}",
                "throughput": metrics.get("throughput", "N/A"),
                "latency": f"{metrics.get('latency', 0):.1f}ms"
            }

            self.log_debug("Metrics", message, service_id=service_id, details=details)

    def log_system_metrics(self, metrics: Dict[str, Any]):
        """Log system-wide metrics."""
        self.current_metrics = {
            **metrics,
            "timestamp": datetime.now()
        }

        # Log every 30 seconds
        if not hasattr(self, "_last_system_log"):
            self._last_system_log = datetime.now()

        if (datetime.now() - self._last_system_log).total_seconds() > 30:
            message = "📈 System Metrics Update"
            details = {
                "total_services": metrics.get("active_services", 0),
                "active_communications": metrics.get("active_communications", 0),
                "total_cost_per_hour": f"${metrics.get('total_cost_per_hour', 0)}",
                "average_utilization": f"{metrics.get('average_utilization', 0):.1f}%",
                "reliability_score": f"{metrics.get('reliability_score', 0):.1f}%"
            }

            self.log_info("System", message, details=details)
            self._last_system_log = datetime.now()

    def _is_significant_metric_change(self, service_id: str, new_metrics: Dict[str, Any]) -> bool:
        """Check if metric change is significant enough to log."""
        if service_id not in self.service_stats:
            return True

        old_metrics = self.service_stats[service_id]

        # Check utilization change > 10%
        old_util = old_metrics.get("utilization", 0)
        new_util = new_metrics.get("utilization", 0)

        return abs(new_util - old_util) > 10

    def _print_communication_details(self, comm_id: str, details: Dict[str, Any], status: str):
        """Print detailed communication information."""
        if not self.show_communications:
            return

        # Create detailed panel
        detail_lines = []
        for key, value in details.items():
            formatted_key = key.replace("_", " ").title()
            detail_lines.append(f"├─ {formatted_key}: {value}")

        if detail_lines:
            detail_lines[-1] = detail_lines[-1].replace("├─", "└─")

        detail_text = "\n".join(detail_lines)

        # Color coding based on status
        if status == "COMPLETED":
            panel_style = "green"
        elif status == "FAILED":
            panel_style = "red"
        else:
            panel_style = "blue"

        # Print to console
        if self.enable_colors:
            panel = Panel(
                detail_text,
                title=f"Communication {status}",
                border_style=panel_style,
                box=ROUNDED
            )
            self.console.print(panel)
        else:
            print(f"\n[{status}] Communication Details:")
            print(detail_text)
            print()

    def print_communication_summary(self):
        """Print a summary of communication statistics."""
        if not self.communication_stats:
            self.console.print("📊 No communications recorded yet.")
            return

        # Create summary table
        table = Table(title="🔄 Communication Summary", box=ROUNDED)
        table.add_column("Communication Type", style="cyan")
        table.add_column("Count", justify="right", style="green")
        table.add_column("Success Rate", justify="right", style="yellow")

        total_communications = sum(self.communication_stats.values())

        for comm_type, count in self.communication_stats.items():
            # Calculate success rate from history
            type_history = [c for c in self.communication_history
                            if c["type"].value == comm_type]
            successful = len([c for c in type_history
                              if c["status"] == CommunicationStatus.COMPLETED])
            success_rate = (successful / len(type_history) * 100) if type_history else 0

            table.add_row(
                comm_type.replace("-", " ").title(),
                str(count),
                f"{success_rate:.1f}%"
            )

        self.console.print(table)

        # Active communications
        if self.active_communications:
            active_table = Table(title="🔄 Active Communications", box=ROUNDED)
            active_table.add_column("From → To", style="cyan")
            active_table.add_column("Type", style="green")
            active_table.add_column("Progress", style="yellow")
            active_table.add_column("Duration", style="blue")

            for comm_id, comm_info in self.active_communications.items():
                duration = (datetime.now() - comm_info["start_time"]).total_seconds()
                progress = comm_info.get("progress", 0)

                active_table.add_row(
                    f"{comm_info['from_service']} → {comm_info['to_service']}",
                    comm_info["type"].value.replace("-", " ").title(),
                    f"{progress*100:.1f}%",
                    f"{duration:.1f}s"
                )

            self.console.print(active_table)

    def print_service_summary(self):
        """Print a summary of service statistics."""
        if not self.service_stats:
            self.console.print("📊 No service metrics recorded yet.")
            return

        # Create service summary table
        table = Table(title="🏗️ Service Summary", box=ROUNDED)
        table.add_column("Service ID", style="cyan")
        table.add_column("Utilization", justify="right", style="green")
        table.add_column("Cost/Hour", justify="right", style="yellow")
        table.add_column("Throughput", justify="right", style="blue")
        table.add_column("Latency", justify="right", style="red")

        for service_id, metrics in self.service_stats.items():
            utilization = metrics.get("utilization", 0)
            cost = metrics.get("cost_per_hour", 0)
            throughput = metrics.get("throughput", "N/A")
            latency = metrics.get("latency", 0)

            # Color code utilization
            util_style = "green" if utilization < 70 else "yellow" if utilization < 90 else "red"

            table.add_row(
                service_id,
                f"[{util_style}]{utilization:.1f}%[/{util_style}]",
                f"${cost}",
                str(throughput),
                f"{latency:.1f}ms"
            )

        self.console.print(table)

    def print_metrics_dashboard(self):
        """Print a real-time metrics dashboard."""
        if not self.current_metrics:
            return

        # Create layout
        layout = Layout()

        # System metrics panel
        metrics_text = []
        for key, value in self.current_metrics.items():
            if key != "timestamp":
                formatted_key = key.replace("_", " ").title()
                metrics_text.append(f"{formatted_key}: {value}")

        metrics_panel = Panel(
            "\n".join(metrics_text),
            title="📊 System Metrics",
            border_style="green"
        )

        # Active communications count
        active_count = len(self.active_communications)
        comm_panel = Panel(
            f"Active: {active_count}\nCompleted: {len(self.communication_history)}",
            title="🔄 Communications",
            border_style="blue"
        )

        # Cost summary
        total_cost = self.current_metrics.get("total_cost_per_hour", 0)
        cost_panel = Panel(
            f"${total_cost}/hour\n${total_cost * 24}/day\n${total_cost * 24 * 30}/month",
            title="💰 Cost Projection",
            border_style="yellow"
        )

        self.console.print(metrics_panel)
        self.console.print(comm_panel)
        self.console.print(cost_panel)

    def _logging_loop(self):
        """Main logging loop."""
        while self._running:
            try:
                # Process log queue
                while not self.log_queue.empty():
                    try:
                        log_entry = self.log_queue.get_nowait()
                        self._process_log_entry(log_entry)
                    except Empty:
                        break

                # Clean up old log entries
                if len(self.log_entries) > self.max_log_entries:
                    self.log_entries = self.log_entries[-self.max_log_entries:]

                time.sleep(0.1)  # Check queue 10 times per second

            except Exception as e:
                print(f"Error in logging loop: {e}")

    def _process_log_entry(self, log_entry: LogEntry):
        """Process a single log entry."""
        self.log_entries.append(log_entry)

        # Format and print log entry
        self._print_log_entry(log_entry)

        # Write to file if configured
        if self.output_file:
            self.output_file.write(str(log_entry) + "\n")
            self.output_file.flush()

    def _print_log_entry(self, log_entry: LogEntry):
        """Print a formatted log entry."""
        timestamp_str = log_entry.timestamp.strftime("%Y-%m-%d %H:%M:%S")

        # Color coding by level
        level_colors = {
            "DEBUG": "dim white",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "bold red"
        }

        if self.enable_colors:
            level_style = level_colors.get(log_entry.level, "white")

            # Create formatted text
            text = Text()
            text.append(f"[{timestamp_str}] ", style="dim")
            text.append(f"{log_entry.level}", style=level_style)

            if log_entry.service_id:
                text.append(f" [{log_entry.service_id}]", style="cyan")
            if log_entry.communication_id:
                text.append(f" [{log_entry.communication_id}]", style="blue")

            text.append(f": {log_entry.message}")

            self.console.print(text)

            # Print details if available
            if log_entry.details and self.level == "DEBUG":
                details_text = []
                for key, value in log_entry.details.items():
                    details_text.append(f"  {key}: {value}")

                if details_text:
                    self.console.print("\n".join(details_text), style="dim")
        else:
            # Plain text output
            prefix = f"[{timestamp_str}] {log_entry.level}"

            if log_entry.service_id:
                prefix += f" [{log_entry.service_id}]"
            if log_entry.communication_id:
                prefix += f" [{log_entry.communication_id}]"

            print(f"{prefix}: {log_entry.message}")

            # Print details
            if log_entry.details and self.level == "DEBUG":
                for key, value in log_entry.details.items():
                    print(f"  {key}: {value}")

    def get_log_entries(self, level: Optional[str] = None, limit: Optional[int] = None) -> List[LogEntry]:
        """Get log entries with optional filtering."""
        entries = self.log_entries

        if level:
            level = level.upper()
            entries = [entry for entry in entries if entry.level == level]

        if limit:
            entries = entries[-limit:]

        return entries

    def export_logs(self, filename: str, format: str = "txt"):
        """Export logs to file."""
        if format.lower() == "json":
            import json

            data = []
            for entry in self.log_entries:
                data.append({
                    "timestamp": entry.timestamp.isoformat(),
                    "level": entry.level,
                    "service_id": entry.service_id,
                    "communication_id": entry.communication_id,
                    "message": entry.message,
                    "details": entry.details
                })

            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
        else:
            # Plain text format
            with open(filename, 'w') as f:
                for entry in self.log_entries:
                    f.write(str(entry) + "\n")
                    if entry.details:
                        for key, value in entry.details.items():
                            f.write(f"  {key}: {value}\n")

        self.log_info("Logger", f"Exported {len(self.log_entries)} log entries to {filename}")

    def clear_logs(self):
        """Clear all log entries."""
        count = len(self.log_entries)
        self.log_entries.clear()
        self.communication_history.clear()
        self.communication_stats.clear()
        self.service_stats.clear()

        self.log_info("Logger", f"Cleared {count} log entries")