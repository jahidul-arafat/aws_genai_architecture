# aws_genai_architecture/cli.py
"""Command line interface for AWS GenAI Architecture Visualizer."""

import argparse
import json
import sys
import time
import signal
from pathlib import Path
from typing import Optional, Dict, Any

from .core.architecture import AWSGenAIArchitecture
from .monitoring.console_logger import ConsoleLogger
from .web.app import create_app

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Configuration file not found: {config_path}")
        return {}
    except json.JSONDecodeError as e:
        print(f"Invalid JSON in configuration file: {e}")
        return {}

def create_default_config() -> Dict[str, Any]:
    """Create default configuration."""
    return {
        "services": {
            "auto_initialize": True,
            "ec2_training": {
                "instance_type": "p4d.24xlarge",
                "count": 2,
                "spot_instances": True
            },
            "sagemaker": {
                "instance_type": "ml.p4d.24xlarge",
                "managed_spot": False
            }
        },
        "communication": {
            "auto_generate": True,
            "frequency": 1.8,
            "types": ["data-transfer", "model-sync", "monitoring"]
        },
        "monitoring": {
            "console_output": True,
            "log_level": "INFO",
            "metrics_interval": 2.0,
            "detailed_logging": True,
            "export_logs": False,
            "log_file": None
        },
        "web": {
            "enabled": True,
            "host": "0.0.0.0",
            "port": 8080,
            "debug": False
        }
    }

def run_console_mode(architecture: AWSGenAIArchitecture, logger: ConsoleLogger, config: Dict[str, Any]):
    """Run in console-only mode."""
    print("🚀 AWS GenAI Architecture Visualizer - Console Mode")
    print("=" * 60)
    print("Commands:")
    print("  start-training  - Start ML training simulation")
    print("  stop-training   - Stop training")
    print("  scale-out       - Add more training instances")
    print("  scale-in        - Remove training instances")
    print("  checkpoint      - Save model checkpoint")
    print("  metrics         - Show current metrics")
    print("  communications  - Show communication summary")
    print("  services        - Show service summary")
    print("  dashboard       - Show metrics dashboard")
    print("  help           - Show this help")
    print("  quit           - Exit")
    print("=" * 60)

    # Start architecture
    architecture.start()

    try:
        while True:
            try:
                command = input("\n🎮 Enter command: ").strip().lower()

                if command in ['quit', 'exit', 'q']:
                    break
                elif command == 'start-training':
                    architecture.start_training()
                    print("✅ Training started")
                elif command == 'stop-training':
                    architecture.stop_training()
                    print("⏹️ Training stopped")
                elif command == 'scale-out':
                    count = input("Number of instances to add (default 2): ").strip()
                    count = int(count) if count.isdigit() else 2
                    architecture.scale_out(count=count)
                    print(f"📈 Added {count} instances")
                elif command == 'scale-in':
                    count = input("Number of instances to remove (default 1): ").strip()
                    count = int(count) if count.isdigit() else 1
                    architecture.scale_in(count=count)
                    print(f"📉 Removed {count} instances")
                elif command == 'checkpoint':
                    architecture.save_checkpoint()
                    print("💾 Checkpoint saved")
                elif command == 'metrics':
                    metrics = architecture.get_current_metrics()
                    print("\n📊 Current Metrics:")
                    for key, value in metrics.items():
                        print(f"  {key.replace('_', ' ').title()}: {value}")
                elif command == 'communications':
                    logger.print_communication_summary()
                elif command == 'services':
                    logger.print_service_summary()
                elif command == 'dashboard':
                    logger.print_metrics_dashboard()
                elif command == 'help':
                    print("Available commands: start-training, stop-training, scale-out, scale-in, checkpoint, metrics, communications, services, dashboard, help, quit")
                else:
                    print(f"Unknown command: {command}. Type 'help' for available commands.")

            except KeyboardInterrupt:
                print("\n\n👋 Shutting down...")
                break
            except EOFError:
                break
            except Exception as e:
                print(f"Error executing command: {e}")

    finally:
        architecture.stop()
        logger.stop()

def run_web_mode(architecture: AWSGenAIArchitecture, config: Dict[str, Any]):
    """Run in web mode."""
    web_config = config.get("web", {})

    app = create_app(architecture)

    host = web_config.get("host", "0.0.0.0")
    port = web_config.get("port", 8080)
    debug = web_config.get("debug", False)

    print(f"🌐 Starting web server on http://{host}:{port}")
    print("Press Ctrl+C to stop")

    try:
        app.run(host=host, port=port, debug=debug)
    except KeyboardInterrupt:
        print("\n👋 Shutting down web server...")
    finally:
        architecture.stop()

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="AWS GenAI Architecture Visualizer - Interactive ML training infrastructure simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  aws-genai-viz --mode web --port 8080
  aws-genai-viz --mode console --config my_config.json
  aws-genai-viz --debug --log-file training.log
        """
    )

    parser.add_argument(
        "--mode",
        choices=["web", "console"],
        default="web",
        help="Run mode: web interface or console only (default: web)"
    )

    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration JSON file"
    )

    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind web server to (default: 0.0.0.0)"
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port for web server (default: 8080)"
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Logging level (default: INFO)"
    )

    parser.add_argument(
        "--log-file",
        type=str,
        help="Optional log file to write to"
    )

    parser.add_argument(
        "--no-colors",
        action="store_true",
        help="Disable colored output"
    )

    parser.add_argument(
        "--export-config",
        type=str,
        help="Export default configuration to file and exit"
    )

    parser.add_argument(
        "--auto-start-training",
        action="store_true",
        help="Automatically start training simulation on startup"
    )

    args = parser.parse_args()

    # Handle config export
    if args.export_config:
        default_config = create_default_config()
        with open(args.export_config, 'w') as f:
            json.dump(default_config, f, indent=2)
        print(f"✅ Default configuration exported to {args.export_config}")
        return 0

    # Load configuration
    config = create_default_config()
    if args.config:
        user_config = load_config(args.config)
        # Merge configurations (user config overrides defaults)
        config.update(user_config)

    # Override config with command line arguments
    if args.debug:
        config["monitoring"]["log_level"] = "DEBUG"
        config["web"]["debug"] = True

    if args.log_level:
        config["monitoring"]["log_level"] = args.log_level

    if args.log_file:
        config["monitoring"]["log_file"] = args.log_file

    if args.host != "0.0.0.0":
        config["web"]["host"] = args.host

    if args.port != 8080:
        config["web"]["port"] = args.port

    # Initialize logger
    logger = ConsoleLogger(
        level=config["monitoring"]["log_level"],
        show_metrics=config["monitoring"].get("detailed_logging", True),
        output_file=config["monitoring"].get("log_file"),
        enable_colors=not args.no_colors
    )

    # Initialize architecture
    try:
        architecture = AWSGenAIArchitecture(config=config, logger=logger)

        # Register signal handlers for graceful shutdown
        def signal_handler(signum, frame):
            print(f"\n🛑 Received signal {signum}, shutting down...")
            architecture.stop()
            logger.stop()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # Auto-start training if requested
        if args.auto_start_training:
            architecture.start_training()

        # Run in specified mode
        if args.mode == "console":
            return run_console_mode(architecture, logger, config)
        else:
            return run_web_mode(architecture, config)

    except Exception as e:
        print(f"❌ Error starting AWS GenAI Architecture: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())