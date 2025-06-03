# aws_genai_architecture/web/app.py
"""Flask web application for AWS GenAI Architecture visualization."""

import json
from datetime import datetime
from typing import Dict, Any, Optional
from flask import Flask, render_template, jsonify, request, send_from_directory
from flask_socketio import SocketIO, emit
import threading
import time

from ..core.architecture import AWSGenAIArchitecture
from ..core.models import CommunicationType, ServiceType


def create_app(architecture: Optional[AWSGenAIArchitecture] = None) -> Flask:
    """Create Flask application with SocketIO support."""

    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'aws-genai-architecture-secret-key-2024'

    # Initialize SocketIO
    socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

    # Store architecture instance
    if architecture is None:
        architecture = AWSGenAIArchitecture()

    app.architecture = architecture

    # Real-time update thread
    update_thread = None

    def start_real_time_updates():
        """Start real-time updates via WebSocket."""
        def update_loop():
            while True:
                try:
                    if hasattr(app, 'architecture'):
                        # Get current metrics
                        metrics = app.architecture.get_current_metrics()
                        socketio.emit('metrics_update', metrics)

                        # Get services data
                        services_data = []
                        for service in app.architecture.get_services():
                            service_data = {
                                'id': service.id,
                                'name': service.name,
                                'type': service.service_type.value,
                                'position': {'x': service.position.x, 'y': service.position.y},
                                'radius': service.radius,
                                'utilization': service.metrics.utilization,
                                'health_status': service.health_status.value,
                                'cost_per_hour': service.metrics.cost_per_hour,
                                'activity_level': service.activity_level,
                                'instance_type': service.config.instance_type,
                                'availability_zone': service.availability_zone
                            }
                            services_data.append(service_data)

                        socketio.emit('services_update', services_data)

                        # Get communications data
                        communications_data = []
                        for comm in app.architecture.get_communications():
                            comm_data = {
                                'id': comm.id,
                                'from_service_id': comm.from_service_id,
                                'to_service_id': comm.to_service_id,
                                'type': comm.communication_type.value,
                                'progress': comm.progress,
                                'data_size': comm.data_size,
                                'throughput': comm.throughput,
                                'latency': comm.latency,
                                'status': comm.status.value
                            }
                            communications_data.append(comm_data)

                        socketio.emit('communications_update', communications_data)

                        # Training status
                        training_status = app.architecture.get_training_status()
                        if training_status:
                            socketio.emit('training_update', training_status)

                    time.sleep(2)  # Update every 2 seconds

                except Exception as e:
                    print(f"Error in real-time update loop: {e}")
                    time.sleep(5)

        nonlocal update_thread
        if update_thread is None or not update_thread.is_alive():
            update_thread = threading.Thread(target=update_loop, daemon=True)
            update_thread.start()

    @app.route('/')
    def index():
        """Main dashboard page."""
        return render_template('index.html')

    @app.route('/api/status')
    def api_status():
        """API status endpoint."""
        return jsonify({
            'status': 'online',
            'timestamp': datetime.now().isoformat(),
            'version': '0.1.0'
        })

    @app.route('/api/services')
    def api_services():
        """Get all services."""
        services = []
        for service in app.architecture.get_services():
            service_data = {
                'id': service.id,
                'name': service.name,
                'type': service.service_type.value,
                'position': {'x': service.position.x, 'y': service.position.y},
                'radius': service.radius,
                'metrics': {
                    'utilization': service.metrics.utilization,
                    'cost_per_hour': service.metrics.cost_per_hour,
                    'throughput': service.metrics.throughput,
                    'latency': service.metrics.latency,
                    'uptime': service.metrics.uptime
                },
                'config': {
                    'instance_type': service.config.instance_type,
                    'vcpus': service.config.vcpus,
                    'memory': service.config.memory
                },
                'health_status': service.health_status.value,
                'availability_zone': service.availability_zone,
                'launch_time': service.launch_time.isoformat(),
                'activity_level': service.activity_level
            }
            services.append(service_data)

        return jsonify(services)

    @app.route('/api/communications')
    def api_communications():
        """Get active communications."""
        communications = []
        for comm in app.architecture.get_communications():
            comm_data = {
                'id': comm.id,
                'from_service_id': comm.from_service_id,
                'to_service_id': comm.to_service_id,
                'type': comm.communication_type.value,
                'status': comm.status.value,
                'progress': comm.progress,
                'start_time': comm.start_time.isoformat(),
                'data_size': comm.data_size,
                'throughput': comm.throughput,
                'latency': comm.latency,
                'protocol': comm.protocol
            }
            communications.append(comm_data)

        return jsonify(communications)

    @app.route('/api/metrics')
    def api_metrics():
        """Get current system metrics."""
        return jsonify(app.architecture.get_current_metrics())

    @app.route('/api/training/start', methods=['POST'])
    def api_training_start():
        """Start training session."""
        data = request.get_json() or {}

        model_name = data.get('model_name', 'LLaMA-70B')
        model_size = data.get('model_size', '70B')
        batch_size = data.get('batch_size', 32)
        learning_rate = data.get('learning_rate', 0.0001)
        total_epochs = data.get('total_epochs', 100)

        session = app.architecture.start_training(
            model_name=model_name,
            model_size=model_size,
            batch_size=batch_size,
            learning_rate=learning_rate,
            total_epochs=total_epochs
        )

        return jsonify({
            'success': True,
            'session_id': session.id,
            'message': f'Training started for {model_name}'
        })

    @app.route('/api/training/stop', methods=['POST'])
    def api_training_stop():
        """Stop training session."""
        app.architecture.stop_training()
        return jsonify({
            'success': True,
            'message': 'Training stopped'
        })

    @app.route('/api/training/pause', methods=['POST'])
    def api_training_pause():
        """Pause training session."""
        app.architecture.pause_training()
        return jsonify({
            'success': True,
            'message': 'Training paused'
        })

    @app.route('/api/training/resume', methods=['POST'])
    def api_training_resume():
        """Resume training session."""
        app.architecture.resume_training()
        return jsonify({
            'success': True,
            'message': 'Training resumed'
        })

    @app.route('/api/training/checkpoint', methods=['POST'])
    def api_training_checkpoint():
        """Save training checkpoint."""
        checkpoint_id = app.architecture.save_checkpoint()
        if checkpoint_id:
            return jsonify({
                'success': True,
                'checkpoint_id': checkpoint_id,
                'message': 'Checkpoint saved successfully'
            })
        else:
            return jsonify({
                'success': False,
                'message': 'No active training session'
            }), 400

    @app.route('/api/training/status')
    def api_training_status():
        """Get training status."""
        status = app.architecture.get_training_status()
        if status:
            return jsonify(status)
        else:
            return jsonify({'active': False})

    @app.route('/api/scale/out', methods=['POST'])
    def api_scale_out():
        """Scale out infrastructure."""
        data = request.get_json() or {}
        instance_type = data.get('instance_type', 'p4d.24xlarge')
        count = data.get('count', 1)

        new_services = app.architecture.scale_out(instance_type=instance_type, count=count)

        return jsonify({
            'success': True,
            'message': f'Added {len(new_services)} instances',
            'new_services': [s.id for s in new_services]
        })

    @app.route('/api/scale/in', methods=['POST'])
    def api_scale_in():
        """Scale in infrastructure."""
        data = request.get_json() or {}
        count = data.get('count', 1)

        removed_count = app.architecture.scale_in(count=count)

        return jsonify({
            'success': True,
            'message': f'Removed {removed_count} instances',
            'removed_count': removed_count
        })

    @app.route('/api/simulate/failure', methods=['POST'])
    def api_simulate_failure():
        """Simulate service failure."""
        data = request.get_json() or {}
        service_id = data.get('service_id')

        success = app.architecture.simulate_failure(service_id)

        if success:
            return jsonify({
                'success': True,
                'message': 'Service failure simulated'
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Failed to simulate failure'
            }), 400

    @app.route('/api/health-check', methods=['POST'])
    def api_health_check():
        """Run system health check."""
        health_summary = app.architecture.run_health_check()
        return jsonify(health_summary)

    @app.route('/api/data/ingest', methods=['POST'])
    def api_data_ingest():
        """Trigger data ingestion."""
        app.architecture.trigger_data_ingestion()
        return jsonify({
            'success': True,
            'message': 'Data ingestion triggered'
        })

    @app.route('/api/data/preprocess', methods=['POST'])
    def api_data_preprocess():
        """Trigger data preprocessing."""
        app.architecture.trigger_preprocessing()
        return jsonify({
            'success': True,
            'message': 'Data preprocessing triggered'
        })

    @app.route('/api/model/validate', methods=['POST'])
    def api_model_validate():
        """Trigger model validation."""
        results = app.architecture.trigger_model_validation()
        if results:
            return jsonify({
                'success': True,
                'message': 'Model validation completed',
                'results': results
            })
        else:
            return jsonify({
                'success': False,
                'message': 'No serving services available for validation'
            }), 400

    @app.route('/api/communications/add', methods=['POST'])
    def api_add_communication():
        """Add a new communication."""
        data = request.get_json() or {}

        comm_type = data.get('type', 'data-transfer')
        from_service_id = data.get('from_service_id')
        to_service_id = data.get('to_service_id')

        app.architecture.add_communication(
            comm_type=comm_type,
            from_service_id=from_service_id,
            to_service_id=to_service_id
        )

        return jsonify({
            'success': True,
            'message': f'Communication {comm_type} added'
        })

    @app.route('/api/communications/clear', methods=['POST'])
    def api_clear_communications():
        """Clear completed communications."""
        app.architecture.clear_communications()
        return jsonify({
            'success': True,
            'message': 'Communications cleared'
        })

    @app.route('/api/architecture/reset', methods=['POST'])
    def api_reset_architecture():
        """Reset the entire architecture."""
        app.architecture.reset_architecture()
        return jsonify({
            'success': True,
            'message': 'Architecture reset completed'
        })

    @app.route('/api/config/export')
    def api_export_config():
        """Export current configuration."""
        filename = f"aws_genai_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        app.architecture.export_configuration(filename)

        return jsonify({
            'success': True,
            'filename': filename,
            'message': 'Configuration exported'
        })

    @app.route('/api/statistics/communications')
    def api_communication_statistics():
        """Get communication statistics."""
        return jsonify(app.architecture.get_communication_statistics())

    @app.route('/api/statistics/services')
    def api_service_statistics():
        """Get service health summary."""
        return jsonify(app.architecture.get_service_health_summary())

    # SocketIO event handlers
    @socketio.on('connect')
    def handle_connect():
        """Handle client connection."""
        print('Client connected')
        start_real_time_updates()
        emit('status', {'msg': 'Connected to AWS GenAI Architecture'})

    @socketio.on('disconnect')
    def handle_disconnect():
        """Handle client disconnection."""
        print('Client disconnected')

    @socketio.on('request_full_update')
    def handle_full_update():
        """Send full system state to client."""
        # Send services
        services_data = []
        for service in app.architecture.get_services():
            service_data = {
                'id': service.id,
                'name': service.name,
                'type': service.service_type.value,
                'position': {'x': service.position.x, 'y': service.position.y},
                'radius': service.radius,
                'utilization': service.metrics.utilization,
                'health_status': service.health_status.value,
                'cost_per_hour': service.metrics.cost_per_hour,
                'activity_level': service.activity_level
            }
            services_data.append(service_data)

        emit('services_update', services_data)

        # Send metrics
        metrics = app.architecture.get_current_metrics()
        emit('metrics_update', metrics)

        # Send training status
        training_status = app.architecture.get_training_status()
        if training_status:
            emit('training_update', training_status)

    # Static file serving
    @app.route('/static/<path:filename>')
    def serve_static(filename):
        """Serve static files."""
        return send_from_directory('static', filename)

    # Start architecture if not already running
    if not app.architecture._running:
        app.architecture.start()

    # Attach SocketIO to app for external access
    app.socketio = socketio

    return app