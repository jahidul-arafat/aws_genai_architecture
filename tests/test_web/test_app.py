import pytest
import json
import os
from unittest.mock import patch
from aws_genai_architecture.web.app import create_app
from aws_genai_architecture import AWSGenAIArchitecture

class TestWebApp:
    """Test cases for web application."""

    @pytest.fixture
    def app(self):
        """Create test app."""
        # Set testing environment
        os.environ['FLASK_ENV'] = 'testing'

        architecture = AWSGenAIArchitecture(config={
            "services": {"auto_initialize": False},
            "communication": {"auto_generate": False}
        })

        app = create_app(architecture)
        app.config['TESTING'] = True
        app.config['WTF_CSRF_ENABLED'] = False

        return app

    @pytest.fixture
    def client(self, app):
        """Create test client."""
        # Patch the werkzeug version issue
        with patch.dict('sys.modules', {'werkzeug': type('MockWerkzeug', (), {'__version__': '2.3.0'})()}):
            return app.test_client()

    def test_index_route(self, client):
        """Test main index route."""
        response = client.get('/')
        assert response.status_code == 200
        assert b'AWS GenAI Architecture Visualizer' in response.data

    def test_api_status(self, client):
        """Test API status endpoint."""
        response = client.get('/api/status')
        assert response.status_code == 200

        data = json.loads(response.data)
        assert data['status'] == 'online'
        assert 'timestamp' in data

    def test_api_services(self, client):
        """Test services API endpoint."""
        response = client.get('/api/services')
        assert response.status_code == 200

        data = json.loads(response.data)
        assert isinstance(data, list)

    def test_api_metrics(self, client):
        """Test metrics API endpoint."""
        response = client.get('/api/metrics')
        assert response.status_code == 200

        data = json.loads(response.data)
        assert 'active_services' in data
        assert 'reliability_score' in data

    def test_training_endpoints(self, client):
        """Test training-related endpoints."""
        # Start training
        response = client.post('/api/training/start',
                               json={'model_name': 'Test Model'})
        assert response.status_code == 200

        data = json.loads(response.data)
        assert data['success']

        # Get training status
        response = client.get('/api/training/status')
        assert response.status_code == 200

        # Stop training
        response = client.post('/api/training/stop')
        assert response.status_code == 200

    def test_scaling_endpoints(self, client):
        """Test scaling endpoints."""
        # Scale out
        response = client.post('/api/scale/out',
                               json={'count': 1, 'instance_type': 'test.instance'})
        assert response.status_code == 200

        # Scale in
        response = client.post('/api/scale/in',
                               json={'count': 1})
        assert response.status_code == 200