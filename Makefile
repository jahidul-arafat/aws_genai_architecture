# 🚨 CORRECT FILE STRUCTURE

## 📁 Keep conftest.py CLEAN - Only pytest fixtures:

# tests/conftest.py (KEEP THIS CLEAN)
"""Pytest configuration and fixtures."""

import pytest
import sys
import os
from unittest.mock import Mock

# Add the package to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from aws_genai_architecture import AWSGenAIArchitecture
from aws_genai_architecture.monitoring.console_logger import ConsoleLogger

@pytest.fixture
def mock_logger():
    """Create a mock logger for testing."""
    logger = Mock(spec=ConsoleLogger)
    logger.log_info = Mock()
    logger.log_error = Mock()
    logger.log_warning = Mock()
    logger.log_debug = Mock()
    logger.start = Mock()
    logger.stop = Mock()
    return logger

@pytest.fixture
def test_config():
    """Create test configuration."""
    return {
        "services": {"auto_initialize": False},
        "communication": {"auto_generate": False},
        "monitoring": {"console_output": False, "log_level": "DEBUG"}
    }

@pytest.fixture
def architecture(mock_logger, test_config):
    """Create a test architecture instance."""
    return AWSGenAIArchitecture(config=test_config, logger=mock_logger)

@pytest.fixture
def running_architecture(architecture):
    """Create a running architecture instance."""
    architecture.start()
    yield architecture
    architecture.stop()

## 📁 CREATE SEPARATE FILES for other configurations:

# 1. CREATE: Makefile (in project root)
cat > Makefile << 'EOF'
# Makefile for AWS GenAI Architecture Tests

.PHONY: test test-unit test-integration test-coverage test-html clean

# Install test dependencies
install-test:
	pip install pytest pytest-cov pytest-mock pytest-xdist pytest-sugar flask-testing

# Run all tests
test:
	pytest tests/ -v

# Run unit tests only
test-unit:
	pytest tests/test_core/ tests/test_monitoring/ -v

# Run integration tests
test-integration:
	pytest tests/test_web/ tests/test_cli/ -v

# Run tests with coverage
test-coverage:
	pytest tests/ -v --cov=aws_genai_architecture --cov-report=term-missing

# Generate HTML coverage report
test-html:
	pytest tests/ --cov=aws_genai_architecture --cov-report=html
	@echo "Coverage report generated in htmlcov/index.html"

# Run tests in parallel
test-parallel:
	pytest tests/ -n auto -v

# Quick smoke test
smoke-test:
	pytest tests/test_core/test_architecture.py::TestAWSGenAIArchitecture::test_initialization -v

# Clean up test artifacts
clean:
	rm -rf .pytest_cache
	rm -rf htmlcov/
	rm -f .coverage
	rm -f test-results.xml
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

help:
	@echo "Available targets:"
	@echo "  install-test    - Install test dependencies"
	@echo "  test           - Run all tests"
	@echo "  test-unit      - Run unit tests only"
	@echo "  test-integration - Run integration tests"
	@echo "  test-coverage  - Run tests with coverage"
	@echo "  test-html      - Generate HTML coverage report"
	@echo "  smoke-test     - Quick smoke test"
	@echo "  clean          - Clean up test artifacts"
EOF

# 2. CREATE: pytest.ini (in project root)
cat > pytest.ini << 'EOF'
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts =
    -v
    --strict-markers
    --tb=short
    --cov=aws_genai_architecture
    --cov-report=term-missing
    --cov-fail-under=80
markers =
    slow: marks tests as slow
    integration: marks tests as integration tests
    unit: marks tests as unit tests
    web: marks tests as web interface tests
    cli: marks tests as CLI tests
filterwarnings =
    ignore::DeprecationWarning
EOF

# 3. CREATE: tox.ini (in project root)
cat > tox.ini << 'EOF'
[tox]
envlist = py38,py39,py310,py311,flake8,mypy
isolated_build = True

[testenv]
deps =
    pytest
    pytest-cov
    pytest-mock
    flask-testing
commands = pytest tests/ -v --cov=aws_genai_architecture

[testenv:flake8]
deps = flake8
commands = flake8 aws_genai_architecture

[testenv:mypy]
deps = mypy
commands = mypy aws_genai_architecture
EOF

# 4. CREATE: .github/workflows/tests.yml (for CI/CD)
mkdir -p .github/workflows
cat > .github/workflows/tests.yml << 'EOF'
name: Tests

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.8", "3.9", "3.10", "3.11"]

    steps:
    - uses: actions/checkout@v4

    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[dev]"
        pip install pytest pytest-cov pytest-mock flask-testing

    - name: Test with pytest
      run: |
        pytest tests/ -v --cov=aws_genai_architecture --cov-report=xml

    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml