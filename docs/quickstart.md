
Quick Start Guide
=================

This guide will get you up and running with the AWS GenAI Architecture Visualizer in minutes.

Basic Usage
-----------

Web Interface (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Start the web interface:

.. code-block:: bash

   aws-genai-viz --mode web --port 8080

Then open http://localhost:8080 in your browser.

Console Mode
~~~~~~~~~~~

For terminal-only usage:

.. code-block:: bash

   aws-genai-viz --mode console

Python API
~~~~~~~~~~

.. code-block:: python

   from aws_genai_architecture import AWSGenAIArchitecture

   # Initialize with default configuration
   architecture = AWSGenAIArchitecture()
   
   # Start the architecture
   architecture.start()
   
   # Start training simulation
   session = architecture.start_training(
       model_name="LLaMA-70B",
       model_size="70B",
       batch_size=32,
       total_epochs=100
   )
   
   # Monitor training progress
   while session.status == "running":
       metrics = architecture.get_current_metrics()
       print(f"Progress: {metrics['training_progress']:.1f}%")
       time.sleep(5)
   
   # Clean shutdown
   architecture.stop()

First Steps Tutorial
-------------------

1. **Start the Visualizer**

   .. code-block:: bash

      aws-genai-viz

2. **Open Web Interface**

   Navigate to http://localhost:8080

3. **Start Training**

   Click "Start Training" in the control panel

4. **Observe Communications**

   Watch real-time service communications in the visualization

5. **Scale Infrastructure**

   Use "Scale Out" to add more training instances

6. **Monitor Metrics**

   Check the metrics panel for cost and performance data

Configuration
-------------

Create a configuration file:

.. code-block:: json

   {
     "services": {
       "ec2_training": {
         "instance_type": "p4d.24xlarge",
         "count": 4,
         "spot_instances": true
       }
     },
     "communication": {
       "auto_generate": true,
       "frequency": 1.5
     },
     "monitoring": {
       "log_level": "INFO",
       "detailed_logging": true
     }
   }

Use with:

.. code-block:: bash

   aws-genai-viz --config myconfig.json

Next Steps
----------

* Read the :doc:`configuration` guide for advanced setup
* Explore the :doc:`api_reference` for programmatic usage
* Check out :doc:`examples` for common use cases
"""

# Build and Deployment Instructions
"""
# Building and Publishing to TestPyPI

## Prerequisites

1. Install build tools:
```bash
pip install build twine
```

2. Set up TestPyPI account and API token at https://test.pypi.org/

## Build Process

1. **Clean previous builds:**
```bash
rm -rf dist/ build/ *.egg-info/
```

2. **Build the package:**
```bash
python -m build
```

3. **Check the package:**
```bash
twine check dist/*
```

4. **Upload to TestPyPI:**
```bash
twine upload --repository testpypi dist/*
```

5. **Test installation:**
```bash
pip install -i https://test.pypi.org/simple/ aws-genai-architecture-viz
```

## Directory Structure


```bash
aws-genai-architecture-viz/
├── aws_genai_architecture/
│   ├── __init__.py
│   ├── __main__.py
│   ├── _version.py
│   ├── cli.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── architecture.py
│   │   ├── communications.py
│   │   ├── metrics.py
│   │   ├── models.py
│   │   └── services.py
│   ├── monitoring/
│   │   ├── __init__.py
│   │   ├── console_logger.py
│   │   └── communication_monitor.py
│   └── web/
│       ├── __init__.py
│       ├── app.py
│       ├── templates/
│       │   └── index.html
│       └── static/
│           └── style.css
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_core/
│   ├── test_monitoring/
│   └── test_web/
├── docs/
│   ├── conf.py
│   ├── index.rst
│   ├── installation.rst
│   └── quickstart.rst
├── setup.py
├── pyproject.toml
├── MANIFEST.in
├── README.md
├── LICENSE
├── CHANGELOG.md
└── requirements.txt
```

## Key Features Summary

### Core Functionality
- **7 AWS Service Types**: EC2, SageMaker, S3, Bedrock, Lambda, ECS, CloudWatch
- **7 Communication Types**: Data transfer, model sync, control signals, checkpoints, monitoring, inference, preprocessing
- **Real-time Visualization**: HTML5 Canvas with WebSocket updates
- **Training Simulation**: Complete ML training lifecycle management
- **Cost Tracking**: Real-time cost analysis and optimization
- **Auto-scaling**: Dynamic infrastructure scaling with spot instances

### Enhanced Console Logging
- **Rich Terminal Output**: Colored, formatted console output with tables
- **Detailed Communication Tracking**: Every service-to-service communication logged
- **Real-time Metrics**: Live system performance monitoring
- **Export Capabilities**: JSON and plain text log export
- **Filtering Options**: Configurable log levels and service filtering

### Web Interface
- **Modern UI**: Responsive web interface with real-time updates
- **Interactive Controls**: Full training and infrastructure management
- **Live Visualization**: Real-time service and communication visualization
- **RESTful API**: Complete HTTP API for integration
- **WebSocket Support**: Real-time bidirectional communication

### CLI Tool
- **Multiple Modes**: Web server and console-only modes
- **Configuration Management**: JSON-based configuration system
- **Debug Support**: Comprehensive debugging and logging options
- **Cross-platform**: Windows, macOS, and Linux support

This package provides a comprehensive, production-ready tool for visualizing and understanding AWS GenAI training architectures with unprecedented detail in service communication monitoring.
