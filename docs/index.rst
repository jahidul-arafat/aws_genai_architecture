
AWS GenAI Architecture Visualizer Documentation
===============================================

Welcome to the AWS GenAI Architecture Visualizer documentation! This tool provides
an interactive visualization of AWS GenAI LLM training architectures with real-time
communication monitoring and detailed console logging.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   api_reference
   configuration
   examples
   contributing
   changelog

Features
--------

* **Interactive Visualization**: Real-time AWS service architecture visualization
* **Communication Monitoring**: Detailed service-to-service communication tracking
* **Training Simulation**: Complete ML training session management
* **Cost Analysis**: Real-time cost tracking and optimization
* **Web Interface**: Modern web-based dashboard with WebSocket updates
* **CLI Tool**: Comprehensive command-line interface
* **Modular Design**: Clean, extensible architecture

Quick Start
-----------

Installation
~~~~~~~~~~~~

.. code-block:: bash

   pip install -i https://test.pypi.org/simple/ aws-genai-architecture-viz

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   from aws_genai_architecture import AWSGenAIArchitecture

   # Initialize architecture
   architecture = AWSGenAIArchitecture()
   architecture.start()

   # Start training simulation
   session = architecture.start_training(
       model_name="LLaMA-70B",
       model_size="70B"
   )

   # Monitor progress
   metrics = architecture.get_current_metrics()
   print(f"Training progress: {metrics['training_progress']}%")

Command Line
~~~~~~~~~~~~

.. code-block:: bash

   # Start web interface
   aws-genai-viz --mode web --port 8080

   # Run in console mode
   aws-genai-viz --mode console --debug

Architecture Overview
--------------------

The AWS GenAI Architecture Visualizer simulates a complete AWS infrastructure
for large language model training, including:

**Core Services**

* **EC2 Training Instances**: GPU-enabled distributed training nodes
* **SageMaker Training Jobs**: Managed ML training with auto-scaling
* **S3 Data Lake**: Scalable data storage and management
* **Bedrock Foundation Models**: Pre-trained model integration
* **Lambda Data Pipeline**: Serverless data processing
* **ECS Model Serving**: Containerized model inference
* **CloudWatch Monitoring**: Comprehensive metrics and logging

**Communication Types**

* **Data Transfer**: Large dataset movement between services
* **Model Synchronization**: Parameter updates across training nodes
* **Control Signals**: Training orchestration and coordination
* **Checkpoints**: Model state persistence and recovery
* **Monitoring**: Health checks and metrics collection
* **Inference**: Real-time model predictions
* **Preprocessing**: Data transformation pipelines

Console Output Example
---------------------

.. code-block:: text

   🚀 AWS GenAI Architecture Started
   📊 Initializing 7 AWS services...

   [2024-06-02 10:30:15] 🔄 DATA-TRANSFER: S3_DATALAKE → SAGEMAKER_TRAINING
   ├─ Data Size: 2.3 GB
   ├─ Throughput: 2.1 GB/s
   ├─ Latency: 12ms
   ├─ Protocol: AWS S3 Transfer Acceleration
   └─ Status: ✅ Transfer Complete (1.1s)

   📈 Real-time Metrics:
   ┌─────────────────────┬─────────────┬─────────────┐
   │ Service             │ Utilization │ Cost/Hour   │
   ├─────────────────────┼─────────────┼─────────────┤
   │ SageMaker Training  │ 92%         │ $458        │
   │ EC2 Training Nodes  │ 89%         │ $434        │
   │ S3 Data Lake        │ 45%         │ $150        │
   └─────────────────────┴─────────────┴─────────────┘

API Reference
-------------

For detailed API documentation, see the :doc:`api_reference` section.

Contributing
------------

We welcome contributions! Please see our :doc:`contributing` guide for details.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
