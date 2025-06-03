# AWS GenAI Architecture Visualizer

An interactive Python package for visualizing AWS GenAI LLM training architectures with real-time communication monitoring and detailed console logging.

## Features

- 🚀 Interactive web-based visualization of AWS GenAI training infrastructure
- 📊 Real-time metrics monitoring (GPU utilization, throughput, costs)
- 🔄 Detailed service-to-service communication tracking
- 💰 Cost optimization with spot instance management
- 🔍 Comprehensive console logging for all operations
- 🎛️ Interactive controls for training operations
- 📈 Scalable architecture simulation
- ⚡ WebSocket-based real-time updates

## Installation

### From TestPyPI

```bash
pip install -i https://test.pypi.org/simple/ aws-genai-architecture-viz
```

### From Source

```bash
git clone https://github.com/yourusername/aws-genai-architecture-viz.git
cd aws-genai-architecture-viz
pip install -e .
```

## Quick Start

### Command Line Interface

```bash
# Start the visualization server
aws-genai-viz --port 8080 --host 0.0.0.0

# Start with custom configuration
aws-genai-viz --config my_config.json --debug
```

### Python API

```python
from aws_genai_architecture import AWSGenAIArchitecture, ConsoleLogger

# Initialize the architecture
architecture = AWSGenAIArchitecture()

# Start training simulation
architecture.start_training()

# Monitor communications
architecture.monitor_communications()

# Scale infrastructure
architecture.scale_out(instance_type="p4d.24xlarge", count=2)
```

### Web Interface

```python
from aws_genai_architecture.web import create_app

app = create_app()
app.run(debug=True, host='0.0.0.0', port=8080)
```

## Architecture Components

### AWS Services Simulated

- **EC2 Training Instances**: Distributed training nodes with GPU support
- **SageMaker Training Jobs**: Managed ML training service
- **S3 Data Lake**: Scalable data storage and management
- **Bedrock Foundation Models**: Pre-trained foundation models
- **Lambda Functions**: Serverless data processing pipeline
- **ECS Model Serving**: Containerized model inference
- **CloudWatch Monitoring**: Comprehensive metrics and logging

### Communication Types

- **Data Transfer**: Large dataset movement between services
- **Model Synchronization**: Parameter updates across training nodes
- **Control Signals**: Training orchestration and coordination
- **Checkpoints**: Model state persistence and recovery
- **Monitoring**: Health checks and metrics collection
- **Inference**: Real-time model predictions
- **Preprocessing**: Data transformation and preparation

## Detailed Usage

### Console Monitoring

```python
from aws_genai_architecture import ConsoleLogger, CommunicationMonitor

# Enable detailed console logging
logger = ConsoleLogger(level="DEBUG", show_metrics=True)

# Monitor specific communication types
monitor = CommunicationMonitor()
monitor.track_communication_type("data-transfer")
monitor.track_service_pair("S3_DATALAKE", "SAGEMAKER_TRAINING")

# Start monitoring
monitor.start_monitoring()
```

### Custom Configuration

```python
# config.json
{
    "services": {
        "ec2_training": {
            "instance_type": "p4d.24xlarge",
            "count": 4,
            "spot_instances": true,
            "cost_per_hour": 434
        },
        "sagemaker": {
            "instance_type": "ml.p4d.24xlarge",
            "managed_spot": false
        }
    },
    "communication": {
        "auto_generate": true,
        "frequency": 1.8,
        "types": ["data-transfer", "model-sync", "monitoring"]
    },
    "monitoring": {
        "console_output": true,
        "metrics_interval": 2.0,
        "detailed_logging": true
    }
}
```

### Training Operations

```python
# Start distributed training
architecture.start_training(
    model_size="70B",
    batch_size=32,
    learning_rate=0.0001
)

# Monitor training progress
for epoch in architecture.training_progress():
    print(f"Epoch {epoch.number}: Loss={epoch.loss:.4f}, GPU={epoch.gpu_util}%")
    
    # Automatic checkpointing
    if epoch.number % 5 == 0:
        architecture.save_checkpoint(f"checkpoint_epoch_{epoch.number}")

# Scale based on performance
if architecture.metrics.gpu_utilization < 70:
    architecture.scale_out(instance_type="p4d.24xlarge", count=2)
```

## Console Output Examples

```
🚀 AWS GenAI Architecture Started
📊 Initializing 7 AWS services...

[2024-06-02 10:30:15] 🔄 DATA-TRANSFER: S3_DATALAKE → SAGEMAKER_TRAINING
├─ Data Size: 2.3 GB
├─ Throughput: 2.1 GB/s
├─ Latency: 12ms
├─ Protocol: AWS S3 Transfer Acceleration
└─ Status: ✅ Transfer Complete (1.1s)

[2024-06-02 10:30:16] 🧠 MODEL-SYNC: SAGEMAKER_TRAINING → EC2_TRAINING_NODES
├─ Parameters: 70B (280 GB)
├─ Compression: 4:1 (FP16)
├─ Network: 25 Gbps Ethernet
├─ Sharding: 8-way tensor parallel
└─ Status: 🔄 Synchronizing... (45% complete)

[2024-06-02 10:30:18] 💾 CHECKPOINT: EC2_TRAINING_NODES → S3_DATALAKE
├─ Checkpoint ID: ckpt_epoch_15_step_2840
├─ Model State: 280 GB
├─ Optimizer State: 560 GB
├─ Compression: Enabled (2.1:1 ratio)
├─ S3 Location: s3://model-checkpoints/llama-70b/
└─ Status: ✅ Saved successfully (45.2s)

📈 Real-time Metrics:
┌─────────────────────┬─────────────┬─────────────┐
│ Service             │ Utilization │ Cost/Hour   │
├─────────────────────┼─────────────┼─────────────┤
│ SageMaker Training  │ 92%         │ $458        │
│ EC2 Training Nodes  │ 89%         │ $434        │
│ S3 Data Lake        │ 45%         │ $150        │
│ Lambda Pipeline     │ 65%         │ $45         │
└─────────────────────┴─────────────┴─────────────┘

⚡ Active Communications: 12
💰 Total Cost: $892/hour
🎯 Training Progress: 67% (Epoch 15/23)
📊 Model Loss: 2.341 (↓ 0.008 from last checkpoint)
```

## API Reference

### Core Classes

#### `AWSGenAIArchitecture`
Main architecture coordinator

```python
class AWSGenAIArchitecture:
    def __init__(self, config: Optional[Dict] = None)
    def start_training(self, **kwargs) -> TrainingSession
    def stop_training(self) -> None
    def scale_out(self, instance_type: str, count: int) -> List[Service]
    def scale_in(self, count: int) -> None
    def add_communication(self, comm_type: str, from_service: str, to_service: str)
    def get_metrics(self) -> MetricsSnapshot
```

#### `Service`
Individual AWS service representation

```python
class Service:
    id: str
    name: str
    aws_service: str
    instance_type: str
    utilization: float
    cost_per_hour: float
    health_status: str
    reliability_score: float
```

#### `Communication`
Service-to-service communication

```python
class Communication:
    id: str
    from_service: Service
    to_service: Service
    comm_type: str
    data_size: str
    throughput: str
    latency: float
    status: str
    progress: float
```

### Web API Endpoints

- `GET /` - Main visualization interface
- `GET /api/services` - List all services
- `GET /api/communications` - List active communications
- `GET /api/metrics` - Current system metrics
- `POST /api/training/start` - Start training session
- `POST /api/training/stop` - Stop training session
- `POST /api/scale/out` - Scale out infrastructure
- `POST /api/scale/in` - Scale in infrastructure
- `WebSocket /ws` - Real-time updates

## Development

### Setting up Development Environment

```bash
git clone https://github.com/yourusername/aws-genai-architecture-viz.git
cd aws-genai-architecture-viz
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e ".[dev]"
```

### Running Tests

```bash
pytest tests/ -v --cov=aws_genai_architecture
```

### Code Formatting

```bash
black aws_genai_architecture/
flake8 aws_genai_architecture/
mypy aws_genai_architecture/
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- AWS for providing the inspiration for service architecture
- The machine learning community for training optimization insights
- Contributors and users providing feedback and improvements

## Support

- 📖 Documentation: https://aws-genai-architecture-viz.readthedocs.io/
- 🐛 Bug Reports: https://github.com/yourusername/aws-genai-architecture-viz/issues
- 💬 Discussions: https://github.com/yourusername/aws-genai-architecture-viz/discussions
- 📧 Email: your.email@example.com