# Cascade - Production-Ready, High-Performance, Asynchronous VAD Library

[中文](./README_zh.md)

[![Python Version](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Development Status](https://img.shields.io/badge/status-beta-orange.svg)](https://github.com/xucailiang/cascade)
[![Powered by Silero VAD](https://img.shields.io/badge/powered%20by-Silero%20VAD-orange.svg)](https://github.com/snakers4/silero-vad)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)](https://github.com/xucailiang/cascade)
[![Code Coverage](https://img.shields.io/badge/coverage-90%25-brightgreen.svg)](https://github.com/xucailiang/cascade)

**Cascade** is a **production-ready**, **high-performance**, and **low-latency** audio stream processing library designed for Voice Activity Detection (VAD). Built upon the excellent [Silero VAD](https://github.com/snakers4/silero-vad) model, Cascade significantly reduces VAD processing latency while maintaining high accuracy through its **1:1:1 binding architecture** and **asynchronous streaming technology**.

## 📊 Performance Benchmarks

Based on our latest streaming VAD performance tests with different chunk sizes:

### Streaming Performance by Chunk Size

| Chunk Size (bytes) | Processing Time (ms) | Throughput (chunks/sec) | Total Test Time (s) | Speech Segments |
|-------------------|---------------------|------------------------|-------------------|-----------------|
| **1024**          | **0.66**            | **92.2**               | 3.15              | 2               |
| **4096**          | 1.66                | 82.4                   | 0.89              | 2               |
| **8192**          | 2.95                | 72.7                   | 0.51              | 2               |

### Key Performance Metrics

| Metric                  | Value         | Description                             |
|-------------------------|---------------|-----------------------------------------|
| **Best Processing Speed** | 0.66ms/chunk | Optimal performance with 1024-byte chunks |
| **Peak Throughput**     | 92.2 chunks/sec | Maximum processing throughput          |
| **Success Rate**        | 100%          | Processing success rate across all tests |
| **Accuracy**            | High          | Guaranteed by the Silero VAD model      |
| **Architecture**        | 1:1:1:1       | Independent model per processor instance |

### Performance Characteristics

- **Excellent performance across chunk sizes**: High throughput and low latency with various chunk sizes
- **Real-time capability**: Sub-millisecond processing enables real-time applications
- **Scalability**: Linear performance scaling with independent processor instances


## ✨ Core Features

### 🚀 High-Performance Engineering

- **Lock-Free Design**: The 1:1:1 binding architecture eliminates lock contention, boosting performance.
- **Frame-Aligned Buffer**: A highly efficient buffer optimized for 512-sample frames.
- **Asynchronous Streaming**: Non-blocking audio stream processing based on `asyncio`.
- **Memory Optimization**: Zero-copy design, object pooling, and cache alignment.
- **Concurrency Optimization**: Dedicated threads, asynchronous queues, and batch processing.

### 🔧 Robust Software Engineering

- **Modular Design**: A component architecture with high cohesion and low coupling.
- **Interface Abstraction**: Dependency inversion through interface-based design.
- **Type System**: Data validation and type checking using Pydantic.
- **Comprehensive Testing**: Unit, integration, and performance tests.
- **Code Standards**: Adherence to PEP 8 style guidelines.

### 🛡️ Production-Ready Reliability

- **Error Handling**: Robust error handling and recovery mechanisms.
- **Resource Management**: Automatic cleanup and graceful shutdown.
- **Monitoring Metrics**: Real-time performance monitoring and statistics.
- **Scalability**: Horizontal scaling by increasing the number of instances.
- **Stability Assurance**: Handles boundary conditions and exceptional cases gracefully.

## 🏗️ Architecture

Cascade employs a **1:1:1:1 independent architecture** to ensure optimal performance and thread safety.

```mermaid
graph TD
    Client --> StreamProcessor
    
    subgraph "1:1:1:1 Independent Architecture"
        StreamProcessor --> |per connection| IndependentProcessor[Independent Processor Instance]
        IndependentProcessor --> |independent loading| VADModel[Silero VAD Model]
        IndependentProcessor --> |independent management| VADIterator[VAD Iterator]
        IndependentProcessor --> |independent buffering| FrameBuffer[Frame-Aligned Buffer]
        IndependentProcessor --> |independent state| StateMachine[State Machine]
    end
    
    subgraph "Asynchronous Processing Flow"
        VADModel --> |asyncio.to_thread| VADInference[VAD Inference]
        VADInference --> StateMachine
        StateMachine --> |None| SingleFrame[Single Frame Output]
        StateMachine --> |start| Collecting[Start Collecting]
        StateMachine --> |end| SpeechSegment[Speech Segment Output]
    end
```

## 🚀 Quick Start

### Installation

```
pip install cascade-vad
```
OR

```bash
# Using uv is recommended
uv venv -p 3.12

source .venv/bin/activate

# Install from PyPI (recommended)
pip install cascade-vad

# Or install from source
git clone https://github.com/xucailiang/cascade.git
cd cascade
pip install -e .
```

### Basic Usage

```python
import cascade
import asyncio

async def basic_example():
    """A basic usage example."""
    
    # Method 1: Simple file processing
    async for result in cascade.process_audio_file("audio.wav"):
        if result.result_type == "segment":
            segment = result.segment
            print(f"🎤 Speech Segment: {segment.start_timestamp_ms:.0f}ms - {segment.end_timestamp_ms:.0f}ms")
        else:
            frame = result.frame
            print(f"🔇 Single Frame: {frame.timestamp_ms:.0f}ms")
    
    # Method 2: Stream processing
    async with cascade.StreamProcessor() as processor:
        async for result in processor.process_stream(audio_stream):
            if result.result_type == "segment":
                segment = result.segment
                print(f"🎤 Speech Segment: {segment.start_timestamp_ms:.0f}ms - {segment.end_timestamp_ms:.0f}ms")
            else:
                frame = result.frame
                print(f"🔇 Single Frame: {frame.timestamp_ms:.0f}ms")

asyncio.run(basic_example())
```

### Advanced Configuration

```python
from cascade.stream import StreamProcessor, create_default_config

async def advanced_example():
    """An advanced configuration example."""
    
    # Custom configuration
    config = create_default_config(
        vad_threshold=0.7,          # Higher detection threshold
        max_instances=3,            # Max 3 concurrent instances
        buffer_size_frames=128      # Larger buffer
    )
    
    # Use the custom config
    async with StreamProcessor(config) as processor:
        # Process audio stream
        async for result in processor.process_stream(audio_stream, "my-stream"):
            # Process results...
            pass
        
        # Get performance statistics
        stats = processor.get_stats()
        print(f"Processing Stats: {stats.summary()}")
        print(f"Throughput: {stats.throughput_chunks_per_second:.1f} chunks/sec")

asyncio.run(advanced_example())
```

## 🧪 Testing

```bash
# Run basic integration tests
python tests/test_simple_vad.py -v

# Run simulated audio stream tests
python tests/test_stream_vad.py -v

# Run performance benchmark tests
python tests/benchmark_performance.py
```

Test Coverage:
- ✅ Basic API Usage
- ✅ Stream Processing
- ✅ File Processing
- ✅ Real Audio VAD
- ✅ Automatic Speech Segment Saving
- ✅ 1:1:1:1 Architecture Validation
- ✅ Performance Benchmarks
- ✅ FrameAlignedBuffer Tests

## 🌐 Web Demo

We provide a complete WebSocket-based web demonstration that showcases Cascade's real-time VAD capabilities with multiple client support.

![Web Demo Screenshot](web_demo/test_image.png)

### Features

- **Real-time Audio Processing**: Capture audio from browser microphone and process with VAD
- **Live VAD Visualization**: Real-time display of VAD detection results
- **Speech Segment Management**: Display detected speech segments with playback support
- **Dynamic VAD Configuration**: Adjust VAD parameters in real-time
- **Multi-client Support**: Independent Cascade instances for each WebSocket connection

### Quick Start

```bash
# Start backend server
cd web_demo
python server.py

# Start frontend (in another terminal)
cd web_demo/frontend
pnpm install && pnpm dev
```

For detailed setup instructions, see [Web Demo Documentation](web_demo/README.md).

## 🔧 Production Deployment

### Best Practices

1.  **Resource Allocation**
    -   Each instance uses approximately 50MB of memory.
    -   Recommended: 2-3 instances per CPU core.
    -   Monitor memory usage to prevent Out-of-Memory (OOM) errors.

2.  **Performance Tuning**
    -   Adjust `max_instances` to match server CPU cores.
    -   Increase `buffer_size_frames` for higher throughput.
    -   Tune `vad_threshold` to balance accuracy and sensitivity.

3.  **Error Handling**
    -   Implement retry mechanisms for transient errors.
    -   Use health checks to monitor service status.
    -   Log detailed information for troubleshooting.

### Monitoring Metrics

```python
# Get performance monitoring metrics
stats = processor.get_stats()

# Key monitoring metrics
print(f"Active Instances: {stats.active_instances}/{stats.total_instances}")
print(f"Average Processing Time: {stats.average_processing_time_ms}ms")
print(f"Success Rate: {stats.success_rate:.2%}")
print(f"Memory Usage: {stats.memory_usage_mb:.1f}MB")
```

## 🔧 Requirements

### Core Dependencies

-   **Python**: 3.12 (recommended)
-   **pydantic**: 2.4.0+ (Data validation)
-   **numpy**: 1.24.0+ (Numerical computation)
-   **scipy**: 1.11.0+ (Signal processing)
-   **silero-vad**: 5.1.2+ (VAD model)
-   **onnxruntime**: 1.22.1+ (ONNX inference)
-   **torchaudio**: 2.7.1+ (Audio processing)

### Development Dependencies

-   **pytest**: Testing framework
-   **black**: Code formatter
-   **ruff**: Linter
-   **mypy**: Type checker
-   **pre-commit**: Git hooks

## 🤝 Contribution Guide

We welcome community contributions! Please follow these steps:

1.  **Fork the project** and create a feature branch.
2.  **Install development dependencies**: `pip install -e .[dev]`
3.  **Run tests**: `pytest`
4.  **Lint your code**: `ruff check . && black --check .`
5.  **Type check**: `mypy cascade`
6.  **Submit a Pull Request** with a clear description of your changes.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

-   **Silero Team**: For their excellent VAD model.
-   **PyTorch Team**: For the deep learning framework.
-   **Pydantic Team**: For the type validation system.
-   **Python Community**: For the rich ecosystem.

## 📞 Contact

-   **Author**: Xucailiang
-   **Email**: xucailiang.ai@gmail.com
-   **Project Homepage**: https://github.com/xucailiang/cascade
-   **Issue Tracker**: https://github.com/xucailiang/cascade/issues
-   **Documentation**: https://cascade-vad.readthedocs.io/

![img_v3_02ra_9845ba4a-a36d-4387-9d01-2b392c94d6cg](https://github.com/user-attachments/assets/1a21b891-d5fc-4319-a70d-f4c95ffaa7dd)

---

**⭐ If you find this project helpful, please give it a star!**
