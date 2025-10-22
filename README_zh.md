[English](./README.md)
# Cascade - 生产级高性能异步并行VAD处理库

[![Python Version](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Development Status](https://img.shields.io/badge/status-beta-orange.svg)](https://github.com/xucailiang/cascade)
[![Silero VAD](https://img.shields.io/badge/powered%20by-Silero%20VAD-orange.svg)](https://github.com/snakers4/silero-vad)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)](https://github.com/xucailiang/cascade)
[![Code Coverage](https://img.shields.io/badge/coverage-90%25-brightgreen.svg)](https://github.com/xucailiang/cascade)

Cascade是一个专为语音活动检测(VAD)设计的**生产级**、**高性能**、**低延迟**音频流处理库。基于优秀的[Silero VAD](https://github.com/snakers4/silero-vad)模型，通过**1:1:1绑定架构**和**异步流式处理技术**，显著降低VAD处理延迟，同时保证检测结果的准确性。

## 📊 性能指标

基于最新流式VAD性能测试的不同块大小测试结果：

### 不同块大小的流式处理性能

| 块大小(字节) | 处理时间(ms) | 吞吐量(块/秒) | 总测试时间(s) | 语音段数 |
|-------------|-------------|-------------|-------------|---------|
| **1024**   | **0.66**   | **92.2**   | 3.15        | 2       |
| **4096**   | 1.66        | 82.4        | 0.89        | 2       |
| **8192**   | 2.95        | 72.7        | 0.51        | 2       |

### 核心性能指标

| 指标 | 数值 | 说明 |
|------|------|------|
| **最佳处理速度** | 0.66ms/块 | 1024字节块大小下的最优性能 |
| **峰值吞吐量** | 92.2块/秒 | 最大处理吞吐量 |
| **成功率** | 100% | 所有测试的处理成功率 |
| **准确性** | 高 | 基于Silero VAD，保证检测准确性 |
| **架构** | 1:1:1:1 | 每个处理器实例独立模型 |

### 性能特性

- **各种块大小下的优异性能**: 在不同块大小下都能保持高吞吐量和低延迟
- **实时处理能力**: 亚毫秒级处理时间支持实时应用
- **可扩展性**: 独立处理器实例实现线性性能扩展

详细测试结果请参见：[架构重构完成报告](docs/architecture_refactoring_completion_report.md)。

## ✨ 核心特性

### 🚀 高性能特性

- **无锁设计**：1:1:1绑定架构消除锁竞争，提升性能
- **帧对齐缓冲区**：专为512样本帧优化的高效缓冲区
- **异步流式处理**：基于asyncio的非阻塞音频流处理
- **内存优化**：零拷贝设计、对象池复用、缓存对齐
- **并发优化**：专用线程、异步队列、批量处理

### 🔧 工程化特性

- **模块化设计**：高内聚低耦合的组件架构
- **接口抽象**：基于接口的依赖倒置设计
- **类型系统**：使用pydantic进行数据验证和类型检查
- **完整测试**：单元测试、集成测试、性能测试
- **代码规范**：符合PEP 8的代码风格

### 🛡️ 生产化特性

- **错误处理**：完善的错误处理和恢复机制
- **资源管理**：自动清理和优雅关闭
- **监控指标**：实时性能监控和统计
- **可扩展性**：通过实例数量水平扩展
- **稳定性保障**：边界条件处理和异常情况恢复

## 🏗️ 架构设计

Cascade采用**1:1:1:1独立架构**，确保最佳性能和线程安全：

```mermaid
graph TD
    Client[客户端] --> StreamProcessor[流式处理器]
    
    subgraph "1:1:1:1独立架构"
        StreamProcessor --> |每个连接| IndependentProcessor[独立处理器实例]
        IndependentProcessor --> |独立加载| VADModel[Silero VAD模型]
        IndependentProcessor --> |独立管理| VADIterator[VAD迭代器]
        IndependentProcessor --> |独立缓冲| FrameBuffer[帧对齐缓冲区]
        IndependentProcessor --> |独立状态| StateMachine[状态机]
    end
    
    subgraph "异步处理流程"
        VADModel --> |asyncio.to_thread| VADInference[VAD推理]
        VADInference --> StateMachine
        StateMachine --> |None| SingleFrame[单帧输出]
        StateMachine --> |start| Collecting[开始收集]
        StateMachine --> |end| SpeechSegment[语音段输出]
    end
```

## 🚀 快速开始

### 安装

```bash
# 建议使用uv
uv venv -p 3.12

source .venv/bin/activate

# 从PyPI安装（推荐）
pip install cascade-vad

# 或从源码安装
git clone https://github.com/xucailiang/cascade.git
cd cascade
pip install -e .
```

### 基础使用

```python
import cascade
import asyncio

async def basic_example():
    """基础使用示例"""
    
    # 方式1：最简单的文件处理
    async for result in cascade.process_audio_file("audio.wav"):
        if result.result_type == "segment":
            segment = result.segment
            print(f"🎤 语音段: {segment.start_timestamp_ms:.0f}ms - {segment.end_timestamp_ms:.0f}ms")
        else:
            frame = result.frame
            print(f"🔇 单帧: {frame.timestamp_ms:.0f}ms")
    
    # 方式2：流式处理
    async with cascade.StreamProcessor() as processor:
        async for result in processor.process_stream(audio_stream):
            if result.result_type == "segment":
                segment = result.segment
                print(f"🎤 语音段: {segment.start_timestamp_ms:.0f}ms - {segment.end_timestamp_ms:.0f}ms")
            else:
                frame = result.frame
                print(f"🔇 单帧: {frame.timestamp_ms:.0f}ms")

asyncio.run(basic_example())
```

### 高级配置

```python
from cascade.stream import StreamProcessor, create_default_config

async def advanced_example():
    """高级配置示例"""
    
    # 自定义配置
    config = create_default_config(
        vad_threshold=0.7,          # 较高的检测阈值
        max_instances=3,            # 最多3个并发实例
        buffer_size_frames=128      # 较大缓冲区
    )
    
    # 使用自定义配置
    async with StreamProcessor(config) as processor:
        # 处理音频流
        async for result in processor.process_stream(audio_stream, "my-stream"):
            # 处理结果...
            pass
        
        # 获取性能统计
        stats = processor.get_stats()
        print(f"处理统计: {stats.summary()}")
        print(f"吞吐量: {stats.throughput_chunks_per_second:.1f} 块/秒")

asyncio.run(advanced_example())
```

## 🧪 测试脚本

```bash
# 运行基础集成测试
python tests/test_simple_vad.py -v

# 运行模拟流式音频测试
python tests/test_stream_vad.py -v

# 运行性能基准测试
python tests/benchmark_performance.py
```

测试覆盖：
- ✅ 基础API使用
- ✅ 流式处理功能
- ✅ 文件处理功能
- ✅ 真实音频VAD检测
- ✅ 语音段自动保存
- ✅ 1:1:1:1架构验证
- ✅ 性能基准测试
- ✅ FrameAlignedBuffer测试

## 🌐 Web演示

我们提供了一个完整的基于WebSocket的Web演示应用，展示Cascade的实时VAD能力和多客户端支持。

![Web演示截图](web_demo/test_image.png)

### 功能特性

- **实时音频处理**：通过浏览器麦克风捕获音频并进行VAD处理
- **实时VAD可视化**：实时显示VAD检测结果
- **语音段管理**：显示检测到的语音段并支持回放
- **动态VAD配置**：实时调整VAD参数
- **多客户端支持**：每个WebSocket连接获得独立的Cascade实例

### 快速启动

```bash
# 启动后端服务器
cd web_demo
python server.py

# 启动前端应用（另开终端）
cd web_demo/frontend
pnpm install && pnpm dev
```

详细的安装和配置说明请参见：[Web演示文档](web_demo/README.md)。

## 🔧 生产环境部署

### 部署最佳实践

1. **资源配置**
   - 每个实例约占用50MB内存
   - 建议每个CPU核心运行2-3个实例
   - 监控内存使用，避免OOM

2. **性能调优**
   - 调整`max_instances`匹配服务器CPU核心数
   - 增大`buffer_size_frames`提高吞吐量
   - 根据需求调整`vad_threshold`平衡准确率和灵敏度

3. **错误处理**
   - 实现重试机制处理临时错误
   - 使用健康检查监控服务状态
   - 记录详细日志便于问题排查

### 监控指标

```python
# 获取性能监控指标
stats = processor.get_stats()

# 关键监控指标
print(f"活跃实例数: {stats.active_instances}/{stats.total_instances}")
print(f"平均处理时间: {stats.average_processing_time_ms}ms")
print(f"处理成功率: {stats.success_rate:.2%}")
print(f"内存使用: {stats.memory_usage_mb:.1f}MB")
```

## 🔧 依赖要求

### 核心依赖

- **Python**: 3.12(推荐)
- **pydantic**: 2.4.0+ (数据验证)
- **numpy**: 1.24.0+ (数值计算)
- **scipy**: 1.11.0+ (信号处理)
- **silero-vad**: 5.1.2+ (VAD模型)
- **onnxruntime**: 1.22.1+ (ONNX推理)
- **torchaudio**: 2.7.1+ (音频处理)

### 开发依赖

- **pytest**: 测试框架
- **black**: 代码格式化
- **ruff**: 代码检查
- **mypy**: 类型检查
- **pre-commit**: Git钩子

## 🤝 贡献指南

我们欢迎社区贡献！请遵循以下步骤：

1. **Fork项目**并创建特性分支
2. **安装开发依赖**: `pip install -e .[dev]`
3. **运行测试**: `pytest`
4. **代码检查**: `ruff check . && black --check .`
5. **类型检查**: `mypy cascade`
6. **提交PR**并描述变更

## 📄 许可证

本项目采用MIT许可证 - 详见 [LICENSE](LICENSE) 文件。

## 🙏 致谢

- **Silero Team**: 提供优秀的VAD模型
- **PyTorch Team**: 深度学习框架支持
- **Pydantic Team**: 类型验证系统
- **Python社区**: 丰富的生态系统

## 📞 联系方式

- **作者**: Xucailiang
- **邮箱**: xucailiang.ai@gmail.com
- **项目主页**: https://github.com/xucailiang/cascade
- **问题反馈**: https://github.com/xucailiang/cascade/issues
- **文档**: https://cascade-vad.readthedocs.io/

## 🗺️ 路线图

### v0.2.0 (计划中)
- [ ] 支持更多音频格式 (MP3, FLAC)
- [ ] 实时麦克风输入支持
- [ ] WebSocket API接口
- [ ] 性能优化和内存减少

### v0.3.0 (计划中)
- [ ] 多语言VAD模型支持
- [ ] 语音分离和增强
- [ ] 云端部署支持
- [ ] 可视化监控界面

---

**⭐ 如果这个项目对您有帮助，请给我们一个Star！**
