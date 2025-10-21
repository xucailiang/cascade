# Cascade 测试指南

这个目录包含了 Cascade 流式VAD库的各种测试脚本。本指南将帮助你了解如何运行这些测试并理解它们的用途。

## 📋 测试文件概览

| 文件名 | 用途 | 难度 | 推荐顺序 |
|--------|------|------|----------|
| `test_simple_vad.py` | 基础VAD功能测试 | ⭐ 简单 | 1️⃣ 首先运行 |
| `test_stream_vad.py` | 流式VAD测试（支持块大小对比） | ⭐⭐ 中等 | 2️⃣ 主要测试 |
| `test_real_multithread_cascade.py` | 多线程并发测试 | ⭐⭐⭐ 复杂 | 3️⃣ 高级测试 |
| `benchmark_performance.py` | 性能基准测试 | ⭐⭐⭐ 复杂 | 4️⃣ 性能分析 |

## 🚀 快速开始

### 前置条件

1. **确保环境已激活**：
   ```bash
   # 如果使用虚拟环境
   source venv/bin/activate  # Linux/Mac
   # 或
   venv\Scripts\activate     # Windows
   ```

2. **准备音频文件**：
   将测试音频文件放在项目根目录，支持的文件名：
   - `我现在开始录音，理论上会有两个文件.wav`
   
   或者使用 `--audio-file` 参数指定自定义音频文件。

3. **音频文件要求**：
   - 格式：WAV
   - 采样率：建议16kHz
   - 声道：单声道
   - 位深：16bit

## 📝 详细测试说明

### 1. 基础功能测试

**文件**：`test_simple_vad.py`  
**用途**：验证基本的VAD检测功能  
**运行时间**：约10-30秒

```bash
# 基础测试
python tests/test_simple_vad.py
```

**预期输出**：
- 检测到的语音段数量
- 每个语音段的时间范围
- 基本的处理统计信息

---

### 2. 流式VAD测试（⭐ 推荐）

**文件**：`test_stream_vad.py`  
**用途**：测试流式处理和不同音频块大小的性能影响  
**运行时间**：单次测试30秒-2分钟，对比测试2-5分钟

#### 基础用法

```bash
# 使用默认4096字节块大小
python tests/test_stream_vad.py

# 测试特定块大小
python tests/test_stream_vad.py --chunk-size 1024
python tests/test_stream_vad.py --chunk-size 8192
```

#### 对比测试（推荐）

```bash
# 使用默认块大小列表进行对比
python tests/test_stream_vad.py --compare

# 自定义块大小列表
python tests/test_stream_vad.py --compare --chunk-sizes "1024,4096,8192"

# 测试极端情况
python tests/test_stream_vad.py --compare --chunk-sizes "512,1024,2048,4096,8192,16384"
```

#### 指定音频文件

```bash
# 使用自定义音频文件
python tests/test_stream_vad.py --audio-file "path/to/your/audio.wav" --compare
```

#### 查看帮助

```bash
python tests/test_stream_vad.py --help
```

**预期输出**：
- 详细的处理统计（处理时间、吞吐量等）
- 语音段检测结果
- 性能对比表格（对比模式）
- 优化建议

**输出文件**：
- `stream_speech_segments_[块大小]/` - 检测到的语音段WAV文件

---

### 3. 多线程并发测试

**文件**：`test_real_multithread_cascade.py`  
**用途**：测试多线程环境下的并发处理能力  
**运行时间**：1-3分钟

```bash
# 多线程测试
python tests/test_real_multithread_cascade.py
```

**预期输出**：
- 多个线程的并发处理结果
- 线程安全性验证
- 并发性能统计

---

### 4. 性能基准测试

**文件**：`benchmark_performance.py`  
**用途**：详细的性能基准测试和分析  
**运行时间**：3-10分钟

```bash
# 性能基准测试
python tests/benchmark_performance.py
```

**预期输出**：
- 详细的性能指标
- 内存使用情况
- CPU利用率
- 处理延迟分析

## 🔧 故障排除

### 常见问题

#### 1. 找不到音频文件
```
❌ 未找到可用的音频文件
```

**解决方案**：
- 确保音频文件在项目根目录
- 使用 `--audio-file` 参数指定文件路径
- 检查文件名是否正确

#### 2. 模块导入错误
```
ModuleNotFoundError: No module named 'cascade'
```

**解决方案**：
```bash
# 确保在项目根目录运行
cd /path/to/cascade

# 确保虚拟环境已激活
source venv/bin/activate

# 安装依赖
pip install -e .
```

#### 3. 音频格式不支持
```
❌ 音频流模拟失败
```

**解决方案**：
- 确保音频文件是WAV格式
- 检查音频文件是否损坏
- 尝试使用其他音频文件

#### 4. 内存不足
```
MemoryError
```

**解决方案**：
- 使用较小的音频文件
- 减少并发线程数
- 使用较小的块大小列表

### 调试技巧

#### 1. 启用详细日志
```bash
# 设置日志级别
export PYTHONPATH=.
python -c "import logging; logging.basicConfig(level=logging.DEBUG)"
python tests/test_stream_vad.py
```

#### 2. 检查系统资源
```bash
# 监控内存使用
htop

# 监控磁盘空间
df -h
```

#### 3. 验证音频文件
```bash
# 使用ffprobe检查音频信息
ffprobe your_audio.wav

# 或使用Python检查
python -c "
import wave
with wave.open('your_audio.wav', 'rb') as f:
    print(f'Channels: {f.getnchannels()}')
    print(f'Sample width: {f.getsampwidth()}')
    print(f'Frame rate: {f.getframerate()}')
    print(f'Frames: {f.getnframes()}')
"
```

## 📊 测试结果解读

### 性能指标说明

- **平均处理时间**：每个音频块的平均处理时间（毫秒）
- **吞吐量**：每秒处理的音频块数量
- **语音段数量**：检测到的语音段总数
- **准确率**：语音检测的准确性（需要人工验证）

### 性能基准参考

| 指标 | 优秀 | 良好 | 一般 | 需优化 |
|------|------|------|------|--------|
| 处理时间 | <1ms | 1-3ms | 3-10ms | >10ms |
| 吞吐量 | >100块/秒 | 50-100块/秒 | 20-50块/秒 | <20块/秒 |
| 内存使用 | <100MB | 100-200MB | 200-500MB | >500MB |

## 🎯 测试建议

### 新用户建议流程

1. **首次测试**：
   ```bash
   python tests/test_simple_vad.py
   ```

2. **流式测试**：
   ```bash
   python tests/test_stream_vad.py
   ```

3. **性能对比**：
   ```bash
   python tests/test_stream_vad.py --compare
   ```

4. **根据结果选择最优配置**

### 开发者测试流程

1. **功能验证**：运行所有基础测试
2. **性能基准**：建立性能基线
3. **回归测试**：代码变更后重新测试
4. **压力测试**：多线程和大文件测试

### CI/CD 集成

```bash
#!/bin/bash
# 自动化测试脚本

echo "开始 Cascade 测试套件..."

# 基础功能测试
python tests/test_simple_vad.py || exit 1

# 流式测试
python tests/test_stream_vad.py --chunk-size 4096 || exit 1

# 性能对比测试
python tests/test_stream_vad.py --compare --chunk-sizes "1024,4096,8192" || exit 1

echo "所有测试通过！"
```

## 📚 相关文档

- [`README_chunk_size_testing.md`](./README_chunk_size_testing.md) - 音频块大小测试详细指南
- 项目根目录的 `README.md` - 项目总体说明
- `docs/` 目录 - 详细的技术文档

## 🤝 贡献指南

如果你发现测试问题或想要添加新的测试：

1. 确保新测试遵循现有的命名规范
2. 添加适当的文档说明
3. 更新这个README文件
4. 提交Pull Request

---

**💡 提示**：建议先运行 `test_stream_vad.py --compare` 来了解你的系统性能特征，然后根据结果选择最适合你应用场景的配置。