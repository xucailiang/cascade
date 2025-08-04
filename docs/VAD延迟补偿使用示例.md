# VAD延迟补偿使用示例

本文档提供VAD延迟补偿功能的详细使用示例和最佳实践。

## 基本使用示例

### 1. 简单启用延迟补偿

```python
from cascade.types import VADConfig, AudioConfig
from cascade.processor import create_vad_processor

# 创建包含延迟补偿的VAD配置
vad_config = VADConfig(
    backend="silero",
    threshold=0.5,
    chunk_duration_ms=500,
    overlap_ms=16,
    compensation_ms=200  # 启用200ms延迟补偿
)

audio_config = AudioConfig(
    sample_rate=16000,
    channels=1,
    dtype="float32"
)

# 创建VAD处理器
processor = await create_vad_processor(audio_config, vad_config)
```

### 2. 处理音频流

```python
import asyncio
import numpy as np

async def process_audio_with_compensation():
    """演示带延迟补偿的音频处理"""
    
    # 模拟音频流
    async def audio_stream():
        for i in range(10):
            # 生成500ms的音频数据
            chunk_size = 8000  # 16kHz * 0.5s
            audio_data = np.random.randn(chunk_size).astype(np.float32) * 0.1
            yield audio_data
            await asyncio.sleep(0.01)
    
    async with processor:
        # 处理音频流并获取结果
        async for vad_result in processor.process_stream(audio_stream()):
            if vad_result.is_speech:
                if vad_result.is_compensated:
                    print(f"🎯 语音开始（已补偿）: "
                          f"原始={vad_result.original_start_ms:.1f}ms, "
                          f"补偿后={vad_result.start_ms:.1f}ms")
                else:
                    print(f"🗣️  语音继续: {vad_result.start_ms:.1f}ms")
            else:
                print(f"🔇 静音: {vad_result.start_ms:.1f}ms")

# 运行示例
await process_audio_with_compensation()
```

## 配置选项详解

### 1. 禁用延迟补偿

```python
# 方法1：设置为0
vad_config = VADConfig(
    backend="silero",
    compensation_ms=0  # 禁用补偿
)

# 方法2：不设置（默认为0）
vad_config = VADConfig(
    backend="silero"
    # compensation_ms 默认为0
)
```

### 2. 不同补偿时长的选择

```python
# 轻度补偿（适合高精度场景）
light_compensation = VADConfig(
    backend="silero",
    compensation_ms=100
)

# 标准补偿（推荐设置）
standard_compensation = VADConfig(
    backend="silero",
    compensation_ms=200
)

# 强补偿（适合实时响应要求高的场景）
strong_compensation = VADConfig(
    backend="silero",
    compensation_ms=300
)
```

## 实际应用场景

### 1. 实时语音识别系统

```python
class RealTimeASR:
    """实时语音识别系统示例"""
    
    def __init__(self):
        self.vad_config = VADConfig(
            backend="silero",
            threshold=0.6,
            compensation_ms=250  # 较强补偿确保捕获语音开头
        )
        self.audio_config = AudioConfig(sample_rate=16000, channels=1)
        
    async def start_recognition(self, audio_stream):
        processor = await create_vad_processor(
            self.audio_config, 
            self.vad_config
        )
        
        async with processor:
            speech_buffer = []
            
            async for vad_result in processor.process_stream(audio_stream):
                if vad_result.is_speech:
                    if vad_result.is_compensated:
                        # 语音开始，开始收集音频
                        print(f"🎤 开始录音（补偿了{vad_result.original_start_ms - vad_result.start_ms:.0f}ms）")
                        speech_buffer = []
                    
                    # 收集语音数据
                    speech_buffer.append(vad_result)
                    
                else:
                    if speech_buffer:
                        # 语音结束，进行识别
                        await self.recognize_speech(speech_buffer)
                        speech_buffer = []
                        
    async def recognize_speech(self, speech_data):
        """模拟语音识别过程"""
        duration = speech_data[-1].end_ms - speech_data[0].start_ms
        print(f"🔄 识别语音段：{duration:.0f}ms")
```

### 2. 语音激活检测

```python
class VoiceActivation:
    """语音激活检测示例"""
    
    def __init__(self, wake_word_threshold=500):
        self.wake_word_threshold = wake_word_threshold  # 最短激活时长
        self.vad_config = VADConfig(
            backend="silero",
            threshold=0.7,  # 较高阈值避免误激活
            compensation_ms=150
        )
        
    async def detect_activation(self, audio_stream):
        processor = await create_vad_processor(
            AudioConfig(sample_rate=16000, channels=1),
            self.vad_config
        )
        
        async with processor:
            speech_start_time = None
            
            async for vad_result in processor.process_stream(audio_stream):
                if vad_result.is_speech:
                    if vad_result.is_compensated:
                        # 记录语音开始时间（使用补偿后的时间）
                        speech_start_time = vad_result.start_ms
                        print(f"👂 检测到语音开始: {speech_start_time:.1f}ms")
                        
                else:
                    if speech_start_time is not None:
                        # 检查语音持续时长
                        duration = vad_result.start_ms - speech_start_time
                        if duration >= self.wake_word_threshold:
                            print(f"✅ 语音激活！持续时长: {duration:.0f}ms")
                            await self.handle_activation()
                        else:
                            print(f"❌ 语音太短，忽略: {duration:.0f}ms")
                        
                        speech_start_time = None
                        
    async def handle_activation(self):
        """处理语音激活事件"""
        print("🚀 系统激活，开始监听命令...")
```

## 高级用法

### 1. 动态调整补偿参数

```python
from cascade.processor.delay_compensator import SimpleDelayCompensator

# 创建独立的补偿器
compensator = SimpleDelayCompensator(compensation_ms=200)

# 动态调整补偿时长
compensator.set_compensation_ms(150)

# 检查当前设置
print(f"当前补偿时长: {compensator.get_compensation_ms()}ms")
print(f"是否启用: {compensator.is_enabled()}")

# 处理VAD结果
vad_result = get_vad_result()  # 获取VAD结果
compensated_result = compensator.process_result(vad_result)
```

### 2. 多音频流处理

```python
async def process_multiple_streams():
    """处理多个音频流的示例"""
    
    # 不同场景使用不同的补偿配置
    configs = {
        "meeting": VADConfig(backend="silero", compensation_ms=100),
        "phone": VADConfig(backend="silero", compensation_ms=250),
        "broadcast": VADConfig(backend="silero", compensation_ms=150),
    }
    
    processors = {}
    for name, config in configs.items():
        processors[name] = await create_vad_processor(
            AudioConfig(sample_rate=16000, channels=1),
            config
        )
    
    try:
        # 并行处理多个流
        tasks = []
        for name, processor in processors.items():
            stream = get_audio_stream(name)  # 获取对应的音频流
            task = asyncio.create_task(
                process_single_stream(name, processor, stream)
            )
            tasks.append(task)
        
        await asyncio.gather(*tasks)
        
    finally:
        # 清理资源
        for processor in processors.values():
            await processor.close()

async def process_single_stream(name, processor, stream):
    """处理单个音频流"""
    async for vad_result in processor.process_stream(stream):
        if vad_result.is_compensated:
            print(f"[{name}] 语音开始（补偿: "
                  f"{vad_result.original_start_ms - vad_result.start_ms:.0f}ms）")
```

## 性能优化建议

### 1. 合理选择补偿时长

```python
# 根据应用场景选择合适的补偿时长
COMPENSATION_PRESETS = {
    "real_time_chat": 300,      # 实时聊天，优先响应速度
    "voice_recording": 200,     # 语音录制，平衡准确性和响应
    "transcription": 150,       # 转录服务，优先准确性
    "voice_commands": 250,      # 语音命令，快速响应
    "meeting_notes": 100,       # 会议记录，高精度要求
}

def create_optimized_config(scenario: str):
    compensation = COMPENSATION_PRESETS.get(scenario, 200)
    return VADConfig(
        backend="silero",
        compensation_ms=compensation
    )
```

### 2. 批处理优化

```python
async def batch_process_with_compensation():
    """批处理模式的优化示例"""
    
    vad_config = VADConfig(
        backend="silero",
        compensation_ms=200,
        chunk_duration_ms=1000,  # 更大的块减少处理频率
        overlap_ms=50            # 适度重叠确保准确性
    )
    
    processor = await create_vad_processor(
        AudioConfig(sample_rate=16000, channels=1),
        vad_config
    )
    
    # 处理逻辑...
```

## 故障排除

### 1. 常见问题

**问题**: 补偿效果不明显
```python
# 解决方案：检查配置
config = VADConfig(compensation_ms=200)
print(f"补偿设置: {config.compensation_ms}ms")

# 验证补偿器是否正常工作
compensator = SimpleDelayCompensator(200)
print(f"补偿器启用: {compensator.is_enabled()}")
```

**问题**: 过度补偿导致时间戳为负
```python
# 内置保护机制会自动处理
vad_result = VADResult(
    is_speech=True,
    start_ms=50.0,  # 小于补偿时长
    end_ms=500.0,
    chunk_id=1
)

compensator = SimpleDelayCompensator(200)
result = compensator.process_result(vad_result)
print(f"补偿后时间: {result.start_ms}ms")  # 输出: 0.0ms (已保护)
```

### 2. 调试模式

```python
# 启用详细日志查看补偿过程
import logging
logging.getLogger('cascade.processor.delay_compensator').setLevel(logging.DEBUG)

# 检查补偿结果
def debug_compensation(vad_result):
    if vad_result.is_compensated:
        compensation = vad_result.original_start_ms - vad_result.start_ms
        print(f"🔧 调试: 原始={vad_result.original_start_ms:.1f}ms, "
              f"补偿={compensation:.1f}ms, "
              f"结果={vad_result.start_ms:.1f}ms")
```

## 最佳实践总结

1. **补偿时长选择**: 
   - 实时应用: 200-300ms
   - 录制应用: 150-200ms
   - 高精度应用: 100-150ms

2. **性能考虑**:
   - 补偿功能几乎零开销
   - 仅在语音开始时进行时间戳调整
   - 不影响流式处理性能

3. **集成建议**:
   - 在配置阶段设置补偿参数
   - 使用工厂函数创建处理器
   - 定期检查补偿效果并调整参数

4. **监控指标**:
   - 语音开始检测延迟
   - 补偿频率和幅度
   - 整体识别准确性

通过合理使用延迟补偿功能，可以显著提升VAD系统的实时性能和用户体验。