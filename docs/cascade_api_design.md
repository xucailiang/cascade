# Cascade API 接口设计

## 设计原则

1. **简洁优先**：API应该简单直观，新用户能快速上手
2. **渐进式复杂度**：从简单用法到高级配置的平滑过渡
3. **异步优先**：所有主要API都是异步的，支持现代Python编程
4. **上下文管理**：支持`async with`语法，自动资源管理
5. **流式处理**：支持`AsyncIterator`，适合实时音频处理
6. **类型安全**：完整的类型注解，支持IDE智能提示

## 核心API设计

### 1. 主要入口点

```python
# cascade/__init__.py - 简洁的顶级API

import cascade

# 方式1：最简单的使用方式
results = await cascade.process_audio_file("audio.wav")

# 方式2：流式处理
async with cascade.StreamProcessor() as processor:
    async for vad_result in processor.process_stream(audio_stream):
        print(f"语音检测: {vad_result.is_speech}")

# 方式3：逐块处理
processor = cascade.StreamProcessor()
await processor.initialize()
result = await processor.process_chunk(audio_data)
await processor.close()
```

### 2. StreamProcessor - 核心处理器

```python
class StreamProcessor:
    """
    Cascade流式VAD处理器
    
    这是Cascade的核心API，提供异步流式音频VAD处理能力。
    每个StreamProcessor实例管理一个独立的处理线程和VAD模型。
    
    Examples:
        基础使用:
        >>> async with cascade.StreamProcessor() as processor:
        ...     async for result in processor.process_stream(audio_stream):
        ...         if result.is_speech:
        ...             print(f"检测到语音: {result.start_ms}ms")
        
        自定义配置:
        >>> config = cascade.Config(sample_rate=8000, threshold=0.7)
        >>> async with cascade.StreamProcessor(config) as processor:
        ...     result = await processor.process_chunk(audio_data)
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        初始化流式处理器
        
        Args:
            config: 可选的配置对象，None时使用默认配置
        """
    
    async def initialize(self) -> None:
        """
        异步初始化处理器
        
        加载VAD模型，创建缓冲区，启动处理线程。
        如果使用上下文管理器，会自动调用此方法。
        
        Raises:
            CascadeError: 初始化失败时抛出
        """
    
    async def process_chunk(self, audio_data: bytes) -> CascadeResult:
        """
        处理单个音频块
        
        Args:
            audio_data: 音频数据（PCM格式，512样本）
            
        Returns:
            Cascade处理结果（单帧或语音段）
            
        Raises:
            CascadeError: 处理失败时抛出
        """
    
    async def process_stream(self,
                           audio_stream: AsyncIterator[bytes]) -> AsyncIterator[CascadeResult]:
        """
        处理音频流
        
        Args:
            audio_stream: 异步音频数据流
            
        Yields:
            Cascade处理结果（单帧或语音段）
            
        Raises:
            CascadeError: 处理失败时抛出
        """
    
    async def process_file(self, file_path: str) -> AsyncIterator[CascadeResult]:
        """
        处理音频文件
        
        Args:
            file_path: 音频文件路径
            
        Yields:
            Cascade处理结果（单帧或语音段）
            
        Raises:
            CascadeError: 文件读取或处理失败时抛出
        """
    
    def get_stats(self) -> ProcessorStats:
        """
        获取处理器统计信息
        
        Returns:
            包含性能指标的统计对象
        """
    
    async def close(self) -> None:
        """
        关闭处理器，释放资源
        
        如果使用上下文管理器，会自动调用此方法。
        """
    
    # 上下文管理器支持
    async def __aenter__(self) -> 'StreamProcessor':
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()
```

### 3. 配置系统

```python
class Config(BaseModel):
    """
    Cascade配置
    
    提供所有可配置的参数，使用合理的默认值。
    
    Examples:
        默认配置:
        >>> config = cascade.Config()
        
        自定义配置:
        >>> config = cascade.Config(
        ...     sample_rate=8000,
        ...     chunk_duration_ms=30,
        ...     vad_threshold=0.7,
        ...     max_instances=3
        ... )
    """
    
    # 音频配置 - 基于silero-vad优化
    sample_rate: int = Field(default=16000, description="采样率(Hz)", frozen=True)  # 固定16kHz
    frame_size: int = Field(default=512, description="VAD帧大小(样本)", frozen=True)  # 固定512样本
    frame_duration_ms: float = Field(default=32.0, description="帧时长(ms)", frozen=True)  # 32ms
    supported_formats: List[str] = Field(default=["wav", "mp3"], description="支持的音频格式", frozen=True)
    
    # VAD配置
    vad_threshold: float = Field(default=0.5, description="VAD检测阈值", ge=0.0, le=1.0)
    min_speech_duration_ms: int = Field(default=100, description="最小语音时长(ms)")
    min_silence_duration_ms: int = Field(default=100, description="最小静音时长(ms)")
    
    # 性能配置
    max_instances: int = Field(default=5, description="最大并发实例数")
    buffer_size_seconds: float = Field(default=2.0, description="缓冲区大小(秒)")
    
    # 高级配置
    enable_logging: bool = Field(default=True, description="是否启用日志")
    log_level: str = Field(default="INFO", description="日志级别")
    enable_profiling: bool = Field(default=False, description="是否启用性能分析")
    
    class Config:
        extra = "forbid"
        json_schema_extra = {
            "examples": [
                {
                    "sample_rate": 16000,
                    "chunk_duration_ms": 30,
                    "vad_threshold": 0.5,
                    "max_instances": 5
                }
            ]
        }
```

### 4. 数据类型

基于VAD状态机设计的新数据类型：

```python
class AudioFrame(BaseModel):
    """
    单个音频帧
    
    表示512样本的音频帧和相关元数据。
    """
    
    # 基础信息
    frame_id: int = Field(description="帧ID")
    audio_data: bytes = Field(description="512样本音频数据")
    timestamp_ms: float = Field(description="时间戳(ms)")
    
    # VAD信息
    vad_result: Optional[Dict[str, Any]] = Field(description="原始VAD结果")
    
    # 元数据
    sample_rate: int = Field(default=16000, description="采样率")
    frame_size: int = Field(default=512, description="帧大小(样本)")
    
    def __str__(self) -> str:
        vad_str = str(self.vad_result) if self.vad_result else "None"
        return f"AudioFrame(id={self.frame_id}, vad={vad_str}, {self.timestamp_ms:.0f}ms)"

class SpeechSegment(BaseModel):
    """
    语音段
    
    表示从VAD检测到start到end之间的完整语音片段。
    """
    
    # 基础信息
    segment_id: int = Field(description="语音段ID")
    audio_data: bytes = Field(description="合并的音频数据")
    
    # 时间信息
    start_timestamp_ms: float = Field(description="开始时间戳(ms)")
    end_timestamp_ms: float = Field(description="结束时间戳(ms)")
    
    # 统计信息
    frame_count: int = Field(description="包含的帧数")
    duration_ms: float = Field(description="时长(ms)")
    
    # VAD信息
    start_vad_result: Dict[str, Any] = Field(description="开始VAD结果")
    end_vad_result: Dict[str, Any] = Field(description="结束VAD结果")
    
    # 元数据
    sample_rate: int = Field(default=16000, description="采样率")
    
    def __str__(self) -> str:
        return f"SpeechSegment(id={self.segment_id}, frames={self.frame_count}, {self.duration_ms:.0f}ms)"

class CascadeResult(BaseModel):
    """
    Cascade输出结果
    
    统一的输出接口，可以是单帧或语音段。
    """
    
    # 结果类型
    result_type: Literal["frame", "segment"] = Field(description="结果类型")
    
    # 结果数据
    frame: Optional[AudioFrame] = Field(default=None, description="单帧结果")
    segment: Optional[SpeechSegment] = Field(default=None, description="语音段结果")
    
    # 处理信息
    processing_time_ms: float = Field(description="处理时间(ms)")
    instance_id: str = Field(description="处理实例ID")
    
    def __str__(self) -> str:
        if self.result_type == "frame":
            return f"CascadeResult(frame: {self.frame})"
        else:
            return f"CascadeResult(segment: {self.segment})"
    
    @property
    def is_speech_segment(self) -> bool:
        """是否为语音段"""
        return self.result_type == "segment"
    
    @property
    def is_single_frame(self) -> bool:
        """是否为单帧"""
        return self.result_type == "frame"

class ProcessorStats(BaseModel):
    """
    处理器统计信息
    """
    
    # 处理统计
    total_chunks_processed: int = Field(description="总处理块数")
    total_processing_time_ms: float = Field(description="总处理时间(ms)")
    average_processing_time_ms: float = Field(description="平均处理时间(ms)")
    
    # 检测统计
    speech_chunks: int = Field(description="语音块数")
    silence_chunks: int = Field(description="静音块数")
    speech_ratio: float = Field(description="语音比例")
    
    # 性能统计
    throughput_chunks_per_second: float = Field(description="吞吐量(块/秒)")
    memory_usage_mb: float = Field(description="内存使用(MB)")
    
    # 错误统计
    error_count: int = Field(description="错误次数")
    error_rate: float = Field(description="错误率")
    
    def summary(self) -> str:
        """返回统计摘要"""
        return (f"处理了{self.total_chunks_processed}个块, "
                f"语音比例{self.speech_ratio:.1%}, "
                f"平均处理时间{self.average_processing_time_ms:.1f}ms")
```

### 5. 便捷函数

```python
# 顶级便捷函数，适合快速使用

async def process_audio_file(file_path: str,
                           config: Optional[Config] = None) -> List[CascadeResult]:
    """
    处理音频文件的便捷函数
    
    Args:
        file_path: 音频文件路径
        config: 可选配置
        
    Returns:
        Cascade结果列表
        
    Example:
        >>> results = await cascade.process_audio_file("audio.wav")
        >>> speech_segments = [r for r in results if r.is_speech_segment]
        >>> print(f"检测到{len(speech_segments)}个语音段")
    """
    async with StreamProcessor(config) as processor:
        results = []
        async for result in processor.process_file(file_path):
            results.append(result)
        return results

async def detect_speech_segments(audio_stream: AsyncIterator[bytes],
                                config: Optional[Config] = None) -> List[SpeechSegment]:
    """
    检测语音段的便捷函数
    
    Args:
        audio_stream: 音频数据流
        config: 可选配置
        
    Returns:
        语音段列表
        
    Example:
        >>> segments = await cascade.detect_speech_segments(audio_stream)
        >>> for segment in segments:
        ...     print(f"语音段: {segment.start_timestamp_ms}ms - {segment.end_timestamp_ms}ms")
    """
    async with StreamProcessor(config) as processor:
        segments = []
        
        async for result in processor.process_stream(audio_stream):
            if result.is_speech_segment:
                # 直接收集语音段
                segments.append(result.segment)
        
        return segments
```

### 6. 错误处理

```python
class CascadeError(Exception):
    """Cascade基础异常"""
    pass

class InitializationError(CascadeError):
    """初始化错误"""
    pass

class ProcessingError(CascadeError):
    """处理错误"""
    pass

class ConfigurationError(CascadeError):
    """配置错误"""
    pass

# 错误处理示例
try:
    async with cascade.StreamProcessor() as processor:
        result = await processor.process_chunk(audio_data)
except cascade.InitializationError as e:
    print(f"初始化失败: {e}")
except cascade.ProcessingError as e:
    print(f"处理失败: {e}")
except cascade.CascadeError as e:
    print(f"Cascade错误: {e}")
```

## API使用示例

### 1. 基础使用

```python
import cascade
import asyncio

async def basic_usage():
    """基础使用示例"""
    
    # 最简单的方式 - 处理文件
    results = await cascade.process_audio_file("audio.wav")
    
    # 统计语音段和单帧
    speech_segments = [r for r in results if r.is_speech_segment]
    single_frames = [r for r in results if r.is_single_frame]
    print(f"检测到 {len(speech_segments)} 个语音段")
    print(f"检测到 {len(single_frames)} 个单帧")
    
    # 计算语音段总时长
    total_speech_duration = sum(s.segment.duration_ms for s in speech_segments)
    print(f"语音总时长: {total_speech_duration:.0f}ms")

asyncio.run(basic_usage())
```

### 2. 流式处理

```python
import cascade
import asyncio

async def stream_processing():
    """流式处理示例"""
    
    # 模拟音频流
    async def audio_stream():
        # 实际应用中，这里可能是麦克风输入或网络流
        with open("audio.wav", "rb") as f:
            while True:
                chunk = f.read(1024)  # 读取1KB
                if not chunk:
                    break
                yield chunk
                await asyncio.sleep(0.01)  # 模拟实时流
    
    # 流式处理
    async with cascade.StreamProcessor() as processor:
        async for result in processor.process_stream(audio_stream()):
            if result.is_speech_segment:
                segment = result.segment
                print(f"🎤 语音段: {segment.start_timestamp_ms:.0f}ms - {segment.end_timestamp_ms:.0f}ms, 帧数: {segment.frame_count}")
            elif result.is_single_frame:
                frame = result.frame
                vad_str = str(frame.vad_result) if frame.vad_result else "None"
                print(f"🔇 单帧: {frame.timestamp_ms:.0f}ms, VAD: {vad_str}")

asyncio.run(stream_processing())
```

### 3. 高级配置

```python
import cascade
import asyncio

async def advanced_usage():
    """高级配置示例"""
    
    # 自定义配置
    config = cascade.Config(
        sample_rate=8000,           # 8kHz采样率
        chunk_duration_ms=20,       # 20ms块大小
        vad_threshold=0.7,          # 较高的检测阈值
        min_speech_duration_ms=200, # 最小语音时长200ms
        max_instances=3,            # 最多3个并发实例
        enable_profiling=True       # 启用性能分析
    )
    
    # 使用自定义配置
    async with cascade.StreamProcessor(config) as processor:
        # 处理音频文件
        results = []
        async for result in processor.process_file("audio.wav"):
            results.append(result)
        
        # 获取统计信息
        stats = processor.get_stats()
        print(f"处理统计: {stats.summary()}")
        print(f"内存使用: {stats.memory_usage_mb:.1f}MB")
        print(f"吞吐量: {stats.throughput_chunks_per_second:.1f} 块/秒")

asyncio.run(advanced_usage())
```

### 4. 语音段检测

```python
import cascade
import asyncio

async def speech_segment_detection():
    """语音段检测示例"""
    
    # 检测语音段
    with open("audio.wav", "rb") as f:
        audio_data = f.read()
    
    async def audio_stream():
        # 将音频数据分块
        chunk_size = 1024
        for i in range(0, len(audio_data), chunk_size):
            yield audio_data[i:i+chunk_size]
            await asyncio.sleep(0.01)
    
    # 使用便捷函数检测语音段
    segments = await cascade.detect_speech_segments(audio_stream())
    
    # 输出结果
    print(f"检测到 {len(segments)} 个语音段:")
    for i, segment in enumerate(segments, 1):
        print(f"  段{i}: {segment.start_timestamp_ms:.0f}ms - {segment.end_timestamp_ms:.0f}ms "
              f"(时长: {segment.duration_ms:.0f}ms, 帧数: {segment.frame_count})")

asyncio.run(speech_segment_detection())
```

### 5. 错误处理

```python
import cascade
import asyncio
import logging

async def error_handling():
    """错误处理示例"""
    
    # 配置日志
    logging.basicConfig(level=logging.INFO)
    
    try:
        # 可能出错的配置
        config = cascade.Config(
            sample_rate=999999,  # 无效的采样率
            vad_threshold=2.0    # 无效的阈值
        )
        
        async with cascade.StreamProcessor(config) as processor:
            result = await processor.process_chunk(b"invalid audio data")
            
    except cascade.ConfigurationError as e:
        print(f"配置错误: {e}")
    except cascade.InitializationError as e:
        print(f"初始化错误: {e}")
    except cascade.ProcessingError as e:
        print(f"处理错误: {e}")
    except Exception as e:
        print(f"未知错误: {e}")

asyncio.run(error_handling())
```

## API兼容性和版本管理

### 版本策略

- 使用语义化版本控制 (SemVer)
- 主要API保持向后兼容
- 新功能通过可选参数添加
- 废弃功能提供迁移指南

### 类型注解

```python
from typing import AsyncIterator, List, Optional, Dict, Any, Literal
from cascade.types import CascadeResult, AudioFrame, SpeechSegment, Config, ProcessorStats

# 所有公共API都有完整的类型注解
async def process_audio_file(
    file_path: str,
    config: Optional[Config] = None
) -> List[CascadeResult]:
    ...

class StreamProcessor:
    async def process_stream(
        self,
        audio_stream: AsyncIterator[bytes]
    ) -> AsyncIterator[CascadeResult]:
        ...
```

这个API设计确保了：

1. **简洁性**：最常用的功能只需要一行代码
2. **渐进性**：从简单到复杂的平滑过渡
3. **异步性**：完全支持现代Python异步编程
4. **类型安全**：完整的类型注解支持
5. **错误处理**：清晰的错误层次和处理机制
6. **可扩展性**：为未来功能扩展预留空间

符合开源项目的最佳实践，用户可以快速上手，同时支持高级用法。