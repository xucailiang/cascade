"""
Cascade 流式处理器模块

提供基于VAD状态机的异步流式音频处理能力。

主要组件：
- StreamProcessor: 核心流式处理器
- CascadeInstance: 单个处理实例
- VADStateMachine: VAD状态机
- SpeechCollector: 语音帧收集器

使用示例：
    ```python
    from cascade.stream import StreamProcessor, Config
    
    # 创建配置
    config = Config()
    
    # 创建处理器
    async with StreamProcessor(config) as processor:
        # 处理音频流
        async for result in processor.process_stream(audio_stream):
            if result.is_speech_segment:
                print(f"检测到语音段: {result.segment}")
            else:
                print(f"单帧结果: {result.frame}")
    ```
"""

from .types import (
    AudioFrame,
    SpeechSegment, 
    CascadeResult,
    Config,
    ProcessorStats,
    AUDIO_SAMPLE_RATE,
    AUDIO_FRAME_SIZE,
    AUDIO_FRAME_DURATION_MS,
    AUDIO_CHANNELS,
    AUDIO_SAMPLE_WIDTH
)

from .collector import SpeechCollector
from .state_machine import VADStateMachine, VADState
from .instance import CascadeInstance
from .processor import StreamProcessor


# 便捷函数
async def process_audio_stream(
    audio_stream,
    config: Config | None = None,
    stream_id: str | None = None
):
    """
    便捷函数：处理音频流
    
    Args:
        audio_stream: 音频数据流
        config: 处理配置，默认使用标准配置
        stream_id: 流标识符
        
    Yields:
        处理结果
    """
    if config is None:
        config = Config()
    
    async with StreamProcessor(config) as processor:
        async for result in processor.process_stream(audio_stream, stream_id):
            yield result


async def process_audio_chunk(
    audio_data: bytes,
    config: Config | None = None
):
    """
    便捷函数：处理单个音频块
    
    Args:
        audio_data: 音频数据
        config: 处理配置，默认使用标准配置
        
    Returns:
        处理结果列表
    """
    if config is None:
        config = Config()
    
    async with StreamProcessor(config) as processor:
        return await processor.process_chunk(audio_data)


def create_default_config(**kwargs) -> Config:
    """
    创建默认配置
    
    Args:
        **kwargs: 配置参数覆盖
        
    Returns:
        配置对象
    """
    return Config(**kwargs)


def create_stream_processor(config: Config | None = None) -> StreamProcessor:
    """
    创建流式处理器
    
    Args:
        config: 处理配置，默认使用标准配置
        
    Returns:
        流式处理器实例
    """
    if config is None:
        config = Config()
    
    return StreamProcessor(config)


__all__ = [
    # 核心类型
    "AudioFrame",
    "SpeechSegment", 
    "CascadeResult",
    "Config",
    "ProcessorStats",
    
    # 常量
    "AUDIO_SAMPLE_RATE",
    "AUDIO_FRAME_SIZE", 
    "AUDIO_FRAME_DURATION_MS",
    "AUDIO_CHANNELS",
    "AUDIO_SAMPLE_WIDTH",
    
    # 核心组件
    "SpeechCollector",
    "VADStateMachine",
    "VADState",
    "CascadeInstance", 
    "StreamProcessor",
    
    # 便捷函数
    "process_audio_stream",
    "process_audio_chunk",
    "create_default_config",
    "create_stream_processor",
]