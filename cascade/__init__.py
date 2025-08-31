"""
Cascade: 高性能异步流式VAD处理库

Cascade是一个专为语音活动检测(VAD)设计的高性能、低延迟音频流处理库。
基于StreamProcessor核心架构，提供简洁的异步流式处理能力。

核心特性:
- 流式处理: 基于VAD状态机的异步流式音频处理
- 语音段检测: 自动检测和收集完整语音段
- 异步设计: 基于asyncio的高并发处理能力
- 低延迟: 优化的缓冲区和处理流程
- 多格式支持: WAV和MP3格式，16kHz采样率
- 多后端支持: ONNX和Silero VAD后端
- 简洁API: 符合现代Python异步编程习惯

快速开始:
    >>> import cascade
    >>> # 零配置使用
    >>> results = await cascade.process_audio_file("audio.wav")
    >>> print(f"检测到 {len(results)} 个结果")
    
    >>> # 流式处理
    >>> async with cascade.StreamProcessor() as processor:
    ...     async for result in processor.process_stream(audio_stream):
    ...         if result.is_speech_segment:
    ...             print(f"语音段: {result.segment.duration_ms:.0f}ms")
    ...         else:
    ...             print(f"单帧: {result.frame.timestamp_ms:.0f}ms")
"""

# 版本信息
__version__ = "0.2.0"
__author__ = "Xucailiang"
__license__ = "MIT"
__email__ = "xucailiang.ai@gmail.com"

import logging
import os
import platform
import sys

logger = logging.getLogger(__name__)

# 核心模块导入
# 流式处理器模块导入
from .stream import (
    AUDIO_FRAME_DURATION_MS,
    AUDIO_FRAME_SIZE,
    # 常量
    AUDIO_SAMPLE_RATE,
    # 数据类型
    AudioFrame,
    CascadeResult,
    Config,
    ProcessorStats,
    SpeechSegment,
    # 核心处理器
    StreamProcessor,
    create_default_config,
    create_stream_processor,
    process_audio_chunk,
    # 便捷函数
    process_audio_stream,
)
from .types import (
    # 数据类型
    AudioChunk,
    # 配置类型
    AudioConfig,
    # 枚举类型
    AudioFormat,
    AudioFormatError,
    BufferError,
    # 异常类型
    CascadeError,
    PerformanceMetrics,
    ProcessingMode,
    VADBackend,
    VADConfig,
    VADProcessingError,
    VADResult,
)


# 主要组件延迟导入
def __getattr__(name: str):
    """延迟导入主要组件"""
    if name == "AudioFormatProcessor":
        from .formats import AudioFormatProcessor
        return AudioFormatProcessor
    elif name == "FrameAlignedBuffer":
        from .buffer import FrameAlignedBuffer
        return FrameAlignedBuffer
    elif name == "process_audio_file":
        # 支持直接导入函数
        return process_audio_file
    else:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

# 公开API
__all__ = [
    # 版本信息
    "__version__",

    # 核心处理器
    "StreamProcessor",

    # 配置类型
    "Config",
    "AudioConfig",
    "VADConfig",

    # 数据类型
    "AudioChunk",
    "AudioFrame",
    "SpeechSegment",
    "CascadeResult",
    "VADResult",
    "PerformanceMetrics",
    "ProcessorStats",

    # 枚举类型
    "AudioFormat",
    "VADBackend",
    "ProcessingMode",

    # 辅助模块（高级用法）
    "AudioFormatProcessor",
    "FrameAlignedBuffer",

    # 异常类型
    "CascadeError",
    "AudioFormatError",
    "BufferError",
    "VADProcessingError",

    # 便捷函数
    "process_audio_stream",
    "process_audio_chunk",
    "process_audio_file",
    "create_default_config",
    "create_stream_processor",
    "create_processor",

    # 常量
    "AUDIO_SAMPLE_RATE",
    "AUDIO_FRAME_SIZE",
    "AUDIO_FRAME_DURATION_MS",
]

# 工厂函数
def create_processor(**kwargs) -> StreamProcessor:
    """
    创建流式处理器的工厂函数
    
    Args:
        **kwargs: 配置参数，覆盖默认值
            - vad_threshold: float = 0.5 (VAD检测阈值)
            - max_instances: int = 5 (最大并发实例数)
            - buffer_size_seconds: float = 2.0 (缓冲区大小)
            - sample_rate: int = 16000 (采样率)
            
    Returns:
        StreamProcessor: 配置好的处理器实例
        
    Example:
        # 默认配置
        processor = cascade.create_processor()
        
        # 自定义配置
        processor = cascade.create_processor(
            vad_threshold=0.7,
            max_instances=3
        )
    """
    # 创建配置，支持参数覆盖
    config = create_default_config(**kwargs)
    return StreamProcessor(config)

# 便捷函数
async def process_audio_file(file_path_or_data, **kwargs):
    """
    处理音频文件的便捷函数（异步迭代器）

    Args:
        file_path_or_data: 音频文件路径或音频数据（bytes）
        **kwargs: 配置参数

    Yields:
        CascadeResult: 处理结果

    Example:
        >>> async for result in cascade.process_audio_file("audio.wav"):
        ...     if result.is_speech_segment:
        ...         print(f"语音段: {result.segment.duration_ms:.0f}ms")
        ...     else:
        ...         print(f"单帧: {result.frame.timestamp_ms:.0f}ms")
    """
    try:

        processor = create_processor(**kwargs)

        # 使用processor的process_file方法进行处理
        async for result in processor.process_file(str(file_path_or_data)):
            yield result
    except Exception as e:
        raise AudioFormatError(f"音频处理失败: {e}") from e

# 兼容性检查
def check_compatibility() -> dict:
    """检查系统兼容性"""

    compatibility_info = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "architecture": platform.architecture(),
        "compatible": True,
        "warnings": [],
        "errors": []
    }

    # 平台检查
    supported_platforms = ["linux", "darwin", "win32"]
    if sys.platform not in supported_platforms:
        compatibility_info["warnings"].append(
            f"平台 {sys.platform} 可能不被完全支持"
        )

    return compatibility_info

# 调试信息
def get_debug_info() -> dict:
    """获取调试信息"""

    debug_info = {
        "version": __version__,
        "python_version": sys.version,
        "install_path": os.path.dirname(__file__),
        "available_backends": [],
        "dependencies": {}
    }

    # 检查可用后端
    try:
        import onnxruntime
        debug_info["available_backends"].append("onnx")
        debug_info["dependencies"]["onnxruntime"] = onnxruntime.__version__
    except ImportError:
        pass

    try:
        import torch
        debug_info["available_backends"].append("silero")
        debug_info["dependencies"]["torch"] = torch.__version__
    except ImportError:
        pass

    # 检查核心依赖
    try:
        import numpy
        debug_info["dependencies"]["numpy"] = numpy.__version__
    except ImportError:
        debug_info["dependencies"]["numpy"] = "未安装"

    try:
        import pydantic
        debug_info["dependencies"]["pydantic"] = pydantic.__version__
    except ImportError:
        debug_info["dependencies"]["pydantic"] = "未安装"

    return debug_info
