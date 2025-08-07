"""
Cascade: 高性能低延迟VAD处理库

Cascade是一个专为语音活动检测(VAD)设计的高性能、低延迟音频流处理库。
采用极简的1:1:1绑定架构，确保有状态VAD模型的上下文连续性，同时提供简洁的API。

核心特性:
- 1:1:1绑定架构: 1个Cascade实例 = 1个VAD Backend = 1个处理会话
- 状态连续性: 确保silero-vad等有状态模型的上下文连续性
- 极简配置: 只需配置3个核心参数(sample_rate, vad_backend, vad_threshold)
- 用户控制: 用户完全控制音频块大小，系统直接处理
- 完整输出: 返回原始音频块+VAD检测结果的完整输出
- 异步设计: 基于asyncio的高并发处理能力
- 低延迟: 优化的缓冲区和处理流程，目标<5ms P99延迟
- 多后端支持: Silero VAD和ONNX两种后端

快速开始 (新架构 - 推荐):
    >>> import cascade
    >>> # 极简使用 - 只需3个参数
    >>> async with cascade.Cascade(cascade.CascadeConfig(
    ...     sample_rate=16000,
    ...     vad_backend="silero"
    ... )) as detector:
    ...     async for result in detector.process_audio_stream(audio_stream):
    ...         if result.is_speech:
    ...             print(f"检测到语音: {result.speech_probability:.2f}")
    
    >>> # 便捷函数
    >>> detector = await cascade.create_cascade(sample_rate=16000, vad_backend="silero")
    >>> result = await detector.process_audio_chunk(audio_data)
    >>> print(f"语音概率: {result.speech_probability}")

兼容性API (旧架构):
    >>> # 继续支持原有VADProcessor API
    >>> processor = cascade.VADProcessor(
    ...     vad_config=cascade.VADConfig(workers=8, threshold=0.7),
    ...     audio_config=cascade.AudioConfig(sample_rate=16000)
    ... )
    >>> async for result in processor.process_stream(audio_stream):
    ...     if result.is_speech:
    ...         print(f"语音: {result.start_ms}ms - {result.end_ms}ms")
"""

# 版本信息
__version__ = "0.1.0"
__author__ = "Xucailiang"
__license__ = "MIT"
__email__ = "xucailiang.ai@gmail.com"

# 核心模块导入
from .types import (
    # 数据类型
    AudioChunk,
    # 配置类型
    AudioConfig,
    VADConfig,
    # 新架构配置类型
    CascadeConfig,
    CascadeVADResult,
    # 枚举类型
    AudioFormat,
    VADBackend,
    ProcessingMode,
    # 结果类型
    VADResult,
    PerformanceMetrics,
    # 异常类型
    CascadeError,
    AudioFormatError,
    BufferError,
    VADProcessingError,
)


# 主要组件导入 (延迟导入以避免循环依赖)
def __getattr__(name: str):
    """延迟导入主要组件"""
    if name == "Cascade":
        from .cascade import Cascade
        return Cascade
    elif name == "create_cascade":
        from .cascade import create_cascade
        return create_cascade
    elif name == "VADProcessor":
        from .processor import VADProcessor
        return VADProcessor
    elif name == "VADProcessorConfig":
        from .processor import VADProcessorConfig
        return VADProcessorConfig
    elif name == "AudioFormatProcessor":
        from .formats import AudioFormatProcessor
        return AudioFormatProcessor
    elif name == "AudioRingBuffer":
        from .buffer import AudioRingBuffer
        return AudioRingBuffer
    elif name == "create_vad_processor":
        # 支持直接导入函数
        return create_vad_processor
    elif name == "process_audio_file":
        # 支持直接导入函数
        return process_audio_file
    else:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

# 公开API
__all__ = [
    # 版本信息
    "__version__",

    # 核心处理器 (新架构 - 推荐)
    "Cascade",
    "create_cascade",

    # 配置类型 (新架构)
    "CascadeConfig",
    "CascadeVADResult",

    # 兼容性处理器 (旧架构)
    "VADProcessor",
    "VADProcessorConfig",

    # 共享配置类型
    "AudioConfig",
    "VADConfig",

    # 数据类型
    "AudioChunk",
    "VADResult",
    "PerformanceMetrics",

    # 枚举类型
    "AudioFormat",
    "VADBackend",
    "ProcessingMode",

    # 辅助模块（高级用法）
    "AudioFormatProcessor",
    "AudioRingBuffer",

    # 异常类型
    "CascadeError",
    "AudioFormatError",
    "BufferError",
    "VADProcessingError",

    # 便捷函数
    "create_vad_processor",  # 旧架构
    "process_audio_file",    # 旧架构
]

# 便捷工厂函数
def create_vad_processor(backend_type: str = "onnx", **kwargs) -> "VADProcessor":
    """
    创建VAD处理器的便捷函数
    
    Args:
        backend_type: VAD后端类型 ("onnx" 或 "vllm")
        **kwargs: 其他配置参数
        
    Returns:
        配置好的VAD处理器实例
        
    Example:
        >>> processor = cascade.create_vad_processor(
        ...     backend_type="onnx",
        ...     workers=4,
        ...     threshold=0.5
        ... )
    """
    from .processor import VADProcessor, VADProcessorConfig
    
    # 创建VAD配置，使用VADBackend枚举
    vad_config = VADConfig(backend=VADBackend(backend_type), **kwargs)
    audio_config = AudioConfig()
    
    # 创建处理器配置
    processor_config = VADProcessorConfig(
        vad_config=vad_config,
        audio_config=audio_config
    )
    
    return VADProcessor(processor_config)

async def process_audio_file(file_path: str, backend_type: str = "onnx", **kwargs) -> list:
    """
    处理音频文件的便捷函数
    
    Args:
        file_path: 音频文件路径
        backend_type: VAD后端类型 ("onnx" 或 "vllm")
        **kwargs: VAD配置参数
        
    Returns:
        VAD结果列表
        
    Example:
        >>> results = await cascade.process_audio_file(
        ...     "audio.wav",
        ...     backend_type="onnx",
        ...     threshold=0.7,
        ...     workers=8
        ... )
        >>> print(f"检测到 {len(results)} 个语音段")
    """
    processor = create_vad_processor(backend_type=backend_type, **kwargs)

    try:
        results = []
        # 注意：VADProcessor可能有process_stream方法而不是process_file
        # 这里需要根据实际实现调整
        async for result in processor.process_stream([]):  # 需要实际的音频数据
            results.append(result)
        return results
    finally:
        await processor.close()

# 兼容性检查
def check_compatibility() -> dict:
    """检查系统兼容性"""
    import platform
    import sys

    compatibility_info = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "architecture": platform.architecture(),
        "compatible": True,
        "warnings": [],
        "errors": []
    }

    # Python版本检查
    if sys.version_info < (3, 11):
        compatibility_info["compatible"] = False
        compatibility_info["errors"].append(
            f"Python版本过低: {sys.version_info.major}.{sys.version_info.minor}, "
            "需要Python 3.11或更高版本"
        )

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
    import os
    import sys

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
        import vllm
        debug_info["available_backends"].append("vllm")
        debug_info["dependencies"]["vllm"] = vllm.__version__
    except ImportError:
        pass

    # 检查核心依赖
    try:
        import numpy
        debug_info["dependencies"]["numpy"] = numpy.__version__
    except ImportError:
        debug_info["dependencies"]["numpy"] = "未安装"

    try:
        import scipy
        debug_info["dependencies"]["scipy"] = scipy.__version__
    except ImportError:
        debug_info["dependencies"]["scipy"] = "未安装"

    try:
        import pydantic
        debug_info["dependencies"]["pydantic"] = pydantic.__version__
    except ImportError:
        debug_info["dependencies"]["pydantic"] = "未安装"

    return debug_info
