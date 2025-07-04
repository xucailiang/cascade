"""
Cascade: 高性能异步并行VAD处理库

Cascade是一个专为语音活动检测(VAD)设计的高性能、低延迟音频流处理库。
通过并行处理技术和优化的架构设计，显著降低VAD处理延迟，同时保证检测结果的准确性。

核心特性:
- 并行处理: 多线程VAD实例并行处理音频块
- 重叠处理: 通过块间重叠区域解决边界问题
- 异步设计: 基于asyncio的高并发处理能力
- 低延迟: 优化的缓冲区和处理流程
- 多格式支持: WAV和PCMA格式，16kHz和8kHz采样率
- 多后端支持: ONNX和VLLM两种VAD后端
- 零拷贝设计: 最小化内存复制，提高处理效率

快速开始:
    >>> import cascade
    >>> # 零配置使用
    >>> results = await cascade.process_audio_file("audio.wav")
    >>> print(f"检测到 {len(results)} 个语音段")
    
    >>> # 高级配置
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
__author__ = "Cascade Team"
__license__ = "Apache-2.0"
__email__ = "team@cascade-vad.org"

# 核心模块导入
from .types import (
    # 配置类型
    AudioConfig,
    VADConfig,
    
    # 数据类型
    AudioChunk,
    VADResult,
    PerformanceMetrics,
    
    # 枚举类型
    AudioFormat,
    VADBackend,
    ProcessingMode,
    
    # 异常类型
    CascadeError,
    AudioFormatError,
    BufferError,
    VADProcessingError,
)

# 主要组件导入 (延迟导入以避免循环依赖)
def __getattr__(name: str):
    """延迟导入主要组件"""
    if name == "VADProcessor":
        from .processor import VADProcessor
        return VADProcessor
    elif name == "AudioFormatProcessor":
        from .formats import AudioFormatProcessor
        return AudioFormatProcessor
    elif name == "AudioRingBuffer":
        from .buffer import AudioRingBuffer
        return AudioRingBuffer
    elif name == "ONNXVADBackend":
        from .backends.onnx import ONNXVADBackend
        return ONNXVADBackend
    elif name == "VLLMVADBackend":
        from .backends.vllm import VLLMVADBackend
        return VLLMVADBackend
    else:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

# 公开API
__all__ = [
    # 版本信息
    "__version__",
    
    # 核心处理器
    "VADProcessor",
    
    # 配置类型
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
    
    # 后端实现
    "ONNXVADBackend", 
    "VLLMVADBackend",
    
    # 辅助模块（高级用法）
    "AudioFormatProcessor", 
    "AudioRingBuffer",
    
    # 异常类型
    "CascadeError",
    "AudioFormatError", 
    "BufferError", 
    "VADProcessingError",
    
    # 便捷函数
    "create_vad_processor",
    "process_audio_file",
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
    from .processor import VADProcessor
    
    vad_config = VADConfig(backend=backend_type, **kwargs)
    audio_config = AudioConfig()
    return VADProcessor(vad_config, audio_config)

async def process_audio_file(file_path: str, **kwargs) -> list:
    """
    处理音频文件的便捷函数
    
    Args:
        file_path: 音频文件路径
        **kwargs: VAD配置参数
        
    Returns:
        VAD结果列表
        
    Example:
        >>> results = await cascade.process_audio_file(
        ...     "audio.wav", 
        ...     threshold=0.7,
        ...     workers=8
        ... )
        >>> print(f"检测到 {len(results)} 个语音段")
    """
    processor = create_vad_processor(**kwargs)
    
    try:
        results = []
        async for result in processor.process_file(file_path):
            results.append(result)
        return results
    finally:
        await processor.close()

# 兼容性检查
def check_compatibility() -> dict:
    """检查系统兼容性"""
    import sys
    import platform
    
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
    import sys
    import os
    
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