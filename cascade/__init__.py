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
__version__ = "0.1.1"
__author__ = "Xucailiang"
__license__ = "MIT"
__email__ = "xucailiang.ai@gmail.com"

import os

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

    # 常量
    "AUDIO_SAMPLE_RATE",
    "AUDIO_FRAME_SIZE",
    "AUDIO_FRAME_DURATION_MS",
]

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
    
    from .stream.instance import CascadeInstance

    # 创建配置
    config = create_default_config(**kwargs)

    try:
        # 判断输入类型
        if isinstance(file_path_or_data, (str, os.PathLike)):
            # 文件路径
            if not os.path.exists(file_path_or_data):
                raise AudioFormatError(f"音频文件不存在: {file_path_or_data}")

            # 1. 读取音频文件
            audio_data = _read_audio_file(str(file_path_or_data), target_sample_rate=16000)

            # 2. 生成音频帧
            audio_frames = list(_generate_audio_frames(audio_data))

            print(f"音频文件处理开始: {file_path_or_data}")
            print(f"总时长: {len(audio_data) / 16000:.2f}秒")
            print(f"处理 {len(audio_frames)} 个音频帧")

        elif isinstance(file_path_or_data, bytes):
            # 直接的音频数据
            audio_frames = list(_generate_audio_frames_from_bytes(file_path_or_data))
            print(f"音频数据处理开始: {len(file_path_or_data)} 字节")
            print(f"处理 {len(audio_frames)} 个音频帧")
        else:
            raise AudioFormatError(f"不支持的输入类型: {type(file_path_or_data)}")

        # 3. 使用CascadeInstance处理
        instance = CascadeInstance("file_processor", config)

        # 初始化VAD后端
        await instance.vad_backend.initialize()

        # 4. 逐帧处理并yield结果
        total_results = 0
        speech_segments = 0
        single_frames = 0

        for frame_data in audio_frames:
            frame_results = instance.process_audio_chunk(frame_data)
            for result in frame_results:
                total_results += 1
                if result.is_speech_segment:
                    speech_segments += 1
                else:
                    single_frames += 1
                yield result

        print("音频处理完成")
        print(f"总结果: {total_results} 个")
        print(f"语音段: {speech_segments} 个")
        print(f"单帧: {single_frames} 个")

    except Exception as e:
        raise AudioFormatError(f"音频处理失败: {e}")


def _read_audio_file(file_path: str, target_sample_rate: int = 16000):
    """
    读取音频文件
    
    Args:
        file_path: 音频文件路径
        target_sample_rate: 目标采样率
        
    Returns:
        音频数据数组
    """
    try:
        # 尝试使用silero-vad的read_audio函数
        try:
            from silero_vad import read_audio
            audio_data = read_audio(file_path, sampling_rate=target_sample_rate)
            return audio_data
        except ImportError:
            # 如果silero-vad不可用，使用基础的音频读取
            import wave

            import numpy as np

            with wave.open(file_path, 'rb') as wav_file:
                frames = wav_file.readframes(-1)
                audio_data = np.frombuffer(frames, dtype=np.int16)

                # 转换为float32并归一化
                # 检查是否是PyTorch Tensor
                if hasattr(audio_data, 'numpy') and callable(getattr(audio_data, 'numpy', None)):
                    # PyTorch Tensor
                    audio_data = audio_data.numpy()
                elif hasattr(audio_data, 'detach') and callable(getattr(audio_data, 'detach', None)):
                    # PyTorch Tensor with gradients
                    audio_data = audio_data.detach().numpy()

                audio_data = audio_data.astype(np.float32) / 32768.0

                # 简单的采样率转换（如果需要）
                source_rate = wav_file.getframerate()
                if source_rate != target_sample_rate:
                    # 简单的线性插值重采样
                    ratio = target_sample_rate / source_rate
                    new_length = int(len(audio_data) * ratio)
                    old_indices = np.linspace(0, len(audio_data) - 1, new_length)
                    audio_data = np.interp(old_indices, np.arange(len(audio_data)), audio_data)

                # 确保返回numpy数组
                if hasattr(audio_data, 'numpy') and callable(getattr(audio_data, 'numpy', None)):
                    audio_data = audio_data.numpy()
                elif hasattr(audio_data, 'detach') and callable(getattr(audio_data, 'detach', None)):
                    audio_data = audio_data.detach().numpy()

                return audio_data.astype(np.float32)

    except Exception as e:
        raise AudioFormatError(f"音频文件读取失败: {e}")


def _generate_audio_frames(audio_data, frame_size: int = 512) -> list:
    """
    生成512样本的音频帧
    
    Args:
        audio_data: 音频数据
        frame_size: 帧大小（样本数）
        
    Returns:
        音频帧列表（bytes格式）
    """
    frames = []

    # 按512样本分块
    for i in range(0, len(audio_data), frame_size):
        frame = audio_data[i:i + frame_size]

        # 如果最后一帧不足512样本，跳过（符合silero-vad要求）
        if len(frame) < frame_size:
            break

        # 转换为int16格式的bytes
        import numpy as np

        # 如果是PyTorch Tensor，先转换为numpy数组
        if hasattr(frame, 'numpy') and callable(getattr(frame, 'numpy', None)):
            frame = frame.numpy()
        elif hasattr(frame, 'detach') and callable(getattr(frame, 'detach', None)):
            frame = frame.detach().numpy()

        frame_int16 = (frame * 32767).astype(np.int16)
        frame_bytes = frame_int16.tobytes()
        frames.append(frame_bytes)

    return frames


def _generate_audio_frames_from_bytes(audio_bytes: bytes, frame_size: int = 512) -> list:
    """
    从音频字节数据生成512样本的音频帧
    
    Args:
        audio_bytes: 音频字节数据（int16格式）
        frame_size: 帧大小（样本数）
        
    Returns:
        音频帧列表（bytes格式）
    """
    import numpy as np

    # 将bytes转换为int16数组
    audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)

    frames = []

    # 按512样本分块
    for i in range(0, len(audio_int16), frame_size):
        frame = audio_int16[i:i + frame_size]

        # 如果最后一帧不足512样本，跳过（符合silero-vad要求）
        if len(frame) < frame_size:
            break

        # 转换为bytes
        frame_bytes = frame.tobytes()
        frames.append(frame_bytes)

    return frames

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
    if sys.version_info < (3, 8):
        compatibility_info["compatible"] = False
        compatibility_info["errors"].append(
            f"Python版本过低: {sys.version_info.major}.{sys.version_info.minor}, "
            "需要Python 3.8或更高版本"
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
