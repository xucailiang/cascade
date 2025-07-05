"""
Cascade 核心类型系统

本模块定义了 `cascade` 项目所有模块共享的核心数据结构和类型。
所有IO边界和模块间通信都应使用此处定义的类型。
"""

# 音频相关类型
from .audio import AudioChunk, AudioConfig, AudioFormat, AudioMetadata

# 后端配置类型
from .config import BackendConfig, ONNXConfig, VLLMConfig

# 错误类型
from .errors import (
    AudioFormatError,
    BackendUnavailableError,
    BufferError,
    BufferFullError,
    ConfigurationError,
    InferenceError,
    InsufficientDataError,
    ModelLoadError,
    PreVADError,
    TimeoutError,
    VADProcessingError,
)

# 通用类型
from .generic import (
    BufferStatus,
    BufferStrategy,
    ErrorCode,
    ErrorInfo,
    ErrorSeverity,
    LogLevel,
    PerformanceMetrics,
    Status,
    SystemStatus,
)

# VAD相关类型
from .vad import (
    OptimizationLevel,
    ProcessingMode,
    VADBackend,
    VADConfig,
    VADResult,
    VADSegment,
)

# 为了向后兼容性，将PreVADError作为CascadeError导出
CascadeError = PreVADError

__all__ = [
    # 音频相关类型
    "AudioChunk", "AudioConfig", "AudioFormat", "AudioMetadata",

    # VAD相关类型
    "VADConfig", "VADResult", "VADSegment", "VADBackend",
    "ProcessingMode", "OptimizationLevel",

    # 通用类型
    "Status", "PerformanceMetrics", "SystemStatus", "BufferStatus",
    "LogLevel", "BufferStrategy", "ErrorCode", "ErrorSeverity", "ErrorInfo",

    # 后端配置类型
    "BackendConfig", "ONNXConfig", "VLLMConfig",

    # 错误类型
    "PreVADError", "CascadeError", "AudioFormatError", "BufferError", "BufferFullError",
    "InsufficientDataError", "VADProcessingError", "ModelLoadError",
    "BackendUnavailableError", "InferenceError", "ConfigurationError", "TimeoutError",
]
