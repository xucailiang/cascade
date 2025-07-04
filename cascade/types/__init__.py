"""
Cascade 核心类型系统

本模块定义了 `cascade` 项目所有模块共享的核心数据结构和类型。
所有IO边界和模块间通信都应使用此处定义的类型。
"""

from .audio import AudioChunk, AudioConfig
from .vad import VADConfig, VADResult
from .generic import Status
from .config import ONNXConfig, VLLMConfig

__all__ = [
    "AudioChunk",
    "AudioConfig",
    "VADConfig",
    "VADResult",
    "Status",
    "ONNXConfig",
    "VLLMConfig",
]