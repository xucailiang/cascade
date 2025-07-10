"""
Cascade VAD 处理器模块

本模块是 `cascade` 库的核心，负责编排整个VAD处理流水线。
"""

from .vad_processor import VADProcessor
from ..types.config import VADProcessorConfig

__all__ = [
    "VADProcessor",
    "VADProcessorConfig",
]
