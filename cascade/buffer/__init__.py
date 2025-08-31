"""
Cascade 音频缓冲模块

本模块负责高效地管理和操作音频数据流。
提供抽象基类和针对1:1:1架构优化的帧对齐缓冲区。

主要组件：
- AudioBuffer: 音频缓冲区抽象基类
- FrameAlignedBuffer: 512样本帧对齐缓冲区（1:1:1架构专用）
"""

from .base import AudioBuffer
from .frame_aligned_buffer import FrameAlignedBuffer

__all__ = [
    "AudioBuffer",
    "FrameAlignedBuffer"
]
