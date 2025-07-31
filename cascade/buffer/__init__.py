"""
Cascade 音频缓冲模块

本模块负责高效地管理和操作音频数据流。
提供抽象基类和高性能的环形缓冲区实现。

主要组件：
- AudioBuffer: 音频缓冲区抽象基类
- AudioRingBuffer: 高性能环形缓冲区实现
- RingBuffer: AudioRingBuffer的便利别名
"""

from .base import AudioBuffer
from .ring_buffer import AudioRingBuffer, RingBuffer

__all__ = [
    "AudioBuffer",
    "AudioRingBuffer",
    "RingBuffer",
]
