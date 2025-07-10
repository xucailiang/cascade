"""
Cascade 音频格式处理模块

本模块通过导入各个具体的格式实现来自动注册格式转换器。
每个格式模块（如 wav.py）负责定义其自己的转换器并调用 registry.register()。
"""

from .base import AudioFormat, FormatConverter, registry
from .utils import get_audio_format

# 导入具体的格式模块，这将触发其中的注册逻辑
from . import wav
from . import pcma


def get_format_converter(format_type: AudioFormat) -> FormatConverter:
    """
    获取指定音频格式的转换器实例

    Args:
        format_type: 音频格式类型

    Returns:
        格式转换器实例

    Raises:
        ValueError: 当没有为所请求的格式注册处理器时
    """
    return registry.get_processor(format_type)


__all__ = [
    "AudioFormat",
    "FormatConverter",
    "get_format_converter",
    "get_audio_format",
]
