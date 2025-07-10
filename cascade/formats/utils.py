"""
音频格式处理工具函数
"""

import os
from typing import Optional

from cascade.types.audio import AudioFormat


def get_audio_format(file_path: str) -> Optional[AudioFormat]:
    """
    根据文件扩展名推断音频格式

    Args:
        file_path: 文件路径

    Returns:
        音频格式枚举成员，如果无法推断则返回None
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".wav":
        return AudioFormat.WAV
    elif ext in [".pcma", ".alaw"]:
        return AudioFormat.PCMA
    return None