"""
PCMA (A-law) 格式音频转换器
"""

import numpy as np

from cascade.formats.base import FormatConverter, registry
from cascade.types.audio import AudioConfig, AudioFormat


class PcmaConverter(FormatConverter):
    """PCMA格式转换器"""

    def __init__(self):
        self._alaw_to_linear_table = self._build_alaw_to_linear_table()

    @staticmethod
    def _build_alaw_to_linear_table() -> np.ndarray:
        """构建A-law到线性PCM的查找表"""
        table = np.zeros(256, dtype=np.int16)
        for i in range(256):
            ex_mask = 0b01010101
            a = i ^ ex_mask
            t = (a & 0x0F) << 4
            seg = (a & 0x70) >> 4
            if seg == 0:
                t += 8
            elif seg > 1:
                t += 0x108
                t <<= seg - 1
            if (a & 0x80) == 0:
                t = -t
            table[i] = t
        return table

    def convert_to_internal(self, audio_data: np.ndarray, config: AudioConfig) -> tuple[np.ndarray, int, int]:
        """将PCMA音频转换为内部处理格式 (float32)"""
        pcm_data = self._alaw_to_linear_table[audio_data.astype(np.uint8)]
        float_data = pcm_data.astype(np.float32) / 32767.0
        return float_data, config.sample_rate, config.channels

    def convert_from_internal(self, audio_data: np.ndarray, config: AudioConfig) -> np.ndarray:
        """将内部格式转换为PCMA (uint8)"""
        # 此处省略线性到A-law的转换实现，因为它在此库中不常用
        raise NotImplementedError("线性PCM到PCMA的转换尚未实现")


# 注册转换器
registry.register(AudioFormat.PCMA, PcmaConverter)
