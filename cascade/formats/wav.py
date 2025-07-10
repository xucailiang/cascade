"""
WAV 格式音频流处理模块

本模块提供对 WAV 格式音频文件的读写支持。
"""

import io
import wave
from typing import Tuple

import numpy as np

from cascade.formats.base import FormatConverter, registry
from cascade.types.audio import AudioConfig, AudioFormat


def read_wav_bytes(wav_bytes: bytes) -> Tuple[np.ndarray, AudioConfig]:
    """
    从WAV字节数据中读取音频。

    Args:
        wav_bytes: WAV文件的字节内容。

    Returns:
        一个元组，包含numpy数组格式的音频数据和音频配置。
    """
    with wave.open(io.BytesIO(wav_bytes), 'rb') as wf:
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        n_frames = wf.getnframes()
        raw_data = wf.readframes(n_frames)

        dtype_map = {1: np.int8, 2: np.int16, 4: np.int32}
        if sample_width not in dtype_map:
            raise ValueError(f"Unsupported sample width: {sample_width}")

        audio_data = np.frombuffer(raw_data, dtype=dtype_map[sample_width])

        if channels > 1:
            audio_data = audio_data.reshape(-1, channels)

    config = AudioConfig(
        format=AudioFormat.WAV,
        sample_rate=sample_rate,
        channels=channels,
        bit_depth=sample_width * 8
    )
    return audio_data, config


class WavFormatConverter(FormatConverter):
    """
    WAV 格式转换器
    
    提供 WAV 格式与内部格式之间的转换。
    """

    def to_internal(self, audio_bytes: bytes) -> Tuple[np.ndarray, AudioConfig]:
        """
        将WAV字节转换为内部处理格式 (float32)。

        Args:
            audio_bytes: WAV文件的字节内容。

        Returns:
            一个元组，包含numpy数组格式的音频数据（float32）和音频配置。
        """
        audio_data, config = read_wav_bytes(audio_bytes)

        # 转换为内部格式 (float32)
        internal_data, _, _ = self.convert_to_internal(audio_data, config)
        return internal_data, config

    def convert_to_internal(self, audio_data: np.ndarray, config: AudioConfig) -> tuple[np.ndarray, int, int]:
        """
        将 WAV 音频数据转换为内部格式 (float32, [-1.0, 1.0])。
        
        Args:
            audio_data: Numpy数组格式的音频数据。
            config: 音频配置。
            
        Returns:
            一个元组 (转换后的 float32 音频数据, 采样率, 通道数)。
        """
        if config.format != AudioFormat.WAV:
            raise ValueError(f"不支持的音频格式: {config.format}，预期为 WAV")

        # 确保数据类型为 float32
        if audio_data.dtype != np.float32:
            # 如果是 int16，归一化到 [-1.0, 1.0] 范围
            if audio_data.dtype == np.int16:
                audio_data = audio_data.astype(np.float32) / 32767.0
            # 如果是 int32，归一化到 [-1.0, 1.0] 范围
            elif audio_data.dtype == np.int32:
                audio_data = audio_data.astype(np.float32) / 2147483647.0
            else:
                audio_data = audio_data.astype(np.float32)

        return audio_data, config.sample_rate, config.channels

    def convert_from_internal(self, audio_data: np.ndarray, config: AudioConfig) -> np.ndarray:
        """
        将内部格式 (float32) 转换为 WAV 格式的目标数据类型。
        
        Args:
            audio_data: 内部格式的音频数据。
            config: 目标音频格式配置。
            
        Returns:
            转换后的音频数据。
        """
        if config.format != AudioFormat.WAV:
            raise ValueError(f"不支持的音频格式: {config.format}，预期为 WAV")

        bit_depth = config.bit_depth
        # 从 float32 转换为目标数据类型
        if bit_depth == 16:
            # 将 [-1.0, 1.0] 范围转换为 int16
            return (audio_data * 32767.0).astype(np.int16)
        elif bit_depth == 32:
            # 将 [-1.0, 1.0] 范围转换为 int32
            return (audio_data * 2147483647.0).astype(np.int32)
        else:
            # 默认为16位
            return (audio_data * 32767.0).astype(np.int16)


# 注册 WAV 格式转换器
registry.register(AudioFormat.WAV, WavFormatConverter)
