"""
音频格式处理模块

提供音频格式验证、转换和优化功能，包括：
- 音频格式验证
- 格式转换 (PCMA到float32, int16到float32等)
- 采样率转换
- 块大小计算
- 内存优化
- 缓存策略

设计原则：
- 高性能实现
- 零拷贝优化
- 智能缓存策略
- 完整的错误处理
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from .._internal.utils import align_to_cache_line, ensure_contiguous, measure_time
from ..types import AudioConfig, AudioFormat, AudioFormatError

# 创建logger
logger = logging.getLogger(__name__)

# 预计算块大小映射表
# 这些常用的块大小预先计算好，避免运行时计算
CHUNK_SIZES = {
    8000: {  # 8kHz采样率
        16: 128,     # 16ms -> 128样本
        32: 256,     # 32ms -> 256样本
        100: 800,    # 100ms -> 800样本
        250: 2000,   # 250ms -> 2000样本
        500: 4000,   # 500ms -> 4000样本
        1000: 8000   # 1000ms -> 8000样本
    },
    16000: { # 16kHz采样率
        16: 256,     # 16ms -> 256样本
        32: 512,     # 32ms -> 512样本
        100: 1600,   # 100ms -> 1600样本
        250: 4000,   # 250ms -> 4000样本
        500: 8000,   # 500ms -> 8000样本
        1000: 16000  # 1000ms -> 16000样本
    },
    22050: { # 22.05kHz采样率
        16: 353,     # 16ms -> 353样本
        32: 705,     # 32ms -> 705样本
        100: 2205,   # 100ms -> 2205样本
        250: 5512,   # 250ms -> 5512样本
        500: 11025,  # 500ms -> 11025样本
        1000: 22050  # 1000ms -> 22050样本
    },
    44100: { # 44.1kHz采样率
        16: 705,     # 16ms -> 705样本
        32: 1411,    # 32ms -> 1411样本
        100: 4410,   # 100ms -> 4410样本
        250: 11025,  # 250ms -> 11025样本
        500: 22050,  # 500ms -> 22050样本
        1000: 44100  # 1000ms -> 44100样本
    },
    48000: { # 48kHz采样率
        16: 768,     # 16ms -> 768样本
        32: 1536,    # 32ms -> 1536样本
        100: 4800,   # 100ms -> 4800样本
        250: 12000,  # 250ms -> 12000样本
        500: 24000,  # 500ms -> 24000样本
        1000: 48000  # 1000ms -> 48000样本
    }
}

# PCMA解码查找表（G.711 A-law）
# 预计算PCMA到线性PCM的转换表，提高转换性能
_PCMA_DECODE_TABLE = None

def _build_pcma_decode_table() -> np.ndarray:
    """构建PCMA解码查找表"""
    global _PCMA_DECODE_TABLE
    if _PCMA_DECODE_TABLE is not None:
        return _PCMA_DECODE_TABLE

    # 创建256个条目的查找表
    table = np.zeros(256, dtype=np.float32)

    for i in range(256):
        # A-law解码算法
        sign = 0 if (i & 0x80) == 0 else 1
        exponent = (i & 0x70) >> 4
        mantissa = i & 0x0F

        if exponent == 0:
            linear = mantissa << 4
        else:
            linear = ((mantissa | 0x10) << (exponent + 3))

        if sign:
            linear = -linear

        # 归一化到[-1, 1]范围
        table[i] = linear / 32768.0

    _PCMA_DECODE_TABLE = table
    return table

class AudioFormatProcessor:
    """
    音频格式处理器 - 纯功能实现
    
    提供音频格式验证、转换和优化功能。
    设计为无状态服务，可以安全地在多线程环境中使用。
    """

    def __init__(self, config: AudioConfig):
        """
        初始化格式处理器
        
        Args:
            config: 音频配置对象
        """
        self.config = config

        # 初始化缓存
        self._chunk_size_cache: dict[tuple[int, int], int] = {}
        self._conversion_cache: dict[str, np.ndarray] = {}

        # 确保PCMA解码表已构建
        _build_pcma_decode_table()

    def validate_format(self,
                       format_type: AudioFormat,
                       sample_rate: int,
                       channels: int) -> bool:
        """
        验证音频格式是否支持
        
        Args:
            format_type: 音频格式
            sample_rate: 采样率
            channels: 通道数
            
        Returns:
            True 如果格式支持，False 否则
            
        Raises:
            AudioFormatError: 如果格式参数无效
        """
        try:
            # 验证格式类型
            if not isinstance(format_type, AudioFormat):
                return False

            # 验证采样率
            supported_rates = [8000, 16000, 22050, 44100, 48000]
            if sample_rate not in supported_rates:
                return False

            # 验证通道数
            if channels not in [1, 2]:
                return False

            # 验证格式特定限制
            if format_type == AudioFormat.PCMA:
                # PCMA格式仅支持8kHz单声道
                if sample_rate != 8000 or channels != 1:
                    return False

            return True

        except Exception as e:
            raise AudioFormatError(
                f"格式验证失败: {e}",
                {
                    "format": format_type.value if isinstance(format_type, AudioFormat) else str(format_type),
                    "sample_rate": sample_rate,
                    "channels": channels
                }
            )

    @measure_time
    def convert_to_internal_format(self,
                                 audio_data: np.ndarray,
                                 format_type: AudioFormat,
                                 sample_rate: int) -> np.ndarray:
        """
        转换为内部处理格式(float32)
        
        Args:
            audio_data: 输入音频数据
            format_type: 音频格式
            sample_rate: 采样率
            
        Returns:
            转换后的float32音频数据
            
        Raises:
            AudioFormatError: 如果转换失败
        """
        try:
            # 验证输入
            if not isinstance(audio_data, np.ndarray):
                raise ValueError("音频数据必须是numpy数组")

            if audio_data.size == 0:
                raise ValueError("音频数据不能为空")

            # 确保数组是连续的
            audio_data = ensure_contiguous(audio_data)

            # 格式转换
            if format_type == AudioFormat.PCMA:
                converted_data = self._pcma_to_float32(audio_data)
            elif audio_data.dtype == np.int16:
                converted_data = self._int16_to_float32(audio_data)
            elif audio_data.dtype == np.int32:
                converted_data = self._int32_to_float32(audio_data)
            elif audio_data.dtype == np.float32:
                converted_data = audio_data
            elif audio_data.dtype == np.float64:
                converted_data = audio_data.astype(np.float32)
            elif audio_data.dtype in [np.int8, np.uint8]:
                # 处理8位数据类型
                converted_data = audio_data.astype(np.float32) / 127.0
            else:
                raise ValueError(f"不支持的数据类型: {audio_data.dtype}")

            # 采样率转换
            if sample_rate != self.config.sample_rate:
                converted_data = self._resample_audio(
                    converted_data, sample_rate, self.config.sample_rate
                )

            # 内存对齐优化
            try:
                converted_data = align_to_cache_line(converted_data)
            except Exception:
                # 如果对齐失败，继续使用原始数据
                pass

            return converted_data

        except Exception as e:
            format_value = format_type.value if hasattr(format_type, 'value') else str(format_type)
            raise AudioFormatError(
                f"格式转换失败: {e}",
                {
                    "format": format_value,
                    "sample_rate": sample_rate,
                    "data_shape": audio_data.shape if hasattr(audio_data, 'shape') else None,
                    "data_dtype": str(audio_data.dtype) if hasattr(audio_data, 'dtype') else None
                }
            )

    def calculate_chunk_size(self, duration_ms: int, sample_rate: int) -> int:
        """
        计算指定时长的块大小
        
        Args:
            duration_ms: 时长（毫秒）
            sample_rate: 采样率
            
        Returns:
            块大小（样本数）
        """
        # 1. 尝试从缓存获取
        cache_key = (duration_ms, sample_rate)
        if cache_key in self._chunk_size_cache:
            return self._chunk_size_cache[cache_key]

        # 2. 尝试从预计算表获取
        if sample_rate in CHUNK_SIZES and duration_ms in CHUNK_SIZES[sample_rate]:
            result = CHUNK_SIZES[sample_rate][duration_ms]
            self._chunk_size_cache[cache_key] = result
            return result

        # 3. 动态计算
        result = int(duration_ms * sample_rate / 1000)
        self._chunk_size_cache[cache_key] = result
        return result

    def calculate_overlap_size(self, overlap_ms: int, sample_rate: int) -> int:
        """
        计算重叠区域大小
        
        Args:
            overlap_ms: 重叠时长（毫秒）
            sample_rate: 采样率
            
        Returns:
            重叠大小（样本数）
        """
        return int(overlap_ms * sample_rate / 1000)

    def _pcma_to_float32(self, pcma_data: np.ndarray) -> np.ndarray:
        """
        PCMA格式转换为float32
        
        使用查找表进行高效转换。
        
        Args:
            pcma_data: PCMA格式数据
            
        Returns:
            float32格式数据
        """
        # 如果不是uint8类型，尝试转换
        if pcma_data.dtype != np.uint8:
            if pcma_data.dtype in [np.int16, np.int32]:
                # 将整数类型转换为uint8范围
                pcma_data = ((pcma_data + 32768) // 256).astype(np.uint8)
            else:
                raise ValueError("PCMA数据必须是uint8类型或可转换的整数类型")

        # 使用查找表进行转换
        decode_table = _build_pcma_decode_table()
        return decode_table[pcma_data]

    def _int16_to_float32(self, int16_data: np.ndarray) -> np.ndarray:
        """
        int16格式转换为float32
        
        Args:
            int16_data: int16格式数据
            
        Returns:
            float32格式数据
        """
        if int16_data.dtype != np.int16:
            raise ValueError("输入数据必须是int16类型")

        # 转换为float32并归一化到[-1, 1]
        return int16_data.astype(np.float32) / 32768.0

    def _int32_to_float32(self, int32_data: np.ndarray) -> np.ndarray:
        """
        int32格式转换为float32
        
        Args:
            int32_data: int32格式数据
            
        Returns:
            float32格式数据
        """
        if int32_data.dtype != np.int32:
            raise ValueError("输入数据必须是int32类型")

        # 转换为float32并归一化到[-1, 1]
        return int32_data.astype(np.float32) / 2147483648.0

    def _resample_audio(self,
                       audio_data: np.ndarray,
                       source_rate: int,
                       target_rate: int) -> np.ndarray:
        """
        音频重采样
        
        使用简单的线性插值进行重采样。
        对于更高质量的重采样，建议使用专业的音频库如librosa。
        
        Args:
            audio_data: 输入音频数据
            source_rate: 源采样率
            target_rate: 目标采样率
            
        Returns:
            重采样后的音频数据
        """
        if source_rate == target_rate:
            return audio_data

        # 计算重采样比例
        ratio = target_rate / source_rate
        new_length = int(len(audio_data) * ratio)

        # 使用线性插值
        old_indices = np.linspace(0, len(audio_data) - 1, new_length)

        # 使用numpy的线性插值
        resampled_data = np.interp(old_indices, np.arange(len(audio_data)), audio_data)

        return resampled_data.astype(np.float32)

    def get_supported_formats(self) -> list[AudioFormat]:
        """获取支持的音频格式列表"""
        return list(AudioFormat)

    def get_format_info(self, format_type: AudioFormat) -> dict[str, Any]:
        """
        获取音频格式信息
        
        Args:
            format_type: 音频格式
            
        Returns:
            格式信息字典
        """
        info = {
            "format": format_type.value,
            "description": "",
            "supported_sample_rates": [],
            "typical_bit_depth": None,
            "compression": "无损" if format_type == AudioFormat.WAV else "有损"
        }

        if format_type == AudioFormat.WAV:
            info.update({
                "description": "无损WAV格式，支持多种采样率和位深度",
                "supported_sample_rates": [8000, 16000, 22050, 44100, 48000],
                "typical_bit_depth": [16, 24, 32]
            })
        elif format_type == AudioFormat.PCMA:
            info.update({
                "description": "G.711 A-law压缩格式，主要用于电话通信",
                "supported_sample_rates": [8000, 16000],
                "typical_bit_depth": 8
            })

        return info

    def clear_cache(self) -> None:
        """清除所有缓存"""
        self._chunk_size_cache.clear()
        self._conversion_cache.clear()

__all__ = [
    "AudioFormatProcessor", "CHUNK_SIZES"
]
