"""
PCMA 格式音频流处理模块

本模块提供对 PCMA (A-law) 格式音频数据的读写支持。
PCMA 是一种常用于电话系统的音频压缩格式，每个样本使用 8 位表示。
"""

import os
from typing import BinaryIO

import numpy as np

from cascade.formats.base import AudioStream, FormatConverter, registry
from cascade.types.audio import AudioConfig, AudioFormat

# PCMA 编解码常量
PCMA_QUANTIZE_MASK = 0xD5  # 1101 0101
PCMA_CLIP_LEVEL = 32635


class PcmaStream(AudioStream):
    """
    PCMA 格式音频流
    
    提供对 PCMA 格式音频文件的读写支持。
    """

    def __init__(self, config: AudioConfig | None = None):
        """
        初始化 PCMA 音频流
        
        Args:
            config: 音频配置，如果为 None，则使用默认配置
        """
        # 默认配置：8kHz 采样率，单声道，PCMA 格式
        default_config = AudioConfig(
            format=AudioFormat.PCMA,
            sample_rate=8000,
            channels=1,
            dtype='int16',
            bit_depth=8
        )

        self._config = config or default_config

        # 验证配置
        if self._config.format != AudioFormat.PCMA:
            self._config.format = AudioFormat.PCMA

        if self._config.sample_rate not in [8000, 16000]:
            raise ValueError("PCMA 格式仅支持 8kHz 和 16kHz 采样率")

        if self._config.channels != 1:
            raise ValueError("当前版本仅支持单声道 PCMA")

        self._file = None
        self._source = None
        self._mode = None
        self._converter = PcmaFormatConverter()
        self._frames = 0
        self._position = 0
        self._file_size = 0

    def open(self, source: str | BinaryIO, mode: str = 'r') -> None:
        """
        打开 PCMA 音频流
        
        Args:
            source: 音频源，可以是文件路径或文件对象
            mode: 打开模式，'r' 表示读取，'w' 表示写入
            
        Raises:
            ValueError: 当音频源无效或模式无效时
            IOError: 当打开音频源失败时
        """
        if self._file is not None:
            self.close()

        if mode not in ('r', 'w'):
            raise ValueError(f"无效的模式: {mode}，必须是 'r' 或 'w'")

        self._mode = mode
        self._source = source

        try:
            # 打开文件
            if isinstance(source, str):
                self._file = open(source, f"{mode}b")
            else:
                self._file = source

            # 读取模式下，获取文件大小和帧数
            if mode == 'r':
                # 获取文件大小
                if isinstance(source, str):
                    self._file_size = os.path.getsize(source)
                else:
                    # 保存当前位置
                    current_pos = source.tell()
                    # 移动到文件末尾
                    source.seek(0, os.SEEK_END)
                    # 获取文件大小
                    self._file_size = source.tell()
                    # 恢复位置
                    source.seek(current_pos, os.SEEK_SET)

                # 计算帧数（每个样本 1 字节）
                self._frames = self._file_size
                self._position = 0
            else:  # mode == 'w'
                self._frames = 0
                self._position = 0
        except OSError as e:
            raise OSError(f"打开 PCMA 文件失败: {str(e)}")

    def close(self) -> None:
        """
        关闭音频流
        
        Raises:
            IOError: 当关闭音频源失败时
        """
        if self._file is not None:
            try:
                # 只有当文件是我们打开的（而不是传入的）时才关闭
                if isinstance(self._source, str):
                    self._file.close()
            except Exception as e:
                raise OSError(f"关闭 PCMA 文件失败: {str(e)}")
            finally:
                self._file = None
                self._source = None
                self._position = 0

    def read(self, frames: int = -1) -> np.ndarray:
        """
        读取指定帧数的音频数据
        
        Args:
            frames: 要读取的帧数，-1 表示读取所有剩余帧
            
        Returns:
            读取的音频数据
            
        Raises:
            IOError: 当读取失败时
            EOFError: 当到达文件末尾时
            ValueError: 当流未打开或不是读取模式时
        """
        if self._file is None:
            raise ValueError("音频流未打开")

        if self._mode != 'r':
            raise ValueError("音频流不是读取模式")

        if frames == -1:
            # 读取所有剩余帧
            frames = self._frames - self._position

        if frames <= 0:
            return np.array([], dtype=np.float32)

        if self._position >= self._frames:
            raise EOFError("已到达文件末尾")

        # 限制读取帧数不超过剩余帧数
        frames = min(frames, self._frames - self._position)

        try:
            # 读取原始字节数据
            raw_data = self._file.read(frames)

            # 转换为 numpy 数组
            pcma_data = np.frombuffer(raw_data, dtype=np.uint8)

            # 更新位置
            self._position += len(pcma_data)

            # 转换为内部格式（float32，范围 [-1.0, 1.0]）
            internal_data, _, _ = self._converter.convert_to_internal(pcma_data, self._config)

            return internal_data
        except Exception as e:
            raise OSError(f"读取 PCMA 数据失败: {str(e)}")

    def write(self, data: np.ndarray) -> int:
        """
        写入音频数据
        
        Args:
            data: 要写入的音频数据
            
        Returns:
            实际写入的帧数
            
        Raises:
            IOError: 当写入失败时
            ValueError: 当流未打开或不是写入模式时
        """
        if self._file is None:
            raise ValueError("音频流未打开")

        if self._mode != 'w':
            raise ValueError("音频流不是写入模式")

        if len(data) == 0:
            return 0

        try:
            # 转换为 PCMA 格式
            pcma_data = self._converter.convert_from_internal(data, self._config)

            # 写入数据
            self._file.write(pcma_data.tobytes())

            # 更新位置和总帧数
            frames_written = len(pcma_data)
            self._position += frames_written
            self._frames += frames_written

            return frames_written
        except Exception as e:
            raise OSError(f"写入 PCMA 数据失败: {str(e)}")

    def seek(self, position: int) -> None:
        """
        设置流位置
        
        Args:
            position: 目标位置（帧数）
            
        Raises:
            IOError: 当设置位置失败时
            ValueError: 当位置无效或流未打开或不是读取模式时
        """
        if self._file is None:
            raise ValueError("音频流未打开")

        if self._mode != 'r':
            raise ValueError("只有读取模式支持定位")

        if position < 0 or position > self._frames:
            raise ValueError(f"无效的位置: {position}，必须在 0 到 {self._frames} 之间")

        try:
            self._file.seek(position, os.SEEK_SET)
            self._position = position
        except Exception as e:
            raise OSError(f"设置 PCMA 文件位置失败: {str(e)}")

    def tell(self) -> int:
        """
        获取当前流位置
        
        Returns:
            当前位置（帧数）
            
        Raises:
            IOError: 当获取位置失败时
            ValueError: 当流未打开时
        """
        if self._file is None:
            raise ValueError("音频流未打开")

        return self._position

    def get_format(self) -> AudioFormat:
        """
        获取音频格式
        
        Returns:
            音频格式
        """
        return AudioFormat.PCMA

    def get_config(self) -> AudioConfig:
        """
        获取音频配置
        
        Returns:
            音频配置
        """
        return self._config

    def get_duration(self) -> float:
        """
        获取音频时长
        
        Returns:
            音频时长（秒）
            
        Raises:
            IOError: 当获取时长失败时
            ValueError: 当流未打开时
        """
        if self._file is None:
            raise ValueError("音频流未打开")

        return self._frames / self._config.sample_rate

    def get_frames(self) -> int:
        """
        获取总帧数
        
        Returns:
            总帧数
            
        Raises:
            IOError: 当获取帧数失败时
            ValueError: 当流未打开时
        """
        if self._file is None:
            raise ValueError("音频流未打开")

        return self._frames

    @property
    def closed(self) -> bool:
        """
        检查流是否已关闭
        
        Returns:
            流是否已关闭
        """
        return self._file is None


class PcmaFormatConverter(FormatConverter):
    """
    PCMA 格式转换器
    
    提供 PCMA 格式与内部格式之间的转换。
    """

    def convert_to_internal(self, audio_data: np.ndarray, config: AudioConfig) -> tuple[np.ndarray, int, int]:
        """
        将 PCMA 格式音频转换为内部处理格式
        
        Args:
            audio_data: 输入音频数据（uint8 类型的 PCMA 编码数据）
            config: 音频格式配置
            
        Returns:
            转换后的音频数据，采样率，通道数
            
        Raises:
            ValueError: 当音频格式不支持时
        """
        if config.format != AudioFormat.PCMA:
            raise ValueError(f"不支持的音频格式: {config.format}，预期为 PCMA")

        # 将 PCMA 解码为线性 PCM
        linear_data = self._pcma_to_linear(audio_data)

        # 转换为 float32，范围 [-1.0, 1.0]
        float_data = linear_data.astype(np.float32) / 32767.0

        return float_data, config.sample_rate, config.channels

    def convert_from_internal(self, audio_data: np.ndarray, config: AudioConfig) -> np.ndarray:
        """
        将内部格式转换为 PCMA 格式
        
        Args:
            audio_data: 内部格式的音频数据（float32，范围 [-1.0, 1.0]）
            config: 目标音频格式配置
            
        Returns:
            转换后的音频数据（uint8 类型的 PCMA 编码数据）
            
        Raises:
            ValueError: 当音频格式不支持时
        """
        if config.format != AudioFormat.PCMA:
            raise ValueError(f"不支持的音频格式: {config.format}，预期为 PCMA")

        # 将 float32 转换为线性 PCM
        linear_data = (audio_data * 32767.0).astype(np.int16)

        # 将线性 PCM 编码为 PCMA
        pcma_data = self._linear_to_pcma(linear_data)

        return pcma_data

    def _pcma_to_linear(self, pcma_data: np.ndarray) -> np.ndarray:
        """
        将 PCMA 编码数据转换为线性 PCM
        
        Args:
            pcma_data: PCMA 编码数据（uint8 类型）
            
        Returns:
            线性 PCM 数据（int16 类型）
        """
        # 创建输出数组
        linear_data = np.zeros(len(pcma_data), dtype=np.int16)

        # 反转 A-law 比特
        pcma_data = np.bitwise_xor(pcma_data, 0x55)

        # 解码每个样本
        for i in range(len(pcma_data)):
            pcma_byte = pcma_data[i]

            # 提取符号位、段号和段内偏移
            sign = 1 - 2 * ((pcma_byte >> 7) & 0x01)  # 1 或 -1
            segment = (pcma_byte >> 4) & 0x07
            quantized = pcma_byte & 0x0F

            # 重建线性样本
            if segment == 0:
                linear_val = quantized << 4
            else:
                linear_val = (0x10 | quantized) << (segment + 3)

            # 应用符号位
            linear_data[i] = sign * linear_val

        return linear_data

    def _linear_to_pcma(self, linear_data: np.ndarray) -> np.ndarray:
        """
        将线性 PCM 数据转换为 PCMA 编码
        
        Args:
            linear_data: 线性 PCM 数据（int16 类型）
            
        Returns:
            PCMA 编码数据（uint8 类型）
        """
        # 创建输出数组
        pcma_data = np.zeros(len(linear_data), dtype=np.uint8)

        # 编码每个样本
        for i in range(len(linear_data)):
            sample = linear_data[i]

            # 确定符号位和绝对值
            sign = 0 if sample >= 0 else 1
            sample_abs = abs(sample)

            # 限制幅度
            sample_abs = min(sample_abs, PCMA_CLIP_LEVEL)

            # 确定段号和段内偏移
            segment = 0
            mask = 0x0080

            while (segment < 8) and ((sample_abs & mask) != mask):
                segment += 1
                mask >>= 1

            if segment < 8:
                # 计算量化值
                if segment == 0:
                    quantized = sample_abs >> 4
                else:
                    quantized = (sample_abs >> (segment + 3)) & 0x0F

                # 组合 PCMA 字节
                pcma_byte = (sign << 7) | (segment << 4) | quantized
            else:
                # 处理极小值
                pcma_byte = sign << 7

            # 反转 A-law 比特
            pcma_data[i] = pcma_byte ^ 0x55

        return pcma_data


# 注册 PCMA 格式处理器
registry.register(AudioFormat.PCMA, PcmaStream)
