"""
WAV 格式音频流处理模块

本模块提供对 WAV 格式音频文件的读写支持。
"""

import wave
from typing import BinaryIO

import numpy as np

from cascade.formats.base import AudioStream, FormatConverter, registry
from cascade.types.audio import AudioConfig, AudioFormat


class WavStream(AudioStream):
    """
    WAV 格式音频流
    
    提供对 WAV 格式音频文件的读写支持。
    """

    def __init__(self, config: AudioConfig | None = None):
        """
        初始化 WAV 音频流
        
        Args:
            config: 音频配置，如果为 None，则在打开文件时从文件中读取
        """
        self._config = config or AudioConfig(format=AudioFormat.WAV)
        self._reader = None  # Wave_read 对象，用于读取
        self._writer = None  # Wave_write 对象，用于写入
        self._source = None
        self._mode = None
        self._converter = WavFormatConverter()
        self._frames = 0
        self._position = 0

    def open(self, source: str | BinaryIO, mode: str = 'r') -> None:
        """
        打开 WAV 音频流
        
        Args:
            source: 音频源，可以是文件路径或文件对象
            mode: 打开模式，'r' 表示读取，'w' 表示写入
            
        Raises:
            ValueError: 当音频源无效或模式无效时
            IOError: 当打开音频源失败时
        """
        self.close()

        if mode not in ('r', 'w'):
            raise ValueError(f"无效的模式: {mode}，必须是 'r' 或 'w'")

        self._mode = mode
        self._source = source

        try:
            # 根据模式打开不同类型的 wave 对象
            if mode == 'r':
                # 读取模式
                if isinstance(source, str):
                    self._reader = wave.open(source, 'rb')
                else:
                    self._reader = wave.open(source, 'rb')

                # 从文件中读取配置
                channels = self._reader.getnchannels()
                sample_rate = self._reader.getframerate()
                sample_width = self._reader.getsampwidth()

                # 根据采样宽度确定数据类型
                dtype = 'int16' if sample_width == 2 else 'int32'

                # 更新配置
                self._config = AudioConfig(
                    format=AudioFormat.WAV,
                    sample_rate=sample_rate,
                    channels=channels,
                    dtype=dtype,
                    bit_depth=sample_width * 8
                )

                # 获取总帧数
                self._frames = self._reader.getnframes()
                self._position = 0
            else:  # mode == 'w'
                # 写入模式
                if isinstance(source, str):
                    self._writer = wave.open(source, 'wb')
                else:
                    self._writer = wave.open(source, 'wb')

                # 设置文件参数
                self._writer.setnchannels(self._config.channels)
                self._writer.setsampwidth(2)  # 使用 16 位采样
                self._writer.setframerate(self._config.sample_rate)

                self._frames = 0
                self._position = 0
        except (OSError, wave.Error) as e:
            raise OSError(f"打开 WAV 文件失败: {str(e)}")

    def close(self) -> None:
        """
        关闭音频流
        
        Raises:
            IOError: 当关闭音频源失败时
        """
        try:
            if self._reader is not None:
                self._reader.close()
                self._reader = None

            if self._writer is not None:
                self._writer.close()
                self._writer = None
        except Exception as e:
            raise OSError(f"关闭 WAV 文件失败: {str(e)}")
        finally:
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
        if self._reader is None:
            raise ValueError("音频流未打开或不是读取模式")

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
            raw_data = self._reader.readframes(frames)

            # 转换为 numpy 数组
            if self._config.bit_depth == 16:
                # 16 位整数
                data = np.frombuffer(raw_data, dtype=np.int16)
            else:
                # 32 位整数
                data = np.frombuffer(raw_data, dtype=np.int32)

            # 更新位置
            self._position += frames

            # 转换为内部格式（float32，范围 [-1.0, 1.0]）
            internal_data, _, _ = self._converter.convert_to_internal(data, self._config)

            return internal_data
        except Exception as e:
            raise OSError(f"读取 WAV 数据失败: {str(e)}")

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
        if self._writer is None:
            raise ValueError("音频流未打开或不是写入模式")

        if len(data) == 0:
            return 0

        try:
            # 转换为 WAV 格式
            wav_data = self._converter.convert_from_internal(data, self._config)

            # 写入数据
            self._writer.writeframes(wav_data.tobytes())

            # 更新位置和总帧数
            frames_written = len(data) // self._config.channels
            self._position += frames_written
            self._frames += frames_written

            return frames_written
        except Exception as e:
            raise OSError(f"写入 WAV 数据失败: {str(e)}")

    def seek(self, position: int) -> None:
        """
        设置流位置
        
        Args:
            position: 目标位置（帧数）
            
        Raises:
            IOError: 当设置位置失败时
            ValueError: 当位置无效或流未打开或不是读取模式时
        """
        if self._reader is None:
            raise ValueError("音频流未打开或不是读取模式")

        if position < 0 or position > self._frames:
            raise ValueError(f"无效的位置: {position}，必须在 0 到 {self._frames} 之间")

        try:
            self._reader.setpos(position)
            self._position = position
        except Exception as e:
            raise OSError(f"设置 WAV 文件位置失败: {str(e)}")

    def tell(self) -> int:
        """
        获取当前流位置
        
        Returns:
            当前位置（帧数）
            
        Raises:
            IOError: 当获取位置失败时
            ValueError: 当流未打开时
        """
        if self._reader is None and self._writer is None:
            raise ValueError("音频流未打开")

        return self._position

    def get_format(self) -> AudioFormat:
        """
        获取音频格式
        
        Returns:
            音频格式
        """
        return AudioFormat.WAV

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
        if self._reader is None and self._writer is None:
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
        if self._reader is None and self._writer is None:
            raise ValueError("音频流未打开")

        return self._frames

    @property
    def closed(self) -> bool:
        """
        检查流是否已关闭
        
        Returns:
            流是否已关闭
        """
        return self._reader is None and self._writer is None


class WavFormatConverter(FormatConverter):
    """
    WAV 格式转换器
    
    提供 WAV 格式与内部格式之间的转换。
    """

    def convert_to_internal(self, audio_data: np.ndarray, config: AudioConfig) -> tuple[np.ndarray, int, int]:
        """
        将 WAV 格式音频转换为内部处理格式
        
        Args:
            audio_data: 输入音频数据
            config: 音频格式配置
            
        Returns:
            转换后的音频数据，采样率，通道数
            
        Raises:
            ValueError: 当音频格式不支持时
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
        将内部格式转换为 WAV 格式
        
        Args:
            audio_data: 内部格式的音频数据
            config: 目标音频格式配置
            
        Returns:
            转换后的音频数据
            
        Raises:
            ValueError: 当音频格式不支持时
        """
        if config.format != AudioFormat.WAV:
            raise ValueError(f"不支持的音频格式: {config.format}，预期为 WAV")

        # 从 float32 转换为目标数据类型
        if config.dtype == 'int16':
            # 将 [-1.0, 1.0] 范围转换为 int16
            return (audio_data * 32767.0).astype(np.int16)
        elif config.dtype == 'int32':
            # 将 [-1.0, 1.0] 范围转换为 int32
            return (audio_data * 2147483647.0).astype(np.int32)
        else:
            # 保持为 float32
            return audio_data


# 注册 WAV 格式处理器
registry.register(AudioFormat.WAV, WavStream)
