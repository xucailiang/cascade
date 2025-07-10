"""
音频格式处理基础模块

本模块定义了音频格式处理的抽象接口，包括音频格式和音频流的基类。
"""

import abc
from typing import Any, BinaryIO

import numpy as np

from cascade.types.audio import AudioConfig
from cascade.types.audio import AudioFormat


class AudioStream(abc.ABC):
    """
    音频流抽象基类
    
    定义了音频流的通用接口，所有具体的音频流实现都应该继承此类。
    """

    @abc.abstractmethod
    def open(self, source: str | BinaryIO) -> None:
        """
        打开音频流
        
        Args:
            source: 音频源，可以是文件路径或文件对象
            
        Raises:
            ValueError: 当音频源无效时
            IOError: 当打开音频源失败时
        """
        pass

    @abc.abstractmethod
    def close(self) -> None:
        """
        关闭音频流
        
        Raises:
            IOError: 当关闭音频源失败时
        """
        pass

    @abc.abstractmethod
    def read(self, frames: int = -1) -> np.ndarray:
        """
        读取指定帧数的音频数据
        
        Args:
            frames: 要读取的帧数，-1表示读取所有剩余帧
            
        Returns:
            读取的音频数据
            
        Raises:
            IOError: 当读取失败时
            EOFError: 当到达文件末尾时
        """
        pass

    @abc.abstractmethod
    def write(self, data: np.ndarray) -> int:
        """
        写入音频数据
        
        Args:
            data: 要写入的音频数据
            
        Returns:
            实际写入的帧数
            
        Raises:
            IOError: 当写入失败时
        """
        pass

    @abc.abstractmethod
    def seek(self, position: int) -> None:
        """
        设置流位置
        
        Args:
            position: 目标位置（帧数）
            
        Raises:
            IOError: 当设置位置失败时
            ValueError: 当位置无效时
        """
        pass

    @abc.abstractmethod
    def tell(self) -> int:
        """
        获取当前流位置
        
        Returns:
            当前位置（帧数）
            
        Raises:
            IOError: 当获取位置失败时
        """
        pass

    @abc.abstractmethod
    def get_format(self) -> AudioFormat:
        """
        获取音频格式
        
        Returns:
            音频格式
        """
        pass

    @abc.abstractmethod
    def get_config(self) -> AudioConfig:
        """
        获取音频配置
        
        Returns:
            音频配置
        """
        pass

    @abc.abstractmethod
    def get_duration(self) -> float:
        """
        获取音频时长
        
        Returns:
            音频时长（秒）
            
        Raises:
            IOError: 当获取时长失败时
        """
        pass

    @abc.abstractmethod
    def get_frames(self) -> int:
        """
        获取总帧数
        
        Returns:
            总帧数
            
        Raises:
            IOError: 当获取帧数失败时
        """
        pass

    @property
    @abc.abstractmethod
    def closed(self) -> bool:
        """
        检查流是否已关闭
        
        Returns:
            流是否已关闭
        """
        pass

    def __enter__(self):
        """上下文管理器入口"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.close()


class FormatConverter(abc.ABC):
    """
    音频格式转换器抽象基类
    
    定义了音频格式转换的通用接口，所有具体的格式转换器都应该继承此类。
    """

    @abc.abstractmethod
    def convert_to_internal(self, audio_data: np.ndarray, config: AudioConfig) -> tuple[np.ndarray, int, int]:
        """
        将输入音频转换为内部处理格式
        
        Args:
            audio_data: 输入音频数据
            config: 音频格式配置
            
        Returns:
            转换后的音频数据，采样率，通道数
            
        Raises:
            ValueError: 当音频格式不支持时
        """
        pass

    @abc.abstractmethod
    def convert_from_internal(self, audio_data: np.ndarray, config: AudioConfig) -> np.ndarray:
        """
        将内部格式转换为输出音频格式
        
        Args:
            audio_data: 内部格式的音频数据
            config: 目标音频格式配置
            
        Returns:
            转换后的音频数据
            
        Raises:
            ValueError: 当音频格式不支持时
        """
        pass


class FormatProcessorRegistry:
    """
    格式处理器注册中心
    
    管理所有注册的音频格式处理器。
    """

    def __init__(self):
        """初始化格式处理器注册中心"""
        self._processors = {}

    def register(self, format_type: AudioFormat, processor_class: type) -> None:
        """
        注册格式处理器
        
        Args:
            format_type: 音频格式类型
            processor_class: 处理器类
        """
        self._processors[format_type] = processor_class

    def get_processor(self, format_type: AudioFormat) -> Any:
        """
        获取格式处理器实例
        
        Args:
            format_type: 音频格式类型
            
        Returns:
            格式处理器实例
            
        Raises:
            ValueError: 当没有注册对应格式的处理器时
        """
        if format_type not in self._processors:
            raise ValueError(f"没有注册处理器用于格式: {format_type}")

        return self._processors[format_type]()

    def supported_formats(self) -> list:
        """
        获取支持的格式列表
        
        Returns:
            支持的格式列表
        """
        return list(self._processors.keys())


# 全局格式处理器注册中心实例
registry = FormatProcessorRegistry()
