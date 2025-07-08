"""
音频缓冲区抽象基类

本模块定义了音频缓冲区的抽象接口，所有具体的缓冲区实现都应该继承此类。
"""

import abc
from enum import Enum

import numpy as np


class BufferStrategy(str, Enum):
    """缓冲区溢出处理策略"""
    BLOCK = "block"       # 阻塞直到有空间
    OVERWRITE = "overwrite"  # 覆盖最旧数据
    REJECT = "reject"     # 拒绝新数据


class BufferFullError(Exception):
    """缓冲区已满异常"""
    pass


class AudioBuffer(abc.ABC):
    """
    音频缓冲区抽象基类
    
    定义了音频缓冲区的通用接口，所有具体的缓冲区实现都应该继承此类。
    """

    @abc.abstractmethod
    def write(self, data: np.ndarray, timeout: float | None = None) -> int:
        """
        写入数据到缓冲区
        
        Args:
            data: 要写入的音频数据
            timeout: 超时时间（秒），None表示无限等待
            
        Returns:
            实际写入的数据量
            
        Raises:
            BufferFullError: 当缓冲区已满且策略为REJECT时
            TimeoutError: 当等待超时时
        """
        pass

    @abc.abstractmethod
    def read(self, size: int, timeout: float | None = None) -> np.ndarray:
        """
        从缓冲区读取指定大小的数据
        
        Args:
            size: 要读取的数据大小（样本数）
            timeout: 超时时间（秒），None表示无限等待
            
        Returns:
            读取的音频数据
            
        Raises:
            ValueError: 当请求的大小无效时
            TimeoutError: 当等待超时时
        """
        pass

    @abc.abstractmethod
    def get_chunk(self, chunk_size: int, overlap_size: int = 0) -> np.ndarray:
        """
        获取指定大小的数据块，可以包含重叠区域
        
        Args:
            chunk_size: 主要块大小（样本数）
            overlap_size: 重叠区域大小（样本数）
            
        Returns:
            包含重叠区域的音频数据
            
        Raises:
            ValueError: 当请求的大小无效时
            BufferError: 当可用数据不足时
        """
        pass

    @abc.abstractmethod
    def get_chunk_with_overlap(self, chunk_size: int, overlap_size: int) -> tuple[np.ndarray, dict]:
        """
        获取指定大小的音频块，包含重叠区域和元数据
        
        Args:
            chunk_size: 主要块大小（样本数）
            overlap_size: 重叠区域大小（样本数）
            
        Returns:
            包含重叠区域的音频数据和元数据字典
            
        Raises:
            ValueError: 当请求的大小无效时
            BufferError: 当可用数据不足时
        """
        pass

    @abc.abstractmethod
    def advance_after_processing(self, chunk_size: int) -> None:
        """
        处理完成后前进读取位置
        
        Args:
            chunk_size: 已处理的块大小（不包括重叠区域）
            
        Raises:
            ValueError: 当请求的大小无效时
        """
        pass

    @abc.abstractmethod
    def peek(self, size: int) -> np.ndarray:
        """
        查看数据但不移动读取位置
        
        Args:
            size: 要查看的数据大小（样本数）
            
        Returns:
            查看的音频数据
            
        Raises:
            ValueError: 当请求的大小无效时
            BufferError: 当可用数据不足时
        """
        pass

    @abc.abstractmethod
    def skip(self, size: int) -> None:
        """
        跳过指定大小的数据
        
        Args:
            size: 要跳过的数据大小（样本数）
            
        Raises:
            ValueError: 当请求的大小无效时
        """
        pass

    @abc.abstractmethod
    def available(self) -> int:
        """
        返回可读取的数据量
        
        Returns:
            可读取的数据量（样本数）
        """
        pass

    @abc.abstractmethod
    def remaining(self) -> int:
        """
        返回剩余可写入的空间
        
        Returns:
            剩余可写入的空间（样本数）
        """
        pass

    @abc.abstractmethod
    def clear(self) -> None:
        """清空缓冲区"""
        pass

    @abc.abstractmethod
    def is_empty(self) -> bool:
        """
        检查缓冲区是否为空
        
        Returns:
            缓冲区是否为空
        """
        pass

    @abc.abstractmethod
    def is_full(self) -> bool:
        """
        检查缓冲区是否已满
        
        Returns:
            缓冲区是否已满
        """
        pass

    @abc.abstractmethod
    def close(self) -> None:
        """关闭缓冲区，释放资源"""
        pass
