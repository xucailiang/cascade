"""
音频缓冲区抽象基类模块

定义音频缓冲区的统一接口和通用行为，为不同类型的缓冲区实现提供基础。
遵循依赖倒置原则，使上层模块依赖抽象而非具体实现。

设计原则：
- 接口隔离：最小化接口，每个接口职责单一
- 依赖倒置：上层模块依赖抽象接口
- 开闭原则：对扩展开放，对修改封闭
- 类型安全：完整的类型注解和运行时验证
"""

from abc import ABC, abstractmethod
from contextlib import contextmanager

import numpy as np

from ..types import AudioChunk, AudioConfig, BufferError, BufferStatus


class AudioBuffer(ABC):
    """
    音频缓冲区抽象基类
    
    定义音频缓冲区的核心接口，所有具体缓冲区实现都必须遵循此接口。
    
    设计目标：
    - 零拷贝数据访问
    - 线程安全操作
    - 高性能内存管理
    - 完整的状态监控
    """

    def __init__(self, config: AudioConfig, capacity_seconds: float):
        """
        初始化音频缓冲区
        
        Args:
            config: 音频配置，定义采样率、格式等基本参数
            capacity_seconds: 缓冲区容量（秒），影响内存分配大小
            
        Raises:
            ValueError: 当配置参数无效时
            BufferError: 当缓冲区初始化失败时
        """
        self._config = config
        self._capacity_seconds = capacity_seconds
        self._capacity_samples = int(capacity_seconds * config.sample_rate)
        self._is_closed = False

        # 子类需要初始化的状态
        self._total_written = 0
        self._total_read = 0

    @property
    def config(self) -> AudioConfig:
        """获取音频配置"""
        return self._config

    @property
    def capacity_seconds(self) -> float:
        """获取容量（秒）"""
        return self._capacity_seconds

    @property
    def capacity_samples(self) -> int:
        """获取容量（样本数）"""
        return self._capacity_samples

    @property
    def is_closed(self) -> bool:
        """检查缓冲区是否已关闭"""
        return self._is_closed

    # === 核心数据操作接口 ===

    @abstractmethod
    def write(self, data: np.ndarray, blocking: bool = True,
              timeout: float | None = None) -> bool:
        """
        写入音频数据到缓冲区
        
        Args:
            data: 音频数据（numpy数组，dtype必须与配置一致）
            blocking: 是否阻塞等待空间可用
            timeout: 超时时间（秒），None表示无限等待
            
        Returns:
            bool: 写入是否成功
            
        Raises:
            BufferError: 当缓冲区状态异常时
            ValueError: 当数据格式不匹配时
            TimeoutError: 当阻塞写入超时时
        """
        pass

    @abstractmethod
    def get_chunk_with_overlap(self, chunk_size: int,
                              overlap_size: int) -> tuple[AudioChunk | None, bool]:
        """
        获取带重叠的音频块（核心接口）
        
        这是与processor模块的主要集成接口，专为VAD边界处理设计。
        
        Args:
            chunk_size: 主要块大小（样本数）
            overlap_size: 重叠区域大小（样本数）
            
        Returns:
            Tuple[AudioChunk | None, bool]: (音频块, 是否有足够数据)
            - 如果数据足够，返回(AudioChunk, True)
            - 如果数据不足，返回(None, False)
            
        Raises:
            ValueError: 当参数无效时（如overlap_size >= chunk_size）
            BufferError: 当缓冲区状态异常时
        """
        pass

    @abstractmethod
    def advance_read_position(self, size: int) -> None:
        """
        前进读取位置（处理完成后调用）
        
        Args:
            size: 已处理的样本数（不包括重叠区域）
            
        Raises:
            ValueError: 当size参数无效时
            BufferError: 当缓冲区状态异常时
        """
        pass

    # === 状态查询接口 ===

    @abstractmethod
    def get_buffer_status(self) -> BufferStatus:
        """
        获取缓冲区状态信息
        
        Returns:
            BufferStatus: 包含容量、使用率、位置等完整状态信息
        """
        pass

    @abstractmethod
    def available_samples(self) -> int:
        """
        获取可读取的样本数
        
        Returns:
            int: 当前可读取的样本数量
        """
        pass

    def remaining_capacity(self) -> int:
        """
        获取剩余容量（样本数）
        
        Returns:
            int: 剩余可写入的样本数量
        """
        return self.capacity_samples - self.available_samples()

    def is_empty(self) -> bool:
        """检查缓冲区是否为空"""
        return self.available_samples() == 0

    def is_full(self) -> bool:
        """检查缓冲区是否已满"""
        return self.available_samples() >= self.capacity_samples

    def utilization_ratio(self) -> float:
        """获取利用率（0.0-1.0）"""
        return self.available_samples() / self.capacity_samples

    # === 管理接口 ===

    @abstractmethod
    def clear(self) -> None:
        """
        清空缓冲区
        
        清除所有数据，重置读写位置，但保持缓冲区可用状态。
        
        Raises:
            BufferError: 当缓冲区已关闭时
        """
        pass

    def resize(self, new_capacity_seconds: float) -> bool:
        """
        动态调整缓冲区大小
        
        默认实现返回False，表示不支持动态调整。
        子类可以重写此方法提供实际的调整功能。
        
        Args:
            new_capacity_seconds: 新的容量（秒）
            
        Returns:
            bool: 调整是否成功
        """
        return False

    @abstractmethod
    def close(self) -> None:
        """
        关闭缓冲区，释放资源
        
        关闭后的缓冲区不能再使用，所有操作都会抛出异常。
        """
        pass

    # === 辅助方法 ===

    def _validate_not_closed(self) -> None:
        """验证缓冲区未关闭"""
        if self._is_closed:
            raise BufferError("缓冲区已关闭")

    def _validate_data_format(self, data: np.ndarray) -> None:
        """
        验证数据格式
        
        Args:
            data: 要验证的数据
            
        Raises:
            ValueError: 当数据格式不匹配时
        """
        if not isinstance(data, np.ndarray):
            raise ValueError("数据必须是numpy数组")

        if len(data.shape) != 1:
            raise ValueError("数据必须是一维数组")

        expected_dtype = np.dtype(self._config.dtype)
        if data.dtype != expected_dtype:
            raise ValueError(f"数据类型不匹配: 期望{expected_dtype}, 实际{data.dtype}")

    def _validate_chunk_params(self, chunk_size: int, overlap_size: int) -> None:
        """
        验证块参数
        
        Args:
            chunk_size: 块大小
            overlap_size: 重叠大小
            
        Raises:
            ValueError: 当参数无效时
        """
        if chunk_size <= 0:
            raise ValueError("块大小必须大于0")

        if overlap_size < 0:
            raise ValueError("重叠大小不能为负数")

        if overlap_size >= chunk_size:
            raise ValueError("重叠大小不能大于等于块大小")

    def get_total_written(self) -> int:
        """获取总写入样本数"""
        return self._total_written

    def get_total_read(self) -> int:
        """获取总读取样本数"""
        return self._total_read

    def get_duration_seconds(self) -> float:
        """获取已写入数据的时长（秒）"""
        return self._total_written / self._config.sample_rate

    # === 上下文管理器支持 ===

    def __enter__(self) -> 'AudioBuffer':
        """上下文管理器入口"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """上下文管理器出口，自动关闭缓冲区"""
        if not self._is_closed:
            self.close()

    # === 线程安全的上下文管理器 ===

    @contextmanager
    def _thread_safe_operation(self, operation_name: str):
        """
        线程安全操作的上下文管理器
        
        子类可以重写此方法提供具体的锁机制。
        
        Args:
            operation_name: 操作名称，用于错误诊断
        """
        self._validate_not_closed()
        try:
            yield
        except Exception as e:
            # 子类可以在这里添加错误恢复逻辑
            raise BufferError(f"{operation_name}操作失败: {str(e)}") from e

    # === 字符串表示 ===

    def __str__(self) -> str:
        """字符串表示"""
        return (f"{self.__class__.__name__}("
                f"capacity={self.capacity_seconds}s/"
                f"{self.capacity_samples}samples, "
                f"available={self.available_samples()}, "
                f"utilization={self.utilization_ratio():.1%})")

    def __repr__(self) -> str:
        """调试表示"""
        return (f"{self.__class__.__name__}("
                f"config={self._config}, "
                f"capacity_seconds={self._capacity_seconds}, "
                f"closed={self._is_closed})")


# === 便利函数 ===

def validate_buffer_config(config: AudioConfig, capacity_seconds: float) -> None:
    """
    验证缓冲区配置的便利函数
    
    Args:
        config: 音频配置
        capacity_seconds: 容量（秒）
        
    Raises:
        ValueError: 当配置无效时
    """
    if capacity_seconds <= 0:
        raise ValueError("容量必须大于0秒")

    if capacity_seconds > 300:  # 5分钟限制
        raise ValueError("容量不能超过300秒")

    # 验证配置本身
    if config.sample_rate <= 0:
        raise ValueError("采样率必须大于0")

    if config.channels != 1:
        raise ValueError("当前仅支持单声道")


__all__ = [
    "AudioBuffer",
    "validate_buffer_config",
]
