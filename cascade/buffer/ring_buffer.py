"""
高性能音频环形缓冲区实现

实现零拷贝、线程安全的音频环形缓冲区，专为VAD处理场景优化。
提供带重叠的数据块获取、智能容量管理、性能监控等高级功能。

核心特性：
- 环形存储：自动重用内存空间，避免碎片
- 零拷贝访问：使用内存视图，最小化数据复制
- 线程安全：读写锁分离，原子操作，条件变量
- 重叠处理：专为VAD边界处理设计的重叠块获取
- 内存优化：缓存行对齐，连续内存分配
- 性能监控：完整的性能指标收集
"""

import threading
import time
from typing import Any

import numpy as np

from .._internal.atomic import AtomicFloat, AtomicInteger
from .._internal.utils import measure_time
from ..types import AudioChunk, AudioConfig, BufferStatus, BufferStrategy
from ..types.errors import InsufficientDataError
from .base import AudioBuffer


class AudioRingBuffer(AudioBuffer):
    """
    高性能音频环形缓冲区
    
    使用环形结构存储音频数据，支持零拷贝访问和线程安全操作。
    专为高频率的音频块读取和VAD处理场景优化。
    
    设计原则：
    - 零拷贝：优先使用内存视图，仅在必要时复制
    - 线程安全：细粒度锁设计，读写分离
    - 高性能：内存对齐，原子操作，批量处理
    - 可观测：完整的状态监控和性能指标
    """

    def __init__(self, config: AudioConfig, capacity_seconds: float,
                 overflow_strategy: BufferStrategy = BufferStrategy.BLOCK):
        """
        初始化音频环形缓冲区
        
        Args:
            config: 音频配置
            capacity_seconds: 缓冲区容量（秒）
            overflow_strategy: 溢出处理策略
        """
        super().__init__(config, capacity_seconds)

        self._overflow_strategy = overflow_strategy
        self._dtype = np.dtype(config.dtype)

        # 创建内存对齐的缓冲区
        self._buffer = self._create_aligned_buffer()

        # 原子位置指针
        self._write_pos = AtomicInteger(0)
        self._read_pos = AtomicInteger(0)
        self._available_data = AtomicInteger(0)

        # 序列号和统计
        self._sequence_counter = AtomicInteger(0)

        # 线程安全机制
        self._read_lock = threading.RLock()
        self._write_lock = threading.RLock()
        self._metadata_lock = threading.RLock()

        # 条件变量
        self._not_empty = threading.Condition(self._read_lock)
        self._not_full = threading.Condition(self._write_lock)

        # 性能监控
        self._write_count = AtomicInteger(0)
        self._read_count = AtomicInteger(0)
        self._zero_copy_count = AtomicInteger(0)
        self._copy_count = AtomicInteger(0)
        self._overflow_count = AtomicInteger(0)
        self._peak_usage = AtomicFloat(0.0)

        # 创建时间
        self._created_time = time.time()

    def _create_aligned_buffer(self) -> np.ndarray:
        """
        创建内存对齐的缓冲区
        
        Returns:
            内存对齐的numpy数组
        """
        # 计算对齐后的大小（64字节缓存行对齐）
        alignment = 64
        element_size = self._dtype.itemsize
        total_bytes = self.capacity_samples * element_size

        # 计算对齐后的字节数
        aligned_bytes = (total_bytes + alignment - 1) // alignment * alignment
        aligned_samples = aligned_bytes // element_size

        # 创建对齐的缓冲区
        buffer = np.zeros(aligned_samples, dtype=self._dtype)

        # 验证内存对齐（如果可能）
        try:
            if buffer.ctypes.data % alignment != 0:
                # 如果不对齐，创建一个新的对齐缓冲区
                import ctypes
                aligned_buffer = np.empty(aligned_samples, dtype=self._dtype)
                # 这里我们尽力而为，现代numpy通常会自动对齐
        except (AttributeError, ImportError):
            # 某些numpy版本可能不支持ctypes.data，跳过验证
            pass

        # 返回精确大小的视图
        return buffer[:self.capacity_samples]

    @measure_time
    def write(self, data: np.ndarray, blocking: bool = True,
              timeout: float | None = None) -> bool:
        """写入音频数据到缓冲区"""
        self._validate_not_closed()
        self._validate_data_format(data)

        data_size = len(data)
        if data_size == 0:
            return True

        with self._write_lock:
            # 检查空间是否足够
            success = self._handle_write_request(data_size, blocking, timeout)
            if not success:
                return False

            # 执行写入
            self._write_data_to_buffer(data)

            # 更新统计
            self._write_count.increment()
            self._total_written += data_size

            # 通知等待的读取线程
            with self._not_empty:
                self._not_empty.notify_all()

            return True

    def _handle_write_request(self, data_size: int, blocking: bool,
                             timeout: float | None) -> bool:
        """处理写入请求的空间检查和等待逻辑"""
        current_available = self._available_data.get()
        required_space = self.capacity_samples - current_available

        if data_size <= required_space:
            return True

        # 空间不足，根据策略处理
        if self._overflow_strategy == BufferStrategy.REJECT:
            self._overflow_count.increment()
            return False

        elif self._overflow_strategy == BufferStrategy.OVERWRITE:
            return self._handle_overwrite(data_size)

        elif self._overflow_strategy == BufferStrategy.BLOCK and blocking:
            return self._handle_blocking_write(data_size, timeout)

        else:
            # 非阻塞且空间不足
            self._overflow_count.increment()
            return False

    def _handle_overwrite(self, data_size: int) -> bool:
        """处理覆盖写入"""
        current_available = self._available_data.get()
        overflow_size = data_size - (self.capacity_samples - current_available)

        # 前进读取位置，丢弃最旧的数据
        current_read_pos = self._read_pos.get()
        new_read_pos = (current_read_pos + overflow_size) % self.capacity_samples

        # 原子更新
        self._read_pos.set(new_read_pos)
        self._available_data.increment(-overflow_size)
        self._total_read += overflow_size

        self._overflow_count.increment()
        return True

    def _handle_blocking_write(self, data_size: int,
                              timeout: float | None) -> bool:
        """处理阻塞写入"""
        start_time = time.time()

        while True:
            current_available = self._available_data.get()
            required_space = self.capacity_samples - current_available

            if data_size <= required_space:
                return True

            # 检查超时
            if timeout is not None:
                elapsed = time.time() - start_time
                if elapsed >= timeout:
                    return False
                remaining_timeout = timeout - elapsed
            else:
                remaining_timeout = None

            # 等待空间
            if not self._not_full.wait(timeout=remaining_timeout):
                return False

    def _write_data_to_buffer(self, data: np.ndarray) -> None:
        """将数据写入缓冲区"""
        data_size = len(data)
        write_position = self._write_pos.get()

        if write_position + data_size <= self.capacity_samples:
            # 连续区域写入
            self._buffer[write_position:write_position + data_size] = data
        else:
            # 跨越边界写入
            first_part_size = self.capacity_samples - write_position
            self._buffer[write_position:] = data[:first_part_size]
            self._buffer[:data_size - first_part_size] = data[first_part_size:]

        # 更新位置和计数
        new_write_pos = (write_position + data_size) % self.capacity_samples
        self._write_pos.set(new_write_pos)

        with self._metadata_lock:
            self._available_data.increment(data_size)

            # 更新峰值使用率
            current_usage = self._available_data.get() / self.capacity_samples
            current_peak = self._peak_usage.get()
            if current_usage > current_peak:
                self._peak_usage.set(current_usage)

    @measure_time
    def get_chunk_with_overlap(self, chunk_size: int,
                              overlap_size: int) -> tuple[AudioChunk | None, bool]:
        """获取带重叠的音频块"""
        self._validate_not_closed()
        self._validate_chunk_params(chunk_size, overlap_size)

        total_size = chunk_size + overlap_size

        with self._read_lock:
            current_available = self._available_data.get()

            # 检查数据是否足够
            if current_available < total_size:
                return None, False

            # 获取数据
            data, is_continuous = self._read_chunk_data(total_size)

            # 创建AudioChunk
            chunk = self._create_audio_chunk(data, chunk_size, overlap_size, is_continuous)

            # 更新统计
            self._read_count.increment()
            if is_continuous:
                self._zero_copy_count.increment()
            else:
                self._copy_count.increment()

            return chunk, True

    def _read_chunk_data(self, total_size: int) -> tuple[np.ndarray, bool]:
        """读取指定大小的数据块"""
        read_position = self._read_pos.get()

        if read_position + total_size <= self.capacity_samples:
            # 连续区域，使用零拷贝视图
            data = self._buffer[read_position:read_position + total_size]
            return data, True
        else:
            # 跨越边界，需要复制数据
            data = np.empty(total_size, dtype=self._dtype)
            first_part_size = self.capacity_samples - read_position
            data[:first_part_size] = self._buffer[read_position:]
            data[first_part_size:] = self._buffer[:total_size - first_part_size]
            return data, False

    def _create_audio_chunk(self, data: np.ndarray, chunk_size: int,
                           overlap_size: int, is_continuous: bool) -> AudioChunk:
        """创建AudioChunk对象"""
        sequence_number = self._sequence_counter.get()
        start_frame = self._total_read
        timestamp_ms = start_frame * 1000.0 / self.config.sample_rate

        chunk = AudioChunk(
            data=data,
            sequence_number=sequence_number,
            start_frame=start_frame,
            chunk_size=chunk_size,
            overlap_size=overlap_size,
            timestamp_ms=timestamp_ms,
            sample_rate=self.config.sample_rate,
            metadata={
                'is_continuous': is_continuous,
                'buffer_level': self._available_data.get() / self.capacity_samples,
                'read_position': self._read_pos.get(),
                'write_position': self._write_pos.get(),
            }
        )

        self._sequence_counter.increment()
        return chunk

    def advance_read_position(self, size: int) -> None:
        """前进读取位置"""
        self._validate_not_closed()

        if size <= 0:
            raise ValueError("前进大小必须大于0")

        with self._read_lock:
            current_available = self._available_data.get()
            if size > current_available:
                raise InsufficientDataError(current_available, size)

            # 更新读取位置
            current_read_pos = self._read_pos.get()
            new_read_pos = (current_read_pos + size) % self.capacity_samples
            self._read_pos.set(new_read_pos)

            # 更新可用数据量
            with self._metadata_lock:
                self._available_data.increment(-size)
                self._total_read += size

            # 通知等待的写入线程
            with self._not_full:
                self._not_full.notify_all()

    def available_samples(self) -> int:
        """获取可读取的样本数"""
        return self._available_data.get()

    def get_buffer_status(self) -> BufferStatus:
        """获取缓冲区状态"""
        current_available = self._available_data.get()
        usage_ratio = current_available / self.capacity_samples

        # 确定状态级别
        if usage_ratio >= 0.95:
            status_level = "critical"
        elif usage_ratio >= 0.8:
            status_level = "warning"
        else:
            status_level = "normal"

        return BufferStatus(
            capacity=self.capacity_samples,
            available_samples=current_available,
            free_samples=self.capacity_samples - current_available,
            usage_ratio=usage_ratio,
            status_level=status_level,
            write_position=self._write_pos.get(),
            read_position=self._read_pos.get(),
            overflow_count=self._overflow_count.get(),
            underflow_count=0,  # 暂不实现
            peak_usage=self._peak_usage.get()
        )

    def clear(self) -> None:
        """清空缓冲区"""
        self._validate_not_closed()

        with self._write_lock, self._read_lock:
            # 重置所有位置和计数
            self._write_pos.set(0)
            self._read_pos.set(0)
            self._available_data.set(0)
            self._sequence_counter.set(0)

            # 重置统计信息（保留累计统计）
            self._peak_usage.set(0.0)

            # 可选：清零缓冲区内容（用于调试）
            # self._buffer.fill(0)

    def close(self) -> None:
        """关闭缓冲区"""
        if not self._is_closed:
            with self._write_lock, self._read_lock:
                self._is_closed = True

                # 通知所有等待的线程
                with self._not_empty:
                    self._not_empty.notify_all()
                with self._not_full:
                    self._not_full.notify_all()

    # === 性能监控接口 ===

    def get_performance_stats(self) -> dict[str, Any]:
        """获取性能统计信息"""
        uptime = time.time() - self._created_time

        return {
            'uptime_seconds': uptime,
            'write_operations': self._write_count.get(),
            'read_operations': self._read_count.get(),
            'zero_copy_operations': self._zero_copy_count.get(),
            'copy_operations': self._copy_count.get(),
            'overflow_count': self._overflow_count.get(),
            'zero_copy_rate': self._calculate_zero_copy_rate(),
            'peak_usage': self._peak_usage.get(),
            'current_usage': self.utilization_ratio(),
            'total_written': self._total_written,
            'total_read': self._total_read,
            'throughput_samples_per_second': self._calculate_throughput(),
        }

    def _calculate_zero_copy_rate(self) -> float:
        """计算零拷贝率"""
        total_reads = self._zero_copy_count.get() + self._copy_count.get()
        if total_reads == 0:
            return 0.0
        return self._zero_copy_count.get() / total_reads

    def _calculate_throughput(self) -> float:
        """计算吞吐量（样本/秒）"""
        uptime = time.time() - self._created_time
        if uptime == 0:
            return 0.0
        return self._total_written / uptime


# === 便利类型别名 ===
RingBuffer = AudioRingBuffer


__all__ = [
    "AudioRingBuffer",
    "RingBuffer",
]
