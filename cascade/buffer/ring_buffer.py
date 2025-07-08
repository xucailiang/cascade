"""
环形音频缓冲区实现

本模块实现了基于环形缓冲区的音频缓冲区，提供高效的音频数据管理。
"""

import threading

import numpy as np

from cascade._internal.atomic import AtomicCounter, AtomicValue
from cascade.buffer.base import AudioBuffer, BufferFullError, BufferStrategy


class RingBuffer(AudioBuffer):
    """
    环形音频缓冲区
    
    基于环形缓冲区设计的音频缓冲区实现，提供线程安全的高效音频数据管理。
    """

    def __init__(self, capacity_seconds: float, sample_rate: int, channels: int = 1,
                 dtype=np.float32, strategy: BufferStrategy = BufferStrategy.BLOCK):
        """
        初始化环形音频缓冲区
        
        Args:
            capacity_seconds: 缓冲区容量（秒）
            sample_rate: 采样率
            channels: 通道数，默认为1
            dtype: 数据类型，默认为np.float32
            strategy: 缓冲区溢出处理策略，默认为BLOCK
        """
        # 基本参数
        self.sample_rate = sample_rate
        self.channels = channels
        self.dtype = dtype
        self.strategy = strategy

        # 计算对齐后的容量
        alignment = 64  # 缓存行大小，通常为64字节
        element_size = np.dtype(dtype).itemsize
        self.capacity = (int(capacity_seconds * sample_rate * channels) * element_size + alignment - 1) // alignment * alignment // element_size

        # 创建缓冲区
        self.buffer = np.zeros(self.capacity, dtype=dtype)

        # 位置和计数器
        self.write_pos = AtomicCounter(0)
        self.read_pos = AtomicCounter(0)
        self.available_data = AtomicCounter(0)

        # 序列计数器和总帧计数
        self.sequence_counter = AtomicCounter(0)
        self.total_frames_read = AtomicCounter(0)

        # 锁和事件
        self._buffer_lock = threading.RLock()
        self._event_not_empty = threading.Event()
        self._event_not_full = threading.Event()
        self._event_not_full.set()  # 初始状态为非满

        # 状态标志
        self._closed = AtomicValue(False)

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
            ValueError: 当缓冲区已关闭时
        """
        if self._closed.get():
            raise ValueError("缓冲区已关闭")

        data_size = len(data)
        if data_size == 0:
            return 0

        with self._buffer_lock:
            # 检查是否有足够的空间
            if self.available_data.get() + data_size > self.capacity:
                # 根据策略处理
                if self.strategy == BufferStrategy.BLOCK:
                    # 阻塞等待
                    self._event_not_full.clear()
                    self._buffer_lock.release()
                    if not self._event_not_full.wait(timeout):
                        self._buffer_lock.acquire()
                        raise TimeoutError("写入操作超时")
                    self._buffer_lock.acquire()

                    # 再次检查空间（可能在等待期间被其他线程写入）
                    if self.available_data.get() + data_size > self.capacity:
                        return 0  # 仍然没有足够空间
                elif self.strategy == BufferStrategy.OVERWRITE:
                    # 覆盖最旧数据
                    overwrite_size = self.available_data.get() + data_size - self.capacity
                    self.read_pos.increment(overwrite_size)
                    self.read_pos.set(self.read_pos.get() % self.capacity)
                    self.available_data.decrement(overwrite_size)
                elif self.strategy == BufferStrategy.REJECT:
                    # 拒绝新数据
                    raise BufferFullError("缓冲区已满，数据被拒绝")

            # 写入数据
            write_pos = self.write_pos.get()
            if write_pos + data_size <= self.capacity:
                # 连续区域
                self.buffer[write_pos:write_pos + data_size] = data
            else:
                # 跨越边界
                first_part_size = self.capacity - write_pos
                self.buffer[write_pos:] = data[:first_part_size]
                self.buffer[:data_size - first_part_size] = data[first_part_size:]

            # 更新写入位置
            self.write_pos.increment(data_size)
            self.write_pos.set(self.write_pos.get() % self.capacity)
            self.available_data.increment(data_size)

            # 通知等待的读取线程
            if self.available_data.get() > 0:
                self._event_not_empty.set()

            return data_size

    def read(self, size: int, timeout: float | None = None) -> np.ndarray:
        """
        从缓冲区读取指定大小的数据
        
        Args:
            size: 要读取的数据大小（样本数）
            timeout: 超时时间（秒），None表示无限等待
            
        Returns:
            读取的音频数据
            
        Raises:
            ValueError: 当请求的大小无效或缓冲区已关闭时
            TimeoutError: 当等待超时时
        """
        if self._closed.get():
            raise ValueError("缓冲区已关闭")

        if size <= 0:
            raise ValueError("读取大小必须大于0")

        with self._buffer_lock:
            # 等待足够的数据
            if self.available_data.get() < size:
                self._event_not_empty.clear()
                self._buffer_lock.release()
                if not self._event_not_empty.wait(timeout):
                    self._buffer_lock.acquire()
                    raise TimeoutError("读取操作超时")
                self._buffer_lock.acquire()

                # 再次检查数据量（可能在等待期间被其他线程读取）
                if self.available_data.get() < size:
                    raise BufferError("可用数据不足")

            # 读取数据
            read_pos = self.read_pos.get()
            if read_pos + size <= self.capacity:
                # 连续区域
                data = self.buffer[read_pos:read_pos + size].copy()
            else:
                # 跨越边界
                data = np.empty(size, dtype=self.dtype)
                first_part_size = self.capacity - read_pos
                data[:first_part_size] = self.buffer[read_pos:]
                data[first_part_size:] = self.buffer[:size - first_part_size]

            # 更新读取位置
            self.read_pos.increment(size)
            self.read_pos.set(self.read_pos.get() % self.capacity)
            self.available_data.decrement(size)
            self.total_frames_read.increment(size)

            # 通知等待的写入线程
            if self.available_data.get() < self.capacity:
                self._event_not_full.set()

            return data

    def get_chunk(self, chunk_size: int, overlap_size: int = 0) -> np.ndarray:
        """
        获取指定大小的数据块，可以包含重叠区域
        
        Args:
            chunk_size: 主要块大小（样本数）
            overlap_size: 重叠区域大小（样本数）
            
        Returns:
            包含重叠区域的音频数据
            
        Raises:
            ValueError: 当请求的大小无效或缓冲区已关闭时
            BufferError: 当可用数据不足时
        """
        if self._closed.get():
            raise ValueError("缓冲区已关闭")

        if chunk_size <= 0:
            raise ValueError("块大小必须大于0")

        if overlap_size < 0:
            raise ValueError("重叠大小不能为负数")

        if overlap_size >= chunk_size:
            raise ValueError("重叠大小不能大于等于块大小")

        total_size = chunk_size + overlap_size

        with self._buffer_lock:
            # 检查可用数据
            if self.available_data.get() < total_size:
                raise BufferError("可用数据不足")

            # 获取数据
            read_pos = self.read_pos.get()
            if read_pos + total_size <= self.capacity:
                # 连续区域
                data = self.buffer[read_pos:read_pos + total_size].copy()
            else:
                # 跨越边界
                data = np.empty(total_size, dtype=self.dtype)
                first_part_size = self.capacity - read_pos
                data[:first_part_size] = self.buffer[read_pos:]
                data[first_part_size:] = self.buffer[:total_size - first_part_size]

            # 更新读取位置（只移动主要块大小，保留重叠区域）
            self.read_pos.increment(chunk_size)
            self.read_pos.set(self.read_pos.get() % self.capacity)
            self.available_data.decrement(chunk_size)
            self.total_frames_read.increment(chunk_size)

            # 通知等待的写入线程
            if self.available_data.get() < self.capacity:
                self._event_not_full.set()

            return data

    def get_chunk_with_overlap(self, chunk_size: int, overlap_size: int) -> tuple[np.ndarray, dict]:
        """
        获取指定大小的音频块，包含重叠区域和元数据
        
        Args:
            chunk_size: 主要块大小（样本数）
            overlap_size: 重叠区域大小（样本数）
            
        Returns:
            包含重叠区域的音频数据和元数据字典
            
        Raises:
            ValueError: 当请求的大小无效或缓冲区已关闭时
            BufferError: 当可用数据不足时
        """
        if self._closed.get():
            raise ValueError("缓冲区已关闭")

        if chunk_size <= 0:
            raise ValueError("块大小必须大于0")

        if overlap_size < 0:
            raise ValueError("重叠大小不能为负数")

        if overlap_size >= chunk_size:
            raise ValueError("重叠大小不能大于等于块大小")

        total_size = chunk_size + overlap_size

        with self._buffer_lock:
            # 检查可用数据
            if self.available_data.get() < total_size:
                raise BufferError("可用数据不足")

            # 获取数据
            read_pos = self.read_pos.get()
            if read_pos + total_size <= self.capacity:
                # 连续区域
                data = self.buffer[read_pos:read_pos + total_size].copy()
            else:
                # 跨越边界
                data = np.empty(total_size, dtype=self.dtype)
                first_part_size = self.capacity - read_pos
                data[:first_part_size] = self.buffer[read_pos:]
                data[first_part_size:] = self.buffer[:total_size - first_part_size]

            # 创建块元数据
            sequence_number = self.sequence_counter.get()
            start_frame = self.total_frames_read.get()
            timestamp_ms = start_frame * 1000.0 / self.sample_rate

            metadata = {
                "sequence_number": sequence_number,
                "start_frame": start_frame,
                "chunk_size": chunk_size,
                "overlap_size": overlap_size,
                "is_last": False,  # 默认不是最后一块
                "timestamp_ms": timestamp_ms,
                "sample_rate": self.sample_rate,
                "dtype": str(self.dtype),
                "channels": self.channels
            }

            # 更新序列计数器
            self.sequence_counter.increment()

            # 更新读取位置（只移动主要块大小，保留重叠区域）
            self.read_pos.increment(chunk_size)
            self.read_pos.set(self.read_pos.get() % self.capacity)
            self.available_data.decrement(chunk_size)
            self.total_frames_read.increment(chunk_size)

            # 通知等待的写入线程
            if self.available_data.get() < self.capacity:
                self._event_not_full.set()

            return data, metadata

    def advance_after_processing(self, chunk_size: int) -> None:
        """
        处理完成后前进读取位置
        
        Args:
            chunk_size: 已处理的块大小（不包括重叠区域）
            
        Raises:
            ValueError: 当请求的大小无效或缓冲区已关闭时
        """
        if self._closed.get():
            raise ValueError("缓冲区已关闭")

        if chunk_size <= 0:
            raise ValueError("块大小必须大于0")

        with self._buffer_lock:
            # 更新读取位置
            self.read_pos.increment(chunk_size)
            self.read_pos.set(self.read_pos.get() % self.capacity)
            self.available_data.decrement(chunk_size)
            self.total_frames_read.increment(chunk_size)

            # 通知等待的写入线程
            if self.available_data.get() < self.capacity:
                self._event_not_full.set()

    def peek(self, size: int) -> np.ndarray:
        """
        查看数据但不移动读取位置
        
        Args:
            size: 要查看的数据大小（样本数）
            
        Returns:
            查看的音频数据
            
        Raises:
            ValueError: 当请求的大小无效或缓冲区已关闭时
            BufferError: 当可用数据不足时
        """
        if self._closed.get():
            raise ValueError("缓冲区已关闭")

        if size <= 0:
            raise ValueError("查看大小必须大于0")

        with self._buffer_lock:
            # 检查可用数据
            if self.available_data.get() < size:
                raise BufferError("可用数据不足")

            # 查看数据
            read_pos = self.read_pos.get()
            if read_pos + size <= self.capacity:
                # 连续区域
                data = self.buffer[read_pos:read_pos + size].copy()
            else:
                # 跨越边界
                data = np.empty(size, dtype=self.dtype)
                first_part_size = self.capacity - read_pos
                data[:first_part_size] = self.buffer[read_pos:]
                data[first_part_size:] = self.buffer[:size - first_part_size]

            return data

    def skip(self, size: int) -> None:
        """
        跳过指定大小的数据
        
        Args:
            size: 要跳过的数据大小（样本数）
            
        Raises:
            ValueError: 当请求的大小无效或缓冲区已关闭时
        """
        if self._closed.get():
            raise ValueError("缓冲区已关闭")

        if size <= 0:
            raise ValueError("跳过大小必须大于0")

        with self._buffer_lock:
            # 检查可用数据
            size = min(size, self.available_data.get())  # 只跳过可用的数据

            # 更新读取位置
            self.read_pos.increment(size)
            self.read_pos.set(self.read_pos.get() % self.capacity)
            self.available_data.decrement(size)
            self.total_frames_read.increment(size)

            # 通知等待的写入线程
            if self.available_data.get() < self.capacity:
                self._event_not_full.set()

    def available(self) -> int:
        """
        返回可读取的数据量
        
        Returns:
            可读取的数据量（样本数）
        """
        return self.available_data.get()

    def remaining(self) -> int:
        """
        返回剩余可写入的空间
        
        Returns:
            剩余可写入的空间（样本数）
        """
        return self.capacity - self.available_data.get()

    def clear(self) -> None:
        """清空缓冲区"""
        with self._buffer_lock:
            self.read_pos.set(0)
            self.write_pos.set(0)
            self.available_data.set(0)
            # 不重置序列计数器和总帧计数，以保持连续性

            # 通知等待的写入线程
            self._event_not_full.set()

    def is_empty(self) -> bool:
        """
        检查缓冲区是否为空
        
        Returns:
            缓冲区是否为空
        """
        return self.available_data.get() == 0

    def is_full(self) -> bool:
        """
        检查缓冲区是否已满
        
        Returns:
            缓冲区是否已满
        """
        return self.available_data.get() == self.capacity

    def close(self) -> None:
        """关闭缓冲区，释放资源"""
        with self._buffer_lock:
            self._closed.set(True)
            self._event_not_empty.set()  # 唤醒所有等待的线程
            self._event_not_full.set()

    def write_batch(self, data_batch: list[np.ndarray], timeout: float | None = None) -> int:
        """
        批量写入数据到缓冲区
        
        Args:
            data_batch: 要写入的音频数据列表
            timeout: 超时时间（秒），None表示无限等待
            
        Returns:
            实际写入的数据量
            
        Raises:
            BufferFullError: 当缓冲区已满且策略为REJECT时
            TimeoutError: 当等待超时时
            ValueError: 当缓冲区已关闭时
        """
        if self._closed.get():
            raise ValueError("缓冲区已关闭")

        if not data_batch:
            return 0

        total_size = sum(len(data) for data in data_batch)
        if total_size == 0:
            return 0

        with self._buffer_lock:
            # 检查是否有足够的空间
            if self.available_data.get() + total_size > self.capacity:
                # 根据策略处理
                if self.strategy == BufferStrategy.BLOCK:
                    # 阻塞等待
                    self._event_not_full.clear()
                    self._buffer_lock.release()
                    if not self._event_not_full.wait(timeout):
                        self._buffer_lock.acquire()
                        raise TimeoutError("写入操作超时")
                    self._buffer_lock.acquire()

                    # 再次检查空间
                    if self.available_data.get() + total_size > self.capacity:
                        return 0  # 仍然没有足够空间
                elif self.strategy == BufferStrategy.OVERWRITE:
                    # 覆盖最旧数据
                    overwrite_size = self.available_data.get() + total_size - self.capacity
                    self.read_pos.increment(overwrite_size)
                    self.read_pos.set(self.read_pos.get() % self.capacity)
                    self.available_data.decrement(overwrite_size)
                elif self.strategy == BufferStrategy.REJECT:
                    # 拒绝新数据
                    raise BufferFullError("缓冲区已满，数据被拒绝")

            # 批量写入
            current_pos = self.write_pos.get()
            written_size = 0

            for data in data_batch:
                data_size = len(data)
                if data_size == 0:
                    continue

                if current_pos + data_size <= self.capacity:
                    # 连续区域
                    self.buffer[current_pos:current_pos + data_size] = data
                else:
                    # 跨越边界
                    first_part_size = self.capacity - current_pos
                    self.buffer[current_pos:] = data[:first_part_size]
                    self.buffer[:data_size - first_part_size] = data[first_part_size:]

                current_pos = (current_pos + data_size) % self.capacity
                written_size += data_size

            # 更新写入位置
            self.write_pos.set(current_pos)
            self.available_data.increment(written_size)

            # 通知等待的读取线程
            if self.available_data.get() > 0:
                self._event_not_empty.set()

            return written_size
