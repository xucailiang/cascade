"""
512样本帧对齐缓冲区 - 1:1:1架构专用版本

基于1:1:1绑定架构设计的简洁缓冲区，专门针对Silero VAD的512样本帧要求优化。
无需考虑多线程安全性，专注核心功能实现。

核心特性：
- 无锁设计：单线程访问，无需同步机制
- 帧对齐优化：专门针对512样本帧
- 简洁实用：避免过度设计
- 高性能：使用bytearray，支持动态扩容
"""

import logging

logger = logging.getLogger(__name__)


class FrameAlignedBuffer:
    """
    512样本帧对齐缓冲区 - 1:1:1架构专用版本
    
    特点：
    - 单线程访问，无锁设计
    - 专门针对512样本帧优化
    - 简洁实用，避免过度设计
    
    设计原则：
    - 基于Silero VAD的512样本帧要求
    - 不足512样本的帧会被保留在缓冲区
    - 避免补0操作，确保帧的完整性
    """

    def __init__(self, max_buffer_samples: int = 32000):
        """
        初始化帧对齐缓冲区
        
        Args:
            max_buffer_samples: 最大缓冲样本数，防止内存无限增长
                               默认16000样本（1秒@16kHz）
        """
        self._buffer = bytearray()  # 内部字节缓冲区，无锁设计
        self._max_buffer_size = max_buffer_samples * 2  # 16bit = 2字节/样本
        self._frame_size_bytes = 512 * 2  # 512样本 * 2字节 = 1024字节
        self._samples_per_frame = 512

        logger.debug(f"FrameAlignedBuffer初始化: max_buffer_size={self._max_buffer_size}字节")

    def write(self, audio_data: bytes) -> None:
        """
        写入音频数据到缓冲区
        
        Args:
            audio_data: 音频数据（任意大小）
        """
        if not audio_data:
            return

        # 直接追加到缓冲区
        self._buffer.extend(audio_data)

        # 防止缓冲区无限增长
        if len(self._buffer) > self._max_buffer_size:
            # 保留最新的数据，丢弃最旧的数据
            excess = len(self._buffer) - self._max_buffer_size
            self._buffer = self._buffer[excess:]
            logger.warning(f"缓冲区溢出，丢弃{excess}字节旧数据")

    def has_complete_frame(self) -> bool:
        """
        检查是否有完整的512样本帧
        
        Returns:
            True如果有完整帧可读，False否则
        """
        return len(self._buffer) >= self._frame_size_bytes

    def read_frame(self) -> bytes | None:
        """
        读取一个完整的512样本帧
        
        Returns:
            512样本的音频数据（1024字节），如果不足则返回None
        """
        if not self.has_complete_frame():
            return None

        # 提取512样本帧
        frame_data = bytes(self._buffer[:self._frame_size_bytes])

        # 从缓冲区移除已读取的数据
        self._buffer = self._buffer[self._frame_size_bytes:]

        return frame_data

    def available_samples(self) -> int:
        """
        返回缓冲区中可用的样本数
        
        Returns:
            可用样本数
        """
        return len(self._buffer) // 2  # 2字节/样本

    def available_frames(self) -> int:
        """
        返回缓冲区中可用的完整帧数
        
        Returns:
            可用的完整帧数
        """
        return len(self._buffer) // self._frame_size_bytes

    def clear(self) -> None:
        """清空缓冲区"""
        self._buffer.clear()
        logger.debug("FrameAlignedBuffer已清空")

    def get_buffer_usage_ratio(self) -> float:
        """
        获取缓冲区使用率
        
        Returns:
            使用率 (0.0-1.0)
        """
        return len(self._buffer) / self._max_buffer_size

    @property
    def buffer_size_bytes(self) -> int:
        """当前缓冲区大小（字节）"""
        return len(self._buffer)

    @property
    def max_buffer_size_bytes(self) -> int:
        """最大缓冲区大小（字节）"""
        return self._max_buffer_size

    @property
    def frame_size_bytes(self) -> int:
        """帧大小（字节）"""
        return self._frame_size_bytes

    @property
    def samples_per_frame(self) -> int:
        """每帧样本数"""
        return self._samples_per_frame

    def __str__(self) -> str:
        return (f"FrameAlignedBuffer(size={len(self._buffer)}B, "
                f"frames={self.available_frames()}, "
                f"usage={self.get_buffer_usage_ratio():.1%})")


__all__ = ["FrameAlignedBuffer"]
