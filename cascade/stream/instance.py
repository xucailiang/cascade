"""
Cascade处理实例

基于1:1:1绑定架构的单个处理实例，集成VAD状态机和音频处理。
使用FrameAlignedBuffer进行帧对齐处理，简化架构设计。
"""

import logging
import time

import numpy as np

from ..backends.silero import SileroVADBackend
from ..buffer.frame_aligned_buffer import FrameAlignedBuffer
from ..types import AudioChunk, VADConfig
from .state_machine import VADStateMachine
from .types import AUDIO_FRAME_SIZE, AudioFrame, CascadeResult, Config

logger = logging.getLogger(__name__)


class CascadeInstance:
    """
    Cascade处理实例
    
    实现1:1:1绑定架构：
    - 一个实例对应一个FrameAlignedBuffer
    - 一个VAD实例进行语音检测
    - 一个状态机管理语音段收集
    
    简化设计原则：
    - 无线程，同步处理
    - 无锁设计，单线程访问
    - 帧对齐优化，专注512样本帧
    """

    def __init__(self, instance_id: str, config: Config):
        """
        初始化Cascade实例
        
        Args:
            instance_id: 实例唯一标识
            config: 配置对象
        """
        self.instance_id = instance_id
        self.config = config

        # 1:1:1绑定：一个实例一个缓冲区（优化版：减小缓冲区大小）
        self.frame_buffer = FrameAlignedBuffer(max_buffer_samples=64000) 

        # 延迟初始化VAD后端
        self._vad_backend = None
        self._initialized = False
        # 确保 chunk_duration_ms 大于 speech_pad_ms，避免验证错误
        self._vad_config = VADConfig(
            threshold=config.vad_threshold,
            speech_pad_ms=config.speech_pad_ms,  # 使用 speech_pad_ms 参数
            min_silence_duration_ms=config.min_silence_duration_ms,
            chunk_duration_ms=max(500, config.speech_pad_ms * 2)  # 确保块时长足够大
        )

        # 立即初始化状态机（无需异步）
        self.state_machine = VADStateMachine(instance_id)

        # 统计信息
        self.frame_counter = 0
        self.total_frames_processed = 0
        self.total_processing_time_ms = 0.0
        self.error_count = 0

        logger.info(f"CascadeInstance {instance_id} 初始化完成")

    async def _ensure_initialized(self):
        """确保VAD后端已初始化"""
        if not self._initialized:
            self._vad_backend = SileroVADBackend(self._vad_config)
            await self._vad_backend.initialize()
            self._initialized = True
            logger.info(f"CascadeInstance {self.instance_id} VAD后端初始化完成")

    @property
    def vad_backend(self):
        """VAD后端属性，保持向后兼容"""
        return self._vad_backend

    def process_audio_chunk(self, audio_data: bytes) -> list[CascadeResult]:
        """
        处理音频块，返回VAD结果列表
        
        Args:
            audio_data: 音频数据（任意大小）
            
        Returns:
            处理结果列表
        """
        if not audio_data:
            return []

        results = []

        try:
            # 1. 写入帧对齐缓冲区
            self.frame_buffer.write(audio_data)

            # 2. 处理所有可用的完整帧
            while self.frame_buffer.has_complete_frame():
                frame_data = self.frame_buffer.read_frame()
                if frame_data:
                    result = self._process_single_frame(frame_data)
                    if result:
                        results.append(result)

        except Exception as e:
            self.error_count += 1
            logger.error(f"CascadeInstance {self.instance_id} 处理音频块失败: {e}")

        return results

    def _process_single_frame(self, frame_data: bytes) -> CascadeResult | None:
        """
        处理单个512样本帧
        
        Args:
            frame_data: 512样本的音频数据（1024字节）
            
        Returns:
            处理结果，可能为None
        """
        start_time = time.time()

        try:
            # 创建音频帧对象
            self.frame_counter += 1
            timestamp_ms = self.frame_counter * 32.0  # 32ms per frame

            # 转换为numpy数组进行VAD检测
            audio_array = np.frombuffer(frame_data, dtype=np.int16).astype(np.float32) / 32768.0

            # 创建AudioChunk用于VAD检测
            audio_chunk = AudioChunk(
                data=audio_array,
                sequence_number=self.frame_counter,
                start_frame=self.frame_counter * AUDIO_FRAME_SIZE,
                chunk_size=AUDIO_FRAME_SIZE,
                timestamp_ms=timestamp_ms,
                sample_rate=16000
            )

            # VAD检测（确保后端已初始化）
            if not self._initialized or self._vad_backend is None:
                logger.warning(f"CascadeInstance {self.instance_id} VAD后端未初始化，跳过处理")
                return None

            vad_result = self._vad_backend.process_chunk(audio_chunk)

            # 转换VAD结果为字典格式
            vad_dict = None
            if vad_result.original_result:
                vad_dict = vad_result.original_result

            # 创建AudioFrame
            frame = AudioFrame(
                frame_id=self.frame_counter,
                audio_data=frame_data,
                timestamp_ms=timestamp_ms,
                vad_result=vad_dict
            )

            # 状态机处理
            result = self.state_machine.process_frame(frame)

            # 更新统计
            processing_time_ms = (time.time() - start_time) * 1000
            self.total_frames_processed += 1
            self.total_processing_time_ms += processing_time_ms

            return result

        except Exception as e:
            self.error_count += 1
            logger.error(f"CascadeInstance {self.instance_id} 帧处理失败: {e}")
            return None

    def reset(self) -> None:
        """重置实例状态"""
        self.state_machine.reset()
        self.frame_buffer.clear()
        self.frame_counter = 0
        self.total_frames_processed = 0
        self.total_processing_time_ms = 0.0
        self.error_count = 0
        logger.info(f"CascadeInstance {self.instance_id} 重置")

    @property
    def buffer_usage_ratio(self) -> float:
        """缓冲区使用率"""
        return self.frame_buffer.get_buffer_usage_ratio()

    @property
    def available_frames(self) -> int:
        """可用帧数"""
        return self.frame_buffer.available_frames()

    @property
    def average_processing_time_ms(self) -> float:
        """平均处理时间"""
        if self.total_frames_processed == 0:
            return 0.0
        return self.total_processing_time_ms / self.total_frames_processed

    @property
    def error_rate(self) -> float:
        """错误率"""
        if self.total_frames_processed == 0:
            return 0.0
        return self.error_count / self.total_frames_processed

    def get_stats(self) -> dict:
        """获取实例统计信息"""
        return {
            'instance_id': self.instance_id,
            'total_frames_processed': self.total_frames_processed,
            'average_processing_time_ms': self.average_processing_time_ms,
            'error_count': self.error_count,
            'error_rate': self.error_rate,
            'buffer_usage_ratio': self.buffer_usage_ratio,
            'available_frames': self.available_frames,
        }

    def __str__(self) -> str:
        return f"CascadeInstance({self.instance_id}, frames={self.total_frames_processed})"
