"""
流式处理器核心模块

实现异步流式VAD处理编排器，管理多个Cascade实例。
基于1:1:1架构优化，简化实例管理。
"""
import logging
import os
import time
from collections.abc import AsyncIterator

import numpy as np
from silero_vad import read_audio

from .instance import CascadeInstance
from .types import CascadeResult, Config, ProcessorStats

logger = logging.getLogger(__name__)


class StreamProcessor:
    """
    流式处理器 - 1:1:1架构优化版本

    核心功能：
    - 管理多个CascadeInstance实例（简化版）
    - 提供异步流式处理接口
    - 实例池管理和复用
    - 统计信息收集
    """

    def __init__(self, config: Config=None):
        """
        初始化流式处理器

        Args:
            config: 处理器配置
        """
        if not config:
            from cascade.stream import create_default_config
            config = create_default_config()

        self.config = config
        self.instances: dict[str, CascadeInstance] = {}
        self.is_running = False
        self._auto_initialized = False  # 新增：自动初始化标志

        # 实例使用统计
        self.instance_last_used: dict[str, float] = {}

        # 统计信息
        self.total_chunks_processed = 0
        self.total_processing_time_ms = 0.0
        self.speech_segments_count = 0
        self.single_frames_count = 0
        self.error_count = 0

        # 性能监控
        self.processing_times = []  # 最近100次处理时间
        self.max_processing_times = 100

        logger.info(f"StreamProcessor 初始化，最大实例数: {config.max_instances}")

    async def start(self) -> None:
        """启动处理器并初始化所有实例"""
        if self.is_running:
            logger.warning("StreamProcessor 已在运行")
            return

        self.is_running = True

        # 初始化所有现有实例
        for instance_id, instance in list(self.instances.items()):
            if not instance._initialized:
                await instance._ensure_initialized()
                logger.info(f"实例 {instance_id} 已初始化")

        logger.info("StreamProcessor 启动完成")

    async def stop(self) -> None:
        """停止处理器"""
        if not self.is_running:
            return

        self.is_running = False

        # 清理所有实例（1:1:1架构下实例无需显式停止）
        self.instances.clear()
        self.instance_last_used.clear()

        logger.info("StreamProcessor 停止")

    async def _ensure_auto_initialized(self):
        """确保处理器已自动初始化"""
        if not self._auto_initialized:
            await self.start()
            self._auto_initialized = True

    async def process_file(self, file_path: str) -> AsyncIterator[CascadeResult]:
        """
        处理音频文件

        Args:
            file_path: 音频文件路径

        Yields:
            CascadeResult: 处理结果
        """
        # 自动初始化
        await self._ensure_auto_initialized()

        # 获取处理实例
        instance = await self._get_or_create_instance("file_processor")

        # 验证文件存在
        if not os.path.exists(file_path):
            from .. import AudioFormatError
            raise AudioFormatError(f"音频文件不存在: {file_path}")

        # 读取和处理音频文件
        audio_data = self._read_audio_file(file_path, target_sample_rate=16000)
        audio_frames = self._generate_audio_frames(audio_data)

        # 逐帧处理
        for frame_data in audio_frames:
            results = instance.process_audio_chunk(frame_data)
            for result in results:
                yield result

    async def process_stream(
        self,
        audio_stream: AsyncIterator[bytes],
        stream_id: str | None = None
    ) -> AsyncIterator[CascadeResult]:
        """
        处理音频流

        Args:
            audio_stream: 音频数据流
            stream_id: 流标识符，用于实例管理

        Yields:
            处理结果
        """
        # 自动初始化
        await self._ensure_auto_initialized()

        # 获取或创建处理实例
        instance = await self._get_or_create_instance(stream_id)

        try:
            # 处理音频流并同时收集结果
            async for audio_chunk in audio_stream:
                if not self.is_running:
                    break

                # 处理音频块并直接获取结果（1:1:1架构）
                start_time = time.time()
                results = instance.process_audio_chunk(audio_chunk)
                processing_time = (time.time() - start_time) * 1000

                self.total_chunks_processed += 1
                self._record_processing_time(processing_time)

                # 返回结果
                for result in results:
                    # 更新统计
                    if result.is_speech_segment:
                        self.speech_segments_count += 1
                    else:
                        self.single_frames_count += 1
                    self.total_processing_time_ms += result.processing_time_ms

                    yield result

        except Exception as e:
            self.error_count += 1
            logger.error(f"处理音频流失败: {e}")
            raise
        finally:
            # 清理实例
            if stream_id:
                await self._cleanup_instance(stream_id)

    async def process_chunk(self, audio_data: bytes) -> list[CascadeResult]:
        """
        处理单个音频块

        Args:
            audio_data: 音频数据

        Returns:
            处理结果列表
        """
        # 自动初始化
        await self._ensure_auto_initialized()

        # 获取可用实例
        instance = await self._get_available_instance()

        try:
            # 处理音频块（1:1:1架构直接返回结果）
            start_time = time.time()
            results = instance.process_audio_chunk(audio_data)
            processing_time = (time.time() - start_time) * 1000

            # 记录处理时间
            self._record_processing_time(processing_time)

            # 更新统计
            self.total_chunks_processed += 1
            for result in results:
                if result.is_speech_segment:
                    self.speech_segments_count += 1
                else:
                    self.single_frames_count += 1
                self.total_processing_time_ms += result.processing_time_ms

            return results

        except Exception as e:
            self.error_count += 1
            logger.error(f"处理音频块失败: {e}")
            raise

    def _read_audio_file(self, file_path: str, target_sample_rate: int = 16000):
        """
        读取音频文件

        Args:
            file_path: 音频文件路径
            target_sample_rate: 目标采样率

        Returns:
            音频数据数组
        """
        try:
            audio_data = read_audio(file_path, sampling_rate=target_sample_rate)
            return audio_data
        except ImportError as err:
            # 如果silero-vad不可用
            raise ImportError("silero-vad未安装，无法读取音频文件") from err


    def _generate_audio_frames(self, audio_data, frame_size: int = 512) -> list:
        """
        生成512样本的音频帧

        Args:
            audio_data: 音频数据
            frame_size: 帧大小（样本数）

        Returns:
            音频帧列表（bytes格式）
        """
        frames = []

        # 按512样本分块
        for i in range(0, len(audio_data), frame_size):
            frame = audio_data[i:i + frame_size]

            # 如果最后一帧不足512样本，跳过
            if len(frame) < frame_size:
                break

            # 如果是PyTorch Tensor，先转换为numpy数组
            if hasattr(frame, 'numpy') and callable(getattr(frame, 'numpy', None)):
                frame = frame.numpy()
            elif hasattr(frame, 'detach') and callable(getattr(frame, 'detach', None)):
                frame = frame.detach().numpy()

            frame_int16 = (frame * 32767).astype(np.int16)
            frame_bytes = frame_int16.tobytes()
            frames.append(frame_bytes)

        return frames


    async def _get_or_create_instance(self, stream_id: str | None) -> CascadeInstance:
        """获取或创建处理实例"""
        if stream_id is None:
            return await self._get_available_instance()

        if stream_id in self.instances:
            # 更新使用时间
            self.instance_last_used[stream_id] = time.time()
            return self.instances[stream_id]

        # 检查实例数量限制
        if len(self.instances) >= self.config.max_instances:
            # 清理最久未使用的实例
            await self._cleanup_oldest_instance()

        # 创建新实例
        try:
            instance = await self._create_instance(stream_id)
            self.instances[stream_id] = instance
            self.instance_last_used[stream_id] = time.time()

            logger.info(f"创建新实例: {stream_id}")
            return instance
        except Exception as e:
            logger.error(f"创建实例失败: {e}")
            # 降级到可用实例
            return await self._get_available_instance()

    async def _get_available_instance(self) -> CascadeInstance:
        """获取可用实例"""
        if not self.instances:
            # 创建默认实例
            instance_id = f"default-{int(time.time())}"
            try:
                instance = await self._create_instance(instance_id)
                self.instances[instance_id] = instance
                self.instance_last_used[instance_id] = time.time()
                return instance
            except Exception as e:
                logger.error(f"创建默认实例失败: {e}")
                raise RuntimeError("无法创建处理实例") from e

        # 简单轮询选择实例（1:1:1架构下实例都是等价的）
        return next(iter(self.instances.values()))

    async def _create_instance(self, instance_id: str) -> CascadeInstance:
        """创建新的处理实例并确保初始化完成"""
        instance = CascadeInstance(
            instance_id=instance_id,
            config=self.config
        )
        # 立即初始化VAD后端，确保实例可用
        await instance._ensure_initialized()
        logger.info(f"CascadeInstance {instance_id} 创建并完成初始化")
        return instance

    async def _cleanup_oldest_instance(self) -> None:
        """清理最久未使用的实例"""
        if not self.instance_last_used:
            return

        # 找到最久未使用的实例
        oldest_id = min(
            self.instance_last_used.keys(),
            key=lambda x: self.instance_last_used[x]
        )

        # 清理实例
        if oldest_id in self.instances:
            self.instances.pop(oldest_id)
            self.instance_last_used.pop(oldest_id, None)
            logger.info(f"清理最久未使用的实例: {oldest_id}")

    def _record_processing_time(self, processing_time_ms: float) -> None:
        """记录处理时间"""
        self.processing_times.append(processing_time_ms)

        # 保持最近的处理时间记录
        if len(self.processing_times) > self.max_processing_times:
            self.processing_times.pop(0)

    async def _cleanup_instance(self, stream_id: str) -> None:
        """清理实例"""
        if stream_id in self.instances:
            self.instances.pop(stream_id)
            self.instance_last_used.pop(stream_id, None)
            logger.info(f"清理实例: {stream_id}")

    def get_stats(self) -> ProcessorStats:
        """获取处理器统计信息"""
        # 计算平均处理时间
        avg_processing_time = 0.0
        if self.total_chunks_processed > 0:
            avg_processing_time = self.total_processing_time_ms / self.total_chunks_processed

        # 计算语音比例
        total_results = self.speech_segments_count + self.single_frames_count
        speech_ratio = 0.0
        if total_results > 0:
            speech_ratio = self.speech_segments_count / total_results

        # 计算吞吐量
        throughput = 0.0
        if self.total_processing_time_ms > 0:
            throughput = self.total_chunks_processed / (self.total_processing_time_ms / 1000.0)

        # 计算错误率
        error_rate = 0.0
        if self.total_chunks_processed > 0:
            error_rate = self.error_count / self.total_chunks_processed

        # 估算内存使用
        memory_usage_mb = len(self.instances) * 50.0  # 粗略估算每个实例50MB

        return ProcessorStats(
            total_chunks_processed=self.total_chunks_processed,
            total_processing_time_ms=self.total_processing_time_ms,
            average_processing_time_ms=avg_processing_time,
            speech_segments=self.speech_segments_count,
            single_frames=self.single_frames_count,
            speech_ratio=speech_ratio,
            throughput_chunks_per_second=throughput,
            memory_usage_mb=memory_usage_mb,
            error_count=self.error_count,
            error_rate=error_rate
        )

    def reset_stats(self) -> None:
        """重置统计信息"""
        self.total_chunks_processed = 0
        self.total_processing_time_ms = 0.0
        self.speech_segments_count = 0
        self.single_frames_count = 0
        self.error_count = 0
        logger.info("统计信息已重置")

    @property
    def active_instances(self) -> int:
        """活跃实例数"""
        return len(self.instances)

    @property
    def is_busy(self) -> bool:
        """是否繁忙（1:1:1架构下简化判断）"""
        return len(self.instances) >= self.config.max_instances

    def __str__(self) -> str:
        status = "running" if self.is_running else "stopped"
        return f"StreamProcessor({status}, instances={self.active_instances})"

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop()
