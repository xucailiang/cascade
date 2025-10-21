"""
流式处理器核心模块 - 简化版

实现1:1:1:1架构：1个StreamProcessor = 1个独立VAD模型 + VADIterator + Buffer + 状态机
完全独立的实例设计，无锁无竞争，真正的简洁高效。
"""
import asyncio
import logging
import os
import time
from collections.abc import AsyncIterator
from typing import Optional

import numpy as np
import torch
from silero_vad import read_audio

from ..buffer.frame_aligned_buffer import FrameAlignedBuffer
from ..types.errors import (
    CascadeError,
    ErrorCode,
    ErrorSeverity,
    AudioFormatError,
)
from .state_machine import VADStateMachine
from .types import (
    AUDIO_FRAME_SIZE,
    AudioFrame,
    CascadeResult,
    Config,
    ProcessorStats,
)

logger = logging.getLogger(__name__)


class StreamProcessor:
    """
    流式处理器 - 简化的1:1:1:1架构
    
    每个实例对应一个WebSocket连接，拥有完全独立的VAD模型和组件。
    无锁无竞争设计，真正的简洁高效。
    
    核心组件：
    - 独立的VAD模型实例
    - 独立的VADIterator实例
    - FrameAlignedBuffer（同步，快速）
    - VADStateMachine（同步，快速）
    
    处理流程：
    1. 音频数据写入FrameAlignedBuffer（同步）
    2. 读取完整512样本帧（同步）
    3. VAD推理（使用asyncio.to_thread在线程池中执行）
    4. 状态机处理（同步）
    5. 返回结果
    """

    def __init__(self, config: Optional[Config] = None):
        """
        初始化流式处理器（不执行实际初始化）
        
        Args:
            config: 处理器配置
        """
        if not config:
            from cascade.stream import create_default_config
            config = create_default_config()
        
        # 验证配置
        self._validate_config(config)
        
        self.config = config
        
        # 1:1:1:1绑定组件
        self.frame_buffer = FrameAlignedBuffer(max_buffer_samples=128000)
        self.state_machine = VADStateMachine("stream_processor")
        
        # VAD组件（延迟初始化）
        self.model = None
        self.vad_iterator = None
        
        # 统计信息
        self.frame_counter = 0
        self.total_chunks_processed = 0
        self.total_processing_time_ms = 0.0
        self.speech_segments_count = 0
        self.single_frames_count = 0
        self.error_count = 0
        
        # 性能监控
        self.processing_times = []  # 最近100次处理时间
        self.max_processing_times = 100
        
        self.is_initialized = False
        
        logger.info("StreamProcessor创建完成（简化版1:1:1:1架构）")
    
    def _validate_config(self, config: Config) -> None:
        """验证配置"""
        if config.vad_threshold < 0 or config.vad_threshold > 1:
            raise CascadeError(
                f"vad_threshold必须在0-1之间，当前值: {config.vad_threshold}",
                ErrorCode.INVALID_CONFIG,
                ErrorSeverity.HIGH
            )
    
    async def initialize(self) -> None:
        """
        异步初始化VAD组件
        
        关键：每个实例加载自己的独立模型，避免并发问题
        """
        if self.is_initialized:
            logger.warning("StreamProcessor已经初始化")
            return
        
        try:
            # 加载独立的VAD模型（在线程池中执行）
            from silero_vad import load_silero_vad, VADIterator
            
            logger.info("开始加载独立VAD模型...")
            self.model = await asyncio.to_thread(
                load_silero_vad,
                onnx=False  # 使用PyTorch模式
            )
            
            # 创建独立的VADIterator实例
            self.vad_iterator = VADIterator(
                self.model,  # 使用独立模型
                sampling_rate=16000,
                threshold=self.config.vad_threshold,
                min_silence_duration_ms=self.config.min_silence_duration_ms,
                speech_pad_ms=self.config.speech_pad_ms
            )
            
            self.is_initialized = True
            logger.info("StreamProcessor初始化完成（独立模型已加载）")
            
        except Exception as e:
            logger.error(f"StreamProcessor初始化失败: {e}")
            raise CascadeError(
                f"初始化失败: {e}",
                ErrorCode.INITIALIZATION_FAILED,
                ErrorSeverity.HIGH
            ) from e
    
    async def process_chunk(self, audio_data: bytes) -> list[CascadeResult]:
        """
        处理音频块
        
        Args:
            audio_data: 音频数据（任意大小）
            
        Returns:
            处理结果列表
        """
        if self.vad_iterator is None:
            raise CascadeError(
                "StreamProcessor未初始化，请先调用initialize()",
                ErrorCode.INVALID_STATE,
                ErrorSeverity.HIGH
            )
        
        results = []
        start_time = time.time()
        
        try:
            # 1. 写入缓冲区（同步，快速）
            self.frame_buffer.write(audio_data)
            
            # 2. 处理所有完整帧
            while self.frame_buffer.has_complete_frame():
                # 读取帧（同步，快速）
                frame_data = self.frame_buffer.read_frame()
                
                if not frame_data:
                    break
                
                # 准备数据（同步，快速）
                audio_array = np.frombuffer(
                    frame_data,
                    dtype=np.int16
                ).astype(np.float32) / 32768.0
                
                audio_tensor = torch.from_numpy(audio_array)
                
                # VAD推理（异步，CPU密集型，在线程池中执行）
                # 由于每个StreamProcessor有独立的model和vad_iterator，
                # 多个StreamProcessor可以并发调用，互不干扰
                vad_result = await asyncio.to_thread(
                    self.vad_iterator,
                    audio_tensor
                )
                
                # 状态机处理（同步，快速逻辑）
                self.frame_counter += 1
                timestamp_ms = self.frame_counter * 32.0  # 32ms per frame
                
                frame = AudioFrame(
                    frame_id=self.frame_counter,
                    audio_data=frame_data,
                    timestamp_ms=timestamp_ms,
                    vad_result=vad_result
                )
                
                result = self.state_machine.process_frame(frame)
                
                if result:
                    # 更新统计
                    if result.is_speech_segment:
                        self.speech_segments_count += 1
                    else:
                        self.single_frames_count += 1
                    
                    results.append(result)
            
            # 记录处理时间
            processing_time = (time.time() - start_time) * 1000
            self.total_chunks_processed += 1
            self.total_processing_time_ms += processing_time
            self._record_processing_time(processing_time)
            
            return results
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"处理音频块失败: {e}")
            raise CascadeError(
                f"处理音频块失败: {e}",
                ErrorCode.PROCESSING_FAILED,
                ErrorSeverity.HIGH
            ) from e
    
    async def process_file(self, file_path: str) -> AsyncIterator[CascadeResult]:
        """
        处理音频文件
        
        Args:
            file_path: 音频文件路径
            
        Yields:
            CascadeResult: 处理结果
        """
        # 确保已初始化
        if not self.is_initialized:
            await self.initialize()
        
        try:
            # 验证文件存在
            if not os.path.exists(file_path):
                raise CascadeError(
                    f"音频文件不存在: {file_path}",
                    ErrorCode.INVALID_INPUT,
                    ErrorSeverity.HIGH,
                    {"file_path": file_path}
                )
            
            # 读取音频文件
            audio_data = self._read_audio_file(file_path, target_sample_rate=16000)
            audio_frames = self._generate_audio_frames(audio_data)
            
            # 逐帧处理
            for frame_data in audio_frames:
                results = await self.process_chunk(frame_data)
                for result in results:
                    yield result
                    
        except Exception as e:
            raise CascadeError(
                f"处理音频文件失败: {e}",
                ErrorCode.PROCESSING_FAILED,
                ErrorSeverity.HIGH,
                {"file_path": file_path}
            ) from e
    
    async def process_stream(
        self,
        audio_stream: AsyncIterator[bytes]
    ) -> AsyncIterator[CascadeResult]:
        """
        处理音频流
        
        Args:
            audio_stream: 音频数据流
            
        Yields:
            处理结果
        """
        # 确保已初始化
        if not self.is_initialized:
            await self.initialize()
        
        try:
            async for audio_chunk in audio_stream:
                results = await self.process_chunk(audio_chunk)
                for result in results:
                    yield result
                    
        except Exception as e:
            raise CascadeError(
                f"处理音频流失败: {e}",
                ErrorCode.PROCESSING_FAILED,
                ErrorSeverity.HIGH
            ) from e
    
    async def close(self) -> None:
        """清理资源"""
        if self.vad_iterator:
            self.vad_iterator.reset_states()
        
        # 清理模型（Python GC会自动处理）
        self.model = None
        self.vad_iterator = None
        
        # 清理其他组件
        self.frame_buffer.clear()
        self.state_machine.reset()
        
        self.is_initialized = False
        logger.info("StreamProcessor已清理")
    
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
    
    def _record_processing_time(self, processing_time_ms: float) -> None:
        """记录处理时间"""
        self.processing_times.append(processing_time_ms)
        
        # 保持最近的处理时间记录
        if len(self.processing_times) > self.max_processing_times:
            self.processing_times.pop(0)
    
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
        
        # 估算内存使用（单实例约80MB）
        memory_usage_mb = 80.0 if self.is_initialized else 0.0
        
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
        self.processing_times.clear()
        logger.info("统计信息已重置")
    
    def __str__(self) -> str:
        status = "initialized" if self.is_initialized else "not initialized"
        return f"StreamProcessor({status}, frames={self.frame_counter})"
    
    async def __aenter__(self):
        """异步上下文管理器入口"""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器退出"""
        await self.close()
