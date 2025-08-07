"""
Cascade主入口类 - 1:1:1绑定架构实现

这是Cascade VAD库的核心类，实现了完整的1:1:1绑定架构：
- 1个Cascade实例
- 1个VAD Backend实例
- 1个处理线程/协程
- 环形缓冲区和音频帧切分逻辑

设计原则：
- 极简配置：只需3个核心参数
- 状态连续性：确保有状态VAD模型(如silero-vad)的上下文连续性
- 环形缓冲区：高效音频数据存储和零拷贝访问
- 帧切分处理：将音频流切分为固定大小的帧，支持重叠处理
- 完整输出：返回原始音频+VAD检测结果
"""

import asyncio
import logging
import threading
import time
from typing import AsyncGenerator, Any, Union, Optional, Tuple

import numpy as np

from .types import (
    CascadeConfig, CascadeVADResult, AudioChunk, AudioConfig, VADConfig,
    CascadeError, ErrorCode, BufferStrategy
)
from .backends import create_vad_backend, VADBackend
from .formats import AudioFormatProcessor
from .buffer import AudioRingBuffer

# 创建logger
logger = logging.getLogger(__name__)


class Cascade:
    """
    Cascade主入口类 - 极简的VAD处理库
    
    实现1:1:1绑定架构，确保VAD状态连续性的流式音频处理。
    
    设计目标：
    - 用户只需配置sample_rate、vad_backend、vad_threshold三个参数
    - 用户控制音频块大小，系统直接处理
    - 返回包含原始音频和VAD结果的完整输出
    - 保证有状态VAD模型的上下文连续性
    
    使用示例:
        >>> config = CascadeConfig(sample_rate=16000, vad_backend="silero")
        >>> cascade = Cascade(config)
        >>> 
        >>> async for result in cascade.process_audio_stream(audio_stream):
        >>>     print(f"语音概率: {result.speech_probability}")
        >>>     print(f"是否语音: {result.is_speech}")
        >>>     # result.audio_data 包含原始音频块
    """
    
    def __init__(self, config: CascadeConfig):
        """
        初始化Cascade实例
        
        Args:
            config: Cascade配置对象，包含sample_rate、vad_backend、vad_threshold
        """
        self.config = config
        self._vad_backend: Union[VADBackend, None] = None
        self._format_processor: Union[AudioFormatProcessor, None] = None
        self._audio_buffer: Union[AudioRingBuffer, None] = None
        self._initialized = False
        
        # 1:1:1绑定 - 确保只有一个Backend实例
        self._audio_config = config.to_audio_config()
        self._vad_config = config.to_vad_config()
        
        # 音频帧处理参数 - 根据VAD要求动态计算
        self._frame_size_samples = None
        self._overlap_size_samples = None
        self._last_chunk_sequence = 0
        
        # 音频流状态标志
        self._audio_stream_finished = False
        
        logger.info(f"Cascade实例已创建，配置: {config}")
    
    async def initialize(self) -> None:
        """
        异步初始化Cascade实例
        
        创建并初始化VAD Backend、格式处理器和环形缓冲区。
        这个方法必须在处理音频前调用。
        
        Raises:
            CascadeError: 当初始化失败时
        """
        if self._initialized:
            logger.warning("Cascade实例已经初始化，跳过重复初始化")
            return
            
        try:
            logger.info("开始初始化Cascade实例...")
            
            # 1. 创建VAD Backend实例（1:1:1绑定的核心）
            self._vad_backend = create_vad_backend(self._vad_config)
            logger.info(f"创建VAD Backend: {self._vad_backend.__class__.__name__}")
            
            # 2. 初始化VAD Backend
            await self._vad_backend.initialize()
            logger.info("VAD Backend初始化完成")
            
            # 3. 创建格式处理器
            self._format_processor = AudioFormatProcessor(self._audio_config)
            logger.info("音频格式处理器创建完成")
            
            # 4. 计算VAD帧参数
            self._calculate_frame_parameters()
            
            # 5. 创建环形缓冲区 - 使用较大的缓冲区以支持音频帧切分和重叠处理
            buffer_capacity_seconds = 5.0  # 5秒缓冲区，足够处理各种场景
            self._audio_buffer = AudioRingBuffer(
                config=self._audio_config,
                capacity_seconds=buffer_capacity_seconds,
                overflow_strategy=BufferStrategy.OVERWRITE  # 自动覆盖旧数据
            )
            logger.info(f"环形缓冲区创建完成，容量: {buffer_capacity_seconds}秒")
            
            # 6. 预热VAD模型（消除冷启动延迟）
            await self._warmup_vad_model()
            
            self._initialized = True
            logger.info("Cascade实例初始化完成")
            
        except Exception as e:
            logger.error(f"Cascade初始化失败: {e}")
            raise CascadeError(
                f"Cascade初始化失败: {e}",
                ErrorCode.INITIALIZATION_FAILED,
                context={"config": self.config.model_dump()}
            )
    
    def _calculate_frame_parameters(self) -> None:
        """
        根据VAD配置计算音频帧参数
        
        计算VAD处理所需的帧大小和重叠大小，确保与VAD模型要求匹配。
        """
        try:
            # 根据VAD配置获取帧参数，如果VAD配置没有明确指定，使用默认值
            if hasattr(self._vad_config, 'chunk_duration_ms'):
                frame_duration_ms = self._vad_config.chunk_duration_ms
            else:
                frame_duration_ms = 512  # 默认512ms窗口
            
            # 计算帧大小（样本数）
            self._frame_size_samples = int(self.config.sample_rate * frame_duration_ms / 1000)
            
            # 计算重叠大小（25%重叠，这是VAD处理的常见做法）
            overlap_ratio = 0.25
            self._overlap_size_samples = int(self._frame_size_samples * overlap_ratio)
            
            logger.info(f"音频帧参数计算完成: 帧大小={self._frame_size_samples}样本, "
                       f"重叠大小={self._overlap_size_samples}样本")
            
        except Exception as e:
            logger.error(f"计算音频帧参数失败: {e}")
            # 使用安全的默认值
            self._frame_size_samples = int(self.config.sample_rate * 0.512)  # 512ms
            self._overlap_size_samples = int(self._frame_size_samples * 0.25)
            logger.warning(f"使用默认帧参数: 帧大小={self._frame_size_samples}样本")

    async def process_audio_stream(
        self,
        audio_stream: AsyncGenerator[np.ndarray, None]
    ) -> AsyncGenerator[CascadeVADResult, None]:
        """
        处理音频流的主要方法 - 使用环形缓冲区和正确的音频帧切分
        
        接受用户提供的音频流，将数据写入环形缓冲区，然后按VAD要求的帧大小
        进行切分和重叠处理，对每个音频帧进行VAD检测。
        
        Args:
            audio_stream: 异步音频流生成器，产生numpy数组格式的音频块
            
        Yields:
            CascadeVADResult: 包含原始音频和VAD检测结果的输出
            
        Raises:
            CascadeError: 当处理失败时
        """
        # 确保已初始化
        if not self._initialized:
            raise CascadeError(
                "Cascade实例未初始化，请先调用initialize()方法",
                ErrorCode.INITIALIZATION_FAILED
            )
        
        try:
            logger.info("开始处理音频流（使用环形缓冲区和帧切分）...")
            
            # 重置音频流结束标志
            self._audio_stream_finished = False
            
            # 启动一个后台任务来喂入音频数据到缓冲区
            feed_task = asyncio.create_task(self._feed_audio_to_buffer(audio_stream))
            
            # 主循环：从缓冲区读取音频帧并进行VAD处理
            async for result in self._process_audio_frames():
                yield result
                
        except Exception as e:
            logger.error(f"音频流处理失败: {e}")
            raise CascadeError(f"音频流处理失败: {e}", ErrorCode.PROCESSING_FAILED)
        finally:
            # 确保后台任务被清理
            if 'feed_task' in locals() and not feed_task.done():
                feed_task.cancel()
                try:
                    await feed_task
                except asyncio.CancelledError:
                    pass
            logger.info("音频流处理完成")
    
    async def _feed_audio_to_buffer(self, audio_stream: AsyncGenerator[np.ndarray, None]) -> None:
        """
        将音频流数据喂入环形缓冲区
        
        Args:
            audio_stream: 音频数据流
        """
        try:
            chunk_count = 0
            async for audio_chunk in audio_stream:
                chunk_count += 1
                logger.debug(f"接收到音频块 {chunk_count}, 大小: {len(audio_chunk)} 样本")
                
                # 格式转换（如果需要）
                processed_audio = self._format_processor.convert_to_internal_format(
                    audio_chunk,
                    self._audio_config.format,
                    self.config.sample_rate
                )
                
                # 写入环形缓冲区（非阻塞，使用覆盖策略）
                success = self._audio_buffer.write(processed_audio, blocking=False)
                if not success:
                    logger.debug("缓冲区满，使用覆盖策略继续写入")
                
                # 短暂让出控制权，避免阻塞
                await asyncio.sleep(0)
                
            logger.info(f"音频流喂入完成，总共处理了 {chunk_count} 个音频块")
            
        except Exception as e:
            logger.error(f"音频流喂入失败: {e}")
            raise
        finally:
            # 标记音频流已结束
            self._audio_stream_finished = True
            logger.debug("音频流结束标志已设置")
    
    async def _process_audio_frames(self) -> AsyncGenerator[CascadeVADResult, None]:
        """
        从缓冲区读取原始音频块，切分为音频帧，并进行VAD处理
        
        环形缓冲区存放原始音频块，在此方法中进行音频帧切分，
        然后将切分好的音频帧传递给VAD后端处理。
        
        Yields:
            CascadeVADResult: VAD检测结果
        """
        processed_frames = 0
        last_frame_time = 0.0
        speech_active = False  # 跟踪当前是否处于语音状态
        
        try:
            while True:
                # 尝试从缓冲区获取一个完整的原始音频块（带重叠）
                # 注意：环形缓冲区存放的是原始音频块，不是音频帧
                raw_chunk, available = self._audio_buffer.get_chunk_with_overlap(
                    self._frame_size_samples,
                    self._overlap_size_samples
                )
                
                if not available or raw_chunk is None:
                    # 检查音频流是否已结束
                    if self._audio_stream_finished:
                        # 音频流已结束，检查是否还有剩余数据需要处理
                        available_samples = self._audio_buffer.available_samples()
                        if available_samples == 0:
                            # 没有剩余数据，如果当前处于语音状态，生成语音结束事件
                            if speech_active:
                                logger.info("检测到音频流结束且当前处于语音状态，生成语音结束事件")
                                # 创建一个空的音频块用于生成语音结束事件
                                empty_data = np.zeros(self._frame_size_samples, dtype=np.float32)
                                empty_chunk = AudioChunk(
                                    data=empty_data,
                                    sequence_number=processed_frames + 1,
                                    start_frame=int(last_frame_time * self.config.sample_rate),
                                    chunk_size=self._frame_size_samples,
                                    timestamp_ms=last_frame_time * 1000,
                                    sample_rate=self.config.sample_rate
                                )
                                
                                # 强制生成语音结束事件
                                # 注意：这里我们需要确保VAD后端能够正确处理这种情况
                                vad_result = self._vad_backend.process_chunk(empty_chunk)
                                
                                # 创建语音结束结果
                                end_result = CascadeVADResult(
                                    audio_data=empty_data,
                                    speech_probability=0.0,  # 语音结束，概率为0
                                    is_speech=False,
                                    start_time=last_frame_time,
                                    end_time=last_frame_time + (self._frame_size_samples / self.config.sample_rate),
                                    confidence=1.0  # 语音结束的置信度为1
                                )
                                
                                processed_frames += 1
                                yield end_result
                                speech_active = False
                                logger.info("语音结束事件已生成")
                            
                            logger.info(f"音频帧处理完成，总共处理了 {processed_frames} 帧")
                            break
                        elif available_samples < (self._frame_size_samples + self._overlap_size_samples):
                            # 剩余数据不足一个完整帧，但还有一些数据，处理最后的不完整帧
                            logger.debug(f"处理最后的不完整音频帧，剩余: {available_samples} 样本")
                            if available_samples > 0:
                                # 获取剩余所有数据作为最后一帧
                                last_chunk, _ = self._audio_buffer.get_chunk_with_overlap(available_samples, 0)
                                if last_chunk:
                                    # 在这里进行音频帧切分 - 将原始音频块切分为VAD所需的音频帧
                                    # 对于最后一个不完整的块，我们直接使用它
                                    
                                    # 处理最后一个不完整的帧
                                    vad_result = self._vad_backend.process_chunk(last_chunk)
                                    frame_duration = available_samples / self.config.sample_rate
                                    cascade_result = CascadeVADResult(
                                        audio_data=last_chunk.data[:available_samples],
                                        speech_probability=vad_result.probability,
                                        is_speech=vad_result.probability >= self.config.vad_threshold,
                                        start_time=last_frame_time,
                                        end_time=last_frame_time + frame_duration,
                                        confidence=vad_result.probability
                                    )
                                    
                                    # 更新语音状态
                                    was_speech_active = speech_active
                                    speech_active = cascade_result.is_speech
                                    
                                    # 清空缓冲区剩余数据
                                    self._audio_buffer.advance_read_position(available_samples)
                                    processed_frames += 1
                                    yield cascade_result
                                    
                                    # 如果当前处于语音状态，生成语音结束事件
                                    if speech_active or was_speech_active:
                                        logger.info("检测到音频流结束且有语音活动，生成语音结束事件")
                                        # 创建一个空的音频块用于生成语音结束事件
                                        empty_data = np.zeros(self._frame_size_samples, dtype=np.float32)
                                        empty_chunk = AudioChunk(
                                            data=empty_data,
                                            sequence_number=processed_frames + 1,
                                            start_frame=int((last_frame_time + frame_duration) * self.config.sample_rate),
                                            chunk_size=self._frame_size_samples,
                                            timestamp_ms=(last_frame_time + frame_duration) * 1000,
                                            sample_rate=self.config.sample_rate
                                        )
                                        
                                        # 强制生成语音结束事件
                                        vad_result = self._vad_backend.process_chunk(empty_chunk)
                                        
                                        # 创建语音结束结果
                                        end_result = CascadeVADResult(
                                            audio_data=empty_data,
                                            speech_probability=0.0,  # 语音结束，概率为0
                                            is_speech=False,
                                            start_time=last_frame_time + frame_duration,
                                            end_time=last_frame_time + frame_duration + (self._frame_size_samples / self.config.sample_rate),
                                            confidence=1.0  # 语音结束的置信度为1
                                        )
                                        
                                        processed_frames += 1
                                        yield end_result
                                        logger.info("语音结束事件已生成")
                            
                            logger.info(f"音频帧处理完成，总共处理了 {processed_frames} 帧")
                            break
                    else:
                        # 音频流还在继续，等待更多数据
                        await asyncio.sleep(0.01)
                        continue
                
                processed_frames += 1
                logger.debug(f"处理音频帧 {processed_frames}, 序列号: {raw_chunk.sequence_number}")
                
                # 在这里进行音频帧切分 - 将原始音频块切分为VAD所需的音频帧
                # 在当前实现中，我们直接使用get_chunk_with_overlap获取的块作为音频帧
                # 这是因为我们已经在获取时指定了正确的帧大小和重叠大小
                audio_frame = raw_chunk
                
                # 记录之前的语音状态，用于检测状态变化
                was_speech_active = speech_active
                
                # 进行VAD检测
                vad_result = self._vad_backend.process_chunk(audio_frame)
                
                # 计算时间信息
                frame_duration = (self._frame_size_samples - self._overlap_size_samples) / self.config.sample_rate
                current_frame_time = last_frame_time
                last_frame_time += frame_duration
                
                # 构造Cascade结果
                cascade_result = CascadeVADResult(
                    audio_data=audio_frame.data[:self._frame_size_samples],  # 返回实际音频帧数据
                    speech_probability=vad_result.probability,
                    is_speech=vad_result.probability >= self.config.vad_threshold,
                    start_time=current_frame_time,
                    end_time=current_frame_time + (self._frame_size_samples / self.config.sample_rate),
                    confidence=vad_result.probability
                )
                
                # 更新语音状态
                speech_active = cascade_result.is_speech
                
                # 检测语音状态变化
                if was_speech_active and not speech_active:
                    logger.info(f"检测到语音结束: {current_frame_time:.2f}s")
                elif not was_speech_active and speech_active:
                    logger.info(f"检测到语音开始: {current_frame_time:.2f}s")
                
                # 推进缓冲区读位置（不包括重叠部分）
                advance_size = self._frame_size_samples - self._overlap_size_samples
                self._audio_buffer.advance_read_position(advance_size)
                
                yield cascade_result
                
                # 让出控制权
                await asyncio.sleep(0)
                
        except Exception as e:
            logger.error(f"音频帧处理失败: {e}")
            raise

    async def process_audio_chunk(self, audio_data: np.ndarray) -> CascadeVADResult:
        """
        处理单个音频块的便捷方法
        
        Args:
            audio_data: 原始音频数据（numpy数组）
            
        Returns:
            CascadeVADResult: VAD检测结果
            
        Raises:
            CascadeError: 当处理失败时
        """
        # 确保已初始化
        if not self._initialized:
            raise CascadeError(
                "Cascade实例未初始化，请先调用initialize()方法",
                ErrorCode.INITIALIZATION_FAILED
            )
        
        # 对于单个音频块处理，我们使用简化的逻辑，直接处理而不使用缓冲区
        return await self._process_single_chunk_direct(audio_data)
    
    async def _process_single_chunk_direct(self, audio_data: np.ndarray) -> CascadeVADResult:
        """
        直接处理单个音频块的内部方法（不使用缓冲区）
        
        这个方法用于process_audio_chunk的单块处理场景。
        
        Args:
            audio_data: 原始音频数据
            
        Returns:
            CascadeVADResult: 处理结果
        """
        try:
            # 1. 格式转换（如果需要）
            processed_audio = self._format_processor.convert_to_internal_format(
                audio_data,
                self._audio_config.format,
                self.config.sample_rate
            )
            
            # 2. 创建AudioChunk对象
            chunk_size = len(processed_audio)
            self._last_chunk_sequence += 1
            
            audio_chunk = AudioChunk(
                data=processed_audio,
                sequence_number=self._last_chunk_sequence,
                start_frame=0,  # 单块处理时从0开始
                chunk_size=chunk_size,
                timestamp_ms=0.0,  # 单块处理时间戳为0
                sample_rate=self.config.sample_rate
            )
            
            # 3. VAD检测（1:1:1绑定确保状态连续性）
            vad_result = self._vad_backend.process_chunk(audio_chunk)
            
            # 4. 计算时间信息
            duration = chunk_size / self.config.sample_rate
            
            # 5. 构造完整结果
            cascade_result = CascadeVADResult(
                audio_data=audio_data,  # 返回原始音频数据
                speech_probability=vad_result.probability,
                is_speech=vad_result.probability >= self.config.vad_threshold,
                start_time=0.0,
                end_time=duration,
                confidence=vad_result.probability  # 使用概率作为置信度
            )
            
            return cascade_result
            
        except Exception as e:
            logger.error(f"处理音频块失败: {e}")
            raise CascadeError(
                f"音频块处理失败: {e}",
                ErrorCode.PROCESSING_FAILED,
                context={
                    "audio_shape": audio_data.shape if hasattr(audio_data, 'shape') else None
                }
            )

    async def _process_single_chunk(
        self, 
        audio_data: np.ndarray, 
        start_time: float,
        chunk_index: int
    ) -> CascadeVADResult:
        """
        处理单个音频块的内部方法
        
        Args:
            audio_data: 原始音频数据
            start_time: 块开始时间
            chunk_index: 块索引
            
        Returns:
            CascadeVADResult: 处理结果
        """
        try:
            # 1. 格式转换（如果需要）
            processed_audio = self._format_processor.convert_to_internal_format(
                audio_data, 
                self._audio_config.format,
                self.config.sample_rate
            )
            
            # 2. 创建AudioChunk对象
            chunk_size = len(processed_audio)
            duration_ms = int(chunk_size * 1000 / self.config.sample_rate)
            
            audio_chunk = AudioChunk(
                data=processed_audio,
                sequence_number=chunk_index,
                start_frame=int(start_time * self.config.sample_rate),
                chunk_size=chunk_size,
                timestamp_ms=start_time * 1000,  # 转换为毫秒
                sample_rate=self.config.sample_rate
            )
            
            # 3. VAD检测（1:1:1绑定确保状态连续性）
            vad_result = self._vad_backend.process_chunk(audio_chunk)
            
            # 4. 计算时间信息
            end_time = start_time + (chunk_size / self.config.sample_rate)
            
            # 5. 构造完整结果
            cascade_result = CascadeVADResult(
                audio_data=audio_data,  # 返回原始音频数据
                speech_probability=vad_result.probability,
                is_speech=vad_result.probability >= self.config.vad_threshold,
                start_time=start_time,
                end_time=end_time,
                confidence=vad_result.probability  # 使用概率作为置信度
            )
            
            return cascade_result
            
        except Exception as e:
            logger.error(f"处理音频块失败: {e}")
            raise CascadeError(
                f"音频块处理失败: {e}",
                ErrorCode.PROCESSING_FAILED,
                context={
                    "chunk_index": chunk_index,
                    "start_time": start_time,
                    "audio_shape": audio_data.shape if hasattr(audio_data, 'shape') else None
                }
            )
    
    async def _warmup_vad_model(self) -> None:
        """
        预热VAD模型，消除冷启动延迟
        """
        try:
            logger.info("开始预热VAD模型...")
            
            # 创建虚拟音频数据（1秒的静音）
            dummy_samples = self.config.sample_rate  # 1秒的样本数
            dummy_audio = np.zeros(dummy_samples, dtype=np.float32)
            
            dummy_chunk = AudioChunk(
                data=dummy_audio,
                sequence_number=0,
                start_frame=0,
                chunk_size=dummy_samples,
                timestamp_ms=0.0,
                sample_rate=self.config.sample_rate
            )
            
            # 预热VAD模型
            self._vad_backend.warmup(dummy_chunk)
            logger.info("VAD模型预热完成")
            
        except Exception as e:
            logger.warning(f"VAD模型预热失败（可忽略）: {e}")
    
    async def close(self) -> None:
        """
        关闭Cascade实例并释放资源
        """
        if not self._initialized:
            return
            
        try:
            logger.info("开始关闭Cascade实例...")
            
            # 关闭VAD Backend
            if self._vad_backend:
                await self._vad_backend.close()
                self._vad_backend = None
                logger.info("VAD Backend已关闭")
            
            # 清理格式处理器
            if self._format_processor:
                self._format_processor.clear_cache()
                self._format_processor = None
                logger.info("格式处理器已关闭")
            
            # 关闭环形缓冲区
            if self._audio_buffer:
                self._audio_buffer.close()
                self._audio_buffer = None
                logger.info("环形缓冲区已关闭")
            
            self._initialized = False
            logger.info("Cascade实例已关闭")
            
        except Exception as e:
            logger.error(f"关闭Cascade实例失败: {e}")
    
    def get_stats(self) -> dict[str, Any]:
        """
        获取实例状态信息
        
        Returns:
            包含状态信息的字典
        """
        stats = {
            "initialized": self._initialized,
            "vad_backend": (
                self._vad_backend.__class__.__name__
                if self._vad_backend else None
            ),
            "config": self.config.model_dump(),
            "frame_parameters": {
                "frame_size_samples": self._frame_size_samples,
                "overlap_size_samples": self._overlap_size_samples
            }
        }
        
        # 添加缓冲区状态信息
        if self._audio_buffer:
            buffer_status = self._audio_buffer.get_buffer_status()
            stats["buffer_status"] = {
                "capacity_samples": buffer_status.capacity,
                "available_samples": buffer_status.available_samples,
                "usage_ratio": buffer_status.usage_ratio,
                "status_level": buffer_status.status_level
            }
        
        return stats
    
    @property
    def is_initialized(self) -> bool:
        """检查是否已初始化"""
        return self._initialized
    
    # 上下文管理器支持
    async def __aenter__(self):
        """异步上下文管理器入口"""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        await self.close()


# 便捷函数
async def create_cascade(
    sample_rate: int = 16000,
    vad_backend: str = "silero", 
    vad_threshold: float = 0.5
) -> Cascade:
    """
    便捷的Cascade实例创建函数
    
    Args:
        sample_rate: 音频采样率
        vad_backend: VAD后端类型
        vad_threshold: VAD检测阈值
        
    Returns:
        初始化完成的Cascade实例
    """
    config = CascadeConfig(
        sample_rate=sample_rate,
        vad_backend=vad_backend,
        vad_threshold=vad_threshold
    )
    
    cascade = Cascade(config)
    await cascade.initialize()
    return cascade


__all__ = [
    "Cascade",
    "create_cascade",
]