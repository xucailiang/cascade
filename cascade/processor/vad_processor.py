"""
语音活动检测处理器

本模块提供语音活动检测（Voice Activity Detection, VAD）功能，
用于检测音频中的语音活动，区分语音和非语音段。

本模块实现了基于深度学习的VAD处理器，支持多种后端（如ONNX、VLLM等），
并通过线程池和线程本地存储实现高性能并行处理。
"""

import asyncio
import logging
import time
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np
from pydantic import BaseModel, Field, model_validator

from cascade.backends.base import VADBackend
from cascade.backends.onnx import ONNXVADBackend
from cascade.backends.thread_pool import VADThreadPool
from cascade.processor.base import (
    AudioChunk,
    AudioProcessor,
    ProcessResult,
)
from cascade.processor.merger import ResultMerger
from cascade.types.audio import AudioConfig
from cascade.types.config import BackendConfig, ONNXConfig, VADProcessorConfig, ProcessorConfig
from cascade.types.vad import (
    MergerConfig,
    VADConfig,
    VADResult,
    VADSegment,
)

# 配置日志
logger = logging.getLogger("cascade.vad.processor")


class VADSensitivity(str, Enum):
    """VAD 灵敏度"""
    LOW = "low"        # 低灵敏度，减少误检
    MEDIUM = "medium"  # 中等灵敏度，平衡误检和漏检
    HIGH = "high"      # 高灵敏度，减少漏检


class VADProcessor(AudioProcessor):
    """
    语音活动检测处理器
    
    基于深度学习的VAD处理器，支持多种后端（如ONNX、VLLM等），
    并通过线程池和线程本地存储实现高性能并行处理。
    """

    def __init__(self, config: VADProcessorConfig | None = None):
        """
        初始化 VAD 处理器
        
        Args:
            config: VAD 处理器配置，如果为 None 则使用默认配置
        """
        super().__init__(config or VADProcessorConfig())
        self.config = self.config if isinstance(self.config, VADProcessorConfig) else VADProcessorConfig()
        
        # 创建后端配置
        self._backend_config = self._create_backend_config()
        
        # 创建后端实例
        self._backend = self._create_backend()
        
        # 创建线程池
        self._thread_pool = VADThreadPool(
            config=VADConfig(
                workers=self.config.workers,
                threshold=self.config.threshold,
                chunk_duration_ms=self.config.chunk_duration_ms,
                overlap_ms=self.config.overlap_ms,
            ),
            backend_template=self._backend
        )
        
        # 创建结果合并器
        self._merger = ResultMerger(
            config=MergerConfig(
                window_size=self.config.smoothing_window,
                threshold=self.config.threshold,
                min_speech_duration_ms=self.config.min_speech_duration_ms,
                min_silence_duration_ms=self.config.min_silence_duration_ms,
            )
        )
        
        # 初始化状态
        self._is_started = False
        self._segments: List[VADSegment] = []
        
        logger.debug(f"VADProcessor初始化完成，配置: {self.config}")

    def _create_backend_config(self) -> BackendConfig:
        """
        创建后端配置
        
        Returns:
            后端配置对象
        """
        if self.config.backend_type.lower() == "onnx":
            return ONNXConfig(
                model_path=self.config.model_path,
                threshold=self.config.threshold,
                chunk_duration_ms=self.config.chunk_duration_ms,
                sample_rate=self.config.sample_rate,
            )
        else:
            raise ValueError(f"不支持的后端类型: {self.config.backend_type}")

    def _create_backend(self) -> VADBackend:
        """
        创建VAD后端实例
        
        Returns:
            VAD后端实例
        """
        if self.config.backend_type.lower() == "onnx":
            return ONNXVADBackend(self._backend_config)
        else:
            raise ValueError(f"不支持的后端类型: {self.config.backend_type}")

    async def start(self) -> None:
        """
        启动VAD处理器
        
        初始化后端并启动线程池。
        
        Raises:
            RuntimeError: 当启动失败时
        """
        if self._is_started:
            logger.warning("VADProcessor已经启动")
            return
            
        logger.info("正在启动VADProcessor...")
        
        try:
            # 初始化后端
            self._backend.initialize()
            
            # 启动线程池
            await self._thread_pool.start()
            
            self._is_started = True
            logger.info("VADProcessor启动成功")
        except Exception as e:
            logger.error(f"VADProcessor启动失败: {e}")
            # 确保资源被释放
            await self.close()
            raise RuntimeError(f"VADProcessor启动失败: {e}")

    async def close(self) -> None:
        """
        关闭VAD处理器
        
        关闭线程池并释放所有资源。
        """
        if not self._is_started:
            logger.debug("VADProcessor尚未启动或已关闭")
            return
            
        logger.info("正在关闭VADProcessor...")
        
        try:
            # 关闭线程池
            await self._thread_pool.close()
            
            self._is_started = False
            logger.info("VADProcessor已关闭")
        except Exception as e:
            logger.error(f"关闭VADProcessor时发生错误: {e}")

    def process_chunk(self, chunk: AudioChunk) -> VADResult:
        """
        处理单个音频块
        
        此方法是同步的，会在当前线程中执行。
        实际处理会委托给线程池中的工作线程。
        
        Args:
            chunk: 音频块
            
        Returns:
            VAD 处理结果
            
        Raises:
            RuntimeError: 当处理器未启动或处理失败时
        """
        if not self._is_started:
            raise RuntimeError("VADProcessor尚未启动，请先调用start()方法")
        
        # 记录开始时间
        start_time = time.time()
        
        # 获取音频数据
        audio_data = chunk.data
        
        # 归一化音频（如果需要）
        if self.config.normalize_audio and len(audio_data) > 0:
            audio_data = audio_data / np.max(np.abs(audio_data))
            # 更新chunk中的数据
            chunk.data = audio_data
        
        # 使用线程池处理音频块
        # 注意：这里应该是异步调用，但process_chunk方法是同步的
        # 因此我们需要在这里运行事件循环来等待异步操作完成
        loop = asyncio.get_event_loop()
        try:
            result = loop.run_until_complete(self._thread_pool.process_chunk(chunk))
        except Exception as e:
            logger.error(f"处理音频块失败: {e}")
            raise RuntimeError(f"处理音频块失败: {e}")
        
        # 计算处理时间
        processing_time = (time.time() - start_time) * 1000  # 转换为毫秒
        
        # 添加能量信息（如果需要）
        if self.config.normalize_audio:
            result.energy_level = self._calculate_energy(audio_data)
        
        # 添加处理时间到结果元数据
        if result.metadata is None:
            result.metadata = {}
        result.metadata["processing_time_ms"] = processing_time
        
        # 添加结果到合并器
        self._merger.add_result(result)
        
        return result

    def _calculate_energy(self, audio_data: np.ndarray) -> float:
        """
        计算音频能量
        
        Args:
            audio_data: 音频数据
            
        Returns:
            能量值
        """
        if len(audio_data) == 0:
            return 0.0

        # 使用均方根（RMS）计算能量
        return float(np.sqrt(np.mean(np.square(audio_data))))

    async def _merge_results(self, context: dict[str, Any]) -> dict[str, Any]:
        """
        合并处理结果
        
        Args:
            context: 处理上下文
            
        Returns:
            更新后的上下文
            
        Raises:
            ValueError: 当合并失败时
        """
        results = context["results"]
        processor_config = context["processor_config"]

        # 按序列号排序
        results.sort(key=lambda r: r.chunk.sequence_number)
        
        # 将所有结果添加到合并器
        for result in results:
            vad_result = result.result_data
            self._merger.add_result(vad_result)
        
        # 获取合并后的语音段
        segments = self._merger.flush()
        self._segments = segments
        
        # 创建新的处理结果
        merged_results = []
        for segment in segments:
            # 找到对应的原始结果
            original_result = next(
                (r for r in results if r.chunk.timestamp_ms <= segment.start_ms and 
                 r.chunk.timestamp_ms + r.chunk.get_duration_ms() >= segment.end_ms),
                results[0] if results else None
            )
            
            if original_result:
                # 创建新的VADResult
                vad_result = VADResult(
                    is_speech=True,
                    probability=segment.peak_probability,
                    start_ms=segment.start_ms,
                    end_ms=segment.end_ms,
                    chunk_id=original_result.chunk.sequence_number,
                    confidence=segment.confidence,
                    energy_level=segment.energy_stats.get("avg") if segment.energy_stats else None,
                )
                
                # 创建新的ProcessResult
                merged_result = ProcessResult(
                    chunk=original_result.chunk,
                    result_data=vad_result,
                    success=True,
                    processing_time_ms=original_result.processing_time_ms,
                    metadata={"segment": segment.model_dump()}
                )
                
                merged_results.append(merged_result)
        
        context["results"] = merged_results
        return context
    
    def get_segments(self) -> List[VADSegment]:
        """
        获取当前的语音段列表
        
        Returns:
            语音段列表
        """
        return self._segments.copy()
    
    def reset(self) -> None:
        """
        重置处理器状态
        
        清空合并器和语音段列表。
        """
        self._merger.reset()
        self._segments = []


class WebRTCVADProcessor(VADProcessor):
    """
    基于 WebRTC VAD 的语音活动检测处理器
    
    使用 WebRTC VAD 算法检测音频中的语音活动。
    需要安装 webrtcvad 包：pip install webrtcvad
    
    注意：此类已被废弃，将在未来版本中移除。
    请使用基于ONNX的VADProcessor代替。
    """

    def __init__(self, config: VADProcessorConfig | None = None):
        """
        初始化 WebRTC VAD 处理器
        
        Args:
            config: VAD 处理器配置，如果为 None 则使用默认配置
            
        Raises:
            ImportError: 当未安装webrtcvad包时抛出
            DeprecationWarning: 此类已被废弃
        """
        import warnings
        warnings.warn(
            "WebRTCVADProcessor已被废弃，将在未来版本中移除。请使用基于ONNX的VADProcessor代替。",
            DeprecationWarning,
            stacklevel=2
        )
        
        super().__init__(config)
        
        try:
            import webrtcvad
            self._webrtc_vad = webrtcvad.Vad()
            
            # 设置 WebRTC VAD 的激进程度（0-3）
            if self.config.sensitivity == VADSensitivity.LOW:
                self._webrtc_vad.set_mode(1)
            elif self.config.sensitivity == VADSensitivity.MEDIUM:
                self._webrtc_vad.set_mode(2)
            else:  # HIGH
                self._webrtc_vad.set_mode(3)
        except ImportError:
            raise ImportError("WebRTC VAD 处理器需要安装 webrtcvad 包：pip install webrtcvad")
    
    async def start(self) -> None:
        """
        启动WebRTC VAD处理器
        
        WebRTC VAD不需要特殊的启动过程，因此这里只是设置状态标志。
        """
        self._is_started = True
        logger.info("WebRTCVADProcessor启动成功")
    
    async def close(self) -> None:
        """
        关闭WebRTC VAD处理器
        
        WebRTC VAD不需要特殊的关闭过程，因此这里只是重置状态标志。
        """
        self._is_started = False
        logger.info("WebRTCVADProcessor已关闭")
    
    def process_chunk(self, chunk: AudioChunk) -> VADResult:
        """
        处理单个音频块
        
        Args:
            chunk: 音频块
            
        Returns:
            VAD 处理结果
            
        Raises:
            ValueError: 当音频块无效时
        """
        # 记录开始时间
        start_time = time.time()
        
        # 获取音频数据
        audio_data = chunk.data
        
        # WebRTC VAD 需要 16 位整数格式的音频
        if audio_data.dtype != np.int16:
            if audio_data.dtype == np.float32:
                audio_data = (audio_data * 32767).astype(np.int16)
            else:
                audio_data = audio_data.astype(np.int16)
        
        # WebRTC VAD 需要特定的帧大小（10、20 或 30 毫秒）
        frame_duration_ms = 30  # 30 毫秒帧
        frames_per_chunk = int(chunk.sample_rate * frame_duration_ms / 1000)
        
        # 将音频分割为帧
        num_frames = len(audio_data) // frames_per_chunk
        speech_frames = 0
        
        for i in range(num_frames):
            frame = audio_data[i * frames_per_chunk:(i + 1) * frames_per_chunk]
            frame_bytes = frame.tobytes()
            
            try:
                is_speech = self._webrtc_vad.is_speech(frame_bytes, chunk.sample_rate)
                if is_speech:
                    speech_frames += 1
            except Exception as e:
                logger.error(f"WebRTC VAD处理帧失败: {e}")
        
        # 计算语音比例
        speech_ratio = speech_frames / num_frames if num_frames > 0 else 0
        
        # 判断是否为语音
        is_speech = speech_ratio > 0.3  # 如果超过30%的帧被检测为语音，则认为整个块是语音
        
        # 创建结果
        result = VADResult(
            is_speech=is_speech,
            probability=speech_ratio,
            start_ms=chunk.timestamp_ms,
            end_ms=chunk.timestamp_ms + chunk.get_duration_ms(),
            chunk_id=chunk.sequence_number,
            confidence=speech_ratio,
            energy_level=self._calculate_energy(chunk.data) if self.config.normalize_audio else None,
        )
        
        # 计算处理时间
        processing_time = (time.time() - start_time) * 1000  # 转换为毫秒
        
        # 添加处理时间到结果元数据
        if result.metadata is None:
            result.metadata = {}
        result.metadata["processing_time_ms"] = processing_time
        
        # 添加结果到合并器
        self._merger.add_result(result)
        
        return result
