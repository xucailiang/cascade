"""
语音活动检测处理器

本模块提供语音活动检测（Voice Activity Detection, VAD）功能，
用于检测音频中的语音活动，区分语音和非语音段。
"""

import time
from enum import Enum
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from cascade.processor.base import (
    AudioChunk,
    AudioProcessor,
    ProcessorConfig,
)


class VADSensitivity(str, Enum):
    """VAD 灵敏度"""
    LOW = "low"        # 低灵敏度，减少误检
    MEDIUM = "medium"  # 中等灵敏度，平衡误检和漏检
    HIGH = "high"      # 高灵敏度，减少漏检


class VADResult(BaseModel):
    """VAD 处理结果"""
    is_speech: bool = Field(description="是否为语音")
    confidence: float = Field(description="置信度", ge=0.0, le=1.0)
    start_time_ms: float = Field(description="开始时间（毫秒）", ge=0.0)
    end_time_ms: float = Field(description="结束时间（毫秒）", ge=0.0)
    energy: float = Field(description="能量值", ge=0.0)
    threshold: float = Field(description="阈值", ge=0.0)
    metadata: dict[str, Any] | None = Field(default=None, description="附加元数据")


class VADProcessorConfig(ProcessorConfig):
    """VAD 处理器配置"""
    sensitivity: VADSensitivity = Field(
        default=VADSensitivity.MEDIUM,
        description="VAD 灵敏度"
    )
    energy_threshold_low: float = Field(
        default=0.01,
        description="低灵敏度能量阈值",
        ge=0.0,
        le=1.0
    )
    energy_threshold_medium: float = Field(
        default=0.005,
        description="中等灵敏度能量阈值",
        ge=0.0,
        le=1.0
    )
    energy_threshold_high: float = Field(
        default=0.002,
        description="高灵敏度能量阈值",
        ge=0.0,
        le=1.0
    )
    min_speech_duration_ms: int = Field(
        default=100,
        description="最小语音持续时间（毫秒）",
        ge=10,
        le=1000
    )
    min_silence_duration_ms: int = Field(
        default=300,
        description="最小静音持续时间（毫秒）",
        ge=10,
        le=1000
    )
    speech_pad_ms: int = Field(
        default=30,
        description="语音段前后填充时间（毫秒）",
        ge=0,
        le=500
    )
    normalize_audio: bool = Field(
        default=True,
        description="是否归一化音频"
    )


class VADProcessor(AudioProcessor):
    """
    语音活动检测处理器
    
    使用能量阈值法检测音频中的语音活动。
    """

    def __init__(self, config: VADProcessorConfig | None = None):
        """
        初始化 VAD 处理器
        
        Args:
            config: VAD 处理器配置，如果为 None 则使用默认配置
        """
        super().__init__(config or VADProcessorConfig())
        self.config = self.config if isinstance(self.config, VADProcessorConfig) else VADProcessorConfig()

        # 根据灵敏度选择阈值
        self.threshold = self._get_threshold_by_sensitivity()

        # 状态变量
        self._last_frame_was_speech = False
        self._speech_start_time = 0.0
        self._silence_start_time = 0.0
        self._current_speech_duration = 0.0
        self._current_silence_duration = 0.0

    def _get_threshold_by_sensitivity(self) -> float:
        """
        根据灵敏度获取阈值
        
        Returns:
            能量阈值
        """
        if self.config.sensitivity == VADSensitivity.LOW:
            return self.config.energy_threshold_low
        elif self.config.sensitivity == VADSensitivity.MEDIUM:
            return self.config.energy_threshold_medium
        else:  # HIGH
            return self.config.energy_threshold_high

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

        # 归一化音频（如果需要）
        if self.config.normalize_audio and len(audio_data) > 0:
            audio_data = audio_data / np.max(np.abs(audio_data))

        # 计算能量
        energy = self._calculate_energy(audio_data)

        # 判断是否为语音
        is_speech = energy > self.threshold

        # 计算置信度
        confidence = min(1.0, energy / (self.threshold * 2)) if is_speech else 0.0

        # 创建结果
        result = VADResult(
            is_speech=is_speech,
            confidence=confidence,
            start_time_ms=chunk.timestamp_ms,
            end_time_ms=chunk.timestamp_ms + chunk.get_duration_ms(),
            energy=energy,
            threshold=self.threshold
        )

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
        return np.sqrt(np.mean(np.square(audio_data)))

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

        # 合并相邻的语音段
        merged_results = []
        current_speech_segment = None

        for result in results:
            vad_result = result.result_data

            if vad_result.is_speech:
                if current_speech_segment is None:
                    # 开始新的语音段
                    current_speech_segment = vad_result
                else:
                    # 扩展当前语音段
                    current_speech_segment.end_time_ms = vad_result.end_time_ms
                    current_speech_segment.confidence = max(current_speech_segment.confidence, vad_result.confidence)
            elif current_speech_segment is not None:
                # 结束当前语音段
                speech_duration = current_speech_segment.end_time_ms - current_speech_segment.start_time_ms

                # 只保留足够长的语音段
                if speech_duration >= self.config.min_speech_duration_ms:
                    # 添加前后填充
                    current_speech_segment.start_time_ms = max(0, current_speech_segment.start_time_ms - self.config.speech_pad_ms)
                    current_speech_segment.end_time_ms += self.config.speech_pad_ms

                    merged_results.append(result._replace(result_data=current_speech_segment))

                current_speech_segment = None

        # 处理最后一个语音段
        if current_speech_segment is not None:
            speech_duration = current_speech_segment.end_time_ms - current_speech_segment.start_time_ms

            if speech_duration >= self.config.min_speech_duration_ms:
                # 添加前后填充
                current_speech_segment.start_time_ms = max(0, current_speech_segment.start_time_ms - self.config.speech_pad_ms)
                current_speech_segment.end_time_ms += self.config.speech_pad_ms

                merged_results.append(results[-1]._replace(result_data=current_speech_segment))

        context["results"] = merged_results
        return context


class WebRTCVADProcessor(AudioProcessor):
    """
    基于 WebRTC VAD 的语音活动检测处理器
    
    使用 WebRTC VAD 算法检测音频中的语音活动。
    需要安装 webrtcvad 包：pip install webrtcvad
    """

    def __init__(self, config: VADProcessorConfig | None = None):
        """
        初始化 WebRTC VAD 处理器
        
        Args:
            config: VAD 处理器配置，如果为 None 则使用默认配置
        """
        super().__init__(config or VADProcessorConfig())
        self.config = self.config if isinstance(self.config, VADProcessorConfig) else VADProcessorConfig()

        try:
            import webrtcvad
            self.vad = webrtcvad.Vad()

            # 设置 WebRTC VAD 的激进程度（0-3）
            if self.config.sensitivity == VADSensitivity.LOW:
                self.vad.set_mode(1)
            elif self.config.sensitivity == VADSensitivity.MEDIUM:
                self.vad.set_mode(2)
            else:  # HIGH
                self.vad.set_mode(3)
        except ImportError:
            raise ImportError("WebRTC VAD 处理器需要安装 webrtcvad 包：pip install webrtcvad")

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
        # 这里我们假设音频块已经是合适的大小
        frame_duration_ms = 30  # 30 毫秒帧
        frames_per_chunk = int(chunk.sample_rate * frame_duration_ms / 1000)

        # 将音频分割为帧
        num_frames = len(audio_data) // frames_per_chunk
        speech_frames = 0

        for i in range(num_frames):
            frame = audio_data[i * frames_per_chunk:(i + 1) * frames_per_chunk]
            frame_bytes = frame.tobytes()

            try:
                is_speech = self.vad.is_speech(frame_bytes, chunk.sample_rate)
                if is_speech:
                    speech_frames += 1
            except Exception:
                # 处理可能的错误
                pass

        # 计算语音比例
        speech_ratio = speech_frames / num_frames if num_frames > 0 else 0

        # 判断是否为语音
        is_speech = speech_ratio > 0.3  # 如果超过 30% 的帧被检测为语音，则认为整个块是语音

        # 创建结果
        result = VADResult(
            is_speech=is_speech,
            confidence=speech_ratio,
            start_time_ms=chunk.timestamp_ms,
            end_time_ms=chunk.timestamp_ms + chunk.get_duration_ms(),
            energy=0.0,  # WebRTC VAD 不使用能量
            threshold=0.3  # 阈值固定为 0.3
        )

        return result
