"""
Cascade 流式处理器数据类型

基于VAD状态机设计的数据类型定义。
"""

from typing import Optional, Dict, Any, List, Literal
from pydantic import BaseModel, Field
import time


# 音频格式常量 - 基于silero-vad要求
AUDIO_SAMPLE_RATE = 16000  # 固定16kHz
AUDIO_FRAME_SIZE = 512     # 固定512样本/帧
AUDIO_FRAME_DURATION_MS = 32.0  # 32ms/帧
AUDIO_CHANNELS = 1         # 单声道
AUDIO_SAMPLE_WIDTH = 2     # 16-bit


class AudioFrame(BaseModel):
    """
    单个音频帧
    
    表示512样本的音频帧和相关元数据。
    """
    
    # 基础信息
    frame_id: int = Field(description="帧ID")
    audio_data: bytes = Field(description="512样本音频数据")
    timestamp_ms: float = Field(description="时间戳(ms)")
    
    # VAD信息
    vad_result: Optional[Dict[str, Any]] = Field(default=None, description="原始VAD结果")
    
    # 元数据
    sample_rate: int = Field(default=AUDIO_SAMPLE_RATE, description="采样率")
    frame_size: int = Field(default=AUDIO_FRAME_SIZE, description="帧大小(样本)")
    
    def __str__(self) -> str:
        vad_str = str(self.vad_result) if self.vad_result else "None"
        return f"AudioFrame(id={self.frame_id}, vad={vad_str}, {self.timestamp_ms:.0f}ms)"


class SpeechSegment(BaseModel):
    """
    语音段
    
    表示从VAD检测到start到end之间的完整语音片段。
    """
    
    # 基础信息
    segment_id: int = Field(description="语音段ID")
    audio_data: bytes = Field(description="合并的音频数据")
    
    # 时间信息
    start_timestamp_ms: float = Field(description="开始时间戳(ms)")
    end_timestamp_ms: float = Field(description="结束时间戳(ms)")
    
    # 统计信息
    frame_count: int = Field(description="包含的帧数")
    
    # VAD信息
    start_vad_result: Dict[str, Any] = Field(description="开始VAD结果")
    end_vad_result: Dict[str, Any] = Field(description="结束VAD结果")
    
    # 元数据
    sample_rate: int = Field(default=AUDIO_SAMPLE_RATE, description="采样率")
    
    @property
    def duration_ms(self) -> float:
        """语音段时长(ms)"""
        return self.end_timestamp_ms - self.start_timestamp_ms
    
    def __str__(self) -> str:
        return f"SpeechSegment(id={self.segment_id}, frames={self.frame_count}, {self.duration_ms:.0f}ms)"


class CascadeResult(BaseModel):
    """
    Cascade输出结果
    
    统一的输出接口，可以是单帧或语音段。
    """
    
    # 结果类型
    result_type: Literal["frame", "segment"] = Field(description="结果类型")
    
    # 结果数据
    frame: Optional[AudioFrame] = Field(default=None, description="单帧结果")
    segment: Optional[SpeechSegment] = Field(default=None, description="语音段结果")
    
    # 处理信息
    processing_time_ms: float = Field(description="处理时间(ms)")
    instance_id: str = Field(description="处理实例ID")
    
    def __str__(self) -> str:
        if self.result_type == "frame":
            return f"CascadeResult(frame: {self.frame})"
        else:
            return f"CascadeResult(segment: {self.segment})"
    
    @property
    def is_speech_segment(self) -> bool:
        """是否为语音段"""
        return self.result_type == "segment"
    
    @property
    def is_single_frame(self) -> bool:
        """是否为单帧"""
        return self.result_type == "frame"


class Config(BaseModel):
    """
    Cascade配置类
    
    基于silero-vad优化，固定关键音频参数，简化配置。
    """
    
    # 音频配置 - 基于silero-vad优化（固定值）
    sample_rate: int = Field(default=AUDIO_SAMPLE_RATE, frozen=True, description="采样率(Hz)")
    frame_size: int = Field(default=AUDIO_FRAME_SIZE, frozen=True, description="VAD帧大小(样本)")
    frame_duration_ms: float = Field(default=AUDIO_FRAME_DURATION_MS, frozen=True, description="帧时长(ms)")
    channels: int = Field(default=AUDIO_CHANNELS, frozen=True, description="声道数")
    sample_width: int = Field(default=AUDIO_SAMPLE_WIDTH, frozen=True, description="采样位宽")
    supported_formats: List[str] = Field(default=["wav", "mp3"], frozen=True, description="支持的音频格式")
    
    # VAD配置
    vad_threshold: float = Field(default=0.5, description="VAD检测阈值", ge=0.0, le=1.0)
    min_speech_duration_ms: int = Field(default=100, description="最小语音时长(ms)")
    min_silence_duration_ms: int = Field(default=100, description="最小静音时长(ms)")
    
    # 性能配置
    max_instances: int = Field(default=5, description="最大并发实例数", ge=1, le=20)
    buffer_size_frames: int = Field(default=64, description="缓冲区大小(帧数)", ge=8, le=256)
    
    # 高级配置
    enable_logging: bool = Field(default=True, description="是否启用日志")
    log_level: str = Field(default="INFO", description="日志级别")
    enable_profiling: bool = Field(default=False, description="是否启用性能分析")
    
    class Config:
        extra = "forbid"
        frozen = True  # 配置不可变
        
    @property
    def buffer_size_seconds(self) -> float:
        """缓冲区大小(秒)"""
        return (self.buffer_size_frames * self.frame_duration_ms) / 1000.0


class ProcessorStats(BaseModel):
    """
    处理器统计信息
    """
    
    # 处理统计
    total_chunks_processed: int = Field(description="总处理块数")
    total_processing_time_ms: float = Field(description="总处理时间(ms)")
    average_processing_time_ms: float = Field(description="平均处理时间(ms)")
    
    # 检测统计
    speech_segments: int = Field(description="语音段数")
    single_frames: int = Field(description="单帧数")
    speech_ratio: float = Field(description="语音比例")
    
    # 性能统计
    throughput_chunks_per_second: float = Field(description="吞吐量(块/秒)")
    memory_usage_mb: float = Field(description="内存使用(MB)")
    
    # 错误统计
    error_count: int = Field(description="错误次数")
    error_rate: float = Field(description="错误率")
    
    def summary(self) -> str:
        """返回统计摘要"""
        return (f"处理了{self.total_chunks_processed}个块, "
                f"语音段{self.speech_segments}个, "
                f"平均处理时间{self.average_processing_time_ms:.1f}ms")