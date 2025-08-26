"""
Cascade核心类型系统

提供整个项目的数据契约基础，包括：
- 音频处理相关类型
- VAD配置和结果类型  
- 性能监控类型
- 错误处理类型
- 状态管理类型

设计原则：
- 零依赖（除pydantic外）
- 类型安全优先
- 完整的验证规则
- 自动文档生成
"""
import sys
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field, field_validator, model_validator

# === 基础枚举类型 ===

class AudioFormat(str, Enum):
    """支持的音频格式"""
    WAV = "wav"
    PCMA = "pcma"

    @classmethod
    def get_supported_formats(cls) -> list[str]:
        """获取支持的格式列表"""
        return [format.value for format in cls]

class VADBackend(str, Enum):
    """支持的VAD后端"""
    ONNX = "onnx"
    VLLM = "vllm"
    SILERO = "silero"  # 新增Silero支持

    @classmethod
    def get_default_backend(cls) -> str:
        """获取默认后端"""
        return cls.SILERO.value

class ProcessingMode(str, Enum):
    """处理模式"""
    STREAMING = "streaming"      # 流式处理
    BATCH = "batch"             # 批量处理
    REALTIME = "realtime"       # 实时处理

class LogLevel(str, Enum):
    """日志级别"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class BufferStrategy(str, Enum):
    """缓冲区溢出策略"""
    BLOCK = "block"             # 阻塞等待
    OVERWRITE = "overwrite"     # 覆盖旧数据
    REJECT = "reject"           # 拒绝新数据

class OptimizationLevel(str, Enum):
    """优化级别"""
    NONE = "none"               # 无优化
    BASIC = "basic"             # 基础优化
    AGGRESSIVE = "aggressive"    # 激进优化
    ALL = "all"                 # 全部优化

# === 音频相关类型 ===

class AudioConfig(BaseModel):
    """
    音频处理配置
    
    定义音频数据的基本参数，影响整个处理流程。
    """
    sample_rate: int = Field(
        default=16000,
        description="采样率（Hz）",
        ge=1000,  # 最小1kHz
        le=48000  # 最大48kHz
    )
    format: AudioFormat = Field(
        default=AudioFormat.WAV,
        description="音频格式"
    )
    channels: int = Field(
        default=1,
        description="声道数",
        ge=1,
        le=2
    )
    dtype: str = Field(
        default="float32",
        description="内部数据类型"
    )
    bit_depth: int | None = Field(
        default=None,
        description="位深度",
        ge=8,
        le=32
    )

    @field_validator('sample_rate')
    @classmethod
    def validate_sample_rate(cls, v):
        """验证采样率"""
        supported_rates = [8000, 16000, 22050, 44100, 48000]
        if v not in supported_rates:
            raise ValueError(f'采样率必须是以下之一: {supported_rates}')
        return v

    @field_validator('channels')
    @classmethod
    def validate_channels(cls, v):
        """验证声道数"""
        if v != 1:
            raise ValueError('当前版本仅支持单声道音频')
        return v

    @field_validator('dtype')
    @classmethod
    def validate_dtype(cls, v):
        """验证数据类型"""
        supported_dtypes = ['float32', 'float64', 'int16', 'int32']
        if v not in supported_dtypes:
            raise ValueError(f'数据类型必须是以下之一: {supported_dtypes}')
        return v

    @model_validator(mode='after')
    def validate_format_compatibility(self):
        """验证格式兼容性"""
        if self.format == AudioFormat.PCMA:
            # PCMA格式限制
            if self.sample_rate not in [8000, 16000]:
                raise ValueError('PCMA格式仅支持8kHz和16kHz采样率')

        return self

    def get_frame_size(self, duration_ms: int) -> int:
        """计算指定时长的帧大小（样本数）"""
        return int(duration_ms * self.sample_rate / 1000)

    def get_bytes_per_second(self) -> int:
        """计算每秒字节数"""
        bytes_per_sample = {
            'float32': 4,
            'float64': 8,
            'int16': 2,
            'int32': 4
        }.get(self.dtype, 4)

        return self.sample_rate * self.channels * bytes_per_sample

    class Config:
        extra = "forbid"
        use_enum_values = True
        json_schema_extra = {
            "examples": [
                {
                    "sample_rate": 16000,
                    "format": "wav",
                    "channels": 1,
                    "dtype": "float32"
                }
            ]
        }

class AudioChunk(BaseModel):
    """
    音频数据块
    
    封装音频数据及其元数据，用于模块间传输。
    """
    data: Any = Field(
        description="音频数据（numpy数组或类似结构）"
    )
    sequence_number: int = Field(
        description="序列号",
        ge=0
    )
    start_frame: int = Field(
        description="起始帧位置",
        ge=0
    )
    chunk_size: int = Field(
        description="主要块大小（样本数）",
        gt=0
    )
    overlap_size: int = Field(
        default=0,
        description="重叠区域大小（样本数）",
        ge=0
    )
    timestamp_ms: float = Field(
        description="时间戳（毫秒）",
        ge=0.0
    )
    sample_rate: int = Field(
        description="采样率",
        gt=0
    )
    is_last: bool = Field(
        default=False,
        description="是否为最后一块"
    )
    metadata: dict[str, Any] | None = Field(
        default=None,
        description="附加元数据"
    )

    @field_validator('overlap_size')
    @classmethod
    def validate_overlap_size(cls, v, info):
        """验证重叠大小"""
        # Note: In Pydantic v2, we can't access other field values in field_validator
        # This validation will be moved to model_validator
        return v

    @model_validator(mode='after')
    def validate_chunk_overlap(self):
        """验证重叠大小与块大小的关系"""
        if self.overlap_size >= self.chunk_size:
            raise ValueError('重叠大小不能大于等于块大小')
        return self

    def get_total_size(self) -> int:
        """获取总大小（包括重叠）"""
        return self.chunk_size + self.overlap_size

    def get_duration_ms(self) -> float:
        """获取块时长（毫秒）"""
        return self.chunk_size * 1000.0 / self.sample_rate

    def get_end_timestamp_ms(self) -> float:
        """获取结束时间戳"""
        return self.timestamp_ms + self.get_duration_ms()

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"  # 允许额外字段以支持扩展

class AudioMetadata(BaseModel):
    """音频元数据"""
    title: str | None = Field(default=None, description="标题")
    duration_seconds: float | None = Field(default=None, description="总时长（秒）", ge=0)
    file_size_bytes: int | None = Field(default=None, description="文件大小（字节）", ge=0)
    encoding: str | None = Field(default=None, description="编码格式")
    bitrate: int | None = Field(default=None, description="比特率", ge=0)
    created_at: datetime | None = Field(default=None, description="创建时间")
    source: str | None = Field(default=None, description="音频源")
    quality_score: float | None = Field(default=None, description="质量评分", ge=0.0, le=1.0)

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

# === VAD配置和结果类型 ===

class VADConfig(BaseModel):
    """
    VAD处理配置
    
    控制VAD处理行为的所有参数。
    """
    backend: VADBackend = Field(
        default=VADBackend.ONNX,
        description="VAD后端类型"
    )
    workers: int = Field(
        default=4,
        description="工作线程数",
        ge=1,
        le=32
    )
    threshold: float = Field(
        default=0.5,
        description="VAD检测阈值",
        ge=0.0,
        le=1.0
    )
    chunk_duration_ms: int = Field(
        default=500,
        description="处理块时长（毫秒）",
        ge=100,
        le=5000
    )
    overlap_ms: int = Field(
        default=16,
        description="重叠区域时长（毫秒）",
        ge=0,
        le=200
    )
    buffer_capacity_seconds: int = Field(
        default=5,
        description="缓冲区容量（秒）",
        ge=1,
        le=60
    )
    processing_mode: ProcessingMode = Field(
        default=ProcessingMode.STREAMING,
        description="处理模式"
    )
    optimization_level: OptimizationLevel = Field(
        default=OptimizationLevel.ALL,
        description="优化级别"
    )

    # 高级参数
    min_speech_duration_ms: int = Field(
        default=100,
        description="最小语音段时长（毫秒）",
        ge=10
    )
    max_silence_duration_ms: int = Field(
        default=500,
        description="最大静音段时长（毫秒）",
        ge=50
    )
    energy_threshold: float | None = Field(
        default=None,
        description="能量阈值",
        ge=0.0
    )
    smoothing_window_ms: int = Field(
        default=50,
        description="平滑窗口大小（毫秒）",
        ge=10,
        le=200
    )

    # 延迟补偿配置
    compensation_ms: int = Field(
        default=30,
        description="语音开始延迟补偿时长（毫秒），0表示关闭",
        ge=0,
        le=500
    )

    @field_validator('workers')
    @classmethod
    def validate_workers(cls, v):
        """验证工作线程数"""
        import os
        max_workers = min(32, (os.cpu_count() or 4) * 2)
        if v > max_workers:
            raise ValueError(f'工作线程数不能超过 {max_workers}')
        return v

    @model_validator(mode='after')
    def validate_timing_consistency(self):
        """验证时间参数一致性"""
        # 验证重叠时长
        if self.overlap_ms >= self.chunk_duration_ms * 0.5:
            raise ValueError('重叠时长不能超过块时长的50%')

        # 验证时间参数一致性
        if self.min_speech_duration_ms > self.chunk_duration_ms:
            raise ValueError('最小语音段时长不能超过块时长')

        if self.max_silence_duration_ms > self.chunk_duration_ms * 2:
            raise ValueError('最大静音段时长过长')

        # 验证延迟补偿参数
        if self.compensation_ms > self.chunk_duration_ms:
            raise ValueError('延迟补偿时长不能超过块时长')

        return self

    def get_chunk_samples(self, sample_rate: int) -> int:
        """计算块样本数"""
        return int(self.chunk_duration_ms * sample_rate / 1000)

    def get_overlap_samples(self, sample_rate: int) -> int:
        """计算重叠样本数"""
        return int(self.overlap_ms * sample_rate / 1000)

    class Config:
        extra = "forbid"
        use_enum_values = True
        json_schema_extra = {
            "examples": [
                {
                    "backend": "onnx",
                    "workers": 4,
                    "threshold": 0.5,
                    "chunk_duration_ms": 500,
                    "overlap_ms": 16
                }
            ]
        }
class VADResult(BaseModel):
    """
    VAD检测结果
    
    包含语音活动检测的所有相关信息。
    """
    audio_chunk: NDArray[np.float32] = Field(
        description="当前检测的音频块",
    )
    original_result: Any = Field(
        description="原始的模型输出",
        default=None
    )

    is_speech: bool = Field(
        description="是否检测到语音"
    )
    probability: float = Field(
        description="语音概率",
        ge=0.0,
        le=1.0
    )
    start_ms: float = Field(
        default=0.0,
        description="开始时间（毫秒）",
        ge=0.0
    )
    end_ms: float = Field(
        default=0.0,
        description="结束时间（毫秒）",
        ge=0.0
    )
    chunk_id: int = Field(
        description="块ID",
        ge=0
    )
    confidence: float = Field(
        default=0.0,
        description="置信度",
        ge=0.0,
        le=1.0
    )
    energy_level: float | None = Field(
        default=None,
        description="能量级别",
        ge=0.0
    )
    snr_db: float | None = Field(
        default=None,
        description="信噪比（dB）"
    )
    speech_type: str | None = Field(
        default=None,
        description="语音类型（如：male, female, child）"
    )
    metadata: dict[str, Any] | None = Field(
        default=None,
        description="附加元数据"
    )

    # 延迟补偿字段
    is_compensated: bool = Field(
        default=True,
        description="是否为延迟补偿后的结果"
    )
    original_start_ms: float | None = Field(
        default=None,
        description="补偿前的原始开始时间（毫秒）",
        ge=0.0
    )

    @model_validator(mode='after')
    def validate_time_order(self):
        """验证时间顺序"""
        if self.end_ms <= self.start_ms:
            raise ValueError('结束时间必须大于开始时间')
        return self

    def get_duration_ms(self) -> float:
        """获取时长（毫秒）"""
        return self.end_ms - self.start_ms

    def get_speech_ratio(self) -> float:
        """获取语音比例（用于统计）"""
        return self.probability if self.is_speech else 0.0

    def is_high_confidence(self, threshold: float = 0.8) -> bool:
        """判断是否为高置信度检测"""
        return self.confidence >= threshold

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"
        json_schema_extra = {
            "examples": [
                {
                    "is_speech": True,
                    "probability": 0.85,
                    "start_ms": 1000.0,
                    "end_ms": 1500.0,
                    "chunk_id": 2,
                    "confidence": 0.9
                }
            ]
        }

class VADSegment(BaseModel):
    """VAD语音段"""
    start_ms: float = Field(description="开始时间（毫秒）", ge=0.0)
    end_ms: float = Field(description="结束时间（毫秒）", ge=0.0)
    confidence: float = Field(description="平均置信度", ge=0.0, le=1.0)
    peak_probability: float = Field(description="峰值概率", ge=0.0, le=1.0)
    chunk_count: int = Field(description="包含的块数", ge=1)
    energy_stats: dict[str, float] | None = Field(default=None, description="能量统计")

    @model_validator(mode='after')
    def validate_duration(self):
        if self.end_ms <= self.start_ms:
            raise ValueError('结束时间必须大于开始时间')
        return self

    def get_duration_ms(self) -> float:
        """获取段时长"""
        return self.end_ms - self.start_ms

# === 后端配置类型 ===

class BackendConfig(BaseModel):
    """VAD后端配置基类"""
    model_path: str | None = Field(
        default=None,
        description="模型文件路径"
    )
    device: str = Field(
        default="cpu",
        description="计算设备"
    )
    optimization_level: OptimizationLevel = Field(
        default=OptimizationLevel.ALL,
        description="优化级别"
    )
    max_batch_size: int = Field(
        default=1,
        description="最大批处理大小",
        ge=1,
        le=64
    )
    warmup_iterations: int = Field(
        default=3,
        description="预热迭代次数",
        ge=0,
        le=10
    )

    class Config:
        extra = "allow"  # 允许后端特定配置

class ONNXConfig(BackendConfig):
    """ONNX后端配置"""
    providers: list[str] = Field(
        default=["CPUExecutionProvider"],
        description="执行提供者列表"
    )
    intra_op_num_threads: int = Field(
        default=1,
        description="线程内操作线程数",
        ge=1,
        le=16
    )
    inter_op_num_threads: int = Field(
        default=1,
        description="线程间操作线程数",
        ge=1,
        le=16
    )
    execution_mode: str = Field(
        default="sequential",
        description="执行模式"
    )
    graph_optimization_level: str = Field(
        default="all",
        description="图优化级别"
    )

    @field_validator('providers')
    @classmethod
    def validate_providers(cls, v):
        """验证执行提供者"""
        valid_providers = [
            "CPUExecutionProvider",
            "CUDAExecutionProvider",
            "TensorrtExecutionProvider",
            "OpenVINOExecutionProvider"
        ]
        for provider in v:
            if provider not in valid_providers:
                raise ValueError(f'无效的执行提供者: {provider}')
        return v

    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "model_path": "/path/to/model.onnx",
                    "providers": ["CPUExecutionProvider"],
                    "intra_op_num_threads": 1
                }
            ]
        }

class VLLMConfig(BackendConfig):
    """VLLM后端配置"""
    tensor_parallel_size: int = Field(
        default=1,
        description="张量并行大小",
        ge=1,
        le=8
    )
    max_model_len: int = Field(
        default=2048,
        description="最大模型长度",
        ge=512,
        le=8192
    )
    gpu_memory_utilization: float = Field(
        default=0.9,
        description="GPU内存利用率",
        ge=0.1,
        le=1.0
    )
    swap_space: int = Field(
        default=4,
        description="交换空间（GB）",
        ge=0,
        le=32
    )
    dtype: str = Field(
        default="auto",
        description="数据类型"
    )

    @field_validator('dtype')
    @classmethod
    def validate_dtype(cls, v):
        """验证数据类型"""
        valid_dtypes = ["auto", "half", "float16", "bfloat16", "float", "float32"]
        if v not in valid_dtypes:
            raise ValueError(f'无效的数据类型: {v}')
        return v

class SileroConfig(BackendConfig):
    """Silero VAD后端配置"""
    onnx: bool = Field(
        default=True,
        description="是否使用ONNX模式，默认使用onnx"
    )
    force_reload: bool = Field(
        default=False,
        description="是否强制重新加载模型（仅torch.hub模式）"
    )
    opset_version: int = Field(
        default=16,
        description="ONNX模型opset版本",
        ge=15,
        le=16
    )
    repo_or_dir: str = Field(
        default="snakers4/silero-vad",
        description="模型仓库或目录（torch.hub模式）"
    )
    model_name: str = Field(
        default="silero_vad",
        description="模型名称（torch.hub模式）"
    )
    use_pip_package: bool = Field(
        default=True,
        description="优先使用silero-vad pip包，失败时回退到torch.hub"
    )
    chunk_size_samples: dict[int, int] = Field(
        default={16000: 512, 8000: 256},
        description="不同采样率的块大小映射"
    )
    return_seconds: bool = Field(
        default=True,
        description="VADIterator是否返回时间戳（秒）"
    )

    @field_validator('opset_version')
    @classmethod
    def validate_opset_version(cls, v):
        """验证opset版本"""
        if v == 15:
            # opset_version=15仅支持16kHz
            pass
        elif v == 16:
            # opset_version=16支持8kHz和16kHz
            pass
        else:
            raise ValueError('opset_version必须是15或16')
        return v

    def get_required_chunk_size(self, sample_rate: int) -> int:
        """获取指定采样率的必需块大小"""
        if sample_rate not in self.chunk_size_samples:
            raise ValueError(f'不支持的采样率: {sample_rate}')
        return self.chunk_size_samples[sample_rate]

    def is_chunk_size_compatible(self, sample_rate: int, chunk_size: int) -> bool:
        """检查块大小是否兼容"""
        required_size = self.get_required_chunk_size(sample_rate)
        return chunk_size >= required_size

    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "onnx": False,
                    "force_reload": False,
                    "opset_version": 16,
                    "use_pip_package": True,
                    "return_seconds": False
                }
            ]
        }

# === 通用状态类型 ===

class Status(str, Enum):
    """通用状态类型"""
    IDLE = "idle"
    RUNNING = "running"
    STOPPED = "stopped"
    ERROR = "error"
    UNKNOWN = "unknown"

# 从子模块导入额外类型
from .errors import (
    AudioFormatError,
    BufferError,
    BufferFullError,
    CascadeError,
    ErrorCode,
    ErrorInfo,
    ErrorSeverity,
    InsufficientDataError,
    ModelLoadError,
    VADProcessingError,
)
from .performance import BufferStatus, PerformanceMetrics, SystemStatus
from .version import CompatibilityInfo, VersionInfo

# === 导出的类型定义 ===

__all__ = [
    # 枚举类型
    "AudioFormat", "VADBackend", "ProcessingMode", "LogLevel",
    "BufferStrategy", "OptimizationLevel", "Status",

    # 音频类型
    "AudioConfig", "AudioChunk", "AudioMetadata",

    # VAD类型
    "VADConfig", "VADResult", "VADSegment",

    # 后端配置
    "BackendConfig", "ONNXConfig", "VLLMConfig", "SileroConfig",

    # 错误处理类型
    "ErrorCode", "ErrorSeverity", "CascadeError", "AudioFormatError",
    "BufferError", "BufferFullError", "InsufficientDataError",
    "VADProcessingError", "ModelLoadError", "ErrorInfo",

    # 性能监控类型
    "PerformanceMetrics", "SystemStatus", "BufferStatus",

    # 版本兼容性类型
    "VersionInfo", "CompatibilityInfo",
]
