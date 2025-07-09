"""
VAD相关类型定义

本模块定义了与语音活动检测(VAD)相关的核心数据类型，包括：
- ProcessingMode: 处理模式枚举
- OptimizationLevel: 优化级别枚举
- VADBackend: VAD后端类型枚举
- VADConfig: VAD处理配置
- VADResult: VAD检测结果
- VADSegment: VAD语音段
"""

import os
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


class ProcessingMode(str, Enum):
    """处理模式"""
    STREAMING = "streaming"      # 流式处理
    BATCH = "batch"             # 批量处理
    REALTIME = "realtime"       # 实时处理


class OptimizationLevel(str, Enum):
    """优化级别"""
    NONE = "none"               # 无优化
    BASIC = "basic"             # 基础优化
    AGGRESSIVE = "aggressive"    # 激进优化
    ALL = "all"                 # 全部优化


class VADBackend(str, Enum):
    """支持的VAD后端"""
    ONNX = "onnx"
    VLLM = "vllm"

    @classmethod
    def get_default_backend(cls) -> str:
        """获取默认后端"""
        return cls.ONNX.value


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

    @field_validator('overlap_ms')
    def validate_overlap(cls, v, info):
        """验证重叠时长"""
        # 在Pydantic V2中，values参数变成了ValidationInfo对象
        chunk_duration = info.data.get('chunk_duration_ms', 500)
        if v >= chunk_duration * 0.5:
            raise ValueError('重叠时长不能超过块时长的50%')
        return v

    @field_validator('workers')
    def validate_workers(cls, v):
        """验证工作线程数"""
        max_workers = min(32, (os.cpu_count() or 4) * 2)
        if v > max_workers:
            raise ValueError(f'工作线程数不能超过 {max_workers}')
        return v

    @model_validator(mode='after')
    def validate_timing_consistency(self):
        """验证时间参数一致性"""
        # 在Pydantic V2中，使用model_validator替代root_validator
        chunk_duration = self.chunk_duration_ms
        min_speech = self.min_speech_duration_ms
        max_silence = self.max_silence_duration_ms

        if min_speech > chunk_duration:
            raise ValueError('最小语音段时长不能超过块时长')

        if max_silence > chunk_duration * 2:
            raise ValueError('最大静音段时长过长')

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
        schema_extra = {
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
    is_speech: bool = Field(
        description="是否检测到语音"
    )
    probability: float = Field(
        description="语音概率",
        ge=0.0,
        le=1.0
    )
    start_ms: float = Field(
        description="开始时间（毫秒）",
        ge=0.0
    )
    end_ms: float = Field(
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

    @field_validator('end_ms')
    def validate_time_order(cls, v, info):
        """验证时间顺序"""
        # 在Pydantic V2中，values参数变成了ValidationInfo对象
        start_ms = info.data.get('start_ms')
        if start_ms is not None and v <= start_ms:
            raise ValueError('结束时间必须大于开始时间')
        return v

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
        extra = "allow"
        schema_extra = {
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

    @field_validator('end_ms')
    def validate_duration(cls, v, info):
        # 在Pydantic V2中，values参数变成了ValidationInfo对象
        start_ms = info.data.get('start_ms')
        if start_ms is not None and v <= start_ms:
            raise ValueError('结束时间必须大于开始时间')
        return v

    def get_duration_ms(self) -> float:
        """获取段时长"""
        return self.end_ms - self.start_ms
