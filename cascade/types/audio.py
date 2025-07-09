"""
音频相关类型定义

本模块定义了与音频处理相关的核心数据类型，包括：
- AudioFormat: 支持的音频格式枚举
- AudioConfig: 音频处理配置
- AudioChunk: 音频数据块
- AudioMetadata: 音频元数据
"""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


class AudioFormat(str, Enum):
    """支持的音频格式"""
    WAV = "wav"
    PCMA = "pcma"

    @classmethod
    def get_supported_formats(cls) -> list[str]:
        """获取支持的格式列表"""
        return [format.value for format in cls]


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
    def validate_sample_rate(cls, v):
        """验证采样率"""
        supported_rates = [8000, 16000, 22050, 44100, 48000]
        if v not in supported_rates:
            raise ValueError(f'采样率必须是以下之一: {supported_rates}')
        return v

    @field_validator('channels')
    def validate_channels(cls, v):
        """验证声道数"""
        if v != 1:
            raise ValueError('当前版本仅支持单声道音频')
        return v

    @field_validator('dtype')
    def validate_dtype(cls, v):
        """验证数据类型"""
        supported_dtypes = ['float32', 'float64', 'int16', 'int32']
        if v not in supported_dtypes:
            raise ValueError(f'数据类型必须是以下之一: {supported_dtypes}')
        return v

    @model_validator(mode='after')
    def validate_format_compatibility(self):
        """验证格式兼容性"""
        format_type = self.format
        sample_rate = self.sample_rate

        if format_type == AudioFormat.PCMA:
            # PCMA格式限制
            if sample_rate not in [8000, 16000]:
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
        schema_extra = {
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
    def validate_overlap_size(cls, v, info):
        """验证重叠大小"""
        chunk_size = info.data.get('chunk_size', 0)
        if v >= chunk_size:
            raise ValueError('重叠大小不能大于等于块大小')
        return v

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
