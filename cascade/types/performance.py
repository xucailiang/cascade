"""
性能监控类型系统

定义Cascade项目中所有性能监控、系统状态和缓冲区状态相关的类型。
"""

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

# === 性能监控类型 ===

class PerformanceMetrics(BaseModel):
    """
    性能监控指标
    
    收集和展示系统性能数据。
    """
    # 延迟指标
    avg_latency_ms: float = Field(
        description="平均延迟（毫秒）",
        ge=0.0
    )
    p50_latency_ms: float = Field(
        description="P50延迟（毫秒）",
        ge=0.0
    )
    p95_latency_ms: float = Field(
        description="P95延迟（毫秒）",
        ge=0.0
    )
    p99_latency_ms: float = Field(
        description="P99延迟（毫秒）",
        ge=0.0
    )
    max_latency_ms: float = Field(
        description="最大延迟（毫秒）",
        ge=0.0
    )

    # 吞吐量指标
    throughput_qps: float = Field(
        description="吞吐量（QPS）",
        ge=0.0
    )
    throughput_mbps: float = Field(
        description="数据吞吐量（MB/s）",
        ge=0.0
    )

    # 错误指标
    error_rate: float = Field(
        description="错误率",
        ge=0.0,
        le=1.0
    )
    success_count: int = Field(
        description="成功次数",
        ge=0
    )
    error_count: int = Field(
        description="错误次数",
        ge=0
    )

    # 资源指标
    memory_usage_mb: float = Field(
        description="内存使用量（MB）",
        ge=0.0
    )
    cpu_usage_percent: float = Field(
        description="CPU使用率（%）",
        ge=0.0,
        le=100.0
    )
    active_threads: int = Field(
        description="活跃线程数",
        ge=0
    )

    # 缓冲区指标
    buffer_utilization: float = Field(
        description="缓冲区利用率",
        ge=0.0,
        le=1.0
    )
    zero_copy_rate: float = Field(
        description="零拷贝率",
        ge=0.0,
        le=1.0
    )
    cache_hit_rate: float = Field(
        description="缓存命中率",
        ge=0.0,
        le=1.0
    )

    # 扩展指标
    additional_metrics: dict[str, Any] = Field(
        default_factory=dict,
        description="额外的指标数据"
    )

    # 时间戳
    measurement_time: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="测量时间"
    )
    collection_duration_seconds: float = Field(
        description="收集时长（秒）",
        gt=0.0
    )

    def get_total_operations(self) -> int:
        """获取总操作数"""
        return self.success_count + self.error_count

    def get_success_rate(self) -> float:
        """获取成功率"""
        total = self.get_total_operations()
        return self.success_count / total if total > 0 else 0.0

    def is_healthy(self,
                   max_error_rate: float = 0.01,
                   min_throughput: float = 1.0,
                   max_latency_p99: float = 100.0) -> bool:
        """判断性能是否健康"""
        return (self.error_rate <= max_error_rate and
                self.throughput_qps >= min_throughput and
                self.p99_latency_ms <= max_latency_p99)

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class SystemStatus(BaseModel):
    """系统状态"""
    status: Literal["healthy", "warning", "critical", "unknown"] = Field(
        description="系统状态"
    )
    uptime_seconds: float = Field(
        description="运行时间（秒）",
        ge=0.0
    )
    total_processed_chunks: int = Field(
        description="已处理块总数",
        ge=0
    )
    total_audio_duration_seconds: float = Field(
        description="已处理音频总时长（秒）",
        ge=0.0
    )
    current_load: float = Field(
        description="当前负载",
        ge=0.0,
        le=1.0
    )
    health_issues: list[str] = Field(
        default=[],
        description="健康问题列表"
    )
    last_error: str | None = Field(
        default=None,
        description="最近的错误"
    )
    performance_summary: PerformanceMetrics | None = Field(
        default=None,
        description="性能摘要"
    )

    def is_operational(self) -> bool:
        """判断是否可操作"""
        return self.status in ["healthy", "warning"]

    def add_health_issue(self, issue: str) -> None:
        """添加健康问题"""
        if issue not in self.health_issues:
            self.health_issues.append(issue)

    def clear_health_issues(self) -> None:
        """清除健康问题"""
        self.health_issues.clear()

class BufferStatus(BaseModel):
    """缓冲区状态"""
    capacity: int = Field(description="总容量（样本数）", gt=0)
    available_samples: int = Field(description="可用样本数", ge=0)
    free_samples: int = Field(description="空闲样本数", ge=0)
    usage_ratio: float = Field(description="使用率", ge=0.0, le=1.0)
    status_level: Literal["normal", "warning", "critical"] = Field(description="状态级别")
    write_position: int = Field(description="写入位置", ge=0)
    read_position: int = Field(description="读取位置", ge=0)
    overflow_count: int = Field(description="溢出次数", ge=0)
    underflow_count: int = Field(description="下溢次数", ge=0)
    peak_usage: float = Field(description="峰值使用率", ge=0.0, le=1.0)

    @field_validator('available_samples')
    @classmethod
    def validate_available_samples(cls, v, info):
        if hasattr(info, 'data') and 'capacity' in info.data:
            capacity = info.data.get('capacity', 0)
            if v > capacity:
                raise ValueError('可用样本数不能超过总容量')
        return v

    def is_healthy(self) -> bool:
        """判断缓冲区是否健康"""
        return self.status_level != "critical" and self.overflow_count == 0

__all__ = [
    "PerformanceMetrics", "SystemStatus", "BufferStatus"
]
