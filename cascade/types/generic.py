"""
通用类型定义

本模块定义了项目中使用的通用数据类型，包括：
- LogLevel: 日志级别枚举
- BufferStrategy: 缓冲区策略枚举
- Status: 系统状态类
- PerformanceMetrics: 性能监控指标
- SystemStatus: 系统状态
- BufferStatus: 缓冲区状态
- ErrorCode: 错误码定义
- ErrorSeverity: 错误严重程度
- ErrorInfo: 错误信息
"""

from datetime import UTC, datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


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


class Status(BaseModel):
    """
    系统状态

    表示操作或系统的当前状态。
    """
    code: int = Field(
        description="状态码",
        default=0
    )
    message: str = Field(
        description="状态消息",
        default="OK"
    )
    success: bool = Field(
        description="是否成功",
        default=True
    )
    timestamp: datetime = Field(
        description="时间戳",
        default_factory=lambda: datetime.now(UTC)
    )
    details: dict[str, Any] | None = Field(
        description="详细信息",
        default=None
    )

    def is_ok(self) -> bool:
        """检查状态是否正常"""
        return self.success and self.code == 0

    def is_error(self) -> bool:
        """检查状态是否为错误"""
        return not self.success or self.code != 0

    @classmethod
    def ok(cls, message: str = "OK", details: dict[str, Any] | None = None) -> 'Status':
        """创建成功状态"""
        return cls(
            code=0,
            message=message,
            success=True,
            details=details
        )

    @classmethod
    def error(cls, code: int, message: str, details: dict[str, Any] | None = None) -> 'Status':
        """创建错误状态"""
        return cls(
            code=code,
            message=message,
            success=False,
            details=details
        )

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


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
    queue_depth: int = Field(
        description="队列深度",
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

    def is_healthy(self) -> bool:
        """判断缓冲区是否健康"""
        return self.status_level != "critical" and self.overflow_count == 0


class ErrorCode(str, Enum):
    """错误码定义"""
    # 通用错误
    UNKNOWN_ERROR = "E0000"
    INVALID_INPUT = "E0001"
    INVALID_CONFIG = "E0002"
    INITIALIZATION_FAILED = "E0003"

    # 音频相关错误
    UNSUPPORTED_FORMAT = "E1001"
    INVALID_SAMPLE_RATE = "E1002"
    INVALID_CHANNELS = "E1003"
    AUDIO_CORRUPTION = "E1004"

    # 缓冲区错误
    BUFFER_FULL = "E2001"
    BUFFER_EMPTY = "E2002"
    INSUFFICIENT_DATA = "E2003"
    BUFFER_CORRUPTION = "E2004"

    # VAD处理错误
    MODEL_LOAD_FAILED = "E3001"
    INFERENCE_FAILED = "E3002"
    RESULT_VALIDATION_FAILED = "E3003"
    BACKEND_UNAVAILABLE = "E3004"

    # 性能相关错误
    TIMEOUT_ERROR = "E4001"
    MEMORY_ERROR = "E4002"
    THREAD_ERROR = "E4003"
    RESOURCE_EXHAUSTED = "E4004"


class ErrorSeverity(str, Enum):
    """错误严重程度"""
    LOW = "low"           # 低：不影响功能
    MEDIUM = "medium"     # 中：影响性能
    HIGH = "high"         # 高：影响功能
    CRITICAL = "critical" # 严重：系统不可用


class ErrorInfo(BaseModel):
    """错误信息"""
    error_code: ErrorCode = Field(description="错误码")
    message: str = Field(description="错误消息")
    severity: ErrorSeverity = Field(description="严重程度")
    timestamp: datetime = Field(description="发生时间")
    context: dict[str, Any] = Field(default={}, description="错误上下文")
    stack_trace: str | None = Field(default=None, description="堆栈跟踪")
    recovery_suggestions: list[str] = Field(default=[], description="恢复建议")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
