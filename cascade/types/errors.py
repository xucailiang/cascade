"""
错误处理类型系统

定义Cascade项目中所有的错误类型、错误码和异常处理相关的类型。
"""

from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

# === 错误处理类型 ===

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

    # 线程池相关错误
    PROCESSING_FAILED = "E5001"
    CLEANUP_FAILED = "E5002"
    INVALID_STATE = "E5003"

class ErrorSeverity(str, Enum):
    """错误严重程度"""
    LOW = "low"           # 低：不影响功能
    MEDIUM = "medium"     # 中：影响性能
    HIGH = "high"         # 高：影响功能
    CRITICAL = "critical" # 严重：系统不可用

class CascadeError(Exception):
    """Cascade错误基类"""
    def __init__(self,
                 message: str,
                 error_code: ErrorCode = ErrorCode.UNKNOWN_ERROR,
                 severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                 context: dict[str, Any] | None = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.severity = severity
        self.context = context or {}
        self.timestamp = datetime.now(UTC)

    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return {
            "message": self.message,
            "error_code": self.error_code.value,
            "severity": self.severity.value,
            "context": self.context,
            "timestamp": self.timestamp.isoformat()
        }

class AudioFormatError(CascadeError):
    """音频格式错误"""
    def __init__(self, message: str, format_info: dict[str, Any] | None = None):
        super().__init__(
            message,
            ErrorCode.UNSUPPORTED_FORMAT,
            ErrorSeverity.HIGH,
            {"format_info": format_info}
        )

class BufferError(CascadeError):
    """缓冲区错误"""
    pass

class BufferFullError(BufferError):
    """缓冲区已满错误"""
    def __init__(self, capacity: int, attempted_size: int):
        super().__init__(
            f"缓冲区已满: 容量={capacity}, 尝试写入={attempted_size}",
            ErrorCode.BUFFER_FULL,
            ErrorSeverity.MEDIUM,
            {"capacity": capacity, "attempted_size": attempted_size}
        )

class InsufficientDataError(BufferError):
    """数据不足错误"""
    def __init__(self, available: int, required: int):
        super().__init__(
            f"数据不足: 可用={available}, 需要={required}",
            ErrorCode.INSUFFICIENT_DATA,
            ErrorSeverity.LOW,
            {"available": available, "required": required}
        )

class VADProcessingError(CascadeError):
    """VAD处理错误"""
    pass

class ModelLoadError(VADProcessingError):
    """模型加载错误"""
    def __init__(self, model_path: str, reason: str):
        super().__init__(
            f"模型加载失败: {model_path}, 原因: {reason}",
            ErrorCode.MODEL_LOAD_FAILED,
            ErrorSeverity.CRITICAL,
            {"model_path": model_path, "reason": reason}
        )

class ErrorInfo(BaseModel):
    """错误信息"""
    error_code: ErrorCode = Field(description="错误码")
    message: str = Field(description="错误消息")
    severity: ErrorSeverity = Field(description="严重程度")
    timestamp: datetime = Field(description="发生时间")
    context: dict[str, Any] = Field(default={}, description="错误上下文")
    stack_trace: str | None = Field(default=None, description="堆栈跟踪")
    recovery_suggestions: list[str] = Field(default=[], description="恢复建议")

    @classmethod
    def from_exception(cls, exc: Exception) -> 'ErrorInfo':
        """从异常创建错误信息"""
        if isinstance(exc, CascadeError):
            return cls(
                error_code=exc.error_code,
                message=exc.message,
                severity=exc.severity,
                timestamp=exc.timestamp,
                context=exc.context
            )
        else:
            return cls(
                error_code=ErrorCode.UNKNOWN_ERROR,
                message=str(exc),
                severity=ErrorSeverity.MEDIUM,
                timestamp=datetime.now(UTC),
                context={"exception_type": type(exc).__name__}
            )

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


__all__ = [
    "ErrorCode", "ErrorSeverity", "CascadeError", "AudioFormatError",
    "BufferError", "BufferFullError", "InsufficientDataError",
    "VADProcessingError", "ModelLoadError", "ErrorInfo"
]
