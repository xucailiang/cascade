"""
错误处理类型定义

本模块定义了项目中使用的异常类型，提供统一的错误处理机制，包括：
- PreVADError: 基础异常类
- AudioFormatError: 音频格式错误
- BufferError: 缓冲区错误
- BufferFullError: 缓冲区已满错误
- InsufficientDataError: 数据不足错误
- VADProcessingError: VAD处理错误
- ModelLoadError: 模型加载错误
"""

from datetime import UTC, datetime
from typing import Any

from .generic import ErrorCode, ErrorSeverity


class PreVADError(Exception):
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


class AudioFormatError(PreVADError):
    """音频格式错误"""
    def __init__(self, message: str, format_info: dict[str, Any] | None = None):
        super().__init__(
            message,
            ErrorCode.UNSUPPORTED_FORMAT,
            ErrorSeverity.HIGH,
            {"format_info": format_info}
        )


class BufferError(PreVADError):
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


class VADProcessingError(PreVADError):
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


class BackendUnavailableError(VADProcessingError):
    """后端不可用错误"""
    def __init__(self, backend_name: str, reason: str):
        super().__init__(
            f"后端不可用: {backend_name}, 原因: {reason}",
            ErrorCode.BACKEND_UNAVAILABLE,
            ErrorSeverity.CRITICAL,
            {"backend_name": backend_name, "reason": reason}
        )


class InferenceError(VADProcessingError):
    """推理错误"""
    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(
            message,
            ErrorCode.INFERENCE_FAILED,
            ErrorSeverity.HIGH,
            details
        )


class ConfigurationError(PreVADError):
    """配置错误"""
    def __init__(self, message: str, config_name: str, details: dict[str, Any] | None = None):
        context = {"config_name": config_name}
        if details:
            context.update(details)

        super().__init__(
            message,
            ErrorCode.INVALID_CONFIG,
            ErrorSeverity.HIGH,
            context
        )


class TimeoutError(PreVADError):
    """超时错误"""
    def __init__(self, operation: str, timeout_seconds: float):
        super().__init__(
            f"操作超时: {operation}, 超时时间: {timeout_seconds}秒",
            ErrorCode.TIMEOUT_ERROR,
            ErrorSeverity.MEDIUM,
            {"operation": operation, "timeout_seconds": timeout_seconds}
        )
