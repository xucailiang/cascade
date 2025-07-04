"""
错误类型单元测试

测试错误处理相关类型的功能。
"""

import pytest
from datetime import datetime, timezone

from cascade.types import (
    ErrorCode, ErrorSeverity, ErrorInfo,
    PreVADError, AudioFormatError, BufferError, BufferFullError,
    InsufficientDataError, VADProcessingError, ModelLoadError,
    BackendUnavailableError, InferenceError, ConfigurationError, TimeoutError
)


class TestPreVADError:
    """测试PreVADError基类"""
    
    def test_basic_properties(self):
        """测试基本属性"""
        error = PreVADError(
            message="测试错误",
            error_code=ErrorCode.UNKNOWN_ERROR,
            severity=ErrorSeverity.MEDIUM,
            context={"key": "value"}
        )
        
        assert error.message == "测试错误"
        assert error.error_code == ErrorCode.UNKNOWN_ERROR
        assert error.severity == ErrorSeverity.MEDIUM
        assert error.context == {"key": "value"}
        assert isinstance(error.timestamp, datetime)
        
        # 确保继承自Exception
        assert isinstance(error, Exception)
    
    def test_default_values(self):
        """测试默认值"""
        error = PreVADError("测试错误")
        
        assert error.message == "测试错误"
        assert error.error_code == ErrorCode.UNKNOWN_ERROR
        assert error.severity == ErrorSeverity.MEDIUM
        assert error.context == {}
    
    def test_to_dict(self):
        """测试转换为字典"""
        error = PreVADError(
            message="测试错误",
            error_code=ErrorCode.INVALID_INPUT,
            severity=ErrorSeverity.HIGH,
            context={"key": "value"}
        )
        
        error_dict = error.to_dict()
        assert error_dict["message"] == "测试错误"
        assert error_dict["error_code"] == "E0001"  # INVALID_INPUT的值
        assert error_dict["severity"] == "high"
        assert error_dict["context"] == {"key": "value"}
        assert "timestamp" in error_dict


class TestAudioFormatError:
    """测试AudioFormatError类"""
    
    def test_basic_properties(self):
        """测试基本属性"""
        format_info = {"format": "mp3", "sample_rate": 44100}
        error = AudioFormatError(
            message="不支持的音频格式",
            format_info=format_info
        )
        
        assert error.message == "不支持的音频格式"
        assert error.error_code == ErrorCode.UNSUPPORTED_FORMAT
        assert error.severity == ErrorSeverity.HIGH
        assert error.context["format_info"] == format_info
        
        # 确保继承自PreVADError
        assert isinstance(error, PreVADError)


class TestBufferError:
    """测试BufferError类"""
    
    def test_inheritance(self):
        """测试继承关系"""
        error = BufferError(
            message="缓冲区错误",
            error_code=ErrorCode.BUFFER_CORRUPTION,
            severity=ErrorSeverity.HIGH
        )
        
        assert isinstance(error, PreVADError)


class TestBufferFullError:
    """测试BufferFullError类"""
    
    def test_basic_properties(self):
        """测试基本属性"""
        error = BufferFullError(capacity=1000, attempted_size=1500)
        
        assert "缓冲区已满" in error.message
        assert "容量=1000" in error.message
        assert "尝试写入=1500" in error.message
        assert error.error_code == ErrorCode.BUFFER_FULL
        assert error.severity == ErrorSeverity.MEDIUM
        assert error.context["capacity"] == 1000
        assert error.context["attempted_size"] == 1500
        
        # 确保继承自BufferError
        assert isinstance(error, BufferError)


class TestInsufficientDataError:
    """测试InsufficientDataError类"""
    
    def test_basic_properties(self):
        """测试基本属性"""
        error = InsufficientDataError(available=500, required=1000)
        
        assert "数据不足" in error.message
        assert "可用=500" in error.message
        assert "需要=1000" in error.message
        assert error.error_code == ErrorCode.INSUFFICIENT_DATA
        assert error.severity == ErrorSeverity.LOW
        assert error.context["available"] == 500
        assert error.context["required"] == 1000
        
        # 确保继承自BufferError
        assert isinstance(error, BufferError)


class TestVADProcessingError:
    """测试VADProcessingError类"""
    
    def test_inheritance(self):
        """测试继承关系"""
        error = VADProcessingError(
            message="VAD处理错误",
            error_code=ErrorCode.INFERENCE_FAILED,
            severity=ErrorSeverity.HIGH
        )
        
        assert isinstance(error, PreVADError)


class TestModelLoadError:
    """测试ModelLoadError类"""
    
    def test_basic_properties(self):
        """测试基本属性"""
        error = ModelLoadError(
            model_path="/path/to/model",
            reason="文件不存在"
        )
        
        assert "模型加载失败" in error.message
        assert "/path/to/model" in error.message
        assert "文件不存在" in error.message
        assert error.error_code == ErrorCode.MODEL_LOAD_FAILED
        assert error.severity == ErrorSeverity.CRITICAL
        assert error.context["model_path"] == "/path/to/model"
        assert error.context["reason"] == "文件不存在"
        
        # 确保继承自VADProcessingError
        assert isinstance(error, VADProcessingError)


class TestBackendUnavailableError:
    """测试BackendUnavailableError类"""
    
    def test_basic_properties(self):
        """测试基本属性"""
        error = BackendUnavailableError(
            backend_name="onnx",
            reason="依赖缺失"
        )
        
        assert "后端不可用" in error.message
        assert "onnx" in error.message
        assert "依赖缺失" in error.message
        assert error.error_code == ErrorCode.BACKEND_UNAVAILABLE
        assert error.severity == ErrorSeverity.CRITICAL
        assert error.context["backend_name"] == "onnx"
        assert error.context["reason"] == "依赖缺失"
        
        # 确保继承自VADProcessingError
        assert isinstance(error, VADProcessingError)


class TestInferenceError:
    """测试InferenceError类"""
    
    def test_basic_properties(self):
        """测试基本属性"""
        details = {"model": "silero_vad", "input_shape": [1, 1000]}
        error = InferenceError(
            message="推理过程中出错",
            details=details
        )
        
        assert error.message == "推理过程中出错"
        assert error.error_code == ErrorCode.INFERENCE_FAILED
        assert error.severity == ErrorSeverity.HIGH
        assert error.context == details
        
        # 确保继承自VADProcessingError
        assert isinstance(error, VADProcessingError)


class TestConfigurationError:
    """测试ConfigurationError类"""
    
    def test_basic_properties(self):
        """测试基本属性"""
        details = {"param": "sample_rate", "value": -1}
        error = ConfigurationError(
            message="配置参数无效",
            config_name="AudioConfig",
            details=details
        )
        
        assert error.message == "配置参数无效"
        assert error.error_code == ErrorCode.INVALID_CONFIG
        assert error.severity == ErrorSeverity.HIGH
        assert error.context["config_name"] == "AudioConfig"
        assert error.context["param"] == "sample_rate"
        assert error.context["value"] == -1
        
        # 确保继承自PreVADError
        assert isinstance(error, PreVADError)


class TestTimeoutError:
    """测试TimeoutError类"""
    
    def test_basic_properties(self):
        """测试基本属性"""
        error = TimeoutError(
            operation="模型推理",
            timeout_seconds=30.0
        )
        
        assert "操作超时" in error.message
        assert "模型推理" in error.message
        assert "30.0秒" in error.message
        assert error.error_code == ErrorCode.TIMEOUT_ERROR
        assert error.severity == ErrorSeverity.MEDIUM
        assert error.context["operation"] == "模型推理"
        assert error.context["timeout_seconds"] == 30.0
        
        # 确保继承自PreVADError
        assert isinstance(error, PreVADError)


class TestErrorInfo:
    """测试ErrorInfo类"""
    
    def test_basic_properties(self):
        """测试基本属性"""
        error_info = ErrorInfo(
            error_code=ErrorCode.INVALID_INPUT,
            message="无效的输入参数",
            severity=ErrorSeverity.HIGH,
            timestamp=datetime.now(timezone.utc),
            context={"param": "sample_rate", "value": -1},
            stack_trace="...",
            recovery_suggestions=["检查输入参数", "参考文档"]
        )
        
        assert error_info.error_code == ErrorCode.INVALID_INPUT
        assert error_info.message == "无效的输入参数"
        assert error_info.severity == ErrorSeverity.HIGH
        assert isinstance(error_info.timestamp, datetime)
        assert error_info.context == {"param": "sample_rate", "value": -1}
        assert error_info.stack_trace == "..."
        assert error_info.recovery_suggestions == ["检查输入参数", "参考文档"]
    
    def test_from_exception(self):
        """测试从异常创建错误信息"""
        # 从PreVADError创建
        error = PreVADError(
            message="测试错误",
            error_code=ErrorCode.INVALID_INPUT,
            severity=ErrorSeverity.HIGH,
            context={"key": "value"}
        )
        
        error_info = ErrorInfo.from_exception(error)
        assert error_info.error_code == ErrorCode.INVALID_INPUT
        assert error_info.message == "测试错误"
        assert error_info.severity == ErrorSeverity.HIGH
        assert error_info.context == {"key": "value"}
        
        # 从普通Exception创建
        std_error = ValueError("标准错误")
        error_info = ErrorInfo.from_exception(std_error)
        assert error_info.error_code == ErrorCode.UNKNOWN_ERROR
        assert error_info.message == "标准错误"
        assert error_info.severity == ErrorSeverity.MEDIUM
        assert "exception_type" in error_info.context
        assert error_info.context["exception_type"] == "ValueError"