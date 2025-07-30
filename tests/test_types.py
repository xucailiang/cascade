"""
Cascade 核心类型系统单元测试

测试所有核心数据类型、验证器和错误处理机制
"""

from datetime import datetime

import numpy as np
import pytest

from cascade.types import (
    AudioChunk,
    # 音频类型
    AudioConfig,
    # 枚举类型
    AudioFormat,
    AudioFormatError,
    BufferFullError,
    CascadeError,
    # 错误类型
    ErrorCode,
    ErrorInfo,
    ErrorSeverity,
    ModelLoadError,
    ONNXConfig,
    VADBackend,
    # VAD类型
    VADConfig,
    VADResult,
    VLLMConfig,
)


class TestAudioConfig:
    """音频配置测试"""

    def test_default_config(self):
        """测试默认配置"""
        config = AudioConfig()
        assert config.sample_rate == 16000
        assert config.format == AudioFormat.WAV
        assert config.channels == 1
        assert config.dtype == "float32"
        assert config.bit_depth is None

    def test_valid_sample_rates(self):
        """测试有效采样率"""
        valid_rates = [8000, 16000, 22050, 44100, 48000]
        for rate in valid_rates:
            config = AudioConfig(sample_rate=rate)
            assert config.sample_rate == rate

    def test_invalid_sample_rate(self):
        """测试无效采样率"""
        with pytest.raises(ValueError, match="采样率必须是以下之一"):
            AudioConfig(sample_rate=12000)

    def test_invalid_channels(self):
        """测试无效声道数"""
        with pytest.raises(ValueError, match="当前版本仅支持单声道"):
            AudioConfig(channels=2)

    def test_invalid_dtype(self):
        """测试无效数据类型"""
        with pytest.raises(ValueError, match="数据类型必须是以下之一"):
            AudioConfig(dtype="float16")

    def test_pcma_format_validation(self):
        """测试PCMA格式验证"""
        # 有效的PCMA配置
        config = AudioConfig(format=AudioFormat.PCMA, sample_rate=8000)
        assert config.format == AudioFormat.PCMA

        config = AudioConfig(format=AudioFormat.PCMA, sample_rate=16000)
        assert config.format == AudioFormat.PCMA

        # 无效的PCMA配置
        with pytest.raises(ValueError, match="PCMA格式仅支持8kHz和16kHz"):
            AudioConfig(format=AudioFormat.PCMA, sample_rate=44100)

    def test_frame_size_calculation(self):
        """测试帧大小计算"""
        config = AudioConfig(sample_rate=16000)
        assert config.get_frame_size(1000) == 16000  # 1秒
        assert config.get_frame_size(500) == 8000    # 0.5秒
        assert config.get_frame_size(100) == 1600    # 0.1秒

    def test_bytes_per_second(self):
        """测试每秒字节数计算"""
        config = AudioConfig(sample_rate=16000, dtype="float32")
        assert config.get_bytes_per_second() == 16000 * 1 * 4  # 64000

        config = AudioConfig(sample_rate=16000, dtype="int16")
        assert config.get_bytes_per_second() == 16000 * 1 * 2  # 32000


class TestAudioChunk:
    """音频数据块测试"""

    def test_basic_chunk(self):
        """测试基本数据块"""
        data = np.array([1, 2, 3, 4, 5], dtype=np.float32)
        chunk = AudioChunk(
            data=data,
            sequence_number=1,
            start_frame=0,
            chunk_size=5,
            timestamp_ms=0.0,
            sample_rate=16000
        )
        assert chunk.sequence_number == 1
        assert chunk.chunk_size == 5
        assert chunk.get_total_size() == 5
        assert chunk.get_duration_ms() == 5 * 1000.0 / 16000

    def test_chunk_with_overlap(self):
        """测试包含重叠的数据块"""
        data = np.array([1, 2, 3, 4, 5, 6, 7], dtype=np.float32)
        chunk = AudioChunk(
            data=data,
            sequence_number=2,
            start_frame=10,
            chunk_size=5,
            overlap_size=2,
            timestamp_ms=100.0,
            sample_rate=16000
        )
        assert chunk.get_total_size() == 7
        assert chunk.get_end_timestamp_ms() == 100.0 + chunk.get_duration_ms()

    def test_invalid_overlap_size(self):
        """测试无效重叠大小"""
        with pytest.raises(ValueError, match="重叠大小不能大于等于块大小"):
            AudioChunk(
                data=np.array([1, 2, 3]),
                sequence_number=1,
                start_frame=0,
                chunk_size=3,
                overlap_size=3,  # 等于块大小
                timestamp_ms=0.0,
                sample_rate=16000
            )


class TestVADConfig:
    """VAD配置测试"""

    def test_default_vad_config(self):
        """测试默认VAD配置"""
        config = VADConfig()
        assert config.backend == VADBackend.ONNX
        assert config.workers == 4
        assert config.threshold == 0.5
        assert config.chunk_duration_ms == 500
        assert config.overlap_ms == 16

    def test_invalid_overlap(self):
        """测试无效重叠时长"""
        # Pydantic V2的验证错误消息格式不同，使用ValidationError
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            VADConfig(chunk_duration_ms=500, overlap_ms=300)

    def test_invalid_timing_consistency(self):
        """测试时间参数一致性"""
        # 最小语音段时长过长
        with pytest.raises(ValueError, match="最小语音段时长不能超过块时长"):
            VADConfig(chunk_duration_ms=500, min_speech_duration_ms=600)

        # 最大静音段时长过长
        with pytest.raises(ValueError, match="最大静音段时长过长"):
            VADConfig(chunk_duration_ms=500, max_silence_duration_ms=1200)

    def test_chunk_samples_calculation(self):
        """测试块样本数计算"""
        config = VADConfig(chunk_duration_ms=500)
        assert config.get_chunk_samples(16000) == 8000
        assert config.get_chunk_samples(8000) == 4000

    def test_overlap_samples_calculation(self):
        """测试重叠样本数计算"""
        config = VADConfig(overlap_ms=16)
        assert config.get_overlap_samples(16000) == 256
        assert config.get_overlap_samples(8000) == 128


class TestVADResult:
    """VAD结果测试"""

    def test_basic_vad_result(self):
        """测试基本VAD结果"""
        result = VADResult(
            is_speech=True,
            probability=0.85,
            start_ms=1000.0,
            end_ms=1500.0,
            chunk_id=2,
            confidence=0.9
        )
        assert result.is_speech
        assert result.get_duration_ms() == 500.0
        assert result.get_speech_ratio() == 0.85
        assert result.is_high_confidence()

    def test_invalid_time_order(self):
        """测试无效时间顺序"""
        with pytest.raises(ValueError, match="结束时间必须大于开始时间"):
            VADResult(
                is_speech=True,
                probability=0.85,
                start_ms=1500.0,
                end_ms=1000.0,  # 结束时间小于开始时间
                chunk_id=2
            )

    def test_non_speech_result(self):
        """测试非语音结果"""
        result = VADResult(
            is_speech=False,
            probability=0.3,
            start_ms=1000.0,
            end_ms=1500.0,
            chunk_id=2,
            confidence=0.8
        )
        assert not result.is_speech
        assert result.get_speech_ratio() == 0.0


class TestErrorHandling:
    """错误处理测试"""

    def test_cascade_error_base(self):
        """测试Cascade错误基类"""
        error = CascadeError(
            "测试错误",
            ErrorCode.INVALID_INPUT,
            ErrorSeverity.HIGH,
            {"test": "context"}
        )
        assert error.message == "测试错误"
        assert error.error_code == ErrorCode.INVALID_INPUT
        assert error.severity == ErrorSeverity.HIGH
        assert error.context["test"] == "context"
        assert isinstance(error.timestamp, datetime)

        # 测试别名
        assert CascadeError == CascadeError

    def test_audio_format_error(self):
        """测试音频格式错误"""
        error = AudioFormatError("不支持的格式", {"format": "mp3"})
        assert error.error_code == ErrorCode.UNSUPPORTED_FORMAT
        assert error.severity == ErrorSeverity.HIGH
        assert error.context["format_info"]["format"] == "mp3"

    def test_buffer_full_error(self):
        """测试缓冲区满错误"""
        error = BufferFullError(capacity=1000, attempted_size=1500)
        assert error.error_code == ErrorCode.BUFFER_FULL
        assert error.context["capacity"] == 1000
        assert error.context["attempted_size"] == 1500

    def test_model_load_error(self):
        """测试模型加载错误"""
        error = ModelLoadError("/path/to/model.onnx", "文件不存在")
        assert error.error_code == ErrorCode.MODEL_LOAD_FAILED
        assert error.severity == ErrorSeverity.CRITICAL
        assert error.context["model_path"] == "/path/to/model.onnx"

    def test_error_info_from_exception(self):
        """测试从异常创建错误信息"""
        # 从CascadeError创建
        cascade_error = CascadeError("测试错误", ErrorCode.INVALID_INPUT)
        error_info = ErrorInfo.from_exception(cascade_error)
        assert error_info.error_code == ErrorCode.INVALID_INPUT
        assert error_info.message == "测试错误"

        # 从普通异常创建
        general_error = ValueError("普通错误")
        error_info = ErrorInfo.from_exception(general_error)
        assert error_info.error_code == ErrorCode.UNKNOWN_ERROR
        assert error_info.message == "普通错误"
        assert error_info.context["exception_type"] == "ValueError"


class TestBackendConfigs:
    """后端配置测试"""

    def test_onnx_config(self):
        """测试ONNX配置"""
        config = ONNXConfig(
            model_path="/path/to/model.onnx",
            providers=["CPUExecutionProvider"],
            intra_op_num_threads=2
        )
        assert config.model_path == "/path/to/model.onnx"
        assert "CPUExecutionProvider" in config.providers
        assert config.intra_op_num_threads == 2

    def test_invalid_onnx_provider(self):
        """测试无效ONNX提供者"""
        with pytest.raises(ValueError, match="无效的执行提供者"):
            ONNXConfig(providers=["InvalidProvider"])

    def test_vllm_config(self):
        """测试VLLM配置"""
        config = VLLMConfig(
            tensor_parallel_size=2,
            max_model_len=4096,
            gpu_memory_utilization=0.8
        )
        assert config.tensor_parallel_size == 2
        assert config.max_model_len == 4096
        assert config.gpu_memory_utilization == 0.8

    def test_invalid_vllm_dtype(self):
        """测试无效VLLM数据类型"""
        with pytest.raises(ValueError, match="无效的数据类型"):
            VLLMConfig(dtype="invalid_dtype")


class TestEnums:
    """枚举类型测试"""

    def test_audio_format_enum(self):
        """测试音频格式枚举"""
        assert AudioFormat.WAV == "wav"
        assert AudioFormat.PCMA == "pcma"
        assert AudioFormat.get_supported_formats() == ["wav", "pcma"]

    def test_vad_backend_enum(self):
        """测试VAD后端枚举"""
        assert VADBackend.ONNX == "onnx"
        assert VADBackend.VLLM == "vllm"
        assert VADBackend.get_default_backend() == "onnx"

    def test_error_code_enum(self):
        """测试错误码枚举"""
        assert ErrorCode.UNKNOWN_ERROR == "E0000"
        assert ErrorCode.UNSUPPORTED_FORMAT == "E1001"
        assert ErrorCode.BUFFER_FULL == "E2001"
        assert ErrorCode.MODEL_LOAD_FAILED == "E3001"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
