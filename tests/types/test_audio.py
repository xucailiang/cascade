"""
音频类型单元测试

测试音频相关类型的功能和验证规则。
"""

import numpy as np
import pytest
from pydantic import ValidationError

from cascade.types import AudioChunk, AudioConfig, AudioFormat, AudioMetadata


class TestAudioFormat:
    """测试AudioFormat枚举"""

    def test_supported_formats(self):
        """测试支持的格式列表"""
        formats = AudioFormat.get_supported_formats()
        assert "wav" in formats
        assert "pcma" in formats
        assert len(formats) == 2


class TestAudioConfig:
    """测试AudioConfig类"""

    def test_default_values(self):
        """测试默认值"""
        config = AudioConfig()
        assert config.sample_rate == 16000
        assert config.format == "wav"
        assert config.channels == 1
        assert config.dtype == "float32"
        assert config.bit_depth is None

    def test_custom_values(self):
        """测试自定义值"""
        config = AudioConfig(
            sample_rate=44100,
            format=AudioFormat.WAV,
            channels=1,
            dtype="float32",
            bit_depth=24
        )
        assert config.sample_rate == 44100
        assert config.format == "wav"
        assert config.channels == 1
        assert config.dtype == "float32"
        assert config.bit_depth == 24

    def test_sample_rate_validation(self):
        """测试采样率验证"""
        # 有效采样率
        for rate in [8000, 16000, 22050, 44100, 48000]:
            config = AudioConfig(sample_rate=rate)
            assert config.sample_rate == rate

        # 无效采样率
        with pytest.raises(ValidationError):
            AudioConfig(sample_rate=10000)

    def test_channels_validation(self):
        """测试声道数验证"""
        # 当前版本仅支持单声道
        with pytest.raises(ValidationError):
            AudioConfig(channels=2)

    def test_dtype_validation(self):
        """测试数据类型验证"""
        # 有效数据类型
        for dtype in ["float32", "float64", "int16", "int32"]:
            config = AudioConfig(dtype=dtype)
            assert config.dtype == dtype

        # 无效数据类型
        with pytest.raises(ValidationError):
            AudioConfig(dtype="uint8")

    def test_format_compatibility(self):
        """测试格式兼容性验证"""
        # PCMA格式仅支持8kHz和16kHz采样率
        AudioConfig(format=AudioFormat.PCMA, sample_rate=8000)
        AudioConfig(format=AudioFormat.PCMA, sample_rate=16000)

        with pytest.raises(ValidationError):
            AudioConfig(format=AudioFormat.PCMA, sample_rate=44100)

    def test_get_frame_size(self):
        """测试获取帧大小"""
        config = AudioConfig(sample_rate=16000)
        assert config.get_frame_size(500) == 8000  # 500ms at 16kHz = 8000 samples
        assert config.get_frame_size(1000) == 16000  # 1s at 16kHz = 16000 samples

    def test_get_bytes_per_second(self):
        """测试获取每秒字节数"""
        # float32, 单声道, 16kHz = 16000 * 1 * 4 = 64000 bytes/s
        config = AudioConfig(sample_rate=16000, dtype="float32", channels=1)
        assert config.get_bytes_per_second() == 64000

        # int16, 单声道, 8kHz = 8000 * 1 * 2 = 16000 bytes/s
        config = AudioConfig(sample_rate=8000, dtype="int16", channels=1)
        assert config.get_bytes_per_second() == 16000


class TestAudioChunk:
    """测试AudioChunk类"""

    def test_basic_properties(self):
        """测试基本属性"""
        # 创建一个简单的音频块
        data = np.zeros(1000, dtype=np.float32)
        chunk = AudioChunk(
            data=data,
            sequence_number=1,
            start_frame=0,
            chunk_size=1000,
            overlap_size=160,
            timestamp_ms=0.0,
            sample_rate=16000,
            is_last=False
        )

        assert chunk.sequence_number == 1
        assert chunk.start_frame == 0
        assert chunk.chunk_size == 1000
        assert chunk.overlap_size == 160
        assert chunk.timestamp_ms == 0.0
        assert chunk.sample_rate == 16000
        assert chunk.is_last is False
        assert chunk.metadata is None

    def test_overlap_validation(self):
        """测试重叠大小验证"""
        data = np.zeros(1000, dtype=np.float32)

        # 有效重叠大小
        AudioChunk(
            data=data,
            sequence_number=1,
            start_frame=0,
            chunk_size=1000,
            overlap_size=160,
            timestamp_ms=0.0,
            sample_rate=16000
        )

        # 重叠大小不能大于等于块大小
        with pytest.raises(ValidationError):
            AudioChunk(
                data=data,
                sequence_number=1,
                start_frame=0,
                chunk_size=1000,
                overlap_size=1000,  # 等于块大小
                timestamp_ms=0.0,
                sample_rate=16000
            )

        with pytest.raises(ValidationError):
            AudioChunk(
                data=data,
                sequence_number=1,
                start_frame=0,
                chunk_size=1000,
                overlap_size=1200,  # 大于块大小
                timestamp_ms=0.0,
                sample_rate=16000
            )

    def test_get_total_size(self):
        """测试获取总大小"""
        data = np.zeros(1160, dtype=np.float32)
        chunk = AudioChunk(
            data=data,
            sequence_number=1,
            start_frame=0,
            chunk_size=1000,
            overlap_size=160,
            timestamp_ms=0.0,
            sample_rate=16000
        )

        assert chunk.get_total_size() == 1160  # 1000 + 160

    def test_get_duration_ms(self):
        """测试获取块时长"""
        data = np.zeros(1000, dtype=np.float32)
        chunk = AudioChunk(
            data=data,
            sequence_number=1,
            start_frame=0,
            chunk_size=1000,
            overlap_size=0,
            timestamp_ms=0.0,
            sample_rate=16000
        )

        # 1000 samples at 16kHz = 62.5ms
        assert chunk.get_duration_ms() == 62.5

    def test_get_end_timestamp_ms(self):
        """测试获取结束时间戳"""
        data = np.zeros(1000, dtype=np.float32)
        chunk = AudioChunk(
            data=data,
            sequence_number=1,
            start_frame=0,
            chunk_size=1000,
            overlap_size=0,
            timestamp_ms=100.0,
            sample_rate=16000
        )

        # 开始时间 + 持续时间 = 100 + 62.5 = 162.5ms
        assert chunk.get_end_timestamp_ms() == 162.5


class TestAudioMetadata:
    """测试AudioMetadata类"""

    def test_basic_properties(self):
        """测试基本属性"""
        metadata = AudioMetadata(
            title="测试音频",
            duration_seconds=10.5,
            file_size_bytes=168000,
            encoding="PCM",
            bitrate=128000,
            source="测试源",
            quality_score=0.95
        )

        assert metadata.title == "测试音频"
        assert metadata.duration_seconds == 10.5
        assert metadata.file_size_bytes == 168000
        assert metadata.encoding == "PCM"
        assert metadata.bitrate == 128000
        assert metadata.source == "测试源"
        assert metadata.quality_score == 0.95

    def test_optional_fields(self):
        """测试可选字段"""
        metadata = AudioMetadata()

        assert metadata.title is None
        assert metadata.duration_seconds is None
        assert metadata.file_size_bytes is None
        assert metadata.encoding is None
        assert metadata.bitrate is None
        assert metadata.created_at is None
        assert metadata.source is None
        assert metadata.quality_score is None

    def test_validation(self):
        """测试验证规则"""
        # 有效值
        AudioMetadata(duration_seconds=10.5, quality_score=0.5)

        # 无效值 - 负时长
        with pytest.raises(ValidationError):
            AudioMetadata(duration_seconds=-1)

        # 无效值 - 质量评分范围
        with pytest.raises(ValidationError):
            AudioMetadata(quality_score=1.5)  # 超过1.0
