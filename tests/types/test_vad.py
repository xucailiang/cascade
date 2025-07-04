"""
VAD类型单元测试

测试VAD相关类型的功能和验证规则。
"""

import pytest
from pydantic import ValidationError
import os

from cascade.types import (
    VADConfig, VADResult, VADSegment, VADBackend,
    ProcessingMode, OptimizationLevel
)


class TestVADBackend:
    """测试VADBackend枚举"""
    
    def test_default_backend(self):
        """测试默认后端"""
        assert VADBackend.get_default_backend() == "onnx"
        assert VADBackend.get_default_backend() == VADBackend.ONNX.value


class TestProcessingMode:
    """测试ProcessingMode枚举"""
    
    def test_enum_values(self):
        """测试枚举值"""
        assert ProcessingMode.STREAMING == "streaming"
        assert ProcessingMode.BATCH == "batch"
        assert ProcessingMode.REALTIME == "realtime"


class TestOptimizationLevel:
    """测试OptimizationLevel枚举"""
    
    def test_enum_values(self):
        """测试枚举值"""
        assert OptimizationLevel.NONE == "none"
        assert OptimizationLevel.BASIC == "basic"
        assert OptimizationLevel.AGGRESSIVE == "aggressive"
        assert OptimizationLevel.ALL == "all"


class TestVADConfig:
    """测试VADConfig类"""
    
    def test_default_values(self):
        """测试默认值"""
        config = VADConfig()
        assert config.backend == "onnx"
        assert config.workers == 4
        assert config.threshold == 0.5
        assert config.chunk_duration_ms == 500
        assert config.overlap_ms == 16
        assert config.buffer_capacity_seconds == 5
        assert config.processing_mode == "streaming"
        assert config.optimization_level == "all"
        assert config.min_speech_duration_ms == 100
        assert config.max_silence_duration_ms == 500
        assert config.energy_threshold is None
        assert config.smoothing_window_ms == 50
    
    def test_custom_values(self):
        """测试自定义值"""
        config = VADConfig(
            backend=VADBackend.VLLM,
            workers=8,
            threshold=0.7,
            chunk_duration_ms=1000,
            overlap_ms=32,
            buffer_capacity_seconds=10,
            processing_mode=ProcessingMode.BATCH,
            optimization_level=OptimizationLevel.BASIC,
            min_speech_duration_ms=200,
            max_silence_duration_ms=1000,
            energy_threshold=0.1,
            smoothing_window_ms=100
        )
        
        assert config.backend == "vllm"
        assert config.workers == 8
        assert config.threshold == 0.7
        assert config.chunk_duration_ms == 1000
        assert config.overlap_ms == 32
        assert config.buffer_capacity_seconds == 10
        assert config.processing_mode == "batch"
        assert config.optimization_level == "basic"
        assert config.min_speech_duration_ms == 200
        assert config.max_silence_duration_ms == 1000
        assert config.energy_threshold == 0.1
        assert config.smoothing_window_ms == 100
    
    def test_overlap_validation(self):
        """测试重叠时长验证"""
        # 有效重叠时长
        VADConfig(chunk_duration_ms=1000, overlap_ms=100)
        
        # 重叠时长不能超过块时长的50%
        with pytest.raises(ValidationError):
            VADConfig(chunk_duration_ms=1000, overlap_ms=500)
        
        with pytest.raises(ValidationError):
            VADConfig(chunk_duration_ms=1000, overlap_ms=600)
    
    def test_workers_validation(self):
        """测试工作线程数验证"""
        # 有效工作线程数
        VADConfig(workers=1)
        VADConfig(workers=8)
        
        # 工作线程数不能超过系统限制
        max_workers = min(32, (os.cpu_count() or 4) * 2)
        
        if max_workers < 32:  # 只有在系统限制小于32时测试
            with pytest.raises(ValidationError):
                VADConfig(workers=max_workers + 1)
    
    def test_timing_consistency(self):
        """测试时间参数一致性"""
        # 最小语音段时长不能超过块时长
        with pytest.raises(ValidationError):
            VADConfig(chunk_duration_ms=500, min_speech_duration_ms=600)
        
        # 最大静音段时长不能超过块时长的2倍
        with pytest.raises(ValidationError):
            VADConfig(chunk_duration_ms=500, max_silence_duration_ms=1100)
    
    def test_get_chunk_samples(self):
        """测试获取块样本数"""
        config = VADConfig(chunk_duration_ms=500)
        assert config.get_chunk_samples(16000) == 8000  # 500ms at 16kHz = 8000 samples
        assert config.get_chunk_samples(8000) == 4000   # 500ms at 8kHz = 4000 samples
    
    def test_get_overlap_samples(self):
        """测试获取重叠样本数"""
        config = VADConfig(overlap_ms=16)
        assert config.get_overlap_samples(16000) == 256  # 16ms at 16kHz = 256 samples
        assert config.get_overlap_samples(8000) == 128   # 16ms at 8kHz = 128 samples


class TestVADResult:
    """测试VADResult类"""
    
    def test_basic_properties(self):
        """测试基本属性"""
        result = VADResult(
            is_speech=True,
            probability=0.85,
            start_ms=1000.0,
            end_ms=1500.0,
            chunk_id=2,
            confidence=0.9
        )
        
        assert result.is_speech is True
        assert result.probability == 0.85
        assert result.start_ms == 1000.0
        assert result.end_ms == 1500.0
        assert result.chunk_id == 2
        assert result.confidence == 0.9
        assert result.energy_level is None
        assert result.snr_db is None
        assert result.speech_type is None
        assert result.metadata is None
    
    def test_time_order_validation(self):
        """测试时间顺序验证"""
        # 有效时间顺序
        VADResult(
            is_speech=True,
            probability=0.85,
            start_ms=1000.0,
            end_ms=1500.0,
            chunk_id=2
        )
        
        # 结束时间必须大于开始时间
        with pytest.raises(ValidationError):
            VADResult(
                is_speech=True,
                probability=0.85,
                start_ms=1500.0,
                end_ms=1000.0,  # 小于开始时间
                chunk_id=2
            )
        
        with pytest.raises(ValidationError):
            VADResult(
                is_speech=True,
                probability=0.85,
                start_ms=1000.0,
                end_ms=1000.0,  # 等于开始时间
                chunk_id=2
            )
    
    def test_get_duration_ms(self):
        """测试获取时长"""
        result = VADResult(
            is_speech=True,
            probability=0.85,
            start_ms=1000.0,
            end_ms=1500.0,
            chunk_id=2
        )
        
        assert result.get_duration_ms() == 500.0  # 1500 - 1000 = 500ms
    
    def test_get_speech_ratio(self):
        """测试获取语音比例"""
        # 语音
        result1 = VADResult(
            is_speech=True,
            probability=0.85,
            start_ms=1000.0,
            end_ms=1500.0,
            chunk_id=2
        )
        assert result1.get_speech_ratio() == 0.85
        
        # 非语音
        result2 = VADResult(
            is_speech=False,
            probability=0.85,
            start_ms=1000.0,
            end_ms=1500.0,
            chunk_id=2
        )
        assert result2.get_speech_ratio() == 0.0
    
    def test_is_high_confidence(self):
        """测试是否为高置信度检测"""
        # 高置信度
        result1 = VADResult(
            is_speech=True,
            probability=0.85,
            start_ms=1000.0,
            end_ms=1500.0,
            chunk_id=2,
            confidence=0.9
        )
        assert result1.is_high_confidence() is True
        assert result1.is_high_confidence(threshold=0.85) is True
        
        # 低置信度
        result2 = VADResult(
            is_speech=True,
            probability=0.85,
            start_ms=1000.0,
            end_ms=1500.0,
            chunk_id=2,
            confidence=0.7
        )
        assert result2.is_high_confidence() is False
        assert result2.is_high_confidence(threshold=0.6) is True


class TestVADSegment:
    """测试VADSegment类"""
    
    def test_basic_properties(self):
        """测试基本属性"""
        segment = VADSegment(
            start_ms=1000.0,
            end_ms=2500.0,
            confidence=0.85,
            peak_probability=0.95,
            chunk_count=3,
            energy_stats={"mean": 0.5, "max": 0.8}
        )
        
        assert segment.start_ms == 1000.0
        assert segment.end_ms == 2500.0
        assert segment.confidence == 0.85
        assert segment.peak_probability == 0.95
        assert segment.chunk_count == 3
        assert segment.energy_stats == {"mean": 0.5, "max": 0.8}
    
    def test_duration_validation(self):
        """测试时长验证"""
        # 有效时长
        VADSegment(
            start_ms=1000.0,
            end_ms=2500.0,
            confidence=0.85,
            peak_probability=0.95,
            chunk_count=3
        )
        
        # 结束时间必须大于开始时间
        with pytest.raises(ValidationError):
            VADSegment(
                start_ms=2500.0,
                end_ms=1000.0,  # 小于开始时间
                confidence=0.85,
                peak_probability=0.95,
                chunk_count=3
            )
        
        with pytest.raises(ValidationError):
            VADSegment(
                start_ms=1000.0,
                end_ms=1000.0,  # 等于开始时间
                confidence=0.85,
                peak_probability=0.95,
                chunk_count=3
            )
    
    def test_get_duration_ms(self):
        """测试获取段时长"""
        segment = VADSegment(
            start_ms=1000.0,
            end_ms=2500.0,
            confidence=0.85,
            peak_probability=0.95,
            chunk_count=3
        )
        
        assert segment.get_duration_ms() == 1500.0  # 2500 - 1000 = 1500ms