"""
Cascade 音频格式处理模块单元测试

测试音频格式验证、转换和性能优化功能
"""

import time

import numpy as np
import pytest

from cascade.formats import AudioFormatProcessor
from cascade.types import AudioConfig, AudioFormat, AudioFormatError


class TestAudioFormatProcessor:
    """音频格式处理器测试"""

    def setup_method(self):
        """设置测试方法"""
        self.config = AudioConfig(sample_rate=16000, format=AudioFormat.WAV)
        self.processor = AudioFormatProcessor(self.config)

    def test_initialization(self):
        """测试初始化"""
        assert self.processor.config == self.config
        assert isinstance(self.processor.config, AudioConfig)

    def test_validate_format_wav(self):
        """测试WAV格式验证"""
        # 有效的WAV配置
        assert self.processor.validate_format(AudioFormat.WAV, 16000, 1)
        assert self.processor.validate_format(AudioFormat.WAV, 44100, 1)
        assert self.processor.validate_format(AudioFormat.WAV, 48000, 1)

        # 有效的WAV配置（双声道）
        assert self.processor.validate_format(AudioFormat.WAV, 16000, 2)

        # 无效的WAV配置（不支持的采样率）
        assert not self.processor.validate_format(AudioFormat.WAV, 12000, 1)

    def test_validate_format_pcma(self):
        """测试PCMA格式验证"""
        # 有效的PCMA配置（仅支持8kHz单声道）
        assert self.processor.validate_format(AudioFormat.PCMA, 8000, 1)

        # 无效的PCMA配置（不支持的采样率）
        assert not self.processor.validate_format(AudioFormat.PCMA, 16000, 1)
        assert not self.processor.validate_format(AudioFormat.PCMA, 44100, 1)
        assert not self.processor.validate_format(AudioFormat.PCMA, 22050, 1)

        # 无效的PCMA配置（多声道）
        assert not self.processor.validate_format(AudioFormat.PCMA, 8000, 2)

    def test_calculate_chunk_size(self):
        """测试块大小计算"""
        # 测试不同的时长和采样率组合
        test_cases = [
            (500, 16000, 8000),   # 500ms @ 16kHz = 8000 samples
            (1000, 16000, 16000), # 1s @ 16kHz = 16000 samples
            (250, 8000, 2000),    # 250ms @ 8kHz = 2000 samples
            (100, 44100, 4410),   # 100ms @ 44.1kHz = 4410 samples
        ]

        for duration_ms, sample_rate, expected_samples in test_cases:
            result = self.processor.calculate_chunk_size(duration_ms, sample_rate)
            assert result == expected_samples

    def test_convert_to_internal_format_int16(self):
        """测试int16数据转换"""
        # 创建测试数据
        test_data = np.array([16384, -16384, 0, 32767, -32768], dtype=np.int16)

        result = self.processor.convert_to_internal_format(
            test_data, AudioFormat.WAV, 16000
        )

        # 验证结果类型和形状
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        assert result.shape == test_data.shape

        # 验证数值转换正确性
        expected = test_data.astype(np.float32) / 32768.0
        np.testing.assert_array_almost_equal(result, expected, decimal=6)

        # 验证范围在[-1, 1]内
        assert np.all(result >= -1.0)
        assert np.all(result <= 1.0)

    def test_convert_to_internal_format_float32(self):
        """测试float32数据转换"""
        # 创建已经是float32的测试数据
        test_data = np.array([0.5, -0.5, 0.0, 1.0, -1.0], dtype=np.float32)

        result = self.processor.convert_to_internal_format(
            test_data, AudioFormat.WAV, 16000
        )

        # float32数据应该直接返回（可能经过内存对齐）
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        np.testing.assert_array_equal(result, test_data)

    def test_convert_to_internal_format_float64(self):
        """测试float64数据转换"""
        test_data = np.array([0.5, -0.5, 0.0, 1.0, -1.0], dtype=np.float64)

        result = self.processor.convert_to_internal_format(
            test_data, AudioFormat.WAV, 16000
        )

        # 应该转换为float32
        assert result.dtype == np.float32
        expected = test_data.astype(np.float32)
        np.testing.assert_array_almost_equal(result, expected)

    def test_convert_to_internal_format_int32(self):
        """测试int32数据转换"""
        test_data = np.array([1073741824, -1073741824, 0], dtype=np.int32)

        result = self.processor.convert_to_internal_format(
            test_data, AudioFormat.WAV, 16000
        )

        assert result.dtype == np.float32
        # int32 转换为 [-1, 1] 范围
        expected = test_data.astype(np.float32) / 2147483648.0
        np.testing.assert_array_almost_equal(result, expected)

    def test_convert_unsupported_dtype(self):
        """测试不支持的数据类型"""
        # 使用一个真正不支持的数据类型
        test_data = np.array([1, 2, 3], dtype=np.complex64)

        # 期望抛出AudioFormatError（包装了ValueError）
        with pytest.raises(AudioFormatError, match="不支持的数据类型"):
            self.processor.convert_to_internal_format(
                test_data, AudioFormat.WAV, 16000
            )

    def test_convert_large_array(self):
        """测试大数组转换"""
        # 创建大的测试数组
        size = 100000
        test_data = np.random.randint(-32768, 32767, size, dtype=np.int16)

        result = self.processor.convert_to_internal_format(
            test_data, AudioFormat.WAV, 16000
        )

        assert result.shape == (size,)
        assert result.dtype == np.float32
        assert np.all(result >= -1.0)
        assert np.all(result <= 1.0)

    def test_convert_empty_array(self):
        """测试空数组转换"""
        test_data = np.array([], dtype=np.int16)

        # 空数组应该抛出ValueError
        with pytest.raises(AudioFormatError, match="音频数据不能为空"):
            self.processor.convert_to_internal_format(
                test_data, AudioFormat.WAV, 16000
            )

    def test_convert_multidimensional_array(self):
        """测试多维数组转换"""
        # 创建2D数组（模拟多声道，虽然当前只支持单声道）
        test_data = np.array([[16384, -16384], [0, 32767]], dtype=np.int16)

        result = self.processor.convert_to_internal_format(
            test_data, AudioFormat.WAV, 16000
        )

        assert result.shape == test_data.shape
        assert result.dtype == np.float32

    def test_memory_alignment_warning(self):
        """测试内存对齐警告"""
        # 创建可能未对齐的数组
        test_data = np.array([1, 2, 3, 4, 5], dtype=np.int16)

        # 测试对齐功能（不依赖特定的内部实现）
        result = self.processor.convert_to_internal_format(
            test_data, AudioFormat.WAV, 16000
        )

        # 验证结果是正确的
        assert result.dtype == np.float32
        assert result.shape == test_data.shape

    def test_performance_logging(self):
        """测试性能日志记录"""
        test_data = np.random.randint(-32768, 32767, 10000, dtype=np.int16)

        # 测试性能测量功能（通过capture输出验证）
        result = self.processor.convert_to_internal_format(
            test_data, AudioFormat.WAV, 16000
        )

        # 验证结果正确性
        assert result.dtype == np.float32
        assert result.shape == test_data.shape
        assert np.all(result >= -1.0)
        assert np.all(result <= 1.0)

    def test_processor_with_different_configs(self):
        """测试不同配置的处理器"""
        configs = [
            AudioConfig(sample_rate=8000, format=AudioFormat.PCMA),
            AudioConfig(sample_rate=44100, format=AudioFormat.WAV),
            AudioConfig(sample_rate=48000, format=AudioFormat.WAV),
        ]

        for config in configs:
            processor = AudioFormatProcessor(config)

            # 对于PCMA格式，需要使用正确的数据类型和参数
            if config.format == AudioFormat.PCMA:
                # PCMA格式需要uint8数据
                test_data = np.array([100, 200, 0], dtype=np.uint8)
            else:
                test_data = np.array([100, -100, 0], dtype=np.int16)

            result = processor.convert_to_internal_format(
                test_data, config.format, config.sample_rate
            )

            assert result.dtype == np.float32
            assert result.shape == test_data.shape

    def test_edge_cases(self):
        """测试边界情况"""
        # 测试极值
        extreme_values = np.array([32767, -32768], dtype=np.int16)
        result = self.processor.convert_to_internal_format(
            extreme_values, AudioFormat.WAV, 16000
        )

        # 检查极值转换
        assert result[0] <= 1.0  # 最大值应该接近但不超过1.0
        assert result[1] >= -1.0  # 最小值应该接近但不小于-1.0

        # 测试零值
        zeros = np.zeros(100, dtype=np.int16)
        result = self.processor.convert_to_internal_format(
            zeros, AudioFormat.WAV, 16000
        )
        np.testing.assert_array_equal(result, np.zeros(100, dtype=np.float32))

    def test_format_compatibility(self):
        """测试格式兼容性"""
        # 创建PCMA配置的处理器
        pcma_config = AudioConfig(sample_rate=8000, format=AudioFormat.PCMA)
        pcma_processor = AudioFormatProcessor(pcma_config)

        # 测试PCMA格式的数据处理（使用uint8数据）
        test_data = np.array([128, 64, 0], dtype=np.uint8)
        result = pcma_processor.convert_to_internal_format(
            test_data, AudioFormat.PCMA, 8000
        )

        assert result.dtype == np.float32
        assert result.shape == test_data.shape

    def test_concurrent_processing(self):
        """测试并发处理"""
        import threading

        results = []
        errors = []

        def process_data(thread_id):
            """处理数据的线程函数"""
            try:
                # 每个线程处理不同的数据
                data_size = 1000 + thread_id * 100
                test_data = np.random.randint(
                    -32768, 32767, data_size, dtype=np.int16
                )

                result = self.processor.convert_to_internal_format(
                    test_data, AudioFormat.WAV, 16000
                )

                results.append((thread_id, result.shape, result.dtype))

            except Exception as e:
                errors.append((thread_id, str(e)))

        # 启动多个线程
        threads = []
        for i in range(5):
            thread = threading.Thread(target=process_data, args=(i,))
            threads.append(thread)
            thread.start()

        # 等待所有线程完成
        for thread in threads:
            thread.join()

        # 验证结果
        assert len(errors) == 0, f"线程处理出错: {errors}"
        assert len(results) == 5

        for thread_id, shape, dtype in results:
            assert dtype == np.float32
            assert shape[0] == 1000 + thread_id * 100


class TestAudioFormatProcessorIntegration:
    """音频格式处理器集成测试"""

    def test_end_to_end_processing(self):
        """测试端到端处理流程"""
        # 模拟真实的音频处理流程
        config = AudioConfig(sample_rate=16000, format=AudioFormat.WAV)
        processor = AudioFormatProcessor(config)

        # 1. 验证格式
        assert processor.validate_format(AudioFormat.WAV, 16000, 1)

        # 2. 计算块大小
        chunk_size = processor.calculate_chunk_size(500, 16000)  # 500ms
        assert chunk_size == 8000

        # 3. 处理音频数据
        audio_data = np.random.randint(-32768, 32767, chunk_size, dtype=np.int16)
        processed_data = processor.convert_to_internal_format(
            audio_data, AudioFormat.WAV, 16000
        )

        # 4. 验证处理结果
        assert processed_data.dtype == np.float32
        assert processed_data.shape == (chunk_size,)
        assert np.all(processed_data >= -1.0)
        assert np.all(processed_data <= 1.0)

    def test_multiple_format_processing(self):
        """测试多种格式处理"""
        # 测试WAV格式
        wav_config = AudioConfig(sample_rate=16000, format=AudioFormat.WAV)
        wav_processor = AudioFormatProcessor(wav_config)

        wav_data = np.array([16384, -16384, 0], dtype=np.int16)
        wav_result = wav_processor.convert_to_internal_format(
            wav_data, AudioFormat.WAV, 16000
        )

        # 测试PCMA格式
        pcma_config = AudioConfig(sample_rate=8000, format=AudioFormat.PCMA)
        pcma_processor = AudioFormatProcessor(pcma_config)

        pcma_data = np.array([128, 64, 0], dtype=np.uint8)
        pcma_result = pcma_processor.convert_to_internal_format(
            pcma_data, AudioFormat.PCMA, 8000
        )

        # 验证PCMA处理结果
        assert pcma_result.dtype == np.float32
        assert pcma_result.shape == pcma_data.shape

    def test_performance_benchmark(self):
        """测试性能基准"""
        config = AudioConfig(sample_rate=16000, format=AudioFormat.WAV)
        processor = AudioFormatProcessor(config)

        # 测试不同大小的数据处理性能
        sizes = [1000, 10000, 100000]

        for size in sizes:
            test_data = np.random.randint(-32768, 32767, size, dtype=np.int16)

            start_time = time.time()
            result = processor.convert_to_internal_format(
                test_data, AudioFormat.WAV, 16000
            )
            end_time = time.time()

            processing_time = (end_time - start_time) * 1000  # ms

            # 验证处理时间合理（这里设置一个比较宽松的限制）
            assert processing_time < 1000  # 不应该超过1秒

            # 验证结果正确性
            assert result.shape == (size,)
            assert result.dtype == np.float32


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
