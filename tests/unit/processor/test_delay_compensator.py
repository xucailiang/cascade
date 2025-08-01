"""
VAD延迟补偿器测试

测试延迟补偿器的核心功能：
- 工厂函数创建
- 延迟补偿逻辑
- 语音开始检测
- 时间戳调整算法
- 边界条件处理
"""


import pytest

from cascade.processor.delay_compensator import (
    SimpleDelayCompensator,
    create_delay_compensator,
)
from cascade.types import VADResult


class TestCreateDelayCompensator:
    """测试延迟补偿器工厂函数"""

    def test_create_disabled_compensator(self):
        """测试创建禁用的补偿器"""
        compensator = create_delay_compensator(0)
        assert compensator is None

        compensator = create_delay_compensator(None)
        assert compensator is None

    def test_create_enabled_compensator(self):
        """测试创建启用的补偿器"""
        compensator = create_delay_compensator(200)
        assert isinstance(compensator, SimpleDelayCompensator)
        assert compensator.compensation_ms == 200

    def test_create_with_negative_value(self):
        """测试使用负值创建补偿器"""
        compensator = create_delay_compensator(-100)
        assert compensator is None


class TestSimpleDelayCompensator:
    """测试简单延迟补偿器"""

    @pytest.fixture
    def compensator(self):
        """创建测试用的补偿器fixture"""
        return SimpleDelayCompensator(compensation_ms=200)

    def test_initialization(self):
        """测试补偿器初始化"""
        compensator = SimpleDelayCompensator(compensation_ms=150)
        assert compensator.compensation_ms == 150
        assert compensator.enabled is True
        assert compensator.previous_is_speech is False

    def test_initialization_disabled(self):
        """测试禁用状态的初始化"""
        compensator = SimpleDelayCompensator(compensation_ms=0)
        assert compensator.compensation_ms == 0
        assert compensator.enabled is False
        assert compensator.previous_is_speech is False

    def test_process_result_no_speech(self, compensator):
        """测试处理非语音结果"""
        # 创建非语音VAD结果
        vad_result = VADResult(
            is_speech=False,
            probability=0.3,
            start_ms=1000.0,
            end_ms=1500.0,
            chunk_id=1,
            confidence=0.4
        )

        processed_result = compensator.process_result(vad_result)

        # 非语音结果应该不被修改
        assert processed_result == vad_result
        assert not processed_result.is_compensated
        assert processed_result.original_start_ms is None
        assert compensator.previous_is_speech is False

    def test_process_result_first_speech_detection(self, compensator):
        """测试首次检测到语音"""
        # 创建语音结果
        vad_result = VADResult(
            is_speech=True,
            probability=0.8,
            start_ms=1000.0,
            end_ms=1500.0,
            chunk_id=1,
            confidence=0.9
        )

        processed_result = compensator.process_result(vad_result)

        # 首次检测到语音应该被补偿
        assert processed_result.is_speech is True
        assert processed_result.probability == 0.8
        assert processed_result.start_ms == 800.0  # 1000 - 200 = 800
        assert processed_result.end_ms == 1500.0  # 不变
        assert processed_result.chunk_id == 1
        assert processed_result.confidence == 0.9
        assert processed_result.is_compensated is True
        assert processed_result.original_start_ms == 1000.0

        # 验证状态更新
        assert compensator.previous_is_speech is True

    def test_process_result_subsequent_speech(self, compensator):
        """测试后续语音块处理"""
        # 先处理第一个语音块
        first_result = VADResult(
            is_speech=True,
            probability=0.8,
            start_ms=1000.0,
            end_ms=1500.0,
            chunk_id=1,
            confidence=0.9
        )
        compensator.process_result(first_result)

        # 处理第二个语音块
        second_result = VADResult(
            is_speech=True,
            probability=0.9,
            start_ms=1500.0,
            end_ms=2000.0,
            chunk_id=2,
            confidence=0.95
        )

        processed_result = compensator.process_result(second_result)

        # 后续语音块不应该被补偿（因为前一个已经是语音）
        assert processed_result == second_result
        assert not processed_result.is_compensated
        assert processed_result.original_start_ms is None

        # 状态应该保持为语音
        assert compensator.previous_is_speech is True

    def test_process_result_speech_after_non_speech(self, compensator):
        """测试非语音后的语音处理"""
        # 处理非语音块
        non_speech_result = VADResult(
            is_speech=False,
            probability=0.2,
            start_ms=1000.0,
            end_ms=1500.0,
            chunk_id=1,
            confidence=0.3
        )
        compensator.process_result(non_speech_result)

        # 处理语音块
        speech_result = VADResult(
            is_speech=True,
            probability=0.85,
            start_ms=1500.0,
            end_ms=2000.0,
            chunk_id=2,
            confidence=0.92
        )

        processed_result = compensator.process_result(speech_result)

        # 非语音后的语音应该被补偿
        assert processed_result.is_compensated is True
        assert processed_result.start_ms == 1300.0  # 1500 - 200
        assert processed_result.original_start_ms == 1500.0

    def test_process_result_edge_cases(self, compensator):
        """测试边界情况"""
        # 测试开始时间为0的情况
        vad_result = VADResult(
            is_speech=True,
            probability=0.8,
            start_ms=0.0,
            end_ms=500.0,
            chunk_id=1,
            confidence=0.9
        )

        processed_result = compensator.process_result(vad_result)

        # 补偿后的开始时间不应该为负数
        assert processed_result.start_ms == 0.0  # max(0 - 200, 0) = 0
        assert processed_result.is_compensated is True
        assert processed_result.original_start_ms == 0.0

    def test_process_result_small_start_time(self, compensator):
        """测试小的开始时间"""
        vad_result = VADResult(
            is_speech=True,
            probability=0.8,
            start_ms=100.0,  # 小于补偿时间200ms
            end_ms=600.0,
            chunk_id=1,
            confidence=0.9
        )

        processed_result = compensator.process_result(vad_result)

        # 补偿后的开始时间不应该为负数
        assert processed_result.start_ms == 0.0  # max(100 - 200, 0) = 0
        assert processed_result.is_compensated is True
        assert processed_result.original_start_ms == 100.0

    def test_compensation_disabled_with_zero_ms(self):
        """测试0ms补偿（实际上是禁用）"""
        compensator = SimpleDelayCompensator(compensation_ms=0)

        vad_result = VADResult(
            is_speech=True,
            probability=0.8,
            start_ms=1000.0,
            end_ms=1500.0,
            chunk_id=1,
            confidence=0.9
        )

        processed_result = compensator.process_result(vad_result)

        # 0ms补偿应该不改变结果
        assert processed_result == vad_result
        assert not processed_result.is_compensated
        assert processed_result.original_start_ms is None

    def test_different_compensation_values(self):
        """测试不同的补偿值"""
        test_cases = [
            (50, 1000.0, 950.0),
            (100, 1000.0, 900.0),
            (300, 1000.0, 700.0),
            (500, 1000.0, 500.0),
        ]

        for compensation_ms, start_ms, expected_start in test_cases:
            compensator = SimpleDelayCompensator(compensation_ms=compensation_ms)

            vad_result = VADResult(
                is_speech=True,
                probability=0.8,
                start_ms=start_ms,
                end_ms=start_ms + 500,
                chunk_id=1,
                confidence=0.9
            )

            processed_result = compensator.process_result(vad_result)

            assert processed_result.start_ms == expected_start
            assert processed_result.is_compensated is True
            assert processed_result.original_start_ms == start_ms

    def test_reset_functionality(self, compensator):
        """测试重置功能"""
        # 处理一个语音结果
        vad_result = VADResult(
            is_speech=True,
            probability=0.8,
            start_ms=1000.0,
            end_ms=1500.0,
            chunk_id=1,
            confidence=0.9
        )
        compensator.process_result(vad_result)
        assert compensator.previous_is_speech is True

        # 重置补偿器
        compensator.reset()
        assert compensator.previous_is_speech is False

        # 重置后第一个语音应该再次被补偿
        processed_result = compensator.process_result(vad_result)
        assert processed_result.is_compensated is True

    def test_utility_methods(self, compensator):
        """测试实用方法"""
        # 测试启用状态检查
        assert compensator.is_enabled() is True

        # 测试获取补偿时间
        assert compensator.get_compensation_ms() == 200

        # 测试动态设置补偿时间
        compensator.set_compensation_ms(300)
        assert compensator.get_compensation_ms() == 300
        assert compensator.enabled is True

        # 测试设置为0（禁用）
        compensator.set_compensation_ms(0)
        assert compensator.get_compensation_ms() == 0
        assert compensator.enabled is False
        assert compensator.is_enabled() is False


class TestDelayCompensatorIntegration:
    """测试延迟补偿器集成场景"""

    def test_realistic_speech_sequence(self):
        """测试真实的语音序列处理"""
        compensator = SimpleDelayCompensator(compensation_ms=150)

        # 模拟一个真实的语音检测序列
        speech_sequence = [
            # 非语音开始
            (False, 0.2, 0.0, 500.0),
            (False, 0.1, 500.0, 1000.0),
            # 语音开始 - 应该被补偿
            (True, 0.8, 1000.0, 1500.0),
            (True, 0.9, 1500.0, 2000.0),
            (True, 0.85, 2000.0, 2500.0),
            # 非语音
            (False, 0.3, 2500.0, 3000.0),
            # 语音重新开始 - 应该被补偿
            (True, 0.88, 3000.0, 3500.0),
            (True, 0.92, 3500.0, 4000.0),
        ]

        results = []
        for i, (is_speech, prob, start, end) in enumerate(speech_sequence):
            result = VADResult(
                is_speech=is_speech,
                probability=prob,
                start_ms=start,
                end_ms=end,
                chunk_id=i,
                confidence=prob
            )

            processed = compensator.process_result(result)
            results.append(processed)

        # 验证补偿行为
        # 第一个语音块（索引2）应该被补偿
        assert results[2].is_compensated is True
        assert results[2].start_ms == 850.0  # 1000 - 150
        assert results[2].original_start_ms == 1000.0

        # 后续语音块不应该被补偿
        assert not results[3].is_compensated
        assert not results[4].is_compensated

        # 非语音后的新语音应该被补偿
        assert results[6].is_compensated is True
        assert results[6].start_ms == 2850.0  # 3000 - 150
        assert results[6].original_start_ms == 3000.0

        # 后续语音不补偿
        assert not results[7].is_compensated


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
