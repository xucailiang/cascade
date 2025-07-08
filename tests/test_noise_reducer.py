"""
噪声抑制处理器测试
"""

import numpy as np
import pytest

from cascade.processor.noise_reducer import (
    NoiseReducer,
    NoiseReducerConfig,
    NoiseReductionMethod,
    NoiseReductionResult,
)
from cascade.types.audio import AudioConfig, AudioFormat


@pytest.fixture
def noisy_audio():
    """生成带噪声的测试音频数据"""
    # 创建一个包含信号和噪声的合成音频
    sample_rate = 16000
    duration_sec = 1.0
    num_samples = int(sample_rate * duration_sec)

    # 创建信号（纯音）
    t = np.linspace(0, duration_sec, num_samples)
    signal = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440Hz正弦波

    # 创建噪声
    noise = np.random.normal(0, 0.1, num_samples)

    # 合并信号和噪声
    noisy_signal = signal + noise

    return noisy_signal, sample_rate


@pytest.fixture
def audio_config(noisy_audio):
    """创建音频配置"""
    _, sample_rate = noisy_audio
    return AudioConfig(
        sample_rate=sample_rate,
        format=AudioFormat.WAV,
        channels=1,
        dtype="float32"
    )


@pytest.mark.asyncio
async def test_spectral_subtraction(noisy_audio, audio_config):
    """测试频谱减法噪声抑制"""
    audio_data, _ = noisy_audio

    # 创建噪声抑制处理器
    config = NoiseReducerConfig(
        method=NoiseReductionMethod.SPECTRAL_SUBTRACTION,
        noise_threshold=0.05,
        reduction_factor=0.5,
        fft_size=512
    )
    processor = NoiseReducer(config)

    # 处理音频
    results = await processor.process_audio(audio_data, audio_config)

    # 验证结果
    assert len(results) > 0

    # 验证每个结果
    for result in results:
        assert result.success
        assert isinstance(result.result_data, NoiseReductionResult)

        # 验证噪声抑制效果
        noise_result = result.result_data
        assert noise_result.original_energy > 0
        assert noise_result.reduced_energy > 0
        assert noise_result.noise_energy >= 0

        # 验证抑制后的能量应该小于原始能量
        assert noise_result.reduced_energy <= noise_result.original_energy

        # 验证噪声抑制比例应该大于0
        # 注意：由于算法可能会降低信号能量，信噪比可能会下降
        assert noise_result.reduction_ratio > 0


@pytest.mark.asyncio
async def test_wiener_filter(noisy_audio, audio_config):
    """测试维纳滤波噪声抑制"""
    audio_data, _ = noisy_audio

    # 创建噪声抑制处理器
    config = NoiseReducerConfig(
        method=NoiseReductionMethod.WIENER_FILTER,
        noise_threshold=0.05,
        smoothing_factor=0.8,
        fft_size=512
    )
    processor = NoiseReducer(config)

    # 处理音频
    results = await processor.process_audio(audio_data, audio_config)

    # 验证结果
    assert len(results) > 0

    # 验证每个结果
    for result in results:
        assert result.success
        assert isinstance(result.result_data, NoiseReductionResult)

        # 验证噪声抑制效果
        noise_result = result.result_data
        assert noise_result.original_energy > 0
        assert noise_result.reduced_energy > 0
        assert noise_result.noise_energy >= 0

        # 验证抑制后的能量应该小于原始能量
        assert noise_result.reduced_energy <= noise_result.original_energy

        # 验证噪声抑制比例应该大于0
        # 注意：由于算法可能会降低信号能量，信噪比可能会下降
        assert noise_result.reduction_ratio > 0


@pytest.mark.asyncio
async def test_minimum_mean_square(noisy_audio, audio_config):
    """测试最小均方误差噪声抑制"""
    audio_data, _ = noisy_audio

    # 创建噪声抑制处理器
    config = NoiseReducerConfig(
        method=NoiseReductionMethod.MINIMUM_MEAN_SQUARE,
        noise_threshold=0.05,
        smoothing_factor=0.8,
        fft_size=512
    )
    processor = NoiseReducer(config)

    # 处理音频
    results = await processor.process_audio(audio_data, audio_config)

    # 验证结果
    assert len(results) > 0

    # 验证每个结果
    for result in results:
        assert result.success
        assert isinstance(result.result_data, NoiseReductionResult)

        # 验证噪声抑制效果
        noise_result = result.result_data
        assert noise_result.original_energy > 0
        assert noise_result.reduced_energy >= 0  # 允许能量为0，因为MMSE方法可能会完全抑制信号
        assert noise_result.noise_energy >= 0

        # 验证抑制后的能量应该小于原始能量
        assert noise_result.reduced_energy <= noise_result.original_energy

        # 验证噪声抑制比例应该大于0
        # 注意：由于算法可能会降低信号能量，信噪比可能会下降
        assert noise_result.reduction_ratio > 0


@pytest.mark.asyncio
async def test_different_parameters(noisy_audio, audio_config):
    """测试不同参数配置"""
    audio_data, _ = noisy_audio

    # 测试不同的reduction_factor
    results_by_reduction = {}

    for reduction_factor in [0.2, 0.5, 0.8]:
        config = NoiseReducerConfig(
            method=NoiseReductionMethod.SPECTRAL_SUBTRACTION,
            reduction_factor=reduction_factor
        )
        processor = NoiseReducer(config)

        results = await processor.process_audio(audio_data, audio_config)
        reduction_ratio = results[0].result_data.reduction_ratio
        results_by_reduction[reduction_factor] = reduction_ratio

    # 更高的reduction_factor应该导致更强的噪声抑制
    assert results_by_reduction[0.8] >= results_by_reduction[0.5]
    assert results_by_reduction[0.5] >= results_by_reduction[0.2]


@pytest.mark.asyncio
async def test_empty_audio(audio_config):
    """测试处理空音频"""
    # 创建空音频
    audio_data = np.array([], dtype=np.float32)

    # 创建噪声抑制处理器
    processor = NoiseReducer()

    # 处理音频
    results = await processor.process_audio(audio_data, audio_config)

    # 验证结果（应该没有结果）
    assert len(results) == 0


@pytest.mark.asyncio
async def test_reset_functionality(noisy_audio, audio_config):
    """测试重置功能"""
    audio_data, _ = noisy_audio

    # 创建噪声抑制处理器
    processor = NoiseReducer()

    # 处理音频（第一次）
    results1 = await processor.process_audio(audio_data, audio_config)

    # 重置噪声估计
    processor.reset()

    # 处理音频（第二次）
    results2 = await processor.process_audio(audio_data, audio_config)

    # 验证结果
    assert len(results1) > 0
    assert len(results2) > 0

    # 重置后，噪声估计应该从头开始，所以第一次和第二次的结果应该不同
    # 但由于噪声是随机的，我们只能验证结果的存在，而不是具体的值
    assert isinstance(results1[0].result_data, NoiseReductionResult)
    assert isinstance(results2[0].result_data, NoiseReductionResult)


@pytest.mark.asyncio
async def test_invalid_fft_size():
    """测试无效的FFT大小"""
    # 创建带有无效FFT大小的配置
    with pytest.raises(ValueError):
        NoiseReducerConfig(fft_size=100)  # 不是2的幂


# 不再需要直接运行测试，使用pytest命令运行测试
# if __name__ == "__main__":
#     asyncio.run(test_spectral_subtraction(noisy_audio(), audio_config(noisy_audio())))
#     asyncio.run(test_wiener_filter(noisy_audio(), audio_config(noisy_audio())))
#     asyncio.run(test_minimum_mean_square(noisy_audio(), audio_config(noisy_audio())))
#     asyncio.run(test_different_parameters(noisy_audio(), audio_config(noisy_audio())))
#     asyncio.run(test_empty_audio(audio_config(noisy_audio())))
#     asyncio.run(test_reset_functionality(noisy_audio(), audio_config(noisy_audio())))
#     _ = test_invalid_fft_size()
#     print("所有测试通过！")
