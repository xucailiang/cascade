"""
VAD 处理器测试
"""

import asyncio

import numpy as np
import pytest

from cascade.processor.vad_processor import (
    VADProcessor,
    VADProcessorConfig,
    VADSensitivity,
)
from cascade.types.audio import AudioConfig, AudioFormat


@pytest.fixture
def sample_audio():
    """生成测试用的音频数据"""
    # 创建一个包含语音和静音的合成音频
    # 1秒16kHz的音频 = 16000个样本
    sample_rate = 16000
    duration_sec = 2.0
    num_samples = int(sample_rate * duration_sec)

    # 创建静音（低能量随机噪声）
    silence = np.random.normal(0, 0.001, num_samples // 4)

    # 创建语音（高能量正弦波）
    t = np.linspace(0, 0.5, num_samples // 2)
    speech = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440Hz正弦波

    # 组合：静音-语音-静音
    audio = np.concatenate([silence, speech, silence])

    return audio, sample_rate


@pytest.mark.asyncio
async def test_vad_processor_basic(sample_audio):
    """测试 VAD 处理器的基本功能"""
    audio_data, sample_rate = sample_audio

    # 创建 VAD 处理器
    config = VADProcessorConfig(
        sensitivity=VADSensitivity.MEDIUM,
        chunk_duration_ms=100,
        overlap_ms=10
    )
    processor = VADProcessor(config)

    # 创建音频配置
    audio_config = AudioConfig(
        sample_rate=sample_rate,
        format=AudioFormat.WAV,
        channels=1,
        dtype="float32"
    )

    # 处理音频
    results = await processor.process_audio(audio_data, audio_config)

    # 验证结果
    assert len(results) > 0

    # 应该至少有一个语音段
    speech_segments = [r for r in results if r.result_data.is_speech]
    assert len(speech_segments) > 0

    # 验证语音段的位置（应该在中间部分）
    for segment in speech_segments:
        # 语音段的时间戳应该在合理范围内
        assert 0 <= segment.result_data.start_time_ms < 2000
        assert 0 < segment.result_data.end_time_ms <= 2000

        # 语音段的能量应该高于阈值
        assert segment.result_data.energy > segment.result_data.threshold


@pytest.mark.asyncio
async def test_vad_processor_sensitivity(sample_audio):
    """测试不同灵敏度下的 VAD 处理器"""
    audio_data, sample_rate = sample_audio

    # 创建音频配置
    audio_config = AudioConfig(
        sample_rate=sample_rate,
        format=AudioFormat.WAV,
        channels=1,
        dtype="float32"
    )

    # 测试不同灵敏度
    results_by_sensitivity = {}

    for sensitivity in [VADSensitivity.LOW, VADSensitivity.MEDIUM, VADSensitivity.HIGH]:
        config = VADProcessorConfig(
            sensitivity=sensitivity,
            chunk_duration_ms=100,
            overlap_ms=10
        )
        processor = VADProcessor(config)

        results = await processor.process_audio(audio_data, audio_config)
        speech_segments = [r for r in results if r.result_data.is_speech]
        results_by_sensitivity[sensitivity] = len(speech_segments)

    # 高灵敏度应该检测到更多语音段
    assert results_by_sensitivity[VADSensitivity.HIGH] >= results_by_sensitivity[VADSensitivity.MEDIUM]
    assert results_by_sensitivity[VADSensitivity.MEDIUM] >= results_by_sensitivity[VADSensitivity.LOW]


@pytest.mark.asyncio
async def test_vad_processor_empty_audio():
    """测试处理空音频"""
    # 创建空音频
    audio_data = np.array([], dtype=np.float32)

    # 创建 VAD 处理器
    processor = VADProcessor()

    # 创建音频配置
    audio_config = AudioConfig(
        sample_rate=16000,
        format=AudioFormat.WAV,
        channels=1,
        dtype="float32"
    )

    # 处理音频
    results = await processor.process_audio(audio_data, audio_config)

    # 验证结果
    assert len(results) == 0


if __name__ == "__main__":
    asyncio.run(test_vad_processor_basic(sample_audio()))
    asyncio.run(test_vad_processor_sensitivity(sample_audio()))
    asyncio.run(test_vad_processor_empty_audio())
    print("所有测试通过！")
