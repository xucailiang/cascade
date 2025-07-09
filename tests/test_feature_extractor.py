"""
特征提取处理器测试
"""

import asyncio

import numpy as np
import pytest

from cascade.processor.base import AudioChunk
from cascade.processor.feature_extractor import (
    FeatureExtractor,
    FeatureExtractorConfig,
    FeatureType,
    FeatureResult,
)
from cascade.types.audio import AudioConfig, AudioFormat


@pytest.fixture
def sample_audio():
    """生成测试用的音频数据"""
    # 创建一个包含语音和静音的合成音频
    # 1秒16kHz的音频 = 16000个样本
    sample_rate = 16000
    duration_sec = 1.0
    num_samples = int(sample_rate * duration_sec)

    # 创建一个简单的正弦波
    t = np.linspace(0, duration_sec, num_samples)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440Hz正弦波

    return audio, sample_rate


@pytest.mark.asyncio
async def test_feature_extractor_basic(sample_audio):
    """测试特征提取处理器的基本功能"""
    audio_data, sample_rate = sample_audio

    # 创建特征提取处理器
    config = FeatureExtractorConfig(
        feature_types=[FeatureType.MFCC, FeatureType.ENERGY, FeatureType.ZERO_CROSSING_RATE],
        n_mfcc=13,
        n_fft=512,
        hop_length=256
    )
    processor = FeatureExtractor(config)

    # 创建音频配置
    audio_config = AudioConfig(
        sample_rate=sample_rate,
        format=AudioFormat.WAV,
        channels=1,
        dtype="float32"
    )

    # 创建音频块
    chunk = AudioChunk(
        data=audio_data,
        sequence_number=0,
        start_frame=0,
        chunk_size=len(audio_data),
        timestamp_ms=0.0,
        sample_rate=sample_rate,
        is_last=True
    )

    # 处理音频块
    result = processor.process_chunk(chunk)

    # 验证结果
    assert result is not None
    assert isinstance(result.features, dict)
    assert isinstance(result.feature_dims, dict)
    assert result.frame_count > 0
    assert result.feature_count > 0

    # 验证MFCC特征
    if "mfcc" in result.features:
        assert result.features["mfcc"].shape[1] == config.n_mfcc

    # 验证能量特征
    if "energy" in result.features:
        assert result.features["energy"].shape[1] == 1

    # 验证过零率特征
    if "zcr" in result.features:
        assert result.features["zcr"].shape[1] == 1


@pytest.mark.asyncio
async def test_feature_extractor_empty_audio():
    """测试处理空音频"""
    # 创建空音频
    audio_data = np.array([], dtype=np.float32)

    # 创建特征提取处理器
    processor = FeatureExtractor()

    # 创建音频块
    chunk = AudioChunk(
        data=audio_data,
        sequence_number=0,
        start_frame=0,
        chunk_size=1,  # 必须大于0
        timestamp_ms=0.0,
        sample_rate=16000,
        is_last=True
    )

    # 处理音频块应该抛出异常
    with pytest.raises(ValueError):
        processor.process_chunk(chunk)


@pytest.mark.asyncio
async def test_feature_extractor_all_features(sample_audio):
    """测试提取所有特征"""
    audio_data, sample_rate = sample_audio

    # 创建特征提取处理器
    config = FeatureExtractorConfig(
        feature_types=[FeatureType.ALL],
        n_mfcc=13,
        n_fft=512,
        hop_length=256
    )
    processor = FeatureExtractor(config)

    # 创建音频配置
    audio_config = AudioConfig(
        sample_rate=sample_rate,
        format=AudioFormat.WAV,
        channels=1,
        dtype="float32"
    )

    # 创建音频块
    chunk = AudioChunk(
        data=audio_data,
        sequence_number=0,
        start_frame=0,
        chunk_size=len(audio_data),
        timestamp_ms=0.0,
        sample_rate=sample_rate,
        is_last=True
    )

    # 处理音频块
    result = processor.process_chunk(chunk)

    # 验证结果
    assert result is not None
    assert isinstance(result.features, dict)

    # 检查是否包含基本特征
    basic_features = ["energy", "log_energy"]
    for feature in basic_features:
        assert feature in result.features or not processor.librosa, f"缺少基本特征: {feature}"


@pytest.mark.asyncio
async def test_feature_extractor_process_audio(sample_audio):
    """测试通过process_audio方法处理音频"""
    audio_data, sample_rate = sample_audio

    # 创建特征提取处理器
    config = FeatureExtractorConfig(
        feature_types=[FeatureType.MFCC],
        n_mfcc=13,
        n_fft=512,
        hop_length=256
    )
    processor = FeatureExtractor(config)

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
    assert results[0].success

    # 获取结果数据
    result_data = results[0].result_data
    
    # 验证结果数据类型
    assert isinstance(result_data, FeatureResult)

    # 检查是否包含MFCC特征
    if processor.librosa:
        assert "mfcc" in result_data.features or not processor.librosa


if __name__ == "__main__":
    asyncio.run(test_feature_extractor_basic(sample_audio()))
    asyncio.run(test_feature_extractor_empty_audio())
    asyncio.run(test_feature_extractor_all_features(sample_audio()))
    asyncio.run(test_feature_extractor_process_audio(sample_audio()))
    print("所有测试通过！")
