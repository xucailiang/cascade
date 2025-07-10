"""
Cascade 顶层API

本模块提供了Cascade库的顶层API，为用户提供简单易用的接口，隐藏底层实现细节。
这些API函数自动管理资源生命周期，提供合理的默认配置，支持异步操作。
"""

import asyncio
import logging
from typing import AsyncGenerator, List, Union

import numpy as np

from cascade.formats.utils import get_audio_format
from cascade.formats import get_format_converter
from cascade.processor import VADProcessor, VADProcessorConfig
from cascade.types.audio import AudioChunk, AudioConfig
from cascade.types.vad import VADResult, VADSegment

# 配置日志
logger = logging.getLogger("cascade.api")


async def process_audio_file(
    file_path: str,
    **kwargs,
) -> List[VADResult]:
    """
    处理单个音频文件

    Args:
        file_path: 音频文件路径
        **kwargs: VAD处理器配置参数

    Returns:
        VAD结果列表
    """
    # 获取音频格式
    audio_format = get_audio_format(file_path)
    if not audio_format:
        raise ValueError(f"无法确定音频文件格式: {file_path}")

    # 获取格式转换器
    converter = get_format_converter(audio_format)
    if not converter:
        raise ValueError(f"不支持的音频格式: {audio_format}")

    # 读取音频数据
    from pathlib import Path
    audio_bytes = Path(file_path).read_bytes()
    audio_data, audio_config = converter.to_internal(audio_bytes)

    # 创建VAD处理器
    config = VADProcessorConfig(**kwargs)
    processor = VADProcessor(config)

    # 启动处理器
    await processor.start()

    # 处理音频
    results = await processor.process_audio(audio_data, audio_config)

    # 关闭处理器
    await processor.close()

    return [res.result_data for res in results]


async def detect_speech_segments(
    file_path: str,
    **kwargs,
) -> List[VADSegment]:
    """
    从音频文件中检测语音段

    Args:
        file_path: 音频文件路径
        **kwargs: VAD处理器配置参数

    Returns:
        语音段列表
    """
    # 创建VAD处理器
    config = VADProcessorConfig(**kwargs)
    processor = VADProcessor(config)

    # 获取音频格式
    audio_format = get_audio_format(file_path)
    if not audio_format:
        raise ValueError(f"无法确定音频文件格式: {file_path}")

    # 获取格式转换器
    converter = get_format_converter(audio_format)
    if not converter:
        raise ValueError(f"不支持的音频格式: {audio_format}")

    # 读取音频数据
    from pathlib import Path
    audio_bytes = Path(file_path).read_bytes()
    audio_data, audio_config = converter.to_internal(audio_bytes)

    # 启动处理器
    await processor.start()

    # 处理音频
    await processor.process_audio(audio_data, audio_config)

    # 获取合并后的语音段
    segments = processor.get_segments()
    
    # 关闭处理器
    await processor.close()

    return segments


async def process_audio_stream(
    stream: AsyncGenerator[bytes, None],
    sample_rate: int,
    **kwargs,
) -> AsyncGenerator[VADResult, None]:
    """
    处理音频流

    Args:
        stream: 异步音频流生成器
        sample_rate: 采样率
        **kwargs: VAD处理器配置参数

    Yields:
        VAD结果
    """
    # 创建VAD处理器
    config = VADProcessorConfig(**kwargs)
    processor = VADProcessor(config)

    # 启动处理器
    await processor.start()

    try:
        sequence = 0
        async for data_chunk in stream:
            # 将字节数据转换为numpy数组 (假设为s16le)
            # 注意: 这里需要更鲁棒的格式处理，但为简化暂定如此
            audio_array = np.frombuffer(data_chunk, dtype=np.int16)

            # 创建 AudioChunk
            chunk = AudioChunk(
                data=audio_array,
                sequence_number=sequence,
                start_frame=0,  # 流式处理中，start_frame可以简化
                chunk_size=len(audio_array),
                timestamp_ms=0,   # 流式处理的时间戳需要外部管理
                sample_rate=sample_rate,
            )

            # 处理块
            result = processor.process_chunk(chunk)
            yield result

            sequence += 1
    finally:
        # 关闭处理器
        await processor.close()

async def process_audio_bytes(
    audio_bytes: bytes,
    sample_rate: int = 16000,
    **kwargs
) -> List[VADResult]:
    """
    处理音频字节数据的便捷函数
    """
    async def bytes_generator():
        yield audio_bytes

    results = []
    async for result in process_audio_stream(
        bytes_generator(),
        sample_rate=sample_rate,
        **kwargs
    ):
        results.append(result)

    return results
