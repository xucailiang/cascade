"""
Cascade 流式VAD处理器使用示例

展示如何使用Cascade流式VAD处理器进行语音活动检测和语音段提取。
"""

import asyncio
import wave
import logging
from typing import AsyncIterator
from pathlib import Path

from cascade.stream import StreamProcessor, create_default_config, SpeechSegment

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def create_audio_stream_from_file(file_path: str) -> AsyncIterator[bytes]:
    """
    从音频文件创建异步音频流
    
    Args:
        file_path: WAV音频文件路径
        
    Yields:
        bytes: 音频数据块（1024字节 = 512样本）
    """
    with wave.open(file_path, 'rb') as wav_file:
        frame_size = 1024  # 512样本 * 2字节
        
        while True:
            chunk = wav_file.readframes(512)  # 读取512样本
            if len(chunk) == 0:
                break
            
            if len(chunk) == frame_size:
                yield chunk
                # 智能延迟：保持VAD时序的最小延迟
                await asyncio.sleep(0.001)


async def basic_usage_example():
    """基础使用示例"""
    logger.info("=== 基础使用示例 ===")
    
    # 1. 创建配置
    config = create_default_config(
        vad_threshold=0.5,  # VAD阈值
        max_instances=1     # 最大实例数
    )
    
    # 2. 创建处理器
    async with StreamProcessor(config) as processor:
        logger.info("流式处理器已启动")
        
        # 3. 创建音频流（这里使用模拟数据）
        async def mock_audio_stream():
            # 模拟音频数据（实际使用中替换为真实音频流）
            for i in range(10):
                # 生成1024字节的模拟音频数据
                mock_data = b'\x00' * 1024
                yield mock_data
                await asyncio.sleep(0.001)
        
        # 4. 处理音频流
        result_count = 0
        speech_count = 0
        
        async for result in processor.process_stream(mock_audio_stream(), "example-stream"):
            result_count += 1
            
            if result.is_speech_segment:
                speech_count += 1
                segment = result.segment
                if segment:
                    logger.info(f"检测到语音段: 时长{segment.duration_ms:.0f}ms")
            
            # 限制示例输出
            if result_count >= 10:
                break
        
        logger.info(f"处理了 {result_count} 个结果，检测到 {speech_count} 个语音段")


async def file_processing_example(audio_file: str):
    """文件处理示例"""
    logger.info("=== 文件处理示例 ===")
    
    if not Path(audio_file).exists():
        logger.warning(f"音频文件不存在: {audio_file}")
        return
    
    # 创建配置
    config = create_default_config()
    
    # 处理音频文件
    async with StreamProcessor(config) as processor:
        audio_stream = create_audio_stream_from_file(audio_file)
        
        speech_segments = []
        
        async for result in processor.process_stream(audio_stream, "file-processing"):
            if result.is_speech_segment and result.segment:
                speech_segments.append(result.segment)
                logger.info(
                    f"语音段 {result.segment.segment_id}: "
                    f"{result.segment.duration_ms:.0f}ms"
                )
        
        logger.info(f"总共检测到 {len(speech_segments)} 个语音段")
        return speech_segments


async def chunk_processing_example():
    """块处理示例"""
    logger.info("=== 块处理示例 ===")
    
    config = create_default_config()
    
    async with StreamProcessor(config) as processor:
        # 模拟音频块
        audio_chunk = b'\x00' * 1024  # 1024字节音频数据
        
        # 处理单个音频块
        results = await processor.process_chunk(audio_chunk)
        
        logger.info(f"处理音频块，产生 {len(results)} 个结果")
        
        for result in results:
            if result.is_speech_segment:
                logger.info("检测到语音段")
            else:
                logger.info("单帧处理结果")


def save_speech_segment(segment: SpeechSegment, output_path: str):
    """
    保存语音段为WAV文件
    
    Args:
        segment: 语音段对象
        output_path: 输出文件路径
    """
    with wave.open(output_path, 'wb') as wav_file:
        wav_file.setnchannels(1)  # 单声道
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(segment.sample_rate)
        wav_file.writeframes(segment.audio_data)
    
    logger.info(f"语音段已保存: {output_path}")


async def main():
    """主函数 - 运行所有示例"""
    logger.info("🚀 Cascade 流式VAD处理器使用示例")
    
    try:
        # 基础使用示例
        await basic_usage_example()
        
        # 块处理示例
        await chunk_processing_example()
        
        # 文件处理示例（如果有音频文件）
        # await file_processing_example("path/to/your/audio.wav")
        
        logger.info("✅ 所有示例运行完成")
        
    except Exception as e:
        logger.error(f"示例运行失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())