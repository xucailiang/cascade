#!/usr/bin/env python3
"""
Cascade VAD 示例程序

本示例展示了如何使用Cascade库的顶层API处理音频文件和音频流。
"""

import asyncio
import logging
import os
import sys
import time
from typing import AsyncGenerator

import numpy as np

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cascade
from cascade.types.vad import VADSegment


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("vad_demo")


async def process_file_demo(file_path: str) -> None:
    """
    处理音频文件示例
    
    Args:
        file_path: 音频文件路径
    """
    logger.info(f"处理音频文件: {file_path}")
    
    # 使用便捷函数处理音频文件
    results = await cascade.process_audio_file(
        file_path,
        threshold=0.5,
        workers=4,
        model_path="/home/justin/opensource/cascade/models/silero-vad/model.onnx"
    )
    
    # 打印结果
    logger.info(f"检测到 {len(results)} 个结果")
    for i, result in enumerate(results):
        if result.is_speech:
            logger.info(f"语音 #{i}: {result.start_ms:.1f}ms - {result.end_ms:.1f}ms, "
                       f"概率: {result.probability:.2f}, 置信度: {result.confidence:.2f}")


async def detect_segments_demo(file_path: str) -> None:
    """
    检测语音段示例
    
    Args:
        file_path: 音频文件路径
    """
    logger.info(f"检测语音段: {file_path}")
    
    # 使用便捷函数检测语音段
    segments = await cascade.detect_speech_segments(
        file_path,
        threshold=0.5,
        min_speech_duration_ms=100,
        min_silence_duration_ms=300,
        workers=4,
        model_path="/home/justin/opensource/cascade/models/silero-vad/model.onnx"
    )
    
    # 打印结果
    logger.info(f"检测到 {len(segments)} 个语音段")
    for i, segment in enumerate(segments):
        duration_ms = segment.end_ms - segment.start_ms
        logger.info(f"语音段 #{i}: {segment.start_ms:.1f}ms - {segment.end_ms:.1f}ms, "
                   f"时长: {duration_ms:.1f}ms, 置信度: {segment.confidence:.2f}")


async def stream_generator(file_path: str, chunk_size: int = 4000) -> AsyncGenerator[bytes, None]:
    """
    模拟音频流生成器
    
    Args:
        file_path: 音频文件路径
        chunk_size: 块大小（字节）
        
    Yields:
        音频字节数据
    """
    with open(file_path, 'rb') as f:
        # 跳过WAV文件头（44字节）
        header = f.read(44)
        
        # 读取数据块
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
                
            # 模拟网络延迟
            await asyncio.sleep(0.1)
            
            yield chunk


async def process_stream_demo(file_path: str) -> None:
    """
    处理音频流示例
    
    Args:
        file_path: 音频文件路径（用于模拟流）
    """
    logger.info(f"处理音频流: {file_path}")
    
    # 创建音频流生成器
    audio_stream = stream_generator(file_path)
    
    # 使用便捷函数处理音频流
    speech_count = 0
    async for result in cascade.process_audio_stream(
        audio_stream,
        sample_rate=16000,
        audio_format="wav",
        threshold=0.5,
        workers=4,
        model_path="/home/justin/opensource/cascade/models/silero-vad/model.onnx"
    ):
        if result.is_speech:
            speech_count += 1
            logger.info(f"检测到语音: {result.start_ms:.1f}ms - {result.end_ms:.1f}ms, "
                       f"概率: {result.probability:.2f}")
    
    logger.info(f"流处理完成，检测到 {speech_count} 个语音块")



async def main():
    """主函数"""
    # 检查命令行参数
    if len(sys.argv) < 2:
        print(f"用法: {sys.argv[0]} <音频文件路径>")
        return
    
    file_path = sys.argv[1]
    
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"错误: 文件不存在: {file_path}")
        return
    
    # 打印Cascade版本信息
    logger.info(f"Cascade 版本: {cascade.__version__}")
    
    # 运行示例
    await process_file_demo(file_path)
    await detect_segments_demo(file_path)
    await process_stream_demo(file_path)
    

if __name__ == "__main__":
    # 运行主函数
    asyncio.run(main())