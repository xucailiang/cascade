#!/usr/bin/env python3
"""Cascade 简洁使用示例"""

import asyncio
import os
import wave

import cascade


async def process_file_example(audio_file):
    """文件处理示例"""
    print(f"处理文件: {audio_file}")

    # 创建处理器并启动
    processor = cascade.create_processor()
    await processor.start()

    # 处理音频文件
    segment_count = 0
    async for result in processor.process_file(audio_file):
        if result.is_speech_segment and result.segment:
            segment_count += 1
            duration = result.segment.duration_ms
            print(f"语音段 {segment_count}: {duration:.1f}ms")

            # 保存语音段
            save_path = f"speech_segments/segment_{segment_count}.wav"
            save_audio_segment(result.segment.audio_data, save_path)

    # 停止处理器
    await processor.stop()
    print(f"文件处理完成，检测到 {segment_count} 个语音段")


async def process_stream_example(audio_file):
    """流式处理示例"""
    print(f"流式处理: {audio_file}")

    # 创建处理器并启动
    processor = cascade.create_processor()
    await processor.start()

    # 从文件创建流
    async def file_to_stream():
        with wave.open(audio_file, 'rb') as wav_file:
            chunk_size = 512 * wav_file.getsampwidth()
            data = wav_file.readframes(512)
            while data:
                yield data
                data = wav_file.readframes(512)
                await asyncio.sleep(0.01)  # 模拟实时流

    # 处理音频流
    segment_count = 0
    async for result in processor.process_stream(file_to_stream()):
        if result.is_speech_segment and result.segment:
            segment_count += 1
            duration = result.segment.duration_ms
            print(f"流式语音段 {segment_count}: {duration:.1f}ms")

            # 保存语音段
            save_path = f"stream_speech_segments/stream_segment_{segment_count}.wav"
            save_audio_segment(result.segment.audio_data, save_path)

    # 停止处理器
    await processor.stop()
    print(f"流式处理完成，检测到 {segment_count} 个语音段")


def save_audio_segment(audio_data, save_path):
    """保存音频段到WAV文件"""
    # 确保目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 保存为WAV文件
    with wave.open(save_path, 'wb') as wav_file:
        wav_file.setnchannels(1)  # 单声道
        wav_file.setsampwidth(2)  # 16位
        wav_file.setframerate(16000)  # 16kHz
        wav_file.writeframes(audio_data)

    print(f"已保存: {save_path}")


async def main():
    """主函数"""
    # 使用真实音频文件
    audio_file = "我现在开始录音，理论上会有两个文件.wav"

    # 文件处理
    await process_file_example(audio_file)

    # 流式处理
    await process_stream_example(audio_file)


if __name__ == "__main__":
    # 运行主函数
    asyncio.run(main())
