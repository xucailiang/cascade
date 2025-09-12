#!/usr/bin/env python3
"""
流式VAD测试脚本
模拟真实的流式音频处理场景，将音频文件切割为4096字节的音频块进行处理
"""

import asyncio
import os
import wave
from collections.abc import AsyncIterator
from pathlib import Path

from cascade.stream import StreamProcessor
from cascade.stream.types import Config


async def simulate_audio_stream(audio_file: str, chunk_size: int = 4096) -> AsyncIterator[bytes]:
    """
    模拟音频流，将音频文件切割为指定大小的音频块
    
    Args:
        audio_file: 音频文件路径
        chunk_size: 音频块大小（字节）
        
    Yields:
        bytes: 音频数据块
    """
    print(f"📡 开始模拟音频流: {audio_file}")
    print(f"🔧 音频块大小: {chunk_size} 字节")

    try:
        with wave.open(audio_file, 'rb') as wav_file:
            # 获取音频信息
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            framerate = wav_file.getframerate()
            total_frames = wav_file.getnframes()

            print("🎵 音频信息:")
            print(f"   - 声道数: {channels}")
            print(f"   - 采样位深: {sample_width * 8} bit")
            print(f"   - 采样率: {framerate} Hz")
            print(f"   - 总帧数: {total_frames}")
            print(f"   - 时长: {total_frames / framerate:.2f} 秒")

            chunk_count = 0
            total_bytes = 0

            # 逐块读取音频数据
            while True:
                audio_chunk = wav_file.readframes(chunk_size // (channels * sample_width))
                if not audio_chunk:
                    break

                chunk_count += 1
                total_bytes += len(audio_chunk)

                print(f"📦 发送音频块 {chunk_count}: {len(audio_chunk)} 字节", end="\r")

                # 模拟网络延迟
                await asyncio.sleep(0.01)

                yield audio_chunk

            print(f"\n✅ 音频流模拟完成: {chunk_count} 个音频块, 总计 {total_bytes} 字节")

    except Exception as e:
        print(f"❌ 音频流模拟失败: {e}")


async def test_stream_vad_processing(audio_file: str):
    """
    测试流式VAD处理
    
    Args:
        audio_file: 音频文件路径
    """
    print(f"🎯 开始流式VAD测试: {audio_file}")

    if not os.path.exists(audio_file):
        print(f"❌ 音频文件不存在: {audio_file}")
        return

    # 创建输出目录
    output_dir = Path("stream_speech_segments")
    output_dir.mkdir(exist_ok=True)

    segment_count = 0
    frame_count = 0
    chunk_count = 0

    try:
        # 使用StreamProcessor进行流式处理
        config = Config()
        async with StreamProcessor(config) as processor:
            print("🚀 StreamProcessor 已启动")

            # 模拟音频流并处理
            audio_stream = simulate_audio_stream(audio_file, chunk_size=4096)

            async for result in processor.process_stream(audio_stream, stream_id="test_stream"):
                if result.result_type == "segment" and result.segment:
                    segment_count += 1
                    segment = result.segment

                    start_ms = segment.start_timestamp_ms
                    end_ms = segment.end_timestamp_ms
                    duration_ms = segment.duration_ms

                    print(f"\n🎤 语音段 {segment_count}: {start_ms:.0f}ms - {end_ms:.0f}ms (时长: {duration_ms:.0f}ms)")

                    # 保存语音段为WAV文件
                    output_file = output_dir / f"stream_speech_segment_{segment_count}_{start_ms:.0f}ms-{end_ms:.0f}ms.wav"
                    save_audio_segment(segment.audio_data, output_file)
                    print(f"💾 已保存: {output_file}")

                else:
                    # 单帧结果
                    frame_count += 1
                    frame = result.frame
                    if frame:  # 确保frame不为None
                        print(f"🔇 单帧 {frame_count}: {frame.timestamp_ms:.0f}ms", end="\r")

            # 获取处理统计
            stats = processor.get_stats()
            print("\n📊 处理统计:")
            print(f"   - 总结果: {segment_count + frame_count} 个")
            print(f"   - 语音段: {segment_count} 个")
            print(f"   - 单帧: {frame_count} 个")
            print(f"   - 处理器统计: {stats.summary()}")

    except Exception as e:
        print(f"❌ 流式处理过程中出错: {e}")
        return

    print("\n✅ 流式VAD测试完成！")
    if segment_count > 0:
        print(f"📁 语音段已保存到: {output_dir.absolute()}")


def save_audio_segment(audio_data: bytes, output_file: Path):
    """
    保存音频段为WAV文件
    
    Args:
        audio_data: 音频数据（16kHz, 16bit, mono）
        output_file: 输出文件路径
    """
    try:
        with wave.open(str(output_file), 'wb') as wav_file:
            # Silero VAD要求的音频格式
            wav_file.setnchannels(1)      # 单声道
            wav_file.setsampwidth(2)      # 16位
            wav_file.setframerate(16000)  # 16kHz采样率
            wav_file.writeframes(audio_data)
    except Exception as e:
        print(f"❌ 保存音频文件失败 {output_file}: {e}")


async def main():
    """主函数"""
    print("🌊 Cascade 流式VAD 测试")
    print("=" * 50)

    # 指定的音频文件
    audio_file = "/home/justin/workspace/cascade/我现在开始录音，理论上会有两个文件.wav"

    # 检查文件是否存在
    if os.path.exists(audio_file):
        await test_stream_vad_processing(audio_file)
    else:
        print(f"❌ 指定的音频文件不存在: {audio_file}")
        print("💡 请确保音频文件存在于指定路径")


if __name__ == "__main__":
    asyncio.run(main())
