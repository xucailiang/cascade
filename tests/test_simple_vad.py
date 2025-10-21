#!/usr/bin/env python3
"""
简化的VAD测试脚本
使用Cascade进行语音活动检测，并保存每个检测到的语音段

改造说明：
- 使用最新的StreamProcessor API
- 使用cascade.Config()创建配置
- 使用async with上下文管理器
- 使用result.is_speech_segment检查结果类型
- 添加统计信息获取
"""

import asyncio
import os
import wave
from pathlib import Path

import cascade


async def test_vad_with_audio_file(audio_file: str):
    """
    测试VAD功能并保存语音段
    
    Args:
        audio_file: 音频文件路径
    """
    print(f"🎵 开始处理音频文件: {audio_file}")

    if not os.path.exists(audio_file):
        print(f"❌ 音频文件不存在: {audio_file}")
        return

    # 创建输出目录
    output_dir = Path("speech_segments")
    output_dir.mkdir(exist_ok=True)

    segment_count = 0
    frame_count = 0

    try:
        # 创建配置
        config = cascade.Config(
            vad_threshold=0.5,
            min_silence_duration_ms=500,
            speech_pad_ms=300
        )
        
        # 使用StreamProcessor处理音频文件
        async with cascade.StreamProcessor(config) as processor:
            print("✅ StreamProcessor已启动")
            
            async for result in processor.process_file(audio_file):
                if result.is_speech_segment and result.segment:
                    segment_count += 1
                    segment = result.segment

                    # 打印语音段信息
                    start_ms = int(segment.start_timestamp_ms)
                    end_ms = int(segment.end_timestamp_ms)
                    duration_ms = int(segment.duration_ms)

                    print(f"🎤 语音段 {segment_count}: {start_ms}ms - {end_ms}ms (时长: {duration_ms}ms)")

                    # 保存语音段为WAV文件
                    output_file = output_dir / f"speech_segment_{segment_count}_{start_ms}ms-{end_ms}ms.wav"
                    await save_audio_segment(segment.audio_data, output_file)
                    print(f"💾 已保存: {output_file}")

                elif result.frame:
                    # 单帧结果
                    frame_count += 1
                    frame = result.frame
                    if frame_count % 50 == 0:  # 每50帧打印一次
                        print(f"🔇 单帧 {frame_count}: {frame.timestamp_ms:.0f}ms", end="\r")
            
            # 获取统计信息
            stats = processor.get_stats()
            print(f"\n📊 处理统计:")
            print(f"   🎤 语音段数量: {segment_count}")
            print(f"   🔇 单帧数量: {frame_count}")
            print(f"   📦 总处理块: {stats.total_chunks_processed}")
            print(f"   ⏱️  平均处理时间: {stats.average_processing_time_ms:.2f}ms")

    except Exception as e:
        print(f"❌ 处理过程中出错: {e}")
        return

    print(f"\n✅ 处理完成！共检测到 {segment_count} 个语音段")
    if segment_count > 0:
        print(f"📁 语音段已保存到: {output_dir.absolute()}")


async def save_audio_segment(audio_data: bytes, output_file: Path):
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
    print("🚀 Cascade VAD 简化测试")
    print("=" * 50)

    # 测试文件列表（按优先级排序）
    test_files = [
        "我现在开始录音，理论上会有两个文件.wav"
    ]

    # 寻找可用的音频文件
    audio_file = None
    for file_path in test_files:
        if os.path.exists(file_path):
            audio_file = file_path
            break

    if audio_file:
        await test_vad_with_audio_file(audio_file)
    else:
        print("❌ 未找到可用的音频文件")
        print("请将音频文件放在项目根目录，支持的文件名:")
        for file_path in test_files:
            print(f"  - {file_path}")
        print("\n💡 提示: 音频文件应为WAV格式，建议16kHz采样率")


if __name__ == "__main__":
    asyncio.run(main())
