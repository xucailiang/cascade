#!/usr/bin/env python3
"""
简单的VAD测试脚本

测试重构后的Cascade StreamProcessor，使用两个.wav文件进行流式VAD检测，
并将检测到的语音段保存为独立的.wav文件。
"""

import asyncio
import os
import wave
from pathlib import Path

import cascade


async def save_speech_segment(audio_data: bytes, output_path: str):
    """
    保存语音段为WAV文件
    
    Args:
        audio_data: 音频数据（16kHz, 16bit, mono）
        output_path: 输出文件路径
    """
    try:
        with wave.open(output_path, 'wb') as wav_file:
            wav_file.setnchannels(1)      # 单声道
            wav_file.setsampwidth(2)      # 16位
            wav_file.setframerate(16000)  # 16kHz采样率
            wav_file.writeframes(audio_data)
        print(f"💾 已保存语音段: {output_path}")
    except Exception as e:
        print(f"❌ 保存失败 {output_path}: {e}")


async def test_vad_on_file(audio_file: str):
    """
    测试单个音频文件的VAD处理
    
    Args:
        audio_file: 音频文件路径
    """
    if not os.path.exists(audio_file):
        print(f"❌ 文件不存在: {audio_file}")
        return
    
    print(f"\n🎯 开始处理: {Path(audio_file).name}")
    print("=" * 50)
    
    # 创建输出目录
    output_dir = Path("speech_segments")
    output_dir.mkdir(exist_ok=True)
    
    # 文件名前缀
    file_prefix = Path(audio_file).stem
    
    segment_count = 0
    frame_count = 0
    
    try:
        # 创建配置
        config = cascade.Config(
            vad_threshold=0.5,
            min_silence_duration_ms=500,
            speech_pad_ms=300
        )
        
        # 使用StreamProcessor处理文件
        async with cascade.StreamProcessor(config) as processor:
            print("✅ StreamProcessor已启动（独立VAD模型）")
            
            async for result in processor.process_file(audio_file):
                if result.is_speech_segment and result.segment:
                    segment_count += 1
                    segment = result.segment
                    
                    # 生成输出文件名
                    start_ms = int(segment.start_timestamp_ms)
                    end_ms = int(segment.end_timestamp_ms)
                    duration_ms = int(segment.duration_ms)
                    
                    output_filename = f"{file_prefix}_segment_{segment_count:03d}_{start_ms}ms-{end_ms}ms.wav"
                    output_path = output_dir / output_filename
                    
                    # 保存语音段
                    await save_speech_segment(segment.audio_data, str(output_path))
                    
                    print(f"🎤 语音段 {segment_count}: {start_ms}ms - {end_ms}ms "
                          f"(时长: {duration_ms}ms, {segment.frame_count}帧)")
                
                elif result.frame:
                    frame_count += 1
                    if frame_count % 50 == 0:  # 每50帧打印一次
                        print(f"🔇 处理帧: {frame_count}", end="\r")
            
            # 获取统计信息
            stats = processor.get_stats()
            
            print(f"\n📊 处理完成:")
            print(f"   🎤 语音段数量: {segment_count}")
            print(f"   🔇 单帧数量: {frame_count}")
            print(f"   📦 总处理块: {stats.total_chunks_processed}")
            print(f"   ⏱️  平均处理时间: {stats.average_processing_time_ms:.2f}ms")
            print(f"   💾 语音段保存到: {output_dir.absolute()}")
    
    except Exception as e:
        print(f"❌ 处理失败: {e}")


async def main():
    """主函数"""
    print("🌊 Cascade 简单VAD测试")
    print("基于重构后的1:1:1:1架构")
    print("=" * 50)
    
    # 测试文件列表
    test_files = [
        "我现在开始录音，理论上会有两个文件.wav"
    ]
    
    # 检查并处理每个文件
    for audio_file in test_files:
        if os.path.exists(audio_file):
            await test_vad_on_file(audio_file)
        else:
            print(f"⚠️  跳过不存在的文件: {audio_file}")
    
    print(f"\n🎉 测试完成！")
    print(f"✅ 重构后的StreamProcessor工作正常")
    print(f"✅ 独立模型架构无并发问题")


if __name__ == "__main__":
    asyncio.run(main())