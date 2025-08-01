#!/usr/bin/env python3
"""
测试Silero VAD流式推理
专注于演示VADIterator的语音段检测功能
"""

import asyncio
import wave
from pathlib import Path

import numpy as np

from cascade.backends import create_vad_backend
from cascade.types import AudioChunk, VADConfig


async def test_silero_streaming_vad():
    """测试Silero VAD流式语音段检测"""
    print("="*80)
    print("🚀 Silero VAD流式语音段检测测试")
    print("="*80)

    # 加载测试音频文件
    audio_file = Path("请问电动汽车和传统汽车比起来哪个更好啊？.wav")

    if not audio_file.exists():
        print(f"❌ 音频文件不存在: {audio_file}")
        return

    # 读取音频文件
    with wave.open(str(audio_file), 'rb') as wav_file:
        frames = wav_file.readframes(wav_file.getnframes())
        sample_rate = wav_file.getframerate()

        # 转换为numpy数组
        audio_data = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
        # 归一化到[-1, 1]
        audio_data = audio_data / 32768.0

    print("📁 音频文件加载成功:")
    print(f"   总样本数: {len(audio_data)}")
    print(f"   采样率: {sample_rate}Hz")
    print(f"   总时长: {len(audio_data) / sample_rate:.2f}秒")
    print()

    # 创建VAD配置（启用流式模式）
    vad_config = VADConfig(
        backend="silero",
        threshold=0.5
    )

    # 创建VAD后端
    try:
        backend = create_vad_backend(vad_config)

        # 启用流式处理模式
        backend._silero_config.streaming_mode = True
        backend._silero_config.return_seconds = False  # 返回样本数而非秒数
        backend._silero_config.onnx = True

        print("✅ VAD后端创建成功")

        # 初始化后端
        await backend.initialize()
        print("✅ VAD后端初始化成功")
        print()

        # 创建完整音频的处理块
        chunk_size = 512  # 16kHz的块大小
        total_chunks = (len(audio_data) + chunk_size - 1) // chunk_size

        print("🎤 开始流式VAD语音段检测:")
        print(f"   处理块数: {total_chunks}")
        print(f"   每块样本数: {chunk_size}")
        print(f"   每块时长: {chunk_size / sample_rate * 1000:.1f}ms")
        print()

        speech_segments = []  # 记录检测到的语音段
        current_segment = None

        for i in range(total_chunks):
            start_idx = i * chunk_size
            end_idx = min(start_idx + chunk_size, len(audio_data))

            chunk_data = audio_data[start_idx:end_idx]

            # 如果最后一块不够大小，填充零
            if len(chunk_data) < chunk_size:
                padded_data = np.zeros(chunk_size, dtype=np.float32)
                padded_data[:len(chunk_data)] = chunk_data
                chunk_data = padded_data

            chunk = AudioChunk(
                data=chunk_data,
                sequence_number=i,
                start_frame=start_idx,
                chunk_size=chunk_size,
                timestamp_ms=i * (chunk_size * 1000.0 / sample_rate),
                sample_rate=sample_rate
            )

            # 处理音频块
            result = backend.process_chunk(chunk)

            # 解析VADIterator的原始结果
            metadata = result.metadata or {}
            streaming_mode = metadata.get('streaming_mode', False)

            # 从后端获取最后的VADIterator结果
            if hasattr(backend, '_thread_local') and hasattr(backend._thread_local, 'last_vad_result'):
                vad_result = backend._thread_local.last_vad_result
                if vad_result:
                    if 'start' in vad_result:
                        start_sample = vad_result['start']
                        start_time = start_sample / sample_rate
                        print(f"🎙️  检测到语音开始: 样本{start_sample} -> {start_time:.3f}秒")
                        current_segment = {'start': start_sample, 'start_time': start_time}

                    elif 'end' in vad_result:
                        end_sample = vad_result['end']
                        end_time = end_sample / sample_rate
                        print(f"🔇 检测到语音结束: 样本{end_sample} -> {end_time:.3f}秒")

                        if current_segment:
                            current_segment['end'] = end_sample
                            current_segment['end_time'] = end_time
                            current_segment['duration'] = end_time - current_segment['start_time']
                            speech_segments.append(current_segment)
                            current_segment = None

        print()
        print("📊 语音段检测结果:")
        if speech_segments:
            for i, segment in enumerate(speech_segments, 1):
                print(f"   语音段 {i}: {segment['start_time']:.3f}s - {segment['end_time']:.3f}s")
                print(f"            时长: {segment['duration']:.3f}s")
                print(f"            样本范围: {segment['start']} - {segment['end']}")
        else:
            print("   未检测到完整的语音段")

        # 关闭后端
        await backend.close()
        print()
        print("✅ VAD后端已关闭")

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

    print()
    print("="*80)
    print("🎉 Silero VAD流式语音段检测测试完成")
    print("="*80)

if __name__ == "__main__":
    asyncio.run(test_silero_streaming_vad())
