#!/usr/bin/env python3
"""
Cascade快速VAD处理示例
针对音频文件: "请问电动汽车和传统汽车比起来哪个更好啊？.wav"

这是一个简化的快速演示，展示Cascade的核心功能：
1. 4线程并行VAD处理
2. Silero VAD模型
3. 实时性能监控
4. 语音段检测结果
"""

import asyncio
import time
from pathlib import Path

import numpy as np

from cascade.backends import create_vad_backend

# Cascade核心导入
from cascade.types import AudioChunk, AudioConfig, AudioFormat, VADConfig


async def quick_vad_demo(audio_file: str = "请问电动汽车和传统汽车比起来哪个更好啊？.wav"):
    """快速VAD处理演示"""

    print("🚀 Cascade快速VAD处理演示")
    print(f"📁 目标文件: {audio_file}")

    # === 1. 配置设置 ===
    print("\n⚙️ 配置Cascade参数...")

    # VAD配置（针对中文优化）
    vad_config = VADConfig(
        backend="silero",                 # 使用Silero后端
        threshold=0.3,                    # 中文语音较低阈值
        chunk_duration_ms=512,            # 512ms块（最优性能）
        overlap_ms=32,                    # 32ms重叠
        min_speech_duration_ms=200,       # 最小语音段200ms
        workers=4                         # 4个工作线程
    )

    # 音频配置
    audio_config = AudioConfig(
        sample_rate=16000,                # 16kHz标准采样率
        channels=1,                       # 单声道
        format=AudioFormat.WAV
    )

    # 处理器配置 - 暂时不使用完整处理器，直接使用VAD后端

    print("✅ 配置完成")

    # === 2. 加载真实音频数据 ===
    print("\n🎵 准备音频数据...")

    # 检查文件是否存在（先检查当前目录，再检查上级目录）
    audio_path = Path(audio_file)
    if not audio_path.exists():
        # 尝试在上级目录查找
        parent_path = Path("..") / audio_file
        if parent_path.exists():
            audio_path = parent_path
            print(f"📂 在上级目录找到音频文件: {parent_path}")
        else:
            print("📝 音频文件不存在，创建模拟语音数据进行演示...")
            audio_data = create_simulation_audio()

    if audio_path.exists():
        print(f"📂 加载真实音频文件: {audio_path}")
        try:
            import wave
            with wave.open(str(audio_path), 'rb') as wav_file:
                # 获取音频参数
                sample_rate = wav_file.getframerate()
                channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                n_frames = wav_file.getnframes()

                print(f"  📊 音频参数: {sample_rate}Hz, {channels}声道, {sample_width*8}位, {n_frames/sample_rate:.2f}秒")

                # 读取音频数据
                raw_audio = wav_file.readframes(n_frames)

                # 转换为numpy数组
                if sample_width == 2:  # 16位
                    audio_data = np.frombuffer(raw_audio, dtype=np.int16).astype(np.float32) / 32768.0
                elif sample_width == 4:  # 32位
                    audio_data = np.frombuffer(raw_audio, dtype=np.int32).astype(np.float32) / 2147483648.0
                else:
                    raise ValueError(f"不支持的位深: {sample_width*8}位")

                # 如果是立体声，转为单声道
                if channels == 2:
                    audio_data = audio_data.reshape(-1, 2).mean(axis=1)

                # 重采样到16kHz（如果需要）
                if sample_rate != 16000:
                    print(f"  🔄 重采样: {sample_rate}Hz -> 16000Hz")
                    from scipy import signal
                    audio_data = signal.resample(audio_data, int(len(audio_data) * 16000 / sample_rate))

                print(f"✅ 音频加载成功: {len(audio_data)/16000:.2f}秒, {len(audio_data)}个采样点")

        except Exception as e:
            print(f"❌ 音频加载失败: {e}")
            print("📝 使用模拟语音数据进行演示...")
            audio_data = create_simulation_audio()

    # === 3. 初始化VAD系统 ===
    print("\n🤖 初始化Silero VAD系统...")

    start_time = time.time()

    # 创建VAD后端
    backend = create_vad_backend(vad_config)
    await backend.initialize()

    init_time = time.time() - start_time
    print(f"✅ 系统初始化完成: {init_time:.3f}秒")

    # === 4. 高性能VAD处理 ===
    print("\n⚡ 开始VAD处理...")

    # 将音频分割为块
    chunk_size = 8192  # 512ms @ 16kHz
    results = []
    speech_segments = []

    print(f"📦 音频分块: {len(audio_data)//chunk_size + 1}个块 x {chunk_size/16000*1000:.0f}ms")

    # 开始处理
    processing_start = time.time()

    print("🔍 VAD检测结果:")
    print("=" * 60)

    for i in range(0, len(audio_data), chunk_size):
        chunk_data = audio_data[i:i+chunk_size]
        if len(chunk_data) < chunk_size:
            # 最后一块补零
            padded_chunk = np.zeros(chunk_size, dtype=np.float32)
            padded_chunk[:len(chunk_data)] = chunk_data
            chunk_data = padded_chunk

        # 创建音频块
        chunk = AudioChunk(
            data=chunk_data,
            sequence_number=i // chunk_size,
            start_frame=i,
            chunk_size=len(chunk_data),
            timestamp_ms=i / 16000 * 1000,
            sample_rate=16000
        )

        # VAD推理
        result = backend.process_chunk(chunk)
        results.append(result)

        # 实时显示结果
        time_str = f"{result.start_ms/1000:.2f}-{result.end_ms/1000:.2f}s"
        if result.is_speech:
            status = "🗣️ 语音"
            speech_segments.append({
                'start': result.start_ms/1000,
                'end': result.end_ms/1000,
                'probability': result.probability
            })
        else:
            status = "🔇 静音"

        print(f"{status} | {time_str} | 概率: {result.probability:.3f}")

    processing_time = time.time() - processing_start

    # === 5. 结果分析 ===
    print("=" * 60)
    print("\n📊 处理结果分析:")

    total_chunks = len(results)
    speech_chunks = sum(1 for r in results if r.is_speech)
    avg_probability = np.mean([r.probability for r in results])

    print("  📈 总体统计:")
    print(f"    - 音频总时长: {len(audio_data)/16000:.2f}秒")
    print(f"    - 处理块数: {total_chunks}")
    print(f"    - 语音块数: {speech_chunks}")
    print(f"    - 静音块数: {total_chunks - speech_chunks}")
    print(f"    - 平均概率: {avg_probability:.3f}")
    print(f"    - 处理耗时: {processing_time:.3f}秒")

    if speech_segments:
        print("\n  🎤 检测到语音段:")
        for i, segment in enumerate(speech_segments, 1):
            duration = segment['end'] - segment['start']
            print(f"    {i}. {segment['start']:.2f}s - {segment['end']:.2f}s "
                  f"(时长: {duration:.2f}s, 概率: {segment['probability']:.3f})")

        total_speech_duration = sum(s['end'] - s['start'] for s in speech_segments)
        speech_ratio = total_speech_duration / (len(audio_data)/16000) * 100
        print(f"    语音比例: {speech_ratio:.1f}%")

    # === 6. 性能指标 ===
    print("\n  ⚡ 性能指标:")
    print(f"    - 平均延迟: {(processing_time/total_chunks)*1000:.2f}ms/块")
    print(f"    - 实时倍率: {(len(audio_data)/16000) / processing_time:.1f}x")
    print(f"    - 吞吐量: {total_chunks / processing_time:.1f} chunks/s")

    await backend.close()

    print("\n🎉 VAD处理完成！")
    print("\n💡 提示:")
    print("  - 将真实的WAV文件放在当前目录，即可处理真实语音")
    print("  - 调整vad_config.threshold可以改变检测敏感度")
    print("  - 支持多种音频格式和采样率")
    print("  - 线程数可根据CPU核心数调整以获得最佳性能")


# === 辅助函数 ===

def create_simulation_audio():
    """创建模拟中文语音数据用于演示"""
    print("🎙️ 生成模拟中文语音信号...")

    # 创建10秒模拟音频：包含3段语音和2段静音
    duration = 10.0  # 10秒
    sample_rate = 16000
    total_samples = int(duration * sample_rate)

    # 生成模拟的中文语音信号
    t = np.linspace(0, duration, total_samples)
    audio_data = np.zeros(total_samples, dtype=np.float32)

    # 语音段1: 0.2-2.5秒 "请问电动汽车"
    mask1 = (t >= 0.2) & (t <= 2.5)
    audio_data[mask1] = (
        0.4 * np.sin(2 * np.pi * 180 * t[mask1]) +  # 基频
        0.3 * np.sin(2 * np.pi * 360 * t[mask1]) +  # 二次谐波
        0.2 * np.sin(2 * np.pi * 540 * t[mask1]) +  # 三次谐波
        0.1 * np.random.randn(np.sum(mask1))        # 噪声
    )

    # 静音段: 2.5-3.5秒

    # 语音段2: 3.5-6秒 "和传统汽车比起来"
    mask2 = (t >= 3.5) & (t <= 6.0)
    audio_data[mask2] = (
        0.3 * np.sin(2 * np.pi * 200 * t[mask2]) +
        0.3 * np.sin(2 * np.pi * 400 * t[mask2]) +
        0.2 * np.sin(2 * np.pi * 600 * t[mask2]) +
        0.1 * np.random.randn(np.sum(mask2))
    )

    # 静音段: 6-6.8秒

    # 语音段3: 6.8-9秒 "哪个更好啊？"
    mask3 = (t >= 6.8) & (t <= 9.0)
    audio_data[mask3] = (
        0.35 * np.sin(2 * np.pi * 220 * t[mask3]) +
        0.25 * np.sin(2 * np.pi * 440 * t[mask3]) +
        0.15 * np.sin(2 * np.pi * 660 * t[mask3]) +
        0.1 * np.random.randn(np.sum(mask3))
    )

    # 添加背景噪声
    background_noise = 0.02 * np.random.randn(total_samples)
    audio_data += background_noise

    # 归一化
    audio_data = audio_data / np.max(np.abs(audio_data)) * 0.8

    print(f"✅ 模拟音频生成完成: {duration}秒, 3个语音段")
    return audio_data.astype(np.float32)


def create_test_audio_file(filename: str = "请问电动汽车和传统汽车比起来哪个更好啊？.wav"):
    """创建测试音频文件（可选）"""
    import wave

    print(f"🎵 创建测试音频文件: {filename}")

    # 生成10秒测试音频
    sample_rate = 16000
    duration = 10.0
    t = np.linspace(0, duration, int(sample_rate * duration))

    # 创建包含多个语音段的测试信号
    signal = np.zeros_like(t)

    # 语音段1: 0.5-3秒
    mask1 = (t >= 0.5) & (t <= 3.0)
    signal[mask1] = 0.3 * (np.sin(2*np.pi*200*t[mask1]) + 0.5*np.sin(2*np.pi*400*t[mask1]))

    # 语音段2: 4-7秒
    mask2 = (t >= 4.0) & (t <= 7.0)
    signal[mask2] = 0.4 * (np.sin(2*np.pi*250*t[mask2]) + 0.3*np.sin(2*np.pi*500*t[mask2]))

    # 语音段3: 8-9.5秒
    mask3 = (t >= 8.0) & (t <= 9.5)
    signal[mask3] = 0.35 * (np.sin(2*np.pi*180*t[mask3]) + 0.4*np.sin(2*np.pi*360*t[mask3]))

    # 添加噪声
    signal += 0.02 * np.random.randn(len(signal))

    # 归一化并转换为16位整数
    signal = signal / np.max(np.abs(signal)) * 0.8
    signal_int16 = (signal * 32767).astype(np.int16)

    # 保存为WAV文件
    with wave.open(filename, 'wb') as wav_file:
        wav_file.setnchannels(1)      # 单声道
        wav_file.setsampwidth(2)      # 16位
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(signal_int16.tobytes())

    print(f"✅ 测试音频文件已创建: {filename}")


async def main():
    """主函数"""
    print("🎬 Cascade高性能并行VAD处理快速演示")
    print("=" * 50)

    # 可选：创建测试音频文件
    # create_test_audio_file()

    # 运行快速演示
    await quick_vad_demo("请问电动汽车和传统汽车比起来哪个更好啊？.wav")


if __name__ == "__main__":
    # 运行演示
    asyncio.run(main())
