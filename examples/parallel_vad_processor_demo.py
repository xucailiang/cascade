#!/usr/bin/env python3
"""
Cascade高并发并行VAD处理器演示
使用真实音频文件: "请问电动汽车和传统汽车比起来哪个更好啊？.wav"

展示Cascade的真正并行处理能力：
1. 完整的VADProcessor流式处理器
2. 多线程并行VAD推理（1:1:1绑定架构）
3. 异步音频流处理
4. 实时性能监控和统计
5. 背压控制和流控机制
6. 零拷贝内存管理
"""

import asyncio
import time
import wave
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import numpy as np

from cascade._internal.thread_pool import VADThreadPoolConfig
from cascade.processor import VADProcessorConfig, create_vad_processor

# Cascade核心导入
from cascade.types import AudioConfig, AudioFormat, VADConfig, VADResult


class AudioStreamGenerator:
    """音频流生成器 - 将音频文件转换为异步流"""

    def __init__(self, audio_data: np.ndarray, chunk_size: int, sample_rate: int):
        self.audio_data = audio_data
        self.chunk_size = chunk_size
        self.sample_rate = sample_rate

    async def generate_stream(self) -> AsyncIterator[np.ndarray]:
        """生成音频流"""
        total_chunks = (len(self.audio_data) + self.chunk_size - 1) // self.chunk_size

        for i in range(total_chunks):
            start = i * self.chunk_size
            end = min(start + self.chunk_size, len(self.audio_data))

            chunk = self.audio_data[start:end]

            yield chunk

        print(f"🎵 音频流生成完成: {total_chunks}个块")


class ParallelPerformanceMonitor:
    """并行处理性能监控器"""

    def __init__(self):
        self.start_time = time.time()
        self.results: list[VADResult] = []
        self.processing_times: list[float] = []
        self.chunk_times: list[float] = []

    def add_result(self, result: VADResult):
        """添加处理结果"""
        self.results.append(result)
        self.chunk_times.append(time.time())

    def get_parallel_statistics(self) -> dict[str, Any]:
        """获取并行处理统计"""
        if not self.results:
            return {}

        total_time = time.time() - self.start_time
        speech_results = [r for r in self.results if r.is_speech]

        # 计算并行度指标
        if len(self.chunk_times) > 1:
            time_intervals = [self.chunk_times[i] - self.chunk_times[i-1] for i in range(1, len(self.chunk_times))]
            avg_interval = np.mean(time_intervals)
            max_interval = np.max(time_intervals)
            min_interval = np.min(time_intervals)
        else:
            avg_interval = max_interval = min_interval = 0.0

        return {
            "总处理时间": f"{total_time:.3f}秒",
            "处理块数": len(self.results),
            "语音块数": len(speech_results),
            "静音块数": len(self.results) - len(speech_results),
            "语音比例": f"{len(speech_results)/len(self.results)*100:.1f}%",
            "平均块间隔": f"{avg_interval*1000:.2f}ms",
            "最大块间隔": f"{max_interval*1000:.2f}ms",
            "最小块间隔": f"{min_interval*1000:.2f}ms",
            "并行吞吐量": f"{len(self.results)/total_time:.1f} chunks/s",
            "实时倍率": f"{(len(self.results) * 0.512) / total_time:.1f}x",  # 假设512ms块
            "平均置信度": f"{np.mean([r.probability for r in self.results]):.3f}"
        }


async def load_real_audio_file(audio_file: str) -> np.ndarray:
    """加载真实音频文件 - 只处理真实文件，不生成模拟数据"""
    audio_path = Path(audio_file)

    if not audio_path.exists():
        raise FileNotFoundError(f"音频文件 '{audio_file}' 不存在！请确保文件在当前目录。")

    print(f"📂 加载真实音频文件: {audio_path}")

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
        return audio_data.astype(np.float32)


async def parallel_vad_demo():
    """高并发并行VAD处理演示"""

    print("🚀 Cascade高并发并行VAD处理器演示")
    print("=" * 60)

    # === 1. 配置高性能并行处理参数 ===
    print("\n⚙️ 配置高性能并行处理参数...")

    # VAD配置 - 优化并行性能
    vad_config = VADConfig(
        backend="silero",                 # 使用Silero后端
        threshold=0.5,                    # 中文语音较低阈值
        chunk_duration_ms=512,            # 512ms块（最优并行性能）
        overlap_ms=32,                    # 32ms重叠
        min_speech_duration_ms=200,       # 最小语音段200ms
        workers=8                         # 8个工作线程（高并发）
    )

    # 音频配置
    audio_config = AudioConfig(
        sample_rate=16000,                # 16kHz标准采样率
        channels=1,                       # 单声道
        format=AudioFormat.WAV
    )

    # 线程池配置 - 1:1:1绑定架构
    thread_pool_config = VADThreadPoolConfig(
        max_workers=8,                    # 8个工作线程
        thread_name_prefix="VADWorker",   # 线程名称前缀
        shutdown_timeout_seconds=30.0,   # 关闭超时30秒
        warmup_enabled=True,              # 启用预热
        warmup_iterations=3,              # 预热3次
        stats_enabled=True                # 启用统计
    )

    # 处理器配置 - 高性能设置
    processor_config = VADProcessorConfig(
        audio_config=audio_config,
        vad_config=vad_config,
        thread_pool_config=thread_pool_config,
        buffer_capacity_seconds=3.0,      # 3秒缓冲区
        max_queue_size=64,                # 大队列支持高并发
        enable_performance_monitoring=True
    )

    print("✅ 并行配置完成")
    print(f"   - 工作线程数: {vad_config.workers}")
    print(f"   - 队列大小: {processor_config.max_queue_size}")
    print(f"   - 缓冲区容量: {processor_config.buffer_capacity_seconds}秒")

    # === 2. 加载真实音频文件 ===
    print("\n🎵 加载真实音频文件...")

    audio_file = "请问电动汽车和传统汽车比起来哪个更好啊？.wav"
    audio_data = await load_real_audio_file(audio_file)

    # === 3. 初始化VAD处理器 ===
    print("\n🤖 初始化高并发VAD处理器...")

    start_time = time.time()

    # 创建处理器
    processor = await create_vad_processor(
        audio_config=audio_config,
        vad_config=vad_config,
        processor_config=processor_config
    )

    init_time = time.time() - start_time
    print(f"✅ 处理器初始化完成: {init_time:.3f}秒")

    try:
        # === 4. 高并发流式处理 ===
        print("\n⚡ 开始高并发并行VAD流式处理...")

        # 创建音频流生成器
        chunk_size = 4096  # 较小的块增加并行度
        stream_generator = AudioStreamGenerator(audio_data, chunk_size, 16000)

        # 性能监控器
        monitor = ParallelPerformanceMonitor()

        print(f"📦 音频流配置: {len(audio_data)//chunk_size + 1}个块 x {chunk_size/16000*1000:.0f}ms")
        print("🔍 并行VAD检测结果:")
        print("=" * 60)

        # 开始并行流式处理
        processing_start = time.time()
        speech_segments = []

        async with processor:
            audio_stream = stream_generator.generate_stream()

            async for result in processor.process_stream(audio_stream):
                monitor.add_result(result)

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

                print(f"{status} | {time_str} | 概率: {result.probability:.3f} | 块ID: {result.chunk_id}")

        processing_time = time.time() - processing_start
        
        print(f"实际处理时间: {processing_time:.3f}秒")

        # === 5. 并行处理结果分析 ===
        print("=" * 60)
        print("\n📊 高并发并行处理结果分析:")

        # 基本统计
        stats = monitor.get_parallel_statistics()
        print("  🎯 基本统计:")
        for key, value in stats.items():
            print(f"    - {key}: {value}")

        # 语音段统计
        if speech_segments:
            print("\n  🎤 检测到的语音段:")
            for i, segment in enumerate(speech_segments, 1):
                duration = segment['end'] - segment['start']
                print(f"    {i}. {segment['start']:.2f}s - {segment['end']:.2f}s "
                      f"(时长: {duration:.2f}s, 概率: {segment['probability']:.3f})")

            total_speech_duration = sum(s['end'] - s['start'] for s in speech_segments)
            speech_ratio = total_speech_duration / (len(audio_data)/16000) * 100
            print(f"    总语音时长: {total_speech_duration:.2f}s ({speech_ratio:.1f}%)")

        # === 6. 处理器性能指标 ===
        metrics = processor.get_performance_metrics()

        print("\n  ⚡ 处理器内部性能指标:")
        print(f"    - 处理块数: {metrics.success_count + metrics.error_count}")
        print(f"    - 平均延迟: {metrics.avg_latency_ms:.2f}ms")
        print(f"    - 吞吐量: {metrics.throughput_qps:.1f} QPS")
        print(f"    - 活跃线程数: {metrics.active_threads}")
        print(f"    - 队列深度: {metrics.queue_depth}")
        print(f"    - 缓冲区利用率: {metrics.buffer_utilization:.1%}")
        print(f"    - 零拷贝率: {metrics.zero_copy_rate:.1%}")
        print(f"    - 错误率: {metrics.error_rate:.3f}")

    finally:
        # 清理资源
        await processor.close()

    print("\n🎉 高并发并行VAD处理完成！")
    print("\n💡 并行处理特性:")
    print("  - 使用完整的VADProcessor进行流式处理")
    print("  - 8个工作线程并行VAD推理（1:1:1绑定架构）")
    print("  - 异步音频流处理和背压控制")
    print("  - 零拷贝内存管理和高性能缓冲区")
    print("  - 实时性能监控和流控机制")


async def main():
    """主函数"""
    print("🎬 Cascade高并发并行VAD处理器演示")
    print("=" * 50)

    # 运行并行演示
    await parallel_vad_demo()


if __name__ == "__main__":
    # 运行并行演示
    asyncio.run(main())
