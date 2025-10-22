#!/usr/bin/env python3
"""
流式VAD测试脚本 - 支持不同音频块大小测试
模拟真实的流式音频处理场景，支持自定义音频块大小进行性能对比测试

改造说明：
- 使用最新的cascade模块API
- 使用cascade.Config()和cascade.StreamProcessor()
- 使用result.is_speech_segment检查结果类型
- 添加统计信息获取
- 移除stream_id参数（新API不需要）
- 支持命令行参数指定音频块大小
- 支持多种块大小的对比测试
"""

import argparse
import asyncio
import os
import time
import wave
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Dict, List

import cascade


class ChunkSizeTestResult:
    """音频块大小测试结果"""
    
    def __init__(self, chunk_size: int):
        self.chunk_size = chunk_size
        self.segment_count = 0
        self.frame_count = 0
        self.total_processing_time = 0.0
        self.average_processing_time = 0.0
        self.total_chunks_processed = 0
        self.start_time = 0.0
        self.end_time = 0.0
        
    @property
    def total_test_time(self) -> float:
        """总测试时间（秒）"""
        return self.end_time - self.start_time
        
    @property
    def throughput_chunks_per_second(self) -> float:
        """吞吐量（块/秒）"""
        if self.total_test_time > 0:
            return self.total_chunks_processed / self.total_test_time
        return 0.0


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


async def test_stream_vad_processing(audio_file: str, chunk_size: int = 4096, save_segments: bool = True) -> ChunkSizeTestResult:
    """
    测试流式VAD处理
    
    Args:
        audio_file: 音频文件路径
        chunk_size: 音频块大小（字节）
        save_segments: 是否保存语音段文件
        
    Returns:
        ChunkSizeTestResult: 测试结果
    """
    print(f"🎯 开始流式VAD测试: {audio_file} (块大小: {chunk_size})")

    if not os.path.exists(audio_file):
        print(f"❌ 音频文件不存在: {audio_file}")
        return ChunkSizeTestResult(chunk_size)

    # 创建测试结果对象
    test_result = ChunkSizeTestResult(chunk_size)
    test_result.start_time = time.time()

    # 创建输出目录（仅在保存语音段时）
    output_dir = None
    if save_segments:
        output_dir = Path(f"stream_speech_segments_{chunk_size}")
        output_dir.mkdir(exist_ok=True)

    try:
        # 创建配置
        config = cascade.Config(
            vad_threshold=0.5,
            min_silence_duration_ms=500,
            speech_pad_ms=300
        )
        
        # 使用StreamProcessor进行流式处理
        async with cascade.StreamProcessor(config) as processor:
            print("🚀 StreamProcessor 已启动")

            # 模拟音频流并处理
            audio_stream = simulate_audio_stream(audio_file, chunk_size=chunk_size)

            async for result in processor.process_stream(audio_stream):
                if result.is_speech_segment and result.segment:
                    test_result.segment_count += 1
                    segment = result.segment

                    start_ms = int(segment.start_timestamp_ms)
                    end_ms = int(segment.end_timestamp_ms)
                    duration_ms = int(segment.duration_ms)

                    print(f"\n🎤 语音段 {test_result.segment_count}: {start_ms}ms - {end_ms}ms (时长: {duration_ms}ms)")

                    # 保存语音段为WAV文件（如果需要）
                    if save_segments and output_dir:
                        output_file = output_dir / f"stream_speech_segment_{test_result.segment_count}_{start_ms}ms-{end_ms}ms.wav"
                        await save_audio_segment(segment.audio_data, output_file)
                        print(f"💾 已保存: {output_file}")

                elif result.frame:
                    # 单帧结果
                    test_result.frame_count += 1
                    frame = result.frame
                    if test_result.frame_count % 50 == 0:  # 每50帧打印一次
                        print(f"🔇 单帧 {test_result.frame_count}: {frame.timestamp_ms:.0f}ms", end="\r")

            # 获取处理统计
            stats = processor.get_stats()
            test_result.total_chunks_processed = stats.total_chunks_processed
            test_result.total_processing_time = stats.total_processing_time_ms
            test_result.average_processing_time = stats.average_processing_time_ms
            
            test_result.end_time = time.time()

            print("\n📊 处理统计:")
            print(f"   - 总结果: {test_result.segment_count + test_result.frame_count} 个")
            print(f"   - 语音段: {test_result.segment_count} 个")
            print(f"   - 单帧: {test_result.frame_count} 个")
            print(f"   - 总处理块: {test_result.total_chunks_processed}")
            print(f"   - 平均处理时间: {test_result.average_processing_time:.2f}ms")
            print(f"   - 总测试时间: {test_result.total_test_time:.2f}s")
            print(f"   - 吞吐量: {test_result.throughput_chunks_per_second:.2f} 块/秒")

    except Exception as e:
        print(f"❌ 流式处理过程中出错: {e}")
        test_result.end_time = time.time()
        return test_result

    print("\n✅ 流式VAD测试完成！")
    if save_segments and test_result.segment_count > 0 and output_dir:
        print(f"📁 语音段已保存到: {output_dir.absolute()}")
    
    return test_result


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


async def run_chunk_size_comparison(audio_file: str, chunk_sizes: List[int]) -> Dict[int, ChunkSizeTestResult]:
    """
    运行不同音频块大小的对比测试
    
    Args:
        audio_file: 音频文件路径
        chunk_sizes: 要测试的音频块大小列表
        
    Returns:
        Dict[int, ChunkSizeTestResult]: 测试结果字典，键为块大小
    """
    print("🔬 开始音频块大小对比测试")
    print("=" * 60)
    
    results = {}
    
    for i, chunk_size in enumerate(chunk_sizes):
        print(f"\n📊 测试 {i+1}/{len(chunk_sizes)}: 块大小 {chunk_size} 字节")
        print("-" * 40)
        
        # 运行测试（除了第一个测试，其他不保存语音段文件以节省空间）
        save_segments = (i == 0)
        result = await test_stream_vad_processing(audio_file, chunk_size, save_segments)
        results[chunk_size] = result
        
        # 简短的结果摘要
        print(f"✅ 完成: {result.segment_count}段, {result.total_chunks_processed}块, "
              f"{result.average_processing_time:.2f}ms/块, {result.throughput_chunks_per_second:.1f}块/秒")
    
    return results


def print_comparison_results(results: Dict[int, ChunkSizeTestResult]):
    """
    打印对比测试结果
    
    Args:
        results: 测试结果字典
    """
    print("\n" + "=" * 80)
    print("📈 音频块大小对比测试结果")
    print("=" * 80)
    
    # 表头
    print(f"{'块大小(字节)':<12} {'语音段':<8} {'总块数':<8} {'平均处理时间(ms)':<16} {'吞吐量(块/秒)':<14} {'总测试时间(s)':<14}")
    print("-" * 80)
    
    # 按块大小排序显示结果
    for chunk_size in sorted(results.keys()):
        result = results[chunk_size]
        print(f"{chunk_size:<12} {result.segment_count:<8} {result.total_chunks_processed:<8} "
              f"{result.average_processing_time:<16.2f} {result.throughput_chunks_per_second:<14.1f} "
              f"{result.total_test_time:<14.2f}")
    
    # 性能分析
    print("\n📊 性能分析:")
    
    # 找出最快和最慢的配置
    fastest_chunk_size = min(results.keys(), key=lambda x: results[x].average_processing_time)
    slowest_chunk_size = max(results.keys(), key=lambda x: results[x].average_processing_time)
    
    fastest_result = results[fastest_chunk_size]
    slowest_result = results[slowest_chunk_size]
    
    print(f"🚀 最快处理: {fastest_chunk_size}字节 ({fastest_result.average_processing_time:.2f}ms/块)")
    print(f"🐌 最慢处理: {slowest_chunk_size}字节 ({slowest_result.average_processing_time:.2f}ms/块)")
    
    # 吞吐量对比
    highest_throughput_chunk_size = max(results.keys(), key=lambda x: results[x].throughput_chunks_per_second)
    lowest_throughput_chunk_size = min(results.keys(), key=lambda x: results[x].throughput_chunks_per_second)
    
    print(f"📈 最高吞吐量: {highest_throughput_chunk_size}字节 ({results[highest_throughput_chunk_size].throughput_chunks_per_second:.1f}块/秒)")
    print(f"📉 最低吞吐量: {lowest_throughput_chunk_size}字节 ({results[lowest_throughput_chunk_size].throughput_chunks_per_second:.1f}块/秒)")
    
    # 建议
    print(f"\n💡 建议:")
    if fastest_chunk_size == highest_throughput_chunk_size:
        print(f"   推荐使用 {fastest_chunk_size} 字节块大小，兼顾处理速度和吞吐量")
    else:
        print(f"   低延迟场景推荐: {fastest_chunk_size} 字节")
        print(f"   高吞吐量场景推荐: {highest_throughput_chunk_size} 字节")


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Cascade 流式VAD测试 - 支持不同音频块大小测试")
    
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=4096,
        help="音频块大小（字节），默认4096"
    )
    
    parser.add_argument(
        "--compare",
        action="store_true",
        help="运行多种块大小的对比测试"
    )
    
    parser.add_argument(
        "--chunk-sizes",
        type=str,
        default="1024,2048,4096,8192,16384",
        help="对比测试的块大小列表，用逗号分隔，默认: 1024,2048,4096,8192,16384"
    )
    
    parser.add_argument(
        "--audio-file",
        type=str,
        help="指定音频文件路径"
    )
    
    return parser.parse_args()


async def main():
    """主函数"""
    args = parse_arguments()
    
    print("🌊 Cascade 流式VAD 测试")
    print("=" * 50)

    # 确定音频文件
    audio_file = args.audio_file
    if not audio_file:
        # 测试文件列表
        test_files = [
            "我现在开始录音，理论上会有两个文件.wav"
        ]
        
        # 寻找可用的音频文件
        for file_path in test_files:
            if os.path.exists(file_path):
                audio_file = file_path
                break

    if not audio_file or not os.path.exists(audio_file):
        print("❌ 未找到可用的音频文件")
        if not args.audio_file:
            print("请将音频文件放在项目根目录，支持的文件名:")
            test_files = [
                "我现在开始录音，理论上会有两个文件.wav"
            ]
            for file_path in test_files:
                print(f"  - {file_path}")
        print("\n💡 提示: 音频文件应为WAV格式，建议16kHz采样率")
        print("💡 或者使用 --audio-file 参数指定音频文件路径")
        return

    if args.compare:
        # 对比测试模式
        try:
            chunk_sizes = [int(x.strip()) for x in args.chunk_sizes.split(",")]
        except ValueError:
            print("❌ 块大小列表格式错误，请使用逗号分隔的整数")
            return
            
        print(f"🎯 对比测试模式: {chunk_sizes}")
        results = await run_chunk_size_comparison(audio_file, chunk_sizes)
        print_comparison_results(results)
    else:
        # 单一测试模式
        print(f"🎯 单一测试模式: 块大小 {args.chunk_size} 字节")
        await test_stream_vad_processing(audio_file, args.chunk_size, save_segments=True)


if __name__ == "__main__":
    asyncio.run(main())
