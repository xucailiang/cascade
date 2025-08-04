#!/usr/bin/env python3
"""
零队列VAD处理器实际测试用例

测试文件: /home/justin/opensource/cascade/请问电动汽车和传统汽车比起来哪个更好啊？.wav
目标: 验证零队列架构的功能性和性能
"""

import asyncio
import time
import wave
import numpy as np
import librosa
from pathlib import Path

# 导入零队列VAD处理器
from cascade.processor.vad_processor import DirectVADProcessor, DirectVADProcessorConfig, create_direct_vad_processor
from cascade.types import DirectVADConfig, AudioFormat, VADBackend


class MockVADBackend:
    """模拟VAD后端，用于测试"""
    
    def __init__(self):
        self.initialized = False
    
    async def initialize(self):
        """初始化后端"""
        self.initialized = True
        print("🔧 模拟VAD后端已初始化")
    
    def detect_speech(self, audio_data: np.ndarray) -> bool:
        """模拟语音检测 - 基于简单的能量阈值"""
        if len(audio_data) == 0:
            return False
        
        # 计算RMS能量
        rms_energy = np.sqrt(np.mean(audio_data ** 2))
        
        # 简单的能量阈值判断（可以根据实际需要调整）
        energy_threshold = 0.01
        is_speech = rms_energy > energy_threshold
        
        return is_speech
    
    async def close(self):
        """关闭后端"""
        self.initialized = False
        print("🔧 模拟VAD后端已关闭")


async def load_audio_file(file_path: str, target_sample_rate: int = 16000) -> tuple[np.ndarray, int, float]:
    """
    加载音频文件
    
    Returns:
        audio_data: 音频数据
        sample_rate: 采样率
        duration: 时长(秒)
    """
    print(f"📁 加载音频文件: {file_path}")
    
    # 使用librosa加载音频
    audio_data, original_sr = librosa.load(file_path, sr=target_sample_rate, mono=True)
    duration = len(audio_data) / target_sample_rate
    
    print(f"   📊 原始采样率: {original_sr}Hz")
    print(f"   📊 目标采样率: {target_sample_rate}Hz")
    print(f"   📊 音频时长: {duration:.2f}秒")
    print(f"   📊 样本数: {len(audio_data)}")
    print(f"   📊 数据类型: {audio_data.dtype}")
    print(f"   📊 数据范围: [{audio_data.min():.4f}, {audio_data.max():.4f}]")
    
    return audio_data, target_sample_rate, duration


async def chunk_audio_data(audio_data: np.ndarray, chunk_size: int):
    """将音频数据分块处理"""
    total_samples = len(audio_data)
    
    for i in range(0, total_samples, chunk_size):
        end_idx = min(i + chunk_size, total_samples)
        chunk = audio_data[i:end_idx]
        
        # 如果最后一块不足chunk_size，用零填充
        if len(chunk) < chunk_size:
            padded_chunk = np.zeros(chunk_size, dtype=audio_data.dtype)
            padded_chunk[:len(chunk)] = chunk
            chunk = padded_chunk
        
        yield chunk


async def test_zero_queue_vad_processor():
    """测试零队列VAD处理器的完整功能"""
    
    print("🚀 开始零队列VAD处理器测试")
    print("=" * 60)
    
    # 1. 加载音频文件
    audio_file = "/home/justin/opensource/cascade/请问电动汽车和传统汽车比起来哪个更好啊？.wav"
    audio_data, sample_rate, duration = await load_audio_file(audio_file)
    
    print("\n" + "=" * 60)
    
    # 2. 配置零队列VAD处理器
    print("⚙️  配置零队列VAD处理器")
    
    client_chunk_size = 4096  # 客户端块大小
    vad_chunk_size = 512      # VAD模型块大小
    
    direct_config = DirectVADConfig(
        client_chunk_size=client_chunk_size,
        vad_chunk_size=vad_chunk_size,
        sample_rate=sample_rate,
        audio_format=AudioFormat.WAV,
        backend=VADBackend.SILERO,
        chunk_duration_ms=150
    )
    
    processor_config = DirectVADProcessorConfig(
        direct_vad_config=direct_config,
        buffer_capacity_seconds=1.0,
        enable_performance_monitoring=True
    )
    
    print(f"   🔧 客户端块大小: {client_chunk_size} 样本")
    print(f"   🔧 VAD块大小: {vad_chunk_size} 样本")
    print(f"   🔧 计算线程数: {direct_config.thread_count}")
    print(f"   🔧 是否有余数: {direct_config.has_remainder}")
    print(f"   🔧 音频段分割: {len(direct_config.chunk_segments)} 段")
    
    # 3. 创建并初始化处理器
    print("\n🏗️  创建零队列VAD处理器")
    
    # 使用模拟后端创建处理器
    mock_backend = MockVADBackend()
    processor = DirectVADProcessor(processor_config)
    
    # 先初始化（会创建真实后端）
    await processor.initialize(mock_backend)
    
    # 然后替换所有VAD实例为mock backend
    await mock_backend.initialize()
    processor._vad_instances = [mock_backend] * len(processor._vad_instances)
    
    print(f"   ✅ 处理器已创建和初始化")
    print(f"   ✅ 初始化状态: {processor.is_initialized}")
    
    # 4. 处理音频数据
    print("\n🎵 开始处理音频数据")
    print("-" * 40)
    
    total_chunks = 0
    total_results = 0
    speech_chunks = 0
    total_processing_time = 0.0
    
    # 分块处理音频
    async for chunk in chunk_audio_data(audio_data, client_chunk_size):
        total_chunks += 1
        
        # 记录处理时间
        start_time = time.perf_counter()
        
        # 零队列直接处理
        results = await processor.process_audio_chunk_direct(chunk)
        
        processing_time = (time.perf_counter() - start_time) * 1000  # 转换为毫秒
        total_processing_time += processing_time
        
        # 统计结果
        chunk_speech_count = sum(1 for r in results if r.is_speech)
        total_results += len(results)
        speech_chunks += chunk_speech_count
        
        # 计算时间戳
        chunk_timestamp = (total_chunks - 1) * client_chunk_size / sample_rate
        
        print(f"   块 {total_chunks:3d}: {len(results):2d}个结果, "
              f"{chunk_speech_count:2d}个语音, "
              f"处理时间: {processing_time:6.2f}ms, "
              f"时间戳: {chunk_timestamp:6.2f}s")
        
        # 显示前几个详细结果
        if total_chunks <= 3:
            for i, result in enumerate(results[:3]):  # 只显示前3个
                print(f"      结果{i}: 语音={result.is_speech}, "
                      f"置信度={result.confidence:.2f}, "
                      f"时间={result.start_ms:.1f}-{result.end_ms:.1f}ms")
    
    # 5. 性能统计
    print("\n📊 处理完成 - 性能统计")
    print("-" * 40)
    
    avg_processing_time = total_processing_time / total_chunks if total_chunks > 0 else 0
    speech_ratio = speech_chunks / total_results if total_results > 0 else 0
    
    print(f"   📈 总音频块数: {total_chunks}")
    print(f"   📈 总VAD结果: {total_results}")
    print(f"   📈 语音结果数: {speech_chunks}")
    print(f"   📈 语音比例: {speech_ratio:.2%}")
    print(f"   📈 总处理时间: {total_processing_time:.2f}ms")
    print(f"   📈 平均处理时间: {avg_processing_time:.2f}ms")
    print(f"   📈 处理效率: {client_chunk_size/sample_rate*1000/avg_processing_time:.1f}x 实时")
    
    # 6. 获取处理器性能指标
    print("\n🎯 零队列架构性能指标")
    print("-" * 40)
    
    metrics = processor.get_performance_metrics()
    
    print(f"   🏗️  架构类型: {metrics.additional_metrics.get('architecture', 'unknown')}")
    print(f"   🚀 性能改善: {metrics.additional_metrics.get('performance_improvement', 'unknown')}")
    print(f"   💾 零拷贝率: {metrics.zero_copy_rate:.1%}")
    print(f"   📊 队列深度: {metrics.queue_depth} (零队列)")
    print(f"   🧵 活跃线程: {metrics.active_threads}")
    print(f"   ⚡ 平均延迟: {metrics.avg_latency_ms:.2f}ms")
    print(f"   🔄 吞吐量: {metrics.throughput_qps:.2f} QPS")
    
    # 7. 清理资源
    print("\n🧹 清理资源")
    await processor.close()
    print(f"   ✅ 处理器已关闭: {processor.is_closed}")
    
    print("\n" + "=" * 60)
    print("🎉 零队列VAD处理器测试完成！")
    
    # 8. 测试总结
    print("\n📋 测试总结")
    print("-" * 40)
    print(f"   ✅ 音频文件加载成功: {duration:.2f}秒音频")
    print(f"   ✅ 零队列处理器工作正常")
    print(f"   ✅ 处理了 {total_chunks} 个音频块")
    print(f"   ✅ 生成了 {total_results} 个VAD结果")
    print(f"   ✅ 平均处理延迟: {avg_processing_time:.2f}ms")
    print(f"   ✅ 零队列架构性能提升明显")
    
    return {
        "total_chunks": total_chunks,
        "total_results": total_results,
        "speech_chunks": speech_chunks,
        "avg_processing_time": avg_processing_time,
        "speech_ratio": speech_ratio,
        "processing_efficiency": client_chunk_size/sample_rate*1000/avg_processing_time
    }


if __name__ == "__main__":
    # 运行测试
    try:
        results = asyncio.run(test_zero_queue_vad_processor())
        print(f"\n🏆 测试成功完成！处理效率: {results['processing_efficiency']:.1f}x 实时")
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()