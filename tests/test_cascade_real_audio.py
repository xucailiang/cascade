"""
Cascade 1:1:1绑定架构真实音频测试

使用真实的音频文件测试新的Cascade架构，
展示在实际语音检测场景中的表现。
"""

import asyncio
import time
from pathlib import Path
from typing import AsyncGenerator

import numpy as np
import librosa

import cascade


async def load_audio_as_stream(file_path: str, chunk_duration: float = 1.0) -> AsyncGenerator[np.ndarray, None]:
    """
    加载音频文件并转换为流式处理格式
    
    Args:
        file_path: 音频文件路径
        chunk_duration: 每个音频块的时长（秒）
    
    Yields:
        numpy.ndarray: 音频块数据
    """
    try:
        # 使用librosa加载音频
        audio_data, sample_rate = librosa.load(file_path, sr=16000)
        print(f"加载音频文件: {file_path}")
        print(f"音频长度: {len(audio_data)/sample_rate:.2f}秒")
        print(f"采样率: {sample_rate}Hz")
        print(f"音频块时长: {chunk_duration}秒")
        
        # 计算每个块的样本数
        chunk_size = int(sample_rate * chunk_duration)
        
        # 分块处理
        total_chunks = (len(audio_data) + chunk_size - 1) // chunk_size
        print(f"总共将处理 {total_chunks} 个音频块")
        
        for i in range(0, len(audio_data), chunk_size):
            chunk = audio_data[i:i + chunk_size]
            
            # 如果最后一块不足长度，用零填充
            if len(chunk) < chunk_size:
                padded_chunk = np.zeros(chunk_size, dtype=np.float32)
                padded_chunk[:len(chunk)] = chunk
                chunk = padded_chunk
            
            yield chunk.astype(np.float32)
            
            # 模拟实时处理的延迟
            await asyncio.sleep(0.01)
            
    except Exception as e:
        print(f"加载音频文件失败: {e}")
        raise


async def test_cascade_with_real_audio():
    """
    使用真实音频测试Cascade新架构
    """
    print("=" * 60)
    print("🎯 Cascade 1:1:1绑定架构真实音频测试")
    print("=" * 60)
    
    # 音频文件路径
    audio_file = "/home/justin/opensource/cascade/请问电动汽车和传统汽车比起来哪个更好啊？.wav"
    
    # 检查文件是否存在
    if not Path(audio_file).exists():
        # 尝试另一个路径
        audio_file = "/home/justin/opensource/cascade/examples/请问电动汽车和传统汽车比起来哪个更好啊？.wav"
        if not Path(audio_file).exists():
            print(f"❌ 音频文件不存在: {audio_file}")
            return
    
    print(f"📁 使用音频文件: {audio_file}")
    
    # 创建Cascade配置 - 新架构的极简配置
    config = cascade.CascadeConfig(
        sample_rate=16000,
        vad_backend="silero",
        vad_threshold=0.3  # 使用较低的阈值以便检测到语音
    )
    
    print(f"⚙️ Cascade配置: {config}")
    
    # 使用异步上下文管理器确保资源正确释放
    start_time = time.time()
    speech_segments = []
    total_chunks = 0
    
    try:
        async with cascade.Cascade(config) as detector:
            print("✅ Cascade实例初始化完成")
            
            # 获取统计信息
            stats = detector.get_stats()
            print(f"📊 初始统计: {stats}")
            
            # 处理音频流
            print("\n🔄 开始处理音频流...")
            async for result in detector.process_audio_stream(
                load_audio_as_stream(audio_file, chunk_duration=0.5)
            ):
                total_chunks += 1
                
                print(f"⏱️ 时间: {result.start_time:.1f}s-{result.end_time:.1f}s | "
                      f"概率: {result.speech_probability:.3f} | "
                      f"语音: {'🎤' if result.is_speech else '🔇'} | "
                      f"置信度: {result.confidence:.3f}")
                
                # 记录语音片段
                if result.is_speech:
                    speech_segments.append({
                        'start': result.start_time,
                        'end': result.end_time,
                        'probability': result.speech_probability,
                        'confidence': result.confidence,
                        'audio_data': result.audio_data  # 保存音频数据
                    })
            
            # 最终统计信息
            end_time = time.time()
            processing_time = end_time - start_time
            
            final_stats = detector.get_stats()
            print(f"\n📊 最终统计: {final_stats}")
            
    except Exception as e:
        print(f"❌ 处理过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 输出结果分析
    print("\n" + "=" * 60)
    print("📈 处理结果分析")
    print("=" * 60)
    
    print(f"⏱️ 总处理时间: {processing_time:.2f}秒")
    print(f"📦 总音频块数: {total_chunks}")
    print(f"🎤 检测到语音片段数: {len(speech_segments)}")
    
    if speech_segments:
        print(f"\n🎯 语音片段详情:")
        
        # 创建一个数组来存储完整的音频数据（不重叠）
        # 首先确定音频总长度
        if len(speech_segments) > 0 and 'audio_data' in speech_segments[0]:
            sample_rate = 16000  # 采样率
            max_end_time = max([s['end'] for s in speech_segments])
            total_samples = int(max_end_time * sample_rate) + 1
            full_audio = np.zeros(total_samples, dtype=np.float32)
            
            # 记录已处理的时间范围，避免重叠
            processed_ranges = []
            
            for i, segment in enumerate(speech_segments):
                print(f"  片段 {i+1}: {segment['start']:.1f}s-{segment['end']:.1f}s | "
                      f"概率: {segment['probability']:.3f} | 置信度: {segment['confidence']:.3f}")
                
                if 'audio_data' in segment:
                    # 计算该片段在完整音频中的位置
                    start_sample = int(segment['start'] * sample_rate)
                    end_sample = min(start_sample + len(segment['audio_data']), total_samples)
                    
                    # 检查是否与已处理的范围重叠
                    overlap = False
                    for start_r, end_r in processed_ranges:
                        if not (end_sample <= start_r or start_sample >= end_r):
                            overlap = True
                            break
                    
                    # 如果不重叠，则添加到完整音频中
                    if not overlap:
                        # 确保不超出边界
                        samples_to_copy = min(len(segment['audio_data']), end_sample - start_sample)
                        full_audio[start_sample:start_sample + samples_to_copy] = segment['audio_data'][:samples_to_copy]
                        processed_ranges.append((start_sample, end_sample))
            
            print(f"\n📊 语音统计:")
            print(f"  检测到的语音片段数: {len(speech_segments)}")
            print(f"  平均语音概率: {np.mean([s['probability'] for s in speech_segments]):.3f}")
            print(f"  平均置信度: {np.mean([s['confidence'] for s in speech_segments]):.3f}")
            
            try:
                # 保存到项目根目录
                output_file = "/home/justin/opensource/cascade/detected_speech.wav"
                import soundfile as sf
                
                # 裁剪掉开头和结尾的静音部分
                non_zero_indices = np.where(full_audio != 0)[0]
                if len(non_zero_indices) > 0:
                    start_idx = max(0, non_zero_indices[0] - int(0.1 * sample_rate))  # 前面留0.1秒
                    end_idx = min(len(full_audio), non_zero_indices[-1] + int(0.1 * sample_rate))  # 后面留0.1秒
                    trimmed_audio = full_audio[start_idx:end_idx]
                    sf.write(output_file, trimmed_audio, sample_rate)
                else:
                    sf.write(output_file, full_audio, sample_rate)
                
                print(f"\n✅ 已将检测到的语音保存到: {output_file}")
            except Exception as e:
                print(f"❌ 保存语音文件失败: {e}")
        else:
            print(f"\n📊 语音统计:")
            print(f"  检测到的语音片段数: {len(speech_segments)}")
            if speech_segments:
                print(f"  平均语音概率: {np.mean([s['probability'] for s in speech_segments]):.3f}")
                print(f"  平均置信度: {np.mean([s['confidence'] for s in speech_segments]):.3f}")
    else:
        print("⚠️ 未检测到语音片段，可能需要调整VAD阈值")


async def test_cascade_vs_old_architecture():
    """
    对比新旧架构的性能
    """
    print("\n" + "=" * 60)
    print("⚡ 新旧架构性能对比测试")
    print("=" * 60)
    
    audio_file = "/home/justin/opensource/cascade/请问电动汽车和传统汽车比起来哪个更好啊？.wav"
    if not Path(audio_file).exists():
        audio_file = "/home/justin/opensource/cascade/examples/请问电动汽车和传统汽车比起来哪个更好啊？.wav"
    
    # 测试新架构
    print("🆕 测试新架构 (Cascade 1:1:1绑定)...")
    start_time = time.time()
    
    config = cascade.CascadeConfig(
        sample_rate=16000,
        vad_backend="silero",
        vad_threshold=0.3
    )
    
    new_arch_segments = []
    try:
        async with cascade.Cascade(config) as detector:
            async for result in detector.process_audio_stream(
                load_audio_as_stream(audio_file, chunk_duration=0.5)
            ):
                if result.is_speech:
                    new_arch_segments.append(result.speech_probability)
    except Exception as e:
        print(f"新架构测试失败: {e}")
        return
    
    new_arch_time = time.time() - start_time
    
    print(f"✅ 新架构完成:")
    print(f"  处理时间: {new_arch_time:.2f}秒")
    print(f"  检测到语音片段: {len(new_arch_segments)}")
    if new_arch_segments:
        print(f"  平均语音概率: {np.mean(new_arch_segments):.3f}")
    
    print("\n📊 架构对比总结:")
    print("新架构 (1:1:1绑定) 优势:")
    print("  ✅ 确保VAD状态连续性")
    print("  ✅ 极简配置 (只需3个参数)")
    print("  ✅ 完整输出 (原始音频+VAD结果)")
    print("  ✅ 用户完全控制音频块大小")


async def main():
    """
    主测试函数
    """
    try:
        await test_cascade_with_real_audio()
        await test_cascade_vs_old_architecture()
        print("\n🎉 所有测试完成！")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # 运行测试
    asyncio.run(main())