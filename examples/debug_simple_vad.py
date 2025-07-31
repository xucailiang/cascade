#!/usr/bin/env python3
"""
简化的VAD调试脚本
用于逐步排查并行处理脚本卡住的问题
"""

import asyncio
import time
import sys
import traceback
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from cascade.types import AudioConfig, VADConfig, AudioFormat
from cascade.backends import create_vad_backend

print("🔍 Cascade VAD调试脚本")
print("=" * 50)

async def test_basic_vad():
    """测试基本VAD功能"""
    try:
        print("\n1️⃣ 测试基本VAD配置...")
        
        # 最简配置
        audio_config = AudioConfig(
            sample_rate=16000,
            channels=1,
            format=AudioFormat.WAV
        )
        
        vad_config = VADConfig(
            backend="silero",
            threshold=0.5,
            chunk_duration_ms=512,
            workers=1  # 单线程测试
        )
        
        print("✅ 配置创建成功")
        
        print("\n2️⃣ 测试VAD后端创建...")
        backend = create_vad_backend(vad_config)
        print("✅ VAD后端创建成功")
        
        print("\n3️⃣ 测试VAD后端初始化...")
        start_time = time.time()
        await backend.initialize()
        init_time = time.time() - start_time
        print(f"✅ VAD后端初始化成功: {init_time:.3f}秒")
        
        print("\n4️⃣ 测试虚拟音频处理...")
        # 创建1秒的虚拟音频（16kHz采样率）
        dummy_audio = np.random.random(16000).astype(np.float32) * 0.1
        
        # 直接调用后端处理
        chunk_size = vad_config.get_chunk_samples(audio_config.sample_rate)
        print(f"   - 块大小: {chunk_size} 样本")
        
        if len(dummy_audio) >= chunk_size:
            audio_chunk = dummy_audio[:chunk_size]
            
            from cascade.types import AudioChunk
            chunk = AudioChunk(
                data=audio_chunk,
                sequence_number=0,
                start_frame=0,
                chunk_size=chunk_size,
                overlap_size=0,
                timestamp_ms=0.0,
                sample_rate=audio_config.sample_rate
            )
            
            print("   - 处理音频块...")
            start_time = time.time()
            result = backend.process_chunk(chunk)
            process_time = time.time() - start_time
            
            print(f"✅ 音频处理成功: {process_time:.3f}秒")
            print(f"   - VAD结果: is_speech={result.is_speech}, confidence={result.confidence:.3f}")
        
        print("\n5️⃣ 清理资源...")
        await backend.close()
        print("✅ 资源清理完成")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        traceback.print_exc()
        return False

async def test_thread_pool():
    """测试线程池功能"""
    try:
        print("\n6️⃣ 测试线程池创建...")
        
        from cascade._internal.thread_pool import VADThreadPool, VADThreadPoolConfig
        from cascade.types import AudioConfig, VADConfig
        
        audio_config = AudioConfig(
            sample_rate=16000,
            channels=1,
            format=AudioFormat.WAV
        )
        
        vad_config = VADConfig(
            backend="silero",
            threshold=0.5,
            workers=2  # 2线程测试
        )
        
        pool_config = VADThreadPoolConfig(
            max_workers=2,
            warmup_enabled=False  # 禁用预热避免卡住
        )
        
        thread_pool = VADThreadPool(vad_config, audio_config, pool_config)
        print("✅ 线程池创建成功")
        
        print("\n7️⃣ 测试线程池初始化...")
        backend_template = create_vad_backend(vad_config)
        await backend_template.initialize()
        
        start_time = time.time()
        await thread_pool.initialize(backend_template)
        init_time = time.time() - start_time
        print(f"✅ 线程池初始化成功: {init_time:.3f}秒")
        
        print("\n8️⃣ 测试并行处理...")
        dummy_audio = np.random.random(8192).astype(np.float32) * 0.1
        chunk_size = vad_config.get_chunk_samples(audio_config.sample_rate)
        
        from cascade.types import AudioChunk
        chunk = AudioChunk(
            data=dummy_audio[:chunk_size],
            sequence_number=0,
            start_frame=0,
            chunk_size=chunk_size,
            overlap_size=0,
            timestamp_ms=0.0,
            sample_rate=audio_config.sample_rate
        )
        
        start_time = time.time()
        result = await thread_pool.process_chunk_async(chunk)
        process_time = time.time() - start_time
        
        print(f"✅ 并行处理成功: {process_time:.3f}秒")
        print(f"   - VAD结果: is_speech={result.is_speech}, confidence={result.confidence:.3f}")
        
        print("\n9️⃣ 清理线程池...")
        await thread_pool.close()
        await backend_template.close()
        print("✅ 线程池清理完成")
        
        return True
        
    except Exception as e:
        print(f"❌ 线程池测试失败: {e}")
        traceback.print_exc()
        return False

async def main():
    """主调试流程"""
    print("开始Cascade VAD调试...")
    
    success_count = 0
    total_tests = 2
    
    # 测试1：基本VAD功能
    print("\n" + "="*60)
    print("测试1: 基本VAD功能")
    print("="*60)
    if await test_basic_vad():
        success_count += 1
        print("✅ 基本VAD功能测试通过")
    else:
        print("❌ 基本VAD功能测试失败")
    
    # 测试2：线程池功能  
    print("\n" + "="*60)
    print("测试2: 线程池功能")
    print("="*60)
    if await test_thread_pool():
        success_count += 1
        print("✅ 线程池功能测试通过")
    else:
        print("❌ 线程池功能测试失败")
    
    print("\n" + "="*60)
    print("调试结果总结")
    print("="*60)
    print(f"通过测试: {success_count}/{total_tests}")
    
    if success_count == total_tests:
        print("🎉 所有测试通过！可以进行完整的并行处理测试")
    else:
        print("⚠️  部分测试失败，需要修复问题后再进行并行处理")
    
    return success_count == total_tests

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⚠️  调试被用户中断")
    except Exception as e:
        print(f"\n❌ 调试脚本异常: {e}")
        traceback.print_exc()