"""
Cascade 简化架构使用示例

展示简化后的1:1:1:1架构使用方式。
每个StreamProcessor实例拥有独立的VAD模型，无锁无竞争。
"""

import asyncio
import cascade


async def example_basic_usage():
    """基础使用示例 - 使用异步上下文管理器"""
    print("=" * 50)
    print("示例1: 基础使用 - 异步上下文管理器")
    print("=" * 50)
    
    # 1. 创建配置
    config = cascade.Config(
        vad_threshold=0.5,
        min_silence_duration_ms=500,
        speech_pad_ms=300
    )
    
    # 2. 使用异步上下文管理器（自动初始化和清理）
    async with cascade.StreamProcessor(config) as processor:
        # 3. 处理音频文件（如果你有音频文件）
        try:
            async for result in processor.process_file("test_audio.wav"):
                if result.is_speech_segment and result.segment:
                    segment = result.segment
                    print(f"✓ 检测到语音段: "
                          f"{segment.start_timestamp_ms:.0f}-{segment.end_timestamp_ms:.0f}ms "
                          f"({segment.duration_ms:.0f}ms, {segment.frame_count}帧)")
                elif result.frame:
                    frame = result.frame
                    print(f"  单帧: {frame.timestamp_ms:.0f}ms")
        except FileNotFoundError:
            print("测试音频文件不存在，跳过文件处理示例")
    
    print("\n处理器已自动清理\n")


async def example_manual_control():
    """手动控制示例 - 显式初始化和清理"""
    print("=" * 50)
    print("示例2: 手动控制")
    print("=" * 50)
    
    # 1. 创建处理器
    processor = cascade.StreamProcessor()
    
    try:
        # 2. 显式初始化
        await processor.initialize()
        print("✓ StreamProcessor初始化完成")
        
        # 3. 处理音频块（模拟）
        # 在实际应用中，audio_chunk来自WebSocket、文件流等
        print("\n模拟处理音频块...")
        
        # 获取统计信息
        stats = processor.get_stats()
        print(f"\n统计信息: {stats.summary()}")
        
    finally:
        # 4. 显式清理
        await processor.close()
        print("✓ StreamProcessor已清理")
    
    print()


async def example_stream_processing():
    """流式处理示例 - 处理音频流"""
    print("=" * 50)
    print("示例3: 流式处理")
    print("=" * 50)
    
    async def mock_audio_stream():
        """模拟音频流生成器"""
        # 模拟3个音频块
        for i in range(3):
            # 每个块1024字节 (512样本 * 2字节)
            yield b'\x00' * 1024
            await asyncio.sleep(0.1)
    
    config = cascade.Config(vad_threshold=0.5)
    
    async with cascade.StreamProcessor(config) as processor:
        print("开始处理音频流...")
        
        result_count = 0
        async for result in processor.process_stream(mock_audio_stream()):
            result_count += 1
            print(f"处理结果 #{result_count}: {result.result_type}")
        
        # 获取统计
        stats = processor.get_stats()
        print(f"\n处理完成! 总共处理 {stats.total_chunks_processed} 个块")
    
    print()


async def example_multiple_processors():
    """多处理器示例 - 展示1:1:1:1架构"""
    print("=" * 50)
    print("示例4: 多处理器并发 (1:1:1:1架构)")
    print("=" * 50)
    print("每个处理器拥有独立的VAD模型，完全隔离，无并发问题\n")
    
    async def simulate_client(client_id: int, processor: cascade.StreamProcessor):
        """模拟一个客户端连接"""
        print(f"客户端 {client_id}: 开始处理")
        
        # 模拟处理几个音频块
        for i in range(2):
            audio_data = b'\x00' * 1024
            results = await processor.process_chunk(audio_data)
            print(f"客户端 {client_id}: 处理块 {i+1}, 得到 {len(results)} 个结果")
            await asyncio.sleep(0.1)
        
        print(f"客户端 {client_id}: 处理完成")
    
    # 创建3个独立的处理器（模拟3个WebSocket连接）
    processors = []
    for i in range(3):
        proc = cascade.StreamProcessor()
        await proc.initialize()
        processors.append(proc)
        print(f"✓ 处理器 {i+1} 初始化完成（独立VAD模型）")
    
    print("\n并发处理...")
    try:
        # 并发处理（每个处理器在独立协程中）
        tasks = [
            simulate_client(i+1, proc)
            for i, proc in enumerate(processors)
        ]
        await asyncio.gather(*tasks)
        
    finally:
        # 清理所有处理器
        print("\n清理处理器...")
        for i, proc in enumerate(processors):
            await proc.close()
            print(f"✓ 处理器 {i+1} 已清理")
    
    print("\n所有处理器已清理，内存已释放\n")


async def example_websocket_pattern():
    """WebSocket使用模式示例"""
    print("=" * 50)
    print("示例5: WebSocket使用模式")
    print("=" * 50)
    
    class SessionManager:
        """会话管理器 - WebSocket服务器端使用"""
        
        def __init__(self):
            # 每个客户端ID对应一个独立的StreamProcessor
            self.processors: dict[str, cascade.StreamProcessor] = {}
        
        async def start_session(self, client_id: str):
            """为新连接创建处理器"""
            config = cascade.Config(vad_threshold=0.5)
            processor = cascade.StreamProcessor(config)
            await processor.initialize()
            self.processors[client_id] = processor
            print(f"✓ 会话 {client_id} 已创建")
        
        async def process_chunk(self, client_id: str, audio_data: bytes):
            """处理音频块"""
            processor = self.processors.get(client_id)
            if not processor:
                print(f"⚠ 会话 {client_id} 不存在")
                return
            
            results = await processor.process_chunk(audio_data)
            for result in results:
                if result.is_speech_segment and result.segment:
                    print(f"  {client_id}: 检测到语音段 "
                          f"{result.segment.duration_ms:.0f}ms")
        
        async def stop_session(self, client_id: str):
            """关闭会话"""
            processor = self.processors.pop(client_id, None)
            if processor:
                await processor.close()
                print(f"✓ 会话 {client_id} 已关闭")
    
    # 模拟WebSocket服务器使用
    manager = SessionManager()
    
    # 客户端1连接
    await manager.start_session("client_1")
    await manager.process_chunk("client_1", b'\x00' * 1024)
    
    # 客户端2连接
    await manager.start_session("client_2")
    await manager.process_chunk("client_2", b'\x00' * 1024)
    
    # 处理更多数据
    await manager.process_chunk("client_1", b'\x00' * 1024)
    
    # 客户端断开
    await manager.stop_session("client_1")
    await manager.stop_session("client_2")
    
    print()


async def main():
    """运行所有示例"""
    print("\n" + "=" * 50)
    print("Cascade 简化架构使用示例集")
    print("=" * 50 + "\n")
    
    # 运行所有示例
    await example_basic_usage()
    await example_manual_control()
    await example_stream_processing()
    await example_multiple_processors()
    await example_websocket_pattern()
    
    print("=" * 50)
    print("所有示例运行完成！")
    print("=" * 50)
    print("\n核心特性:")
    print("✓ 1:1:1:1架构 - 每个StreamProcessor拥有独立模型")
    print("✓ 无锁设计 - 完全独立，无并发冲突")
    print("✓ 异步处理 - 使用asyncio.to_thread执行VAD推理")
    print("✓ 简洁API - 易于理解和使用")
    print("✓ 自动资源管理 - 异步上下文管理器支持\n")


if __name__ == "__main__":
    # 运行示例
    asyncio.run(main())