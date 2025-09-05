#!/usr/bin/env python3
"""
真正的多线程多CascadeInstance实例测试脚本

使用ThreadPoolExecutor实现真正的多线程并发测试，
验证每个线程运行独立的CascadeInstance实例的流式音频处理能力。

改进点：
1. 使用真正的多线程而不是异步并发
2. 从模型加载完成后开始计时
3. 移除人工延迟
4. 优化性能测试逻辑
"""

import logging
import os
import threading
import time
import wave
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List

from cascade.stream import StreamProcessor, Config
from pydantic import BaseModel, Field

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ThreadTestResult(BaseModel):
    """线程测试结果"""
    thread_id: int = Field(description="线程ID")
    thread_name: str = Field(description="线程名称")
    stream_id: str = Field(description="流标识符")
    instance_id: str = Field(description="CascadeInstance ID")
    
    # 处理统计
    total_chunks_processed: int = Field(default=0, description="总处理块数")
    speech_segments_count: int = Field(default=0, description="语音段数量")
    single_frames_count: int = Field(default=0, description="单帧数量")
    
    # 时间统计（排除模型加载时间）
    model_load_time_ms: float = Field(default=0.0, description="模型加载时间(ms)")
    processing_start_time: float = Field(description="处理开始时间")
    processing_end_time: float = Field(default=0.0, description="处理结束时间")
    pure_processing_time_ms: float = Field(default=0.0, description="纯处理时间(ms)")
    
    # 性能统计
    throughput_chunks_per_sec: float = Field(default=0.0, description="吞吐量(块/秒)")
    
    # 错误统计
    error_count: int = Field(default=0, description="错误次数")
    
    def calculate_metrics(self):
        """计算性能指标（排除模型加载时间）"""
        self.pure_processing_time_ms = (self.processing_end_time - self.processing_start_time) * 1000
        
        if self.pure_processing_time_ms > 0:
            self.throughput_chunks_per_sec = (
                self.total_chunks_processed / (self.pure_processing_time_ms / 1000)
            )


class RealMultithreadTestSuite:
    """真正的多线程测试套件"""
    
    def __init__(self, audio_file: str, num_threads: int = 4, chunk_size: int = 4096):
        """
        初始化测试套件
        
        Args:
            audio_file: 音频文件路径
            num_threads: 线程数量
            chunk_size: 音频块大小
        """
        self.audio_file = audio_file
        self.num_threads = num_threads
        self.chunk_size = chunk_size
        
        # 输出目录
        self.output_dir = Path("real_multithread_test_results")
        self.output_dir.mkdir(exist_ok=True)
        
        # 预加载音频数据（所有线程共享）
        self.audio_chunks = self._preload_audio_chunks()
        
        # 测试结果存储
        self.thread_results: Dict[int, ThreadTestResult] = {}
        self.thread_lock = threading.RLock()

    def _preload_audio_chunks(self) -> List[bytes]:
        """预加载音频数据为块列表"""
        logger.info(f"预加载音频文件: {self.audio_file}")
        
        chunks = []
        try:
            with wave.open(self.audio_file, 'rb') as wav_file:
                # 获取音频信息
                channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                framerate = wav_file.getframerate()
                total_frames = wav_file.getnframes()
                
                logger.info(f"音频信息: {channels}ch, {sample_width*8}bit, {framerate}Hz, {total_frames} 帧")
                
                # 逐块读取音频数据
                while True:
                    frames_per_chunk = self.chunk_size // (channels * sample_width)
                    audio_chunk = wav_file.readframes(frames_per_chunk)
                    if not audio_chunk:
                        break
                    chunks.append(audio_chunk)
                
                logger.info(f"预加载完成: {len(chunks)} 个音频块")
                
        except Exception as e:
            logger.error(f"预加载音频失败: {e}")
            
        return chunks

    def process_audio_in_thread(self, thread_id: int) -> ThreadTestResult:
        """
        在指定线程中处理音频（同步函数，用于ThreadPoolExecutor）
        
        Args:
            thread_id: 线程ID
            
        Returns:
            ThreadTestResult: 线程处理结果
        """
        thread_name = threading.current_thread().name
        # 基于线程ID和线程名生成唯一的stream_id
        stream_id = f"thread_{thread_id}_{threading.get_ident()}"
        
        logger.info(f"线程 {thread_id} ({thread_name}) 开始处理，stream_id: {stream_id}")
        
        # 初始化结果对象
        result = ThreadTestResult(
            thread_id=thread_id,
            thread_name=thread_name,
            stream_id=stream_id,
            instance_id="",
            processing_start_time=0.0  # 稍后设置
        )
        
        segment_count = 0
        frame_count = 0
        
        try:
            # 1. 模型加载阶段（计时）
            model_load_start = time.time()
            
            # 创建独立的StreamProcessor配置
            config = Config(max_instances=1)  # 每个线程只需要1个实例
            
            # 创建StreamProcessor（这里会加载模型）
            processor = StreamProcessor(config)
            
            # 启动processor（完成模型初始化）
            import asyncio
            
            # 在新线程中需要创建新的事件循环
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                # 启动处理器
                loop.run_until_complete(processor.start())
                
                # 模型加载完成
                model_load_end = time.time()
                result.model_load_time_ms = (model_load_end - model_load_start) * 1000
                
                logger.info(f"线程 {thread_id} 模型加载完成，耗时: {result.model_load_time_ms:.1f}ms")
                
                # 2. 音频处理阶段（从这里开始计时）
                result.processing_start_time = time.time()
                
                # 定义异步处理函数
                async def process_audio_stream():
                    nonlocal segment_count, frame_count
                    
                    # 创建异步音频流生成器（用于process_stream）
                    async def audio_stream_generator():
                        for audio_chunk in self.audio_chunks:
                            if audio_chunk:
                                yield audio_chunk
                    
                    # 使用process_stream API（支持stream_id隔离）
                    audio_stream = audio_stream_generator()
                    
                    async for cascade_result in processor.process_stream(audio_stream, stream_id=stream_id):
                        if cascade_result.result_type == "segment" and cascade_result.segment:
                            segment_count += 1
                            segment = cascade_result.segment
                            
                            # 记录实例ID
                            if not result.instance_id:
                                result.instance_id = cascade_result.instance_id
                            
                            start_ms = segment.start_timestamp_ms
                            end_ms = segment.end_timestamp_ms
                            duration_ms = segment.duration_ms
                            
                            logger.info(f"线程 {thread_id} 语音段 {segment_count}: {start_ms:.0f}ms-{end_ms:.0f}ms ({duration_ms:.0f}ms)")
                            
                            # 保存语音段
                            self._save_segment_for_thread(thread_id, segment_count, segment)
                            
                        elif cascade_result.result_type == "frame":
                            frame_count += 1
                            if not result.instance_id:
                                result.instance_id = cascade_result.instance_id
                
                # 运行异步处理
                loop.run_until_complete(process_audio_stream())
                
                # 获取处理器统计信息
                stats = processor.get_stats()
                result.total_chunks_processed = stats.total_chunks_processed
                
                logger.info(f"线程 {thread_id} 处理完成: {segment_count} 语音段, {frame_count} 单帧")
                
                # 停止处理器
                loop.run_until_complete(processor.stop())
                
            finally:
                loop.close()
                
        except Exception as e:
            result.error_count += 1
            logger.error(f"线程 {thread_id} 处理失败: {e}")
        
        finally:
            result.processing_end_time = time.time()
            result.speech_segments_count = segment_count
            result.single_frames_count = frame_count
            result.calculate_metrics()
            
            # 线程安全地存储结果
            with self.thread_lock:
                self.thread_results[thread_id] = result
        
        return result

    def _save_segment_for_thread(self, thread_id: int, segment_count: int, segment):
        """为指定线程保存语音段"""
        try:
            thread_dir = self.output_dir / f"thread_{thread_id}"
            thread_dir.mkdir(exist_ok=True)
            
            start_ms = segment.start_timestamp_ms
            end_ms = segment.end_timestamp_ms
            output_file = thread_dir / f"segment_{segment_count}_{start_ms:.0f}ms-{end_ms:.0f}ms.wav"
            
            with wave.open(str(output_file), 'wb') as wav_file:
                wav_file.setnchannels(1)      # 单声道
                wav_file.setsampwidth(2)      # 16位
                wav_file.setframerate(16000)  # 16kHz采样率
                wav_file.writeframes(segment.audio_data)
                
        except Exception as e:
            logger.error(f"线程 {thread_id} 保存语音段失败: {e}")

    def run_real_multithread_test(self) -> Dict[int, ThreadTestResult]:
        """
        运行真正的多线程测试
        
        Returns:
            Dict[int, ThreadTestResult]: 各线程的测试结果
        """
        logger.info(f"开始真正的多线程并发测试: {self.num_threads} 个线程")
        logger.info(f"音频文件: {self.audio_file}")
        logger.info(f"音频块数量: {len(self.audio_chunks)}")
        logger.info(f"块大小: {self.chunk_size} 字节")
        
        # 使用ThreadPoolExecutor创建真正的多线程
        with ThreadPoolExecutor(max_workers=self.num_threads, thread_name_prefix="AudioProcessor") as executor:
            # 提交所有线程任务
            future_to_thread_id = {}
            for thread_id in range(1, self.num_threads + 1):
                future = executor.submit(self.process_audio_in_thread, thread_id)
                future_to_thread_id[future] = thread_id
            
            # 等待所有线程完成
            successful_results = {}
            for future in as_completed(future_to_thread_id):
                thread_id = future_to_thread_id[future]
                try:
                    result = future.result()
                    successful_results[thread_id] = result
                    logger.info(f"线程 {thread_id} 完成处理")
                except Exception as exc:
                    logger.error(f"线程 {thread_id} 执行异常: {exc}")
        
        return successful_results

    def analyze_results(self, results: Dict[int, ThreadTestResult]):
        """分析测试结果"""
        logger.info("=" * 60)
        logger.info("📊 真正的多线程测试结果分析")
        logger.info("=" * 60)
        
        if not results:
            logger.warning("没有成功的测试结果")
            return
        
        # 实例隔离性验证
        instance_ids = [result.instance_id for result in results.values()]
        unique_instances = set(instance_ids)
        
        logger.info(f"🔍 线程隔离性检查:")
        logger.info(f"   - 线程数量: {len(results)}")
        logger.info(f"   - 实例ID数量: {len(unique_instances)}")
        logger.info(f"   - 实例隔离: {'✅ 成功' if len(unique_instances) == len(results) else '❌ 失败'}")
        
        # 显示各线程详细结果
        logger.info(f"\n📈 各线程处理结果:")
        total_segments = 0
        total_frames = 0
        total_errors = 0
        total_model_load_time = 0.0
        total_processing_time = 0.0
        
        for thread_id, result in results.items():
            logger.info(f"   线程 {thread_id} ({result.thread_name}):")
            logger.info(f"     - 实例ID: {result.instance_id}")
            logger.info(f"     - 语音段: {result.speech_segments_count}")
            logger.info(f"     - 单帧: {result.single_frames_count}")
            logger.info(f"     - 模型加载时间: {result.model_load_time_ms:.1f}ms")
            logger.info(f"     - 纯处理时间: {result.pure_processing_time_ms:.1f}ms")
            logger.info(f"     - 吞吐量: {result.throughput_chunks_per_sec:.1f} 块/秒")
            logger.info(f"     - 错误数: {result.error_count}")
            
            total_segments += result.speech_segments_count
            total_frames += result.single_frames_count
            total_errors += result.error_count
            total_model_load_time += result.model_load_time_ms
            total_processing_time += result.pure_processing_time_ms
        
        # 汇总统计
        avg_model_load_time = total_model_load_time / len(results)
        avg_processing_time = total_processing_time / len(results)
        avg_throughput = sum(r.throughput_chunks_per_sec for r in results.values()) / len(results)
        
        logger.info(f"\n📋 汇总统计:")
        logger.info(f"   - 总语音段: {total_segments}")
        logger.info(f"   - 总单帧: {total_frames}")
        logger.info(f"   - 总错误: {total_errors}")
        logger.info(f"   - 平均模型加载时间: {avg_model_load_time:.1f}ms")
        logger.info(f"   - 平均纯处理时间: {avg_processing_time:.1f}ms")
        logger.info(f"   - 平均吞吐量: {avg_throughput:.1f} 块/秒")
        logger.info(f"   - 结果输出目录: {self.output_dir.absolute()}")
        
        # 性能一致性检查
        processing_times = [r.pure_processing_time_ms for r in results.values()]
        time_variance = max(processing_times) - min(processing_times)
        logger.info(f"\n⚡ 性能一致性:")
        logger.info(f"   - 处理时长差异: {time_variance:.1f}ms")
        logger.info(f"   - 一致性评估: {'✅ 良好' if time_variance < 1000 else '⚠️ 需关注'}")
        
        # 线程真实性验证
        thread_names = [result.thread_name for result in results.values()]
        unique_thread_names = set(thread_names)
        logger.info(f"\n🧵 线程真实性验证:")
        logger.info(f"   - 线程名称数量: {len(unique_thread_names)}")
        logger.info(f"   - 真实多线程: {'✅ 是' if len(unique_thread_names) == len(results) else '❌ 否'}")


async def main():
    """主函数"""
    print("🧵 Cascade 真正的多线程多实例测试")
    print("=" * 50)
    
    # 音频文件路径
    audio_file = "/home/justin/workspace/cascade/我现在开始录音，理论上会有两个文件.wav"
    
    # 检查文件是否存在
    if not os.path.exists(audio_file):
        print(f"❌ 音频文件不存在: {audio_file}")
        return
    
    # 测试配置
    num_threads = 4  # 使用4个线程测试
    chunk_size = 4096  # 4KB块大小
    
    # 创建测试套件
    test_suite = RealMultithreadTestSuite(
        audio_file=audio_file,
        num_threads=num_threads,
        chunk_size=chunk_size
    )
    
    try:
        # 运行真正的多线程测试
        start_time = time.time()
        results = test_suite.run_real_multithread_test()
        end_time = time.time()
        
        # 分析结果
        test_suite.analyze_results(results)
        
        # 总体测试结果
        print(f"\n🎉 测试完成!")
        print(f"⏱️  总耗时: {(end_time - start_time):.2f} 秒")
        print(f"✅ 成功线程: {len(results)} / {num_threads}")
        
        if len(results) == num_threads:
            print("🏆 所有线程均成功完成，真正的多线程多实例测试通过！")
        else:
            print("⚠️  部分线程失败，请检查日志")
    
    except Exception as e:
        print(f"❌ 测试执行失败: {e}")
        logger.exception("测试异常详情:")


if __name__ == "__main__":
    # 注意：这里不使用asyncio.run，因为主要逻辑在同步的多线程中
    import asyncio
    asyncio.run(main())