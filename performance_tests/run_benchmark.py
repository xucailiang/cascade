#!/usr/bin/env python3
"""
Cascade 性能测试脚本

测试流式音频处理接口和高并发调用的性能
"""

import asyncio
import os
import sys
import time
import wave
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import psutil

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import cascade


@dataclass
class TestResult:
    """测试结果"""
    test_name: str
    metrics: list[dict[str, Any]] = field(default_factory=list)


class CascadePerformanceTester:
    """Cascade性能测试器"""

    def __init__(self):
        self.results: list[TestResult] = []
        self.process = psutil.Process()

    def load_test_audio(self, file_path: str) -> bytes:
        """加载测试音频文件"""
        with wave.open(file_path, 'rb') as wf:
            return wf.readframes(wf.getnframes())

    def measure_memory_usage(self) -> float:
        """测量当前内存使用"""
        return self.process.memory_info().rss / 1024 / 1024

    def measure_cpu_usage(self, interval: float | None = None) -> float:
        """测量CPU使用"""
        return self.process.cpu_percent(interval=interval)

    async def run_all_tests(self):
        """运行所有测试"""
        print("🚀 开始Cascade性能测试")

        audio_file = "/home/justin/workspace/cascade/新能源汽车和燃油车相比有哪些优缺点？.wav"
        if not os.path.exists(audio_file):
            print(f"❌ 测试音频文件不存在: {audio_file}")
            return

        audio_data = self.load_test_audio(audio_file)

        # 1. 流式音频处理接口测试
        print("\n📊 [1/3] 测试流式音频处理接口性能...")
        await self.run_streaming_interface_tests(audio_data)

        # 2. 高并发调用测试
        print("\n📊 [2/3] 测试高并发调用性能...")
        await self.run_concurrency_tests(audio_data)

        # 3. 生成报告
        print("\n📊 [3/3] 生成测试报告...")
        self.generate_report()

    async def run_streaming_interface_tests(self, audio_data: bytes):
        # 场景1：不同音频块大小测试
        test_result = TestResult("不同音频块大小测试")
        test_result.metrics = await self.test_chunk_size_performance(audio_data)
        self.results.append(test_result)

    async def run_concurrency_tests(self, audio_data: bytes):
        # 场景1：独立处理器模式
        test_result_ind = TestResult("并发测试（独立处理器）")
        test_result_ind.metrics = await self.test_independent_processors_concurrency(audio_data)
        self.results.append(test_result_ind)

        # 场景2：共享处理器模式
        test_result_shared = TestResult("并发测试（共享处理器）")
        test_result_shared.metrics = await self.test_shared_processor_concurrency(audio_data)
        self.results.append(test_result_shared)

    async def test_chunk_size_performance(self, audio_data: bytes):
        """测试不同音频块大小对性能的影响"""
        chunk_sizes = [512, 1024, 2048, 4096, 8192]
        metrics = []

        for chunk_size in chunk_sizes:
            processor = cascade.StreamProcessor()
            await processor.start()

            latencies = []
            start_time = time.time()
            total_frames = 0

            for i in range(0, len(audio_data), chunk_size):
                chunk = audio_data[i : i + chunk_size]
                chunk_start_time = time.time()
                results = await processor.process_chunk(chunk)
                latencies.append((time.time() - chunk_start_time) * 1000)
                total_frames += len(results)

            total_duration = time.time() - start_time
            await processor.stop()

            metrics.append({
                "chunk_size": chunk_size,
                "avg_latency_ms": np.mean(latencies),
                "fps": total_frames / total_duration
            })
        return metrics

    async def test_independent_processors_concurrency(self, audio_data: bytes):
        """测试每个并发任务使用独立处理器"""
        concurrency_levels = [1, 2, 4, 8, 12, 16, 24, 32]
        metrics = []

        for concurrency in concurrency_levels:
            # 准备阶段
            start_mem = self.measure_memory_usage()
            self.measure_cpu_usage(interval=None)  # 初始化CPU测量

            async def task():
                proc = cascade.StreamProcessor()
                await proc.start()
                count = 0
                for i in range(0, len(audio_data), 4096):
                    res = await proc.process_chunk(audio_data[i:i+4096])
                    count += len(res)
                await proc.stop()
                return count

            # 测试阶段
            start_time = time.time()
            tasks = [task() for _ in range(concurrency)]
            total_frames_list = await asyncio.gather(*tasks)
            total_duration = time.time() - start_time

            # 测量结果
            cpu_usage = self.measure_cpu_usage(interval=None)
            end_mem = self.measure_memory_usage()

            metrics.append({
                "concurrency": concurrency,
                "total_frames": sum(total_frames_list),
                "total_fps": sum(total_frames_list) / total_duration,
                "mem_usage_mb": end_mem - start_mem,
                "cpu_usage_percent": cpu_usage,
            })
        return metrics

    async def test_shared_processor_concurrency(self, audio_data: bytes):
        """测试所有并发任务共享一个处理器"""
        concurrency_levels = [1, 2, 4, 8, 12, 16, 24, 32]
        metrics = []

        for concurrency in concurrency_levels:
            # 准备阶段
            start_mem = self.measure_memory_usage()
            self.measure_cpu_usage(interval=None) # 初始化CPU测量

            # 创建共享的处理器
            processor = cascade.StreamProcessor()
            await processor.start()

            async def task():
                count = 0
                for i in range(0, len(audio_data), 4096):
                    res = await processor.process_chunk(audio_data[i:i+4096])
                    count += len(res)
                return count

            # 测试阶段
            start_time = time.time()
            tasks = [task() for _ in range(concurrency)]
            total_frames_list = await asyncio.gather(*tasks)
            total_duration = time.time() - start_time

            await processor.stop()

            # 测量结果
            cpu_usage = self.measure_cpu_usage(interval=None)
            end_mem = self.measure_memory_usage()

            metrics.append({
                "concurrency": concurrency,
                "total_frames": sum(total_frames_list),
                "total_fps": sum(total_frames_list) / total_duration,
                "mem_usage_mb": end_mem - start_mem,
                "cpu_usage_percent": cpu_usage,
            })
        return metrics

    def generate_report(self):
        """生成并打印测试报告"""
        print("\n" + "="*80)
        print("🏆 Cascade 性能测试报告 🏆")
        print("="*80)

        for result in self.results:
            print(f"\n### 📊 {result.test_name} ###\n")
            if result.test_name == "不同音频块大小测试":
                print(f"{'Chunk Size (B)':<20} {'Avg Latency (ms)':<20} {'Throughput (fps)':<20}")
                print("-"*60)
                for m in result.metrics:
                    print(f"{m['chunk_size']:<20} {m['avg_latency_ms']:.2f} {m['fps']:.2f}")
            elif "并发测试" in result.test_name:
                print(f"{'Concurrency':<15} {'Total FPS':<15} {'Mem Usage (MB)':<20} {'CPU Usage (%)':<15}")
                print("-"*65)
                for m in result.metrics:
                    print(f"{m['concurrency']:<15} {m['total_fps']:.2f} {m['mem_usage_mb']:.2f} {m['cpu_usage_percent']:.2f}")

        print("\n" + "="*80)
        print("✅ 测试结束")
        print("="*80)


if __name__ == "__main__":
    tester = CascadePerformanceTester()
    asyncio.run(tester.run_all_tests())
