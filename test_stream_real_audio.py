"""
Cascade 流式VAD处理器 - 智能模式测试

使用真实音频文件测试流式VAD处理器的核心功能。
采用智能模式：保持VAD时序要求的同时显著提升处理速度。
"""

import asyncio
import wave
import logging
import time
from typing import AsyncIterator
from pathlib import Path
import os
from datetime import datetime

from cascade.stream import StreamProcessor, create_default_config, SpeechSegment

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 音频文件路径
# AUDIO_FILE = "/home/justin/workspace/cascade/我现在开始录音，理论上会有两个文件.wav"

AUDIO_FILE = "/home/justin/workspace/cascade/新能源汽车和燃油车相比有哪些优缺点？.wav"

# 输出目录
OUTPUT_DIR = "speech_segments"

# 智能模式配置：最小延迟保持VAD时序
SMART_DELAY = 0.001  # 1ms最小延迟


class PerformanceTimer:
    """性能计时器"""
    
    def __init__(self, name: str):
        self.name = name
        self.start_time: float = 0.0
        self.end_time: float = 0.0
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        logger.info(f"⏱️ 开始: {self.name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        duration = self.end_time - self.start_time
        logger.info(f"⏱️ 完成: {self.name} - 耗时: {duration:.3f}秒")
        return duration


def load_audio_file(file_path: str) -> tuple[bytes, int, int]:
    """
    加载音频文件
    
    Returns:
        (audio_data, sample_rate, channels)
    """
    with PerformanceTimer("加载音频文件"):
        logger.info(f"加载音频文件: {file_path}")
        
        with wave.open(file_path, 'rb') as wav_file:
            # 获取音频参数
            frames = wav_file.getnframes()
            sample_rate = wav_file.getframerate()
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            
            # 计算音频时长
            duration_seconds = frames / sample_rate
            
            logger.info(f"音频参数: {frames}帧, {sample_rate}Hz, {channels}声道, {sample_width}字节/样本")
            logger.info(f"音频时长: {duration_seconds:.2f}秒")
            
            # 读取音频数据
            audio_data = wav_file.readframes(frames)
            
            return audio_data, sample_rate, channels


async def create_audio_stream(audio_data: bytes, frame_size: int = 1024) -> AsyncIterator[bytes]:
    """
    将音频数据转换为异步流（智能模式）
    
    Args:
        audio_data: 音频数据
        frame_size: 每次发送的字节数（默认1024字节 = 512样本 * 2字节）
    """
    logger.info(f"创建音频流，总长度: {len(audio_data)}字节，帧大小: {frame_size}字节")
    
    total_chunks = len(audio_data) // frame_size
    logger.info(f"🎯 智能模式：总共 {total_chunks} 个音频块，使用 {SMART_DELAY*1000:.1f}ms 智能延迟")
    
    chunk_count = 0
    for i in range(0, len(audio_data), frame_size):
        chunk = audio_data[i:i + frame_size]
        
        # 确保块大小是512样本的倍数（1024字节）
        if len(chunk) == frame_size:
            chunk_count += 1
            yield chunk
            
            # 智能延迟：保持VAD时序要求的最小延迟
            await asyncio.sleep(SMART_DELAY)
            
            # 进度提示
            if chunk_count % 100 == 0:
                logger.info(f"📊 已处理 {chunk_count}/{total_chunks} 个音频块")
        else:
            # 最后一块可能不足，跳过
            logger.info(f"跳过最后不完整的块，大小: {len(chunk)}字节")
            break


def save_speech_segment_to_wav(segment: SpeechSegment, output_dir: str = OUTPUT_DIR) -> str:
    """
    保存语音段为WAV文件
    
    Args:
        segment: 语音段对象
        output_dir: 输出目录
        
    Returns:
        保存的文件路径
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成文件名：segment_ID_时间戳.wav
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"segment_{segment.segment_id:03d}_{timestamp}.wav"
    filepath = os.path.join(output_dir, filename)
    
    # 保存为WAV文件
    with wave.open(filepath, 'wb') as wav_file:
        # 设置音频参数
        wav_file.setnchannels(1)  # 单声道
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(segment.sample_rate)  # 采样率
        
        # 写入音频数据
        wav_file.writeframes(segment.audio_data)
    
    # 计算文件大小
    file_size = os.path.getsize(filepath)
    
    logger.info(
        f"💾 保存语音段 {segment.segment_id}: {filename} "
        f"({file_size} 字节, {segment.duration_ms:.0f}ms)"
    )
    
    return filepath


async def test_stream_processing():
    """测试流式处理（智能模式）"""
    with PerformanceTimer("完整流式处理测试"):
        logger.info("=== Cascade 流式VAD处理器 - 智能模式测试 ===")
        
        # 检查音频文件是否存在
        if not Path(AUDIO_FILE).exists():
            logger.error(f"音频文件不存在: {AUDIO_FILE}")
            return False
        
        try:
            # 加载音频文件
            audio_data, sample_rate, channels = load_audio_file(AUDIO_FILE)
            
            # 验证音频格式
            if sample_rate != 16000:
                logger.warning(f"音频采样率为 {sample_rate}Hz，建议使用16kHz")
            
            if channels != 1:
                logger.warning(f"音频为 {channels} 声道，建议使用单声道")
            
            # 创建配置
            with PerformanceTimer("创建处理器配置"):
                config = create_default_config(
                    vad_threshold=0.5,
                    max_instances=1
                )
            
            # 创建处理器
            with PerformanceTimer("初始化流式处理器"):
                async with StreamProcessor(config) as processor:
                    logger.info("流式处理器已启动")
                    
                    # 创建音频流
                    audio_stream = create_audio_stream(audio_data, frame_size=1024)  # 512样本 * 2字节
                    
                    # 处理音频流
                    result_count = 0
                    speech_segments = 0
                    single_frames = 0
                    
                    logger.info("开始处理音频流...")
                    stream_start_time = time.perf_counter()
                    
                    with PerformanceTimer("音频流处理"):
                        async for result in processor.process_stream(audio_stream, "smart-audio-test"):
                            result_count += 1
                            
                            if result.is_speech_segment:
                                speech_segments += 1
                                segment = result.segment
                                if segment:
                                    logger.info(
                                        f"🎤 检测到语音段 {segment.segment_id}: "
                                        f"时长 {segment.duration_ms:.0f}ms, "
                                        f"包含 {segment.frame_count} 帧"
                                    )
                                    
                                    # 保存语音段为WAV文件
                                    try:
                                        with PerformanceTimer(f"保存语音段{segment.segment_id}"):
                                            saved_path = save_speech_segment_to_wav(segment)
                                        logger.info(f"✅ 语音段已保存: {saved_path}")
                                    except Exception as e:
                                        logger.error(f"❌ 保存语音段失败: {e}")
                            else:
                                single_frames += 1
                    
                    stream_end_time = time.perf_counter()
                    stream_duration = stream_end_time - stream_start_time
                    
                    # 获取统计信息
                    stats = processor.get_stats()
                    
                    logger.info("=== 处理完成 ===")
                    logger.info(f"总结果数: {result_count}")
                    logger.info(f"语音段数: {speech_segments}")
                    logger.info(f"单帧数: {single_frames}")
                    logger.info(f"流处理耗时: {stream_duration:.3f}秒")
                    logger.info(f"处理统计: {stats.summary()}")
                    
                    return True
                    
        except Exception as e:
            logger.error(f"测试失败: {e}")
            import traceback
            traceback.print_exc()
            return False


async def main():
    """主函数"""
    with PerformanceTimer("整个测试程序"):
        logger.info("🚀 Cascade 流式VAD处理器 - 智能模式测试")
        logger.info(f"📁 语音段将保存到目录: {OUTPUT_DIR}")
        logger.info(f"🎯 智能延迟: {SMART_DELAY*1000:.1f}ms（保持VAD时序的最小延迟）")
        
        try:
            # 测试流式处理
            success = await test_stream_processing()
            
            if success:
                logger.info("✅ 智能模式测试成功")
            else:
                logger.error("❌ 智能模式测试失败")
                
        except Exception as e:
            logger.error(f"测试运行失败: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())