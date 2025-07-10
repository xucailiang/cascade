# Cascade 示例程序

本目录包含了Cascade库的示例程序，展示了如何使用Cascade库的各种功能。

## VAD示例程序

`vad_demo.py`展示了如何使用Cascade库的顶层API处理音频文件和音频流，检测语音活动。

### 运行示例

```bash
# 安装依赖
pip install numpy

# 运行示例
python vad_demo.py path/to/audio.wav
```

### 功能展示

该示例程序展示了以下功能：

1. **处理音频文件**：使用`process_audio_file`函数处理整个音频文件
2. **检测语音段**：使用`detect_speech_segments`函数检测音频中的语音段
3. **处理音频流**：使用`process_audio_stream`函数处理模拟的音频流
4. **转录音频**：使用`transcribe_audio`函数转录音频中的语音（需要安装whisper）

### 代码示例

#### 处理音频文件

```python
import asyncio
import cascade

async def process_file_example(file_path):
    # 使用便捷函数处理音频文件
    results = await cascade.process_audio_file(
        file_path,
        threshold=0.5,
        workers=4
    )
    
    # 打印结果
    print(f"检测到 {len(results)} 个结果")
    for result in results:
        if result.is_speech:
            print(f"语音: {result.start_ms}ms - {result.end_ms}ms")

# 运行示例
asyncio.run(process_file_example("audio.wav"))
```

#### 检测语音段

```python
import asyncio
import cascade

async def detect_segments_example(file_path):
    # 使用便捷函数检测语音段
    segments = await cascade.detect_speech_segments(
        file_path,
        threshold=0.5,
        min_speech_duration_ms=100,
        min_silence_duration_ms=300
    )
    
    # 打印结果
    print(f"检测到 {len(segments)} 个语音段")
    for segment in segments:
        print(f"语音段: {segment.start_ms}ms - {segment.end_ms}ms")

# 运行示例
asyncio.run(detect_segments_example("audio.wav"))
```

#### 处理音频流

```python
import asyncio
import cascade

async def stream_generator(file_path, chunk_size=4000):
    """模拟音频流生成器"""
    with open(file_path, 'rb') as f:
        # 跳过WAV文件头（44字节）
        header = f.read(44)
        
        # 读取数据块
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
                
            # 模拟网络延迟
            await asyncio.sleep(0.1)
            
            yield chunk

async def process_stream_example(file_path):
    # 创建音频流生成器
    audio_stream = stream_generator(file_path)
    
    # 使用便捷函数处理音频流
    async for result in cascade.process_audio_stream(
        audio_stream,
        sample_rate=16000,
        audio_format="wav",
        threshold=0.5
    ):
        if result.is_speech:
            print(f"检测到语音: {result.start_ms}ms - {result.end_ms}ms")

# 运行示例
asyncio.run(process_stream_example("audio.wav"))
```

#### 转录音频

```python
import asyncio
import cascade

async def transcribe_example(file_path):
    # 使用便捷函数转录音频
    results = await cascade.transcribe_audio(
        file_path,
        threshold=0.5,
        min_speech_duration_ms=100,
        min_silence_duration_ms=300,
        language="zh"
    )
    
    # 打印结果
    print(f"转录完成，检测到 {len(results)} 个语音段")
    for result in results:
        print(f"{result['start_ms']}ms - {result['end_ms']}ms: {result['text']}")

# 运行示例（需要安装whisper）
# pip install openai-whisper
asyncio.run(transcribe_example("audio.wav"))
```

## 高级用法

除了顶层API外，Cascade库还提供了更底层的API，允许用户更精细地控制VAD处理过程。

```python
import asyncio
import cascade
from cascade.types.audio import AudioConfig
from cascade.types.vad import VADConfig

async def advanced_example(file_path):
    # 创建音频配置
    audio_config = AudioConfig(
        sample_rate=16000,
        format="wav"
    )
    
    # 创建VAD配置
    vad_config = VADConfig(
        backend="onnx",
        workers=4,
        threshold=0.5,
        chunk_duration_ms=500,
        overlap_ms=16
    )
    
    # 创建VAD处理器
    processor = cascade.VADProcessor(
        vad_config=vad_config,
        audio_config=audio_config
    )
    
    try:
        # 启动处理器
        await processor.start()
        
        # 处理文件
        async for result in processor.process_file(file_path):
            if result.is_speech:
                print(f"语音: {result.start_ms}ms - {result.end_ms}ms")
                
        # 获取语音段
        segments = processor.get_segments()
        print(f"检测到 {len(segments)} 个语音段")
        
    finally:
        # 关闭处理器
        await processor.close()

# 运行示例
asyncio.run(advanced_example("audio.wav"))