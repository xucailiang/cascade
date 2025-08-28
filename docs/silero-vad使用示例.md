# Silero VAD 使用示例

本文档展示了如何使用 Silero VAD (Voice Activity Detection) 模型进行语音活动检测的各种示例。

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/snakers4/silero-vad/blob/master/examples.ipynb)

## 安装和导入依赖

```python
# 安装和导入依赖
# 假设您已经安装了相关版本的 PyTorch
!pip install -q torchaudio

SAMPLING_RATE = 16000

import torch
torch.set_num_threads(1)

from IPython.display import Audio
from pprint import pprint

# 下载示例音频文件
torch.hub.download_url_to_file('https://models.silero.ai/vad_models/en.wav', 'en_example.wav')
```

## 模型加载配置

```python
USE_PIP = True  # 使用 pip 包下载模型或使用 torch.hub
USE_ONNX = False  # 如果要测试 ONNX 模型，请将此设置为 True

# ONNX 模型支持 opset_version 15 和 16（默认为 16）
# 将参数 opset_version 传递给 load_silero_vad (pip) 或 torch.hub.load (torchhub)
# !!! opset_version=15 的 ONNX 模型仅支持 16000 采样率 !!!

if USE_ONNX:
    !pip install -q onnxruntime

if USE_PIP:
    !pip install -q silero-vad
    from silero_vad import (
        load_silero_vad,
        read_audio,
        get_speech_timestamps,
        save_audio,
        VADIterator,
        collect_chunks
    )
    model = load_silero_vad(onnx=USE_ONNX)
else:
    model, utils = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad',
        force_reload=True,
        onnx=USE_ONNX,
        opset_version=16
    )
    
    (get_speech_timestamps,
     save_audio,
     read_audio,
     VADIterator,
     collect_chunks) = utils
```

## 从完整音频获取语音时间戳

```python
# 读取音频文件
wav = read_audio('en_example.wav', sampling_rate=SAMPLING_RATE)

# 从完整音频文件获取语音时间戳
speech_timestamps = get_speech_timestamps(wav, model, sampling_rate=SAMPLING_RATE)
print(speech_timestamps) # # [{'start': 40992, 'end': 66528}, {'start': 91168, 'end': 126432}]

# 将所有语音块合并为一个音频
save_audio('only_speech.wav',
           collect_chunks(speech_timestamps, wav), 
           sampling_rate=SAMPLING_RATE)

# 播放处理后的音频
Audio('only_speech.wav')
```

## 完整音频推理

```python
wav = read_audio('en_example.wav', sampling_rate=SAMPLING_RATE)

# 块大小为 32 ms，每秒音频包含 31.25 个块
# 目前仅使用大小为 512 的块用于 16 kHz，256 用于 8 kHz
# 例如：512 / 16000 = 256 / 8000 = 0.032 s = 32.0 ms
predicts = model.audio_forward(wav, sr=SAMPLING_RATE)
```

## 流式处理模拟示例

### 使用 VADIterator 类

```python
vad_iterator = VADIterator(model, sampling_rate=SAMPLING_RATE)
wav = read_audio('en_example.wav', sampling_rate=SAMPLING_RATE)

window_size_samples = 512 if SAMPLING_RATE == 16000 else 256

for i in range(0, len(wav), window_size_samples):
    chunk = wav[i: i + window_size_samples]
    if len(chunk) < window_size_samples:
        break
    speech_dict = vad_iterator(chunk, return_seconds=True)
    if speech_dict:
        print(speech_dict, end=' ')

# 每个音频处理完后重置模型状态
vad_iterator.reset_states()
```

### 仅获取概率值

```python
wav = read_audio('en_example.wav', sampling_rate=SAMPLING_RATE)
speech_probs = []
window_size_samples = 512 if SAMPLING_RATE == 16000 else 256

for i in range(0, len(wav), window_size_samples):
    chunk = wav[i: i + window_size_samples]
    if len(chunk) < window_size_samples:
        break
    speech_prob = model(chunk, SAMPLING_RATE).item()
    speech_probs.append(speech_prob)

# 每个音频处理完后重置模型状态
model.reset_states()

# 打印前 10 个块的预测结果
print(speech_probs[:10])
```

## 注意事项

- 确保音频采样率与模型要求匹配（通常为 16kHz 或 8kHz）
- 在处理新音频前记得重置模型状态
- ONNX 模型在某些配置下可能有采样率限制
- 块大小固定为 512 样本（16kHz）或 256 样本（8kHz）