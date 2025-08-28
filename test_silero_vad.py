# import cascade
# import asyncio

# async def basic_example():
#     """基础使用示例"""

#     # 方式1：最简单的文件处理
#     results = await cascade.process_audio_file("/home/justin/workspace/cascade/我现在开始录音，理论上会有两个文件.wav")
#     speech_segments = [r for r in results if r.is_speech_segment]
#     print(f"检测到 {len(speech_segments)} 个语音段")

    # 方式2：流式处理
    # async with cascade.StreamProcessor() as processor:
    #     async for result in processor.process_stream(audio_stream):
    #         if result.is_speech_segment:
    #             segment = result.segment
    #             print(f"🎤 语音段: {segment.start_timestamp_ms:.0f}ms - {segment.end_timestamp_ms:.0f}ms")
    #         else:
    #             frame = result.frame
    #             print(f"🔇 单帧: {frame.timestamp_ms:.0f}ms")

# asyncio.run(basic_example())







from silero_vad import (
    collect_chunks,
    get_speech_timestamps,
    load_silero_vad,
    read_audio,
    save_audio,
)

model = load_silero_vad(onnx=True)
wav = read_audio('/home/justin/workspace/cascade/我现在开始录音，理论上会有两个文件.wav', sampling_rate=16000)

speech_timestamps = get_speech_timestamps(wav, model, sampling_rate=16000)

print(speech_timestamps) # list of dicts: [{'start': 40992, 'end': 66528}, {'start': 91168, 'end': 126432}]

# 将所有语音块合并为一个音频
save_audio('only_speech.wav', collect_chunks(speech_timestamps, wav), sampling_rate=16000)


















