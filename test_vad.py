SAMPLING_RATE = 16000

from pprint import pprint
import time


from silero_vad import (load_silero_vad,
                          read_audio,
                          get_speech_timestamps,
                          save_audio,
                          VADIterator,
                          collect_chunks)
model = load_silero_vad(onnx="/home/justin/opensource/cascade/models/silero-vad/model.onnx")





## using VADIterator class

vad_iterator = VADIterator(model, sampling_rate=SAMPLING_RATE)


start_time = time.time()

wav = read_audio(f'/home/justin/opensource/cascade/请问电动汽车和传统汽车比起来哪个更好啊？.wav', sampling_rate=SAMPLING_RATE)


window_size_samples = 512 if SAMPLING_RATE == 16000 else 256
for i in range(0, len(wav), window_size_samples):
    chunk = wav[i: i+ window_size_samples]
    if len(chunk) < window_size_samples:
      break
    speech_dict = vad_iterator(chunk, return_seconds=False)
    if speech_dict:
        print(speech_dict, end=' ')

end_time = time.time()
print(f"\nProcessing time: {end_time - start_time:.2f} seconds")

vad_iterator.reset_states() # reset model states after each audio




















