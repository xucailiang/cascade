Examples
Open In Colab

Imports
#@title Install and Import Dependencies

# this assumes that you have a relevant version of PyTorch installed
!pip install -q torchaudio

SAMPLING_RATE = 16000

import torch
torch.set_num_threads(1)

from IPython.display import Audio
from pprint import pprint
# download example
torch.hub.download_url_to_file('https://models.silero.ai/vad_models/en.wav', 'en_example.wav')


USE_PIP = True # download model using pip package or torch.hub
USE_ONNX = False # change this to True if you want to test onnx model
# ONNX model supports opset_version 15 and 16 (default is 16). 
# Pass argument opset_version to load_silero_vad (pip) or torch.hub.load (torchhub).
# !!! ONNX model with opset_version=15 supports only 16000 sampling rate !!!
if USE_ONNX:
  !pip install -q onnxruntime
if USE_PIP:
  !pip install -q silero-vad
  from silero_vad import (load_silero_vad,
                          read_audio,
                          get_speech_timestamps,
                          save_audio,
                          VADIterator,
                          collect_chunks)
  model = load_silero_vad(onnx=USE_ONNX, opset_version=16)
else:
  model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                model='silero_vad',
                                force_reload=True,
                                onnx=USE_ONNX,
                                opset_version=16)

  (get_speech_timestamps,
  save_audio,
  read_audio,
  VADIterator,
  collect_chunks) = utils
Speech timestamps from full audio
wav = read_audio('en_example.wav', sampling_rate=SAMPLING_RATE)
# get speech timestamps from full audio file
speech_timestamps = get_speech_timestamps(wav, model, sampling_rate=SAMPLING_RATE)
pprint(speech_timestamps)
# merge all speech chunks to one audio
save_audio('only_speech.wav',
           collect_chunks(speech_timestamps, wav), sampling_rate=SAMPLING_RATE) 
Audio('only_speech.wav')
Entire audio inference
wav = read_audio('en_example.wav', sampling_rate=SAMPLING_RATE)
# chunk size is 32 ms, and each second of the audio contains 31.25 chunks
# currently only chunks of size 512 are used for 16 kHz and 256 for 8 kHz
# e.g. 512 / 16000 = 256 / 8000 = 0.032 s = 32.0 ms
predicts = model.audio_forward(wav, sr=SAMPLING_RATE)
Stream imitation example
## using VADIterator class

vad_iterator = VADIterator(model, sampling_rate=SAMPLING_RATE)
wav = read_audio(f'en_example.wav', sampling_rate=SAMPLING_RATE)

window_size_samples = 512 if SAMPLING_RATE == 16000 else 256
for i in range(0, len(wav), window_size_samples):
    chunk = wav[i: i+ window_size_samples]
    if len(chunk) < window_size_samples:
      break
    speech_dict = vad_iterator(chunk, return_seconds=True)
    if speech_dict:
        print(speech_dict, end=' ')
vad_iterator.reset_states() # reset model states after each audio
## just probabilities

wav = read_audio('en_example.wav', sampling_rate=SAMPLING_RATE)
speech_probs = []
window_size_samples = 512 if SAMPLING_RATE == 16000 else 256
for i in range(0, len(wav), window_size_samples):
    chunk = wav[i: i+window_size_samples]
    if len(chunk) < window_size_samples:
        break
    speech_prob = model(chunk, SAMPLING_RATE).item()
    speech_probs.append(speech_prob)
model.reset_states() # reset model states after each audio

print(speech_probs[:10]) # first 10 chunks predicts