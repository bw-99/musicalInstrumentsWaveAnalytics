import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import wave
from pydub import AudioSegment

file_full_name = 'tb_bad.mp3'
file_name = file_full_name.split('.')[0]

if (file_full_name.split('.')[-1] == 'mp3'):
    audSeg = AudioSegment.from_mp3(file_full_name)
    audSeg.export(file_name, format="wav")

wav_obj = wave.open(file_name, 'rb')
sample_freq = wav_obj.getframerate()
n_samples = wav_obj.getnframes()

signal_wave = wav_obj.readframes(n_samples)
signal_array = np.frombuffer(signal_wave, dtype=np.int16)
t_audio = n_samples/sample_freq
times = np.linspace(0, t_audio, num=n_samples)

l_channel = signal_array[0::2]
r_channel = signal_array[1::2]

plt.figure(figsize=(15, 5))
plt.plot(times, signal_array)
plt.title('Left Channel')
plt.ylabel('Signal Value')
plt.xlabel('Time (s)')
plt.xlim(0, t_audio)
plt.show()