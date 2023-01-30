import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import wave
from pydub import AudioSegment
import sys
import os

WAV_LIST = []

class WAV_FILE:
    def __init__(self, file_full_name):
        file_name = file_full_name.split('.')[0]
        wav_file_name = os.path.join("wav",file_name+".wav")
        wav_obj = wave.open(wav_file_name, 'rb')
        sample_freq = wav_obj.getframerate()
        n_samples = wav_obj.getnframes()
        signal_wave = wav_obj.readframes(n_samples)
        signal_array = np.frombuffer(signal_wave, dtype=np.int16)
        t_audio = n_samples/sample_freq
        times = np.linspace(0, t_audio, num=n_samples)
        l_channel = signal_array[0::2]
        r_channel = signal_array[1::2]

        self.wav_obj = wav_obj
        self.sample_freq = sample_freq
        self.n_samples = n_samples
        self.signal_wave = signal_wave
        self.signal_array = signal_array
        self.t_audio = t_audio
        self.times = times
        self.l_channel = l_channel
        self.r_channel = r_channel


def save_mp3_to_wav(file_full_name):
    file_name = file_full_name.split('.')[0]
    audSeg = AudioSegment.from_mp3(os.path.join("src",file_full_name))
    audSeg.export(os.path.join("wav",file_name+".wav"), format="wav")
    return os.path.join("wav",file_name+".wav")


def get_wav_list():
    for i in range(len(sys.argv)):
        file_full_name = sys.argv[i]
        print(file_full_name)
        file_name = file_full_name.split('.')[0]
        if (file_full_name.split('.')[-1] == 'mp3'):
            save_mp3_to_wav(file_full_name=file_full_name)
            wav_file = WAV_FILE(file_full_name=file_full_name)
            WAV_LIST.append(wav_file)


if __name__=="__main__":
    if len(sys.argv) < 3:
        print("python main.py [first wav file] [second wav file] ...")
        exit(1)

    get_wav_list()
    
    ax_list = []
    fig, ax_list = plt.subplots(len(WAV_LIST))

    for index, wav in enumerate(WAV_LIST):
        ax_list[index].plot(wav.times, wav.l_channel)
    
    plt.show()