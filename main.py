import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import wave
from pydub import AudioSegment
import sys
import os
import pyaudio
import struct
import time
from scipy.fftpack import fft

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
    file_list = sys.argv[2:]
    for i in range(len(file_list)):
        file_full_name = file_list[i]
        file_name = file_full_name.split('.')[0]
        if (file_full_name.split('.')[-1] == 'mp3'):
            save_mp3_to_wav(file_full_name=file_full_name)
        wav_file = WAV_FILE(file_full_name=file_full_name)
        WAV_LIST.append(wav_file)


def draw_time_amp():
    ax_list = []
    fig, ax_list = plt.subplots(len(WAV_LIST))
    for index, wav in enumerate(WAV_LIST):
        ax_list[index].plot(wav.times, wav.l_channel)
    
    plt.show()

def draw_freq_amp():
    fig, (ax1, ax2) = plt.subplots(2, figsize=(15, 7))

    CHUNK = 1024 * 2             # samples per frame
    FORMAT = pyaudio.paInt16     # audio format (bytes per sample?)
    CHANNELS = 1                 # single channel for microphone
    RATE = 44100                 # samples per second
    p = pyaudio.PyAudio()
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        output=True,
        frames_per_buffer=CHUNK
    )

    # variable for plotting
    x = np.arange(0, 2 * CHUNK, 2)       # samples (waveform)
    xf = np.linspace(0, RATE, CHUNK)     # frequencies (spectrum)

    # create a line object with random data
    line, = ax1.plot(x, np.random.rand(CHUNK), '-', lw=2)

    # create semilogx line for spectrum
    line_fft, = ax2.semilogx(xf, np.random.rand(CHUNK), '-', lw=2)

    # format waveform axes
    ax1.set_title('AUDIO WAVEFORM')
    ax1.set_xlabel('samples')
    ax1.set_ylabel('volume')
    ax1.set_ylim(0, 255)
    ax1.set_xlim(0, 2 * CHUNK)
    plt.setp(ax1, xticks=[0, CHUNK, 2 * CHUNK], yticks=[0, 128, 255])
    plt.show(block=False)
    # format spectrum axes
    ax2.set_xlim(20, RATE / 2)

    print('stream started')

    # for measuring frame rate
    frame_count = 0
    start_time = time.time()

    while True:
        # binary data
        data = stream.read(CHUNK)  
        
        # convert data to integers, make np array, then offset it by 127
        data_int = struct.unpack(str(2 * CHUNK) + 'B', data)
        
        # create np array and offset by 128
        data_np = np.array(data_int).astype('b')[::2] + 128
        
        line.set_ydata(data_np)
        
        # compute FFT and update line
        yf = fft(data_int)
        line_fft.set_ydata(np.abs(yf[0:CHUNK])  / (128 * CHUNK))
        
        # update figure canvas
        try:
            fig.canvas.draw()
            fig.canvas.flush_events()
            frame_count += 1
        except:
            # calculate average frame rate
            frame_rate = frame_count / (time.time() - start_time)
            
            print('stream stopped')
            print('average frame rate = {:.0f} FPS'.format(frame_rate))
            break


if __name__=="__main__":

    if(sys.argv[1] == "time_amp"):
        if len(sys.argv) < 4:
            print("python main.py option [first wav file] [second wav file] ...")
            exit(1)

        get_wav_list()
        draw_time_amp()
    elif(sys.argv[1] == "amp_freq"):
        draw_freq_amp()
    else:
        print("python main.py option [first wav file] [second wav file] ...")
