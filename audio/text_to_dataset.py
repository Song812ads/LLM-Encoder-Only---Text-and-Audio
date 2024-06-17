import numpy as np
import librosa
import io
import os
import librosa.display
import matplotlib.pyplot as plt
from gtts import gTTS  # You'll need to install gtts package for text-to-speech
import soundfile as sf
import time
from responsive_voice import ResponsiveVoice
from responsive_voice.voices import VietnameseMale 

# engine = ResponsiveVoice()
viet = VietnameseMale()
dataset = 'dataset_VN.txt'
data_fol = 'dataset/'

def text_to_logmelspectrogram(text, label, index, lang='vi'):
    # Convert text to speech
    tts = gTTS(text=text, lang=lang)
    # audio_data = io.BytesIO()
    save_file = f"{data_fol}{label}_{index}.wav"
    tts.save(save_file)
    # tts.save()
      # Rewind the buffer to the beginning

    # Step 2: Load the audio from the in-memory buffer using soundfile and librosa
    # buffer.seek(0)  # Ensure the buffer is at the beginning
    # y, sr = sf.read(buffer)
    # if sr != sample_rate:
    #     y = librosa.resample(y, orig_sr=sr, target_sr=sample_rate)
    #     sr = sample_rate

    # viet = VietnameseMale()
    # audio_bytes = viet.get_mp3(text, mp3_file = 'temp')

    # # Load the audio from in-memory bytes using librosa and soundfile
    # buffer = io.BytesIO(audio_bytes)
    # buffer.seek(0)  # Ensure the buffer is at the beginning
    # y, sr = librosa.load(buffer, sr=sample_rate)

    # y, sr = librosa.load('temp.mp3', sr=sample_rate, mono=True)
    
    # Compute the log-mel spectrogram
    # S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    # log_S = librosa.power_to_db(S, ref=np.max)


#10s = (128,431)
#20s = (128,862)
#30s = (128,323


if __name__ == "__main__":
    # text = "Chào em, anh đứng đây từ chiều "
    store_file = 'model_spectrogram/dataset.npy'
    X_dataset = np.loadtxt(dataset, delimiter=',', dtype='str', usecols=[1], encoding='utf-8')
    y_dataset = np.loadtxt(dataset, delimiter=',', dtype='int32', usecols=[0], encoding='utf-8')
    store_data = np.zeros((X_dataset.shape[0],128,323))
    if os.path.exists(store_file):
        store_data = np.load(store_file)
    print(store_data.shape)
    i = 0
    while (i<X_dataset.shape[0]):
        try:
            sentence = X_dataset[i]
            label = y_dataset[i]
            text_to_logmelspectrogram(sentence, label,i)
            # padded_array = np.pad(data, ((0, 0), (0, 323-data.shape[1])), mode='constant', constant_values=0)
            # store_data[i] = padded_array
            i+=1
            # time.sleep(0.1)
        except:
            # np.save(store_file,store_data)
            print(f'Loop stop at {i}')
            break
            # time.sleep(0.1)
        # print(i)
    # np.save(store_file, store_data)
    # check_data = np.load('model_spectrogram/dataset.npy')
    # print(store_data == check_data)
    

        


