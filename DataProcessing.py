#Import needed tools
import librosa
import soundfile
import os, glob
import numpy as np
from sklearn.model_selection import train_test_split

#Create class for data processing
class DataProcessor:

    def __init__(self):
        self.emotions = {
            '01' : 'neutral',
            '02' : 'calm',
            '03' : 'happy',
            '04' : 'sad',
            '05' : 'angry',
            '06' : 'fearful',
            '07' : 'disgust',
            '08' : 'surprised'
        }
        self.observed_emotions = ['calm', 'happy', 'fearful', 'disgust']

    #Extract features from a sound file
    def extract_feature(self, file_name, mfcc, chroma, mel):
        with soundfile.SoundFile(file_name) as sf:
            data = sf.read(dtype = 'float32')
            sample_rate = sf.samplerate
            if chroma:
                stft = np.abs(librosa.stft(data))
            result = np.array([])
            if mfcc:
                mfccs = np.mean(librosa.feature.mfcc(y = data, sr = sample_rate, n_mfcc = 40).T, axis = 0)
                result = np.hstack((result, mfccs))
            if chroma:
                chroma = np.mean(librosa.feature.chroma_stft(S = stft, sr = sample_rate).T, axis = 0)
                result = np.hstack((result, chroma))
            if mel:
                mel = np.mean(librosa.feature.melspectrogram(data, sr = sample_rate).T, axis = 0)
                result = np.hstack((result, mel))
        return result

    #Load data
    def load_data(self, test_size):
        x, y = [], []
        for file in glob.glob("Dataset\\Actor_*\\*.wav"):
            file_name = os.path.basename(file)
            emotion = self.emotions[file_name.split('-')[2]]
            if emotion in self.observed_emotions:
                feature = self.extract_feature(file, mfcc = True, chroma = True, mel = True)
                x.append(feature)
                y.append(emotion)
        return train_test_split(np.array(x), y, test_size = test_size, random_state = 9)
