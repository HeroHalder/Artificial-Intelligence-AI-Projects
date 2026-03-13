# src/audio_utils.py
import numpy as np
import librosa

def load_audio(path, sr=16000):
    y, _ = librosa.load(path, sr=sr)
    return y

def extract_mfcc(y, sr=16000, n_mfcc=40, hop_length=256, n_fft=512):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc,
                                n_fft=n_fft, hop_length=hop_length)
    mfcc = (mfcc - np.mean(mfcc, axis=1, keepdims=True)) / (np.std(mfcc, axis=1, keepdims=True) + 1e-9)
    return mfcc.T  # frames x n_mfcc
