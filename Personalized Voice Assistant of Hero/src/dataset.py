# src/dataset.py
import os
import numpy as np
from tensorflow.keras.utils import Sequence
from .audio_utils import load_audio, extract_mfcc

class AudioDataset(Sequence):
    def __init__(self, samples, batch_size=32, sr=16000, max_len=160):
        """
        samples: list of tuples (path, label_int)
        max_len: frames to pad/truncate to (e.g., 160)
        """
        self.samples = samples
        self.batch_size = batch_size
        self.sr = sr
        self.max_len = max_len

    def __len__(self):
        return int(np.ceil(len(self.samples) / self.batch_size))

    def __getitem__(self, idx):
        batch = self.samples[idx*self.batch_size:(idx+1)*self.batch_size]
        X = []
        y = []
        for path, label in batch:
            y_audio = load_audio(path, sr=self.sr)
            mfcc = extract_mfcc(y_audio, sr=self.sr)
            if mfcc.shape[0] < self.max_len:
                pad = np.zeros((self.max_len - mfcc.shape[0], mfcc.shape[1]))
                mfcc = np.vstack([mfcc, pad])
            else:
                mfcc = mfcc[:self.max_len, :]
            X.append(mfcc)
            y.append(label)
        X = np.array(X)  # shape: (B, frames, n_mfcc)
        y = np.array(y)
        return X, y
