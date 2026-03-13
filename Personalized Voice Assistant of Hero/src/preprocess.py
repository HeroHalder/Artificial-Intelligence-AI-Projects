# preprocess.py
import librosa
import numpy as np

SAMPLE_RATE = 16000
N_MFCC = 40
MAX_LEN = 160  # number of frames (tweakable)

def load_wav(path, sr=SAMPLE_RATE):
    y, sr = librosa.load(path, sr=sr)
    return y

def extract_mfcc(y, sr=SAMPLE_RATE, n_mfcc=N_MFCC, max_len=MAX_LEN):
    # hop_length and n_fft chosen so that frames ≈ max_len for 1s signals
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=512, hop_length=256)
    mfcc = mfcc.T  # shape: (frames, n_mfcc)
    if mfcc.shape[0] < max_len:
        pad = np.zeros((max_len - mfcc.shape[0], mfcc.shape[1]), dtype=np.float32)
        mfcc = np.vstack([mfcc, pad])
    else:
        mfcc = mfcc[:max_len, :]
    # Optionally normalize per-sample
    mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-9)
    return mfcc.astype(np.float32)
