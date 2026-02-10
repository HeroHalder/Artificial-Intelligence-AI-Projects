# src/audio_utils.py
import numpy as np
import librosa

def load_audio(path, sr=16000):
    y, _ = librosa.load(path, sr=sr)
    return y

def extract_mfcc(y, sr=16000, n_mfcc=40, hop_length=256, n_fft=512):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc,
                                n_fft=n_fft, hop_length=hop_length)
    # shape: (n_mfcc, frames)
    # normalize
    mfcc = (mfcc - np.mean(mfcc, axis=1, keepdims=True)) / (np.std(mfcc, axis=1, keepdims=True) + 1e-9)
    return mfcc.T  # return frames × n_mfcc






# src/dataset.py
import os
import numpy as np
from tensorflow.keras.utils import Sequence
from .audio_utils import load_audio, extract_mfcc

class AudioDataset(Sequence):
    def __init__(self, samples, labels, batch_size=32, sr=16000, max_len=160):
        self.samples = samples
        self.labels = labels
        self.batch_size = batch_size
        self.sr = sr
        self.max_len = max_len  # number of frames for padding/trunc
    def __len__(self):
        return int(np.ceil(len(self.samples) / self.batch_size))
    def __getitem__(self, idx):
        batch_samples = self.samples[idx*self.batch_size:(idx+1)*self.batch_size]
        X = []
        y = []
        for p,l in batch_samples:
            y_audio = load_audio(p, sr=self.sr)
            mfcc = extract_mfcc(y_audio, sr=self.sr)
            # pad/truncate
            if mfcc.shape[0] < self.max_len:
                pad = np.zeros((self.max_len - mfcc.shape[0], mfcc.shape[1]))
                mfcc = np.vstack([mfcc, pad])
            else:
                mfcc = mfcc[:self.max_len,:]
            X.append(mfcc)
            y.append(l)
        return np.array(X), np.array(y)






# src/model.py
import tensorflow as tf
from tensorflow.keras import layers, models

def build_model(input_shape=(160,40), num_classes=10):
    inp = layers.Input(shape=input_shape)  # frames x n_mfcc
    x = layers.Reshape((*input_shape,1))(inp)
    x = layers.Conv2D(32, (3,3), activation='relu', padding='same')(x)
    x = layers.MaxPool2D((2,2))(x)
    x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = layers.MaxPool2D((2,2))(x)
    x = layers.TimeDistributed(layers.Flatten())(x)
    x = layers.GRU(128, return_sequences=False)(x)
    x = layers.Dense(128, activation='relu')(x)
    out = layers.Dense(num_classes, activation='softmax')(x)
    model = models.Model(inputs=inp, outputs=out)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model








# src/train.py
import glob, random
from src.dataset import AudioDataset
from src.model import build_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

def load_samples_from_folder(folder, label):
    files = glob.glob(folder + '/*.wav')
    return [(f,label) for f in files]

def main():
    # Example: prepare list of (path,label)
    samples = []
    # add dataset folders -> map to numeric labels
    labels_map = {'play_music':0, 'set_alarm':1, 'weather':2, 'stop':3, 'next':4}
    for cmd,idx in labels_map.items():
        samples += load_samples_from_folder(f'data/commands/{cmd}', idx)
    random.shuffle(samples)
    split = int(0.8*len(samples))
    train = samples[:split]
    val = samples[split:]
    train_ds = AudioDataset(train, None, batch_size=32)
    val_ds = AudioDataset(val, None, batch_size=32)
    model = build_model(input_shape=(160,40), num_classes=len(labels_map))
    model.summary()
    ckpt = ModelCheckpoint('models/baseline.h5', save_best_only=True, monitor='val_accuracy', mode='max')
    es = EarlyStopping(patience=8, monitor='val_accuracy', mode='max')
    model.fit(train_ds, validation_data=val_ds, epochs=50, callbacks=[ckpt,es])
if __name__=='__main__':
    main()




# src/finetune.py
import tensorflow as tf
from src.model import build_model
from src.dataset import AudioDataset
import glob

def finetune_model(user_folder, base_model_path='models/baseline.h5'):
    # assume user_folder has subfolders for intents with wavs
    labels_map = {'play_music':0, 'set_alarm':1, 'weather':2, 'stop':3, 'next':4}
    samples=[]
    for cmd,idx in labels_map.items():
        files = glob.glob(f'{user_folder}/{cmd}/*.wav')
        samples += [(f,idx) for f in files]
    if len(samples) < 10:
        print("Provide at least 10 user samples for fine-tuning.")
        return
    ds = AudioDataset(samples, None, batch_size=8)
    model = build_model(input_shape=(160,40), num_classes=len(labels_map))
    model.load_weights(base_model_path)
    # optionally freeze early layers
    for layer in model.layers[:6]:
        layer.trainable = False
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(ds, epochs=10)
    model.save(f'models/finetuned_{user_folder.split("/")[-1]}.h5')
    print("Fine-tuning complete.")


