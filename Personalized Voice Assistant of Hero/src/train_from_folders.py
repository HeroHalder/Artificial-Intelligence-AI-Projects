# train_from_folders.py
import os, glob, random
import numpy as np
from preprocess import load_wav, extract_mfcc
from model import build_model
from sklearn.model_selection import train_test_split
import tensorflow as tf

DATA_ROOT = r"D:\datasets\speech_commands_v0.02"  # change to your extracted path
TARGET_WORDS = ['yes','no','up','down','left','right','on','off','stop','go']

def collect_files(limit_per_class=1000):
    files=[]
    for idx, w in enumerate(TARGET_WORDS):
        folder = os.path.join(DATA_ROOT, w)
        all_files = glob.glob(os.path.join(folder, '*.wav'))
        random.shuffle(all_files)
        selected = all_files[:limit_per_class]
        files += [(p, idx) for p in selected]
    random.shuffle(files)
    return files

def build_X_y(files):
    X = np.zeros((len(files), 160, 40), dtype=np.float32)
    y = np.zeros((len(files),), dtype=np.int32)
    for i,(p,label) in enumerate(files):
        y_raw = load_wav(p)  # using librosa
        mf = extract_mfcc(y_raw)
        X[i] = mf
        y[i] = label
    return X, y

if __name__ == "__main__":
    files = collect_files(limit_per_class=400)  # adjust per RAM
    X, y = build_X_y(files)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.12, random_state=42, stratify=y)
    model = build_model(input_shape=(160,40), num_classes=len(TARGET_WORDS))
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint('models/google_sc_best.h5', save_best_only=True, monitor='val_accuracy', mode='max'),
        tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=6, mode='max', restore_best_weights=True)
    ]
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=25, batch_size=32, callbacks=callbacks)
    model.save('models/google_sc_final.h5')
    print("Saved models/google_sc_final.h5")

