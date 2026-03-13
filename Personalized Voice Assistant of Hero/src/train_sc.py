# src/infer.py
import numpy as np
from tensorflow.keras.models import load_model
from src.preprocess import load_wav, extract_mfcc   # <-- এখানে src. যোগ করলাম

MODEL_PATH = 'models/google_sc_best.h5'  # পরে ঠিক মডেল নাম দাও
model = load_model(MODEL_PATH)  # load once

LABELS = ['yes','no','up','down','left','right','on','off','stop','go']  # index->name mapping

def predict(path, max_len=160):
    y = load_wav(path)
    mf = extract_mfcc(y, max_len=max_len)
    X = np.expand_dims(mf, axis=0)
    probs = model.predict(X, verbose=0)[0]
    label = int(np.argmax(probs))
    prob = float(probs[label])
    return label, prob