# src/infer.py
from src.tts import speak
import os
import sys
import numpy as np
from tensorflow.keras.models import load_model
from src.preprocess import load_wav, extract_mfcc

# সঠিক মডেল path দাও (baseline.h5 বা voice_assistant_model.h5)
MODEL_PATH = 'models/baseline.h5'

# মডেল একবারই লোড করো
model = load_model(MODEL_PATH)

# সব ফোল্ডার থেকে labels auto-detect করো
LABELS = sorted([
    f for f in os.listdir("data/commands")
    if os.path.isdir(os.path.join("data/commands", f))
])

def predict(path, max_len=160):
    """একটা wav ফাইল থেকে prediction করো"""
    y = load_wav(path)
    mf = extract_mfcc(y, max_len=max_len)
    X = np.expand_dims(mf, axis=0)  # batch dimension যোগ করো
    probs = model.predict(X, verbose=0)[0]
    label_idx = int(np.argmax(probs))
    prob = float(probs[label_idx])
    return LABELS[label_idx], prob

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: py -m src.infer <wav_file>")
    else:
        wav_path = sys.argv[1]
        if not os.path.exists(wav_path):
            print(f"Error: File not found -> {wav_path}")
            sys.exit(1)
        label, prob = predict(wav_path)
        print(f"Detected: {label} (p={prob:.2f})")