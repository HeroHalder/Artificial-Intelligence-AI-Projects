# src/augment.py
import os
import librosa
import numpy as np
import soundfile as sf

SAMPLE_RATE = 16000
THRESHOLD = 1000   # 1000 এর নিচে হলে Augmentation চালাবে

def augment_and_save(y, sr, out_path, base_name):
    # Time stretch
    y_fast = librosa.effects.time_stretch(y, rate=1.1)
    sf.write(f"{out_path}/{base_name}_fast.wav", y_fast, sr)

    y_slow = librosa.effects.time_stretch(y, rate=0.9)
    sf.write(f"{out_path}/{base_name}_slow.wav", y_slow, sr)

    # Pitch shift (keyword arguments ব্যবহার করো)
    y_up = librosa.effects.pitch_shift(y, sr=sr, n_steps=2)
    sf.write(f"{out_path}/{base_name}_up.wav", y_up, sr)

    y_down = librosa.effects.pitch_shift(y, sr=sr, n_steps=-2)
    sf.write(f"{out_path}/{base_name}_down.wav", y_down, sr)

    # Noise
    noise = np.random.normal(0, 0.005, len(y))
    y_noisy = y + noise
    sf.write(f"{out_path}/{base_name}_noisy.wav", y_noisy, sr)
    
def process_folder(folder):
    files = [f for f in os.listdir(folder) if f.endswith(".wav")]
    if len(files) < THRESHOLD:
        print(f"Augmenting folder: {folder} (found {len(files)} files)")
        out_path = folder + "_aug"
        os.makedirs(out_path, exist_ok=True)

        for fname in files:
            path = os.path.join(folder, fname)
            y, sr = librosa.load(path, sr=SAMPLE_RATE)
            base_name = os.path.splitext(fname)[0]
            augment_and_save(y, sr, out_path, base_name)
    else:
        print(f"Skipping folder: {folder} (found {len(files)} files)")

if __name__ == "__main__":
    root = "data/commands"
    for sub in os.listdir(root):
        folder = os.path.join(root, sub)
        if os.path.isdir(folder):
            process_folder(folder)