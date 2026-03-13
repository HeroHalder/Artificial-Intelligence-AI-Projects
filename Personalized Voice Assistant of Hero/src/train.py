# src/train.py
import os, glob, random
import numpy as np
from src.dataset import AudioDataset
from src.model import build_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

def get_labels_map(root="data/commands"):
    """সব ফোল্ডার থেকে লেবেল নাও"""
    subfolders = [f for f in os.listdir(root) if os.path.isdir(os.path.join(root, f))]
    labels_map = {name: idx for idx, name in enumerate(sorted(subfolders))}
    return labels_map

def load_samples(folder_map):
    """প্রতিটি ফোল্ডার থেকে সব wav ফাইল লোড করো"""
    samples = []
    for label_name, label_idx in folder_map.items():
        files = glob.glob(os.path.join('data', 'commands', label_name, '*.wav'))
        # Augmented ফোল্ডারও include করো যদি থাকে
        aug_folder = os.path.join('data', 'commands', label_name + "_aug")
        if os.path.exists(aug_folder):
            files += glob.glob(os.path.join(aug_folder, '*.wav'))
        samples += [(f, label_idx) for f in files]
    return samples

def main():
    # reproducibility
    random.seed(42)
    np.random.seed(42)

    # সব ফোল্ডার থেকে লেবেল নাও
    labels_map = get_labels_map("data/commands")

    # Load samples
    samples = load_samples(labels_map)
    random.shuffle(samples)

    # Train/Validation split
    split = int(0.8 * len(samples))
    train = samples[:split]
    val = samples[split:]

    # Create datasets
    train_ds = AudioDataset(train, batch_size=64, max_len=160)  # batch size বড় করা হলো
    val_ds = AudioDataset(val, batch_size=64, max_len=160)

    # Build model
    model = build_model(input_shape=(160, 40), num_classes=len(labels_map))

    # Create models folder if not exists
    os.makedirs('models', exist_ok=True)

    # Callbacks
    ckpt = ModelCheckpoint('models/baseline.h5', save_best_only=True, monitor='val_accuracy', mode='max')
    es = EarlyStopping(patience=10, monitor='val_accuracy', mode='max', restore_best_weights=True)
    lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)

    # Train
    model.fit(train_ds, validation_data=val_ds, epochs=50, callbacks=[ckpt, es, lr])

    # Save final model
    model.save("voice_assistant_model.h5")
    print("✅ Model saved successfully!")

if __name__ == '__main__':
    main()