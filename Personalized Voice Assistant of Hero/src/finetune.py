# src/finetune.py
import os, glob
import tensorflow as tf
from src.model import build_model
from src.dataset import AudioDataset

def finetune_user(user_folder, base_model_path='models/baseline.h5', user_name='user'):
    labels_map = {'play_music':0, 'set_alarm':1, 'weather':2, 'stop':3, 'next':4}
    samples=[]
    for label_name, idx in labels_map.items():
        files = glob.glob(os.path.join(user_folder, label_name, '*.wav'))
        samples += [(f, idx) for f in files]
    if len(samples) < 8:
        print("At least 8 user samples recommended for fine-tuning.")
        return
    ds = AudioDataset(samples, batch_size=8, max_len=160)
    model = build_model(input_shape=(160,40), num_classes=len(labels_map))
    model.load_weights(base_model_path)
    # freeze early layers
    for layer in model.layers[:6]:
        layer.trainable = False
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(ds, epochs=10)
    os.makedirs('models', exist_ok=True)
    out_path = f'models/finetuned_{user_name}.h5'
    model.save(out_path)
    print("Saved fine-tuned model to:", out_path)

if __name__ == '__main__':
    # Example: python src/finetune.py user_profiles/user_hero hero
    import sys
    if len(sys.argv) >= 3:
        user_folder = sys.argv[1]
        user_name = sys.argv[2]
        finetune_user(user_folder, base_model_path='models/baseline.h5', user_name=user_name)
    else:
        print("Usage: python src/finetune.py <user_folder> <user_name>")
