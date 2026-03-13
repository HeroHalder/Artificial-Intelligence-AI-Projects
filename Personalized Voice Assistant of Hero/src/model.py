# model.py
from tensorflow.keras import layers, models

def build_model(input_shape=(160,40), num_classes=10):
    inp = layers.Input(shape=input_shape)
    x = layers.Reshape((*input_shape,1))(inp)  # (160,40,1)
    x = layers.Conv2D(32,(3,3),activation='relu',padding='same')(x)
    x = layers.MaxPool2D((2,2))(x)
    x = layers.Conv2D(64,(3,3),activation='relu',padding='same')(x)
    x = layers.MaxPool2D((2,2))(x)
    x = layers.Conv2D(128,(3,3),activation='relu',padding='same')(x)
    x = layers.MaxPool2D((2,2))(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(num_classes, activation='softmax')(x)
    model = models.Model(inp, out)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model
