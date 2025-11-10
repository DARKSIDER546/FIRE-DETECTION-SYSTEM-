import tensorflow as tf
from tensorflow.keras import layers, models
import os

DATA_DIR = "data/sample"
IMG_SIZE = (128,128)
BATCH = 8
EPOCHS = 10
MODEL_PATH = "models/fire_smoke_cnn.h5"

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATA_DIR,
    labels="inferred",
    label_mode="binary",
    image_size=IMG_SIZE,
    batch_size=BATCH,
    shuffle=True,
    validation_split=0.2,
    subset="training",
    seed=123
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATA_DIR,
    labels="inferred",
    label_mode="binary",
    image_size=IMG_SIZE,
    batch_size=BATCH,
    shuffle=True,
    validation_split=0.2,
    subset="validation",
    seed=123
)

normalization = layers.Rescaling(1./255)
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
])

model = models.Sequential([
    layers.Input(shape=(*IMG_SIZE,3)),
    data_augmentation,
    normalization,
    layers.Conv2D(16,3,activation='relu'),
    layers.MaxPool2D(),
    layers.Conv2D(32,3,activation='relu'),
    layers.MaxPool2D(),
    layers.Conv2D(64,3,activation='relu'),
    layers.MaxPool2D(),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)
os.makedirs("models", exist_ok=True)
model.save(MODEL_PATH)
print("Saved model to", MODEL_PATH)