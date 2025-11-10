"""
detect_image.py
Loads the trained model and runs detection on a single image.
Usage: python detect_image.py path/to/image.jpg
"""
import sys, os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

MODEL_PATH = "models/fire_smoke_cnn.h5"

def predict(img_path):
    if not os.path.exists(MODEL_PATH):
        print("Model not found. Train model first: python train.py")
        return
    model = load_model(MODEL_PATH)
    img = image.load_img(img_path, target_size=(128,128))
    arr = image.img_to_array(img)/255.0
    arr = np.expand_dims(arr, axis=0)
    pred = model.predict(arr)[0][0]
    label = "fire" if pred>=0.5 else "smoke"
    print(f"Prediction: {label} (score={pred:.3f})")

if __name__ == "__main__":
    if len(sys.argv)<2:
        print("Usage: python detect_image.py path/to/image.jpg")
    else:
        predict(sys.argv[1])