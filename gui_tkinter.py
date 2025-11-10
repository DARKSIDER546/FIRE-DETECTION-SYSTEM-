"""
gui_tkinter.py
Simple Tkinter GUI to load an image and display prediction.
"""
import tkinter as tk
from tkinter import filedialog, Label
from PIL import Image, ImageTk
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

MODEL_PATH = "models/fire_smoke_cnn.h5"

def predict(path):
    if not os.path.exists(MODEL_PATH):
        return "Model not found. Run train.py first."
    model = load_model(MODEL_PATH)
    img = image.load_img(path, target_size=(128,128))
    arr = image.img_to_array(img)/255.0
    arr = np.expand_dims(arr, axis=0)
    pred = model.predict(arr)[0][0]
    return "FIRE" if pred>=0.5 else "SMOKE"

def open_file():
    path = filedialog.askopenfilename()
    if not path:
        return
    img = Image.open(path).resize((300,300))
    tkimg = ImageTk.PhotoImage(img)
    img_label.config(image=tkimg)
    img_label.image = tkimg
    result = predict(path)
    result_label.config(text=f"Prediction: {result}")

root = tk.Tk()
root.title("Fire/Smoke Detection GUI")
btn = tk.Button(root, text="Open Image", command=open_file)
btn.pack()
img_label = Label(root)
img_label.pack()
result_label = Label(root, text="Prediction: -")
result_label.pack()
root.mainloop()