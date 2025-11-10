import tensorflow as tf
import cv2
import numpy as np
import os

MODEL_PATH = "models/fire_smoke_cnn.h5"
IMG_SIZE = (128, 128)
THRESHOLD = 0.3  

def main():
    if not os.path.exists(MODEL_PATH):
        print("Model not found at:", MODEL_PATH)
        return
    print("Loading model from:", MODEL_PATH)
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully!")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Webcam not found.")
        return
    print("Webcam started... Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame.")
            break

        img = cv2.resize(frame, IMG_SIZE)
        x = np.expand_dims(img, axis=0) / 255.0 

        pred = model.predict(x)
        confidence = pred[0][0]

        if confidence > THRESHOLD:
            label = f"Fire/Smoke ({confidence:.2f})"
            color = (0, 0, 255) 
        else:
            label = f"Normal ({confidence:.2f})"
            color = (0, 255, 0)  

        print("Prediction:", confidence)

        cv2.putText(frame, label, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
        cv2.imshow("Fire & Smoke Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Detection stopped.")

if __name__ == "__main__":
    main()
