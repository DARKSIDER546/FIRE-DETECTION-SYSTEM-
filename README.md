# Fire & Smoke Detection - Demo Project

This demo project contains scripts to train and run a simple Fire vs Smoke classifier using TensorFlow/Keras.
It includes:
- `train.py` : Train a small CNN on the `data/sample/` synthetic images.
- `detect_image.py` : Run prediction on a single image.
- `detect_webcam.py` : Real-time webcam detection.
- `app/flask_app.py` : Simple Flask web app to upload images and get predictions.
- `gui_tkinter.py` : Simple Tkinter GUI for desktop usage.
- `requirements.txt` : Python dependencies.
- `models/` : Saved model will be placed here after training.

## How to use (quick)
1. Create a Python virtual environment and install requirements:
   ```bash
   python -m venv venv
   source venv/bin/activate   # on Windows use: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. Train the demo model (this uses the tiny synthetic dataset included):
   ```bash
   python train.py
   ```

3. Test on an image:
   ```bash
   python detect_image.py data/sample/fire/fire_1.jpg
   ```

4. Real-time webcam:
   ```bash
   python detect_webcam.py
   ```

5. Run the Flask app:
   ```bash
   cd app
   python flask_app.py
   ```

6. GUI:
   ```bash
   python gui_tkinter.py
   ```

## Notes & Next Steps (for a production-quality project)
- Replace the small synthetic dataset with a larger labeled dataset (Kaggle Fire/Smoke datasets).
- Use transfer learning (MobileNetV2, EfficientNet) for better accuracy and faster convergence.
- Add object detection (YOLOv5/YOLOv8 or SSD) to localize fire/smoke regions, not just classify whole image.
- Improve robustness with hard negative mining, more data augmentation, and validation on real CCTV footage.
- Add logging, alerting (email/SMS), and integration with cloud storage for production deployment.