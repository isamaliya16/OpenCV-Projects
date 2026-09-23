import os
import cv2
import numpy as np
import config

path = os.path.join(config.DATASET_DIR, config.PERSON_DIR_NAME)

if not hasattr(cv2, "face"):
    raise SystemExit("Install opencv-contrib-python.")

if not os.path.isdir(path):
    raise SystemExit(f"Dataset not found: {path}")

faces = []

for file in os.listdir(path):
    if file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
        img = cv2.imread(os.path.join(path, file), 0)
        if img is not None:
            faces.append(img)

if not faces:
    raise SystemExit("No valid face images found.")

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(
    faces,
    np.full(len(faces), config.PERSON_LABEL, dtype=np.int32)
)

os.makedirs(config.TRAINER_DIR, exist_ok=True)
recognizer.write(config.MODEL_PATH)

print(f"Model saved: {config.MODEL_PATH}")
print(f"Images used: {len(faces)}")