import os
import cv2
import config

if not hasattr(cv2, "face"):
    raise SystemExit("Install opencv-contrib-python.")

if not os.path.isfile(config.MODEL_PATH):
    raise SystemExit("Model not found. Train the model first.")

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(config.MODEL_PATH)

cascade = config.CASCADE_PATH
if not os.path.isfile(cascade):
    cascade = r"D:\OpenCV\face_recognition_project\haarcascade_frontalface_default.xml"

detector = cv2.CascadeClassifier(cascade)
cap = cv2.VideoCapture(config.CAMERA_INDEX)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector.detectMultiScale(
        gray,
        config.SCALE_FACTOR,
        config.MIN_NEIGHBORS,
        minSize=config.MIN_FACE_SIZE
    )

    for x, y, w, h in faces:
        face = cv2.resize(gray[y:y+h, x:x+w], config.FACE_SIZE)
        label, distance = recognizer.predict(face)

        known = (
            label == config.PERSON_LABEL
            and distance <= config.RECOGNITION_THRESHOLD
        )

        name = config.PERSON_NAME if known else "Unknown"
        color = (0, 255, 0) if known else (0, 0, 255)

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(
            frame, f"{name} {distance:.1f}",
            (x, y+h+25),
            cv2.FONT_HERSHEY_SIMPLEX, .7, color, 2
        )

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
        break

cap.release()
cv2.destroyAllWindows()