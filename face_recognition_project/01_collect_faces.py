import os
import time
import cv2
import config

os.makedirs(
    f"{config.DATASET_DIR}/{config.PERSON_DIR_NAME}",
    exist_ok=True
)

cascade = config.CASCADE_PATH
if not os.path.isfile(cascade):
    cascade = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

detector = cv2.CascadeClassifier(cascade)
cap = cv2.VideoCapture(config.CAMERA_INDEX)

count, last = 0, 0

while count < config.TARGET_SAMPLES:
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

    if len(faces):
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

        if time.time() - last > .2:
            face = cv2.resize(gray[y:y+h, x:x+w], config.FACE_SIZE)
            count += 1
            cv2.imwrite(
                f"{config.DATASET_DIR}/{config.PERSON_DIR_NAME}/"
                f"{config.PERSON_DIR_NAME}_{count:03}.jpg",
                face
            )
            last = time.time()

        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

    cv2.putText(
        frame, f"{count}/{config.TARGET_SAMPLES}",
        (10,30), cv2.FONT_HERSHEY_SIMPLEX, .8, (0,255,0), 2
    )

    cv2.imshow("Face Collection", frame)

    if cv2.waitKey(1) & 255 in (ord("q"), ord("Q")):
        break

cap.release()
cv2.destroyAllWindows()