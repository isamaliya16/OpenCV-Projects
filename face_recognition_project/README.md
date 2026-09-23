# Face Recognition Project (Python + OpenCV)

Local, real-time face recognition. No cloud APIs, no external services.

- **Detection:** Haar Cascade
- **Recognition:** LBPH (Local Binary Patterns Histograms)

Two ways to use this project:
- **CLI scripts** (below) — single person, capture/train/recognize from the command line.
- **Web app** ([jump to section](#web-app-multi-user-registration--sign-in-with-your-face)) — any number of people can register through the browser and sign in with their face.

## Project Structure

```
face_recognition_project/
│
├── dataset/
│   └── ayush/              # your captured face images (auto-created)
│
├── trainer/
│   └── face_model.yml      # trained model (auto-created)
│
├── 01_collect_faces.py     # Step 1: capture face images from webcam
├── 02_train_model.py       # Step 2: train the LBPH model
├── 03_face_recognition.py  # Step 3: real-time recognition
├── config.py                # all settings in one place
└── requirements.txt
```

## 1. Install dependencies (Windows PowerShell)

```powershell
pip uninstall opencv-python opencv-contrib-python -y
pip install opencv-contrib-python numpy
```

`opencv-python` and `opencv-contrib-python` conflict when both are installed — the uninstall step prevents that.

## 2. Verify the install

```powershell
python -c "import cv2; print(cv2.__version__); print(hasattr(cv2, 'face'))"
```

The second line **must print `True`**. That confirms `cv2.face` (the LBPH recognizer) is available. If it prints `False`, re-run step 1 — `opencv-python` is likely still installed and shadowing `opencv-contrib-python`.

## 3. Collect your face images

```powershell
python 01_collect_faces.py
```

Look at the camera and move your head slightly (angle, expression) for variety. It stops automatically at 30 images, or press `Q` to stop early. Images are saved to `dataset/ayush/`.

## 4. Train the model

```powershell
python 02_train_model.py
```

Trains an LBPH recognizer on your images and saves it to `trainer/face_model.yml`.

## 5. Run real-time recognition

```powershell
python 03_face_recognition.py
```

Draws a bounding box around each detected face and shows:
- **Ayush Isamaliya** — distance ≤ threshold (match)
- **Unknown** — distance > threshold (no match)

Press `Q` to quit.

## How the threshold works

`RECOGNITION_THRESHOLD` in `config.py` (default `70`) is the cutoff on LBPH's distance score, where **lower = more confident**:

- distance ≤ threshold → labeled `Ayush Isamaliya`
- distance > threshold → labeled `Unknown`

Lower the threshold for stricter matching; raise it if you're being labeled Unknown too often.

## Troubleshooting

| Problem | Fix |
|---|---|
| `hasattr(cv2, 'face')` prints `False` | Re-run the uninstall/reinstall commands in step 1 |
| Webcam won't open | Close other apps using the camera; try `CAMERA_INDEX = 1` in `config.py` |
| No face detected | Improve lighting; face the camera more directly |
| You're labeled Unknown | Raise `RECOGNITION_THRESHOLD`, or re-collect images with more angle/lighting variety |
| Strangers labeled with your name | Lower `RECOGNITION_THRESHOLD` |

---

## Web app: multi-user registration + sign in with your face

The web app doesn't need steps 1-2 above — it has its own in-browser
registration flow that captures face samples and trains the model for you.
(If you *have* already run the CLI scripts, your existing `dataset/ayush/`
and `trainer/face_model.yml` are automatically imported as the first
account on first launch, so nothing is lost.)

```bash
pip install -r requirements.txt
python run_web.py
```

Open **http://localhost:5000** (camera access needs `localhost` or HTTPS —
your browser won't grant it over a plain LAN IP).

- **New account:** click **Create an account**, pick a username and enter
  your name, then capture ~20 face photos as you move your head slightly.
  If the face looks like it's already registered under another account,
  you'll get a warning before anything is saved.
- **Sign in:** press **Start camera** and hold still. The server checks a
  few consecutive frames against *every* registered account — once one
  account matches enough times in a row, you're signed in and taken to a
  dashboard showing the match score and recent sign-in activity for your
  account specifically.

Every account you register is folded into the same trained model, so
multiple people can register and each will only ever be recognised as
themselves.

Design notes, the full registration/recognition/session/lockout pipeline,
and this system's real limitations (no liveness detection — a photo of an
enrolled user's face would still pass) are all in
**[`docs/SYSTEM_DESIGN.md`](docs/SYSTEM_DESIGN.md)**.

Run the test suite (63 tests, including full registration→login flows
against real face images) with:

```bash
pytest tests/ -q
```
