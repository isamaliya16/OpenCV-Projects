# VisionScan - object and face detection

Upload a photo and get every detected **object** (YOLO26, 80 everyday classes) and **face**
(OpenCV YuNet), drawn on the image with a confidence score.

- Drag and drop or pick a JPG, JPEG, PNG or WEBP file (up to 15 MB)
- Adjustable confidence for objects and faces
- Annotated image, per-class summary and a list of every detection
- Download the result or delete it right away
- Works offline once the models are on disk

## Quick start

You need **Python 3.9 or newer** (3.10 to 3.12 recommended).

**Windows (PowerShell)**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python download_models.py
python app.py
```

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python download_models.py
python app.py
```

Then open <http://127.0.0.1:5000>.

If PowerShell refuses to run `Activate.ps1`, run
`Set-ExecutionPolicy -Scope Process RemoteSigned` once in that window, or use `.venv\Scripts\activate.bat` from `cmd`.

`python download_models.py` is optional but recommended: it fetches the small face model now (about 230 KB) so the
first scan does not have to. It prints exactly what to do if the download is blocked.
The status badge in the top-right corner shows whether the models are loaded.

## Settings

Everything has a sensible default. Change a setting with an environment variable before starting the app.

| Variable | Default | Meaning |
|----------|---------|---------|
| `VISION_MODEL` | `yolo26m.pt` | YOLO weights: `yolo26n.pt` (fastest), `yolo26s.pt`, `yolo26m.pt`, `yolo26l.pt` (most accurate) |
| `OBJECT_CONF` / `FACE_CONF` | `0.25` / `0.75` | Default confidence for objects / faces (also adjustable on the page) |
| `MAX_UPLOAD_MB` | `15` | Largest accepted upload |
| `MAX_IMAGE_SIDE` | `2560` | Larger photos are scaled down to this many pixels on the long side |
| `MAX_IMAGE_PIXELS` | `40000000` | Photos with more pixels than this are rejected |
| `RESULT_TTL_MINUTES` | `60` | Results are deleted automatically after this time |
| `HOST` / `PORT` | `127.0.0.1` / `5000` | Where the server listens. Use `HOST=0.0.0.0` to allow other devices on your network |
| `SECRET_KEY` | random | Set it to keep sessions valid across restarts |
| `DEBUG` | `0` | `1` turns on Flask's debug mode. Never use it on a shared network |
| `MODEL_DIR` | `./models` | Folder that holds the model files |
| `YUNET_URL` | GitHub | Alternative download address for the face model |
| `WARMUP` | `1` | `0` skips loading the models in the background at start-up |
| `LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, ... |

```powershell
# Windows PowerShell
$env:VISION_MODEL = "yolo26n.pt"; python app.py
```
```bash
# macOS / Linux
VISION_MODEL=yolo26n.pt python app.py
```

## Running for real users

`python app.py` uses Flask's development server. To serve other people, use a production server:

```bash
pip install waitress
waitress-serve --listen=127.0.0.1:5000 app:app
```

Put it behind HTTPS (for example a reverse proxy) and set `SECRET_KEY`.

## Privacy

The original upload is decoded in memory and **never written to disk**. Only the annotated result is stored in
`results/`, under a random address, and it is deleted automatically after `RESULT_TTL_MINUTES` (or immediately with
the *Delete result* button).

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest
```

The tests use fake models, so they need no internet, no GPU and no model files.

## Project layout

```
app.py               Flask app: routes, upload checks, security
detector.py          Model loading, detection, drawing, result storage
download_models.py   Fetches and checks the model files
models/              yolo26m.pt (included) and the face model (downloaded)
results/             Annotated results (auto-deleted)
static/              CSS and JavaScript
templates/index.html The page
tests/               Automated tests
```

## Troubleshooting

| Problem | What to do |
|---------|------------|
| Badge says *Ready, face detection unavailable* | The face model could not be downloaded. Run `python download_models.py` and follow what it prints. Object detection keeps working meanwhile. |
| Badge says *Object model unavailable* | Run `pip install -r requirements.txt`, and check that `models/yolo26m.pt` exists. |
| The first scan is slow | Models load once at start-up; wait for the badge to say *ready*. CPU-only PCs can use `VISION_MODEL=yolo26n.pt`. |
| "Address already in use" | Another program uses port 5000. Start with `PORT=5001` (`$env:PORT = "5001"` in PowerShell). |
| Face boxes look too strict or too loose | Change *Face confidence* under *Detection settings*. |
| Details of any failure | Read the console window where you started the app; errors are logged there, not shown on the page. |

## Licence note

Ultralytics YOLO and the YOLO26 weights are licensed under **AGPL-3.0** (a commercial licence is available from
Ultralytics). If you offer this app to other people over a network, AGPL obligations apply to you. The YuNet face
model comes from the OpenCV Model Zoo (MIT licence). This is not legal advice.
