"""Detection core for VisionScan: YOLO26 objects + OpenCV YuNet faces.

Everything that touches models, images or result files lives here, so app.py can
stay a thin web layer and the logic can be unit-tested without PyTorch
(Ultralytics is imported lazily, only when the object model is first needed).
"""
from __future__ import annotations

import http.client
import io
import json
import logging
import os
import re
import threading
import time
import urllib.request
import uuid
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger("visionscan.detector")

# --------------------------------------------------------------------------- config
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = Path(os.getenv("MODEL_DIR") or BASE_DIR / "models")


def _env_num(name, default, cast):
    try:
        return cast(os.environ[name])
    except (KeyError, ValueError):
        return default


OBJECT_MODEL = os.getenv("VISION_MODEL", "yolo26m.pt")
DEFAULT_OBJECT_CONF = _env_num("OBJECT_CONF", 0.25, float)
DEFAULT_FACE_CONF = _env_num("FACE_CONF", 0.75, float)
IMGSZ = _env_num("IMGSZ", 640, int)
MAX_SIDE = _env_num("MAX_IMAGE_SIDE", 2560, int)  # bigger images are scaled down
MAX_PIXELS = _env_num("MAX_IMAGE_PIXELS", 40_000_000, int)  # rejects decompression bombs
FACE_MAX_SIDE = _env_num("FACE_MAX_SIDE", 1280, int)  # faces are searched at this size

YUNET_FILE = "face_detection_yunet_2023mar.onnx"
YUNET_PATH = MODEL_DIR / YUNET_FILE
YUNET_DEFAULT_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/" + YUNET_FILE
)
DOWNLOAD_TIMEOUT = 20  # seconds without data before a download is abandoned

# Colours are written as RGB (what a designer expects) and converted for OpenCV.
# They match the legend dots in static/css/style.css (.dot.object / .dot.face).
OBJECT_RGB = (230, 160, 0)
FACE_RGB = (21, 148, 71)
HEADER_BG_RGB = (245, 248, 252)
HEADER_TEXT_RGB = (35, 45, 55)
FONT = cv2.FONT_HERSHEY_SIMPLEX


class DetectionError(Exception):
    """A problem with the uploaded image. The message is safe to show to users."""


class ModelUnavailable(RuntimeError):
    """A model could not be loaded. The message is safe to show to users
    (it never contains file-system paths); details go to the server log."""


# --------------------------------------------------------------------------- model loading
class LazyModel:
    """Load a model once, thread-safely, on first use or from a warm-up thread.

    After a failure the error is remembered for `retry_after` seconds, so a broken
    model does not make every request wait for another failing download."""

    def __init__(self, name, loader, retry_after=30.0):
        self.name = name
        self._loader = loader
        self._retry_after = retry_after
        self._lock = threading.Lock()
        self._obj = None
        self._failed_at = 0.0
        self.state = "idle"  # idle | loading | ready | error
        self.error = None

    def get(self):
        if self._obj is not None:
            return self._obj
        with self._lock:
            if self._obj is not None:  # another thread finished while we waited
                return self._obj
            if self.state == "error" and time.monotonic() - self._failed_at < self._retry_after:
                raise ModelUnavailable(self.error)
            self.state = "loading"
            try:
                self._obj = self._loader()
            except ModelUnavailable as exc:
                self._fail(str(exc))
                raise
            except Exception as exc:
                log.exception("Loading the %s model failed", self.name)
                self._fail("unexpected error while loading the model (see the server log)")
                raise ModelUnavailable(self.error) from exc
            self.state, self.error = "ready", None
            return self._obj

    def _fail(self, message):
        self.state, self.error, self._failed_at = "error", message, time.monotonic()
        log.warning("%s model unavailable: %s", self.name, message)

    def status(self):
        return {"state": self.state, "error": self.error}


def _download(url, dest, timeout=DOWNLOAD_TIMEOUT, total_timeout=180, min_bytes=100_000):
    """Download `url` to `dest` safely.

    The data goes to a temporary file that is moved into place only when it is
    complete, so an interrupted download can never leave a broken model behind.
    Raises OSError on any problem."""
    dest = Path(dest)
    tmp = dest.with_name(dest.name + ".part")
    deadline = time.monotonic() + total_timeout
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "VisionScan/2.0"})
        with urllib.request.urlopen(request, timeout=timeout) as resp, open(tmp, "wb") as out:
            expected = resp.headers.get("Content-Length")
            received, head = 0, b""
            while True:
                chunk = resp.read(64 * 1024)
                if not chunk:
                    break
                if not head:
                    head = chunk[:64]
                out.write(chunk)
                received += len(chunk)
                if time.monotonic() > deadline:
                    raise TimeoutError("download took too long")
        if expected and received != int(expected):
            raise OSError(f"download incomplete ({received} of {expected} bytes)")
        if head.startswith(b"version https://git-lfs"):
            raise OSError("server sent a Git-LFS pointer instead of the model file")
        if received < min_bytes:
            raise OSError(f"downloaded file is too small ({received} bytes)")
        os.replace(tmp, dest)
    except http.client.HTTPException as exc:  # e.g. IncompleteRead
        raise OSError(f"download failed: {exc}") from exc
    finally:
        tmp.unlink(missing_ok=True)  # no-op after a successful replace


class FaceDetector:
    """Thin, thread-safe wrapper around cv2.FaceDetectorYN."""

    def __init__(self, path):
        self._det = cv2.FaceDetectorYN.create(str(path), "", (320, 320), 0.75, 0.3, 5000)
        self._lock = threading.Lock()

    def detect(self, image, score_threshold):
        """Return an (N, 15) array: x, y, w, h, 5 landmarks (x, y), score."""
        height, width = image.shape[:2]
        with self._lock:  # setInputSize + detect must not interleave between threads
            self._det.setInputSize((width, height))
            self._det.setScoreThreshold(float(score_threshold))
            _, faces = self._det.detect(image)
        return np.empty((0, 15), np.float32) if faces is None else faces


def _validate_yunet(path):
    """Raise cv2.error if `path` is not a loadable YuNet model."""
    cv2.FaceDetectorYN.create(str(path), "", (320, 320), 0.75, 0.3, 5000)


def ensure_face_model(force=False):
    """Return the path of the YuNet model, downloading it first if it is missing."""
    if YUNET_PATH.exists() and not force:
        return YUNET_PATH
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    problems = []
    for url in [u for u in (os.getenv("YUNET_URL"), YUNET_DEFAULT_URL) if u]:
        try:
            log.info("Downloading the face model from %s", url)
            _download(url, YUNET_PATH)
            _validate_yunet(YUNET_PATH)
            return YUNET_PATH
        except Exception as exc:  # noqa: BLE001 - any failure means "try the next source"
            YUNET_PATH.unlink(missing_ok=True)
            problems.append(f"{url}: {exc}")
    log.error("Could not get the face model:\n  %s", "\n  ".join(problems))
    raise ModelUnavailable(
        "the face model could not be downloaded (check the internet connection, or copy "
        f"{YUNET_FILE} into the models folder - see the README)"
    )


def _load_face_detector():
    for attempt in (1, 2):
        path = ensure_face_model()
        try:
            return FaceDetector(path)
        except cv2.error:
            # A corrupt file, e.g. left over from an interrupted download in an older
            # version: delete it so the next pass downloads a fresh copy.
            log.warning("Face model %s is corrupt; deleting it (attempt %d)", path, attempt)
            path.unlink(missing_ok=True)
    raise ModelUnavailable("the face model file was corrupt and could not be replaced")


def _resolve_weights(name):
    """Look for the YOLO weights next to the app, not in the current directory."""
    p = Path(name)
    if p.is_absolute():
        return str(p)
    for folder in (MODEL_DIR, BASE_DIR):
        if (folder / p.name).is_file():
            return str(folder / p.name)
    return name  # a well-known name such as yolo26n.pt: Ultralytics downloads it


def _load_object_model():
    try:
        from ultralytics import YOLO  # lazy: slow import, and not needed by the tests
    except ImportError as exc:
        raise ModelUnavailable(
            "the 'ultralytics' package is not installed (run: pip install -r requirements.txt)"
        ) from exc
    weights = _resolve_weights(OBJECT_MODEL)
    log.info("Loading object model: %s", weights)
    try:
        return YOLO(weights)
    except Exception as exc:  # noqa: BLE001
        log.exception("Could not load the YOLO weights %s", weights)
        raise ModelUnavailable(
            f"the YOLO weights '{Path(OBJECT_MODEL).name}' could not be loaded "
            "(missing file or no internet connection - see the server log)"
        ) from exc


object_model = LazyModel("object", _load_object_model)
face_model = LazyModel("face", _load_face_detector)


def models_status():
    return {
        "object_model": {**object_model.status(), "weights": Path(OBJECT_MODEL).name},
        "face_model": face_model.status(),
    }


def warm_up():
    """Load both models. Runs in a background thread when the server starts."""
    for model in (object_model, face_model):
        try:
            model.get()
        except ModelUnavailable:
            pass  # already logged; the UI shows the state


# --------------------------------------------------------------------------- image helpers
def decode_image(data):
    """Turn uploaded bytes into a BGR image, or raise DetectionError."""
    if not data:
        raise DetectionError("The uploaded file is empty.")
    try:
        from PIL import Image

        with Image.open(io.BytesIO(data)) as im:  # reads the header only
            width, height = im.size
    except ImportError:
        width = height = 0  # Pillow missing: skip the cheap size pre-check
    except Exception:  # noqa: BLE001
        raise DetectionError("The uploaded file is not a valid image.") from None
    if width * height > MAX_PIXELS:
        raise DetectionError(
            f"The image is too large ({width}x{height} px). "
            f"The limit is {MAX_PIXELS // 1_000_000} megapixels."
        )
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)  # honours EXIF rotation
    if img is None:
        raise DetectionError("The uploaded file could not be read as an image.")
    return _limit_side(img, MAX_SIDE)


def _limit_side(img, max_side):
    height, width = img.shape[:2]
    longest = max(height, width)
    if longest <= max_side:
        return img
    scale = max_side / longest
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


def clamp_box(x1, y1, x2, y2, width, height):
    """Clip a box that may stick out of the image. Returns [x1, y1, x2, y2] or None.

    The far corner must be computed *before* clipping (x2 = x + w), otherwise a
    face cut off by the image edge gets a box that is too big."""
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    cx1 = max(0, min(width - 1, int(round(x1))))
    cy1 = max(0, min(height - 1, int(round(y1))))
    cx2 = max(0, min(width - 1, int(round(x2))))
    cy2 = max(0, min(height - 1, int(round(y2))))
    if cx2 <= cx1 or cy2 <= cy1:
        return None
    return [cx1, cy1, cx2, cy2]


def _np(value):
    return value.cpu().numpy() if hasattr(value, "cpu") else np.asarray(value)


# --------------------------------------------------------------------------- detection
def _detect_objects(img, conf):
    model = object_model.get()  # may raise ModelUnavailable
    # Ultralytics serialises predict() calls internally, so sharing one model is safe.
    result = model.predict(source=img, conf=conf, imgsz=IMGSZ, verbose=False)[0]
    if result.boxes is None:
        return []
    xyxy = _np(result.boxes.xyxy).reshape(-1, 4)
    scores = _np(result.boxes.conf).reshape(-1)
    classes = _np(result.boxes.cls).reshape(-1).astype(int)
    height, width = img.shape[:2]
    found = []
    for box, score, cls in zip(xyxy, scores, classes):
        clipped = clamp_box(*box, width, height)
        if clipped is not None:
            found.append(
                {"type": "object", "label": str(result.names[int(cls)]),
                 "confidence": float(score), "box": clipped}
            )
    return found


def _detect_faces(img, conf):
    detector = face_model.get()  # may raise ModelUnavailable
    height, width = img.shape[:2]
    scale, probe = 1.0, img
    if max(height, width) > FACE_MAX_SIDE:  # YuNet gets very slow on huge images
        scale = FACE_MAX_SIDE / max(height, width)
        probe = cv2.resize(
            img, (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    found = []
    for row in detector.detect(probe, conf):
        x, y, w, h = (float(v) / scale for v in row[:4])
        clipped = clamp_box(x, y, x + w, y + h, width, height)
        if clipped is not None:
            found.append(
                {"type": "face", "label": "face", "confidence": float(row[-1]), "box": clipped}
            )
    return found


# --------------------------------------------------------------------------- drawing
def _bgr(rgb):
    return (rgb[2], rgb[1], rgb[0])


def _ink(rgb):
    """Readable label text colour (RGB) for a given chip colour."""
    luminance = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
    return (23, 33, 43) if luminance > 150 else (255, 255, 255)


def _draw_detection(canvas, det):
    height, width = canvas.shape[:2]
    longest = max(height, width)
    line = max(2, round(longest / 400))  # stroke and text scale with the image
    font_scale = max(0.6, longest / 1100)
    text_thickness = max(1, round(font_scale * 2))
    rgb = FACE_RGB if det["type"] == "face" else OBJECT_RGB
    x1, y1, x2, y2 = det["box"]
    cv2.rectangle(canvas, (x1, y1), (x2, y2), _bgr(rgb), line, cv2.LINE_AA)

    text = f'{det["label"]} {det["confidence"]:.0%}'
    (tw, th), baseline = cv2.getTextSize(text, FONT, font_scale, text_thickness)
    pad = max(3, line + 1)
    chip_w, chip_h = tw + 2 * pad, th + baseline + 2 * pad
    cx = max(0, min(x1, width - chip_w))  # keep the label inside the image
    cy = y1 - chip_h
    if cy < 0:  # no room above the box: put the label just inside its top edge
        cy = y1
    cy = max(0, min(cy, height - chip_h))
    cv2.rectangle(canvas, (cx, cy), (cx + chip_w, cy + chip_h), _bgr(rgb), -1)
    cv2.putText(canvas, text, (cx + pad, cy + pad + th), FONT, font_scale,
                _bgr(_ink(rgb)), text_thickness, cv2.LINE_AA)


def header_layout(texts, width, height):
    """Pick the first header text that fits `width`, shrinking the font if needed."""
    margin = max(8, width // 60)
    base_scale = max(0.6, max(width, height) / 1300)
    thickness = max(1, round(base_scale * 2))
    fallback = None
    for text in texts:
        scale = base_scale
        while scale >= 0.4:
            (tw, th), baseline = cv2.getTextSize(text, FONT, scale, thickness)
            fallback = (text, scale, thickness, tw, th, baseline, margin)
            if tw <= width - 2 * margin:
                return fallback
            scale -= 0.05
    return fallback  # even the shortest text does not fit (tiny image): it is clipped


def _add_header(img, texts):
    """Add a summary strip ABOVE the photo, so it never hides any detections."""
    height, width = img.shape[:2]
    text, scale, thickness, _, th, baseline, margin = header_layout(texts, width, height)
    strip = th + baseline + 2 * margin
    canvas = cv2.copyMakeBorder(img, strip, 0, 0, 0, cv2.BORDER_CONSTANT, value=_bgr(HEADER_BG_RGB))
    cv2.putText(canvas, text, (margin, margin + th), FONT, scale,
                _bgr(HEADER_TEXT_RGB), thickness, cv2.LINE_AA)
    return canvas


# --------------------------------------------------------------------------- results on disk
_ID_RE = re.compile(r"^[0-9a-f]{32}$")


def valid_id(rid):
    return isinstance(rid, str) and bool(_ID_RE.match(rid))


def _write_atomic(path, payload):
    tmp = path.with_name(path.name + ".part")
    try:
        tmp.write_bytes(payload)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def detect_image(data, original_name, outdir, object_conf=None, face_conf=None):
    """Run both detectors on the uploaded bytes and store the annotated result.

    Only the annotated JPEG and a small JSON file are written - the original
    upload is never stored. Returns the result metadata (see load_result)."""
    started = time.perf_counter()
    object_conf = DEFAULT_OBJECT_CONF if object_conf is None else object_conf
    face_conf = DEFAULT_FACE_CONF if face_conf is None else face_conf
    img = decode_image(data)
    height, width = img.shape[:2]

    try:
        objects = _detect_objects(img, object_conf)
    except ModelUnavailable as exc:
        raise DetectionError(f"Object detection is unavailable: {exc}.") from exc

    warnings = []
    try:
        faces = _detect_faces(img, face_conf)
    except ModelUnavailable as exc:  # degrade gracefully: keep the object results
        faces = []
        warnings.append(f"Face detection is unavailable: {exc}. Showing object results only.")

    detections = sorted(objects + faces, key=lambda d: d["confidence"], reverse=True)
    canvas = img.copy()
    for det in reversed(detections):  # draw the most confident boxes last (on top)
        _draw_detection(canvas, det)
    canvas = _add_header(canvas, [
        f"Scan complete | Objects: {len(objects)} | Faces: {len(faces)}",
        f"Objects {len(objects)} | Faces {len(faces)}",
    ])

    ok, encoded = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise RuntimeError("could not encode the result image")

    rid = uuid.uuid4().hex
    meta = {
        "id": rid,
        "original_name": original_name,
        "image_width": int(canvas.shape[1]),
        "image_height": int(canvas.shape[0]),
        "object_count": len(objects),
        "face_count": len(faces),
        "total": len(objects) + len(faces),
        "summary": [{"label": k, "count": v}
                    for k, v in Counter(d["label"] for d in objects).most_common()],
        "detections": detections,
        "warnings": warnings,
        "object_conf": object_conf,
        "face_conf": face_conf,
        "created": time.time(),
        "elapsed_ms": int((time.perf_counter() - started) * 1000),
    }
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    _write_atomic(outdir / f"{rid}.jpg", encoded.tobytes())
    _write_atomic(outdir / f"{rid}.json", json.dumps(meta).encode("utf-8"))  # written last
    return meta


def load_result(outdir, rid):
    """Return the stored metadata for a result id, or None if unknown/expired."""
    if not valid_id(rid):
        return None
    outdir = Path(outdir)
    try:
        meta = json.loads((outdir / f"{rid}.json").read_text("utf-8"))
    except (OSError, ValueError):
        return None
    return meta if (outdir / f"{rid}.jpg").is_file() else None


def delete_result(outdir, rid):
    if not valid_id(rid):
        return
    for suffix in (".jpg", ".json"):
        (Path(outdir) / f"{rid}{suffix}").unlink(missing_ok=True)


def cleanup_results(outdir, max_age_seconds):
    """Delete results (and stray temp files) older than `max_age_seconds`."""
    now, removed = time.time(), 0
    for path in Path(outdir).glob("*"):
        if path.suffix not in (".jpg", ".json", ".part"):
            continue
        try:
            if now - path.stat().st_mtime > max_age_seconds:
                path.unlink()
                removed += 1
        except OSError:
            pass  # file vanished or is in use: try again next time
    return removed
