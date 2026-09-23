"""Face detection + multi-user verification + (re)training.

Same primitives as the original CLI scripts — Haar cascade to find the face,
grayscale crop resized to FACE_SIZE, LBPH ``predict``/``train`` — packaged so
the web server can call them per camera frame and retrain in place whenever a
new user finishes registering.

Preprocessing is deliberately identical to ``01_collect_faces.py`` (no
histogram equalisation etc.) because that's what the model was trained on.
"""
from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .settings import Settings

MAX_DETECT_WIDTH = 640  # frames wider than this are scaled down (matches typical webcam size)


@dataclass
class Detection:
    box: tuple          # (x, y, w, h) in pixels, on the (possibly downscaled) frame that was analysed
    confidence: float    # Haar reject-level weight; higher = more face-like


@dataclass
class FaceResult:
    # "no_face" | "no_match" | "match"
    status: str
    box: Optional[dict] = None          # normalised 0..1: x, y, w, h
    distance: Optional[float] = None    # LBPH distance, lower = more similar
    label: Optional[int] = None
    elapsed_ms: float = 0.0
    frame_size: tuple = field(default_factory=tuple)


@dataclass
class EnrollResult:
    # "no_face" | "multiple_faces" | "ok"
    status: str
    box: Optional[dict] = None
    crop: Optional[np.ndarray] = None   # grayscale, FACE_SIZE — ready to save/train on


class FaceEngine:
    """Wraps one Haar cascade + one LBPH recognizer for the whole process.

    The recognizer may start with **no trained model** (a fresh install with
    zero registered users) — ``ready`` is False in that case and ``analyze``
    always returns "no_face"/"no_match" rather than erroring, so the app can
    still serve the registration flow.
    """

    def __init__(self, settings: type[Settings] = Settings):
        if not hasattr(cv2, "face"):
            raise RuntimeError(
                "cv2.face is missing. Install opencv-contrib-python "
                "(and uninstall opencv-python if it is present)."
            )
        if not settings.CASCADE_PATH.is_file():
            raise RuntimeError(f"Haar cascade not found at {settings.CASCADE_PATH}")

        self.s = settings
        self.detector = cv2.CascadeClassifier(str(settings.CASCADE_PATH))
        if self.detector.empty():
            raise RuntimeError(f"Could not load Haar cascade from {settings.CASCADE_PATH}")

        # OpenCV objects are not guaranteed thread-safe; serialise access,
        # including the brief moment a retrain swaps self.recognizer.
        self._lock = threading.Lock()

        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.ready = False
        self.training_images = 0
        self.label_counts: dict[int, int] = {}
        if settings.MODEL_PATH.is_file():
            self._load(settings.MODEL_PATH)

    def _load(self, path: Path) -> None:
        self.recognizer.read(str(path))
        hist = self.recognizer.getHistograms()
        labels = self.recognizer.getLabels().flatten().tolist()
        self.training_images = len(hist)
        counts: dict[int, int] = {}
        for lb in labels:
            counts[int(lb)] = counts.get(int(lb), 0) + 1
        self.label_counts = counts
        self.ready = True

    # ------------------------------------------------------------------ io
    @staticmethod
    def decode(jpeg_bytes: bytes) -> Optional[np.ndarray]:
        """Decode raw JPEG/PNG bytes into a BGR image, or None if invalid."""
        if not jpeg_bytes:
            return None
        arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None or img.ndim != 3:
            return None
        h, w = img.shape[:2]
        if w < 160 or h < 120 or w > 4096 or h > 4096:
            return None
        return img

    # ------------------------------------------------------------ detection
    def _detect(self, frame_bgr: np.ndarray) -> tuple[list[Detection], np.ndarray, int, int]:
        """Downscale, grayscale, and run the cascade. Returns detections
        sorted by confidence (best first), the grayscale frame, and its size."""
        h, w = frame_bgr.shape[:2]
        if w > MAX_DETECT_WIDTH:
            scale = MAX_DETECT_WIDTH / w
            frame_bgr = cv2.resize(frame_bgr, (MAX_DETECT_WIDTH, int(h * scale)), interpolation=cv2.INTER_AREA)
            h, w = frame_bgr.shape[:2]
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        with self._lock:
            rects, _levels, weights = self.detector.detectMultiScale3(
                gray, self.s.SCALE_FACTOR, self.s.MIN_NEIGHBORS, 0, self.s.MIN_FACE_SIZE,
                outputRejectLevels=True,
            )
        dets = sorted(
            (Detection(tuple(int(v) for v in r), float(c)) for r, c in zip(rects, np.ravel(weights))),
            key=lambda d: (d.confidence, d.box[2] * d.box[3]),
            reverse=True,
        )
        return dets, gray, w, h

    @staticmethod
    def _norm_box(box: tuple, w: int, h: int) -> dict:
        x, y, fw, fh = box
        return {"x": x / w, "y": y / h, "w": fw / w, "h": fh / h}

    # ------------------------------------------------------------- analyse
    def analyze(self, frame_bgr: np.ndarray) -> FaceResult:
        """Login path: verify the single best face detection against every
        enrolled user. Only the top detection is ever considered — a
        bystander in frame can at worst cause a "no_match", never a match,
        because we never fall back to a second, lower-confidence face."""
        t0 = time.perf_counter()
        dets, gray, w, h = self._detect(frame_bgr)

        if not dets:
            return FaceResult(status="no_face", elapsed_ms=self._ms(t0), frame_size=(w, h))

        x, y, fw, fh = dets[0].box
        box = self._norm_box(dets[0].box, w, h)

        if not self.ready:
            return FaceResult(status="no_match", box=box, elapsed_ms=self._ms(t0), frame_size=(w, h))

        crop = cv2.resize(gray[y:y + fh, x:x + fw], self.s.FACE_SIZE)
        with self._lock:
            label, distance = self.recognizer.predict(crop)
        distance = float(distance)
        matched = distance <= self.s.MATCH_THRESHOLD
        return FaceResult(
            status="match" if matched else "no_match",
            box=box, distance=round(distance, 1), label=int(label),
            elapsed_ms=self._ms(t0), frame_size=(w, h),
        )

    # ----------------------------------------------------------- enrolment
    def detect_for_enrollment(self, frame_bgr: np.ndarray) -> EnrollResult:
        """Registration path: stricter than login. Exactly one person must be
        in frame — a second face that's nearly as confident/large as the
        primary one aborts the sample, so a bystander can never get folded
        into someone else's enrollment."""
        dets, gray, w, h = self._detect(frame_bgr)
        if not dets:
            return EnrollResult(status="no_face")

        best = dets[0]
        for other in dets[1:]:
            same_conf = other.confidence >= self.s.ENROLL_SECOND_FACE_CONF_RATIO * best.confidence
            same_size = (other.box[2] * other.box[3]) >= self.s.ENROLL_SECOND_FACE_SIZE_RATIO * (best.box[2] * best.box[3])
            if same_conf and same_size:
                return EnrollResult(status="multiple_faces", box=self._norm_box(best.box, w, h))

        x, y, fw, fh = best.box
        crop = cv2.resize(gray[y:y + fh, x:x + fw], self.s.FACE_SIZE)
        return EnrollResult(status="ok", box=self._norm_box(best.box, w, h), crop=crop)

    def predict_crop(self, crop: np.ndarray) -> tuple[Optional[int], Optional[float]]:
        """Run the recognizer on an already-cropped FACE_SIZE grayscale image
        (used for the duplicate-account guard during registration)."""
        if not self.ready:
            return None, None
        with self._lock:
            label, distance = self.recognizer.predict(crop)
        return int(label), float(distance)

    # -------------------------------------------------------------- train
    def retrain(self, dataset_dirs: list[tuple[int, str]]) -> None:
        """Rebuild the LBPH model from every registered user's saved face
        crops and swap it in atomically. ``dataset_dirs`` is a list of
        (label, username) pairs; images are read from
        ``DATASET_ROOT/<username>/*``.
        """
        faces: list[np.ndarray] = []
        labels: list[int] = []
        counts: dict[int, int] = {}

        for label, username in dataset_dirs:
            folder = self.s.DATASET_ROOT / username
            if not folder.is_dir():
                continue
            n = 0
            for f in sorted(folder.iterdir()):
                if f.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp"):
                    continue
                img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                if img.shape != self.s.FACE_SIZE:
                    img = cv2.resize(img, self.s.FACE_SIZE)
                faces.append(img)
                labels.append(label)
                n += 1
            counts[label] = n

        if not faces:
            with self._lock:
                self.recognizer = cv2.face.LBPHFaceRecognizer_create()
                self.ready = False
                self.training_images = 0
                self.label_counts = {}
            return

        new_recognizer = cv2.face.LBPHFaceRecognizer_create()
        new_recognizer.train(faces, np.array(labels, dtype=np.int32))

        self.s.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.s.MODEL_PATH.with_suffix(".tmp")
        new_recognizer.write(str(tmp_path))
        os.replace(tmp_path, self.s.MODEL_PATH)   # atomic on POSIX: readers never see a half-written file

        with self._lock:
            self.recognizer = new_recognizer
            self.ready = True
            self.training_images = len(faces)
            self.label_counts = counts

    @staticmethod
    def _ms(t0: float) -> float:
        return round((time.perf_counter() - t0) * 1000, 1)
