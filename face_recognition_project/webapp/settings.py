"""Web-app settings.

Shared recognition parameters (threshold, cascade parameters, face size) are
read from the project's existing ``config.py`` so the web app and the
command-line scripts always agree on how a face is detected and cropped.
Identity itself is no longer a single hardcoded person: the app now supports
many registered users, stored in the SQLite users table (see ``db.py``). The
very first user is the account already produced by ``01_collect_faces.py`` /
``02_train_model.py`` (``config.PERSON_NAME`` / ``config.PERSON_LABEL``) —
that dataset and model keep working, they're just no longer special-cased in
code beyond a one-time import (see ``webapp.users_db.sync_legacy_user``).

Anything web-specific lives here and can be overridden with environment
variables.
"""
from __future__ import annotations

import importlib.util
import os
import re
import secrets
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent


def _load_project_config():
    # Load config.py by path so we never clash with another module named "config".
    spec = importlib.util.spec_from_file_location("project_config", ROOT / "config.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_cfg = _load_project_config()


def _resolve(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else ROOT / p


def _env_float(name: str, default: float) -> float:
    return float(os.getenv(name, default))


def _env_int(name: str, default: int) -> int:
    return int(os.getenv(name, default))


def _secret_key() -> str:
    """Use SECRET_KEY if provided; otherwise create one once and reuse it."""
    if os.getenv("SECRET_KEY"):
        return os.environ["SECRET_KEY"]
    key_file = ROOT / "webapp" / "data" / ".secret_key"
    if key_file.exists():
        return key_file.read_text().strip()
    key = secrets.token_hex(32)
    key_file.parent.mkdir(parents=True, exist_ok=True)
    key_file.write_text(key)
    return key


class Settings:
    # ---- shared detection/recognition parameters (from config.py) ---------
    FACE_SIZE: tuple = tuple(_cfg.FACE_SIZE)
    SCALE_FACTOR: float = _cfg.SCALE_FACTOR
    MIN_NEIGHBORS: int = _cfg.MIN_NEIGHBORS
    MIN_FACE_SIZE: tuple = tuple(_cfg.MIN_FACE_SIZE)

    MODEL_PATH: Path = _resolve(_cfg.MODEL_PATH)
    DATASET_ROOT: Path = _resolve(_cfg.DATASET_DIR)        # dataset/<username>/*.jpg per user

    # The pre-existing single-user dataset this project shipped with.
    # Imported once as the first user; see users_db.sync_legacy_user().
    LEGACY_USERNAME: str = _cfg.PERSON_DIR_NAME
    LEGACY_DISPLAY_NAME: str = _cfg.PERSON_NAME
    LEGACY_LABEL: int = _cfg.PERSON_LABEL
    LEGACY_DATASET_PATH: Path = DATASET_ROOT / _cfg.PERSON_DIR_NAME

    _cascade = _resolve(_cfg.CASCADE_PATH)
    CASCADE_PATH: Path = (
        _cascade
        if _cascade.is_file()
        else Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    )

    # LBPH distance: lower = more similar. Accept when distance <= threshold.
    # Defaults to config.RECOGNITION_THRESHOLD; set FACE_LOGIN_THRESHOLD to be
    # stricter for login than for the demo scripts.
    MATCH_THRESHOLD: float = _env_float("FACE_LOGIN_THRESHOLD", _cfg.RECOGNITION_THRESHOLD)

    # ---- login policy --------------------------------------------------------
    REQUIRED_STREAK: int = _env_int("FACE_REQUIRED_STREAK", 5)      # consecutive matching frames...
    MAX_NO_FACE_GAP: int = 3            # tolerated consecutive "no face" frames before streak resets
    MAX_MISMATCH_FRAMES: int = 15       # frames that show a face but do not match -> attempt fails
    MAX_FRAMES_PER_ATTEMPT: int = 120
    ATTEMPT_TTL_SECONDS: int = 40
    MIN_FRAME_INTERVAL: float = 0.10    # seconds between frames per attempt (server-side throttle)

    MAX_FAILED_ATTEMPTS: int = _env_int("FACE_MAX_FAILED_ATTEMPTS", 5)
    LOCKOUT_WINDOW_SECONDS: int = 600
    LOCKOUT_SECONDS: int = _env_int("FACE_LOCKOUT_SECONDS", 300)

    # ---- registration / enrollment policy -------------------------------------
    USERNAME_RE = re.compile(r"^[a-z0-9](?:[a-z0-9._-]{1,30}[a-z0-9])?$")
    USERNAME_MIN: int = 3
    USERNAME_MAX: int = 32
    DISPLAY_NAME_MAX: int = 80
    EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

    ENROLL_TARGET_SAMPLES: int = _env_int("FACE_ENROLL_SAMPLES", 20)   # face photos captured at signup
    ENROLL_MIN_SAMPLE_INTERVAL: float = 0.35   # seconds between accepted samples (natural pose variety)
    ENROLL_MIN_FRAME_INTERVAL: float = 0.10    # server-side per-request throttle, like login
    ENROLL_TTL_SECONDS: int = 240              # registration gets more time than login (must move the head)
    ENROLL_MAX_FRAMES: int = 500

    # A second face nearly as confident/large as the primary one aborts an
    # enrollment frame — during signup we want exactly one person in frame,
    # unlike login where only the best detection is ever considered.
    ENROLL_SECOND_FACE_CONF_RATIO: float = 0.5
    ENROLL_SECOND_FACE_SIZE_RATIO: float = 0.35

    # "Is this face already enrolled under someone else's account?" check,
    # run once a session reaches its target sample count. Stricter than the
    # login threshold on purpose: this only has to be confident enough to
    # warn, not to reject outright (the person can still confirm and continue).
    DUPLICATE_GUARD_ENABLED: bool = True
    DUPLICATE_GUARD_THRESHOLD: float = _env_float("FACE_DUPLICATE_THRESHOLD", 45.0)

    MAX_REGISTRATIONS_PER_IP: int = _env_int("FACE_MAX_REGISTRATIONS_PER_IP", 8)
    REGISTRATION_WINDOW_SECONDS: int = 3600

    # ---- web / session -----------------------------------------------------
    SECRET_KEY: str = _secret_key()
    SESSION_MINUTES: int = _env_int("SESSION_MINUTES", 30)
    COOKIE_SECURE: bool = os.getenv("COOKIE_SECURE", "0") == "1"   # set to 1 behind HTTPS
    MAX_CONTENT_LENGTH: int = 1_000_000                              # 1 MB per frame
    TRUST_PROXY: bool = os.getenv("TRUST_PROXY", "0") == "1"         # set to 1 behind nginx/Caddy
    DB_PATH: Path = Path(os.getenv("AUTH_DB_PATH", ROOT / "webapp" / "data" / "auth.db"))
