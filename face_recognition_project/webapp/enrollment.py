"""Registration (enrollment) session state.

Mirrors security.Attempt/AttemptStore in shape, but for the sign-up camera
flow instead of the login one: collect a run of good face samples in memory,
then — once the target count is reached — the caller decides whether to
persist them (see routes.py's /api/register/complete), so nothing touches
disk or the trained model until the person confirms.
"""
from __future__ import annotations

import secrets
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .face_engine import EnrollResult
from .settings import Settings


def validate_registration(username: str, display_name: str, email: str, s: type[Settings]) -> dict:
    """Return {field: message} for anything invalid; empty dict means OK.
    Uniqueness is checked separately by the caller (it needs the DB)."""
    errors = {}

    u = (username or "").strip().lower()
    if not (s.USERNAME_MIN <= len(u) <= s.USERNAME_MAX):
        errors["username"] = f"Username must be {s.USERNAME_MIN}-{s.USERNAME_MAX} characters."
    elif not s.USERNAME_RE.match(u):
        errors["username"] = "Use lowercase letters, numbers, dots, dashes or underscores only."

    name = (display_name or "").strip()
    if not name:
        errors["display_name"] = "Enter your name."
    elif len(name) > s.DISPLAY_NAME_MAX:
        errors["display_name"] = f"Keep it under {s.DISPLAY_NAME_MAX} characters."

    e = (email or "").strip()
    if e and not s.EMAIL_RE.match(e):
        errors["email"] = "That doesn't look like a valid email address."

    return errors


@dataclass
class DuplicateCandidate:
    label: int
    username: str
    display_name: str
    distance: float


@dataclass
class EnrollmentSession:
    id: str
    ip: str
    username: str
    display_name: str
    email: Optional[str]
    created: float
    target: int
    last_frame: float = 0.0
    last_accepted: float = 0.0
    frames: int = 0
    samples: list = field(default_factory=list)   # list[np.ndarray], grayscale FACE_SIZE crops
    state: str = "collecting"    # collecting | ready | duplicate_hold | completed | expired | cancelled
    duplicate: Optional[DuplicateCandidate] = None

    def accept_frame(self, result: EnrollResult, s: type[Settings]) -> str:
        self.frames += 1
        if self.state != "collecting":
            return self.state

        if result.status == "ok":
            now = time.time()
            if now - self.last_accepted >= s.ENROLL_MIN_SAMPLE_INTERVAL:
                self.samples.append(result.crop)
                self.last_accepted = now
                if len(self.samples) >= self.target:
                    self.state = "ready"

        if self.frames >= s.ENROLL_MAX_FRAMES and self.state == "collecting":
            self.state = "expired"
        return self.state

    @property
    def progress(self) -> int:
        return len(self.samples)


class EnrollmentStore:
    def __init__(self, settings: type[Settings] = Settings):
        self.s = settings
        self._sessions: dict[str, EnrollmentSession] = {}
        self._lock = threading.Lock()

    def start(self, ip: str, username: str, display_name: str, email: Optional[str]) -> EnrollmentSession:
        now = time.time()
        with self._lock:
            self._prune(now)
            for sess in [s for s in self._sessions.values() if s.ip == ip and s.state == "collecting"]:
                sess.state = "cancelled"
            sess = EnrollmentSession(
                id=secrets.token_urlsafe(24), ip=ip, username=username, display_name=display_name,
                email=email, created=now, target=self.s.ENROLL_TARGET_SAMPLES,
            )
            self._sessions[sess.id] = sess
            return sess

    def get(self, enrollment_id: Optional[str]) -> Optional[EnrollmentSession]:
        if not enrollment_id:
            return None
        with self._lock:
            return self._sessions.get(enrollment_id)

    def drop(self, enrollment_id: str) -> None:
        with self._lock:
            self._sessions.pop(enrollment_id, None)

    def _prune(self, now: float) -> None:
        cutoff = now - max(self.s.ENROLL_TTL_SECONDS * 3, 300)
        for k in [k for k, s in self._sessions.items() if s.created < cutoff]:
            del self._sessions[k]
