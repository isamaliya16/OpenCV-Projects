"""Login-attempt state and brute-force protection.

Everything that decides "is this person verified?" lives on the server. The
browser only sends camera frames; it never reports a result. State is kept in
memory, so run a single worker process (see docs/SYSTEM_DESIGN.md for the
Redis-based scale-out path).
"""
from __future__ import annotations

import secrets
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Optional

from .face_engine import FaceResult
from .settings import Settings


@dataclass
class Attempt:
    id: str
    ip: str
    created: float
    last_frame: float = 0.0
    frames: int = 0
    streak: int = 0
    matched_label: Optional[int] = None   # the user the current streak is building toward
    gap: int = 0                      # consecutive "no face" frames
    mismatches: int = 0               # frames with a face that did not match
    best_distance: Optional[float] = None
    best_label: Optional[int] = None      # label nearest to best_distance — the closest "near miss", if any
    streak_distances: list = field(default_factory=list)
    state: str = "active"             # active | verified | failed | expired

    def register(self, result: FaceResult, s: type[Settings]) -> str:
        """Fold one analysed frame into the attempt; return the new state.

        The streak is per-identity: every frame in a row must match the SAME
        registered user (result.label), not just "someone enrolled". If the
        best-matching identity changes mid-streak, that streak is abandoned
        and a new one starts from this frame — a run of matches can never be
        assembled by borrowing frames across two different people.
        """
        self.frames += 1

        if result.distance is not None:
            if self.best_distance is None or result.distance < self.best_distance:
                self.best_distance = result.distance
                self.best_label = result.label

        if result.status == "match":
            if self.streak and result.label != self.matched_label:
                self.streak_distances.clear()
                self.streak = 0
            self.matched_label = result.label
            self.streak += 1
            self.gap = 0
            self.streak_distances.append(result.distance)
        elif result.status == "no_match":
            self.streak = 0
            self.matched_label = None
            self.gap = 0
            self.mismatches += 1
            self.streak_distances.clear()
        else:  # no_face: tolerate brief detector flicker, but not a long absence
            self.gap += 1
            if self.gap > s.MAX_NO_FACE_GAP:
                self.streak = 0
                self.matched_label = None
                self.streak_distances.clear()

        if self.streak >= s.REQUIRED_STREAK:
            self.state = "verified"
        elif self.mismatches >= s.MAX_MISMATCH_FRAMES or self.frames >= s.MAX_FRAMES_PER_ATTEMPT:
            self.state = "failed"
        return self.state

    @property
    def mean_distance(self) -> Optional[float]:
        d = self.streak_distances
        return round(sum(d) / len(d), 1) if d else None


class AttemptStore:
    def __init__(self, settings: type[Settings] = Settings):
        self.s = settings
        self._attempts: dict[str, Attempt] = {}
        self._lock = threading.Lock()

    def start(self, ip: str) -> Attempt:
        now = time.time()
        with self._lock:
            self._prune(now)
            # One live attempt per client: starting a new one retires the old one.
            for a in [a for a in self._attempts.values() if a.ip == ip and a.state == "active"]:
                a.state = "expired"
            attempt = Attempt(id=secrets.token_urlsafe(24), ip=ip, created=now)
            self._attempts[attempt.id] = attempt
            return attempt

    def get(self, attempt_id: Optional[str]) -> Optional[Attempt]:
        if not attempt_id:
            return None
        with self._lock:
            return self._attempts.get(attempt_id)

    def _prune(self, now: float) -> None:
        cutoff = now - max(self.s.ATTEMPT_TTL_SECONDS * 3, 120)
        for k in [k for k, a in self._attempts.items() if a.created < cutoff]:
            del self._attempts[k]


class LockoutTracker:
    """Temporarily blocks a client after too many failed attempts."""

    def __init__(self, settings: type[Settings] = Settings):
        self.s = settings
        self._failures: dict[str, deque] = defaultdict(deque)
        self._locked_until: dict[str, float] = {}
        self._lock = threading.Lock()

    def seconds_left(self, ip: str) -> int:
        now = time.time()
        with self._lock:
            until = self._locked_until.get(ip, 0)
            if until <= now:
                self._locked_until.pop(ip, None)
                return 0
            return int(until - now) + 1

    def record_failure(self, ip: str) -> bool:
        """Record a failed attempt. Returns True if the client is now locked."""
        now = time.time()
        with self._lock:
            q = self._failures[ip]
            q.append(now)
            while q and q[0] < now - self.s.LOCKOUT_WINDOW_SECONDS:
                q.popleft()
            if len(q) >= self.s.MAX_FAILED_ATTEMPTS:
                self._locked_until[ip] = now + self.s.LOCKOUT_SECONDS
                q.clear()
                return True
            return False

    def reset(self, ip: str) -> None:
        with self._lock:
            self._failures.pop(ip, None)
            self._locked_until.pop(ip, None)


class RegistrationThrottle:
    """Caps how many new accounts one IP can create per hour — registration
    has no lockout/penalty concept, just a rolling-window cap."""

    def __init__(self, settings: type[Settings] = Settings):
        self.s = settings
        self._events: dict[str, deque] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, ip: str) -> bool:
        """Record an attempt to start registering and report whether it's allowed."""
        now = time.time()
        with self._lock:
            q = self._events[ip]
            while q and q[0] < now - self.s.REGISTRATION_WINDOW_SECONDS:
                q.popleft()
            if len(q) >= self.s.MAX_REGISTRATIONS_PER_IP:
                return False
            q.append(now)
            return True
