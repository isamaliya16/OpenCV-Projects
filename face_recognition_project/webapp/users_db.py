"""Registered-users store (SQLite) — maps an LBPH label to an account.

Lives in the same database file as the audit log (``db.py``), as its own
table. Kept in a separate class/file because it has a distinct job: identity
and enrollment bookkeeping, not activity history.
"""
from __future__ import annotations

import sqlite3
import threading
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    label         INTEGER PRIMARY KEY,     -- the LBPH label used by the recognizer
    username      TEXT    NOT NULL UNIQUE, -- lowercase slug, also the dataset folder name
    display_name  TEXT    NOT NULL,
    email         TEXT,
    created_at    TEXT    NOT NULL,        -- ISO-8601 UTC
    sample_count  INTEGER NOT NULL DEFAULT 0,
    source        TEXT    NOT NULL DEFAULT 'web'   -- 'web' (registered here) | 'legacy' (imported)
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_users_username_ci ON users (username);
"""


@dataclass
class User:
    label: int
    username: str
    display_name: str
    email: Optional[str]
    created_at: str
    sample_count: int
    source: str


class UsernameTaken(Exception):
    pass


class EmailTaken(Exception):
    pass


class UsersStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()  # serialises label allocation
        with closing(self._connect()) as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript(SCHEMA)
            db.commit()

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path, timeout=5)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        return db

    @staticmethod
    def _row(r) -> User:
        return User(
            label=r["label"], username=r["username"], display_name=r["display_name"],
            email=r["email"], created_at=r["created_at"], sample_count=r["sample_count"],
            source=r["source"],
        )

    # ------------------------------------------------------------- lookups
    def get_by_label(self, label: int) -> Optional[User]:
        with closing(self._connect()) as db:
            r = db.execute("SELECT * FROM users WHERE label = ?", (label,)).fetchone()
        return self._row(r) if r else None

    def get_by_username(self, username: str) -> Optional[User]:
        with closing(self._connect()) as db:
            r = db.execute(
                "SELECT * FROM users WHERE username = ? COLLATE NOCASE", (username,)
            ).fetchone()
        return self._row(r) if r else None

    def username_taken(self, username: str) -> bool:
        return self.get_by_username(username) is not None

    def email_taken(self, email: str) -> bool:
        if not email:
            return False
        with closing(self._connect()) as db:
            r = db.execute(
                "SELECT 1 FROM users WHERE email = ? COLLATE NOCASE", (email,)
            ).fetchone()
        return r is not None

    def list_all(self) -> list[User]:
        with closing(self._connect()) as db:
            rows = db.execute("SELECT * FROM users ORDER BY label").fetchall()
        return [self._row(r) for r in rows]

    def count(self) -> int:
        with closing(self._connect()) as db:
            return db.execute("SELECT COUNT(*) FROM users").fetchone()[0]

    def dataset_dirs(self) -> list[tuple[int, str]]:
        """(label, username) pairs, for retraining against dataset/<username>/."""
        with closing(self._connect()) as db:
            rows = db.execute("SELECT label, username FROM users").fetchall()
        return [(r["label"], r["username"]) for r in rows]

    # -------------------------------------------------------------- writes
    def create(self, username: str, display_name: str, email: Optional[str], sample_count: int,
               source: str = "web", label: Optional[int] = None) -> User:
        """Insert a new user, allocating the next free label unless one is given
        (used only for the one-time legacy import, which must reuse config.PERSON_LABEL)."""
        with self._lock:
            if self.username_taken(username):
                raise UsernameTaken(username)
            if email and self.email_taken(email):
                raise EmailTaken(email)
            with closing(self._connect()) as db:
                if label is None:
                    row = db.execute("SELECT COALESCE(MAX(label), 0) + 1 FROM users").fetchone()
                    label = row[0]
                db.execute(
                    "INSERT INTO users (label, username, display_name, email, created_at, "
                    "sample_count, source) VALUES (?,?,?,?,?,?,?)",
                    (label, username, display_name, email or None,
                     datetime.now(timezone.utc).isoformat(timespec="seconds"), sample_count, source),
                )
                db.commit()
        return self.get_by_label(label)

    def bump_sample_count(self, label: int, added: int) -> None:
        with closing(self._connect()) as db:
            db.execute("UPDATE users SET sample_count = sample_count + ? WHERE label = ?", (added, label))
            db.commit()

    def update_profile(self, label: int, display_name: str, email: Optional[str]) -> User:
        """Update the editable parts of a profile (display name, email).
        Username and dataset are untouched — renaming those would mean
        moving files on disk and is out of scope here."""
        with self._lock:
            current = self.get_by_label(label)
            if current is None:
                raise LookupError(label)
            if email and email.lower() != (current.email or "").lower() and self.email_taken(email):
                raise EmailTaken(email)
            with closing(self._connect()) as db:
                db.execute(
                    "UPDATE users SET display_name = ?, email = ? WHERE label = ?",
                    (display_name, email or None, label),
                )
                db.commit()
        return self.get_by_label(label)

    def delete(self, label: int) -> None:
        with closing(self._connect()) as db:
            db.execute("DELETE FROM users WHERE label = ?", (label,))
            db.commit()


def slugify_username(raw: str) -> str:
    """Normalise a proposed username the same way both client and server check it."""
    return (raw or "").strip().lower()


def sync_legacy_user(users: UsersStore, settings) -> None:
    """One-time import of the pre-existing single-user dataset (config.py /
    01_collect_faces.py / 02_train_model.py) as the system's first account,
    so it keeps working unchanged after upgrading to multi-user registration.
    No-op once that user (or any user) already exists.
    """
    if users.count() > 0:
        return
    if not settings.LEGACY_DATASET_PATH.is_dir():
        return
    images = [
        f for f in settings.LEGACY_DATASET_PATH.iterdir()
        if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")
    ]
    if not images:
        return
    users.create(
        username=slugify_username(settings.LEGACY_USERNAME),
        display_name=settings.LEGACY_DISPLAY_NAME,
        email=None,
        sample_count=len(images),
        source="legacy",
        label=settings.LEGACY_LABEL,
    )
