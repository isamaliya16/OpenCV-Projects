"""Audit log of sign-in and registration activity (SQLite).

Only metadata is stored: time, outcome, which account it relates to (by LBPH
label, nullable for events that predate an identity — e.g. a lockout can
happen before any face has matched), match distance, frame count, IP and
browser string. Camera frames are analysed in memory and never written to disk.
"""
from __future__ import annotations

import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS login_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          TEXT    NOT NULL,          -- ISO-8601, UTC
    event       TEXT    NOT NULL,          -- success | failed | timeout | locked | logout | registered
    label       INTEGER,                   -- which user this relates to, if known
    ip          TEXT,
    user_agent  TEXT,
    distance    REAL,                      -- LBPH distance (lower = closer match)
    frames      INTEGER                    -- frames analysed in the attempt
);
CREATE INDEX IF NOT EXISTS idx_login_events_ts ON login_events (ts DESC);
CREATE INDEX IF NOT EXISTS idx_login_events_label ON login_events (label, id DESC);
"""

_SIGNIN_EVENTS = ("success", "failed", "locked", "timeout")


class AuditLog:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript(SCHEMA)
            db.commit()

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path, timeout=5)
        db.row_factory = sqlite3.Row
        return db

    def record(self, event: str, ip: str = "", user_agent: str = "", distance=None, frames=None,
               label: int | None = None) -> None:
        with closing(self._connect()) as db:
            db.execute(
                "INSERT INTO login_events (ts, event, label, ip, user_agent, distance, frames) "
                "VALUES (?,?,?,?,?,?,?)",
                (datetime.now(timezone.utc).isoformat(timespec="seconds"), event, label, ip,
                 (user_agent or "")[:160], distance, frames),
            )
            db.commit()

    def recent_for_label(self, label: int, limit: int = 8) -> list[dict]:
        placeholders = ",".join("?" * len(_SIGNIN_EVENTS))
        with closing(self._connect()) as db:
            rows = db.execute(
                f"SELECT ts, event, ip, distance, frames FROM login_events "
                f"WHERE label = ? AND event IN ({placeholders}) ORDER BY id DESC LIMIT ?",
                (label, *_SIGNIN_EVENTS, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def stats_for_label(self, label: int) -> dict:
        with closing(self._connect()) as db:
            row = db.execute(
                "SELECT "
                " SUM(event='success') AS ok, "
                " SUM(event='failed')  AS bad, "
                " MAX(CASE WHEN event='success' THEN ts END) AS last_ok "
                "FROM login_events WHERE label = ?",
                (label,),
            ).fetchone()
            prev = db.execute(
                "SELECT ts FROM login_events WHERE label = ? AND event='success' "
                "ORDER BY id DESC LIMIT 1 OFFSET 1",
                (label,),
            ).fetchone()
        return {
            "success": row["ok"] or 0,
            "failed": row["bad"] or 0,
            "last_success": row["last_ok"],
            "previous_success": prev["ts"] if prev else None,
        }
