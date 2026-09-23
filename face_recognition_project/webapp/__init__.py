"""FaceKey — face sign-in web app built on the project's OpenCV pipeline."""
from __future__ import annotations

from datetime import timedelta
from types import SimpleNamespace

from flask import Flask, request
from werkzeug.middleware.proxy_fix import ProxyFix

from .db import AuditLog
from .enrollment import EnrollmentStore
from .face_engine import FaceEngine
from .security import AttemptStore, LockoutTracker, RegistrationThrottle
from .settings import Settings
from .users_db import UsersStore, sync_legacy_user

CSP = (
    "default-src 'self'; "
    "img-src 'self' data:; "
    "style-src 'self'; "
    "script-src 'self'; "
    "font-src 'self'; "
    "connect-src 'self'; "
    "media-src 'self'; "
    "frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
)


def create_app(settings: type[Settings] = Settings, engine: FaceEngine | None = None) -> Flask:
    app = Flask(__name__)
    app.config.update(
        SECRET_KEY=settings.SECRET_KEY,
        MAX_CONTENT_LENGTH=settings.MAX_CONTENT_LENGTH,
        SESSION_COOKIE_NAME="aperture_session",
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
        SESSION_COOKIE_SECURE=settings.COOKIE_SECURE,
        PERMANENT_SESSION_LIFETIME=timedelta(minutes=settings.SESSION_MINUTES),
        SESSION_REFRESH_EACH_REQUEST=False,   # absolute lifetime, not sliding
        JSON_SORT_KEYS=False,
    )

    if settings.TRUST_PROXY:  # only enable behind a reverse proxy you control
        app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)

    users = UsersStore(settings.DB_PATH)
    sync_legacy_user(users, settings)   # one-time import of the pre-existing single-user dataset

    face_engine = engine or FaceEngine(settings)
    if not face_engine.ready and users.count() > 0:
        # Users exist in the DB (e.g. the legacy import above) but the model
        # file was missing or unreadable — rebuild it from their saved crops.
        face_engine.retrain(users.dataset_dirs())

    app.extensions["aperture"] = SimpleNamespace(
        settings=settings,
        engine=face_engine,
        users=users,
        attempts=AttemptStore(settings),
        lockout=LockoutTracker(settings),
        enrollments=EnrollmentStore(settings),
        reg_throttle=RegistrationThrottle(settings),
        audit=AuditLog(settings.DB_PATH),
    )

    from .routes import bp
    app.register_blueprint(bp)

    @app.after_request
    def _headers(resp):
        resp.headers.setdefault("Content-Security-Policy", CSP)
        resp.headers.setdefault("X-Content-Type-Options", "nosniff")
        resp.headers.setdefault("Referrer-Policy", "same-origin")
        resp.headers.setdefault("Permissions-Policy", "camera=(self), microphone=(), geolocation=()")
        if not request.path.startswith("/static/"):
            resp.headers["Cache-Control"] = "no-store"
        return resp

    return app
