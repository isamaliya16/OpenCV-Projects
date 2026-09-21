"""VisionScan web app (Flask): upload an image, get objects and faces detected."""
from __future__ import annotations

import logging
import math
import os
import re
import secrets
import threading
import time
from pathlib import Path

from flask import (Flask, abort, flash, jsonify, redirect, render_template, request,
                   send_from_directory, session, url_for)

import detector
from detector import DetectionError

BASE_DIR = Path(__file__).resolve().parent
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}
DEBUG = os.getenv("DEBUG", "0").lower() in ("1", "true", "yes")

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")


def allowed(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def display_name(filename):
    """A safe name to *show* (never used as a path), keeping non-ASCII letters."""
    name = _CONTROL_CHARS.sub("", os.path.basename(filename.replace("\\", "/"))).strip()
    return name[:120] or "image"


def _float_env(name, default):
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return default


def _size_label(num_bytes):
    return f"{num_bytes / (1024 * 1024):g} MB"


def create_app(test_config=None):
    app = Flask(__name__)
    app.config.update(
        # Set SECRET_KEY in the environment to keep sessions across restarts.
        SECRET_KEY=os.getenv("SECRET_KEY") or secrets.token_hex(32),
        MAX_CONTENT_LENGTH=int(_float_env("MAX_UPLOAD_MB", 15) * 1024 * 1024),
        RESULT_DIR=str(BASE_DIR / "results"),
        RESULT_TTL_MINUTES=_float_env("RESULT_TTL_MINUTES", 60.0),
        WARMUP=os.getenv("WARMUP", "1") != "0",
        SESSION_COOKIE_SAMESITE="Lax",
    )
    if test_config:
        app.config.update(test_config)

    result_dir = Path(app.config["RESULT_DIR"])
    result_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ housekeeping
    cleanup_state = {"last": float("-inf")}
    cleanup_lock = threading.Lock()

    def cleanup():
        """Delete expired results; runs at most once a minute."""
        with cleanup_lock:
            now = time.monotonic()
            if now - cleanup_state["last"] < 60:
                return
            cleanup_state["last"] = now
        detector.cleanup_results(result_dir, app.config["RESULT_TTL_MINUTES"] * 60)

    cleanup()

    # Load the models in the background so the first scan is not slow. With the
    # debug reloader only the child process (the one serving requests) does this.
    reloader_parent = DEBUG and os.environ.get("WERKZEUG_RUN_MAIN") != "true"
    if app.config["WARMUP"] and not reloader_parent:
        threading.Thread(target=detector.warm_up, name="model-warmup", daemon=True).start()

    # ------------------------------------------------------------------ CSRF + headers
    def csrf_token():
        if "_csrf" not in session:
            session["_csrf"] = secrets.token_hex(16)
        return session["_csrf"]

    @app.context_processor
    def template_globals():
        limit = app.config["MAX_CONTENT_LENGTH"]
        return {
            "csrf_token": csrf_token,
            "max_bytes": limit,
            "max_label": _size_label(limit),
            "result_ttl": f"{app.config['RESULT_TTL_MINUTES']:g}",
        }

    @app.before_request
    def protect_forms():
        if request.method == "POST":
            sent = request.form.get("_csrf", "")
            if not sent or not secrets.compare_digest(sent, session.get("_csrf", "")):
                abort(400)

    @app.after_request
    def security_headers(response):
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        response.headers.setdefault(
            "Content-Security-Policy",
            "default-src 'self'; img-src 'self' blob: data:; frame-ancestors 'none'; "
            "form-action 'self'",
        )
        return response

    # ------------------------------------------------------------------ errors
    def home_with_message(message, status):
        flash(message)
        return render_template("index.html", result=None), status

    @app.errorhandler(413)
    def too_large(_exc):
        limit = _size_label(app.config["MAX_CONTENT_LENGTH"])
        return home_with_message(
            f"That file is too large. The limit is {limit}. Choose a smaller image.", 413
        )

    @app.errorhandler(400)
    def bad_request(_exc):
        return home_with_message("The form expired. Please choose your image again.", 400)

    # ------------------------------------------------------------------ routes
    def percent_field(name, default):
        """Read a 5-95 % form field as a 0-1 fraction; bad input falls back to `default`."""
        try:
            value = float(request.form.get(name, ""))
        except ValueError:
            return default
        if not math.isfinite(value):
            return default
        return max(5.0, min(95.0, value)) / 100

    @app.get("/")
    def index():
        cleanup()
        return render_template(
            "index.html",
            result=None,
            object_default=round(detector.DEFAULT_OBJECT_CONF * 100),
            face_default=round(detector.DEFAULT_FACE_CONF * 100),
        )

    @app.post("/")
    def scan():
        cleanup()
        upload = request.files.get("image")
        if not upload or not upload.filename:
            flash("Choose an image first.")
            return redirect(url_for("index"))
        if not allowed(upload.filename):
            flash("Unsupported format. Use JPG, JPEG, PNG or WEBP.")
            return redirect(url_for("index"))
        try:
            meta = detector.detect_image(
                upload.read(),
                display_name(upload.filename),
                result_dir,
                object_conf=percent_field("object_conf", detector.DEFAULT_OBJECT_CONF),
                face_conf=percent_field("face_conf", detector.DEFAULT_FACE_CONF),
            )
        except DetectionError as exc:  # user-facing, safe to show
            flash(str(exc))
            return redirect(url_for("index"))
        except Exception:  # noqa: BLE001 - never show internals to the user
            app.logger.exception("Scan failed")
            flash("Something went wrong while scanning this image. Try another image.")
            return redirect(url_for("index"))
        return redirect(url_for("result", rid=meta["id"]))  # Post/Redirect/Get

    @app.get("/result/<rid>")
    def result(rid):
        meta = detector.load_result(result_dir, rid)
        if meta is None:
            flash("That result has expired or does not exist. Scan the image again.")
            return redirect(url_for("index"))
        return render_template("index.html", result=meta)

    @app.get("/results/<rid>.jpg")
    def result_image(rid):
        if not detector.valid_id(rid):
            abort(404)
        response = send_from_directory(result_dir, f"{rid}.jpg", mimetype="image/jpeg")
        response.headers["Cache-Control"] = "private, max-age=300"
        return response

    @app.get("/result/<rid>/download")
    def result_download(rid):
        meta = detector.load_result(result_dir, rid)
        if meta is None:
            abort(404)
        stem = Path(meta["original_name"]).stem or "image"
        return send_from_directory(
            result_dir, f"{rid}.jpg", mimetype="image/jpeg",
            as_attachment=True, download_name=f"{stem}_detected.jpg",
        )

    @app.post("/result/<rid>/delete")
    def result_delete(rid):
        detector.delete_result(result_dir, rid)
        flash("Result deleted.")
        return redirect(url_for("index"))

    @app.get("/health")
    def health():
        response = jsonify(detector.models_status())
        response.headers["Cache-Control"] = "no-store"
        return response

    return app


app = create_app()

if __name__ == "__main__":
    app.run(
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "5000")),
        debug=DEBUG,
    )
