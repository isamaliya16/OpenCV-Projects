from __future__ import annotations

import shutil
import time
from datetime import datetime, timezone
from functools import wraps

import cv2
from flask import (
    Blueprint, current_app, jsonify, redirect, render_template, request, session, url_for,
)

from .enrollment import DuplicateCandidate, validate_registration
from .users_db import UsernameTaken, EmailTaken, slugify_username

bp = Blueprint("aperture", __name__)


def _ctx():
    return current_app.extensions["aperture"]


def _ip() -> str:
    return request.remote_addr or "unknown"


def _current_user():
    """Return the signed-in session dict, or None if absent/expired."""
    user = session.get("user")
    if not user:
        return None
    if time.time() >= user.get("expires_at", 0):
        session.clear()
        return None
    return user


def login_required(view):
    @wraps(view)
    def wrapper(*args, **kwargs):
        if _current_user() is None:
            if request.path.startswith("/api/"):
                return jsonify(error="unauthorised"), 401
            return redirect(url_for("aperture.login"))
        return view(*args, **kwargs)
    return wrapper


# ================================================================== pages
@bp.get("/")
def index():
    return redirect(url_for("aperture.dashboard" if _current_user() else "aperture.login"))


@bp.get("/login")
def login():
    if _current_user():
        return redirect(url_for("aperture.dashboard"))
    c = _ctx()
    return render_template(
        "login.html",
        required=c.settings.REQUIRED_STREAK,
        has_users=c.users.count() > 0,
        just_registered=request.args.get("registered") == "1",
        registered_name=request.args.get("name", ""),
    )


@bp.get("/register")
def register():
    if _current_user():
        return redirect(url_for("aperture.dashboard"))
    c = _ctx()
    return render_template(
        "register.html",
        target=c.settings.ENROLL_TARGET_SAMPLES,
        username_min=c.settings.USERNAME_MIN,
        username_max=c.settings.USERNAME_MAX,
    )


@bp.get("/dashboard")
@login_required
def dashboard():
    c = _ctx()
    sess_user = _current_user()
    user = c.users.get_by_label(sess_user["label"])
    if user is None:   # account was removed server-side mid-session
        session.clear()
        return redirect(url_for("aperture.login"))

    return render_template(
        "dashboard.html",
        user=user,
        first_name=user.display_name.split()[0],
        initials="".join(p[0] for p in user.display_name.split()[:2]).upper(),
        events=c.audit.recent_for_label(user.label, 8),
        stats=c.audit.stats_for_label(user.label),
        threshold=c.settings.MATCH_THRESHOLD,
        required=c.settings.REQUIRED_STREAK,
        verified_at=sess_user["verified_at"],
        verified_distance=sess_user["distance"],
        verified_frames=sess_user["frames"],
        total_users=c.users.count(),
        seconds_left=max(0, int(sess_user["expires_at"] - time.time())),
    )


@bp.post("/logout")
def logout():
    user = _current_user()
    if user:
        _ctx().audit.record("logout", _ip(), request.user_agent.string, label=user["label"])
    session.clear()
    return redirect(url_for("aperture.login"))


@bp.post("/api/account/update")
@login_required
def account_update():
    c = _ctx()
    s = c.settings
    sess_user = _current_user()
    body = request.get_json(silent=True) or {}
    display_name = (body.get("display_name") or "").strip()
    email = (body.get("email") or "").strip() or None

    errors = {}
    if not display_name:
        errors["display_name"] = "Enter your name."
    elif len(display_name) > s.DISPLAY_NAME_MAX:
        errors["display_name"] = f"Keep it under {s.DISPLAY_NAME_MAX} characters."
    if email and not s.EMAIL_RE.match(email):
        errors["email"] = "That doesn't look like a valid email address."
    if errors:
        return jsonify(errors=errors), 422

    try:
        user = c.users.update_profile(sess_user["label"], display_name, email)
    except EmailTaken:
        return jsonify(errors={"email": "That email is already registered."}), 409
    except LookupError:
        session.clear()
        return jsonify(error="unauthorised"), 401

    return jsonify(display_name=user.display_name, email=user.email)


@bp.post("/api/account/delete")
@login_required
def account_delete():
    c = _ctx()
    sess_user = _current_user()
    user = c.users.get_by_label(sess_user["label"])
    if user is None:
        session.clear()
        return jsonify(error="unauthorised"), 401

    folder = c.settings.DATASET_ROOT / user.username
    shutil.rmtree(folder, ignore_errors=True)
    c.users.delete(user.label)
    c.engine.retrain(c.users.dataset_dirs())
    c.audit.record("deleted", _ip(), request.user_agent.string, label=user.label)

    session.clear()
    return jsonify(state="done", redirect=url_for("aperture.login"))


@bp.get("/healthz")
def healthz():
    c = _ctx()
    return jsonify(
        status="ok", ready=c.engine.ready, users=c.users.count(),
        training_images=c.engine.training_images, threshold=c.settings.MATCH_THRESHOLD,
    )


# =========================================================== registration API
@bp.post("/api/register/start")
def register_start():
    c = _ctx()
    s = c.settings
    ip = _ip()

    if not c.reg_throttle.allow(ip):
        return jsonify(error="too_many_registrations"), 429

    body = request.get_json(silent=True) or {}
    username = slugify_username(body.get("username", ""))
    display_name = (body.get("display_name") or "").strip()
    email = (body.get("email") or "").strip() or None

    errors = validate_registration(username, display_name, email, s)
    if not errors:
        if c.users.username_taken(username):
            errors["username"] = "That username is already taken."
        elif email and c.users.email_taken(email):
            errors["email"] = "That email is already registered."
    if errors:
        return jsonify(errors=errors), 422

    sess = c.enrollments.start(ip, username, display_name, email)
    return jsonify(
        enrollment_id=sess.id, target=sess.target, ttl=s.ENROLL_TTL_SECONDS,
    )


@bp.get("/api/register/username-available")
def register_username_available():
    c = _ctx()
    username = slugify_username(request.args.get("u", ""))
    if not c.settings.USERNAME_RE.match(username) or not (
        c.settings.USERNAME_MIN <= len(username) <= c.settings.USERNAME_MAX
    ):
        return jsonify(available=False, reason="invalid")
    return jsonify(available=not c.users.username_taken(username))


@bp.post("/api/register/frame")
def register_frame():
    c = _ctx()
    s = c.settings
    ip = _ip()

    sess = c.enrollments.get(request.headers.get("X-Enrollment-Id"))
    if sess is None or sess.ip != ip:
        return jsonify(state="expired"), 410
    if sess.state not in ("collecting",):
        return jsonify(state=sess.state, count=sess.progress, target=sess.target)

    now = time.time()
    if now - sess.created > s.ENROLL_TTL_SECONDS:
        sess.state = "expired"
        return jsonify(state="expired"), 200
    if now - sess.last_frame < s.ENROLL_MIN_FRAME_INTERVAL:
        return jsonify(error="slow_down"), 429
    sess.last_frame = now

    frame = c.engine.decode(request.get_data(cache=False))
    if frame is None:
        return jsonify(error="bad_image"), 400

    result = c.engine.detect_for_enrollment(frame)
    state = sess.accept_frame(result, s)

    payload = {
        "state": state if state != "collecting" else ("searching" if result.status == "no_face" else result.status),
        "face": result.box,
        "count": sess.progress,
        "target": sess.target,
    }

    if state == "ready":
        _check_duplicate(sess, c)
        payload["state"] = sess.state
        if sess.duplicate:
            payload["duplicate"] = {
                "username": sess.duplicate.username,
                "display_name": sess.duplicate.display_name,
            }

    return jsonify(payload)


def _check_duplicate(sess, c) -> None:
    """Flag (not block) when the captured face closely matches an existing
    account — run once, right as a session reaches its target sample count."""
    s = c.settings
    if not s.DUPLICATE_GUARD_ENABLED or not c.engine.ready or not sess.samples:
        return
    probe = sess.samples[:: max(1, len(sess.samples) // 5)][:5]
    votes: dict[int, list] = {}
    for crop in probe:
        label, distance = c.engine.predict_crop(crop)
        if label is not None:
            votes.setdefault(label, []).append(distance)
    if not votes:
        return
    best_label = min(votes, key=lambda lb: sum(votes[lb]) / len(votes[lb]))
    mean_d = sum(votes[best_label]) / len(votes[best_label])
    if len(votes[best_label]) >= max(2, len(probe) - 1) and mean_d <= s.DUPLICATE_GUARD_THRESHOLD:
        other = c.users.get_by_label(best_label)
        if other:
            sess.duplicate = DuplicateCandidate(other.label, other.username, other.display_name, round(mean_d, 1))
            sess.state = "duplicate_hold"


@bp.post("/api/register/complete")
def register_complete():
    c = _ctx()
    s = c.settings
    ip = _ip()

    body = request.get_json(silent=True) or {}
    force = bool(body.get("force"))
    sess = c.enrollments.get(body.get("enrollment_id"))
    if sess is None or sess.ip != ip:
        return jsonify(error="expired"), 410
    if sess.state == "duplicate_hold" and not force:
        return jsonify(error="duplicate_unconfirmed"), 409
    if sess.state not in ("ready", "duplicate_hold"):
        return jsonify(error="not_ready", state=sess.state), 409
    if c.users.username_taken(sess.username):   # race with another signup using the same name
        c.enrollments.drop(sess.id)
        return jsonify(error="username_taken"), 409

    folder = s.DATASET_ROOT / sess.username
    folder.mkdir(parents=True, exist_ok=True)
    for i, crop in enumerate(sess.samples, start=1):
        cv2.imwrite(str(folder / f"{sess.username}_{i:03}.jpg"), crop)

    try:
        user = c.users.create(
            username=sess.username, display_name=sess.display_name, email=sess.email,
            sample_count=len(sess.samples), source="web",
        )
    except (UsernameTaken, EmailTaken) as e:
        return jsonify(error="conflict", field="username" if isinstance(e, UsernameTaken) else "email"), 409

    c.engine.retrain(c.users.dataset_dirs())
    c.audit.record("registered", ip, request.user_agent.string, label=user.label)
    sess.state = "completed"
    c.enrollments.drop(sess.id)

    return jsonify(state="done", redirect=url_for("aperture.login", registered=1, name=user.display_name))


# ================================================================== login API
@bp.post("/api/login/start")
def login_start():
    c = _ctx()
    if c.users.count() == 0:
        return jsonify(state="no_users"), 409
    wait = c.lockout.seconds_left(_ip())
    if wait:
        return jsonify(state="locked", retry_after=wait), 429
    attempt = c.attempts.start(_ip())
    return jsonify(
        attempt_id=attempt.id,
        required=c.settings.REQUIRED_STREAK,
        ttl=c.settings.ATTEMPT_TTL_SECONDS,
        threshold=c.settings.MATCH_THRESHOLD,
    )


@bp.post("/api/login/frame")
def login_frame():
    """Receive one camera frame (raw JPEG body), verify it, return the attempt state."""
    c = _ctx()
    s = c.settings
    ip = _ip()

    wait = c.lockout.seconds_left(ip)
    if wait:
        return jsonify(state="locked", retry_after=wait), 429

    attempt = c.attempts.get(request.headers.get("X-Attempt-Id"))
    if attempt is None or attempt.ip != ip:
        return jsonify(state="expired"), 410

    if attempt.state != "active":
        return jsonify(state=attempt.state), 200

    now = time.time()
    if now - attempt.created > s.ATTEMPT_TTL_SECONDS:
        attempt.state = "expired"
        c.audit.record("timeout", ip, request.user_agent.string, frames=attempt.frames, label=attempt.best_label)
        return jsonify(state="expired"), 200

    if now - attempt.last_frame < s.MIN_FRAME_INTERVAL:
        return jsonify(error="slow_down"), 429
    attempt.last_frame = now

    frame = c.engine.decode(request.get_data(cache=False))
    if frame is None:
        return jsonify(error="bad_image"), 400

    result = c.engine.analyze(frame)
    state = attempt.register(result, s)

    payload = {
        "state": {"no_face": "searching", "match": "checking", "no_match": "mismatch"}[result.status],
        "face": result.box,
        "distance": result.distance,
        "streak": attempt.streak,
        "required": s.REQUIRED_STREAK,
    }

    if state == "verified":
        user = c.users.get_by_label(attempt.matched_label)
        if user is None:   # shouldn't happen, but never issue a session for an unknown label
            attempt.state = "failed"
            c.audit.record("failed", ip, request.user_agent.string, attempt.best_distance, attempt.frames)
            payload.update(state="failed")
            return jsonify(payload)

        c.lockout.reset(ip)
        distance = attempt.mean_distance
        c.audit.record("success", ip, request.user_agent.string, distance, attempt.frames, label=user.label)
        session.clear()  # fresh session on privilege change (prevents fixation)
        session.permanent = True
        session["user"] = {
            "label": user.label,
            "verified_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "distance": distance,
            "frames": attempt.frames,
            "expires_at": now + s.SESSION_MINUTES * 60,
        }
        payload.update(state="verified", name=user.display_name, redirect=url_for("aperture.dashboard"))

    elif state == "failed":
        c.audit.record("failed", ip, request.user_agent.string, attempt.best_distance, attempt.frames,
                        label=attempt.best_label)
        if c.lockout.record_failure(ip):
            c.audit.record("locked", ip, request.user_agent.string, label=attempt.best_label)
            payload.update(state="locked", retry_after=c.lockout.seconds_left(ip))
        else:
            payload.update(state="failed")

    return jsonify(payload)


# ==================================================================== errors
@bp.app_errorhandler(413)
def too_large(_e):
    return jsonify(error="frame_too_large"), 413


@bp.app_errorhandler(404)
def not_found(_e):
    if request.path.startswith("/api/"):
        return jsonify(error="not_found"), 404
    return render_template("error.html", code=404, title="Page not found",
                           message="That page doesn't exist."), 404
