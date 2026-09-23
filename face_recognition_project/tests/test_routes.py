import cv2

from tests._frames import owner_frame, stranger_frame, two_face_frame
from tests.conftest import REAL_DATASET


def _jpeg(frame):
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    assert ok
    return buf.tobytes()


def _post_login_frame(client, attempt_id, frame):
    return client.post(
        "/api/login/frame", data=_jpeg(frame),
        headers={"Content-Type": "image/jpeg", "X-Attempt-Id": attempt_id},
    )


def _post_reg_frame(client, enrollment_id, frame):
    return client.post(
        "/api/register/frame", data=_jpeg(frame),
        headers={"Content-Type": "image/jpeg", "X-Enrollment-Id": enrollment_id},
    )


def _register(client, settings, username, display_name, frame_source, email=None, force=None):
    """Drive a full registration to completion; frame_source() returns a fresh frame each call."""
    start = client.post("/api/register/start", json={
        "username": username, "display_name": display_name, "email": email,
    })
    assert start.status_code == 200, start.get_json()
    enrollment_id = start.get_json()["enrollment_id"]

    last = None
    for _ in range(settings.ENROLL_TARGET_SAMPLES + 2):
        last = _post_reg_frame(client, enrollment_id, frame_source()).get_json()
        if last["state"] in ("ready", "duplicate_hold"):
            break

    body = {"enrollment_id": enrollment_id}
    if force is not None:
        body["force"] = force
    return client.post("/api/register/complete", json=body)


def _login(client, settings, frame_source):
    start = client.post("/api/login/start")
    assert start.status_code == 200, start.get_json()
    attempt_id = start.get_json()["attempt_id"]
    last = None
    for _ in range(settings.REQUIRED_STREAK + 2):
        last = _post_login_frame(client, attempt_id, frame_source()).get_json()
        if last["state"] in ("verified", "failed", "locked"):
            break
    return last


# ============================================================ basic pages
def test_dashboard_requires_login(client):
    resp = client.get("/dashboard")
    assert resp.status_code == 302
    assert "/login" in resp.headers["Location"]


def test_healthz(client):
    resp = client.get("/healthz")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "ok"
    assert data["users"] == 1   # the seeded legacy "ayush" account


def test_security_headers_present(client):
    resp = client.get("/login")
    assert "Content-Security-Policy" in resp.headers
    assert resp.headers["X-Content-Type-Options"] == "nosniff"


def test_login_page_shows_cold_start_message_when_no_users(client_empty):
    resp = client_empty.get("/login")
    assert resp.status_code == 200
    assert b'data-has-users="false"' in resp.data
    assert b"register" in resp.data.lower()


def test_login_page_ready_state_when_users_exist(client):
    resp = client.get("/login")
    assert b'data-has-users="true"' in resp.data


# ============================================================== login (legacy user)
def test_login_as_preexisting_legacy_user_succeeds(client, settings):
    idx = iter(range(0, len(REAL_DATASET), 5))
    result = _login(client, settings, lambda: owner_frame(REAL_DATASET[next(idx) % len(REAL_DATASET)], scale=1.4))
    assert result["state"] == "verified"
    assert result["name"] == settings.LEGACY_DISPLAY_NAME

    dash = client.get("/dashboard")
    assert dash.status_code == 200
    assert b"Welcome back" in dash.data
    assert settings.LEGACY_USERNAME.encode() in dash.data


def test_stranger_is_rejected_and_never_reaches_dashboard(client, settings):
    result = _login(client, settings, stranger_frame)
    assert result["state"] in ("failed", "locked")
    assert client.get("/dashboard").status_code == 302


def test_login_with_no_users_in_system(client_empty):
    resp = client_empty.post("/api/login/start")
    assert resp.status_code == 409
    assert resp.get_json()["state"] == "no_users"


def test_frame_without_attempt_id_is_rejected(client):
    resp = _post_login_frame(client, "not-a-real-attempt", stranger_frame())
    assert resp.status_code == 410


def test_lockout_after_repeated_failed_attempts(client, settings):
    locked = False
    for _ in range(settings.MAX_FAILED_ATTEMPTS + 1):
        start = client.post("/api/login/start")
        if start.status_code == 429:
            locked = True
            break
        result = _login(client, settings, stranger_frame)
        if result["state"] == "locked":
            locked = True
            break
    assert locked


def test_logout_clears_session(client, settings):
    idx = iter(range(0, len(REAL_DATASET), 5))
    result = _login(client, settings, lambda: owner_frame(REAL_DATASET[next(idx) % len(REAL_DATASET)], scale=1.4))
    assert result["state"] == "verified"
    assert client.get("/dashboard").status_code == 200
    client.post("/logout")
    assert client.get("/dashboard").status_code == 302


# ============================================================ registration
def test_register_new_user_end_to_end_then_login(client, settings):
    resp = _register(client, settings, "newperson", "New Person", stranger_frame, email="new@example.com")
    assert resp.status_code == 200, resp.get_json()
    assert resp.get_json()["state"] == "done"

    # a fresh session (cookie cleared) proves login, not the leftover registration state
    client.delete_cookie("aperture_session")
    result = _login(client, settings, stranger_frame)
    assert result["state"] == "verified"
    assert result["name"] == "New Person"

    dash = client.get("/dashboard")
    assert b"newperson" in dash.data


def test_registering_does_not_break_existing_users_login(client, settings):
    _register(client, settings, "sidekick", "Side Kick", stranger_frame)
    client.delete_cookie("aperture_session")

    idx = iter(range(0, len(REAL_DATASET), 5))
    result = _login(client, settings, lambda: owner_frame(REAL_DATASET[next(idx) % len(REAL_DATASET)], scale=1.4))
    assert result["state"] == "verified"
    assert result["name"] == settings.LEGACY_DISPLAY_NAME


def test_two_registered_users_are_not_confused(client, settings):
    _register(client, settings, "impostor_free", "Someone New", stranger_frame)
    client.delete_cookie("aperture_session")

    # astronaut face must log in as the new user, never as the legacy one
    result = _login(client, settings, stranger_frame)
    assert result["state"] == "verified"
    assert result["name"] == "Someone New"
    client.post("/logout")

    # and the legacy user's face must still log in as themselves
    idx = iter(range(0, len(REAL_DATASET), 5))
    result2 = _login(client, settings, lambda: owner_frame(REAL_DATASET[next(idx) % len(REAL_DATASET)], scale=1.4))
    assert result2["state"] == "verified"
    assert result2["name"] == settings.LEGACY_DISPLAY_NAME


def test_username_already_taken_is_rejected(client, settings):
    resp = client.post("/api/register/start", json={
        "username": settings.LEGACY_USERNAME, "display_name": "Someone Else", "email": None,
    })
    assert resp.status_code == 422
    assert "username" in resp.get_json()["errors"]


def test_invalid_username_is_rejected(client, settings):
    resp = client.post("/api/register/start", json={
        "username": "N O", "display_name": "Bad Name Case", "email": None,
    })
    assert resp.status_code == 422
    assert "username" in resp.get_json()["errors"]


def test_missing_display_name_is_rejected(client, settings):
    resp = client.post("/api/register/start", json={
        "username": "someoneok", "display_name": "  ", "email": None,
    })
    assert resp.status_code == 422
    assert "display_name" in resp.get_json()["errors"]


def test_username_availability_endpoint(client, settings):
    taken = client.get(f"/api/register/username-available?u={settings.LEGACY_USERNAME}")
    assert taken.get_json()["available"] is False
    free = client.get("/api/register/username-available?u=totally_free_name")
    assert free.get_json()["available"] is True


def test_registration_frame_rejects_two_people_in_view(client, settings):
    start = client.post("/api/register/start", json={
        "username": "crowded", "display_name": "Crowded Person", "email": None,
    })
    enrollment_id = start.get_json()["enrollment_id"]

    composite = two_face_frame(REAL_DATASET[20], REAL_DATASET[90])
    resp = _post_reg_frame(client, enrollment_id, composite)
    assert resp.get_json()["state"] == "multiple_faces"


def test_duplicate_guard_flags_reregistration_of_same_face(client, settings):
    """Registering a new account with the legacy user's own face should be
    flagged, not silently allowed through."""
    idx = iter(range(0, len(REAL_DATASET), 3))
    resp = _register(client, settings, "second_account_same_face", "Duplicate Attempt",
                      lambda: owner_frame(REAL_DATASET[next(idx) % len(REAL_DATASET)], scale=1.4))
    assert resp.status_code == 409
    assert resp.get_json()["error"] == "duplicate_unconfirmed"


def test_duplicate_guard_can_be_overridden_with_force(client, settings):
    idx = iter(range(0, len(REAL_DATASET), 3))
    resp = _register(client, settings, "second_account_forced", "Duplicate But Forced",
                      lambda: owner_frame(REAL_DATASET[next(idx) % len(REAL_DATASET)], scale=1.4),
                      force=True)
    assert resp.status_code == 200
    assert resp.get_json()["state"] == "done"


def test_register_on_empty_system_becomes_first_user_and_can_log_in(client_empty, settings_no_legacy):
    resp = _register(client_empty, settings_no_legacy, "pioneer", "First Person", stranger_frame)
    assert resp.status_code == 200, resp.get_json()

    client_empty.delete_cookie("aperture_session")
    result = _login(client_empty, settings_no_legacy, stranger_frame)
    assert result["state"] == "verified"
    assert result["name"] == "First Person"


def test_registration_expired_enrollment_id_is_rejected(client):
    resp = _post_reg_frame(client, "not-a-real-session", stranger_frame())
    assert resp.status_code == 410


# ============================================================ account management
def test_account_update_requires_login(client):
    resp = client.post("/api/account/update", json={"display_name": "New Name"})
    assert resp.status_code == 401


def test_account_update_changes_name_and_email(client, settings):
    idx = iter(range(0, len(REAL_DATASET), 5))
    _login(client, settings, lambda: owner_frame(REAL_DATASET[next(idx) % len(REAL_DATASET)], scale=1.4))

    resp = client.post("/api/account/update", json={
        "display_name": "Ayush Updated", "email": "ayush.updated@example.com",
    })
    assert resp.status_code == 200, resp.get_json()
    assert resp.get_json()["display_name"] == "Ayush Updated"

    dash = client.get("/dashboard")
    assert b"Ayush Updated" in dash.data


def test_account_update_rejects_empty_name(client, settings):
    idx = iter(range(0, len(REAL_DATASET), 5))
    _login(client, settings, lambda: owner_frame(REAL_DATASET[next(idx) % len(REAL_DATASET)], scale=1.4))
    resp = client.post("/api/account/update", json={"display_name": "  ", "email": ""})
    assert resp.status_code == 422
    assert "display_name" in resp.get_json()["errors"]


def test_account_update_rejects_email_taken_by_another_account(client, settings):
    _register(client, settings, "sidekick2", "Side Kick Two", stranger_frame, email="taken@example.com")
    client.delete_cookie("aperture_session")

    idx = iter(range(0, len(REAL_DATASET), 5))
    _login(client, settings, lambda: owner_frame(REAL_DATASET[next(idx) % len(REAL_DATASET)], scale=1.4))
    resp = client.post("/api/account/update", json={"display_name": "Ayush", "email": "taken@example.com"})
    assert resp.status_code == 409
    assert "email" in resp.get_json()["errors"]


def test_account_delete_requires_login(client):
    resp = client.post("/api/account/delete")
    assert resp.status_code == 401


def test_account_delete_removes_account_and_logs_out(client, settings):
    _register(client, settings, "throwaway", "Throw Away", stranger_frame)
    client.delete_cookie("aperture_session")
    _login(client, settings, stranger_frame)
    assert client.get("/dashboard").status_code == 200

    resp = client.post("/api/account/delete")
    assert resp.status_code == 200
    assert resp.get_json()["state"] == "done"

    # session is gone
    assert client.get("/dashboard").status_code == 302
    # and the account can no longer log in
    result = _login(client, settings, stranger_frame)
    assert result["state"] in ("failed", "locked")


def test_account_delete_does_not_affect_other_users(client, settings):
    _register(client, settings, "temp_user", "Temp User", stranger_frame)
    client.delete_cookie("aperture_session")
    _login(client, settings, stranger_frame)
    client.post("/api/account/delete")

    # the legacy user must still be able to log in after the retrain
    idx = iter(range(0, len(REAL_DATASET), 5))
    result = _login(client, settings, lambda: owner_frame(REAL_DATASET[next(idx) % len(REAL_DATASET)], scale=1.4))
    assert result["state"] == "verified"
    assert result["name"] == settings.LEGACY_DISPLAY_NAME


def test_username_freed_after_delete_can_be_reregistered(client, settings):
    _register(client, settings, "recyclable", "First Owner", stranger_frame)
    client.delete_cookie("aperture_session")
    _login(client, settings, stranger_frame)
    client.post("/api/account/delete")
    client.delete_cookie("aperture_session")

    resp = _register(client, settings, "recyclable", "Second Owner", stranger_frame)
    assert resp.status_code == 200, resp.get_json()
