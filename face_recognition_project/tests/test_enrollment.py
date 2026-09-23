import numpy as np

from webapp.enrollment import EnrollmentSession, EnrollmentStore, validate_registration
from webapp.face_engine import EnrollResult


def _crop(settings):
    return np.zeros(settings.FACE_SIZE, dtype="uint8")


def test_accepts_samples_until_target(settings):
    sess = EnrollmentSession(id="e1", ip="1.1.1.1", username="jane", display_name="Jane",
                              email=None, created=0, target=settings.ENROLL_TARGET_SAMPLES)
    state = "collecting"
    for _ in range(settings.ENROLL_TARGET_SAMPLES):
        state = sess.accept_frame(EnrollResult(status="ok", box={}, crop=_crop(settings)), settings)
    assert state == "ready"
    assert sess.progress == settings.ENROLL_TARGET_SAMPLES


def test_no_face_and_multiple_faces_do_not_count_as_samples(settings):
    sess = EnrollmentSession(id="e1", ip="1.1.1.1", username="jane", display_name="Jane",
                              email=None, created=0, target=settings.ENROLL_TARGET_SAMPLES)
    sess.accept_frame(EnrollResult(status="no_face"), settings)
    sess.accept_frame(EnrollResult(status="multiple_faces", box={}), settings)
    assert sess.progress == 0
    assert sess.state == "collecting"


def test_frames_over_cap_expire_the_session(settings):
    sess = EnrollmentSession(id="e1", ip="1.1.1.1", username="jane", display_name="Jane",
                              email=None, created=0, target=999999)   # unreachable target
    state = "collecting"
    for _ in range(settings.ENROLL_MAX_FRAMES):
        state = sess.accept_frame(EnrollResult(status="no_face"), settings)
    assert state == "expired"


def test_enrollment_store_one_session_per_ip(settings):
    store = EnrollmentStore(settings)
    first = store.start("2.2.2.2", "jane", "Jane", None)
    second = store.start("2.2.2.2", "jane2", "Jane Two", None)
    assert store.get(first.id).state == "cancelled"
    assert store.get(second.id).state == "collecting"


def test_validate_registration_rejects_bad_username(settings):
    errors = validate_registration("a", "Jane Doe", "", settings)
    assert "username" in errors


def test_validate_registration_rejects_uppercase_or_symbols(settings):
    errors = validate_registration("Jane Doe!", "Jane Doe", "", settings)
    assert "username" in errors


def test_validate_registration_requires_display_name(settings):
    errors = validate_registration("janedoe", "  ", "", settings)
    assert "display_name" in errors


def test_validate_registration_checks_email_format(settings):
    errors = validate_registration("janedoe", "Jane Doe", "not-an-email", settings)
    assert "email" in errors


def test_validate_registration_allows_blank_email(settings):
    errors = validate_registration("janedoe", "Jane Doe", "", settings)
    assert errors == {}


def test_validate_registration_accepts_good_input(settings):
    errors = validate_registration("jane.doe-2", "Jane Doe", "jane@example.com", settings)
    assert errors == {}
