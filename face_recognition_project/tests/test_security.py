from webapp.face_engine import FaceResult
from webapp.security import Attempt, AttemptStore, LockoutTracker, RegistrationThrottle


def test_streak_of_matches_verifies(settings):
    a = Attempt(id="x", ip="1.2.3.4", created=0)
    state = None
    for _ in range(settings.REQUIRED_STREAK):
        state = a.register(FaceResult(status="match", distance=20.0, label=1), settings)
    assert state == "verified"
    assert a.matched_label == 1
    assert a.mean_distance == 20.0


def test_mismatch_resets_streak(settings):
    a = Attempt(id="x", ip="1.2.3.4", created=0)
    a.register(FaceResult(status="match", distance=20.0, label=1), settings)
    a.register(FaceResult(status="match", distance=20.0, label=1), settings)
    state = a.register(FaceResult(status="no_match", distance=90.0, label=1), settings)
    assert state == "active"
    assert a.streak == 0
    assert a.matched_label is None


def test_streak_does_not_cross_identities(settings):
    """A run of matches for user A followed by matches for user B must not
    combine into a single verifying streak for either of them."""
    a = Attempt(id="x", ip="1.2.3.4", created=0)
    a.register(FaceResult(status="match", distance=20.0, label=1), settings)
    a.register(FaceResult(status="match", distance=20.0, label=1), settings)
    state = a.register(FaceResult(status="match", distance=22.0, label=2), settings)
    assert state == "active"
    assert a.streak == 1          # restarted, not 3
    assert a.matched_label == 2


def test_too_many_mismatches_fails_attempt(settings):
    a = Attempt(id="x", ip="1.2.3.4", created=0)
    state = "active"
    for _ in range(settings.MAX_MISMATCH_FRAMES):
        state = a.register(FaceResult(status="no_match", distance=90.0, label=1), settings)
    assert state == "failed"


def test_best_label_tracks_closest_ever_seen(settings):
    a = Attempt(id="x", ip="1.2.3.4", created=0)
    a.register(FaceResult(status="no_match", distance=90.0, label=1), settings)
    a.register(FaceResult(status="no_match", distance=60.0, label=2), settings)
    a.register(FaceResult(status="no_match", distance=75.0, label=1), settings)
    assert a.best_distance == 60.0
    assert a.best_label == 2


def test_brief_no_face_gap_does_not_reset_streak(settings):
    a = Attempt(id="x", ip="1.2.3.4", created=0)
    a.register(FaceResult(status="match", distance=20.0, label=1), settings)
    for _ in range(settings.MAX_NO_FACE_GAP):
        a.register(FaceResult(status="no_face"), settings)
    assert a.streak == 1  # tolerated, not reset
    a.register(FaceResult(status="no_face"), settings)  # exceeds the tolerated gap
    assert a.streak == 0


def test_attempt_store_starting_new_expires_old(settings):
    store = AttemptStore(settings)
    first = store.start("9.9.9.9")
    second = store.start("9.9.9.9")
    assert store.get(first.id).state == "expired"
    assert store.get(second.id).state == "active"


def test_lockout_after_max_failures(settings):
    lock = LockoutTracker(settings)
    ip = "5.5.5.5"
    for _ in range(settings.MAX_FAILED_ATTEMPTS - 1):
        assert lock.record_failure(ip) is False
    assert lock.record_failure(ip) is True
    assert lock.seconds_left(ip) > 0


def test_lockout_reset_clears_state(settings):
    lock = LockoutTracker(settings)
    ip = "6.6.6.6"
    for _ in range(settings.MAX_FAILED_ATTEMPTS):
        lock.record_failure(ip)
    assert lock.seconds_left(ip) > 0
    lock.reset(ip)
    assert lock.seconds_left(ip) == 0


def test_registration_throttle_allows_up_to_limit(settings):
    throttle = RegistrationThrottle(settings)
    ip = "7.7.7.7"
    for _ in range(settings.MAX_REGISTRATIONS_PER_IP):
        assert throttle.allow(ip) is True
    assert throttle.allow(ip) is False
