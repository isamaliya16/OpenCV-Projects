import glob
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from webapp.settings import Settings  # noqa: E402
from webapp.face_engine import FaceEngine  # noqa: E402
from webapp import create_app  # noqa: E402

REAL_DATASET = sorted(glob.glob(str(ROOT / "dataset" / "ayush" / "*.jpg")))


@pytest.fixture(scope="session")
def engine():
    """A FaceEngine over the project's real, pre-trained model — read-only.
    Used for detection/recognition unit tests that don't need to register
    anyone or retrain (retraining happens only against the isolated `app`
    fixture below, so these tests never touch the real trained model file)."""
    return FaceEngine(Settings)


def _make_test_settings(tmp_path, extra: dict | None = None):
    class TestSettings(Settings):
        REQUIRED_STREAK = 3
        MAX_MISMATCH_FRAMES = 4
        MAX_FAILED_ATTEMPTS = 3
        LOCKOUT_SECONDS = 2
        LOCKOUT_WINDOW_SECONDS = 60
        MIN_FRAME_INTERVAL = 0.0
        ATTEMPT_TTL_SECONDS = 40
        COOKIE_SECURE = False

        ENROLL_TARGET_SAMPLES = 6
        ENROLL_MIN_SAMPLE_INTERVAL = 0.0
        ENROLL_MIN_FRAME_INTERVAL = 0.0
        ENROLL_TTL_SECONDS = 60
        MAX_REGISTRATIONS_PER_IP = 20
        REGISTRATION_WINDOW_SECONDS = 3600
        DUPLICATE_GUARD_THRESHOLD = 45.0

    ts = TestSettings
    ts.DB_PATH = tmp_path / "auth.db"
    ts.DATASET_ROOT = tmp_path / "dataset"
    ts.MODEL_PATH = tmp_path / "trainer" / "face_model.yml"
    ts.LEGACY_DATASET_PATH = ts.DATASET_ROOT / ts.LEGACY_USERNAME
    for k, v in (extra or {}).items():
        setattr(ts, k, v)
    return ts


@pytest.fixture()
def settings_no_legacy(tmp_path):
    """Isolated settings with a completely empty dataset/DB — a fresh install."""
    ts = _make_test_settings(tmp_path)
    ts.DATASET_ROOT.mkdir(parents=True, exist_ok=True)
    return ts


@pytest.fixture()
def settings(tmp_path):
    """Isolated settings seeded with a small copy of the real 'ayush' dataset,
    so the legacy-import path runs and there's a first, pre-existing user."""
    ts = _make_test_settings(tmp_path)
    legacy_dir = ts.DATASET_ROOT / ts.LEGACY_USERNAME
    legacy_dir.mkdir(parents=True, exist_ok=True)
    for f in REAL_DATASET[::4]:   # ~25 images: enough to train/recognise reliably, fast to retrain
        shutil.copy(f, legacy_dir / Path(f).name)
    return ts


@pytest.fixture()
def app(settings):
    application = create_app(settings)
    application.config.update(TESTING=True)
    return application


@pytest.fixture()
def app_empty(settings_no_legacy):
    application = create_app(settings_no_legacy)
    application.config.update(TESTING=True)
    return application


@pytest.fixture()
def client(app):
    return app.test_client()


@pytest.fixture()
def client_empty(app_empty):
    return app_empty.test_client()
