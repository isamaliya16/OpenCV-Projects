import os
import sys
from pathlib import Path

# Importing `app` builds the module-level app object. Switch the background model
# warm-up off *before* that import so the tests never load models or touch the network.
os.environ["WARMUP"] = "0"

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import detector  # noqa: E402
from app import create_app  # noqa: E402
from helpers import FakeFaces, FakeModel, FakeYOLO  # noqa: E402


@pytest.fixture
def models(monkeypatch):
    """Install fake models. Call it again inside a test to change what they return.

    Returns the FakeYOLO so tests can inspect how it was called."""

    def install(objects=(), faces=(), face_error=None, object_error=None, predict_error=None):
        yolo = FakeYOLO(objects, error=predict_error)
        monkeypatch.setattr(detector, "object_model", FakeModel(yolo, object_error))
        monkeypatch.setattr(detector, "face_model", FakeModel(FakeFaces(faces), face_error))
        return yolo

    install()
    return install


@pytest.fixture
def app(tmp_path):
    return create_app({
        "TESTING": True,
        "WARMUP": False,
        "SECRET_KEY": "test-key",
        "RESULT_DIR": str(tmp_path / "results"),
    })


@pytest.fixture
def client(app):
    return app.test_client()


@pytest.fixture
def results_dir(app):
    return Path(app.config["RESULT_DIR"])
