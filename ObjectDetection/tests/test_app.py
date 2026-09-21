import io
import os
import re
import time

import pytest

import detector
from app import allowed, create_app, display_name
from helpers import csrf_token, face_row, image_bytes, upload

SCENE = dict(objects=[(50, 60, 200, 220, 0.9, 2), (300, 40, 380, 200, 0.8, 0)],
             faces=[face_row(310, 50, 40, 40, 0.9)])


def alerts(response):
    return re.findall(r'class="alert" role="alert">\s*([^<]*?)\s*<', response.get_data(as_text=True))


# ------------------------------------------------------------------ pages and security
def test_home_page_renders_the_form(client):
    page = client.get("/").get_data(as_text=True)
    assert 'name="_csrf"' in page
    assert ".jpg,.jpeg,.png,.webp" in page  # the picker no longer offers GIF/BMP
    assert "display:none" not in page


def test_post_without_csrf_token_is_rejected(client):
    response = client.post("/", data={"image": (io.BytesIO(image_bytes()), "a.png")},
                           content_type="multipart/form-data")
    assert response.status_code == 400
    assert "expired" in response.get_data(as_text=True)


def test_security_headers_are_set(client):
    headers = client.get("/").headers
    assert headers["X-Content-Type-Options"] == "nosniff"
    assert "default-src 'self'" in headers["Content-Security-Policy"]


def test_secret_key_is_not_hard_coded(monkeypatch):
    monkeypatch.delenv("SECRET_KEY", raising=False)
    keys = {create_app({"WARMUP": False}).config["SECRET_KEY"] for _ in range(2)}
    assert len(keys) == 2, "each start-up must generate its own random key"
    assert "visionscan-local" not in keys


def test_health_reports_model_state(client, models):
    data = client.get("/health").get_json()
    assert data["object_model"]["state"] == "ready"
    assert client.get("/health").headers["Cache-Control"] == "no-store"


# ------------------------------------------------------------------ the scan flow
def test_successful_scan_redirects_then_shows_the_result(client, models, results_dir):
    models(**SCENE)
    response = upload(client, image_bytes(600, 400))
    assert response.status_code == 302  # Post/Redirect/Get: refreshing does not re-upload
    assert "/result/" in response.headers["Location"]

    page = client.get(response.headers["Location"])
    text = page.get_data(as_text=True)
    assert page.status_code == 200
    assert "Detection result" in text and "photo.png" in text
    assert re.search(r"<strong>3</strong><span>Detections", text)
    assert "Car" in text and "Person" in text and "Face" in text

    rid = response.headers["Location"].rsplit("/", 1)[1]
    image = client.get(f"/results/{rid}.jpg")
    assert image.status_code == 200 and image.mimetype == "image/jpeg"
    assert image.data[:2] == b"\xff\xd8"

    download = client.get(f"/result/{rid}/download")
    assert "attachment" in download.headers["Content-Disposition"]
    assert "photo_detected.jpg" in download.headers["Content-Disposition"]


def test_original_upload_is_never_written_to_disk(client, models, results_dir):
    models()
    rid = upload(client, image_bytes()).headers["Location"].rsplit("/", 1)[1]
    assert sorted(p.name for p in results_dir.iterdir()) == sorted([f"{rid}.jpg", f"{rid}.json"])


def test_delete_button_removes_the_result(client, models, results_dir):
    models()
    rid = upload(client, image_bytes()).headers["Location"].rsplit("/", 1)[1]
    response = client.post(f"/result/{rid}/delete", data={"_csrf": csrf_token(client)},
                           follow_redirects=True)
    assert "Result deleted." in response.get_data(as_text=True)
    assert list(results_dir.iterdir()) == []
    assert client.get(f"/results/{rid}.jpg").status_code == 404


def test_delete_requires_the_csrf_token(client, models, results_dir):
    models()
    rid = upload(client, image_bytes()).headers["Location"].rsplit("/", 1)[1]
    assert client.post(f"/result/{rid}/delete").status_code == 400
    assert len(list(results_dir.iterdir())) == 2


def test_unknown_or_expired_result_redirects_home(client):
    response = client.get("/result/" + "a" * 32, follow_redirects=True)
    assert "expired" in response.get_data(as_text=True)
    assert client.get("/results/not-an-id.jpg").status_code == 404


def test_scan_shows_the_face_warning_but_keeps_objects(client, models):
    models(objects=[(50, 60, 200, 220, 0.9, 2)],
           face_error=detector.ModelUnavailable("the face model could not be downloaded"))
    response = upload(client, image_bytes())
    page = client.get(response.headers["Location"]).get_data(as_text=True)
    assert 'class="notice"' in page and "Face detection is unavailable" in page
    assert "Car" in page


# ------------------------------------------------------------------ validation and errors
@pytest.mark.parametrize("name", ["x.gif", "x.bmp", "noext", "archive.tar", ".jpg.exe"])
def test_unsupported_extensions_are_rejected(client, models, name):
    response = upload(client, image_bytes(), name=name)
    follow = client.get(response.headers["Location"])
    assert "Unsupported format" in follow.get_data(as_text=True)


@pytest.mark.parametrize("name", ["a.jpg", "A.JPG", "a.jpeg", "a.png", "a.webp", "x.tar.jpg"])
def test_supported_extensions_are_accepted(name):
    assert allowed(name)


def test_file_that_is_not_an_image_is_rejected_and_nothing_is_stored(client, models, results_dir):
    response = upload(client, b"just some text", name="fake.jpg", )
    follow = client.get(response.headers["Location"])
    assert "not a valid image" in follow.get_data(as_text=True)
    assert list(results_dir.iterdir()) == []


def test_oversized_upload_gets_a_friendly_message(tmp_path):
    """Old bug: a bare 'Request Entity Too Large' page."""
    app = create_app({"TESTING": True, "WARMUP": False, "RESULT_DIR": str(tmp_path),
                      "MAX_CONTENT_LENGTH": 1024 * 1024})
    client = app.test_client()
    token = csrf_token(client)
    response = client.post("/", data={"_csrf": token, "image": (io.BytesIO(b"0" * 2_000_000), "big.jpg")},
                           content_type="multipart/form-data")
    text = response.get_data(as_text=True)
    assert response.status_code == 413
    assert "too large" in text and "limit is 1 MB" in text
    assert "<title>413" not in text  # not the stock Werkzeug page


def test_internal_errors_are_logged_but_not_shown(client, models, caplog):
    """Old bug: 'Detection failed: <exception text with file paths>'."""
    models(predict_error=RuntimeError(r"cannot open C:\Users\secret\models\face.onnx"))
    response = upload(client, image_bytes())
    text = client.get(response.headers["Location"]).get_data(as_text=True)
    assert "Something went wrong" in text
    assert "secret" not in text and "face.onnx" not in text
    assert "secret" in caplog.text  # ...but the operator can still find it in the log


def test_slider_values_are_used_and_clamped(client, models):
    yolo = models()
    upload(client, image_bytes(), object_conf="60", face_conf="40")
    assert yolo.calls[-1]["conf"] == 0.6
    upload(client, image_bytes(), object_conf="abc")
    assert yolo.calls[-1]["conf"] == detector.DEFAULT_OBJECT_CONF
    upload(client, image_bytes(), object_conf="500")
    assert yolo.calls[-1]["conf"] == 0.95
    upload(client, image_bytes(), object_conf="nan")
    assert yolo.calls[-1]["conf"] == detector.DEFAULT_OBJECT_CONF


# ------------------------------------------------------------------ filenames
@pytest.mark.parametrize("name", ["फोटो.jpg", "照片.webp", "Ünï café.png", "my photo (1).jpg"])
def test_non_english_filenames_are_shown_correctly(client, models, name):
    """Old bug: secure_filename() reduced these to just 'jpg' / 'webp'."""
    models()
    response = upload(client, image_bytes(), name=name)
    page = client.get(response.headers["Location"]).get_data(as_text=True)
    assert name in page


def test_display_name_strips_paths_and_control_characters():
    assert display_name("../../etc/passwd.jpg") == "passwd.jpg"
    assert display_name("C:\\Users\\me\\pic.png") == "pic.png"
    assert display_name("a\x00b\nc.jpg") == "abc.jpg"
    assert display_name("///") == "image"
    assert len(display_name("x" * 500 + ".jpg")) == 120


def test_filenames_are_html_escaped(client, models):
    models()
    response = upload(client, image_bytes(), name='<script>alert(1)</script>.jpg')
    page = client.get(response.headers["Location"]).get_data(as_text=True)
    assert "<script>alert(1)" not in page


# ------------------------------------------------------------------ housekeeping
def test_expired_results_are_removed_automatically(models, tmp_path):
    """Old bug: every upload and result stayed on disk forever."""
    models()
    old = detector.detect_image(image_bytes(), "old.png", tmp_path)
    past = time.time() - 3600
    for suffix in (".jpg", ".json"):
        os.utime(tmp_path / f"{old['id']}{suffix}", (past, past))
    app2 = create_app({"TESTING": True, "WARMUP": False, "SECRET_KEY": "k",
                       "RESULT_DIR": str(tmp_path), "RESULT_TTL_MINUTES": 10})  # startup cleanup
    assert detector.load_result(tmp_path, old["id"]) is None
    assert app2.test_client().get("/").status_code == 200


# ------------------------------------------------------------------ start-up
def test_models_are_warmed_up_in_the_background_only_when_enabled(monkeypatch, tmp_path):
    import threading

    started = threading.Event()
    monkeypatch.setattr(detector, "warm_up", started.set)
    create_app({"WARMUP": False, "RESULT_DIR": str(tmp_path)})
    assert not started.wait(0.3)
    create_app({"WARMUP": True, "RESULT_DIR": str(tmp_path)})
    assert started.wait(2)
