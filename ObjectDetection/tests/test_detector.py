import http.client
import os
import threading
import time

import pytest

import detector
from detector import DetectionError, LazyModel, ModelUnavailable
from helpers import (LocalServer, colour_pixels, face_row, image_bytes, read_result)

CAR = (100, 8, 300, 120, 0.91, 2)  # a box that starts 8 px from the top edge


# ------------------------------------------------------------------ drawing
def test_top_of_photo_is_not_covered_by_the_header(models, tmp_path):
    """Old bug: a solid 58 px banner painted over the top of the photo."""
    models(objects=[CAR])
    meta = detector.detect_image(image_bytes(600, 335), "a.png", tmp_path)
    out = read_result(tmp_path, meta)
    strip = out.shape[0] - 335
    assert strip > 0, "the summary must be a strip added above the photo"
    top_of_photo = out[strip:strip + 58]
    assert colour_pixels(top_of_photo, detector.OBJECT_RGB) > 200


def test_box_colours_match_the_ui_legend(models, tmp_path):
    """Old bug: BGR/RGB mix-up drew object boxes blue while the legend was orange."""
    models(objects=[(100, 150, 300, 300, 0.9, 2)], faces=[face_row(400, 150, 60, 60)])
    meta = detector.detect_image(image_bytes(600, 400), "a.png", tmp_path)
    out = read_result(tmp_path, meta)
    assert colour_pixels(out, detector.OBJECT_RGB) > 300
    assert colour_pixels(out, detector.FACE_RGB) > 100
    assert colour_pixels(out, (51, 167, 240)) == 0  # the old, wrong blue


@pytest.mark.parametrize("width", [200, 260, 320, 480, 640, 1280, 3000])
def test_header_text_always_fits(width):
    """Old bug: the banner text was cut off on images narrower than ~570 px."""
    texts = ["Scan complete | Objects: 12 | Faces: 3", "Objects 12 | Faces 3"]
    _, _, _, text_width, _, _, margin = detector.header_layout(texts, width, int(width * 0.7))
    assert text_width <= width - 2 * margin


def test_labels_stay_inside_the_image(models, tmp_path):
    models(objects=[(560, 300, 599, 334, 0.9, 2)])  # box in the bottom-right corner
    meta = detector.detect_image(image_bytes(600, 335), "a.png", tmp_path)
    out = read_result(tmp_path, meta)
    assert out.shape[1] == 600  # nothing widened the canvas
    assert colour_pixels(out, detector.OBJECT_RGB) > 50


# ------------------------------------------------------------------ boxes
def test_face_cut_off_by_left_edge_is_not_enlarged(models, tmp_path):
    """Old bug: x was clamped before x + w, so the box grew by the hidden part."""
    models(faces=[face_row(-20, 150, 60, 60)])
    meta = detector.detect_image(image_bytes(600, 335), "a.png", tmp_path)
    face = next(d for d in meta["detections"] if d["type"] == "face")
    assert face["box"] == [0, 150, 40, 210]


def test_face_cut_off_by_top_edge_is_not_enlarged(models, tmp_path):
    models(faces=[face_row(200, -30, 50, 100)])
    meta = detector.detect_image(image_bytes(600, 335), "a.png", tmp_path)
    face = next(d for d in meta["detections"] if d["type"] == "face")
    assert face["box"] == [200, 0, 250, 70]


def test_clamp_box_cases():
    assert detector.clamp_box(-20, -10, 40, 30, 100, 100) == [0, 0, 40, 30]
    assert detector.clamp_box(90, 90, 200, 200, 100, 100) == [90, 90, 99, 99]
    assert detector.clamp_box(50, 60, 10, 20, 100, 100) == [10, 20, 50, 60]  # swapped corners
    assert detector.clamp_box(-50, 10, -5, 40, 100, 100) is None  # completely outside
    assert detector.clamp_box(10, 10, 10, 40, 100, 100) is None  # zero width


def test_faces_are_found_at_reduced_size_but_boxes_use_original_pixels(models, monkeypatch, tmp_path):
    monkeypatch.setattr(detector, "FACE_MAX_SIDE", 500)
    # In the 500 px probe image (scale 0.5 of a 1000 px one) the face is at 100,100 50x50.
    models(faces=[face_row(100, 100, 50, 50)])
    meta = detector.detect_image(image_bytes(1000, 600), "big.png", tmp_path)
    face = next(d for d in meta["detections"] if d["type"] == "face")
    assert face["box"] == [200, 200, 300, 300]


# ------------------------------------------------------------------ pipeline behaviour
def test_face_failure_keeps_the_object_results(models, tmp_path):
    models(objects=[CAR], face_error=ModelUnavailable("the face model could not be downloaded"))
    meta = detector.detect_image(image_bytes(600, 335), "a.png", tmp_path)
    assert meta["object_count"] == 1 and meta["face_count"] == 0
    assert len(meta["warnings"]) == 1
    assert "Face detection is unavailable" in meta["warnings"][0]


def test_object_model_failure_is_reported_as_a_user_safe_error(models, tmp_path):
    models(object_error=ModelUnavailable("the YOLO weights could not be loaded"))
    with pytest.raises(DetectionError, match="Object detection is unavailable"):
        detector.detect_image(image_bytes(), "a.png", tmp_path)


def test_confidence_settings_reach_the_models(models, tmp_path):
    yolo = models(objects=[(10, 10, 100, 100, 0.5, 2), (120, 10, 220, 100, 0.8, 0)],
                  faces=[face_row(300, 10, 40, 40, 0.6), face_row(400, 10, 40, 40, 0.9)])
    meta = detector.detect_image(image_bytes(), "a.png", tmp_path, object_conf=0.6, face_conf=0.7)
    assert yolo.calls[0]["conf"] == 0.6
    assert meta["object_count"] == 1 and meta["face_count"] == 1


def test_summary_counts_classes(models, tmp_path):
    models(objects=[(10, 10, 60, 60, 0.9, 2), (70, 10, 120, 60, 0.8, 2), (130, 10, 180, 60, 0.7, 0)])
    meta = detector.detect_image(image_bytes(), "a.png", tmp_path)
    assert meta["summary"] == [{"label": "car", "count": 2}, {"label": "person", "count": 1}]
    assert meta["total"] == 3


def test_only_the_annotated_result_is_stored(models, tmp_path):
    models()
    meta = detector.detect_image(image_bytes(), "a.png", tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted([f"{meta['id']}.jpg", f"{meta['id']}.json"])


# ------------------------------------------------------------------ decoding
def test_decode_rejects_bad_input():
    with pytest.raises(DetectionError, match="empty"):
        detector.decode_image(b"")
    with pytest.raises(DetectionError, match="not a valid image"):
        detector.decode_image(b"this is not an image")


def test_decode_rejects_decompression_bombs(monkeypatch):
    monkeypatch.setattr(detector, "MAX_PIXELS", 1000)
    with pytest.raises(DetectionError, match="too large"):
        detector.decode_image(image_bytes(100, 100))


def test_huge_images_are_scaled_down(monkeypatch):
    monkeypatch.setattr(detector, "MAX_SIDE", 200)
    img = detector.decode_image(image_bytes(800, 400))
    assert img.shape[:2] == (100, 200)


# ------------------------------------------------------------------ face-model download
def test_interrupted_download_leaves_nothing_behind(tmp_path):
    """Old bug: a truncated file stayed on disk and broke every later scan."""
    server = LocalServer(b"x" * 1000, declared_length=230_000)  # promises 230 KB, sends 1 KB
    try:
        dest = tmp_path / "model.onnx"
        with pytest.raises((OSError, http.client.HTTPException)):
            detector._download(server.url, dest)
    finally:
        server.close()
    assert list(tmp_path.iterdir()) == []


def test_complete_download_is_moved_into_place(tmp_path):
    body = bytes(range(256)) * 1000  # 256 KB
    server = LocalServer(body)
    try:
        dest = tmp_path / "model.onnx"
        detector._download(server.url, dest)
    finally:
        server.close()
    assert dest.read_bytes() == body
    assert [p.name for p in tmp_path.iterdir()] == ["model.onnx"]


def test_git_lfs_pointer_is_rejected(tmp_path):
    server = LocalServer(b"version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 1\n")
    try:
        with pytest.raises(OSError, match="LFS"):
            detector._download(server.url, tmp_path / "model.onnx", min_bytes=10)
    finally:
        server.close()
    assert list(tmp_path.iterdir()) == []


def test_download_gives_up_when_the_server_stalls(tmp_path):
    import socket

    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)  # accepts the connection but never answers
    port = listener.getsockname()[1]
    started = time.monotonic()
    try:
        with pytest.raises(OSError):
            detector._download(f"http://127.0.0.1:{port}/m.onnx", tmp_path / "m.onnx", timeout=1)
    finally:
        listener.close()
    assert time.monotonic() - started < 10


def test_corrupt_face_model_from_an_old_run_is_deleted(tmp_path, monkeypatch):
    """Self-healing for the poisoned file the old version could leave behind."""
    monkeypatch.setattr(detector, "MODEL_DIR", tmp_path)
    monkeypatch.setattr(detector, "YUNET_PATH", tmp_path / detector.YUNET_FILE)
    (tmp_path / detector.YUNET_FILE).write_bytes(b"truncated garbage" * 10)

    def offline(*args, **kwargs):
        raise OSError("no internet")

    monkeypatch.setattr(detector, "_download", offline)
    with pytest.raises(ModelUnavailable):
        detector._load_face_detector()
    assert not (tmp_path / detector.YUNET_FILE).exists()


def test_failed_download_message_contains_no_local_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(detector, "MODEL_DIR", tmp_path)
    monkeypatch.setattr(detector, "YUNET_PATH", tmp_path / detector.YUNET_FILE)
    monkeypatch.setattr(detector, "_download", lambda *a, **k: (_ for _ in ()).throw(OSError("boom")))
    with pytest.raises(ModelUnavailable) as info:
        detector.ensure_face_model()
    assert str(tmp_path) not in str(info.value)


# ------------------------------------------------------------------ lazy loading
def test_lazy_model_loads_once_even_with_many_threads():
    calls = []

    def loader():
        calls.append(1)
        time.sleep(0.05)
        return object()

    model = LazyModel("t", loader)
    results = []
    threads = [threading.Thread(target=lambda: results.append(model.get())) for _ in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert len(calls) == 1 and len({id(r) for r in results}) == 1
    assert model.status()["state"] == "ready"


def test_lazy_model_remembers_failure_then_retries():
    attempts = []

    def loader():
        attempts.append(1)
        if len(attempts) == 1:
            raise ModelUnavailable("offline")
        return "model"

    model = LazyModel("t", loader, retry_after=0.2)
    with pytest.raises(ModelUnavailable):
        model.get()
    with pytest.raises(ModelUnavailable):  # inside the window: no new attempt
        model.get()
    assert len(attempts) == 1 and model.status()["state"] == "error"
    time.sleep(0.25)
    assert model.get() == "model" and len(attempts) == 2


def test_lazy_model_hides_unexpected_error_details():
    def loader():
        raise RuntimeError(r"C:\Users\secret\path")

    model = LazyModel("t", loader)
    with pytest.raises(ModelUnavailable) as info:
        model.get()
    assert "secret" not in str(info.value)


def test_weights_are_found_next_to_the_app_not_in_the_current_directory(tmp_path, monkeypatch):
    (tmp_path / "yolo26m.pt").write_bytes(b"x")
    monkeypatch.setattr(detector, "MODEL_DIR", tmp_path)
    elsewhere = tmp_path / "other"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    assert detector._resolve_weights("yolo26m.pt") == str(tmp_path / "yolo26m.pt")
    assert detector._resolve_weights("not-here.pt") == "not-here.pt"  # left for Ultralytics


# ------------------------------------------------------------------ stored results
def test_result_roundtrip_delete_and_cleanup(models, tmp_path):
    models()
    keep = detector.detect_image(image_bytes(), "keep.png", tmp_path)
    old = detector.detect_image(image_bytes(), "old.png", tmp_path)
    assert detector.load_result(tmp_path, keep["id"])["original_name"] == "keep.png"

    past = time.time() - 7200
    for suffix in (".jpg", ".json"):
        os.utime(tmp_path / f"{old['id']}{suffix}", (past, past))
    (tmp_path / "stale.part").write_bytes(b"x")
    os.utime(tmp_path / "stale.part", (past, past))

    assert detector.cleanup_results(tmp_path, 3600) == 3
    assert detector.load_result(tmp_path, old["id"]) is None
    assert detector.load_result(tmp_path, keep["id"]) is not None

    detector.delete_result(tmp_path, keep["id"])
    assert detector.load_result(tmp_path, keep["id"]) is None


@pytest.mark.parametrize("bad", ["", "../etc/passwd", "abc", "Z" * 32, None, "a" * 33])
def test_invalid_result_ids_are_rejected(bad, tmp_path):
    assert not detector.valid_id(bad)
    assert detector.load_result(tmp_path, bad) is None
