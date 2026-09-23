import cv2
import numpy as np

from tests._frames import owner_frame, stranger_frame, two_face_frame
from tests.conftest import REAL_DATASET


def _encode(frame):
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    assert ok
    return buf.tobytes()


def test_engine_loads_real_pretrained_model(engine):
    assert engine.ready
    assert engine.training_images > 0
    assert not engine.detector.empty()


def test_owner_face_matches(engine):
    frame = owner_frame(REAL_DATASET[10], scale=1.4)
    result = engine.analyze(frame)
    assert result.status == "match"
    assert result.distance is not None and result.distance <= engine.s.MATCH_THRESHOLD


def test_stranger_face_does_not_match(engine):
    result = engine.analyze(stranger_frame())
    assert result.status == "no_match"
    assert result.distance is not None and result.distance > engine.s.MATCH_THRESHOLD


def test_blank_frame_reports_no_face(engine):
    blank = np.full((480, 640, 3), 120, dtype="uint8")
    result = engine.analyze(blank)
    assert result.status == "no_face"
    assert result.box is None


def test_decode_rejects_garbage_bytes(engine):
    assert engine.decode(b"not an image") is None
    assert engine.decode(b"") is None


def test_decode_accepts_valid_jpeg(engine):
    frame = engine.decode(_encode(stranger_frame()))
    assert frame is not None
    assert frame.ndim == 3


def test_enrollment_accepts_single_clear_face(engine):
    frame = owner_frame(REAL_DATASET[5], scale=1.4)
    result = engine.detect_for_enrollment(frame)
    assert result.status == "ok"
    assert result.crop is not None
    assert result.crop.shape == engine.s.FACE_SIZE


def test_enrollment_rejects_two_people_in_frame(engine):
    composite = two_face_frame(REAL_DATASET[20], REAL_DATASET[90])
    result = engine.detect_for_enrollment(composite)
    assert result.status == "multiple_faces"


def test_enrollment_reports_no_face_on_blank(engine):
    blank = np.full((480, 640, 3), 120, dtype="uint8")
    result = engine.detect_for_enrollment(blank)
    assert result.status == "no_face"


def test_predict_crop_matches_owner_and_rejects_stranger(engine):
    frame = owner_frame(REAL_DATASET[7], scale=1.4)
    enrolled = engine.detect_for_enrollment(frame)
    label, distance = engine.predict_crop(enrolled.crop)
    assert label is not None
    assert distance <= engine.s.MATCH_THRESHOLD
