"""Fakes and utilities shared by the tests (no PyTorch or model files needed)."""
from __future__ import annotations

import http.server
import re
import socketserver
import threading
import types

import cv2
import numpy as np


class FakeModel:
    """Stands in for detector.LazyModel."""

    def __init__(self, obj=None, error=None):
        self._obj, self._error = obj, error

    def get(self):
        if self._error is not None:
            raise self._error
        return self._obj

    def status(self):
        return {"state": "error" if self._error else "ready",
                "error": str(self._error) if self._error else None}


class FakeYOLO:
    """Mimics ultralytics.YOLO.predict(); boxes are (x1, y1, x2, y2, conf, cls)."""

    names = {0: "person", 2: "car"}

    def __init__(self, boxes=(), error=None):
        self.boxes, self.error, self.calls = list(boxes), error, []

    def predict(self, source=None, conf=0.25, imgsz=640, verbose=False):
        self.calls.append({"conf": conf, "shape": source.shape})
        if self.error:
            raise self.error
        kept = np.array([b for b in self.boxes if b[4] >= conf], dtype=float).reshape(-1, 6)
        boxes = types.SimpleNamespace(xyxy=kept[:, :4], conf=kept[:, 4], cls=kept[:, 5])
        return [types.SimpleNamespace(boxes=boxes, names=self.names)]


class FakeFaces:
    """Mimics detector.FaceDetector."""

    def __init__(self, rows=()):
        self.rows = list(rows)

    def detect(self, image, score_threshold):
        kept = [r for r in self.rows if r[-1] >= score_threshold]
        return np.array(kept, dtype=np.float32).reshape(-1, 15)


def face_row(x, y, w, h, score=0.9):
    """A YuNet output row: x, y, w, h, 5 landmark points, score."""
    return [x, y, w, h] + [0] * 10 + [score]


def image_bytes(width=600, height=400, value=128, ext=".png"):
    ok, encoded = cv2.imencode(ext, np.full((height, width, 3), value, np.uint8))
    assert ok
    return encoded.tobytes()


def read_result(outdir, meta):
    return cv2.imread(str(outdir / f"{meta['id']}.jpg"))


def colour_pixels(image, rgb, tolerance=45):
    """Count pixels close to an RGB colour (results are lossy JPEG, so allow slack)."""
    bgr = np.array(rgb[::-1], dtype=int)
    return int(np.all(np.abs(image.astype(int) - bgr) < tolerance, axis=2).sum())


def csrf_token(client):
    page = client.get("/").get_data(as_text=True)
    return re.search(r'name="_csrf" value="([0-9a-f]+)"', page).group(1)


def upload(client, data, name="photo.png", **fields):
    import io

    payload = {"_csrf": csrf_token(client), "image": (io.BytesIO(data), name), **fields}
    return client.post("/", data=payload, content_type="multipart/form-data")


class LocalServer:
    """A tiny HTTP server that serves one canned response, for download tests."""

    def __init__(self, body: bytes, declared_length: int | None = None):
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802
                self.send_response(200)
                length = len(body) if declared_length is None else declared_length
                self.send_header("Content-Length", str(length))
                self.end_headers()
                self.wfile.write(body)
                self.wfile.flush()
                self.close_connection = True

            def log_message(self, *args):
                pass

        self._server = socketserver.TCPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self._server.server_address[1]}/model.onnx"
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    def close(self):
        self._server.shutdown()
        self._server.server_close()
