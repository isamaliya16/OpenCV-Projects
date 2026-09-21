# Models

VisionScan uses two models. Both are looked up in this folder.

| File | Used for | How you get it |
|------|----------|----------------|
| `yolo26m.pt` | Objects (80 COCO classes) | Included in the project. If it is missing, Ultralytics downloads it. |
| `face_detection_yunet_2023mar.onnx` | Faces (OpenCV YuNet) | Downloaded on first start-up, or run `python download_models.py`. |

Set `VISION_MODEL=yolo26n.pt` (smaller, faster) or `yolo26l.pt` (more accurate) to use another YOLO26 size.

## Getting the face model manually

If the automatic download fails (firewall, offline PC), download this file in a browser and save it in this folder
under the exact name above:

https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx

The file should be about 230 KB. A much smaller file means the download was blocked or is a Git-LFS pointer.
