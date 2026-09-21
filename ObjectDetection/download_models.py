"""Download and check the models VisionScan needs.

Run this once after installing the requirements, so the first scan is instant and
the app keeps working offline afterwards:

    python download_models.py            # fetch whatever is missing
    python download_models.py --force    # re-download the face model

Exit code 0 = everything is ready, 1 = something is missing (details are printed).
"""
from __future__ import annotations

import argparse
import sys

import detector
from detector import ModelUnavailable


def check_face_model(force):
    print(f"Face model ({detector.YUNET_FILE})")
    try:
        path = detector.ensure_face_model(force=force)
        detector.FaceDetector(path)  # make sure it really loads
    except ModelUnavailable as exc:
        print(f"  FAILED: {exc}")
        print("  Manual fix: download the file from")
        print(f"    {detector.YUNET_DEFAULT_URL}")
        print(f"  and save it as: {detector.YUNET_PATH}")
        return False
    except Exception as exc:  # noqa: BLE001
        print(f"  FAILED: the file exists but cannot be loaded ({exc}).")
        print("  Delete it and run this script again with --force.")
        return False
    print(f"  OK: {path}")
    return True


def check_object_model():
    print(f"Object model ({detector.OBJECT_MODEL})")
    try:
        detector.object_model.get()
    except ModelUnavailable as exc:
        print(f"  FAILED: {exc}")
        print(f"  Put the weights file in: {detector.MODEL_DIR}")
        return False
    print(f"  OK: {detector._resolve_weights(detector.OBJECT_MODEL)}")
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--force", action="store_true", help="re-download the face model")
    parser.add_argument("--skip-object", action="store_true",
                        help="only check the face model (skips loading PyTorch)")
    args = parser.parse_args(argv)

    ok = check_face_model(args.force)
    if not args.skip_object:
        ok = check_object_model() and ok
    print("\nAll models are ready." if ok else "\nSome models are missing - see above.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
