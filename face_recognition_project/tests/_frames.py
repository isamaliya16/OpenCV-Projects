import cv2, numpy as np, glob
from skimage import data

def owner_frame(path, scale=1.5, dx=0, dy=0, size=(640,480)):
    face = cv2.imread(path, 0)
    face = cv2.resize(face, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    bg = np.full((size[1], size[0]), int(face.mean()), np.uint8)
    # soft noise so the background is not a perfectly flat colour
    bg = cv2.add(bg, np.random.default_rng(1).integers(0, 6, bg.shape, dtype=np.uint8))
    fh, fw = face.shape
    x0 = (size[0]-fw)//2 + dx; y0 = (size[1]-fh)//2 + dy
    canvas = bg.copy(); canvas[y0:y0+fh, x0:x0+fw] = face
    # feather the edge of the pasted crop so Haar does not latch onto a hard rectangle
    mask = np.zeros_like(bg, np.float32); mask[y0:y0+fh, x0:x0+fw] = 1
    mask = cv2.GaussianBlur(mask, (0,0), 6)[..., None]
    out = (canvas[..., None]*mask + bg[..., None]*(1-mask)).astype(np.uint8)
    return cv2.cvtColor(out[..., 0], cv2.COLOR_GRAY2BGR)

def stranger_frame(size=(640,480)):
    img = cv2.cvtColor(data.astronaut(), cv2.COLOR_RGB2BGR)  # bundled sample photo, not a project image
    img = cv2.resize(img, (480,480))
    bg = np.full((size[1], size[0], 3), 128, np.uint8)
    x0 = (size[0]-480)//2
    bg[:, x0:x0+480] = img
    return bg

def two_face_frame(path1, path2, scale=1.5, size=(800,480)):
    """Two independent, undistorted face crops placed side by side — used to
    verify the enrollment path rejects a frame with a second person in it.
    (A composite built by literally splicing two frames down the middle tends
    to produce a weak, low-confidence second detection near the seam, which
    under-represents a real bystander — this keeps both faces intact.)"""
    face1 = cv2.resize(cv2.imread(path1, 0), None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    face2 = cv2.resize(cv2.imread(path2, 0), None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    bg = np.full((size[1], size[0]), int(face1.mean()), np.uint8)
    bg = cv2.add(bg, np.random.default_rng(1).integers(0, 6, bg.shape, dtype=np.uint8))
    canvas = bg.copy()
    h1, w1 = face1.shape
    h2, w2 = face2.shape
    y1, x1 = (size[1]-h1)//2, size[0]//4 - w1//2
    y2, x2 = (size[1]-h2)//2, 3*size[0]//4 - w2//2
    canvas[y1:y1+h1, x1:x1+w1] = face1
    canvas[y2:y2+h2, x2:x2+w2] = face2
    return cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
