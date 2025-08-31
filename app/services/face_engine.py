import os
import sys
import threading
import math
import cv2
import numpy as np
import requests

# ---------- Model files (OpenCV Zoo) ----------
YUNET_FILE  = "face_detection_yunet_2023mar.onnx"
SFACE_FILE  = "face_recognition_sface_2021dec.onnx"

# A couple of public mirrors (either works). If blocked, place files manually.
YUNET_URLS = [
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "https://raw.githubusercontent.com/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
]
SFACE_URLS = [
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
    "https://raw.githubusercontent.com/opencv/opencv_zoo/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
]

# ---------- Paths that also work inside PyInstaller bundles ----------
def _resource_base():
    # When packaged, _MEIPASS points to the temp bundle dir
    return getattr(sys, "_MEIPASS", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def _model_path(name: str) -> str:
    # Keep models in app/models/
    base = _resource_base()
    path = os.path.join(base, "app", "models", name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path

YUNET_PATH = _model_path(YUNET_FILE)
SFACE_PATH = _model_path(SFACE_FILE)

# ---------- Globals ----------
_lock = threading.Lock()
_detector = None
_recognizer = None

def _download(urls, dest) -> bool:
    for url in urls:
        try:
            r = requests.get(url, timeout=60, stream=True)
            r.raise_for_status()
            tmp = dest + ".part"
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(1 << 15):
                    if chunk:
                        f.write(chunk)
            os.replace(tmp, dest)
            return True
        except Exception:
            continue
    return False

def _ensure_models():
    ok = True
    if not os.path.exists(YUNET_PATH):
        ok = _download(YUNET_URLS, YUNET_PATH) and ok
    if not os.path.exists(SFACE_PATH):
        ok = _download(SFACE_URLS, SFACE_PATH) and ok
    if not ok:
        raise RuntimeError(
            "Could not obtain YuNet/SFace models.\n"
            f"Place them manually:\n  {YUNET_PATH}\n  {SFACE_PATH}\n"
        )

def _ensure_engine():
    global _detector, _recognizer
    with _lock:
        if _detector is None or _recognizer is None:
            _ensure_models()
            # YuNet detector; input size is set dynamically per frame
            _detector = cv2.FaceDetectorYN_create(
                YUNET_PATH, "",
                (320, 320),
                score_threshold=0.6,
                nms_threshold=0.3,
                top_k=5000
            )
            # SFace recognizer (aligns + extracts features)
            _recognizer = cv2.FaceRecognizerSF_create(SFACE_PATH, "")
    return _detector, _recognizer

# ---------- Helpers ----------
def _largest_face(faces: np.ndarray):
    # faces shape: Nx15 [x, y, w, h, l0x, l0y, ..., l4x, l4y, score]
    if faces is None or len(faces) == 0:
        return None
    areas = faces[:, 2] * faces[:, 3]
    idx = int(np.argmax(areas))
    return faces[idx]

def _l2_normalize(v: np.ndarray, eps=1e-12):
    n = np.linalg.norm(v) + eps
    return (v / n).astype(np.float32)

# ---------- Public API (compatible with your app) ----------
def get_face_embedding_bgr(frame_bgr: np.ndarray):
    """
    Return (embedding, bbox, (face_w, face_h)).
    If no face/too small => (None, bbox_or_None, None)
    """
    detector, recognizer = _ensure_engine()
    h, w = frame_bgr.shape[:2]
    detector.setInputSize((w, h))
    success, faces = detector.detect(frame_bgr)
    if not success or faces is None or len(faces) == 0:
        return None, None, None

    face = _largest_face(faces)
    x, y, fw, fh = face[:4].astype(int)
    bbox = (max(0, x), max(0, y), min(w, x+fw), min(h, y+fh))

    # reject tiny faces (keeps your UI behavior consistent)
    if fw < 110 or fh < 110:
        return None, bbox, (fw, fh)

    # SFace handles alignment internally
    face_aligned = recognizer.alignCrop(frame_bgr, face)
    feat = recognizer.feature(face_aligned).flatten()
    emb = _l2_normalize(feat)
    return emb, bbox, (fw, fh)

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    # Both are L2-normalized → dot product is cosine similarity in [-1, 1]
    return float(np.dot(a, b))

def best_match(probe_emb: np.ndarray, gallery):
    if probe_emb is None or not gallery:
        return None
    sims = [(sid, name, matric, cosine_similarity(probe_emb, gemb)) for sid, name, matric, gemb in gallery]
    sims.sort(key=lambda x: x[3], reverse=True)
    return sims[0]
