# app/services/face_engine.py
import os
import sys
import threading
import cv2
import numpy as np
import shutil
import requests

# ---------- Model files (OpenCV Zoo) ----------
YUNET_FILE = "face_detection_yunet_2023mar.onnx"
SFACE_FILE = "face_recognition_sface_2021dec.onnx"

YUNET_URLS = [
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "https://raw.githubusercontent.com/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
]
SFACE_URLS = [
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
    "https://raw.githubusercontent.com/opencv/opencv_zoo/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
]

# ---------- Paths / Env helpers ----------
def _is_frozen():
    return getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS")

def _resource_base():
    """
    Read-only base for packaged resources:
      - Frozen: sys._MEIPASS (PyInstaller temp)
      - Dev:    app/ (this file is in app/services → go up one level)
    """
    if _is_frozen():
        return sys._MEIPASS
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../app

def _user_cache_dir():
    """
    User-writable cache for downloads/copies.
    """
    appname = "FaceRecognitionAttendance"
    if sys.platform.startswith("win"):
        root = os.getenv("LOCALAPPDATA") or os.path.expanduser(r"~\AppData\Local")
        path = os.path.join(root, appname, "cache")
    elif sys.platform == "darwin":
        path = os.path.expanduser(f"~/Library/Caches/{appname}")
    else:
        path = os.path.expanduser(f"~/.cache/{appname}")
    os.makedirs(path, exist_ok=True)
    return path

# Paths for packaged models (read-only) and local dev models
# NOTE: in your repo the models live at app/app/models
_PACKAGED_MODELS_DIR = os.path.join(_resource_base(), "app", "app", "models")
_DEV_MODELS_DIR      = _PACKAGED_MODELS_DIR if _is_frozen() else _PACKAGED_MODELS_DIR
_CACHE_MODELS_DIR    = os.path.join(_user_cache_dir(), "models")

# Globals that hold the actual model paths used at runtime
_YUNET_PATH = None
_SFACE_PATH = None

# ---------- Concurrency ----------
_lock = threading.Lock()
_detector = None
_recognizer = None

# ---------- IO helpers ----------
def _exists_file(path: str) -> bool:
    try:
        return os.path.isfile(path)
    except Exception:
        return False

def _safe_copy(src: str, dst: str):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = dst + ".part"
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)

def _download(urls, dest) -> bool:
    try:
        os.makedirs(os.path.dirname(dest), exist_ok=True)
    except Exception:
        # If we can't create the parent, it's likely Program Files → bail
        return False
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

# ---------- Model resolution ----------
def _resolve_model_paths():
    """
    Decide where to load models from (no writes into Program Files):
      1) Use PACKAGED models if present (read-only).
      2) Else, if running frozen → download to user cache.
      3) Else (dev) → try repo path; if missing, download to dev path (writable).
    Sets _YUNET_PATH and _SFACE_PATH accordingly.
    """
    global _YUNET_PATH, _SFACE_PATH

    packaged_yunet = os.path.join(_PACKAGED_MODELS_DIR, YUNET_FILE)
    packaged_sface = os.path.join(_PACKAGED_MODELS_DIR, SFACE_FILE)

    # Case 1: packaged (bundled via PyInstaller datas)
    if _exists_file(packaged_yunet) and _exists_file(packaged_sface):
        _YUNET_PATH, _SFACE_PATH = packaged_yunet, packaged_sface
        return

    # Case 2: frozen but not packaged → download to user cache (writable)
    if _is_frozen():
        cache_yunet = os.path.join(_CACHE_MODELS_DIR, YUNET_FILE)
        cache_sface = os.path.join(_CACHE_MODELS_DIR, SFACE_FILE)
        ok1 = _exists_file(cache_yunet) or _download(YUNET_URLS, cache_yunet)
        ok2 = _exists_file(cache_sface) or _download(SFACE_URLS, cache_sface)
        if not (ok1 and ok2):
            raise RuntimeError(
                "Could not obtain YuNet/SFace models.\n"
                f"Tried user cache:\n  {cache_yunet}\n  {cache_sface}\n"
                "Check your internet connection or pre-bundle models in the installer."
            )
        _YUNET_PATH, _SFACE_PATH = cache_yunet, cache_sface
        return

    # Case 3: dev (not frozen)
    dev_yunet = os.path.join(_DEV_MODELS_DIR, YUNET_FILE)
    dev_sface = os.path.join(_DEV_MODELS_DIR, SFACE_FILE)

    # If present in repo, use them
    if _exists_file(dev_yunet) and _exists_file(dev_sface):
        _YUNET_PATH, _SFACE_PATH = dev_yunet, dev_sface
        return

    # Else download to repo path (dev is usually writable)
    ok1 = _exists_file(dev_yunet) or _download(YUNET_URLS, dev_yunet)
    ok2 = _exists_file(dev_sface) or _download(SFACE_URLS, dev_sface)
    if not (ok1 and ok2):
        raise RuntimeError(
            "Could not obtain YuNet/SFace models in dev.\n"
            f"Tried repo path:\n  {dev_yunet}\n  {dev_sface}\n"
            "Check your internet connection or download the files manually."
        )
    _YUNET_PATH, _SFACE_PATH = dev_yunet, dev_sface

def _ensure_engine():
    global _detector, _recognizer
    with _lock:
        if _detector is None or _recognizer is None:
            if _YUNET_PATH is None or _SFACE_PATH is None:
                _resolve_model_paths()
            # YuNet detector; input size is set dynamically per frame
            _detector = cv2.FaceDetectorYN_create(
                _YUNET_PATH, "",
                (320, 320),
                score_threshold=0.6,
                nms_threshold=0.3,
                top_k=5000
            )
            # SFace recognizer (aligns + extracts features)
            _recognizer = cv2.FaceRecognizerSF_create(_SFACE_PATH, "")
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

# ---------- Public API ----------
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
    bbox = (max(0, x), max(0, y), min(w, x + fw), min(h, y + fh))

    # reject tiny faces (keeps your UI behavior consistent)
    if fw < 110 or fh < 110:
        return None, bbox, (fw, fh)

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
