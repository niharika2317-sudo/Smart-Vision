"""Tunable project settings. Camera coordinates are a demo fallback."""
from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent
EVIDENCE_DIR = BASE_DIR / "evidence"
CAMERA_ID = "camera-01"
CAMERA_NAME = "VantaWatch Camera 01"
# CHANGE THIS to the surveyed fixed location of the installed camera.
CAMERA_LAT = 28.6139
CAMERA_LON = 77.2090
PORT = int(os.environ.get("SV_PORT", "5050"))
SERVER_URL = f"http://127.0.0.1:{PORT}"
FRAME_WIDTH = 640
DEVICE = "auto"  # "auto" selects CUDA when available, otherwise CPU.
PERSON_IMAGE_SIZE = 640
WEAPON_IMAGE_SIZE = 640
POSE_IMAGE_SIZE = 640
PERSON_TRACKER = "botsort.yaml"
PERSON_BEHAVIOR_EVERY_N_FRAMES = 3
WEAPON_EVERY_N_FRAMES = 1
PERSON_CONFIDENCE = 0.30
POSE_DETECT_CONFIDENCE = 0.40
WEAPON_CONFIDENCE = 0.25 # Lowest swept point with no police alerts on the reduced Normal sample.
WEAPON_CLASS_CONFIDENCE = {"pistol": 0.35, "knife": 0.25}
WEAPON_MAX_BOX_AREA_RATIO = 0.10
WEAPON_REQUIRE_PERSON_CONTACT_FOR_POLICE = True
WEAPON_CONFIRM_FRAMES = 1
WEAPON_TRACK_IOU = 0.30
HOLD_MARGIN = 0.15
POSE_MIN_IOU = 0.30
KEYPOINT_CONFIDENCE = 0.25
FALL_ANGLE_DEGREES = 60
RAISED_HAND_WIDTH_RATIO = 0.30
MOTION_WINDOW_SECONDS = 1.0
MOTION_JITTER_RATIO = 0.12  # Ignore small person-box movement from detector jitter.
TRACK_FORGET_SECONDS = 5.0
WALKING_SPEED = 0.8
RUNNING_SPEED = 1.4
AGGRESSIVE_SPEED = 3.0
SCORE_LOW = 30
SCORE_HIGH = 65
POLICE_CONFIRM_FRAMES = 3
POLICE_COOLDOWN_SECONDS = 30
MEDICAL_COOLDOWN_SECONDS = 60
MEDICAL_FALL_SECONDS = 2.0
MEDICAL_BLOOD_SECONDS = 1.5
MEDICAL_BLOOD_MIN_RATIO = 0.015
MEDICAL_BLOOD_MAX_RATIO = 0.30
LYING_ASPECT = 1.25
MEDICAL_MIN_TRACK_AGE_SECONDS = 1.0
MEDICAL_MIN_BOX_HEIGHT_RATIO = 0.15
MEDICAL_EDGE_MARGIN_RATIO = 0.01
MEDICAL_MAX_OCCLUSION_RATIO = 0.45
MEDICAL_STATIONARY_SPEED = 0.35
MEDICAL_POSE_KEYPOINT_CONFIDENCE = 0.50
MEDICAL_POSE_FALL_MIN_KEYPOINTS = 6
BLOOD_HUE_LOW = 8
BLOOD_HUE_HIGH = 172
BLOOD_SAT_MIN = 110
BLOOD_VALUE_MIN = 40
BLOOD_VALUE_MAX = 190
GPS_LIVE_SECONDS = 60
EVIDENCE_PRE_SECONDS = 3.0
EVIDENCE_POST_SECONDS = 5.0
EVIDENCE_MAX_SECONDS = 60.0
EVIDENCE_MAX_GB = 5.0
EVIDENCE_CAP_CHECK_SECONDS = 30.0
LOCAL_SIREN = True
EVIDENCE_FPS_FALLBACK = 20.0


_RUNTIME_DEVICE = None


def resolve_device():
    """Prefer CUDA, then Apple MPS, then CPU when DEVICE is auto."""
    global _RUNTIME_DEVICE
    if DEVICE != "auto":
        return DEVICE
    if _RUNTIME_DEVICE is not None:
        return _RUNTIME_DEVICE
    import torch
    if torch.cuda.is_available():
        _RUNTIME_DEVICE = 0
    elif getattr(getattr(torch, "backends", None), "mps", None) is not None and torch.backends.mps.is_available():
        _RUNTIME_DEVICE = "mps"
    else:
        _RUNTIME_DEVICE = "cpu"
    return _RUNTIME_DEVICE


def run_model_with_device_fallback(operation):
    """Run an Ultralytics op on the selected device; retry on CPU after MPS failure."""
    global _RUNTIME_DEVICE
    device = resolve_device()
    try:
        return operation(device)
    except Exception as exc:
        if device != "mps":
            raise
        _RUNTIME_DEVICE = "cpu"
        print(f"[DEVICE] MPS model operation failed ({exc}); retrying on CPU.", flush=True)
        return operation("cpu")


def model_options(device, image_size):
    """Build consistent Ultralytics options; use half precision only with CUDA."""
    options = {"imgsz": image_size, "device": device}
    if isinstance(device, int) or str(device).startswith("cuda"):
        options["half"] = True
    return options
