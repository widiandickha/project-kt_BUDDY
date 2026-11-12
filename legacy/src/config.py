from pathlib import Path
from typing import Any, Dict
import yaml

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = ROOT_DIR / "config.yaml"

DEFAULT_CONFIG: Dict[str, Any] = {
    "app": {
        "name": "Kucing Teman Kerja",
        "theme": "light",
        "active_hours_enabled": False,
        "active_hour_start": "08:00",
        "active_hour_end": "22:00",
        "sensitivity": 1.0,
    },
    "camera": {
        "device_index": 0,
        "target_fps": 15,
        "frame_width": 640,
        "frame_height": 480,
    },
    "models": {
        "face_yolo_path": "models/face_expression_yolov8n.pt",
        "pose_yolo_path": "models/yolov8n-pose.pt",
        "use_two_class_fallback": False,
        "confidence_threshold": 0.45,
    },
    "calibration": {
        "duration_seconds": 10,
        "default_ear": 0.28,
        "default_mar": 0.15,
        "default_brow_ratio": 0.24,
        "default_posture_ratio": 0.55,
    },
    "analysis": {
        "sliding_window_seconds": 60,
        "burnout_window_minutes": 15,
        "db_log_interval_seconds": 5,
        "ear_closed_factor": 0.75,
        "mar_yawn_threshold": 0.55,
        "brow_furrow_factor": 0.88,
        "posture_slouch_factor": 0.82,
        "hand_to_head_ratio_threshold": 0.48,
        "weights": {
            "eye_closed": 0.30,
            "yawn": 0.20,
            "brow_furrow": 0.15,
            "posture_slouch": 0.15,
            "hand_to_head": 0.10,
            "head_restlessness": 0.10,
        },
        "thresholds": {
            "waspada": 40.0,
            "tinggi": 70.0,
            "burnout_avg": 60.0,
        },
    },
    "buddy": {
        "cooldown_seconds": 180,
        "snooze_minutes": 15,
        "overlay_duration_seconds": 8,
    },
}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(config_path: Path = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    if not config_path.exists():
        save_config(DEFAULT_CONFIG, config_path)
        return dict(DEFAULT_CONFIG)
    with open(config_path, "r", encoding="utf-8") as fh:
        loaded = yaml.safe_load(fh) or {}
    return _deep_merge(DEFAULT_CONFIG, loaded)


def save_config(config_data: Dict[str, Any], config_path: Path = DEFAULT_CONFIG_PATH) -> None:
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(config_data, fh, sort_keys=False, allow_unicode=True)
