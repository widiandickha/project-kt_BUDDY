"""MediaPipe face landmarks -> EAR (spec 4.2), MAR, brow ratio, face size.

EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||) with Face Mesh indices
left eye [362, 385, 387, 263, 373, 380], right eye [33, 160, 158, 133, 153, 144].
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.core.config import resolve_path

LEFT_EYE: tuple[int, ...] = (362, 385, 387, 263, 373, 380)
RIGHT_EYE: tuple[int, ...] = (33, 160, 158, 133, 153, 144)
MOUTH_INNER_V: tuple[int, int] = (13, 14)
MOUTH_CORNERS: tuple[int, int] = (78, 308)
FACE_SIDES: tuple[int, int] = (234, 454)
EYE_OUTER: tuple[int, int] = (33, 263)
INNER_BROWS: tuple[int, int] = (107, 336)
BROW_EYELID_LEFT: tuple[int, int] = (295, 386)
BROW_EYELID_RIGHT: tuple[int, int] = (65, 159)
NOSE_TIP: int = 1

LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def compute_ear(pts: np.ndarray, idx: tuple[int, ...]) -> float:
    """Eye aspect ratio of one eye from 6 landmarks p1..p6."""
    p1, p2, p3, p4, p5, p6 = (pts[i] for i in idx)
    horiz = _dist(p1, p4)
    if horiz < 1e-6:
        return 0.0
    return (_dist(p2, p6) + _dist(p3, p5)) / (2.0 * horiz)


def compute_mar(pts: np.ndarray) -> float:
    """Inner-lip mouth aspect ratio (vertical opening / mouth width)."""
    horiz = _dist(pts[MOUTH_CORNERS[0]], pts[MOUTH_CORNERS[1]])
    if horiz < 1e-6:
        return 0.0
    return _dist(pts[MOUTH_INNER_V[0]], pts[MOUTH_INNER_V[1]]) / horiz


def compute_brow_ratio(pts: np.ndarray) -> float:
    """Brow tension proxy: inner-brow gap and brow-eyelid gap over eye span (smaller = more furrowed)."""
    span = _dist(pts[EYE_OUTER[0]], pts[EYE_OUTER[1]])
    if span < 1e-6:
        return 0.0
    inner = _dist(pts[INNER_BROWS[0]], pts[INNER_BROWS[1]]) / span
    lift = (_dist(pts[BROW_EYELID_LEFT[0]], pts[BROW_EYELID_LEFT[1]])
            + _dist(pts[BROW_EYELID_RIGHT[0]], pts[BROW_EYELID_RIGHT[1]])) / (2.0 * span)
    return 0.5 * inner + 0.5 * lift


@dataclass
class LandmarkResult:
    """Landmark-derived measurements of one frame."""

    face_found: bool
    ear_left: float | None = None
    ear_right: float | None = None
    ear_mean: float | None = None
    mar: float | None = None
    brow_ratio: float | None = None
    face_size_ratio: float | None = None
    face_box: tuple[int, int, int, int] | None = None
    nose_xy: tuple[float, float] | None = None


def measure(pts: np.ndarray, width: int, height: int) -> LandmarkResult:
    """All measurements from pixel landmarks (N, 2)."""
    left, right = compute_ear(pts, LEFT_EYE), compute_ear(pts, RIGHT_EYE)
    x1, y1 = pts[:, 0].min(), pts[:, 1].min()
    x2, y2 = pts[:, 0].max(), pts[:, 1].max()
    nose = pts[NOSE_TIP]
    return LandmarkResult(
        face_found=True,
        ear_left=left,
        ear_right=right,
        ear_mean=(left + right) / 2.0,
        mar=compute_mar(pts),
        brow_ratio=compute_brow_ratio(pts),
        face_size_ratio=float((x2 - x1) / max(width, 1)),
        face_box=(int(max(0, x1)), int(max(0, y1)), int(min(width - 1, x2)), int(min(height - 1, y2))),
        nose_xy=(float(nose[0] / max(width, 1)), float(nose[1] / max(height, 1))),
    )


def ensure_landmarker(path: Path) -> Path:
    """Download the official MediaPipe face landmarker model if it is missing."""
    if not path.exists():
        import urllib.request

        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(LANDMARKER_URL, path)  # noqa: S310 (fixed official URL)
    return path


class EarCalculator:
    """MediaPipe FaceLandmarker in video mode (tracking between frames)."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        model = ensure_landmarker(resolve_path(cfg["landmark"]["model"]))
        options = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._last_ms = -1

    def process(self, frame_bgr: np.ndarray, ts: float | None = None) -> LandmarkResult:
        """Measure one BGR frame; ts in seconds (monotonic) for the video-mode tracker."""
        import mediapipe as mp

        ms = int((ts if ts is not None else time.monotonic()) * 1000)
        ms = max(ms, self._last_ms + 1)  # MediaPipe requires strictly increasing timestamps
        self._last_ms = ms
        h, w = frame_bgr.shape[:2]
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        res = self._landmarker.detect_for_video(image, ms)
        if not res.face_landmarks:
            return LandmarkResult(face_found=False)
        pts = np.array([[lm.x * w, lm.y * h] for lm in res.face_landmarks[0]], dtype=np.float32)
        return measure(pts, w, h)

    def close(self) -> None:
        """Release native resources."""
        self._landmarker.close()


class ImageLandmarker:
    """FaceLandmarker in image mode (independent pictures: dataset tools, evaluation)."""

    def __init__(self, model_path: Path) -> None:
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        options = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)

    def points(self, bgr: np.ndarray) -> np.ndarray | None:
        """Pixel landmarks (N, 2) of the first face, or None."""
        import mediapipe as mp

        h, w = bgr.shape[:2]
        res = self._landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
        if not res.face_landmarks:
            return None
        return np.array([[lm.x * w, lm.y * h] for lm in res.face_landmarks[0]], dtype=np.float32)

    def close(self) -> None:
        """Release native resources."""
        self._landmarker.close()
