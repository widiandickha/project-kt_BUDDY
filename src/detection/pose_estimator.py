"""yolov8n-pose (pretrained, no retraining) -> hand-to-forehead, shoulder slouch, head pitch (spec 4.1).

COCO keypoints: 0 nose, 1/2 eyes, 3/4 ears, 5/6 shoulders, 9/10 wrists.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.core.config import resolve_path
from src.core.device import select_device

NOSE, L_EYE, R_EYE, L_SH, R_SH, L_WR, R_WR = 0, 1, 2, 5, 6, 9, 10
MIN_SHOULDER_PX = 1.0


@dataclass
class PoseResult:
    """Pose measurements of the main person in one frame."""

    person_found: bool
    keypoints: np.ndarray | None = None  # (17, 3) x, y, conf in pixels
    hand_near_forehead: bool = False     # raw, before the consecutive-frame rule
    shoulder_slouch: float | None = None
    head_pitch_proxy: float | None = None
    nose_xy: tuple[float, float] | None = None


def analyze_keypoints(kpts: np.ndarray, width: int, height: int, min_conf: float, hand_face_ratio: float) -> PoseResult:
    """Pure geometry on one person's keypoints.

    * hand_near_forehead: any wrist with dist(wrist, eye midpoint) < hand_face_ratio * shoulder width
      and wrist_y <= nose_y (image y grows downward).
    * shoulder_slouch: (mean shoulder_y - nose_y) / shoulder width; smaller = head lower (D-018).
    * head_pitch_proxy: (nose_y - mean shoulder_y) / frame height.
    """
    ok = kpts[:, 2] >= min_conf
    res = PoseResult(person_found=True, keypoints=kpts)
    if ok[NOSE]:
        res.nose_xy = (float(kpts[NOSE, 0] / width), float(kpts[NOSE, 1] / height))
    if not (ok[NOSE] and ok[L_SH] and ok[R_SH]):
        return res
    sh_w = float(np.linalg.norm(kpts[L_SH, :2] - kpts[R_SH, :2]))
    if sh_w < MIN_SHOULDER_PX:
        return res
    sh_y = float((kpts[L_SH, 1] + kpts[R_SH, 1]) / 2.0)
    nose_y = float(kpts[NOSE, 1])
    res.shoulder_slouch = (sh_y - nose_y) / sh_w
    res.head_pitch_proxy = (nose_y - sh_y) / height

    if ok[L_EYE] and ok[R_EYE]:
        eye_mid = (kpts[L_EYE, :2] + kpts[R_EYE, :2]) / 2.0
    else:
        eye_mid = kpts[NOSE, :2]
    for wr in (L_WR, R_WR):
        if ok[wr] and kpts[wr, 1] <= nose_y and np.linalg.norm(kpts[wr, :2] - eye_mid) < hand_face_ratio * sh_w:
            res.hand_near_forehead = True
            break
    return res


class ConsecutiveFrames:
    """True only after `min_frames` consecutive True inputs (pose.hand_min_frames)."""

    def __init__(self, min_frames: int) -> None:
        self.min_frames = max(1, min_frames)
        self._run = 0

    def update(self, value: bool) -> bool:
        """Feed one frame; returns the debounced state."""
        self._run = self._run + 1 if value else 0
        return self._run >= self.min_frames


class PoseEstimator:
    """Runs yolov8n-pose and keeps the largest person."""

    def __init__(self, cfg: dict[str, Any], device: str | None = None) -> None:
        from ultralytics import YOLO
        from ultralytics.utils.downloads import attempt_download_asset

        pose = cfg["pose"]
        self.model = YOLO(str(attempt_download_asset(str(resolve_path(pose["model"])))))
        self.min_conf = float(pose["min_kp_conf"])
        self.hand_face_ratio = float(pose["hand_face_ratio"])
        self.imgsz = int(cfg["inference"]["imgsz"])
        self.device = select_device(device or cfg["inference"]["device"]).device
        self._hand = ConsecutiveFrames(int(pose["hand_min_frames"]))

    def process(self, frame_bgr: np.ndarray) -> PoseResult:
        """Analyze one frame (the hand rule is debounced across calls)."""
        h, w = frame_bgr.shape[:2]
        res = self.model.predict(frame_bgr, imgsz=self.imgsz, device=self.device, verbose=False)[0]
        if res.keypoints is None or res.boxes is None or len(res.boxes) == 0:
            self._hand.update(False)
            return PoseResult(person_found=False)
        boxes = res.boxes.xyxy.cpu().numpy()
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        i = int(np.argmax(areas))
        xy = res.keypoints.xy[i].cpu().numpy()
        conf = res.keypoints.conf[i].cpu().numpy() if res.keypoints.conf is not None else np.ones(len(xy))
        kpts = np.concatenate([xy, conf[:, None]], axis=1).astype(np.float32)
        out = analyze_keypoints(kpts, w, h, self.min_conf, self.hand_face_ratio)
        out.hand_near_forehead = self._hand.update(out.hand_near_forehead)
        return out
