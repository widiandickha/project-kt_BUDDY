"""Combine landmark, (optional) YOLO face and pose results into one FrameFeatures (spec 4.2).

When no trained face detector is available (models/best.pt missing, D-016), yawn and brow
furrow fall back to landmark rules: MAR > landmark.mar_yawn, and brow ratio below the
calibrated baseline by landmark.brow_drop. Classes listed in det.landmark_classes keep using the
landmark rule even when YOLO runs, because the detector was not trained on them (D-031).
"""

from __future__ import annotations

from typing import Any

from src.analysis.calibration import Calibration
from src.core.events import FrameFeatures
from src.detection.ear_calculator import LandmarkResult
from src.detection.pose_estimator import PoseResult


class FeatureExtractor:
    """Stateless apart from the current calibration."""

    def __init__(self, cfg: dict[str, Any], calibration: Calibration) -> None:
        self.mar_yawn = float(cfg["landmark"]["mar_yawn"])
        self.brow_drop = float(cfg["landmark"]["brow_drop"])
        self.landmark_classes = frozenset(cfg["det"].get("landmark_classes", []))
        self.calibration = calibration

    def set_calibration(self, calibration: Calibration) -> None:
        """Swap in a new user calibration."""
        self.calibration = calibration

    def extract(
        self,
        ts: float,
        landmark: LandmarkResult | None,
        faces: Any | None = None,
        pose: PoseResult | None = None,
    ) -> FrameFeatures:
        """Build FrameFeatures; `faces` is a FaceDetections or None when YOLO is unavailable."""
        lm = landmark if landmark is not None and landmark.face_found else None
        person = pose if pose is not None and pose.person_found else None
        f = FrameFeatures(ts=ts, user_present=bool(lm or person))

        if lm is not None:
            f.ear_left, f.ear_right, f.ear_mean = lm.ear_left, lm.ear_right, lm.ear_mean
            f.mar, f.brow_ratio, f.face_size_ratio = lm.mar, lm.brow_ratio, lm.face_size_ratio
            f.nose_xy = lm.nose_xy

        if faces is not None:
            f.det_available = True
            f.eye_closed_yolo = faces.conf("mata_tertutup")
            f.yawn = self._yawn_rule(lm) if "menguap" in self.landmark_classes else faces.conf("menguap")
            f.brow_furrow = (self._furrow_rule(lm) if "kerutan_dahi" in self.landmark_classes
                             else faces.conf("kerutan_dahi"))
        elif lm is not None:
            f.yawn = self._yawn_rule(lm)
            f.brow_furrow = self._furrow_rule(lm)

        if person is not None:
            f.hand_to_forehead = person.hand_near_forehead
            f.shoulder_slouch = person.shoulder_slouch
            f.head_pitch_proxy = person.head_pitch_proxy
            f.pose_keypoints = person.keypoints
            if f.nose_xy is None:
                f.nose_xy = person.nose_xy
        return f

    def _yawn_rule(self, lm: LandmarkResult | None) -> float:
        """1.0 when the mouth aspect ratio is above landmark.mar_yawn."""
        return 1.0 if lm is not None and (lm.mar or 0.0) > self.mar_yawn else 0.0

    def _furrow_rule(self, lm: LandmarkResult | None) -> float:
        """1.0 when the brows sit landmark.brow_drop below the calibrated baseline (needs calibration)."""
        base = self.calibration.brow_ratio
        if lm is None or not base or lm.brow_ratio is None:
            return 0.0
        return 1.0 if lm.brow_ratio < base * (1.0 - self.brow_drop) else 0.0
