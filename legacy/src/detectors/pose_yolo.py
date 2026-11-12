from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
import numpy as np

from src.config import ROOT_DIR

try:
    from ultralytics import YOLO
    HAS_ULTRALYTICS = True
except Exception:
    YOLO = None
    HAS_ULTRALYTICS = False


@dataclass
class PoseFeatures:
    pose_detected: bool
    posture_ratio: float
    hand_to_head: bool
    shoulder_tilt: float
    head_point: Optional[Tuple[float, float]] = None
    wrist_point: Optional[Tuple[float, float]] = None


class PoseAnalyzer:
    """Menganalisis postur bungkuk dan gestur tangan ke kepala memakai yolov8n-pose."""

    def __init__(
        self,
        model_rel_path: str = "models/yolov8n-pose.pt",
        hand_to_head_threshold: float = 0.48,
    ) -> None:
        self.hand_to_head_threshold = hand_to_head_threshold
        self.model_path = self._resolve_path(model_rel_path)
        self._model = None
        self._load_or_init()

    def _resolve_path(self, raw_path: str) -> Path:
        p = Path(raw_path)
        if not p.is_absolute():
            p = ROOT_DIR / p
        return p

    def _load_or_init(self) -> None:
        if not HAS_ULTRALYTICS:
            return
        try:
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            if self.model_path.exists():
                self._model = YOLO(str(self.model_path))
            else:
                # Ultralytics downloads yolov8n-pose.pt automatically on first use.
                self._model = YOLO("yolov8n-pose.pt")
        except Exception:
            self._model = None

    def analyze(self, frame_bgr: np.ndarray) -> PoseFeatures:
        if self._model is None:
            return PoseFeatures(
                pose_detected=False,
                posture_ratio=0.55,
                hand_to_head=False,
                shoulder_tilt=0.0,
            )

        h, w = frame_bgr.shape[:2]
        preds = self._model.predict(source=frame_bgr, conf=0.35, verbose=False)
        if not preds or preds[0].keypoints is None or len(preds[0].keypoints.xy) == 0:
            return PoseFeatures(
                pose_detected=False,
                posture_ratio=0.55,
                hand_to_head=False,
                shoulder_tilt=0.0,
            )

        kpts_xy = preds[0].keypoints.xy[0].cpu().numpy()
        kpts_conf = (
            preds[0].keypoints.conf[0].cpu().numpy()
            if preds[0].keypoints.conf is not None
            else np.ones(len(kpts_xy), dtype=np.float32)
        )
        if len(kpts_xy) < 11:
            return PoseFeatures(
                pose_detected=False,
                posture_ratio=0.55,
                hand_to_head=False,
                shoulder_tilt=0.0,
            )

        nose = kpts_xy[0]
        l_sh, r_sh = kpts_xy[5], kpts_xy[6]
        l_wr, r_wr = kpts_xy[9], kpts_xy[10]

        if kpts_conf[5] < 0.25 or kpts_conf[6] < 0.25 or kpts_conf[0] < 0.25:
            return PoseFeatures(
                pose_detected=False,
                posture_ratio=0.55,
                hand_to_head=False,
                shoulder_tilt=0.0,
            )

        shoulder_width = float(np.linalg.norm(l_sh - r_sh))
        if shoulder_width < 10.0:
            return PoseFeatures(
                pose_detected=False,
                posture_ratio=0.55,
                hand_to_head=False,
                shoulder_tilt=0.0,
            )

        shoulder_mid = 0.5 * (l_sh + r_sh)
        neck_vertical = max(0.0, float(shoulder_mid[1] - nose[1]))
        shoulder_tilt = abs(float(l_sh[1] - r_sh[1])) / shoulder_width
        posture_ratio = (neck_vertical / shoulder_width) * max(0.5, 1.0 - 0.5 * shoulder_tilt)

        # Forehead center sits slightly above the nose/eye line relative to shoulder span.
        forehead_target = np.array([nose[0], nose[1] - 0.18 * shoulder_width], dtype=np.float32)
        hand_near_head = False
        active_wrist: Optional[Tuple[float, float]] = None

        for wrist_idx, wrist_pt in ((9, l_wr), (10, r_wr)):
            if kpts_conf[wrist_idx] >= 0.30 and wrist_pt[0] > 1 and wrist_pt[1] > 1:
                dist_ratio = float(np.linalg.norm(wrist_pt - forehead_target)) / shoulder_width
                if dist_ratio <= self.hand_to_head_threshold:
                    hand_near_head = True
                    active_wrist = (float(wrist_pt[0] / max(w, 1)), float(wrist_pt[1] / max(h, 1)))
                    break

        head_norm = (float(nose[0] / max(w, 1)), float(nose[1] / max(h, 1)))
        return PoseFeatures(
            pose_detected=True,
            posture_ratio=float(posture_ratio),
            hand_to_head=hand_near_head,
            shoulder_tilt=float(shoulder_tilt),
            head_point=head_norm,
            wrist_point=active_wrist,
        )
