from collections import deque
from dataclasses import dataclass
import time
from typing import Any, Deque, Dict, Optional, Tuple
import numpy as np

from src.analysis.calibration import UserBaseline


@dataclass
class FrameSample:
    timestamp: float
    ear: float
    eye_closed: bool
    yawn: bool
    brow_furrow: bool
    slouch: bool
    hand_to_head: bool
    head_point: Optional[Tuple[float, float]]


@dataclass
class WindowSummary:
    ear_current: float
    blink_rate_per_min: float
    eye_closed_ratio: float
    yawn_ratio: float
    brow_furrow_ratio: float
    slouch_ratio: float
    hand_to_head_ratio: float
    head_restlessness: float
    stress_score: float
    stress_level: str
    moving_avg_15m: float
    burnout_detected: bool
    dominant_reason: str


class StressScoreEngine:
    """Mengumpulkan indikator ke dalam sliding window 60 detik dan menghitung skor stres."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.update_config(config)
        self._window: Deque[FrameSample] = deque()
        self._burnout_history: Deque[Tuple[float, float]] = deque()
        self._blink_timestamps: Deque[float] = deque()
        self._prev_eye_closed: bool = False
        self._closed_Run_frames: int = 0

    def update_config(self, config: Dict[str, Any]) -> None:
        analysis_cfg = config.get("analysis", {})
        app_cfg = config.get("app", {})
        self.window_seconds = float(analysis_cfg.get("sliding_window_seconds", 60.0))
        self.burnout_seconds = float(analysis_cfg.get("burnout_window_minutes", 15.0)) * 60.0
        self.ear_closed_factor = float(analysis_cfg.get("ear_closed_factor", 0.75))
        self.mar_yawn_threshold = float(analysis_cfg.get("mar_yawn_threshold", 0.55))
        self.brow_furrow_factor = float(analysis_cfg.get("brow_furrow_factor", 0.88))
        self.posture_slouch_factor = float(analysis_cfg.get("posture_slouch_factor", 0.82))
        self.sensitivity = float(app_cfg.get("sensitivity", 1.0))

        weights = analysis_cfg.get("weights", {})
        self.w_eye = float(weights.get("eye_closed", 0.30))
        self.w_yawn = float(weights.get("yawn", 0.20))
        self.w_brow = float(weights.get("brow_furrow", 0.15))
        self.w_slouch = float(weights.get("posture_slouch", 0.15))
        self.w_hand = float(weights.get("hand_to_head", 0.10))
        self.w_jitter = float(weights.get("head_restlessness", 0.10))

        thresholds = analysis_cfg.get("thresholds", {})
        self.th_waspada = float(thresholds.get("waspada", 40.0))
        self.th_tinggi = float(thresholds.get("tinggi", 70.0))
        self.th_burnout = float(thresholds.get("burnout_avg", 60.0))

    def process_frame(
        self,
        baseline: UserBaseline,
        ear: float,
        mar: float,
        brow_ratio: float,
        posture_ratio: float,
        hand_to_head: bool,
        head_point: Optional[Tuple[float, float]],
        yolo_eye_closed: bool = False,
        yolo_yawn: bool = False,
        yolo_brow_furrow: bool = False,
        now_ts: Optional[float] = None,
    ) -> WindowSummary:
        ts = now_ts if now_ts is not None else time.monotonic()

        eye_closed = yolo_eye_closed or (ear < baseline.ear * self.ear_closed_factor)
        yawn = yolo_yawn or (mar > self.mar_yawn_threshold)
        brow_furrow = yolo_brow_furrow or (brow_ratio < baseline.brow_ratio * self.brow_furrow_factor)
        slouch = posture_ratio < (baseline.posture_ratio * self.posture_slouch_factor)

        if eye_closed:
            self._closed_Run_frames += 1
        else:
            if 1 <= self._closed_Run_frames <= 8 and self._prev_eye_closed:
                self._blink_timestamps.append(ts)
            self._closed_Run_frames = 0
        self._prev_eye_closed = eye_closed

        self._window.append(
            FrameSample(
                timestamp=ts,
                ear=ear,
                eye_closed=eye_closed,
                yawn=yawn,
                brow_furrow=brow_furrow,
                slouch=slouch,
                hand_to_head=hand_to_head,
                head_point=head_point,
            )
        )

        cutoff_window = ts - self.window_seconds
        while self._window and self._window[0].timestamp < cutoff_window:
            self._window.popleft()

        cutoff_blinks = ts - 60.0
        while self._blink_timestamps and self._blink_timestamps[0] < cutoff_blinks:
            self._blink_timestamps.popleft()

        n = max(1, len(self._window))
        eye_closed_ratio = sum(1 for s in self._window if s.eye_closed) / n
        yawn_ratio = sum(1 for s in self._window if s.yawn) / n
        brow_ratio_win = sum(1 for s in self._window if s.brow_furrow) / n
        slouch_ratio = sum(1 for s in self._window if s.slouch) / n
        hand_ratio = sum(1 for s in self._window if s.hand_to_head) / n

        head_pts = [s.head_point for s in self._window if s.head_point is not None]
        if len(head_pts) >= 4:
            arr = np.array(head_pts, dtype=np.float32)
            diffs = np.linalg.norm(np.diff(arr, axis=0), axis=1)
            # Scale frame-to-frame head movement into a [0, 1] restlessness indicator.
            head_restlessness = float(min(1.0, np.mean(diffs) * 28.0))
        else:
            head_restlessness = 0.0

        elapsed_win = max(10.0, min(60.0, ts - self._window[0].timestamp + 0.1))
        blink_rate = len(self._blink_timestamps) * (60.0 / elapsed_win)

        # Normal blink rate sits around 12-20/min; staring (<8/min) adds dry-eye strain.
        dry_eye_strain = max(0.0, (10.0 - blink_rate) / 10.0) if elapsed_win >= 20.0 else 0.0
        eye_component = min(1.0, max(eye_closed_ratio * 2.2, dry_eye_strain * 0.65))
        yawn_component = min(1.0, yawn_ratio * 3.0)
        brow_component = min(1.0, brow_ratio_win * 2.0)
        slouch_component = min(1.0, slouch_ratio * 1.6)
        hand_component = min(1.0, hand_ratio * 2.2)

        raw_score = 100.0 * (
            self.w_eye * eye_component
            + self.w_yawn * yawn_component
            + self.w_brow * brow_component
            + self.w_slouch * slouch_component
            + self.w_hand * hand_component
            + self.w_jitter * head_restlessness
        )
        stress_score = float(np.clip(raw_score * self.sensitivity, 0.0, 100.0))

        self._burnout_history.append((ts, stress_score))
        cutoff_burnout = ts - self.burnout_seconds
        while self._burnout_history and self._burnout_history[0][0] < cutoff_burnout:
            self._burnout_history.popleft()

        moving_avg_15m = float(np.mean([s for _, s in self._burnout_history]))
        burnout_span = (
            self._burnout_history[-1][0] - self._burnout_history[0][0]
            if len(self._burnout_history) >= 2
            else 0.0
        )
        burnout_detected = (
            burnout_span >= min(120.0, self.burnout_seconds * 0.4)
            and moving_avg_15m >= self.th_burnout
        )

        if stress_score >= self.th_tinggi:
            level = "Tinggi"
        elif stress_score >= self.th_waspada:
            level = "Waspada"
        else:
            level = "Normal"

        dominant_reason = self._pick_dominant_reason(
            burnout_detected=burnout_detected,
            eye_closed_ratio=eye_closed_ratio,
            yawn_ratio=yawn_ratio,
            brow_ratio_win=brow_ratio_win,
            slouch_ratio=slouch_ratio,
            hand_ratio=hand_ratio,
            dry_eye_strain=dry_eye_strain,
        )

        return WindowSummary(
            ear_current=ear,
            blink_rate_per_min=blink_rate,
            eye_closed_ratio=eye_closed_ratio,
            yawn_ratio=yawn_ratio,
            brow_furrow_ratio=brow_ratio_win,
            slouch_ratio=slouch_ratio,
            hand_to_head_ratio=hand_ratio,
            head_restlessness=head_restlessness,
            stress_score=stress_score,
            stress_level=level,
            moving_avg_15m=moving_avg_15m,
            burnout_detected=burnout_detected,
            dominant_reason=dominant_reason,
        )

    def _pick_dominant_reason(
        self,
        burnout_detected: bool,
        eye_closed_ratio: float,
        yawn_ratio: float,
        brow_ratio_win: float,
        slouch_ratio: float,
        hand_ratio: float,
        dry_eye_strain: float,
    ) -> str:
        if burnout_detected:
            return "burnout"
        if eye_closed_ratio >= 0.25 or yawn_ratio >= 0.12:
            return "kantuk"
        if brow_ratio_win >= 0.25 or hand_ratio >= 0.20:
            return "tegang_dahi"
        if slouch_ratio >= 0.30:
            return "postur_bungkuk"
        if dry_eye_strain >= 0.35:
            return "jarang_berkedip"
        return "normal"
