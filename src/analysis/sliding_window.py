"""Time-based sliding window over FrameFeatures (spec 4.3 + 4.4).

Per window: PERCLOS (only closures >= ear.microsleep_ms count), blink rate (closures between
blink_min_ms and microsleep_ms), yawn count (>= yawn_min_sec, debounced), furrow ratio,
hand-to-forehead ratio/count, and trajectory motion (std of the nose position).
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.core.events import FrameFeatures

ACTIVE = 0.5  # a YOLO/landmark signal in 0..1 counts as active above this


def eye_closed(f: FrameFeatures, ear_threshold: float, cfg: dict[str, Any]) -> bool | None:
    """Fused eye state (spec 4.3); None when neither source is available.

    Both available: w_ear * [ear < threshold] + w_yolo * conf_yolo >= ear.fusion_threshold.
    Only one: that source alone. With the spec weights (0.6/0.4) EAR decides when it is present.
    """
    ear_cfg = cfg["ear"]
    ear_sig = None if f.ear_mean is None else (1.0 if f.ear_mean < ear_threshold else 0.0)
    if f.det_available and ear_sig is not None:
        fused = float(ear_cfg["fusion_w_ear"]) * ear_sig + float(ear_cfg["fusion_w_yolo"]) * f.eye_closed_yolo
        return fused >= float(ear_cfg["fusion_threshold"])
    if ear_sig is not None:
        return ear_sig >= ACTIVE
    if f.det_available and f.user_present:
        return f.eye_closed_yolo > float(cfg["det"]["conf_eye"])
    return None


@dataclass
class _Sample:
    ts: float
    present: bool
    closed: bool | None
    yawn: bool
    furrow: bool
    hand: bool
    nose: tuple[float, float] | None


@dataclass
class WindowStats:
    """Indicators of the short window."""

    span_sec: float
    present_ratio: float
    perclos: float
    blink_rate: float | None
    yawn_count: int
    furrow_ratio: float
    hand_forehead_ratio: float
    hand_forehead_count: int
    trajectory_motion: float
    eyes_closed_now: bool


def _runs(samples: list[_Sample], attr: str, end_ts: float) -> list[tuple[float, float]]:
    """(start, duration) of consecutive runs where attr is True; a run ends at the next sample."""
    runs: list[tuple[float, float]] = []
    start: float | None = None
    for i, s in enumerate(samples):
        on = bool(getattr(s, attr))
        if on and start is None:
            start = s.ts
        if not on and start is not None:
            runs.append((start, s.ts - start))
            start = None
        if i == len(samples) - 1 and start is not None:
            runs.append((start, max(end_ts, s.ts) - start))
    return runs


class SlidingWindow:
    """Thread-safe: the inference thread adds, the analysis tick reads."""

    def __init__(self, cfg: dict[str, Any], ear_threshold: float) -> None:
        self.cfg = cfg
        self.span = float(cfg["window"]["short_sec"])
        self.ear_threshold = ear_threshold
        ear = cfg["ear"]
        self.blink_min = float(ear["blink_min_ms"]) / 1000.0
        self.microsleep = float(ear["microsleep_ms"]) / 1000.0
        self.yawn_min = float(cfg["window"]["yawn_min_sec"])
        self.yawn_debounce = float(cfg["window"]["yawn_debounce_sec"])
        self._samples: deque[_Sample] = deque()
        self._lock = threading.Lock()
        self._last_yawn_counted: float = float("-inf")
        self._counted_yawns: deque[float] = deque()
        self._hand_prev = False
        self.total_yawns = 0          # monotonic counters, used to log events (FASE 7)
        self.total_hand_episodes = 0

    def set_threshold(self, ear_threshold: float) -> None:
        """Use a newly calibrated EAR threshold."""
        self.ear_threshold = ear_threshold

    def add(self, f: FrameFeatures) -> None:
        """Append one frame and drop samples older than the window."""
        s = _Sample(
            ts=f.ts,
            present=f.user_present,
            closed=eye_closed(f, self.ear_threshold, self.cfg) if f.user_present else None,
            yawn=f.user_present and f.yawn > ACTIVE,
            furrow=f.user_present and f.brow_furrow > ACTIVE,
            hand=f.user_present and f.hand_to_forehead,
            nose=f.nose_xy if f.user_present else None,
        )
        if s.hand and not self._hand_prev:
            self.total_hand_episodes += 1
        self._hand_prev = s.hand
        with self._lock:
            self._samples.append(s)
            cutoff = f.ts - self.span
            while self._samples and self._samples[0].ts < cutoff:
                self._samples.popleft()

    def _update_yawns(self, samples: list[_Sample], now: float) -> int:
        """Count yawn episodes >= yawn_min_sec, at most one per yawn_debounce_sec; counts persist
        across ticks so an episode is never counted twice."""
        for start, dur in _runs(samples, "yawn", now):
            if dur >= self.yawn_min and start - self._last_yawn_counted >= self.yawn_debounce:
                self._last_yawn_counted = start
                self._counted_yawns.append(start)
                self.total_yawns += 1
        while self._counted_yawns and self._counted_yawns[0] < now - self.span:
            self._counted_yawns.popleft()
        return len(self._counted_yawns)

    def stats(self, now: float) -> WindowStats:
        """Compute the window indicators at time `now`."""
        with self._lock:
            samples = list(self._samples)
        if not samples:
            return WindowStats(0.0, 0.0, 0.0, None, 0, 0.0, 0.0, 0, 0.0, False)
        span = max(1e-6, min(self.span, now - samples[0].ts))
        present = [s for s in samples if s.present]
        present_ratio = len(present) / len(samples)

        closed_runs = _runs([s for s in samples if s.closed is not None], "closed", now)
        perclos = sum(d for _, d in closed_runs if d >= self.microsleep) / span
        blinks = sum(1 for _, d in closed_runs if self.blink_min <= d < self.microsleep)
        min_span = float(self.cfg["score"]["min_window_sec"])
        blink_rate = blinks * 60.0 / span if span >= min_span and present_ratio > 0 else None

        def time_ratio(attr: str) -> float:
            return min(1.0, sum(d for _, d in _runs(samples, attr, now)) / span)

        hand_runs = _runs(samples, "hand", now)
        noses = np.array([s.nose for s in present if s.nose is not None], dtype=np.float64)
        motion = float(noses.std(axis=0).mean()) if len(noses) >= 3 else 0.0
        return WindowStats(
            span_sec=span,
            present_ratio=present_ratio,
            perclos=min(1.0, perclos),
            blink_rate=blink_rate,
            yawn_count=self._update_yawns(samples, now),
            furrow_ratio=time_ratio("furrow"),
            hand_forehead_ratio=time_ratio("hand"),
            hand_forehead_count=len(hand_runs),
            trajectory_motion=motion,
            eyes_closed_now=bool(samples[-1].closed),
        )
