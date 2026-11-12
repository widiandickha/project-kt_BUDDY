"""Rule-based stress score (spec 4.5): normalized indicators x weights, EMA, level hysteresis.

score = 100 * sum(w_i * clamp(x_i / norm_i)); levels rendah < low <= sedang <= high < tinggi.
Going up a level needs the score above the threshold for up_hold_sec; going down needs it below
threshold - down_margin for down_hold_sec.
"""

from __future__ import annotations

from typing import Any

from src.analysis.sliding_window import WindowStats
from src.core.events import LEVELS, StressSnapshot


def clamp01(x: float) -> float:
    """Clamp to [0, 1]."""
    return 0.0 if x < 0 else 1.0 if x > 1 else float(x)


def blink_abnormality(blink_rate: float | None, normal: tuple[float, float]) -> float:
    """Distance (blinks/min) outside the normal range; 0 inside it or when unknown."""
    if blink_rate is None:
        return 0.0
    lo, hi = normal
    return max(0.0, lo - blink_rate, blink_rate - hi)


def normalized_indicators(stats: WindowStats, cfg: dict[str, Any]) -> dict[str, float]:
    """Each indicator mapped to 0..1 as in the spec table."""
    sc = cfg["score"]
    n = sc["norm"]
    normal = (float(sc["blink_normal"][0]), float(sc["blink_normal"][1]))
    return {
        "perclos": clamp01(stats.perclos / float(n["perclos"])),
        "yawn": clamp01(stats.yawn_count / float(n["yawn"])),
        "furrow": clamp01(stats.furrow_ratio / float(n["furrow"])),
        "hand": clamp01(stats.hand_forehead_count / float(n["hand"])),
        "blink": clamp01(blink_abnormality(stats.blink_rate, normal) / float(n["blink"])),
        "motion": clamp01(stats.trajectory_motion / float(n["motion"])),
    }


def level_of(score: float, low: float, high: float) -> str:
    """Level without hysteresis."""
    if score > high:
        return "tinggi"
    return "sedang" if score >= low else "rendah"


class StressScorer:
    """Call update() once per second."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        sc = cfg["score"]
        self.cfg = cfg
        self.weights = {k: float(v) for k, v in sc["weights"].items()}
        self.low = float(sc["thresholds"]["low"])
        self.high = float(sc["thresholds"]["high"])
        self.alpha = float(sc["ema_alpha"])
        self.up_hold = float(sc["up_hold_sec"])
        self.down_hold = float(sc["down_hold_sec"])
        self.margin = float(sc["down_margin"])
        self.min_span = float(sc["min_window_sec"])
        self.level = "rendah"
        self.ema: float | None = None
        self._up_since: float | None = None
        self._down_since: float | None = None

    def raw_score(self, stats: WindowStats) -> tuple[float, dict[str, float]]:
        """Score before smoothing and the contribution (score points) of each indicator."""
        norm = normalized_indicators(stats, self.cfg)
        components = {k: 100.0 * self.weights[k] * norm[k] for k in self.weights}
        return sum(components.values()), components

    def _lower_bound(self, level: str) -> float:
        return {"rendah": float("-inf"), "sedang": self.low, "tinggi": self.high}[level]

    def _apply_hysteresis(self, score: float, now: float) -> None:
        target = level_of(score, self.low, self.high)
        rank = LEVELS.index
        if rank(target) > rank(self.level):
            self._down_since = None
            if self._up_since is None:
                self._up_since = now
            if now - self._up_since >= self.up_hold:
                self.level, self._up_since = target, None
            return
        self._up_since = None
        if score < self._lower_bound(self.level) - self.margin:
            if self._down_since is None:
                self._down_since = now
            if now - self._down_since >= self.down_hold:
                self.level = level_of(score + self.margin, self.low, self.high)
                self._down_since = None
        else:
            self._down_since = None

    def update(self, stats: WindowStats, now: float, burnout_trend: float = 0.0, burnout_risk: bool = False) -> StressSnapshot | None:
        """New snapshot, or None while the window holds too little data or nobody is present."""
        if stats.span_sec < self.min_span or stats.present_ratio <= 0.0:
            return None
        raw, components = self.raw_score(stats)
        self.ema = raw if self.ema is None else self.alpha * raw + (1.0 - self.alpha) * self.ema
        self._apply_hysteresis(self.ema, now)
        return StressSnapshot(
            ts=now, score=round(self.ema, 2), level=self.level, components={k: round(v, 2) for k, v in components.items()},
            burnout_trend=burnout_trend, raw_score=round(raw, 2), burnout_risk=burnout_risk,
        )
