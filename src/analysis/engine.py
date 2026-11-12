"""Analysis engine: SlidingWindow + StressScorer + BurnoutTrend + HabitEvaluator behind one API.

The inference thread calls add(features) per frame; the analysis worker calls tick(now) once
per second and gets the snapshot plus habit/presence events. If an EventBus is given, results
are also published under the spec event names. (Extension module, D-017.)
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from src.analysis.burnout_trend import BurnoutStatus, BurnoutTrend
from src.analysis.calibration import Calibration
from src.analysis.habit_evaluator import HabitEvaluator
from src.analysis.sliding_window import SlidingWindow, WindowStats
from src.analysis.stress_scorer import StressScorer
from src.core.events import (
    EVENT_FRAME_FEATURES,
    EVENT_HABIT_ALERT,
    EVENT_STRESS_SNAPSHOT,
    EventBus,
    FrameFeatures,
    HabitAlert,
    PresenceEvent,
    StressSnapshot,
)


@dataclass
class TickResult:
    """Everything computed in one analysis tick."""

    ts: float
    stats: WindowStats
    snapshot: StressSnapshot | None
    burnout: BurnoutStatus
    presence: list[PresenceEvent] = field(default_factory=list)
    alerts: list[HabitAlert] = field(default_factory=list)


class AnalysisEngine:
    """Thread-safe add(); tick() must be called from a single thread."""

    def __init__(
        self,
        cfg: dict[str, Any],
        calibration: Calibration,
        bus: EventBus | None = None,
        clock: Callable[[float], datetime] | None = None,
    ) -> None:
        self.cfg = cfg
        self.bus = bus
        self.calibration = calibration
        self.window = SlidingWindow(cfg, calibration.ear_threshold)
        self.scorer = StressScorer(cfg)
        self.burnout = BurnoutTrend(cfg)
        self.habits = HabitEvaluator(cfg, calibration, clock=clock)
        self.last_snapshot: StressSnapshot | None = None

    def set_calibration(self, calibration: Calibration) -> None:
        """Apply a new user calibration everywhere."""
        self.calibration = calibration
        self.window.set_threshold(calibration.ear_threshold)
        self.habits.set_calibration(calibration)

    def add(self, f: FrameFeatures) -> None:
        """Feed one frame."""
        self.window.add(f)
        self.habits.observe(f)
        if self.bus is not None:
            self.bus.publish(EVENT_FRAME_FEATURES, f)

    def tick(self, now: float) -> TickResult:
        """Run the once-per-second analysis."""
        stats = self.window.stats(now)
        presence, alerts = self.habits.tick(now, stats, self.last_snapshot)
        burnout = self.burnout.status(now)
        snapshot = None
        if not self.habits.away:
            snapshot = self.scorer.update(stats, now, burnout.slope_per_hour, burnout.risk)
            if snapshot is not None:
                self.burnout.add(now, snapshot.score)
                self.last_snapshot = snapshot
        if self.bus is not None:
            for p in presence:
                self.bus.publish(p.kind, p)
            for a in alerts:
                self.bus.publish(EVENT_HABIT_ALERT, a)
            if snapshot is not None:
                self.bus.publish(EVENT_STRESS_SNAPSHOT, snapshot)
        return TickResult(now, stats, snapshot, burnout, presence, alerts)
