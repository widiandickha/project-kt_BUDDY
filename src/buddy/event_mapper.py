"""Turn analysis outputs (stress snapshots, habit alerts, presence) into BuddyEvents (D-021).

kind == state-machine trigger; message_key == advice bank key ("" = no bubble).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from src.core.events import EVENT_USER_AWAY, EVENT_USER_RETURNED, BuddyEvent, HabitAlert, PresenceEvent, StressSnapshot

SEVERITY: dict[str, int] = {
    "stress_sedang": 2, "stress_tinggi": 3, "burnout_risk": 3, "need_break": 3, "late_night": 3,
    "bad_posture": 2, "too_close": 2, "dry_eyes": 2, "sleepy": 2,
    "good_break": 1, "doing_great": 1, "user_returned": 1, "celebrate": 1, "stress_calm": 1, "user_away_long": 1,
}
SILENT: frozenset[str] = frozenset({"stress_calm", "user_away_long"})
HABIT_KINDS: frozenset[str] = frozenset({"need_break", "bad_posture", "too_close", "dry_eyes", "late_night", "good_break", "doing_great"})


class BuddyEventMapper:
    """Stateful: detects level changes, calm periods, sleepiness edges and daily celebrations."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        b = cfg["buddy"]
        self.calm_sec = float(b["calm_sec"])
        self.away_sleep = float(b["away_sleep_sec"])
        self.sleepy_level = float(b["sleepy_component"])
        self.celebrate_breaks = int(b["celebrate_breaks"])
        self.weights = {k: float(v) for k, v in cfg["score"]["weights"].items()}
        self._level = "rendah"
        self._calm_since: float | None = None
        self._elevated = False
        self._burnout = False
        self._sleepy = False
        self._away_since: float | None = None
        self._slept = False
        self._breaks: dict[str, int] = {}
        self._celebrated: set[str] = set()

    @staticmethod
    def _event(kind: str, ts: float) -> BuddyEvent:
        return BuddyEvent(kind=kind, severity=SEVERITY[kind], message_key="" if kind in SILENT else kind, ts=ts)

    def _norm(self, snap: StressSnapshot, key: str) -> float:
        w = self.weights.get(key, 0.0)
        return snap.components.get(key, 0.0) / (100.0 * w) if w > 0 else 0.0

    def on_snapshot(self, snap: StressSnapshot) -> list[BuddyEvent]:
        """Level transitions, calm-down, burnout and sleepiness."""
        out: list[BuddyEvent] = []
        order = ("rendah", "sedang", "tinggi")
        if order.index(snap.level) > order.index(self._level):
            out.append(self._event(f"stress_{snap.level}", snap.ts))
        self._level = snap.level
        if snap.level == "rendah":
            if self._elevated:
                self._calm_since = self._calm_since or snap.ts
                if snap.ts - self._calm_since >= self.calm_sec:
                    out.append(self._event("stress_calm", snap.ts))
                    self._elevated, self._calm_since = False, None
        else:
            self._elevated, self._calm_since = True, None

        if snap.burnout_risk and not self._burnout:
            out.append(self._event("burnout_risk", snap.ts))
        self._burnout = snap.burnout_risk

        sleepy = max(self._norm(snap, "perclos"), self._norm(snap, "yawn")) >= self.sleepy_level
        if sleepy and not self._sleepy:
            out.append(self._event("sleepy", snap.ts))
            self._elevated = True
        self._sleepy = sleepy
        return out

    def on_habit(self, alert: HabitAlert) -> list[BuddyEvent]:
        """Habit alerts map 1:1; enough good breaks in a day add a celebration."""
        if alert.kind not in HABIT_KINDS:
            return []
        out = [self._event(alert.kind, alert.ts)]
        if alert.kind == "good_break":
            day = datetime.fromtimestamp(alert.ts).date().isoformat()
            self._breaks[day] = self._breaks.get(day, 0) + 1
            if self._breaks[day] >= self.celebrate_breaks and day not in self._celebrated:
                self._celebrated.add(day)
                out.append(self._event("celebrate", alert.ts))
        return out

    def on_presence(self, p: PresenceEvent) -> list[BuddyEvent]:
        """Remember when the user left; greet on return."""
        if p.kind == EVENT_USER_AWAY:
            self._away_since, self._slept = p.ts - p.away_sec, False
            return []
        if p.kind == EVENT_USER_RETURNED:
            self._away_since = None
            return [self._event("user_returned", p.ts)]
        return []

    def tick(self, now: float) -> list[BuddyEvent]:
        """Fall asleep after buddy.away_sleep_sec away."""
        if self._away_since is not None and not self._slept and now - self._away_since >= self.away_sleep:
            self._slept = True
            return [self._event("user_away_long", now)]
        return []
