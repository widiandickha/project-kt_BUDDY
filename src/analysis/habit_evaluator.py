"""Habit evaluation (spec 4.8), separate from the stress score.

| habit              | rule                                                         | event        |
|--------------------|--------------------------------------------------------------|--------------|
| sitting too long   | no user_away >= away_min_sec for work_break_min              | need_break   |
| slouching          | shoulder_slouch < baseline * (1 - slouch_drop) for 3 min     | bad_posture  |
| too close          | face_size_ratio > baseline * too_close_ratio for 2 min       | too_close    |
| rare blinking      | blink_rate < low_blink for 3 min                             | dry_eyes     |
| late night         | local hour >= late_hour while active (once per night)        | late_night   |
| good break         | away 5..15 min after a long work stretch                     | good_break   |
| doing great        | 2 h at level "rendah"                                        | doing_great  |

Presence: user_away when no face/person for > away_after_sec, user_returned when seen again.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timedelta
from typing import Any

from src.analysis.calibration import Calibration
from src.analysis.sliding_window import WindowStats
from src.core.events import EVENT_USER_AWAY, EVENT_USER_RETURNED, FrameFeatures, HabitAlert, PresenceEvent, StressSnapshot

SEC_PER_MIN = 60.0
SEC_PER_HOUR = 3600.0
NIGHT_END_HOUR = 4  # hours before this still count as the previous night


class Sustained:
    """Fires once when a condition has held for `hold_sec`; re-arms when it becomes false."""

    def __init__(self, hold_sec: float) -> None:
        self.hold = hold_sec
        self._since: float | None = None
        self._fired = False

    def update(self, now: float, condition: bool) -> bool:
        """Feed the condition; True exactly once per sustained episode."""
        if not condition:
            self._since, self._fired = None, False
            return False
        if self._since is None:
            self._since = now
        if not self._fired and now - self._since >= self.hold:
            self._fired = True
            return True
        return False

    def reset(self) -> None:
        """Forget the current episode."""
        self._since, self._fired = None, False


class HabitEvaluator:
    """Call observe() for every frame and tick() once per second."""

    def __init__(self, cfg: dict[str, Any], calibration: Calibration, clock: Callable[[float], datetime] | None = None) -> None:
        h = cfg["habit"]
        self.calibration = calibration
        self.work_break = float(h["work_break_min"]) * SEC_PER_MIN
        self.away_min = float(h["away_min_sec"])
        self.slouch_drop = float(h["slouch_drop"])
        self.too_close_ratio = float(h["too_close_ratio"])
        self.low_blink = float(h["low_blink"])
        self.late_hour = int(h["late_hour"])
        self.away_after = float(h["away_after_sec"])
        self.good_break_max = float(h["good_break_max_min"]) * SEC_PER_MIN
        self.long_work = float(h["long_work_min"]) * SEC_PER_MIN
        self._clock = clock or (lambda ts: datetime.fromtimestamp(ts))
        self._posture = Sustained(float(h["posture_hold_min"]) * SEC_PER_MIN)
        self._close = Sustained(float(h["close_hold_min"]) * SEC_PER_MIN)
        self._dry = Sustained(float(h["blink_hold_min"]) * SEC_PER_MIN)
        self._great = Sustained(float(h["great_hours"]) * SEC_PER_HOUR)
        self._last_seen: float | None = None
        self._away_since: float | None = None
        self._work_start: float | None = None
        self._last_break_alert: float | None = None
        self._late_night_key: str | None = None
        self._slouch: float | None = None
        self._face_size: float | None = None

    def set_calibration(self, calibration: Calibration) -> None:
        """Use a new baseline."""
        self.calibration = calibration

    def observe(self, f: FrameFeatures) -> None:
        """Track presence and the latest posture/distance values."""
        if f.user_present:
            self._last_seen = f.ts
            if f.shoulder_slouch is not None:
                self._slouch = f.shoulder_slouch
            if f.face_size_ratio is not None:
                self._face_size = f.face_size_ratio

    def reset_presence(self) -> None:
        """Forget presence after a pause (pausing the camera is not 'leaving the desk')."""
        self._last_seen = None
        self._away_since = None

    @property
    def away(self) -> bool:
        """True while the user is considered away."""
        return self._away_since is not None

    def _presence(self, now: float) -> tuple[list[PresenceEvent], list[HabitAlert]]:
        presence: list[PresenceEvent] = []
        alerts: list[HabitAlert] = []
        if self._last_seen is None:
            return presence, alerts
        gone_for = now - self._last_seen
        if self._away_since is None and gone_for > self.away_after:
            self._away_since = self._last_seen
            presence.append(PresenceEvent(EVENT_USER_AWAY, now, gone_for))
            for s in (self._posture, self._close, self._dry):
                s.reset()
        elif self._away_since is not None and gone_for <= self.away_after:
            away_sec = self._last_seen - self._away_since
            presence.append(PresenceEvent(EVENT_USER_RETURNED, now, away_sec))
            worked = (self._away_since - self._work_start) if self._work_start is not None else 0.0
            if self.away_min <= away_sec <= self.good_break_max and worked >= self.long_work:
                alerts.append(HabitAlert("good_break", now, positive=True,
                                         detail={"away_min": round(away_sec / SEC_PER_MIN, 1)}))
            if away_sec >= self.away_min:
                self._work_start, self._last_break_alert = None, None
            self._away_since = None
        if self._away_since is None and self._work_start is None:
            self._work_start = self._last_seen
        return presence, alerts

    def tick(self, now: float, stats: WindowStats, snapshot: StressSnapshot | None) -> tuple[list[PresenceEvent], list[HabitAlert]]:
        """Evaluate all rules at time `now`."""
        presence, alerts = self._presence(now)
        if self.away or self._last_seen is None:
            return presence, alerts

        if self._work_start is not None:
            ref = self._last_break_alert or self._work_start
            if now - ref >= self.work_break:
                self._last_break_alert = now
                alerts.append(HabitAlert("need_break", now, detail={"work_min": round((now - self._work_start) / SEC_PER_MIN)}))

        base_slouch = self.calibration.shoulder_slouch
        if base_slouch and self._slouch is not None and self._posture.update(now, self._slouch < base_slouch * (1.0 - self.slouch_drop)):
            alerts.append(HabitAlert("bad_posture", now, detail={"slouch": round(self._slouch, 3)}))

        base_face = self.calibration.face_size_ratio
        if base_face and self._face_size is not None and self._close.update(now, self._face_size > base_face * self.too_close_ratio):
            alerts.append(HabitAlert("too_close", now, detail={"face_ratio": round(self._face_size, 3)}))

        if stats.blink_rate is not None and self._dry.update(now, stats.blink_rate < self.low_blink):
            alerts.append(HabitAlert("dry_eyes", now, detail={"blink_rate": round(stats.blink_rate, 1)}))

        local = self._clock(now)
        if local.hour >= self.late_hour or local.hour < NIGHT_END_HOUR:
            night = (local - timedelta(hours=NIGHT_END_HOUR)).date().isoformat()  # 23:00 and 01:00 = same night
            if self._late_night_key != night:
                self._late_night_key = night
                alerts.append(HabitAlert("late_night", now, detail={"hour": local.hour}))

        if snapshot is not None and self._great.update(now, snapshot.level == "rendah"):
            alerts.append(HabitAlert("doing_great", now, positive=True))
        return presence, alerts
