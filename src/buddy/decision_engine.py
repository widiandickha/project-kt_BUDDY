from datetime import datetime
import time
from typing import Any, Dict, Optional

from src.analysis.scoring import WindowSummary
from src.buddy.messages import BuddyAdvice, pick_advice_for_reason


class BuddyDecisionEngine:
    """Mengatur kapan Virtual Buddy muncul dengan memperhitungkan prioritas, cooldown, dan snooze."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.update_config(config)
        self._last_trigger_ts: float = 0.0
        self._snooze_until_ts: float = 0.0

    def update_config(self, config: Dict[str, Any]) -> None:
        buddy_cfg = config.get("buddy", {})
        app_cfg = config.get("app", {})
        analysis_cfg = config.get("analysis", {})
        thresholds = analysis_cfg.get("thresholds", {})

        self.cooldown_seconds = float(buddy_cfg.get("cooldown_seconds", 180.0))
        self.snooze_minutes = float(buddy_cfg.get("snooze_minutes", 15.0))
        self.waspada_threshold = float(thresholds.get("waspada", 40.0))
        self.tinggi_threshold = float(thresholds.get("tinggi", 70.0))

        self.active_hours_enabled = bool(app_cfg.get("active_hours_enabled", False))
        self.active_hour_start = str(app_cfg.get("active_hour_start", "08:00"))
        self.active_hour_end = str(app_cfg.get("active_hour_end", "22:00"))

    def snooze(self, minutes: Optional[float] = None) -> None:
        duration_min = minutes if minutes is not None else self.snooze_minutes
        self._snooze_until_ts = time.monotonic() + (duration_min * 60.0)

    def clear_snooze(self) -> None:
        self._snooze_until_ts = 0.0

    @property
    def is_snoozed(self) -> bool:
        return time.monotonic() < self._snooze_until_ts

    @property
    def snooze_remaining_seconds(self) -> int:
        rem = self._snooze_until_ts - time.monotonic()
        return max(0, int(rem))

    def _within_active_hours(self) -> bool:
        if not self.active_hours_enabled:
            return True
        now_hm = datetime.now().strftime("%H:%M")
        if self.active_hour_start <= self.active_hour_end:
            return self.active_hour_start <= now_hm <= self.active_hour_end
        return now_hm >= self.active_hour_start or now_hm <= self.active_hour_end

    def evaluate(
        self,
        summary: WindowSummary,
        now_ts: Optional[float] = None,
    ) -> Optional[BuddyAdvice]:
        ts = now_ts if now_ts is not None else time.monotonic()
        if ts < self._snooze_until_ts:
            return None
        if not self._within_active_hours():
            return None

        # High urgency (Tinggi or Burnout) uses half the normal cooldown so critical drowsiness is not missed.
        effective_cooldown = (
            self.cooldown_seconds * 0.5
            if (summary.burnout_detected or summary.stress_score >= self.tinggi_threshold)
            else self.cooldown_seconds
        )
        if self._last_trigger_ts > 0.0 and (ts - self._last_trigger_ts) < effective_cooldown:
            return None

        if summary.burnout_detected:
            self._last_trigger_ts = ts
            return pick_advice_for_reason("burnout")

        if summary.stress_score >= self.waspada_threshold and summary.dominant_reason != "normal":
            self._last_trigger_ts = ts
            return pick_advice_for_reason(summary.dominant_reason)

        return None
