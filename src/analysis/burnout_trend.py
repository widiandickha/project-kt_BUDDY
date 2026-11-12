"""Burnout trend (spec 4.7): per-minute mean stress, 30-min and 4-h moving averages, slope.

burnout_trend = linear-regression slope (score points per hour) of the 30-min moving average
over the last ma_long_hours (D-019). burnout_risk = slope >= slope_min and 4-h MA >= min_level,
both holding for >= min_duration_min.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

SEC_PER_MIN = 60.0
MIN_PER_HOUR = 60.0


@dataclass
class BurnoutStatus:
    """Current trend values."""

    ma_short: float | None
    ma_long: float | None
    slope_per_hour: float
    risk: bool
    minutes: int


class BurnoutTrend:
    """Feed one stress score per second (or minute averages via seed())."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        b = cfg["burnout"]
        self.short_min = int(b["ma_short_min"])
        self.long_min = int(float(b["ma_long_hours"]) * MIN_PER_HOUR)
        self.slope_min = float(b["slope_min"])
        self.min_level = float(b["min_level"])
        self.min_duration = float(b["min_duration_min"]) * SEC_PER_MIN
        self._minutes: deque[tuple[float, float]] = deque(maxlen=self.long_min)  # (minute_start_ts, mean)
        self._cur_minute: float | None = None
        self._cur_values: list[float] = []
        self._risk_since: float | None = None

    def seed(self, minute_means: list[tuple[float, float]]) -> None:
        """Load earlier minute averages (e.g. from the database after a restart)."""
        for ts, mean in sorted(minute_means)[-self.long_min:]:
            self._minutes.append((ts, mean))

    def add(self, ts: float, score: float) -> None:
        """Add one score sample."""
        minute = ts - (ts % SEC_PER_MIN)
        if self._cur_minute is not None and minute != self._cur_minute and self._cur_values:
            self._minutes.append((self._cur_minute, float(np.mean(self._cur_values))))
            self._cur_values = []
        self._cur_minute = minute
        self._cur_values.append(score)

    def minute_means(self) -> list[tuple[float, float]]:
        """Completed minute averages (oldest first)."""
        return list(self._minutes)

    def status(self, now: float) -> BurnoutStatus:
        """Moving averages, slope and risk flag at `now`."""
        means = np.array([m for _, m in self._minutes], dtype=np.float64)
        n = len(means)
        if n == 0:
            self._risk_since = None
            return BurnoutStatus(None, None, 0.0, False, 0)
        ma_short = float(means[-self.short_min:].mean())
        ma_long = float(means.mean())
        slope = 0.0
        if n >= 2:
            k = min(self.short_min, n)
            ma_series = np.convolve(means, np.ones(k) / k, mode="valid") if n >= k else means
            if len(ma_series) >= 2:
                hours = np.arange(len(ma_series)) / MIN_PER_HOUR
                slope = float(np.polyfit(hours, ma_series, 1)[0])
        condition = slope >= self.slope_min and ma_long >= self.min_level
        if condition:
            if self._risk_since is None:
                self._risk_since = now
        else:
            self._risk_since = None
        risk = self._risk_since is not None and now - self._risk_since >= self.min_duration
        return BurnoutStatus(ma_short, ma_long, slope, risk, n)
