"""First-run calibration (spec 4.6): open-eye EAR, closed-eye EAR, upright posture, normal distance.

ear.closed_threshold = open_ear - 0.4 * (open_ear - closed_ear). Values are stored as key/value
rows in the user_calibration table (FASE 6).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from src.core.events import FrameFeatures

THRESHOLD_FRACTION = 0.4  # spec 4.6


@dataclass
class Calibration:
    """Per-user baseline."""

    open_ear: float
    closed_ear: float
    shoulder_slouch: float | None
    face_size_ratio: float | None
    brow_ratio: float | None
    ear_threshold: float

    @classmethod
    def from_values(
        cls, open_ear: float, closed_ear: float, shoulder_slouch: float | None,
        face_size_ratio: float | None, brow_ratio: float | None,
    ) -> Calibration:
        """Build a calibration and derive the EAR threshold."""
        return cls(open_ear, closed_ear, shoulder_slouch, face_size_ratio, brow_ratio,
                   open_ear - THRESHOLD_FRACTION * (open_ear - closed_ear))

    @classmethod
    def default(cls, cfg: dict[str, Any]) -> Calibration:
        """Fallback before the wizard has run: config threshold, no posture/distance baseline."""
        c = cfg["calibration"]
        return cls(float(c["default_open_ear"]), float(c["default_closed_ear"]), None, None, None,
                   float(cfg["ear"]["closed_threshold"]))

    def to_rows(self) -> dict[str, float]:
        """Key/value rows for the user_calibration table (None values are skipped)."""
        return {k: float(v) for k, v in asdict(self).items() if v is not None}

    @classmethod
    def from_rows(cls, rows: dict[str, float], cfg: dict[str, Any]) -> Calibration | None:
        """Inverse of to_rows; None when the required keys are missing."""
        if "open_ear" not in rows or "closed_ear" not in rows:
            return None
        base = cls.default(cfg)
        return cls(
            rows["open_ear"], rows["closed_ear"], rows.get("shoulder_slouch"), rows.get("face_size_ratio"),
            rows.get("brow_ratio"), rows.get("ear_threshold", base.ear_threshold),
        )


class CalibrationSession:
    """Collects features during the wizard: first eyes open, then eyes closed."""

    def __init__(self, cfg: dict[str, Any], start_ts: float) -> None:
        c = cfg["calibration"]
        self.total_sec = float(c["total_sec"])
        self.closed_sec = float(c["closed_sec"])
        self.min_samples = int(c["min_samples"])
        self.start_ts = start_ts
        self._open: list[FrameFeatures] = []
        self._closed: list[float] = []

    def phase(self, now: float) -> str:
        """'open', 'closed' or 'done'."""
        t = now - self.start_ts
        if t < self.total_sec - self.closed_sec:
            return "open"
        return "closed" if t < self.total_sec else "done"

    def progress(self, now: float) -> float:
        """0..1."""
        return float(min(1.0, max(0.0, (now - self.start_ts) / self.total_sec)))

    def add(self, f: FrameFeatures) -> None:
        """Add one frame with a visible face."""
        if f.ear_mean is None:
            return
        ph = self.phase(f.ts)
        if ph == "open":
            self._open.append(f)
        elif ph == "closed":
            self._closed.append(f.ear_mean)

    def result(self) -> Calibration | None:
        """Median-based calibration, or None when too few valid samples were seen."""
        if len(self._open) < self.min_samples or len(self._closed) < max(3, self.min_samples // 3):
            return None

        def med(values: list[float | None]) -> float | None:
            vals = [v for v in values if v is not None]
            return float(np.median(vals)) if vals else None

        open_ear = med([f.ear_mean for f in self._open])
        closed_ear = float(np.percentile(self._closed, 25))  # most-closed part of the closed phase
        if open_ear is None or closed_ear >= open_ear:
            return None
        return Calibration.from_values(
            open_ear, closed_ear,
            med([f.shoulder_slouch for f in self._open]),
            med([f.face_size_ratio for f in self._open]),
            med([f.brow_ratio for f in self._open]),
        )
