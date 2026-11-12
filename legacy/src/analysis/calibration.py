from dataclasses import dataclass
import time
from typing import List
import numpy as np


@dataclass
class UserBaseline:
    ear: float = 0.28
    mar: float = 0.15
    brow_ratio: float = 0.24
    posture_ratio: float = 0.55
    calibrated: bool = False


class BaselineCalibrator:
    """Mengumpulkan indikator awal pengguna di awal sesi untuk menentukan nilai baseline."""

    def __init__(
        self,
        duration_seconds: float = 10.0,
        default_ear: float = 0.28,
        default_mar: float = 0.15,
        default_brow_ratio: float = 0.24,
        default_posture_ratio: float = 0.55,
    ) -> None:
        self.duration_seconds = max(2.0, duration_seconds)
        self.default_baseline = UserBaseline(
            ear=default_ear,
            mar=default_mar,
            brow_ratio=default_brow_ratio,
            posture_ratio=default_posture_ratio,
            calibrated=False,
        )
        self.baseline = UserBaseline(
            ear=default_ear,
            mar=default_mar,
            brow_ratio=default_brow_ratio,
            posture_ratio=default_posture_ratio,
            calibrated=False,
        )
        self._start_ts: float = 0.0
        self._active: bool = False
        self._ears: List[float] = []
        self._mars: List[float] = []
        self._brows: List[float] = []
        self._postures: List[float] = []

    def start(self) -> None:
        self._start_ts = time.monotonic()
        self._active = True
        self._ears.clear()
        self._mars.clear()
        self._brows.clear()
        self._postures.clear()

    @property
    def is_calibrating(self) -> bool:
        return self._active

    @property
    def progress(self) -> float:
        if not self._active:
            return 1.0 if self.baseline.calibrated else 0.0
        elapsed = time.monotonic() - self._start_ts
        return min(1.0, max(0.0, elapsed / self.duration_seconds))

    def add_sample(
        self,
        ear: float,
        mar: float,
        brow_ratio: float,
        posture_ratio: float,
        valid_face: bool = True,
        valid_pose: bool = True,
    ) -> bool:
        """Menambahkan sampel selama kalibrasi; mengembalikan True tepat saat kalibrasi selesai."""
        if not self._active:
            return False

        if valid_face and 0.18 <= ear <= 0.45:
            self._ears.append(ear)
            self._mars.append(mar)
            self._brows.append(brow_ratio)
        if valid_pose and 0.25 <= posture_ratio <= 0.95:
            self._postures.append(posture_ratio)

        if (time.monotonic() - self._start_ts) >= self.duration_seconds:
            self._finalize()
            return True
        return False

    def _finalize(self) -> None:
        self._active = False
        # Median ignores transient blinks or brief posture shifts during the calibration window.
        ear_val = float(np.median(self._ears)) if len(self._ears) >= 5 else self.default_baseline.ear
        mar_val = float(np.median(self._mars)) if len(self._mars) >= 5 else self.default_baseline.mar
        brow_val = (
            float(np.median(self._brows))
            if len(self._brows) >= 5
            else self.default_baseline.brow_ratio
        )
        posture_val = (
            float(np.median(self._postures))
            if len(self._postures) >= 5
            else self.default_baseline.posture_ratio
        )
        self.baseline = UserBaseline(
            ear=round(ear_val, 4),
            mar=round(mar_val, 4),
            brow_ratio=round(brow_val, 4),
            posture_ratio=round(posture_val, 4),
            calibrated=True,
        )
