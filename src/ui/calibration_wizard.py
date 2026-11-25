"""First-run calibration wizard (spec 4.6): ~20 s, first sit upright with eyes open, then close
the eyes briefly. Features arrive from the pipeline via add_features() (main thread)."""

from __future__ import annotations

import time
from typing import Any

from PySide6.QtCore import QTimer, Signal
from PySide6.QtWidgets import QDialog, QFrame, QHBoxLayout, QLabel, QProgressBar, QPushButton, QVBoxLayout

from src.analysis.calibration import Calibration, CalibrationSession
from src.core.events import FrameFeatures
from src.ui import theme

INSTRUCTIONS = {
    "open": "👀  Duduk tegak seperti biasa bekerja dan lihat ke layar dengan mata terbuka.",
    "closed": "😌  Sekarang pejamkan mata sebentar sampai bar penuh.",
    "done": "✅  Kalibrasi selesai!",
}
UI_TICK_MS = 100


class CalibrationWizard(QDialog):
    """Emits finished_calibration(Calibration | None)."""

    finished_calibration = Signal(object)

    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()
        self.cfg = cfg
        self.setWindowTitle("Kalibrasi Stress Buddy")
        self.setStyleSheet(theme.STYLESHEET)
        self.setMinimumWidth(460)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(20, 18, 20, 18)
        lay.setSpacing(12)

        title = QLabel("🎯  Kalibrasi Singkat")
        title.setObjectName("title")
        lay.addWidget(title)

        intro = QLabel(
            f"Hanya butuh {int(cfg['calibration']['total_sec'])} detik agar deteksi mata, postur, dan jarak layar "
            "sesuai denganmu. Tidak ada foto atau video yang disimpan."
        )
        intro.setWordWrap(True)
        intro.setObjectName("muted")
        lay.addWidget(intro)

        card = QFrame()
        card.setObjectName("hero")
        card_lay = QVBoxLayout(card)
        card_lay.setContentsMargins(16, 14, 16, 14)
        card_lay.setSpacing(10)
        self.instruction = QLabel("Klik Mulai saat kamu sudah duduk nyaman di depan layar.")
        self.instruction.setObjectName("heroText")
        self.instruction.setWordWrap(True)
        card_lay.addWidget(self.instruction)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setFixedHeight(14)
        card_lay.addWidget(self.progress)

        self.face_label = QLabel("Menunggu dimulai...")
        self.face_label.setObjectName("muted")
        card_lay.addWidget(self.face_label)
        lay.addWidget(card)

        row = QHBoxLayout()
        self.skip_btn = QPushButton("Lewati")
        self.skip_btn.setObjectName("secondary")
        self.skip_btn.clicked.connect(self._skip)
        self.start_btn = QPushButton("Mulai Kalibrasi")
        self.start_btn.clicked.connect(self.start)
        row.addWidget(self.skip_btn)
        row.addStretch(1)
        row.addWidget(self.start_btn)
        lay.addLayout(row)

        self.session: CalibrationSession | None = None
        self._faces = 0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._update)

    def start(self) -> None:
        """Begin collecting."""
        self.session = CalibrationSession(self.cfg, time.time())
        self._faces = 0
        self.start_btn.setEnabled(False)
        self._timer.start(UI_TICK_MS)

    def add_features(self, f: FrameFeatures) -> None:
        """Feed one frame (no-op unless running)."""
        if self.session is not None and self.session.phase(f.ts) != "done":
            self.session.add(f)
            self._faces += int(f.ear_mean is not None)

    def _update(self) -> None:
        if self.session is None:
            return
        now = time.time()
        phase = self.session.phase(now)
        self.instruction.setText(INSTRUCTIONS[phase])
        self.progress.setValue(int(self.session.progress(now) * 100))
        self.face_label.setText(f"Frame wajah terbaca: {self._faces}")
        if phase == "done":
            self._timer.stop()
            result = self.session.result()
            self.session = None
            if result is None:
                self.instruction.setText("⚠️ Wajah kurang terbaca atau mata belum terpejam. Coba lagi dengan cahaya cukup.")
                self.start_btn.setText("Coba lagi")
                self.start_btn.setEnabled(True)
                return
            self.finished_calibration.emit(result)
            self.accept()

    def _skip(self) -> None:
        self._timer.stop()
        self.finished_calibration.emit(None)
        self.reject()

    @staticmethod
    def describe(cal: Calibration) -> str:
        """Human-readable summary."""
        return f"EAR terbuka {cal.open_ear:.3f}, tertutup {cal.closed_ear:.3f}, ambang {cal.ear_threshold:.3f}"
