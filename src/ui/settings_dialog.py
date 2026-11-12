"""Settings dialog: Buddy size/position/monitor, performance mode, work-break interval,
data retention, recalibration, delete all data. Saved to data/settings.yaml (D-024)."""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Signal
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from src.ui import theme


class SettingsDialog(QDialog):
    """Emits saved(changes) with a nested dict of changed config values."""

    saved = Signal(dict)
    recalibrate = Signal()
    delete_all = Signal()

    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()
        self.cfg = cfg
        self.setWindowTitle("Pengaturan Stress Buddy")
        self.setStyleSheet(theme.STYLESHEET)
        lay = QVBoxLayout(self)
        form = QFormLayout()
        b = cfg["buddy"]
        self.scale = QDoubleSpinBox()
        self.scale.setRange(0.8, 1.2)  # 128..192 px with base 160
        self.scale.setSingleStep(0.05)
        self.scale.setValue(float(b["scale"]))
        form.addRow("Ukuran kucing", self.scale)
        self.position = QComboBox()
        self.position.addItems(["right", "center"])
        self.position.setCurrentText(str(b.get("position", "right")))
        form.addRow("Posisi (kanan / tengah bawah)", self.position)
        self.monitor = QComboBox()
        for i, s in enumerate(QGuiApplication.screens()):
            self.monitor.addItem(f"{i}: {s.name()}", i)
        self.monitor.setCurrentIndex(min(int(b.get("monitor", 0)), max(0, self.monitor.count() - 1)))
        form.addRow("Monitor", self.monitor)
        self.overlays = QCheckBox("Efek animasi (zZz, keringat, konfeti)")
        self.overlays.setChecked(bool(b["overlays_enabled"]))
        form.addRow("", self.overlays)
        self.perf = QCheckBox("Mode hemat (pose tiap 3 frame, deteksi tiap 2 frame)")
        self.perf.setChecked(bool(cfg["inference"]["performance_mode"]))
        form.addRow("", self.perf)
        self.work_break = QSpinBox()
        self.work_break.setRange(20, 120)
        self.work_break.setValue(int(cfg["habit"]["work_break_min"]))
        self.work_break.setSuffix(" menit")
        form.addRow("Ingatkan istirahat setelah", self.work_break)
        self.retention = QSpinBox()
        self.retention.setRange(7, 365)
        self.retention.setValue(int(cfg["storage"]["retention_days"]))
        self.retention.setSuffix(" hari")
        form.addRow("Simpan riwayat selama", self.retention)
        lay.addLayout(form)
        note = QLabel("Ukuran, posisi, dan monitor berlaku setelah aplikasi dibuka ulang.")
        note.setObjectName("muted")
        lay.addWidget(note)

        row = QHBoxLayout()
        recal = QPushButton("Kalibrasi ulang")
        recal.setObjectName("secondary")
        recal.clicked.connect(self._recalibrate)
        delete = QPushButton("Hapus semua data")
        delete.setObjectName("secondary")
        delete.clicked.connect(self._confirm_delete)
        save = QPushButton("Simpan")
        save.clicked.connect(self._save)
        row.addWidget(recal)
        row.addWidget(delete)
        row.addStretch(1)
        row.addWidget(save)
        lay.addLayout(row)

    def changes(self) -> dict[str, Any]:
        """Nested dict of the form values."""
        return {
            "buddy": {"scale": round(self.scale.value(), 2), "position": self.position.currentText(),
                      "monitor": int(self.monitor.currentData() or 0), "overlays_enabled": self.overlays.isChecked()},
            "inference": {"performance_mode": self.perf.isChecked()},
            "habit": {"work_break_min": self.work_break.value()},
            "storage": {"retention_days": self.retention.value()},
        }

    def _save(self) -> None:
        self.saved.emit(self.changes())
        self.accept()

    def _recalibrate(self) -> None:
        self.recalibrate.emit()
        self.accept()

    def _confirm_delete(self) -> None:
        ok = QMessageBox.question(
            self, "Hapus semua data",
            "Semua riwayat skor, event, saran, dan kalibrasi akan dihapus permanen. Lanjutkan?",
        )
        if ok == QMessageBox.StandardButton.Yes:
            self.delete_all.emit()
