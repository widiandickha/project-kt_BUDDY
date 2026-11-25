"""System tray icon (spec 5.1 / FASE 7): camera indicator, pause/resume, dashboard, preview,
settings, recalibration, quit. The icon shows a red dot while the camera is active."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject, QRectF, Qt, Signal
from PySide6.QtGui import QAction, QColor, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import QMenu, QSystemTrayIcon

from src.buddy.procedural import make_pixmap
from src.ui import theme

ICON_PX = 64
DOT_COLOR = QColor(220, 38, 38)
IDLE_SPRITE = Path(__file__).resolve().parents[1] / "buddy" / "assets" / "cat" / "idle.png"


def tray_icon(camera_active: bool) -> QIcon:
    """Cat icon (uses the real sticker sprite if present), with a red dot when the camera is on."""
    pm = QPixmap(str(IDLE_SPRITE))
    if pm.isNull():
        pm = QPixmap(make_pixmap("idle", ICON_PX))
    else:
        pm = pm.scaled(ICON_PX, ICON_PX, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
    if camera_active:
        p = QPainter(pm)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(QColor("white"))
        p.setBrush(DOT_COLOR)
        p.drawEllipse(QRectF(ICON_PX * 0.66, ICON_PX * 0.04, ICON_PX * 0.28, ICON_PX * 0.28))
        p.end()
    return QIcon(pm)


class Tray(QObject):
    """Wraps QSystemTrayIcon; works (as a no-op) where no tray is available."""

    toggle_pause = Signal()
    open_dashboard = Signal()
    open_preview = Signal()
    open_settings = Signal()
    recalibrate = Signal()
    quit_requested = Signal()

    def __init__(self, show_indicator: bool = True) -> None:
        super().__init__()
        self.show_indicator = show_indicator
        self.available = QSystemTrayIcon.isSystemTrayAvailable()
        self.icon = QSystemTrayIcon()
        self.menu = QMenu()
        self.menu.setStyleSheet(theme.MENU_STYLESHEET)
        self.status_action = QAction("🐾  Kamera: memulai...")
        self.status_action.setEnabled(False)
        self.pause_action = QAction("⏸️  Jeda Kamera")
        self.pause_action.triggered.connect(self.toggle_pause.emit)
        self.menu.addAction(self.status_action)
        self.menu.addSeparator()
        for text, sig in (
            ("📊  Buka Dashboard", self.open_dashboard),
            ("📷  Lihat Kamera & Deteksi", self.open_preview),
        ):
            act = QAction(text, self.menu)
            act.triggered.connect(sig.emit)
            self.menu.addAction(act)
        self.menu.addSeparator()
        self.menu.addAction(self.pause_action)
        for text, sig in (
            ("🎯  Kalibrasi Ulang", self.recalibrate),
            ("⚙️  Pengaturan", self.open_settings),
        ):
            act = QAction(text, self.menu)
            act.triggered.connect(sig.emit)
            self.menu.addAction(act)
        self.menu.addSeparator()
        quit_act = QAction("🚪  Keluar", self.menu)
        quit_act.triggered.connect(self.quit_requested.emit)
        self.menu.addAction(quit_act)
        self.icon.setContextMenu(self.menu)
        self.icon.activated.connect(self._activated)
        self.set_camera_state("starting", "")
        if self.available:
            self.icon.show()

    def _activated(self, reason: QSystemTrayIcon.ActivationReason) -> None:
        if reason == QSystemTrayIcon.ActivationReason.DoubleClick:
            self.open_dashboard.emit()

    def set_camera_state(self, status: str, message: str = "") -> None:
        """Update icon, tooltip and menu text from the camera status."""
        texts = {
            "active": "🟢  Kamera aktif · lokal, tidak disimpan",
            "paused": "⏸️  Kamera dijeda (perangkat dilepas)",
            "unavailable": f"⚠️  Kamera tidak tersedia · {message}".strip(" ·"),
            "stopped": "⏹️  Kamera mati",
            "starting": "🐾  Kamera: memulai...",
            "fullscreen": "🌙  Mode jangan ganggu (fullscreen)",
        }
        text = texts.get(status, status)
        self.status_action.setText(text)
        self.pause_action.setText("▶️  Lanjutkan Kamera" if status == "paused" else "⏸️  Jeda Kamera")
        self.icon.setToolTip(f"Stress Buddy\n{text}")
        self.icon.setIcon(tray_icon(self.show_indicator and status == "active"))

    def notify(self, title: str, message: str) -> None:
        """System notification (if supported)."""
        if self.available and self.icon.supportsMessages():
            self.icon.showMessage(title, message, QSystemTrayIcon.MessageIcon.Information, 4000)

    def hide(self) -> None:
        """Remove the icon."""
        self.icon.hide()
