from typing import Optional
from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QMenu, QSystemTrayIcon, QWidget

from src.ui.cat_canvas import build_cat_icon


class BuddyTrayIcon(QObject):
    """Ikon system tray untuk menjeda kamera, menunda peringatan, atau membuka dashboard."""

    open_dashboard_requested = Signal()
    toggle_pause_requested = Signal()
    snooze_requested = Signal(int)
    preview_buddy_requested = Signal()
    quit_requested = Signal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.tray = QSystemTrayIcon(build_cat_icon(64), parent)
        self.tray.setToolTip("Kucing Teman Kerja: Kamera Aktif (Lokal)")

        self.menu = QMenu(parent)
        self.act_open = QAction("Buka Jendela Utama", self.menu)
        self.act_open.triggered.connect(self.open_dashboard_requested.emit)
        self.menu.addAction(self.act_open)

        self.act_pause = QAction("Jeda Pemantauan Kamera", self.menu)
        self.act_pause.triggered.connect(self.toggle_pause_requested.emit)
        self.menu.addAction(self.act_pause)

        self.act_snooze = QAction("Tunda Buddy 15 Menit", self.menu)
        self.act_snooze.triggered.connect(lambda: self.snooze_requested.emit(15))
        self.menu.addAction(self.act_snooze)

        self.act_preview = QAction("Tampilkan Virtual Buddy Sekarang", self.menu)
        self.act_preview.triggered.connect(self.preview_buddy_requested.emit)
        self.menu.addAction(self.act_preview)

        self.menu.addSeparator()
        self.act_quit = QAction("Keluar dari Aplikasi", self.menu)
        self.act_quit.triggered.connect(self.quit_requested.emit)
        self.menu.addAction(self.act_quit)

        self.tray.setContextMenu(self.menu)
        self.tray.activated.connect(self._on_activated)
        if QSystemTrayIcon.isSystemTrayAvailable():
            self.tray.show()

    def _on_activated(self, reason: QSystemTrayIcon.ActivationReason) -> None:
        if reason in (QSystemTrayIcon.Trigger, QSystemTrayIcon.DoubleClick):
            self.open_dashboard_requested.emit()

    def update_camera_state(self, is_paused: bool) -> None:
        if is_paused:
            self.act_pause.setText("Lanjutkan Pemantauan Kamera")
            self.tray.setToolTip("Kucing Teman Kerja: Kamera Dijeda")
        else:
            self.act_pause.setText("Jeda Pemantauan Kamera")
            self.tray.setToolTip("Kucing Teman Kerja: Kamera Aktif (Pemrosesan Lokal)")
