from typing import Optional
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QGuiApplication, QKeyEvent
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from src.buddy.messages import BuddyAdvice
from src.ui.cat_canvas import CatBuddyWidget
from src.ui.theme import build_stylesheet


class BuddyOverlayWindow(QWidget):
    """Jendela kecil di sudut layar yang menampilkan saran Virtual Buddy dan hilang otomatis."""

    action_recorded = Signal(int, str)
    snooze_requested = Signal(int)

    def __init__(self, theme_name: str = "light", parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowFlags(
            Qt.WindowStaysOnTopHint
            | Qt.FramelessWindowHint
            | Qt.Tool
        )
        self.setAttribute(Qt.WA_ShowWithoutActivating, False)
        self.setFixedWidth(410)

        self._current_log_id: int = -1
        self._remaining_sec: int = 8
        self._snooze_minutes: int = 15

        self._timer = QTimer(self)
        self._timer.setInterval(1000)
        self._timer.timeout.connect(self._on_tick)

        self._build_ui()
        self.apply_theme(theme_name)

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        self.card = QFrame(self)
        self.card.setObjectName("SurfacePanel")
        card_layout = QVBoxLayout(self.card)
        card_layout.setContentsMargins(16, 14, 16, 14)
        card_layout.setSpacing(10)

        top_row = QHBoxLayout()
        top_row.setSpacing(14)

        self.cat_widget = CatBuddyWidget(expression="istirahat", size=92, parent=self.card)
        top_row.addWidget(self.cat_widget, alignment=Qt.AlignTop)

        text_col = QVBoxLayout()
        text_col.setSpacing(4)

        self.lbl_badge = QLabel("Kucing Teman Kerja", self.card)
        self.lbl_badge.setObjectName("MutedText")
        text_col.addWidget(self.lbl_badge)

        self.lbl_title = QLabel("Waktunya Istirahat Sejenak", self.card)
        self.lbl_title.setObjectName("HeadingMedium")
        self.lbl_title.setWordWrap(True)
        text_col.addWidget(self.lbl_title)

        self.lbl_message = QLabel("", self.card)
        self.lbl_message.setWordWrap(True)
        text_col.addWidget(self.lbl_message)

        self.lbl_reason = QLabel("", self.card)
        self.lbl_reason.setObjectName("MutedText")
        self.lbl_reason.setWordWrap(True)
        text_col.addWidget(self.lbl_reason)

        top_row.addLayout(text_col, stretch=1)
        card_layout.addLayout(top_row)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)

        self.lbl_countdown = QLabel("Menutup otomatis dalam 8 dtk", self.card)
        self.lbl_countdown.setObjectName("MutedText")
        btn_row.addWidget(self.lbl_countdown, stretch=1)

        self.btn_snooze = QPushButton("Tunda 15 Menit", self.card)
        self.btn_snooze.clicked.connect(self._on_snooze_clicked)
        btn_row.addWidget(self.btn_snooze)

        self.btn_ack = QPushButton("Mengerti (Esc)", self.card)
        self.btn_ack.setObjectName("PrimaryButton")
        self.btn_ack.clicked.connect(self._on_ack_clicked)
        btn_row.addWidget(self.btn_ack)

        card_layout.addLayout(btn_row)
        outer.addWidget(self.card)

    def apply_theme(self, theme_name: str) -> None:
        self.setStyleSheet(build_stylesheet(theme_name))

    def show_advice(
        self,
        advice: BuddyAdvice,
        duration_seconds: int = 8,
        snooze_minutes: int = 15,
        buddy_log_id: int = -1,
    ) -> None:
        self._current_log_id = buddy_log_id
        self._remaining_sec = max(3, int(duration_seconds))
        self._snooze_minutes = max(1, int(snooze_minutes))

        self.cat_widget.set_expression(advice.expression)
        self.lbl_badge.setText(f"Virtual Buddy | Ekspresi: {advice.expression.capitalize()}")
        self.lbl_title.setText(advice.title)
        self.lbl_message.setText(advice.message)
        self.lbl_reason.setText(f"Pemicu: {advice.trigger_reason}")
        self.btn_snooze.setText(f"Tunda {self._snooze_minutes} Menit")
        self.lbl_countdown.setText(f"Menutup otomatis dalam {self._remaining_sec} dtk")

        self.adjustSize()
        self._position_bottom_right()
        self.show()
        self.raise_()
        self._timer.start()

    def _position_bottom_right(self) -> None:
        screen = QGuiApplication.primaryScreen()
        if screen is None:
            return
        avail = screen.availableGeometry()
        x = avail.right() - self.width() - 24
        y = avail.bottom() - self.height() - 24
        self.move(max(avail.left() + 12, x), max(avail.top() + 12, y))

    def _on_tick(self) -> None:
        self._remaining_sec -= 1
        if self._remaining_sec <= 0:
            self._timer.stop()
            if self._current_log_id > 0:
                self.action_recorded.emit(self._current_log_id, "otomatis_selesai")
            self.hide()
        else:
            self.lbl_countdown.setText(f"Menutup otomatis dalam {self._remaining_sec} dtk")

    def _on_ack_clicked(self) -> None:
        self._timer.stop()
        if self._current_log_id > 0:
            self.action_recorded.emit(self._current_log_id, "diterima")
        self.hide()

    def _on_snooze_clicked(self) -> None:
        self._timer.stop()
        if self._current_log_id > 0:
            self.action_recorded.emit(self._current_log_id, f"snooze_{self._snooze_minutes}m")
        self.snooze_requested.emit(self._snooze_minutes)
        self.hide()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key_Escape:
            self._on_ack_clicked()
            event.accept()
            return
        super().keyPressEvent(event)
