"""Speech bubble above the cat's head (spec 5.5).

Fades/scales in over 200 ms, hides itself after buddy.bubble_sec (response "diabaikan") or
when clicked ("ditutup"). Optional buttons: Oke, Nanti (snooze), Jangan hari ini.
Never takes keyboard focus away from the user's work.
"""

from __future__ import annotations

from PySide6.QtCore import QEasingCurve, QPoint, QPropertyAnimation, QRect, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QGuiApplication, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

ANIM_MS = 200
WIDTH = 270
TAIL = 10
RADIUS = 12
BG = QColor(255, 253, 248)
BORDER = QColor(40, 50, 110)
TEXT = "#1C1917"
BUTTONS: tuple[tuple[str, str], ...] = (("Oke", "oke"), ("Nanti", "nanti"), ("Jangan hari ini", "jangan_hari_ini"))


class SpeechBubble(QWidget):
    """Top-level, frameless, translucent bubble window."""

    responded = Signal(str, str)  # (message_key, response)

    def __init__(self, bubble_sec: float) -> None:
        super().__init__(None, Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint
                         | Qt.WindowType.Tool | Qt.WindowType.WindowDoesNotAcceptFocus)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.bubble_ms = int(bubble_sec * 1000)
        self.key = ""
        self._label = QLabel()
        self._label.setWordWrap(True)
        self._label.setStyleSheet(f"color: {TEXT}; font-size: 13px;")
        self._buttons = QHBoxLayout()
        self._buttons.setSpacing(6)
        self._button_widgets: list[QPushButton] = []
        for text, response in BUTTONS:
            b = QPushButton(text)
            b.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            b.setStyleSheet("QPushButton { background: #F0ECE1; border: 1px solid #78716C; border-radius: 6px;"
                            " padding: 2px 8px; color: #1C1917; font-size: 11px; }"
                            " QPushButton:hover { background: #E7E1D3; }")
            b.clicked.connect(lambda _=False, r=response: self._finish(r))
            self._buttons.addWidget(b)
            self._button_widgets.append(b)
        self._buttons.addStretch(1)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 10, 14, 10 + TAIL)
        lay.addWidget(self._label)
        lay.addLayout(self._buttons)
        self.setFixedWidth(WIDTH)
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(lambda: self._finish("diabaikan"))
        self._fade = QPropertyAnimation(self, b"windowOpacity", self)
        self._grow = QPropertyAnimation(self, b"geometry", self)
        for a in (self._fade, self._grow):
            a.setDuration(ANIM_MS)
            a.setEasingCurve(QEasingCurve.Type.OutCubic)

    def show_message(self, key: str, text: str, anchor: QRect, with_buttons: bool = True) -> None:
        """Show text above `anchor` (the cat window geometry)."""
        if self.isVisible() and self.key:
            self._timer.stop()
            self.responded.emit(self.key, "diabaikan")
        self.key = key
        self._label.setText(text)
        for b in self._button_widgets:
            b.setVisible(with_buttons)
        self.adjustSize()
        screen = QGuiApplication.screenAt(anchor.center()) or QGuiApplication.primaryScreen()
        avail = screen.availableGeometry()
        x = min(max(avail.left(), anchor.center().x() - self.width() // 2), avail.right() - self.width())
        y = max(avail.top(), anchor.top() - self.height() + 6)
        end = QRect(QPoint(x, y), self.size())
        start = QRect(end.center().x() - int(end.width() * 0.45), end.center().y() - int(end.height() * 0.45),
                      int(end.width() * 0.9), int(end.height() * 0.9))
        self._tail_x = anchor.center().x() - x
        self.setWindowOpacity(0.0)
        self.setGeometry(start)
        self.show()
        self._fade.setStartValue(0.0)
        self._fade.setEndValue(1.0)
        self._grow.setStartValue(start)
        self._grow.setEndValue(end)
        self._fade.start()
        self._grow.start()
        self._timer.start(self.bubble_ms)

    def _finish(self, response: str) -> None:
        self._timer.stop()
        key, self.key = self.key, ""
        self.hide()
        if key:
            self.responded.emit(key, response)

    def dismiss(self) -> None:
        """Hide without recording a response (e.g. fullscreen / pause)."""
        self._timer.stop()
        self.key = ""
        self.hide()

    def mousePressEvent(self, event) -> None:  # noqa: N802 (Qt API)
        """Clicking the bubble body closes it."""
        self._finish("ditutup")

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt API)
        """Rounded bubble with a tail pointing at the cat."""
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        body = QRectF(1, 1, self.width() - 2, self.height() - TAIL - 2)
        path = QPainterPath()
        path.addRoundedRect(body, RADIUS, RADIUS)
        tx = min(max(getattr(self, "_tail_x", self.width() / 2), RADIUS + TAIL), self.width() - RADIUS - TAIL)
        tail = QPainterPath()
        tail.moveTo(tx - TAIL, body.bottom() - 1)
        tail.lineTo(tx, self.height() - 1)
        tail.lineTo(tx + TAIL, body.bottom() - 1)
        tail.closeSubpath()
        path = path.united(tail)
        p.setPen(QPen(BORDER, 1.5))
        p.setBrush(BG)
        p.drawPath(path)
