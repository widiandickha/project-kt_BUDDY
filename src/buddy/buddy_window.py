"""The cat as a desktop widget (spec 5.1).

Frameless, always on top, Qt.Tool (no taskbar entry), never takes focus, translucent.
Sits at the bottom edge of QScreen.availableGeometry() (above the taskbar) on the configured
monitor, right or center. Clicks on transparent pixels pass through (setMask from the sprite
alpha, refreshed <= buddy.mask_hz). Left click = react, right click = menu, drag = move along
the bottom edge.
"""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import QPoint, QRect, Qt, QTimer, Signal
from PySide6.QtGui import QBitmap, QGuiApplication, QPainter, QRegion
from PySide6.QtWidgets import QMenu, QWidget

from src.buddy.sprite_player import SpriteAnimator

DRAG_THRESHOLD_PX = 4
MENU_ACTIONS: tuple[tuple[str, str], ...] = (
    ("pause", "Jeda 30 menit"),
    ("silence_today", "Diam hari ini"),
    ("dashboard", "Buka dashboard"),
    ("settings", "Pengaturan"),
    ("quit", "Keluar"),
)


class BuddyWindow(QWidget):
    """Displays the SpriteAnimator and handles mouse interaction."""

    clicked = Signal()
    menu_action = Signal(str)
    moved = Signal(int)

    def __init__(self, buddy_cfg: dict[str, Any], animator: SpriteAnimator | None = None) -> None:
        super().__init__(None, Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint
                         | Qt.WindowType.Tool | Qt.WindowType.WindowDoesNotAcceptFocus)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.cfg = buddy_cfg
        self.animator = animator or SpriteAnimator(buddy_cfg, parent=self)
        self.animator.frame_changed.connect(self.update)
        size = int(float(buddy_cfg["base_px"]) * float(buddy_cfg["scale"]))
        self.setFixedSize(size, size)
        self.margin = int(buddy_cfg["margin_px"])
        self._press: QPoint | None = None
        self._dragging = False
        self._x_offset: int | None = None
        self._mask_timer = QTimer(self)
        self._mask_timer.timeout.connect(self.update_mask)
        self._mask_timer.start(int(1000 / max(1.0, float(buddy_cfg["mask_hz"]))))

    # -- placement ------------------------------------------------------------------
    def screen_geometry(self) -> QRect:
        """Available geometry of the configured monitor (excludes taskbar/dock)."""
        screens = QGuiApplication.screens()
        idx = int(self.cfg.get("monitor", 0))
        screen = screens[idx] if 0 <= idx < len(screens) else QGuiApplication.primaryScreen()
        return screen.availableGeometry()

    def place(self) -> None:
        """Snap to the bottom edge, at the remembered x or the configured position."""
        avail = self.screen_geometry()
        if self._x_offset is None:
            if self.cfg.get("position", "right") == "center":
                x = avail.center().x() - self.width() // 2
            else:
                x = avail.right() - self.width() - self.margin
        else:
            x = avail.left() + self._x_offset
        x = min(max(avail.left(), x), avail.right() - self.width())
        self.move(x, avail.bottom() - self.height() + 1)

    def showEvent(self, event) -> None:  # noqa: N802 (Qt API)
        """Position on first show."""
        self.place()
        super().showEvent(event)

    # -- painting & mask --------------------------------------------------------------
    def paintEvent(self, event) -> None:  # noqa: N802 (Qt API)
        """Draw the current animation frame."""
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.animator.paint(p, self.rect())

    def update_mask(self) -> None:
        """Make transparent pixels click-through."""
        img = self.animator.render_image(self.width())
        mask = QBitmap.fromImage(img.createAlphaMask())
        region = QRegion(mask)
        if not region.isEmpty():
            self.setMask(region)

    # -- mouse ------------------------------------------------------------------------
    def mousePressEvent(self, event) -> None:  # noqa: N802 (Qt API)
        """Start a click or drag."""
        if event.button() == Qt.MouseButton.LeftButton:
            self._press = event.globalPosition().toPoint()
            self._dragging = False

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 (Qt API)
        """Drag horizontally along the bottom edge."""
        if self._press is None:
            return
        pos = event.globalPosition().toPoint()
        if not self._dragging and abs(pos.x() - self._press.x()) > DRAG_THRESHOLD_PX:
            self._dragging = True
        if self._dragging:
            avail = self.screen_geometry()
            x = min(max(avail.left(), pos.x() - self.width() // 2), avail.right() - self.width())
            self._x_offset = x - avail.left()
            self.move(x, avail.bottom() - self.height() + 1)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 (Qt API)
        """Finish: a short press without movement is a click."""
        if event.button() == Qt.MouseButton.LeftButton and self._press is not None:
            if self._dragging and self._x_offset is not None:
                self.moved.emit(self._x_offset)
            else:
                self.clicked.emit()
        self._press, self._dragging = None, False

    def contextMenuEvent(self, event) -> None:  # noqa: N802 (Qt API)
        """Right-click menu."""
        menu = QMenu(self)
        for key, label in MENU_ACTIONS:
            act = menu.addAction(label)
            act.triggered.connect(lambda _=False, k=key: self.menu_action.emit(k))
        menu.exec(event.globalPos())
