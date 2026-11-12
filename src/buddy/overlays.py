"""Code-drawn overlays (Lampiran A.5): zzz, sweat, exclaim, sparkle, confetti, question.

Drawn with QPainter on top of the sprite, never baked into the PNGs. `rect` is the sprite
rectangle; the head is assumed in its upper half. `t` is the animation time 0..1 and
`elapsed_ms` the absolute time, so looping effects keep moving across animation cycles.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen, QPolygonF

NAVY = QColor(40, 50, 110)
BLUE = QColor(70, 150, 235)
RED = QColor(220, 50, 50)
YELLOW = QColor(250, 205, 60)
CONFETTI_COLORS = (QColor(240, 90, 120), QColor(80, 180, 240), QColor(250, 205, 60), QColor(110, 200, 120), QColor(170, 110, 230))
CONFETTI_COUNT = 30
CONFETTI_SEED = 42


def _text(p: QPainter, text: str, center: QPointF, size: float, color: QColor, opacity: float = 1.0, rotate: float = 0.0) -> None:
    p.save()
    p.setOpacity(p.opacity() * opacity)
    font = QFont()
    font.setBold(True)
    font.setPixelSize(max(6, int(size)))
    p.setFont(font)
    p.translate(center)
    p.rotate(rotate)
    p.setPen(QPen(Qt.GlobalColor.white, max(1.0, size / 10)))
    path = QPainterPath()
    path.addText(-size * 0.3, size * 0.35, font, text)
    p.strokePath(path, QPen(Qt.GlobalColor.white, max(2.0, size / 6)))
    p.fillPath(path, color)
    p.restore()


def zzz(p: QPainter, rect: QRectF, t: float, elapsed_ms: float) -> None:
    """Three 'Z's rising and fading in turn, growing, top-right of the head."""
    cycle = (elapsed_ms / 2400.0) % 1.0
    for i in range(3):
        u = (cycle + i / 3.0) % 1.0
        size = rect.height() * (0.08 + 0.04 * i)
        x = rect.left() + rect.width() * (0.66 + 0.08 * u)
        y = rect.top() + rect.height() * (0.38 - 0.30 * u)
        _text(p, "Z", QPointF(x, y), size, NAVY, opacity=1.0 - u)


def sweat(p: QPainter, rect: QRectF, t: float, elapsed_ms: float) -> None:
    """One blue drop at the temple falling slowly, then reappearing."""
    u = (elapsed_ms / 1800.0) % 1.0
    r = rect.height() * 0.035
    cx = rect.left() + rect.width() * 0.72
    cy = rect.top() + rect.height() * (0.30 + 0.10 * u)
    p.save()
    p.setOpacity(p.opacity() * (1.0 - 0.6 * u))
    path = QPainterPath()
    path.moveTo(cx, cy - 2.2 * r)
    path.quadTo(cx + 1.2 * r, cy - 0.2 * r, cx, cy + r)
    path.quadTo(cx - 1.2 * r, cy - 0.2 * r, cx, cy - 2.2 * r)
    p.fillPath(path, BLUE)
    p.restore()


def exclaim(p: QPainter, rect: QRectF, t: float, elapsed_ms: float) -> None:
    """Red '!' above the head with a pop."""
    u = min(1.0, t * 6.0)
    scale = 0.6 + 0.6 * u if u < 0.7 else 1.2 - 0.2 * (u - 0.7) / 0.3
    _text(p, "!", QPointF(rect.center().x() + rect.width() * 0.05, rect.top() + rect.height() * 0.06),
          rect.height() * 0.2 * scale, RED)


def _star(p: QPainter, c: QPointF, r: float, color: QColor) -> None:
    pts = []
    for k in range(8):
        rr = r if k % 2 == 0 else r * 0.4
        a = math.pi / 4 * k - math.pi / 2
        pts.append(QPointF(c.x() + rr * math.cos(a), c.y() + rr * math.sin(a)))
    p.setBrush(color)
    p.setPen(Qt.PenStyle.NoPen)
    p.drawPolygon(QPolygonF(pts))


def sparkle(p: QPainter, rect: QRectF, t: float, elapsed_ms: float) -> None:
    """Four small yellow stars blinking around the head."""
    spots = ((0.18, 0.15), (0.82, 0.12), (0.12, 0.45), (0.88, 0.40))
    p.save()
    for i, (fx, fy) in enumerate(spots):
        a = 0.5 + 0.5 * math.sin(elapsed_ms / 150.0 + i * 1.7)
        p.setOpacity(a)
        _star(p, QPointF(rect.left() + rect.width() * fx, rect.top() + rect.height() * fy), rect.height() * 0.05, YELLOW)
    p.restore()


def confetti(p: QPainter, rect: QRectF, t: float, elapsed_ms: float) -> None:
    """~30 colored particles falling with gravity for about 2 s (deterministic)."""
    rng = random.Random(CONFETTI_SEED)
    sec = min(elapsed_ms, 2000.0) / 1000.0
    p.save()
    p.setPen(Qt.PenStyle.NoPen)
    for i in range(CONFETTI_COUNT):
        x0 = rect.left() + rect.width() * rng.uniform(0.05, 0.95)
        vx = rect.width() * rng.uniform(-0.15, 0.15)
        vy = -rect.height() * rng.uniform(0.2, 0.6)
        g = rect.height() * 1.4
        x = x0 + vx * sec
        y = rect.top() + rect.height() * 0.15 + vy * sec + 0.5 * g * sec * sec
        p.setOpacity(max(0.0, 1.0 - sec / 2.0))
        p.setBrush(CONFETTI_COLORS[i % len(CONFETTI_COLORS)])
        s = rect.height() * 0.025
        p.drawRect(QRectF(x, y, s, s * 0.6))
    p.restore()


def question(p: QPainter, rect: QRectF, t: float, elapsed_ms: float) -> None:
    """Blue '?' above the head, swaying lightly."""
    _text(p, "?", QPointF(rect.center().x() + rect.width() * 0.18, rect.top() + rect.height() * 0.08),
          rect.height() * 0.2, BLUE, rotate=8.0 * math.sin(elapsed_ms / 250.0))


OVERLAYS: dict[str, Callable[[QPainter, QRectF, float, float], None]] = {
    "zzz": zzz,
    "sweat": sweat,
    "exclaim": exclaim,
    "sparkle": sparkle,
    "confetti": confetti,
    "question": question,
}


def draw_overlay(name: str | None, p: QPainter, rect: QRectF, t: float, elapsed_ms: float) -> None:
    """Draw a named overlay (no-op for None/unknown)."""
    fn = OVERLAYS.get(name or "")
    if fn is not None:
        fn(p, rect, t, elapsed_ms)
