"""Procedural cat drawn with QPainter: fallback when the 12 PNG sprites are missing or invalid.

Expression parameters mirror scripts/generate_cat_sprites.py (eyes, mouth, brow, body).
Includes tabby stripes, red collar, and golden bell so fallback art looks just as polished.
"""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPainterPath, QPen, QPixmap, QPolygonF

ORANGE = QColor("#F59E3F")
ORANGE_DARK = QColor("#D87724")
CREAM = QColor("#FFF7EA")
STRIPE = QColor("#B8551E")
OUTLINE = QColor("#4A250D")
PINK = QColor("#F78697")
COLLAR = QColor("#D9383A")
BELL = QColor("#FACC15")
IRIS = QColor("#5AB84B")

EXPRESSION_PARAMS: dict[str, dict[str, str]] = {
    "idle": dict(eyes="open", mouth="smile", brow="none", body="sit"),
    "happy": dict(eyes="happy", mouth="open", brow="none", body="sit"),
    "sleepy": dict(eyes="half", mouth="yawn", brow="none", body="sit"),
    "worried": dict(eyes="open", mouth="flat", brow="worried", body="sit"),
    "alert": dict(eyes="wide", mouth="open", brow="none", body="stand"),
    "concerned": dict(eyes="open", mouth="flat", brow="worried", body="sit"),
    "stretch": dict(eyes="closed", mouth="open", brow="none", body="stretch"),
    "curious": dict(eyes="wide", mouth="smile", brow="none", body="sit"),
    "eye_care": dict(eyes="closed", mouth="smile", brow="none", body="sit"),
    "sleeping": dict(eyes="closed", mouth="none", brow="none", body="curl"),
    "wake_greet": dict(eyes="happy", mouth="open", brow="none", body="stand"),
    "celebrate": dict(eyes="happy", mouth="open", brow="none", body="stand"),
}


def _layout(body: str, s: float) -> tuple[QRectF, QPointF, float]:
    cx, base = s / 2.0, s * 0.96
    if body == "curl":
        return QRectF(cx - s * .34, base - s * .34, s * .60, s * .34), QPointF(cx - s * .14, base - s * .28), s * .19
    if body == "stand":
        return QRectF(cx - s * .22, base - s * .46, s * .44, s * .46), QPointF(cx, base - s * .56), s * .25
    if body == "stretch":
        return QRectF(cx - s * .34, base - s * .30, s * .64, s * .30), QPointF(cx - s * .22, base - s * .38), s * .22
    return QRectF(cx - s * .27, base - s * .44, s * .54, s * .44), QPointF(cx, base - s * .50), s * .26


def draw_cat(p: QPainter, size: int, expression: str) -> None:
    """Draw the cat for `expression` into a size x size area at the painter origin."""
    spec = EXPRESSION_PARAMS.get(expression, EXPRESSION_PARAMS["idle"])
    s = float(size)
    body, hc, r = _layout(spec["body"], s)
    w = max(2.0, s / 100.0)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)

    # Tail
    p.setPen(QPen(ORANGE, s * 0.065, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
    p.drawArc(QRectF(body.right() - s * .12, body.bottom() - s * .42, s * .32, s * .32), 270 * 16, 190 * 16)

    # Body & chest
    pen = QPen(OUTLINE, w, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
    p.setPen(pen)
    p.setBrush(ORANGE)
    p.drawEllipse(body)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(CREAM)
    p.drawEllipse(QRectF(body.center().x() - body.width() * 0.24, body.top() + body.height() * 0.12,
                         body.width() * 0.48, body.height() * 0.50))

    # Red collar & golden bell
    p.setBrush(COLLAR)
    p.drawRoundedRect(QRectF(hc.x() - r * 0.62, hc.y() + r * 0.68, r * 1.24, r * 0.22), r * 0.1, r * 0.1)
    p.setPen(QPen(OUTLINE, w * 0.6))
    p.setBrush(BELL)
    p.drawEllipse(QPointF(hc.x(), hc.y() + r * 0.85), r * 0.16, r * 0.16)

    # Ears
    for sx in (-1, 1):
        ex = hc.x() + sx * r * .62
        p.setPen(pen)
        p.setBrush(ORANGE)
        p.drawPolygon(QPolygonF([QPointF(ex - sx * r * .35, hc.y() - r * .55), QPointF(ex + sx * r * .05, hc.y() - r * 1.28),
                                 QPointF(ex + sx * r * .45, hc.y() - r * .55)]))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(PINK)
        p.drawPolygon(QPolygonF([QPointF(ex - sx * r * .15, hc.y() - r * .62), QPointF(ex + sx * r * .05, hc.y() - r * 1.02),
                                 QPointF(ex + sx * r * .28, hc.y() - r * .62)]))

    # Head
    p.setPen(pen)
    p.setBrush(ORANGE)
    p.drawEllipse(QRectF(hc.x() - r, hc.y() - r * .9, 2 * r, 1.8 * r))

    # Forehead tabby stripes
    p.setPen(QPen(STRIPE, w * 1.1, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
    for dx in (-r * 0.22, 0.0, r * 0.22):
        p.drawLine(QPointF(hc.x() + dx, hc.y() - r * 0.80), QPointF(hc.x() + dx * 0.7, hc.y() - r * 0.52))

    # Blush cheeks
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QColor(255, 110, 125, 110))
    for sx in (-1, 1):
        p.drawEllipse(QRectF(hc.x() + sx * r * .68 - r * .18, hc.y() + r * .08, r * .36, r * .22))

    # Eyes
    for sx in (-1, 1):
        x, y, er = hc.x() + sx * r * .44, hc.y() - r * .04, r * .17
        style = spec["eyes"]
        if style in ("open", "half"):
            p.setPen(QPen(OUTLINE, w * 0.7))
            p.setBrush(IRIS)
            p.drawEllipse(QPointF(x, y), er, er * 1.1)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(OUTLINE)
            p.drawEllipse(QPointF(x, y), er * 0.55, er * 0.85)
            p.setBrush(Qt.GlobalColor.white)
            p.drawEllipse(QPointF(x - er * 0.28, y - er * 0.32), er * 0.32, er * 0.32)
            if style == "half":
                p.setBrush(ORANGE_DARK)
                p.drawRect(QRectF(x - er * 1.2, y - er * 1.2, er * 2.4, er * 1.1))
                p.setPen(pen)
                p.drawLine(QPointF(x - er * 1.2, y - er * .1), QPointF(x + er * 1.2, y - er * .1))
        elif style == "wide":
            p.setPen(pen)
            p.setBrush(IRIS)
            p.drawEllipse(QPointF(x, y), er * 1.35, er * 1.35)
            p.setBrush(OUTLINE)
            p.drawEllipse(QPointF(x, y), er * .75, er * .75)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(Qt.GlobalColor.white)
            p.drawEllipse(QPointF(x - er * 0.35, y - er * 0.35), er * 0.38, er * 0.38)
        else:  # closed / happy arcs
            p.setPen(pen)
            p.setBrush(Qt.BrushStyle.NoBrush)
            arc = QRectF(x - er * 1.2, y - er * 0.8, er * 2.4, er * 2.0)
            p.drawArc(arc, (200 if style == "closed" else 20) * 16, 140 * 16)

    if spec["brow"] == "worried":
        p.setPen(pen)
        for sx in (-1, 1):
            x, y = hc.x() + sx * r * .45, hc.y() - r * .32
            p.drawLine(QPointF(x - sx * r * .18, y - r * .10), QPointF(x + sx * r * .18, y + r * .06))

    mx, my = hc.x(), hc.y() + r * .40
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(PINK)
    p.drawEllipse(QRectF(mx - r * .07, my - r * .24, r * .14, r * .10))  # nose
    mouth = spec["mouth"]
    if mouth in ("open", "yawn"):
        k = 1.35 if mouth == "yawn" else 1.0
        p.setBrush(OUTLINE)
        p.drawEllipse(QRectF(mx - r * .14 * k, my - r * .02, r * .28 * k, r * .30 * k))
        p.setBrush(PINK)
        p.drawEllipse(QRectF(mx - r * .09 * k, my + r * .10 * k, r * .18 * k, r * .15 * k))
    elif mouth == "smile":
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawArc(QRectF(mx - r * .22, my - r * .15, r * .44, r * .37), 200 * 16, 140 * 16)
    elif mouth == "flat":
        p.setPen(pen)
        p.drawLine(QPointF(mx - r * .16, my + r * .05), QPointF(mx + r * .16, my + r * .05))


def make_pixmap(expression: str, size: int) -> QPixmap:
    """Transparent square pixmap with the procedural cat."""
    img = QImage(size, size, QImage.Format.Format_ARGB32_Premultiplied)
    img.fill(Qt.GlobalColor.transparent)
    p = QPainter(img)
    draw_cat(p, size, expression)
    p.end()
    return QPixmap.fromImage(img)


def outline_path(size: int) -> QPainterPath:
    """Rough body outline (unused by the renderer; handy for hit tests)."""
    path = QPainterPath()
    path.addEllipse(QRectF(size * .2, size * .3, size * .6, size * .66))
    return path
