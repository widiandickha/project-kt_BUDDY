"""Procedural cat drawn with QPainter: fallback when the 12 PNG sprites are missing or invalid.

Expression parameters mirror scripts/generate_placeholder_sprites.py (eyes, mouth, brow, body).
"""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPainterPath, QPen, QPixmap, QPolygonF

ORANGE = QColor(253, 186, 100)
ORANGE_DARK = QColor(250, 128, 100)
OUTLINE = QColor(40, 50, 110)
PINK = QColor(240, 110, 150)

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
    w = max(2.0, s / 110.0)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setPen(Qt.PenStyle.NoPen)

    p.setPen(QPen(ORANGE, s * 0.06, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))  # tail
    p.drawArc(QRectF(body.right() - s * .12, body.bottom() - s * .42, s * .32, s * .32), 270 * 16, 190 * 16)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(ORANGE)
    p.drawEllipse(body)
    for sx in (-1, 1):  # ears
        ex = hc.x() + sx * r * .62
        p.setBrush(ORANGE)
        p.drawPolygon(QPolygonF([QPointF(ex - sx * r * .35, hc.y() - r * .55), QPointF(ex + sx * r * .05, hc.y() - r * 1.25),
                                 QPointF(ex + sx * r * .45, hc.y() - r * .55)]))
        p.setBrush(ORANGE_DARK)
        p.drawPolygon(QPolygonF([QPointF(ex - sx * r * .15, hc.y() - r * .62), QPointF(ex + sx * r * .05, hc.y() - r * 1.0),
                                 QPointF(ex + sx * r * .28, hc.y() - r * .62)]))
    p.setBrush(ORANGE)
    p.drawEllipse(QRectF(hc.x() - r, hc.y() - r * .9, 2 * r, 1.8 * r))
    p.setBrush(ORANGE_DARK)
    for sx in (-1, 1):  # cheeks
        p.drawEllipse(QRectF(hc.x() + sx * r * .72 - r * .16, hc.y() + r * .08, r * .32, r * .24))

    pen = QPen(OUTLINE, w, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
    for sx in (-1, 1):  # eyes
        x, y, er = hc.x() + sx * r * .45, hc.y() - r * .02, r * .13
        style = spec["eyes"]
        if style in ("open", "half"):
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(OUTLINE)
            p.drawEllipse(QPointF(x, y), er, er)
            if style == "half":
                p.setBrush(ORANGE)
                p.drawRect(QRectF(x - er * 1.2, y - er * 1.2, er * 2.4, er * 1.1))
                p.setPen(pen)
                p.drawLine(QPointF(x - er * 1.2, y - er * .1), QPointF(x + er * 1.2, y - er * .1))
        elif style == "wide":
            p.setPen(pen)
            p.setBrush(Qt.GlobalColor.white)
            p.drawEllipse(QPointF(x, y), er * 1.5, er * 1.5)
            p.setBrush(OUTLINE)
            p.drawEllipse(QPointF(x, y), er * .75, er * .75)
        else:  # closed / happy arcs
            p.setPen(pen)
            p.setBrush(Qt.BrushStyle.NoBrush)
            arc = QRectF(x - er * 1.4, y - er, er * 2.8, er * 2.4)
            p.drawArc(arc, (200 if style == "closed" else 20) * 16, 140 * 16)
    if spec["brow"] == "worried":
        p.setPen(pen)
        for sx in (-1, 1):
            x, y = hc.x() + sx * r * .45, hc.y() - r * .30
            p.drawLine(QPointF(x - sx * r * .18, y - r * .10), QPointF(x + sx * r * .18, y + r * .06))
    mx, my = hc.x(), hc.y() + r * .40
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(OUTLINE)
    p.drawEllipse(QRectF(mx - r * .05, my - r * .22, r * .1, r * .08))  # nose
    mouth = spec["mouth"]
    if mouth in ("open", "yawn"):
        k = 1.4 if mouth == "yawn" else 1.0
        p.drawEllipse(QRectF(mx - r * .14 * k, my - r * .02, r * .28 * k, r * .30 * k))
        p.setBrush(PINK)
        p.drawEllipse(QRectF(mx - r * .08 * k, my + r * .12 * k, r * .16 * k, r * .14 * k))
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
