#!/usr/bin/env python3
"""Draw the Virtual Buddy cat (orange tabby, sticker style) for all 12 expressions.

    python scripts/generate_cat_sprites.py                    # overwrite src/buddy/assets/cat/<state>.png
    python scripts/generate_cat_sprites.py --out /tmp/cat     # somewhere else
    python scripts/generate_cat_sprites.py --only idle happy  # just a few

Original vector artwork drawn with QPainter; replaces the spec placeholder sprites (Lampiran B,
scripts/generate_placeholder_sprites.py). The asset contract is unchanged: 512x512 RGBA PNG,
feet on the bottom-centre, file name = state name, so animations.yaml, the sprite player and the
code-drawn overlays (zzz, sweat, ...) keep working. Your own art can still replace any PNG.

Coordinates below are on a 100x100 canvas; the image is rendered at 2x and scaled down.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "src" / "buddy" / "assets" / "cat"

FUR, FUR_LIGHT, FUR_SHADE = "#F4A54A", "#FFCB7E", "#DB8530"
STRIPE, CREAM, OUTLINE = "#C0622A", "#FFF4E2", "#5B3216"
EAR_IN, NOSE, MOUTH_IN, TONGUE = "#F5A3A6", "#E8677A", "#7A2E36", "#F48B9B"
IRIS, IRIS_DARK, PUPIL = "#9BCB5C", "#4E8A2C", "#22180F"
LW = 1.5  # outline width (canvas units)


@dataclass(frozen=True)
class Look:
    pose: str = "sit"       # sit | stretch | curl
    eyes: str = "open"      # open | wide | closed | happy | half
    mouth: str = "smile"    # smile | open | yawn | flat | none
    brows: str = "none"     # none | worried
    ears: str = "normal"    # normal | up | back
    paws: str = "down"      # down | wave | up
    tail: str = "curl"      # curl | up | question
    blush: bool = True


LOOKS: dict[str, Look] = {
    "idle": Look(),
    "happy": Look(eyes="happy", mouth="open", tail="up"),
    "sleepy": Look(eyes="half", mouth="yawn", ears="back"),
    "worried": Look(brows="worried", mouth="flat", ears="back", blush=False),
    "alert": Look(eyes="wide", mouth="open", ears="up", tail="up", blush=False),
    "concerned": Look(brows="worried", mouth="flat"),
    "stretch": Look(pose="stretch", eyes="closed", mouth="open", tail="up"),
    "curious": Look(eyes="wide", tail="question"),
    "eye_care": Look(eyes="closed"),
    "sleeping": Look(pose="curl", eyes="closed", mouth="none", ears="back"),
    "wake_greet": Look(eyes="happy", mouth="open", paws="wave", tail="up"),
    "celebrate": Look(eyes="happy", mouth="open", paws="up", tail="up"),
}


def _qt():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtCore, QtGui

    QtGui.QGuiApplication.instance() or QtGui.QGuiApplication([])
    return QtCore, QtGui


QtCore, QtGui = _qt()
Qt, QPointF, QRectF = QtCore.Qt, QtCore.QPointF, QtCore.QRectF
QColor, QPainterPath, QPen, QBrush = QtGui.QColor, QtGui.QPainterPath, QtGui.QPen, QtGui.QBrush


# --- primitives ---------------------------------------------------------------------------

def _pen(color: str | QColor, w: float = LW) -> QPen:
    p = QPen(QColor(color), w)
    p.setCapStyle(Qt.RoundCap)
    p.setJoinStyle(Qt.RoundJoin)
    return p


def _fill(p, path, brush, outline: bool = True, w: float = LW) -> None:
    p.setPen(_pen(OUTLINE, w) if outline else Qt.NoPen)
    p.setBrush(brush if isinstance(brush, QBrush) else QBrush(QColor(brush)))
    p.drawPath(path)


def _stroke(p, path, color: str | QColor, w: float) -> None:
    p.setPen(_pen(color, w))
    p.setBrush(Qt.NoBrush)
    p.drawPath(path)


def _ellipse(cx: float, cy: float, rx: float, ry: float) -> QPainterPath:
    path = QPainterPath()
    path.addEllipse(QPointF(cx, cy), rx, ry)
    return path


def _path(*cmds) -> QPainterPath:
    """Tiny path DSL: ('M', x, y) ('L', x, y) ('Q', cx, cy, x, y) ('C', c1x, c1y, c2x, c2y, x, y) ('Z',)."""
    path = QPainterPath()
    for c in cmds:
        op, a = c[0], c[1:]
        if op == "M":
            path.moveTo(*a)
        elif op == "L":
            path.lineTo(*a)
        elif op == "Q":
            path.quadTo(*a)
        elif op == "C":
            path.cubicTo(*a)
        elif op == "Z":
            path.closeSubpath()
    return path


def _fur(y0: float, y1: float) -> QBrush:
    g = QtGui.QLinearGradient(0, y0, 0, y1)
    g.setColorAt(0.0, QColor(FUR_LIGHT))
    g.setColorAt(0.45, QColor(FUR))
    g.setColorAt(1.0, QColor(FUR_SHADE))
    return QBrush(g)


def _thick(p, path: QPainterPath, w: float, color: str = FUR) -> None:
    """Thick fur stroke with an outline (tail, raised arm)."""
    _stroke(p, path, OUTLINE, w + 2 * LW)
    _stroke(p, path, color, w)


def _bands(p, path: QPainterPath, ts: tuple[float, ...], w: float) -> None:
    """Tabby stripes across a thick stroke at path percentages `ts`."""
    for t in ts:
        pt = path.pointAtPercent(t)
        a = math.radians(path.angleAtPercent(t))
        nx, ny, h = math.sin(a), math.cos(a), w / 2 - 0.4
        _stroke(p, _path(("M", pt.x() - nx * h, pt.y() - ny * h), ("L", pt.x() + nx * h, pt.y() + ny * h)), STRIPE, 1.9)


def _shadow(p, cx: float, rx: float) -> None:
    _fill(p, _ellipse(cx, 95.8, rx, 2.8), QColor(70, 35, 10, 45), outline=False)


def _paw(p, cx: float, cy: float, rx: float = 5.6, ry: float = 3.6) -> None:
    _fill(p, _ellipse(cx, cy, rx, ry), CREAM)
    for dx in (-1.7, 1.7):
        _stroke(p, _path(("M", cx + dx, cy - 1.0), ("L", cx + dx, cy + ry - 0.6)), OUTLINE, 0.8)


# --- head ---------------------------------------------------------------------------------

def _ears(p, kind: str) -> None:
    for sx in (-1, 1):
        p.save()
        p.scale(sx, 1)
        p.translate(-13.5, -12)
        if kind == "back":
            p.rotate(-26)
        elif kind == "up":
            p.scale(1.0, 1.14)
        p.translate(13.5, 12)
        outer = _path(("M", -22.5, -5), ("L", -24.6, -28), ("Q", -25.2, -32.6, -21.2, -30.2), ("L", -4.5, -16.5), ("Z",))
        _fill(p, outer, FUR)
        inner = _path(("M", -20, -9), ("L", -21.6, -25.5), ("Q", -21.5, -27, -20, -26), ("L", -9.5, -16.8), ("Z",))
        _fill(p, inner, EAR_IN, outline=False)
        p.restore()


def _eye(p, x: float, y: float, kind: str) -> None:
    if kind == "closed":  # content, U shape
        _stroke(p, _path(("M", x - 4.4, y - 0.6), ("Q", x, y + 3.2, x + 4.4, y - 0.6)), OUTLINE, 1.6)
        return
    if kind == "happy":  # ^ shape
        _stroke(p, _path(("M", x - 4.4, y + 1.6), ("Q", x, y - 3.4, x + 4.4, y + 1.6)), OUTLINE, 1.6)
        return
    wide = kind == "wide"
    rx, ry = (5.4, 6.0) if wide else (4.4, 5.2)
    eye = _ellipse(x, y, rx, ry)
    g = QtGui.QRadialGradient(QPointF(x, y - ry * 0.4), ry * 1.3)
    g.setColorAt(0.0, QColor(IRIS))
    g.setColorAt(1.0, QColor(IRIS_DARK))
    _fill(p, eye, QBrush(g), w=1.1)
    pupil = _ellipse(x, y, 3.0, 3.2) if wide else _ellipse(x, y + 0.2, 1.8, 3.9)
    _fill(p, pupil, PUPIL, outline=False)
    hl = 1.7 if wide else 1.35
    _fill(p, _ellipse(x - rx * 0.33, y - ry * 0.38, hl, hl), "#FFFFFF", outline=False)
    _fill(p, _ellipse(x + rx * 0.32, y + ry * 0.38, hl * 0.45, hl * 0.45), "#FFFFFF", outline=False)
    if kind == "half":  # heavy eyelid over the top half
        p.save()
        p.setClipPath(eye)
        _fill(p, _path(("M", x - 7, y - 7), ("L", x + 7, y - 7), ("L", x + 7, y - 0.4),
                       ("Q", x, y - 1.6, x - 7, y - 0.4), ("Z",)), FUR_SHADE, outline=False)
        p.restore()
        _stroke(p, _path(("M", x - 4.8, y - 0.5), ("Q", x, y - 1.7, x + 4.8, y - 0.5)), OUTLINE, 1.4)


def _mouth(p, kind: str) -> None:
    philtrum_end = 9.6 if kind in ("open", "yawn") else 9.0
    _stroke(p, _path(("M", 0, 7.6), ("L", 0, philtrum_end)), OUTLINE, 1.0)
    if kind in ("smile", "none"):
        k = 1.0 if kind == "smile" else 0.7
        for sx in (-1, 1):
            _stroke(p, _path(("M", 0, 9), ("Q", 0, 9 + 1.8 * k, sx * 2.4 * k, 9 + 1.8 * k),
                             ("Q", sx * 4.2 * k, 9 + 1.7 * k, sx * 4.8 * k, 9 + 0.6 * k)), OUTLINE, 1.1)
    elif kind == "flat":
        _stroke(p, _path(("M", -3.2, 10.8), ("Q", -1.6, 10.0, 0, 10.6), ("Q", 1.6, 11.2, 3.2, 10.4)), OUTLINE, 1.1)
    elif kind in ("open", "yawn"):
        if kind == "open":
            m = _path(("M", -3.8, 9.6), ("Q", 0, 11.0, 3.8, 9.6), ("Q", 3.6, 15.2, 0, 15.5), ("Q", -3.6, 15.2, -3.8, 9.6), ("Z",))
            tongue = _ellipse(0, 14.8, 2.7, 1.9)
        else:
            m = _ellipse(0, 14.4, 4.4, 5.0)
            tongue = _ellipse(0, 18.0, 3.3, 2.3)
        _fill(p, m, MOUTH_IN, w=1.0)
        p.save()
        p.setClipPath(m)
        _fill(p, tongue, TONGUE, outline=False)
        p.restore()


def _head(p, lk: Look, cx: float, cy: float, scale: float = 1.0, tilt: float = 0.0) -> None:
    p.save()
    p.translate(cx, cy)
    p.rotate(tilt)
    p.scale(scale, scale)
    _ears(p, lk.ears)
    head = _path(("M", 0, -20), ("C", 16, -20, 25, -11, 25, 0), ("C", 25, 4, 26.5, 6, 28, 8.5), ("L", 23.6, 8.9),
                 ("L", 25.4, 12), ("C", 20, 17.5, 10, 20, 0, 20), ("C", -10, 20, -20, 17.5, -25.4, 12),
                 ("L", -23.6, 8.9), ("L", -28, 8.5), ("C", -26.5, 6, -25, 4, -25, 0), ("C", -25, -11, -16, -20, 0, -20), ("Z",))
    g = QtGui.QRadialGradient(QPointF(0, -7), 31)
    g.setColorAt(0.0, QColor(FUR_LIGHT))
    g.setColorAt(0.55, QColor(FUR))
    g.setColorAt(1.0, QColor(FUR_SHADE))
    _fill(p, head, QBrush(g))
    # tabby marks: forehead and cheeks
    for path in (_path(("M", 0, -19), ("L", 0, -13.6)), _path(("M", -6, -18.6), ("Q", -5.3, -16, -4.5, -13.8)),
                 _path(("M", 6, -18.6), ("Q", 5.3, -16, 4.5, -13.8))):
        _stroke(p, path, STRIPE, 2.0)
    for sx in (-1, 1):
        _stroke(p, _path(("M", sx * 24.4, 1.4), ("L", sx * 20.2, 2.4)), STRIPE, 1.7)
        _stroke(p, _path(("M", sx * 24.8, 4.8), ("L", sx * 20.8, 5.4)), STRIPE, 1.7)
    # muzzle, blush, eyes, brows
    for sx in (-1, 1):
        _fill(p, _ellipse(sx * 3.7, 9.2, 5.1, 3.9), CREAM, outline=False)
    if lk.blush:
        for sx in (-1, 1):
            _fill(p, _ellipse(sx * 14.8, 8.6, 3.9, 2.3), QColor(255, 120, 120, 95), outline=False)
    for sx in (-1, 1):
        _eye(p, sx * 9.6, -0.6, lk.eyes)
    if lk.brows == "worried":
        for sx in (-1, 1):
            _stroke(p, _path(("M", sx * 13.4, -7.6), ("Q", sx * 10.2, -8.6, sx * 6.6, -10.4)), OUTLINE, 1.3)
    # whiskers
    wc = QColor(OUTLINE)
    wc.setAlpha(170)
    for sx in (-1, 1):
        for ey in (5.0, 8.6, 12.2):
            _stroke(p, _path(("M", sx * 11.5, 7.6 + (ey - 8.6) * 0.25), ("Q", sx * 20, ey - 0.6, sx * 29, ey)), wc, 0.6)
    # nose and mouth
    _fill(p, _path(("M", -2.3, 5.0), ("L", 2.3, 5.0), ("Q", 2.9, 5.2, 2.3, 5.9), ("L", 0.5, 7.6),
                   ("Q", 0, 8.1, -0.5, 7.6), ("L", -2.3, 5.9), ("Q", -2.9, 5.2, -2.3, 5.0), ("Z",)), NOSE, w=0.7)
    _mouth(p, lk.mouth)
    p.restore()


# --- poses --------------------------------------------------------------------------------

def _tail(p, kind: str) -> None:
    if kind == "up":
        t = _path(("M", 62, 90), ("C", 78, 90, 84, 76, 82, 58), ("Q", 81.4, 50, 86.5, 47))
    elif kind == "question":
        t = _path(("M", 62, 90), ("C", 80, 92, 86, 76, 80, 62), ("C", 76, 52, 88, 44, 90, 52))
    else:
        t = _path(("M", 62, 90), ("C", 80, 94, 90, 78, 84, 64), ("Q", 81, 57, 76, 60))
    _thick(p, t, 7.6)
    _bands(p, t, (0.58, 0.74, 0.9), 7.6)


def _arm(p, sx: int) -> None:
    arm = _path(("M", 50 + sx * 8, 63), ("Q", 50 + sx * 24, 62, 50 + sx * 29, 47))
    _thick(p, arm, 8.6)
    cx, cy = 50 + sx * 29.4, 43.6
    _fill(p, _ellipse(cx, cy, 5.4, 5.0), CREAM)
    _fill(p, _ellipse(cx, cy + 1.0, 2.3, 1.8), EAR_IN, outline=False)
    for dx, dy in ((-2.7, -2.2), (0, -3.3), (2.7, -2.2)):
        _fill(p, _ellipse(cx + dx, cy + dy, 1.05, 1.05), EAR_IN, outline=False)


def _sit(p, lk: Look) -> None:
    _shadow(p, 50, 27)
    _tail(p, lk.tail)
    body = _path(("M", 37, 54), ("C", 26, 61, 21, 80, 25, 91), ("Q", 27, 95.6, 35, 95.6), ("L", 65, 95.6),
                 ("Q", 73, 95.6, 75, 91), ("C", 79, 80, 74, 61, 63, 54), ("Z",))
    _fill(p, body, _fur(50, 96))
    for sx in (-1, 1):
        _fill(p, _ellipse(50 + sx * 17.5, 87.6, 9.6, 8.0), _fur(79, 96))
        for y in (66, 72, 78):
            _stroke(p, _path(("M", 50 + sx * 23.6, y), ("Q", 50 + sx * 20, y - 1.2, 50 + sx * 17.2, y + 1.0)), STRIPE, 2.1)
    _fill(p, _ellipse(50, 67, 10.5, 11.5), CREAM, outline=False)
    raised = {"wave": {1}, "up": {-1, 1}}.get(lk.paws, set())
    for sx in (-1, 1):
        if sx in raised:
            continue
        x = 50 + sx * 6.2
        leg = QPainterPath()
        leg.addRoundedRect(QRectF(x - 4.6, 70, 9.2, 23.5), 4.6, 4.6)
        _fill(p, leg, _fur(70, 94))
        _paw(p, x, 92.6)
    _head(p, lk, 50, 40)
    for sx in sorted(raised):
        _arm(p, sx)


def _stretch(p, lk: Look) -> None:
    _shadow(p, 50, 40)
    t = _path(("M", 80, 70), ("C", 90, 66, 93, 54, 87, 45), ("Q", 84.5, 41, 88.5, 37.5))
    _thick(p, t, 7.4)
    _bands(p, t, (0.55, 0.72, 0.9), 7.4)
    hind = QPainterPath()
    hind.addRoundedRect(QRectF(73.6, 78, 9.6, 16), 4.8, 4.8)
    _fill(p, hind, _fur(78, 95))
    body = _path(("M", 20, 86), ("C", 20, 76, 32, 74, 46, 76), ("C", 58, 77.5, 63, 61, 76, 61),
                 ("C", 87, 61, 90, 74, 87.5, 86), ("C", 86.5, 92, 83, 95.6, 77, 95.6), ("L", 30, 95.6),
                 ("C", 24, 95.6, 20, 92, 20, 86), ("Z",))
    _fill(p, body, _fur(60, 96))
    for x, y in ((56, 73.5), (62, 68.5), (69, 63.5)):
        _stroke(p, _path(("M", x, y), ("Q", x + 1.6, y + 2.6, x + 1.0, y + 5.2)), STRIPE, 2.1)
    _fill(p, _ellipse(36, 89, 12, 5.2), CREAM, outline=False)
    _paw(p, 79.5, 93.4, 5.4, 3.2)
    for y in (86.6, 91.4):
        arm = QPainterPath()
        arm.addRoundedRect(QRectF(6.5, y - 3.6, 26, 7.2), 3.6, 3.6)
        _fill(p, arm, _fur(y - 4, y + 4))
        _paw(p, 9.2, y, 4.8, 3.4)
    _head(p, lk, 28, 69, scale=0.8, tilt=-8)


def _curl(p, lk: Look) -> None:
    _shadow(p, 52, 38)
    body = _path(("M", 20, 86), ("C", 20, 70, 40, 63, 58, 63), ("C", 76, 63, 89, 72, 89, 84),
                 ("C", 89, 92, 85, 95.6, 77, 95.6), ("L", 30, 95.6), ("C", 24, 95.6, 20, 92, 20, 86), ("Z",))
    _fill(p, body, _fur(62, 96))
    for x in (54, 62, 70, 78):
        _stroke(p, _path(("M", x, 64.2 + (x - 54) * 0.12), ("Q", x + 1.5, 67.5, x + 0.6, 70.5)), STRIPE, 2.1)
    _paw(p, 42, 91.6, 5.2, 3.2)
    t = _path(("M", 87, 87), ("C", 84, 97.5, 56, 98.2, 38, 94.4))
    _thick(p, t, 7.2)
    _bands(p, t, (0.55, 0.72, 0.88), 7.2)
    _head(p, lk, 31, 75.5, scale=0.78, tilt=-10)


POSES = {"sit": _sit, "stretch": _stretch, "curl": _curl}


# --- output -------------------------------------------------------------------------------

def render(name: str, size: int = 512):
    """QImage (ARGB32) of one expression."""
    big = size * 2
    img = QtGui.QImage(big, big, QtGui.QImage.Format_ARGB32_Premultiplied)
    img.fill(Qt.transparent)
    p = QtGui.QPainter(img)
    p.setRenderHint(QtGui.QPainter.Antialiasing)
    p.scale(big / 100.0, big / 100.0)
    lk = LOOKS[name]
    POSES[lk.pose](p, lk)
    p.end()
    return img.scaled(size, size, Qt.IgnoreAspectRatio, Qt.SmoothTransformation).convertToFormat(QtGui.QImage.Format_ARGB32)


def preview_sheet(folder: Path, names: list[str], cell: int = 200) -> Path:
    """Contact sheet of the sprites in `folder` with their names."""
    cols = 4
    rows = math.ceil(len(names) / cols)
    sheet = QtGui.QImage(cols * cell, rows * (cell + 26), QtGui.QImage.Format_ARGB32)
    sheet.fill(QColor("#F3EFE8"))
    p = QtGui.QPainter(sheet)
    p.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)
    font = QtGui.QFont()
    font.setPixelSize(14)
    p.setFont(font)
    p.setPen(QColor("#44403C"))
    for i, name in enumerate(names):
        x, y = (i % cols) * cell, (i // cols) * (cell + 26)
        p.drawImage(QRectF(x, y, cell, cell), QtGui.QImage(str(folder / f"{name}.png")))
        p.drawText(QRectF(x, y + cell, cell, 22), Qt.AlignCenter, name)
    p.end()
    out = folder / "_preview_sheet.png"
    sheet.save(str(out))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--only", nargs="*", choices=sorted(LOOKS), help="hanya ekspresi ini")
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    names = args.only or list(LOOKS)
    for name in names:
        render(name, args.size).save(str(args.out / f"{name}.png"))
    sheet = preview_sheet(args.out, [n for n in LOOKS if (args.out / f"{n}.png").exists()])
    print(f"{len(names)} sprite -> {args.out}  (preview: {sheet.name})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
