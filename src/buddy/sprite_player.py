"""SpriteAnimator (Lampiran A.6): plays the 12 cat animations from assets/cat/animations.yaml.

* One PNG per expression (static_pose) + code transforms (transforms.py) + code overlays
  (overlays.py). Replacing a PNG with a real sprite of the same name needs no code change.
* On start every PNG is validated (exists, RGBA, square, == canvas_size); otherwise a clear
  error is logged and the procedural QPainter cat is used instead.
* Render timer: anim_fps (16 ms) while a non-looping animation or crossfade runs, idle_fps
  (33 ms) for looping ones.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from PySide6.QtCore import QObject, QPointF, QRect, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QImage, QPainter, QPixmap, QTransform

from src.buddy import procedural
from src.buddy.buddy_state import PRIORITY, BuddyState
from src.buddy.overlays import draw_overlay
from src.buddy.transforms import transform_at

logger = logging.getLogger(__name__)

ASSETS_DIR = Path(__file__).resolve().parent / "assets" / "cat"
EXPRESSIONS: tuple[str, ...] = tuple(s.value for s in BuddyState)


@dataclass(frozen=True)
class AnimationSpec:
    """One entry of animations.yaml."""

    name: str
    pose: str
    mode: str
    transform: str | None
    overlay: str | None
    loop: bool
    duration_ms: float
    crossfade_ms: float
    next: str | None


def load_animations(path: Path) -> tuple[int, dict[str, AnimationSpec], str]:
    """Parse animations.yaml -> (canvas_size, specs, default_state)."""
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    specs = {}
    for name, a in doc["animations"].items():
        specs[name] = AnimationSpec(
            name=name, pose=a["pose"], mode=a.get("mode", "static_pose"), transform=a.get("transform"),
            overlay=a.get("overlay"), loop=bool(a.get("loop", False)), duration_ms=float(a.get("duration_ms", 1000)),
            crossfade_ms=float(a.get("crossfade_ms", 150)), next=a.get("next"),
        )
    return int(doc["canvas_size"]), specs, str(doc.get("default_state", "idle"))


def validate_assets(assets_dir: Path, specs: dict[str, AnimationSpec], canvas: int) -> list[str]:
    """Problems with the sprite files (empty list = all 12 OK)."""
    problems = []
    missing = [e for e in EXPRESSIONS if e not in specs]
    if missing:
        problems.append(f"animations.yaml tidak punya: {', '.join(missing)}")
    for spec in specs.values():
        if spec.mode != "static_pose":
            continue
        img = QImage(str(assets_dir / spec.pose))
        if img.isNull():
            problems.append(f"{spec.pose}: tidak ada / tidak terbaca")
        elif not img.hasAlphaChannel():
            problems.append(f"{spec.pose}: bukan RGBA (butuh background transparan)")
        elif img.width() != img.height() or img.width() != canvas:
            problems.append(f"{spec.pose}: ukuran {img.width()}x{img.height()}, harus {canvas}x{canvas}")
    return problems


class SpriteAnimator(QObject):
    """Animation state + painting. Call paint() from the widget's paintEvent."""

    finished = Signal(str)
    frame_changed = Signal()

    def __init__(self, buddy_cfg: dict[str, Any], assets_dir: Path = ASSETS_DIR, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.assets_dir = assets_dir
        self.overlays_enabled = bool(buddy_cfg.get("overlays_enabled", True))
        self.active_interval = max(1, int(1000 / float(buddy_cfg.get("anim_fps", 60))))
        self.idle_interval = max(1, int(1000 / float(buddy_cfg.get("idle_fps", 30))))
        self.canvas, self.specs, default = load_animations(assets_dir / "animations.yaml")
        self.problems = validate_assets(assets_dir, self.specs, self.canvas)
        self.procedural = bool(self.problems)
        if self.procedural:
            logger.error("Sprite Buddy tidak valid, memakai kucing prosedural: %s", "; ".join(self.problems))
        self._pixmaps: dict[str, QPixmap] = {}
        self._current = default
        self._prev: str | None = None
        self._start = time.monotonic()
        self._fade_start = 0.0
        self._done = False
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(self.idle_interval)

    # -- assets -------------------------------------------------------------------
    def pixmap(self, name: str) -> QPixmap:
        """Cached pose pixmap for an animation (procedural when assets are invalid)."""
        if name not in self._pixmaps:
            spec = self.specs[name]
            pm = QPixmap() if self.procedural else QPixmap(str(self.assets_dir / spec.pose))
            if pm.isNull():
                pm = procedural.make_pixmap(name, self.canvas)
            self._pixmaps[name] = pm
        return self._pixmaps[name]

    # -- playback -----------------------------------------------------------------
    def current(self) -> str:
        """Name of the animation playing now."""
        return self._current

    def _running_non_loop(self) -> bool:
        spec = self.specs[self._current]
        return not spec.loop and not self._done

    def play(self, name: str, *, interrupt: bool = False) -> bool:
        """Start an animation; a running higher-priority non-loop animation is kept unless interrupt."""
        if name not in self.specs:
            raise ValueError(f"Animasi tidak dikenal: {name}")
        if not interrupt and self._running_non_loop() and name != self._current:
            if PRIORITY[BuddyState(name)] < PRIORITY[BuddyState(self._current)]:
                return False
        if name != self._current:
            self._prev = self._current
            self._fade_start = time.monotonic()
        self._current = name
        self._start = time.monotonic()
        self._done = False
        self._timer.setInterval(self.active_interval)
        self.frame_changed.emit()
        return True

    def elapsed_ms(self) -> float:
        """Time since the current animation started."""
        return (time.monotonic() - self._start) * 1000.0

    def progress(self) -> float:
        """Normalized time t of the current cycle."""
        spec = self.specs[self._current]
        e = self.elapsed_ms()
        if spec.loop:
            return (e % spec.duration_ms) / spec.duration_ms
        return min(1.0, e / spec.duration_ms)

    def _fade(self) -> float:
        spec = self.specs[self._current]
        if self._prev is None or spec.crossfade_ms <= 0:
            return 1.0
        return min(1.0, (time.monotonic() - self._fade_start) * 1000.0 / spec.crossfade_ms)

    def _tick(self) -> None:
        spec = self.specs[self._current]
        if not spec.loop and not self._done and self.elapsed_ms() >= spec.duration_ms:
            self._done = True
            name = self._current
            self.finished.emit(name)
            if spec.next and self._current == name:
                self.play(spec.next, interrupt=True)
        if self._fade() >= 1.0:
            self._prev = None
        busy = self._prev is not None or self._running_non_loop()
        self._timer.setInterval(self.active_interval if busy else self.idle_interval)
        self.frame_changed.emit()

    # -- painting -----------------------------------------------------------------
    def _draw_pose(self, p: QPainter, name: str, rect: QRectF, opacity: float, t: float) -> None:
        spec = self.specs[name]
        tr = transform_at(spec.transform, t, spec.duration_ms)
        if spec.transform == "slow_blink":  # crossfade from idle (flat pose cannot blink)
            self._blit(p, self.pixmap("idle"), rect, opacity * (1.0 - tr.opacity), tr)
            self._blit(p, self.pixmap(name), rect, opacity * tr.opacity, tr)
        else:
            self._blit(p, self.pixmap(name), rect, opacity * tr.opacity, tr)

    @staticmethod
    def _blit(p: QPainter, pm: QPixmap, rect: QRectF, opacity: float, tr: Any) -> None:
        if opacity <= 0.0:
            return
        p.save()
        p.setOpacity(opacity)
        pivot = QPointF(rect.center().x(), rect.bottom())
        m = QTransform()
        m.translate(pivot.x() + tr.dx * rect.height(), pivot.y() + tr.dy * rect.height())
        m.rotate(tr.rotate_deg)
        m.scale(tr.scale_x, tr.scale_y)
        m.translate(-pivot.x(), -pivot.y())
        p.setTransform(m, True)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        p.drawPixmap(rect.toRect(), pm)
        p.restore()

    def paint(self, painter: QPainter, rect: QRect | QRectF) -> None:
        """Draw the current frame (crossfade + transform + overlay) into rect."""
        r = QRectF(rect)
        fade = self._fade()
        if self._prev is not None and fade < 1.0:
            self._draw_pose(painter, self._prev, r, 1.0 - fade, 0.0)
        self._draw_pose(painter, self._current, r, fade, self.progress())
        if self.overlays_enabled:
            draw_overlay(self.specs[self._current].overlay, painter, r, self.progress(), self.elapsed_ms())

    def render_image(self, size: int) -> QImage:
        """Current frame as an ARGB image (used for the click-through mask and tests)."""
        img = QImage(size, size, QImage.Format.Format_ARGB32_Premultiplied)
        img.fill(Qt.GlobalColor.transparent)
        p = QPainter(img)
        self.paint(p, QRect(0, 0, size, size))
        p.end()
        return img
