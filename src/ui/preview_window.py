"""Live camera preview (D-032): your own face with the detector boxes, pose points and the
indicators the stress score uses.

Frames arrive from the inference thread only while this window is open (Pipeline
.set_preview_listener); they live in memory, are drawn with QPainter and are dropped on close.
Nothing is ever written to disk and there is no screenshot or record button.
"""

from __future__ import annotations

import threading
from typing import Any

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen
from PySide6.QtWidgets import QCheckBox, QFrame, QGridLayout, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from src.analysis.sliding_window import eye_closed
from src.ui import theme

REFRESH_MS = 66  # ~15 Hz; inference is capped lower anyway
CLASS_COLORS: dict[str, str] = {
    "mata_tertutup": "#DC2626", "mata_terbuka": "#16A34A", "menguap": "#EA580C",
}
LANDMARK_COLOR = "#38BDF8"
POSE_COLOR = "#FACC15"
# COCO keypoints: nose 0, eyes 1-2, ears 3-4, shoulders 5-6, elbows 7-8, wrists 9-10
POSE_EDGES = ((5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (0, 1), (0, 2), (1, 3), (2, 4))
OK, WARN, BAD = theme.LEVEL_COLORS["rendah"], theme.LEVEL_COLORS["sedang"], theme.LEVEL_COLORS["tinggi"]


def indicators(frame: Any, calibration: Any, cfg: dict[str, Any], snapshot: Any | None) -> dict[str, tuple[str, str]]:
    """Human-readable status per indicator: key -> (text, colour). Pure, so it is unit-tested."""
    f = frame.features
    out: dict[str, tuple[str, str]] = {}
    if not f.user_present:
        out["wajah"] = ("tidak terlihat", BAD)
    else:
        out["wajah"] = ("terdeteksi", OK)
    closed = eye_closed(f, calibration.ear_threshold, cfg) if f.user_present else None
    out["mata"] = ("–", theme.TEXT_2) if closed is None else (("TERTUTUP", BAD) if closed else ("terbuka", OK))
    out["menguap"] = ("YA", WARN) if f.yawn >= 0.5 else ("tidak", OK)
    out["tangan"] = ("di dahi", WARN) if f.hand_to_forehead else ("tidak", OK)
    h = cfg["habit"]
    if f.shoulder_slouch is None:
        out["postur"] = ("–", theme.TEXT_2)
    elif calibration.shoulder_slouch:
        bad = f.shoulder_slouch < calibration.shoulder_slouch * (1.0 - float(h["slouch_drop"]))
        out["postur"] = ("membungkuk", WARN) if bad else ("tegak", OK)
    else:
        out["postur"] = (f"{f.shoulder_slouch:.2f} (perlu kalibrasi)", theme.TEXT_2)
    if f.face_size_ratio is None:
        out["jarak"] = ("–", theme.TEXT_2)
    elif calibration.face_size_ratio:
        close = f.face_size_ratio > calibration.face_size_ratio * float(h["too_close_ratio"])
        out["jarak"] = ("terlalu dekat", WARN) if close else ("pas", OK)
    else:
        out["jarak"] = (f"{f.face_size_ratio:.2f} (perlu kalibrasi)", theme.TEXT_2)
    if snapshot is None:
        out["skor"] = ("menghitung…", theme.TEXT_2)
    else:
        out["skor"] = (f"{snapshot.score:.0f} · {snapshot.level.capitalize()}", theme.LEVEL_COLORS.get(snapshot.level, theme.TEXT))
    out["fps"] = (f"{frame.fps:.1f}", theme.TEXT)
    return out


def to_qimage(bgr: np.ndarray) -> QImage:
    """BGR ndarray -> QImage (copied, so the array can be released)."""
    rgb = np.ascontiguousarray(bgr[..., ::-1])
    h, w = rgb.shape[:2]
    return QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()


class VideoView(QWidget):
    """Draws the latest frame and the overlays, aspect-ratio preserved."""

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumSize(480, 360)
        self.image: QImage | None = None
        self.frame: Any | None = None
        self.message = "Menunggu kamera…"
        self.show_yolo = self.show_landmark = self.show_pose = True
        self.mirror = True

    def set_frame(self, frame: Any | None, image: QImage | None) -> None:
        self.frame, self.image = frame, image
        self.update()

    def paintEvent(self, _event) -> None:  # noqa: N802 (Qt API)
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        path_bg = QRectF(self.rect())
        p.setPen(Qt.NoPen)
        p.setBrush(QColor("#141210"))
        p.drawRoundedRect(path_bg, 14, 14)
        if self.image is None or self.frame is None:
            p.setPen(QColor("#E7E5E4"))
            p.drawText(self.rect(), Qt.AlignCenter, self.message)
            p.end()
            return
        iw, ih = self.image.width(), self.image.height()
        scale = min(self.width() / iw, self.height() / ih)
        ox, oy = (self.width() - iw * scale) / 2, (self.height() - ih * scale) / 2
        target = QRectF(ox, oy, iw * scale, ih * scale)
        p.drawImage(target, self.image.flipped(Qt.Orientation.Horizontal) if self.mirror else self.image)

        def pt(x: float, y: float) -> QPointF:
            x = iw - x if self.mirror else x
            return QPointF(ox + x * scale, oy + y * scale)

        def box(xyxy, color: str, label: str | None, width: float = 2.0, dashed: bool = False) -> None:
            x1, y1, x2, y2 = xyxy
            a, b = pt(x1, y1), pt(x2, y2)
            r = QRectF(a, b).normalized()
            pen = QPen(QColor(color), width)
            if dashed:
                pen.setStyle(Qt.DashLine)
            p.setPen(pen)
            p.setBrush(Qt.NoBrush)
            p.drawRoundedRect(r, 5, 5)
            if label:
                f = QFont(self.font())
                f.setPixelSize(12)
                f.setBold(True)
                p.setFont(f)
                tw = p.fontMetrics().horizontalAdvance(label) + 12
                tag = QRectF(r.left(), max(oy, r.top() - 20), tw, 19)
                p.setPen(Qt.NoPen)
                p.setBrush(QColor(color))
                p.drawRoundedRect(tag, 5, 5)
                p.setPen(QColor("white"))
                p.drawText(tag, Qt.AlignCenter, label)

        fr = self.frame
        lm = fr.landmark
        if self.show_landmark and lm is not None and lm.face_found:
            if lm.face_box:
                box(lm.face_box, LANDMARK_COLOR, "wajah", 1.8, dashed=True)
            for b in (lm.eye_box, lm.mouth_box):
                if b:
                    box(b, LANDMARK_COLOR, None, 1.3, dashed=True)
        if self.show_pose and fr.pose is not None and fr.pose.keypoints is not None:
            k = fr.pose.keypoints
            ok = k[:, 2] >= 0.5
            p.setPen(QPen(QColor(POSE_COLOR), 2.2))
            for i, j in POSE_EDGES:
                if ok[i] and ok[j]:
                    p.drawLine(pt(k[i, 0], k[i, 1]), pt(k[j, 0], k[j, 1]))
            p.setBrush(QColor(POSE_COLOR))
            for i in np.flatnonzero(ok):
                p.drawEllipse(pt(k[i, 0], k[i, 1]), 3.8, 3.8)
        if self.show_yolo and fr.faces is not None:
            for d in fr.faces.boxes:
                box(d.xyxy, CLASS_COLORS.get(d.cls_name, "#FFFFFF"), f"{d.cls_name} {d.conf:.2f}", 2.5)
        # privacy badge
        f = QFont(self.font())
        f.setPixelSize(11)
        f.setBold(True)
        p.setFont(f)
        badge = QRectF(target.right() - 168, target.top() + 10, 158, 24)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(15, 14, 13, 175))
        p.drawRoundedRect(badge, 12, 12)
        p.setBrush(QColor("#EF4444"))
        p.drawEllipse(QPointF(badge.left() + 13, badge.center().y()), 4, 4)
        p.setPen(QColor("white"))
        p.drawText(badge.adjusted(22, 0, 0, 0), Qt.AlignVCenter, "LIVE · tidak disimpan")
        p.end()


class PreviewWindow(QWidget):
    """Window shell: video, compact indicator cards, overlay toggles. Feed it with `push()` from any thread."""

    closed = Signal()
    ROWS = (
        ("wajah", "👤  Wajah"),
        ("mata", "👁️  Mata"),
        ("menguap", "🥱  Menguap"),
        ("tangan", "✋  Tangan"),
        ("postur", "🪑  Postur"),
        ("jarak", "📏  Jarak layar"),
        ("fps", "⚡  FPS"),
    )

    def __init__(self, cfg: dict[str, Any], calibration_fn, snapshot_fn, camera_active_fn=None) -> None:
        super().__init__()
        self.cfg = cfg
        self.calibration_fn, self.snapshot_fn = calibration_fn, snapshot_fn
        self.camera_active_fn = camera_active_fn or (lambda: True)
        self._lock = threading.Lock()
        self._pending: Any | None = None
        self.setWindowTitle("Stress Buddy · Kamera & Deteksi")
        self.setStyleSheet(theme.STYLESHEET)
        self.resize(960, 580)

        root = QHBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(14)

        # Left video card
        video_card = QFrame()
        video_card.setObjectName("card")
        left = QVBoxLayout(video_card)
        left.setContentsMargins(12, 12, 12, 10)
        left.setSpacing(10)
        self.view = VideoView()
        left.addWidget(self.view, 1)

        toggles = QHBoxLayout()
        toggles.setSpacing(14)
        for attr, text in (("show_yolo", "Kotak YOLO"), ("show_landmark", "Landmark"), ("show_pose", "Postur"),
                           ("mirror", "Cermin")):
            cb = QCheckBox(text)
            cb.setChecked(getattr(self.view, attr))
            cb.toggled.connect(lambda on, a=attr: (setattr(self.view, a, on), self.view.update()))
            toggles.addWidget(cb)
            setattr(self, f"cb_{attr}", cb)
        toggles.addStretch(1)
        left.addLayout(toggles)
        root.addWidget(video_card, 3)

        # Right status column
        right_col = QVBoxLayout()
        right_col.setSpacing(10)

        # Hero score card
        score_card = QFrame()
        score_card.setObjectName("hero")
        score_card.setFixedWidth(290)
        sl = QVBoxLayout(score_card)
        sl.setContentsMargins(16, 12, 16, 12)
        sl.setSpacing(4)
        sc_lbl = QLabel("SKOR STRES SAAT INI")
        sc_lbl.setStyleSheet(f"color: {theme.ACCENT}; font-size: 11px; font-weight: 800;")
        self.values: dict[str, QLabel] = {}
        val_skor = QLabel("–")
        val_skor.setStyleSheet("font-size: 24px; font-weight: 800;")
        self.values["skor"] = val_skor
        sl.addWidget(sc_lbl)
        sl.addWidget(val_skor)
        right_col.addWidget(score_card)

        # Indicators card
        panel = QFrame()
        panel.setObjectName("card")
        panel.setFixedWidth(290)
        pl = QVBoxLayout(panel)
        pl.setContentsMargins(16, 14, 16, 14)
        pl.setSpacing(10)
        title = QLabel("Indikator Terdeteksi")
        title.setObjectName("h2")
        pl.addWidget(title)

        grid = QGridLayout()
        grid.setVerticalSpacing(9)
        grid.setHorizontalSpacing(10)
        for r, (key, label) in enumerate(self.ROWS):
            cap = QLabel(label)
            cap.setObjectName("cardLabel")
            val = QLabel("–")
            val.setWordWrap(True)
            val.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            val.setStyleSheet("font-weight: 700;")
            grid.addWidget(cap, r, 0)
            grid.addWidget(val, r, 1)
            self.values[key] = val
        pl.addLayout(grid)
        pl.addStretch(1)

        note = QLabel("🔒 Diproses lokal di memori, tidak disimpan")
        note.setObjectName("muted")
        note.setWordWrap(True)
        pl.addWidget(note)
        close = QPushButton("Tutup")
        close.clicked.connect(self.close)
        pl.addWidget(close)
        right_col.addWidget(panel, 1)
        root.addLayout(right_col)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh)

    # -- feeding (any thread) ---------------------------------------------------------------
    def push(self, frame: Any) -> None:
        """Called from the inference thread: keep only the newest frame."""
        with self._lock:
            self._pending = frame

    # -- main thread ------------------------------------------------------------------------
    def showEvent(self, event) -> None:  # noqa: N802 (Qt API)
        self.timer.start(REFRESH_MS)
        super().showEvent(event)

    def refresh(self) -> None:
        """Draw the newest frame and update the panel."""
        with self._lock:
            frame, self._pending = self._pending, None
        if frame is None:
            if not self.camera_active_fn():
                self.view.message = "Kamera dijeda / tidak tersedia"
                self.view.set_frame(None, None)
            return
        self.view.set_frame(frame, to_qimage(frame.image))
        for key, (text, color) in indicators(frame, self.calibration_fn(), self.cfg, self.snapshot_fn()).items():
            if key in self.values:
                self.values[key].setText(text)
                size = "22px" if key == "skor" else "13px"
                self.values[key].setStyleSheet(f"font-size: {size}; font-weight: 700; color: {color};")

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt API)
        self.timer.stop()
        with self._lock:
            self._pending = None
        self.view.set_frame(None, None)  # drop the last frame from memory
        self.closed.emit()
        super().closeEvent(event)
