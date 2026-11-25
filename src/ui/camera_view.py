"""Live camera & detection process viewer (simple, visual, minimal text).

Displays the real-time webcam frame with detection overlays (face box, MediaPipe
landmarks, YOLO boxes, pose keypoints) alongside a compact visual summary of the
detection state and stress score.

Frames are processed in memory only and never written to disk.
"""

from __future__ import annotations

import cv2
import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from src.analysis.engine import TickResult
from src.analysis.sliding_window import eye_closed
from src.core.pipeline import DetectionFrame, Pipeline
from src.detection import ear_calculator as ec
from src.detection import pose_estimator as pe
from src.detection.face_detector import CLASS_COLORS
from src.ui import theme

POSE_EDGES: tuple[tuple[int, int], ...] = (
    (pe.L_SH, pe.R_SH),
    (pe.L_SH, pe.L_WR),
    (pe.R_SH, pe.R_WR),
)

COMPONENT_LABELS: tuple[tuple[str, str, int], ...] = (
    ("perclos", "Mata (PERCLOS)", 30),
    ("yawn", "Menguap", 20),
    ("furrow", "Kerutan Dahi", 20),
    ("hand", "Tangan ke Dahi", 15),
    ("blink", "Kedipan", 10),
    ("motion", "Gerak Kepala", 5),
)


def render_detection_overlay(
    df: DetectionFrame,
    ear_threshold: float,
    cfg: dict,
    *,
    show_landmarks: bool = True,
    show_yolo: bool = True,
    show_pose: bool = True,
    show_hud: bool = True,
) -> np.ndarray:
    """Draw face box, landmarks, YOLO boxes, pose keypoints, and a minimal HUD badge on a copy of the frame."""
    canvas = df.image.copy()
    h, w = canvas.shape[:2]
    lm = df.landmark
    f = df.features
    closed = eye_closed(f, ear_threshold, cfg) if f.user_present else None

    # 1. MediaPipe Face Box & Landmarks
    if show_landmarks and lm is not None and lm.face_found:
        if lm.face_box is not None:
            x1, y1, x2, y2 = lm.face_box
            box_color = (60, 60, 230) if closed else (60, 185, 75)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), box_color, 2)

        pts = getattr(lm, "landmarks", None)
        if pts is not None and len(pts) > 454:
            for pt in pts[::6]:
                cv2.circle(canvas, (int(pt[0]), int(pt[1])), 1, (190, 215, 190), -1, cv2.LINE_AA)

            eye_color = (50, 50, 235) if closed else (50, 220, 100)
            for eye_idx in (ec.LEFT_EYE, ec.RIGHT_EYE):
                poly = np.array([[int(pts[i][0]), int(pts[i][1])] for i in eye_idx], dtype=np.int32)
                cv2.polylines(canvas, [poly], True, eye_color, 2, cv2.LINE_AA)

            mouth_color = (30, 145, 245) if f.yawn > 0.5 else (235, 190, 70)
            m_top = (int(pts[ec.MOUTH_INNER_V[0]][0]), int(pts[ec.MOUTH_INNER_V[0]][1]))
            m_bot = (int(pts[ec.MOUTH_INNER_V[1]][0]), int(pts[ec.MOUTH_INNER_V[1]][1]))
            m_left = (int(pts[ec.MOUTH_CORNERS[0]][0]), int(pts[ec.MOUTH_CORNERS[0]][1]))
            m_right = (int(pts[ec.MOUTH_CORNERS[1]][0]), int(pts[ec.MOUTH_CORNERS[1]][1]))
            cv2.line(canvas, m_left, m_right, mouth_color, 1, cv2.LINE_AA)
            cv2.line(canvas, m_top, m_bot, mouth_color, 2, cv2.LINE_AA)

            brow_color = (210, 60, 210) if f.brow_furrow > 0.5 else (220, 180, 120)
            b_l = (int(pts[ec.INNER_BROWS[0]][0]), int(pts[ec.INNER_BROWS[0]][1]))
            b_r = (int(pts[ec.INNER_BROWS[1]][0]), int(pts[ec.INNER_BROWS[1]][1]))
            cv2.line(canvas, b_l, b_r, brow_color, 2, cv2.LINE_AA)
            cv2.circle(canvas, b_l, 3, brow_color, -1, cv2.LINE_AA)
            cv2.circle(canvas, b_r, 3, brow_color, -1, cv2.LINE_AA)

    # 2. YOLOv8 Face Detections
    if show_yolo and df.faces is not None:
        for d in getattr(df.faces, "boxes", ()):
            color = CLASS_COLORS.get(d.cls_name, (255, 255, 255))
            x1, y1, x2, y2 = d.xyxy
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                canvas, f"{d.cls_name} {d.conf:.2f}",
                (x1, min(h - 8, y2 + 16)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2, cv2.LINE_AA,
            )

    # 3. YOLOv8-Pose Skeleton & Keypoints
    pose = df.pose
    if show_pose and pose is not None and pose.person_found and pose.keypoints is not None:
        kpts = pose.keypoints
        min_conf = float(cfg["pose"]["min_kp_conf"])
        for a, b in POSE_EDGES:
            if a < len(kpts) and b < len(kpts) and kpts[a, 2] >= min_conf and kpts[b, 2] >= min_conf:
                pa = (int(kpts[a, 0]), int(kpts[a, 1]))
                pb = (int(kpts[b, 0]), int(kpts[b, 1]))
                cv2.line(canvas, pa, pb, (240, 175, 50), 2, cv2.LINE_AA)
        for idx in (pe.L_SH, pe.R_SH, pe.L_WR, pe.R_WR):
            if idx < len(kpts) and kpts[idx, 2] >= min_conf:
                px, py = int(kpts[idx, 0]), int(kpts[idx, 1])
                is_wrist = idx in (pe.L_WR, pe.R_WR)
                kp_color = (40, 60, 240) if (is_wrist and f.hand_to_forehead) else (255, 200, 60)
                cv2.circle(canvas, (px, py), 5 if is_wrist else 4, kp_color, -1, cv2.LINE_AA)

    # 4. Minimal top-left HUD pill
    if show_hud:
        ear_txt = f"{f.ear_mean:.2f}" if f.ear_mean is not None else "-"
        mar_txt = f"{f.mar:.2f}" if f.mar is not None else "-"
        line = f"{df.fps:4.1f} FPS  |  EAR {ear_txt}  |  MAR {mar_txt}"
        overlay = canvas.copy()
        cv2.rectangle(overlay, (8, 8), (min(w - 8, 285), 34), (24, 22, 20), -1)
        cv2.addWeighted(overlay, 0.65, canvas, 0.35, 0, canvas)
        cv2.putText(canvas, line, (14, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (248, 245, 240), 1, cv2.LINE_AA)

    return canvas


def _badge_tile(icon: str, title: str) -> tuple[QFrame, QLabel]:
    frame = QFrame()
    frame.setObjectName("card")
    lay = QVBoxLayout(frame)
    lay.setContentsMargins(10, 8, 10, 8)
    lay.setSpacing(2)
    cap = QLabel(f"{icon}  {title}")
    cap.setObjectName("cardLabel")
    val = QLabel("–")
    val.setStyleSheet("font-size: 14px; font-weight: 700;")
    lay.addWidget(cap)
    lay.addWidget(val)
    return frame, val


class CameraView(QWidget):
    """Compact live webcam & detection viewer with minimal text."""

    closed = Signal()
    toggle_pause_requested = Signal()
    recalibrate_requested = Signal()

    def __init__(self, pipeline: Pipeline, cfg: dict) -> None:
        super().__init__()
        self.pipeline = pipeline
        self.cfg = cfg
        self.setWindowTitle("Stress Buddy · Kamera & Deteksi")
        self.setStyleSheet(theme.STYLESHEET)
        self.resize(900, 560)
        self._last_df: DetectionFrame | None = None

        # Hidden compatibility labels kept concise for tests / callers
        self.val_face = QLabel("–")
        self.val_ear = QLabel("–")
        self.val_win_span = QLabel("–")

        outer = QVBoxLayout(self)
        outer.setContentsMargins(16, 14, 16, 14)
        outer.setSpacing(10)

        outer.addLayout(self._build_header())

        body = QHBoxLayout()
        body.setSpacing(12)
        body.addWidget(self._build_video_panel(), 3)
        body.addWidget(self._build_side_panel(), 2)
        outer.addLayout(body, 1)

    def _build_header(self) -> QHBoxLayout:
        head = QHBoxLayout()
        title = QLabel("Kamera & Deteksi")
        title.setObjectName("title")
        head.addWidget(title)

        self.status_pill = QLabel("Menunggu...")
        self.status_pill.setObjectName("pill")
        self.status_pill.setStyleSheet(f"background: {theme.SURFACE}; color: {theme.TEXT};")
        head.addWidget(self.status_pill, 0, Qt.AlignmentFlag.AlignVCenter)
        head.addStretch(1)

        self.pause_btn = QPushButton("Jeda")
        self.pause_btn.setObjectName("secondary")
        self.pause_btn.clicked.connect(self.toggle_pause_requested.emit)
        head.addWidget(self.pause_btn)

        recal_btn = QPushButton("Kalibrasi")
        recal_btn.setObjectName("secondary")
        recal_btn.clicked.connect(self.recalibrate_requested.emit)
        head.addWidget(recal_btn)
        return head

    def _build_video_panel(self) -> QFrame:
        card = QFrame()
        card.setObjectName("card")
        lay = QVBoxLayout(card)
        lay.setContentsMargins(12, 12, 12, 10)
        lay.setSpacing(8)

        self.video_label = QLabel("Menyiapkan kamera...")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(480, 360)
        self.video_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.video_label.setStyleSheet(
            "background: #1C1917; color: #F7F3EC; border-radius: 10px; font-size: 13px; font-weight: 600;"
        )
        lay.addWidget(self.video_label, 1)

        toggles = QHBoxLayout()
        toggles.setSpacing(12)
        self.chk_landmarks = QCheckBox("Wajah")
        self.chk_landmarks.setChecked(True)
        self.chk_yolo = QCheckBox("YOLO")
        self.chk_yolo.setChecked(True)
        self.chk_pose = QCheckBox("Postur")
        self.chk_pose.setChecked(True)
        self.chk_hud = QCheckBox("FPS")
        self.chk_hud.setChecked(True)

        for chk in (self.chk_landmarks, self.chk_yolo, self.chk_pose, self.chk_hud):
            chk.toggled.connect(self._rerender_last)
            toggles.addWidget(chk)
        toggles.addStretch(1)

        lock_lbl = QLabel("🔒 Lokal, tidak disimpan")
        lock_lbl.setObjectName("muted")
        toggles.addWidget(lock_lbl)
        lay.addLayout(toggles)
        return card

    def _build_side_panel(self) -> QWidget:
        panel = QWidget()
        col = QVBoxLayout(panel)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(10)

        # 1. Score card
        score_card = QFrame()
        score_card.setObjectName("hero")
        sl = QVBoxLayout(score_card)
        sl.setContentsMargins(14, 12, 14, 12)
        sl.setSpacing(4)
        top_row = QHBoxLayout()
        cap = QLabel("Skor Stres")
        cap.setObjectName("cardLabel")
        self.val_level = QLabel("Menunggu")
        self.val_level.setObjectName("pill")
        self.val_level.setStyleSheet(f"background: {theme.SURFACE}; color: {theme.TEXT_2};")
        top_row.addWidget(cap)
        top_row.addStretch(1)
        top_row.addWidget(self.val_level)
        sl.addLayout(top_row)

        self.val_score = QLabel("–")
        self.val_score.setStyleSheet("font-size: 28px; font-weight: 800;")
        sl.addWidget(self.val_score)
        col.addWidget(score_card)

        # 2. Live status tiles (2x3 grid, visual & short)
        grid = QGridLayout()
        grid.setSpacing(8)
        tiles = (
            ("eyes", "👁️", "Mata"),
            ("mouth", "👄", "Mulut"),
            ("brow", "🤨", "Alis"),
            ("posture", "🪑", "Postur"),
            ("hand", "✋", "Tangan"),
            ("blink", "⚡", "Kedipan"),
        )
        self.tiles: dict[str, QLabel] = {}
        for i, (key, icon, label) in enumerate(tiles):
            f_tile, v_lbl = _badge_tile(icon, label)
            self.tiles[key] = v_lbl
            grid.addWidget(f_tile, i // 2, i % 2)
        col.addLayout(grid)

        # 3. Compact component bars
        bars_card = QFrame()
        bars_card.setObjectName("card")
        bl = QVBoxLayout(bars_card)
        bl.setContentsMargins(12, 10, 12, 10)
        bl.setSpacing(6)
        bh = QLabel("Indikator (60 dtk)")
        bh.setObjectName("cardLabel")
        bl.addWidget(bh)

        self.comp_bars: dict[str, tuple[QProgressBar, QLabel]] = {}
        comp_grid = QGridLayout()
        comp_grid.setHorizontalSpacing(8)
        comp_grid.setVerticalSpacing(5)
        for i, (key, title, max_pt) in enumerate(COMPONENT_LABELS):
            lbl = QLabel(title)
            lbl.setObjectName("muted")
            bar = QProgressBar()
            bar.setRange(0, max_pt * 10)
            bar.setValue(0)
            bar.setTextVisible(False)
            bar.setFixedHeight(7)
            bar.setStyleSheet(
                f"QProgressBar {{ background: {theme.SURFACE}; border: none; border-radius: 3px; }}"
                f"QProgressBar::chunk {{ background: {theme.ACCENT}; border-radius: 3px; }}"
            )
            pts_lbl = QLabel("0")
            pts_lbl.setObjectName("muted")
            pts_lbl.setFixedWidth(28)
            pts_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            comp_grid.addWidget(lbl, i, 0)
            comp_grid.addWidget(bar, i, 1)
            comp_grid.addWidget(pts_lbl, i, 2)
            self.comp_bars[key] = (bar, pts_lbl)
        bl.addLayout(comp_grid)
        col.addWidget(bars_card)
        col.addStretch(1)
        return panel

    # -- live updates ----------------------------------------------------------------------
    def update_pause_state(self, paused: bool) -> None:
        self.pause_btn.setText("Lanjutkan" if paused else "Jeda")
        if paused:
            self.status_pill.setText("Dijeda")
            self.status_pill.setStyleSheet(f"background: {theme.LEVEL_SOFT['sedang']}; color: {theme.LEVEL_COLORS['sedang']};")
            self.video_label.setPixmap(QPixmap())
            self.video_label.setText("Kamera dijeda")

    def _rerender_last(self) -> None:
        if self._last_df is not None and not self.pipeline.paused:
            self._render_video(self._last_df)

    def _render_video(self, df: DetectionFrame) -> None:
        cal = self.pipeline.engine.calibration
        bgr = render_detection_overlay(
            df,
            cal.ear_threshold,
            self.cfg,
            show_landmarks=self.chk_landmarks.isChecked(),
            show_yolo=self.chk_yolo.isChecked(),
            show_pose=self.chk_pose.isChecked(),
            show_hud=self.chk_hud.isChecked(),
        )
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()
        pix = QPixmap.fromImage(qimg).scaled(
            self.video_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.video_label.setPixmap(pix)

    def _set_tile(self, key: str, text: str, alert: bool = False) -> None:
        lbl = self.tiles[key]
        lbl.setText(text)
        color = theme.LEVEL_COLORS["tinggi"] if alert else theme.TEXT
        lbl.setStyleSheet(f"font-size: 14px; font-weight: 700; color: {color};")

    def on_detection_frame(self, df: DetectionFrame) -> None:
        if self.pipeline.paused:
            return
        self._last_df = df
        self._render_video(df)

        cal = self.pipeline.engine.calibration
        f = df.features
        lm = df.landmark
        closed = eye_closed(f, cal.ear_threshold, self.cfg) if f.user_present else None

        if lm is not None and lm.face_found:
            self.val_face.setText("Terdeteksi")
            self.val_ear.setText(f"{lm.ear_left:.2f} / {lm.ear_right:.2f} / {lm.ear_mean:.2f}")
            self.status_pill.setText(f"Wajah Terdeteksi · {df.fps:.0f} FPS")
            self.status_pill.setStyleSheet(
                f"background: {theme.LEVEL_SOFT['rendah']}; color: {theme.LEVEL_COLORS['rendah']};"
            )
        else:
            self.val_face.setText("Tidak terlihat")
            self.val_ear.setText("–")
            self.status_pill.setText("Wajah Tidak Terlihat")
            self.status_pill.setStyleSheet(
                f"background: {theme.LEVEL_SOFT['sedang']}; color: {theme.LEVEL_COLORS['sedang']};"
            )

        # Tile 1: Mata
        if closed is None:
            self._set_tile("eyes", "–")
        elif closed:
            self._set_tile("eyes", f"Tertutup ({f.ear_mean or 0:.2f})", alert=True)
        else:
            self._set_tile("eyes", f"Terbuka ({f.ear_mean or 0:.2f})")

        # Tile 2: Mulut
        if f.mar is None:
            self._set_tile("mouth", "–")
        elif f.yawn > 0.5:
            self._set_tile("mouth", "Menguap", alert=True)
        else:
            self._set_tile("mouth", "Normal")

        # Tile 3: Alis
        if f.brow_ratio is None:
            self._set_tile("brow", "–")
        elif f.brow_furrow > 0.5:
            self._set_tile("brow", "Berkerut", alert=True)
        else:
            self._set_tile("brow", "Rileks")

        # Tile 4: Postur
        if f.shoulder_slouch is None:
            self._set_tile("posture", "–")
        elif cal.shoulder_slouch and f.shoulder_slouch < cal.shoulder_slouch * (1.0 - float(self.cfg["habit"]["slouch_drop"])):
            self._set_tile("posture", "Bungkuk", alert=True)
        else:
            self._set_tile("posture", "Tegak")

        # Tile 5: Tangan
        if not f.user_present:
            self._set_tile("hand", "–")
        elif f.hand_to_forehead:
            self._set_tile("hand", "Ke Dahi", alert=True)
        else:
            self._set_tile("hand", "Normal")

    def on_tick_result(self, res: TickResult | None) -> None:
        if res is None:
            return
        st = res.stats
        self.val_win_span.setText(f"{st.span_sec:.0f} dtk")

        # Tile 6: Kedipan
        if st.blink_rate is None:
            self._set_tile("blink", f"{st.span_sec:.0f} dtk...")
        else:
            low_b = float(self.cfg["habit"]["low_blink"])
            self._set_tile("blink", f"{st.blink_rate:.0f} / mnt", alert=st.blink_rate < low_b)

        snap = res.snapshot
        if snap is not None:
            self.val_score.setText(f"{snap.score:.0f}")
            lv = snap.level
            self.val_level.setText(lv.capitalize())
            self.val_level.setStyleSheet(
                f"background: {theme.LEVEL_SOFT[lv]}; color: {theme.LEVEL_COLORS[lv]};"
            )
            for key, (bar, lbl) in self.comp_bars.items():
                pts = float(snap.components.get(key, 0.0))
                bar.setValue(int(round(pts * 10)))
                lbl.setText(f"{pts:.0f}")
        else:
            min_w = int(self.cfg["score"]["min_window_sec"])
            rem = max(0, int(round(min_w - st.span_sec)))
            self.val_score.setText("–")
            self.val_level.setText(f"{rem} dtk" if rem > 0 else "Kosong")
            self.val_level.setStyleSheet(f"background: {theme.SURFACE}; color: {theme.TEXT_2};")

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt API)
        self.closed.emit()
        super().closeEvent(event)
