import math
import time
from typing import Any, Dict, Optional
import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage

from src.analysis.calibration import BaselineCalibrator
from src.analysis.scoring import StressScoreEngine
from src.buddy.decision_engine import BuddyDecisionEngine
from src.database import DatabaseManager
from src.detectors.ear_landmark import FaceLandmarkAnalyzer
from src.detectors.face_yolo import CustomFaceYOLODetector
from src.detectors.pose_yolo import PoseAnalyzer


class CameraWorker(QThread):
    """Thread terpisah untuk membaca frame kamera, menjalankan detektor AI, dan mencatat ke SQLite."""

    frame_processed = Signal(QImage, dict)
    calibration_completed = Signal(dict)
    buddy_triggered = Signal(object, int)
    camera_error = Signal(str)
    status_changed = Signal(str)

    def __init__(
        self,
        config: Dict[str, Any],
        db: DatabaseManager,
        simulate: bool = False,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._config = config
        self._db = db
        self._simulate = simulate
        self._running = False
        self._paused = False
        self._recalibrate_requested = False
        self._cap: Optional[cv2.VideoCapture] = None

        cal_cfg = config.get("calibration", {})
        self.calibrator = BaselineCalibrator(
            duration_seconds=float(cal_cfg.get("duration_seconds", 10.0)),
            default_ear=float(cal_cfg.get("default_ear", 0.28)),
            default_mar=float(cal_cfg.get("default_mar", 0.15)),
            default_brow_ratio=float(cal_cfg.get("default_brow_ratio", 0.24)),
            default_posture_ratio=float(cal_cfg.get("default_posture_ratio", 0.55)),
        )
        self.scoring = StressScoreEngine(config)
        self.buddy_engine = BuddyDecisionEngine(config)
        self.session_id: Optional[int] = None

    def update_live_config(self, config: Dict[str, Any]) -> None:
        self._config = config
        self.scoring.update_config(config)
        self.buddy_engine.update_config(config)

    def set_paused(self, paused: bool) -> None:
        self._paused = paused
        if paused:
            self.status_changed.emit("paused")
        elif self.calibrator.is_calibrating:
            self.status_changed.emit("calibrating")
        else:
            self.status_changed.emit("simulation" if self._simulate else "active")

    @property
    def is_paused(self) -> bool:
        return self._paused

    @property
    def is_simulated(self) -> bool:
        return self._simulate

    def enable_simulation_fallback(self) -> None:
        self._simulate = True

    def open_camera_on_main_thread(self) -> bool:
        """Membuka kamera dari main thread agar macOS AVFoundation dapat meminta izin akses kamera."""
        if self._simulate:
            return False

        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None

        cam_cfg = self._config.get("camera", {})
        dev_idx = int(cam_cfg.get("device_index", 0))
        width = int(cam_cfg.get("frame_width", 640))
        height = int(cam_cfg.get("frame_height", 480))

        cap = cv2.VideoCapture(dev_idx, cv2.CAP_AVFOUNDATION)
        if not cap.isOpened():
            cap = cv2.VideoCapture(dev_idx)

        if not cap.isOpened():
            self.camera_error.emit(
                f"Kamera indeks {dev_idx} belum diizinkan oleh macOS. "
                "Berikan izin kamera untuk Terminal/IDE di System Settings > Privacy & Security > Camera, "
                "lalu klik tombol 'Gunakan Kamera Asli'."
            )
            self._simulate = True
            return False

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        # Warm up macOS AVFoundation sensor so the first worker read succeeds immediately.
        for _ in range(15):
            ok, frame = cap.read()
            if ok and frame is not None:
                break
            time.sleep(0.04)
        self._cap = cap
        self._simulate = False
        return True

    def request_recalibration(self) -> None:
        self._recalibrate_requested = True

    def snooze_buddy(self, minutes: int) -> None:
        self.buddy_engine.snooze(float(minutes))

    def stop_worker(self) -> None:
        self._running = False
        self.wait(3000)

    def _build_simulated_frame(self, step: int) -> tuple[np.ndarray, dict]:
        w, h = 640, 480
        canvas = np.full((h, w, 3), (235, 232, 224), dtype=np.uint8)
        phase = (step // 45) % 4

        if phase == 0:
            ear = 0.29 + 0.01 * math.sin(step * 0.3)
            if step % 28 in (0, 1):
                ear = 0.16
            mar = 0.15
            brow = 0.24
            posture = 0.56
            hand_to_head = False
            label_phase = "Simulasi: Kondisi Normal"
        elif phase == 1:
            ear = 0.17 if (step % 10 < 5) else 0.25
            mar = 0.62 if (step % 30 < 8) else 0.18
            brow = 0.23
            posture = 0.52
            hand_to_head = False
            label_phase = "Simulasi: Mengantuk & Menguap"
        elif phase == 2:
            ear = 0.26
            mar = 0.16
            brow = 0.18
            posture = 0.41
            hand_to_head = True
            label_phase = "Simulasi: Dahi Tegang & Tangan ke Kepala"
        else:
            ear = 0.18
            mar = 0.58 if (step % 20 < 6) else 0.19
            brow = 0.19
            posture = 0.39
            hand_to_head = (step % 15) < 7
            label_phase = "Simulasi: Kelelahan Tinggi"

        cx = int(w * 0.5 + 12 * math.sin(step * 0.17))
        cy = int(h * 0.44 + (24 if posture < 0.45 else 0))

        # Draw simulated upper body and face landmarks on the preview frame.
        cv2.ellipse(canvas, (cx, cy + 165), (130, 85), 0, 180, 360, (90, 95, 105), -1)
        cv2.circle(canvas, (cx, cy), 72, (205, 185, 165), -1)
        cv2.circle(canvas, (cx, cy), 72, (50, 45, 40), 2)

        eye_h = 3 if ear < 0.21 else 9
        cv2.ellipse(canvas, (cx - 26, cy - 10), (14, eye_h), 0, 0, 360, (30, 30, 30), -1)
        cv2.ellipse(canvas, (cx + 26, cy - 10), (14, eye_h), 0, 0, 360, (30, 30, 30), -1)

        brow_offset = -16 if brow < 0.21 else -24
        cv2.line(canvas, (cx - 42, cy + brow_offset), (cx - 12, cy + brow_offset + 4), (40, 35, 30), 3)
        cv2.line(canvas, (cx + 12, cy + brow_offset + 4), (cx + 42, cy + brow_offset), (40, 35, 30), 3)

        mouth_h = 20 if mar > 0.50 else 5
        cv2.ellipse(canvas, (cx, cy + 32), (18, mouth_h), 0, 0, 360, (120, 45, 45), -1)

        if hand_to_head:
            cv2.circle(canvas, (cx + 58, cy - 28), 20, (190, 165, 145), -1)
            cv2.circle(canvas, (cx + 58, cy - 28), 20, (50, 45, 40), 2)

        cv2.putText(
            canvas,
            label_phase,
            (18, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (28, 25, 23),
            2,
        )

        head_pt = (cx / float(w), cy / float(h))
        return canvas, {
            "ear": ear,
            "mar": mar,
            "brow_ratio": brow,
            "posture_ratio": posture,
            "hand_to_head": hand_to_head,
            "head_point": head_pt,
            "yolo_eye_closed": ear < 0.20,
            "yolo_yawn": mar > 0.50,
            "yolo_brow_furrow": brow < 0.20,
            "face_detected": True,
            "pose_detected": True,
        }

    def _draw_buddy_on_head(self, frame_bgr: np.ndarray, x1: int, y1: int, x2: int, expression: str = "istirahat") -> None:
        """Menggambar karakter kucing Virtual Buddy lucu di atas kepala pengguna."""
        h, w = frame_bgr.shape[:2]
        face_cx = int((x1 + x2) * 0.5)
        box_w = max(40, x2 - x1)
        cat_size = int(max(36, min(80, box_w * 0.38)))

        # Posisikan kucing tepat di atas batas atas kepala (y1)
        cat_cx = face_cx
        cat_cy = y1 - int(cat_size * 0.65)
        if cat_cy - cat_size < 10:
            cat_cy = max(cat_size, y1 + 10)

        # Warna kucing (oranye hangat, telinga krem, outline batu)
        fur = (54, 119, 217)       # BGR untuk #D97736
        inner_ear = (168, 198, 243) # BGR untuk #F3C6A8
        dark = (23, 25, 28)        # BGR untuk #1C1917
        muzzle = (235, 243, 250)   # BGR untuk #FAF3EB

        r = cat_size // 2

        # Telinga kucing
        ear_h = int(r * 0.75)
        pts_l = np.array([
            [cat_cx - int(r * 0.8), cat_cy - int(r * 0.2)],
            [cat_cx - int(r * 0.9), cat_cy - r - ear_h],
            [cat_cx - int(r * 0.2), cat_cy - int(r * 0.7)]
        ], np.int32)
        cv2.fillPoly(frame_bgr, [pts_l], fur)
        cv2.polylines(frame_bgr, [pts_l], True, dark, 2)

        pts_r = np.array([
            [cat_cx + int(r * 0.8), cat_cy - int(r * 0.2)],
            [cat_cx + int(r * 0.9), cat_cy - r - ear_h],
            [cat_cx + int(r * 0.2), cat_cy - int(r * 0.7)]
        ], np.int32)
        cv2.fillPoly(frame_bgr, [pts_r], fur)
        cv2.polylines(frame_bgr, [pts_r], True, dark, 2)

        # Telinga dalam (pinkish/krem)
        in_l = np.array([
            [cat_cx - int(r * 0.7), cat_cy - int(r * 0.3)],
            [cat_cx - int(r * 0.8), cat_cy - r - int(ear_h * 0.6)],
            [cat_cx - int(r * 0.3), cat_cy - int(r * 0.6)]
        ], np.int32)
        cv2.fillPoly(frame_bgr, [in_l], inner_ear)

        in_r = np.array([
            [cat_cx + int(r * 0.7), cat_cy - int(r * 0.3)],
            [cat_cx + int(r * 0.8), cat_cy - r - int(ear_h * 0.6)],
            [cat_cx + int(r * 0.3), cat_cy - int(r * 0.6)]
        ], np.int32)
        cv2.fillPoly(frame_bgr, [in_r], inner_ear)

        # Kepala bulat kucing
        cv2.circle(frame_bgr, (cat_cx, cat_cy), r, fur, -1)
        cv2.circle(frame_bgr, (cat_cx, cat_cy), r, dark, 2)

        # Moncong putih
        cv2.ellipse(frame_bgr, (cat_cx, cat_cy + int(r * 0.25)), (int(r * 0.42), int(r * 0.28)), 0, 0, 360, muzzle, -1)
        cv2.ellipse(frame_bgr, (cat_cx, cat_cy + int(r * 0.25)), (int(r * 0.42), int(r * 0.28)), 0, 0, 360, dark, 1)

        # Hidung segitiga kecil
        pts_nose = np.array([
            [cat_cx - 3, cat_cy + int(r * 0.15)],
            [cat_cx + 3, cat_cy + int(r * 0.15)],
            [cat_cx, cat_cy + int(r * 0.25)]
        ], np.int32)
        cv2.fillPoly(frame_bgr, [pts_nose], (18, 52, 154)) # BGR terakota

        # Mata kucing berdasarkan ekspresi
        eye_y = cat_cy - int(r * 0.12)
        eye_lx = cat_cx - int(r * 0.35)
        eye_rx = cat_cx + int(r * 0.35)

        if expression == "menguap":
            # Mata menyipit mengantuk (garis melengkung)
            cv2.ellipse(frame_bgr, (eye_lx, eye_y), (4, 3), 0, 180, 360, dark, 2)
            cv2.ellipse(frame_bgr, (eye_rx, eye_y), (4, 3), 0, 180, 360, dark, 2)
            # Mulut menguap bulat merah
            cv2.circle(frame_bgr, (cat_cx, cat_cy + int(r * 0.42)), 4, (27, 27, 153), -1)
        else:
            # Mata ceria (titik bulat dengan kilau putih)
            cv2.circle(frame_bgr, (eye_lx, eye_y), max(2, int(r * 0.14)), dark, -1)
            cv2.circle(frame_bgr, (eye_rx, eye_y), max(2, int(r * 0.14)), dark, -1)
            cv2.circle(frame_bgr, (eye_lx - 1, eye_y - 1), 1, (255, 255, 255), -1)
            cv2.circle(frame_bgr, (eye_rx - 1, eye_y - 1), 1, (255, 255, 255), -1)

        # Kumis kecil kiri dan kanan
        whisker_y = cat_cy + int(r * 0.25)
        cv2.line(frame_bgr, (cat_cx - int(r * 0.8), whisker_y - 2), (cat_cx - int(r * 0.4), whisker_y), dark, 1)
        cv2.line(frame_bgr, (cat_cx - int(r * 0.8), whisker_y + 4), (cat_cx - int(r * 0.4), whisker_y + 2), dark, 1)
        cv2.line(frame_bgr, (cat_cx + int(r * 0.4), whisker_y), (cat_cx + int(r * 0.8), whisker_y - 2), dark, 1)
        cv2.line(frame_bgr, (cat_cx + int(r * 0.4), whisker_y + 2), (cat_cx + int(r * 0.8), whisker_y + 4), dark, 1)

    def run(self) -> None:
        self._running = True
        cam_cfg = self._config.get("camera", {})
        models_cfg = self._config.get("models", {})
        analysis_cfg = self._config.get("analysis", {})

        target_fps = max(5, int(cam_cfg.get("target_fps", 15)))
        frame_interval = 1.0 / float(target_fps)
        log_interval = float(analysis_cfg.get("db_log_interval_seconds", 5.0))

        landmark_analyzer = FaceLandmarkAnalyzer()
        face_yolo = CustomFaceYOLODetector(
            model_rel_path=str(models_cfg.get("face_yolo_path", "models/face_expression_yolov8n.pt")),
            conf_threshold=float(models_cfg.get("confidence_threshold", 0.45)),
            use_two_class_fallback=bool(models_cfg.get("use_two_class_fallback", False)),
        )
        pose_analyzer = PoseAnalyzer(
            model_rel_path=str(models_cfg.get("pose_yolo_path", "models/yolov8n-pose.pt")),
            hand_to_head_threshold=float(analysis_cfg.get("hand_to_head_ratio_threshold", 0.48)),
        )

        if not self._simulate and (self._cap is None or not self._cap.isOpened()):
            self.camera_error.emit(
                "Kamera belum terbuka dari main thread. Beralih sementara ke Mode Simulasi."
            )
            self._simulate = True

        self.calibrator.start()
        b = self.calibrator.baseline
        self.session_id = self._db.start_session(b.ear, b.brow_ratio, b.posture_ratio)
        self.status_changed.emit("calibrating")

        last_db_log_ts = time.monotonic()
        sim_step = 0
        consecutive_failures = 0

        while self._running:
            t0 = time.monotonic()
            if self._paused:
                time.sleep(0.15)
                continue

            if self._recalibrate_requested:
                self._recalibrate_requested = False
                self.calibrator.start()
                self.status_changed.emit("calibrating")

            if self._simulate or self._cap is None:
                sim_step += 1
                frame_bgr, obs = self._build_simulated_frame(sim_step)
                ear = obs["ear"]
                mar = obs["mar"]
                brow_ratio = obs["brow_ratio"]
                posture_ratio = obs["posture_ratio"]
                hand_to_head = obs["hand_to_head"]
                head_pt = obs["head_point"]
                yolo_eye = obs["yolo_eye_closed"]
                yolo_yawn = obs["yolo_yawn"]
                yolo_brow = obs["yolo_brow_furrow"]
                face_ok = obs["face_detected"]
                pose_ok = obs["pose_detected"]
                yolo_loaded = face_yolo.is_ready
            else:
                ret, frame_bgr = self._cap.read()
                if not ret or frame_bgr is None:
                    consecutive_failures += 1
                    if consecutive_failures > 15:
                        self.camera_error.emit(
                            "Pembacaan frame kamera gagal berulang. Periksa izin kamera macOS lalu klik 'Gunakan Kamera Asli'."
                        )
                        self._simulate = True
                        self.status_changed.emit("simulation")
                    time.sleep(0.05)
                    continue
                consecutive_failures = 0

                # Flip horizontal agar tampilan kamera natural seperti cermin (mirror preview)
                frame_bgr = cv2.flip(frame_bgr, 1)

                lm = landmark_analyzer.analyze(frame_bgr)
                pose = pose_analyzer.analyze(frame_bgr)
                yolo_res = face_yolo.detect(frame_bgr)

                ear = lm.ear
                mar = lm.mar
                brow_ratio = lm.brow_ratio
                posture_ratio = pose.posture_ratio
                hand_to_head = pose.hand_to_head
                head_pt = pose.head_point or lm.head_point
                yolo_eye = yolo_res.mata_tertutup
                yolo_yawn = yolo_res.menguap
                yolo_brow = yolo_res.kerutan_dahi
                face_ok = lm.face_detected
                pose_ok = pose.pose_detected
                yolo_loaded = yolo_res.model_loaded

                if lm.bbox is not None:
                    x1, y1, x2, y2 = lm.bbox
                    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (22, 101, 52), 2)
                    # Gambar ikon kucing Virtual Buddy lucu tepat di atas kepala
                    cat_expr = "menguap" if yolo_yawn or mar > 0.48 else ("istirahat" if ear < 0.20 else "minum")
                    self._draw_buddy_on_head(frame_bgr, x1, y1, x2, expression=cat_expr)

            if self.calibrator.is_calibrating:
                just_done = self.calibrator.add_sample(
                    ear=ear,
                    mar=mar,
                    brow_ratio=brow_ratio,
                    posture_ratio=posture_ratio,
                    valid_face=face_ok,
                    valid_pose=pose_ok,
                )
                if just_done and self.session_id is not None:
                    base = self.calibrator.baseline
                    self._db.update_session_baseline(
                        self.session_id,
                        base.ear,
                        base.brow_ratio,
                        base.posture_ratio,
                    )
                    self.calibration_completed.emit(
                        {
                            "ear": base.ear,
                            "mar": base.mar,
                            "brow_ratio": base.brow_ratio,
                            "posture_ratio": base.posture_ratio,
                        }
                    )
                    self.status_changed.emit("simulation" if self._simulate else "active")

            summary = self.scoring.process_frame(
                baseline=self.calibrator.baseline,
                ear=ear,
                mar=mar,
                brow_ratio=brow_ratio,
                posture_ratio=posture_ratio,
                hand_to_head=hand_to_head,
                head_point=head_pt,
                yolo_eye_closed=yolo_eye,
                yolo_yawn=yolo_yawn,
                yolo_brow_furrow=yolo_brow,
            )

            now_ts = time.monotonic()
            if self.session_id is not None and (now_ts - last_db_log_ts) >= log_interval:
                last_db_log_ts = now_ts
                self._db.log_score(
                    session_id=self.session_id,
                    ear_value=summary.ear_current,
                    blink_rate_per_min=summary.blink_rate_per_min,
                    eye_closed_ratio=summary.eye_closed_ratio,
                    yawn_ratio=summary.yawn_ratio,
                    brow_furrow_ratio=summary.brow_furrow_ratio,
                    slouch_ratio=summary.slouch_ratio,
                    hand_to_head_ratio=summary.hand_to_head_ratio,
                    head_restlessness=summary.head_restlessness,
                    stress_score=summary.stress_score,
                    stress_level=summary.stress_level,
                    moving_avg_15m=summary.moving_avg_15m,
                    burnout_detected=summary.burnout_detected,
                )
                if summary.stress_level == "Tinggi":
                    self._db.log_event(
                        session_id=self.session_id,
                        event_type=summary.dominant_reason,
                        severity="Tinggi",
                        details=f"Skor stres mencapai {summary.stress_score:.1f}",
                    )

            if not self.calibrator.is_calibrating and self.session_id is not None:
                advice = self.buddy_engine.evaluate(summary, now_ts=now_ts)
                if advice is not None:
                    log_id = self._db.log_buddy(
                        session_id=self.session_id,
                        expression=advice.expression,
                        message=advice.message,
                        trigger_reason=advice.trigger_reason,
                    )
                    self.buddy_triggered.emit(advice, log_id)

            elapsed_ms = (time.monotonic() - t0) * 1000.0
            actual_fps = 1000.0 / max(1.0, max(elapsed_ms, frame_interval * 1000.0))

            rgb_view = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            qh, qw, ch = rgb_view.shape
            qimg = QImage(rgb_view.data, qw, qh, ch * qw, QImage.Format_RGB888).copy()

            payload = {
                "ear": summary.ear_current,
                "blink_rate": summary.blink_rate_per_min,
                "eye_closed_ratio": summary.eye_closed_ratio,
                "yawn_ratio": summary.yawn_ratio,
                "brow_furrow_ratio": summary.brow_furrow_ratio,
                "slouch_ratio": summary.slouch_ratio,
                "hand_to_head_ratio": summary.hand_to_head_ratio,
                "head_restlessness": summary.head_restlessness,
                "stress_score": summary.stress_score,
                "stress_level": summary.stress_level,
                "moving_avg_15m": summary.moving_avg_15m,
                "burnout_detected": summary.burnout_detected,
                "dominant_reason": summary.dominant_reason,
                "calibrating": self.calibrator.is_calibrating,
                "calibration_progress": self.calibrator.progress,
                "baseline_ear": self.calibrator.baseline.ear,
                "baseline_brow": self.calibrator.baseline.brow_ratio,
                "baseline_posture": self.calibrator.baseline.posture_ratio,
                "fps": actual_fps,
                "latency_ms": elapsed_ms,
                "yolo_custom_loaded": yolo_loaded,
                "snooze_remaining": self.buddy_engine.snooze_remaining_seconds,
            }
            self.frame_processed.emit(qimg, payload)

            sleep_dur = max(0.005, frame_interval - (time.monotonic() - t0))
            time.sleep(sleep_dur)

        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None
        landmark_analyzer.close()
        if self.session_id is not None:
            self._db.end_session(self.session_id)
