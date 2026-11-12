"""Runtime orchestration (spec FASE 7):

    CaptureThread -> InferenceThread (landmarks + pose + optional YOLO -> FrameFeatures)
                  -> AnalysisEngine (window, score, habits, burnout)  [tick once per second]
                  -> EventBus / BuddyEventMapper -> Buddy UI, and DbWriter -> SQLite

Threads: A capture, B inference (models are created inside it), the 1 s analysis tick runs in
the Qt main thread (QTimer) so Buddy updates are thread-safe. Inference adapts its frame skip
when FPS < runtime.min_fps, restarts its models after repeated errors, and is restarted by the
watchdog if the thread dies.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.analysis.calibration import Calibration
from src.analysis.engine import AnalysisEngine, TickResult
from src.analysis.feature_extractor import FeatureExtractor
from src.analysis.replay import compact
from src.buddy.event_mapper import BuddyEventMapper
from src.capture.camera import STATUS_ACTIVE, CaptureThread
from src.core.config import resolve_path
from src.core.events import EventBus, FrameFeatures, BuddyEvent
from src.storage.repository import Repository, StressRow

logger = logging.getLogger(__name__)

FPS_EMA = 0.1
RECOVER_FACTOR = 1.2


@dataclass
class Models:
    """Detectors used by the inference thread."""

    ear: Any
    pose: Any
    face: Any | None


def default_models(cfg: dict[str, Any]) -> Models:
    """Create MediaPipe + pose (+ YOLO face when models/best.pt exists)."""
    from src.detection.ear_calculator import EarCalculator
    from src.detection.pose_estimator import PoseEstimator

    face = None
    if resolve_path(cfg["det"]["model"]).exists():
        from src.detection.face_detector import FaceDetector

        face = FaceDetector(cfg)
    else:
        logger.info("models/best.pt belum ada: memakai landmark MediaPipe saja (D-016)")
    return Models(EarCalculator(cfg), PoseEstimator(cfg), face)


class InferenceThread(threading.Thread):
    """Thread B: newest frame -> FrameFeatures -> on_features callback."""

    def __init__(self, cfg: dict[str, Any], capture: CaptureThread, extractor: FeatureExtractor,
                 on_features: Callable[[FrameFeatures], None],
                 model_factory: Callable[[dict[str, Any]], Models] = default_models) -> None:
        super().__init__(name="InferenceThread", daemon=True)
        rt = cfg["runtime"]
        self.cfg, self.capture, self.extractor = cfg, capture, extractor
        self.on_features = on_features
        self.model_factory = model_factory
        self.min_fps = float(rt["min_fps"])
        self.target_fps = float(cfg["inference"]["target_fps"])
        self.max_pose_every, self.max_det_every = int(rt["max_pose_every"]), int(rt["max_det_every"])
        self.restart_after = int(rt["restart_after_errors"])
        self.performance_mode = bool(cfg["inference"]["performance_mode"])
        self.pose_every = int(rt["perf_pose_every"]) if self.performance_mode else 1
        self.det_every = int(rt["perf_det_every"]) if self.performance_mode else 1
        self.fps = 0.0
        self.errors = 0
        self.restarts = 0
        self.frames = 0
        self.det_available = False
        self._paused = threading.Event()
        self._stop_event = threading.Event()

    def pause(self) -> None:
        """Skip inference (fullscreen / pause) to save CPU."""
        self._paused.set()

    def resume(self) -> None:
        """Continue inference."""
        self._paused.clear()

    def stop(self) -> None:
        """End the thread."""
        self._stop_event.set()
        if self.is_alive():
            self.join(timeout=5.0)

    def _adapt(self) -> None:
        """Adaptive frame skip: run pose/YOLO less often when FPS drops below min_fps."""
        if self.fps and self.fps < self.min_fps:
            if self.pose_every < self.max_pose_every:
                self.pose_every += 1
            elif self.det_every < self.max_det_every:
                self.det_every += 1
        elif self.fps > self.target_fps * RECOVER_FACTOR and not self.performance_mode:
            self.pose_every = max(1, self.pose_every - 1)
            self.det_every = max(1, self.det_every - 1)

    def run(self) -> None:
        """Inference loop."""
        models = self.model_factory(self.cfg)
        self.det_available = models.face is not None
        pose_res, faces = None, None
        last = time.monotonic()
        while not self._stop_event.is_set():
            if self._paused.is_set():
                self._stop_event.wait(0.2)
                continue
            frame = self.capture.get_latest(timeout=0.5)
            if frame is None:
                continue
            try:
                self.frames += 1
                lm = models.ear.process(frame.image, frame.ts)
                if pose_res is None or self.frames % self.pose_every == 0:
                    pose_res = models.pose.process(frame.image)
                if models.face is not None and (faces is None or self.frames % self.det_every == 0):
                    faces = models.face.predict(frame.image)
                self.on_features(self.extractor.extract(time.time(), lm, faces, pose_res))
                now = time.monotonic()
                inst = 1.0 / max(1e-3, now - last)
                last = now
                self.fps = inst if self.fps == 0 else (1 - FPS_EMA) * self.fps + FPS_EMA * inst
                if self.frames % 30 == 0:
                    self._adapt()
                self.errors = 0
            except Exception:  # noqa: BLE001 - keep the app alive, restart models if it persists
                self.errors += 1
                logger.exception("Error inferensi (%d berturut-turut)", self.errors)
                if self.errors >= self.restart_after:
                    logger.warning("Memuat ulang model inferensi")
                    models = self.model_factory(self.cfg)
                    pose_res, faces = None, None
                    self.errors = 0
                    self.restarts += 1
        for m in (models.ear,):
            close = getattr(m, "close", None)
            if close:
                close()


@dataclass
class TickOutput:
    """What one analysis tick produced."""

    result: TickResult | None
    buddy_events: list[BuddyEvent] = field(default_factory=list)


class Pipeline:
    """Owns capture, inference, analysis, mapping and logging. Call tick() every second."""

    def __init__(self, cfg: dict[str, Any], repo: Repository, calibration: Calibration,
                 bus: EventBus | None = None, source: str | None = None, record_path: Path | None = None,
                 model_factory: Callable[[dict[str, Any]], Models] = default_models,
                 on_camera_status: Callable[[str, str], None] | None = None, source_fps: float | None = None) -> None:
        self.cfg, self.repo = cfg, repo
        self.bus = bus or EventBus()
        self.engine = AnalysisEngine(cfg, calibration, self.bus)
        self.extractor = FeatureExtractor(cfg, calibration)
        self.mapper = BuddyEventMapper(cfg)
        cam = cfg["camera"]
        self.capture = CaptureThread(
            source if source is not None else str(cam["index"]), int(cam["width"]), int(cam["height"]),
            float(source_fps or cam["fps"]), float(cam["retry_sec"]), int(cfg["runtime"]["queue_size"]),
            on_status=on_camera_status,
        )
        self.model_factory = model_factory
        self.inference = self._new_inference()
        self.log_every = float(cfg["storage"]["log_every_sec"])
        self.session_id: int | None = None
        self.feature_listener: Callable[[FrameFeatures], None] | None = None
        self.paused = False
        self.inference_paused = False
        self.watchdog_restarts = 0
        self._last_log = 0.0
        self._last_level: str | None = None
        self._burnout = False
        self._yawns = 0
        self._hands = 0
        self._record = record_path.open("a", encoding="utf-8") if record_path else None
        self._record_lock = threading.Lock()

    def _new_inference(self) -> InferenceThread:
        return InferenceThread(self.cfg, self.capture, self.extractor, self._on_features, self.model_factory)

    # -- lifecycle --------------------------------------------------------------------------
    def start(self) -> None:
        """Start a session and the worker threads."""
        self.session_id = self.repo.start_session()
        self.capture.start()
        self.inference.start()

    def stop(self) -> None:
        """Stop threads, close the session, flush the recording."""
        self.inference.stop()
        self.capture.stop()
        if self.session_id is not None:
            self.repo.end_session(self.session_id)
        if self._record:
            with self._record_lock:
                self._record.close()
                self._record = None

    def pause(self) -> None:
        """'Jeda': release the camera and stop inference."""
        self.paused = True
        self.capture.pause()
        self.inference.pause()

    def resume(self) -> None:
        """Continue after a pause."""
        self.paused = False
        self.engine.habits.reset_presence()
        self.capture.resume()
        if not self.inference_paused:
            self.inference.resume()

    def set_inference_paused(self, paused: bool) -> None:
        """Fullscreen/DND: keep the camera but skip inference to save CPU."""
        self.inference_paused = paused
        if paused:
            self.inference.pause()
        elif not self.paused:
            self.engine.habits.reset_presence()
            self.inference.resume()

    def set_calibration(self, calibration: Calibration) -> None:
        """Apply a new calibration and store it."""
        self.engine.set_calibration(calibration)
        self.extractor.set_calibration(calibration)
        self.repo.set_calibration(calibration.to_rows())

    @property
    def camera_active(self) -> bool:
        """True while frames are being captured."""
        return self.capture.status == STATUS_ACTIVE and not self.paused

    # -- per frame (inference thread) --------------------------------------------------------
    def _on_features(self, f: FrameFeatures) -> None:
        self.engine.add(f)
        listener = self.feature_listener
        if listener is not None:
            listener(f)
        if self._record is not None:
            with self._record_lock:
                if self._record is not None:
                    self._record.write(json.dumps(compact(f.to_dict()), separators=(",", ":")) + "\n")

    # -- per second (main thread) -------------------------------------------------------------
    def _watchdog(self) -> None:
        if not self.inference.is_alive() and not self.inference._stop_event.is_set():
            logger.error("InferenceThread berhenti tak terduga, dijalankan ulang")
            self.inference = self._new_inference()
            if self.paused or self.inference_paused:
                self.inference.pause()
            self.inference.start()
            self.watchdog_restarts += 1

    def tick(self, now: float | None = None) -> TickOutput:
        """Analysis + logging + Buddy events for this second."""
        self._watchdog()
        if self.paused or self.inference_paused:
            return TickOutput(None)
        now = now if now is not None else time.time()
        res = self.engine.tick(now)
        sid = self.session_id
        events: list[BuddyEvent] = []
        for p in res.presence:
            self.repo.add_event(now, sid, p.kind, 1, {"away_sec": round(p.away_sec, 1)})
            events += self.mapper.on_presence(p)
        for a in res.alerts:
            self.repo.add_event(now, sid, a.kind, 1 if a.positive else 2, a.detail)
            events += self.mapper.on_habit(a)
        snap = res.snapshot
        if snap is not None:
            if snap.level != self._last_level and self._last_level is not None:
                self.repo.add_event(now, sid, f"level_{snap.level}", 3 if snap.level == "tinggi" else 2, {"score": snap.score})
            self._last_level = snap.level
            if snap.burnout_risk and not self._burnout:
                self.repo.add_event(now, sid, "burnout_risk", 3, {"trend": round(snap.burnout_trend, 2)})
            self._burnout = snap.burnout_risk
            events += self.mapper.on_snapshot(snap)
        window = self.engine.window
        for _ in range(window.total_yawns - self._yawns):
            self.repo.add_event(now, sid, "yawn", 1, {})
        for _ in range(window.total_hand_episodes - self._hands):
            self.repo.add_event(now, sid, "hand_to_forehead", 1, {})
        self._yawns, self._hands = window.total_yawns, window.total_hand_episodes
        events += self.mapper.tick(now)

        if now - self._last_log >= self.log_every and sid is not None:
            self._last_log = now
            st = res.stats
            self.repo.add_stress(StressRow(
                ts=now, session_id=sid, score=snap.score if snap else None, level=snap.level if snap else None,
                perclos=round(st.perclos, 4), blink_rate=None if st.blink_rate is None else round(st.blink_rate, 2),
                yawn_count=st.yawn_count, furrow_ratio=round(st.furrow_ratio, 4), hand_count=st.hand_forehead_count,
                user_present=not self.engine.habits.away and st.present_ratio > 0,
            ))
        return TickOutput(res, events)
