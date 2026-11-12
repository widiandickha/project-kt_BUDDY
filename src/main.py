"""Stress Buddy entry point (spec FASE 7).

    python -m src.main                                   # webcam from config
    python -m src.main --source path/video.mp4           # video file or folder of images
    python -m src.main --record-features rec.jsonl       # also save FrameFeatures (numbers only)
    python -m src.main --skip-intro --seconds 60         # unattended test run

Flow: load config (+ data/settings.yaml), logging to logs/app.log, open DB (migrate, retention
cleanup, batching writer), first-run disclaimer, start capture + inference threads, show the
Buddy and tray, run the calibration wizard when no calibration exists, analyse every second.
On exit: stop threads, write today's summary, flush the DB.
"""

from __future__ import annotations

import argparse
import logging
import logging.handlers
import sys
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, QTimer, Signal
from PySide6.QtWidgets import QApplication, QMessageBox

from src.analysis.calibration import Calibration
from src.buddy.controller import BuddyController
from src.core.config import load_app_config, resolve_path, save_user_settings
from src.core.events import BuddyEvent, FrameFeatures
from src.core.pipeline import Pipeline
from src.storage import daily_summary as ds
from src.storage.db import Database, DbWriter
from src.storage.repository import Repository

logger = logging.getLogger("stress_buddy")

TICK_MS = 1000
MIDNIGHT_CHECK_MS = 60_000
LOG_MAX_BYTES = 1_000_000
LOG_BACKUPS = 3
DISCLAIMER_KEY = "disclaimer_ack"
DISCLAIMER = (
    "Stress Buddy adalah alat bantu kesadaran diri, bukan alat diagnosis medis.\n\n"
    "Kamera hanya diproses di perangkat ini. Tidak ada gambar atau video yang disimpan atau dikirim; "
    "yang dicatat hanya angka indikator (skor, kedipan, menguap, postur).\n\n"
    "Kamu bisa menjeda kamera kapan saja lewat ikon tray atau klik kanan pada kucing."
)


def setup_logging(cfg: dict[str, Any]) -> None:
    """Rotating file log (no frames, no personal data) + console."""
    path = resolve_path(cfg["logging"]["file"])
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.handlers.RotatingFileHandler(path, maxBytes=LOG_MAX_BYTES, backupCount=LOG_BACKUPS, encoding="utf-8")
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(threadName)s %(name)s: %(message)s")
    handler.setFormatter(fmt)
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    root = logging.getLogger()
    root.handlers[:] = [handler, console]
    root.setLevel(str(cfg["logging"]["level"]).upper())


class Bridge(QObject):
    """Moves callbacks from worker threads into the Qt main thread (queued signals)."""

    features = Signal(object)
    camera_status = Signal(str, str)


class App(QObject):
    """Wires pipeline, Buddy, tray, dashboard, settings and calibration together."""

    def __init__(self, qapp: QApplication, cfg: dict[str, Any], args: argparse.Namespace,
                 model_factory: Any = None) -> None:
        super().__init__()
        self.qapp, self.cfg, self.args = qapp, cfg, args
        st = cfg["storage"]
        self.db = Database(resolve_path(args.db or st["db_path"]))
        removed = self.db.cleanup(int(st["retention_days"]))
        logger.info("Retensi %s hari: %d baris lama dihapus", st["retention_days"], removed)
        self.writer = DbWriter(self.db, float(st["flush_sec"]))
        self.writer.start()
        self.repo = Repository(self.db, self.writer)
        cal = Calibration.from_rows(self.repo.get_calibration(), cfg)
        self.needs_calibration = cal is None
        self.bridge = Bridge()
        self.pipeline = Pipeline(
            cfg, self.repo, cal or Calibration.default(cfg), source=args.source,
            record_path=Path(args.record_features) if args.record_features else None,
            on_camera_status=self.bridge.camera_status.emit, source_fps=args.source_fps,
            **({"model_factory": model_factory} if model_factory else {}),
        )
        self.buddy = BuddyController(cfg)
        from src.ui.tray import Tray

        self.tray = Tray(bool(cfg["privacy"]["show_camera_indicator"]))
        self.dashboard = None
        self.wizard = None
        self._advice_ts: dict[str, float] = {}
        self._camera_lost = False
        self._today = date.today()
        self._quitting = False

        self.bridge.camera_status.connect(self.on_camera_status)
        self.bridge.features.connect(self._on_features_main)
        b = self.buddy
        b.advice_shown.connect(self.on_advice_shown)
        b.advice_response.connect(self.on_advice_response)
        b.pause_requested.connect(self.pause_for)
        b.fullscreen_changed.connect(self.on_fullscreen)
        b.open_dashboard.connect(self.show_dashboard)
        b.open_settings.connect(self.show_settings)
        b.quit_requested.connect(self.quit)
        self.tray.toggle_pause.connect(self.toggle_pause)
        self.tray.open_dashboard.connect(self.show_dashboard)
        self.tray.open_settings.connect(self.show_settings)
        self.tray.recalibrate.connect(self.show_calibration)
        self.tray.quit_requested.connect(self.quit)
        self.qapp.aboutToQuit.connect(self.shutdown)

        self.tick_timer = QTimer(self)
        self.tick_timer.timeout.connect(self.tick)
        self.midnight_timer = QTimer(self)
        self.midnight_timer.timeout.connect(self.check_midnight)
        self.resume_timer = QTimer(self)
        self.resume_timer.setSingleShot(True)
        self.resume_timer.timeout.connect(self.resume)

    # -- startup ------------------------------------------------------------------------------
    def start(self) -> None:
        """Disclaimer, threads, UI, timers, calibration."""
        if self.repo.get_state(DISCLAIMER_KEY) != "1":
            if not self.args.skip_intro:
                QMessageBox.information(None, "Tentang Stress Buddy", DISCLAIMER)
            self.repo.set_state(DISCLAIMER_KEY, "1")
        self.pipeline.start()
        self.buddy.show()
        self.tick_timer.start(TICK_MS)
        self.midnight_timer.start(MIDNIGHT_CHECK_MS)
        if self.needs_calibration and not self.args.skip_intro:
            QTimer.singleShot(1500, self.show_calibration)
        if self.args.seconds:
            QTimer.singleShot(int(self.args.seconds * 1000), self.quit)
        logger.info("Stress Buddy berjalan (YOLO wajah: %s)", "ya" if self.pipeline.inference.det_available else "belum")

    # -- periodic -----------------------------------------------------------------------------
    def tick(self) -> None:
        """Once per second: analysis, logging, Buddy reactions."""
        out = self.pipeline.tick()
        for ev in out.buddy_events:
            self.buddy.handle_event(ev)

    def check_midnight(self) -> None:
        """Write yesterday's summary when the date changes."""
        if date.today() != self._today:
            self.repo.flush()
            ds.aggregate(self.repo, self._today, self.cfg)
            self._today = date.today()

    # -- camera / pause ------------------------------------------------------------------------
    def on_camera_status(self, status: str, message: str) -> None:
        """Tray indicator; Buddy sleeps while the camera is unavailable."""
        if self.pipeline.inference_paused and status == "active":
            status = "fullscreen"
        self.tray.set_camera_state(status, message)
        now = time.time()
        if status == "unavailable" and not self._camera_lost:
            self._camera_lost = True
            self.buddy.handle_event(BuddyEvent("user_away_long", 1, "", now))
        elif status == "active" and self._camera_lost:
            self._camera_lost = False
            self.buddy.handle_event(BuddyEvent("user_returned", 1, "", now))

    def toggle_pause(self) -> None:
        """Tray: pause/resume the camera."""
        if self.pipeline.paused:
            self.resume()
        else:
            self.pause_for(0)

    def pause_for(self, minutes: int) -> None:
        """Release the camera; resume automatically after `minutes` (0 = until resumed)."""
        self.pipeline.pause()
        self.tray.set_camera_state("paused")
        if minutes > 0:
            self.resume_timer.start(minutes * 60_000)

    def resume(self) -> None:
        """Re-open the camera."""
        self.resume_timer.stop()
        self.buddy.advice.unmute()
        self.pipeline.resume()

    def on_fullscreen(self, active: bool) -> None:
        """Another app is fullscreen: stop inference to save CPU."""
        self.pipeline.set_inference_paused(active)
        self.tray.set_camera_state("fullscreen" if active else self.pipeline.capture.status)

    # -- advice logging ------------------------------------------------------------------------
    def on_advice_shown(self, adv: Any) -> None:
        """Log the bubble."""
        self._advice_ts[adv.key] = adv.ts
        self.repo.add_advice(adv.ts, adv.key, adv.expression)

    def on_advice_response(self, key: str, response: str) -> None:
        """Log the user's response."""
        ts = self._advice_ts.pop(key, None)
        if ts is not None:
            self.repo.set_advice_response(ts, key, response)

    # -- windows -------------------------------------------------------------------------------
    def show_dashboard(self) -> None:
        """Open (or refresh) the dashboard."""
        from src.ui.dashboard import Dashboard

        if self.dashboard is None:
            self.dashboard = Dashboard(self.repo, self.cfg)
        else:
            self.dashboard.refresh()
        self.dashboard.show()
        self.dashboard.raise_()

    def show_settings(self) -> None:
        """Open the settings dialog."""
        from src.ui.settings_dialog import SettingsDialog

        dlg = SettingsDialog(self.cfg)
        dlg.saved.connect(self.apply_settings)
        dlg.recalibrate.connect(lambda: QTimer.singleShot(0, self.show_calibration))
        dlg.delete_all.connect(self.delete_all)
        dlg.exec()

    def apply_settings(self, changes: dict[str, Any]) -> None:
        """Persist settings; apply the ones that work live."""
        save_user_settings(changes)
        self.cfg = load_app_config()
        self.pipeline.engine.habits.work_break = float(self.cfg["habit"]["work_break_min"]) * 60.0
        self.buddy.animator.overlays_enabled = bool(self.cfg["buddy"]["overlays_enabled"])
        logger.info("Pengaturan disimpan")

    def delete_all(self) -> None:
        """'Hapus semua data' (keeps the disclaimer flag so it is not shown again)."""
        self.repo.flush()
        self.db.delete_all()
        self.repo.set_state(DISCLAIMER_KEY, "1")
        logger.info("Semua data dihapus atas permintaan pengguna")

    def show_calibration(self) -> None:
        """Run the calibration wizard with live features."""
        from src.ui.calibration_wizard import CalibrationWizard

        if self.wizard is not None:
            return
        self.wizard = CalibrationWizard(self.cfg)
        self.pipeline.feature_listener = self.bridge.features.emit
        self.wizard.finished_calibration.connect(self.on_calibrated)
        self.wizard.exec()
        self.pipeline.feature_listener = None
        self.wizard = None

    def _on_features_main(self, f: FrameFeatures) -> None:
        if self.wizard is not None:
            self.wizard.add_features(f)

    def on_calibrated(self, cal: Calibration | None) -> None:
        """Store and apply a new calibration (None = keep defaults)."""
        if cal is None:
            logger.info("Kalibrasi dilewati, memakai nilai bawaan")
            return
        self.pipeline.set_calibration(cal)
        self.needs_calibration = False
        logger.info("Kalibrasi tersimpan (ambang EAR %.3f)", cal.ear_threshold)

    # -- shutdown ------------------------------------------------------------------------------
    def quit(self) -> None:
        """Quit the application."""
        self.qapp.quit()

    def shutdown(self) -> None:
        """Clean exit: stop threads, summarise today, flush the DB."""
        if self._quitting:
            return
        self._quitting = True
        self.tick_timer.stop()
        self.midnight_timer.stop()
        self.pipeline.stop()
        self.repo.flush()
        try:
            ds.aggregate(self.repo, date.today(), self.cfg)
            if date.today() != self._today:
                ds.aggregate(self.repo, date.today() - timedelta(days=1), self.cfg)
        except Exception:  # noqa: BLE001 - never block exit on a summary error
            logger.exception("Ringkasan harian gagal dibuat")
        self.writer.stop()
        self.buddy.close()
        self.tray.hide()
        self.db.close()
        logger.info("Stress Buddy ditutup dengan bersih")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Command-line arguments."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default=None, help="indeks kamera, file video, atau folder gambar")
    ap.add_argument("--source-fps", type=float, default=None, help="kecepatan putar untuk file/folder (default camera.fps)")
    ap.add_argument("--record-features", default=None, help="simpan FrameFeatures ke .jsonl (angka saja)")
    ap.add_argument("--db", default=None, help="path database alternatif")
    ap.add_argument("--seconds", type=float, default=None, help="keluar otomatis setelah N detik")
    ap.add_argument("--skip-intro", action="store_true", help="tanpa dialog disclaimer & wizard kalibrasi")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    args = parse_args(argv)
    cfg = load_app_config()
    setup_logging(cfg)
    qapp = QApplication.instance() or QApplication(sys.argv)
    qapp.setApplicationName("Stress Buddy")
    qapp.setQuitOnLastWindowClosed(False)
    app = App(qapp, cfg, args)
    app.start()
    code = qapp.exec()
    app.shutdown()
    return code


if __name__ == "__main__":
    sys.exit(main())
