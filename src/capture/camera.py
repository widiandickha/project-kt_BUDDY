"""CaptureThread (spec 4.1 thread A): reads frames into a queue.Queue(maxsize=2), always
keeping the newest (old frames are dropped). Pause releases the camera device completely.
If the camera is missing or busy, it retries every camera.retry_sec.

Sources: camera index ("0"), a video file, or a folder of images (played at camera.fps,
looping) for repeatable tests. Frames live only in memory.
"""

from __future__ import annotations

import logging
import queue
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
STATUS_ACTIVE = "active"
STATUS_PAUSED = "paused"
STATUS_UNAVAILABLE = "unavailable"
STATUS_STOPPED = "stopped"


@dataclass
class Frame:
    """A captured frame with its monotonic timestamp."""

    ts: float
    image: np.ndarray


class _FolderSource:
    """Images of a folder, looping, paced at fps."""

    def __init__(self, folder: Path, fps: float) -> None:
        self.files = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        self.dt = 1.0 / max(fps, 1e-3)
        self.i = 0
        self._next = time.monotonic()

    def isOpened(self) -> bool:  # noqa: N802 (cv2-like API)
        return bool(self.files)

    def read(self) -> tuple[bool, np.ndarray | None]:
        wait = self._next - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        self._next = max(self._next + self.dt, time.monotonic())
        img = cv2.imread(str(self.files[self.i % len(self.files)]))
        self.i += 1
        return img is not None, img

    def release(self) -> None:
        self.files = []


class CaptureThread(threading.Thread):
    """Producer thread. Consumers call get_latest()."""

    def __init__(self, source: str, width: int, height: int, fps: float, retry_sec: float,
                 queue_size: int = 2, on_status: Callable[[str, str], None] | None = None,
                 mirror: bool = True) -> None:
        super().__init__(name="CaptureThread", daemon=True)
        self.source = source
        self.width, self.height, self.fps = width, height, fps
        self.retry_sec = retry_sec
        self.mirror = mirror and source.isdigit()
        self.frames: queue.Queue[Frame] = queue.Queue(maxsize=queue_size)
        self.on_status = on_status or (lambda status, msg: None)
        self._stop_event = threading.Event()
        self._paused = threading.Event()
        self._cap = None
        self.status = STATUS_STOPPED
        self.dropped = 0
        self.captured = 0

    # -- control ------------------------------------------------------------------------
    def pause(self) -> None:
        """Stop capturing and release the device (camera light off)."""
        self._paused.set()

    def resume(self) -> None:
        """Re-open the device and continue."""
        self._paused.clear()

    @property
    def paused(self) -> bool:
        """True while paused."""
        return self._paused.is_set()

    def stop(self) -> None:
        """End the thread and release the device."""
        self._stop_event.set()
        self._paused.clear()
        if self.is_alive():
            self.join(timeout=3.0)
        self._release()

    # -- internals ----------------------------------------------------------------------
    def _set_status(self, status: str, msg: str = "") -> None:
        if status != self.status:
            self.status = status
            logger.info("Kamera: %s %s", status, msg)
            self.on_status(status, msg)

    def _open(self):
        src = self.source
        if src.isdigit():
            backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
            cap = cv2.VideoCapture(int(src), backend)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        elif Path(src).is_dir():
            cap = _FolderSource(Path(src), self.fps)
        else:
            cap = cv2.VideoCapture(src)
        return cap if cap.isOpened() else None

    def _release(self) -> None:
        if self._cap is not None:
            try:
                self._cap.release()
            finally:
                self._cap = None

    def _push(self, frame: Frame) -> None:
        while True:
            try:
                self.frames.put_nowait(frame)
                return
            except queue.Full:
                try:
                    self.frames.get_nowait()  # drop the oldest, keep the newest
                    self.dropped += 1
                except queue.Empty:
                    pass

    def get_latest(self, timeout: float = 0.5) -> Frame | None:
        """Newest frame or None if none arrived within timeout."""
        try:
            return self.frames.get(timeout=timeout)
        except queue.Empty:
            return None

    def run(self) -> None:
        """Capture loop with pause/retry handling."""
        is_file = not self.source.isdigit() and not Path(self.source).is_dir()
        min_dt = 1.0 / self.fps if is_file else 0.0
        while not self._stop_event.is_set():
            if self._paused.is_set():
                self._release()
                self._set_status(STATUS_PAUSED)
                self._stop_event.wait(0.2)
                continue
            if self._cap is None:
                self._cap = self._open()
                if self._cap is None:
                    self._set_status(STATUS_UNAVAILABLE, f"mencoba lagi dalam {self.retry_sec:.0f} detik")
                    self._stop_event.wait(self.retry_sec)
                    continue
                self._set_status(STATUS_ACTIVE)
            t0 = time.monotonic()
            ok, img = self._cap.read()
            if not ok or img is None:
                if is_file:  # loop video files
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                self._release()
                self._set_status(STATUS_UNAVAILABLE, "frame gagal dibaca")
                self._stop_event.wait(self.retry_sec)
                continue
            if self.mirror:
                img = cv2.flip(img, 1)
            self.captured += 1
            self._push(Frame(time.monotonic(), img))
            if min_dt:
                time.sleep(max(0.0, min_dt - (time.monotonic() - t0)))
        self._release()
        self._set_status(STATUS_STOPPED)
