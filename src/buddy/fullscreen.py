"""Best-effort detection of another application running fullscreen (game / presentation).

macOS: Quartz window list (pyobjc-framework-Quartz) - a normal-layer window of another
process covering a whole display. Windows: foreground window rect == its monitor rect.
Other platforms or missing libraries: always False (feature disabled, logged once).
"""

from __future__ import annotations

import logging
import os
import sys

logger = logging.getLogger(__name__)
_warned = False
TOLERANCE_PX = 2


def _warn_once(msg: str) -> None:
    global _warned
    if not _warned:
        logger.info(msg)
        _warned = True


def covers(window: tuple[float, float, float, float], screen: tuple[float, float, float, float]) -> bool:
    """True when window (x, y, w, h) covers screen (x, y, w, h) within a small tolerance."""
    wx, wy, ww, wh = window
    sx, sy, sw, sh = screen
    return (wx <= sx + TOLERANCE_PX and wy <= sy + TOLERANCE_PX
            and wx + ww >= sx + sw - TOLERANCE_PX and wy + wh >= sy + sh - TOLERANCE_PX)


def _mac() -> bool:
    try:
        import Quartz  # type: ignore[import-not-found]
    except ImportError:
        _warn_once("pyobjc-framework-Quartz tidak terpasang: deteksi fullscreen nonaktif")
        return False
    displays = Quartz.CGGetActiveDisplayList(16, None, None)[1] or []
    screens = []
    for d in displays:
        b = Quartz.CGDisplayBounds(d)
        screens.append((b.origin.x, b.origin.y, b.size.width, b.size.height))
    wins = Quartz.CGWindowListCopyWindowInfo(
        Quartz.kCGWindowListOptionOnScreenOnly | Quartz.kCGWindowListExcludeDesktopElements, Quartz.kCGNullWindowID
    ) or []
    me = os.getpid()
    for w in wins:
        if w.get("kCGWindowLayer", 1) != 0 or w.get("kCGWindowOwnerPID") == me:
            continue
        b = w.get("kCGWindowBounds") or {}
        rect = (b.get("X", 0), b.get("Y", 0), b.get("Width", 0), b.get("Height", 0))
        if any(covers(rect, s) for s in screens):
            return True
    return False


def _windows() -> bool:
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32  # type: ignore[attr-defined]
    hwnd = user32.GetForegroundWindow()
    if not hwnd or hwnd in (user32.GetDesktopWindow(), user32.GetShellWindow()):
        return False
    rect = wintypes.RECT()
    user32.GetWindowRect(hwnd, ctypes.byref(rect))

    class MONITORINFO(ctypes.Structure):
        _fields_ = [("cbSize", wintypes.DWORD), ("rcMonitor", wintypes.RECT), ("rcWork", wintypes.RECT), ("dwFlags", wintypes.DWORD)]

    info = MONITORINFO()
    info.cbSize = ctypes.sizeof(MONITORINFO)
    user32.GetMonitorInfoW(user32.MonitorFromWindow(hwnd, 2), ctypes.byref(info))
    m = info.rcMonitor
    return covers((rect.left, rect.top, rect.right - rect.left, rect.bottom - rect.top),
                  (m.left, m.top, m.right - m.left, m.bottom - m.top))


def other_app_fullscreen() -> bool:
    """True if another application appears to be fullscreen right now."""
    try:
        if sys.platform == "darwin":
            return _mac()
        if sys.platform.startswith("win"):
            return _windows()
    except Exception:  # noqa: BLE001 - detection must never crash the app
        logger.exception("Deteksi fullscreen gagal")
        return False
    _warn_once("Deteksi fullscreen tidak didukung di platform ini")
    return False
