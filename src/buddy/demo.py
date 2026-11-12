"""Buddy demo with a debug panel (FASE 5 DoD).

    python -m src.buddy.demo                     # cat at the bottom of the screen + debug panel
    python -m src.buddy.demo --sheet out.png     # render all 12 expressions to an image and exit

The panel has one button per expression (12), buttons for every event/trigger, and a switch
to ignore advice cooldowns while testing.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from PySide6.QtCore import QRect, Qt, QTimer
from PySide6.QtGui import QImage, QPainter
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QGridLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from src.buddy.advice_engine import AdviceEngine
from src.buddy.buddy_state import BuddyState
from src.buddy.controller import BuddyController
from src.buddy.event_mapper import SEVERITY
from src.buddy.sprite_player import SpriteAnimator
from src.core.config import load_config
from src.core.events import BuddyEvent

EVENT_BUTTONS: tuple[str, ...] = (
    "stress_sedang", "stress_tinggi", "stress_calm", "burnout_risk", "need_break", "bad_posture", "too_close",
    "dry_eyes", "sleepy", "late_night", "good_break", "doing_great", "celebrate", "user_away_long", "user_returned",
)
SHEET_COLS = 4
SHEET_TILE = 192
SHEET_SETTLE_MS = 400


class DebugPanel(QWidget):
    """Buttons to trigger expressions and events."""

    def __init__(self, controller: BuddyController) -> None:
        super().__init__()
        self.setWindowTitle("Buddy debug panel")
        self.c = controller
        lay = QVBoxLayout(self)
        self.state_label = QLabel()
        lay.addWidget(self.state_label)

        expr_box = QGroupBox("Ekspresi (12)")
        grid = QGridLayout(expr_box)
        for i, st in enumerate(BuddyState):
            b = QPushButton(st.value)
            b.clicked.connect(lambda _=False, n=st.value: self.c.play_expression(n))
            grid.addWidget(b, i // 4, i % 4)
        lay.addWidget(expr_box)

        ev_box = QGroupBox("Event (state machine + saran)")
        grid2 = QGridLayout(ev_box)
        for i, kind in enumerate(EVENT_BUTTONS):
            b = QPushButton(kind)
            b.clicked.connect(lambda _=False, k=kind: self.fire(k))
            grid2.addWidget(b, i // 3, i % 3)
        lay.addWidget(ev_box)

        self.ignore_cd = QCheckBox("Abaikan cooldown saran (debug)")
        self.ignore_cd.setChecked(True)
        lay.addWidget(self.ignore_cd)
        fs = QPushButton("Simulasikan fullscreen on/off")
        fs.clicked.connect(lambda: self.c.set_fullscreen(not self.c.fullscreen))
        lay.addWidget(fs)
        self.log = QLabel("")
        self.log.setWordWrap(True)
        lay.addWidget(self.log)

        self.c.state_changed.connect(lambda a, b: self.refresh())
        self.c.advice_shown.connect(lambda adv: self.log.setText(f"Saran [{adv.key}/{adv.tone}]: {adv.text}"))
        self.c.advice_response.connect(lambda k, r: self.log.setText(f"Respons {k}: {r}"))
        self.c.window.menu_action.connect(lambda k: self.log.setText(f"Menu: {k}"))
        self.refresh()

    def fire(self, kind: str) -> None:
        """Send one BuddyEvent through the controller."""
        if self.ignore_cd.isChecked():
            self.c.advice = AdviceEngine(self.c.cfg["buddy"])
        key = "" if kind in ("stress_calm", "user_away_long") else kind
        adv = self.c.handle_event(BuddyEvent(kind, SEVERITY.get(kind, 1), key, time.time()))
        if adv is None and key:
            self.log.setText(f"{kind}: tidak ada bubble ({self.c.advice.blocked_reason(key, time.time())})")
        self.refresh()

    def refresh(self) -> None:
        """Show the state machine status."""
        m = self.c.machine
        self.state_label.setText(f"State: {m.state.value} (base: {m.base.value}) | animasi: {self.c.animator.current()}")


def render_sheet(path: Path) -> None:
    """Render every expression mid-animation into one contact sheet (headless check)."""
    cfg = load_config()
    anim = SpriteAnimator(cfg["buddy"])
    names = [s.value for s in BuddyState]
    rows = (len(names) + SHEET_COLS - 1) // SHEET_COLS
    sheet = QImage(SHEET_COLS * SHEET_TILE, rows * SHEET_TILE, QImage.Format.Format_ARGB32)
    sheet.fill(Qt.GlobalColor.white)
    p = QPainter(sheet)
    for i, name in enumerate(names):
        anim.play(name, interrupt=True)
        anim._start -= SHEET_SETTLE_MS / 1000.0  # sample a moment into the animation
        anim._prev = None
        x, y = (i % SHEET_COLS) * SHEET_TILE, (i // SHEET_COLS) * SHEET_TILE
        anim.paint(p, QRect(x, y, SHEET_TILE, SHEET_TILE))
        p.drawText(x + 6, y + 14, name)
    p.end()
    sheet.save(str(path))
    print(f"{len(names)} ekspresi dirender ke {path} (prosedural: {anim.procedural})")


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sheet", type=Path, help="render 12 ekspresi ke PNG lalu keluar")
    ap.add_argument("--seconds", type=float, default=None, help="tutup otomatis setelah N detik")
    args = ap.parse_args(argv)
    app = QApplication.instance() or QApplication(sys.argv)
    if args.sheet:
        render_sheet(args.sheet)
        return 0
    cfg = load_config()
    controller = BuddyController(cfg)
    controller.quit_requested.connect(app.quit)
    controller.show()
    panel = DebugPanel(controller)
    panel.show()
    if args.seconds:
        QTimer.singleShot(int(args.seconds * 1000), app.quit)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
