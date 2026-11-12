"""BuddyController: state machine + animator + advice engine + speech bubble + window.

Input: BuddyEvents (from BuddyEventMapper). Output Qt signals for the rest of the app:
advice_shown / advice_response (logged to the DB in FASE 6-7), menu requests, fullscreen changes.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta
from typing import Any

from PySide6.QtCore import QObject, QTimer, Signal

from src.buddy.advice_engine import Advice, AdviceEngine
from src.buddy.buddy_state import PRIORITY, BuddyState, BuddyStateMachine, Transition
from src.buddy.buddy_window import BuddyWindow
from src.buddy.fullscreen import other_app_fullscreen
from src.buddy.speech_bubble import SpeechBubble
from src.core.events import BuddyEvent

INTERRUPT_PRIORITY = 3
SEC_PER_MIN = 60.0


class BuddyController(QObject):
    """Owns the Buddy UI objects; all methods run in the Qt main thread."""

    advice_shown = Signal(object)          # Advice
    advice_response = Signal(str, str)     # key, response
    state_changed = Signal(str, str)       # previous, current
    pause_requested = Signal(int)          # minutes
    open_dashboard = Signal()
    open_settings = Signal()
    quit_requested = Signal()
    fullscreen_changed = Signal(bool)

    def __init__(self, cfg: dict[str, Any], window: BuddyWindow | None = None,
                 advice: AdviceEngine | None = None, poll_fullscreen: bool = True) -> None:
        super().__init__()
        b = cfg["buddy"]
        self.cfg = cfg
        self.pause_min = int(b["pause_min"])
        self.window = window or BuddyWindow(b)
        self.animator = self.window.animator
        self.machine = BuddyStateMachine()
        self.advice = advice or AdviceEngine(b)
        self.bubble = SpeechBubble(float(b["bubble_sec"]))
        self.fullscreen = False
        self.window.clicked.connect(self.on_click)
        self.window.menu_action.connect(self.on_menu)
        self.animator.finished.connect(self.on_animation_finished)
        self.bubble.responded.connect(self.on_response)
        self._poll = QTimer(self)
        self._poll.timeout.connect(self.check_fullscreen)
        if poll_fullscreen:
            self._poll.start(int(float(b["fullscreen_poll_sec"]) * 1000))

    def show(self) -> None:
        """Show the cat."""
        self.window.show()

    # -- state ----------------------------------------------------------------------
    def _apply(self, tr: Transition | None) -> None:
        if tr is None:
            return
        self.animator.play(tr.current.value, interrupt=PRIORITY[tr.current] >= INTERRUPT_PRIORITY or tr.trigger == "click")
        self.state_changed.emit(tr.previous.value, tr.current.value)

    def play_expression(self, name: str) -> None:
        """Debug: force an expression regardless of the state machine."""
        self.machine.state = BuddyState(name)
        self.animator.play(name, interrupt=True)

    def handle_event(self, ev: BuddyEvent) -> Advice | None:
        """React to one event: change state and maybe show advice."""
        self._apply(self.machine.handle(ev.kind))
        if not ev.message_key or self.fullscreen:
            return None
        adv = self.advice.request(ev.message_key, time.time())
        if adv is not None:
            self.show_advice(adv)
        return adv

    def show_advice(self, adv: Advice) -> None:
        """Display a bubble and announce it."""
        self.bubble.show_message(adv.key, adv.text, self.window.frameGeometry(), with_buttons=adv.severity >= 2)
        self.advice_shown.emit(adv)

    # -- user interaction --------------------------------------------------------------
    def on_click(self) -> None:
        """Left click: happy reaction + a meow (not rate limited)."""
        self._apply(self.machine.handle("click"))
        if "click" in self.advice.bank and not self.fullscreen:
            self.bubble.show_message("click", self.advice.pick_text("click"), self.window.frameGeometry(), with_buttons=False)

    def on_response(self, key: str, response: str) -> None:
        """Bubble answered or timed out."""
        if key == "click":
            return
        self.advice.respond(key, response, time.time())
        self.advice_response.emit(key, response)
        if response in ("oke", "ditutup") and self.machine.state == BuddyState.ALERT:
            self._apply(self.machine.handle("advice_done"))
        if key == "need_break" and response == "oke":
            self._apply(self.machine.handle("stretch"))

    def on_animation_finished(self, name: str) -> None:
        """Non-looping animation ended: return to the base state."""
        tr = self.machine.animation_finished(name)
        if tr is not None:
            self.animator.play(tr.current.value, interrupt=True)
            self.state_changed.emit(tr.previous.value, tr.current.value)

    def on_menu(self, key: str) -> None:
        """Right-click menu actions."""
        if key == "pause":
            self.advice.mute_until(time.time() + self.pause_min * SEC_PER_MIN)
            self.pause_requested.emit(self.pause_min)
        elif key == "silence_today":
            self.silence_today()
        elif key == "dashboard":
            self.open_dashboard.emit()
        elif key == "settings":
            self.open_settings.emit()
        elif key == "quit":
            self.quit_requested.emit()

    def silence_today(self) -> None:
        """'Diam hari ini': no advice until midnight."""
        tomorrow = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
        self.advice.mute_until(tomorrow.timestamp())
        self.bubble.dismiss()

    # -- Do Not Disturb --------------------------------------------------------------------
    def set_fullscreen(self, active: bool) -> None:
        """Hide the cat and hold advice while another app is fullscreen."""
        if active == self.fullscreen:
            return
        self.fullscreen = active
        if active:
            self.bubble.dismiss()
            self.window.hide()
        else:
            self.window.show()
        self.fullscreen_changed.emit(active)

    def check_fullscreen(self) -> None:
        """Poll the platform fullscreen detector."""
        self.set_fullscreen(other_app_fullscreen())

    def close(self) -> None:
        """Hide everything."""
        self._poll.stop()
        self.bubble.dismiss()
        self.window.close()
