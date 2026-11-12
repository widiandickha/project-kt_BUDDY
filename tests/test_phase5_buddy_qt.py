"""FASE 5 (Qt, offscreen): SpriteAnimator, overlays, window flags, controller, sprite replacement."""

from __future__ import annotations

import shutil
import time
from pathlib import Path

import pytest
from PIL import Image
from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QImage, QPainter

from src.buddy import overlays
from src.buddy.buddy_state import BuddyState
from src.buddy.buddy_window import BuddyWindow
from src.buddy.controller import BuddyController
from src.buddy.sprite_player import ASSETS_DIR, SpriteAnimator, load_animations
from src.core.config import load_config
from src.core.events import BuddyEvent

CFG = load_config()


def advance(anim: SpriteAnimator, ms: float) -> None:
    """Pretend `ms` passed since the animation started, then run one timer tick."""
    anim._start -= ms / 1000.0
    anim._fade_start -= ms / 1000.0
    anim._tick()


def test_yaml_schema_has_12_animations() -> None:
    canvas, specs, default = load_animations(ASSETS_DIR / "animations.yaml")
    assert canvas == 512 and default == "idle" and set(specs) == {s.value for s in BuddyState}


def test_animator_uses_real_sprites(qapp) -> None:
    anim = SpriteAnimator(CFG["buddy"])
    assert anim.problems == [] and not anim.procedural
    assert anim.current() == "idle"


def test_non_loop_finishes_and_plays_next(qapp) -> None:
    anim = SpriteAnimator(CFG["buddy"])
    done: list[str] = []
    anim.finished.connect(done.append)
    anim.play("happy")
    advance(anim, 1300)
    assert done == ["happy"] and anim.current() == "idle"


def test_play_respects_priority_unless_interrupt(qapp) -> None:
    anim = SpriteAnimator(CFG["buddy"])
    anim.play("alert")
    assert anim.play("happy") is False and anim.current() == "alert"
    assert anim.play("happy", interrupt=True) is True and anim.current() == "happy"


def test_timer_rate_drops_when_idle(qapp) -> None:
    anim = SpriteAnimator(CFG["buddy"])
    anim.play("happy")
    assert anim._timer.interval() == anim.active_interval
    advance(anim, 2000)  # happy -> idle, crossfade over
    advance(anim, 500)
    assert anim._timer.interval() == anim.idle_interval


def test_every_expression_paints_something(qapp) -> None:
    anim = SpriteAnimator(CFG["buddy"])
    for st in BuddyState:
        anim.play(st.value, interrupt=True)
        anim._prev = None
        img = anim.render_image(128)
        assert img.pixelColor(64, 100).alpha() > 0, st.value


@pytest.mark.parametrize("name", sorted(overlays.OVERLAYS))
def test_overlays_draw(qapp, name: str) -> None:
    img = QImage(100, 100, QImage.Format.Format_ARGB32_Premultiplied)
    img.fill(Qt.GlobalColor.transparent)
    p = QPainter(img)
    overlays.draw_overlay(name, p, QRectF(0, 0, 100, 100), 0.5, 300.0)
    p.end()
    assert any(img.pixelColor(x, y).alpha() > 0 for x in range(0, 100, 2) for y in range(0, 100, 2))


def test_replacing_a_png_needs_no_code_change(qapp, tmp_path: Path) -> None:
    assets = tmp_path / "cat"
    shutil.copytree(ASSETS_DIR, assets)
    Image.new("RGBA", (512, 512), (10, 200, 30, 255)).save(assets / "idle.png")  # "real" sprite
    anim = SpriteAnimator(CFG["buddy"], assets_dir=assets)
    assert not anim.procedural
    anim._prev = None
    color = anim.render_image(64).pixelColor(32, 40)
    assert (color.red(), color.green()) == (10, 200)


def test_invalid_assets_fall_back_to_procedural(qapp, tmp_path: Path) -> None:
    assets = tmp_path / "cat"
    shutil.copytree(ASSETS_DIR, assets)
    (assets / "sleepy.png").unlink()
    Image.new("RGBA", (300, 300)).save(assets / "happy.png")
    anim = SpriteAnimator(CFG["buddy"], assets_dir=assets)
    assert anim.procedural and len(anim.problems) == 2
    assert anim.render_image(64).pixelColor(32, 50).alpha() > 0


def test_window_flags_and_size(qapp) -> None:
    w = BuddyWindow(CFG["buddy"])
    flags = w.windowFlags()
    for f in (Qt.WindowType.FramelessWindowHint, Qt.WindowType.WindowStaysOnTopHint, Qt.WindowType.Tool,
              Qt.WindowType.WindowDoesNotAcceptFocus):
        assert flags & f
    assert w.testAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
    assert 128 <= w.width() <= 192
    w.place()
    avail = w.screen_geometry()
    assert w.geometry().bottom() <= avail.bottom() + 1 and avail.contains(w.geometry().center())


def test_controller_event_flow(qapp) -> None:
    c = BuddyController(CFG, poll_fullscreen=False)
    shown = []
    c.advice_shown.connect(shown.append)
    adv = c.handle_event(BuddyEvent("need_break", 3, "need_break", time.time()))
    assert adv is not None and c.machine.state == BuddyState.ALERT and c.animator.current() == "alert"
    assert shown and c.bubble.isVisible()
    responses = []
    c.advice_response.connect(lambda k, r: responses.append((k, r)))
    c.bubble._finish("oke")
    assert responses == [("need_break", "oke")]
    assert c.machine.state == BuddyState.STRETCH  # accepted break advice -> stretch
    c.set_fullscreen(True)
    assert not c.window.isVisible()
    assert c.handle_event(BuddyEvent("bad_posture", 2, "bad_posture", time.time())) is None  # DND
    c.set_fullscreen(False)
    c.close()


def test_menu_actions_emit(qapp) -> None:
    c = BuddyController(CFG, poll_fullscreen=False)
    got = []
    c.pause_requested.connect(lambda m: got.append(("pause", m)))
    c.open_dashboard.connect(lambda: got.append("dash"))
    c.quit_requested.connect(lambda: got.append("quit"))
    for k in ("pause", "dashboard", "silence_today", "quit"):
        c.on_menu(k)
    assert got == [("pause", 30), "dash", "quit"] and c.advice.muted()
    c.close()


def test_click_reacts(qapp) -> None:
    c = BuddyController(CFG, poll_fullscreen=False)
    c.on_click()
    assert c.machine.state == BuddyState.HAPPY and c.bubble.key == "click"
    c.close()
