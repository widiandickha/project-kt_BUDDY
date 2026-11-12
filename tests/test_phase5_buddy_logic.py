"""FASE 5 (no Qt): transforms, state machine, advice engine, event mapper, asset files."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
from PIL import Image

from src.buddy.advice_engine import AdviceEngine, load_bank, sentence_count, validate_bank
from src.buddy.buddy_state import PRIORITY, BuddyState, BuddyStateMachine
from src.buddy.event_mapper import BuddyEventMapper
from src.buddy.transforms import IDENTITY, TRANSFORMS, transform_at
from src.core.config import load_config
from src.core.events import HabitAlert, PresenceEvent, StressSnapshot

CFG = load_config()
ASSETS = Path(__file__).resolve().parents[1] / "src" / "buddy" / "assets" / "cat"
LOOPING = ("breathe", "breathe_slow", "sway", "nod")


# --- transforms -------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(TRANSFORMS))
def test_transforms_are_deterministic(name: str) -> None:
    for t in (0.0, 0.13, 0.5, 0.77, 1.0):
        assert transform_at(name, t, 1500) == transform_at(name, t, 1500)


@pytest.mark.parametrize("name", LOOPING)
def test_looping_transforms_join_at_t0_t1(name: str) -> None:
    a, b = transform_at(name, 0.0, 2400), transform_at(name, 1.0, 2400)
    for field in ("scale_x", "scale_y", "rotate_deg", "dx", "dy", "opacity"):
        assert getattr(a, field) == pytest.approx(getattr(b, field), abs=1e-6)


def test_transform_shapes() -> None:
    assert transform_at("breathe", 0.5, 2400).scale_y == pytest.approx(1.03)
    assert transform_at("breathe_slow", 0.5, 4000).scale_y == pytest.approx(1.02)
    assert transform_at("pop", 0.0, 1500).scale_x == pytest.approx(0.8)
    assert transform_at("pop", 1.0, 1500).scale_x == pytest.approx(1.0)
    assert min(transform_at("bounce", t / 100, 1200).dy for t in range(100)) == pytest.approx(-0.12, abs=0.01)
    assert transform_at("bounce", 0.49, 1200).scale_y == pytest.approx(0.92)  # landing squash
    assert transform_at("tilt", 0.5, 1600).rotate_deg == pytest.approx(-10.0)
    assert transform_at("stretch", 0.5, 2000).scale_x == pytest.approx(1.08)
    assert transform_at("slow_blink", 0.3, 2200).opacity == pytest.approx(1.0)
    assert max(abs(transform_at("wave", t / 100, 1800).rotate_deg) for t in range(100)) == pytest.approx(6.0, abs=0.1)
    assert transform_at("unknown", 0.3, 100) == IDENTITY


# --- assets -------------------------------------------------------------------------

def test_twelve_rgba_square_sprites_same_size() -> None:
    sizes = set()
    for st in BuddyState:
        img = Image.open(ASSETS / f"{st.value}.png")
        assert img.mode == "RGBA"
        assert img.width == img.height
        sizes.add(img.size)
    assert sizes == {(512, 512)}


# --- state machine ------------------------------------------------------------------

def test_stress_path_idle_worried_alert_and_back() -> None:
    m = BuddyStateMachine()
    assert m.handle("stress_sedang").current == BuddyState.WORRIED
    assert m.handle("stress_tinggi").current == BuddyState.ALERT
    assert m.animation_finished("alert").current == BuddyState.WORRIED  # alert pop returns to base
    assert m.handle("stress_calm").current == BuddyState.IDLE


def test_alert_advice_done_goes_idle() -> None:
    m = BuddyStateMachine()
    m.handle("need_break")
    assert m.handle("advice_done").current == BuddyState.IDLE


def test_sleep_and_wake() -> None:
    m = BuddyStateMachine()
    assert m.handle("user_away_long").current == BuddyState.SLEEPING
    assert m.handle("stress_tinggi") is None and m.handle("click") is None  # only user_returned wakes it
    assert m.handle("user_returned").current == BuddyState.WAKE_GREET
    assert m.animation_finished("wake_greet").current == BuddyState.IDLE


def test_priorities() -> None:
    m = BuddyStateMachine()
    m.handle("need_break")  # ALERT (3)
    assert m.handle("bad_posture") is None  # WORRIED (2) cannot cut ALERT
    assert m.handle("good_break") is None   # positive (1) cannot cut ALERT
    assert m.handle("burnout_risk").current == BuddyState.CONCERNED  # same priority 3
    m2 = BuddyStateMachine()
    m2.handle("burnout_risk")  # CONCERNED base
    assert m2.handle("sleepy") is None  # lower-priority mood does not replace it
    assert m2.handle("good_break").current == BuddyState.HAPPY  # short reaction is fine
    assert m2.animation_finished("happy").current == BuddyState.CONCERNED  # back to base


def test_click_always_reacts_and_returns() -> None:
    m = BuddyStateMachine()
    m.handle("stress_sedang")
    assert m.handle("click").current == BuddyState.HAPPY
    assert m.animation_finished("happy").current == BuddyState.WORRIED


def test_unknown_trigger_raises() -> None:
    with pytest.raises(ValueError):
        BuddyStateMachine().handle("dance")
    assert PRIORITY[BuddyState.ALERT] == PRIORITY[BuddyState.CONCERNED] == 3


# --- advice engine ------------------------------------------------------------------

def engine(**over) -> AdviceEngine:
    b = copy.deepcopy(CFG["buddy"])
    b.update(over)
    return AdviceEngine(b, seed=1)


def test_bank_rules() -> None:
    bank = load_bank()
    assert validate_bank(bank, CFG["buddy"]["max_chars"]) == []
    assert sentence_count("Sudah 50 menit nih. Berdiri sebentar yuk!") == 2
    bad = {"x": {"severity": 2, "expression": "idle", "lembut": ["a", "b", "c"], "tegas": ["Satu. Dua. Tiga."]}}
    assert len(validate_bank(bad, 90)) == 2


def test_key_and_global_cooldown() -> None:
    e = engine()
    assert e.request("bad_posture", 0) is not None
    assert e.request("bad_posture", 60) is None                 # per-key 15 min
    assert e.blocked_reason("too_close", 60) == "cooldown global"  # 3 min between bubbles
    assert e.request("too_close", 181) is not None
    assert e.request("bad_posture", 15 * 60 + 1) is not None


def test_severity3_skips_global_but_not_key_cooldown() -> None:
    e = engine()
    e.request("bad_posture", 0)
    assert e.request("need_break", 10) is not None
    assert e.request("need_break", 20) is None


def test_max_per_hour() -> None:
    e = engine(cooldown_global_min=0, cooldown_key_min=0)
    shown = sum(e.request("need_break", float(i)) is not None for i in range(20))
    assert shown == 8
    assert e.request("need_break", 3601.0) is not None


def test_escalation_after_two_ignores_and_reset() -> None:
    e = engine(cooldown_key_min=0, cooldown_global_min=0)
    assert e.request("dry_eyes", 0).tone == "lembut"
    e.respond("dry_eyes", "diabaikan", 1)
    e.respond("dry_eyes", "diabaikan", 2)
    assert e.request("dry_eyes", 3).tone == "tegas"
    e.respond("dry_eyes", "oke", 4)
    assert e.request("dry_eyes", 5).tone == "lembut"


def test_snooze_mute_today_and_global_mute() -> None:
    e = engine(cooldown_key_min=0, cooldown_global_min=0)
    e.request("bad_posture", 0)
    e.respond("bad_posture", "nanti", 0)
    assert e.request("bad_posture", 9 * 60) is None and e.request("bad_posture", 10 * 60 + 1) is not None
    e.respond("too_close", "jangan_hari_ini", 1000)
    assert e.blocked_reason("too_close", 1001) == "dimatikan hari ini"
    e.mute_until(5000)
    assert e.request("need_break", 4000) is None and e.request("need_break", 5001) is not None
    with pytest.raises(ValueError):
        e.respond("x", "maybe")


def test_variants_do_not_repeat_back_to_back() -> None:
    e = engine()
    texts = [e.pick_text("need_break") for _ in range(10)]
    assert all(a != b for a, b in zip(texts, texts[1:]))


# --- event mapper --------------------------------------------------------------------

def snap(ts: float, level: str = "rendah", **comp) -> StressSnapshot:
    return StressSnapshot(ts=ts, score=0, level=level, components=comp, burnout_trend=0.0)


def test_mapper_levels_calm_and_burnout() -> None:
    m = BuddyEventMapper(CFG)
    assert [e.kind for e in m.on_snapshot(snap(0, "sedang"))] == ["stress_sedang"]
    assert [e.kind for e in m.on_snapshot(snap(1, "tinggi"))] == ["stress_tinggi"]
    assert m.on_snapshot(snap(2, "rendah")) == []
    assert [e.kind for e in m.on_snapshot(snap(62, "rendah"))] == ["stress_calm"]
    s = snap(70)
    s.burnout_risk = True
    out = m.on_snapshot(s)
    assert [e.kind for e in out] == ["burnout_risk"] and out[0].severity == 3 and out[0].message_key == "burnout_risk"


def test_mapper_sleepy_edge() -> None:
    m = BuddyEventMapper(CFG)
    assert [e.kind for e in m.on_snapshot(snap(0, perclos=20.0))] == ["sleepy"]  # 20/30 >= 0.5
    assert m.on_snapshot(snap(1, perclos=20.0)) == []


def test_mapper_presence_and_celebrate() -> None:
    m = BuddyEventMapper(CFG)
    assert m.on_presence(PresenceEvent("user_away", 100.0, away_sec=10.0)) == []
    assert m.tick(150.0) == []
    sleep = m.tick(211.0)
    assert [e.kind for e in sleep] == ["user_away_long"] and sleep[0].message_key == ""
    assert [e.kind for e in m.on_presence(PresenceEvent("user_returned", 300.0, 200.0))] == ["user_returned"]
    kinds = []
    for i in range(4):
        kinds += [e.kind for e in m.on_habit(HabitAlert("good_break", 1_700_000_000.0 + i, positive=True))]
    assert kinds.count("celebrate") == 1
