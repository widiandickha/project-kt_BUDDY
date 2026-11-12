"""Sprite transforms (Lampiran A.4): pure functions of normalized time t (0..1).

Every transform returns Transform(scale_x, scale_y, rotate_deg, dx, dy, opacity). Pivot is the
bottom-center of the sprite; dx/dy are fractions of the sprite height (negative dy = up).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class Transform:
    """Geometric + opacity state of the sprite at one instant."""

    scale_x: float = 1.0
    scale_y: float = 1.0
    rotate_deg: float = 0.0
    dx: float = 0.0
    dy: float = 0.0
    opacity: float = 1.0


IDENTITY = Transform()


def _ease_out(u: float) -> float:
    return 1.0 - (1.0 - u) ** 2


def _ease_in(u: float) -> float:
    return u * u


def _ease_in_out(u: float) -> float:
    return 0.5 - 0.5 * math.cos(math.pi * u)


def _clamp01(u: float) -> float:
    return 0.0 if u < 0 else 1.0 if u > 1 else u


def breathe(t: float, duration_ms: float) -> Transform:
    """scale_y 1.00 -> 1.03 -> 1.00 (sine)."""
    return Transform(scale_y=1.0 + 0.03 * math.sin(math.pi * t))


def breathe_slow(t: float, duration_ms: float) -> Transform:
    """Like breathe, amplitude 0.02."""
    return Transform(scale_y=1.0 + 0.02 * math.sin(math.pi * t))


BOUNCE_HEIGHT = 0.12
SQUASH_MS = 80.0


def bounce(t: float, duration_ms: float) -> Transform:
    """Two jumps: rise to 0.12 (ease-out), fall (ease-in), 80 ms squash on landing."""
    half = duration_ms / 2.0
    u_ms = (t * duration_ms) % half
    air = max(1.0, half - SQUASH_MS)
    if u_ms >= air:  # landing squash
        return Transform(scale_x=1.06, scale_y=0.92)
    u = u_ms / air
    h = _ease_out(u / 0.5) if u < 0.5 else 1.0 - _ease_in((u - 0.5) / 0.5)
    return Transform(dy=-BOUNCE_HEIGHT * h)


def sway(t: float, duration_ms: float) -> Transform:
    """rotate ±4° (sine)."""
    return Transform(rotate_deg=4.0 * math.sin(2.0 * math.pi * t))


def nod(t: float, duration_ms: float) -> Transform:
    """0 -> +6° slowly, back quickly; dy +0.02 with the same envelope."""
    env = _ease_in_out(t / 0.75) if t < 0.75 else 1.0 - _ease_out((t - 0.75) / 0.25)
    return Transform(rotate_deg=6.0 * env, dy=0.02 * env)


POP_MS = 250.0


def pop(t: float, duration_ms: float) -> Transform:
    """scale 0.80 -> 1.10 -> 1.00 within 250 ms, then still."""
    u = _clamp01(t * duration_ms / POP_MS)
    s = 0.8 + 0.3 * _ease_out(u / 0.6) if u < 0.6 else 1.1 - 0.1 * _ease_in_out((u - 0.6) / 0.4)
    return Transform(scale_x=s, scale_y=s)


TILT_MS = 300.0


def tilt(t: float, duration_ms: float) -> Transform:
    """rotate 0 -> -10° (ease-out 300 ms), hold, back at the end."""
    ms = t * duration_ms
    k = TILT_MS / max(duration_ms, 1.0)
    if t < k:
        a = _ease_out(ms / TILT_MS)
    elif t > 1.0 - k:
        a = 1.0 - _ease_in_out((ms - (duration_ms - TILT_MS)) / TILT_MS)
    else:
        a = 1.0
    return Transform(rotate_deg=-10.0 * a)


STRETCH_HOLD_MS = 500.0


def stretch(t: float, duration_ms: float) -> Transform:
    """scale_x 1.00 -> 1.08 and scale_y 1.00 -> 0.94, hold 500 ms, back."""
    ramp = max(1.0, (duration_ms - STRETCH_HOLD_MS) / 2.0)
    ms = t * duration_ms
    if ms < ramp:
        a = _ease_in_out(ms / ramp)
    elif ms < ramp + STRETCH_HOLD_MS:
        a = 1.0
    else:
        a = 1.0 - _ease_in_out(_clamp01((ms - ramp - STRETCH_HOLD_MS) / ramp))
    return Transform(scale_x=1.0 + 0.08 * a, scale_y=1.0 - 0.06 * a)


BLINK_FADE_MS = 300.0
BLINK_HOLD_MS = 800.0


def slow_blink(t: float, duration_ms: float) -> Transform:
    """Crossfade idle -> this pose (300 ms), hold 800 ms, back to idle. opacity = weight of this pose."""
    ms = t * duration_ms
    if ms < BLINK_FADE_MS:
        a = ms / BLINK_FADE_MS
    elif ms < BLINK_FADE_MS + BLINK_HOLD_MS:
        a = 1.0
    else:
        a = 1.0 - _clamp01((ms - BLINK_FADE_MS - BLINK_HOLD_MS) / BLINK_FADE_MS)
    return Transform(opacity=a)


def wave(t: float, duration_ms: float) -> Transform:
    """rotate back and forth ±6° quickly, 3 times."""
    return Transform(rotate_deg=6.0 * math.sin(2.0 * math.pi * 3.0 * t))


def none(t: float, duration_ms: float) -> Transform:
    """No movement."""
    return IDENTITY


TRANSFORMS: dict[str, Callable[[float, float], Transform]] = {
    "breathe": breathe,
    "breathe_slow": breathe_slow,
    "bounce": bounce,
    "sway": sway,
    "nod": nod,
    "pop": pop,
    "tilt": tilt,
    "stretch": stretch,
    "slow_blink": slow_blink,
    "wave": wave,
    "none": none,
}


def transform_at(name: str | None, t: float, duration_ms: float) -> Transform:
    """Look up a transform by name (unknown/None = identity)."""
    fn = TRANSFORMS.get(name or "none", none)
    return fn(_clamp01(t), float(duration_ms))
