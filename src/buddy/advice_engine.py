"""Advice selection with spam control (spec 5.6).

* Bank: assets/advice_id.yaml, >= 4 gentle variants per key, <= 2 sentences / <= 90 chars.
* Cooldown per key (buddy.cooldown_key_min) and global between bubbles (buddy.cooldown_global_min).
* At most buddy.max_per_hour bubbles per hour. Severity 3 may skip the global cooldown, never
  the per-key cooldown.
* Escalation: after buddy.escalate_after ignored bubbles of a key, use the firmer variants.
* Responses: oke | nanti (snooze buddy.snooze_min) | jangan_hari_ini (mute key today) |
  diabaikan (timed out) | ditutup (clicked away).
"""

from __future__ import annotations

import random
import re
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

BANK_PATH = Path(__file__).resolve().parent / "assets" / "advice_id.yaml"
RESPONSES: tuple[str, ...] = ("oke", "nanti", "jangan_hari_ini", "diabaikan", "ditutup")
MIN_VARIANTS = 4
MAX_SENTENCES = 2
SEC_PER_MIN = 60.0
HOUR = 3600.0
_SENTENCE_END = re.compile(r"[.!?]+(?=\s|$)")


@dataclass(frozen=True)
class Advice:
    """One bubble to show."""

    key: str
    text: str
    expression: str
    severity: int
    tone: str
    ts: float


def load_bank(path: Path = BANK_PATH) -> dict[str, dict[str, Any]]:
    """Read the advice bank."""
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def sentence_count(text: str) -> int:
    """Number of sentences (terminal . ! ? groups)."""
    return max(1, len(_SENTENCE_END.findall(text.strip())))


def validate_bank(bank: dict[str, dict[str, Any]], max_chars: int) -> list[str]:
    """Rule violations in the bank (empty = OK)."""
    problems = []
    for key, entry in bank.items():
        gentle = entry.get("lembut", [])
        if len(gentle) < MIN_VARIANTS:
            problems.append(f"{key}: hanya {len(gentle)} saran lembut (min {MIN_VARIANTS})")
        for tone in ("lembut", "tegas"):
            for text in entry.get(tone, []):
                if len(text) > max_chars:
                    problems.append(f"{key}/{tone}: {len(text)} karakter > {max_chars}: {text}")
                if sentence_count(text) > MAX_SENTENCES:
                    problems.append(f"{key}/{tone}: lebih dari {MAX_SENTENCES} kalimat: {text}")
        if entry.get("severity") not in (1, 2, 3):
            problems.append(f"{key}: severity harus 1..3")
    return problems


class AdviceEngine:
    """Decides whether and what to say. Time is epoch seconds (injectable for tests)."""

    def __init__(self, buddy_cfg: dict[str, Any], bank: dict[str, dict[str, Any]] | None = None,
                 seed: int | None = None, clock: Callable[[], float] = time.time) -> None:
        self.bank = bank if bank is not None else load_bank()
        self.cooldown_key = float(buddy_cfg["cooldown_key_min"]) * SEC_PER_MIN
        self.cooldown_global = float(buddy_cfg["cooldown_global_min"]) * SEC_PER_MIN
        self.max_per_hour = int(buddy_cfg["max_per_hour"])
        self.snooze = float(buddy_cfg["snooze_min"]) * SEC_PER_MIN
        self.escalate_after = int(buddy_cfg["escalate_after"])
        self._rng = random.Random(seed)
        self._clock = clock
        self._last_global = float("-inf")
        self._next_allowed: dict[str, float] = {}
        self._ignored: dict[str, int] = {}
        self._muted_day: dict[str, str] = {}
        self._muted_until = 0.0
        self._last_text: dict[str, str] = {}
        self._shown: deque[float] = deque()

    # -- muting ---------------------------------------------------------------
    def mute_until(self, ts: float) -> None:
        """Silence every bubble until ts (pause / 'Diam hari ini' / Do Not Disturb)."""
        self._muted_until = max(self._muted_until, ts)

    def unmute(self) -> None:
        """Cancel a global mute."""
        self._muted_until = 0.0

    def muted(self, now: float | None = None) -> bool:
        """True while globally muted."""
        return (now if now is not None else self._clock()) < self._muted_until

    @staticmethod
    def _day(ts: float) -> str:
        return datetime.fromtimestamp(ts).date().isoformat()

    # -- selection ------------------------------------------------------------
    def pick_text(self, key: str, tone: str = "lembut") -> str:
        """Random variant, avoiding the one shown last time for this key."""
        variants = self.bank[key].get(tone) or self.bank[key]["lembut"]
        choices = [v for v in variants if v != self._last_text.get(key)] or variants
        text = self._rng.choice(choices)
        self._last_text[key] = text
        return text

    def blocked_reason(self, key: str, now: float) -> str | None:
        """Why a bubble for key would not be shown now (None = allowed)."""
        if key not in self.bank:
            return "kunci tidak dikenal"
        if self.muted(now):
            return "dibisukan"
        if self._muted_day.get(key) == self._day(now):
            return "dimatikan hari ini"
        if now < self._next_allowed.get(key, float("-inf")):
            return "cooldown kunci"
        while self._shown and self._shown[0] <= now - HOUR:
            self._shown.popleft()
        if len(self._shown) >= self.max_per_hour:
            return "batas per jam"
        if int(self.bank[key]["severity"]) < 3 and now - self._last_global < self.cooldown_global:
            return "cooldown global"
        return None

    def request(self, key: str, now: float | None = None) -> Advice | None:
        """Advice to show for key, or None when spam control blocks it."""
        now = now if now is not None else self._clock()
        if self.blocked_reason(key, now) is not None:
            return None
        entry = self.bank[key]
        tone = "tegas" if self._ignored.get(key, 0) >= self.escalate_after and entry.get("tegas") else "lembut"
        advice = Advice(key, self.pick_text(key, tone), str(entry["expression"]), int(entry["severity"]), tone, now)
        self._last_global = now
        self._next_allowed[key] = now + self.cooldown_key
        self._shown.append(now)
        return advice

    def respond(self, key: str, response: str, now: float | None = None) -> None:
        """Record what the user did with a bubble."""
        if response not in RESPONSES:
            raise ValueError(f"Respons tidak dikenal: {response}")
        now = now if now is not None else self._clock()
        if response == "oke":
            self._ignored[key] = 0
        elif response == "nanti":
            self._ignored[key] = self._ignored.get(key, 0) + 1
            self._next_allowed[key] = now + self.snooze
        elif response == "diabaikan":
            self._ignored[key] = self._ignored.get(key, 0) + 1
        elif response == "jangan_hari_ini":
            self._muted_day[key] = self._day(now)

    def ignored_count(self, key: str) -> int:
        """Consecutive ignored bubbles of key."""
        return self._ignored.get(key, 0)
