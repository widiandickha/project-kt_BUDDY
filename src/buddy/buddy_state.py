"""Explicit Buddy state machine (spec 5.4): enum + transition table + priorities.

    IDLE --(stress_sedang)--> WORRIED --(stress_tinggi)--> ALERT
      ^                          |                          |
      +---- (stress_calm) -------+------(advice_done)-------+
    IDLE --(user_away_long)--> SLEEPING --(user_returned)--> WAKE_GREET --> IDLE
    any  --(positive event)--> HAPPY / CELEBRATE --(anim done)--> previous base state

Priority when events collide: ALERT/CONCERNED (3) > WORRIED/CURIOUS/EYE_CARE/SLEEPY/STRETCH (2)
> HAPPY/CELEBRATE/WAKE_GREET (1) > IDLE (0). SLEEPING (4) only leaves on user_returned.
A trigger whose target has a lower priority than the current state is ignored, except the
user's own click. Non-looping states return to the stored base state when their animation ends.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class BuddyState(str, Enum):
    """The 12 expressions; the value is the animation/sprite name."""

    IDLE = "idle"
    HAPPY = "happy"
    SLEEPY = "sleepy"
    WORRIED = "worried"
    ALERT = "alert"
    CONCERNED = "concerned"
    STRETCH = "stretch"
    CURIOUS = "curious"
    EYE_CARE = "eye_care"
    SLEEPING = "sleeping"
    WAKE_GREET = "wake_greet"
    CELEBRATE = "celebrate"


S = BuddyState

PRIORITY: dict[BuddyState, int] = {
    S.IDLE: 0,
    S.HAPPY: 1, S.CELEBRATE: 1, S.WAKE_GREET: 1,
    S.WORRIED: 2, S.CURIOUS: 2, S.EYE_CARE: 2, S.SLEEPY: 2, S.STRETCH: 2,
    S.ALERT: 3, S.CONCERNED: 3,
    S.SLEEPING: 4,
}

# Persistent (looping) states that non-looping reactions return to.
BASE_STATES: frozenset[BuddyState] = frozenset({S.IDLE, S.WORRIED, S.CONCERNED, S.SLEEPY, S.SLEEPING})

ANY = "*"
# (from_state | ANY, trigger) -> to_state. Specific rows win over ANY rows.
TRANSITIONS: dict[tuple[str, str], BuddyState] = {
    (S.IDLE.value, "stress_sedang"): S.WORRIED,
    (S.SLEEPY.value, "stress_sedang"): S.WORRIED,
    (S.WORRIED.value, "stress_tinggi"): S.ALERT,
    (S.IDLE.value, "stress_tinggi"): S.ALERT,
    (S.SLEEPY.value, "stress_tinggi"): S.ALERT,
    (S.WORRIED.value, "stress_calm"): S.IDLE,
    (S.ALERT.value, "stress_calm"): S.IDLE,
    (S.SLEEPY.value, "stress_calm"): S.IDLE,
    (S.CONCERNED.value, "stress_calm"): S.IDLE,
    (S.ALERT.value, "advice_done"): S.IDLE,
    (S.SLEEPING.value, "user_returned"): S.WAKE_GREET,
    (ANY, "user_away_long"): S.SLEEPING,
    (ANY, "burnout_risk"): S.CONCERNED,
    (ANY, "late_night"): S.CONCERNED,
    (ANY, "need_break"): S.ALERT,
    (ANY, "bad_posture"): S.WORRIED,
    (ANY, "too_close"): S.CURIOUS,
    (ANY, "dry_eyes"): S.EYE_CARE,
    (ANY, "sleepy"): S.SLEEPY,
    (ANY, "stretch"): S.STRETCH,
    (ANY, "good_break"): S.HAPPY,
    (ANY, "doing_great"): S.HAPPY,
    (ANY, "click"): S.HAPPY,
    (ANY, "celebrate"): S.CELEBRATE,
}
TRIGGERS: frozenset[str] = frozenset(t for _, t in TRANSITIONS)
USER_TRIGGERS: frozenset[str] = frozenset({"click"})


@dataclass(frozen=True)
class Transition:
    """Result of handling a trigger."""

    previous: BuddyState
    current: BuddyState
    trigger: str


class BuddyStateMachine:
    """Tracks the current state and the base state to return to."""

    def __init__(self) -> None:
        self.state = S.IDLE
        self.base = S.IDLE

    def target_for(self, trigger: str) -> BuddyState | None:
        """Table lookup without side effects."""
        return TRANSITIONS.get((self.state.value, trigger)) or TRANSITIONS.get((ANY, trigger))

    def _allowed(self, trigger: str, target: BuddyState) -> bool:
        """Priority rules. Rows written for the current state are always allowed; wildcard rows
        are filtered: SLEEPING ignores them, a running reaction is not cut by a lower-priority
        one, and a base mood is not replaced by a lower-priority mood. Lower-priority short
        reactions (e.g. HAPPY while WORRIED) may play and then return to the base state."""
        if (self.state.value, trigger) in TRANSITIONS:
            return True
        if self.state == S.SLEEPING:
            return False
        if trigger in USER_TRIGGERS:
            return True
        lower = PRIORITY[target] < PRIORITY[self.state]
        if self.state not in BASE_STATES:
            return not lower
        return not (lower and target in BASE_STATES)

    def handle(self, trigger: str) -> Transition | None:
        """Apply a trigger; None if it is unknown here, already active, or outranked."""
        if trigger not in TRIGGERS:
            raise ValueError(f"Unknown trigger '{trigger}'")
        target = self.target_for(trigger)
        if target is None or target == self.state or not self._allowed(trigger, target):
            return None
        prev = self.state
        self.state = target
        if target in BASE_STATES:
            self.base = target
        elif trigger in ("stress_calm", "advice_done"):
            self.base = S.IDLE
        return Transition(prev, target, trigger)

    def animation_finished(self, name: str) -> Transition | None:
        """A non-looping animation ended: go back to the base state."""
        if name != self.state.value or self.state in BASE_STATES:
            return None
        prev = self.state
        self.state = self.base if self.state != S.WAKE_GREET else S.IDLE
        if self.state == S.IDLE:
            self.base = S.IDLE
        return Transition(prev, self.state, "anim_finished")
