"""Data contracts between modules (spec 4.2) and a small thread-safe EventBus.

Modules never call each other directly: producers publish dataclasses under one of the
EVENT_* names and consumers subscribe. Fields marked "extension" are additions to the spec
contract, documented in docs/DECISIONS.md D-017.
"""

from __future__ import annotations

import logging
import threading
from collections import defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

EVENT_FRAME_FEATURES = "frame_features"
EVENT_STRESS_SNAPSHOT = "stress_snapshot"
EVENT_HABIT_ALERT = "habit_alert"
EVENT_BUDDY_EVENT = "buddy_event"
EVENT_USER_AWAY = "user_away"
EVENT_USER_RETURNED = "user_returned"
EVENT_NAMES: tuple[str, ...] = (
    EVENT_FRAME_FEATURES,
    EVENT_STRESS_SNAPSHOT,
    EVENT_HABIT_ALERT,
    EVENT_BUDDY_EVENT,
    EVENT_USER_AWAY,
    EVENT_USER_RETURNED,
)

LEVELS: tuple[str, ...] = ("rendah", "sedang", "tinggi")


@dataclass
class FrameFeatures:
    """Per-frame features produced by the inference thread."""

    ts: float
    user_present: bool
    eye_closed_yolo: float = 0.0
    yawn: float = 0.0
    brow_furrow: float = 0.0
    ear_left: float | None = None
    ear_right: float | None = None
    ear_mean: float | None = None
    hand_to_forehead: bool = False
    head_pitch_proxy: float | None = None
    shoulder_slouch: float | None = None
    face_size_ratio: float | None = None
    pose_keypoints: np.ndarray | None = None
    # extensions (D-017)
    det_available: bool = False
    mar: float | None = None
    brow_ratio: float | None = None
    nose_xy: tuple[float, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly dict (keypoints as nested lists)."""
        d = asdict(self)
        d["pose_keypoints"] = None if self.pose_keypoints is None else np.asarray(self.pose_keypoints).round(4).tolist()
        d["nose_xy"] = None if self.nose_xy is None else list(self.nose_xy)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> FrameFeatures:
        """Inverse of to_dict; unknown keys are ignored."""
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        if known.get("pose_keypoints") is not None:
            known["pose_keypoints"] = np.asarray(known["pose_keypoints"], dtype=np.float32)
        if known.get("nose_xy") is not None:
            known["nose_xy"] = tuple(known["nose_xy"])
        return cls(**known)


@dataclass
class StressSnapshot:
    """Stress score computed once per second from the short window."""

    ts: float
    score: float
    level: str
    components: dict[str, float]
    burnout_trend: float
    # extensions (D-017)
    raw_score: float = 0.0
    burnout_risk: bool = False


@dataclass
class HabitAlert:
    """Habit event (spec 4.8): need_break, bad_posture, too_close, dry_eyes, late_night, good_break, doing_great."""

    kind: str
    ts: float
    positive: bool = False
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class PresenceEvent:
    """user_away / user_returned payload."""

    kind: str
    ts: float
    away_sec: float = 0.0


@dataclass
class BuddyEvent:
    """Something the Buddy should react to (spec 4.2 / 5.4)."""

    kind: str
    severity: int
    message_key: str
    ts: float


Handler = Callable[[Any], None]


class EventBus:
    """Synchronous publish/subscribe. Handlers run in the publisher's thread, so they must be
    quick; the Qt bridge (ui) re-emits into the main thread via a queued signal.
    A failing handler is logged and does not stop the others.
    """

    def __init__(self) -> None:
        self._handlers: dict[str, list[Handler]] = defaultdict(list)
        self._lock = threading.Lock()

    def subscribe(self, event: str, handler: Handler) -> Callable[[], None]:
        """Register a handler; returns a function that unsubscribes it."""
        if event not in EVENT_NAMES:
            raise ValueError(f"Unknown event '{event}', expected one of {EVENT_NAMES}")
        with self._lock:
            self._handlers[event].append(handler)

        def unsubscribe() -> None:
            with self._lock:
                if handler in self._handlers[event]:
                    self._handlers[event].remove(handler)

        return unsubscribe

    def publish(self, event: str, payload: Any) -> None:
        """Deliver payload to every handler of event."""
        if event not in EVENT_NAMES:
            raise ValueError(f"Unknown event '{event}'")
        with self._lock:
            handlers = list(self._handlers[event])
        for h in handlers:
            try:
                h(payload)
            except Exception:  # noqa: BLE001 - one bad consumer must not break the pipeline
                logger.exception("Handler for %s failed", event)
