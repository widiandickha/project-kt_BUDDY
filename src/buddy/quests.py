"""Interactive wellness Quests for the Virtual Buddy.

When the Buddy gives advice (or when the user picks a quest from the right-click menu),
the user can start an interactive timed Quest — e.g. 5-minute power nap, drinking water,
20-20-20 eye rest, shoulder stretch, or deep breathing.

While a Quest is running:
- A floating QuestBanner sits above the cat showing the quest icon, title, live MM:SS
  countdown, animated progress bar, and Selesai / Batal buttons.
- The cat adopts the matching expression (e.g. sleeping during a 5-min rest, eye_care during
  eye rest, stretch during stretching).
- When the countdown reaches 00:00, the cat celebrates and notifies the user with a completion
  message!
"""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import QPoint, QRect, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QGuiApplication, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QHBoxLayout, QLabel, QProgressBar, QPushButton, QVBoxLayout, QWidget

from src.ui import theme

BANNER_WIDTH = 288
TAIL = 10
RADIUS = 14
BG = QColor("#FFFDF9")
BORDER = QColor("#E2C5AE")
SHADOW = QColor(28, 25, 23, 30)


@dataclass(frozen=True)
class QuestSpec:
    """Definition of an interactive Quest."""

    id: str
    title: str
    icon: str
    duration_sec: int
    active_expression: str
    button_label: str
    active_hint: str
    done_text: str


QUESTS: dict[str, QuestSpec] = {
    "nap_5m": QuestSpec(
        id="nap_5m",
        title="Tidur / Rehat 5 Menit",
        icon="😴",
        duration_sec=300,
        active_expression="sleeping",
        button_label="😴 Quest Tidur 5m",
        active_hint="Pejamkan mata & rebahkan badan, aku hitung waktunya...",
        done_text="⏰ Waktu tidur 5 menit selesai! Selamat bangun, semoga badanmu kembali segar!",
    ),
    "drink_water": QuestSpec(
        id="drink_water",
        title="Minum Air Putih",
        icon="💧",
        duration_sec=45,
        active_expression="happy",
        button_label="💧 Quest Minum Air",
        active_hint="Ambil segelas air putih dan minum perlahan...",
        done_text="💧 Quest Minum Air selesai! Tubuh terhidrasi bikin fokus lebih tajam.",
    ),
    "eye_rest": QuestSpec(
        id="eye_rest",
        title="Istirahat Mata 20 Detik",
        icon="👀",
        duration_sec=20,
        active_expression="eye_care",
        button_label="👀 Quest Mata 20d",
        active_hint="Pandang objek jauh (6 meter) atau pejamkan mata...",
        done_text="✨ 20 detik selesai! Matamu sudah lebih rileks, siap lanjut lagi.",
    ),
    "stretch_body": QuestSpec(
        id="stretch_body",
        title="Peregangan Badan 1 Menit",
        icon="🧘",
        duration_sec=60,
        active_expression="stretch",
        button_label="🧘 Quest Peregangan 1m",
        active_hint="Berdiri, putar bahu, dan regangkan punggungmu...",
        done_text="🎉 Peregangan 1 menit selesai! Otot bahu dan punggung pasti lebih ringan.",
    ),
    "breathe_calm": QuestSpec(
        id="breathe_calm",
        title="Napas Relaksasi 1 Menit",
        icon="🌬️",
        duration_sec=60,
        active_expression="eye_care",
        button_label="🌬️ Quest Napas 1m",
        active_hint="Tarik napas 4 detik, tahan 4 detik, hembuskan perlahan...",
        done_text="🌿 Relaksasi napas selesai! Pikiran lebih tenang dan siap fokus kembali.",
    ),
    "posture_hold": QuestSpec(
        id="posture_hold",
        title="Duduk Tegak & Atur Jarak",
        icon="🪑",
        duration_sec=30,
        active_expression="idle",
        button_label="🪑 Quest Postur 30d",
        active_hint="Tegakkan punggung, rilekskan bahu, jaga jarak satu lengan...",
        done_text="👍 Bagus sekali! Pertahankan posisi duduk ergonomis ini ya.",
    ),
}

# Maps an advice key to the most fitting interactive Quest
ADVICE_QUEST_MAP: dict[str, str] = {
    "stress_tinggi": "nap_5m",
    "burnout_risk": "nap_5m",
    "late_night": "nap_5m",
    "sleepy": "drink_water",
    "need_break": "stretch_body",
    "dry_eyes": "eye_rest",
    "stress_sedang": "breathe_calm",
    "bad_posture": "posture_hold",
    "too_close": "posture_hold",
}


def quest_for_advice(advice_key: str) -> QuestSpec | None:
    """Return the interactive Quest associated with an advice message key, if any."""
    qid = ADVICE_QUEST_MAP.get(advice_key)
    return QUESTS.get(qid) if qid else None


def format_mmss(seconds: int) -> str:
    """Format remaining seconds as MM:SS."""
    s = max(0, int(seconds))
    m, sec = divmod(s, 60)
    return f"{m:02d}:{sec:02d}"


class QuestBanner(QWidget):
    """Floating countdown HUD above the cat while a Quest is active."""

    completed = Signal(str)       # quest_id (timer reached 0 or user clicked Selesai)
    cancelled = Signal(str)       # quest_id
    tick_updated = Signal(int)    # remaining_sec

    def __init__(self) -> None:
        super().__init__(
            None,
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowDoesNotAcceptFocus,
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setFixedWidth(BANNER_WIDTH)

        self.quest: QuestSpec | None = None
        self.remaining_sec: int = 0
        self.total_sec: int = 1
        self._tail_x: float = BANNER_WIDTH / 2

        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 12, 16, 12 + TAIL)
        lay.setSpacing(6)

        # Top row: quest badge + big MM:SS timer
        top = QHBoxLayout()
        top.setSpacing(8)
        self.title_lbl = QLabel("🎯 Quest Aktif")
        self.title_lbl.setStyleSheet(f"color: {theme.ACCENT}; font-size: 12px; font-weight: 800;")
        self.time_lbl = QLabel("05:00")
        self.time_lbl.setStyleSheet(
            f"background: {theme.ACCENT_SOFT}; color: {theme.ACCENT_HOVER};"
            " border: 1px solid #F2C7A5; border-radius: 8px; padding: 2px 8px;"
            " font-size: 15px; font-weight: 800;"
        )
        top.addWidget(self.title_lbl, 1)
        top.addWidget(self.time_lbl, 0)
        lay.addLayout(top)

        self.hint_lbl = QLabel("")
        self.hint_lbl.setWordWrap(True)
        self.hint_lbl.setStyleSheet(f"color: {theme.TEXT_2}; font-size: 11px; font-weight: 500;")
        lay.addWidget(self.hint_lbl)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(7)
        self.progress.setStyleSheet(
            f"QProgressBar {{ background: {theme.SURFACE}; border: none; border-radius: 3px; }}"
            f"QProgressBar::chunk {{ background: {theme.LEVEL_COLORS['rendah']}; border-radius: 3px; }}"
        )
        lay.addWidget(self.progress)

        btns = QHBoxLayout()
        btns.setSpacing(6)
        self.done_btn = QPushButton("✓ Selesai Sekarang")
        self.done_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.done_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.done_btn.setStyleSheet(
            f"QPushButton {{ background: {theme.LEVEL_COLORS['rendah']}; color: white; border: none;"
            " border-radius: 7px; padding: 4px 10px; font-size: 11px; font-weight: 700; }"
        )
        self.done_btn.clicked.connect(self.finish_now)

        self.cancel_btn = QPushButton("Batal")
        self.cancel_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.cancel_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.cancel_btn.setStyleSheet(
            f"QPushButton {{ background: {theme.SURFACE}; color: {theme.TEXT_2}; border: 1px solid {theme.BORDER};"
            " border-radius: 7px; padding: 4px 9px; font-size: 11px; font-weight: 600; }"
        )
        self.cancel_btn.clicked.connect(self.cancel)

        btns.addWidget(self.done_btn)
        btns.addWidget(self.cancel_btn)
        btns.addStretch(1)
        lay.addLayout(btns)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._on_tick)

    def start_quest(self, quest: QuestSpec, anchor: QRect) -> None:
        """Start counting down `quest` and position the banner above `anchor`."""
        self._timer.stop()
        self.quest = quest
        self.total_sec = max(1, quest.duration_sec)
        self.remaining_sec = quest.duration_sec
        self.title_lbl.setText(f"{quest.icon}  {quest.title}")
        self.hint_lbl.setText(quest.active_hint)
        self._sync_ui()
        self.place_above(anchor)
        self.show()
        self._timer.start(1000)

    def place_above(self, anchor: QRect) -> None:
        """Position the banner directly above the cat window."""
        self.adjustSize()
        screen = QGuiApplication.screenAt(anchor.center()) or QGuiApplication.primaryScreen()
        avail = screen.availableGeometry()
        x = min(max(avail.left(), anchor.center().x() - self.width() // 2), avail.right() - self.width())
        y = max(avail.top(), anchor.top() - self.height() + 6)
        self._tail_x = anchor.center().x() - x
        self.setGeometry(QRect(QPoint(x, y), self.size()))

    def _sync_ui(self) -> None:
        self.time_lbl.setText(format_mmss(self.remaining_sec))
        elapsed = self.total_sec - self.remaining_sec
        pct = int(round(100 * elapsed / max(1, self.total_sec)))
        self.progress.setValue(max(0, min(100, pct)))

    def _on_tick(self) -> None:
        if self.quest is None:
            self._timer.stop()
            return
        self.remaining_sec -= 1
        self._sync_ui()
        self.tick_updated.emit(self.remaining_sec)
        if self.remaining_sec <= 0:
            self.finish_now()

    def finish_now(self) -> None:
        """Complete the active quest immediately (or when the timer reaches 0)."""
        self._timer.stop()
        q = self.quest
        self.quest = None
        self.hide()
        if q is not None:
            self.completed.emit(q.id)

    def cancel(self) -> None:
        """Cancel the active quest without completion."""
        self._timer.stop()
        q = self.quest
        self.quest = None
        self.hide()
        if q is not None:
            self.cancelled.emit(q.id)

    def paintEvent(self, _event) -> None:  # noqa: N802 (Qt API)
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        body = QRectF(2, 2, self.width() - 4, self.height() - TAIL - 4)
        path = QPainterPath()
        path.addRoundedRect(body, RADIUS, RADIUS)
        tx = min(max(self._tail_x, RADIUS + TAIL), self.width() - RADIUS - TAIL)
        tail = QPainterPath()
        tail.moveTo(tx - TAIL, body.bottom() - 1)
        tail.lineTo(tx, self.height() - 2)
        tail.lineTo(tx + TAIL, body.bottom() - 1)
        tail.closeSubpath()
        path = path.united(tail)

        p.save()
        p.translate(0, 1.5)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(SHADOW)
        p.drawPath(path)
        p.restore()

        p.setPen(QPen(BORDER, 1.6))
        p.setBrush(BG)
        p.drawPath(path)
