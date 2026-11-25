"""Daily dashboard (spec FASE 6): summary cards, stress per hour, breaks over 7 days,
7-day comparison, automatic insights, CSV export.

    python -m src.ui.dashboard                  # real database (data/stress_buddy.db)
    python -m src.ui.dashboard --demo-data      # temporary database with 7 days of dummy data
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import QDate, QRectF, Qt, Signal, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QDateEdit,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from src.storage import daily_summary as ds
from src.storage.repository import Repository
from src.ui import theme

DISCLAIMER = "Alat bantu kesadaran diri, bukan alat diagnosis medis."
CAT_DIR = Path(__file__).resolve().parents[1] / "buddy" / "assets" / "cat"
HARI = ("Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu")
BULAN = ("Januari", "Februari", "Maret", "April", "Mei", "Juni", "Juli", "Agustus", "September",
         "Oktober", "November", "Desember")


def fmt_minutes(m: float) -> str:
    """'3 j 25 m' style duration."""
    h, mm = divmod(int(round(m)), 60)
    return f"{h} j {mm} m" if h else f"{mm} m"


def fmt_day(d: date) -> str:
    """'Kamis, 19 November 2026'."""
    return f"{HARI[d.weekday()]}, {d.day} {BULAN[d.month - 1]} {d.year}"


def stress_level(score: float | None, cfg: dict[str, Any]) -> str | None:
    """rendah / sedang / tinggi from the configured thresholds."""
    if score is None:
        return None
    th = cfg["score"]["thresholds"]
    return "rendah" if score < float(th["low"]) else "sedang" if score < float(th["high"]) else "tinggi"


def health_band(score: float | None) -> tuple[str, str]:
    """(label, colour) for the health score ring."""
    if score is None:
        return "Belum ada data", theme.BORDER_STRONG
    if score >= 80:
        return "Baik", theme.LEVEL_COLORS["rendah"]
    if score >= 60:
        return "Cukup", theme.LEVEL_COLORS["sedang"]
    return "Perlu perhatian", theme.LEVEL_COLORS["tinggi"]


def buddy_message(t: ds.DailySummary | None) -> tuple[str, str]:
    """(cat sprite, sentence) shown in the hero card."""
    if t is None or t.work_minutes <= 0:
        return "sleeping", "Belum ada data untuk hari ini. Nyalakan Stress Buddy dan aku akan menemanimu."
    if t.burnout_flag:
        return "worried", "Stres tinggi bertahan berjam-jam. Ambil jeda yang cukup panjang, ya."
    hs = t.health_score or 0.0
    if hs >= 80:
        return "happy", "Hari yang bagus! Ritme kerja dan istirahatmu seimbang."
    if hs >= 60:
        return "idle", "Lumayan. Coba selipkan jeda singkat lebih sering."
    return "concerned", "Hari yang berat. Istirahat sejenak dan regangkan badan, ya."


def insight_icon(text: str) -> str:
    """Emoji that matches an automatic insight sentence."""
    t = text.lower()
    for key, icon in (("burnout", "⚠️"), ("belum ada istirahat", "☕"), ("tanpa jeda", "⏳"), ("saran", "💬"),
                      ("puncak", "📈"), ("lebih rendah", "👍"), ("lebih tinggi", "📊"), ("belum ada data", "🐾")):
        if key in t:
            return icon
    return "✨"


class HealthRing(QWidget):
    """Circular gauge 0..100 with the score in the middle."""

    def __init__(self) -> None:
        super().__init__()
        self.score: float | None = None
        self.setMinimumSize(150, 150)

    def set_score(self, score: float | None) -> None:
        self.score = score
        self.update()

    def paintEvent(self, _event) -> None:  # noqa: N802 (Qt API)
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        side = min(self.width(), self.height()) - 16
        rect = QRectF((self.width() - side) / 2, (self.height() - side) / 2, side, side)
        label, color = health_band(self.score)
        pen = QPen(QColor(theme.SURFACE), 12)
        pen.setCapStyle(Qt.RoundCap)
        p.setPen(pen)
        p.drawArc(rect, 0, 360 * 16)
        if self.score is not None:
            pen.setColor(QColor(color))
            p.setPen(pen)
            p.drawArc(rect, 90 * 16, int(-360 * 16 * max(0.0, min(100.0, self.score)) / 100))
        p.setPen(QColor(theme.TEXT))
        f = QFont(self.font())
        f.setPixelSize(int(side * 0.26))
        f.setBold(True)
        p.setFont(f)
        p.drawText(rect.adjusted(0, -side * 0.08, 0, -side * 0.08), Qt.AlignCenter,
                   "–" if self.score is None else f"{self.score:.0f}")
        f.setPixelSize(max(11, int(side * 0.09)))
        f.setBold(False)
        p.setFont(f)
        p.setPen(QColor(color))
        p.drawText(rect.adjusted(0, side * 0.3, 0, 0), Qt.AlignHCenter | Qt.AlignVCenter, label)
        p.end()


class InsightList(QWidget):
    """Insight sentences as soft cards (keeps QListWidget's count() for callers/tests)."""

    def __init__(self) -> None:
        super().__init__()
        self.lay = QVBoxLayout(self)
        self.lay.setContentsMargins(0, 0, 0, 0)
        self.lay.setSpacing(8)
        self._items: list[str] = []

    def count(self) -> int:
        return len(self._items)

    def items(self) -> list[str]:
        return list(self._items)

    def set_items(self, texts: list[str]) -> None:
        while self.lay.count():
            w = self.lay.takeAt(0).widget()
            if w is not None:
                w.deleteLater()
        self._items = list(texts)
        for text in texts:
            row = QFrame()
            row.setObjectName("insight")
            h = QHBoxLayout(row)
            h.setContentsMargins(12, 10, 12, 10)
            icon = QLabel(insight_icon(text))
            icon.setObjectName("cardIcon")
            icon.setFixedWidth(28)
            body = QLabel(text)
            body.setWordWrap(True)
            h.addWidget(icon, 0, Qt.AlignTop)
            h.addWidget(body, 1)
            self.lay.addWidget(row)


def _card(object_name: str = "card") -> tuple[QFrame, QVBoxLayout]:
    frame = QFrame()
    frame.setObjectName(object_name)
    lay = QVBoxLayout(frame)
    lay.setContentsMargins(16, 14, 16, 14)
    lay.setSpacing(4)
    return frame, lay


def _stat_card(icon: str, label: str) -> tuple[QFrame, QLabel]:
    frame, lay = _card()
    top = QHBoxLayout()
    ic = QLabel(icon)
    ic.setObjectName("cardIcon")
    cap = QLabel(label)
    cap.setObjectName("cardLabel")
    top.addWidget(ic)
    top.addWidget(cap, 1)
    lay.addLayout(top)
    value = QLabel("–")
    value.setObjectName("cardValue")
    lay.addWidget(value)
    return frame, value


def _chart_card(title: str, height: int) -> tuple[QFrame, Figure, FigureCanvasQTAgg]:
    frame, lay = _card()
    h = QLabel(title)
    h.setObjectName("h2")
    lay.addWidget(h)
    fig = Figure(figsize=(5, height / 100), tight_layout=True, facecolor=theme.CARD)
    canvas = FigureCanvasQTAgg(fig)
    canvas.setMinimumHeight(height)
    canvas.setStyleSheet(f"background: {theme.CARD};")
    lay.addWidget(canvas)
    return frame, fig, canvas


def _style_axes(ax) -> None:
    ax.set_facecolor(theme.CARD)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(theme.BORDER)
    ax.tick_params(colors=theme.TEXT_2, labelsize=8, length=0)
    ax.grid(axis="y", color=theme.BORDER, lw=0.6, alpha=0.8)
    ax.set_axisbelow(True)


class Dashboard(QWidget):
    """Reads from the repository; call refresh() to recompute."""

    open_preview = Signal()

    STATS: tuple[tuple[str, str, str], ...] = (
        ("work", "⏱️", "Waktu kerja"), ("peak", "📈", "Stres puncak"), ("breaks", "☕", "Istirahat ≥ 5 menit"),
        ("stretch", "⏳", "Kerja terlama tanpa jeda"), ("advice", "💬", "Saran diikuti"), ("yawn", "🥱", "Menguap"),
    )

    def __init__(self, repo: Repository, cfg: dict[str, Any]) -> None:
        super().__init__()
        self.repo, self.cfg = repo, cfg
        self.setWindowTitle("Stress Buddy · Dashboard")
        self.setStyleSheet(theme.STYLESHEET)
        self.resize(1080, 860)
        self.values: dict[str, QLabel] = {}

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        outer.addWidget(scroll)
        page = QWidget()
        scroll.setWidget(page)
        root = QVBoxLayout(page)
        root.setContentsMargins(24, 20, 24, 20)
        root.setSpacing(14)

        root.addLayout(self._build_header())
        root.addLayout(self._build_hero())

        grid = QGridLayout()
        grid.setSpacing(12)
        for i, (key, icon, label) in enumerate(self.STATS):
            frame, value = _stat_card(icon, label)
            self.values[key] = value
            grid.addWidget(frame, i // 3, i % 3)
        root.addLayout(grid)

        card, self.fig_hourly, self.canvas_hourly = _chart_card("Stres per jam", 230)
        root.addWidget(card)
        row = QHBoxLayout()
        row.setSpacing(12)
        card, self.fig_breaks, self.canvas_breaks = _chart_card("Istirahat per hari (7 hari)", 210)
        row.addWidget(card, 1)
        card, self.fig_trend, self.canvas_trend = _chart_card("Tren 7 hari", 210)
        row.addWidget(card, 1)
        root.addLayout(row)

        card, lay = _card()
        h = QLabel("Insight otomatis")
        h.setObjectName("h2")
        lay.addWidget(h)
        lay.addSpacing(4)
        self.insights = InsightList()
        lay.addWidget(self.insights)
        root.addWidget(card)

        note = QLabel("🔒 " + DISCLAIMER + " Hanya angka indikator yang disimpan, tanpa gambar atau video.")
        note.setObjectName("muted")
        note.setWordWrap(True)
        root.addWidget(note)
        root.addStretch(1)

        self.today: ds.DailySummary | None = None
        self.history: list[ds.DailySummary] = []
        self.refresh()

    # -- layout ---------------------------------------------------------------------------
    def _build_header(self) -> QHBoxLayout:
        head = QHBoxLayout()
        titles = QVBoxLayout()
        titles.setSpacing(0)
        title = QLabel("Ringkasan harian")
        title.setObjectName("title")
        self.subtitle = QLabel("")
        self.subtitle.setObjectName("subtitle")
        titles.addWidget(title)
        titles.addWidget(self.subtitle)
        head.addLayout(titles)
        head.addStretch(1)

        prev = QPushButton("‹")
        prev.setObjectName("icon")
        prev.setToolTip("Hari sebelumnya")
        prev.clicked.connect(lambda: self.shift_day(-1))
        self.date_edit = QDateEdit(QDate.currentDate())
        self.date_edit.setCalendarPopup(True)
        self.date_edit.setDisplayFormat("dd MMM yyyy")
        self.date_edit.setMaximumDate(QDate.currentDate())
        self.date_edit.dateChanged.connect(lambda _: self.refresh())
        nxt = QPushButton("›")
        nxt.setObjectName("icon")
        nxt.setToolTip("Hari berikutnya")
        nxt.clicked.connect(lambda: self.shift_day(1))
        today = QPushButton("Hari ini")
        today.setObjectName("secondary")
        today.clicked.connect(lambda: self.date_edit.setDate(QDate.currentDate()))
        reload_ = QPushButton("⟳")
        reload_.setObjectName("icon")
        reload_.setToolTip("Muat ulang")
        reload_.clicked.connect(self.refresh)
        camera_btn = QPushButton("Pratinjau kamera")
        camera_btn.setObjectName("secondary")
        camera_btn.clicked.connect(self.open_preview.emit)
        export = QPushButton("Ekspor CSV")
        export.clicked.connect(self.export_csv)
        for w in (prev, self.date_edit, nxt, today, reload_):
            head.addWidget(w)
        head.addSpacing(6)
        head.addWidget(camera_btn)
        head.addWidget(export)
        return head

    def _build_hero(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(12)
        hero, lay = _card("hero")
        h = QHBoxLayout()
        h.setSpacing(16)
        self.cat = QLabel()
        self.cat.setFixedSize(136, 136)
        h.addWidget(self.cat, 0, Qt.AlignVCenter)
        text = QVBoxLayout()
        text.setSpacing(8)
        self.hero_text = QLabel("")
        self.hero_text.setObjectName("heroText")
        self.hero_text.setWordWrap(True)
        text.addWidget(self.hero_text)
        line = QHBoxLayout()
        line.setSpacing(8)
        cap = QLabel("Rata-rata stres")
        cap.setObjectName("cardLabel")
        self.values["avg"] = QLabel("–")
        self.values["avg"].setObjectName("cardValue")
        self.level_pill = QLabel("")
        self.level_pill.setObjectName("pill")
        self.level_pill.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        line.addWidget(cap)
        line.addWidget(self.values["avg"])
        line.addWidget(self.level_pill, 0, Qt.AlignVCenter)
        line.addStretch(1)
        text.addLayout(line)
        h.addLayout(text, 1)
        lay.addLayout(h)
        row.addWidget(hero, 3)

        ring_card, rl = _card()
        cap = QLabel("Health score")
        cap.setObjectName("h2")
        rl.addWidget(cap, 0, Qt.AlignHCenter)
        self.ring = HealthRing()
        rl.addWidget(self.ring, 1)
        self.values["health"] = QLabel("–")  # numeric copy for callers; the ring paints it
        self.values["health"].setVisible(False)
        rl.addWidget(self.values["health"])
        ring_card.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        row.addWidget(ring_card, 1)
        return row

    # -- data -----------------------------------------------------------------------------
    def selected_day(self) -> date:
        """Date chosen in the picker."""
        return self.date_edit.date().toPython()

    def shift_day(self, days: int) -> None:
        """Move the picker by `days` (not past today)."""
        self.date_edit.setDate(QDate(self.selected_day() + timedelta(days=days)))

    def refresh(self) -> None:
        """Recompute summaries and redraw."""
        self.repo.flush()
        day = self.selected_day()
        self.history = ds.last_days(self.repo, day, self.cfg)
        self.today = self.history[-1]
        t = self.today
        v = self.values
        self.subtitle.setText(fmt_day(day))
        v["work"].setText(fmt_minutes(t.work_minutes))
        v["avg"].setText("–" if t.avg_stress is None else f"{t.avg_stress:.0f}")
        v["peak"].setText("–" if t.peak_stress is None else f"{t.peak_stress:.0f} · {t.peak_time}")
        v["breaks"].setText(str(t.break_count))
        v["health"].setText("–" if t.health_score is None else f"{t.health_score:.0f}")
        v["advice"].setText(f"{t.advice_followed}/{t.advice_count}")
        v["yawn"].setText(str(t.yawn_total))
        v["stretch"].setText(fmt_minutes(t.longest_stretch_min))
        self.ring.set_score(t.health_score)

        level = stress_level(t.avg_stress, self.cfg)
        self.level_pill.setVisible(level is not None)
        if level:
            self.level_pill.setText(level.capitalize())
            self.level_pill.setStyleSheet(f"background: {theme.LEVEL_SOFT[level]}; color: {theme.LEVEL_COLORS[level]};")
        sprite, sentence = buddy_message(t)
        self.hero_text.setText(sentence)
        pix = QPixmap(str(CAT_DIR / f"{sprite}.png"))
        if not pix.isNull():
            self.cat.setPixmap(pix.scaled(self.cat.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

        self.insights.set_items(ds.insights(t, self.history))
        self._draw(day)

    def _draw(self, day: date) -> None:
        low = float(self.cfg["score"]["thresholds"]["low"])
        high = float(self.cfg["score"]["thresholds"]["high"])

        self.fig_hourly.clear()
        ax = self.fig_hourly.add_subplot(1, 1, 1)
        _style_axes(ax)
        for y0, y1, lv in ((0, low, "rendah"), (low, high, "sedang"), (high, 100, "tinggi")):
            ax.axhspan(y0, y1, color=theme.LEVEL_COLORS[lv], alpha=0.06, lw=0)
            ax.text(1.005, (y0 + y1) / 2, lv, transform=ax.get_yaxis_transform(), fontsize=8,
                    color=theme.LEVEL_COLORS[lv], va="center")
        hourly = ds.hourly_stress(self.repo, day)
        if hourly:
            hours, scores = zip(*hourly, strict=True)
            ax.fill_between(hours, scores, color=theme.ACCENT, alpha=0.12, lw=0)
            ax.plot(hours, scores, color=theme.ACCENT, lw=2, marker="o", ms=4)
            ax.set_xticks(list(hours))
            ax.set_xticklabels([f"{h:02d}.00" for h in hours])
        else:
            ax.text(0.5, 0.5, "Belum ada data", transform=ax.transAxes, ha="center", color=theme.TEXT_2)
            ax.set_xticks([])
        ax.set_ylim(0, 100)
        ax.set_ylabel("skor", fontsize=8, color=theme.TEXT_2)
        self.canvas_hourly.draw_idle()

        labels = [f"{int(d.date[8:])}/{int(d.date[5:7])}" for d in self.history]
        self.fig_breaks.clear()
        ax = self.fig_breaks.add_subplot(1, 1, 1)
        _style_axes(ax)
        counts = [d.break_count for d in self.history]
        colors = [theme.CHART_MUTED] * (len(counts) - 1) + [theme.LEVEL_COLORS["rendah"]]
        bars = ax.bar(labels, counts, color=colors, width=0.6)
        ax.bar_label(bars, fontsize=8, color=theme.TEXT_2, padding=2)
        ax.set_ylim(0, max(counts + [1]) * 1.25)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        self.canvas_breaks.draw_idle()

        self.fig_trend.clear()
        ax = self.fig_trend.add_subplot(1, 1, 1)
        _style_axes(ax)
        nan = float("nan")
        ax.plot(labels, [d.avg_stress if d.avg_stress is not None else nan for d in self.history],
                marker="o", ms=4, lw=2, color=theme.ACCENT, label="rata-rata stres")
        ax.plot(labels, [d.health_score if d.health_score is not None else nan for d in self.history],
                marker="s", ms=4, lw=2, color=theme.LEVEL_COLORS["rendah"], label="health score")
        ax.set_ylim(0, 105)
        ax.legend(fontsize=8, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
        self.canvas_trend.draw_idle()

    def export_csv(self, folder: Path | None = None) -> list[Path]:
        """Export the selected day (stress_log, events, advice_log) and all daily summaries."""
        if folder is None:
            chosen = QFileDialog.getExistingDirectory(self, "Pilih folder ekspor")
            if not chosen:
                return []
            folder = Path(chosen)
        day = self.selected_day()
        start, end = ds.day_bounds(day)
        out = []
        for table in ("stress_log", "events", "advice_log"):
            p = folder / f"{table}_{day.isoformat()}.csv"
            self.repo.export_csv(table, p, start, end)
            out.append(p)
        p = folder / "daily_summary.csv"
        self.repo.export_csv("daily_summary", p)
        out.append(p)
        return out


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    from src.core.config import load_config, resolve_path
    from src.storage.db import Database
    from src.storage.demo_data import fill_demo_data

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--demo-data", action="store_true", help="pakai database sementara berisi data dummy")
    args = ap.parse_args(argv)
    cfg = load_config()
    app = QApplication.instance() or QApplication(sys.argv)
    if args.demo_data:
        db = Database(Path(tempfile.mkdtemp()) / "demo.db")
        repo = Repository(db)
        fill_demo_data(repo, date.today())
    else:
        repo = Repository(Database(resolve_path(cfg["storage"]["db_path"])))
    w = Dashboard(repo, cfg)
    w.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
