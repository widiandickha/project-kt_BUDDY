"""Daily dashboard (spec FASE 6): summary cards, stress per hour, breaks over 7 days,
7-day comparison, automatic insights, CSV export.

    python -m src.ui.dashboard                  # real database (data/stress_buddy.db)
    python -m src.ui.dashboard --demo-data      # temporary database with 7 days of dummy data
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from datetime import date
from pathlib import Path
from typing import Any

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import QDate
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QDateEdit,
)

from src.storage import daily_summary as ds
from src.storage.repository import Repository
from src.ui import theme

DISCLAIMER = "Alat bantu kesadaran diri, bukan alat diagnosis medis."


def _card(label: str) -> tuple[QFrame, QLabel]:
    frame = QFrame()
    frame.setObjectName("card")
    lay = QVBoxLayout(frame)
    value = QLabel("–")
    value.setObjectName("cardValue")
    cap = QLabel(label)
    cap.setObjectName("cardLabel")
    lay.addWidget(value)
    lay.addWidget(cap)
    return frame, value


def fmt_minutes(m: float) -> str:
    """'3 j 25 m' style duration."""
    h, mm = divmod(int(round(m)), 60)
    return f"{h} j {mm} m" if h else f"{mm} m"


class Dashboard(QWidget):
    """Reads from the repository; call refresh() to recompute."""

    CARDS: tuple[tuple[str, str], ...] = (
        ("work", "Waktu kerja"), ("avg", "Rata-rata stres"), ("peak", "Stres puncak"), ("breaks", "Istirahat ≥ 5 menit"),
        ("health", "Health score"), ("advice", "Saran diikuti"), ("yawn", "Menguap"), ("stretch", "Kerja terlama tanpa jeda"),
    )

    def __init__(self, repo: Repository, cfg: dict[str, Any]) -> None:
        super().__init__()
        self.repo, self.cfg = repo, cfg
        self.setWindowTitle("Stress Buddy · Dashboard")
        self.setStyleSheet(theme.STYLESHEET)
        self.resize(980, 760)
        root = QVBoxLayout(self)

        head = QHBoxLayout()
        title = QLabel("Ringkasan harian")
        title.setObjectName("title")
        head.addWidget(title)
        head.addStretch(1)
        self.date_edit = QDateEdit(QDate.currentDate())
        self.date_edit.setCalendarPopup(True)
        self.date_edit.dateChanged.connect(lambda _: self.refresh())
        head.addWidget(self.date_edit)
        refresh = QPushButton("Muat ulang")
        refresh.setObjectName("secondary")
        refresh.clicked.connect(self.refresh)
        head.addWidget(refresh)
        export = QPushButton("Ekspor CSV")
        export.clicked.connect(self.export_csv)
        head.addWidget(export)
        root.addLayout(head)

        grid = QGridLayout()
        self.values: dict[str, QLabel] = {}
        for i, (key, label) in enumerate(self.CARDS):
            frame, value = _card(label)
            self.values[key] = value
            grid.addWidget(frame, i // 4, i % 4)
        root.addLayout(grid)

        self.figure = Figure(figsize=(9, 4.2), tight_layout=True, facecolor=theme.BG)
        self.canvas = FigureCanvasQTAgg(self.figure)
        root.addWidget(self.canvas, 1)

        bottom = QHBoxLayout()
        box = QVBoxLayout()
        box.addWidget(QLabel("Insight otomatis"))
        self.insights = QListWidget()
        box.addWidget(self.insights)
        bottom.addLayout(box, 1)
        root.addLayout(bottom)
        note = QLabel(DISCLAIMER + " Hanya angka indikator yang disimpan, tanpa gambar atau video.")
        note.setObjectName("muted")
        root.addWidget(note)
        self.today: ds.DailySummary | None = None
        self.history: list[ds.DailySummary] = []
        self.refresh()

    def selected_day(self) -> date:
        """Date chosen in the picker."""
        return self.date_edit.date().toPython()

    def refresh(self) -> None:
        """Recompute summaries and redraw."""
        self.repo.flush()
        day = self.selected_day()
        self.history = ds.last_days(self.repo, day, self.cfg)
        self.today = self.history[-1]
        t = self.today
        v = self.values
        v["work"].setText(fmt_minutes(t.work_minutes))
        v["avg"].setText("–" if t.avg_stress is None else f"{t.avg_stress:.0f}")
        v["peak"].setText("–" if t.peak_stress is None else f"{t.peak_stress:.0f} · {t.peak_time}")
        v["breaks"].setText(str(t.break_count))
        v["health"].setText("–" if t.health_score is None else f"{t.health_score:.0f}")
        v["advice"].setText(f"{t.advice_followed}/{t.advice_count}")
        v["yawn"].setText(str(t.yawn_total))
        v["stretch"].setText(fmt_minutes(t.longest_stretch_min))
        self.insights.clear()
        self.insights.addItems(ds.insights(t, self.history))
        self._draw(day)

    def _draw(self, day: date) -> None:
        self.figure.clear()
        ax1 = self.figure.add_subplot(1, 3, 1)
        hourly = ds.hourly_stress(self.repo, day)
        if hourly:
            hours, scores = zip(*hourly, strict=True)
            ax1.plot(hours, scores, marker="o", color=theme.ACCENT)
        ax1.axhline(float(self.cfg["score"]["thresholds"]["low"]), color=theme.LEVEL_COLORS["sedang"], ls="--", lw=0.8)
        ax1.axhline(float(self.cfg["score"]["thresholds"]["high"]), color=theme.LEVEL_COLORS["tinggi"], ls="--", lw=0.8)
        ax1.set(title="Stres per jam", xlabel="jam", ylabel="skor", ylim=(0, 100))

        labels = [d.date[5:] for d in self.history]
        ax2 = self.figure.add_subplot(1, 3, 2)
        ax2.bar(labels, [d.break_count for d in self.history], color=theme.LEVEL_COLORS["rendah"])
        ax2.set(title="Istirahat per hari")
        ax2.tick_params(axis="x", labelrotation=45)

        ax3 = self.figure.add_subplot(1, 3, 3)
        ax3.plot(labels, [d.avg_stress if d.avg_stress is not None else float("nan") for d in self.history],
                 marker="o", color=theme.ACCENT, label="rata-rata stres")
        ax3.plot(labels, [d.health_score if d.health_score is not None else float("nan") for d in self.history],
                 marker="s", color=theme.LEVEL_COLORS["rendah"], label="health score")
        ax3.set(title="Perbandingan 7 hari", ylim=(0, 100))
        ax3.tick_params(axis="x", labelrotation=45)
        ax3.legend(fontsize=8)
        self.canvas.draw_idle()

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
