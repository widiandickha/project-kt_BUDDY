"""Calm, high-contrast palette (from legacy/DESIGN.md, WCAG AA) shared by the windows."""

from __future__ import annotations

BG = "#FAF8F5"
SURFACE = "#F0ECE1"
TEXT = "#1C1917"
TEXT_2 = "#44403C"
BORDER = "#78716C"
ACCENT = "#9A3412"
LEVEL_COLORS: dict[str, str] = {"rendah": "#166534", "sedang": "#92400E", "tinggi": "#991B1B"}

STYLESHEET = f"""
QWidget {{ background: {BG}; color: {TEXT}; font-size: 13px; }}
QLabel#title {{ font-size: 20px; font-weight: 600; }}
QLabel#muted {{ color: {TEXT_2}; font-size: 12px; }}
QFrame#card {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 8px; }}
QLabel#cardValue {{ font-size: 22px; font-weight: 600; background: transparent; }}
QLabel#cardLabel {{ color: {TEXT_2}; font-size: 12px; background: transparent; }}
QPushButton {{ background: {ACCENT}; color: white; border: none; border-radius: 6px; padding: 6px 14px; }}
QPushButton:hover {{ background: #7C2D12; }}
QPushButton#secondary {{ background: {SURFACE}; color: {TEXT}; border: 1px solid {BORDER}; }}
QListWidget {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 8px; padding: 4px; }}
QDateEdit, QSpinBox, QDoubleSpinBox, QComboBox {{ background: white; border: 1px solid {BORDER}; border-radius: 4px; padding: 2px 6px; }}
"""
