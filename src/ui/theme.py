"""Calm, warm, high-contrast palette (from legacy/DESIGN.md, WCAG AA) shared by the windows.

The accent follows the Buddy cat (orange tabby); text colours keep AA contrast on BG and CARD.
"""

from __future__ import annotations

BG = "#F7F3EC"
SURFACE = "#F0EADF"
CARD = "#FFFFFF"
TEXT = "#1C1917"
TEXT_2 = "#57534E"
BORDER = "#E4DCCF"
BORDER_STRONG = "#A8A29E"
ACCENT = "#9A3412"
ACCENT_HOVER = "#7C2D12"
ACCENT_SOFT = "#FDEBD8"
LEVEL_COLORS: dict[str, str] = {"rendah": "#166534", "sedang": "#92400E", "tinggi": "#991B1B"}
LEVEL_SOFT: dict[str, str] = {"rendah": "#DCFCE7", "sedang": "#FEF3C7", "tinggi": "#FEE2E2"}
CHART_MUTED = "#D9CFBF"

STYLESHEET = f"""
QWidget {{ background: {BG}; color: {TEXT}; font-size: 13px; }}  /* system font: SF on macOS, Segoe on Windows */
QScrollArea, QScrollArea > QWidget > QWidget {{ background: {BG}; border: none; }}
QLabel {{ background: transparent; }}
QLabel#title {{ font-size: 22px; font-weight: 700; }}
QLabel#subtitle {{ color: {TEXT_2}; font-size: 13px; }}
QLabel#h2 {{ font-size: 15px; font-weight: 600; }}
QLabel#muted {{ color: {TEXT_2}; font-size: 12px; }}
QFrame#card {{ background: {CARD}; border: 1px solid {BORDER}; border-radius: 14px; }}
QFrame#hero {{ background: {ACCENT_SOFT}; border: 1px solid #F5D3B3; border-radius: 16px; }}
QFrame#insight {{ background: {SURFACE}; border: none; border-radius: 10px; }}
QLabel#cardValue {{ font-size: 24px; font-weight: 700; }}
QLabel#cardLabel {{ color: {TEXT_2}; font-size: 12px; }}
QLabel#cardIcon {{ font-size: 18px; }}
QLabel#heroText {{ font-size: 16px; font-weight: 600; }}
QLabel#pill {{ border-radius: 10px; padding: 2px 10px; font-size: 12px; font-weight: 600; }}
QPushButton {{ background: {ACCENT}; color: white; border: none; border-radius: 8px; padding: 7px 14px; font-weight: 600; }}
QPushButton:hover {{ background: {ACCENT_HOVER}; }}
QPushButton#secondary, QPushButton#icon {{ background: {CARD}; color: {TEXT}; border: 1px solid {BORDER}; font-weight: 500; }}
QPushButton#secondary:hover, QPushButton#icon:hover {{ background: {SURFACE}; }}
QPushButton#icon {{ padding: 4px 10px; min-width: 14px; font-size: 17px; }}
QListWidget {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 8px; padding: 4px; }}
QDateEdit, QSpinBox, QDoubleSpinBox, QComboBox {{ background: {CARD}; border: 1px solid {BORDER}; border-radius: 8px; padding: 5px 8px; }}
"""
