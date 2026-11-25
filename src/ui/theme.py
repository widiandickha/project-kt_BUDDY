"""Calm, warm, modern palette (WCAG AA) shared by all windows, menus, and dialogs.

The accent follows the Buddy cat (orange tabby); text colours keep AA contrast on BG and CARD.
"""

from __future__ import annotations

BG = "#F8F5EE"
SURFACE = "#EFEAE0"
CARD = "#FFFFFF"
TEXT = "#1C1917"
TEXT_2 = "#57534E"
BORDER = "#E6DEC8"
BORDER_STRONG = "#A8A29E"
ACCENT = "#C2410C"
ACCENT_HOVER = "#9A3412"
ACCENT_SOFT = "#FFF1E5"
LEVEL_COLORS: dict[str, str] = {"rendah": "#15803D", "sedang": "#B45309", "tinggi": "#B91C1C"}
LEVEL_SOFT: dict[str, str] = {"rendah": "#DCFCE7", "sedang": "#FEF3C7", "tinggi": "#FEE2E2"}
CHART_MUTED = "#DDD4C6"

MENU_STYLESHEET = f"""
QMenu {{
    background: {CARD};
    color: {TEXT};
    border: 1px solid {BORDER};
    border-radius: 12px;
    padding: 6px;
    font-size: 13px;
}}
QMenu::item {{
    padding: 8px 18px 8px 12px;
    border-radius: 8px;
    margin: 1px 2px;
    font-weight: 500;
}}
QMenu::item:selected {{
    background: {ACCENT_SOFT};
    color: {ACCENT_HOVER};
    font-weight: 600;
}}
QMenu::item:disabled {{
    color: {TEXT_2};
    font-size: 12px;
    font-weight: 600;
}}
QMenu::separator {{
    height: 1px;
    background: {BORDER};
    margin: 5px 8px;
}}
"""

STYLESHEET = f"""
QWidget {{ background: {BG}; color: {TEXT}; font-size: 13px; }}
QScrollArea, QScrollArea > QWidget > QWidget {{ background: {BG}; border: none; }}
QLabel {{ background: transparent; }}
QLabel#title {{ font-size: 22px; font-weight: 800; color: {TEXT}; }}
QLabel#subtitle {{ color: {TEXT_2}; font-size: 13px; }}
QLabel#h2 {{ font-size: 15px; font-weight: 700; color: {TEXT}; }}
QLabel#muted {{ color: {TEXT_2}; font-size: 12px; }}
QFrame#card {{ background: {CARD}; border: 1px solid {BORDER}; border-radius: 16px; }}
QFrame#hero {{ background: {ACCENT_SOFT}; border: 1px solid #F5D0B5; border-radius: 18px; }}
QFrame#conclusion {{ background: {CARD}; border: 1.5px solid #F2C7A5; border-radius: 16px; }}
QFrame#insight {{ background: {SURFACE}; border: none; border-radius: 12px; }}
QLabel#cardValue {{ font-size: 24px; font-weight: 800; color: {TEXT}; }}
QLabel#cardLabel {{ color: {TEXT_2}; font-size: 12px; font-weight: 500; }}
QLabel#cardIcon {{ font-size: 18px; }}
QLabel#heroText {{ font-size: 17px; font-weight: 700; color: {TEXT}; }}
QLabel#heroSub {{ font-size: 13px; color: {TEXT_2}; }}
QLabel#pill {{ border-radius: 10px; padding: 3px 11px; font-size: 12px; font-weight: 700; }}
QPushButton {{
    background: {ACCENT};
    color: white;
    border: none;
    border-radius: 10px;
    padding: 8px 16px;
    font-weight: 600;
}}
QPushButton:hover {{ background: {ACCENT_HOVER}; }}
QPushButton#secondary, QPushButton#icon {{
    background: {CARD};
    color: {TEXT};
    border: 1px solid {BORDER};
    font-weight: 600;
}}
QPushButton#secondary:hover, QPushButton#icon:hover {{ background: {SURFACE}; }}
QPushButton#icon {{ padding: 5px 11px; min-width: 14px; font-size: 16px; }}
QListWidget {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 10px; padding: 4px; }}
QDateEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    background: {CARD};
    border: 1px solid {BORDER};
    border-radius: 9px;
    padding: 6px 10px;
    font-weight: 500;
}}
QProgressBar {{
    background: {SURFACE};
    border: none;
    border-radius: 6px;
    text-align: center;
    font-weight: 600;
}}
QProgressBar::chunk {{
    background: {ACCENT};
    border-radius: 6px;
}}
{MENU_STYLESHEET}
"""
