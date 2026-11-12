from typing import Dict

PALETTES: Dict[str, Dict[str, str]] = {
    "light": {
        "bg": "#FAF8F5",
        "surface": "#F0ECE1",
        "text": "#1C1917",
        "text_muted": "#44403C",
        "border": "#78716C",
        "accent": "#9A3412",
        "accent_text": "#FFFFFF",
        "normal": "#166534",
        "waspada": "#92400E",
        "tinggi": "#991B1B",
    },
    "dark": {
        "bg": "#18181B",
        "surface": "#27272A",
        "text": "#F4F4F5",
        "text_muted": "#D4D4D8",
        "border": "#A1A1AA",
        "accent": "#FDBA74",
        "accent_text": "#18181B",
        "normal": "#86EFAC",
        "waspada": "#FCD34D",
        "tinggi": "#FCA5A5",
    },
}


def get_palette(theme_name: str) -> Dict[str, str]:
    return PALETTES.get(theme_name, PALETTES["light"])


def build_stylesheet(theme_name: str) -> str:
    p = get_palette(theme_name)
    return f"""
    QMainWindow, QWidget {{
        background-color: {p['bg']};
        color: {p['text']};
        font-family: "Helvetica Neue", "SF Pro Text", "Segoe UI", sans-serif;
        font-size: 13px;
    }}
    QLabel, QCheckBox {{
        background-color: transparent;
    }}
    QFrame#SurfacePanel {{
        background-color: {p['surface']};
        border: 1px solid {p['border']};
        border-radius: 6px;
    }}
    QFrame#TopStatusBar {{
        background-color: {p['surface']};
        border-bottom: 2px solid {p['border']};
    }}
    QLabel#HeadingLarge {{
        font-size: 20px;
        font-weight: 700;
        color: {p['text']};
    }}
    QLabel#HeadingMedium {{
        font-size: 15px;
        font-weight: 600;
        color: {p['text']};
    }}
    QLabel#ScoreFocal {{
        font-size: 38px;
        font-weight: 800;
        color: {p['text']};
    }}
    QLabel#MutedText {{
        color: {p['text_muted']};
        font-size: 12px;
    }}
    QPushButton {{
        background-color: {p['surface']};
        color: {p['text']};
        border: 1px solid {p['border']};
        border-radius: 5px;
        padding: 7px 14px;
        min-height: 28px;
        font-weight: 600;
    }}
    QPushButton:hover {{
        border: 2px solid {p['accent']};
    }}
    QPushButton:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus,
    QLineEdit:focus, QCheckBox:focus, QTableWidget:focus {{
        border: 2px solid {p['accent']};
        outline: 2px solid {p['accent']};
    }}
    QPushButton#PrimaryButton {{
        background-color: {p['accent']};
        color: {p['accent_text']};
        border: 1px solid {p['accent']};
        font-weight: 700;
    }}
    QPushButton#DangerButton {{
        background-color: {p['surface']};
        color: {p['tinggi']};
        border: 2px solid {p['tinggi']};
        font-weight: 700;
    }}
    QTabWidget::pane {{
        border: 1px solid {p['border']};
        background-color: {p['bg']};
        top: -1px;
    }}
    QTabBar::tab {{
        background-color: {p['surface']};
        color: {p['text_muted']};
        border: 1px solid {p['border']};
        padding: 9px 18px;
        margin-right: 4px;
        font-weight: 600;
        border-top-left-radius: 5px;
        border-top-right-radius: 5px;
    }}
    QTabBar::tab:selected {{
        background-color: {p['bg']};
        color: {p['text']};
        border-bottom: 3px solid {p['accent']};
    }}
    QTableWidget {{
        background-color: {p['bg']};
        color: {p['text']};
        gridline-color: {p['border']};
        border: 1px solid {p['border']};
    }}
    QHeaderView::section {{
        background-color: {p['surface']};
        color: {p['text']};
        border: 1px solid {p['border']};
        padding: 6px;
        font-weight: 700;
    }}
    QSpinBox, QDoubleSpinBox, QLineEdit, QComboBox {{
        background-color: {p['bg']};
        color: {p['text']};
        border: 1px solid {p['border']};
        border-radius: 4px;
        padding: 5px 8px;
        min-height: 26px;
    }}
    QProgressBar {{
        border: 1px solid {p['border']};
        border-radius: 4px;
        background-color: {p['bg']};
        text-align: center;
        color: {p['text']};
        font-weight: 700;
        min-height: 22px;
    }}
    QProgressBar::chunk {{
        background-color: {p['surface']};
        border-right: 3px solid {p['accent']};
    }}
    """
