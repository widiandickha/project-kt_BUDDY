from pathlib import Path
from typing import Optional
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QIcon, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import QWidget

from src.config import ROOT_DIR


class CatBuddyWidget(QWidget):
    """Menggambar karakter kucing Virtual Buddy dengan 5 ekspresi fungsional.

    Jika file gambar kustom tersedia di assets/buddy/<expression>.png, widget akan
    menampilkan aset tersebut. Jika belum ada, widget menggambar ilustrasi vektor kucing.
    """

    def __init__(self, expression: str = "istirahat", size: int = 110, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._expression = expression
        self.setFixedSize(size, size)

    def set_expression(self, expression: str) -> None:
        self._expression = expression
        self.update()

    @property
    def expression(self) -> str:
        return self._expression

    def _custom_pixmap(self) -> Optional[QPixmap]:
        asset_dir = ROOT_DIR / "assets" / "buddy"
        for ext in (".png", ".jpg", ".svg"):
            candidate = asset_dir / f"{self._expression}{ext}"
            if candidate.exists():
                pix = QPixmap(str(candidate))
                if not pix.isNull():
                    return pix
        return None

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        w = float(self.width())
        h = float(self.height())

        custom = self._custom_pixmap()
        if custom is not None:
            scaled = custom.scaled(
                int(w - 8),
                int(h - 8),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
            painter.drawPixmap(int((w - scaled.width()) / 2), int((h - scaled.height()) / 2), scaled)
            painter.end()
            return

        fur_color = QColor("#D97736")
        inner_ear_color = QColor("#F3C6A8")
        outline_color = QColor("#1C1917")
        muzzle_color = QColor("#FAF3EB")
        pen_outline = QPen(outline_color, 2.4)
        painter.setPen(pen_outline)

        # Left and right ears
        left_ear = QPainterPath()
        left_ear.moveTo(w * 0.22, h * 0.42)
        left_ear.lineTo(w * 0.16, h * 0.14)
        left_ear.lineTo(w * 0.40, h * 0.28)
        left_ear.closeSubpath()
        painter.setBrush(fur_color)
        painter.drawPath(left_ear)

        right_ear = QPainterPath()
        right_ear.moveTo(w * 0.78, h * 0.42)
        right_ear.lineTo(w * 0.84, h * 0.14)
        right_ear.lineTo(w * 0.60, h * 0.28)
        right_ear.closeSubpath()
        painter.drawPath(right_ear)

        # Inner ear triangles
        painter.setBrush(inner_ear_color)
        painter.setPen(Qt.NoPen)
        inner_l = QPainterPath()
        inner_l.moveTo(w * 0.24, h * 0.37)
        inner_l.lineTo(w * 0.20, h * 0.20)
        inner_l.lineTo(w * 0.35, h * 0.29)
        inner_l.closeSubpath()
        painter.drawPath(inner_l)

        inner_r = QPainterPath()
        inner_r.moveTo(w * 0.76, h * 0.37)
        inner_r.lineTo(w * 0.80, h * 0.20)
        inner_r.lineTo(w * 0.65, h * 0.29)
        inner_r.closeSubpath()
        painter.drawPath(inner_r)

        # Head circle
        painter.setPen(pen_outline)
        painter.setBrush(fur_color)
        head_rect = QRectF(w * 0.18, h * 0.24, w * 0.64, h * 0.56)
        painter.drawEllipse(head_rect)

        # Muzzle pad
        painter.setBrush(muzzle_color)
        painter.setPen(QPen(outline_color, 1.5))
        painter.drawEllipse(QRectF(w * 0.34, h * 0.50, w * 0.32, h * 0.22))

        # Nose triangle
        painter.setBrush(QColor("#9A3412"))
        nose = QPainterPath()
        nose.moveTo(w * 0.46, h * 0.54)
        nose.lineTo(w * 0.54, h * 0.54)
        nose.lineTo(w * 0.50, h * 0.59)
        nose.closeSubpath()
        painter.drawPath(nose)

        # Whiskers
        painter.setPen(QPen(outline_color, 1.6))
        painter.drawLine(QPointF(w * 0.12, h * 0.54), QPointF(w * 0.30, h * 0.56))
        painter.drawLine(QPointF(w * 0.12, h * 0.62), QPointF(w * 0.30, h * 0.60))
        painter.drawLine(QPointF(w * 0.88, h * 0.54), QPointF(w * 0.70, h * 0.56))
        painter.drawLine(QPointF(w * 0.88, h * 0.62), QPointF(w * 0.70, h * 0.60))

        # Expression-specific eyes, mouth, and paws
        painter.setPen(QPen(outline_color, 2.4))
        if self._expression == "menguap":
            # Sleepy curved eyes + wide open yawning mouth
            painter.drawArc(QRectF(w * 0.30, h * 0.40, w * 0.14, h * 0.08), 0, 180 * 16)
            painter.drawArc(QRectF(w * 0.56, h * 0.40, w * 0.14, h * 0.08), 0, 180 * 16)
            painter.setBrush(QColor("#991B1B"))
            painter.drawEllipse(QRectF(w * 0.44, h * 0.60, w * 0.12, h * 0.13))
        elif self._expression == "minum":
            # Alert eyes + holding a water cup
            painter.setBrush(outline_color)
            painter.drawEllipse(QRectF(w * 0.33, h * 0.40, w * 0.08, h * 0.09))
            painter.drawEllipse(QRectF(w * 0.59, h * 0.40, w * 0.08, h * 0.09))
            painter.setBrush(QColor("#3B82F6"))
            cup_rect = QRectF(w * 0.38, h * 0.66, w * 0.24, h * 0.26)
            painter.drawRoundedRect(cup_rect, 4.0, 4.0)
            painter.setPen(QPen(QColor("#FFFFFF"), 1.6))
            font = QFont()
            font.setPointSize(8)
            font.setBold(True)
            painter.setFont(font)
            painter.drawText(cup_rect, Qt.AlignCenter, "AIR")
        elif self._expression == "olahraga":
            # Stretching both paws high above the head
            painter.setBrush(outline_color)
            painter.drawEllipse(QRectF(w * 0.33, h * 0.40, w * 0.08, h * 0.09))
            painter.drawEllipse(QRectF(w * 0.59, h * 0.40, w * 0.08, h * 0.09))
            painter.drawArc(QRectF(w * 0.43, h * 0.58, w * 0.14, h * 0.08), 180 * 16, 180 * 16)
            painter.setBrush(fur_color)
            painter.drawRoundedRect(QRectF(w * 0.06, h * 0.22, w * 0.12, h * 0.34), 6.0, 6.0)
            painter.drawRoundedRect(QRectF(w * 0.82, h * 0.22, w * 0.12, h * 0.34), 6.0, 6.0)
        elif self._expression == "jalan":
            # Stepping paws + upright tail on the right
            painter.setBrush(outline_color)
            painter.drawEllipse(QRectF(w * 0.33, h * 0.40, w * 0.08, h * 0.09))
            painter.drawEllipse(QRectF(w * 0.59, h * 0.40, w * 0.08, h * 0.09))
            painter.drawArc(QRectF(w * 0.43, h * 0.58, w * 0.14, h * 0.08), 180 * 16, 180 * 16)
            painter.setBrush(fur_color)
            painter.drawEllipse(QRectF(w * 0.24, h * 0.77, w * 0.18, h * 0.14))
            painter.drawEllipse(QRectF(w * 0.58, h * 0.72, w * 0.18, h * 0.14))
            tail = QPainterPath()
            tail.moveTo(w * 0.80, h * 0.72)
            tail.cubicTo(w * 0.96, h * 0.62, w * 0.94, h * 0.36, w * 0.86, h * 0.30)
            painter.setBrush(Qt.NoBrush)
            painter.setPen(QPen(fur_color, 5.0, Qt.SolidLine, Qt.RoundCap))
            painter.drawPath(tail)
        else:
            # istirahat: calm closed eyes and resting front paws
            painter.drawArc(QRectF(w * 0.30, h * 0.42, w * 0.14, h * 0.07), 180 * 16, 180 * 16)
            painter.drawArc(QRectF(w * 0.56, h * 0.42, w * 0.14, h * 0.07), 180 * 16, 180 * 16)
            painter.drawArc(QRectF(w * 0.43, h * 0.59, w * 0.14, h * 0.06), 180 * 16, 180 * 16)
            painter.setBrush(fur_color)
            painter.drawEllipse(QRectF(w * 0.28, h * 0.75, w * 0.18, h * 0.12))
            painter.drawEllipse(QRectF(w * 0.54, h * 0.75, w * 0.18, h * 0.12))

        painter.end()


def build_cat_icon(size: int = 64) -> QIcon:
    widget = CatBuddyWidget(expression="istirahat", size=size)
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.transparent)
    widget.render(pixmap)
    return QIcon(pixmap)
