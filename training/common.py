"""Shared dataset helpers: raw file naming scheme, YOLO label I/O, image discovery.

Raw file name scheme (docs/DECISIONS.md D-008):

    <subject>__<hint>__<condition>__<frame_id>.<ext>

* subject   : person id, lowercase [a-z0-9-], e.g. ``self-andi``, ``nthu-001``, ``uta-12``.
              The split is done per subject, so it must identify one person.
* hint      : what the person was asked to do / source folder (``mata_tertutup``, ``menguap``,
              ``normal`` ...). Only a hint; the real class comes from the label file.
* condition : free tag for robustness tests (``glasses``, ``redup``, ``unknown`` ...).
* frame_id  : unique within subject.

The ``__`` separators survive annotation tools that append suffixes (e.g. Roboflow's
``_jpg.rf.<hash>``), so the subject can always be recovered from the file name.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

IMAGE_EXTS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".bmp"})
SEP = "__"
_TOKEN_RE = re.compile(r"[^a-z0-9-]+")


@dataclass(frozen=True)
class RawName:
    """Parsed raw file name."""

    subject: str
    hint: str
    condition: str
    frame_id: str


@dataclass(frozen=True)
class YoloBox:
    """One YOLO label line: class id and normalized center/size."""

    cls: int
    cx: float
    cy: float
    w: float
    h: float

    def to_line(self) -> str:
        """Format as a YOLO txt line."""
        return f"{self.cls} {self.cx:.6f} {self.cy:.6f} {self.w:.6f} {self.h:.6f}"


def slug(text: str) -> str:
    """Normalize a token to lowercase [a-z0-9-] (underscores and spaces become '-')."""
    cleaned = _TOKEN_RE.sub("-", text.strip().lower()).strip("-")
    if not cleaned:
        raise ValueError(f"Token '{text}' is empty after normalization")
    return cleaned


def make_raw_name(subject: str, hint: str, condition: str, frame_id: str | int, ext: str = ".jpg") -> str:
    """Build a raw file name following the scheme above."""
    hint_token = hint.strip().lower()
    if not re.fullmatch(r"[a-z0-9_]+", hint_token):
        raise ValueError(f"Hint '{hint}' must match [a-z0-9_]+")
    parts = [slug(subject), hint_token, slug(condition), slug(str(frame_id))]
    return SEP.join(parts) + ext


def parse_raw_name(path: str | Path) -> RawName:
    """Recover subject/hint/condition/frame from a (possibly tool-renamed) raw file name."""
    stem = Path(path).name
    parts = stem.split(SEP)
    if len(parts) < 4:
        raise ValueError(f"File name does not follow the raw scheme: {stem}")
    frame_id = re.split(r"[._]", parts[3], maxsplit=1)[0]
    return RawName(subject=parts[0], hint=parts[1], condition=parts[2], frame_id=frame_id)


def iter_images(root: Path) -> list[Path]:
    """All image files under root, sorted, skipping hidden files."""
    return sorted(
        p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS and not p.name.startswith(".")
    )


def label_path_for(image: Path) -> Path:
    """Label next to the image, or in a sibling labels/ dir (Roboflow/YOLO export layout)."""
    beside = image.with_suffix(".txt")
    if beside.exists():
        return beside
    parts = list(image.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            return Path(*parts).with_suffix(".txt")
    return beside


def read_labels(path: Path, num_classes: int) -> list[YoloBox]:
    """Read and validate a YOLO txt file (empty file = negative sample)."""
    boxes: list[YoloBox] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"{path}:{lineno}: expected 5 fields, got {len(fields)}")
        cls = int(fields[0])
        cx, cy, w, h = (float(v) for v in fields[1:])
        if not 0 <= cls < num_classes:
            raise ValueError(f"{path}:{lineno}: class {cls} out of range 0..{num_classes - 1}")
        if not all(0.0 <= v <= 1.0 for v in (cx, cy, w, h)) or w <= 0 or h <= 0:
            raise ValueError(f"{path}:{lineno}: coordinates must be normalized to (0, 1]")
        boxes.append(YoloBox(cls, cx, cy, w, h))
    return boxes


def write_labels(path: Path, boxes: list[YoloBox]) -> None:
    """Write a YOLO txt file (empty when there are no boxes)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(b.to_line() + "\n" for b in boxes), encoding="utf-8")


def xyxy_to_yolo(cls: int, x1: float, y1: float, x2: float, y2: float, width: int, height: int) -> YoloBox | None:
    """Pixel corners to a clipped, normalized YOLO box; None if nothing is left after clipping."""
    x1, x2 = max(0.0, min(x1, x2)), min(float(width), max(x1, x2))
    y1, y2 = max(0.0, min(y1, y2)), min(float(height), max(y1, y2))
    if x2 - x1 < 1.0 or y2 - y1 < 1.0:
        return None
    return YoloBox(
        cls=cls,
        cx=(x1 + x2) / 2.0 / width,
        cy=(y1 + y2) / 2.0 / height,
        w=(x2 - x1) / width,
        h=(y2 - y1) / height,
    )
