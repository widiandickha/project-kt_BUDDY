"""Final class list of the face detector (spec 2.1). Never reorder: indices are baked into labels."""

from __future__ import annotations

CLASS_NAMES: tuple[str, ...] = ("mata_tertutup", "mata_terbuka", "menguap", "kerutan_dahi")
CLASS_INDEX: dict[str, int] = {name: idx for idx, name in enumerate(CLASS_NAMES)}

MATA_TERTUTUP: int = CLASS_INDEX["mata_tertutup"]
MATA_TERBUKA: int = CLASS_INDEX["mata_terbuka"]
MENGUAP: int = CLASS_INDEX["menguap"]
KERUTAN_DAHI: int = CLASS_INDEX["kerutan_dahi"]
