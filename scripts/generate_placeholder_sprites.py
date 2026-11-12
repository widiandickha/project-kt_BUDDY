#!/usr/bin/env python3
"""
generate_placeholder_sprites.py
Membuat sprite PLACEHOLDER untuk Virtual Buddy (kucing) + animations.yaml.

Pakai:
    python scripts/generate_placeholder_sprites.py
    python scripts/generate_placeholder_sprites.py --out src/buddy/assets/cat --size 512
    python scripts/generate_placeholder_sprites.py --force      # timpa file yang sudah ada

Aturan penting:
  * File sprite ASLI milikmu TIDAK akan ditimpa kecuali pakai --force.
    Cukup ganti file <nama>.png dengan gambar aslimu (nama sama, ukuran kanvas sama).
  * Semua sprite: PNG transparan, kanvas persegi, titik pijakan kaki = bawah-tengah.
Butuh: pip install pillow pyyaml
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import yaml
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Definisi 12 ekspresi (nama file = nama key di sini)
#   eyes : open | closed | half | wide | happy | wink
#   mouth: smile | open | flat | yawn | none
#   brow : none | worried | angry
#   transform: efek gerak yang dijalankan sprite_player (kode, bukan gambar)
#   overlay  : efek tambahan yang digambar kode (zzz, sweat, dll)
# ---------------------------------------------------------------------------
EXPRESSIONS: dict[str, dict] = {
    "idle":       dict(eyes="open",   mouth="smile", brow="none",    body="sit",
                       transform="breathe", overlay=None,       loop=True,  duration_ms=2400),
    "happy":      dict(eyes="happy",  mouth="open",  brow="none",    body="sit",
                       transform="bounce",  overlay="sparkle",  loop=False, duration_ms=1200, next="idle"),
    "sleepy":     dict(eyes="half",   mouth="yawn",  brow="none",    body="sit",
                       transform="nod",     overlay=None,       loop=True,  duration_ms=3000),
    "worried":    dict(eyes="open",   mouth="flat",  brow="worried", body="sit",
                       transform="sway",    overlay="sweat",    loop=True,  duration_ms=1800),
    "alert":      dict(eyes="wide",   mouth="open",  brow="none",    body="stand",
                       transform="pop",     overlay="exclaim",  loop=False, duration_ms=1500, next="idle"),
    "concerned":  dict(eyes="open",   mouth="flat",  brow="worried", body="sit",
                       transform="breathe", overlay=None,       loop=True,  duration_ms=2600),
    "stretch":    dict(eyes="closed", mouth="open",  brow="none",    body="stretch",
                       transform="stretch", overlay=None,       loop=False, duration_ms=2000, next="idle"),
    "curious":    dict(eyes="wide",   mouth="smile", brow="none",    body="sit",
                       transform="tilt",    overlay="question", loop=False, duration_ms=1600, next="idle"),
    "eye_care":   dict(eyes="closed", mouth="smile", brow="none",    body="sit",
                       transform="slow_blink", overlay=None,    loop=False, duration_ms=2200, next="idle"),
    "sleeping":   dict(eyes="closed", mouth="none",  brow="none",    body="curl",
                       transform="breathe_slow", overlay="zzz", loop=True,  duration_ms=4000),
    "wake_greet": dict(eyes="happy",  mouth="open",  brow="none",    body="stand",
                       transform="wave",    overlay=None,       loop=False, duration_ms=1800, next="idle"),
    "celebrate":  dict(eyes="happy",  mouth="open",  brow="none",    body="stand",
                       transform="bounce",  overlay="confetti", loop=False, duration_ms=2000, next="idle"),
}

ORANGE = (253, 186, 100, 255)
ORANGE_DARK = (250, 128, 100, 255)
OUTLINE = (40, 50, 110, 255)
PINK = (240, 110, 150, 255)
LABEL_BG = (255, 255, 255, 215)
LABEL_FG = (60, 60, 60, 255)


def _font(size: int) -> ImageFont.ImageFont:
    for name in ("DejaVuSans-Bold.ttf", "Arial.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size)  # Pillow >= 10.1
    except TypeError:
        return ImageFont.load_default()


def _body_params(kind: str, s: int):
    """Return (body_box, head_center, head_r) dalam koordinat kanvas."""
    cx = s // 2
    base = int(s * 0.96)          # garis pijakan kaki
    if kind == "curl":            # meringkuk: lebar & pendek
        body = (cx - s * .34, base - s * .34, cx + s * .26, base)
        head = (cx - s * .14, base - s * .28, s * .19)
    elif kind == "stand":
        body = (cx - s * .22, base - s * .46, cx + s * .22, base)
        head = (cx, base - s * .56, s * .25)
    elif kind == "stretch":
        body = (cx - s * .34, base - s * .30, cx + s * .30, base)
        head = (cx - s * .22, base - s * .38, s * .22)
    else:                          # sit
        body = (cx - s * .27, base - s * .44, cx + s * .27, base)
        head = (cx, base - s * .50, s * .26)
    return body, head[:2], head[2]


def _draw_eyes(d: ImageDraw.ImageDraw, hc, r, style: str, w: int):
    ex = r * 0.45
    ey = hc[1] - r * 0.02
    for sx in (-1, 1):
        x, y = hc[0] + sx * ex, ey
        er = r * 0.13
        if style == "open":
            d.ellipse((x - er, y - er, x + er, y + er), fill=OUTLINE)
        elif style == "wide":
            er *= 1.5
            d.ellipse((x - er, y - er, x + er, y + er), fill=(255, 255, 255, 255), outline=OUTLINE, width=w)
            d.ellipse((x - er * .5, y - er * .5, x + er * .5, y + er * .5), fill=OUTLINE)
        elif style == "closed":
            d.arc((x - er * 1.4, y - er, x + er * 1.4, y + er * 1.4), 10, 170, fill=OUTLINE, width=w)
        elif style == "half":
            d.ellipse((x - er, y - er, x + er, y + er), fill=OUTLINE)
            d.rectangle((x - er * 1.2, y - er * 1.2, x + er * 1.2, y - er * 0.1), fill=ORANGE)
            d.line((x - er * 1.2, y - er * .1, x + er * 1.2, y - er * .1), fill=OUTLINE, width=w)
        elif style == "happy":  # ^ ^
            d.arc((x - er * 1.4, y - er * .6, x + er * 1.4, y + er * 1.6), 190, 350, fill=OUTLINE, width=w)
        elif style == "wink":
            if sx < 0:
                d.arc((x - er * 1.4, y - er, x + er * 1.4, y + er * 1.4), 10, 170, fill=OUTLINE, width=w)
            else:
                d.ellipse((x - er, y - er, x + er, y + er), fill=OUTLINE)


def _draw_mouth(d: ImageDraw.ImageDraw, hc, r, style: str, w: int):
    x, y = hc[0], hc[1] + r * 0.40
    if style == "smile":
        d.arc((x - r * .22, y - r * .15, x + r * .22, y + r * .22), 20, 160, fill=OUTLINE, width=w)
    elif style == "open":
        d.ellipse((x - r * .14, y - r * .02, x + r * .14, y + r * .28), fill=OUTLINE)
        d.ellipse((x - r * .08, y + r * .12, x + r * .08, y + r * .26), fill=PINK)
    elif style == "yawn":
        d.ellipse((x - r * .20, y - r * .05, x + r * .20, y + r * .42), fill=OUTLINE)
        d.ellipse((x - r * .12, y + r * .20, x + r * .12, y + r * .40), fill=PINK)
    elif style == "flat":
        d.line((x - r * .16, y + r * .05, x + r * .16, y + r * .05), fill=OUTLINE, width=w)
    # none -> tidak digambar
    d.ellipse((x - r * .05, y - r * .22, x + r * .05, y - r * .14), fill=OUTLINE)  # hidung


def _draw_brows(d: ImageDraw.ImageDraw, hc, r, style: str, w: int):
    if style == "none":
        return
    ex, y = r * 0.45, hc[1] - r * 0.30
    for sx in (-1, 1):
        x = hc[0] + sx * ex
        inner = (x - sx * r * .18, y - r * .10)   # sisi dalam
        outer = (x + sx * r * .18, y + r * .04)
        if style == "worried":
            inner, outer = (x - sx * r * .18, y - r * .10), (x + sx * r * .18, y + r * .06)
        elif style == "angry":
            inner, outer = (x - sx * r * .18, y + r * .06), (x + sx * r * .18, y - r * .10)
        d.line((inner, outer), fill=OUTLINE, width=w)


def make_sprite(name: str, spec: dict, size: int) -> Image.Image:
    # gambar pada kanvas 2x lalu perkecil -> tepi halus
    S = size * 2
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    w = max(3, S // 110)

    body, hc, r = _body_params(spec["body"], S)

    # ekor
    tx = body[2] - S * .02
    ty = body[3] - S * .12
    d.arc((tx - S * .10, ty - S * .30, tx + S * .22, ty + S * .02), 270, 100, fill=ORANGE, width=int(S * .06))

    # badan
    d.ellipse(body, fill=ORANGE)
    # belang badan
    bx = (body[0] + body[2]) / 2
    for i in (-1, 0, 1):
        d.line((bx + i * S * .07, body[1] + S * .03, bx + i * S * .07 + S * .02, body[1] + S * .10),
               fill=ORANGE_DARK, width=int(S * .02))

    # telinga
    for sx in (-1, 1):
        ex = hc[0] + sx * r * .62
        d.polygon([(ex - sx * r * .35, hc[1] - r * .55),
                   (ex + sx * r * .05, hc[1] - r * 1.25),
                   (ex + sx * r * .45, hc[1] - r * .55)], fill=ORANGE)
        d.polygon([(ex - sx * r * .15, hc[1] - r * .62),
                   (ex + sx * r * .05, hc[1] - r * 1.0),
                   (ex + sx * r * .28, hc[1] - r * .62)], fill=ORANGE_DARK)

    # kepala
    d.ellipse((hc[0] - r, hc[1] - r * .9, hc[0] + r, hc[1] + r * .9), fill=ORANGE)
    # pipi
    for sx in (-1, 1):
        cx_, cy_ = hc[0] + sx * r * .72, hc[1] + r * .20
        d.ellipse((cx_ - r * .16, cy_ - r * .12, cx_ + r * .16, cy_ + r * .12), fill=ORANGE_DARK)

    _draw_eyes(d, hc, r, spec["eyes"], w)
    _draw_brows(d, hc, r, spec["brow"], w)
    _draw_mouth(d, hc, r, spec["mouth"], w)

    img = img.resize((size, size), Image.LANCZOS)

    # label placeholder (ganti PNG-nya dengan sprite asli untuk menghapus label)
    d2 = ImageDraw.Draw(img)
    f = _font(max(12, size // 22))
    text = f"PLACEHOLDER\n{name}"
    bbox = d2.multiline_textbbox((0, 0), text, font=f, align="center")
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    pad = 6
    x0, y0 = (size - tw) // 2, int(size * 0.02)
    d2.rounded_rectangle((x0 - pad, y0 - pad, x0 + tw + pad, y0 + th + pad), 8, fill=LABEL_BG)
    d2.multiline_text((x0, y0), text, font=f, fill=LABEL_FG, align="center")
    return img


def build_animations_yaml(size: int) -> dict:
    anims = {}
    for name, spec in EXPRESSIONS.items():
        a = {
            "pose": f"{name}.png",          # ganti PNG ini dengan sprite asli
            "mode": "static_pose",          # static_pose | frames (kalau nanti ada frame animasi)
            "transform": spec["transform"],  # diimplementasi di sprite_player.py
            "overlay": spec["overlay"],      # digambar kode: zzz, sweat, exclaim, sparkle, confetti, question
            "loop": spec["loop"],
            "duration_ms": spec["duration_ms"],
            "crossfade_ms": 150,
        }
        if "next" in spec:
            a["next"] = spec["next"]
        anims[name] = a
    return {
        "canvas_size": size,
        "anchor": "bottom_center",  # titik pijakan kaki
        "default_state": "idle",
        "animations": anims,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate placeholder sprites Virtual Buddy")
    ap.add_argument("--out", default="src/buddy/assets/cat", help="folder output")
    ap.add_argument("--size", type=int, default=512, help="ukuran kanvas persegi (px)")
    ap.add_argument("--force", action="store_true", help="timpa file yang sudah ada")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    created, skipped = [], []
    for name, spec in EXPRESSIONS.items():
        p = out / f"{name}.png"
        if p.exists() and not args.force:
            skipped.append(name)
            continue
        make_sprite(name, spec, args.size).save(p)
        created.append(name)

    yml = out / "animations.yaml"
    if not yml.exists() or args.force:
        with open(yml, "w", encoding="utf-8") as f:
            yaml.safe_dump(build_animations_yaml(args.size), f, sort_keys=False, allow_unicode=True)
        print(f"[ok] {yml}")
    else:
        print(f"[skip] {yml} sudah ada (pakai --force untuk menimpa)")

    # contact sheet untuk preview cepat
    cols = 4
    rows = math.ceil(len(EXPRESSIONS) / cols)
    th = 192
    sheet = Image.new("RGBA", (cols * th, rows * th), (235, 235, 240, 255))
    for i, name in enumerate(EXPRESSIONS):
        im = Image.open(out / f"{name}.png").convert("RGBA").resize((th, th), Image.LANCZOS)
        sheet.alpha_composite(im, ((i % cols) * th, (i // cols) * th))
    sheet.convert("RGB").save(out / "_preview_sheet.png")

    print(f"[ok] dibuat : {len(created)} -> {', '.join(created) or '-'}")
    print(f"[skip] ada  : {len(skipped)} -> {', '.join(skipped) or '-'}")
    print(f"[ok] preview: {out / '_preview_sheet.png'}")


if __name__ == "__main__":
    main()
