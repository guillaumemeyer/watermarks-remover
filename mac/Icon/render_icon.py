#!/usr/bin/env python3
"""Render the Watermarker app icon to an edge-to-edge 1024x1024 PNG.

The shipped ``appicon-1024.png`` is the output of this script, so the artwork
is source-controlled as code rather than as an opaque binary. To use different
artwork instead, drop a square PNG at ``mac/Icon/appicon-source.png``; the
build prefers that file and auto-crops any uniform border off it (see
``mac/Scripts/crop_icon.py``).

Requires Pillow. Nothing else in the build needs it -- the rendered PNG is
committed, and the .icns is produced from the PNG by sips/iconutil.

    python3 mac/Icon/render_icon.py mac/Icon/appicon-1024.png
"""

from __future__ import annotations

import math
import random
import sys

from PIL import Image, ImageDraw, ImageFilter

S = 4  # supersampling factor
N = 1024
W = N * S


def px(v: float) -> float:
    return v * S


def lerp(a: tuple[int, int, int], b: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    return tuple(round(a[i] + (b[i] - a[i]) * t) for i in range(3))


NAVY_LIGHT = (62, 92, 121)
NAVY_DARK = (25, 42, 60)
STEEL_LIGHT = (233, 239, 243)
STEEL = (176, 190, 201)
STEEL_DARK = (108, 124, 138)
MESH_DARK = (86, 106, 126)
PAPER_LIGHT = (166, 197, 226)
PAPER_DARK = (74, 112, 152)
INK = (196, 220, 242)


def squircle_mask(size: int, radius: float) -> Image.Image:
    """A rounded-square mask with an Apple-ish superellipse falloff."""
    mask = Image.new("L", (size, size), 0)
    d = ImageDraw.Draw(mask)
    d.rounded_rectangle([0, 0, size - 1, size - 1], radius=radius, fill=255)
    return mask.filter(ImageFilter.GaussianBlur(size / 900))


def background(size: int) -> Image.Image:
    """Diagonal navy gradient, lighter at the top-left."""
    bg = Image.new("RGB", (size, size))
    p = bg.load()
    for y in range(size):
        for x in range(0, size, 8):
            t = x / size * 0.45 + y / size * 0.55
            c = lerp(NAVY_LIGHT, NAVY_DARK, min(1.0, t**0.9))
            for k in range(8):
                if x + k < size:
                    p[x + k, y] = c
    return bg


def sheet(
    draw: ImageDraw.ImageDraw,
    cx: float,
    cy: float,
    w: float,
    h: float,
    angle: float,
    fill: tuple[int, int, int],
    lines: bool,
) -> None:
    """One document page, rotated about its centre."""
    a = math.radians(angle)
    corners = []
    for dx, dy in ((-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)):
        corners.append(
            (cx + dx * math.cos(a) - dy * math.sin(a), cy + dx * math.sin(a) + dy * math.cos(a))
        )
    draw.polygon(corners, fill=fill)
    if not lines:
        return
    random.seed(7)
    rows = 9
    for r in range(rows):
        dy = -h / 2 + h * (r + 1.1) / (rows + 1.6)
        segs = 2 if r % 3 == 2 else 1
        x0 = -w / 2 + w * 0.09
        avail = w * 0.82
        for s in range(segs):
            frac = 0.55 if segs == 2 else random.uniform(0.62, 0.94)
            seg_w = avail * frac / segs
            x1 = x0 + seg_w
            pts = []
            for dx, ddy in (
                (x0, dy - h * 0.019),
                (x1, dy - h * 0.019),
                (x1, dy + h * 0.019),
                (x0, dy + h * 0.019),
            ):
                pts.append(
                    (
                        cx + dx * math.cos(a) - ddy * math.sin(a),
                        cy + dx * math.sin(a) + ddy * math.cos(a),
                    )
                )
            draw.polygon(pts, fill=INK)
            x0 = x1 + avail * 0.06


def _linear_gradient(size, a, b, horizontal=True):
    """A two-stop linear gradient the size of the canvas."""
    g = Image.new("RGB", (size[0] // (S * 2), size[1] // (S * 2)))
    px_ = g.load()
    for y in range(g.height):
        for x in range(g.width):
            t = (x / max(1, g.width - 1)) if horizontal else (y / max(1, g.height - 1))
            px_[x, y] = lerp(a, b, t)
    return g.resize(size, Image.BILINEAR).convert("RGBA")


def bowl(img: Image.Image) -> None:
    """The strainer: mesh bowl, steel rim, side loop, and W-capped handle."""
    cx, cy = px(496), px(452)
    rx, ry = px(272), px(114)
    depth = px(322)

    body = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(body)

    # Left loop handle goes down first so the bowl covers where it joins.
    lx = cx - rx - px(6)
    d.ellipse(
        [lx - px(62), cy + px(28), lx + px(58), cy + px(122)],
        outline=STEEL_LIGHT + (255,),
        width=int(px(16)),
    )

    # Right handle: a tapered bar out to a rounded cap carrying a W.
    hx0, hy0 = cx + rx - px(40), cy - px(34)
    hx1, hy1 = cx + rx + px(168), cy - px(214)
    d.line([(hx0, hy0), (hx1, hy1)], fill=STEEL_DARK + (255,), width=int(px(30)))
    d.line([(hx0, hy0 - px(4)), (hx1, hy1 - px(4))], fill=STEEL_LIGHT + (255,), width=int(px(13)))

    # Bowl body: the lower half of an ellipse, so it reads as a hemisphere.
    shape = Image.new("L", img.size, 0)
    sd = ImageDraw.Draw(shape)
    sd.pieslice([cx - rx, cy - depth, cx + rx, cy + depth], 0, 180, fill=255)
    sd.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=255)
    lit = _linear_gradient(img.size, STEEL_LIGHT, STEEL_DARK)
    body.paste(lit, (0, 0), shape)

    # Bottom vignette: the underside of a bowl never catches the light.
    vign = Image.new("L", img.size, 0)
    vd = ImageDraw.Draw(vign)
    for i in range(60):
        t = i / 59
        vd.pieslice(
            [
                cx - rx * (1 - t * 0.05),
                cy - depth + depth * t * 1.55,
                cx + rx * (1 - t * 0.05),
                cy + depth,
            ],
            0,
            180,
            fill=int(120 * t),
        )
    body.paste(
        Image.new("RGBA", img.size, STEEL_DARK + (255,)),
        (0, 0),
        vign.filter(ImageFilter.GaussianBlur(px(10))),
    )

    # Mesh interior: cross-hatch clipped to the inside of the rim.
    inner = Image.new("L", img.size, 0)
    ImageDraw.Draw(inner).ellipse(
        [cx - rx * 0.925, cy - ry * 0.9, cx + rx * 0.925, cy + ry * 0.9], fill=255
    )
    mesh = Image.new("RGBA", img.size, MESH_DARK + (255,))
    md = ImageDraw.Draw(mesh)
    step = int(px(20))
    for i in range(-int(rx * 3), int(rx * 3), step):
        md.line(
            [(cx + i, cy - px(420)), (cx + i + px(420), cy + px(420))],
            fill=STEEL_LIGHT + (255,),
            width=int(px(3.4)),
        )
        md.line(
            [(cx + i, cy + px(420)), (cx + i + px(420), cy - px(420))],
            fill=STEEL_LIGHT + (255,),
            width=int(px(3.4)),
        )
    body.paste(mesh, (0, 0), inner)

    # Rim: a bright steel ring over the seam between mesh and bowl.
    d.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], outline=STEEL_LIGHT + (255,), width=int(px(23)))
    d.ellipse(
        [cx - rx * 0.95, cy - ry * 0.9, cx + rx * 0.95, cy + ry * 0.9],
        outline=STEEL_DARK + (170,),
        width=int(px(5)),
    )

    # The handle cap, last, so it sits above the rim.
    ang = math.atan2(hy1 - hy0, hx1 - hx0)
    cap_w, cap_h = px(158), px(116)
    ccx, ccy = hx1 + math.cos(ang) * px(56), hy1 + math.sin(ang) * px(56)
    cap = Image.new("RGBA", (int(cap_w * 1.7), int(cap_h * 1.9)), (0, 0, 0, 0))
    cd = ImageDraw.Draw(cap)
    ox, oy = (cap.width - cap_w) / 2, (cap.height - cap_h) / 2
    cd.rounded_rectangle(
        [ox, oy, ox + cap_w, oy + cap_h],
        radius=px(36),
        fill=STEEL + (255,),
        outline=STEEL_LIGHT + (255,),
        width=int(px(10)),
    )
    # "W" drawn from strokes so the icon needs no font file at build time.
    wl, wr = ox + cap_w * 0.27, ox + cap_w * 0.73
    wt, wb = oy + cap_h * 0.33, oy + cap_h * 0.69
    mid = (wl + wr) / 2
    cd.line(
        [
            (wl, wt),
            (wl + (mid - wl) * 0.58, wb),
            (mid, wt + (wb - wt) * 0.40),
            (wr - (wr - mid) * 0.58, wb),
            (wr, wt),
        ],
        fill=NAVY_DARK + (255,),
        width=int(px(12)),
        joint="curve",
    )
    cap = cap.rotate(-math.degrees(ang), resample=Image.BICUBIC, expand=False)
    body.alpha_composite(cap, (int(ccx - cap.width / 2), int(ccy - cap.height / 2)))

    img.alpha_composite(body)


def marks(draw: ImageDraw.ImageDraw) -> None:
    """Watermark glyphs streaming up and out of the strainer."""
    random.seed(19)
    for i in range(36):
        t = i / 35
        x = px(196) + t * px(420) + random.uniform(-px(62), px(62))
        y = px(372) - t * px(268) + random.uniform(-px(52), px(52))
        if y > px(392):
            continue
        scale = px(random.uniform(7, 20))
        alpha = int(255 * (0.35 + 0.6 * (1 - abs(t - 0.45) * 1.1)))
        col = STEEL_LIGHT + (max(70, min(240, alpha)),)
        rot = random.uniform(-1.1, 1.1)
        strokes = random.choice(
            (
                [(-1, 1), (-0.45, -1), (0, 0.4), (0.45, -1), (1, 1)],  # W
                [(-0.7, 1), (0, -1), (0.7, 1)],  # V / A
                [(-0.8, -1), (-0.8, 1)],  # I
                [(-0.9, 1), (-0.9, -1), (0.1, 1), (0.9, -1), (0.9, 1)],  # M
                [(-0.8, -0.7), (0.6, -0.9), (-0.6, 0.9), (0.8, 0.7)],  # S
            )
        )
        pts = [
            (
                x + (dx * math.cos(rot) - dy * math.sin(rot)) * scale,
                y + (dx * math.sin(rot) + dy * math.cos(rot)) * scale,
            )
            for dx, dy in strokes
        ]
        draw.line(pts, fill=col, width=int(px(random.uniform(2.4, 4.2))), joint="curve")


def render() -> Image.Image:
    img = background(W).convert("RGBA")
    draw = ImageDraw.Draw(img)

    # Documents first, so the strainer sits in front of them.
    sheet(draw, px(462), px(792), px(524), px(366), -7.0, PAPER_DARK, False)
    sheet(draw, px(506), px(779), px(524), px(366), -3.2, lerp(PAPER_DARK, PAPER_LIGHT, 0.3), False)
    sheet(draw, px(550), px(766), px(524), px(366), 0.6, PAPER_LIGHT, True)

    bowl(img)
    marks(draw)

    # Top-left sheen, then clip everything to the rounded square.
    sheen = Image.new("L", img.size, 0)
    ImageDraw.Draw(sheen).ellipse([-px(340), -px(620), px(880), px(300)], fill=30)
    img.paste(
        Image.new("RGBA", img.size, (255, 255, 255, 255)),
        (0, 0),
        sheen.filter(ImageFilter.GaussianBlur(px(60))),
    )

    img.putalpha(squircle_mask(W, px(226)))
    return img.resize((N, N), Image.LANCZOS)


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "appicon-1024.png"
    render().save(out)
    print(f"wrote {out}")
