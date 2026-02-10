#!/usr/bin/env python3
"""
Convert an image into an edit.tf frame URL (and optional Telstar frame JSON).

The output frame is 40x25 teletext cells, encoded as:
http://edit.tf/#0:<base64url_payload>

This script builds mosaic graphics only:
- Column 0: mosaic colour control
- Column 1: contiguous graphics control (0x19)
- Column 2: hold graphics control (0x1E)
- Columns 3..39: mosaic characters (0x20..0x3F)
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
from typing import Iterable, Tuple

from PIL import Image, ImageOps


FRAME_COLS = 40
FRAME_ROWS = 25
DEFAULT_RESERVED_COLS = 3

CTRL_CONTIGUOUS = 0x19
CTRL_HOLD = 0x1E


TELETEXT_COLOURS = {
    "red": (0x11, (255, 0, 0)),
    "green": (0x12, (0, 255, 0)),
    "yellow": (0x13, (255, 255, 0)),
    "blue": (0x14, (0, 0, 255)),
    "magenta": (0x15, (255, 0, 255)),
    "cyan": (0x16, (0, 255, 255)),
    "white": (0x17, (255, 255, 255)),
}

PALETTE = list(TELETEXT_COLOURS.values())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create edit.tf frames from images (mosaic graphics)."
    )
    parser.add_argument("input_image", type=Path, help="Input image path.")
    parser.add_argument(
        "--fit",
        choices=("cover", "contain", "stretch"),
        default="cover",
        help="How to fit the source image into the mosaic pixel grid.",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=128,
        help="Mask threshold (0..255). Higher = fewer pixels lit.",
    )
    parser.add_argument(
        "--dither",
        action="store_true",
        help="Use Floyd-Steinberg dithering for the binary mosaic mask.",
    )
    parser.add_argument(
        "--invert",
        action="store_true",
        help="Invert the binary mosaic mask.",
    )
    parser.add_argument(
        "--reserved-cols",
        type=int,
        default=DEFAULT_RESERVED_COLS,
        help="Columns reserved for controls per row (default: 3).",
    )
    parser.add_argument(
        "--default-colour",
        choices=tuple(TELETEXT_COLOURS.keys()),
        default="cyan",
        help="Row colour when a row has no lit pixels.",
    )
    parser.add_argument(
        "--background",
        choices=("black", "white"),
        default="black",
        help="Background used with --fit contain.",
    )
    parser.add_argument(
        "--page-no",
        type=int,
        default=200,
        help="Page number for JSON output.",
    )
    parser.add_argument(
        "--frame-id",
        type=str,
        default="a",
        help="Frame id for JSON output.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Optional output frame JSON path.",
    )
    parser.add_argument(
        "--output-url",
        type=Path,
        help="Optional output text file containing only the edit.tf URL.",
    )
    return parser.parse_args()


def nearest_teletext_colour(rgb: Tuple[int, int, int]) -> int:
    best_ctrl = PALETTE[0][0]
    best_dist = float("inf")
    r, g, b = rgb
    for ctrl, (cr, cg, cb) in PALETTE:
        dr = r - cr
        dg = g - cg
        db = b - cb
        dist = dr * dr + dg * dg + db * db
        if dist < best_dist:
            best_dist = dist
            best_ctrl = ctrl
    return best_ctrl


def fit_image(src: Image.Image, width: int, height: int, fit_mode: str, bg: str) -> Image.Image:
    src = src.convert("RGB")
    if fit_mode == "stretch":
        return src.resize((width, height), Image.Resampling.LANCZOS)

    if fit_mode == "cover":
        return ImageOps.fit(src, (width, height), method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))

    # contain
    contained = ImageOps.contain(src, (width, height), method=Image.Resampling.LANCZOS)
    bg_rgb = (0, 0, 0) if bg == "black" else (255, 255, 255)
    canvas = Image.new("RGB", (width, height), bg_rgb)
    ox = (width - contained.width) // 2
    oy = (height - contained.height) // 2
    canvas.paste(contained, (ox, oy))
    return canvas


def binary_mask(img: Image.Image, threshold: int, dither: bool, invert: bool) -> Image.Image:
    gray = img.convert("L")
    if dither:
        mask = gray.convert("1")
    else:
        threshold = max(0, min(255, threshold))
        mask = gray.point(lambda p: 255 if p >= threshold else 0, mode="1")
    if invert:
        mask = ImageOps.invert(mask.convert("L")).convert("1")
    return mask


def average_rgb(values: Iterable[Tuple[int, int, int]]) -> Tuple[int, int, int]:
    total_r = 0
    total_g = 0
    total_b = 0
    count = 0
    for r, g, b in values:
        total_r += r
        total_g += g
        total_b += b
        count += 1
    if count == 0:
        return (0, 0, 0)
    return (total_r // count, total_g // count, total_b // count)


def encode_rows_to_payload(rows: list[list[int]]) -> str:
    flat = [v & 0x7F for row in rows for v in row]
    if len(flat) != FRAME_COLS * FRAME_ROWS:
        raise ValueError("Internal error: frame is not 40x25 cells")

    bits: list[int] = []
    for value in flat:
        for bit in range(6, -1, -1):
            bits.append((value >> bit) & 1)

    packed = bytearray()
    for idx in range(0, len(bits), 8):
        byte = 0
        for bit in bits[idx:idx + 8]:
            byte = (byte << 1) | bit
        packed.append(byte)

    return base64.urlsafe_b64encode(bytes(packed)).decode("ascii").rstrip("=")


def build_frame_codes(
    rgb_img: Image.Image,
    mask_img: Image.Image,
    reserved_cols: int,
    default_colour_name: str,
) -> list[list[int]]:
    if not (0 <= reserved_cols <= 10):
        raise ValueError("--reserved-cols must be between 0 and 10")

    active_cols = FRAME_COLS - reserved_cols
    if active_cols <= 0:
        raise ValueError("--reserved-cols leaves no drawable columns")

    rgb_px = rgb_img.load()
    mask_px = mask_img.load()

    rows: list[list[int]] = []
    default_ctrl = TELETEXT_COLOURS[default_colour_name][0]

    for row_idx in range(FRAME_ROWS):
        y0 = row_idx * 3
        lit_colours: list[Tuple[int, int, int]] = []

        # Pick a colour per teletext row from lit source pixels.
        for py in range(y0, y0 + 3):
            for px in range(active_cols * 2):
                if mask_px[px, py] != 0:
                    lit_colours.append(rgb_px[px, py])

        if lit_colours:
            row_ctrl = nearest_teletext_colour(average_rgb(lit_colours))
        else:
            row_ctrl = default_ctrl

        row_codes: list[int] = []
        if reserved_cols >= 1:
            row_codes.append(row_ctrl)
        if reserved_cols >= 2:
            row_codes.append(CTRL_CONTIGUOUS)
        if reserved_cols >= 3:
            row_codes.append(CTRL_HOLD)
        if reserved_cols > 3:
            row_codes.extend([0x20] * (reserved_cols - 3))

        # Build mosaic chars from 2x3 subpixel blocks.
        for cell_x in range(active_cols):
            x0 = cell_x * 2
            bits = 0

            if mask_px[x0, y0] != 0:
                bits |= 0x01
            if mask_px[x0 + 1, y0] != 0:
                bits |= 0x02
            if mask_px[x0, y0 + 1] != 0:
                bits |= 0x04
            if mask_px[x0 + 1, y0 + 1] != 0:
                bits |= 0x08
            if mask_px[x0, y0 + 2] != 0:
                bits |= 0x10
            if mask_px[x0 + 1, y0 + 2] != 0:
                bits |= 0x20

            row_codes.append(0x20 | bits)

        if len(row_codes) != FRAME_COLS:
            raise ValueError(f"Internal error: row {row_idx} has {len(row_codes)} columns")

        rows.append(row_codes)

    return rows


def main() -> None:
    args = parse_args()

    if not args.input_image.exists():
        raise SystemExit(f"Input image not found: {args.input_image}")

    active_cols = FRAME_COLS - args.reserved_cols
    target_w = active_cols * 2
    target_h = FRAME_ROWS * 3

    src = Image.open(args.input_image)
    rgb = fit_image(src, target_w, target_h, args.fit, args.background)
    mask = binary_mask(rgb, args.threshold, args.dither, args.invert)

    rows = build_frame_codes(
        rgb_img=rgb,
        mask_img=mask,
        reserved_cols=args.reserved_cols,
        default_colour_name=args.default_colour,
    )
    payload = encode_rows_to_payload(rows)
    url = f"http://edit.tf/#0:{payload}"

    if args.output_url:
        args.output_url.parent.mkdir(parents=True, exist_ok=True)
        args.output_url.write_text(url + "\n", encoding="utf-8")

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        frame_obj = {
            "pid": {"page-no": args.page_no, "frame-id": args.frame_id},
            "visible": True,
            "frame-type": "information",
            "content": {"data": url, "type": "edit.tf"},
        }
        args.output_json.write_text(json.dumps(frame_obj, indent=2) + "\n", encoding="utf-8")

    print(url)


if __name__ == "__main__":
    main()
