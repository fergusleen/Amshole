#!/usr/bin/env python3
"""
Convert an image into an edit.tf frame URL (and optional Telstar frame JSON).

Corrected for Teletext G1 Mosaic bit mapping:
Pixel 1 (0,0): bit 0
Pixel 2 (1,0): bit 1
Pixel 3 (0,1): bit 2
Pixel 4 (1,1): bit 3
Pixel 5 (0,2): bit 4
Pixel 6 (1,2): bit 6  <-- Corrected from bit 5
Bit 5 is always 1 for the mosaic range (0x20-0x3F, 0x60-0x7F).
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
from typing import Iterable, Tuple

from PIL import Image, ImageOps, ImageFilter


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
        "--cell-mode",
        choices=("threshold", "match"),
        default="threshold",
        help="threshold: binary subpixels; match: optimal mosaic patterns.",
    )
    parser.add_argument(
        "--match-smoothness",
        type=float,
        default=0.15,
        help="Higher prefers less fragmented patterns in match mode.",
    )
    parser.add_argument(
        "--fit",
        choices=("cover", "contain", "stretch"),
        default="cover",
        help="How to fit the source image.",
    )
    parser.add_argument(
        "--resample",
        choices=("nearest", "bilinear", "bicubic", "lanczos"),
        default="lanczos",
        help="Resampling kernel.",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=128,
        help="Mask threshold (0..255).",
    )
    parser.add_argument(
        "--dither",
        action="store_true",
        help="Use Floyd-Steinberg dithering for the binary mosaic mask.",
    )
    parser.add_argument(
        "--dilation",
        type=int,
        default=0,
        help="Dilation radius to thicken the logo (e.g., 3).",
    )
    parser.add_argument(
        "--invert",
        action="store_true",
        help="Invert the binary mosaic mask.",
    )
    parser.add_argument(
        "--no-crop",
        action="store_true",
        help="Disable automatic margin cropping.",
    )
    parser.add_argument(
        "--reserved-cols",
        type=int,
        default=DEFAULT_RESERVED_COLS,
        help="Columns reserved for controls per row.",
    )
    parser.add_argument(
        "--subpixel-scale",
        type=int,
        default=1,
        help="Supersample factor (1..8).",
    )
    parser.add_argument(
        "--coverage-threshold",
        type=float,
        default=0.5,
        help="Fraction of lit supersamples needed to switch a subpixel on.",
    )
    parser.add_argument(
        "--solid-centre",
        action="store_true",
        help="Promote interior cells to full blocks (0x7F).",
    )
    parser.add_argument(
        "--default-colour",
        choices=tuple(TELETEXT_COLOURS.keys()),
        default="cyan",
        help="Row colour when a row has no lit pixels.",
    )
    parser.add_argument(
        "--force-colour",
        choices=tuple(TELETEXT_COLOURS.keys()),
        help="Force all rows to use this mosaic colour.",
    )
    parser.add_argument(
        "--background",
        choices=("black", "white"),
        default="black",
        help="Background for 'contain' fit.",
    )
    parser.add_argument("--page-no", type=int, default=200)
    parser.add_argument("--frame-id", type=str, default="a")
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-url", type=Path)
    return parser.parse_args()


def get_resample_filter(name: str) -> Image.Resampling:
    mapping = {
        "nearest": Image.Resampling.NEAREST,
        "bilinear": Image.Resampling.BILINEAR,
        "bicubic": Image.Resampling.BICUBIC,
        "lanczos": Image.Resampling.LANCZOS,
    }
    return mapping[name]


def nearest_teletext_colour(rgb: Tuple[int, int, int]) -> int:
    best_ctrl = PALETTE[0][0]
    best_dist = float("inf")
    r, g, b = rgb
    for ctrl, (cr, cg, cb) in PALETTE:
        dr, dg, db = r - cr, g - cg, b - cb
        dist = dr*dr + dg*dg + db*db
        if dist < best_dist:
            best_dist, best_ctrl = dist, ctrl
    return best_ctrl


def fit_image(
    src: Image.Image,
    width: int,
    height: int,
    fit_mode: str,
    bg: str,
    resample_name: str,
    crop: bool,
) -> Image.Image:
    src = src.convert("RGB")
    
    if crop:
        gray = src.convert("L")
        if sum(gray.getdata()) / len(gray.getdata()) > 128:
            gray = ImageOps.invert(gray)
        bbox = gray.getbbox()
        if bbox:
            # 2px padding to avoid edge clipping
            bbox = (bbox[0]-2, bbox[1]-2, bbox[2]+2, bbox[3]+2)
            src = src.crop(bbox)

    resample = get_resample_filter(resample_name)
    if fit_mode == "stretch":
        return src.resize((width, height), resample)
    if fit_mode == "cover":
        return ImageOps.fit(src, (width, height), method=resample)

    contained = ImageOps.contain(src, (width, height), method=resample)
    canvas = Image.new("RGB", (width, height), (0,0,0) if bg == "black" else (255,255,255))
    canvas.paste(contained, ((width - contained.width)//2, (height - contained.height)//2))
    return canvas


def binary_mask(img: Image.Image, threshold: int, dither: bool, invert: bool, dilation: int) -> Image.Image:
    gray = img.convert("L")
    if dither:
        mask = gray.convert("1")
    else:
        mask = gray.point(lambda p: 255 if p >= threshold else 0, mode="1")
    
    if dilation > 0:
        mask = mask.convert("L").filter(ImageFilter.MaxFilter(dilation)).convert("1")

    if invert:
        mask = ImageOps.invert(mask.convert("L")).convert("1")
    return mask


def grayscale_levels(img: Image.Image, invert: bool) -> Image.Image:
    gray = img.convert("L")
    return ImageOps.invert(gray) if invert else gray


def downsample_binary_mask(mask_img: Image.Image, scale: int, coverage_threshold: float) -> Image.Image:
    if scale == 1: return mask_img.convert("1")
    width, height = mask_img.size
    out_w, out_h = width // scale, height // scale
    src = mask_img.convert("L").load()
    out = Image.new("1", (out_w, out_h))
    out_px = out.load()
    cutoff = coverage_threshold * (scale * scale)

    for oy in range(out_h):
        for ox in range(out_w):
            lit = sum(src[ox*scale + dx, oy*scale + dy] != 0 for dy in range(scale) for dx in range(scale))
            out_px[ox, oy] = 255 if lit >= cutoff else 0
    return out


def choose_best_bits_from_levels(levels: list[float], smoothness: float) -> int:
    best_bits, best_score = 0, float("inf")
    # i represents 6 bits of data mapped to Teletext bits 0,1,2,3,4,6
    for i in range(64):
        b = [(i >> j) & 1 for j in range(6)]
        mse = sum((levels[idx] - float(b[idx]))**2 for idx in range(6))
        # Fragmentation pairs: TL-TR, TL-ML, TR-MR, ML-MR, ML-BL, MR-BR, BL-BR
        pairs = [(0,1), (0,2), (1,3), (2,3), (2,4), (3,5), (4,5)]
        frag = sum(abs(b[p1]-b[p2]) for p1, p2 in pairs)
        score = mse + smoothness * frag
        if score < best_score:
            best_score = score
            best_bits = b[0] | (b[1]<<1) | (b[2]<<2) | (b[3]<<3) | (b[4]<<4) | (b[5]<<6)
    return best_bits


def promote_interior_to_full_blocks(cell_bits_rows: list[list[int]]) -> list[list[int]]:
    if not cell_bits_rows: return cell_bits_rows
    FULL_BLOCK = 0x5F # Bits 0,1,2,3,4,6
    rows, cols = len(cell_bits_rows), len(cell_bits_rows[0])
    promoted = [row[:] for row in cell_bits_rows]
    for y in range(1, rows - 1):
        for x in range(1, cols - 1):
            if cell_bits_rows[y][x] == 0 or cell_bits_rows[y][x] == FULL_BLOCK: continue
            if sum(1 for dy in (-1,0,1) for dx in (-1,0,1) if (dx!=0 or dy!=0) and cell_bits_rows[y+dy][x+dx] != 0) >= 6:
                promoted[y][x] = FULL_BLOCK
    return promoted


def encode_rows_to_payload(rows: list[list[int]]) -> str:
    bits = []
    for row in rows:
        for val in row:
            v = val & 0x7F
            for i in range(6, -1, -1):
                bits.append((v >> i) & 1)
    packed = bytearray()
    for i in range(0, len(bits), 8):
        chunk = bits[i:i+8]
        byte = 0
        for bit in chunk: byte = (byte << 1) | bit
        if len(chunk) < 8: byte <<= (8 - len(chunk))
        packed.append(byte)
    return base64.urlsafe_b64encode(packed).decode("ascii").rstrip("=")


def build_frame_codes(
    rgb_img: Image.Image, mask_img: Image.Image, reserved_cols: int,
    default_colour_name: str, force_colour_name: str | None, solid_centre: bool,
) -> list[list[int]]:
    active_cols = FRAME_COLS - reserved_cols
    rgb_px, mask_px = rgb_img.load(), mask_img.load()
    row_colours, raw_cell_bits = [], []
    default_ctrl = TELETEXT_COLOURS[default_colour_name][0]
    forced_ctrl = TELETEXT_COLOURS[force_colour_name][0] if force_colour_name else None

    for ry in range(FRAME_ROWS):
        y0 = ry * 3
        lit_colours = [rgb_px[px, py] for py in range(y0, y0+3) for px in range(active_cols*2) if mask_px[px, py]]
        row_colours.append(forced_ctrl if forced_ctrl else (nearest_teletext_colour(sum(c[0] for c in lit_colours)//len(lit_colours), sum(c[1] for c in lit_colours)//len(lit_colours), sum(c[2] for c in lit_colours)//len(lit_colours)) if lit_colours else default_ctrl))
        
        bits_row = []
        for cx in range(active_cols):
            x0, b = cx * 2, 0
            if mask_px[x0, y0]: b |= 0x01
            if mask_px[x0+1, y0]: b |= 0x02
            if mask_px[x0, y0+1]: b |= 0x04
            if mask_px[x0+1, y0+1]: b |= 0x08
            if mask_px[x0, y0+2]: b |= 0x10
            if mask_px[x0+1, y0+2]: b |= 0x40 # Correct bit 6
            bits_row.append(b)
        raw_cell_bits.append(bits_row)

    if solid_centre: raw_cell_bits = promote_interior_to_full_blocks(raw_cell_bits)
    
    rows = []
    for ry in range(FRAME_ROWS):
        row = [row_colours[ry]]
        if reserved_cols >= 2: row.append(CTRL_CONTIGUOUS)
        if reserved_cols >= 3: row.append(CTRL_HOLD)
        if reserved_cols > 3: row.extend([0x20]*(reserved_cols-3))
        row.extend([0x20 | b for b in raw_cell_bits[ry]])
        rows.append(row)
    return rows


def build_frame_codes_matched(
    rgb_img: Image.Image, level_img: Image.Image, reserved_cols: int,
    default_colour_name: str, force_colour_name: str | None, solid_centre: bool, match_smoothness: float
) -> list[list[int]]:
    active_cols = FRAME_COLS - reserved_cols
    rgb_px, lv_px = rgb_img.load(), level_img.load()
    row_colours, raw_cell_bits = [], []
    default_ctrl = TELETEXT_COLOURS[default_colour_name][0]
    forced_ctrl = TELETEXT_COLOURS[force_colour_name][0] if force_colour_name else None

    for ry in range(FRAME_ROWS):
        y0 = ry * 3
        lit_colours = [rgb_px[px, py] for py in range(y0, y0+3) for px in range(active_cols*2) if lv_px[px, py] >= 128]
        if forced_ctrl: row_ctrl = forced_ctrl
        elif lit_colours:
            avg = (sum(c[0] for c in lit_colours)//len(lit_colours), sum(c[1] for c in lit_colours)//len(lit_colours), sum(c[2] for c in lit_colours)//len(lit_colours))
            row_ctrl = nearest_teletext_colour(avg)
        else: row_ctrl = default_ctrl
        row_colours.append(row_ctrl)

        bits_row = []
        for cx in range(active_cols):
            x0 = cx * 2
            levels = [lv_px[x0, y0]/255., lv_px[x0+1, y0]/255., lv_px[x0, y0+1]/255., lv_px[x0+1, y0+1]/255., lv_px[x0, y0+2]/255., lv_px[x0+1, y0+2]/255.]
            bits_row.append(choose_best_bits_from_levels(levels, match_smoothness))
        raw_cell_bits.append(bits_row)

    if solid_centre: raw_cell_bits = promote_interior_to_full_blocks(raw_cell_bits)
    
    rows = []
    for ry in range(FRAME_ROWS):
        row = [row_colours[ry]]
        if reserved_cols >= 2: row.append(CTRL_CONTIGUOUS)
        if reserved_cols >= 3: row.append(CTRL_HOLD)
        if reserved_cols > 3: row.extend([0x20]*(reserved_cols-3))
        row.extend([0x20 | b for b in raw_cell_bits[ry]])
        rows.append(row)
    return rows


def main() -> None:
    args = parse_args()
    if not args.input_image.exists(): raise SystemExit(f"Missing image: {args.input_image}")
    active_cols = FRAME_COLS - args.reserved_cols
    base_w, base_h = active_cols * 2, FRAME_ROWS * 3
    src = Image.open(args.input_image)
    rgb = fit_image(src, base_w, base_h, args.fit, args.background, args.resample, not args.no_crop)

    if args.cell_mode == "match":
        levels = grayscale_levels(fit_image(src, base_w*args.subpixel_scale, base_h*args.subpixel_scale, args.fit, args.background, args.resample, not args.no_crop), args.invert).resize((base_w, base_h), Image.Resampling.BOX)
        rows = build_frame_codes_matched(rgb, levels, args.reserved_cols, args.default_colour, args.force_colour, args.solid_centre, args.match_smoothness)
    else:
        mask = downsample_binary_mask(binary_mask(fit_image(src, base_w*args.subpixel_scale, base_h*args.subpixel_scale, args.fit, args.background, args.resample, not args.no_crop), args.threshold, args.dither, args.invert, args.dilation), args.subpixel_scale, args.coverage_threshold)
        rows = build_frame_codes(rgb, mask, args.reserved_cols, args.default_colour, args.force_colour, args.solid_centre)

    payload = encode_rows_to_payload(rows)
    url = f"http://edit.tf/#0:{payload}"
    if args.output_url:
        args.output_url.parent.mkdir(parents=True, exist_ok=True)
        args.output_url.write_text(url + "\n")
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps({"pid": {"page-no": args.page_no, "frame-id": args.frame_id}, "visible": True, "frame-type": "information", "content": {"data": url, "type": "edit.tf"}}, indent=2) + "\n")
    print(url)

if __name__ == "__main__":
    main()
