from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np


def save_rgb_png(path: str | Path, image: np.ndarray) -> Path:
    path = Path(path)
    image = np.asarray(image, dtype=np.uint8)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("save_rgb_png expects an HxWx3 uint8 image")
    height, width, _ = image.shape
    raw = b"".join(b"\x00" + image[row].tobytes() for row in range(height))

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(raw, level=6))
    png += chunk(b"IEND", b"")
    path.write_bytes(png)
    return path


def draw_line(image: np.ndarray, p0: tuple[int, int], p1: tuple[int, int], color, width: int = 1) -> None:
    x0, y0 = p0
    x1, y1 = p1
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    while True:
        draw_circle(image, (x0, y0), max(0, width // 2), color)
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy


def draw_circle(image: np.ndarray, center: tuple[int, int], radius: int, color) -> None:
    cx, cy = center
    h, w, _ = image.shape
    color = np.asarray(color, dtype=np.uint8)
    r2 = radius * radius
    for yy in range(max(0, cy - radius), min(h, cy + radius + 1)):
        for xx in range(max(0, cx - radius), min(w, cx + radius + 1)):
            if (xx - cx) ** 2 + (yy - cy) ** 2 <= r2:
                image[yy, xx] = color

