#!/usr/bin/env python3
"""Bake a tileable PNG preview using the same 2D Worley FBM as the HLSL shader.

The noise is built on a torus (cell indices wrap modulo the period and every
octave transform uses an integer scale), so the four edges of the PNG line up
and the texture can be tiled with fract() in the preview without seams.
"""

import argparse
import math
import struct
import zlib


def fract(value):
    return value - math.floor(value)


def hash22(x, y):
    p = (fract(x * 0.1031), fract(y * 0.1030), fract(x * 0.0973))
    dot_value = sum(p[i] * (p[(i + 1) % 3] + 33.33) for i in range(3))
    p = tuple(value + dot_value for value in p)
    return (
        fract((p[0] + p[1]) * p[2]),
        fract((p[0] + p[2]) * p[1]),
    )


def hash12(x, y):
    p = (fract(x * 0.1031), fract(y * 0.1031), fract(x * 0.1031))
    dot_value = (p[0] * (p[1] + 33.33) +
                 p[1] * (p[2] + 33.33) +
                 p[2] * (p[0] + 33.33))
    p = tuple(value + dot_value for value in p)
    return fract((p[0] + p[1]) * p[2])


def worley_2d(x, y, period):
    # Cell indices wrap modulo the period so the lattice forms a torus.
    cell_x = math.floor(x) % period
    cell_y = math.floor(y) % period
    local_x = fract(x)
    local_y = fract(y)
    minimum_distance = 1.0e6
    for offset_y in (-1, 0, 1):
        for offset_x in (-1, 0, 1):
            feature_x, feature_y = hash22(
                (cell_x + offset_x) % period, (cell_y + offset_y) % period
            )
            delta_x = offset_x + feature_x - local_x
            delta_y = offset_y + feature_y - local_y
            minimum_distance = min(
                minimum_distance, delta_x * delta_x + delta_y * delta_y
            )
    return math.sqrt(minimum_distance)


def worley_fbm_2d(x, y, period):
    total = 0.0
    amplitude = 0.5
    normalization = 0.0
    for _ in range(3):
        total += (1.0 - min(max(worley_2d(x, y, period), 0.0), 1.0)) * amplitude
        normalization += amplitude
        # Integer scale keeps every octave periodic on the period torus.
        x = (x * 2 + 17) % period
        y = (y * 2 + 9) % period
        amplitude *= 0.5
    return total / normalization


def value_noise_2d(x, y, period):
    cell_x = math.floor(x) % period
    cell_y = math.floor(y) % period
    fx = fract(x)
    fy = fract(y)
    wx = fx * fx * (3.0 - 2.0 * fx)
    wy = fy * fy * (3.0 - 2.0 * fy)

    def sample(offset_x, offset_y):
        return hash12((cell_x + offset_x) % period, (cell_y + offset_y) % period)

    return (smooth_lerp(sample(0, 0), sample(1, 0), wx) * (1.0 - wy) +
            smooth_lerp(sample(0, 1), sample(1, 1), wx) * wy)


def smooth_lerp(a, b, t):
    return a + (b - a) * t


def value_fbm_2d(x, y, period):
    total = 0.0
    amplitude = 0.5
    normalization = 0.0
    for _ in range(3):
        total += value_noise_2d(x, y, period) * amplitude
        normalization += amplitude
        x = (x * 2 + 23) % period
        y = (y * 2 + 41) % period
        amplitude *= 0.5
    return total / normalization


def cloud_coverage_2d(x, y, period):
    # Both bands stay on the period torus; the value band at scale 1 joins
    # nearby Worley cells into larger, connected cloud systems.
    worley = worley_fbm_2d((x + 7.0) % period, (y + 19.0) % period, period)
    broad = value_fbm_2d((x + 13.0) % period, (y + 29.0) % period, period)
    return min(max(worley * 0.74 + broad * 0.26, 0.0), 1.0)


def smoothstep(edge0, edge1, value):
    t = min(max((value - edge0) / max(edge1 - edge0, 1.0e-6), 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


def png_chunk(kind, data):
    payload = kind + data
    return struct.pack(">I", len(data)) + payload + struct.pack(">I", zlib.crc32(payload) & 0xFFFFFFFF)


def write_png(path, width, height, rows):
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    raw = b"".join(b"\x00" + row for row in rows)
    png = b"\x89PNG\r\n\x1a\n" + png_chunk(b"IHDR", header)
    png += png_chunk(b"IDAT", zlib.compress(raw, 9))
    png += png_chunk(b"IEND", b"")
    with open(path, "wb") as output:
        output.write(png)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=512)
    # Integer frequency: the texture then spans exactly one noise period, so
    # the left/right and top/bottom edges line up (tileable texture).
    parser.add_argument("--frequency", type=int, default=8)
    parser.add_argument("--coverage", type=float, default=0.58)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output", default="cloud_distribution_preview.png")
    parser.add_argument("--raw-output", default="")
    args = parser.parse_args()

    rows = []
    raw_rows = []
    seed = args.seed * 17.17
    frequency = max(args.frequency, 1)
    # Keep the weather map broad enough to form connected cloud systems. The
    # runtime still applies a second 3D density threshold for billowy volume.
    threshold = 0.68 - 0.14 * max(min(args.coverage, 1.0), 0.0)
    for pixel_y in range(args.height):
        row = bytearray()
        raw_row = bytearray()
        for pixel_x in range(args.width):
            uv_x = (pixel_x + 0.5) / args.width
            uv_y = (pixel_y + 0.5) / args.height
            value = cloud_coverage_2d(
                uv_x * frequency + seed,
                uv_y * frequency + seed * 1.37,
                frequency,
            )
            mask = smoothstep(threshold - 0.035, threshold + 0.055, value)
            raw_level = int(round(value * 255.0))
            level = int(round(mask * 255.0))
            row.extend((level, level, level))
            raw_row.extend((raw_level, raw_level, raw_level))
        rows.append(bytes(row))
        raw_rows.append(bytes(raw_row))

    write_png(args.output, args.width, args.height, rows)
    print(f"Wrote {args.output} ({args.width}x{args.height})")
    if args.raw_output:
        write_png(args.raw_output, args.width, args.height, raw_rows)
        print(f"Wrote {args.raw_output} ({args.width}x{args.height})")


if __name__ == "__main__":
    main()
