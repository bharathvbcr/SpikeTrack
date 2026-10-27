"""Offline generator for st-video's JPEG decode fixtures (not run by any test).

Writes, next to this file:

  cases/<name>_<W>x<H>.jpg   a small JPEG (made by Pillow's libjpeg-turbo encoder or by `cjpeg`)
  cases/<name>_<W>x<H>.rgb   Pillow 13.0.0's decode of it: `Image.open(p).convert("RGB").tobytes()`
  fallback/...               JPEG variants the native decoder refuses, and a PNG, made by Pillow

`tests/jpeg_pillow.rs` decodes every cases/*.jpg with st_video::decode_jpeg and requires the bytes
of the matching .rgb at zero tolerance. The cases are small and chosen for what the 640x480 VISEM
frames never exercise: partial MCUs at the right and bottom edges, odd sizes, chroma planes of 1 to
3 samples (where libjpeg-turbo's fancy upsampler gives way to the box filter), every upsampler
(h2v1, h1v2, h2v2 and whole-factor replication), grayscale, restart intervals, optimised Huffman
tables, and a multi-scan (one component per scan) sequential file.

Every case is also decoded with `djpeg` (libjpeg-turbo 3.2.0) and must equal Pillow's decode, so a
fixture is only written when the two oracles agree.

    tools/reference/.venv/bin/python -s tools/reference/jpeg_cases.py

Needs Pillow 13.0.0 (libjpeg-turbo 3.2.0), and `cjpeg` and `djpeg` from libjpeg-turbo 3.2.0 on PATH.
Offline fixture generation only: nothing here runs in the shipped binary.
"""

import io
import random
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, features

# The fixtures live with the Rust tests that read them.
HERE = Path(__file__).resolve().parents[2] / "crates" / "st-video" / "tests" / "jpeg_cases"
CASES = HERE / "cases"
FALLBACK = HERE / "fallback"


def texture(w: int, h: int, seed: int, gray: bool = False) -> Image.Image:
    """A deterministic picture with smooth ramps, hard colour edges and noise, so chroma
    upsampling, edge handling and clamping all show up in the output."""
    rng = random.Random(seed)
    px = []
    for y in range(h):
        for x in range(w):
            r = (x * 255) // max(w - 1, 1)
            g = (y * 255) // max(h - 1, 1)
            b = 255 - (r + g) // 2
            if ((x // 5) + (y // 4)) % 3 == 0:  # blocky colour edges
                r, g, b = 255 - r, b, g
            if rng.random() < 0.08:  # salt: saturated pixels exercise range limiting
                r, g, b = rng.choice([(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 255), (0, 0, 0)])
            n = rng.randint(-14, 14)
            px.append((max(0, min(255, r + n)), max(0, min(255, g - n)), max(0, min(255, b + n // 2))))
    im = Image.new("RGB", (w, h))
    im.putdata(px)
    return im.convert("L") if gray else im


def pillow_rgb(data: bytes) -> bytes:
    with Image.open(io.BytesIO(data)) as im:
        assert im.format == "JPEG"
        return im.convert("RGB").tobytes()


def djpeg_rgb(data: bytes, w: int, h: int) -> bytes:
    with tempfile.TemporaryDirectory() as d:
        src = Path(d) / "in.jpg"
        src.write_bytes(data)
        run = subprocess.run(["djpeg", "-ppm", str(src)], capture_output=True)
        assert run.returncode in (0, 2), run.stderr  # 2 means warnings only (e.g. an odd Adobe transform)
        out = run.stdout
    # "P6\nW H\n255\n" then RGB pixels; a grayscale JPEG comes out as "P5" (one byte per pixel),
    # which is replicated to three channels here as Pillow's convert("RGB") does.
    parts = out.split(b"\n", 3)
    assert parts[1] == f"{w} {h}".encode() and parts[2] == b"255", parts[:3]
    if parts[0] == b"P5":
        return b"".join(bytes([g, g, g]) for g in parts[3])
    assert parts[0] == b"P6", parts[:3]
    return parts[3]


def cjpeg(im: Image.Image, *args: str) -> bytes:
    with tempfile.TemporaryDirectory() as d:
        src = Path(d) / ("in.pgm" if im.mode == "L" else "in.ppm")
        im.save(src)
        return subprocess.run(["cjpeg", *args, str(src)], check=True, capture_output=True).stdout


def pillow_save(im: Image.Image, **kw) -> bytes:
    buf = io.BytesIO()
    im.save(buf, "JPEG", **kw)
    return buf.getvalue()


def split_segments(jpeg: bytes):
    """(marker, bytes of the whole segment including ff xx and length) for every segment before
    the scan data, then the scan data from the SOS marker on, as the last element's remainder."""
    assert jpeg[:2] == b"\xff\xd8"
    out, i = [], 2
    while True:
        assert jpeg[i] == 0xFF
        marker = jpeg[i + 1]
        length = int.from_bytes(jpeg[i + 2 : i + 4], "big")
        out.append((marker, jpeg[i : i + 2 + length]))
        i += 2 + length
        if marker == 0xDA:
            return out, jpeg[i:]


def restamp(jpeg: bytes, drop=(), insert=(), ids=None) -> bytes:
    """Rewrite a 3-component JPEG's marker segments: drop segments by marker, insert APP segments
    right after SOI, and optionally replace the component ids in the SOF and SOS headers."""
    segs, tail = split_segments(jpeg)
    body = b"".join(insert)
    for marker, seg in segs:
        if marker in drop:
            continue
        if ids is not None and marker == 0xC0:
            seg = bytearray(seg)
            for k, cid in enumerate(ids):
                seg[2 + 2 + 6 + 3 * k] = cid
            seg = bytes(seg)
        if ids is not None and marker == 0xDA:
            seg = bytearray(seg)
            for k, cid in enumerate(ids):
                seg[2 + 2 + 1 + 2 * k] = cid
            seg = bytes(seg)
        body += seg
    return b"\xff\xd8" + body + tail


def adobe(transform: int) -> bytes:
    payload = b"Adobe" + (100).to_bytes(2, "big") + bytes(4) + bytes([transform])
    return b"\xff\xee" + (len(payload) + 2).to_bytes(2, "big") + payload


JFIF, ADOBE = 0xE0, 0xEE


def write_case(name: str, jpeg: bytes, w: int, h: int) -> None:
    ours = pillow_rgb(jpeg)
    assert len(ours) == w * h * 3, (name, len(ours))
    theirs = djpeg_rgb(jpeg, w, h)
    assert ours == theirs, f"{name}: Pillow and djpeg disagree"
    stem = f"{name}_{w}x{h}"
    (CASES / f"{stem}.jpg").write_bytes(jpeg)
    (CASES / f"{stem}.rgb").write_bytes(ours)
    print(f"{stem}: {len(jpeg)} byte JPEG, {len(ours)} byte RGB")


def main() -> int:
    if features.version("libjpeg_turbo") != "3.2.0":
        print(f"Pillow's libjpeg-turbo is {features.version('libjpeg_turbo')}, expected 3.2.0", file=sys.stderr)
        return 1
    CASES.mkdir(exist_ok=True)
    FALLBACK.mkdir(exist_ok=True)

    seed = 1

    def pic(w, h, gray=False):
        nonlocal seed
        seed += 1
        return texture(w, h, seed, gray)

    # Pillow's encoder: grayscale, and each chroma subsampling it offers.
    write_case("gray", pillow_save(pic(17, 13, True), quality=85), 17, 13)
    write_case("gray", pillow_save(pic(64, 48, True), quality=60), 64, 48)
    write_case("s444", pillow_save(pic(19, 11), quality=90, subsampling="4:4:4"), 19, 11)
    write_case("s422", pillow_save(pic(33, 17), quality=80, subsampling="4:2:2"), 33, 17)
    write_case("s420", pillow_save(pic(35, 21), quality=75, subsampling="4:2:0"), 35, 21)
    write_case("s420", pillow_save(pic(130, 70), quality=50, subsampling="4:2:0"), 130, 70)
    write_case("s420", pillow_save(pic(32, 32), quality=95, subsampling="4:2:0"), 32, 32)
    write_case("s411", pillow_save(pic(41, 9), quality=80, subsampling="4:1:1"), 41, 9)
    # Tiny pictures: chroma planes 2, 3, 1 and 2 samples wide.
    write_case("s420", pillow_save(pic(3, 3), quality=90, subsampling="4:2:0"), 3, 3)
    write_case("s420", pillow_save(pic(5, 5), quality=90, subsampling="4:2:0"), 5, 5)
    write_case("s420", pillow_save(pic(1, 1), quality=90, subsampling="4:2:0"), 1, 1)
    write_case("s422", pillow_save(pic(4, 2), quality=90, subsampling="4:2:2"), 4, 2)
    write_case("s420", pillow_save(pic(2, 15), quality=90, subsampling="4:2:0"), 2, 15)
    # Restart intervals (every MCU row; every 3 MCUs) and per-image optimised Huffman tables.
    write_case("s420rst", pillow_save(pic(64, 40), quality=80, subsampling="4:2:0", restart_marker_rows=1), 64, 40)
    write_case("s444rst", pillow_save(pic(45, 27), quality=80, subsampling="4:4:4", restart_marker_blocks=3), 45, 27)
    write_case("s420opt", pillow_save(pic(51, 37), quality=70, subsampling="4:2:0", optimize=True), 51, 37)
    write_case("s420q100", pillow_save(pic(24, 24), quality=100, subsampling="4:2:0"), 24, 24)
    write_case("s420q5", pillow_save(pic(40, 40), quality=5, subsampling="4:2:0"), 40, 40)

    # cjpeg: sampling Pillow cannot write (h1v2), a 4:2:2 file with a vertical-only Cb
    # (h2v2 luma, 1x2 chroma ratios), and sequential files with one component per scan.
    write_case("h1v2", cjpeg(pic(29, 23), "-quality", "85", "-baseline", "-sample", "1x2,1x1,1x1"), 29, 23)
    write_case("h2v2mix", cjpeg(pic(37, 27), "-quality", "85", "-baseline", "-sample", "2x2,1x2,2x1"), 37, 27)
    write_case("h4v1", cjpeg(pic(50, 21), "-quality", "85", "-baseline", "-sample", "4x1,1x1,2x1"), 50, 21)
    write_case("graycj", cjpeg(pic(27, 19, True), "-quality", "85", "-baseline"), 27, 19)
    scans = HERE / "_scans.txt"
    scans.write_text("0: 0 63 0 0;\n1: 0 63 0 0;\n2: 0 63 0 0;\n")
    try:
        write_case("multiscan420", cjpeg(pic(37, 27), "-quality", "80", "-sample", "2x2,1x1,1x1", "-scans", str(scans)), 37, 27)
        write_case("multiscan444", cjpeg(pic(21, 14), "-quality", "80", "-sample", "1x1,1x1,1x1", "-scans", str(scans)), 21, 14)
        scans.write_text("0 1: 0 63 0 0;\n2: 0 63 0 0;\n")
        write_case("twoscan420", cjpeg(pic(40, 30), "-quality", "80", "-sample", "2x2,1x1,1x1", "-scans", str(scans)), 40, 30)
    finally:
        scans.unlink()

    # Colour space detection (`default_decompress_parms`): the same YCbCr picture under each
    # marker / component-id combination. JFIF means YCbCr; else Adobe transform 0 means RGB and
    # 1 or any other value YCbCr; else ids 1,2,3 mean YCbCr, 'R','G','B' mean RGB, other ids YCbCr.
    base = pillow_save(pic(23, 17), quality=90, subsampling="4:2:0")
    write_case("cs_jfif", base, 23, 17)
    write_case("cs_noid123", restamp(base, drop=(JFIF,)), 23, 17)
    write_case("cs_idsRGB", restamp(base, drop=(JFIF,), ids=b"RGB"), 23, 17)
    write_case("cs_idsother", restamp(base, drop=(JFIF,), ids=bytes([7, 8, 9])), 23, 17)
    write_case("cs_adobe0", restamp(base, drop=(JFIF,), insert=(adobe(0),)), 23, 17)
    write_case("cs_adobe1", restamp(base, drop=(JFIF,), insert=(adobe(1),)), 23, 17)
    write_case("cs_adobe2", restamp(base, drop=(JFIF,), insert=(adobe(2),)), 23, 17)
    write_case("cs_adobe0idsRGB", restamp(base, drop=(JFIF,), insert=(adobe(0),), ids=b"RGB"), 23, 17)
    write_case("cs_jfifandadobe0", restamp(base, insert=(adobe(0),)), 23, 17)
    write_case("cs_cjpegrgb444", cjpeg(pic(26, 14), "-rgb", "-quality", "90", "-baseline"), 26, 14)
    write_case("cs_cjpegrgb420", cjpeg(pic(26, 14), "-rgb", "-quality", "90", "-baseline", "-sample", "2x2,1x1,1x1"), 26, 14)

    # EXIF orientation 6 (rotate 90 clockwise to display): Image.open does not apply it, and neither
    # does the decoder, so the stored pixels come back.
    exif = Image.Exif()
    exif[0x0112] = 6
    write_case("exif6", pillow_save(pic(23, 17), quality=90, subsampling="4:2:0", exif=exif), 23, 17)

    # What the native decoder refuses: progressive and CMYK (Pillow), plus a PNG.
    fb = pic(24, 16)
    (FALLBACK / "progressive_24x16.jpg").write_bytes(pillow_save(fb, quality=85, progressive=True))
    cm = io.BytesIO()
    fb.convert("CMYK").save(cm, "JPEG", quality=85)
    (FALLBACK / "cmyk_24x16.jpg").write_bytes(cm.getvalue())
    pn = io.BytesIO()
    pic(12, 8).save(pn, "PNG")
    (FALLBACK / "gradient_12x8.png").write_bytes(pn.getvalue())
    print("wrote fallback/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
