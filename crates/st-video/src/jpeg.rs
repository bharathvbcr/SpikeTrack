//! A baseline JPEG decoder whose 8-bit RGB output is byte-identical to Pillow
//! 13.0.0 on libjpeg-turbo 3.2.0 (`Image.open(path).convert("RGB")`).
//!
//! This software is based in part on the work of the Independent JPEG Group.
//! It is a Rust port of the algorithms of libjpeg-turbo 3.2.0's default
//! decompression path (IJG licence and BSD; see NOTICE), written from the
//! 3.2.0 sources: `jdmarker.c`, `jdinput.c`, `jdhuff.c`, `jidctint.c`,
//! `jdsample.c`, `jdmainct.c`, `jdcolor.c`/`jdcolext.c` and `jdapimin.c`.
//! Std only, no `unsafe`, no third-party crate.
//!
//! # What is decoded
//!
//! Sequential Huffman JPEG (SOF0, and SOF1 at 8-bit precision, which share one
//! pipeline), 8-bit samples, **1 component** (grayscale) or **3 components**
//! (YCbCr, or RGB when the file says so), sampling factors 1 to 4 in each
//! direction, interleaved or one-component-per-scan layouts, restart intervals
//! (DRI/RSTn), APPn and COM segments skipped. Everything else is refused by
//! name, as an [`Unsupported`] value and never a panic: progressive (SOF2),
//! arithmetic coding (SOF9 to SOF11), lossless (SOF3), hierarchical or
//! differential (SOF5 to SOF7, SOF13 to SOF15, DHP), a precision other than 8,
//! a component count other than 1 or 3 (CMYK and YCCK are four), and sampling
//! ratios that are not whole numbers.
//!
//! # Where the exact bits come from
//!
//! What Pillow asks libjpeg-turbo for is read from Pillow 13.0.0's
//! `src/libImaging/JpegDecode.c`: no draft mode (so `do_fancy_upsampling` stays
//! TRUE and `dct_method` stays `JDCT_DEFAULT`, which `jpeglib.h` defines as
//! `JDCT_ISLOW`), `jpeg_color_space` left to libjpeg's own guess, and the output
//! colour space RGB (or grayscale for one component). Merged upsampling
//! (`jdmerge.c`) is therefore never selected: `use_merged_upsample` in
//! `jdmaster.c` returns FALSE whenever `do_fancy_upsampling` is TRUE.
//!
//! Stages, each ported from the file named:
//!
//! 1. Entropy decoding, `jdhuff.c`: canonical Huffman tables built and checked
//!    as `jpeg_make_d_derived_tbl` does, `HUFF_EXTEND`, DC prediction kept as a
//!    32-bit running sum and stored as a 16-bit `JCOEF`.
//! 2. Dequantisation and inverse DCT, `jidctint.c` `_jpeg_idct_islow`
//!    (`CONST_BITS` 13, `PASS1_BITS` 2), with the post-IDCT range limit
//!    `range_limit[x & 1023]` expressed in closed form ([`range_limit_idct`]).
//!    The DC-only shortcuts of the C code are exact equivalents of the full
//!    path and are kept for speed.
//! 3. Upsampling, `jdsample.c`, chosen per component exactly as
//!    `_jinit_upsampler` chooses: full-size, `h2v1` fancy, `h1v2` fancy, `h2v2`
//!    fancy (the triangle filters, "3/4 nearer + 1/4 further" with the
//!    alternating rounding bias), and the box filters when
//!    `downsampled_width <= 2` or for other integral ratios. Context rows come
//!    from `jdmainct.c`: at the top of the image the row above row 0 is row 0
//!    itself, at the bottom the row below the last *real* component row
//!    (`downsampled_height - 1`, not the last row of the padded block) is that
//!    row itself. Fancy upsampling reads `downsampled_width` columns, so the
//!    padding columns of a partial MCU never enter the output.
//! 4. Colour conversion, `jdcolor.c`/`jdcolext.c`: YCbCr to RGB with 16-bit
//!    fixed-point tables (`FIX(1.40200)`, `FIX(1.77200)`, `FIX(0.71414)`,
//!    `FIX(0.34414)`), range limited; grayscale replicated to three channels
//!    (what Pillow's `convert("RGB")` does to mode `L`); RGB passed through.
//!    The colour space is `default_decompress_parms` from `jdapimin.c`: a JFIF
//!    marker means YCbCr; else an Adobe marker with transform 0 means RGB and
//!    anything else YCbCr; else component ids 1,2,3 mean YCbCr, 'R','G','B'
//!    mean RGB, any other ids YCbCr.
//!
//! libjpeg-turbo's SIMD routines (what the Pillow wheel runs) are specified to
//! match the C routines bit for bit for in-range data; the C sources are the
//! reference here, and `tests/decode.rs` checks the result against Pillow's own
//! output on VISEM frames.
//!
//! # Where this decoder is stricter than libjpeg
//!
//! libjpeg repairs damaged data and warns; this decoder returns an error, so a
//! damaged file is never silently turned into a picture. None of these fire on
//! a well-formed file: a Huffman code that does not exist (libjpeg returns 0),
//! a run that leaves the 64-coefficient block (libjpeg writes position 63),
//! entropy data that ends before the scan does (libjpeg pads zero bits), a
//! restart marker that is not the expected `RSTn` (libjpeg resynchronises), a
//! file with no end-of-image marker, a component that no scan covers, and a
//! zero-length marker segment. Scan parameters Ss, Se, Ah and Al are ignored,
//! as libjpeg ignores them for sequential files.
//!
//! # Bounds
//!
//! Dimensions are limited to libjpeg's 65500 and the pixel count to
//! [`MAX_PIXELS`] (2^27, the same bound as the ImageIO path; larger is
//! [`JpegError::TooLarge`]). Component planes are allocated only at the first
//! scan header, and only after checking that the data left in the file could
//! hold the image: every 8x8 block needs at least two bits (a one-bit DC code
//! and a one-bit end-of-block code), so a header that claims a large image over
//! a few bytes is refused before anything large is allocated. Every loop is
//! bounded by the input length or by the validated block grid.

use std::fmt;

use crate::resize::Rgb8;

/// The largest image, in pixels, that [`decode_jpeg`] accepts (2^27, as the
/// ImageIO path in `decode.rs`; an RGB result of this size is 384 MiB).
pub const MAX_PIXELS: usize = 1 << 27;

/// `JPEG_MAX_DIMENSION` in libjpeg: the largest width or height.
const MAX_DIMENSION: usize = 65500;

/// Position in the 8x8 block (row-major) of the i'th coefficient in zigzag
/// order (`jpeg_natural_order` in `jutils.c`).
const NATURAL_ORDER: [usize; 64] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20,
    13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59,
    52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
];

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// A well-formed JPEG this decoder refuses, by name. The caller may hand such a
/// file to another decoder.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Unsupported {
    /// SOF2: progressive Huffman.
    Progressive,
    /// SOF9, SOF10 or SOF11: arithmetic coding.
    Arithmetic,
    /// SOF3: lossless Huffman.
    Lossless,
    /// SOF5 to SOF7, SOF13 to SOF15 or a DHP segment: hierarchical (differential).
    Hierarchical,
    /// A sample precision other than 8 bits (12 and 16 are the real ones).
    Precision(u8),
    /// A component count other than 1 or 3 (4 is CMYK or YCCK).
    Components(u8),
    /// A component whose sampling ratio to the largest is not a whole number.
    FractionalSampling,
    /// A reserved marker (JPG, SOF8) that starts a frame.
    ReservedMarker(u8),
}

impl fmt::Display for Unsupported {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Progressive => write!(f, "progressive JPEG (SOF2)"),
            Self::Arithmetic => write!(f, "arithmetic-coded JPEG"),
            Self::Lossless => write!(f, "lossless JPEG (SOF3)"),
            Self::Hierarchical => write!(f, "hierarchical (differential) JPEG"),
            Self::Precision(p) => write!(f, "{p}-bit sample precision (only 8 is decoded)"),
            Self::Components(4) => write!(f, "4-component JPEG (CMYK or YCCK)"),
            Self::Components(n) => write!(f, "{n}-component JPEG (only 1 and 3 are decoded)"),
            Self::FractionalSampling => write!(f, "sampling factors with a non-integral ratio"),
            Self::ReservedMarker(m) => write!(f, "reserved frame marker ff{m:02x}"),
        }
    }
}

/// Why [`decode_jpeg`] returned no picture.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum JpegError {
    /// The data does not start with the SOI marker `ff d8`.
    NotJpeg,
    /// A valid-looking file in a variant this decoder refuses.
    Unsupported(Unsupported),
    /// More than [`MAX_PIXELS`] pixels.
    TooLarge { width: usize, height: usize },
    /// The data ends before the picture does; the detail says where.
    Truncated(&'static str),
    /// A marker, table, header or entropy-coded value that is invalid.
    Malformed(String),
}

impl fmt::Display for JpegError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotJpeg => write!(f, "not a JPEG (no SOI marker)"),
            Self::Unsupported(u) => write!(f, "unsupported: {u}"),
            Self::TooLarge { width, height } => {
                write!(f, "{width}x{height} exceeds {MAX_PIXELS} pixels")
            }
            Self::Truncated(what) => write!(f, "truncated: {what}"),
            Self::Malformed(what) => write!(f, "malformed: {what}"),
        }
    }
}

impl std::error::Error for JpegError {}

fn malformed(what: impl Into<String>) -> JpegError {
    JpegError::Malformed(what.into())
}

// ---------------------------------------------------------------------------
// Huffman tables and the bit reader (jdhuff.c)
// ---------------------------------------------------------------------------

/// A table as a DHT segment defines it: counts of codes per length and the
/// symbols in code order.
#[derive(Clone, Copy)]
struct HuffSpec {
    bits: [u8; 17],
    vals: [u8; 256],
}

/// Bits resolved by the direct lookup (`HUFF_LOOKAHEAD` in `jdhuff.c`).
const LOOKAHEAD: u32 = 9;

/// `d_derived_tbl`: the decoding form of a [`HuffSpec`].
struct HuffTable {
    /// Largest code of each length, -1 where there is none (index 17 unused).
    maxcode: [i32; 18],
    /// `huffval` index of a code of each length minus its first code.
    valoffset: [i32; 18],
    vals: [u8; 256],
    /// For every 9-bit prefix, `(length << 8) | symbol` when the code is at
    /// most 9 bits long, else 0.
    lookup: [u16; 1 << LOOKAHEAD],
}

impl HuffTable {
    /// `jpeg_make_d_derived_tbl`: validates that the counts form a legal
    /// prefix code (no code of all ones) and, for a DC table, that every
    /// symbol is a size category 0 to 15.
    fn build(spec: &HuffSpec, is_dc: bool) -> Result<HuffTable, JpegError> {
        let mut huffsize = [0u8; 257];
        let mut p = 0usize;
        for l in 1..=16usize {
            let n = usize::from(spec.bits[l]);
            if p + n > 256 {
                return Err(malformed("Huffman table has more than 256 codes"));
            }
            for _ in 0..n {
                huffsize[p] = l as u8;
                p += 1;
            }
        }
        let numsymbols = p;
        let mut huffcode = [0u32; 257];
        let mut code = 0u32;
        let mut si = u32::from(huffsize[0]);
        let mut p = 0usize;
        while huffsize[p] != 0 {
            while u32::from(huffsize[p]) == si {
                huffcode[p] = code;
                p += 1;
                code += 1;
            }
            // `code` is one more than the last code of length `si`; it must
            // still fit in `si` bits, since no code may be all ones.
            if code >= (1u32 << si) {
                return Err(malformed("Huffman table is not a legal prefix code"));
            }
            code <<= 1;
            si += 1;
        }
        let mut t = HuffTable {
            maxcode: [-1; 18],
            valoffset: [0; 18],
            vals: spec.vals,
            lookup: [0; 1 << LOOKAHEAD],
        };
        let mut p = 0usize;
        for l in 1..=16usize {
            let n = usize::from(spec.bits[l]);
            if n > 0 {
                t.valoffset[l] = p as i32 - huffcode[p] as i32;
                p += n;
                t.maxcode[l] = huffcode[p - 1] as i32;
            }
        }
        t.maxcode[17] = 0xF_FFFF;
        let mut p = 0usize;
        for l in 1..=LOOKAHEAD as usize {
            for _ in 0..spec.bits[l] {
                let first = (huffcode[p] as usize) << (LOOKAHEAD as usize - l);
                for slot in &mut t.lookup[first..first + (1 << (LOOKAHEAD as usize - l))] {
                    *slot = ((l as u16) << 8) | u16::from(spec.vals[p]);
                }
                p += 1;
            }
        }
        if is_dc && spec.vals[..numsymbols].iter().any(|&s| s > 15) {
            return Err(malformed("DC Huffman table has a symbol above 15"));
        }
        Ok(t)
    }
}

/// The entropy-coded segment's bit source: removes byte stuffing (`ff 00` is
/// a data byte), stops at a marker, and supplies zero bits after it while
/// counting them so that a block that consumed any of them is detected.
struct BitReader<'a> {
    data: &'a [u8],
    /// Next byte to read; at a marker it stays at the marker's first `ff`.
    pos: usize,
    buf: u64,
    nbits: u32,
    /// How many of the low `nbits` bits are zero padding, not data.
    pad: u32,
    ended: bool,
}

impl<'a> BitReader<'a> {
    fn new(data: &'a [u8], pos: usize) -> Self {
        BitReader {
            data,
            pos,
            buf: 0,
            nbits: 0,
            pad: 0,
            ended: false,
        }
    }

    /// Discard every buffered bit (a restart boundary).
    fn reset(&mut self) {
        self.buf = 0;
        self.nbits = 0;
        self.pad = 0;
        self.ended = false;
    }

    /// Top the buffer up to more than 48 bits.
    fn fill(&mut self) {
        while self.nbits <= 48 {
            let mut byte = 0u8;
            let mut real = false;
            if !self.ended {
                match self.data.get(self.pos) {
                    None => self.ended = true,
                    Some(&0xff) => {
                        // `ff` run: stuffed zero (a data byte `ff`) or a marker.
                        let mut j = self.pos + 1;
                        while self.data.get(j) == Some(&0xff) {
                            j += 1;
                        }
                        match self.data.get(j) {
                            Some(&0) => {
                                byte = 0xff;
                                real = true;
                                self.pos = j + 1;
                            }
                            _ => self.ended = true,
                        }
                    }
                    Some(&b) => {
                        byte = b;
                        real = true;
                        self.pos += 1;
                    }
                }
            }
            self.buf = (self.buf << 8) | u64::from(byte);
            self.nbits += 8;
            if !real {
                self.pad += 8;
            }
        }
    }

    #[inline]
    fn ensure(&mut self, n: u32) {
        if self.nbits < n {
            self.fill();
        }
    }

    /// The next `n` (at most 16) bits as a number.
    #[inline]
    fn get_bits(&mut self, n: u32) -> u32 {
        self.ensure(n);
        self.nbits -= n;
        ((self.buf >> self.nbits) & ((1u64 << n) - 1)) as u32
    }

    fn decode(&mut self, t: &HuffTable) -> Result<u8, JpegError> {
        self.ensure(16);
        let look = ((self.buf >> (self.nbits - LOOKAHEAD)) & ((1 << LOOKAHEAD) - 1)) as usize;
        let e = t.lookup[look];
        if e != 0 {
            self.nbits -= u32::from(e >> 8);
            return Ok(e as u8);
        }
        let v = ((self.buf >> (self.nbits - 16)) & 0xffff) as i32;
        for l in (LOOKAHEAD as usize + 1)..=16 {
            let code = v >> (16 - l);
            if code <= t.maxcode[l] {
                self.nbits -= l as u32;
                return Ok(t.vals[((code + t.valoffset[l]) & 0xff) as usize]);
            }
        }
        Err(malformed(
            "entropy data holds a Huffman code that is not in the table",
        ))
    }

    /// A block that consumed padding bits ran past the end of its segment.
    fn check_not_overrun(&self) -> Result<(), JpegError> {
        if self.nbits < self.pad {
            Err(JpegError::Truncated(
                "the entropy-coded data ends before the scan does",
            ))
        } else {
            Ok(())
        }
    }
}

/// `HUFF_EXTEND`: the signed value of an `s`-bit magnitude code `r`.
#[inline]
fn extend(r: u32, s: u32) -> i32 {
    if r < (1 << (s - 1)) {
        r as i32 - (1 << s) + 1
    } else {
        r as i32
    }
}

/// Decode one block's 64 coefficients into natural (row-major) order,
/// not dequantised. `decode_mcu_slow` in `jdhuff.c`.
fn decode_block(
    br: &mut BitReader<'_>,
    dc: &HuffTable,
    ac: &HuffTable,
    last_dc: &mut i32,
    out: &mut [i16; 64],
) -> Result<(), JpegError> {
    *out = [0; 64];
    let s = u32::from(br.decode(dc)?);
    let mut diff = 0i32;
    if s != 0 {
        diff = extend(br.get_bits(s), s);
    }
    // libjpeg keeps the running DC sum as an `int` and stores it as `JCOEF`.
    *last_dc = last_dc.wrapping_add(diff);
    out[0] = *last_dc as i16;
    let mut k = 1usize;
    while k < 64 {
        let sym = u32::from(br.decode(ac)?);
        let (r, s) = ((sym >> 4) as usize, sym & 15);
        if s != 0 {
            k += r;
            if k > 63 {
                return Err(malformed("a coefficient run leaves the 8x8 block"));
            }
            out[NATURAL_ORDER[k]] = extend(br.get_bits(s), s) as i16;
            k += 1;
        } else if r == 15 {
            k += 16;
        } else {
            break;
        }
    }
    br.check_not_overrun()
}

// ---------------------------------------------------------------------------
// Inverse DCT (jidctint.c, jpeg_idct_islow)
// ---------------------------------------------------------------------------

const CONST_BITS: u32 = 13;
const PASS1_BITS: u32 = 2;
const FIX_0_298631336: i64 = 2446;
const FIX_0_390180644: i64 = 3196;
const FIX_0_541196100: i64 = 4433;
const FIX_0_765366865: i64 = 6270;
const FIX_0_899976223: i64 = 7373;
const FIX_1_175875602: i64 = 9633;
const FIX_1_501321110: i64 = 12299;
const FIX_1_847759065: i64 = 15137;
const FIX_1_961570560: i64 = 16069;
const FIX_2_053119869: i64 = 16819;
const FIX_2_562915447: i64 = 20995;
const FIX_3_072711026: i64 = 25172;

/// `DESCALE`: right shift with rounding (an arithmetic shift of the biased value).
#[inline]
fn descale(x: i64, n: u32) -> i64 {
    (x + (1i64 << (n - 1))) >> n
}

/// `range_limit[x & RANGE_MASK]` on the post-IDCT table that starts at
/// `sample_range_limit + CENTERJSAMPLE`: take the low 10 bits as a signed
/// value, add 128, clamp to 0..=255 (`jdmaster.c`, `prepare_range_limit_table`).
#[inline]
fn range_limit_idct(x: i32) -> u8 {
    let s = ((x & 1023) << 22) >> 22;
    (s + 128).clamp(0, 255) as u8
}

/// Dequantise and inverse-transform one block (`coef` and `quant` in natural
/// order) into 8 rows of 8 samples at the start of `out`, row pitch `stride`.
fn idct_islow(coef: &[i16; 64], quant: &[i32; 64], out: &mut [u8], stride: usize) {
    let mut ws = [0i32; 64];
    for col in 0..8 {
        let c = |row: usize| i64::from(coef[row * 8 + col]);
        let dq = |row: usize| c(row) * i64::from(quant[row * 8 + col]);
        if (1..8).all(|row| coef[row * 8 + col] == 0) {
            // AC terms all zero: every output is the DC value, scaled.
            let dc = (dq(0) << PASS1_BITS) as i32;
            for row in 0..8 {
                ws[row * 8 + col] = dc;
            }
            continue;
        }
        // Even part.
        let (z2, z3) = (dq(2), dq(6));
        let z1 = (z2 + z3) * FIX_0_541196100;
        let tmp2 = z1 + z3 * -FIX_1_847759065;
        let tmp3 = z1 + z2 * FIX_0_765366865;
        let (z2, z3) = (dq(0), dq(4));
        let tmp0 = (z2 + z3) << CONST_BITS;
        let tmp1 = (z2 - z3) << CONST_BITS;
        let (tmp10, tmp13, tmp11, tmp12) = (tmp0 + tmp3, tmp0 - tmp3, tmp1 + tmp2, tmp1 - tmp2);
        // Odd part.
        let (t0, t1, t2, t3) = (dq(7), dq(5), dq(3), dq(1));
        let (z1, z2, z3, z4) = (t0 + t3, t1 + t2, t0 + t2, t1 + t3);
        let z5 = (z3 + z4) * FIX_1_175875602;
        let (t0, t1, t2, t3) = (
            t0 * FIX_0_298631336,
            t1 * FIX_2_053119869,
            t2 * FIX_3_072711026,
            t3 * FIX_1_501321110,
        );
        let (z1, z2, z3, z4) = (
            z1 * -FIX_0_899976223,
            z2 * -FIX_2_562915447,
            z3 * -FIX_1_961570560 + z5,
            z4 * -FIX_0_390180644 + z5,
        );
        let (t0, t1, t2, t3) = (t0 + z1 + z3, t1 + z2 + z4, t2 + z2 + z3, t3 + z1 + z4);
        let sh = CONST_BITS - PASS1_BITS;
        ws[col] = descale(tmp10 + t3, sh) as i32;
        ws[7 * 8 + col] = descale(tmp10 - t3, sh) as i32;
        ws[8 + col] = descale(tmp11 + t2, sh) as i32;
        ws[6 * 8 + col] = descale(tmp11 - t2, sh) as i32;
        ws[2 * 8 + col] = descale(tmp12 + t1, sh) as i32;
        ws[5 * 8 + col] = descale(tmp12 - t1, sh) as i32;
        ws[3 * 8 + col] = descale(tmp13 + t0, sh) as i32;
        ws[4 * 8 + col] = descale(tmp13 - t0, sh) as i32;
    }
    for row in 0..8 {
        let w = &ws[row * 8..row * 8 + 8];
        let o = &mut out[row * stride..row * stride + 8];
        if w[1..].iter().all(|&v| v == 0) {
            let dc = range_limit_idct(descale(i64::from(w[0]), PASS1_BITS + 3) as i32);
            o.fill(dc);
            continue;
        }
        let w = |i: usize| i64::from(w[i]);
        let (z2, z3) = (w(2), w(6));
        let z1 = (z2 + z3) * FIX_0_541196100;
        let tmp2 = z1 + z3 * -FIX_1_847759065;
        let tmp3 = z1 + z2 * FIX_0_765366865;
        let tmp0 = (w(0) + w(4)) << CONST_BITS;
        let tmp1 = (w(0) - w(4)) << CONST_BITS;
        let (tmp10, tmp13, tmp11, tmp12) = (tmp0 + tmp3, tmp0 - tmp3, tmp1 + tmp2, tmp1 - tmp2);
        let (t0, t1, t2, t3) = (w(7), w(5), w(3), w(1));
        let (z1, z2, z3, z4) = (t0 + t3, t1 + t2, t0 + t2, t1 + t3);
        let z5 = (z3 + z4) * FIX_1_175875602;
        let (t0, t1, t2, t3) = (
            t0 * FIX_0_298631336,
            t1 * FIX_2_053119869,
            t2 * FIX_3_072711026,
            t3 * FIX_1_501321110,
        );
        let (z1, z2, z3, z4) = (
            z1 * -FIX_0_899976223,
            z2 * -FIX_2_562915447,
            z3 * -FIX_1_961570560 + z5,
            z4 * -FIX_0_390180644 + z5,
        );
        let (t0, t1, t2, t3) = (t0 + z1 + z3, t1 + z2 + z4, t2 + z2 + z3, t3 + z1 + z4);
        let sh = CONST_BITS + PASS1_BITS + 3;
        o[0] = range_limit_idct(descale(tmp10 + t3, sh) as i32);
        o[7] = range_limit_idct(descale(tmp10 - t3, sh) as i32);
        o[1] = range_limit_idct(descale(tmp11 + t2, sh) as i32);
        o[6] = range_limit_idct(descale(tmp11 - t2, sh) as i32);
        o[2] = range_limit_idct(descale(tmp12 + t1, sh) as i32);
        o[5] = range_limit_idct(descale(tmp12 - t1, sh) as i32);
        o[3] = range_limit_idct(descale(tmp13 + t0, sh) as i32);
        o[4] = range_limit_idct(descale(tmp13 - t0, sh) as i32);
    }
}

// ---------------------------------------------------------------------------
// Frame geometry, upsampling (jdsample.c) and colour conversion (jdcolor.c)
// ---------------------------------------------------------------------------

/// The upsampler `_jinit_upsampler` picks for one component.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Upsample {
    /// Same size as the output: no work.
    Full,
    H2V1Fancy,
    H2V1Box,
    H1V2Fancy,
    H2V2Fancy,
    H2V2Box,
    /// `int_upsample`: replicate by whole factors.
    Int {
        h: usize,
        v: usize,
    },
}

/// `_jinit_upsampler`'s dispatch, with `do_fancy_upsampling` TRUE and the
/// DCT scale 8 for every component. `None` is `JERR_FRACT_SAMPLE_NOTIMPL`.
fn choose_upsampler(
    h_in: usize,
    v_in: usize,
    h_out: usize,
    v_out: usize,
    downsampled_width: usize,
) -> Option<Upsample> {
    if h_in == h_out && v_in == v_out {
        Some(Upsample::Full)
    } else if h_in * 2 == h_out && v_in == v_out {
        Some(if downsampled_width > 2 {
            Upsample::H2V1Fancy
        } else {
            Upsample::H2V1Box
        })
    } else if h_in == h_out && v_in * 2 == v_out {
        Some(Upsample::H1V2Fancy)
    } else if h_in * 2 == h_out && v_in * 2 == v_out {
        Some(if downsampled_width > 2 {
            Upsample::H2V2Fancy
        } else {
            Upsample::H2V2Box
        })
    } else if h_out % h_in == 0 && v_out % v_in == 0 {
        Some(Upsample::Int {
            h: h_out / h_in,
            v: v_out / v_in,
        })
    } else {
        None
    }
}

struct Component {
    id: u8,
    h: usize,
    v: usize,
    tq: usize,
    /// `width_in_blocks` and `height_in_blocks`: blocks that cover the image.
    wib: usize,
    hib: usize,
    /// `downsampled_width` and `downsampled_height`: real samples of this component.
    dw: usize,
    dh: usize,
    /// Plane pitch, and rows, rounded up to whole MCUs.
    stride: usize,
    rows: usize,
    up: Upsample,
    plane: Vec<u8>,
    /// Dequantisation table, natural order, latched at the component's first scan.
    quant: Option<[i32; 64]>,
    scanned: bool,
}

struct Frame {
    width: usize,
    height: usize,
    /// Interleaved MCU grid.
    mcus_x: usize,
    mcus_y: usize,
    comps: Vec<Component>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ColorSpace {
    Gray,
    YCbCr,
    Rgb,
}

fn div_ceil(a: usize, b: usize) -> usize {
    a.div_ceil(b)
}

/// `FIX(x) = (JLONG)(x * 65536 + 0.5)` in `jdcolor.c`.
const FIX_1_40200: i32 = 91881;
const FIX_1_77200: i32 = 116130;
const FIX_0_71414: i32 = 46802;
const FIX_0_34414: i32 = 22554;

/// The four YCbCr to RGB tables of `build_ycc_rgb_table`.
struct YccTables {
    cr_r: [i32; 256],
    cb_b: [i32; 256],
    cr_g: [i32; 256],
    cb_g: [i32; 256],
}

impl YccTables {
    fn new() -> Self {
        let mut t = YccTables {
            cr_r: [0; 256],
            cb_b: [0; 256],
            cr_g: [0; 256],
            cb_g: [0; 256],
        };
        for i in 0..256usize {
            let x = i as i32 - 128;
            t.cr_r[i] = (FIX_1_40200 * x + (1 << 15)) >> 16;
            t.cb_b[i] = (FIX_1_77200 * x + (1 << 15)) >> 16;
            t.cr_g[i] = -FIX_0_71414 * x;
            // ONE_HALF is added here so the inner loop need not.
            t.cb_g[i] = -FIX_0_34414 * x + (1 << 15);
        }
        t
    }
}

#[inline]
fn clamp_u8(v: i32) -> u8 {
    v.clamp(0, 255) as u8
}

impl Component {
    /// Row `y` of this component, upsampled to the full image width, into
    /// `out[..width]`. `tmp` is scratch of at least `2 * dw` entries.
    fn upsampled_row(&self, y: usize, width: usize, out: &mut [u8], tmp: &mut [i32]) {
        let row = |r: usize| &self.plane[r * self.stride..r * self.stride + self.dw];
        let dw = self.dw;
        match self.up {
            Upsample::Full => out[..width].copy_from_slice(&row(y)[..width]),
            Upsample::H2V1Box | Upsample::H2V2Box => {
                let src = row(if self.up == Upsample::H2V1Box {
                    y
                } else {
                    y / 2
                });
                for x in 0..width {
                    out[x] = src[x / 2];
                }
            }
            Upsample::Int { h, v } => {
                let src = row(y / v);
                for x in 0..width {
                    out[x] = src[x / h];
                }
            }
            Upsample::H1V2Fancy => {
                // Nearest row weighs 3, the next nearest (above for the upper
                // output row, below for the lower) weighs 1; the context rows
                // at the image edges are the edge rows themselves.
                let r = y / 2;
                let (nr, bias) = if y % 2 == 0 {
                    (r.saturating_sub(1), 1)
                } else {
                    ((r + 1).min(self.dh - 1), 2)
                };
                let (a, b) = (row(r), row(nr));
                for x in 0..width {
                    out[x] = ((i32::from(a[x]) * 3 + i32::from(b[x]) + bias) >> 2) as u8;
                }
            }
            Upsample::H2V1Fancy => {
                let s = row(y);
                let mut o = vec![0u8; 2 * dw];
                o[0] = s[0];
                o[1] = ((i32::from(s[0]) * 3 + i32::from(s[1]) + 2) >> 2) as u8;
                for i in 1..dw - 1 {
                    let v = i32::from(s[i]) * 3;
                    o[2 * i] = ((v + i32::from(s[i - 1]) + 1) >> 2) as u8;
                    o[2 * i + 1] = ((v + i32::from(s[i + 1]) + 2) >> 2) as u8;
                }
                let l = dw - 1;
                o[2 * l] = ((i32::from(s[l]) * 3 + i32::from(s[l - 1]) + 1) >> 2) as u8;
                o[2 * l + 1] = s[l];
                out[..width].copy_from_slice(&o[..width]);
            }
            Upsample::H2V2Fancy => {
                let r = y / 2;
                let nr = if y % 2 == 0 {
                    r.saturating_sub(1)
                } else {
                    (r + 1).min(self.dh - 1)
                };
                let (a, b) = (row(r), row(nr));
                let sum = &mut tmp[..dw];
                for i in 0..dw {
                    sum[i] = i32::from(a[i]) * 3 + i32::from(b[i]);
                }
                let mut o = vec![0u8; 2 * dw];
                o[0] = ((sum[0] * 4 + 8) >> 4) as u8;
                o[1] = ((sum[0] * 3 + sum[1] + 7) >> 4) as u8;
                for i in 1..dw - 1 {
                    o[2 * i] = ((sum[i] * 3 + sum[i - 1] + 8) >> 4) as u8;
                    o[2 * i + 1] = ((sum[i] * 3 + sum[i + 1] + 7) >> 4) as u8;
                }
                let l = dw - 1;
                o[2 * l] = ((sum[l] * 3 + sum[l - 1] + 8) >> 4) as u8;
                o[2 * l + 1] = ((sum[l] * 4 + 7) >> 4) as u8;
                out[..width].copy_from_slice(&o[..width]);
            }
        }
    }
}

impl Frame {
    /// Upsample every component and convert to interleaved RGB.
    fn to_rgb(&self, color: ColorSpace) -> Rgb8 {
        let (w, h) = (self.width, self.height);
        let mut rgb = vec![0u8; w * h * 3];
        let tables = YccTables::new();
        let mut rows: Vec<Vec<u8>> = self.comps.iter().map(|_| vec![0u8; w]).collect();
        let mut tmp = vec![0i32; self.comps.iter().map(|c| c.dw).max().unwrap_or(0)];
        for y in 0..h {
            for (c, row) in self.comps.iter().zip(rows.iter_mut()) {
                c.upsampled_row(y, w, row, &mut tmp);
            }
            let line = &mut rgb[y * w * 3..(y + 1) * w * 3];
            match color {
                ColorSpace::Gray => {
                    for (px, &g) in line.chunks_exact_mut(3).zip(rows[0].iter()) {
                        px.fill(g);
                    }
                }
                ColorSpace::Rgb => {
                    for (x, px) in line.chunks_exact_mut(3).enumerate() {
                        px.copy_from_slice(&[rows[0][x], rows[1][x], rows[2][x]]);
                    }
                }
                ColorSpace::YCbCr => {
                    for (x, px) in line.chunks_exact_mut(3).enumerate() {
                        let (y, cb, cr) = (
                            i32::from(rows[0][x]),
                            usize::from(rows[1][x]),
                            usize::from(rows[2][x]),
                        );
                        px[0] = clamp_u8(y + tables.cr_r[cr]);
                        px[1] = clamp_u8(y + ((tables.cb_g[cb] + tables.cr_g[cr]) >> 16));
                        px[2] = clamp_u8(y + tables.cb_b[cb]);
                    }
                }
            }
        }
        Rgb8::new(w, h, rgb)
    }
}

// ---------------------------------------------------------------------------
// Markers (jdmarker.c) and scans
// ---------------------------------------------------------------------------

struct Decoder<'a> {
    data: &'a [u8],
    pos: usize,
    dc_specs: [Option<HuffSpec>; 4],
    ac_specs: [Option<HuffSpec>; 4],
    quant: [Option<[u16; 64]>; 4],
    restart_interval: usize,
    frame: Option<Frame>,
    saw_jfif: bool,
    saw_adobe: bool,
    adobe_transform: u8,
    color: Option<ColorSpace>,
    scans: usize,
}

/// Decode a JPEG file's bytes to interleaved 8-bit RGB, byte-identical to
/// Pillow 13.0.0 with libjpeg-turbo 3.2.0 for every variant it accepts. See the
/// module documentation for what is accepted, refused and bounded.
pub fn decode_jpeg(bytes: &[u8]) -> Result<Rgb8, JpegError> {
    if bytes.len() < 2 || bytes[0] != 0xff || bytes[1] != 0xd8 {
        return Err(JpegError::NotJpeg);
    }
    let mut d = Decoder {
        data: bytes,
        pos: 2,
        dc_specs: [None; 4],
        ac_specs: [None; 4],
        quant: [None; 4],
        restart_interval: 0,
        frame: None,
        saw_jfif: false,
        saw_adobe: false,
        adobe_transform: 0,
        color: None,
        scans: 0,
    };
    d.run()
}

/// The next marker code at or after `*pos`, as `next_marker` in `jdmarker.c`:
/// bytes that are not `ff` are skipped, runs of `ff` are fill, and `ff 00`
/// (a stuffed zero) is not a marker. `*pos` ends just after the marker code.
fn next_marker(data: &[u8], pos: &mut usize) -> Result<u8, JpegError> {
    loop {
        while *data
            .get(*pos)
            .ok_or(JpegError::Truncated("no end-of-image marker"))?
            != 0xff
        {
            *pos += 1;
        }
        while data.get(*pos) == Some(&0xff) {
            *pos += 1;
        }
        let c = *data
            .get(*pos)
            .ok_or(JpegError::Truncated("the data ends inside a marker"))?;
        *pos += 1;
        if c != 0 {
            return Ok(c);
        }
    }
}

/// `default_decompress_parms` in `jdapimin.c`, evaluated at the first scan
/// header (a JFIF or Adobe marker after that point is not seen by libjpeg).
fn guess_color(
    saw_jfif: bool,
    saw_adobe: bool,
    adobe_transform: u8,
    comps: &[Component],
) -> ColorSpace {
    if comps.len() == 1 {
        ColorSpace::Gray
    } else if saw_jfif {
        ColorSpace::YCbCr
    } else if saw_adobe {
        if adobe_transform == 0 {
            ColorSpace::Rgb
        } else {
            ColorSpace::YCbCr
        }
    } else if (comps[0].id, comps[1].id, comps[2].id) == (b'R', b'G', b'B') {
        ColorSpace::Rgb
    } else {
        ColorSpace::YCbCr
    }
}

impl<'a> Decoder<'a> {
    /// The parameters of the segment whose marker was just read: the bytes
    /// after its two-byte length field.
    fn segment(&mut self) -> Result<&'a [u8], JpegError> {
        let len = self
            .data
            .get(self.pos..self.pos + 2)
            .map(|l| usize::from(u16::from_be_bytes([l[0], l[1]])))
            .ok_or(JpegError::Truncated(
                "the data ends inside a segment header",
            ))?;
        if len < 2 {
            return Err(malformed(format!("segment length {len} is below 2")));
        }
        let end = self.pos + len;
        let payload = self
            .data
            .get(self.pos + 2..end)
            .ok_or(JpegError::Truncated("a segment is longer than the data"))?;
        self.pos = end;
        Ok(payload)
    }

    fn run(&mut self) -> Result<Rgb8, JpegError> {
        loop {
            let marker = next_marker(self.data, &mut self.pos)?;
            match marker {
                0xd8 => return Err(malformed("a second SOI marker")),
                0xc0 | 0xc1 => {
                    let p = self.segment()?;
                    self.sof(p)?;
                }
                0xc2 => return Err(JpegError::Unsupported(Unsupported::Progressive)),
                0xc3 => return Err(JpegError::Unsupported(Unsupported::Lossless)),
                0xc9..=0xcb => return Err(JpegError::Unsupported(Unsupported::Arithmetic)),
                0xc5..=0xc7 | 0xcd..=0xcf | 0xde => {
                    return Err(JpegError::Unsupported(Unsupported::Hierarchical))
                }
                0xc8 => return Err(JpegError::Unsupported(Unsupported::ReservedMarker(marker))),
                0xc4 => {
                    let p = self.segment()?;
                    self.dht(p)?;
                }
                0xdb => {
                    let p = self.segment()?;
                    self.dqt(p)?;
                }
                0xdd => {
                    let p = self.segment()?;
                    if p.len() != 2 {
                        return Err(malformed("DRI segment is not 4 bytes"));
                    }
                    self.restart_interval = usize::from(u16::from_be_bytes([p[0], p[1]]));
                }
                0xda => {
                    let p = self.segment()?;
                    self.scan(p)?;
                }
                0xd9 => return self.finish(),
                0xe0 => {
                    let p = self.segment()?;
                    // JFIF APP0: "JFIF\0" within the first 14 data bytes.
                    if p.len() >= 14 && p[..5] == *b"JFIF\0" {
                        self.saw_jfif = true;
                    }
                }
                0xee => {
                    let p = self.segment()?;
                    // Adobe APP14: "Adobe" and the transform byte at offset 11.
                    if p.len() >= 12 && p[..5] == *b"Adobe" {
                        self.saw_adobe = true;
                        self.adobe_transform = p[11];
                    }
                }
                // Other APPn, COM, and DAC (arithmetic conditioning, which only
                // an arithmetic file has and which is refused at its SOF).
                0xe1..=0xed | 0xef | 0xfe | 0xcc => {
                    self.segment()?;
                }
                // Parameterless: restart markers outside a scan and TEM.
                0xd0..=0xd7 | 0x01 => {}
                m => return Err(malformed(format!("unknown marker ff{m:02x}"))),
            }
        }
    }

    fn sof(&mut self, p: &[u8]) -> Result<(), JpegError> {
        if self.frame.is_some() {
            return Err(malformed("a second SOF marker"));
        }
        if p.len() < 6 {
            return Err(malformed("SOF segment is too short"));
        }
        let (precision, height, width, n) = (
            p[0],
            usize::from(u16::from_be_bytes([p[1], p[2]])),
            usize::from(u16::from_be_bytes([p[3], p[4]])),
            usize::from(p[5]),
        );
        if height == 0 || width == 0 || n == 0 {
            return Err(malformed("the image has no pixels (a height of 0 is DNL-defined, which is not supported either)"));
        }
        if p.len() != 6 + 3 * n {
            return Err(malformed(
                "SOF segment length does not match its component count",
            ));
        }
        if precision != 8 {
            return Err(JpegError::Unsupported(Unsupported::Precision(precision)));
        }
        if n != 1 && n != 3 {
            return Err(JpegError::Unsupported(Unsupported::Components(n as u8)));
        }
        if width > MAX_DIMENSION || height > MAX_DIMENSION {
            return Err(malformed(format!(
                "{width}x{height} exceeds {MAX_DIMENSION} in a dimension"
            )));
        }
        if width * height > MAX_PIXELS {
            return Err(JpegError::TooLarge { width, height });
        }
        let mut comps = Vec::with_capacity(n);
        for c in p[6..].chunks_exact(3) {
            let (h, v) = (usize::from(c[1] >> 4), usize::from(c[1] & 15));
            if !(1..=4).contains(&h) || !(1..=4).contains(&v) {
                return Err(malformed("sampling factor outside 1..=4"));
            }
            comps.push((c[0], h, v, usize::from(c[2])));
        }
        let max_h = comps.iter().map(|c| c.1).max().unwrap_or(1);
        let max_v = comps.iter().map(|c| c.2).max().unwrap_or(1);
        let (mcus_x, mcus_y) = (div_ceil(width, max_h * 8), div_ceil(height, max_v * 8));
        let mut out = Vec::with_capacity(n);
        for (id, h, v, tq) in comps {
            let dw = div_ceil(width * h, max_h);
            let up = choose_upsampler(h, v, max_h, max_v, dw)
                .ok_or(JpegError::Unsupported(Unsupported::FractionalSampling))?;
            out.push(Component {
                id,
                h,
                v,
                tq,
                wib: div_ceil(width * h, max_h * 8),
                hib: div_ceil(height * v, max_v * 8),
                dw,
                dh: div_ceil(height * v, max_v),
                stride: mcus_x * h * 8,
                rows: mcus_y * v * 8,
                up,
                plane: Vec::new(),
                quant: None,
                scanned: false,
            });
        }
        self.frame = Some(Frame {
            width,
            height,
            mcus_x,
            mcus_y,
            comps: out,
        });
        Ok(())
    }

    fn dht(&mut self, mut p: &[u8]) -> Result<(), JpegError> {
        while p.len() > 16 {
            let index = usize::from(p[0]);
            let mut bits = [0u8; 17];
            bits[1..].copy_from_slice(&p[1..17]);
            let count: usize = bits.iter().map(|&b| usize::from(b)).sum();
            p = &p[17..];
            if count > 256 || count > p.len() {
                return Err(malformed("DHT counts exceed the segment or 256 codes"));
            }
            let mut vals = [0u8; 256];
            vals[..count].copy_from_slice(&p[..count]);
            p = &p[count..];
            let spec = HuffSpec { bits, vals };
            let (is_ac, i) = if index & 0x10 != 0 {
                (true, index - 0x10)
            } else {
                (false, index)
            };
            if i >= 4 {
                return Err(malformed(format!(
                    "DHT table index {index:#x} is out of range"
                )));
            }
            if is_ac {
                self.ac_specs[i] = Some(spec);
            } else {
                self.dc_specs[i] = Some(spec);
            }
        }
        if !p.is_empty() {
            return Err(malformed("DHT segment length does not match its tables"));
        }
        Ok(())
    }

    fn dqt(&mut self, mut p: &[u8]) -> Result<(), JpegError> {
        while let Some(&n) = p.first() {
            let (wide, id) = (n >> 4 != 0, usize::from(n & 15));
            if id >= 4 {
                return Err(malformed(format!("DQT table index {id} is out of range")));
            }
            let need = 1 + if wide { 128 } else { 64 };
            if p.len() < need {
                return Err(malformed("DQT segment length does not match its tables"));
            }
            let mut table = [0u16; 64];
            for (i, &pos) in NATURAL_ORDER.iter().enumerate() {
                table[pos] = if wide {
                    u16::from_be_bytes([p[1 + 2 * i], p[2 + 2 * i]])
                } else {
                    u16::from(p[1 + i])
                };
            }
            self.quant[id] = Some(table);
            p = &p[need..];
        }
        Ok(())
    }

    /// One scan: its header, then the entropy-coded data up to the next marker.
    fn scan(&mut self, p: &[u8]) -> Result<(), JpegError> {
        let frame = self
            .frame
            .as_mut()
            .ok_or_else(|| malformed("a scan before any frame header"))?;
        let n = usize::from(*p.first().ok_or_else(|| malformed("empty SOS segment"))?);
        if !(1..=4).contains(&n) || p.len() != 1 + 2 * n + 3 {
            return Err(malformed(
                "SOS segment length does not match its component count",
            ));
        }
        let mut members: Vec<(usize, usize, usize)> = Vec::with_capacity(n); // (component, dc table, ac table)
        for s in p[1..1 + 2 * n].chunks_exact(2) {
            let ci = frame
                .comps
                .iter()
                .enumerate()
                .position(|(i, c)| c.id == s[0] && !members.iter().any(|m| m.0 == i))
                .ok_or_else(|| {
                    malformed(format!("scan names component {} twice or not at all", s[0]))
                })?;
            members.push((ci, usize::from(s[1] >> 4), usize::from(s[1] & 15)));
        }
        if self.color.is_none() {
            self.color = Some(guess_color(
                self.saw_jfif,
                self.saw_adobe,
                self.adobe_transform,
                &frame.comps,
            ));
        }
        // Planes are allocated here, once, after checking that the data left
        // could hold the image: two bits at least per block.
        if self.scans == 0 {
            let blocks: usize = frame.comps.iter().map(|c| c.wib * c.hib).sum();
            let left = self.data.len().saturating_sub(self.pos);
            if blocks.saturating_mul(2) > left.saturating_mul(8) {
                return Err(JpegError::Truncated(
                    "the data after the first scan header is too short for the declared image",
                ));
            }
            for c in &mut frame.comps {
                c.plane = vec![0u8; c.stride * c.rows];
            }
        }
        // Quantisation tables are latched at a component's first scan.
        let mut quants: Vec<[i32; 64]> = Vec::with_capacity(n);
        for &(ci, ..) in &members {
            let c = &mut frame.comps[ci];
            let latched = match c.quant {
                Some(t) => t,
                None => {
                    let q = self.quant.get(c.tq).copied().flatten().ok_or_else(|| {
                        malformed(format!(
                            "component {} has no quantisation table {}",
                            c.id, c.tq
                        ))
                    })?;
                    let mut t = [0i32; 64];
                    for (d, &s) in t.iter_mut().zip(q.iter()) {
                        *d = i32::from(s);
                    }
                    c.quant = Some(t);
                    t
                }
            };
            quants.push(latched);
        }
        let mut tables: Vec<(HuffTable, HuffTable)> = Vec::with_capacity(n);
        for &(_, dc, ac) in &members {
            let dcs = self.dc_specs.get(dc).copied().flatten().ok_or_else(|| {
                malformed(format!(
                    "scan uses DC Huffman table {dc}, which no DHT defined"
                ))
            })?;
            let acs = self.ac_specs.get(ac).copied().flatten().ok_or_else(|| {
                malformed(format!(
                    "scan uses AC Huffman table {ac}, which no DHT defined"
                ))
            })?;
            tables.push((
                HuffTable::build(&dcs, true)?,
                HuffTable::build(&acs, false)?,
            ));
        }
        // MCU layout (`per_scan_setup` in jdinput.c).
        let interleaved = n > 1;
        let (mcu_cols, mcu_rows) = if interleaved {
            (frame.mcus_x, frame.mcus_y)
        } else {
            (frame.comps[members[0].0].wib, frame.comps[members[0].0].hib)
        };
        if interleaved {
            let blocks: usize = members
                .iter()
                .map(|m| frame.comps[m.0].h * frame.comps[m.0].v)
                .sum();
            if blocks > 10 {
                return Err(malformed(format!(
                    "an MCU of {blocks} blocks (the limit is 10)"
                )));
            }
        }
        let mut br = BitReader::new(self.data, self.pos);
        let mut last_dc = [0i32; 4];
        let mut to_go = self.restart_interval;
        let mut next_rst = 0u8;
        let mut coef = [0i16; 64];
        for my in 0..mcu_rows {
            for mx in 0..mcu_cols {
                if self.restart_interval != 0 {
                    if to_go == 0 {
                        br.reset();
                        let m = next_marker(self.data, &mut br.pos)?;
                        if m != 0xd0 + next_rst {
                            return Err(malformed(format!(
                                "expected restart marker RST{next_rst}, found ff{m:02x}"
                            )));
                        }
                        next_rst = (next_rst + 1) & 7;
                        last_dc = [0; 4];
                        to_go = self.restart_interval;
                    }
                    to_go -= 1;
                }
                for (mi, &(ci, _, _)) in members.iter().enumerate() {
                    let c = &mut frame.comps[ci];
                    let (bw, bh) = if interleaved { (c.h, c.v) } else { (1, 1) };
                    let q = &quants[mi];
                    for by in 0..bh {
                        for bx in 0..bw {
                            decode_block(
                                &mut br,
                                &tables[mi].0,
                                &tables[mi].1,
                                &mut last_dc[mi],
                                &mut coef,
                            )?;
                            let (x, y) = ((mx * bw + bx) * 8, (my * bh + by) * 8);
                            idct_islow(&coef, q, &mut c.plane[y * c.stride + x..], c.stride);
                        }
                    }
                }
            }
        }
        for &(ci, ..) in &members {
            frame.comps[ci].scanned = true;
        }
        self.scans += 1;
        self.pos = br.pos;
        Ok(())
    }

    /// EOI reached: the picture must be complete.
    fn finish(&mut self) -> Result<Rgb8, JpegError> {
        let frame = self
            .frame
            .as_ref()
            .ok_or_else(|| malformed("end of image before any frame header"))?;
        if self.scans == 0 {
            return Err(malformed("end of image before any scan"));
        }
        if let Some(c) = frame.comps.iter().find(|c| !c.scanned) {
            return Err(malformed(format!("no scan covers component {}", c.id)));
        }
        let color = self
            .color
            .ok_or_else(|| malformed("end of image before any scan"))?;
        Ok(frame.to_rgb(color))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Huffman -------------------------------------------------------

    fn spec(counts: &[(usize, u8)], vals: &[u8]) -> HuffSpec {
        let mut s = HuffSpec {
            bits: [0; 17],
            vals: [0; 256],
        };
        for &(l, n) in counts {
            s.bits[l] = n;
        }
        s.vals[..vals.len()].copy_from_slice(vals);
        s
    }

    #[test]
    fn huffman_decodes_canonical_codes_short_and_long() {
        // Lengths 1, 2, 3 and two codes of length 12 (past the 9-bit lookup):
        // codes 0, 10, 110, 1110_0000_0000, 1110_0000_0001.
        let s = spec(&[(1, 1), (2, 1), (3, 1), (12, 2)], &[7, 8, 9, 10, 11]);
        let t = HuffTable::build(&s, false).expect("legal table");
        // 0 | 10 | 110 | 111000000000 | 111000000001 | 0, padded with 1s to a byte boundary.
        let bits = "0".to_string() + "10" + "110" + "111000000000" + "111000000001" + "0";
        let mut bytes = Vec::new();
        let padded = format!("{bits:1<48}");
        for chunk in padded.as_bytes().chunks(8) {
            let byte = chunk.iter().fold(0u8, |a, &c| (a << 1) | (c - b'0'));
            bytes.push(byte);
            if byte == 0xff {
                bytes.push(0); // stuffed zero
            }
        }
        let mut br = BitReader::new(&bytes, 0);
        let got: Vec<u8> = (0..6).map(|_| br.decode(&t).expect("symbol")).collect();
        assert_eq!(got, [7, 8, 9, 10, 11, 7]);
    }

    #[test]
    fn huffman_table_validation() {
        // All-ones code: two codes of length 1 leave no room (code 1 = all ones is
        // legal for the second, but a third code of length 1 overflows).
        assert!(
            HuffTable::build(&spec(&[(1, 3)], &[0, 1, 2]), false).is_err(),
            "three 1-bit codes"
        );
        // Two 1-bit codes use '0' and '1', and '1' is all ones: refused.
        assert!(
            HuffTable::build(&spec(&[(1, 2)], &[0, 1]), false).is_err(),
            "a code of all ones"
        );
        assert!(HuffTable::build(&spec(&[(2, 3)], &[0, 1, 2]), false).is_ok());
        // A DC symbol above 15.
        assert!(HuffTable::build(&spec(&[(2, 2)], &[3, 16]), true).is_err());
        assert!(
            HuffTable::build(&spec(&[(2, 2)], &[3, 16]), false).is_ok(),
            "AC symbols are not range checked"
        );
    }

    #[test]
    fn a_code_that_is_not_in_the_table_is_an_error() {
        // One 2-bit code '00'; the data starts with 1s.
        let t = HuffTable::build(&spec(&[(2, 1)], &[5]), false).expect("table");
        let mut br = BitReader::new(&[0xff, 0x00, 0xff, 0x00, 0xff, 0x00], 0);
        assert!(matches!(br.decode(&t), Err(JpegError::Malformed(_))));
    }

    #[test]
    fn extend_matches_the_standard_sign_extension() {
        assert_eq!(extend(0, 1), -1);
        assert_eq!(extend(1, 1), 1);
        assert_eq!(extend(0b011, 3), -4);
        assert_eq!(extend(0b100, 3), 4);
        assert_eq!(extend(0, 11), -2047);
        assert_eq!(extend(2047, 11), 2047);
    }

    #[test]
    fn the_bit_reader_stops_at_a_marker_and_counts_padding() {
        let data = [0xab, 0xff, 0x00, 0x12, 0xff, 0xd9];
        let mut br = BitReader::new(&data, 0);
        assert_eq!(br.get_bits(8), 0xab);
        assert_eq!(br.get_bits(8), 0xff, "ff 00 is a data byte");
        assert_eq!(br.get_bits(8), 0x12);
        assert!(br.check_not_overrun().is_ok());
        assert_eq!(br.pos, 4, "the reader stays at the marker's ff");
        assert_eq!(br.get_bits(8), 0, "zero bits follow a marker");
        assert!(matches!(
            br.check_not_overrun(),
            Err(JpegError::Truncated(_))
        ));
    }

    // ---- IDCT ----------------------------------------------------------

    /// A tiny xorshift generator: the tests need no crate for random blocks.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn range(&mut self, lo: i64, hi: i64) -> i64 {
            lo + (self.next() % (hi - lo + 1) as u64) as i64
        }
    }

    /// The textbook inverse DCT in f64, level shifted (128 added), not clamped.
    fn idct_reference(coef: &[i16; 64], quant: &[i32; 64]) -> [f64; 64] {
        let mut out = [0f64; 64];
        for y in 0..8 {
            for x in 0..8 {
                let mut s = 0f64;
                for v in 0..8 {
                    for u in 0..8 {
                        let cu = if u == 0 {
                            std::f64::consts::FRAC_1_SQRT_2
                        } else {
                            1.0
                        };
                        let cv = if v == 0 {
                            std::f64::consts::FRAC_1_SQRT_2
                        } else {
                            1.0
                        };
                        let f = f64::from(coef[v * 8 + u]) * f64::from(quant[v * 8 + u]);
                        s += cu
                            * cv
                            * f
                            * ((2 * x + 1) as f64 * u as f64 * std::f64::consts::PI / 16.0).cos()
                            * ((2 * y + 1) as f64 * v as f64 * std::f64::consts::PI / 16.0).cos();
                    }
                }
                out[y * 8 + x] = s / 4.0 + 128.0;
            }
        }
        out
    }

    #[test]
    fn islow_idct_is_within_one_level_of_the_exact_transform_on_random_blocks() {
        let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
        let (mut worst, mut compared, mut inside) = (0i32, 0usize, 0usize);
        for case in 0..2000 {
            // Realistic content: quantised coefficients, sparse AC, bounded so the
            // result mostly stays inside 0..=255 (clamping hides nothing either way).
            let mut quant = [0i32; 64];
            for q in &mut quant {
                *q = rng.range(1, 40) as i32;
            }
            let mut coef = [0i16; 64];
            coef[0] = rng.range(-60, 60) as i16;
            let nonzero = rng.range(0, 20);
            for _ in 0..nonzero {
                let k = rng.range(1, 63) as usize;
                coef[k] = rng.range(-12, 12) as i16;
            }
            let mut out = [0u8; 64];
            idct_islow(&coef, &quant, &mut out, 8);
            let want = idct_reference(&coef, &quant);
            for i in 0..64 {
                // libjpeg masks the descaled value to 10 bits, so a sample more than
                // 512 away from mid-grey wraps around; that is corrupt-data behaviour
                // (checked by the range-limit table test), not what is measured here.
                if (want[i] - 128.0).abs() >= 500.0 {
                    continue;
                }
                compared += 1;
                inside += usize::from(want[i] > 1.0 && want[i] < 254.0);
                let exact = want[i].round().clamp(0.0, 255.0) as i32;
                let d = (i32::from(out[i]) - exact).abs();
                worst = worst.max(d);
                assert!(
                    d <= 1,
                    "case {case}: sample {i}: islow {} vs exact {}",
                    out[i],
                    want[i]
                );
            }
        }
        assert!(
            compared > 100_000 && inside * 2 > compared,
            "{inside} of {compared} samples are unsaturated"
        );
        assert!(worst <= 1);
    }

    #[test]
    fn idct_shortcuts_equal_the_full_path_and_a_flat_block_is_flat() {
        // A DC-only block: every sample is the rounded DC / 8 + 128.
        let quant = [3i32; 64];
        for dc in [-300i16, -17, -1, 0, 1, 5, 99, 310] {
            let mut coef = [0i16; 64];
            coef[0] = dc;
            let mut out = [0u8; 64];
            idct_islow(&coef, &quant, &mut out, 8);
            // libjpeg rounds ties upward (a biased arithmetic shift), not away from zero.
            let want = (f64::from(dc) * 3.0 / 8.0 + 128.5)
                .floor()
                .clamp(0.0, 255.0) as u8;
            assert!(out.iter().all(|&v| v == want), "dc {dc}: {out:?} vs {want}");
        }
    }

    #[test]
    fn the_post_idct_range_limit_matches_the_libjpeg_table() {
        // The 1024-entry table of `prepare_range_limit_table`, built literally.
        let mut table = vec![0u8; 1024];
        for i in 0..128 {
            table[i] = 128 + i as u8;
        }
        for t in table.iter_mut().take(512).skip(128) {
            *t = 255;
        }
        // 512..896 stay 0; 896..1024 hold 0..128.
        for i in 0..128 {
            table[896 + i] = i as u8;
        }
        for x in -4096i32..4096 {
            assert_eq!(range_limit_idct(x), table[(x & 1023) as usize], "x = {x}");
        }
    }

    #[test]
    fn natural_order_is_a_zigzag_permutation() {
        let mut seen = [false; 64];
        for &p in &NATURAL_ORDER {
            assert!(!seen[p]);
            seen[p] = true;
        }
        // Adjacent zigzag positions are neighbours on the grid.
        for w in NATURAL_ORDER.windows(2) {
            let ((r0, c0), (r1, c1)) = ((w[0] / 8, w[0] % 8), (w[1] / 8, w[1] % 8));
            assert!(r0.abs_diff(r1) <= 1 && c0.abs_diff(c1) <= 1);
        }
    }

    // ---- Colour conversion ---------------------------------------------

    #[test]
    fn ycc_constants_are_fix_of_the_stated_coefficients() {
        let fix = |x: f64| (x * 65536.0 + 0.5) as i32;
        assert_eq!(
            (FIX_1_40200, FIX_1_77200, FIX_0_71414, FIX_0_34414),
            (fix(1.40200), fix(1.77200), fix(0.71414), fix(0.34414))
        );
        let t = YccTables::new();
        // Neutral chroma maps to no change; Cr = 255 adds 1.402 * 127 = 178.05 to R.
        assert_eq!(
            (t.cr_r[128], t.cb_b[128], t.cr_g[128], t.cb_g[128]),
            (0, 0, 0, 1 << 15)
        );
        assert_eq!(t.cr_r[255], 178);
        assert_eq!(t.cb_b[255], 225);
    }

    #[test]
    fn colour_space_guess_follows_default_decompress_parms() {
        let c = |ids: [u8; 3]| -> Vec<Component> {
            ids.iter()
                .map(|&id| Component {
                    id,
                    ..comp(Upsample::Full, &[&[0]])
                })
                .collect()
        };
        let g = |jfif, adobe, t, ids| guess_color(jfif, adobe, t, &c(ids));
        assert_eq!(
            guess_color(false, false, 0, &c([1, 2, 3])[..1]),
            ColorSpace::Gray
        );
        // JFIF wins over everything, including an Adobe transform of 0.
        assert_eq!(g(true, true, 0, [b'R', b'G', b'B']), ColorSpace::YCbCr);
        // Adobe: transform 0 is RGB, 1 and anything else YCbCr.
        assert_eq!(g(false, true, 0, [1, 2, 3]), ColorSpace::Rgb);
        assert_eq!(g(false, true, 1, [b'R', b'G', b'B']), ColorSpace::YCbCr);
        assert_eq!(g(false, true, 2, [1, 2, 3]), ColorSpace::YCbCr);
        // No marker: ids decide.
        assert_eq!(g(false, false, 0, [1, 2, 3]), ColorSpace::YCbCr);
        assert_eq!(g(false, false, 0, [b'R', b'G', b'B']), ColorSpace::Rgb);
        assert_eq!(g(false, false, 0, [b'R', b'G', b'C']), ColorSpace::YCbCr);
        assert_eq!(g(false, false, 0, [0, 1, 2]), ColorSpace::YCbCr);
    }

    // ---- Upsampling ----------------------------------------------------

    /// A component over a small hand-written plane (`rows` of `dw` samples).
    fn comp(up: Upsample, rows: &[&[u8]]) -> Component {
        let (dw, dh) = (rows[0].len(), rows.len());
        Component {
            id: 1,
            h: 1,
            v: 1,
            tq: 0,
            wib: 1,
            hib: 1,
            dw,
            dh,
            stride: dw,
            rows: dh,
            up,
            plane: rows.concat(),
            quant: None,
            scanned: true,
        }
    }

    fn row_of(c: &Component, y: usize, width: usize) -> Vec<u8> {
        let mut out = vec![0u8; width];
        let mut tmp = vec![0i32; c.dw];
        c.upsampled_row(y, width, &mut out, &mut tmp);
        out
    }

    #[test]
    fn h2v1_fancy_matches_hand_computation() {
        // in = 10 20 30 40. First column: 10, (30+20+2)>>2 = 13. Then for 20:
        // (60+10+1)>>2 = 17, (60+30+2)>>2 = 23. For 30: (90+20+1)>>2 = 27,
        // (90+40+2)>>2 = 33. Last column 40: (120+30+1)>>2 = 37, 40.
        let c = comp(Upsample::H2V1Fancy, &[&[10, 20, 30, 40]]);
        assert_eq!(row_of(&c, 0, 8), [10, 13, 17, 23, 27, 33, 37, 40]);
        // Output width 7 (odd image width) drops the last sample.
        assert_eq!(row_of(&c, 0, 7), [10, 13, 17, 23, 27, 33, 37]);
    }

    #[test]
    fn h2v2_fancy_matches_hand_computation() {
        // Two rows a = [8, 16, 24], b = [40, 32, 16]. Output row 0 is nearest a, next b? No:
        // row group 0 has input row 0 (nearest) and the replicated row above (= row 0),
        // so output row 0 uses colsum = 3a + a = 4a = [32, 64, 96].
        let c = comp(Upsample::H2V2Fancy, &[&[8, 16, 24], &[40, 32, 16]]);
        // y = 0: sums 32 64 96: o0 = (32*4+8)>>4 = 8; o1 = (32*3+64+7)>>4 = 10; o2 = (64*3+32+8)>>4 = 14;
        // o3 = (64*3+96+7)>>4 = 18; o4 = (96*3+64+8)>>4 = 22; o5 = (96*4+7)>>4 = 24.
        assert_eq!(row_of(&c, 0, 6), [8, 10, 14, 18, 22, 24]);
        // y = 1: nearest a, next nearest b (below): sums 3a+b = 64 80 88.
        // o0 = (64*4+8)>>4 = 16; o1 = (64*3+80+7)>>4 = 17; o2 = (80*3+64+8)>>4 = 19;
        // o3 = (80*3+88+7)>>4 = 20; o4 = (88*3+80+8)>>4 = 22; o5 = (88*4+7)>>4 = 22.
        assert_eq!(row_of(&c, 1, 6), [16, 17, 19, 20, 22, 22]);
        // y = 2: nearest b, next nearest a (above): sums 3b+a = 128 112 72.
        assert_eq!(
            row_of(&c, 2, 6),
            [
                (128 * 4 + 8) >> 4,
                (128 * 3 + 112 + 7) >> 4,
                (112 * 3 + 128 + 8) >> 4,
                (112 * 3 + 72 + 7) >> 4,
                (72 * 3 + 112 + 8) >> 4,
                (72 * 4 + 7) >> 4
            ]
            .map(|v| v as u8)
        );
        // y = 3: nearest b, below is the replicated last row (= b): 4b = 160 128 64.
        assert_eq!(
            row_of(&c, 3, 6),
            [
                (160 * 4 + 8) >> 4,
                (160 * 3 + 128 + 7) >> 4,
                (128 * 3 + 160 + 8) >> 4,
                (128 * 3 + 64 + 7) >> 4,
                (64 * 3 + 128 + 8) >> 4,
                (64 * 4 + 7) >> 4
            ]
            .map(|v| v as u8)
        );
    }

    #[test]
    fn h1v2_fancy_uses_bias_1_above_and_2_below() {
        let c = comp(Upsample::H1V2Fancy, &[&[4, 8], &[12, 20]]);
        // y=0: nearest row0, above = row0: (3*4+4+1)>>2 = 4, (3*8+8+1)>>2 = 8.
        assert_eq!(row_of(&c, 0, 2), [4, 8]);
        // y=1: nearest row0, below row1, bias 2: (12+12+2)>>2 = 6, (24+20+2)>>2 = 11.
        assert_eq!(row_of(&c, 1, 2), [6, 11]);
        // y=2: nearest row1, above row0, bias 1: (36+4+1)>>2 = 10, (60+8+1)>>2 = 17.
        assert_eq!(row_of(&c, 2, 2), [10, 17]);
        // y=3: nearest row1, below = row1, bias 2: (36+12+2)>>2 = 12, (60+20+2)>>2 = 20.
        assert_eq!(row_of(&c, 3, 2), [12, 20]);
    }

    #[test]
    fn narrow_planes_fall_back_to_the_box_filter() {
        // downsampled_width <= 2 never gets the fancy filter.
        assert_eq!(choose_upsampler(1, 1, 2, 1, 2), Some(Upsample::H2V1Box));
        assert_eq!(choose_upsampler(1, 1, 2, 2, 2), Some(Upsample::H2V2Box));
        assert_eq!(choose_upsampler(1, 1, 2, 1, 3), Some(Upsample::H2V1Fancy));
        assert_eq!(choose_upsampler(1, 1, 2, 2, 3), Some(Upsample::H2V2Fancy));
        let c = comp(Upsample::H2V2Box, &[&[7, 9], &[1, 3]]);
        assert_eq!(row_of(&c, 0, 4), [7, 7, 9, 9]);
        assert_eq!(row_of(&c, 1, 4), [7, 7, 9, 9]);
        assert_eq!(row_of(&c, 2, 4), [1, 1, 3, 3]);
    }

    #[test]
    fn upsampler_dispatch_follows_jinit_upsampler() {
        assert_eq!(choose_upsampler(2, 2, 2, 2, 100), Some(Upsample::Full));
        assert_eq!(choose_upsampler(1, 2, 1, 4, 100), Some(Upsample::H1V2Fancy));
        assert_eq!(
            choose_upsampler(1, 1, 4, 1, 100),
            Some(Upsample::Int { h: 4, v: 1 })
        );
        assert_eq!(
            choose_upsampler(1, 1, 3, 3, 100),
            Some(Upsample::Int { h: 3, v: 3 })
        );
        assert_eq!(
            choose_upsampler(2, 1, 3, 1, 100),
            None,
            "a 3:2 ratio is fractional"
        );
        // 4x2 against 2x1 is still a factor-of-two pair after dividing out.
        assert_eq!(choose_upsampler(2, 1, 4, 2, 100), Some(Upsample::H2V2Fancy));
    }

    // ---- Refusals and malformed input ----------------------------------

    /// SOI, then one segment `ff <marker> <len> <payload>`.
    fn stream(segments: &[(u8, &[u8])]) -> Vec<u8> {
        let mut v = vec![0xff, 0xd8];
        for (m, p) in segments {
            v.extend_from_slice(&[0xff, *m]);
            v.extend_from_slice(&((p.len() + 2) as u16).to_be_bytes());
            v.extend_from_slice(p);
        }
        v
    }

    /// SOF payload: precision, height, width, then `(id, sampling, tq)` per component.
    fn sof(precision: u8, h: u16, w: u16, comps: &[(u8, u8, u8)]) -> Vec<u8> {
        let mut p = vec![precision];
        p.extend_from_slice(&h.to_be_bytes());
        p.extend_from_slice(&w.to_be_bytes());
        p.push(comps.len() as u8);
        for c in comps {
            p.extend_from_slice(&[c.0, c.1, c.2]);
        }
        p
    }

    const YCC: [(u8, u8, u8); 3] = [(1, 0x22, 0), (2, 0x11, 1), (3, 0x11, 1)];

    fn refused(bytes: &[u8]) -> Unsupported {
        match decode_jpeg(bytes) {
            Err(JpegError::Unsupported(u)) => u,
            other => panic!("expected a refusal, got {other:?}"),
        }
    }

    #[test]
    fn every_unsupported_variant_is_refused_by_name() {
        let p = sof(8, 16, 16, &YCC);
        assert_eq!(refused(&stream(&[(0xc2, &p)])), Unsupported::Progressive);
        assert_eq!(refused(&stream(&[(0xc3, &p)])), Unsupported::Lossless);
        for m in [0xc9, 0xca, 0xcb] {
            assert_eq!(
                refused(&stream(&[(m, &p)])),
                Unsupported::Arithmetic,
                "SOF marker {m:#x}"
            );
        }
        for m in [0xc5, 0xc6, 0xc7, 0xcd, 0xce, 0xcf, 0xde] {
            assert_eq!(
                refused(&stream(&[(m, &p)])),
                Unsupported::Hierarchical,
                "marker {m:#x}"
            );
        }
        assert_eq!(
            refused(&stream(&[(0xc8, &p)])),
            Unsupported::ReservedMarker(0xc8)
        );
        assert_eq!(
            refused(&stream(&[(0xc0, &sof(12, 16, 16, &YCC))])),
            Unsupported::Precision(12)
        );
        assert_eq!(
            refused(&stream(&[(0xc1, &sof(16, 16, 16, &YCC))])),
            Unsupported::Precision(16)
        );
        let cmyk = [(1, 0x11, 0), (2, 0x11, 0), (3, 0x11, 0), (4, 0x11, 0)];
        assert_eq!(
            refused(&stream(&[(0xc0, &sof(8, 16, 16, &cmyk))])),
            Unsupported::Components(4)
        );
        assert_eq!(
            refused(&stream(&[(0xc0, &sof(8, 16, 16, &cmyk[..2]))])),
            Unsupported::Components(2)
        );
        // Sampling 3x1 against 2x1: not a whole ratio.
        let frac = [(1, 0x31, 0), (2, 0x21, 1), (3, 0x21, 1)];
        assert_eq!(
            refused(&stream(&[(0xc0, &sof(8, 16, 16, &frac))])),
            Unsupported::FractionalSampling
        );
        // Each name has words a person can read.
        assert!(Unsupported::Components(4).to_string().contains("CMYK"));
    }

    fn malformed_kind(bytes: &[u8]) -> JpegError {
        decode_jpeg(bytes).expect_err("must not decode")
    }

    #[test]
    fn malformed_headers_are_errors_not_panics() {
        assert_eq!(malformed_kind(&[]), JpegError::NotJpeg);
        assert_eq!(malformed_kind(&[0xff]), JpegError::NotJpeg);
        assert_eq!(malformed_kind(b"GIF89a"), JpegError::NotJpeg);
        assert!(
            matches!(malformed_kind(&[0xff, 0xd8]), JpegError::Truncated(_)),
            "SOI alone"
        );
        assert!(
            matches!(
                malformed_kind(&[0xff, 0xd8, 0xff, 0xd9]),
                JpegError::Malformed(_)
            ),
            "EOI before any frame"
        );
        // Zero dimensions, zero components.
        assert!(matches!(
            malformed_kind(&stream(&[(0xc0, &sof(8, 0, 16, &YCC))])),
            JpegError::Malformed(_)
        ));
        assert!(matches!(
            malformed_kind(&stream(&[(0xc0, &sof(8, 16, 0, &YCC))])),
            JpegError::Malformed(_)
        ));
        assert!(matches!(
            malformed_kind(&stream(&[(0xc0, &sof(8, 16, 16, &[]))])),
            JpegError::Malformed(_)
        ));
        // Sampling factors out of range (0 and 5), and a length that disagrees.
        for s in [0x00u8, 0x05, 0x50, 0x10, 0x01] {
            let comps = [(1, s, 0), (2, 0x11, 1), (3, 0x11, 1)];
            assert!(
                matches!(
                    malformed_kind(&stream(&[(0xc0, &sof(8, 16, 16, &comps))])),
                    JpegError::Malformed(_)
                ),
                "sampling {s:#x}"
            );
        }
        let mut bad = sof(8, 16, 16, &YCC);
        bad.pop();
        assert!(
            matches!(
                malformed_kind(&stream(&[(0xc0, &bad)])),
                JpegError::Malformed(_)
            ),
            "short SOF"
        );
        // Two frames.
        let p = sof(8, 16, 16, &YCC);
        assert!(matches!(
            malformed_kind(&stream(&[(0xc0, &p), (0xc0, &p)])),
            JpegError::Malformed(_)
        ));
        // A scan before a frame; an unknown marker; a segment longer than the data.
        assert!(matches!(
            malformed_kind(&stream(&[(0xda, &[1, 1, 0, 0, 63, 0])])),
            JpegError::Malformed(_)
        ));
        assert!(matches!(
            malformed_kind(&stream(&[(0x02, &[0, 0])])),
            JpegError::Malformed(_)
        ));
        let mut long = stream(&[(0xe0, &[0u8; 4])]);
        long[5] = 200;
        assert!(matches!(malformed_kind(&long), JpegError::Truncated(_)));
        let mut tiny = stream(&[(0xe0, &[0u8; 4])]);
        tiny[4] = 0;
        tiny[5] = 1;
        assert!(
            matches!(malformed_kind(&tiny), JpegError::Malformed(_)),
            "segment length below 2"
        );
    }

    #[test]
    fn absurd_sizes_are_refused_before_anything_is_allocated() {
        // 65535 x 65535 = 2^32 pixels: over the dimension limit.
        assert!(matches!(
            malformed_kind(&stream(&[(0xc0, &sof(8, 65535, 65535, &YCC))])),
            JpegError::Malformed(_)
        ));
        // 20000 x 20000 = 4e8 > 2^27 within the dimension limit.
        assert_eq!(
            malformed_kind(&stream(&[(0xc0, &sof(8, 20000, 20000, &YCC))])),
            JpegError::TooLarge {
                width: 20000,
                height: 20000
            }
        );
        // A 8000 x 8000 header (6.4e7 pixels, under the cap) over a few bytes of
        // data: the scan header is reached and the image is refused as truncated
        // before any plane exists.
        let mut q = vec![0u8];
        q.extend_from_slice(&[8u8; 64]);
        let mut dht = vec![0x00u8, 0, 1];
        dht.extend_from_slice(&[0; 14]);
        dht.push(0);
        let s = stream(&[
            (0xdb, &q),
            (0xc0, &sof(8, 8000, 8000, &YCC)),
            (0xc4, &dht),
            (0xda, &[3, 1, 0, 2, 0, 3, 0, 0, 63, 0]),
        ]);
        assert!(
            matches!(malformed_kind(&s), JpegError::Truncated(_)),
            "{:?}",
            malformed_kind(&s)
        );
    }

    #[test]
    fn missing_tables_are_errors() {
        let mut dqt = vec![0u8];
        dqt.extend_from_slice(&[1u8; 64]);
        let sos = [3u8, 1, 0, 2, 0x11, 3, 0x11, 0, 63, 0];
        // No DHT at all: the scan is refused (after the plane-size check, so pad the data).
        let mut s = stream(&[
            (0xdb, &dqt),
            (
                0xc0,
                &sof(8, 8, 8, &[(1, 0x11, 0), (2, 0x11, 0), (3, 0x11, 0)]),
            ),
            (0xda, &sos),
        ]);
        s.extend_from_slice(&[0u8; 64]);
        assert!(matches!(malformed_kind(&s), JpegError::Malformed(m) if m.contains("Huffman")));
        // No DQT: the quantisation table is missing.
        let mut s = stream(&[
            (
                0xc0,
                &sof(8, 8, 8, &[(1, 0x11, 0), (2, 0x11, 0), (3, 0x11, 0)]),
            ),
            (0xda, &sos),
        ]);
        s.extend_from_slice(&[0u8; 64]);
        assert!(
            matches!(malformed_kind(&s), JpegError::Malformed(m) if m.contains("quantisation"))
        );
    }

    #[test]
    fn malformed_tables_are_errors() {
        // DHT whose counts run past the segment, a leftover tail, a bad index.
        let mut over = vec![0x00u8];
        over.extend_from_slice(&[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 200]);
        assert!(matches!(
            malformed_kind(&stream(&[(0xc4, &over)])),
            JpegError::Malformed(_)
        ));
        let mut tail = vec![0x00u8, 0, 1];
        tail.extend_from_slice(&[0; 14]);
        tail.push(0); // the one symbol
        tail.push(9); // a stray byte
        assert!(matches!(
            malformed_kind(&stream(&[(0xc4, &tail)])),
            JpegError::Malformed(_)
        ));
        let mut idx = vec![0x04u8];
        idx.extend_from_slice(&[0; 16]);
        assert!(matches!(
            malformed_kind(&stream(&[(0xc4, &idx)])),
            JpegError::Malformed(_)
        ));
        // DQT: a table id of 4, and a table cut short.
        let mut q = vec![4u8];
        q.extend_from_slice(&[1; 64]);
        assert!(matches!(
            malformed_kind(&stream(&[(0xdb, &q)])),
            JpegError::Malformed(_)
        ));
        let mut q = vec![0u8];
        q.extend_from_slice(&[1; 40]);
        assert!(matches!(
            malformed_kind(&stream(&[(0xdb, &q)])),
            JpegError::Malformed(_)
        ));
        // DRI of the wrong size.
        assert!(matches!(
            malformed_kind(&stream(&[(0xdd, &[0, 1, 2])])),
            JpegError::Malformed(_)
        ));
    }

    // ---- A hand-assembled 8x8 grayscale file ----------------------------

    /// A one-block grayscale JPEG: quantiser all 1s, DC table with one symbol
    /// pair, AC table with only end-of-block. `dc_diff` is coded as category
    /// `size` with the magnitude bits `mag`.
    fn gray_8x8(size: u8, mag: u16) -> Vec<u8> {
        // DC table: symbols [size] with code '0' (length 2: '00'), plus a dummy so '00' is legal.
        let mut dht_dc = vec![0x00u8];
        let mut counts = [0u8; 16];
        counts[1] = 2; // two 2-bit codes: '00' and '01'
        dht_dc.extend_from_slice(&counts);
        dht_dc.extend_from_slice(&[size, 0]);
        // AC table: EOB 0x00 as '00', ZRL as '01'.
        let mut dht_ac = vec![0x10u8];
        dht_ac.extend_from_slice(&counts);
        dht_ac.extend_from_slice(&[0x00, 0xf0]);
        let mut dqt = vec![0u8];
        dqt.extend_from_slice(&[1u8; 64]);
        let mut v = stream(&[
            (0xdb, &dqt),
            (0xc0, &sof(8, 8, 8, &[(1, 0x11, 0)])),
            (0xc4, &dht_dc),
            (0xc4, &dht_ac),
            (0xda, &[1, 1, 0x00, 0, 63, 0]),
        ]);
        // Bits: DC code '00', then `size` magnitude bits, then EOB '00'.
        let mut bits = String::from("00");
        for i in (0..size).rev() {
            bits.push(if (mag >> i) & 1 == 1 { '1' } else { '0' });
        }
        bits.push_str("00");
        let padded = format!("{bits:1<16}");
        for chunk in padded.as_bytes().chunks(8) {
            let byte = chunk.iter().fold(0u8, |a, &c| (a << 1) | (c - b'0'));
            v.push(byte);
            if byte == 0xff {
                v.push(0);
            }
        }
        v.extend_from_slice(&[0xff, 0xd9]);
        v
    }

    #[test]
    fn a_hand_assembled_grayscale_block_decodes_to_the_expected_flat_level() {
        // DC diff +16 (size 5, bits 10000): coefficient 16, quantiser 1, so every
        // sample is 128 + 16/8 = 130, replicated to R, G and B.
        let img = decode_jpeg(&gray_8x8(5, 0b10000)).expect("decodes");
        assert_eq!((img.width, img.height), (8, 8));
        assert!(img.data.iter().all(|&v| v == 130), "{:?}", &img.data[..6]);
        // DC diff -16 (size 5, bits 01111): 128 - 2 = 126.
        let img = decode_jpeg(&gray_8x8(5, 0b01111)).expect("decodes");
        assert!(img.data.iter().all(|&v| v == 126));
    }

    #[test]
    fn the_hand_assembled_file_errors_when_cut_or_damaged() {
        let whole = gray_8x8(5, 0b10000);
        assert!(decode_jpeg(&whole).is_ok());
        for cut in 2..whole.len() {
            assert!(
                decode_jpeg(&whole[..cut]).is_err(),
                "prefix of {cut} bytes decoded"
            );
        }
        // Trailing bytes after EOI are not an error (libjpeg stops at EOI).
        let mut more = whole.clone();
        more.extend_from_slice(b"trailing");
        assert!(decode_jpeg(&more).is_ok());
        // A DC size category of 12 against a DC table that allows at most 15 is fine,
        // but 16 in the table is refused.
        let mut bad = gray_8x8(5, 0b10000);
        let at = bad
            .windows(2)
            .position(|w| w == [5, 0])
            .expect("the DC symbols");
        bad[at] = 16;
        assert!(matches!(decode_jpeg(&bad), Err(JpegError::Malformed(_))));
    }
}
