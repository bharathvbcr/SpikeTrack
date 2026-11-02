//! A minimal PNG encoder: 8-bit RGB, no interlace, filter 0 on every row, and
//! the zlib stream made of *stored* (uncompressed) deflate blocks. The files are
//! about as large as the raw pixels; what the encoder buys is that it needs
//! nothing but this file (RFC 2083 chunks with CRC-32, RFC 1950 zlib with
//! Adler-32, RFC 1951 stored blocks).
//!
//! An `sRGB` chunk (rendering intent 0, perceptual) is written so a decoder that
//! colour-manages treats the samples as sRGB and does not apply a gamma
//! conversion of its own.

use std::fmt;

#[derive(Debug, PartialEq, Eq)]
pub enum PngError {
    /// `data.len()` is not `width * height * 3`.
    Size {
        width: usize,
        height: usize,
        len: usize,
    },
    /// Width or height is zero or does not fit the format's 31 bits.
    Dimensions { width: usize, height: usize },
}

impl fmt::Display for PngError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Size { width, height, len } => {
                write!(f, "PNG: {len} bytes is not {width} x {height} x 3")
            }
            Self::Dimensions { width, height } => {
                write!(f, "PNG: {width} x {height} is not a valid size")
            }
        }
    }
}

impl std::error::Error for PngError {}

const SIGNATURE: [u8; 8] = *b"\x89PNG\r\n\x1a\n";
/// The most data bytes one stored deflate block carries (`LEN` is 16 bits).
const STORED_MAX: usize = 65_535;

/// CRC-32 (IEEE 802.3, reflected, polynomial 0xEDB88320) of `bytes`, as PNG
/// chunks use it: the CRC over chunk type and data.
pub fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = u32::MAX;
    for &b in bytes {
        crc ^= u32::from(b);
        for _ in 0..8 {
            crc = if crc & 1 != 0 {
                (crc >> 1) ^ 0xEDB8_8320
            } else {
                crc >> 1
            };
        }
    }
    !crc
}

/// Adler-32 (RFC 1950) of `bytes`.
pub fn adler32(bytes: &[u8]) -> u32 {
    const MOD: u32 = 65_521;
    let (mut a, mut b) = (1u32, 0u32);
    // 5552 is the most bytes that can be added before the sums could overflow a u32.
    for chunk in bytes.chunks(5552) {
        for &x in chunk {
            a += u32::from(x);
            b += a;
        }
        a %= MOD;
        b %= MOD;
    }
    (b << 16) | a
}

fn chunk(out: &mut Vec<u8>, tag: &[u8; 4], data: &[u8]) {
    out.extend_from_slice(&(data.len() as u32).to_be_bytes());
    let start = out.len();
    out.extend_from_slice(tag);
    out.extend_from_slice(data);
    let crc = crc32(&out[start..]);
    out.extend_from_slice(&crc.to_be_bytes());
}

/// A zlib stream holding `raw` in stored blocks.
fn zlib_stored(raw: &[u8]) -> Vec<u8> {
    let blocks = raw.len().div_ceil(STORED_MAX).max(1);
    let mut z = Vec::with_capacity(raw.len() + blocks * 5 + 6);
    z.extend_from_slice(&[0x78, 0x01]); // CMF: deflate, 32 KiB window; FLG: no dictionary, level 0 (0x7801 % 31 == 0)
    let mut chunks = raw.chunks(STORED_MAX).peekable();
    if chunks.peek().is_none() {
        z.extend_from_slice(&[1, 0, 0, 0xff, 0xff]); // one empty final block
    }
    while let Some(c) = chunks.next() {
        let last = chunks.peek().is_none();
        z.push(u8::from(last)); // BFINAL, BTYPE = 00 (stored)
        z.extend_from_slice(&(c.len() as u16).to_le_bytes());
        z.extend_from_slice(&(!(c.len() as u16)).to_le_bytes());
        z.extend_from_slice(c);
    }
    z.extend_from_slice(&adler32(raw).to_be_bytes());
    z
}

/// Encode 8-bit RGB (`width * height * 3` bytes, row-major) as a PNG file.
pub fn encode_rgb(width: usize, height: usize, rgb: &[u8]) -> Result<Vec<u8>, PngError> {
    let max = i32::MAX as usize;
    if width == 0 || height == 0 || width > max || height > max {
        return Err(PngError::Dimensions { width, height });
    }
    if width.checked_mul(height).and_then(|p| p.checked_mul(3)) != Some(rgb.len()) {
        return Err(PngError::Size {
            width,
            height,
            len: rgb.len(),
        });
    }
    let row = width * 3;
    let mut raw = Vec::with_capacity(height * (row + 1));
    for y in 0..height {
        raw.push(0); // filter type 0: None
        raw.extend_from_slice(&rgb[y * row..(y + 1) * row]);
    }
    let mut out = Vec::with_capacity(raw.len() + 100);
    out.extend_from_slice(&SIGNATURE);
    let mut ihdr = Vec::with_capacity(13);
    ihdr.extend_from_slice(&(width as u32).to_be_bytes());
    ihdr.extend_from_slice(&(height as u32).to_be_bytes());
    ihdr.extend_from_slice(&[8, 2, 0, 0, 0]); // 8 bits, truecolour, deflate, adaptive filtering, no interlace
    chunk(&mut out, b"IHDR", &ihdr);
    chunk(&mut out, b"sRGB", &[0]);
    chunk(&mut out, b"IDAT", &zlib_stored(&raw));
    chunk(&mut out, b"IEND", &[]);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use st_video::decode_image;

    #[test]
    fn checksums_match_their_published_check_values() {
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926); // the CRC-32 catalogue check value
        assert_eq!(crc32(b""), 0);
        assert_eq!(adler32(b"Wikipedia"), 0x11E6_0398); // RFC 1950's worked example
        assert_eq!(adler32(b""), 1);
        // The sums are reduced in blocks of 5552: a long run of 0xff must match the definition.
        let long = vec![0xffu8; 100_000];
        let (mut a, mut b) = (1u64, 0u64);
        for &x in &long {
            a = (a + u64::from(x)) % 65_521;
            b = (b + a) % 65_521;
        }
        assert_eq!(adler32(&long), ((b << 16) | a) as u32);
    }

    #[test]
    fn the_empty_chunk_has_the_well_known_iend_crc() {
        let mut out = Vec::new();
        chunk(&mut out, b"IEND", &[]);
        assert_eq!(
            out,
            [0, 0, 0, 0, b'I', b'E', b'N', b'D', 0xae, 0x42, 0x60, 0x82]
        );
    }

    #[test]
    fn stored_blocks_split_at_65535_and_end_with_the_adler_sum() {
        let raw: Vec<u8> = (0..70_000u32).map(|i| (i % 251) as u8).collect();
        let z = zlib_stored(&raw);
        assert_eq!(&z[..2], &[0x78, 0x01]);
        assert_eq!((u16::from(z[0]) << 8 | u16::from(z[1])) % 31, 0);
        // First block: not final, 65535 bytes; second: final, 4465 bytes.
        assert_eq!(z[2], 0);
        assert_eq!(u16::from_le_bytes([z[3], z[4]]), 65_535);
        assert_eq!(u16::from_le_bytes([z[5], z[6]]), 0);
        let second = 2 + 5 + 65_535;
        assert_eq!(z[second], 1);
        assert_eq!(u16::from_le_bytes([z[second + 1], z[second + 2]]), 4_465);
        assert_eq!(z.len(), 2 + 5 + 65_535 + 5 + 4_465 + 4);
        assert_eq!(&z[z.len() - 4..], &adler32(&raw).to_be_bytes());
    }

    #[test]
    fn bad_sizes_are_errors() {
        assert_eq!(
            encode_rgb(2, 2, &[0; 11]),
            Err(PngError::Size {
                width: 2,
                height: 2,
                len: 11
            })
        );
        assert_eq!(
            encode_rgb(0, 2, &[]),
            Err(PngError::Dimensions {
                width: 0,
                height: 2
            })
        );
    }

    #[test]
    fn it_round_trips_through_imageio() {
        // A size whose raw stream spans several stored blocks, with every byte value present.
        let (w, h) = (301usize, 97usize);
        let rgb: Vec<u8> = (0..w * h * 3).map(|i| (i * 7 + i / 255) as u8).collect();
        assert!(h * (w * 3 + 1) > STORED_MAX);
        let png = encode_rgb(w, h, &rgb).expect("encode");
        let dir = std::env::temp_dir().join(format!("spiketrack-png-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("dir");
        let path = dir.join("roundtrip.png");
        std::fs::write(&path, &png).expect("write");
        let back = decode_image(&path).expect("ImageIO decodes our PNG");
        let _ = std::fs::remove_dir_all(&dir);
        assert_eq!((back.width, back.height), (w, h));
        assert!(back.data == rgb, "pixels differ after the round trip");
    }
}
