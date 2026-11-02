//! SAM 3's CLIP byte-level BPE tokenizer (`sam3/model/tokenizer_ve.py`,
//! `SimpleTokenizer` with `clean="lower"`), read from the Hugging Face
//! `vocab.json` and `merges.txt` that ship with `facebook/sam3.1`.
//!
//! Text cleaning follows upstream: `ftfy.fix_text`, `html.unescape` twice,
//! whitespace collapse, lowercase. Of `ftfy.fix_text` this implements the
//! defaults that change well-formed text: curly quotes to straight, Latin
//! ligatures, full-width characters, `\r\n` line breaks and control
//! characters. Mojibake repair is not implemented, and NFC normalization is
//! not either, so text with combining marks is refused rather than tokenized
//! differently from upstream. `html.unescape` covers numeric references and
//! the XML entities; other named entities stay as written.

use std::collections::HashMap;
use std::fmt;
use std::path::Path;

pub const SOT: u32 = 49406;
pub const EOT: u32 = 49407;
pub const CONTEXT: usize = 32;
const SPECIALS: [&str; 2] = ["<start_of_text>", "<end_of_text>"];

#[derive(Debug)]
pub enum TokenizerError {
    Io(String),
    Vocab(String),
    /// Input this tokenizer cannot clean the way upstream does.
    Unsupported(String),
}

impl fmt::Display for TokenizerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(s) => write!(f, "tokenizer files: {s}"),
            Self::Vocab(s) => write!(f, "tokenizer vocabulary: {s}"),
            Self::Unsupported(s) => write!(f, "tokenizer input: {s}"),
        }
    }
}

impl std::error::Error for TokenizerError {}

/// GPT-2/CLIP's byte alphabet: printable bytes map to themselves, the rest to
/// U+0100 onwards, in byte order.
fn bytes_to_unicode() -> [char; 256] {
    let mut table = ['\0'; 256];
    let printable = |b: u32| {
        (0x21..=0x7e).contains(&b) || (0xa1..=0xac).contains(&b) || (0xae..=0xff).contains(&b)
    };
    let mut n = 0;
    for b in 0..256u32 {
        table[b as usize] = if printable(b) {
            char::from_u32(b).expect("byte is a scalar")
        } else {
            n += 1;
            char::from_u32(255 + n).expect("U+0100.. are scalars")
        };
    }
    table
}

pub struct Tokenizer {
    encoder: HashMap<String, u32>,
    ranks: HashMap<(String, String), usize>,
    byte_enc: [char; 256],
}

impl Tokenizer {
    /// `vocab.json` (piece -> id) and `merges.txt` (a `#version` line, then
    /// one merge per line, in rank order).
    pub fn from_hf_files(vocab: &Path, merges: &Path) -> Result<Self, TokenizerError> {
        let io = |p: &Path, e: std::io::Error| TokenizerError::Io(format!("{}: {e}", p.display()));
        let vocab_text = std::fs::read_to_string(vocab).map_err(|e| io(vocab, e))?;
        let merges_text = std::fs::read_to_string(merges).map_err(|e| io(merges, e))?;
        Self::from_hf_text(&vocab_text, &merges_text)
    }

    pub fn from_hf_text(vocab_json: &str, merges_txt: &str) -> Result<Self, TokenizerError> {
        let bad = |s: String| TokenizerError::Vocab(s);
        let v = ojas_io::parse_json(vocab_json)
            .map_err(|e| bad(format!("vocab.json: {}", e.detail())))?;
        let obj = v
            .as_object()
            .map_err(|e| bad(format!("vocab.json: {}", e.detail())))?;
        let mut encoder = HashMap::with_capacity(obj.len());
        for (piece, id) in obj {
            let id = id
                .as_u64()
                .map_err(|e| bad(format!("id of {piece:?}: {}", e.detail())))?;
            let id = u32::try_from(id)
                .map_err(|_| bad(format!("id {id} of {piece:?} overflows u32")))?;
            encoder.insert(piece.clone(), id);
        }
        // Upstream builds these two from its own list; the HF file spells them
        // `<|startoftext|>` / `<|endoftext|>` at the same ids.
        for (name, hf, id) in [
            (SPECIALS[0], "<|startoftext|>", SOT),
            (SPECIALS[1], "<|endoftext|>", EOT),
        ] {
            match encoder.get(hf) {
                Some(&got) if got == id => {
                    encoder.insert(name.to_string(), id);
                }
                other => return Err(bad(format!("{hf} is {other:?}, expected {id}"))),
            }
        }
        let mut ranks = HashMap::new();
        for (rank, line) in merges_txt
            .lines()
            .skip(1)
            .filter(|l| !l.is_empty())
            .enumerate()
        {
            let mut parts = line.split(' ');
            let (Some(a), Some(b), None) = (parts.next(), parts.next(), parts.next()) else {
                return Err(bad(format!("merges.txt line {}: {line:?}", rank + 2)));
            };
            ranks.insert((a.to_string(), b.to_string()), rank);
        }
        Ok(Self {
            encoder,
            ranks,
            byte_enc: bytes_to_unicode(),
        })
    }

    /// `[SOT] + encode(text) + [EOT]`, truncated to `CONTEXT` with EOT kept
    /// last, zero-padded to `CONTEXT`.
    pub fn tokenize(&self, text: &str) -> Result<[u32; CONTEXT], TokenizerError> {
        let mut ids = vec![SOT];
        ids.extend(self.encode(text)?);
        ids.push(EOT);
        if ids.len() > CONTEXT {
            ids.truncate(CONTEXT);
            ids[CONTEXT - 1] = EOT;
        }
        let mut out = [0u32; CONTEXT];
        out[..ids.len()].copy_from_slice(&ids);
        Ok(out)
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>, TokenizerError> {
        let text = clean_lower(text)?;
        let mut out = Vec::new();
        for token in pre_tokenize(&text) {
            if SPECIALS.contains(&token) {
                out.push(self.encoder[token]);
                continue;
            }
            let mapped: String = token.bytes().map(|b| self.byte_enc[b as usize]).collect();
            for piece in self.bpe(&mapped) {
                let id = self
                    .encoder
                    .get(&piece)
                    .ok_or_else(|| TokenizerError::Vocab(format!("no id for piece {piece:?}")))?;
                out.push(*id);
            }
        }
        Ok(out)
    }

    /// Upstream's `bpe`: the last symbol carries `</w>`; repeatedly merge
    /// every occurrence of the lowest-ranked adjacent pair.
    fn bpe(&self, token: &str) -> Vec<String> {
        let chars: Vec<char> = token.chars().collect();
        let mut word: Vec<String> = chars.iter().map(|c| c.to_string()).collect();
        let last = word.len() - 1;
        word[last].push_str("</w>");
        while word.len() > 1 {
            let best = word
                .windows(2)
                .filter_map(|p| {
                    self.ranks
                        .get(&(p[0].clone(), p[1].clone()))
                        .map(|&r| (r, p[0].clone(), p[1].clone()))
                })
                .min_by_key(|t| t.0);
            let Some((_, first, second)) = best else {
                break;
            };
            let mut merged = Vec::with_capacity(word.len());
            let mut i = 0;
            while i < word.len() {
                if i + 1 < word.len() && word[i] == first && word[i + 1] == second {
                    merged.push(format!("{first}{second}"));
                    i += 2;
                } else {
                    merged.push(word[i].clone());
                    i += 1;
                }
            }
            word = merged;
        }
        word
    }
}

/// The subset of `ftfy.fix_text` defaults that alters well-formed text.
fn ftfy_subset(text: &str) -> Result<String, TokenizerError> {
    let mut out = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        let u = c as u32;
        if (0x0300..=0x036f).contains(&u)
            || (0x1ab0..=0x1aff).contains(&u)
            || (0x20d0..=0x20ff).contains(&u)
        {
            return Err(TokenizerError::Unsupported(format!(
                "combining mark U+{u:04X}: upstream NFC-normalizes (ftfy) and this tokenizer does not"
            )));
        }
        match c {
            '\u{2018}' | '\u{2019}' | '\u{201a}' | '\u{201b}' => out.push('\''),
            '\u{201c}' | '\u{201d}' | '\u{201e}' | '\u{201f}' => out.push('"'),
            '\u{fb00}' => out.push_str("ff"),
            '\u{fb01}' => out.push_str("fi"),
            '\u{fb02}' => out.push_str("fl"),
            '\u{fb03}' => out.push_str("ffi"),
            '\u{fb04}' => out.push_str("ffl"),
            '\u{fb05}' | '\u{fb06}' => out.push_str("st"),
            '\u{3000}' => out.push(' '),
            '\u{ff01}'..='\u{ff5e}' => out.push(char::from_u32(u - 0xff01 + 0x21).expect("ASCII")),
            '\r' => {
                if chars.peek() != Some(&'\n') {
                    out.push('\n');
                }
            }
            '\t' | '\n' => out.push(c),
            c if c.is_control() => {}
            c => out.push(c),
        }
    }
    Ok(out)
}

/// `html.unescape` for numeric references and the XML entities.
fn html_unescape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(i) = rest.find('&') {
        out.push_str(&rest[..i]);
        let tail = &rest[i..];
        let named = [
            ("&amp;", '&'),
            ("&lt;", '<'),
            ("&gt;", '>'),
            ("&quot;", '"'),
            ("&apos;", '\''),
            ("&#39;", '\''),
        ];
        if let Some((pat, ch)) = named.iter().find(|(p, _)| tail.starts_with(p)) {
            out.push(*ch);
            rest = &tail[pat.len()..];
            continue;
        }
        let numeric = tail.strip_prefix("&#").and_then(|t| {
            let (radix, digits) = match t.strip_prefix(['x', 'X']) {
                Some(h) => (16, h),
                None => (10, t),
            };
            let end = digits.find(';')?;
            let v = u32::from_str_radix(&digits[..end], radix).ok()?;
            let used = tail.len() - digits.len() + end + 1;
            Some((char::from_u32(v)?, used))
        });
        match numeric {
            Some((ch, used)) => {
                out.push(ch);
                rest = &tail[used..];
            }
            None => {
                out.push('&');
                rest = &tail[1..];
            }
        }
    }
    out.push_str(rest);
    out
}

/// `whitespace_clean(basic_clean(text)).lower()`.
fn clean_lower(text: &str) -> Result<String, TokenizerError> {
    let t = ftfy_subset(text)?;
    let t = html_unescape(&html_unescape(&t));
    let collapsed: Vec<&str> = t.split_whitespace().collect();
    Ok(collapsed.join(" ").to_lowercase())
}

/// Upstream's split regex,
/// `<start_of_text>|<end_of_text>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+`,
/// on already-lowercased text. Letters are `char::is_alphabetic` (a superset
/// of `\p{L}` that differs only on rare marks) and numbers `char::is_numeric`.
fn pre_tokenize(text: &str) -> Vec<&str> {
    let letter = |c: char| c.is_alphabetic();
    let number = |c: char| c.is_numeric();
    let other = |c: char| !c.is_whitespace() && !letter(c) && !number(c);
    let mut out = Vec::new();
    let mut i = 0;
    while i < text.len() {
        let rest = &text[i..];
        let c = rest.chars().next().expect("non-empty");
        if let Some(s) = SPECIALS.iter().find(|s| rest.starts_with(**s)) {
            out.push(&rest[..s.len()]);
            i += s.len();
            continue;
        }
        if c == '\'' {
            if let Some(n) = ["s", "t", "re", "ve", "m", "ll", "d"]
                .iter()
                .find(|s| rest[1..].starts_with(**s))
                .map(|s| 1 + s.len())
            {
                out.push(&rest[..n]);
                i += n;
                continue;
            }
        }
        let run = |pred: &dyn Fn(char) -> bool| {
            rest.char_indices()
                .find(|(_, ch)| !pred(*ch))
                .map_or(rest.len(), |(j, _)| j)
        };
        let n = if letter(c) {
            run(&letter)
        } else if number(c) {
            c.len_utf8()
        } else if other(c) {
            run(&other)
        } else {
            i += c.len_utf8(); // whitespace: matched by nothing
            continue;
        };
        out.push(&rest[..n]);
        i += n;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_like_the_upstream_regex() {
        assert_eq!(
            pre_tokenize("sperm-cell, (pinhead)!"),
            ["sperm", "-", "cell", ",", "(", "pinhead", ")!"]
        );
        assert_eq!(
            pre_tokenize("it's they're 42x"),
            ["it", "'s", "they", "'re", "4", "2", "x"]
        );
        assert_eq!(pre_tokenize("a!'s"), ["a", "!'", "s"]);
    }

    #[test]
    fn cleaning() {
        assert_eq!(
            clean_lower("  White\u{2019}s   SPACE ").unwrap(),
            "white's space"
        );
        assert_eq!(clean_lower("&amp;amp; &#65;").unwrap(), "& a");
        assert!(clean_lower("cafe\u{301}").is_err());
    }

    #[test]
    fn byte_alphabet_is_a_bijection() {
        let t = bytes_to_unicode();
        let mut seen: Vec<char> = t.to_vec();
        seen.sort_unstable();
        seen.dedup();
        assert_eq!(seen.len(), 256);
        assert_eq!(t[b'a' as usize], 'a');
        assert_eq!(t[b' ' as usize], '\u{120}');
    }
}
