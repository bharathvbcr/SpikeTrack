//! JSON output helpers. Reading goes through `ojas_io::parse_json` (limits, typed
//! getters); this is only the writing side: string escaping and finite numbers.

use std::fmt::Write as _;

/// `s` as a JSON string literal (RFC 8259 escaping).
pub fn string(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// A JSON number for a finite `f64`; an error naming `what` for NaN or infinity
/// (JSON has no literal for them, and a quiet `null` would hide the bug).
pub fn number(v: f64, what: &str) -> Result<String, String> {
    if v.is_finite() {
        Ok(format!("{v}"))
    } else {
        Err(format!("{what} is {v}, which JSON cannot represent"))
    }
}

/// A JSON number for a finite `f32`, printed with `f32`'s own shortest
/// round-trip form (not widened to `f64`, which would print 0.8999999761581421
/// for 0.9).
pub fn number32(v: f32, what: &str) -> Result<String, String> {
    if v.is_finite() {
        Ok(format!("{v}"))
    } else {
        Err(format!("{what} is {v}, which JSON cannot represent"))
    }
}

/// `[a, b, c, ...]` of finite numbers.
pub fn number_array(values: &[f64], what: &str) -> Result<String, String> {
    let parts: Result<Vec<String>, String> = values.iter().map(|&v| number(v, what)).collect();
    Ok(format!("[{}]", parts?.join(", ")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ojas_io::parse_json;

    #[test]
    fn strings_are_escaped_and_parse_back() {
        let s = "a \"quoted\" back\\slash\nnew\tline \u{1} é 猫";
        let parsed = parse_json(&string(s)).expect("parse");
        assert_eq!(parsed.as_str().unwrap(), s);
    }

    #[test]
    fn numbers_must_be_finite() {
        assert_eq!(number(0.5, "x").unwrap(), "0.5");
        assert_eq!(number(1e-9, "x").unwrap().parse::<f64>().unwrap(), 1e-9);
        assert!(number(f64::NAN, "score").unwrap_err().contains("score"));
        assert!(number(f64::INFINITY, "x").is_err());
        assert_eq!(number_array(&[1.0, 2.5], "b").unwrap(), "[1, 2.5]");
        assert!(number_array(&[1.0, f64::NAN], "b").is_err());
    }
}
