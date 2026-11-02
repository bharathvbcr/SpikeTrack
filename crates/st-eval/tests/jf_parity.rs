//! st-eval J and F against davis2017-evaluation on tools/reference/jf_cases.json
//! (tools/reference/jf_oracle.py).

use ojas_io::{parse_json, JsonValue};
use st_eval::jf::{boundary_f, region_j, Mask, DAVIS_BOUND_TH};

const CASES: &str = include_str!("../../../tools/reference/jf_cases.json");

fn field<'a>(o: &'a [(String, JsonValue)], k: &str) -> &'a JsonValue {
    &o.iter()
        .find(|(key, _)| key == k)
        .unwrap_or_else(|| panic!("no field {k:?}"))
        .1
}

fn unrle(v: &JsonValue, w: usize, h: usize) -> Mask {
    let mut data = Vec::with_capacity(w * h);
    let mut bit = false;
    for run in v.as_array().expect("rle") {
        let n = run.as_u64().expect("run") as usize;
        data.extend(std::iter::repeat_n(bit, n));
        bit = !bit;
    }
    Mask::new(w, h, data)
}

#[test]
fn matches_davis2017_evaluation() {
    let top = parse_json(CASES).expect("jf_cases.json");
    let cases = field(top.as_object().expect("object"), "cases")
        .as_array()
        .expect("cases");
    let (mut worst_j, mut worst_f) = (0f64, 0f64);
    for (i, c) in cases.iter().enumerate() {
        let c = c.as_object().expect("case");
        let (h, w) = (
            field(c, "h").as_u64().unwrap() as usize,
            field(c, "w").as_u64().unwrap() as usize,
        );
        let gt = unrle(field(c, "gt"), w, h);
        let seg = unrle(field(c, "seg"), w, h);
        let void = match field(c, "void") {
            JsonValue::Null => None,
            v => Some(unrle(v, w, h)),
        };
        let j = region_j(&gt, &seg, void.as_ref());
        let f = boundary_f(&seg, &gt, void.as_ref(), DAVIS_BOUND_TH);
        let (wj, wf) = (
            field(c, "j").as_f64().unwrap(),
            field(c, "f").as_f64().unwrap(),
        );
        worst_j = worst_j.max((j - wj).abs());
        worst_f = worst_f.max((f - wf).abs());
        assert!(
            (j - wj).abs() < 1e-12,
            "case {i} ({w}x{h}): J {j} vs DAVIS {wj}"
        );
        assert!(
            (f - wf).abs() < 1e-12,
            "case {i} ({w}x{h}): F {f} vs DAVIS {wf}"
        );
    }
    println!(
        "{} cases, worst |dJ| {worst_j:.2e}, worst |dF| {worst_f:.2e}",
        cases.len()
    );
}
