//! tracker/associate.rs against upstream's own NMS, boundary suppression and
//! association functions on tools/reference/assoc_cases.json
//! (tools/reference/assoc_oracle.py). Exact: every flag and index must agree.

use ojas_io::{parse_json, JsonValue};
use st_model::tracker::associate::{
    associate, nms_iom, not_close_to_boundary, AssocParams, Bitmask,
};

const CASES: &str = include_str!("../../../tools/reference/assoc_cases.json");

fn field<'a>(o: &'a [(String, JsonValue)], k: &str) -> &'a JsonValue {
    &o.iter()
        .find(|(key, _)| key == k)
        .unwrap_or_else(|| panic!("no field {k:?}"))
        .1
}

fn arr<'a>(o: &'a [(String, JsonValue)], k: &str) -> &'a [JsonValue] {
    field(o, k)
        .as_array()
        .unwrap_or_else(|_| panic!("{k} is not an array"))
}

fn bools(o: &[(String, JsonValue)], k: &str) -> Vec<bool> {
    arr(o, k)
        .iter()
        .map(|v| v.as_bool().expect("bool"))
        .collect()
}

fn f32s(o: &[(String, JsonValue)], k: &str) -> Vec<f32> {
    arr(o, k)
        .iter()
        .map(|v| v.as_f64().expect("number") as f32)
        .collect()
}

fn unrle(v: &JsonValue, pixels: usize) -> Bitmask {
    let mut bits = Vec::with_capacity(pixels);
    let mut bit = false;
    for run in v.as_array().expect("rle") {
        bits.extend(std::iter::repeat_n(
            bit,
            run.as_u64().expect("run") as usize,
        ));
        bit = !bit;
    }
    assert_eq!(bits.len(), pixels, "rle length");
    Bitmask::from_fn(pixels, |i| bits[i])
}

fn masks(o: &[(String, JsonValue)], k: &str, pixels: usize) -> Vec<Bitmask> {
    arr(o, k).iter().map(|v| unrle(v, pixels)).collect()
}

#[test]
fn matches_upstream_nms_boundary_and_association() {
    let top = parse_json(CASES).expect("assoc_cases.json");
    let top = top.as_object().expect("object");
    let pixels =
        (field(top, "h").as_u64().expect("h") * field(top, "w").as_u64().expect("w")) as usize;

    let nms = arr(top, "nms");
    for (i, c) in nms.iter().enumerate() {
        let c = c.as_object().expect("case");
        let got = nms_iom(&f32s(c, "probs"), &masks(c, "masks", pixels), 0.4, 0.1);
        assert_eq!(got, bools(c, "keep"), "NMS case {i}");
    }

    let bnd = arr(top, "boundary");
    for (i, c) in bnd.iter().enumerate() {
        let c = c.as_object().expect("case");
        let boxes: Vec<[f32; 4]> = arr(c, "boxes")
            .iter()
            .map(|b| {
                let b: Vec<f32> = b
                    .as_array()
                    .expect("box")
                    .iter()
                    .map(|v| v.as_f64().expect("coord") as f32)
                    .collect();
                [b[0], b[1], b[2], b[3]]
            })
            .collect();
        assert_eq!(
            not_close_to_boundary(&boxes, 0.025),
            bools(c, "keep"),
            "boundary case {i}"
        );
    }

    let assoc = arr(top, "assoc");
    let p = AssocParams::default();
    for (i, c) in assoc.iter().enumerate() {
        let c = c.as_object().expect("case");
        let a = associate(
            &masks(c, "det_masks", pixels),
            &f32s(c, "det_scores"),
            &bools(c, "det_keep"),
            &masks(c, "trk_masks", pixels),
            &p,
        );
        assert_eq!(
            a.trk_is_unmatched,
            bools(c, "trk_is_unmatched"),
            "case {i} trk_is_unmatched"
        );
        assert_eq!(
            a.trk_is_nonempty,
            bools(c, "trk_is_nonempty"),
            "case {i} trk_is_nonempty"
        );
        assert_eq!(a.is_new_det, bools(c, "is_new_det"), "case {i} is_new_det");
        let want: Vec<usize> = arr(c, "det_to_max_trk")
            .iter()
            .map(|v| v.as_u64().expect("index") as usize)
            .collect();
        assert_eq!(a.det_to_max_trk, want, "case {i} det_to_max_trk");
        assert_eq!(
            a.det_is_high_conf,
            bools(c, "det_is_high_conf"),
            "case {i} det_is_high_conf"
        );
        assert_eq!(
            a.det_is_high_overlap,
            bools(c, "det_is_high_overlap"),
            "case {i} det_is_high_overlap"
        );
        assert_eq!(a.im_mask, bools(c, "im_mask"), "case {i} im_mask");
    }
    println!(
        "{} NMS, {} boundary and {} association cases match upstream exactly",
        nms.len(),
        bnd.len(),
        assoc.len()
    );
}
