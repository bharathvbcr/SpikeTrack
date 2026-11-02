//! st-eval HOTA against TrackEval on the synthetic cases in
//! tools/reference/hota_cases.json (tools/reference/hota_oracle.py).

use ojas_io::{parse_json, JsonValue};
use st_eval::hota::{alpha, ALPHAS};
use st_eval::{combine, hota_sequence, Frame, Hota, Xywh};

const CASES: &str = include_str!("../../../tools/reference/hota_cases.json");

fn field<'a>(o: &'a [(String, JsonValue)], k: &str) -> &'a JsonValue {
    &o.iter()
        .find(|(key, _)| key == k)
        .unwrap_or_else(|| panic!("no field {k:?}"))
        .1
}

fn f64s(v: &JsonValue) -> Vec<f64> {
    v.as_array()
        .expect("array")
        .iter()
        .map(|x| x.as_f64().expect("number"))
        .collect()
}

fn boxes(v: &JsonValue) -> Vec<(u64, Xywh)> {
    v.as_array()
        .expect("boxes")
        .iter()
        .map(|b| {
            let b = f64s(b);
            (
                b[0] as u64,
                Xywh {
                    x: b[1],
                    y: b[2],
                    w: b[3],
                    h: b[4],
                },
            )
        })
        .collect()
}

/// Our per-alpha value of a TrackEval field.
fn ours(h: &Hota, name: &str, a: usize) -> f64 {
    match name {
        "HOTA_TP" => h.tp[a],
        "HOTA_FN" => h.fn_[a],
        "HOTA_FP" => h.fp[a],
        "AssA" => h.ass_a[a],
        "AssRe" => h.ass_re[a],
        "AssPr" => h.ass_pr[a],
        "LocA" => h.loc_a[a],
        "HOTA" => h.hota(a),
        "DetA" => h.det_a(a),
        "DetRe" => h.det_re(a),
        "DetPr" => h.det_pr(a),
        other => panic!("unknown field {other}"),
    }
}

fn check(tag: &str, h: &Hota, expected: &[(String, JsonValue)]) -> f64 {
    let mut worst = 0f64;
    for (name, v) in expected {
        let want = f64s(v);
        assert_eq!(want.len(), ALPHAS, "{tag} {name}");
        for (a, w) in want.iter().enumerate() {
            let got = ours(h, name, a);
            let d = (got - w).abs();
            worst = worst.max(d);
            assert!(
                d < 1e-9,
                "{tag} {name} alpha {:.2}: ours {got} TrackEval {w}",
                alpha(a)
            );
        }
    }
    worst
}

#[test]
fn matches_trackeval_per_alpha_on_synthetic_cases() {
    let top = parse_json(CASES).expect("hota_cases.json");
    let top = top.as_object().expect("object");
    let alphas = f64s(field(top, "alphas"));
    for (a, &x) in alphas.iter().enumerate() {
        assert!(
            (alpha(a) - x).abs() < 1e-12,
            "alpha {a}: {} vs TrackEval {x}",
            alpha(a)
        );
    }
    let mut seqs = Vec::new();
    for (i, case) in field(top, "cases")
        .as_array()
        .expect("cases")
        .iter()
        .enumerate()
    {
        let case = case.as_object().expect("case");
        let frames: Vec<Frame<u64>> = field(case, "frames")
            .as_array()
            .expect("frames")
            .iter()
            .map(|f| {
                let f = f.as_object().expect("frame");
                Frame {
                    gt: boxes(field(f, "gt")),
                    pred: boxes(field(f, "pred")),
                }
            })
            .collect();
        let h = hota_sequence(&frames);
        let worst = check(
            &format!("case {i}"),
            &h,
            field(case, "expected").as_object().expect("expected"),
        );
        println!(
            "case {i}: HOTA {:.4} (worst |d| {worst:.2e})",
            h.summary().hota * 100.0
        );
        seqs.push(h);
    }
    let comb = combine(&seqs);
    check(
        "combined",
        &comb,
        field(top, "combined").as_object().expect("combined"),
    );
    println!("combined HOTA {:.4}", comb.summary().hota * 100.0);
}
