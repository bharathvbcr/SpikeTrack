//! tracker/bookkeeping.rs against upstream's own planning phase on
//! tools/reference/bookkeeping_cases.json (tools/reference/bookkeeping_oracle.py), which runs
//! `run_tracker_update_planning_phase` and the `propagate_in_video` yield loop unmodified over
//! synthetic multi-frame sequences. Exact: every id, index, flag, counter and hidden-id list of
//! every frame must agree.
//!
//! Each frame is fed twice over: the `Association` upstream computed is what `plan_frame`
//! consumes (so this test isolates the bookkeeping), and `associate()` over the recorded masks
//! must reproduce that same `Association` (so the two ports are chained end to end).

use std::collections::BTreeMap;

use ojas_io::{parse_json, JsonValue};
use st_model::tracker::associate::{associate, Association, AssocParams, Bitmask};
use st_model::tracker::bookkeeping::{select_births, sigmoid_f32, suppress_by_recent_occlusion, BookParams, FrameInput, HideLog, TrackBook, YieldBuffer};

const CASES: &str = include_str!("../../../tools/reference/bookkeeping_cases.json");

type Obj = [(String, JsonValue)];

fn field<'a>(o: &'a Obj, k: &str) -> &'a JsonValue {
    &o.iter().find(|(key, _)| key == k).unwrap_or_else(|| panic!("no field {k:?}")).1
}

fn obj<'a>(o: &'a Obj, k: &str) -> &'a Obj {
    field(o, k).as_object().unwrap_or_else(|_| panic!("{k} is not an object"))
}

fn arr<'a>(o: &'a Obj, k: &str) -> &'a [JsonValue] {
    field(o, k).as_array().unwrap_or_else(|_| panic!("{k} is not an array"))
}

fn bools(o: &Obj, k: &str) -> Vec<bool> {
    arr(o, k).iter().map(|v| v.as_bool().expect("bool")).collect()
}

fn f32s(o: &Obj, k: &str) -> Vec<f32> {
    arr(o, k).iter().map(|v| v.as_f64().expect("number") as f32).collect()
}

fn i64s(o: &Obj, k: &str) -> Vec<i64> {
    arr(o, k).iter().map(|v| v.as_i64().expect("integer")).collect()
}

fn u32s(o: &Obj, k: &str) -> Vec<u32> {
    arr(o, k).iter().map(|v| v.as_u64().expect("count") as u32).collect()
}

fn usizes(o: &Obj, k: &str) -> Vec<usize> {
    arr(o, k).iter().map(|v| v.as_u64().expect("index") as usize).collect()
}

/// `[[a, b], ...]` with an integer first and second element.
fn pairs(o: &Obj, k: &str) -> Vec<(i64, i64)> {
    arr(o, k)
        .iter()
        .map(|p| {
            let p = p.as_array().expect("pair");
            (p[0].as_i64().expect("first"), p[1].as_i64().expect("second"))
        })
        .collect()
}

fn unrle(v: &JsonValue, pixels: usize) -> Bitmask {
    let mut bits = Vec::with_capacity(pixels);
    let mut bit = false;
    for run in v.as_array().expect("rle") {
        bits.extend(std::iter::repeat_n(bit, run.as_u64().expect("run") as usize));
        bit = !bit;
    }
    assert_eq!(bits.len(), pixels, "rle length");
    Bitmask::from_fn(pixels, |i| bits[i])
}

fn masks(o: &Obj, k: &str, pixels: usize) -> Vec<Bitmask> {
    arr(o, k).iter().map(|v| unrle(v, pixels)).collect()
}

/// The `Association` upstream recorded. Upstream's "no tracks" answer is -1 per detection where
/// associate.rs says `usize::MAX`.
fn recorded_association(a: &Obj) -> Association {
    Association {
        trk_is_unmatched: bools(a, "trk_is_unmatched"),
        trk_is_nonempty: bools(a, "trk_is_nonempty"),
        is_new_det: bools(a, "is_new_det"),
        det_to_max_trk: i64s(a, "det_to_max_trk").into_iter().map(|v| if v < 0 { usize::MAX } else { v as usize }).collect(),
        det_is_high_conf: bools(a, "det_is_high_conf"),
        det_is_high_overlap: bools(a, "det_is_high_overlap"),
        im_mask: bools(a, "im_mask"),
    }
}

#[test]
fn sigmoid_gate_matches_torch_around_the_threshold() {
    let top = parse_json(CASES).expect("bookkeeping_cases.json");
    let probe = obj(top.as_object().expect("object"), "sigmoid_probe");
    let xs = f32s(probe, "logits");
    let contexts = obj(probe, "gt");
    assert!(xs.len() > 50 && !contexts.is_empty());
    for (len, expected) in contexts {
        let expected: Vec<bool> = expected.as_array().expect("array").iter().map(|v| v.as_bool().expect("bool")).collect();
        let got: Vec<bool> = xs.iter().map(|&x| sigmoid_f32(x) > 0.8).collect();
        assert_eq!(got, expected, "sigmoid(x) > 0.8 over tensors of length {len}");
    }
}

#[test]
fn occlusion_iou_threshold_matches_torch_on_exact_ratios() {
    let top = parse_json(CASES).expect("bookkeeping_cases.json");
    let top = top.as_object().expect("object");
    let pixels = (field(top, "h").as_u64().expect("h") * field(top, "w").as_u64().expect("w")) as usize;
    let cases = arr(top, "iou_cases");
    let at_threshold = cases.iter().filter(|c| field(c.as_object().expect("case"), "b").as_u64().expect("b") % 7 == 0).count();
    assert!(cases.len() > 100 && at_threshold >= 40);
    for (i, c) in cases.iter().enumerate() {
        let c = c.as_object().expect("case");
        // Mask a covers the first `a` pixels, mask b the first `b`: IoU = b / a.
        let (a, b) = (field(c, "a").as_u64().expect("a") as usize, field(c, "b").as_u64().expect("b") as usize);
        let masks = [Bitmask::from_fn(pixels, |p| p < a), Bitmask::from_fn(pixels, |p| p < b)];
        assert_eq!(suppress_by_recent_occlusion(&masks, &i64s(c, "last"), 0.7), bools(c, "suppress"), "iou case {i}: {b}/{a}");
    }
}

#[test]
fn birth_selection_matches_upstream_cap_order_and_ids() {
    let top = parse_json(CASES).expect("bookkeeping_cases.json");
    let cases = arr(top.as_object().expect("object"), "birth_cases");
    let (mut capped, mut tied) = (0, 0);
    for (i, c) in cases.iter().enumerate() {
        let c = c.as_object().expect("case");
        let new_inds = usizes(c, "new_inds");
        let scores = f32s(c, "scores");
        let prev = field(c, "prev").as_u64().expect("prev") as usize;
        let (births, dropped) = select_births(&new_inds, &scores, prev, field(c, "max_obj_id").as_i64().expect("max id"), BookParams::default().max_num_objects);
        let got: Vec<(usize, i64)> = births.iter().map(|b| (b.det, b.id)).collect();
        let want: Vec<(usize, i64)> = pairs(c, "births").into_iter().map(|(d, id)| (d as usize, id)).collect();
        assert_eq!(got, want, "birth case {i}");
        assert_eq!(dropped as u64, field(c, "dropped").as_u64().expect("dropped"), "birth case {i}: dropped");
        if dropped > 0 {
            capped += 1;
            let mut seen: Vec<f32> = new_inds.iter().map(|&d| scores[d]).collect();
            seen.sort_by(f32::total_cmp);
            seen.dedup();
            tied += usize::from(seen.len() < new_inds.len());
        }
    }
    assert!(capped >= 20 && tied >= 5, "{capped} capped cases, {tied} with tied scores");
}

#[test]
fn matches_upstream_planning_phase_and_yield_hiding() {
    let top = parse_json(CASES).expect("bookkeeping_cases.json");
    let top = top.as_object().expect("object");
    let pixels = (field(top, "h").as_u64().expect("h") * field(top, "w").as_u64().expect("w")) as usize;
    let sequences = arr(top, "sequences");
    let (mut frames_seen, mut births, mut removals, mut suppressions, mut reconditions, mut caps) = (0, 0, 0, 0, 0, 0);

    for seq in sequences {
        let seq = seq.as_object().expect("sequence");
        let name = field(seq, "name").as_str().expect("name").to_string();
        let start = field(seq, "start").as_i64().expect("start");
        let last = field(seq, "last").as_i64().expect("last");
        let num_frames = field(seq, "num_frames").as_i64().expect("num_frames");
        let mut book = TrackBook::new(BookParams::default());
        let mut log = HideLog::new(BookParams::default().confirm_consecutive, num_frames);
        let mut buffer = YieldBuffer::new(BookParams::default().hotstart_delay as usize);
        let mut released: Vec<(i64, i64, Option<Vec<i64>>)> = Vec::new();

        for fr in arr(seq, "frames") {
            let fr = fr.as_object().expect("frame");
            let f = field(fr, "f").as_i64().expect("f");
            let at = |what: &str| format!("{name} frame {f}: {what}");
            assert_eq!(book.obj_ids(), i64s(fr, "prev_ids").as_slice(), "{}", at("previous ids"));

            let det_masks = masks(fr, "det_masks", pixels);
            let det_scores = f32s(fr, "det_scores");
            let det_keep = bools(fr, "det_keep");
            let trk_masks = masks(fr, "trk_masks", pixels);
            let recorded = recorded_association(obj(fr, "assoc"));
            let derived = associate(&det_masks, &det_scores, &det_keep, &trk_masks, &AssocParams::default());
            assert_eq!(derived, recorded, "{}", at("associate() vs upstream's association"));

            let plan = book.plan_frame(&FrameInput {
                frame_idx: f,
                association: &recorded,
                det_keep: &det_keep,
                det_scores: &det_scores,
                det_masks: &det_masks,
                trk_masks: &trk_masks,
                trk_score_logits: &f32s(fr, "trk_score_logits"),
            });

            let out = obj(fr, "out");
            let want_births: Vec<(usize, i64)> = pairs(out, "births").into_iter().map(|(d, id)| (d as usize, id)).collect();
            let got_births: Vec<(usize, i64)> = plan.births.iter().map(|b| (b.det, b.id)).collect();
            assert_eq!(got_births, want_births, "{}", at("births"));
            assert_eq!(plan.realized.new_det_fa_inds, usizes(out, "new_det_fa_inds_precap"), "{}", at("new-detection candidates"));
            assert_eq!(plan.num_dropped_due_to_limit as u64, field(out, "dropped").as_u64().expect("dropped"), "{}", at("dropped by the cap"));
            assert_eq!(plan.removed_ids, i64s(out, "removed_ids"), "{}", at("removed ids"));
            assert_eq!(plan.realized.unmatched_trk_ids, i64s(out, "unmatched_ids"), "{}", at("unmatched ids"));
            assert_eq!(plan.realized.empty_trk_ids, i64s(out, "empty_ids"), "{}", at("empty ids"));
            let want_matched: Vec<(usize, Vec<i64>)> = arr(out, "det_to_matched")
                .iter()
                .map(|p| {
                    let p = p.as_array().expect("pair");
                    (p[0].as_u64().expect("det") as usize, p[1].as_array().expect("ids").iter().map(|v| v.as_i64().expect("id")).collect())
                })
                .collect();
            assert_eq!(plan.realized.det_to_matched_trk_ids, want_matched, "{}", at("detection -> matched track ids"));
            let want_pairs: Vec<(i64, usize)> = pairs(out, "trk_to_det").into_iter().map(|(id, d)| (id, d as usize)).collect();
            assert_eq!(plan.realized.trk_id_to_max_iou_high_conf_det, want_pairs, "{}", at("track -> high-confidence detection"));
            let mut got_recond = plan.recondition.clone();
            got_recond.sort();
            let want_recond: Vec<(i64, usize)> = pairs(out, "recondition").into_iter().map(|(id, d)| (id, d as usize)).collect();
            assert_eq!(got_recond, want_recond, "{}", at("reconditioned tracks"));
            assert_eq!(plan.suppressed, usizes(out, "suppressed"), "{}", at("occlusion-suppressed rows"));
            assert_eq!(plan.obj_ids, i64s(out, "obj_ids"), "{}", at("updated ids"));
            assert_eq!(plan.unconfirmed_ids, i64s(out, "unconfirmed"), "{}", at("unconfirmed ids"));

            let meta = obj(fr, "meta");
            assert_eq!(book.max_obj_id(), field(meta, "max_obj_id").as_i64().expect("max id"), "{}", at("max_obj_id"));
            let want_scores: BTreeMap<i64, f32> = arr(meta, "scores")
                .iter()
                .map(|p| {
                    let p = p.as_array().expect("pair");
                    (p[0].as_i64().expect("id"), p[1].as_f64().expect("score") as f32)
                })
                .collect();
            assert_eq!(book.scores(), &want_scores, "{}", at("obj_id_to_score"));
            assert_eq!(book.first_frame(), i64s(meta, "first_frame").as_slice(), "{}", at("obj_first_frame"));
            assert_eq!(book.unmatch_count(), u32s(meta, "unmatch_count").as_slice(), "{}", at("unmatch counts"));
            assert_eq!(book.overlap_pair_counts(), u32s(meta, "overlap").as_slice(), "{}", at("overlap pair counts"));
            assert_eq!(book.last_occluded(), i64s(meta, "last_occluded").as_slice(), "{}", at("last occluded"));
            assert_eq!(book.consecutive_det(), u32s(meta, "consecutive_det").as_slice(), "{}", at("consecutive detections"));
            assert_eq!(book.confirmed(), bools(meta, "confirmed").as_slice(), "{}", at("confirmed"));

            frames_seen += 1;
            births += plan.births.len();
            removals += plan.removed_ids.len();
            suppressions += plan.suppressed.len();
            reconditions += plan.recondition.len();
            caps += usize::from(plan.num_dropped_due_to_limit > 0);

            // The yield loop, as of the moment frame `f` has been processed.
            log.record(f, &plan.unconfirmed_ids);
            for y in buffer.push(f, f == last) {
                released.push((y, f - start + 1, log.hidden_for(y).map(<[i64]>::to_vec)));
            }
        }

        let want: Vec<(i64, i64, Option<Vec<i64>>)> = arr(seq, "yields")
            .iter()
            .map(|y| {
                let y = y.as_object().expect("yield");
                let hide = field(y, "hide");
                (
                    field(y, "frame").as_i64().expect("frame"),
                    field(y, "after").as_i64().expect("after"),
                    if hide.is_null() { None } else { Some(hide.as_array().expect("ids").iter().map(|v| v.as_i64().expect("id")).collect()) },
                )
            })
            .collect();
        assert_eq!(released, want, "{name}: yielded frames, release time and hidden ids");
    }

    // The sequences must actually exercise the rules, or a pass means little.
    assert!(sequences.len() >= 30 && frames_seen >= 600, "{} sequences, {frames_seen} frames", sequences.len());
    assert!(births > 100 && removals > 0 && suppressions > 0 && reconditions > 0 && caps > 0, "births {births} removals {removals} suppressions {suppressions} reconditions {reconditions} caps {caps}");
    println!("{} sequences, {frames_seen} frames match upstream exactly ({births} births, {removals} removals, {suppressions} suppressions, {reconditions} reconditions, {caps} capped frames)", sequences.len());
}
