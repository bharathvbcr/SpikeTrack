//! SAM 3.1 Object Multiplex's text-prompted video tracking session: the detector, the
//! multiplex tracker and the per-frame bookkeeping, driven one preprocessed frame at a time,
//! as upstream's builder path runs them (`build_sam3_multiplex_video_predictor`:
//! `Sam3MultiplexTrackingWithInteractivity` over `Sam3MultiplexBase`, with
//! `Sam3VideoTrackingMultiplexDemo` as the tracker). The text prompt is given on frame 0 and
//! tracking runs forward only (`add_prompt(frame_index=0)` then `propagate_in_video` from 0);
//! a prompt on a later frame and the reverse leg are not ported. One device, world size 1.
//!
//! The per-frame order (`_det_track_one_frame_impl`, `sam3_multiplex_base.py:477-642`; file:line
//! cites are to the CPU-patched checkout) as [`Session::push_frame`] runs it:
//! 1. detect (`run_backbone_and_detection` `:646-809`): trunk, detector neck, detector with the
//!    prompt; `prob = sigmoid(pred_logits)` where `pred_logits` already carries the presence
//!    score (`supervise_joint_box_scores=True`, `sam3_image.py:345-353`); mask-IoM NMS over
//!    `prob > 0.4` at IoM `> 0.1` (`sam3_multiplex_detector.py:828-848`, `_nms_masks_core_batched`)
//!    and `pred_logits -= 1e4` for every query it did not keep; keep = `prob > 0.4` and box
//!    centre not within 0.025 of an edge (`:744-749`);
//!    (Both tracker necks, interactive and propagation, are rounded through bf16 right after the
//!    neck, `sam3_multiplex_detector.py:563-586`; everything downstream computes in f32.)
//! 2. propagate every tracker state one frame without memory encoding (`run_tracker_propagation`
//!    `:811-863`, `_propogate_tracker_one_frame_local_gpu` `:1731-1889`); rows concatenated and
//!    sorted by object id;
//! 3. move the kept detections first (`:545-552`; stable here, see Ties);
//! 4. plan (`run_tracker_update_planning_phase` `:1018-1420`), exactly [`TrackBook::plan_frame`]
//!    over [`associate`]: association, hot-start removal, the reconditioning picks, occlusion
//!    suppression, births, confirmation. The decisions only need binary masks, so the plan is
//!    made first and its model work is carried out in upstream's order:
//!    a. recondition (`_recondition_masklets` `:867-1008`, every 16th frame): the track's mask
//!       becomes `where((det > 0) == (trk > 0), trk, det)`; the states holding the objects get
//!       `add_new_masks(reconditioning=True)` on the detector mask at 1152 and a preflight;
//!    b. suppressed tracks' masks are overwritten with -10 (`:1510-1512`);
//!    c. memory update (`_tracker_update_memories` `:2496-2666`): masks to 1152 (bilinear), the
//!       area-shrinkage suppression (`_suppress_object_pw_area_shrinkage` `VD:3324-3368`), the
//!       +-10 area-proxy scores, then each state encodes its own objects (found by id, the
//!       dynamic-multiplex routing of `:2535-2580`);
//! 5. execute (`run_tracker_update_execution_phase` `sam3_video_base.py:1176-1223`): births go to
//!    the state with the fewest empty slots that still fits them, else a new state
//!    (`_tracker_add_new_objects` `:2668-2798`: detector 288 logits -> bilinear 1152 -> `> 0`,
//!    `add_new_masks`, preflight); then hot-start removals leave every state
//!    (`_tracker_remove_objects` `:2800-2822`; a state left empty is dropped);
//! 6. outputs (`build_outputs` `:1544-1631`): previous ids from the final (reconditioned,
//!    suppressed) track masks, new ids from the detector masks, each bilinear to the video size
//!    and `> 0`.
//!
//! The prompt frame is visited twice (`add_prompt` runs the whole step once, then
//! `propagate_in_video` starts at the same frame): both visits run here on the same trunk and
//! detections (upstream detects the second one in a 16-frame batch, so its detector numerics
//! differ slightly; ours are identical by construction), and only the second visit feeds the
//! yield buffer and the hide log, as upstream's `add_prompt` output is not part of the stream.
//!
//! Output (`propagate_in_video` `sam3_multiplex_tracking.py:302-583`, `_postprocess_output`
//! `:668-797`): a frame is released when [`YieldBuffer`] holds `hotstart_delay` (15) frames, or
//! at the last frame, hiding the ids [`HideLog`] lists as unconfirmed two frames later; the ids
//! that are hidden or have an empty mask are dropped, boxes come from the remaining masks
//! (inclusive pixel extents, `xywh` with `w = x1 - x0`, normalized by the video size), and the
//! masks are then made disjoint by the tracker's score (winner takes the overlap, lowest id on
//! ties). Probabilities are the ids' birth scores (`-1e4` once removed).
//!
//! Ties (stable and documented, upstream leaves them unspecified): detections that are kept
//! come first in query order, then the others in query order (`argsort` on a bool tensor);
//! `argmax` over detections, tracks, masks and objects takes the first index.
//!
//! Dead on this path, therefore not ported: the hot-start `removed`/`suppressed` hide sets
//! (always empty, see `bookkeeping.rs`), `trk_keep_alive`, `reapply_no_object_pointer`,
//! hole filling (`fill_hole_area=0`), the detector's 16-frame chunking.

use std::collections::BTreeMap;

use ojas_core::{Backend, Tensor};

use crate::detector::{Detector, DetectorTaps, D as DET_D, QUERIES};
use crate::host;
use crate::layers::{host_of, up};
use crate::text::Text;
use crate::tokenizer::Tokenizer;
use crate::tracker::associate::{associate, nms_iom, not_close_to_boundary, track_detections, AssocParams, Association, Bitmask};
use crate::tracker::bookkeeping::{BookParams, FrameInput, FramePlan, HideLog, TrackBook, YieldBuffer};
use crate::tracker::mask_decoder::MUX;
use crate::tracker::state::{err, FrameCtx, TrackerState};
use crate::tracker::vos::Vos;
use crate::{ModelError, Neck, Taps};

/// The builder's session parameters (`model_builder.py:1170-1202`).
#[derive(Clone, Debug)]
pub struct SessionConfig {
    /// Frames of the video (upstream's `inference_state["num_frames"]`): sets the object-pointer
    /// window `min(num_frames, 16)`, the look-ahead clamp and the last-frame flush.
    pub num_frames: usize,
    /// Video `(height, width)` the output masks are produced at.
    pub video_hw: (usize, usize),
    pub book: BookParams,
    pub assoc: AssocParams,
    /// `score_threshold_detection`.
    pub score_threshold: f32,
    /// `det_nms_thresh` (mask IoM).
    pub nms_iom: f32,
    /// `_suppress_detections_close_to_boundary` margin.
    pub boundary_margin: f32,
}

impl SessionConfig {
    pub fn new(num_frames: usize, height: usize, width: usize) -> Self {
        Self { num_frames, video_hw: (height, width), book: BookParams::default(), assoc: AssocParams::default(), score_threshold: 0.4, nms_iom: 0.1, boundary_margin: 0.025 }
    }
}

/// The detector's raw output for one frame, host side.
pub struct Detections {
    /// `[200]` joint class-and-presence logits, before NMS.
    pub logits: Vec<f32>,
    /// `[200, 4]` boxes, centre x, centre y, width, height in `[0, 1]`.
    pub boxes_cxcywh: Vec<f32>,
    /// `[200, 288 * 288]` mask logits.
    pub masks: Vec<f32>,
}

/// One released frame (`out_obj_ids`, `out_probs`, `out_boxes_xywh`, `out_binary_masks`).
pub struct YieldedFrame {
    pub frame: usize,
    /// Ascending.
    pub obj_ids: Vec<i64>,
    pub probs: Vec<f32>,
    /// `[n, 4]` normalized `x, y, w, h`.
    pub boxes_xywh: Vec<f32>,
    /// One `height * width` mask per id.
    pub masks: Vec<Bitmask>,
    /// Not upstream: per id, the normalized `x0, y0, x1, y1` box of the detection this frame
    /// assigned to it (a newborn's own detection; for a track, [`track_detections`]), or `None`.
    /// What `spiketrack track --boxes detector` reports in place of the mask box.
    pub det_boxes_xyxy: Vec<Option<[f32; 4]>>,
}

/// A tracker state after a visit: what it stored for the visit's frame.
pub struct StateTrace {
    pub obj_ids: Vec<i64>,
    pub slots: Vec<i32>,
    pub maskmem: Option<Vec<f32>>,
    pub obj_ptr: Vec<f32>,
    pub cond_objs: Vec<usize>,
}

/// Everything one visit of `_det_track_one_frame_impl` decided and produced, for parity checks.
/// Recorded only after [`Session::set_trace`]`(true)`.
pub struct VisitTrace {
    pub frame: usize,
    pub visit: usize,
    /// Detector queries, query order: probabilities after NMS, the keep flags, boxes as xyxy.
    pub det_probs: Vec<f32>,
    pub det_keep: Vec<bool>,
    pub det_boxes_xyxy: Vec<f32>,
    /// `det_order[k]` is the query at position `k` after the kept-first reorder; association
    /// and plan indices are in that space.
    pub det_order: Vec<usize>,
    /// Mask logits of the kept queries in query order, `[kept, 288 * 288]`.
    pub det_kept_masks: Vec<f32>,
    /// Propagated tracks before any change: ids, `[n, 288 * 288]` logits, score logits.
    pub prop_ids: Vec<i64>,
    pub prop_masks: Vec<f32>,
    pub prop_scores: Vec<f32>,
    pub association: Association,
    pub plan: FramePlan,
    /// The masks handed to the memory update (after recondition and suppression) and the
    /// area-proxy scores it derived. Empty when no track existed.
    pub mem_masks: Vec<f32>,
    pub mem_scores: Vec<f32>,
    pub states: Vec<StateTrace>,
    /// `build_outputs`: id and video-size mask, previous ids first, then newborns.
    pub outputs: Vec<(i64, Bitmask)>,
    /// The tracker score per id of the frame (`obj_id_to_sam2_score_frame_wise`).
    pub sam2_scores: Vec<(i64, f32)>,
}

/// A frame waiting in the yield buffer.
struct Buffered {
    frame: usize,
    /// Ascending ids with their mask, birth score, tracker score and assigned detection box.
    items: Vec<(i64, Bitmask, f32, f32, Option<[f32; 4]>)>,
}

struct VisitOut {
    buffered: Buffered,
    unconfirmed: Vec<i64>,
}

/// Detections after NMS, the boundary filter and the kept-first reorder.
struct DetStage {
    /// After NMS, query order.
    probs: Vec<f32>,
    /// `pos_pred_mask`, query order.
    keep: Vec<bool>,
    boxes_xyxy: Vec<f32>,
    order: Vec<usize>,
    /// Reordered.
    bits: Vec<Bitmask>,
    scores: Vec<f32>,
    keep_re: Vec<bool>,
}

pub struct Session<'a, B: Backend + ?Sized> {
    vos: &'a Vos<'a, B>,
    detector: &'a Detector<'a, B>,
    /// The prompt's valid text tokens `[1, L, 256]`.
    prompt: Tensor,
    cfg: SessionConfig,
    states: Vec<TrackerState>,
    book: TrackBook,
    next_frame: usize,
    yield_buf: YieldBuffer,
    hide: HideLog,
    buffered: BTreeMap<i64, Buffered>,
    trace: bool,
    traces: Vec<VisitTrace>,
    marker: Option<StageMarker<'a>>,
}

/// Called with a fixed stage name as each phase of a frame begins (see [`Session::set_stage_marker`]).
pub type StageMarker<'a> = Box<dyn FnMut(&str) + 'a>;

/// The stage names [`Session::push_frame`] marks, in the order a frame meets them. A state's
/// propagation is marked once per tracker state, as `tracker propagate, state <i>`, then per phase
/// as `tracker propagate, state <i>: <phase>` (`state::PROPAGATE_STAGES`); between the detector
/// neck and its readback the detector marks its own (`detector::DETECTOR_STAGES`).
pub const STAGES: [&str; 12] = [
    "trunk",
    "detector neck",
    "detector mask readback",
    "propagation neck",
    "tracker propagate",
    "detections + association (host)",
    "interactive neck",
    "recondition",
    "memory encode: upsample + readback + host area rule",
    "memory encode: encode",
    "births",
    "outputs + release",
];

/// The memory update's host-facing parts, marked inside `STAGES[8]` as `<STAGES[8]>: <part>`:
/// `STAGES[8]` itself carries the upload of the final low-res masks and the device upsample to
/// the input-mask side, then the upsampled logits come back, then the host applies the area rule.
/// The memory encoder that follows is marked per state as `<STAGES[9]>, state <i>: <phase>`
/// (`state::ENCODE_STAGES`).
pub const MEMORY_UPDATE_STAGES: [&str; 2] = ["upsampled readback", "area rule (host)"];

fn mark(marker: &mut Option<StageMarker<'_>>, stage: &str) {
    if let Some(m) = marker.as_mut() {
        m(stage);
    }
}

/// `_suppress_object_pw_area_shrinkage` followed by the area-proxy score of `_tracker_update_memories`:
/// with more than one object, an object whose area at pixels it wins (first index on ties)
/// falls below 0.3 of its area is clamped to at most -10; each object's score is then `+10` if
/// any pixel is positive, else `-10`. `hr`: `[n, px]` logits, edited in place.
fn suppress_area_shrinkage(hr: &mut [f32], n: usize, px: usize) -> Vec<f32> {
    if n > 1 {
        let mut winner = vec![0usize; px];
        for (i, w) in winner.iter_mut().enumerate() {
            let mut best = hr[i];
            for o in 1..n {
                let v = hr[o * px + i];
                if v > best {
                    best = v;
                    *w = o;
                }
            }
        }
        for o in 0..n {
            let plane = &mut hr[o * px..(o + 1) * px];
            let before = plane.iter().filter(|&&v| v > 0.0).count();
            let after = plane.iter().enumerate().filter(|&(i, &v)| winner[i] == o && v > 0.0).count();
            let ratio = after as f32 / before.max(1) as f32;
            if !(ratio >= 0.3) {
                plane.iter_mut().for_each(|v| *v = v.min(-10.0));
            }
        }
    }
    (0..n).map(|o| if hr[o * px..(o + 1) * px].iter().any(|&v| v > 0.0) { 10.0 } else { -10.0 }).collect()
}

/// Round a tensor's values through bfloat16 and keep computing in f32. Upstream's video detector
/// casts both tracker necks' FPN levels to bf16 before the tracker sees them
/// (`sam3_multiplex_detector.py:563-586`), i.e. before `conv_s0`/`conv_s1`, memory attention, the
/// decoders and the memory encoder. The VOS driver (`vos.rs`) does not do this.
///
/// `Backend::cast_bf16` rounds on the device (Metal `ojas_round_bf16`; no host read, so no wait):
/// nearest even, as `vos::bf16_round` and torch, for every non-NaN value; a NaN stays a quiet NaN.
fn round_bf16<B: Backend + ?Sized>(be: &B, t: &Tensor) -> Result<Tensor, ModelError> {
    Ok(be.cast_bf16(t)?)
}

fn round_neck<B: Backend + ?Sized>(be: &B, neck: [Tensor; 3]) -> Result<[Tensor; 3], ModelError> {
    let [a, b, c] = neck;
    Ok([round_bf16(be, &a)?, round_bf16(be, &b)?, round_bf16(be, &c)?])
}

/// Mask logits of query `q`: one of `QUERIES` equal rows (`step_frame` checks the count against
/// the geometry's mask size before any row is read).
fn det_row(det: &Detections, q: usize) -> &[f32] {
    let px = det.masks.len() / QUERIES;
    &det.masks[q * px..(q + 1) * px]
}

/// `masks_to_boxes` (inclusive extents) then `box_xyxy_to_xywh` and the normalization by the video size.
fn mask_box_xywh(m: &Bitmask, h: usize, w: usize) -> [f32; 4] {
    let (mut x0, mut y0, mut x1, mut y1) = (w, h, 0usize, 0usize);
    for y in 0..h {
        for x in 0..w {
            if m.get(y * w + x) {
                x0 = x0.min(x);
                x1 = x1.max(x);
                y0 = y0.min(y);
                y1 = y1.max(y);
            }
        }
    }
    let (x0, y0, x1, y1) = (x0 as f32, y0 as f32, x1 as f32, y1 as f32);
    [x0 / w as f32, y0 / h as f32, (x1 - x0) / w as f32, (y1 - y0) / h as f32]
}

impl<'a, B: Backend + ?Sized> Session<'a, B> {
    /// Tokenize and encode the prompt, keeping its valid (non-padding) tokens `[1, L, 256]`.
    pub fn encode_prompt(be: &B, text: &Text<'_, B>, tokenizer: &Tokenizer, prompt: &str) -> Result<Tensor, ModelError> {
        let ids = tokenizer.tokenize(prompt).map_err(|e| err(format!("tokenizer: {e}")))?;
        let tf = text.encode(&[ids])?;
        let resized = host_of(&tf.resized, be)?;
        let valid: Vec<f32> = (0..tf.padding.len()).filter(|&t| !tf.padding[t]).flat_map(|t| resized[t * DET_D..(t + 1) * DET_D].to_vec()).collect();
        Ok(up(be, &valid, &[1, valid.len() / DET_D, DET_D])?)
    }

    /// `prompt`: from [`Self::encode_prompt`]. The tracker `vos` brings the backbone.
    pub fn new(vos: &'a Vos<'a, B>, detector: &'a Detector<'a, B>, prompt: Tensor, cfg: SessionConfig) -> Self {
        let book = TrackBook::new(cfg.book);
        let yield_buf = YieldBuffer::new(cfg.book.hotstart_delay.max(0) as usize);
        let hide = HideLog::new(cfg.book.confirm_consecutive, cfg.num_frames as i64);
        Self { vos, detector, prompt, cfg, states: Vec::new(), book, next_frame: 0, yield_buf, hide, buffered: BTreeMap::new(), trace: false, traces: Vec::new(), marker: None }
    }

    /// Call `marker` with the stage name as each phase of a frame begins ([`STAGES`]), for
    /// profiling (`track --profile`). It sees names only; timing and syncing are its business.
    pub fn set_stage_marker(&mut self, marker: impl FnMut(&str) + 'a) {
        self.marker = Some(Box::new(marker));
    }

    /// Record a [`VisitTrace`] per visit (large: tens of megabytes per frame).
    pub fn set_trace(&mut self, on: bool) {
        self.trace = on;
    }

    pub fn take_traces(&mut self) -> Vec<VisitTrace> {
        std::mem::take(&mut self.traces)
    }

    pub fn states(&self) -> &[TrackerState] {
        &self.states
    }

    pub fn book(&self) -> &TrackBook {
        &self.book
    }

    /// Frames fed so far.
    pub fn frames_done(&self) -> usize {
        self.next_frame
    }

    /// Feed the next frame (`[1, 3, 1008, 1008]`, upstream's preprocessing) and return the frames
    /// the yield buffer releases.
    pub fn push_frame(&mut self, frame: &Tensor) -> Result<Vec<YieldedFrame>, ModelError> {
        mark(&mut self.marker, STAGES[0]);
        let trunk = self.vos.backbone.trunk(frame, &[], &mut Taps::default())?;
        mark(&mut self.marker, STAGES[1]);
        let fpn = self.vos.backbone.neck(Neck::Detector, &trunk)?;
        let marker = &mut self.marker;
        let out = self.detector.forward_marked(&fpn, &self.prompt, &mut DetectorTaps::default(), &mut |s| mark(marker, s))?;
        mark(&mut self.marker, STAGES[2]);
        let det = Detections { logits: out.pred_logits, boxes_cxcywh: out.pred_boxes, masks: host_of(&out.pred_masks, self.vos.be)? };
        self.step_frame(&trunk, &det)
    }

    /// As [`Self::push_frame`] with the detector's output supplied (for isolating the tracker).
    pub fn push_frame_with_detections(&mut self, frame: &Tensor, det: &Detections) -> Result<Vec<YieldedFrame>, ModelError> {
        let trunk = self.vos.backbone.trunk(frame, &[], &mut Taps::default())?;
        self.step_frame(&trunk, det)
    }

    /// Release every frame still buffered (upstream flushes at the last frame; this is for a
    /// stream that stops early).
    pub fn finish(&mut self) -> Vec<YieldedFrame> {
        let frames: Vec<i64> = self.buffered.keys().copied().collect();
        frames.into_iter().map(|f| self.release(f)).collect()
    }

    fn step_frame(&mut self, trunk: &Tensor, det: &Detections) -> Result<Vec<YieldedFrame>, ModelError> {
        let t = self.next_frame;
        if t >= self.cfg.num_frames {
            return Err(err(format!("frame {t} is past the {} frames the session was built for", self.cfg.num_frames)));
        }
        let mask = self.vos.geo.low_res();
        if det.logits.len() != QUERIES || det.boxes_cxcywh.len() != QUERIES * 4 || det.masks.len() != QUERIES * mask * mask {
            return Err(err(format!("detections do not hold {QUERIES} queries of {mask}x{mask} masks (the tracker's geometry is {} pixels; was the detector built for the same size?)", self.vos.geo.pixels())));
        }
        mark(&mut self.marker, STAGES[3]);
        let prop = round_neck(self.vos.be, self.vos.backbone.neck(Neck::Propagation, trunk)?)?;
        let mut ctx = self.vos.frame_ctx(t, &prop)?;
        // The prompt frame: `add_prompt`'s visit, then `propagate_in_video`'s.
        let visits = if t == 0 { 2 } else { 1 };
        let mut last = None;
        for v in 0..visits {
            last = Some(self.visit(t, v, trunk, &mut ctx, det)?);
        }
        mark(&mut self.marker, STAGES[11]);
        let out = last.expect("at least one visit");
        for st in self.states.iter_mut() {
            st.prune(t);
        }
        self.next_frame += 1;
        self.hide.record(t as i64, &out.unconfirmed);
        self.buffered.insert(t as i64, out.buffered);
        let released = self.yield_buf.push(t as i64, t + 1 == self.cfg.num_frames);
        Ok(released.into_iter().map(|f| self.release(f)).collect())
    }

    /// `_postprocess_output` for one released frame.
    fn release(&mut self, frame: i64) -> YieldedFrame {
        let buf = self.buffered.remove(&frame).expect("a released frame is buffered");
        let (h, w) = self.cfg.video_hw;
        let hw = h * w;
        let hidden: Vec<i64> = self.hide.hidden_for(frame).map(|v| v.to_vec()).unwrap_or_default();
        let kept: Vec<&(i64, Bitmask, f32, f32, Option<[f32; 4]>)> = buf.items.iter().filter(|(id, m, _, _, _)| m.area() > 0 && !hidden.contains(id)).collect();
        let boxes: Vec<f32> = kept.iter().flat_map(|(_, m, _, _, _)| mask_box_xywh(m, h, w)).collect();
        // Object-wise non-overlap: at each pixel only the covering object with the highest
        // tracker score keeps it (first index on ties), and only if that score is positive.
        let masks: Vec<Bitmask> = if kept.len() > 1 {
            let mut winner = vec![0usize; hw];
            for (p, win) in winner.iter_mut().enumerate() {
                let mut best = if kept[0].1.get(p) { kept[0].3 } else { 0.0 };
                for (o, item) in kept.iter().enumerate().skip(1) {
                    let v = if item.1.get(p) { item.3 } else { 0.0 };
                    if v > best {
                        best = v;
                        *win = o;
                    }
                }
            }
            kept.iter().enumerate().map(|(o, item)| Bitmask::from_fn(hw, |p| item.1.get(p) && winner[p] == o && item.3 > 0.0)).collect()
        } else {
            kept.iter().map(|item| item.1.clone()).collect()
        };
        YieldedFrame {
            frame: buf.frame,
            obj_ids: kept.iter().map(|k| k.0).collect(),
            probs: kept.iter().map(|k| k.2).collect(),
            boxes_xywh: boxes,
            masks,
            det_boxes_xyxy: kept.iter().map(|k| k.4).collect(),
        }
    }

    /// The frame's interactive neck, computed once (marked as its own stage), then `next` marked.
    fn ensure_inter_marked(&mut self, trunk: &Tensor, ctx: &mut FrameCtx, next: &str) -> Result<(), ModelError> {
        if ctx.inter.is_none() {
            mark(&mut self.marker, STAGES[6]);
            let inter = round_neck(self.vos.be, self.vos.backbone.neck(Neck::Interactive, trunk)?)?;
            ctx.inter = Some(self.vos.interactive_pix(&inter)?);
        }
        mark(&mut self.marker, next);
        Ok(())
    }

    /// NMS, thresholds, boundary filter and the kept-first reorder over the detector's queries.
    fn process_detections(&self, det: &Detections) -> DetStage {
        let px = det.masks.len() / QUERIES;
        let probs0: Vec<f32> = det.logits.iter().map(|&l| host::sigmoid(l)).collect();
        let bits0: Vec<Bitmask> = (0..QUERIES).map(|q| Bitmask::from_logits(&det.masks[q * px..(q + 1) * px])).collect();
        let nms_keep = nms_iom(&probs0, &bits0, self.cfg.score_threshold, self.cfg.nms_iom);
        // Queries NMS did not keep (including those at or under the threshold) lose 1e4 of logit.
        let probs: Vec<f32> = (0..QUERIES).map(|q| host::sigmoid(det.logits[q] - 1e4 * if nms_keep[q] { 0.0 } else { 1.0 })).collect();
        let boxes_xyxy: Vec<f32> = det
            .boxes_cxcywh
            .chunks_exact(4)
            .flat_map(|b| [b[0] - 0.5 * b[2], b[1] - 0.5 * b[3], b[0] + 0.5 * b[2], b[1] + 0.5 * b[3]])
            .collect();
        let corners: Vec<[f32; 4]> = boxes_xyxy.chunks_exact(4).map(|b| [b[0], b[1], b[2], b[3]]).collect();
        let inside = not_close_to_boundary(&corners, self.cfg.boundary_margin);
        let keep: Vec<bool> = (0..QUERIES).map(|q| probs[q] > self.cfg.score_threshold && inside[q]).collect();
        let mut order: Vec<usize> = (0..QUERIES).filter(|&q| keep[q]).collect();
        order.extend((0..QUERIES).filter(|&q| !keep[q]));
        DetStage {
            bits: order.iter().map(|&q| bits0[q].clone()).collect(),
            scores: order.iter().map(|&q| probs[q]).collect(),
            keep_re: order.iter().map(|&q| keep[q]).collect(),
            probs,
            keep,
            boxes_xyxy,
            order,
        }
    }

    /// One visit of `_det_track_one_frame_impl`.
    fn visit(&mut self, t: usize, visit: usize, trunk: &Tensor, ctx: &mut FrameCtx, det: &Detections) -> Result<VisitOut, ModelError> {
        let vos = self.vos;
        let be = vos.be;
        let (h, w) = self.cfg.video_hw;
        let (low_res, input_mask) = (vos.geo.low_res(), vos.geo.input_mask());
        let px = low_res * low_res;
        let num_frames = self.cfg.num_frames;
        let prev_ids: Vec<i64> = self.book.obj_ids().to_vec();
        let n = prev_ids.len();

        // 1. Propagate every state; rows sorted by object id.
        let mut entries: Vec<(i64, usize, usize)> = Vec::new();
        let mut props = Vec::with_capacity(self.states.len());
        for (si, st) in self.states.iter_mut().enumerate() {
            if self.marker.is_some() {
                mark(&mut self.marker, &format!("{}, state {si}", STAGES[4]));
            }
            let marker = &mut self.marker;
            let p = st.propagate_one_frame(vos, ctx, num_frames, &mut |s| {
                if marker.is_some() {
                    mark(marker, &format!("{}, state {si}: {s}", STAGES[4]));
                }
            })?;
            for (k, &id) in st.obj_ids().iter().enumerate() {
                entries.push((id, si, k));
            }
            props.push(p);
        }
        entries.sort_by_key(|e| e.0);
        if entries.iter().map(|e| e.0).collect::<Vec<_>>() != prev_ids {
            return Err(err(format!("tracker states hold ids {:?} but the book lists {prev_ids:?}", entries.iter().map(|e| e.0).collect::<Vec<_>>())));
        }
        let mut trk: Vec<f32> = Vec::with_capacity(n * px);
        let mut trk_scores: Vec<f32> = Vec::with_capacity(n);
        for &(_, si, k) in &entries {
            trk.extend_from_slice(&props[si].low_res[k * px..(k + 1) * px]);
            trk_scores.push(props[si].scores[k]);
        }
        let prop_masks = if self.trace { trk.clone() } else { Vec::new() };

        // 2-3. Detections: NMS, keep, reorder.
        mark(&mut self.marker, STAGES[5]);
        let ds = self.process_detections(det);

        // 4. Plan.
        let trk_bits: Vec<Bitmask> = (0..n).map(|i| Bitmask::from_logits(&trk[i * px..(i + 1) * px])).collect();
        let association = associate(&ds.bits, &ds.scores, &ds.keep_re, &trk_bits, &self.cfg.assoc);
        let trk_det = track_detections(&association, &ds.scores, &ds.keep_re, n);
        let plan = self.book.plan_frame(&FrameInput {
            frame_idx: t as i64,
            association: &association,
            det_keep: &ds.keep_re,
            det_scores: &ds.scores,
            det_masks: &ds.bits,
            trk_masks: &trk_bits,
            trk_score_logits: &trk_scores,
        });

        // 4a. Recondition.
        let mut per_state: Vec<(usize, Vec<i64>, Vec<usize>)> = Vec::new();
        for &(id, d) in &plan.recondition {
            let row = prev_ids.iter().position(|&x| x == id).ok_or_else(|| err(format!("reconditioned id {id} is not a track")))?;
            let dl = det_row(det, ds.order[d]);
            for (p, &dv) in dl.iter().enumerate() {
                if (dv > 0.0) != (trk[row * px + p] > 0.0) {
                    trk[row * px + p] = dv;
                }
            }
            let si = self.states.iter().position(|s| s.obj_ids().contains(&id)).ok_or_else(|| err(format!("reconditioned id {id} is in no tracker state")))?;
            match per_state.iter_mut().find(|e| e.0 == si) {
                Some(e) => {
                    e.1.push(id);
                    e.2.push(d);
                }
                None => per_state.push((si, vec![id], vec![d])),
            }
        }
        if !per_state.is_empty() {
            self.ensure_inter_marked(trunk, ctx, STAGES[7])?;
            for (si, ids, ds_idx) in &per_state {
                let dl: Vec<f32> = ds_idx.iter().flat_map(|&d| det_row(det, ds.order[d]).iter().copied()).collect();
                let masks = self.binary_input_masks(&dl, ds_idx.len())?;
                self.states[*si].add_new_masks(vos, ctx, ids, &masks, (h, w), true)?;
            }
            for st in self.states.iter_mut() {
                // Only the states a recondition touched hold a pending consolidation.
                st.preflight(vos, ctx, false)?;
            }
        }

        // 4b. Occlusion suppression.
        for &r in &plan.suppressed {
            trk[r * px..(r + 1) * px].iter_mut().for_each(|v| *v = -10.0);
        }

        // 4c. Memory update for every existing track.
        mark(&mut self.marker, STAGES[8]);
        let (mem_masks, mem_scores) = if n > 0 {
            let low = up(be, &trk, &[n, 1, low_res, low_res])?;
            let upsampled = be.interpolate_bilinear_forward(&low, [input_mask, input_mask], false)?;
            if self.marker.is_some() {
                mark(&mut self.marker, &format!("{}: {}", STAGES[8], MEMORY_UPDATE_STAGES[0]));
            }
            // The host tensor the readback made is the working copy: no second copy into a Vec. On a
            // host backend `to_host` is a clone sharing `upsampled`'s storage, so `upsampled` is
            // dropped first to leave the allocation uniquely owned for the in-place area rule.
            let mut hr_host = upsampled.to_host(be.budget())?;
            drop(upsampled);
            let hr = hr_host.f32_slice_mut()?;
            if self.marker.is_some() {
                mark(&mut self.marker, &format!("{}: {}", STAGES[8], MEMORY_UPDATE_STAGES[1]));
            }
            let proxy = suppress_area_shrinkage(hr, n, input_mask * input_mask);
            mark(&mut self.marker, STAGES[9]);
            for (si, st) in self.states.iter_mut().enumerate() {
                if st.is_empty() {
                    continue;
                }
                let rows = st
                    .obj_ids()
                    .iter()
                    .map(|id| prev_ids.binary_search(id).map_err(|_| err(format!("state object {id} is not a track"))))
                    .collect::<Result<Vec<usize>, ModelError>>()?;
                let local_scores: Vec<f32> = rows.iter().map(|&r| proxy[r]).collect();
                let marker = &mut self.marker;
                st.encode_memory(vos, ctx, hr, &rows, &local_scores, &mut |s| {
                    if marker.is_some() {
                        mark(marker, &format!("{}, state {si}: {s}", STAGES[9]));
                    }
                })?;
            }
            (if self.trace { trk.clone() } else { Vec::new() }, proxy)
        } else {
            (Vec::new(), Vec::new())
        };

        // 5a. Births.
        let k = plan.births.len();
        if k > 0 {
            self.ensure_inter_marked(trunk, ctx, STAGES[10])?;
            let dl: Vec<f32> = plan.births.iter().flat_map(|b| det_row(det, ds.order[b.det]).iter().copied()).collect();
            let masks = self.binary_input_masks(&dl, k)?;
            let ids: Vec<i64> = plan.births.iter().map(|b| b.id).collect();
            // At most MUX births fit one state (one 16-slot bucket). Under upstream's builder cap
            // (`max_num_objects` 16) a frame never births more, and this is one group. A larger
            // cap (`BookParams::max_num_objects`, not the builder's) can birth more in one frame;
            // they are placed MUX at a time, in birth order. Upstream would put them in extra
            // buckets of one state instead, so beyond 16 objects the grouping, and with it the
            // multiplex decoder's joint decode, is not parity-checked.
            let px_in = input_mask * input_mask;
            for (g, group) in ids.chunks(MUX).enumerate() {
                let kg = group.len();
                let gm = &masks[g * MUX * px_in..(g * MUX + kg) * px_in];
                // The state with the fewest empty slots that still fits them, the first on ties.
                let mut best: Option<(usize, usize)> = None;
                for (i, st) in self.states.iter().enumerate() {
                    let a = st.available_slots();
                    if a >= kg && best.is_none_or(|(_, b)| a < b) {
                        best = Some((i, a));
                    }
                }
                let si = match best {
                    Some((i, _)) => i,
                    None => {
                        self.states.push(TrackerState::new());
                        self.states.len() - 1
                    }
                };
                self.states[si].add_new_masks(vos, ctx, group, gm, (h, w), false)?;
                self.states[si].preflight(vos, ctx, true)?;
            }
        }

        // 5b. Removals.
        if !plan.removed_ids.is_empty() {
            for st in self.states.iter_mut() {
                st.remove_objects(&plan.removed_ids)?;
            }
            self.states.retain(|s| !s.is_empty());
        }

        // 6. Outputs: previous ids from the final track masks, newborns from the detector masks.
        mark(&mut self.marker, STAGES[11]);
        let total = n + k;
        let mut all = trk.clone();
        for b in &plan.births {
            all.extend_from_slice(det_row(det, ds.order[b.det]));
        }
        let mut sam2_scores: Vec<(i64, f32)> = prev_ids.iter().zip(&trk_scores).map(|(&id, &s)| (id, host::sigmoid(s))).collect();
        sam2_scores.extend(plan.births.iter().map(|b| (b.id, ds.scores[b.det])));
        let mut outputs: Vec<(i64, Bitmask)> = Vec::with_capacity(total);
        if total > 0 {
            let src = up(be, &all, &[total, 1, low_res, low_res])?;
            let video = host_of(&be.interpolate_bilinear_forward(&src, [h, w], false)?, be)?;
            let ids = prev_ids.iter().copied().chain(plan.births.iter().map(|b| b.id));
            for (i, id) in ids.enumerate() {
                outputs.push((id, Bitmask::from_logits(&video[i * h * w..(i + 1) * h * w])));
            }
        }
        // The detection box each output id reports under `--boxes detector` (reordered index -> query).
        let det_box = |d: usize| -> [f32; 4] {
            let q = ds.order[d];
            [ds.boxes_xyxy[q * 4], ds.boxes_xyxy[q * 4 + 1], ds.boxes_xyxy[q * 4 + 2], ds.boxes_xyxy[q * 4 + 3]]
        };
        let det_boxes: Vec<Option<[f32; 4]>> = trk_det.iter().map(|d| d.map(det_box)).chain(plan.births.iter().map(|b| Some(det_box(b.det)))).collect();
        let mut items: Vec<(i64, Bitmask, f32, f32, Option<[f32; 4]>)> = Vec::with_capacity(total);
        for (i, (id, mask)) in outputs.iter().enumerate() {
            let birth = *self.book.scores().get(id).ok_or_else(|| err(format!("no score for id {id}")))?;
            let sam2 = sam2_scores.iter().find(|(i, _)| i == id).map(|s| s.1).unwrap_or(0.0);
            items.push((*id, mask.clone(), birth, sam2, det_boxes[i]));
        }
        items.sort_by_key(|i| i.0);

        if self.trace {
            let kept_q: Vec<usize> = (0..QUERIES).filter(|&q| ds.keep[q]).collect();
            self.traces.push(VisitTrace {
                frame: t,
                visit,
                det_kept_masks: kept_q.iter().flat_map(|&q| det_row(det, q).iter().copied()).collect(),
                det_probs: ds.probs.clone(),
                det_keep: ds.keep.clone(),
                det_boxes_xyxy: ds.boxes_xyxy.clone(),
                det_order: ds.order.clone(),
                prop_ids: prev_ids.clone(),
                prop_masks,
                prop_scores: trk_scores.clone(),
                association: association.clone(),
                plan: plan.clone(),
                mem_masks,
                mem_scores,
                states: self
                    .states
                    .iter()
                    .map(|st| {
                        let stored = st.stored(t);
                        StateTrace {
                            obj_ids: st.obj_ids().to_vec(),
                            slots: st.slots().to_vec(),
                            maskmem: stored.and_then(|s| s.maskmem.as_ref().map(|m| m.nchw.clone())),
                            obj_ptr: stored.map(|s| s.obj_ptr.clone()).unwrap_or_default(),
                            cond_objs: stored.map(|s| s.cond_objs.iter().copied().collect()).unwrap_or_default(),
                        }
                    })
                    .collect(),
                outputs: outputs.clone(),
                sam2_scores: sam2_scores.clone(),
            });
        }
        Ok(VisitOut { buffered: Buffered { frame: t, items }, unconfirmed: plan.unconfirmed_ids.clone() })
    }

    /// Detector mask logits `[k, M * M]` to the tracker's input masks: bilinear to `I = 16 * grid`
    /// (no antialiasing) and `> 0`, as 0/1 floats `[k, I * I]` (`M` = 288, `I` = 1152 at 1008).
    fn binary_input_masks(&self, logits: &[f32], k: usize) -> Result<Vec<f32>, ModelError> {
        let be = self.vos.be;
        let (low_res, input_mask) = (self.vos.geo.low_res(), self.vos.geo.input_mask());
        let src = up(be, logits, &[k, 1, low_res, low_res])?;
        let hi = host_of(&be.interpolate_bilinear_forward(&src, [input_mask, input_mask], false)?, be)?;
        Ok(hi.iter().map(|&v| if v > 0.0 { 1.0 } else { 0.0 }).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tracker::vos::bf16_round;

    /// `round_bf16` (Backend::cast_bf16) equals the host rounding it replaced, `vos::bf16_round`,
    /// bit for bit on every non-NaN input: random values over the whole exponent range, ties
    /// (exactly half an ulp, even and odd), subnormals, the largest finite values, infinities and
    /// zeros. (A NaN goes to a quiet NaN under cast_bf16; the tracker features are finite.) On
    /// Metal, Ojas pins cast_bf16 to the same host rounding bit for bit over every upper half-word
    /// and the ties (`ojas-metal/tests/cast_bf16.rs`, cast_bf16_matches_the_host_round_...), so
    /// Metal equals the old host path by transitivity; the Metal session parity gate is unchanged
    /// (32/32 id sets, IoU 0.99855, bench/results/parity-tracker/metal-f32-detect-track-visem22.log).
    #[test]
    fn device_bf16_rounding_equals_the_host_rounding_it_replaced() {
        let be = ojas_cpu::CpuBackend::new(ojas_core::Budget::new(64 << 20));
        let mut v: Vec<f32> = Vec::new();
        let mut x = 0x2545_f491u32;
        while v.len() < 20_000 {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            let f = f32::from_bits(x);
            if !f.is_nan() {
                v.push(f);
            }
        }
        for hi in [0x3f80u32, 0x3f81, 0x0001, 0x7f7f, 0x807f, 0xff7f] {
            v.extend([0x0000u32, 0x7fff, 0x8000, 0x8001, 0xffff].map(|lo| f32::from_bits((hi << 16) | lo)));
        }
        v.extend([f32::MAX, f32::MIN, f32::MIN_POSITIVE, f32::INFINITY, f32::NEG_INFINITY, 0.0, -0.0, 1e-40, -1e-40]);
        let n = v.len();
        let t = Tensor::from_f32(&v, &[n], be.budget()).expect("tensor");
        let ours = round_bf16(&be, &t).expect("round").to_f32_vec().expect("f32");
        let mut want = v.clone();
        bf16_round(&mut want);
        let differ: Vec<(f32, u32, u32)> = v.iter().zip(&ours).zip(&want).filter(|((_, a), b)| a.to_bits() != b.to_bits()).map(|((x, a), b)| (*x, a.to_bits(), b.to_bits())).take(5).collect();
        assert!(differ.is_empty(), "cast_bf16 differs from bf16_round: {differ:?}");
    }

    #[test]
    fn shrinkage_suppression_clamps_an_object_that_keeps_under_30_percent_of_its_area() {
        // Two objects over 10 pixels: object 0 is positive on all 10, object 1 on 4 of which object 1 wins 1.
        let px = 10;
        let mut hr = vec![0f32; 2 * px];
        for i in 0..10 {
            hr[i] = 5.0;
        }
        for i in 0..4 {
            hr[px + i] = 6.0 + if i == 0 { 1.0 } else { -5.5 };
        }
        // Pixel 0: object 1 (7.0) beats 5.0. Pixels 1-3: object 0 (5.0) beats 0.5.
        let scores = suppress_area_shrinkage(&mut hr, 2, px);
        // Object 1 keeps 1 of its 4 positive pixels: 0.25 < 0.3, so its whole plane is clamped to at
        // most -10 and its proxy score is -10; object 0 keeps 9 of 10 and stays at +10.
        assert_eq!(scores, vec![10.0, -10.0]);
        assert!(hr[px..].iter().all(|&v| v <= -10.0));
        assert!(hr[..px].iter().all(|&v| v == 5.0));
    }

    #[test]
    fn a_single_object_is_untouched_and_an_empty_mask_scores_minus_ten() {
        let mut hr = vec![-3.0f32; 4];
        assert_eq!(suppress_area_shrinkage(&mut hr, 1, 4), vec![-10.0]);
        assert_eq!(hr, vec![-3.0; 4]);
        let mut hr = vec![1.0, -1.0, 2.0, -2.0];
        assert_eq!(suppress_area_shrinkage(&mut hr, 1, 4), vec![10.0]);
    }

    #[test]
    fn mask_boxes_are_inclusive_extents_normalized_with_no_plus_one() {
        let m = Bitmask::from_fn(8 * 4, |i| {
            let (x, y) = (i % 8, i / 8);
            (2..=5).contains(&x) && (1..=2).contains(&y)
        });
        assert_eq!(mask_box_xywh(&m, 4, 8), [2.0 / 8.0, 1.0 / 4.0, 3.0 / 8.0, 1.0 / 4.0]);
    }
}
