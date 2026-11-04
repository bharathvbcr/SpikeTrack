//! The multiplex tracker's inference state for one bucket of up to 16 objects, as upstream's
//! `Sam3VideoTrackingMultiplexDemo` keeps it (`video_tracking_multiplex_demo.py`,
//! `video_tracking_multiplex.py`, `multiplex_utils.py`), and the operations the text-prompted
//! video session drives on it. Built from the pieces `vos.rs` ports (memory attention, the
//! multiplex decoder, the interactive path, the memory encoder); this file adds what the
//! semi-supervised VOS driver does not need: several states side by side, objects entering
//! and leaving a state, per-frame stored outputs that later frames and later steps of the same
//! frame read back, and the slot map between objects and bucket slots.
//!
//! What a state holds (upstream `inference_state`): object ids in insertion order (always
//! ascending: ids are handed out increasing), the slot assignment (`MultiplexState`:
//! `PADDING = -1` for an empty slot, `REMOVED = -1116` for a removed object whose slot is never
//! reused), a conditioning and a non-conditioning output per frame (`output_dict`), the frames
//! whose stored output the next propagation reuses (`consolidated_frame_inds`), the frames
//! already propagated (`frames_already_tracked`), the frames on which each object received a
//! mask input (`mask_inputs_per_obj`) and the per-object temporary masks between
//! `add_new_masks` and `propagate_in_video_preflight` (`temp_output_dict_per_obj`).
//!
//! Operations (upstream function -> method):
//! - `add_new_masks` (`VD:1665-1904`, births, `reconditioning=False`) and its recondition twin
//!   (`reconditioning=True`): [`TrackerState::add_new_masks`], with `add_new_masks_to_existing_state`
//!   and `recondition_masks_in_existing_state` (`VM:3142-3428`);
//! - `propagate_in_video_preflight` (`VD:2277-2361`, `_consolidate_temp_output_across_obj`
//!   `VD:1933-2274`): [`TrackerState::preflight`];
//! - `propagate_in_video` for one frame without memory encoding (`VD:3394-3481`, `track_step`
//!   propagation mode `VM:2100-2126`, `_prepare_memory_conditioned_features` `VM:1304-1643`):
//!   [`TrackerState::propagate_one_frame`];
//! - `_run_memory_encoder` as `_tracker_update_memories` calls it (`B:2496-2666`, `VD:2869-2931`):
//!   [`TrackerState::encode_memory`];
//! - `remove_objects` (`VD:2989-3184`, `MU:242-344`): [`TrackerState::remove_objects`].
//!
//! Not ported because the builder's path never reaches it: more than one bucket per state (the
//! 16-object cap and best-fit placement keep every state at one bucket), more than 4
//! conditioning frames per state (a state has exactly one, its creation frame),
//! `use_memory_selection`, `reapply_no_object_pointer` (class default False, the builder does not
//! set it: `B:255`), reverse tracking.
//!
//! What a stored output keeps that matters later: `maskmem`, `obj_ptr` and the frame's image
//! features (memory attention of later frames), the conditioning set (the memory encoder's
//! condition channel), and `pred_masks` only on the frame being processed (the prompt frame's
//! second visit reads it back, and a birth in the same frame re-encodes it). Older frames drop
//! their `pred_masks` and, past the memory and pointer horizon, the whole output.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, OnceLock};

use ojas_core::{Backend, Conv2dParams, DType, OjasError, Tensor};

use crate::host;
use crate::layers::{host_slice, up};
use crate::tracker::mask_decoder::{high_res, MASKS_PER_OBJ, MUX};
use crate::tracker::memory_attention::{FrameKv, FramesInputs, HostFrame, HostMemInputs};
use crate::tracker::vos::{
    bf16_round, nchw_to_tokens, sine_1d, Vos, D, LN2D_EPS, MAX_PTRS, NO_OBJ_SCORE, NUM_MASKMEM,
};
use crate::ModelError;

/// An empty slot of a bucket (`_PADDING_NUM`).
pub const PADDING: i32 = -1;
/// The slot of a removed object (`_REMOVED_NUM`); never reused.
pub const REMOVED: i32 = -1116;
/// `max_cond_frames_in_attn`.
const MAX_COND_IN_ATTN: usize = 4;
/// The phases [`TrackerState::propagate_one_frame`] marks when it computes a frame: the host
/// builds the memory and pointer tokens and uploads them; memory attention runs on the device;
/// the multiplex decoder runs on the device with its small heads on the host; then the masks
/// come back and the host picks each object's mask and pointer. A frame with no stored memory
/// skips the attention mark.
pub const PROPAGATE_STAGES: [&str; 4] = [
    "memory + pointer tokens (host) + upload",
    "memory attention (device)",
    "multiplex decoder",
    "mask readback + per-object pick + pointers (host)",
];
/// The phases [`TrackerState::encode_memory`] marks: the host builds the encoder's mask channels
/// (sigmoid of each object's logits, the condition planes); they are uploaded; the memory encoder
/// runs on the device and its features come back; the host adds `no_obj_embed_spatial`, rounds to
/// bf16 and uploads the tokens.
pub const ENCODE_STAGES: [&str; 4] = [
    "mask channels (host sigmoid + fill)",
    "mask upload",
    "encoder (device) + feature readback",
    "no-object add + bf16 + token upload (host)",
];

pub(crate) fn err(detail: impl Into<String>) -> ModelError {
    ModelError::Ojas(OjasError::OutOfRange {
        op: "st_model::tracker::state",
        detail: detail.into(),
    })
}

/// One frame's propagation-neck features as device tokens `[1, G * G, 256]` (`G` = 72 at 1008); shared by every
/// state that stores a memory for the frame (upstream stores the same tensor in each state's output).
pub struct FrameFeats {
    pub tokens: Tensor,
    /// `tokens` on the host, read once by the first frame that needs it (an external memory
    /// attention only).
    pub(crate) host: OnceLock<Vec<f32>>,
}

impl FrameFeats {
    pub fn new(tokens: Tensor) -> Self {
        Self {
            tokens,
            host: OnceLock::new(),
        }
    }

    fn host(&self, budget: &ojas_core::Budget) -> Result<&[f32], ModelError> {
        if self.host.get().is_none() {
            let h = self.tokens.to_host(budget)?;
            let _ = self.host.set(h.f32_slice()?.to_vec());
        }
        Ok(self.host.get().map(Vec::as_slice).unwrap_or_default())
    }
}

/// A stored frame's memory features (`maskmem_features`, rounded through bf16 as upstream stores
/// them): `nchw` `[256 * G * G]` on the host for traces and parity, and the same values token-major
/// on the device (`[1, G * G, 256]`), which memory attention concatenates every frame.
#[derive(Clone)]
pub struct MaskMem {
    pub nchw: Vec<f32>,
    pub tokens: Tensor,
    /// Its projections in memory attention, with the image features they were computed from:
    /// filled by the first frame that attends this memory ([`MaskMem::kv`]), shared by clones,
    /// and dropped by [`TrackerState::prune`] once no later frame can attend it.
    kv: Arc<OnceLock<(Arc<FrameFeats>, FrameKv)>>,
    /// `nchw` token-major on the host, built once for an external memory attention.
    host_tokens: Arc<OnceLock<Vec<f32>>>,
}

impl MaskMem {
    fn host_tokens(&self) -> &[f32] {
        self.host_tokens.get_or_init(|| {
            let tokens = self.nchw.len() / D;
            nchw_to_tokens(&self.nchw, D, tokens)
        })
    }

    /// This memory's memory-attention projections with `feats`, the stored frame's own image
    /// features: computed on the first call, then reused.
    fn kv<B: Backend + ?Sized>(
        &self,
        vos: &Vos<'_, B>,
        feats: &Arc<FrameFeats>,
    ) -> Result<&FrameKv, ModelError> {
        if self.kv.get().is_none() {
            let kv = vos.mem_attn.frame_kv(vos.be, &feats.tokens, &self.tokens)?;
            // A second setter would hold the same values; the first one stays.
            let _ = self.kv.set((Arc::clone(feats), kv));
        }
        match self.kv.get() {
            Some((with, kv)) if Arc::ptr_eq(with, feats) => Ok(kv),
            Some(_) => Err(err(
                "a mask memory's projections were computed with another frame's image features",
            )),
            None => Err(err("a mask memory's projections were not stored")),
        }
    }
}

/// Interactive-path inputs of the current frame (`_get_interactive_pix_mem` and the two
/// high-resolution skips after the interactive decoder's `conv_s0`/`conv_s1`).
pub struct InterPix {
    pix: Tensor,
    s0: Tensor,
    s1: Tensor,
}

/// Everything the tracker reads about the current frame (upstream's `feature_cache[frame_idx]`).
pub struct FrameCtx {
    pub frame: usize,
    /// Propagation neck, top level `[1, 256, G, G]`.
    pub prop72: Tensor,
    /// `conv_s0` / `conv_s1` of the multiplex decoder on the propagation neck's finer levels.
    pub ps0: Tensor,
    pub ps1: Tensor,
    pub feats: Arc<FrameFeats>,
    /// Filled by [`Vos::interactive_pix`] before the first birth or recondition of the frame.
    pub inter: Option<InterPix>,
}

/// A frame's stored output (upstream's compact `current_out`).
#[derive(Clone)]
pub struct Stored {
    /// Object ids of the rows of `pred_masks` and `scores` (`local_obj_id_to_idx`).
    pub row_ids: Vec<i64>,
    /// `[rows, M * M]` logits, `M` = 4 x the grid, 288 at 1008 (`-1024` for an object the tracker scored as absent).
    pub pred_masks: Vec<f32>,
    /// Object-score logits per row.
    pub scores: Vec<f32>,
    /// Muxed object pointers `[16, 256]`.
    pub obj_ptr: Vec<f32>,
    /// State-local object indices that are conditioned on this frame.
    pub cond_objs: BTreeSet<usize>,
    /// `None` until a memory is encoded.
    pub maskmem: Option<MaskMem>,
    pub feats: Option<Arc<FrameFeats>>,
}

/// Masks the tracker predicted for one frame.
pub struct PropOut {
    /// `[n, M * M]`, state object order.
    pub low_res: Vec<f32>,
    /// `[n]` object-score logits.
    pub scores: Vec<f32>,
}

/// Temporary per-object masks between `add_new_masks` and the preflight.
struct Pending {
    frame: usize,
    /// Whether the frame went to the conditioning storage.
    is_cond: bool,
    /// `(object index, foreground after the same-frame suppression at video resolution)`.
    masks: Vec<(usize, Vec<bool>)>,
    video: (usize, usize),
}

/// One tracker inference state.
#[derive(Default)]
pub struct TrackerState {
    obj_ids: Vec<i64>,
    slots: Vec<i32>,
    cond: BTreeMap<usize, Stored>,
    non_cond: BTreeMap<usize, Stored>,
    consolidated_cond: BTreeSet<usize>,
    consolidated_non_cond: BTreeSet<usize>,
    tracked: BTreeSet<usize>,
    inputs: Vec<BTreeSet<usize>>,
    pending: Option<Pending>,
    /// Set when an object removal left no conditioning frame: upstream then wipes the stored
    /// outputs, and its next `propagate_in_video` raises `RuntimeError` ("No points are
    /// provided"). The port reports that as an error instead of continuing.
    broken: bool,
}

/// `multiplex_state.mux` for one bucket: object-order rows to slot order, zeros elsewhere.
fn mux(slot_of: &[usize], x: &[f32], dim: usize) -> Vec<f32> {
    let mut out = vec![0f32; MUX * dim];
    for (o, &s) in slot_of.iter().enumerate() {
        out[s * dim..(s + 1) * dim].copy_from_slice(&x[o * dim..(o + 1) * dim]);
    }
    out
}

/// `multiplex_state.demux` for one bucket: slot order to object order, padding and removed slots dropped.
fn demux(slot_of: &[usize], x: &[f32], dim: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(slot_of.len() * dim);
    for &s in slot_of {
        out.extend_from_slice(&x[s * dim..(s + 1) * dim]);
    }
    out
}

/// Plane `k` of `dst` is `sigmoid(x) * 2 - 1` of plane `src_rows[k]` of `hr`, `px` values per
/// plane: the memory encoder's mask input. Threaded with [`host::for_each_chunk_mut`]; each
/// element comes from its own source element alone, so the split does not change a bit.
fn fill_sigmoid_planes(dst: &mut [f32], hr: &[f32], src_rows: &[usize], px: usize) {
    assert_eq!(
        dst.len(),
        src_rows.len() * px,
        "fill_sigmoid_planes: destination size"
    );
    host::for_each_chunk_mut(dst, FILL_MIN_PER_THREAD, |start, piece| {
        let (mut j, mut rest) = (start, piece);
        while !rest.is_empty() {
            let (k, i) = (j / px, j % px);
            let len = (px - i).min(rest.len());
            let (head, tail) = std::mem::take(&mut rest).split_at_mut(len);
            let src = &hr[src_rows[k] * px + i..src_rows[k] * px + i + len];
            for (d, &x) in head.iter_mut().zip(src) {
                *d = host::sigmoid(x) * 2.0 - 1.0;
            }
            (j, rest) = (j + len, tail);
        }
    });
}

/// The elements per thread from which [`fill_sigmoid_planes`] splits its work.
const FILL_MIN_PER_THREAD: usize = 1 << 14;

impl TrackerState {
    pub fn new() -> Self {
        Self {
            slots: vec![PADDING; MUX],
            ..Self::default()
        }
    }

    /// Object ids in state order (ascending).
    pub fn obj_ids(&self) -> &[i64] {
        &self.obj_ids
    }

    pub fn len(&self) -> usize {
        self.obj_ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.obj_ids.is_empty()
    }

    /// `multiplex_state.available_slots`: empty slots of the bucket (removed slots do not count).
    pub fn available_slots(&self) -> usize {
        self.slots.iter().filter(|&&s| s == PADDING).count()
    }

    /// The bucket's assignment: object index, [`PADDING`] or [`REMOVED`] per slot.
    pub fn slots(&self) -> &[i32] {
        &self.slots
    }

    /// The stored output of `frame` (conditioning storage first, as upstream looks it up).
    pub fn stored(&self, frame: usize) -> Option<&Stored> {
        self.cond.get(&frame).or_else(|| self.non_cond.get(&frame))
    }

    /// Slot of each valid object, in object order.
    fn slot_of(&self) -> Vec<usize> {
        let mut out = vec![usize::MAX; self.obj_ids.len()];
        for (s, &o) in self.slots.iter().enumerate() {
            if o >= 0 {
                out[o as usize] = s;
            }
        }
        out
    }

    /// `add_new_masks`: bring objects `ids` into this state on the current frame from their
    /// binary masks at the model's input size (`masks`: `[n, I * I]`, `I` = 16 x the grid, 1152 at 1008; 0 or 1).
    ///
    /// `reconditioning == false`: births. A fresh state takes the objects as its conditioning
    /// frame (`multiplex_controller.get_state`: object k to slot k); an existing state that has
    /// already propagated this frame appends them to its stored output as non-conditioning
    /// (`add_all_frames_to_correct_as_cond` is False), filling the first empty slots.
    /// `reconditioning == true`: the objects exist; their stored mask, score and pointer on this
    /// frame are replaced and they join the frame's conditioning set.
    /// Neither re-encodes memory (`run_mem_encoder=False`); call [`Self::preflight`] next.
    ///
    /// Chain per upstream: `masks` -> (antialiased bilinear) video-size mask `> 0.5` -> `+-1024`,
    /// where pixels any *other* object of the call covers are forced to `-1024`; the
    /// interactive decoder turns the 1152 mask into the object pointer (double no-object gating:
    /// the decoder's own score, then the mask's emptiness); the stored low-res mask is the
    /// antialiased 288 resize of `mask * 20 - 10`.
    pub fn add_new_masks<B: Backend + ?Sized>(
        &mut self,
        vos: &Vos<'_, B>,
        ctx: &FrameCtx,
        ids: &[i64],
        masks: &[f32],
        video: (usize, usize),
        reconditioning: bool,
    ) -> Result<(), ModelError> {
        let n = ids.len();
        if n == 0 {
            return Ok(());
        }
        let frame = ctx.frame;
        let ip = ctx.inter.as_ref().ok_or_else(|| {
            err("add_new_masks: the frame's interactive features are not computed")
        })?;
        let (h, w) = video;
        let mx = vos.geo.input_mask();
        if masks.len() != n * mx * mx {
            return Err(err(format!(
                "add_new_masks: {} values for {n} masks of {mx}x{mx}",
                masks.len()
            )));
        }
        if self.pending.is_some() {
            return Err(err(
                "add_new_masks: the previous call was not consolidated by preflight",
            ));
        }
        if self.slots.is_empty() {
            self.slots = vec![PADDING; MUX];
        }
        let is_new_state = self.obj_ids.is_empty();
        let idxs: Vec<usize> = if reconditioning {
            ids.iter()
                .map(|id| {
                    self.obj_ids.iter().position(|x| x == id).ok_or_else(|| {
                        err(format!(
                            "recondition of object id {id}, which is not in this state"
                        ))
                    })
                })
                .collect::<Result<_, _>>()?
        } else {
            (self.obj_ids.len()..self.obj_ids.len() + n).collect()
        };
        if !reconditioning && n > self.available_slots() {
            return Err(err(format!("{n} new objects but {} empty slots; upstream would open a second bucket, which best-fit placement never needs", self.available_slots())));
        }

        let (ptrs, appearing, low) = vos.pointers_from_masks(ip, masks, n)?;
        let scores: Vec<f32> = appearing
            .iter()
            .map(|&a| if a { 10.0 } else { -10.0 })
            .collect();
        let vid = st_video::resize_bilinear_aa_f32(masks, n, mx, mx, h, w);
        let hw = h * w;
        let bin: Vec<Vec<bool>> = (0..n)
            .map(|o| vid[o * hw..(o + 1) * hw].iter().map(|&v| v > 0.5).collect())
            .collect();

        let is_cond = !self.tracked.contains(&frame);
        let low_res = vos.geo.low_res();
        let px = low_res * low_res;
        if is_new_state {
            for k in 0..n {
                self.slots[k] = k as i32;
            }
            self.obj_ids = ids.to_vec();
            let slot_of = self.slot_of();
            let out = Stored {
                row_ids: ids.to_vec(),
                pred_masks: low,
                scores,
                obj_ptr: mux(&slot_of, &ptrs, D),
                cond_objs: (0..n).collect(),
                maskmem: None,
                feats: None,
            };
            self.inputs = vec![BTreeSet::new(); n];
            if is_cond {
                if self.non_cond.remove(&frame).is_some() {
                    self.consolidated_non_cond.remove(&frame);
                }
                self.cond.insert(frame, out);
            } else {
                self.non_cond.insert(frame, out);
            }
        } else {
            let old_slot_of = self.slot_of();
            if !reconditioning {
                for &idx in &idxs {
                    let s = self
                        .slots
                        .iter()
                        .position(|&v| v == PADDING)
                        .ok_or_else(|| err("no empty slot"))?;
                    self.slots[s] = idx as i32;
                }
                self.obj_ids.extend_from_slice(ids);
            }
            let new_slot_of = self.slot_of();
            let in_cond = self.cond.contains_key(&frame);
            let existing = (if in_cond {
                self.cond.get_mut(&frame)
            } else {
                self.non_cond.get_mut(&frame)
            })
            .ok_or_else(|| {
                err(format!(
                    "no stored output on frame {frame} to modify in place"
                ))
            })?;
            let mut pointers = demux(&old_slot_of, &existing.obj_ptr, D);
            if reconditioning {
                for (k, &idx) in idxs.iter().enumerate() {
                    existing.pred_masks[idx * px..(idx + 1) * px]
                        .copy_from_slice(&low[k * px..(k + 1) * px]);
                    existing.scores[idx] = scores[k];
                    pointers[idx * D..(idx + 1) * D].copy_from_slice(&ptrs[k * D..(k + 1) * D]);
                }
            } else {
                existing.pred_masks.extend_from_slice(&low);
                existing.scores.extend_from_slice(&scores);
                existing.row_ids.extend_from_slice(ids);
                pointers.extend_from_slice(&ptrs);
            }
            existing.obj_ptr = mux(&new_slot_of, &pointers, D);
            existing.cond_objs.extend(idxs.iter().copied());
            // `compact_current_out` is a fresh dict stored under the frame's storage key.
            let copy = existing.clone();
            if is_cond {
                if self.non_cond.remove(&frame).is_some() {
                    self.consolidated_non_cond.remove(&frame);
                }
                self.cond.insert(frame, copy);
            } else {
                self.non_cond.insert(frame, copy);
            }
        }
        if is_cond {
            self.consolidated_cond.insert(frame);
        } else {
            self.consolidated_non_cond.insert(frame);
        }
        self.inputs.resize(self.obj_ids.len(), BTreeSet::new());
        for &idx in &idxs {
            self.inputs[idx].insert(frame);
        }

        // Temporary per-object masks: each object loses the pixels any other object of this call covers.
        let mut cover = vec![0u16; hw];
        for b in &bin {
            for (c, &v) in cover.iter_mut().zip(b) {
                *c += u16::from(v);
            }
        }
        let masks_pending = idxs
            .iter()
            .enumerate()
            .map(|(k, &idx)| {
                (
                    idx,
                    (0..hw)
                        .map(|p| bin[k][p] && cover[p] - u16::from(bin[k][p]) == 0)
                        .collect::<Vec<bool>>(),
                )
            })
            .collect();
        self.pending = Some(Pending {
            frame,
            is_cond,
            masks: masks_pending,
            video,
        });
        Ok(())
    }

    /// `propagate_in_video_preflight`: consolidate the pending per-object masks into the frame's
    /// stored output and, with `run_mem_encoder`, re-encode its memory from them.
    ///
    /// The consolidated low-res masks take the pending objects' masks (video size `+-1024`,
    /// antialiased to 288) and every other object's stored row unchanged; the memory encode
    /// resizes those to 1008 (bilinear), applies the hard pixel-wise non-overlap constraint,
    /// scores each object by its *stored* object-score logit and conditions the objects of the
    /// stored conditioning set. A frame that ends up in the conditioning storage loses any copy
    /// in the non-conditioning storage (`VD:2324-2333`): a recondition of the state's own
    /// conditioning frame is therefore stored by the in-place update alone.
    ///
    /// `run_mem_encoder = false` is for the recondition: the memory it would encode is always
    /// overwritten by the frame's memory update before anything reads it.
    pub fn preflight<B: Backend + ?Sized>(
        &mut self,
        vos: &Vos<'_, B>,
        ctx: &FrameCtx,
        run_mem_encoder: bool,
    ) -> Result<(), ModelError> {
        let Some(p) = self.pending.take() else {
            return Ok(());
        };
        let be = vos.be;
        let frame = p.frame;
        let (low_res, image) = (vos.geo.low_res(), vos.geo.pixels());
        let px = low_res * low_res;
        let n = self.obj_ids.len();
        let all_out = self
            .cond
            .get(&frame)
            .or_else(|| self.non_cond.get(&frame))
            .ok_or_else(|| err(format!("preflight: no stored output on frame {frame}")))?;
        if all_out.row_ids.len() != n {
            return Err(err(format!(
                "preflight: stored output has {} rows for {n} objects",
                all_out.row_ids.len()
            )));
        }
        let mut merged = all_out.clone();
        let (h, w) = p.video;
        for (idx, fg) in &p.masks {
            let plane: Vec<f32> = fg
                .iter()
                .map(|&b| if b { -NO_OBJ_SCORE } else { NO_OBJ_SCORE })
                .collect();
            let row = st_video::resize_bilinear_aa_f32(&plane, 1, h, w, low_res, low_res);
            merged.pred_masks[idx * px..(idx + 1) * px].copy_from_slice(&row);
        }
        if run_mem_encoder {
            let low = up(be, &merged.pred_masks, &[n, 1, low_res, low_res])?;
            let upsampled = be.interpolate_bilinear_forward(&low, [image, image], false)?;
            drop(low);
            let mut hr_host = upsampled.to_host(be.budget())?;
            drop(upsampled);
            let hr = hr_host.f32_slice_mut()?;
            crate::tracker::vos::non_overlap(hr, n, image * image);
            let slot_of = self.slot_of();
            let rows: Vec<usize> = (0..n).collect();
            merged.maskmem = Some(vos.encode_mask_memory(
                ctx,
                hr,
                &rows,
                image,
                &merged.scores,
                &merged.cond_objs,
                &slot_of,
                &mut |_| {},
            )?);
            merged.feats = Some(ctx.feats.clone());
        }
        if p.is_cond {
            self.cond.insert(frame, merged);
        } else {
            self.non_cond.insert(frame, merged);
        }
        let cond_frames: Vec<usize> = self.cond.keys().copied().collect();
        for f in cond_frames {
            if self.non_cond.remove(&f).is_some() {
                self.consolidated_non_cond.remove(&f);
            }
        }
        Ok(())
    }

    /// `_run_memory_encoder` as the frame's memory update calls it: encode the state's masks at
    /// 1152 with the given object-score logits (state object order) and store the result in the
    /// frame's output (both storages, as upstream loops over them). `hr` holds `1152 * 1152`
    /// logit planes, and the state's object `k` is plane `rows[k]`, so a caller holding every
    /// track's planes passes them without gathering a copy. The conditioning set is the stored
    /// output's (`conditioning_objects=None` lookup). `mark` is called with each
    /// [`ENCODE_STAGES`] name as that phase begins (a profiler's marker; `&mut |_| {}` otherwise).
    pub fn encode_memory<B: Backend + ?Sized>(
        &mut self,
        vos: &Vos<'_, B>,
        ctx: &FrameCtx,
        hr: &[f32],
        rows: &[usize],
        scores: &[f32],
        mark: &mut dyn FnMut(&str),
    ) -> Result<(), ModelError> {
        let frame = ctx.frame;
        let cond = self
            .stored(frame)
            .map(|s| s.cond_objs.clone())
            .ok_or_else(|| err(format!("conditioning objects not found at frame {frame}")))?;
        let slot_of = self.slot_of();
        let mm = vos.encode_mask_memory(
            ctx,
            hr,
            rows,
            vos.geo.input_mask(),
            scores,
            &cond,
            &slot_of,
            mark,
        )?;
        for map in [&mut self.cond, &mut self.non_cond] {
            if let Some(st) = map.get_mut(&frame) {
                st.maskmem = Some(mm.clone());
                st.feats = Some(ctx.feats.clone());
            }
        }
        Ok(())
    }

    /// `propagate_in_video` for one frame with no memory encoding. A frame whose stored output
    /// was consolidated (the state was created or modified on it) returns that output; any other
    /// frame runs memory attention over the stored memories and pointers, then the multiplex
    /// decoder, and stores the result as a non-conditioning output without memory.
    /// `num_frames` is upstream's video length (the object-pointer window is `min(num_frames, 16)`).
    /// `mark` is called with each [`PROPAGATE_STAGES`] name as that phase begins (a profiler's
    /// marker; `&mut |_| {}` otherwise).
    pub fn propagate_one_frame<B: Backend + ?Sized>(
        &mut self,
        vos: &Vos<'_, B>,
        ctx: &FrameCtx,
        num_frames: usize,
        mark: &mut dyn FnMut(&str),
    ) -> Result<PropOut, ModelError> {
        let frame = ctx.frame;
        if self.broken || self.cond.is_empty() {
            return Err(err("propagate: no conditioning frame in this state (upstream raises RuntimeError: No points are provided; please add points first)"));
        }
        let n = self.obj_ids.len();
        let reuse = if self.consolidated_cond.contains(&frame) {
            Some(self.cond.get(&frame).ok_or_else(|| {
                err(format!(
                    "frame {frame} is consolidated but has no conditioning output"
                ))
            })?)
        } else if self.consolidated_non_cond.contains(&frame) {
            Some(self.non_cond.get(&frame).ok_or_else(|| {
                err(format!(
                    "frame {frame} is consolidated but has no non-conditioning output"
                ))
            })?)
        } else {
            None
        };
        let out = match reuse {
            Some(st) => {
                if st.row_ids.len() != n
                    || st.pred_masks.len() != n * vos.geo.low_res() * vos.geo.low_res()
                {
                    return Err(err(format!(
                        "stored output of frame {frame} has {} rows for {n} objects",
                        st.row_ids.len()
                    )));
                }
                PropOut {
                    low_res: st.pred_masks.clone(),
                    scores: st.scores.clone(),
                }
            }
            None => {
                let st = self.compute_propagation(vos, ctx, num_frames, mark)?;
                let out = PropOut {
                    low_res: st.pred_masks.clone(),
                    scores: st.scores.clone(),
                };
                self.non_cond.insert(frame, st);
                out
            }
        };
        self.tracked.insert(frame);
        Ok(out)
    }

    /// Memory-conditioned features, multiplex decode and object pointers for the current frame.
    fn compute_propagation<B: Backend + ?Sized>(
        &self,
        vos: &Vos<'_, B>,
        ctx: &FrameCtx,
        num_frames: usize,
        mark: &mut dyn FnMut(&str),
    ) -> Result<Stored, ModelError> {
        mark(PROPAGATE_STAGES[0]);
        let be = vos.be;
        let frame = ctx.frame;
        if self.cond.len() > MAX_COND_IN_ATTN {
            return Err(err(format!("{} conditioning frames; selecting the closest {MAX_COND_IN_ATTN} is not ported (a state has one)", self.cond.len())));
        }
        let tpos = |t_pos: i64| -> usize {
            if t_pos <= 0 || t_pos >= NUM_MASKMEM as i64 {
                NUM_MASKMEM - 1
            } else {
                NUM_MASKMEM - t_pos as usize - 1
            }
        };
        // Memory frames: the conditioning frames (t_pos = distance), then t_pos 1..6 -> frame - (7 - t_pos).
        let mut entries: Vec<(&Stored, i64)> = Vec::new();
        for (&c, st) in &self.cond {
            if st.maskmem.is_some() {
                entries.push((st, frame as i64 - c as i64));
            }
        }
        for t_pos in 1..NUM_MASKMEM {
            let prev = frame as i64 - (NUM_MASKMEM - t_pos) as i64;
            if prev < 0 {
                continue;
            }
            if let Some(st) = self
                .non_cond
                .get(&(prev as usize))
                .filter(|s| s.maskmem.is_some())
            {
                entries.push((st, t_pos as i64));
            }
        }
        let feat = vos.geo.grid;
        let image_dev = &ctx.feats.tokens;
        let pix = if entries.is_empty() {
            // No memory to attend to: upstream falls back to the frame's own features.
            ctx.prop72.clone()
        } else {
            // Each stored memory brings its projections, computed once by the first frame that
            // attends it, and its temporal position row; the pointer block is uploaded after them.
            // An external memory attention takes the stored frames' raw inputs on the host instead.
            let ext = vos.ext_mem_attn.as_deref();
            let mut frames: Vec<(&FrameKv, &Tensor)> = Vec::with_capacity(entries.len());
            let mut host_frames: Vec<HostFrame<'_>> = Vec::new();
            for &(st, t_pos) in &entries {
                let feats = st
                    .feats
                    .as_ref()
                    .ok_or_else(|| err("a stored memory without its image features"))?;
                let mm = st.maskmem.as_ref().expect("filtered on maskmem");
                if ext.is_some() {
                    host_frames.push(HostFrame {
                        image: feats.host(be.budget())?,
                        memory: mm.host_tokens(),
                        pos: &vos.memory_pos_host[tpos(t_pos)],
                    });
                } else {
                    frames.push((mm.kv(vos, feats)?, &vos.memory_pos_tokens[tpos(t_pos)]));
                }
            }
            // Object pointers: the conditioning frames at |distance|, then frames before the current one.
            let max_ptr = num_frames.min(MAX_PTRS);
            let mut ptr_frames: Vec<(&Stored, usize)> = self
                .cond
                .iter()
                .map(|(&c, st)| (st, frame.abs_diff(c)))
                .collect();
            for t_diff in 1..max_ptr {
                if t_diff > frame {
                    break;
                }
                if let Some(st) = self.non_cond.get(&(frame - t_diff)) {
                    ptr_frames.push((st, t_diff));
                }
            }
            let t_diff_max = (max_ptr - 1).max(1) as f32;
            let (mut ptrs, mut ptrs_pos) = (
                Vec::with_capacity(ptr_frames.len() * MUX * D),
                Vec::with_capacity(ptr_frames.len() * MUX * D),
            );
            for &(st, pos) in &ptr_frames {
                let pe = vos
                    .obj_ptr_tpos_proj
                    .host(&sine_1d(pos as f32 / t_diff_max, D), 1);
                ptrs.extend_from_slice(&st.obj_ptr);
                for _ in 0..MUX {
                    ptrs_pos.extend_from_slice(&pe);
                }
            }
            let n_ptr = ptr_frames.len() * MUX;
            let fused = if let Some(ext) = ext {
                let image = ctx.feats.host(be.budget())?;
                mark(PROPAGATE_STAGES[1]);
                let out = ext.run(&HostMemInputs {
                    image,
                    src_pos: &vos.pos_host,
                    frames: &host_frames,
                    ptrs: &ptrs,
                    ptr_pos: &ptrs_pos,
                })?;
                up(be, &out, &[1, feat * feat, D])?
            } else {
                let ptr_dev = if n_ptr > 0 {
                    Some((
                        up(be, &ptrs, &[1, n_ptr, D])?,
                        up(be, &ptrs_pos, &[1, n_ptr, D])?,
                    ))
                } else {
                    None
                };
                mark(PROPAGATE_STAGES[1]);
                vos.mem_attn.fwd_frames(
                    be,
                    &FramesInputs {
                        image: image_dev,
                        src_pos: &vos.pos_tokens,
                        frames: &frames,
                        ptrs: ptr_dev.as_ref().map(|(p, pp)| (p, pp)),
                    },
                )?
            };
            be.permute(&fused, &[0, 2, 1])?
                .reshape(&[1, D, feat, feat])?
        };

        // Multiplex decode: slots with an object get the valid embedding, the rest the invalid one.
        mark(PROPAGATE_STAGES[2]);
        let n = self.obj_ids.len();
        let slot_of = self.slot_of();
        let extra: Vec<f32> = (0..MUX)
            .flat_map(|s| {
                if self.slots[s] >= 0 {
                    vos.output_valid[s * D..(s + 1) * D].to_vec()
                } else {
                    vos.output_invalid[s * D..(s + 1) * D].to_vec()
                }
            })
            .collect();
        let out = vos
            .mux_decoder
            .fwd(be, &pix, &vos.prop_pe, &ctx.ps0, &ctx.ps1, &extra)?;
        drop(pix);
        mark(PROPAGATE_STAGES[3]);
        let masks_view = host_slice(&out.masks, be)?; // [1, 16, 3, M, M]
        let masks = masks_view.as_slice();
        let px = vos.geo.low_res() * vos.geo.low_res();
        let mut low = vec![0f32; n * px];
        let mut ptrs = vec![0f32; n * D];
        let mut scores = vec![0f32; n];
        for o in 0..n {
            let s = slot_of[o];
            let ious = &out.iou_pred[s * MASKS_PER_OBJ..(s + 1) * MASKS_PER_OBJ];
            // torch.argmax: the first index of the maximum.
            let mut best = 0;
            for m in 1..MASKS_PER_OBJ {
                if ious[m] > ious[best] {
                    best = m;
                }
            }
            let score = out.object_score_logits[s];
            scores[o] = score;
            let src = &masks[(s * MASKS_PER_OBJ + best) * px..(s * MASKS_PER_OBJ + best + 1) * px];
            let dst = &mut low[o * px..(o + 1) * px];
            if score > 0.0 {
                dst.copy_from_slice(src);
            } else {
                dst.iter_mut().for_each(|v| *v = NO_OBJ_SCORE);
            }
            let tok =
                &out.sam_tokens[(s * MASKS_PER_OBJ + best) * D..(s * MASKS_PER_OBJ + best + 1) * D];
            let ptr = vos.obj_ptr_proj.host(tok, 1);
            let ptr = if score > 0.0 {
                ptr
            } else {
                vos.no_obj_ptr_linear.host(&ptr, 1)
            };
            ptrs[o * D..(o + 1) * D].copy_from_slice(&ptr);
        }
        drop(masks_view);
        drop(out);
        Ok(Stored {
            row_ids: self.obj_ids.clone(),
            pred_masks: low,
            scores,
            obj_ptr: mux(&slot_of, &ptrs, D),
            cond_objs: BTreeSet::new(),
            maskmem: None,
            feats: None,
        })
    }

    /// `remove_objects` (`strict=False`, `need_output=False`). Returns whether the state still
    /// holds objects; the caller drops a state that does not (`_tracker_remove_objects`).
    ///
    /// Order as upstream: (0) clear each removed object's mask inputs; a frame that no object
    /// has an input on any more stops being a conditioning frame (its output moves to the
    /// non-conditioning storage), and when no conditioning frame is left everything stored is
    /// wiped ([`Self::propagate_one_frame`] then reports upstream's `RuntimeError`); (1) renumber
    /// the objects densely; (2) mark the slots [`REMOVED`] (never reused) and renumber the
    /// remaining objects in the assignment; (3) drop the removed rows from every stored output.
    /// Stored memories and pointers are not edited: one bucket stays one bucket, so past frames
    /// keep the removed object's contribution in `maskmem` and its stale pointer in its slot.
    pub fn remove_objects(&mut self, ids: &[i64]) -> Result<bool, ModelError> {
        let mut rm: Vec<usize> = ids
            .iter()
            .filter_map(|id| self.obj_ids.iter().position(|x| x == id))
            .collect();
        rm.sort_unstable();
        rm.dedup();
        if rm.is_empty() {
            return Ok(!self.obj_ids.is_empty());
        }
        let n_before = self.obj_ids.len();
        self.inputs.resize(n_before, BTreeSet::new());
        for &idx in &rm {
            let frames: Vec<usize> = self.inputs[idx].iter().copied().collect();
            for f in frames {
                self.inputs[idx].remove(&f);
                if (0..n_before).any(|j| self.inputs[j].contains(&f)) {
                    continue;
                }
                self.consolidated_cond.remove(&f);
                self.consolidated_non_cond.remove(&f);
                if let Some(out) = self.cond.remove(&f) {
                    self.non_cond.insert(f, out);
                    self.tracked.remove(&f);
                }
                if self.cond.is_empty() {
                    self.cond.clear();
                    self.non_cond.clear();
                    self.consolidated_cond.clear();
                    self.consolidated_non_cond.clear();
                    self.tracked.clear();
                    self.inputs.iter_mut().for_each(|s| s.clear());
                    self.broken = true;
                }
            }
        }
        let remaining: Vec<usize> = (0..n_before).filter(|i| !rm.contains(i)).collect();
        if remaining.is_empty() {
            self.obj_ids.clear();
            return Ok(false);
        }
        let removed_ids: BTreeSet<i64> = rm.iter().map(|&i| self.obj_ids[i]).collect();
        let rank: BTreeMap<usize, usize> = remaining
            .iter()
            .enumerate()
            .map(|(new, &old)| (old, new))
            .collect();
        for s in self.slots.iter_mut() {
            if *s >= 0 {
                *s = match rank.get(&(*s as usize)) {
                    Some(&new) => new as i32,
                    None => REMOVED,
                };
            }
        }
        self.obj_ids = remaining.iter().map(|&i| self.obj_ids[i]).collect();
        self.inputs = remaining
            .iter()
            .map(|&i| std::mem::take(&mut self.inputs[i]))
            .collect();
        for map in [&mut self.cond, &mut self.non_cond] {
            for st in map.values_mut() {
                let keep: Vec<usize> = (0..st.row_ids.len())
                    .filter(|&r| !removed_ids.contains(&st.row_ids[r]))
                    .collect();
                if !st.pred_masks.is_empty() {
                    // One row is one mask's logits; its length is whatever resolution stored it.
                    let px = st.pred_masks.len() / st.row_ids.len();
                    st.pred_masks = keep
                        .iter()
                        .flat_map(|&r| st.pred_masks[r * px..(r + 1) * px].iter().copied())
                        .collect();
                }
                st.scores = keep.iter().map(|&r| st.scores[r]).collect();
                let local: BTreeMap<usize, usize> = keep
                    .iter()
                    .enumerate()
                    .map(|(new, &old)| (old, new))
                    .collect();
                st.cond_objs = st
                    .cond_objs
                    .iter()
                    .filter_map(|o| local.get(o).copied())
                    .collect();
                st.row_ids = keep.iter().map(|&r| st.row_ids[r]).collect();
            }
        }
        Ok(true)
    }

    /// Drop what no later step can read once `current` is done: older frames' `pred_masks`,
    /// the memory-attention projections of non-conditioning frames before the next frame's
    /// 7-frame memory, and non-conditioning outputs beyond the 16-frame pointer window.
    pub fn prune(&mut self, current: usize) {
        for map in [&mut self.cond, &mut self.non_cond] {
            for (&f, st) in map.iter_mut() {
                if f < current {
                    st.pred_masks = Vec::new();
                }
            }
        }
        // Frame current + 1 attends non-conditioning frames current - 5 ..= current.
        for (&f, st) in self.non_cond.iter_mut() {
            if f + NUM_MASKMEM <= current + 1 {
                if let Some(mm) = st.maskmem.as_mut() {
                    mm.kv = Arc::default();
                }
            }
        }
        self.non_cond.retain(|&f, _| f + MAX_PTRS > current);
    }
}

/// Outputs of [`Vos::interactive_pix`] and friends for the session.
impl<'b, B: Backend + ?Sized> Vos<'b, B> {
    /// The frame's propagation-side inputs.
    pub fn frame_ctx(&self, frame: usize, prop: &[Tensor; 3]) -> Result<FrameCtx, ModelError> {
        let (ps0, ps1) = high_res(self.be, &self.mux_decoder, &prop[0], &prop[1])?;
        let tokens = self
            .be
            .permute(&prop[2].reshape(&[1, D, self.geo.tokens()])?, &[0, 2, 1])?;
        Ok(FrameCtx {
            frame,
            prop72: prop[2].clone(),
            ps0,
            ps1,
            feats: Arc::new(FrameFeats::new(tokens)),
            inter: None,
        })
    }

    /// The interactive features a birth or a recondition needs (`inter` is the interactive neck
    /// of the frame): the top level plus `interactivity_no_mem_embed`, and the two skips.
    pub fn interactive_pix(&self, inter: &[Tensor; 3]) -> Result<InterPix, ModelError> {
        let be = self.be;
        let pix = be.broadcast_add_forward(&inter[2], &self.no_mem_embed_device)?;
        let p1 = Conv2dParams::new();
        let s0 = be.conv2d_forward(&inter[0], &self.i_conv_s0.0, Some(&self.i_conv_s0.1), p1)?;
        let s1 = be.conv2d_forward(&inter[1], &self.i_conv_s1.0, Some(&self.i_conv_s1.1), p1)?;
        Ok(InterPix { pix, s0, s1 })
    }

    /// `_use_mask_as_output` for `n` binary masks `[n, I * I]` (`I` = 1152 at 1008): object pointers
    /// `[n, 256]`, whether each mask has any foreground, and the low-res mask logits
    /// `[n, M * M]` (`M` = 288; the antialiased resize of `mask * 20 - 10`).
    pub(crate) fn pointers_from_masks(
        &self,
        ip: &InterPix,
        masks: &[f32],
        n: usize,
    ) -> Result<(Vec<f32>, Vec<bool>, Vec<f32>), ModelError> {
        let be = self.be;
        let (mx, low_res) = (self.geo.input_mask(), self.geo.low_res());
        let appearing: Vec<bool> = (0..n)
            .map(|o| {
                masks[o * mx * mx..(o + 1) * mx * mx]
                    .iter()
                    .any(|&v| v > 0.0)
            })
            .collect();
        let hi: Vec<f32> = masks.iter().map(|&m| m * 20.0 - 10.0).collect();
        let low = st_video::resize_bilinear_aa_f32(&hi, n, mx, mx, low_res, low_res);
        drop(hi);

        let mut k4 = Conv2dParams::new();
        k4.stride = [4, 4];
        let masks_dev = up(be, masks, &[n, 1, mx, mx])?;
        let down = be.conv2d_forward(
            &masks_dev,
            &self.mask_downsample.w,
            Some(&self.mask_downsample.b),
            k4,
        )?;
        drop(masks_dev);
        let ds = down.shape()[2];
        let mut x = if ds == low_res {
            down
        } else {
            let down_host = host_slice(&down, be)?;
            let resized = st_video::resize_bilinear_aa_f32(&down_host, n, ds, ds, low_res, low_res);
            drop((down_host, down));
            up(be, &resized, &[n, 1, low_res, low_res])?
        };
        let p1 = Conv2dParams::new();
        let mut k2 = Conv2dParams::new();
        k2.stride = [2, 2];
        for (i, (conv, ln)) in self.pe_downscale.iter().enumerate() {
            x = be.conv2d_forward(&x, &conv.w, Some(&conv.b), if i < 2 { k2 } else { p1 })?;
            if let Some((lw, lb)) = ln {
                x = be.gelu_erf_forward(&be.channel_layer_norm_forward(&x, lw, lb, LN2D_EPS)?)?;
            }
        }
        let sparse: Vec<f32> = (0..2 * n).flat_map(|_| self.not_a_point.clone()).collect();
        let io =
            self.interactive
                .fwd(be, &ip.pix, &self.inter_pe, &sparse, 2, &x, &ip.s0, &ip.s1)?;
        drop(x);
        let mut ptrs = vec![0f32; n * D];
        for o in 0..n {
            let tok = &io.mask_tokens[o * 4 * D..o * 4 * D + D]; // mask token 0
            let mut ptr = self.interactive_obj_ptr_proj.host(tok, 1);
            // `_forward_sam_heads` gates on the decoder's own object score, `_use_mask_as_output` again on the mask.
            if !(io.object_score_logits[o] > 0.0) {
                ptr = self.no_obj_ptr_linear.host(&ptr, 1);
            }
            if !appearing[o] {
                ptr = self.no_obj_ptr_linear.host(&ptr, 1);
            }
            ptrs[o * D..(o + 1) * D].copy_from_slice(&ptr);
        }
        Ok((ptrs, appearing, low))
    }

    /// `_encode_new_memory` + bf16 storage for `n` objects' mask logits, object `o` being plane
    /// `rows[o]` of `hr` (`size * size` values per plane)
    /// (`size` = the image side, 1008, for a consolidated frame, 1152 for the frame memory update;
    /// the encoder resizes to `16 * grid`). `scores`: the object-score logits that decide which slots get
    /// `no_obj_embed_spatial` (padding and removed slots count as score 0). `slot_of[o]`: slot of
    /// object `o`. `cond`: objects whose condition channel is 1.0. `mark`: as in
    /// [`TrackerState::encode_memory`].
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_mask_memory(
        &self,
        ctx: &FrameCtx,
        hr: &[f32],
        rows: &[usize],
        size: usize,
        scores: &[f32],
        cond: &BTreeSet<usize>,
        slot_of: &[usize],
        mark: &mut dyn FnMut(&str),
    ) -> Result<MaskMem, ModelError> {
        mark(ENCODE_STAGES[0]);
        let be = self.be;
        let tokens = self.geo.tokens();
        let px = size * size;
        let n = slot_of.len();
        if rows.len() != n || scores.len() != n || rows.iter().any(|&r| (r + 1) * px > hr.len()) {
            return Err(err(format!(
                "encode_mask_memory: rows {rows:?} of {} mask values and {} scores for {n} objects of {size}x{size}",
                hr.len(),
                scores.len()
            )));
        }
        let mut obj_of_slot = [None; MUX];
        for (o, &s) in slot_of.iter().enumerate() {
            if s >= MUX || obj_of_slot[s].replace(o).is_some() {
                return Err(err(format!(
                    "encode_mask_memory: slots {slot_of:?} are not distinct slots below {MUX}"
                )));
            }
        }
        let mut feat = match self.ext_mem_enc.as_deref() {
            Some(ext) if size == self.geo.input_mask() => {
                mark(ENCODE_STAGES[1]);
                mark(ENCODE_STAGES[2]);
                let pix = host_slice(&ctx.prop72, be)?;
                let mut flags = [0f32; MUX];
                for (s, o) in obj_of_slot.iter().enumerate() {
                    if o.is_some_and(|o| cond.contains(&o)) {
                        flags[s] = 1.0;
                    }
                }
                // The 16 slot planes in slot order, written straight into the encoder's input:
                // each slot's sigmoid mask, zeros for an empty slot, as the Ojas path builds them.
                let feat = ext.run(
                    pix.as_slice(),
                    &mut |buf| {
                        for (plane, o) in buf.chunks_exact_mut(px).zip(obj_of_slot) {
                            match o {
                                Some(o) => fill_sigmoid_planes(plane, hr, &[rows[o]], px),
                                None => plane.fill(0.0),
                            }
                        }
                    },
                    &flags,
                )?;
                if feat.len() != D * tokens {
                    return Err(err(format!(
                        "encode_mask_memory: the external encoder returned {} values, want {}",
                        feat.len(),
                        D * tokens
                    )));
                }
                let [e, o] = self.mem_enc_calls.get();
                self.mem_enc_calls.set([e + 1, o]);
                feat
            }
            _ => {
                // The encoder's input is 2 * MUX planes in slot order: each slot's sigmoid mask (0 for an
                // empty slot), then each slot's condition plane (1 for a conditioning object, else 0). Only
                // the n object planes cross to the device, in slot order; the constant planes are device
                // fills and the concat moves bits, so the input is the one the host used to build.
                let by_slot: Vec<usize> = obj_of_slot.iter().flatten().copied().collect();
                let objs = if n > 0 {
                    let mut t = Tensor::zeros(&[n, size, size], DType::F32, be.budget())?;
                    let src_rows: Vec<usize> = by_slot.iter().map(|&o| rows[o]).collect();
                    fill_sigmoid_planes(t.f32_slice_mut()?, hr, &src_rows, px);
                    mark(ENCODE_STAGES[1]);
                    Some(be.upload(&t)?)
                } else {
                    mark(ENCODE_STAGES[1]);
                    None
                };
                let zero = be.fill(&[1, size, size], DType::F32, 0.0)?;
                let one = if cond.is_empty() {
                    None
                } else {
                    Some(be.fill(&[1, size, size], DType::F32, 1.0)?)
                };
                let mut parts: Vec<Tensor> = Vec::with_capacity(2 * MUX);
                let (mut s, mut k) = (0, 0);
                while s < MUX {
                    if obj_of_slot[s].is_none() {
                        parts.push(zero.clone());
                        s += 1;
                        continue;
                    }
                    let start = k;
                    while s < MUX && obj_of_slot[s].is_some() {
                        s += 1;
                        k += 1;
                    }
                    let objs = objs.as_ref().ok_or_else(|| {
                        err("encode_mask_memory: an occupied slot with no object planes")
                    })?;
                    parts.push(if k - start == n {
                        objs.clone()
                    } else {
                        be.slice(objs, 0, start, k - start)?
                    });
                }
                for o in obj_of_slot {
                    parts.push(match (o, &one) {
                        (Some(o), Some(one)) if cond.contains(&o) => one.clone(),
                        _ => zero.clone(),
                    });
                }
                let refs: Vec<&Tensor> = parts.iter().collect();
                let masks = be.concat(&refs, 0)?.reshape(&[1, 2 * MUX, size, size])?;
                drop(refs);
                drop((parts, objs, zero, one));
                mark(ENCODE_STAGES[2]);
                let enc = self.mem_enc.fwd(be, &ctx.prop72, &masks)?;
                drop(masks);
                let feat_slice = host_slice(&enc.features, be)?;
                let feat = feat_slice.to_vec();
                drop((feat_slice, enc));
                let [e, o] = self.mem_enc_calls.get();
                self.mem_enc_calls.set([e, o + 1]);
                feat
            }
        };
        mark(ENCODE_STAGES[3]);
        let mut score_of_slot = [0f32; MUX];
        for (o, &s) in slot_of.iter().enumerate() {
            score_of_slot[s] = scores[o];
        }
        let mut add = vec![0f32; D];
        for (s, &score) in score_of_slot.iter().enumerate() {
            if !(score > 0.0) {
                add.iter_mut()
                    .zip(&self.no_obj_embed_spatial[s * D..(s + 1) * D])
                    .for_each(|(a, b)| *a += b);
            }
        }
        for c in 0..D {
            feat[c * tokens..(c + 1) * tokens]
                .iter_mut()
                .for_each(|v| *v += add[c]);
        }
        bf16_round(&mut feat);
        let tokens = up(be, &nchw_to_tokens(&feat, D, tokens), &[1, tokens, D])?;
        Ok(MaskMem {
            nchw: feat,
            tokens,
            kv: Arc::default(),
            host_tokens: Arc::default(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tracker::vos::LOW;

    #[test]
    fn threaded_sigmoid_fill_is_bit_identical_to_the_serial_one() {
        // Plane sizes that do not divide by the thread count and cross FILL_MIN_PER_THREAD, rows
        // out of order and repeated, a single plane, and values from -100 to 100 with the zeros,
        // infinities and a subnormal.
        let mut s = 0x2545_f491u32;
        for &(px, ref rows) in &[
            (70_001usize, vec![2usize, 0, 1]),
            (1152 * 1152 / 9 + 1, vec![0]),
            (13, vec![1, 1, 0]),
            (1, vec![0]),
        ] {
            let planes = rows.iter().max().map_or(0, |m| m + 1);
            let mut hr: Vec<f32> = (0..planes * px)
                .map(|_| {
                    s ^= s << 13;
                    s ^= s >> 17;
                    s ^= s << 5;
                    (s as f32 / u32::MAX as f32 - 0.5) * 200.0
                })
                .collect();
            hr.iter_mut()
                .zip([0.0, -0.0, f32::INFINITY, f32::NEG_INFINITY, 1e-40])
                .for_each(|(v, e)| *v = e);
            let mut want = Vec::with_capacity(rows.len() * px);
            for &r in rows {
                want.extend(
                    hr[r * px..(r + 1) * px]
                        .iter()
                        .map(|&x| host::sigmoid(x) * 2.0 - 1.0),
                );
            }
            let mut got = vec![f32::NAN; rows.len() * px];
            fill_sigmoid_planes(&mut got, &hr, rows, px);
            let differ = got
                .iter()
                .zip(&want)
                .position(|(a, b)| a.to_bits() != b.to_bits());
            assert_eq!(differ, None, "px {px} rows {rows:?}");
        }
        assert!(
            3 * 70_001 >= 2 * FILL_MIN_PER_THREAD,
            "the threaded path is exercised"
        );
    }

    #[test]
    fn mux_and_demux_are_inverse_on_valid_slots() {
        let slot_of = [3usize, 0, 7];
        let x: Vec<f32> = (0..3 * 2).map(|v| v as f32).collect();
        let m = mux(&slot_of, &x, 2);
        assert_eq!(m.len(), MUX * 2);
        assert_eq!(&m[6..8], &[0.0, 1.0]);
        assert_eq!(&m[0..2], &[2.0, 3.0]);
        assert_eq!(&m[14..16], &[4.0, 5.0]);
        assert!(m
            .iter()
            .enumerate()
            .all(|(i, &v)| matches!(i / 2, 0 | 3 | 7) || v == 0.0));
        assert_eq!(demux(&slot_of, &m, 2), x);
    }

    fn state_with(ids: &[i64]) -> TrackerState {
        let mut s = TrackerState::new();
        for (k, &id) in ids.iter().enumerate() {
            s.slots[k] = k as i32;
            s.obj_ids.push(id);
        }
        s.inputs = vec![BTreeSet::from([0usize]); ids.len()];
        let rows = ids.len();
        s.cond.insert(
            0,
            Stored {
                row_ids: ids.to_vec(),
                pred_masks: vec![0.0; rows * LOW * LOW],
                scores: (0..rows).map(|r| r as f32).collect(),
                obj_ptr: vec![0.0; MUX * D],
                cond_objs: (0..rows).collect(),
                maskmem: None,
                feats: None,
            },
        );
        s.consolidated_cond.insert(0);
        s
    }

    #[test]
    fn removal_marks_the_slot_removed_renumbers_and_slices_rows() {
        let mut s = state_with(&[10, 11, 12]);
        assert_eq!(s.available_slots(), MUX - 3);
        assert!(s.remove_objects(&[11, 99]).expect("remove"));
        assert_eq!(s.obj_ids(), &[10, 12]);
        assert_eq!(&s.slots()[..3], &[0, REMOVED, 1]);
        // A removed slot is not reusable: 13 free slots stay 13.
        assert_eq!(s.available_slots(), MUX - 3);
        let st = s.stored(0).expect("cond frame");
        assert_eq!(st.row_ids, vec![10, 12]);
        assert_eq!(st.scores, vec![0.0, 2.0]);
        assert_eq!(st.cond_objs, BTreeSet::from([0, 1]));
        assert_eq!(s.slot_of(), vec![0, 2]);
    }

    #[test]
    fn removing_every_object_empties_the_state() {
        let mut s = state_with(&[1, 2]);
        assert!(!s.remove_objects(&[1, 2]).expect("remove"));
        assert!(s.is_empty());
    }

    #[test]
    fn removing_the_last_object_with_an_input_on_the_conditioning_frame_wipes_the_state() {
        // Object 3 was born later (no input on frame 0); removing the two that were conditioned on
        // frame 0 leaves no conditioning frame, which upstream turns into a RuntimeError on the next propagate.
        let mut s = state_with(&[0, 1, 3]);
        s.inputs[2] = BTreeSet::from([5usize]);
        assert!(s.remove_objects(&[0, 1]).expect("remove"));
        assert!(s.broken);
        assert!(s.cond.is_empty() && s.non_cond.is_empty());
        // Keeping one conditioned object keeps the frame conditioning.
        let mut t = state_with(&[0, 1, 3]);
        t.inputs[2] = BTreeSet::from([5usize]);
        assert!(t.remove_objects(&[0]).expect("remove"));
        assert!(!t.broken && t.cond.contains_key(&0));
    }

    #[test]
    fn prune_drops_old_masks_and_frames_outside_the_window() {
        let mut s = state_with(&[0]);
        for f in 1..=20usize {
            let template = s.cond.get(&0).expect("cond").clone();
            s.non_cond.insert(f, template);
        }
        s.prune(20);
        assert!(s.non_cond.contains_key(&20) && s.non_cond.contains_key(&5));
        assert!(!s.non_cond.contains_key(&4));
        assert!(s.non_cond.get(&19).expect("19").pred_masks.is_empty());
        assert!(!s.non_cond.get(&20).expect("20").pred_masks.is_empty());
    }

    #[test]
    fn prune_drops_projections_before_the_next_frames_memory_only() {
        let mm = || MaskMem {
            nchw: Vec::new(),
            tokens: Tensor::from_f32(&[0.0], &[1, 1, 1], &ojas_core::Budget::new(1 << 20))
                .expect("tensor"),
            kv: Arc::default(),
            host_tokens: Arc::default(),
        };
        let mut s = state_with(&[0]);
        s.cond.get_mut(&0).expect("cond").maskmem = Some(mm());
        for f in 1..=20usize {
            let mut st = s.cond.get(&0).expect("cond").clone();
            st.maskmem = Some(mm());
            s.non_cond.insert(f, st);
        }
        let cache = |st: &Stored| Arc::clone(&st.maskmem.as_ref().expect("maskmem").kv);
        let before: BTreeMap<usize, _> = s.non_cond.iter().map(|(&f, st)| (f, cache(st))).collect();
        let cond_before = cache(s.cond.get(&0).expect("cond"));
        s.prune(20);
        // Frame 21 attends 15..=20: those keep their projections; 5..=14 stay for pointers only.
        for (f, st) in &s.non_cond {
            let kept = Arc::ptr_eq(&before[f], &cache(st));
            assert_eq!(kept, *f >= 15, "frame {f}: projections kept {kept}");
        }
        assert!(Arc::ptr_eq(
            &cond_before,
            &cache(s.cond.get(&0).expect("cond"))
        ));
    }
}
