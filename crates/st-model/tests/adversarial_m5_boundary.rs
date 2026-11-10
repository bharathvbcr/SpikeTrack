//! Adversarial Boundary & Stress Verification Test for Milestone M5
//! Tests input shape validation, zero/empty states, non-finite handling, and edge conditions.

use ojas_core::{BackendFloat, BackendModule, Budget, OjasError, Tensor};
use ojas_cpu::CpuBackend;
use ojas_metal::MetalBackend;
use st_model::tracker::associate::{
    associate, mask_iom, nms_iom, not_close_to_boundary, AssocParams, Bitmask,
};
use st_model::tracker::bookkeeping::{BookParams, TrackBook};
use st_model::tracker::session::SessionConfig;

#[test]
fn test_bitmask_boundary_and_zero_pixel_cases() {
    // 0 pixels
    let b0 = Bitmask::empty(0);
    assert_eq!(b0.area(), 0);
    assert_eq!(b0.pixels(), 0);

    // Empty 100 pixels
    let b1 = Bitmask::empty(100);
    let b2 = Bitmask::empty(100);
    assert_eq!(b1.area(), 0);
    let iom_empty = mask_iom(&[b1.clone()], &[b2.clone()]);
    assert_eq!(iom_empty[0], 0.0);

    // Disjoint bitmasks
    let m1 = Bitmask::from_fn(128, |i| i < 64);
    let m2 = Bitmask::from_fn(128, |i| i >= 64);
    assert_eq!(m1.area(), 64);
    assert_eq!(m2.area(), 64);
    assert_eq!(m1.intersection(&m2), 0);
    let iom_disjoint = mask_iom(&[m1.clone()], &[m2.clone()]);
    assert_eq!(iom_disjoint[0], 0.0);

    // Full overlap
    assert_eq!(m1.intersection(&m1), 64);
    let iom_overlap = mask_iom(&[m1.clone()], &[m1.clone()]);
    assert!((iom_overlap[0] - 1.0).abs() < 1e-6);
}

#[test]
fn test_nms_iom_boundary_and_edge_cases() {
    // 1. Empty inputs
    let keep = nms_iom(&[], &[], 0.4, 0.1);
    assert!(keep.is_empty());

    // 2. All below threshold
    let probs = vec![0.1, 0.2, 0.399];
    let masks = vec![Bitmask::from_fn(64, |_| true); 3];
    let keep = nms_iom(&probs, &masks, 0.4, 0.1);
    assert_eq!(keep, vec![false, false, false]);

    // 3. Exactly on threshold (strict > 0.4 required)
    let probs = vec![0.4];
    let masks = vec![Bitmask::from_fn(64, |_| true)];
    let keep = nms_iom(&probs, &masks, 0.4, 0.1);
    assert_eq!(keep, vec![false]);

    // 4. Overlapping masks: highest prob wins, duplicate suppressed
    let probs = vec![0.9, 0.85, 0.5];
    let masks = vec![Bitmask::from_fn(64, |_| true); 3];
    let keep = nms_iom(&probs, &masks, 0.4, 0.1);
    assert_eq!(keep, vec![true, false, false]);
}

#[test]
fn test_boundary_suppression_extreme_coordinates_xyxy() {
    // Empty boxes
    assert!(not_close_to_boundary(&[], 0.025).is_empty());

    // Boxes in xyxy format: margin = 0.025
    let boxes = vec![
        [0.5, 0.5, 0.6, 0.6],   // center = (0.55, 0.55) -> kept
        [0.01, 0.5, 0.02, 0.6], // center x = 0.015 < 0.025 -> suppressed
        [0.98, 0.5, 0.99, 0.6], // center x = 0.985 > 0.975 -> suppressed
        [0.5, 0.01, 0.6, 0.02], // center y = 0.015 < 0.025 -> suppressed
        [0.5, 0.98, 0.6, 0.99], // center y = 0.985 > 0.975 -> suppressed
    ];
    let keep = not_close_to_boundary(&boxes, 0.025);
    assert_eq!(keep, vec![true, false, false, false, false]);
}

#[test]
fn test_associate_zero_elements_and_extreme_permutations() {
    let p = AssocParams::default();

    // 1. 0 detections, 0 tracks
    let a0 = associate(&[], &[], &[], &[], &p);
    assert!(a0.trk_is_unmatched.is_empty());
    assert!(a0.trk_is_nonempty.is_empty());
    assert!(a0.is_new_det.is_empty());
    assert!(a0.det_to_max_trk.is_empty());

    // 2. 2 detections, 0 tracks
    let det_masks = vec![Bitmask::from_fn(64, |_| true), Bitmask::empty(64)];
    let det_scores = vec![0.8, 0.9];
    let det_keep = vec![true, true];
    let a1 = associate(&det_masks, &det_scores, &det_keep, &[], &p);
    assert_eq!(a1.is_new_det, vec![true, true]);
    assert_eq!(a1.det_to_max_trk, vec![usize::MAX, usize::MAX]);

    // 3. 0 detections, 2 tracks
    let trk_masks = vec![Bitmask::from_fn(64, |_| true), Bitmask::empty(64)];
    let a2 = associate(&[], &[], &[], &trk_masks, &p);
    assert_eq!(a2.trk_is_unmatched, vec![true, true]);
    assert_eq!(a2.trk_is_nonempty, vec![true, false]);
    assert!(a2.det_to_max_trk.is_empty());
}

#[test]
fn test_track_book_and_session_config_validation() {
    let cfg = SessionConfig::new(32, 480, 640);
    assert_eq!(cfg.num_frames, 32);
    assert_eq!(cfg.video_hw, (480, 640));

    // Zero-frame edge case
    let cfg_zero = SessionConfig::new(0, 0, 0);
    assert_eq!(cfg_zero.num_frames, 0);
    assert_eq!(cfg_zero.video_hw, (0, 0));

    let book = TrackBook::new(BookParams::default());
    assert_eq!(book.obj_ids().len(), 0);
    assert_eq!(book.max_obj_id(), -1);
}

#[test]
fn test_tensor_non_finite_rejection_fail_closed() {
    let budget = Budget::new(16 << 20);
    let cpu = CpuBackend::new(budget.clone());

    // Upload NaN tensor
    let nan_data = vec![f32::NAN, 1.0, 2.0, 3.0];
    let t = Tensor::from_f32(&nan_data, &[1, 4], &budget).unwrap();
    assert_eq!(t.shape(), &[1, 4]);

    // ReLU on NaN fails closed with typed OjasError::NonFinite
    let relu_res = cpu.relu_forward(&t);
    assert!(
        matches!(relu_res, Err(OjasError::NonFinite { op }) if op == "relu_forward"),
        "CPU backend must fail-closed on non-finite operand with OjasError::NonFinite"
    );

    // LayerNorm on NaN also fails closed
    let gamma = Tensor::from_f32(&[1.0; 4], &[4], &budget).unwrap();
    let beta = Tensor::from_f32(&[0.0; 4], &[4], &budget).unwrap();
    let ln_res = cpu.layer_norm_forward(&t, &gamma, &beta, 1e-5);
    assert!(
        matches!(ln_res, Err(OjasError::NonFinite { op }) if op == "layer_norm_forward"),
        "CPU backend must fail-closed on non-finite operand with OjasError::NonFinite"
    );
}

#[test]
fn test_metal_tensor_budget_refusal_and_capacity_exceeded() {
    let budget = Budget::new(16 << 20);
    if let Ok(_metal) = MetalBackend::new(budget) {
        // Out-of-budget allocation refusal
        let small_budget = Budget::new(64); // 64 bytes
        let huge_data = vec![0.0f32; 1024]; // 4096 bytes
        let huge_res = Tensor::from_f32(&huge_data, &[1024], &small_budget);
        assert!(
            matches!(huge_res, Err(OjasError::CapacityExceeded { .. })),
            "Allocation exceeding budget must refuse with CapacityExceeded"
        );
    }
}
