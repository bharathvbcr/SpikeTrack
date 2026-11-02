//! Adversarial Verification Test for Milestone M3: Lin & Norm Heap Elimination & Bit-Identical Parity
#![cfg(target_os = "macos")]
#![allow(dead_code)]

use ojas_core::{Budget, Tensor};
use ojas_cpu::CpuBackend;

// Re-export needed by layers.rs
pub use st_model::host;
pub use st_model::ModelError;
pub mod detector {
    pub const D: usize = 256;
    pub const HEADS: usize = 8;
    pub const LN_EPS: f32 = 1e-5;
}

#[path = "../src/layers.rs"]
mod layers;

use layers::{Lin, Norm};

#[test]
fn lin_and_norm_have_zero_heap_vector_overhead() {
    // 1. Check struct sizes
    // Tensor is Arc-backed handle (size_of::<Tensor>() is 16 bytes: 2 usize).
    // Lin previously had: w: Tensor (16), b: Tensor (16), out: usize (8), hw: Vec<f32> (24), hb: Vec<f32> (24) -> 88 bytes.
    // Now Lin has: w: Tensor (16), b: Tensor (16), out: usize (8) -> 40 bytes.
    let expected_lin_size = 2 * std::mem::size_of::<Tensor>() + std::mem::size_of::<usize>();
    assert_eq!(
        std::mem::size_of::<Lin>(),
        expected_lin_size,
        "Lin struct size should be exactly {expected_lin_size} bytes (no hw/hb Vec fields), found {}",
        std::mem::size_of::<Lin>()
    );

    // Norm previously had: w: Tensor (16), b: Tensor (16), hw: Vec<f32> (24), hb: Vec<f32> (24) -> 80 bytes.
    // Now Norm has: w: Tensor (16), b: Tensor (16) -> 32 bytes.
    let expected_norm_size = 2 * std::mem::size_of::<Tensor>();
    assert_eq!(
        std::mem::size_of::<Norm>(),
        expected_norm_size,
        "Norm struct size should be exactly {expected_norm_size} bytes (no hw/hb Vec fields), found {}",
        std::mem::size_of::<Norm>()
    );
}

#[test]
fn lin_host_returns_exact_bit_identical_results_to_reference() {
    let budget = Budget::new(64 << 20);
    let cpu = CpuBackend::new(budget.clone());

    let (inp, out) = (64, 128);
    let n = 8; // batch/tokens

    // Deterministic weights and biases
    let w_data: Vec<f32> = (0..out * inp).map(|i| (i as f32 * 0.03125).sin()).collect();
    let b_data: Vec<f32> = (0..out).map(|i| (i as f32 * 0.0625).cos()).collect();
    let x_data: Vec<f32> = (0..n * inp)
        .map(|i| (i as f32 * 0.015625 + 0.1).sin())
        .collect();

    // 1. Golden reference directly using host::linear
    let golden = st_model::host::linear(&x_data, n, &w_data, &b_data, out);

    // 2. Lin on CpuBackend
    let lin_cpu = Lin::from_parts(&cpu, &w_data, &b_data, out, inp).expect("Lin::from_parts CPU");
    let host_out_cpu = lin_cpu.host(&x_data, n);

    assert_eq!(
        host_out_cpu.len(),
        golden.len(),
        "Lin::host CPU length mismatch"
    );
    for i in 0..golden.len() {
        assert_eq!(
            host_out_cpu[i].to_bits(),
            golden[i].to_bits(),
            "CPU Lin::host mismatch at index {i}: got {}, expected {}",
            host_out_cpu[i],
            golden[i]
        );
    }

    // 3. Lin on MetalBackend (if available)
    if let Ok(metal) = ojas_metal::MetalBackend::new(budget.clone()) {
        let lin_metal =
            Lin::from_parts(&metal, &w_data, &b_data, out, inp).expect("Lin::from_parts Metal");
        let host_out_metal = lin_metal.host(&x_data, n);

        assert_eq!(
            host_out_metal.len(),
            golden.len(),
            "Lin::host Metal length mismatch"
        );
        for i in 0..golden.len() {
            assert_eq!(
                host_out_metal[i].to_bits(),
                golden[i].to_bits(),
                "Metal Lin::host mismatch at index {i}: got {}, expected {}",
                host_out_metal[i],
                golden[i]
            );
        }
    }
}

#[test]
fn norm_host_returns_exact_bit_identical_results_to_reference() {
    let budget = Budget::new(64 << 20);
    let cpu = CpuBackend::new(budget.clone());
    let d = detector::D;

    // Deterministic weights and biases
    let w_data: Vec<f32> = (0..d).map(|i| 1.0 + (i as f32 * 0.01).sin()).collect();
    let b_data: Vec<f32> = (0..d).map(|i| (i as f32 * 0.02).cos()).collect();
    let x_data: Vec<f32> = (0..4 * d).map(|i| (i as f32 * 0.05).sin()).collect();

    // 1. Golden reference directly using host::layer_norm
    let golden = st_model::host::layer_norm(&x_data, d, &w_data, &b_data, detector::LN_EPS);

    // 2. Norm on CpuBackend
    let norm_cpu = Norm {
        w: layers::up(&cpu, &w_data, &[d]).unwrap(),
        b: layers::up(&cpu, &b_data, &[d]).unwrap(),
    };
    let host_out_cpu = norm_cpu.host(&x_data);

    assert_eq!(
        host_out_cpu.len(),
        golden.len(),
        "Norm::host CPU length mismatch"
    );
    for i in 0..golden.len() {
        assert_eq!(
            host_out_cpu[i].to_bits(),
            golden[i].to_bits(),
            "CPU Norm::host mismatch at index {i}: got {}, expected {}",
            host_out_cpu[i],
            golden[i]
        );
    }

    // 3. Norm on MetalBackend (if available)
    if let Ok(metal) = ojas_metal::MetalBackend::new(budget) {
        let norm_metal = Norm {
            w: layers::up(&metal, &w_data, &[d]).unwrap(),
            b: layers::up(&metal, &b_data, &[d]).unwrap(),
        };
        let host_out_metal = norm_metal.host(&x_data);

        assert_eq!(
            host_out_metal.len(),
            golden.len(),
            "Norm::host Metal length mismatch"
        );
        for i in 0..golden.len() {
            assert_eq!(
                host_out_metal[i].to_bits(),
                golden[i].to_bits(),
                "Metal Norm::host mismatch at index {i}: got {}, expected {}",
                host_out_metal[i],
                golden[i]
            );
        }
    }
}
