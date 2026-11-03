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

#[test]
fn test_load_host_zero_waits_and_fused_mlp_host_bit_identity() {
    use layers::Mlp;
    use st_weights::{TensorGroup, Weight};

    let d = detector::D;
    let mut g = TensorGroup {
        prefix: "mlp".to_string(),
        tensors: std::collections::BTreeMap::new(),
    };
    let mut fill = |name: &str, shape: &[usize], seed: usize| {
        let len: usize = shape.iter().product();
        let data: Vec<f32> = (0..len)
            .map(|i| (((i + seed) as f32) * 0.03125).sin() * 0.25)
            .collect();
        g.tensors.insert(
            name.to_string(),
            Weight {
                name: name.to_string(),
                shape: shape.to_vec(),
                data,
            },
        );
    };
    fill("mlp.layers.0.weight", &[d, d], 1);
    fill("mlp.layers.0.bias", &[d], 2);
    fill("mlp.layers.1.weight", &[d, d], 3);
    fill("mlp.layers.1.bias", &[d], 4);
    fill("mlp.layers.2.weight", &[d, d], 5);
    fill("mlp.layers.2.bias", &[d], 6);
    fill("mlp.out_norm.weight", &[d], 7);
    fill("mlp.out_norm.bias", &[d], 8);

    let budget = Budget::new(64 << 20);
    let cpu = ojas_cpu::CpuBackend::new(budget.clone());
    let mlp_host = Mlp::load_host(&cpu, &g, "mlp", &[d, d, d, d], true, true).unwrap();

    // Unfused reference: layer-by-layer Lin::host + relu_in_place + residual + Norm::host.
    let n = 67usize; // crosses multiple 32-row tiles and THREAD_MIN_MACS
    let x: Vec<f32> = (0..n * d).map(|i| ((i as f32) * 0.017).cos()).collect();
    let mut want = x.clone();
    for (i, l) in mlp_host.layers.iter().enumerate() {
        want = l.host(&want, n);
        if i + 1 < mlp_host.layers.len() {
            st_model::host::relu_in_place(&mut want);
        }
    }
    want.iter_mut().zip(&x).for_each(|(a, b)| *a += b);
    want = mlp_host.out_norm.as_ref().unwrap().host(&want);

    let got = mlp_host.host(&x, n);
    assert_eq!(got.len(), want.len());
    for (i, (gv, wv)) in got.iter().zip(&want).enumerate() {
        assert_eq!(gv.to_bits(), wv.to_bits(), "Mlp::host mismatch at {i}");
    }
    let mut got_rows = vec![0f32; n * d];
    mlp_host.host_rows(&x, n, &mut got_rows);
    for (i, (gv, wv)) in got_rows.iter().zip(&want).enumerate() {
        assert_eq!(gv.to_bits(), wv.to_bits(), "Mlp::host_rows mismatch at {i}");
    }

    if let Ok(metal) = ojas_metal::MetalBackend::new(budget) {
        let d0 = metal.dispatches();
        let w0 = metal.waits();
        let mlp_m = Mlp::load_host(&metal, &g, "mlp", &[d, d, d, d], true, true).unwrap();
        for l in &mlp_m.layers {
            assert!(
                l.w.device().is_none() && l.b.device().is_none(),
                "Mlp::load_host weights must remain host-only"
            );
        }
        let nm = mlp_m.out_norm.as_ref().unwrap();
        assert!(
            nm.w.device().is_none() && nm.b.device().is_none(),
            "Mlp::load_host out_norm must remain host-only"
        );
        let got_m = mlp_m.host(&x, n);
        let d1 = metal.dispatches();
        let w1 = metal.waits();
        assert_eq!(d1, d0, "Mlp::load_host + host() must not dispatch GPU ops");
        assert_eq!(w1, w0, "Mlp::load_host + host() must not trigger GPU waits");
        for (i, (gv, wv)) in got_m.iter().zip(&want).enumerate() {
            assert_eq!(
                gv.to_bits(),
                wv.to_bits(),
                "Metal Mlp::host mismatch at {i}"
            );
        }
    }

    // Empty-layers MLP must match Mlp::fwd (identity + optional residual + out_norm), not drop x.
    let empty_mlp = Mlp {
        layers: Vec::new(),
        residual: true,
        out_norm: None,
    };
    assert_eq!(empty_mlp.host(&[1.5, -2.0], 1), vec![3.0, -4.0]);
}

#[test]
#[should_panic(expected = "Mlp input")]
fn mlp_host_panics_on_nonempty_input_when_n_is_zero() {
    use layers::Mlp;
    let budget = Budget::new(1 << 20);
    let cpu = CpuBackend::new(budget);
    let l0 = Lin::from_parts(&cpu, &[1.0, 2.0, 3.0, 4.0], &[0.0, 0.0], 2, 2).unwrap();
    let mlp = Mlp {
        layers: vec![l0],
        residual: false,
        out_norm: None,
    };
    let _ = mlp.host(&[1.0, 2.0], 0);
}

#[test]
fn view_f32_and_host_of_respect_subview_window_strided_and_bf16_on_metal() {
    use ojas_core::{
        Backend, BackendId, DType, DeviceBuffer, FenceError, HostLease, HostLeaseMut, MappedBuffer,
        OjasError, Timeline,
    };
    use std::any::Any;
    use std::sync::Arc;

    #[derive(Debug)]
    struct DummyTimeline;
    impl Timeline for DummyTimeline {
        fn submitted(&self) -> Result<u64, FenceError> {
            Ok(0)
        }
        fn current_value(&self) -> u64 {
            0
        }
        fn wait(&self, _v: u64) -> Result<(), FenceError> {
            Ok(())
        }
    }
    struct DummyLease {
        bytes: Vec<u8>,
        f32s: Vec<f32>,
    }
    impl HostLease for DummyLease {
        fn as_bytes(&self) -> &[u8] {
            &self.bytes
        }
        fn as_f32(&self) -> Result<&[f32], OjasError> {
            Ok(&self.f32s)
        }
    }
    #[derive(Debug)]
    struct DummyMapped {
        bytes: Vec<u8>,
        f32s: Vec<f32>,
        tl: DummyTimeline,
    }
    impl DeviceBuffer for DummyMapped {
        fn backend(&self) -> BackendId {
            BackendId::Metal
        }
        fn byte_len(&self) -> usize {
            self.bytes.len()
        }
        fn read_bytes(&self, offset: usize, len: usize) -> Result<Vec<u8>, OjasError> {
            Ok(self.bytes[offset..offset + len].to_vec())
        }
        fn as_any(&self) -> &dyn Any {
            self
        }
    }
    impl MappedBuffer for DummyMapped {
        fn host_ptr(&self) -> *mut u8 {
            std::ptr::null_mut()
        }
        fn gpu_address(&self) -> u64 {
            0
        }
        fn timeline(&self) -> &dyn Timeline {
            &self.tl
        }
        fn host_read(&self) -> Result<Box<dyn HostLease>, FenceError> {
            Ok(Box::new(DummyLease {
                bytes: self.bytes.clone(),
                f32s: self.f32s.clone(),
            }))
        }
        fn host_write(&self) -> Result<Box<dyn HostLeaseMut>, FenceError> {
            Err(FenceError::Busy {
                detail: "read-only test mock".to_string(),
            })
        }
    }

    let budget = Budget::new(16 << 20);
    let cpu = CpuBackend::new(budget.clone());
    let data = [10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0];
    let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_ne_bytes()).collect();
    let mbuf = Arc::new(DummyMapped {
        bytes,
        f32s: data.to_vec(),
        tl: DummyTimeline,
    });
    let mapped = Tensor::from_mapped(mbuf.clone(), &[2, 3], DType::F32, &budget).unwrap();

    // 1. Contiguous subview at non-zero byte_offset (elements 2..4 -> [30.0, 40.0])
    let sub = mapped.view(&[2], &[1], 2 * 4).unwrap();
    assert_eq!(layers::view_f32(&sub).unwrap().as_slice(), &[30.0, 40.0]);
    assert_eq!(layers::host_of(&sub, &cpu).unwrap(), vec![30.0, 40.0]);

    // 2. Non-contiguous strided view ([3, 2] transpose): must refuse cleanly, NEVER return un-transposed [10..60]!
    let transposed = mapped.view(&[3, 2], &[1, 3], 0).unwrap();
    assert!(
        layers::view_f32(&transposed).is_err(),
        "non-contiguous view must not return un-transposed underlying storage"
    );
    assert!(
        layers::host_of(&transposed, &cpu).is_err(),
        "non-contiguous view must not return un-transposed underlying storage"
    );

    // 3. BF16 mapped tensor (4 elements = 8 bytes, multiple of 4!): must refuse dtype mismatch, NEVER reinterpret bf16 bytes as f32s!
    let bf16_mapped = Tensor::from_mapped(mbuf, &[4], DType::Bf16, &budget).unwrap();
    assert!(
        layers::view_f32(&bf16_mapped).is_err(),
        "BF16 mapped tensor must not reinterpret raw bytes as f32"
    );
    assert!(
        layers::host_of(&bf16_mapped, &cpu).is_err(),
        "BF16 mapped tensor must not reinterpret raw bytes as f32"
    );

    // 4. Also check real MetalBackend Payload::Device subviews, strided views, and BF16!
    if let Ok(metal) = ojas_metal::MetalBackend::new(budget.clone()) {
        let dev = metal
            .upload(&Tensor::from_f32(&data, &[2, 3], metal.budget()).unwrap())
            .unwrap();
        let sub_dev = dev.view(&[2], &[1], 2 * 4).unwrap();
        assert_eq!(
            layers::view_f32(&sub_dev).unwrap().as_slice(),
            &[30.0, 40.0]
        );
        assert_eq!(layers::host_of(&sub_dev, &metal).unwrap(), vec![30.0, 40.0]);
        let tr_dev = dev.view(&[3, 2], &[1, 3], 0).unwrap();
        assert!(layers::view_f32(&tr_dev).is_err());
        let bf16_dev = metal
            .upload(
                &Tensor::from_bf16_bits(&[0x3f80, 0x4000, 0x4040, 0x4080], &[4], &budget).unwrap(),
            )
            .unwrap();
        assert!(layers::view_f32(&bf16_dev).is_err());
    }
}
