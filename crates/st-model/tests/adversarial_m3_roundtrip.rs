//! Adversarial Verification Test for Milestone M3: Zero-Copy Frame Pipeline & On-Device Broadcast/Concat Parity
#![cfg(target_os = "macos")]

use ojas_core::{BackendCore, BackendFloat, Budget, DType, Tensor};
use ojas_cpu::CpuBackend;
use ojas_metal::MetalBackend;

#[test]
fn adversarial_mask_decoder_pe_and_concat_multibatch() {
    let budget = Budget::new(64 << 20);
    let cpu = CpuBackend::new(budget.clone());

    for (b, feat, d) in [(1, 16, 64), (2, 16, 64), (4, 8, 32)] {
        let hw = feat * feat;
        let pe_data: Vec<f32> = (0..d * hw).map(|i| (i as f32 * 0.01).sin()).collect();

        // 1. CPU Reference
        let pe_cpu = Tensor::from_f32(&pe_data, &[1, d, hw], &budget).unwrap();
        let pe1_cpu = cpu.permute(&pe_cpu, &[0, 2, 1]).unwrap(); // [1, hw, d]
        let out_cpu = if b == 1 {
            pe1_cpu
        } else {
            let zeros = cpu.fill(&[b, 1, 1], DType::F32, 0.0).unwrap();
            cpu.broadcast_add_forward(&pe1_cpu, &zeros).unwrap()
        };
        assert_eq!(out_cpu.shape(), &[b, hw, d]);
        let cpu_slice = out_cpu.f32_slice().unwrap();

        // 2. Metal Device
        if let Ok(metal) = MetalBackend::new(budget.clone()) {
            let pe_metal = metal
                .upload(&Tensor::from_f32(&pe_data, &[1, d, hw], metal.budget()).unwrap())
                .unwrap();
            let pe1_metal = metal.permute(&pe_metal, &[0, 2, 1]).unwrap();
            let out_metal = if b == 1 {
                pe1_metal
            } else {
                let zeros = metal.fill(&[b, 1, 1], DType::F32, 0.0).unwrap();
                metal.broadcast_add_forward(&pe1_metal, &zeros).unwrap()
            };
            assert_eq!(out_metal.shape(), &[b, hw, d]);
            assert!(out_metal.device().is_some(), "Must remain on device");

            // Verify bit-exact parity with CPU
            let metal_slice = out_metal
                .to_host(metal.budget())
                .unwrap()
                .to_f32_vec()
                .unwrap();
            assert_eq!(metal_slice.len(), cpu_slice.len());
            for i in 0..cpu_slice.len() {
                assert_eq!(
                    metal_slice[i].to_bits(),
                    cpu_slice[i].to_bits(),
                    "Mismatch for b={b} at index {i}"
                );
            }

            // 3. Concat test for b > 1
            let mut mask_tensors = Vec::with_capacity(b);
            let n = 32;
            for bi in 0..b {
                let m = metal
                    .fill(&[1, 4, 3, n, n], DType::F32, (bi + 1) as f32)
                    .unwrap();
                mask_tensors.push(m);
            }
            let refs: Vec<&Tensor> = mask_tensors.iter().collect();
            let concat_masks = if b == 1 {
                mask_tensors.pop().unwrap()
            } else {
                metal.concat(&refs, 0).unwrap()
            };
            assert_eq!(concat_masks.shape(), &[b, 4, 3, n, n]);
            assert!(concat_masks.device().is_some());
        }
    }
}

#[test]
fn adversarial_interactive_broadcast_repeat() {
    let budget = Budget::new(64 << 20);
    let cpu = CpuBackend::new(budget.clone());

    for (n, feat, d) in [(1, 16, 64), (3, 16, 64)] {
        let x_data: Vec<f32> = (0..d * feat * feat)
            .map(|i| (i as f32 * 0.05).cos())
            .collect();

        // CPU repeat logic
        let x_cpu = Tensor::from_f32(&x_data, &[1, d, feat, feat], &budget).unwrap();
        let rep_cpu = if n == 1 {
            x_cpu
        } else {
            let mut s = vec![1; 4];
            s[0] = n;
            let zeros = cpu.fill(&s, DType::F32, 0.0).unwrap();
            cpu.broadcast_add_forward(&x_cpu, &zeros).unwrap()
        };
        assert_eq!(rep_cpu.shape(), &[n, d, feat, feat]);
        let cpu_slice = rep_cpu.f32_slice().unwrap();

        // Metal repeat logic
        if let Ok(metal) = MetalBackend::new(budget.clone()) {
            let x_metal = metal
                .upload(&Tensor::from_f32(&x_data, &[1, d, feat, feat], metal.budget()).unwrap())
                .unwrap();
            let rep_metal = if n == 1 {
                x_metal
            } else {
                let mut s = vec![1; 4];
                s[0] = n;
                let zeros = metal.fill(&s, DType::F32, 0.0).unwrap();
                metal.broadcast_add_forward(&x_metal, &zeros).unwrap()
            };
            assert_eq!(rep_metal.shape(), &[n, d, feat, feat]);
            assert!(rep_metal.device().is_some(), "Must remain on device");

            let metal_slice = rep_metal
                .to_host(metal.budget())
                .unwrap()
                .to_f32_vec()
                .unwrap();
            assert_eq!(metal_slice.len(), cpu_slice.len());
            for i in 0..cpu_slice.len() {
                assert_eq!(
                    metal_slice[i].to_bits(),
                    cpu_slice[i].to_bits(),
                    "Mismatch for n={n} at index {i}"
                );
            }
        }
    }
}

#[test]
fn adversarial_detector_prompt_and_query_concat() {
    let budget = Budget::new(64 << 20);

    if let Ok(metal) = MetalBackend::new(budget.clone()) {
        let (d, text_len, geo_len) = (256, 32, 1);
        let text_data: Vec<f32> = (0..text_len * d).map(|i| i as f32).collect();
        let geo_data: Vec<f32> = (0..geo_len * d).map(|i| (i * 2) as f32).collect();

        let text = metal
            .upload(&Tensor::from_f32(&text_data, &[1, text_len, d], metal.budget()).unwrap())
            .unwrap();
        let geo = metal
            .upload(&Tensor::from_f32(&geo_data, &[1, geo_len, d], metal.budget()).unwrap())
            .unwrap();

        // On-device concat of prompt: [1, text_len + geo_len, d]
        let prompt = metal.concat(&[&text, &geo], 1).unwrap();
        assert_eq!(prompt.shape(), &[1, text_len + geo_len, d]);
        assert!(prompt.device().is_some());

        // Presence token [1, 1, d] and query embed [1, 300, d]
        let (queries, presence_len) = (300, 1);
        let presence = metal.fill(&[1, presence_len, d], DType::F32, 1.0).unwrap();
        let query_embed = metal.fill(&[1, queries, d], DType::F32, 2.0).unwrap();

        let x = metal.concat(&[&presence, &query_embed], 1).unwrap();
        assert_eq!(x.shape(), &[1, presence_len + queries, d]);
        assert!(x.device().is_some());
    }
}
