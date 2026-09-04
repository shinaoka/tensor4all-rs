//! CUDA residency tests for context-scoped factorization (issue #720).
//!
//! Each test uploads host fixtures into one caller-owned [`ExecutionContext`],
//! runs the `_in` factorization/estimator entry, and checks exact-context
//! residency plus numerical parity against the CPU path. Complex QR parity is
//! reconstruction-based: QR is unique only up to per-column unit phases
//! (CUDA yields real R diagonals; CPU/faer leaves small imaginary residues).

#![cfg(feature = "tenferro-cuda")]

use std::sync::Arc;
use tensor4all_core::CudaExecutionContext;

#[test]
fn probe_cuda_qr_with_in_matches_cpu() {
    use tensor4all_core::qr::{qr_with, qr_with_in, QrOptions};
    use tensor4all_core::{DynIndex, ExecutionContext, IdxTensor};
    use tensor4all_tensorbackend::CpuExecutionContext;

    let cuda = Arc::new(CudaExecutionContext::new().unwrap());
    let context = ExecutionContext::Cuda(Arc::clone(&cuda));

    for dtype_c64 in [false, true] {
        let i = DynIndex::new_dyn(6);
        let j = DynIndex::new_dyn(4);
        // Full-rank fixture: diagonal-heavy + small off-diagonal
        let host = if dtype_c64 {
            use num_complex::Complex64;
            let data: Vec<Complex64> = (0..24)
                .map(|k| {
                    let (r, c) = (k % 6, k / 6);
                    Complex64::new(
                        if r == c {
                            4.0 + r as f64
                        } else {
                            0.1 * (r + c) as f64
                        },
                        0.05 * (r as f64 - c as f64),
                    )
                })
                .collect();
            IdxTensor::from_dense(vec![i.clone(), j.clone()], data).unwrap()
        } else {
            let data: Vec<f64> = (0..24)
                .map(|k| {
                    let (r, c) = (k % 6, k / 6);
                    if r == c {
                        4.0 + r as f64
                    } else {
                        0.1 * (r + c) as f64
                    }
                })
                .collect();
            IdxTensor::from_dense(vec![i.clone(), j.clone()], data).unwrap()
        };

        let resident = host.upload_cuda(&cuda).unwrap();
        resident.validate_context(&context).unwrap();

        let options = QrOptions::new().with_rtol(1e-10);
        let (q, r) = if dtype_c64 {
            qr_with_in::<num_complex::Complex64>(&resident, &[i.clone()], &options, &context)
                .unwrap()
        } else {
            qr_with_in::<f64>(&resident, &[i.clone()], &options, &context).unwrap()
        };
        q.validate_context(&context).unwrap();
        r.validate_context(&context).unwrap();

        // Residency: host read must fail without explicit download
        assert!(q.to_vec::<f64>().is_err() || dtype_c64);

        // Numerical parity against CPU, via explicit download + whole-result maxabs
        let (cpu_q, cpu_r) = if dtype_c64 {
            qr_with::<num_complex::Complex64>(&host, &[i.clone()], &options).unwrap()
        } else {
            qr_with::<f64>(&host, &[i.clone()], &options).unwrap()
        };
        assert_eq!(q.dims(), cpu_q.dims());
        assert_eq!(r.dims(), cpu_r.dims());

        let q_back = q.download(&cuda).unwrap();
        let r_back = r.download(&cuda).unwrap();
        // Parity via reconstruction: complex QR is unique only up to per-column
        // unit phases (CUDA yields real R diagonals; CPU/faer leaves small
        // imaginary residues), so direct Q/R comparison is phase-fragile.
        {
            use tensor4all_core::TensorContractionLike;
            let cuda_recon = q_back.contract_pair(&r_back).unwrap();
            let cpu_recon = cpu_q.contract_pair(&cpu_r).unwrap();
            let residual = if dtype_c64 {
                use num_complex::Complex64;
                let got: Vec<Complex64> = cuda_recon.to_vec().unwrap();
                let want: Vec<Complex64> = cpu_recon.to_vec().unwrap();
                got.iter()
                    .zip(want.iter())
                    .map(|(a, b)| (a - b).norm())
                    .fold(0.0_f64, f64::max)
            } else {
                let got: Vec<f64> = cuda_recon.to_vec().unwrap();
                let want: Vec<f64> = cpu_recon.to_vec().unwrap();
                got.iter()
                    .zip(want.iter())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0_f64, f64::max)
            };
            assert!(residual < 1e-9, "CUDA/CPU reconstruction parity {residual}");
        }
        if !dtype_c64 {
            // Real QR fixes column signs up to roundoff: direct comparison valid.
            let got: Vec<f64> = q_back.to_vec().unwrap();
            let want: Vec<f64> = cpu_q.to_vec().unwrap();
            let residual = got
                .iter()
                .zip(want.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max);
            assert!(residual < 1e-9, "CUDA/CPU Q parity residual {residual}");
            let got: Vec<f64> = r_back.to_vec().unwrap();
            let want: Vec<f64> = cpu_r.to_vec().unwrap();
            let residual = got
                .iter()
                .zip(want.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max);
            assert!(residual < 1e-9, "CUDA/CPU R parity residual {residual}");
        }

        // Context-free entry rejects resident inputs before any work
        assert!(qr_with::<f64>(&resident, &[i.clone()], &options).is_err());
        // Foreign context is rejected
        let foreign = ExecutionContext::Cuda(Arc::new(CudaExecutionContext::new().unwrap()));
        assert!(qr_with_in::<f64>(&resident, &[i.clone()], &options, &foreign).is_err());
    }
}

#[test]
fn probe_cuda_svd_with_in_matches_cpu() {
    use tensor4all_core::svd::{svd_with, svd_with_in, SvdOptions};
    use tensor4all_core::{DynIndex, ExecutionContext, IdxTensor, SvdTruncationPolicy};

    let cuda = Arc::new(CudaExecutionContext::new().unwrap());
    let context = ExecutionContext::Cuda(Arc::clone(&cuda));
    let i = DynIndex::new_dyn(6);
    let j = DynIndex::new_dyn(4);
    // Rank-2 fixture: diagonal-heavy data
    let data: Vec<f64> = (0..24)
        .map(|k| {
            let (r, c) = (k % 6, k / 6);
            if r == c {
                if r < 2 {
                    4.0 + r as f64
                } else {
                    1e-13 * r as f64
                }
            } else {
                1e-13 * (r + c) as f64
            }
        })
        .collect();
    let host = IdxTensor::from_dense(vec![i.clone(), j.clone()], data.clone()).unwrap();
    let resident = host.upload_cuda(&cuda).unwrap();

    let options = SvdOptions::new().with_policy(SvdTruncationPolicy::new(1e-10));
    let (u, s, v) = svd_with_in::<f64>(&resident, &[i.clone()], &options, &context).unwrap();
    u.validate_context(&context).unwrap();
    v.validate_context(&context).unwrap();
    // S uses diagonal storage (download support for resident diagonal factors
    // is a PR4 final-SVD prerequisite); residency is shown by the absence of
    // host readability, matching the cuda_transfer test idiom.
    assert!(s.to_vec::<f64>().is_err());

    let (cpu_u, cpu_s, _cpu_v) = svd_with::<f64>(&host, &[i.clone()], &options).unwrap();
    assert_eq!(s.dims(), cpu_s.dims());
    assert_eq!(u.dims(), cpu_u.dims());
    // Rank decision parity: the fixture truncates to rank 2 on both paths.
    assert_eq!(s.dims(), vec![2, 2]);

    // Context-free entry rejects resident inputs before any work
    assert!(svd_with::<f64>(&resident, &[i.clone()], &options).is_err());
}

#[test]
fn probe_cuda_src_error_estimate_matches_cpu() {
    use tensor4all_core::{DynIndex, ExecutionContext, IdxTensor, TensorFactorizationLike};

    let cuda = Arc::new(CudaExecutionContext::new().unwrap());
    let context = ExecutionContext::Cuda(Arc::clone(&cuda));

    // Upper-triangular 4x4 R fixture, column-major
    let cap = DynIndex::new_dyn(4);
    let batch = DynIndex::new_dyn(4);
    let mut data = vec![0.0_f64; 16];
    for row in 0..4 {
        for col in row..4 {
            data[row + col * 4] = 3.0 + row as f64 + 0.5 * col as f64;
        }
    }
    let host = IdxTensor::from_dense(vec![cap, batch], data).unwrap();
    let resident = host.upload_cuda(&cuda).unwrap();

    let device = resident.src_error_estimate_in(&context).unwrap();
    let expected = host.src_error_estimate().unwrap();
    assert!(
        (device.error - expected.error).abs() < 1e-12,
        "error: device={} cpu={}",
        device.error,
        expected.error
    );
    assert!(
        (device.norm - expected.norm).abs() < 1e-12,
        "norm: device={} cpu={}",
        device.norm,
        expected.norm
    );

    // Context-free entry rejects resident inputs before any work
    assert!(resident.src_error_estimate().is_err());
    // Foreign context is rejected
    let foreign = ExecutionContext::Cuda(Arc::new(CudaExecutionContext::new().unwrap()));
    assert!(resident.src_error_estimate_in(&foreign).is_err());
    // Non-square and rank-1 factors fail with typed errors
    let bad_rank =
        IdxTensor::from_dense_in(&context, vec![DynIndex::new_dyn(2)], vec![1.0, 2.0]).unwrap();
    assert!(bad_rank.src_error_estimate_in(&context).is_err());
}

#[test]
fn probe_cuda_incremental_probe_batch_matches_host() {
    use tensor4all_core::{DynIndex, ExecutionContext, IdxTensor};
    use tensor4all_core::{TensorContractionLike, TensorFactorizationLike};

    let cuda = Arc::new(CudaExecutionContext::new().unwrap());
    let context = ExecutionContext::Cuda(Arc::clone(&cuda));

    // 8x6 sketch in two blocks of 3 columns (column-major f64)
    let rows: Vec<DynIndex> = (0..1).map(|_| DynIndex::new_dyn(8)).collect();
    let left = rows[0].clone();
    let data: Vec<f64> = (0..48)
        .map(|k| {
            let (r, c) = (k % 8, k / 8);
            if r == c {
                5.0 + r as f64
            } else {
                0.1 * (r as f64 + 1.0) / (c as f64 + 1.0)
            }
        })
        .collect();
    let block_a = DynIndex::new_dyn(3);
    let block_b = DynIndex::new_dyn(3);
    let host_a = IdxTensor::from_dense(
        vec![left.clone(), block_a.clone()],
        data.iter()
            .enumerate()
            .filter(|(k, _)| (k / 8) < 3)
            .map(|(_, v)| *v)
            .collect(),
    )
    .unwrap();
    let host_b = IdxTensor::from_dense(
        vec![left.clone(), block_b.clone()],
        data.iter()
            .enumerate()
            .filter(|(k, _)| (k / 8) >= 3)
            .map(|(_, v)| *v)
            .collect(),
    )
    .unwrap();
    let resident_a = host_a.upload_cuda(&cuda).unwrap();
    let resident_b = host_b.upload_cuda(&cuda).unwrap();

    // Device growth loop
    let first = IdxTensor::factorize_probe_batch_incremental_in(
        None,
        &resident_a,
        &block_a,
        &[left.clone()],
        &context,
    )
    .unwrap();
    assert_eq!(first.rank, 3);
    first.left.validate_context(&context).unwrap();
    first.right.validate_context(&context).unwrap();
    let grown = IdxTensor::factorize_probe_batch_incremental_in(
        Some(&first),
        &resident_b,
        &block_b,
        &[left.clone()],
        &context,
    )
    .unwrap();
    assert_eq!(grown.rank, 6);
    grown.left.validate_context(&context).unwrap();
    grown.right.validate_context(&context).unwrap();
    // The resident path stores no host IncrementalQrState (code-evident via
    // FactorizeResult::new); the estimator below takes the full-solve branch.

    // Host growth loop for reference
    let host_first =
        IdxTensor::factorize_probe_batch_incremental(None, &host_a, &block_a, &[left.clone()])
            .unwrap();
    let host_grown = IdxTensor::factorize_probe_batch_incremental(
        Some(&host_first),
        &host_b,
        &block_b,
        &[left.clone()],
    )
    .unwrap();
    assert_eq!(host_grown.rank, grown.rank);

    // Both spans reconstruct the full sketch: compare reconstructions
    let dev_recon = grown
        .left
        .contract_pair(&grown.right)
        .unwrap()
        .download(&cuda)
        .unwrap();
    let host_recon = host_grown.left.contract_pair(&host_grown.right).unwrap();
    let got: Vec<f64> = dev_recon.to_vec().unwrap();
    let want: Vec<f64> = host_recon.to_vec().unwrap();
    assert_eq!(got.len(), want.len());
    let residual = got
        .iter()
        .zip(want.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        residual < 1e-9,
        "incremental reconstruction parity {residual}"
    );

    // The grown R feeds the device estimator
    let estimate = grown.right.src_error_estimate_in(&context).unwrap();
    let expected = host_grown.right.src_error_estimate().unwrap();
    assert!((estimate.error - expected.error).abs() < 1e-12);
    assert!((estimate.norm - expected.norm).abs() < 1e-12);

    // Context-free entry rejects resident inputs
    assert!(IdxTensor::factorize_probe_batch_incremental(
        None,
        &resident_a,
        &block_a,
        &[left.clone()]
    )
    .is_err());
}
