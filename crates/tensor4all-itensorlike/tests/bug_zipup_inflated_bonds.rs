//! Bug: truncate with small rtol produces wrong results on non-canonical TTs
//! from axpby (direct sum).
//!
//! After axpby, the TT has inflated bond dimensions and is not in canonical
//! form. Calling `truncate(svd().with_svd_policy(tensor4all_core::SvdTruncationPolicy::new(1e-15)))` should remove only
//! near-zero singular values (effectively lossless for well-conditioned data).
//! But truncate incorrectly drops significant singular values, changing the
//! represented function.
//!
//! This bug affects downstream operations (e.g., matrix multiplication via
//! zipup) that depend on correct truncation to compress intermediate results.
//!
//! Example: f = 0.8*I + 0*ones + 0.1*ones (accumulated via axpby, bonddim=3)
//! requires bonddim=2 to represent exactly. truncate(rtol=1e-15) drops to
//! bonddim=1, losing 15% of the function.

use tensor4all_core::{AnyScalar, DynIndex, IndexLike, TensorDynLen};
use tensor4all_itensorlike::{ContractOptions, TensorTrain, TruncateOptions};

/// Create identity MPO with bond dim 1.
fn create_identity_mpo(row_indices: &[DynIndex], col_indices: &[DynIndex]) -> TensorTrain {
    let nsites = row_indices.len();
    assert_eq!(nsites, col_indices.len());
    let bonds: Vec<DynIndex> = (0..nsites - 1).map(|_| DynIndex::new_dyn(1)).collect();

    let mut tensors = Vec::with_capacity(nsites);
    for i in 0..nsites {
        let dim = row_indices[i].dim();
        let mut data = vec![0.0_f64; dim * dim];
        for d in 0..dim {
            data[d * dim + d] = 1.0;
        }
        let mut indices = Vec::new();
        if i > 0 {
            indices.push(bonds[i - 1].clone());
        }
        indices.push(row_indices[i].clone());
        indices.push(col_indices[i].clone());
        if i < nsites - 1 {
            indices.push(bonds[i].clone());
        }
        tensors.push(TensorDynLen::from_dense(indices, data).unwrap());
    }
    TensorTrain::new(tensors).unwrap()
}

/// Create ones MPO with bond dim 1.
fn create_ones_mpo(row_indices: &[DynIndex], col_indices: &[DynIndex]) -> TensorTrain {
    let nsites = row_indices.len();
    assert_eq!(nsites, col_indices.len());
    let bonds: Vec<DynIndex> = (0..nsites - 1).map(|_| DynIndex::new_dyn(1)).collect();

    let mut tensors = Vec::with_capacity(nsites);
    for i in 0..nsites {
        let mut indices = Vec::new();
        if i > 0 {
            indices.push(bonds[i - 1].clone());
        }
        indices.push(row_indices[i].clone());
        indices.push(col_indices[i].clone());
        if i < nsites - 1 {
            indices.push(bonds[i].clone());
        }
        let data = vec![1.0_f64; row_indices[i].dim() * col_indices[i].dim()];
        tensors.push(TensorDynLen::from_dense(indices, data).unwrap());
    }
    TensorTrain::new(tensors).unwrap()
}

fn bond_dims(tt: &TensorTrain) -> Vec<usize> {
    (0..tt.len().saturating_sub(1))
        .map(|s| tt.linkind(s).map(|l| l.dim()).unwrap_or(0))
        .collect()
}

/// Minimal reproduction: truncate(rtol=1e-15) on TT from axpby changes
/// the represented function.
///
/// f = 0.8*I + 0*ones + 0.1*ones (3-term accumulation via axpby)
///   = 0.8*I + 0.1*ones (requires bonddim=2)
///
/// After axpby: bonddim=3 (non-canonical, with one zero singular value)
/// After truncate(rtol=1e-15): bonddim=1 (WRONG — should be 2)
///
/// The truncation incorrectly drops a significant singular value,
/// producing rel_err ≈ 0.15 (15% error).
#[test]
fn test_truncate_drops_significant_singular_values() {
    let nbit = 2;
    let row: Vec<DynIndex> = (0..nbit)
        .map(|s| DynIndex::new_dyn_with_tag(2, &format!("row={}", s + 1)).unwrap())
        .collect();
    let col: Vec<DynIndex> = (0..nbit)
        .map(|s| DynIndex::new_dyn_with_tag(2, &format!("col={}", s + 1)).unwrap())
        .collect();

    let identity = create_identity_mpo(&row, &col);
    let ones = create_ones_mpo(&row, &col);

    // Accumulate like block_conv: f = 0.8*I + 0*ones + 0.1*ones
    let zero_ones = ones.scale(AnyScalar::new_real(0.0)).unwrap();
    let step1 = identity.scale(AnyScalar::new_real(0.8)).unwrap();
    let step2 = step1
        .axpby(
            AnyScalar::new_real(1.0),
            &zero_ones,
            AnyScalar::new_real(1.0),
        )
        .unwrap();
    let f = step2
        .axpby(AnyScalar::new_real(1.0), &ones, AnyScalar::new_real(0.1))
        .unwrap();

    let f_dense_before = f.to_dense().unwrap();
    eprintln!(
        "before truncate: bonds={:?} norm={:.6e}",
        bond_dims(&f),
        f.norm().unwrap()
    );

    // truncate with rtol=1e-15: should keep bonddim=2
    let mut f_truncated = f.clone();
    f_truncated
        .truncate(
            &TruncateOptions::svd()
                .with_svd_policy(tensor4all_core::SvdTruncationPolicy::new(1e-15)),
        )
        .unwrap();

    let f_dense_after = f_truncated.to_dense().unwrap();
    eprintln!(
        "after truncate(rtol=1e-15): bonds={:?} norm={:.6e}",
        bond_dims(&f_truncated),
        f_truncated.norm().unwrap()
    );

    let diff = f_dense_before
        .add(&f_dense_after.scale(AnyScalar::new_real(-1.0)).unwrap())
        .unwrap();
    let rel_err = diff.norm() / f_dense_before.norm();
    eprintln!("rel_err = {:.6e}", rel_err);

    // truncate with rtol=0.0 should also preserve the function
    let mut f_trunc0 = f.clone();
    f_trunc0
        .truncate(
            &TruncateOptions::svd().with_svd_policy(tensor4all_core::SvdTruncationPolicy::new(0.0)),
        )
        .unwrap();
    let f_dense_0 = f_trunc0.to_dense().unwrap();
    let diff_0 = f_dense_before
        .add(&f_dense_0.scale(AnyScalar::new_real(-1.0)).unwrap())
        .unwrap();
    let rel_err_0 = diff_0.norm() / f_dense_before.norm();
    eprintln!(
        "truncate(rtol=0.0): bonds={:?} rel_err={:.6e}",
        bond_dims(&f_trunc0),
        rel_err_0
    );
    assert!(
        rel_err_0 < 1e-12,
        "truncate(rtol=0.0) changed the function! rel_err={:.6e}",
        rel_err_0,
    );

    assert!(
        rel_err < 1e-12,
        "truncate(rtol=1e-15) changed the function! \
         bonds: {:?} → {:?}, rel_err={:.6e}",
        bond_dims(&f),
        bond_dims(&f_truncated),
        rel_err,
    );
}

/// Verify the issue scales: more sites, different parameters.
#[test]
fn test_truncate_drops_sv_various_sizes() {
    for nbit in [2, 3, 4] {
        let row: Vec<DynIndex> = (0..nbit)
            .map(|s| DynIndex::new_dyn_with_tag(2, &format!("row={}", s + 1)).unwrap())
            .collect();
        let col: Vec<DynIndex> = (0..nbit)
            .map(|s| DynIndex::new_dyn_with_tag(2, &format!("col={}", s + 1)).unwrap())
            .collect();

        let identity = create_identity_mpo(&row, &col);
        let ones = create_ones_mpo(&row, &col);

        // f = I + 0*ones + ones (accumulated via axpby, bonddim = 3)
        let zero_ones = ones.scale(AnyScalar::new_real(0.0)).unwrap();
        let f = identity
            .axpby(
                AnyScalar::new_real(1.0),
                &zero_ones,
                AnyScalar::new_real(1.0),
            )
            .unwrap()
            .axpby(AnyScalar::new_real(1.0), &ones, AnyScalar::new_real(1.0))
            .unwrap();

        let f_dense_before = f.to_dense().unwrap();

        let mut f_truncated = f.clone();
        f_truncated
            .truncate(
                &TruncateOptions::svd()
                    .with_svd_policy(tensor4all_core::SvdTruncationPolicy::new(1e-15)),
            )
            .unwrap();

        let f_dense_after = f_truncated.to_dense().unwrap();
        let diff = f_dense_before
            .add(&f_dense_after.scale(AnyScalar::new_real(-1.0)).unwrap())
            .unwrap();
        let rel_err = diff.norm() / f_dense_before.norm();

        eprintln!(
            "nbit={}: bonds {:?} → {:?}, rel_err={:.6e}",
            nbit,
            bond_dims(&f),
            bond_dims(&f_truncated),
            rel_err,
        );

        assert!(
            rel_err < 1e-12,
            "nbit={}: truncate(rtol=1e-15) changed the function! rel_err={:.6e}",
            nbit,
            rel_err,
        );
    }
}

/// Zipup should give identical results for direct and accumulated TTs.
/// Compare using dense tensors to avoid catastrophic cancellation in
/// TT norm of the direct-sum difference.
#[test]
fn test_zipup_small_error_with_accumulated_tt() {
    for nbit in [3, 4] {
        let row: Vec<DynIndex> = (0..nbit)
            .map(|s| DynIndex::new_dyn_with_tag(2, &format!("row={}", s + 1)).unwrap())
            .collect();
        let shared: Vec<DynIndex> = (0..nbit)
            .map(|s| DynIndex::new_dyn_with_tag(2, &format!("shared={}", s + 1)).unwrap())
            .collect();
        let col: Vec<DynIndex> = (0..nbit)
            .map(|s| DynIndex::new_dyn_with_tag(2, &format!("col={}", s + 1)).unwrap())
            .collect();

        let identity = create_identity_mpo(&row, &shared);
        let ones = create_ones_mpo(&row, &shared);
        let g = create_ones_mpo(&shared, &col);

        // Direct: bonddim=2
        let f_direct = identity
            .axpby(AnyScalar::new_real(0.8), &ones, AnyScalar::new_real(0.1))
            .unwrap();

        // Accumulated with zero term: bonddim=3
        let zero_ones = ones.scale(AnyScalar::new_real(0.0)).unwrap();
        let f_accumulated = identity
            .scale(AnyScalar::new_real(0.8))
            .unwrap()
            .axpby(
                AnyScalar::new_real(1.0),
                &zero_ones,
                AnyScalar::new_real(1.0),
            )
            .unwrap()
            .axpby(AnyScalar::new_real(1.0), &ones, AnyScalar::new_real(0.1))
            .unwrap();

        let options = ContractOptions::zipup();
        let result_direct = f_direct.contract(&g, &options).unwrap();
        let result_accum = f_accumulated.contract(&g, &options).unwrap();

        // Compare dense representations to avoid catastrophic cancellation
        let dense_direct = result_direct.to_dense().unwrap();
        let dense_accum = result_accum.to_dense().unwrap();
        let diff = dense_direct
            .add(&dense_accum.scale(AnyScalar::new_real(-1.0)).unwrap())
            .unwrap();
        let rel_err = diff.norm() / dense_direct.norm();

        eprintln!(
            "nbit={}: bonds direct={:?} accum={:?} rel_err={:.6e}",
            nbit,
            bond_dims(&f_direct),
            bond_dims(&f_accumulated),
            rel_err,
        );

        assert!(
            rel_err < 1e-10,
            "nbit={}: zipup gives different results! rel_err={:.6e}",
            nbit,
            rel_err,
        );
    }
}
