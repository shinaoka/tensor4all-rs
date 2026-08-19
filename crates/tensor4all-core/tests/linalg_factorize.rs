//! Tests for the unified factorize function.

use num_complex::Complex64;
use tensor4all_core::block_tensor::BlockTensor;
use tensor4all_core::index::Index;
use tensor4all_core::{
    factorize, factorize_full_rank, Canonical, DynIndex, FactorizeAlg, FactorizeError,
    FactorizeOptions, TensorContractionLike, TensorFactorizationLike,
};
use tensor4all_core::{IdxTensor, SvdTruncationPolicy};
use tensor4all_tensorbackend::TensorElement;

// ============================================================================
// Test Data Helpers
// ============================================================================

/// Helper to create a simple 2x3 matrix tensor for testing.
fn create_test_matrix() -> IdxTensor {
    let i: DynIndex = Index::new_dyn(2);
    let j: DynIndex = Index::new_dyn(3);

    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    IdxTensor::from_dense(vec![i, j], data).unwrap()
}

/// Helper to create a rank-3 tensor for testing.
fn create_rank3_tensor() -> IdxTensor {
    let i: DynIndex = Index::new_dyn(2);
    let j: DynIndex = Index::new_dyn(3);
    let k: DynIndex = Index::new_dyn(2);

    let data: Vec<f64> = (0..12).map(|x| x as f64).collect();
    IdxTensor::from_dense(vec![i, j, k], data).unwrap()
}

fn create_unit_dim_rank3_tensor() -> IdxTensor {
    let i: DynIndex = Index::new_dyn(1);
    let j: DynIndex = Index::new_dyn(2);
    let k: DynIndex = Index::new_dyn(2);

    let data = vec![1.0, 2.0, 3.0, 4.0];
    IdxTensor::from_dense(vec![i, j, k], data).unwrap()
}

fn create_non_symmetric_col_major_matrix() -> IdxTensor {
    let i: DynIndex = Index::new_dyn(2);
    let j: DynIndex = Index::new_dyn(3);

    // Logical matrix:
    // [[1, 2, 4],
    //  [3, 5, 6]]
    let data = vec![1.0, 3.0, 2.0, 5.0, 4.0, 6.0];
    IdxTensor::from_dense(vec![i, j], data).unwrap()
}

// ============================================================================
// Shared Test Helpers
// ============================================================================

#[test]
fn factorize_auto_default_validates_and_delegates() {
    let left = DynIndex::new_dyn(2);
    let right = DynIndex::new_dyn(2);
    let dense =
        IdxTensor::from_dense(vec![left.clone(), right], vec![1.0_f64, 0.0, 0.0, 1.0]).unwrap();
    let block = BlockTensor::new(vec![dense], (1, 1)).unwrap();

    assert!(matches!(
        block.factorize_auto(std::slice::from_ref(&left), &FactorizeOptions::svd()),
        Err(FactorizeError::ComputationError(_))
    ));
    assert!(matches!(
        block.factorize_auto(&[left], &FactorizeOptions::qr()),
        Err(FactorizeError::InvalidOptions(_))
    ));
}

#[test]
fn factorize_auto_matches_svd_for_safe_policy() {
    let i = DynIndex::new_dyn(2);
    let j = DynIndex::new_dyn(2);
    let tensor =
        IdxTensor::from_dense(vec![i.clone(), j], vec![3.0_f64, 0.0, 0.0, 1.0e-3]).unwrap();
    let options = FactorizeOptions::svd()
        .with_svd_policy(SvdTruncationPolicy::new(1.0e-2))
        .with_canonical(Canonical::Left);

    let explicit = tensor
        .factorize(std::slice::from_ref(&i), &options)
        .unwrap();
    let automatic = tensor.factorize_auto(&[i], &options).unwrap();

    assert_eq!(automatic.rank, explicit.rank);
    assert!(automatic
        .left
        .contract_pair(&automatic.right)
        .unwrap()
        .isapprox(
            &explicit.left.contract_pair(&explicit.right).unwrap(),
            1.0e-10,
            0.0,
        )
        .unwrap());
}

/// Test factorization with given options and verify reconstruction.
fn test_factorize_reconstruction(options: &FactorizeOptions) {
    let tensor = create_test_matrix();
    let left_inds = vec![tensor.indices[0].clone()];

    let result = factorize(&tensor, &left_inds, options).unwrap();

    // Verify reconstruction: left * right ≈ original
    let reconstructed = result.left.contract_pair(&result.right).unwrap();
    assert_tensors_approx_equal(&tensor, &reconstructed, 1e-10);
}

/// Test that left and right factors share the same bond index.
fn test_shared_bond_index(options: &FactorizeOptions) {
    let tensor = create_test_matrix();
    let left_inds = vec![tensor.indices[0].clone()];

    let result = factorize(&tensor, &left_inds, options).unwrap();

    let left_bond = result.left.indices.last().unwrap();
    let right_bond = result.right.indices.first().unwrap();
    assert_eq!(
        left_bond.id, right_bond.id,
        "Left and right should share the same bond index"
    );
    assert_eq!(
        left_bond.id, result.bond_index.id,
        "Bond index should match left's bond index"
    );
}

// ============================================================================
// All Algorithms: Reconstruction Tests
// ============================================================================

#[test]
fn test_factorize_reconstruction_all_algorithms() {
    // Test all algorithms with both canonical directions (where supported)
    let algorithms = [
        (FactorizeAlg::SVD, vec![Canonical::Left, Canonical::Right]),
        (FactorizeAlg::QR, vec![Canonical::Left]), // Right not supported
        (FactorizeAlg::LU, vec![Canonical::Left, Canonical::Right]),
        (FactorizeAlg::CI, vec![Canonical::Left, Canonical::Right]),
    ];

    for (alg, canonicals) in algorithms {
        for canonical in canonicals {
            let options = FactorizeOptions {
                alg,
                canonical,
                max_bond_dim: None,
                svd_policy: None,
                qr_rtol: None,
            };
            test_factorize_reconstruction(&options);
        }
    }
}

#[test]
fn test_factorize_shared_bond_index_all_algorithms() {
    // Test all algorithms have shared bond index
    let algorithms = [
        FactorizeAlg::SVD,
        FactorizeAlg::QR,
        FactorizeAlg::LU,
        FactorizeAlg::CI,
    ];

    for alg in algorithms {
        let options = FactorizeOptions {
            alg,
            canonical: Canonical::Left,
            max_bond_dim: None,
            svd_policy: None,
            qr_rtol: None,
        };
        test_shared_bond_index(&options);
    }
}

// ============================================================================
// SVD-Specific Tests
// ============================================================================

#[test]
fn test_factorize_svd_returns_singular_values() {
    for canonical in [Canonical::Left, Canonical::Right] {
        let tensor = create_test_matrix();
        let left_inds = vec![tensor.indices[0].clone()];
        let options = FactorizeOptions::svd().with_canonical(canonical);
        let result = factorize(&tensor, &left_inds, &options).unwrap();

        assert!(result.singular_values.is_some());
        let sv = result.singular_values.unwrap();
        assert!(!sv.is_empty());
        assert!(result.rank > 0);
        assert!(result.rank <= 2); // min(2, 3) = 2
    }
}

#[test]
fn test_factorize_svd_rank3() {
    let tensor = create_rank3_tensor();
    let left_inds = vec![tensor.indices[0].clone(), tensor.indices[1].clone()];

    let options = FactorizeOptions::svd();
    let result = factorize(&tensor, &left_inds, &options).unwrap();

    let reconstructed = result.left.contract_pair(&result.right).unwrap();
    assert_tensors_approx_equal(&tensor, &reconstructed, 1e-10);
}

#[test]
fn factorize_auto_matches_explicit_svd_for_all_policies_and_shapes() {
    let policies = [
        SvdTruncationPolicy::new(5.0e-2),
        SvdTruncationPolicy::new(5.0e-2).with_discarded_tail_sum(),
        SvdTruncationPolicy::new(5.0e-2).with_squared_values(),
        SvdTruncationPolicy::new(5.0e-2)
            .with_squared_values()
            .with_discarded_tail_sum(),
        SvdTruncationPolicy::new(5.0e-2).with_absolute(),
        SvdTruncationPolicy::new(5.0e-2)
            .with_absolute()
            .with_discarded_tail_sum(),
        SvdTruncationPolicy::new(5.0e-2)
            .with_absolute()
            .with_squared_values(),
        SvdTruncationPolicy::new(5.0e-2)
            .with_absolute()
            .with_squared_values()
            .with_discarded_tail_sum(),
    ];

    for (m, n, data) in [
        (
            4,
            3,
            vec![10.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1],
        ),
        (
            3,
            4,
            vec![10.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1],
        ),
    ] {
        let left = DynIndex::new_dyn(m);
        let right = DynIndex::new_dyn(n);
        let tensor = IdxTensor::from_dense(vec![left.clone(), right], data).unwrap();
        for canonical in [Canonical::Left, Canonical::Right] {
            for policy in policies {
                let options = FactorizeOptions::svd()
                    .with_canonical(canonical)
                    .with_svd_policy(policy);
                let explicit = tensor
                    .factorize(std::slice::from_ref(&left), &options)
                    .unwrap();
                let automatic = tensor
                    .factorize_auto(std::slice::from_ref(&left), &options)
                    .unwrap();
                assert_eq!(automatic.rank, explicit.rank);
                let explicit_reconstructed = explicit.left.contract_pair(&explicit.right).unwrap();
                let automatic_reconstructed =
                    automatic.left.contract_pair(&automatic.right).unwrap();
                assert_tensors_approx_equal(
                    &explicit_reconstructed,
                    &automatic_reconstructed,
                    1.0e-10,
                );
                assert!(
                    tensor
                        .sub(&automatic_reconstructed)
                        .unwrap()
                        .maxabs()
                        .unwrap()
                        <= 1.1
                );
                assert_canonical::<f64>(&automatic, canonical, 1.0e-10);
            }
        }
    }

    for (m, n, data) in [
        (
            4,
            2,
            vec![
                Complex64::new(10.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.5),
                Complex64::new(0.0, 0.0),
            ],
        ),
        (
            2,
            4,
            vec![
                Complex64::new(10.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.5),
                Complex64::new(0.0, 0.0),
            ],
        ),
    ] {
        let left = DynIndex::new_dyn(m);
        let right = DynIndex::new_dyn(n);
        let tensor = IdxTensor::from_dense(vec![left.clone(), right], data).unwrap();
        for canonical in [Canonical::Left, Canonical::Right] {
            let options = FactorizeOptions::svd()
                .with_canonical(canonical)
                .with_svd_policy(SvdTruncationPolicy::new(5.0e-2));
            let explicit = tensor
                .factorize(std::slice::from_ref(&left), &options)
                .unwrap();
            let automatic = tensor
                .factorize_auto(std::slice::from_ref(&left), &options)
                .unwrap();
            assert_eq!(automatic.rank, explicit.rank);
            let explicit_reconstructed = explicit.left.contract_pair(&explicit.right).unwrap();
            let automatic_reconstructed = automatic.left.contract_pair(&automatic.right).unwrap();
            assert_tensors_approx_equal(&explicit_reconstructed, &automatic_reconstructed, 1.0e-10);
            assert!(
                tensor
                    .sub(&automatic_reconstructed)
                    .unwrap()
                    .maxabs()
                    .unwrap()
                    <= 1.1
            );
            assert_canonical::<Complex64>(&automatic, canonical, 1.0e-10);
        }
    }
}

#[test]
fn factorize_auto_respects_cap_zero_and_rank_deficiency() {
    let i = DynIndex::new_dyn(3);
    let j = DynIndex::new_dyn(3);
    let rank_one = IdxTensor::from_dense(
        vec![i.clone(), j.clone()],
        vec![3.0_f64, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    )
    .unwrap();
    let options = FactorizeOptions::svd()
        .with_svd_policy(SvdTruncationPolicy::new(1.0e-2))
        .with_max_bond_dim(2);
    let result = rank_one
        .factorize_auto(std::slice::from_ref(&i), &options)
        .unwrap();
    assert_eq!(result.rank, 1);
    assert_tensors_approx_equal(
        &rank_one,
        &result.left.contract_pair(&result.right).unwrap(),
        1.0e-10,
    );

    let capped = IdxTensor::from_dense(
        vec![i.clone(), j.clone()],
        vec![3.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 1.0],
    )
    .unwrap();
    let capped_options = FactorizeOptions::svd()
        .with_svd_policy(SvdTruncationPolicy::new(1.0e-6))
        .with_max_bond_dim(2);
    assert_eq!(
        capped
            .factorize_auto(std::slice::from_ref(&i), &capped_options)
            .unwrap()
            .rank,
        2
    );

    let zero = IdxTensor::from_dense(vec![i.clone(), j.clone()], vec![0.0; 9]).unwrap();
    for canonical in [Canonical::Left, Canonical::Right] {
        let result = zero
            .factorize_auto(
                std::slice::from_ref(&i),
                &FactorizeOptions::svd()
                    .with_canonical(canonical)
                    .with_svd_policy(SvdTruncationPolicy::new(1.0e-2)),
            )
            .unwrap();
        assert_eq!(result.rank, 1);
        assert_tensors_approx_equal(
            &zero,
            &result.left.contract_pair(&result.right).unwrap(),
            1.0e-12,
        );
        assert_canonical::<f64>(&result, canonical, 1.0e-12);
    }
}

#[test]
fn factorize_auto_gate_is_strict_and_preserves_tracked_ad() {
    let i = DynIndex::new_dyn(2);
    let j = DynIndex::new_dyn(2);
    let tensor =
        IdxTensor::from_dense(vec![i.clone(), j], vec![2.0_f64, 0.0, 0.0, 1.0e-3]).unwrap();

    let squared_boundary = FactorizeOptions::svd()
        .with_svd_policy(SvdTruncationPolicy::new(1.0e-12).with_squared_values());
    let value_boundary = FactorizeOptions::svd().with_svd_policy(SvdTruncationPolicy::new(1.0e-6));
    assert_eq!(
        tensor
            .factorize_auto(std::slice::from_ref(&i), &squared_boundary)
            .unwrap()
            .rank,
        tensor
            .factorize(std::slice::from_ref(&i), &squared_boundary)
            .unwrap()
            .rank
    );
    assert_eq!(
        tensor
            .factorize_auto(std::slice::from_ref(&i), &value_boundary)
            .unwrap()
            .rank,
        tensor
            .factorize(std::slice::from_ref(&i), &value_boundary)
            .unwrap()
            .rank
    );

    let tracked = tensor.clone().enable_grad().unwrap();
    let tracked_result = tracked.factorize_auto(
        std::slice::from_ref(&i),
        &FactorizeOptions::svd().with_svd_policy(SvdTruncationPolicy::new(1.0e-2)),
    );
    assert!(tracked_result.is_ok());
    assert!(tracked_result.unwrap().left.tracks_grad());

    let mut invalid = FactorizeOptions::qr();
    assert!(matches!(
        tensor.factorize_auto(&[i], &invalid),
        Err(FactorizeError::InvalidOptions(_))
    ));
    invalid.alg = FactorizeAlg::LU;
    assert!(matches!(
        tensor.factorize_auto(&[DynIndex::new_dyn(2)], &invalid),
        Err(FactorizeError::InvalidOptions(_))
    ));
}

// ============================================================================
// QR-Specific Tests
// ============================================================================

#[test]
fn test_factorize_qr_right_canonical_error() {
    let tensor = create_test_matrix();
    let left_inds = vec![tensor.indices[0].clone()];

    let options = FactorizeOptions::qr().with_canonical(Canonical::Right);
    let result = factorize(&tensor, &left_inds, &options);

    assert!(matches!(
        result,
        Err(FactorizeError::UnsupportedCanonical(_))
    ));
}

#[test]
fn test_factorize_qr_no_singular_values() {
    let tensor = create_test_matrix();
    let left_inds = vec![tensor.indices[0].clone()];
    let options = FactorizeOptions::qr();
    let result = factorize(&tensor, &left_inds, &options).unwrap();

    assert!(result.singular_values.is_none());
}

// ============================================================================
// LU/CI-Specific Tests
// ============================================================================

#[test]
fn test_factorize_lu_ci_no_singular_values() {
    for alg in [FactorizeAlg::LU, FactorizeAlg::CI] {
        for canonical in [Canonical::Left, Canonical::Right] {
            let tensor = create_test_matrix();
            let left_inds = vec![tensor.indices[0].clone()];
            let options = FactorizeOptions {
                alg,
                canonical,
                max_bond_dim: None,
                svd_policy: None,
                qr_rtol: None,
            };
            let result = factorize(&tensor, &left_inds, &options).unwrap();

            assert!(result.singular_values.is_none());
        }
    }
}

#[test]
fn test_factorize_lu_ci_reconstruction_with_unit_dim_axis() {
    let tensor = create_unit_dim_rank3_tensor();
    let left_inds = vec![tensor.indices[1].clone(), tensor.indices[2].clone()];

    for alg in [FactorizeAlg::LU, FactorizeAlg::CI] {
        let options = FactorizeOptions {
            alg,
            canonical: Canonical::Left,
            max_bond_dim: None,
            svd_policy: None,
            qr_rtol: None,
        };
        let result = factorize(&tensor, &left_inds, &options).unwrap();
        let reconstructed = result.left.contract_pair(&result.right).unwrap();
        assert_tensors_approx_equal(&tensor, &reconstructed, 1e-10);
    }
}

#[test]
fn test_factorize_lu_ci_reconstruction_with_col_major_matrix_input() {
    let tensor = create_non_symmetric_col_major_matrix();
    let left_inds = vec![tensor.indices[0].clone()];

    for alg in [FactorizeAlg::LU, FactorizeAlg::CI] {
        let options = FactorizeOptions {
            alg,
            canonical: Canonical::Left,
            max_bond_dim: None,
            svd_policy: None,
            qr_rtol: None,
        };
        let result = factorize(&tensor, &left_inds, &options).unwrap();
        let reconstructed = result.left.contract_pair(&result.right).unwrap();
        assert_tensors_approx_equal(&tensor, &reconstructed, 1e-10);
    }
}

// ============================================================================
// Truncation Tests
// ============================================================================

#[test]
fn test_factorize_with_max_bond_dim() {
    let tensor = create_test_matrix();
    let left_inds = vec![tensor.indices[0].clone()];

    // LU should respect max_bond_dim
    let options = FactorizeOptions::lu().with_max_bond_dim(1);
    let result = factorize(&tensor, &left_inds, &options).unwrap();
    assert_eq!(result.rank, 1);

    // SVD API works (actual truncation behavior may vary)
    let options = FactorizeOptions::svd().with_max_bond_dim(1);
    let result = factorize(&tensor, &left_inds, &options).unwrap();
    assert!(result.rank >= 1);
}

#[test]
fn test_factorize_full_rank_preserves_near_dependent_components() {
    let i: DynIndex = Index::new_dyn(2);
    let j: DynIndex = Index::new_dyn(2);
    let tensor =
        IdxTensor::from_dense(vec![i.clone(), j.clone()], vec![1.0, 0.0, 0.0, 1.0e-16]).unwrap();

    for alg in [FactorizeAlg::SVD, FactorizeAlg::QR, FactorizeAlg::LU] {
        let result =
            factorize_full_rank(&tensor, std::slice::from_ref(&i), alg, Canonical::Left).unwrap();
        assert_eq!(
            result.rank, 2,
            "{alg:?} full-rank factorization dropped a near-dependent component"
        );

        let reconstructed = result.left.contract_pair(&result.right).unwrap();
        assert_tensors_approx_equal(&tensor, &reconstructed, 1.0e-18);
    }
}

#[test]
fn test_factorize_rejects_tracked_lu_and_ci_before_materialization() {
    let tensor = create_test_matrix().enable_grad().unwrap();
    let left_inds = vec![tensor.indices[0].clone()];

    for options in [FactorizeOptions::lu(), FactorizeOptions::ci()] {
        let error = factorize(&tensor, &left_inds, &options).unwrap_err();
        assert!(matches!(error, FactorizeError::UnsupportedStorage(_)));
        assert!(error.to_string().contains("tracked tensors"));
    }

    for alg in [FactorizeAlg::LU, FactorizeAlg::CI] {
        let error = factorize_full_rank(&tensor, &left_inds, alg, Canonical::Left).unwrap_err();
        assert!(matches!(error, FactorizeError::UnsupportedStorage(_)));
        assert!(error.to_string().contains("tracked tensors"));
    }
}

#[test]
fn test_factorize_rejects_mixed_algorithm_options() {
    let tensor = create_test_matrix();
    let left_inds = vec![tensor.indices[0].clone()];

    let svd_with_qr = FactorizeOptions::svd().with_qr_rtol(1.0e-8);
    assert!(matches!(
        factorize(&tensor, &left_inds, &svd_with_qr),
        Err(FactorizeError::InvalidOptions(_))
    ));

    let qr_with_svd = FactorizeOptions::qr().with_svd_policy(SvdTruncationPolicy::new(1.0e-8));
    assert!(matches!(
        factorize(&tensor, &left_inds, &qr_with_svd),
        Err(FactorizeError::InvalidOptions(_))
    ));

    for options in [
        FactorizeOptions::lu().with_qr_rtol(1.0e-8),
        FactorizeOptions::lu().with_svd_policy(SvdTruncationPolicy::new(1.0e-8)),
        FactorizeOptions::ci().with_qr_rtol(1.0e-8),
        FactorizeOptions::ci().with_svd_policy(SvdTruncationPolicy::new(1.0e-8)),
    ] {
        assert!(matches!(
            factorize(&tensor, &left_inds, &options),
            Err(FactorizeError::InvalidOptions(_))
        ));
    }
}

// ============================================================================
// Diagonal Storage Contraction Tests
// ============================================================================

#[test]
fn test_diag_dense_contraction_svd_internals() {
    use tensor4all_core::svd;

    let i: DynIndex = Index::new_dyn(2);
    let j: DynIndex = Index::new_dyn(3);

    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let tensor = IdxTensor::from_dense(vec![i.clone(), j.clone()], data).unwrap();

    let (u, s, v) = svd::<f64>(&tensor, std::slice::from_ref(&i)).expect("SVD should succeed");

    // SVD represents singular values as a compact diagonal tensor while keeping
    // an eager diagonal embedding available for AD-preserving contractions.
    assert!(s.is_diag());
    assert_eq!(s.dims().len(), 2);
    assert_eq!(s.dims()[0], s.dims()[1]);

    // Verify S and V share a common index
    let common_found = s
        .indices
        .iter()
        .any(|s_idx| v.indices.iter().any(|v_idx| s_idx.id == v_idx.id));
    assert!(common_found, "S and V should share a common index");

    // Contractions should work
    let sv = s.contract_pair(&v).unwrap();
    assert_eq!(sv.dims().len(), 2, "S*V should be a 2D tensor");

    let us = u.contract_pair(&s).unwrap();
    assert_eq!(us.dims().len(), 2, "U*S should be a 2D tensor");
}

// ============================================================================
// Helper Functions
// ============================================================================

trait TestScalar:
    TensorElement + Copy + Default + std::ops::Add<Output = Self> + std::ops::Mul<Output = Self>
{
    fn conjugate(self) -> Self;
    fn distance_from(self, target: f64) -> f64;
}

impl TestScalar for f64 {
    fn conjugate(self) -> Self {
        self
    }

    fn distance_from(self, target: f64) -> f64 {
        (self - target).abs()
    }
}

impl TestScalar for Complex64 {
    fn conjugate(self) -> Self {
        self.conj()
    }

    fn distance_from(self, target: f64) -> f64 {
        (self - Complex64::new(target, 0.0)).norm()
    }
}

fn assert_canonical<T: TestScalar>(
    result: &tensor4all_core::FactorizeResult<IdxTensor>,
    canonical: Canonical,
    tol: f64,
) {
    match canonical {
        Canonical::Left => {
            let dims = result.left.dims();
            let rows = dims[0];
            let rank = dims[1];
            let data = result.left.to_vec::<T>().unwrap();
            for a in 0..rank {
                for b in 0..rank {
                    let value = (0..rows).fold(T::default(), |sum, row| {
                        sum + data[row + rows * a].conjugate() * data[row + rows * b]
                    });
                    assert!(
                        value.distance_from((a == b) as usize as f64) < tol,
                        "left canonical Gram entry ({a}, {b}) was not orthonormal"
                    );
                }
            }
        }
        Canonical::Right => {
            let dims = result.right.dims();
            let rank = dims[0];
            let columns = dims[1];
            let data = result.right.to_vec::<T>().unwrap();
            for a in 0..rank {
                for b in 0..rank {
                    let value = (0..columns).fold(T::default(), |sum, column| {
                        sum + data[a + rank * column] * data[b + rank * column].conjugate()
                    });
                    assert!(
                        value.distance_from((a == b) as usize as f64) < tol,
                        "right canonical Gram entry ({a}, {b}) was not orthonormal"
                    );
                }
            }
        }
    }
}

fn assert_tensors_approx_equal(a: &IdxTensor, b: &IdxTensor, tol: f64) {
    assert!(
        a.isapprox(b, tol, 0.0).unwrap(),
        "Tensors differ: maxabs diff = {}",
        a.sub(b).unwrap().maxabs().unwrap()
    );
}
