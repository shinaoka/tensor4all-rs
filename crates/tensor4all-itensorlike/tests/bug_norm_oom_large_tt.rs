//! Regression tests for scalable TensorTrain norms and residual checks.
//!
//! Long tensor-train tests must not rely on wall-clock thresholds or dense
//! `maxabs()` comparisons. They use local transfer-matrix references, structural
//! assertions, and norm-based residuals instead.

use tensor4all_core::defaults::tensordynlen::TensorDynLen;
use tensor4all_core::{AnyScalar, DynIndex, IndexLike};
use tensor4all_itensorlike::TensorTrain;

/// Create a TT with `n_sites` sites, each of physical dimension 2,
/// and bond dimension 2.
fn make_tt(n_sites: usize) -> TensorTrain {
    let mut tensors = Vec::with_capacity(n_sites);
    for k in 0..n_sites {
        let site_idx = DynIndex::new_dyn_with_tag(2, &format!("s={}", k + 1)).unwrap();
        if k == 0 {
            let bond_r = DynIndex::new_dyn(2);
            let t =
                TensorDynLen::from_dense(vec![site_idx, bond_r], vec![1.0, 0.5, 0.3, 1.0]).unwrap();
            tensors.push(t);
        } else if k == n_sites - 1 {
            let bond_l = tensors[k - 1].indices().last().unwrap().clone();
            let t =
                TensorDynLen::from_dense(vec![bond_l, site_idx], vec![1.0, 0.2, 0.7, 1.0]).unwrap();
            tensors.push(t);
        } else {
            let bond_l = tensors[k - 1].indices().last().unwrap().clone();
            let bond_r = DynIndex::new_dyn(2);
            let t = TensorDynLen::from_dense(
                vec![bond_l, site_idx, bond_r],
                vec![1.0, 0.0, 0.5, 0.3, 0.0, 1.0, 0.2, 0.8],
            )
            .unwrap();
            tensors.push(t);
        }
    }
    TensorTrain::new(tensors).expect("Failed to create TensorTrain")
}

fn local_tensor_value(
    data: &[f64],
    left_dim: usize,
    physical_dim: usize,
    left: usize,
    physical: usize,
    right: usize,
) -> f64 {
    data[left + left_dim * (physical + physical_dim * right)]
}

fn reference_norm_squared(tt: &TensorTrain) -> f64 {
    assert!(!tt.is_empty());

    let mut current = Vec::new();
    for site in 0..tt.len() {
        let tensor = tt.tensor(site).unwrap();
        let data = tensor.to_vec::<f64>().unwrap();
        let left_dim = if site == 0 {
            1
        } else {
            tt.linkind(site - 1).unwrap().dim()
        };
        let right_dim = if site + 1 == tt.len() {
            1
        } else {
            tt.linkind(site).unwrap().dim()
        };
        let physical_dim = tensor.dims().iter().product::<usize>() / (left_dim * right_dim);

        if site == 0 {
            current = vec![0.0; right_dim * right_dim];
            for physical in 0..physical_dim {
                for right in 0..right_dim {
                    let value =
                        local_tensor_value(&data, left_dim, physical_dim, 0, physical, right);
                    for right_conj in 0..right_dim {
                        let idx = right * right_dim + right_conj;
                        current[idx] += value
                            * local_tensor_value(
                                &data,
                                left_dim,
                                physical_dim,
                                0,
                                physical,
                                right_conj,
                            );
                    }
                }
            }
            continue;
        }

        let mut next = vec![0.0; right_dim * right_dim];
        for left in 0..left_dim {
            for left_conj in 0..left_dim {
                let env = current[left * left_dim + left_conj];
                for physical in 0..physical_dim {
                    for right in 0..right_dim {
                        let value = local_tensor_value(
                            &data,
                            left_dim,
                            physical_dim,
                            left,
                            physical,
                            right,
                        );
                        for right_conj in 0..right_dim {
                            let idx = right * right_dim + right_conj;
                            next[idx] += env
                                * value
                                * local_tensor_value(
                                    &data,
                                    left_dim,
                                    physical_dim,
                                    left_conj,
                                    physical,
                                    right_conj,
                                );
                        }
                    }
                }
            }
        }
        current = next;
    }

    current[0].max(0.0)
}

fn assert_close_relative(actual: f64, expected: f64) {
    let scale = expected.abs().max(1.0);
    assert!(
        (actual - expected).abs() <= 1e-10 * scale,
        "actual={actual:.12e}, expected={expected:.12e}"
    );
}

/// Sanity check: norm() on a small TT returns a positive finite value.
#[test]
fn test_norm_small_tt_works() {
    let tt = make_tt(4);
    let n = tt.norm().unwrap();
    assert!(n > 0.0, "norm should be positive, got {n}");
    assert!(n.is_finite(), "norm should be finite, got {n}");
    assert_close_relative(n * n, reference_norm_squared(&tt));
}

#[test]
fn test_norm_25_site_tt_matches_local_reference() {
    let tt = make_tt(25);
    let n = tt.norm().unwrap();
    assert!(n > 0.0, "norm should be positive, got {n}");
    assert!(n.is_finite(), "norm should be finite, got {n}");
    assert_close_relative(n * n, reference_norm_squared(&tt));
}

#[test]
fn test_norm_90_site_tt_uses_scalable_structured_path() {
    let tt = make_tt(90);
    assert_eq!(tt.len(), 90);
    assert_eq!(tt.maxbonddim(), 2);

    let n = tt.norm().unwrap();
    assert!(n > 0.0, "norm should be positive, got {n}");
    assert!(n.is_finite(), "norm should be finite, got {n}");
    assert_close_relative(n * n, reference_norm_squared(&tt));
}

#[test]
fn test_long_tt_residual_uses_norm_without_dense_maxabs() {
    let tt = make_tt(25);
    let residual = tt
        .axpby(AnyScalar::new_real(1.0), &tt, AnyScalar::new_real(-1.0))
        .unwrap();

    assert_eq!(residual.len(), tt.len());
    assert!(residual.norm().unwrap() <= 1e-12 * tt.norm().unwrap().max(1.0));
}
