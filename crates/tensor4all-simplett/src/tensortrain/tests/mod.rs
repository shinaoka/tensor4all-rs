use super::*;
use num_complex::Complex64;
use tenferro_tensor::TensorScalar;

// Generic test functions for f64 and Complex64

fn test_tensortrain_zeros_generic<T: TTScalar>() {
    let tt = TensorTrain::<T>::zeros(&[2, 3, 2]);
    assert_eq!(tt.len(), 3);
    assert_eq!(tt.site_dims(), vec![2, 3, 2]);
    assert_eq!(tt.rank(), 1);
}

fn test_tensortrain_constant_generic<T: TTScalar>() {
    let tt = TensorTrain::<T>::constant(&[2, 2], T::from_f64(5.0));
    assert_eq!(tt.len(), 2);

    // Sum should be 5.0 * 2 * 2 = 20.0
    let sum = tt.sum();
    assert!(sum.abs_sq().sqrt() - 20.0 < 1e-10);
}

fn test_tensortrain_constant_single_site_generic<T: TTScalar>() {
    let tt = TensorTrain::<T>::constant(&[3], T::from_f64(5.0));
    assert_eq!(tt.len(), 1);
    assert_eq!(tt.site_dims(), vec![3]);

    let sum = tt.sum();
    assert!((sum.abs_sq().sqrt() - 15.0).abs() < 1e-10);

    let (data, shape) = tt.full_tensor();
    assert_eq!(shape, vec![3]);
    assert_eq!(data.len(), 3);
    for value in data {
        assert!((value - T::from_f64(5.0)).abs_sq().sqrt() < 1e-10);
    }
}

fn test_tensortrain_evaluate_generic<T: TTScalar>() {
    // Create a simple tensor train that returns the product of indices + 1
    let _site_dims = [2, 3];

    // First tensor: values are 1 for index 0, 2 for index 1
    let mut t0: Tensor3<T> = tensor3_zeros(1, 2, 1);
    t0.set3(0, 0, 0, T::from_f64(1.0));
    t0.set3(0, 1, 0, T::from_f64(2.0));

    // Second tensor: values are 1, 2, 3 for indices 0, 1, 2
    let mut t1: Tensor3<T> = tensor3_zeros(1, 3, 1);
    t1.set3(0, 0, 0, T::from_f64(1.0));
    t1.set3(0, 1, 0, T::from_f64(2.0));
    t1.set3(0, 2, 0, T::from_f64(3.0));

    let tt = TensorTrain::new(vec![t0, t1]).unwrap();

    // tt([0, 0]) = 1 * 1 = 1
    let val00 = tt.evaluate(&[0, 0]).unwrap();
    assert!((val00 - T::from_f64(1.0)).abs_sq().sqrt() < 1e-10);
    // tt([1, 2]) = 2 * 3 = 6
    let val12 = tt.evaluate(&[1, 2]).unwrap();
    assert!((val12 - T::from_f64(6.0)).abs_sq().sqrt() < 1e-10);
}

fn test_tensortrain_scale_generic<T: TTScalar>() {
    let mut tt = TensorTrain::<T>::constant(&[2, 2], T::from_f64(1.0));
    tt.scale_mut(T::from_f64(3.0));

    // Sum should be 3.0 * 2 * 2 = 12.0
    let sum = tt.sum();
    assert!((sum.abs_sq().sqrt() - 12.0).abs() < 1e-10);
}

fn test_tensortrain_reverse_generic<T: TTScalar>() {
    let mut t0: Tensor3<T> = tensor3_zeros(1, 2, 1);
    t0.set3(0, 0, 0, T::from_f64(1.0));
    t0.set3(0, 1, 0, T::from_f64(2.0));

    let mut t1: Tensor3<T> = tensor3_zeros(1, 3, 1);
    t1.set3(0, 0, 0, T::from_f64(1.0));
    t1.set3(0, 1, 0, T::from_f64(2.0));
    t1.set3(0, 2, 0, T::from_f64(3.0));

    let tt = TensorTrain::new(vec![t0, t1]).unwrap();
    let tt_rev = tt.reverse();

    assert_eq!(tt_rev.len(), 2);
    assert_eq!(tt_rev.site_dims(), vec![3, 2]);

    // Reversed evaluation: tt_rev([2, 1]) should equal tt([1, 2])
    let diff = tt_rev.evaluate(&[2, 1]).unwrap() - tt.evaluate(&[1, 2]).unwrap();
    assert!(diff.abs_sq().sqrt() < 1e-10);
}

fn test_full_tensor_generic<T: TTScalar>() {
    let tt = TensorTrain::<T>::constant(&[2, 3], T::from_f64(5.0));
    let (data, shape) = tt.full_tensor();

    assert_eq!(shape, vec![2, 3]);
    assert_eq!(data.len(), 6);

    // All elements should be 5.0
    for val in &data {
        assert!((*val - T::from_f64(5.0)).abs_sq().sqrt() < 1e-10);
    }
}

fn test_full_tensor_matches_evaluate_generic<T: TTScalar>() {
    let mut t0: Tensor3<T> = tensor3_zeros(1, 2, 1);
    t0.set3(0, 0, 0, T::from_f64(1.0));
    t0.set3(0, 1, 0, T::from_f64(2.0));

    let mut t1: Tensor3<T> = tensor3_zeros(1, 3, 1);
    t1.set3(0, 0, 0, T::from_f64(1.0));
    t1.set3(0, 1, 0, T::from_f64(2.0));
    t1.set3(0, 2, 0, T::from_f64(3.0));

    let tt = TensorTrain::new(vec![t0, t1]).unwrap();
    let (data, shape) = tt.full_tensor();

    assert_eq!(shape, vec![2, 3]);

    // Check each element matches evaluate
    for i in 0..2 {
        for j in 0..3 {
            let idx = i + 2 * j;
            let expected = tt.evaluate(&[i, j]).unwrap();
            let diff = data[idx] - expected;
            assert!(diff.abs_sq().sqrt() < 1e-10, "Mismatch at [{}, {}]", i, j);
        }
    }
}

fn test_log_norm_matches_norm_generic<T: TTScalar>() {
    let tt = TensorTrain::<T>::constant(&[2, 3], T::from_f64(2.0));

    let norm = tt.norm();
    let log_norm = tt.log_norm();

    // log_norm should equal ln(norm)
    assert!(
        (log_norm - norm.ln()).abs() < 1e-10,
        "log_norm={}, ln(norm)={}",
        log_norm,
        norm.ln()
    );
}

fn test_log_norm_with_varied_values_generic<T: TTScalar>() {
    let mut t0: Tensor3<T> = tensor3_zeros(1, 2, 2);
    t0.set3(0, 0, 0, T::from_f64(1.0));
    t0.set3(0, 0, 1, T::from_f64(0.5));
    t0.set3(0, 1, 0, T::from_f64(2.0));
    t0.set3(0, 1, 1, T::from_f64(1.0));

    let mut t1: Tensor3<T> = tensor3_zeros(2, 2, 1);
    t1.set3(0, 0, 0, T::from_f64(1.0));
    t1.set3(0, 1, 0, T::from_f64(2.0));
    t1.set3(1, 0, 0, T::from_f64(0.5));
    t1.set3(1, 1, 0, T::from_f64(1.5));

    let tt = TensorTrain::new(vec![t0, t1]).unwrap();

    let norm = tt.norm();
    let log_norm = tt.log_norm();

    assert!(
        (log_norm - norm.ln()).abs() < 1e-10,
        "log_norm={}, ln(norm)={}",
        log_norm,
        norm.ln()
    );
}

fn test_log_norm_zero_tensor_generic<T: TTScalar>() {
    let tt = TensorTrain::<T>::zeros(&[2, 3]);
    let log_norm = tt.log_norm();

    assert!(log_norm.is_infinite() && log_norm < 0.0);
}

// f64 tests
#[test]
fn test_tensortrain_zeros_f64() {
    test_tensortrain_zeros_generic::<f64>();
}

#[test]
fn test_tensortrain_constant_f64() {
    test_tensortrain_constant_generic::<f64>();
}

#[test]
fn test_tensortrain_constant_single_site_f64() {
    test_tensortrain_constant_single_site_generic::<f64>();
}

#[test]
fn test_tensortrain_evaluate_f64() {
    test_tensortrain_evaluate_generic::<f64>();
}

#[test]
fn test_tensortrain_scale_f64() {
    test_tensortrain_scale_generic::<f64>();
}

#[test]
fn test_tensortrain_reverse_f64() {
    test_tensortrain_reverse_generic::<f64>();
}

#[test]
fn test_full_tensor_f64() {
    test_full_tensor_generic::<f64>();
}

#[test]
fn test_full_tensor_matches_evaluate_f64() {
    test_full_tensor_matches_evaluate_generic::<f64>();
}

#[test]
fn test_log_norm_matches_norm_f64() {
    test_log_norm_matches_norm_generic::<f64>();
}

#[test]
fn test_log_norm_with_varied_values_f64() {
    test_log_norm_with_varied_values_generic::<f64>();
}

#[test]
fn test_log_norm_zero_tensor_f64() {
    test_log_norm_zero_tensor_generic::<f64>();
}

// Complex64 tests
#[test]
fn test_tensortrain_zeros_c64() {
    test_tensortrain_zeros_generic::<Complex64>();
}

#[test]
fn test_tensortrain_constant_c64() {
    test_tensortrain_constant_generic::<Complex64>();
}

#[test]
fn test_tensortrain_constant_single_site_c64() {
    test_tensortrain_constant_single_site_generic::<Complex64>();
}

#[test]
fn test_tensortrain_evaluate_c64() {
    test_tensortrain_evaluate_generic::<Complex64>();
}

#[test]
fn test_tensortrain_scale_c64() {
    test_tensortrain_scale_generic::<Complex64>();
}

#[test]
fn test_tensortrain_reverse_c64() {
    test_tensortrain_reverse_generic::<Complex64>();
}

#[test]
fn test_full_tensor_c64() {
    test_full_tensor_generic::<Complex64>();
}

#[test]
fn test_full_tensor_matches_evaluate_c64() {
    test_full_tensor_matches_evaluate_generic::<Complex64>();
}

#[test]
fn test_log_norm_matches_norm_c64() {
    test_log_norm_matches_norm_generic::<Complex64>();
}

#[test]
fn test_log_norm_with_varied_values_c64() {
    test_log_norm_with_varied_values_generic::<Complex64>();
}

#[test]
fn test_log_norm_zero_tensor_c64() {
    test_log_norm_zero_tensor_generic::<Complex64>();
}

// ============================================================================
// partial_sum tests
// ============================================================================

fn test_partial_sum_all_dims_generic<
    T: TTScalar + tensor4all_tcicore::Scalar + Default + std::fmt::Debug,
>() {
    // f(i,j,k) = 1.0 for all indices → sum = 2*3*2 = 12
    let tt = TensorTrain::<T>::constant(&[2, 3, 2], T::from_f64(1.0));
    let full_sum = tt.sum();

    let result = tt.partial_sum(&[0, 1, 2]).unwrap();
    // Should be 1-site TT with a single scalar
    assert_eq!(result.len(), 1);
    let partial_val = result.sum();
    assert!(
        (full_sum - partial_val).abs_sq().sqrt() < 1e-12,
        "partial_sum all dims: expected {full_sum:?}, got {partial_val:?}"
    );
}

fn test_partial_sum_no_dims_generic<
    T: TTScalar + tensor4all_tcicore::Scalar + Default + std::fmt::Debug,
>() {
    let tt = TensorTrain::<T>::constant(&[2, 3, 2], T::from_f64(1.0));
    let result = tt.partial_sum(&[]).unwrap();
    assert_eq!(result.len(), 3);
    // Should evaluate identically to original
    for i in 0..2 {
        for j in 0..3 {
            for k in 0..2 {
                let expected = tt.evaluate(&[i, j, k]).unwrap();
                let actual = result.evaluate(&[i, j, k]).unwrap();
                assert!(
                    (expected - actual).abs_sq().sqrt() < 1e-12,
                    "partial_sum no dims: mismatch at ({i},{j},{k})"
                );
            }
        }
    }
}

fn test_partial_sum_single_dim_generic<
    T: TTScalar + tensor4all_tcicore::Scalar + Default + std::fmt::Debug,
>() {
    // f(i,j,k) = (1+i) * (1+j) * (1+k), site_dims = [3, 4, 2]
    // Build TT via rank-1 product structure
    let mut t0 = tensor3_zeros::<T>(1, 3, 1);
    for s in 0..3 {
        t0.set3(0, s, 0, T::from_f64((1 + s) as f64));
    }
    let mut t1 = tensor3_zeros::<T>(1, 4, 1);
    for s in 0..4 {
        t1.set3(0, s, 0, T::from_f64((1 + s) as f64));
    }
    let mut t2 = tensor3_zeros::<T>(1, 2, 1);
    for s in 0..2 {
        t2.set3(0, s, 0, T::from_f64((1 + s) as f64));
    }
    let tt = TensorTrain::new(vec![t0, t1, t2]).unwrap();

    // Sum over dim 1 (j): sum_{j=0}^{3} (1+j) = 1+2+3+4 = 10
    // Result should be TT for g(i,k) = (1+i) * 10 * (1+k)
    let result = tt.partial_sum(&[1]).unwrap();
    assert_eq!(result.len(), 2); // dims 0 and 2 remain

    for i in 0..3 {
        for k in 0..2 {
            let expected = T::from_f64((1 + i) as f64 * 10.0 * (1 + k) as f64);
            let actual = result.evaluate(&[i, k]).unwrap();
            assert!(
                (expected - actual).abs_sq().sqrt() < 1e-10,
                "partial_sum dim=1: mismatch at ({i},{k}): expected {expected:?}, got {actual:?}"
            );
        }
    }
}

fn test_partial_sum_multiple_dims_generic<
    T: TTScalar + tensor4all_tcicore::Scalar + Default + std::fmt::Debug,
>() {
    // Same rank-1 TT: f(i,j,k) = (1+i)*(1+j)*(1+k)
    let mut t0 = tensor3_zeros::<T>(1, 3, 1);
    for s in 0..3 {
        t0.set3(0, s, 0, T::from_f64((1 + s) as f64));
    }
    let mut t1 = tensor3_zeros::<T>(1, 4, 1);
    for s in 0..4 {
        t1.set3(0, s, 0, T::from_f64((1 + s) as f64));
    }
    let mut t2 = tensor3_zeros::<T>(1, 2, 1);
    for s in 0..2 {
        t2.set3(0, s, 0, T::from_f64((1 + s) as f64));
    }
    let tt = TensorTrain::new(vec![t0, t1, t2]).unwrap();

    // Sum over dims 0 and 2: sum_i (1+i) = 6, sum_k (1+k) = 3
    // Result: g(j) = 6 * (1+j) * 3 = 18 * (1+j)
    let result = tt.partial_sum(&[0, 2]).unwrap();
    assert_eq!(result.len(), 1); // only dim 1 remains

    for j in 0..4 {
        let expected = T::from_f64(18.0 * (1 + j) as f64);
        let actual = result.evaluate(&[j]).unwrap();
        assert!(
            (expected - actual).abs_sq().sqrt() < 1e-10,
            "partial_sum dims=[0,2]: mismatch at j={j}: expected {expected:?}, got {actual:?}"
        );
    }
}

#[test]
fn test_partial_sum_all_dims_f64() {
    test_partial_sum_all_dims_generic::<f64>();
}
#[test]
fn test_partial_sum_all_dims_c64() {
    test_partial_sum_all_dims_generic::<Complex64>();
}
#[test]
fn test_partial_sum_no_dims_f64() {
    test_partial_sum_no_dims_generic::<f64>();
}
#[test]
fn test_partial_sum_no_dims_c64() {
    test_partial_sum_no_dims_generic::<Complex64>();
}
#[test]
fn test_partial_sum_single_dim_f64() {
    test_partial_sum_single_dim_generic::<f64>();
}
#[test]
fn test_partial_sum_single_dim_c64() {
    test_partial_sum_single_dim_generic::<Complex64>();
}
#[test]
fn test_partial_sum_multiple_dims_f64() {
    test_partial_sum_multiple_dims_generic::<f64>();
}
#[test]
fn test_partial_sum_multiple_dims_c64() {
    test_partial_sum_multiple_dims_generic::<Complex64>();
}

// ============================================================================
// TT arithmetic tests (port of test_tensortrain.jl)
// ============================================================================

fn test_tt_addition_generic<
    T: TTScalar + tensor4all_tcicore::Scalar + Default + std::fmt::Debug,
>() {
    // Build two rank-1 TTs: f(i,j,k) = (1+i), g(i,j,k) = (1+j)
    let mut t0_a = tensor3_zeros::<T>(1, 3, 1);
    let mut t1_a = tensor3_zeros::<T>(1, 3, 1);
    let mut t2_a = tensor3_zeros::<T>(1, 2, 1);
    for s in 0..3 {
        t0_a.set3(0, s, 0, T::from_f64((1 + s) as f64));
    }
    for s in 0..3 {
        t1_a.set3(0, s, 0, T::from_f64(1.0));
    }
    for s in 0..2 {
        t2_a.set3(0, s, 0, T::from_f64(1.0));
    }
    let tt_a = TensorTrain::new(vec![t0_a, t1_a, t2_a]).unwrap();

    let mut t0_b = tensor3_zeros::<T>(1, 3, 1);
    let mut t1_b = tensor3_zeros::<T>(1, 3, 1);
    let mut t2_b = tensor3_zeros::<T>(1, 2, 1);
    for s in 0..3 {
        t0_b.set3(0, s, 0, T::from_f64(1.0));
    }
    for s in 0..3 {
        t1_b.set3(0, s, 0, T::from_f64((1 + s) as f64));
    }
    for s in 0..2 {
        t2_b.set3(0, s, 0, T::from_f64(1.0));
    }
    let tt_b = TensorTrain::new(vec![t0_b, t1_b, t2_b]).unwrap();

    // tt_a + tt_b
    let tt_add = tt_a.add(&tt_b).unwrap();
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..2 {
                let expected = T::from_f64((1 + i + 1 + j) as f64);
                let actual = tt_add.evaluate(&[i, j, k]).unwrap();
                assert!(
                    (expected - actual).abs_sq().sqrt() < 1e-12,
                    "add: mismatch at ({i},{j},{k})"
                );
            }
        }
    }

    // tt_a - tt_b
    let tt_sub = tt_a.sub(&tt_b).unwrap();
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..2 {
                let expected = T::from_f64((i as i64 - j as i64) as f64);
                let actual = tt_sub.evaluate(&[i, j, k]).unwrap();
                assert!(
                    (expected - actual).abs_sq().sqrt() < 1e-12,
                    "sub: mismatch at ({i},{j},{k})"
                );
            }
        }
    }

    // scalar multiplication
    let tt_scaled = tt_a.scale(T::from_f64(2.5));
    for i in 0..3 {
        let expected = T::from_f64(2.5 * (1 + i) as f64);
        let actual = tt_scaled.evaluate(&[i, 0, 0]).unwrap();
        assert!(
            (expected - actual).abs_sq().sqrt() < 1e-12,
            "scale: mismatch at ({i},0,0)"
        );
    }
}

#[test]
fn test_tt_addition_f64() {
    test_tt_addition_generic::<f64>();
}
#[test]
fn test_tt_addition_c64() {
    test_tt_addition_generic::<Complex64>();
}

// ============================================================================
// SVD compression with tolerance (port of test_tensortrain.jl "compress! (SVD)")
// ============================================================================

fn test_svd_compression_tolerance_generic<T>()
where
    T: TTScalar
        + tensor4all_tcicore::Scalar
        + tensor4all_tcicore::MatrixLuciScalar
        + Default
        + std::fmt::Debug,
    f64: From<<T as TensorScalar>::Real>,
{
    use crate::compression::{CompressionMethod, CompressionOptions};

    // Build a random-ish TT with known structure: sum of two rank-1 terms
    // f(i,j,k) = (1+i)*(1+j)*(1+k) + (2+i)*(3+j)*(4+k)
    let build_tt = || -> TensorTrain<T> {
        let mut t0 = tensor3_zeros::<T>(1, 4, 2);
        let mut t1 = tensor3_zeros::<T>(2, 4, 2);
        let mut t2 = tensor3_zeros::<T>(2, 4, 1);
        for s in 0..4 {
            t0.set3(
                0,
                s,
                0,
                <T as tensor4all_tcicore::Scalar>::from_f64((1 + s) as f64),
            );
            t0.set3(
                0,
                s,
                1,
                <T as tensor4all_tcicore::Scalar>::from_f64((2 + s) as f64),
            );
            t1.set3(
                0,
                s,
                0,
                <T as tensor4all_tcicore::Scalar>::from_f64((1 + s) as f64),
            );
            t1.set3(
                1,
                s,
                1,
                <T as tensor4all_tcicore::Scalar>::from_f64((3 + s) as f64),
            );
            t2.set3(
                0,
                s,
                0,
                <T as tensor4all_tcicore::Scalar>::from_f64((1 + s) as f64),
            );
            t2.set3(
                1,
                s,
                0,
                <T as tensor4all_tcicore::Scalar>::from_f64((4 + s) as f64),
            );
        }
        TensorTrain::new(vec![t0, t1, t2]).unwrap()
    };

    let tt_orig = build_tt();
    let orig_sum = tt_orig.sum();

    let mut tt_svd = build_tt();
    tt_svd
        .compress(&CompressionOptions {
            method: CompressionMethod::SVD,
            tolerance: 1e-10,
            max_bond_dim: usize::MAX,
            normalize_error: true,
        })
        .unwrap();

    let svd_sum = tt_svd.sum();
    assert!(
        <T as tensor4all_tcicore::Scalar>::abs_sq(orig_sum - svd_sum).sqrt() < 1e-8,
        "SVD compression changed sum: orig={orig_sum:?}, svd={svd_sum:?}"
    );

    // Verify element-wise
    for i in 0..4 {
        for j in 0..4 {
            for k in 0..4 {
                let expected = tt_orig.evaluate(&[i, j, k]).unwrap();
                let actual = tt_svd.evaluate(&[i, j, k]).unwrap();
                assert!(
                    <T as tensor4all_tcicore::Scalar>::abs_sq(expected - actual).sqrt() < 1e-8,
                    "SVD compression error at ({i},{j},{k})"
                );
            }
        }
    }
}

#[test]
fn test_svd_compression_tolerance_f64() {
    test_svd_compression_tolerance_generic::<f64>();
}
#[test]
fn test_svd_compression_tolerance_c64() {
    test_svd_compression_tolerance_generic::<Complex64>();
}
