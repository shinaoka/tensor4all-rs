use super::*;
use crate::error::TCIError;
use std::cell::Cell;
use std::rc::Rc;
use tensor4all_simplett::AbstractTensorTrain;

#[test]
fn test_sweep1site_preserves_accuracy() {
    // Build a TCI2 for f(i,j,k) = (i+1)*(j+1)*(k+1)
    let f = |idx: &MultiIndex| ((idx[0] + 1) * (idx[1] + 1) * (idx[2] + 1)) as f64;
    let local_dims = vec![3, 3, 3];
    let first_pivot = vec![vec![1, 1, 1]];
    let options = TCI2Options {
        tolerance: 1e-12,
        max_iter: 20,
        max_bond_dim: usize::MAX,
        ..Default::default()
    };

    let crate::TCI2OptimizationResult {
        mut tci,
        ranks: _ranks,
        errors: _errors,
        ..
    } = crossinterpolate2::<f64, _, fn(&[MultiIndex]) -> Vec<f64>>(
        f,
        None,
        local_dims,
        first_pivot,
        options,
    )
    .unwrap();

    // Perform a forward 1-site sweep with tensor updates
    tci.sweep1site(&f, true, 1e-14, 0.0, usize::MAX, true)
        .unwrap();

    // Verify accuracy is preserved
    let tt = tci.to_tensor_train().unwrap();
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                let expected = ((i + 1) * (j + 1) * (k + 1)) as f64;
                let actual = tt.evaluate(&[i, j, k]).unwrap();
                assert!(
                    (actual - expected).abs() < 1e-10,
                    "f({},{},{}) = {} but got {}",
                    i,
                    j,
                    k,
                    expected,
                    actual
                );
            }
        }
    }
}

#[test]
fn test_make_canonical() {
    let f = |idx: &MultiIndex| idx[0] as f64 + idx[1] as f64 * 0.5 + idx[2] as f64 * 0.25;
    let local_dims = vec![4, 4, 4];
    let first_pivot = vec![vec![1, 1, 1]];
    let options = TCI2Options {
        tolerance: 1e-12,
        max_iter: 20,
        max_bond_dim: usize::MAX,
        ..Default::default()
    };

    let crate::TCI2OptimizationResult { mut tci, .. } = crossinterpolate2::<
        f64,
        _,
        fn(&[MultiIndex]) -> Vec<f64>,
    >(
        f, None, local_dims, first_pivot, options
    )
    .unwrap();

    // Add global pivots then canonicalize
    tci.add_global_pivots(&[vec![3, 3, 3], vec![0, 2, 1]])
        .unwrap();
    tci.make_canonical(&f, 1e-14, 0.0, usize::MAX).unwrap();

    // Verify accuracy after canonicalization
    let tt = tci.to_tensor_train().unwrap();
    for i in 0..4 {
        for j in 0..4 {
            for k in 0..4 {
                let expected = f(&vec![i, j, k]);
                let actual = tt.evaluate(&[i, j, k]).unwrap();
                assert!(
                    (actual - expected).abs() < 1e-10,
                    "f({},{},{}) = {} but got {}",
                    i,
                    j,
                    k,
                    expected,
                    actual
                );
            }
        }
    }
}

#[test]
fn test_fill_site_tensors() {
    let f = |idx: &MultiIndex| (idx[0] + idx[1] + 1) as f64;
    let local_dims = vec![3, 3];
    let first_pivot = vec![vec![1, 1]];
    let options = TCI2Options {
        tolerance: 1e-12,
        max_iter: 20,
        max_bond_dim: usize::MAX,
        ..Default::default()
    };

    let crate::TCI2OptimizationResult { mut tci, .. } = crossinterpolate2::<
        f64,
        _,
        fn(&[MultiIndex]) -> Vec<f64>,
    >(
        f, None, local_dims, first_pivot, options
    )
    .unwrap();

    // Fill site tensors from function
    tci.fill_site_tensors(&f).unwrap();

    // Verify site tensors have correct dimensions
    for p in 0..tci.len() {
        let t = tci.site_tensor(p);
        assert!(t.left_dim() > 0);
        assert!(t.right_dim() > 0);
        assert_eq!(t.site_dim(), 3);
    }
}

#[test]
fn test_tensorci2_new() {
    let tci = TensorCI2::<f64>::new(vec![2, 3, 2]).unwrap();
    assert_eq!(tci.len(), 3);
    assert_eq!(tci.local_dims(), &[2, 3, 2]);
}

#[test]
fn test_tensorci2_requires_two_sites() {
    let result = TensorCI2::<f64>::new(vec![2]);
    assert!(result.is_err());
}

#[test]
fn test_crossinterpolate2_constant() {
    let f = |_: &MultiIndex| 1.0f64;
    let local_dims = vec![2, 2];
    let first_pivot = vec![vec![0, 0]];
    let options = TCI2Options::default();

    let crate::TCI2OptimizationResult {
        tci,
        ranks: _ranks,
        errors: _errors,
        ..
    } = crossinterpolate2::<f64, _, fn(&[MultiIndex]) -> Vec<f64>>(
        f,
        None,
        local_dims,
        first_pivot,
        options,
    )
    .unwrap();

    assert_eq!(tci.len(), 2);
    assert!(tci.rank() >= 1);
}

#[test]
fn test_crossinterpolate2_reports_convergence_and_rank_cap_separately() {
    let options = |max_bond_dim| TCI2Options {
        max_bond_dim,
        ncheck_history: 1,
        nsearch: 0,
        max_nglobal_pivot: 0,
        ..TCI2Options::default()
    };

    let converged = crossinterpolate2::<f64, _, fn(&[MultiIndex]) -> Vec<f64>>(
        |_| 1.0,
        None,
        vec![2, 2],
        vec![vec![0, 0]],
        options(usize::MAX),
    )
    .unwrap();
    assert_eq!(converged.termination, TCI2Termination::Converged);

    let capped = crossinterpolate2::<f64, _, fn(&[MultiIndex]) -> Vec<f64>>(
        |_| 1.0,
        None,
        vec![2, 2],
        vec![vec![0, 0]],
        options(1),
    )
    .unwrap();
    assert_eq!(capped.termination, TCI2Termination::MaxBondDimension);
}

#[test]
fn test_crossinterpolate2_with_batch_function() {
    // Use batch evaluation
    let f = |idx: &MultiIndex| (idx[0] + idx[1] + 1) as f64;
    let batched_f = |indices: &[MultiIndex]| -> Vec<f64> {
        indices
            .iter()
            .map(|idx| (idx[0] + idx[1] + 1) as f64)
            .collect()
    };

    let local_dims = vec![3, 3];
    let first_pivot = vec![vec![1, 1]];
    let options = TCI2Options::default();

    let crate::TCI2OptimizationResult {
        tci,
        ranks: _ranks,
        errors: _errors,
        ..
    } = crossinterpolate2(f, Some(batched_f), local_dims, first_pivot, options).unwrap();

    assert_eq!(tci.len(), 2);
}

#[test]
fn test_crossinterpolate2_rank2_function() {
    // f(i, j) = i + j
    let f = |idx: &MultiIndex| (idx[0] + idx[1]) as f64;
    let local_dims = vec![4, 4];
    let first_pivot = vec![vec![1, 1]];
    let options = TCI2Options {
        tolerance: 1e-12,
        max_iter: 10,
        ..Default::default()
    };

    let crate::TCI2OptimizationResult {
        tci,
        ranks: _ranks,
        errors: _errors,
        ..
    } = crossinterpolate2::<f64, _, fn(&[MultiIndex]) -> Vec<f64>>(
        f,
        None,
        local_dims,
        first_pivot,
        options,
    )
    .unwrap();

    // Should converge to rank 2
    assert!(tci.rank() <= 2, "Expected rank <= 2, got {}", tci.rank());

    // Verify full-grid reconstruction
    let tt = tci.to_tensor_train().unwrap();
    let mut max_error = 0.0f64;
    for i in 0..4 {
        for j in 0..4 {
            let expected = (i + j) as f64;
            let actual = tt.evaluate(&[i, j]).unwrap();
            max_error = max_error.max((actual - expected).abs());
        }
    }
    assert!(
        max_error < 1e-10,
        "Rank-2 reconstruction error too large: {max_error:.2e}"
    );
}

#[test]
fn test_crossinterpolate2_rank2_function_rook_search() {
    let f = |idx: &MultiIndex| (idx[0] + idx[1]) as f64;
    let local_dims = vec![4, 4];
    let first_pivot = vec![vec![1, 1]];
    let options = TCI2Options {
        tolerance: 1e-12,
        max_iter: 10,
        pivot_search: PivotSearchStrategy::Rook,
        ..Default::default()
    };

    let crate::TCI2OptimizationResult {
        tci,
        ranks: _ranks,
        errors,
        ..
    } = crossinterpolate2::<f64, _, fn(&[MultiIndex]) -> Vec<f64>>(
        f,
        None,
        local_dims,
        first_pivot,
        options,
    )
    .unwrap();

    assert!(tci.rank() <= 3, "Expected rank <= 3, got {}", tci.rank());
    let final_error = errors.last().copied().unwrap_or(f64::INFINITY);
    assert!(
        final_error < 0.1,
        "Expected small error, got {}",
        final_error
    );
}

/// Port of Julia test_tensorci2.jl: "pivoterrors".
#[test]
fn test_crossinterpolate2_pivot_errors_match_diagonal_values() {
    let diagonal = [1.0_f64, 1e-5, 0.0];
    let f = |idx: &MultiIndex| {
        if idx[0] == idx[1] {
            diagonal[idx[0]]
        } else {
            0.0
        }
    };
    let crate::TCI2OptimizationResult {
        tci,
        ranks: _ranks,
        errors: _errors,
        ..
    } = crossinterpolate2::<f64, _, fn(&[MultiIndex]) -> Vec<f64>>(
        f,
        None,
        vec![3, 3],
        vec![vec![0, 0]],
        TCI2Options {
            tolerance: 1e-8,
            ..TCI2Options::default()
        },
    )
    .unwrap();

    assert_eq!(tci.pivot_errors().len(), diagonal.len());
    for (&actual, &expected) in tci.pivot_errors().iter().zip(diagonal.iter()) {
        assert!((actual - expected).abs() < 1e-14);
    }
}

#[test]
fn test_crossinterpolate2_rook_search_uses_partial_batch_requests() {
    let f = |idx: &MultiIndex| ((idx[0] + 1) * (idx[1] + 1)) as f64;
    let max_batch = Rc::new(Cell::new(0usize));
    let total_requested = Rc::new(Cell::new(0usize));
    let batched_f = {
        let max_batch = Rc::clone(&max_batch);
        let total_requested = Rc::clone(&total_requested);
        move |indices: &[MultiIndex]| -> Vec<f64> {
            max_batch.set(max_batch.get().max(indices.len()));
            total_requested.set(total_requested.get() + indices.len());
            indices
                .iter()
                .map(|idx| ((idx[0] + 1) * (idx[1] + 1)) as f64)
                .collect()
        }
    };

    let local_dims = vec![8, 8];
    let first_pivot = vec![vec![0, 0]];
    let options = TCI2Options {
        max_iter: 1,
        max_bond_dim: 2,
        pivot_search: PivotSearchStrategy::Rook,
        ..Default::default()
    };

    let crate::TCI2OptimizationResult {
        tci: _tci,
        ranks: _ranks,
        errors: _errors,
        ..
    } = crossinterpolate2(f, Some(batched_f), local_dims, first_pivot, options).unwrap();

    assert!(
        max_batch.get() < 64,
        "rook search should request partial batches, got full batch of {} entries",
        max_batch.get()
    );
    assert!(
        total_requested.get() < 64,
        "rook search should avoid evaluating the full Pi matrix, requested {} entries",
        total_requested.get()
    );
}

#[test]
fn test_crossinterpolate2_rook_search_rejects_bad_batch_length() {
    let f = |idx: &MultiIndex| ((idx[0] + 1) * (idx[1] + 1)) as f64;
    let bad_batched_f = |_indices: &[MultiIndex]| -> Vec<f64> { vec![1.0] };
    let local_dims = vec![4, 4];
    let first_pivot = vec![vec![0, 0]];
    let options = TCI2Options {
        max_iter: 1,
        max_bond_dim: 2,
        pivot_search: PivotSearchStrategy::Rook,
        ..Default::default()
    };

    let result = crossinterpolate2(f, Some(bad_batched_f), local_dims, first_pivot, options);
    assert!(matches!(result, Err(TCIError::InvalidOperation { .. })));
}

#[test]
fn test_crossinterpolate2_full_search_rejects_bad_batch_length() {
    let f = |idx: &MultiIndex| ((idx[0] + 1) * (idx[1] + 1)) as f64;
    let bad_batched_f = |_indices: &[MultiIndex]| -> Vec<f64> { vec![1.0] };
    let local_dims = vec![4, 4];
    let first_pivot = vec![vec![0, 0]];
    let options = TCI2Options {
        max_iter: 1,
        max_bond_dim: 2,
        pivot_search: PivotSearchStrategy::Full,
        ..Default::default()
    };

    let result = crossinterpolate2(f, Some(bad_batched_f), local_dims, first_pivot, options);
    assert!(matches!(result, Err(TCIError::InvalidOperation { .. })));
}

#[test]
fn test_crossinterpolate2_3sites_constant() {
    // 3-site constant function: f(i, j, k) = 1.0
    let f = |_: &MultiIndex| 1.0f64;
    let local_dims = vec![2, 2, 2];
    let first_pivot = vec![vec![0, 0, 0]];
    let options = TCI2Options::default();

    let result = crossinterpolate2::<f64, _, fn(&[MultiIndex]) -> Vec<f64>>(
        f,
        None,
        local_dims,
        first_pivot,
        options,
    );

    assert!(
        result.is_ok(),
        "crossinterpolate2 failed: {:?}",
        result.err()
    );
    let crate::TCI2OptimizationResult { tci, .. } = result.unwrap();

    assert_eq!(tci.len(), 3);
    assert!(tci.rank() >= 1);

    // Verify site tensor dimensions
    for p in 0..tci.len() {
        let t = tci.site_tensor(p);
        assert!(t.left_dim() > 0, "Site {} left_dim should be > 0", p);
        assert!(t.right_dim() > 0, "Site {} right_dim should be > 0", p);
    }

    // Test to_tensor_train conversion
    let tt_result = tci.to_tensor_train();
    assert!(
        tt_result.is_ok(),
        "to_tensor_train failed: {:?}",
        tt_result.err()
    );

    let tt = tt_result.unwrap();
    assert_eq!(tt.len(), 3);

    // Verify TT can be evaluated
    let val = tt.evaluate(&[0, 0, 0]).unwrap();
    assert!((val - 1.0).abs() < 1e-10, "Expected 1.0, got {}", val);
}

#[test]
fn test_crossinterpolate2_4sites_product() {
    // 4-site product function: f(i, j, k, l) = (1+i) * (1+j) * (1+k) * (1+l)
    let f = |idx: &MultiIndex| {
        (1 + idx[0]) as f64 * (1 + idx[1]) as f64 * (1 + idx[2]) as f64 * (1 + idx[3]) as f64
    };
    let local_dims = vec![2, 2, 2, 2];
    let first_pivot = vec![vec![0, 0, 0, 0]];
    let options = TCI2Options::default();

    let result = crossinterpolate2::<f64, _, fn(&[MultiIndex]) -> Vec<f64>>(
        f,
        None,
        local_dims,
        first_pivot,
        options,
    );

    assert!(
        result.is_ok(),
        "crossinterpolate2 failed: {:?}",
        result.err()
    );
    let crate::TCI2OptimizationResult { tci, .. } = result.unwrap();

    assert_eq!(tci.len(), 4);

    // Test to_tensor_train conversion
    let tt_result = tci.to_tensor_train();
    assert!(
        tt_result.is_ok(),
        "to_tensor_train failed: {:?}",
        tt_result.err()
    );

    let tt = tt_result.unwrap();
    assert_eq!(tt.len(), 4);

    // Verify evaluations
    let val = tt.evaluate(&[0, 0, 0, 0]).unwrap();
    assert!((val - 1.0).abs() < 1e-10, "f(0,0,0,0) = 1, got {}", val);

    let val = tt.evaluate(&[1, 1, 1, 1]).unwrap();
    assert!((val - 16.0).abs() < 1e-10, "f(1,1,1,1) = 16, got {}", val);
}

#[test]
fn test_crossinterpolate2_5sites_constant() {
    // 5-site constant function
    let f = |_: &MultiIndex| 2.5f64;
    let local_dims = vec![2, 2, 2, 2, 2];
    let first_pivot = vec![vec![0, 0, 0, 0, 0]];
    let options = TCI2Options::default();

    let result = crossinterpolate2::<f64, _, fn(&[MultiIndex]) -> Vec<f64>>(
        f,
        None,
        local_dims,
        first_pivot,
        options,
    );

    assert!(
        result.is_ok(),
        "crossinterpolate2 failed: {:?}",
        result.err()
    );
    let crate::TCI2OptimizationResult { tci, .. } = result.unwrap();

    assert_eq!(tci.len(), 5);

    let tt_result = tci.to_tensor_train();
    assert!(
        tt_result.is_ok(),
        "to_tensor_train failed: {:?}",
        tt_result.err()
    );

    let tt = tt_result.unwrap();

    // Sum should be 2.5 * 2^5 = 80
    let sum = tt.sum();
    assert!((sum - 80.0).abs() < 1e-8, "Expected sum=80, got {}", sum);
}

/// Regression test for issue #227:
/// crossinterpolate2 panics with NaN in LU for oscillatory functions (e.g. sin)
/// Reproduces the exact scenario from the issue: sin(10*x) on a quantics grid.
#[test]
fn test_crossinterpolate2_sin_quantics() {
    let r = 6;
    let local_dims = vec![2; r];

    // Quantics-to-coordinate mapping: indices [q0,..,q5] (each 0 or 1)
    // -> integer = sum q_i * 2^(R-1-i), coordinate x = integer / 2^R
    let f = |indices: &MultiIndex| -> f64 {
        let mut int_idx: usize = 0;
        for (i, &q) in indices.iter().enumerate() {
            int_idx += q * (1 << (r - 1 - i));
        }
        let x = int_idx as f64 / (1u64 << r) as f64;
        (10.0 * x).sin()
    };

    // Use [0,1,0,0,0,0] as initial pivot so f != 0
    // (x = 0.5, sin(10*0.5) = sin(5) ≈ -0.959)
    let first_pivot = vec![vec![0, 1, 0, 0, 0, 0]];
    let options = TCI2Options {
        tolerance: 1e-10,
        max_bond_dim: usize::MAX,
        max_iter: 20,
        ..Default::default()
    };

    // This previously panicked with "NaN in L matrix" inside rrlu_inplace
    let result = crossinterpolate2::<f64, _, fn(&[MultiIndex]) -> Vec<f64>>(
        f,
        None,
        local_dims,
        first_pivot,
        options,
    );

    assert!(
        result.is_ok(),
        "crossinterpolate2 failed for sin(10x): {:?}",
        result.err()
    );
}

#[test]
fn test_crossinterpolate2_rank2_full_grid_reconstruction() {
    // Reproduce issue #259: f(i,j) = i + j should be reconstructed exactly
    let f = |idx: &MultiIndex| (idx[0] + idx[1]) as f64;
    let local_dims = vec![4, 4];
    let first_pivot = vec![vec![1, 1]];
    let options = TCI2Options {
        tolerance: 1e-12,
        max_iter: 20,
        max_bond_dim: usize::MAX,
        ..Default::default()
    };

    let crate::TCI2OptimizationResult {
        tci,
        ranks: _ranks,
        errors: _errors,
        ..
    } = crossinterpolate2::<f64, _, fn(&[MultiIndex]) -> Vec<f64>>(
        f,
        None,
        local_dims,
        first_pivot,
        options,
    )
    .unwrap();

    let tt = tci.to_tensor_train().unwrap();

    let mut max_error = 0.0f64;
    for i in 0..4 {
        for j in 0..4 {
            let expected = (i + j) as f64;
            let actual = tt.evaluate(&[i, j]).unwrap();
            max_error = max_error.max((actual - expected).abs());
        }
    }
    eprintln!("rank={} max_error={:.6e}", tci.rank(), max_error);
    assert!(
        max_error < 1e-10,
        "Full-grid reconstruction error too large: {max_error:.6e} (rank={})",
        tci.rank()
    );
}

/// Test non-strictly-nested mode and rook pivot search don't cause runtime errors.
/// Port of Julia test_tensorci2.jl: pivotsearch/strictlynested parameter variants.
#[test]
fn test_crossinterpolate2_non_strictly_nested() {
    let f = |idx: &MultiIndex| {
        let sum_sq: f64 = idx.iter().map(|&v| (v as f64).powi(2)).sum();
        1.0 / (sum_sq + 1.0)
    };
    let local_dims = vec![4; 3];
    let first_pivot = vec![vec![1; 3]];

    // Non-strictly-nested (default) with Full pivot search
    let options = TCI2Options {
        tolerance: 1e-8,
        max_iter: 10,
        strictly_nested: false,
        pivot_search: PivotSearchStrategy::Full,
        ..Default::default()
    };
    let crate::TCI2OptimizationResult {
        tci,
        ranks: _ranks,
        errors: _errors,
        ..
    } = crossinterpolate2::<f64, _, fn(&[MultiIndex]) -> Vec<f64>>(
        f,
        None,
        local_dims.clone(),
        first_pivot.clone(),
        options,
    )
    .unwrap();

    let tt = tci.to_tensor_train().unwrap();
    let val = tt.evaluate(&[0, 0, 0]).unwrap();
    assert!(
        (val - 1.0).abs() < 1e-6,
        "non-strictly-nested Full: f(0,0,0)={val}"
    );

    // Strictly-nested with Full pivot search
    let options = TCI2Options {
        tolerance: 1e-8,
        max_iter: 10,
        strictly_nested: true,
        pivot_search: PivotSearchStrategy::Full,
        ..Default::default()
    };
    let crate::TCI2OptimizationResult {
        tci,
        ranks: _ranks,
        errors: _errors,
        ..
    } = crossinterpolate2::<f64, _, fn(&[MultiIndex]) -> Vec<f64>>(
        f,
        None,
        local_dims.clone(),
        first_pivot.clone(),
        options,
    )
    .unwrap();

    let tt = tci.to_tensor_train().unwrap();
    let val = tt.evaluate(&[0, 0, 0]).unwrap();
    assert!(
        (val - 1.0).abs() < 1e-6,
        "strictly-nested Full: f(0,0,0)={val}"
    );

    // Non-strictly-nested with Rook pivot search
    let options = TCI2Options {
        tolerance: 1e-8,
        max_iter: 10,
        strictly_nested: false,
        pivot_search: PivotSearchStrategy::Rook,
        ..Default::default()
    };
    let crate::TCI2OptimizationResult {
        tci,
        ranks: _ranks,
        errors: _errors,
        ..
    } = crossinterpolate2::<f64, _, fn(&[MultiIndex]) -> Vec<f64>>(
        f,
        None,
        local_dims.clone(),
        first_pivot.clone(),
        options,
    )
    .unwrap();

    let tt = tci.to_tensor_train().unwrap();
    let val = tt.evaluate(&[0, 0, 0]).unwrap();
    assert!(
        (val - 1.0).abs() < 1e-6,
        "non-strictly-nested Rook: f(0,0,0)={val}"
    );

    // Strictly-nested with Rook pivot search
    let options = TCI2Options {
        tolerance: 1e-8,
        max_iter: 10,
        strictly_nested: true,
        pivot_search: PivotSearchStrategy::Rook,
        ..Default::default()
    };
    let crate::TCI2OptimizationResult {
        tci,
        ranks: _ranks,
        errors: _errors,
        ..
    } = crossinterpolate2::<f64, _, fn(&[MultiIndex]) -> Vec<f64>>(
        f,
        None,
        local_dims,
        first_pivot,
        options,
    )
    .unwrap();

    let tt = tci.to_tensor_train().unwrap();
    let val = tt.evaluate(&[0, 0, 0]).unwrap();
    assert!(
        (val - 1.0).abs() < 1e-6,
        "strictly-nested Rook: f(0,0,0)={val}"
    );
}

/// Port of Julia test: Lorentz MPS with Float64
/// From test_tensorci2.jl: coeff=1.0, localdims=fill(10, 5)
#[test]
fn test_crossinterpolate2_lorentz_f64() {
    // Lorentz function: 1/(sum(v^2) + 1)
    let f = |idx: &MultiIndex| {
        let sum_sq: f64 = idx.iter().map(|&v| (v as f64).powi(2)).sum();
        1.0 / (sum_sq + 1.0)
    };
    let local_dims = vec![10; 5];
    let first_pivot = vec![vec![1; 5]];
    let options = TCI2Options {
        tolerance: 1e-8,
        max_iter: 20,
        max_bond_dim: usize::MAX,
        ..Default::default()
    };

    let crate::TCI2OptimizationResult {
        tci,
        ranks: _ranks,
        errors,
        ..
    } = crossinterpolate2::<f64, _, fn(&[MultiIndex]) -> Vec<f64>>(
        f,
        None,
        local_dims.clone(),
        first_pivot,
        options,
    )
    .unwrap();

    // Verify convergence
    let final_error = *errors.last().unwrap();
    assert!(
        final_error < 1e-6,
        "Lorentz f64 did not converge: error = {:.2e}",
        final_error
    );

    // Verify reconstruction at several points
    let tt = tci.to_tensor_train().unwrap();
    for indices in &[
        vec![0, 0, 0, 0, 0],
        vec![1, 2, 3, 4, 5],
        vec![9, 9, 9, 9, 9],
    ] {
        let expected = f(indices);
        let actual = tt.evaluate(indices).unwrap();
        let abs_err = (actual - expected).abs();
        assert!(
            abs_err < 1e-6,
            "Lorentz f64 error at {:?}: expected {}, got {}, err={}",
            indices,
            expected,
            actual,
            abs_err
        );
    }
}

/// Port of Julia test: Lorentz MPS with Complex64
/// From test_tensorci2.jl: coeff=0.5-1.0im
#[test]
fn test_crossinterpolate2_lorentz_c64() {
    use num_complex::Complex64;

    let coeff = Complex64::new(0.5, -1.0);
    let f = |idx: &MultiIndex| {
        let sum_sq: f64 = idx.iter().map(|&v| (v as f64).powi(2)).sum();
        coeff / Complex64::new(sum_sq + 1.0, 0.0)
    };
    let local_dims = vec![10; 5];
    let first_pivot = vec![vec![1; 5]];
    let options = TCI2Options {
        tolerance: 1e-8,
        max_iter: 20,
        max_bond_dim: usize::MAX,
        ..Default::default()
    };

    let crate::TCI2OptimizationResult {
        tci,
        ranks: _ranks,
        errors,
        ..
    } = crossinterpolate2::<Complex64, _, fn(&[MultiIndex]) -> Vec<Complex64>>(
        f,
        None,
        local_dims,
        first_pivot,
        options,
    )
    .unwrap();

    let final_error = *errors.last().unwrap();
    assert!(
        final_error < 1e-6,
        "Lorentz c64 did not converge: error = {:.2e}",
        final_error
    );

    // Verify reconstruction
    let tt = tci.to_tensor_train().unwrap();
    let test_idx = vec![1, 2, 3, 4, 5];
    let expected = f(&test_idx);
    let actual = tt.evaluate(&test_idx).unwrap();
    let abs_err = (actual - expected).norm();
    assert!(
        abs_err < 1e-6,
        "Lorentz c64 error: expected {}, got {}, err={}",
        expected,
        actual,
        abs_err
    );
}

/// Port of Julia test: convergence criterion
/// From test_tensorci2.jl: convergencecriterion function
#[test]
fn test_convergence_criterion() {
    // Not converged: too few history points
    assert_eq!(
        super::convergence_criterion(&[2], &[1e-10], &[0], 1e-8, 100, 3),
        None
    );

    // Converged: 3 iterations with low error, no global pivots, stable rank
    assert_eq!(
        super::convergence_criterion(&[5, 5, 5], &[1e-10, 1e-10, 1e-10], &[0, 0, 0], 1e-8, 100, 3,),
        Some(TCI2Termination::Converged)
    );

    // Normalized errors are compared with the configured relative tolerance,
    // not with an absolute tolerance rescaled by a large function magnitude.
    assert_eq!(
        super::convergence_criterion(&[5, 5, 5], &[1e-6, 1e-6, 1e-6], &[0, 0, 0], 1e-8, 100, 3,),
        None
    );

    // Not converged: global pivots still being added
    assert_eq!(
        super::convergence_criterion(&[5, 5, 5], &[1e-10, 1e-10, 1e-10], &[0, 1, 0], 1e-8, 100, 3,),
        None
    );

    // Reaching the rank cap is a limit, not convergence.
    assert_eq!(
        super::convergence_criterion(&[100, 100, 100], &[0.1, 0.1, 0.1], &[5, 5, 5], 1e-8, 100, 3,),
        Some(TCI2Termination::MaxBondDimension)
    );

    // Not converged: rank still growing
    assert_eq!(
        super::convergence_criterion(&[3, 4, 5], &[1e-10, 1e-10, 1e-10], &[0, 0, 0], 1e-8, 100, 3,),
        None
    );
}

/// Port of Julia test: global search with oscillatory function
/// From test_globalsearch.jl
#[test]
fn test_global_search_oscillatory() {
    use crate::globalsearch::estimate_true_error;

    // Oscillatory function on quantics grid
    let r = 10;
    let local_dims = vec![2; r];

    let f = |indices: &MultiIndex| -> f64 {
        let mut int_idx: usize = 0;
        for (i, &q) in indices.iter().enumerate() {
            int_idx += q * (1 << (r - 1 - i));
        }
        let x = int_idx as f64 / (1u64 << r) as f64;
        (-x).exp() + 1e-3 * (1000.0 * x).sin()
    };

    let first_pivot = vec![vec![0; r]];
    // Use non-zero initial pivot
    let options = TCI2Options {
        tolerance: 1e-4,
        max_iter: 20,
        max_bond_dim: usize::MAX,
        normalize_error: false,
        ..Default::default()
    };

    let crate::TCI2OptimizationResult {
        tci,
        ranks: _ranks,
        errors: _errors,
        ..
    } = crossinterpolate2::<f64, _, fn(&[MultiIndex]) -> Vec<f64>>(
        f,
        None,
        local_dims,
        first_pivot,
        options,
    )
    .unwrap();

    let tt = tci.to_tensor_train().unwrap();

    // Estimate true error
    let mut rng = rand::rng();
    let pivot_errors = estimate_true_error(&tt, &f, 20, None, &mut rng);

    // Verify errors are sorted in descending order
    for i in 0..pivot_errors.len().saturating_sub(1) {
        assert!(
            pivot_errors[i].1 >= pivot_errors[i + 1].1,
            "Errors not sorted at position {}: {} < {}",
            i,
            pivot_errors[i].1,
            pivot_errors[i + 1].1
        );
    }

    // Verify error consistency
    for (pivot, error) in &pivot_errors {
        let f_val = f(pivot);
        let tt_val = tt.evaluate(pivot).unwrap();
        let actual_error = (f_val - tt_val).abs();
        assert!(
            (actual_error - error).abs() < 1e-10,
            "Error inconsistency: reported={}, actual={}",
            error,
            actual_error
        );
    }
}

/// Port of Julia test_tensorci2.jl: "custom global pivot finder"
///
/// Tests that the GlobalPivotFinder trait can be implemented with a custom
/// strategy. Uses a random pivot finder (returns random multi-indices).
#[test]
fn test_custom_global_pivot_finder() {
    use crate::globalpivot::{GlobalPivotFinder, GlobalPivotSearchInput};
    use rand::Rng;

    // Custom finder: returns random pivots (same as Julia's CustomGlobalPivotFinder)
    struct RandomPivotFinder {
        npivots: usize,
    }

    impl GlobalPivotFinder for RandomPivotFinder {
        fn find_global_pivots<T, F>(
            &self,
            input: &GlobalPivotSearchInput<T>,
            _f: &F,
            _abs_tol: f64,
            rng: &mut impl Rng,
        ) -> Vec<MultiIndex>
        where
            T: tensor4all_tcicore::Scalar + tensor4all_simplett::TTScalar,
            F: Fn(&MultiIndex) -> T,
        {
            (0..self.npivots)
                .map(|_| {
                    input
                        .local_dims
                        .iter()
                        .map(|&d| rng.random_range(0..d))
                        .collect()
                })
                .collect()
        }
    }

    // f(x) = exp(-x) on quantics grid with R=8 bits
    let r = 8;
    let local_dims = vec![2; r];
    let f = |indices: &MultiIndex| -> f64 {
        let mut int_idx: usize = 0;
        for (i, &q) in indices.iter().enumerate() {
            int_idx += q * (1 << (r - 1 - i));
        }
        let x = int_idx as f64 / (1u64 << r) as f64;
        (-x).exp()
    };

    // Use default crossinterpolate2 (which uses DefaultGlobalPivotFinder internally)
    let first_pivot = vec![vec![0; r], {
        let mut p = vec![0; r];
        p[0] = 1;
        p
    }];
    let options = TCI2Options {
        tolerance: 1e-4,
        max_iter: 10,
        max_bond_dim: usize::MAX,
        normalize_error: false,
        ..Default::default()
    };

    let crate::TCI2OptimizationResult {
        tci,
        ranks: _ranks,
        errors: _errors,
        ..
    } = crossinterpolate2::<f64, _, fn(&[MultiIndex]) -> Vec<f64>>(
        f,
        None,
        local_dims.clone(),
        first_pivot,
        options,
    )
    .unwrap();

    let tt = tci.to_tensor_train().unwrap();

    // Now test that the custom finder can be used to find pivots
    let finder = RandomPivotFinder { npivots: 10 };
    let input = GlobalPivotSearchInput {
        local_dims,
        current_tt: tt.clone(),
        max_sample_value: tci.max_sample_value(),
        i_set: (0..tci.len()).map(|p| tci.i_set(p).to_vec()).collect(),
        j_set: (0..tci.len()).map(|p| tci.j_set(p).to_vec()).collect(),
    };

    let mut rng = rand::rng();
    let pivots = finder.find_global_pivots(&input, &f, 1e-4, &mut rng);

    // Custom finder should return npivots random pivots
    assert_eq!(pivots.len(), 10);
    for pivot in &pivots {
        assert_eq!(pivot.len(), r);
        for (i, &idx) in pivot.iter().enumerate() {
            assert!(idx < input.local_dims[i], "Index out of bounds at site {i}");
        }
    }

    // Verify the TT can evaluate at the custom pivots
    for pivot in &pivots {
        let tt_val = tt.evaluate(pivot).unwrap();
        let f_val = f(pivot);
        // Error should be bounded by tolerance
        assert!(
            (tt_val - f_val).abs() < 1e-2,
            "Large error at custom pivot {:?}: tt={}, f={}, err={}",
            pivot,
            tt_val,
            f_val,
            (tt_val - f_val).abs()
        );
    }
}

#[test]
fn test_optimize_with_finder_invokes_custom_finder() {
    use crate::globalpivot::{GlobalPivotFinder, GlobalPivotSearchInput};
    use rand::Rng;
    use std::cell::Cell;
    use std::rc::Rc;

    struct CountingFinder {
        calls: Rc<Cell<usize>>,
    }

    impl GlobalPivotFinder for CountingFinder {
        fn find_global_pivots<T, F>(
            &self,
            input: &GlobalPivotSearchInput<T>,
            _f: &F,
            _abs_tol: f64,
            _rng: &mut impl Rng,
        ) -> Vec<MultiIndex>
        where
            T: tensor4all_tcicore::Scalar + tensor4all_simplett::TTScalar,
            F: Fn(&MultiIndex) -> T,
        {
            self.calls.set(self.calls.get() + 1);
            vec![vec![input.local_dims[0] - 1, input.local_dims[1] - 1]]
        }
    }

    let f = |idx: &MultiIndex| ((idx[0] + 1) * (idx[1] + 2)) as f64;
    let mut tci = TensorCI2::<f64>::new(vec![4, 4]).unwrap();
    tci.add_global_pivots(&[vec![0, 0]]).unwrap();

    let calls = Rc::new(Cell::new(0));
    let finder = CountingFinder {
        calls: Rc::clone(&calls),
    };
    let crate::TCI2OptimizationResult {
        tci,
        ranks,
        errors,
        termination,
    } = optimize_with_finder::<f64, _, fn(&[MultiIndex]) -> Vec<f64>, _>(
        tci,
        f,
        None,
        TCI2Options {
            max_iter: 3,
            ncheck_history: 1,
            ..TCI2Options::default()
        },
        finder,
    )
    .unwrap();

    assert_eq!(ranks.len(), 3);
    assert_eq!(errors.len(), 3);
    assert!(tci.rank() >= 1);
    assert_eq!(calls.get(), 3);
    assert_eq!(termination, TCI2Termination::MaxIterations);
}
