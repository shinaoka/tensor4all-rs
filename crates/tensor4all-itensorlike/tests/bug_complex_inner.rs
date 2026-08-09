//! Regression tests for TensorTrain::inner(self, self) with non-standard
//! site tensor index ordering.
//!
//! <x|x> = Σ |x_i|² should always be non-negative real, but inner() returns
//! the correct value even when site tensors are not already arranged as
//! [BOND_left, SITE..., BOND_right].
//!
//! This happens in practice when loading MPS/MPO data from HDF5 files written by
//! ITensors.jl, which doesn't enforce a particular in-memory index ordering.

use num_complex::Complex64;
use tensor4all_core::{DynIndex, TensorDynLen};
use tensor4all_itensorlike::TensorTrain;

fn c(re: f64, im: f64) -> Complex64 {
    Complex64::new(re, im)
}

/// Minimal reproduction: 2-site complex TT.
/// Same data, same indices, different index ordering → same inner() result.
#[test]
fn test_inner_wrong_with_nonstandard_index_order() {
    let s0 = DynIndex::new_dyn_with_tag(2, "s=1").unwrap();
    let s1 = DynIndex::new_dyn_with_tag(2, "s=2").unwrap();
    let b = DynIndex::new_dyn(2);

    let data0 = vec![c(1.0, 2.0), c(3.0, -1.0), c(-0.5, 0.7), c(2.0, 1.5)];
    let data1 = vec![c(0.5, -0.3), c(1.0, 0.8), c(-1.2, 0.4), c(0.3, -0.9)];

    // Standard ordering: site 0 = [s0, b], site 1 = [b, s1] → PASSES
    let tt_std = TensorTrain::new(vec![
        TensorDynLen::from_dense(vec![s0.clone(), b.clone()], data0.clone()).unwrap(),
        TensorDynLen::from_dense(vec![b.clone(), s1.clone()], data1.clone()).unwrap(),
    ])
    .unwrap();

    // Non-standard ordering: site 1 = [s1, b] (site index first)
    let tt_ns = TensorTrain::new(vec![
        TensorDynLen::from_dense(vec![s0.clone(), b.clone()], data0).unwrap(),
        TensorDynLen::from_dense(vec![s1.clone(), b.clone()], data1).unwrap(),
    ])
    .unwrap();

    // Both TTs represent the same tensor (same indices, same data layout
    // relative to each index), so they must have the same dense norm
    let dense_std = tt_std.to_dense().unwrap();
    let dense_ns = tt_ns.to_dense().unwrap();
    let norm_sq_std = dense_std.norm() * dense_std.norm();
    let norm_sq_ns = dense_ns.norm() * dense_ns.norm();

    eprintln!("dense norm² std={:.6e}, ns={:.6e}", norm_sq_std, norm_sq_ns);

    // inner(x, x) for both must be real, non-negative, and match dense norm²
    let inner_std = tt_std.inner(&tt_std).unwrap();
    let inner_ns = tt_ns.inner(&tt_ns).unwrap();
    let norm_sq_std_tt = tt_std.norm_squared().unwrap();
    let norm_sq_ns_tt = tt_ns.norm_squared().unwrap();

    eprintln!("inner std={:?}", inner_std);
    eprintln!("inner ns ={:?}", inner_ns);

    // Standard ordering works
    assert!(
        inner_std.imag().abs() < 1e-10,
        "std: inner(x,x) should be real, got imag={:.6e}",
        inner_std.imag()
    );
    assert!(
        (inner_std.real() - norm_sq_std).abs() / norm_sq_std < 1e-10,
        "std: inner mismatch"
    );
    assert!(
        (norm_sq_std_tt - norm_sq_std).abs() / norm_sq_std < 1e-10,
        "std: norm_squared mismatch: TT={:.6e}, dense={:.6e}",
        norm_sq_std_tt,
        norm_sq_std
    );

    assert!(
        inner_ns.imag().abs() < 1e-10,
        "ns: inner(x,x) should be real, got imag={:.6e}",
        inner_ns.imag()
    );
    assert!(
        inner_ns.real() >= -1e-10,
        "ns: inner(x,x) should be non-negative, got real={:.6e}",
        inner_ns.real()
    );
    assert!(
        (inner_ns.real() - norm_sq_ns).abs() / norm_sq_ns < 1e-6,
        "ns: inner mismatch: TT={:.6e}, dense={:.6e}",
        inner_ns.real(),
        norm_sq_ns
    );
    assert!(
        (norm_sq_ns_tt - norm_sq_ns).abs() / norm_sq_ns < 1e-6,
        "ns: norm_squared mismatch: TT={:.6e}, dense={:.6e}",
        norm_sq_ns_tt,
        norm_sq_ns
    );
}

/// Same bug with 3 sites: mixed non-standard ordering
#[test]
fn test_inner_wrong_3site_nonstandard() {
    let s0 = DynIndex::new_dyn_with_tag(2, "s=1").unwrap();
    let s1 = DynIndex::new_dyn_with_tag(2, "s=2").unwrap();
    let s2 = DynIndex::new_dyn_with_tag(2, "s=3").unwrap();
    let b0 = DynIndex::new_dyn(2);
    let b1 = DynIndex::new_dyn(2);

    let data0 = vec![c(1.0, 0.5), c(-0.3, 1.2), c(0.7, -0.8), c(2.1, 0.3)];
    let data1 = vec![
        c(0.5, -0.1),
        c(1.0, 0.3),
        c(-0.4, 0.9),
        c(0.2, -0.7),
        c(1.3, 0.2),
        c(-0.6, 0.5),
        c(0.8, -1.1),
        c(-0.3, 0.4),
    ];
    let data2 = vec![c(-0.9, 0.6), c(0.4, 1.1), c(1.5, -0.2), c(-0.7, 0.8)];

    // Non-standard: [s0, b0], [s1, b0, b1], [s2, b1]
    let tt = TensorTrain::new(vec![
        TensorDynLen::from_dense(vec![s0.clone(), b0.clone()], data0).unwrap(),
        TensorDynLen::from_dense(vec![s1.clone(), b0.clone(), b1.clone()], data1).unwrap(),
        TensorDynLen::from_dense(vec![s2.clone(), b1.clone()], data2).unwrap(),
    ])
    .unwrap();

    let inner = tt.inner(&tt).unwrap();
    let norm_sq_tt = tt.norm_squared().unwrap();
    let dense = tt.to_dense().unwrap();
    let dense_norm_sq = dense.norm() * dense.norm();

    eprintln!("inner={:?}, dense_norm²={:.6e}", inner, dense_norm_sq);

    assert!(
        inner.imag().abs() < 1e-10,
        "inner(x,x) should be real, got imag={:.6e}",
        inner.imag()
    );
    assert!(
        inner.real() >= -1e-10,
        "inner(x,x) should be non-negative, got real={:.6e}",
        inner.real()
    );
    assert!(
        (inner.real() - dense_norm_sq).abs() / dense_norm_sq < 1e-6,
        "inner mismatch: TT={:.6e}, dense={:.6e}",
        inner.real(),
        dense_norm_sq
    );
    assert!(
        (norm_sq_tt - dense_norm_sq).abs() / dense_norm_sq < 1e-6,
        "norm_squared mismatch: TT={:.6e}, dense={:.6e}",
        norm_sq_tt,
        dense_norm_sq
    );
}

/// Same issue for MPO-like site tensors with two site indices per site.
#[test]
fn test_inner_wrong_with_two_site_indices_per_site_nonstandard_order() {
    let s0_in = DynIndex::new_dyn_with_tag(2, "s=1").unwrap();
    let s0_out = DynIndex::new_dyn_with_tag(2, "so=1").unwrap();
    let s1_in = DynIndex::new_dyn_with_tag(2, "s=2").unwrap();
    let s1_out = DynIndex::new_dyn_with_tag(2, "so=2").unwrap();
    let b = DynIndex::new_dyn(2);

    let data0 = vec![
        c(1.0, 0.1),
        c(-0.2, 0.3),
        c(0.4, -0.5),
        c(0.7, 0.2),
        c(-0.6, 0.8),
        c(0.9, -0.4),
        c(0.3, 0.6),
        c(-0.1, -0.7),
    ];
    let data1 = vec![
        c(-0.5, 0.2),
        c(0.8, -0.1),
        c(0.6, 0.7),
        c(-0.4, 0.9),
        c(0.2, -0.3),
        c(1.1, 0.5),
        c(-0.7, 0.4),
        c(0.3, -0.8),
    ];

    let tt = TensorTrain::new(vec![
        TensorDynLen::from_dense(vec![s0_in, s0_out, b.clone()], data0).unwrap(),
        TensorDynLen::from_dense(vec![s1_in, s1_out, b], data1).unwrap(),
    ])
    .unwrap();

    let inner = tt.inner(&tt).unwrap();
    let norm_sq_tt = tt.norm_squared().unwrap();
    let dense = tt.to_dense().unwrap();
    let dense_norm_sq = dense.norm() * dense.norm();

    assert!(
        inner.imag().abs() < 1e-10,
        "inner(x,x) should be real, got imag={:.6e}",
        inner.imag()
    );
    assert!(
        (inner.real() - dense_norm_sq).abs() / dense_norm_sq < 1e-10,
        "inner mismatch: TT={:.12e}, dense={:.12e}",
        inner.real(),
        dense_norm_sq
    );
    assert!(
        (norm_sq_tt - dense_norm_sq).abs() / dense_norm_sq < 1e-10,
        "norm_squared mismatch: TT={:.12e}, dense={:.12e}",
        norm_sq_tt,
        dense_norm_sq
    );
}
