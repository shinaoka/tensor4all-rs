//! Test utilities for partitioned tensor train tests.
//!
//! This module provides common test helper functions to reduce duplication
//! across test modules.

#![allow(dead_code)]

use num_complex::Complex64;
use tensor4all_core::index::Index;
use tensor4all_core::{DynIndex, IdxTensor, TensorElement};
use tensor4all_itensorlike::TensorTrain;

/// Trait for types that can be used as test scalars (f64 or Complex64).
pub trait TestScalar: TensorElement + From<f64> + Clone {
    /// Create test data for a tensor with the given size.
    fn make_test_data(size: usize) -> Vec<Self>;
}

impl TestScalar for f64 {
    fn make_test_data(size: usize) -> Vec<Self> {
        (0..size).map(|i| (i + 1) as f64).collect()
    }
}

impl TestScalar for Complex64 {
    fn make_test_data(size: usize) -> Vec<Self> {
        (0..size)
            .map(|i| Complex64::new((i + 1) as f64, (i as f64) * 0.1))
            .collect()
    }
}

/// Create a new dynamic index with the given size.
pub fn make_index(size: usize) -> DynIndex {
    Index::new_dyn(size)
}

/// Create a tensor with incrementing f64 values for the given indices.
pub fn make_tensor(indices: Vec<DynIndex>) -> IdxTensor {
    make_tensor_generic::<f64>(indices)
}

/// Create a tensor with test data for the given scalar type.
pub fn make_tensor_generic<T: TestScalar>(indices: Vec<DynIndex>) -> IdxTensor {
    let dims: Vec<usize> = indices.iter().map(|i| i.dim).collect();
    let size: usize = dims.iter().product();
    let data = T::make_test_data(size);
    IdxTensor::from_dense(indices, data).unwrap()
}

/// Create shared indices for a 2-site tensor train.
///
/// Returns (site_indices, link_index) where:
/// - site_indices[0] has dimension 2
/// - site_indices[1] has dimension 2
/// - link_index has dimension 3
pub fn make_shared_indices() -> (Vec<DynIndex>, DynIndex) {
    let s0 = make_index(2);
    let l01 = make_index(3);
    let s1 = make_index(2);
    (vec![s0, s1], l01)
}

/// Create a 2-site tensor train using the provided indices.
pub fn make_tt_with_indices(site_inds: &[DynIndex], link_ind: &DynIndex) -> TensorTrain {
    make_tt_with_indices_generic::<f64>(site_inds, link_ind)
}

/// Create a 2-site tensor train using the provided indices and scalar type.
pub fn make_tt_with_indices_generic<T: TestScalar>(
    site_inds: &[DynIndex],
    link_ind: &DynIndex,
) -> TensorTrain {
    let t0 = make_tensor_generic::<T>(vec![site_inds[0].clone(), link_ind.clone()]);
    let t1 = make_tensor_generic::<T>(vec![link_ind.clone(), site_inds[1].clone()]);
    TensorTrain::new(vec![t0, t1]).unwrap()
}

/// Create a simple 2-site tensor train with default indices.
///
/// Returns (tensor_train, site_indices).
pub fn make_simple_tt() -> (TensorTrain, Vec<DynIndex>) {
    let (site_inds, link_ind) = make_shared_indices();
    let tt = make_tt_with_indices(&site_inds, &link_ind);
    (tt, site_inds)
}

/// Create indices for contraction tests.
///
/// Returns (s0, l01, s1, l12, s2) for two TTs:
/// - TT1: s0 -- l01 -- s1
/// - TT2: s1 -- l12 -- s2
pub fn make_contraction_indices() -> (DynIndex, DynIndex, DynIndex, DynIndex, DynIndex) {
    let s0 = make_index(2);
    let l01 = make_index(3);
    let s1 = make_index(2);
    let l12 = make_index(3);
    let s2 = make_index(2);
    (s0, l01, s1, l12, s2)
}

/// Create the first tensor train for contraction tests.
pub fn make_tt1<T: TestScalar>(s0: &DynIndex, l01: &DynIndex, s1: &DynIndex) -> TensorTrain {
    let t0 = make_tensor_generic::<T>(vec![s0.clone(), l01.clone()]);
    let t1 = make_tensor_generic::<T>(vec![l01.clone(), s1.clone()]);
    TensorTrain::new(vec![t0, t1]).unwrap()
}

/// Create the second tensor train for contraction tests.
pub fn make_tt2<T: TestScalar>(s1: &DynIndex, l12: &DynIndex, s2: &DynIndex) -> TensorTrain {
    let t0 = make_tensor_generic::<T>(vec![s1.clone(), l12.clone()]);
    let t1 = make_tensor_generic::<T>(vec![l12.clone(), s2.clone()]);
    TensorTrain::new(vec![t0, t1]).unwrap()
}
