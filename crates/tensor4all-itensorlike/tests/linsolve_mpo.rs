//! Regression test for #351: itensorlike linsolve with MPO that has
//! distinct input/output site indices.
//!
//! Previously, `linsolve` failed with index mismatch errors because
//! it did not pass IndexMapping to the treetn solver.

use num_complex::Complex64;
use tensor4all_core::{AnyScalar, DynIndex, IdxTensor};
use tensor4all_itensorlike::{LinsolveOptions, TensorTrain};

/// Build a 2-site identity MPO where input indices are the SAME objects
/// as the MPS site indices, and output indices are distinct (new_dyn).
///
/// Convention: MPO input index = s (shared with MPS), output index = s_out (distinct).
#[test]
fn test_linsolve_identity_mpo_distinct_output_indices() {
    let phys_dim = 2;

    // Physical indices for the MPS (shared with MPO input)
    let s0 = DynIndex::new_dyn(phys_dim);
    let s1 = DynIndex::new_dyn(phys_dim);

    // MPO output indices (distinct IDs from s0/s1)
    let s0_out = DynIndex::new_dyn(phys_dim);
    let s1_out = DynIndex::new_dyn(phys_dim);

    // Bond indices
    let b_mps = DynIndex::new_dyn(phys_dim);
    let b_mpo = DynIndex::new_dyn(1);

    // Build MPS for RHS: b = [1, 2, 3, 4]
    // Site 0: [s0, b_mps] = identity-like
    let mut data0 = vec![0.0; phys_dim * phys_dim];
    for i in 0..phys_dim {
        data0[i * phys_dim + i] = 1.0;
    }
    let t0_mps = IdxTensor::from_dense(vec![s0.clone(), b_mps.clone()], data0).unwrap();

    // Site 1: [b_mps, s1] = values
    let t1_mps =
        IdxTensor::from_dense(vec![b_mps.clone(), s1.clone()], vec![1.0, 2.0, 3.0, 4.0]).unwrap();

    let rhs = TensorTrain::new(vec![t0_mps.clone(), t1_mps.clone()]).unwrap();
    let init = TensorTrain::new(vec![t0_mps, t1_mps]).unwrap();

    // Build identity MPO:
    // Input indices (s0, s1) are the SAME as MPS site indices.
    // Output indices (s0_out, s1_out) are new distinct indices.
    // Site 0: [s0_out, s0, b_mpo] - identity
    let mut id_data = vec![0.0; phys_dim * phys_dim];
    for i in 0..phys_dim {
        id_data[i * phys_dim + i] = 1.0;
    }
    let t0_mpo = IdxTensor::from_dense(
        vec![s0_out.clone(), s0.clone(), b_mpo.clone()],
        id_data.clone(),
    )
    .unwrap();

    // Site 1: [b_mpo, s1_out, s1] - identity
    let t1_mpo =
        IdxTensor::from_dense(vec![b_mpo.clone(), s1_out.clone(), s1.clone()], id_data).unwrap();

    let operator = TensorTrain::new(vec![t0_mpo, t1_mpo]).unwrap();

    // This previously failed with "Index count mismatch" or "index structure mismatch"
    let options = LinsolveOptions::new(3)
        .with_gmres_tol(1e-10)
        .with_gmres_restart_dim(10)
        .with_max_bond_dim(4);

    let result = operator.linsolve(&rhs, init, &options).unwrap();

    // For identity operator, solution should match RHS
    assert_eq!(result.len(), 2);
}

#[test]
fn test_linsolve_identity_mpo_accepts_complex_coefficients() {
    let phys_dim = 2;

    let s0 = DynIndex::new_dyn(phys_dim);
    let s1 = DynIndex::new_dyn(phys_dim);
    let s0_out = DynIndex::new_dyn(phys_dim);
    let s1_out = DynIndex::new_dyn(phys_dim);
    let b_mps = DynIndex::new_dyn(phys_dim);
    let b_mpo = DynIndex::new_dyn(1);

    let mut data0 = vec![Complex64::new(0.0, 0.0); phys_dim * phys_dim];
    for i in 0..phys_dim {
        data0[i * phys_dim + i] = Complex64::new(1.0, 0.0);
    }
    let t0_mps = IdxTensor::from_dense(vec![s0.clone(), b_mps.clone()], data0).unwrap();

    let rhs_values = vec![
        Complex64::new(1.0, 0.5),
        Complex64::new(2.0, -1.0),
        Complex64::new(-0.25, 0.75),
        Complex64::new(0.5, 1.5),
    ];
    let t1_mps =
        IdxTensor::from_dense(vec![b_mps.clone(), s1.clone()], rhs_values.clone()).unwrap();
    let rhs = TensorTrain::new(vec![t0_mps.clone(), t1_mps]).unwrap();

    let zero0 = t0_mps.scale(AnyScalar::new_real(0.0)).unwrap();
    let zero1 = IdxTensor::from_dense(
        vec![b_mps.clone(), s1.clone()],
        vec![Complex64::new(0.0, 0.0); phys_dim * phys_dim],
    )
    .unwrap();
    let init = TensorTrain::new(vec![zero0, zero1]).unwrap();

    let mut id_data = vec![Complex64::new(0.0, 0.0); phys_dim * phys_dim];
    for i in 0..phys_dim {
        id_data[i * phys_dim + i] = Complex64::new(1.0, 0.0);
    }
    let t0_mpo = IdxTensor::from_dense(
        vec![s0_out.clone(), s0.clone(), b_mpo.clone()],
        id_data.clone(),
    )
    .unwrap();
    let t1_mpo = IdxTensor::from_dense(vec![b_mpo, s1_out, s1.clone()], id_data).unwrap();
    let operator = TensorTrain::new(vec![t0_mpo, t1_mpo]).unwrap();

    let options = LinsolveOptions::new(3)
        .with_gmres_tol(1e-10)
        .with_gmres_restart_dim(10)
        .with_max_bond_dim(4)
        .with_coefficients(0.0, Complex64::new(0.0, 1.0));

    let result = operator.linsolve(&rhs, init, &options).unwrap();
    let dense = result.to_dense().unwrap().to_vec::<Complex64>().unwrap();
    let expected = rhs_values
        .iter()
        .map(|value| Complex64::new(0.0, -1.0) * value)
        .collect::<Vec<_>>();

    for (actual, expected) in dense.iter().zip(expected.iter()) {
        assert!(
            (*actual - *expected).norm() < 1e-8,
            "actual={actual:?}, expected={expected:?}"
        );
    }
}
