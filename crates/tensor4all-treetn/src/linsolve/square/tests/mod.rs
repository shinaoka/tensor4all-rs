use super::*;
use tensor4all_core::{DynIndex, IdxTensor};

fn create_simple_2site_mps() -> TreeTN<IdxTensor, String> {
    let mut mps = TreeTN::<IdxTensor, String>::new();
    let s0 = DynIndex::new_dyn(2);
    let s1 = DynIndex::new_dyn(2);
    let b01 = DynIndex::new_dyn(2);

    let t0 = IdxTensor::from_dense(vec![s0.clone(), b01.clone()], vec![1.0; 4]).unwrap();
    let t1 = IdxTensor::from_dense(vec![b01.clone(), s1.clone()], vec![1.0; 4]).unwrap();

    let n0 = mps.add_tensor("site0".to_string(), t0).unwrap();
    let n1 = mps.add_tensor("site1".to_string(), t1).unwrap();
    mps.connect(n0, &b01, n1, &b01).unwrap();

    mps
}

fn create_simple_2site_mpo() -> TreeTN<IdxTensor, String> {
    let mut mpo = TreeTN::<IdxTensor, String>::new();
    let s0_in = DynIndex::new_dyn(2);
    let s0_out = DynIndex::new_dyn(2);
    let s1_in = DynIndex::new_dyn(2);
    let s1_out = DynIndex::new_dyn(2);
    let b_mpo = DynIndex::new_dyn(1);

    let id_data = vec![1.0, 0.0, 0.0, 1.0]; // Identity matrix
    let t0_mpo = IdxTensor::from_dense(
        vec![s0_out.clone(), s0_in.clone(), b_mpo.clone()],
        id_data.clone(),
    )
    .unwrap();
    let t1_mpo =
        IdxTensor::from_dense(vec![b_mpo.clone(), s1_out.clone(), s1_in.clone()], id_data).unwrap();

    let n0_mpo = mpo.add_tensor("site0".to_string(), t0_mpo).unwrap();
    let n1_mpo = mpo.add_tensor("site1".to_string(), t1_mpo).unwrap();
    mpo.connect(n0_mpo, &b_mpo, n1_mpo, &b_mpo).unwrap();

    mpo
}

#[test]
fn test_validate_linsolve_inputs_success() {
    let operator = create_simple_2site_mpo();
    let rhs = create_simple_2site_mps();
    let init = create_simple_2site_mps();

    // Should succeed for compatible inputs
    let result = validate_linsolve_inputs(&operator, &rhs, &init);
    assert!(result.is_ok());
}

#[test]
fn test_validate_linsolve_inputs_incompatible_dimensions() {
    let operator = create_simple_2site_mpo();
    let rhs = create_simple_2site_mps();

    // Create init with different dimensions
    let mut init = TreeTN::<IdxTensor, String>::new();
    let s0 = DynIndex::new_dyn(3); // Different dimension
    let s1 = DynIndex::new_dyn(3);
    let b01 = DynIndex::new_dyn(2);

    let t0 = IdxTensor::from_dense(vec![s0.clone(), b01.clone()], vec![1.0; 6]).unwrap();
    let t1 = IdxTensor::from_dense(vec![b01.clone(), s1.clone()], vec![1.0; 6]).unwrap();

    let n0 = init.add_tensor("site0".to_string(), t0).unwrap();
    let n1 = init.add_tensor("site1".to_string(), t1).unwrap();
    init.connect(n0, &b01, n1, &b01).unwrap();

    // Should fail due to incompatible dimensions
    let result = validate_linsolve_inputs(&operator, &rhs, &init);
    assert!(result.is_err());
}

#[test]
fn test_square_linsolve_zero_sweeps_returns_solution_wrapper() {
    let operator = create_simple_2site_mpo();
    let rhs = create_simple_2site_mps();
    let init = create_simple_2site_mps();

    let result = square_linsolve(
        &operator,
        &rhs,
        init,
        &"site0".to_string(),
        LinsolveOptions::new(0),
        None,
        None,
    )
    .unwrap();

    assert_eq!(result.solution.node_count(), 2);
    assert_eq!(result.sweeps, 0);
    assert!(result.residual.is_some_and(|residual| residual < 1.0e-12));
    assert!(!result.converged);
}

#[test]
fn test_square_linsolve_can_skip_final_residual() {
    let operator = create_simple_2site_mpo();
    let rhs = create_simple_2site_mps();
    let init = create_simple_2site_mps();

    let result = square_linsolve(
        &operator,
        &rhs,
        init,
        &"site0".to_string(),
        LinsolveOptions::new(0).with_residual_check(false),
        None,
        None,
    )
    .unwrap();

    assert_eq!(result.solution.node_count(), 2);
    assert_eq!(result.sweeps, 0);
    assert!(result.residual.is_none());
    assert!(!result.converged);
}

#[test]
fn test_square_linsolve_validation_error_bubbles_up() {
    let operator = create_simple_2site_mpo();
    let rhs = create_simple_2site_mps();

    let mut init = TreeTN::<IdxTensor, String>::new();
    let s0 = DynIndex::new_dyn(3);
    let s1 = DynIndex::new_dyn(3);
    let b01 = DynIndex::new_dyn(2);

    let t0 = IdxTensor::from_dense(vec![s0.clone(), b01.clone()], vec![1.0; 6]).unwrap();
    let t1 = IdxTensor::from_dense(vec![b01.clone(), s1.clone()], vec![1.0; 6]).unwrap();

    let n0 = init.add_tensor("site0".to_string(), t0).unwrap();
    let n1 = init.add_tensor("site1".to_string(), t1).unwrap();
    init.connect(n0, &b01, n1, &b01).unwrap();

    let err = square_linsolve(
        &operator,
        &rhs,
        init,
        &"site0".to_string(),
        LinsolveOptions::new(0),
        None,
        None,
    )
    .unwrap_err()
    .to_string();

    assert!(
        err.contains("Operator cannot act on init")
            || err.contains("Result of operator action is not compatible with RHS"),
        "unexpected error: {err}"
    );
}

// Note: square_linsolve requires MPO and MPS to have compatible index structures.
// The MPO should have input indices matching the MPS site indices.
// These tests are simplified to only test validation, as full linsolve requires
// proper index mappings which are tested in integration tests.
