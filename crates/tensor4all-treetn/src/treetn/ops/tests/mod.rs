use num_complex::Complex64;
use tensor4all_core::{AnyScalar, ColMajorArrayRef, DynIndex, TensorDynLen};

use crate::{TreeTN, TreeTNEvaluator};

fn make_three_node_chain() -> TreeTN<TensorDynLen, usize> {
    let s0 = DynIndex::new_dyn(2);
    let s1 = DynIndex::new_dyn(2);
    let s2 = DynIndex::new_dyn(2);
    let bond01 = DynIndex::new_dyn(2);
    let bond12 = DynIndex::new_dyn(2);

    let t0 =
        TensorDynLen::from_dense(vec![s0, bond01.clone()], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
    let t1 = TensorDynLen::from_dense(
        vec![bond01, s1, bond12.clone()],
        vec![1.0_f64, -1.0, 0.5, 2.0, 1.5, 0.25, -0.75, 3.0],
    )
    .unwrap();
    let t2 = TensorDynLen::from_dense(vec![bond12, s2], vec![2.0_f64, 0.0, 1.0, 3.0]).unwrap();

    TreeTN::from_tensors(vec![t0, t1, t2], vec![0, 1, 2]).unwrap()
}

#[test]
fn test_inner_matches_norm_squared_for_three_node_chain() {
    let tn = make_three_node_chain();
    let inner = tn.inner(&tn).unwrap();

    let mut tn_for_norm = tn.clone();
    let norm_squared = tn_for_norm.norm_squared().unwrap();

    assert!((inner.real() - norm_squared).abs() < 1.0e-10);
    assert!(inner.imag().abs() < 1.0e-10);
}

#[test]
fn test_scale_doubles_norm() {
    let mut tn = make_three_node_chain();
    let original_norm = tn.norm().unwrap();

    tn.scale_mut(AnyScalar::new_real(2.0)).unwrap();
    let scaled_norm = tn.norm().unwrap();

    assert!((scaled_norm - 2.0 * original_norm).abs() < 1.0e-10);
}

#[test]
fn test_inner_preserves_complex_phase() {
    let site = DynIndex::new_dyn(2);
    let left = TensorDynLen::from_dense(
        vec![site.clone()],
        vec![Complex64::new(1.0, 1.0), Complex64::new(2.0, -1.0)],
    )
    .unwrap();
    let right = TensorDynLen::from_dense(
        vec![site],
        vec![Complex64::new(3.0, -2.0), Complex64::new(-1.0, 4.0)],
    )
    .unwrap();

    let left_tn = TreeTN::from_tensors(vec![left], vec![0usize]).unwrap();
    let right_tn = TreeTN::from_tensors(vec![right], vec![0usize]).unwrap();

    let inner = left_tn.inner(&right_tn).unwrap();

    assert!((inner.real() + 5.0).abs() < 1.0e-10);
    assert!((inner.imag() - 2.0).abs() < 1.0e-10);
}

#[test]
fn test_evaluate_accepts_full_indices_for_same_id_prime_pair() {
    let input = DynIndex::new_dyn(2);
    let output = input.prime();
    let tensor = TensorDynLen::from_dense(
        vec![input.clone(), output.clone()],
        vec![10.0, 20.0, 30.0, 40.0],
    )
    .unwrap();
    let tn = TreeTN::from_tensors(vec![tensor], vec![0usize]).unwrap();

    let indices = vec![input, output];
    let values = [0usize, 0usize, 1usize, 1usize];
    let shape = [2usize, 2usize];
    let result = tn
        .evaluate(&indices, ColMajorArrayRef::new(&values, &shape).unwrap())
        .unwrap();

    assert_eq!(result.len(), 2);
    assert!((result[0].real() - 10.0).abs() < 1.0e-10);
    assert!((result[1].real() - 40.0).abs() < 1.0e-10);
}

#[test]
fn evaluator_evaluate_batched_matches_evaluate() {
    let tn = make_three_node_chain();
    let (indices, _) = tn.all_site_indices().unwrap();
    let values = [0usize, 0, 0, 1, 1, 1, 0, 1, 0];
    let shape = [indices.len(), 3usize];
    let values_ref = ColMajorArrayRef::new(&values, &shape).unwrap();

    let evaluator = TreeTNEvaluator::new(&tn, &indices).unwrap();
    let direct = evaluator.evaluate_batched(values_ref).unwrap();
    let sugar = tn.evaluate(&indices, values_ref).unwrap();

    assert_eq!(direct.len(), 3);
    assert_eq!(sugar.len(), 3);
    for (left, right) in direct.iter().zip(sugar.iter()) {
        assert!((left.real() - right.real()).abs() < 1.0e-12);
        assert!((left.imag() - right.imag()).abs() < 1.0e-12);
    }
}

#[test]
fn evaluate_point_matches_single_column_batch() {
    let tn = make_three_node_chain();
    let (indices, _) = tn.all_site_indices().unwrap();
    let values = [1usize, 0, 1];
    let shape = [indices.len(), 1usize];

    let point = tn.evaluate_point(&indices, &values).unwrap();
    let batch = tn
        .evaluate(&indices, ColMajorArrayRef::new(&values, &shape).unwrap())
        .unwrap();

    assert_eq!(batch.len(), 1);
    assert!((point.real() - batch[0].real()).abs() < 1.0e-12);
    assert!((point.imag() - batch[0].imag()).abs() < 1.0e-12);
}

#[test]
fn evaluate_point_rejects_value_count_mismatch() {
    let tn = make_three_node_chain();
    let (indices, _) = tn.all_site_indices().unwrap();

    let err = tn.evaluate_point(&indices, &[0usize, 1]).unwrap_err();

    assert!(err.to_string().contains("values.len()"));
}

#[test]
fn link_dims_reports_bond_dimensions_in_node_order() {
    let s0 = DynIndex::new_dyn(2);
    let s1 = DynIndex::new_dyn(2);
    let s2 = DynIndex::new_dyn(2);
    let bond01 = DynIndex::new_dyn(2);
    let bond12 = DynIndex::new_dyn(3);

    let t0 = TensorDynLen::from_dense(vec![s0, bond01.clone()], vec![1.0_f64; 2 * 2]).unwrap();
    let t1 = TensorDynLen::from_dense(vec![bond01, s1, bond12.clone()], vec![1.0_f64; 2 * 2 * 3])
        .unwrap();
    let t2 = TensorDynLen::from_dense(vec![bond12, s2], vec![1.0_f64; 3 * 2]).unwrap();
    let tn = TreeTN::from_tensors(vec![t0, t1, t2], vec![0usize, 1, 2]).unwrap();

    assert_eq!(tn.link_dims(), vec![2, 3]);
}

#[test]
fn evaluator_rejects_incomplete_site_index_list() {
    let tn = make_three_node_chain();
    let (indices, _) = tn.all_site_indices().unwrap();

    let err = TreeTNEvaluator::new(&tn, &indices[..indices.len() - 1]).unwrap_err();

    assert!(err.to_string().contains("indices.len()"));
}
