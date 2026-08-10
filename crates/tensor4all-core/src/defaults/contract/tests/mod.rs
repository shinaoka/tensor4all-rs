use super::*;
use crate::defaults::{Index, TagSet};
use num_complex::Complex64;
use std::ffi::OsString;
use std::time::Duration;
use tensor4all_tensorbackend::StorageKind;

struct ScopedEnvVar {
    key: &'static str,
    previous: Option<OsString>,
}

impl ScopedEnvVar {
    fn set(key: &'static str, value: &str) -> Self {
        let previous = std::env::var_os(key);
        unsafe {
            std::env::set_var(key, value);
        }
        Self { key, previous }
    }
}

impl Drop for ScopedEnvVar {
    fn drop(&mut self) {
        match &self.previous {
            Some(value) => unsafe {
                std::env::set_var(self.key, value);
            },
            None => unsafe {
                std::env::remove_var(self.key);
            },
        }
    }
}

fn make_test_tensor(shape: &[usize], ids: &[u64]) -> TensorDynLen {
    let indices: Vec<DynIndex> = ids
        .iter()
        .zip(shape.iter())
        .map(|(&id, &dim)| Index::new(DynId(id), dim))
        .collect();
    let total_size: usize = shape.iter().product();
    let data: Vec<Complex64> = (0..total_size)
        .map(|i| Complex64::new(i as f64, 0.0))
        .collect();
    TensorDynLen::from_dense(indices, data).unwrap()
}

fn make_test_tensor_from_data(
    shape: &[usize],
    indices: Vec<DynIndex>,
    data: Vec<f64>,
) -> TensorDynLen {
    assert_eq!(shape.iter().product::<usize>(), data.len());
    assert_eq!(shape.len(), indices.len());
    TensorDynLen::from_dense(indices, data).unwrap()
}

fn col_major_offset(coords: &[usize], dims: &[usize]) -> usize {
    let mut stride = 1usize;
    let mut offset = 0usize;
    for (&coord, &dim) in coords.iter().zip(dims.iter()) {
        assert!(coord < dim);
        offset += coord * stride;
        stride *= dim;
    }
    offset
}

// ========================================================================
// contract tests
// ========================================================================

#[test]
fn test_contract_empty() {
    let tensors: Vec<&TensorDynLen> = vec![];
    let result = contract(&tensors);
    assert!(result.is_err());
}

#[test]
fn test_contract_single() {
    let tensor = make_test_tensor(&[2, 3], &[1, 2]);
    let result = contract(&[&tensor]).unwrap();
    assert_eq!(result.dims(), tensor.dims());
}

#[test]
fn test_contract_pair_nary_entry() {
    // A[i,j] * B[j,k] -> C[i,k]
    let a = make_test_tensor(&[2, 3], &[1, 2]); // i=1, j=2
    let b = make_test_tensor(&[3, 4], &[2, 3]); // j=2, k=3
    let result = contract(&[&a, &b]).unwrap();
    assert_eq!(result.dims(), vec![2, 4]); // i, k
}

#[test]
fn test_contract_default_entry_rejects_disconnected_inputs() {
    let i = DynIndex::new_dyn(2);
    let j = DynIndex::new_dyn(3);

    let a = TensorDynLen::from_dense(vec![i], vec![1.0_f64, 2.0]).unwrap();
    let b = TensorDynLen::from_dense(vec![j], vec![3.0_f64, 4.0, 5.0]).unwrap();

    let err = contract(&[&a, &b]).unwrap_err();
    assert!(err.to_string().contains("Disconnected"));
}

#[test]
fn test_contract_same_id_different_tags_is_disconnected() {
    let id = DynId(90);
    let site = Index::new_with_tags(id, 2, TagSet::from_str("site").unwrap());
    let link = Index::new_with_tags(id, 2, TagSet::from_str("link").unwrap());

    let a = TensorDynLen::from_dense(vec![site], vec![1.0_f64, 2.0]).unwrap();
    let b = TensorDynLen::from_dense(vec![link], vec![3.0_f64, 4.0]).unwrap();

    let err = contract(&[&a, &b]).unwrap_err();
    assert!(err.to_string().contains("Disconnected"));
}

#[test]
fn test_outer_product_is_explicit_disconnected_entry() {
    let i = DynIndex::new_dyn(2);
    let j = DynIndex::new_dyn(3);

    let a = TensorDynLen::from_dense(vec![i.clone()], vec![1.0_f64, 2.0]).unwrap();
    let b = TensorDynLen::from_dense(vec![j.clone()], vec![3.0_f64, 4.0, 5.0]).unwrap();

    let result = outer_product(&a, &b).unwrap();
    assert_eq!(result.indices(), &[i, j]);
    assert_eq!(
        result.to_vec::<f64>().unwrap(),
        vec![3.0, 6.0, 4.0, 8.0, 5.0, 10.0]
    );
}

#[test]
fn test_contract_owned_rejects_disconnected_inputs() {
    let i = DynIndex::new_dyn(2);
    let j = DynIndex::new_dyn(2);

    let a = TensorDynLen::from_dense(vec![i], vec![1.0_f64, 2.0]).unwrap();
    let b = TensorDynLen::from_dense(vec![j], vec![3.0_f64, 4.0]).unwrap();

    let err = contract_owned(vec![a, b]).unwrap_err();
    assert!(err.to_string().contains("disconnected"));
}

#[test]
fn test_contract_diag_diag_partial_preserves_diagonal_storage() {
    let i = Index::new(DynId(1), 3);
    let j = Index::new(DynId(2), 3);
    let k = Index::new(DynId(3), 3);

    let a = TensorDynLen::from_diag(vec![i.clone(), j.clone()], vec![1.0_f64, 2.0, 3.0]).unwrap();
    let b = TensorDynLen::from_diag(vec![j, k.clone()], vec![4.0_f64, 5.0, 6.0]).unwrap();

    let result = contract(&[&a, &b]).unwrap();

    assert_eq!(result.dims(), vec![3, 3]);
    assert!(result.is_diag());
    assert_eq!(
        result.storage().unwrap().storage_kind(),
        StorageKind::Diagonal
    );

    let expected = TensorDynLen::from_diag(vec![i, k], vec![4.0_f64, 10.0, 18.0]).unwrap();
    assert!(result.isapprox(&expected, 1e-12, 0.0).unwrap());
}

#[test]
fn test_contract_three() {
    // A[i,j] * B[j,k] * C[k,l] -> D[i,l]
    let a = make_test_tensor(&[2, 3], &[1, 2]); // i=1, j=2
    let b = make_test_tensor(&[3, 4], &[2, 3]); // j=2, k=3
    let c = make_test_tensor(&[4, 5], &[3, 4]); // k=3, l=4
    let result = contract(&[&a, &b, &c]).unwrap();
    let mut sorted_dims = result.dims();
    sorted_dims.sort();
    assert_eq!(sorted_dims, vec![2, 5]); // i=2, l=5
}

#[test]
fn test_contract_four() {
    // A[i,j] * B[j,k] * C[k,l] * D[l,m] -> E[i,m]
    let a = make_test_tensor(&[2, 3], &[1, 2]);
    let b = make_test_tensor(&[3, 4], &[2, 3]);
    let c = make_test_tensor(&[4, 5], &[3, 4]);
    let d = make_test_tensor(&[5, 6], &[4, 5]);
    let result = contract(&[&a, &b, &c, &d]).unwrap();
    let mut sorted_dims = result.dims();
    sorted_dims.sort();
    assert_eq!(sorted_dims, vec![2, 6]); // i=2, m=6
}

#[test]
fn test_outer_product_matrix_matrix() {
    // A[i,j] * B[k,l] (no common indices) -> outer product C[i,j,k,l]
    let a = make_test_tensor(&[2, 3], &[1, 2]);
    let b = make_test_tensor(&[4, 5], &[3, 4]);
    let result = outer_product(&a, &b).unwrap();
    let result_dims = result.dims();
    let total_elements: usize = result_dims.iter().product();
    assert_eq!(total_elements, 2 * 3 * 4 * 5);
    assert_eq!(result_dims.len(), 4);
}

#[test]
fn test_outer_product_vector_vector() {
    // A[i] * B[j] (no common indices) -> outer product C[i,j]
    let a = make_test_tensor(&[2], &[1]); // i=1
    let b = make_test_tensor(&[3], &[2]); // j=2
    let result = outer_product(&a, &b).unwrap();
    let result_dims = result.dims();
    let total_elements: usize = result_dims.iter().product();
    assert_eq!(total_elements, 2 * 3);
    assert_eq!(result.dims().len(), 2);
}

#[test]
fn test_contract_disconnected_error() {
    let a = make_test_tensor(&[2, 3], &[1, 2]);
    let b = make_test_tensor(&[4, 5], &[3, 4]);
    let result = contract(&[&a, &b]);
    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .to_string()
        .to_lowercase()
        .contains("disconnected"));
}

#[test]
fn test_contract_with_options_retains_shared_batch_index() {
    let batch = Index::new(DynId(10), 2);
    let i = Index::new(DynId(11), 2);
    let k = Index::new(DynId(12), 3);
    let j = Index::new(DynId(13), 2);

    let a = make_test_tensor_from_data(
        &[2, 2, 3],
        vec![batch.clone(), i.clone(), k.clone()],
        (1..=12).map(|x| x as f64).collect(),
    );
    let b = make_test_tensor_from_data(
        &[2, 3, 2],
        vec![batch.clone(), k.clone(), j.clone()],
        (1..=12).map(|x| 0.5 * x as f64).collect(),
    );

    let retain_indices = [batch.clone()];
    let options = ContractionOptions::new().with_retain_indices(&retain_indices);
    let result = contract_with_options(&[&a, &b], options).unwrap();

    assert_eq!(result.indices(), &[batch.clone(), i.clone(), j.clone()]);
    assert_eq!(result.dims(), vec![2, 2, 2]);

    let a_data = a.to_vec::<f64>().unwrap();
    let b_data = b.to_vec::<f64>().unwrap();
    let a_dims = a.dims();
    let b_dims = b.dims();
    let mut expected = Vec::new();
    for j_idx in 0..2 {
        for i_idx in 0..2 {
            for b_idx in 0..2 {
                let mut sum = 0.0;
                for k_idx in 0..3 {
                    let a_offset = col_major_offset(&[b_idx, i_idx, k_idx], &a_dims);
                    let b_offset = col_major_offset(&[b_idx, k_idx, j_idx], &b_dims);
                    sum += a_data[a_offset] * b_data[b_offset];
                }
                expected.push(sum);
            }
        }
    }

    assert_eq!(result.to_vec::<f64>().unwrap(), expected);
}

#[test]
fn test_tensor_contract_with_options_retains_shared_index() {
    let batch = Index::new(DynId(60), 2);
    let i = Index::new(DynId(61), 2);
    let k = Index::new(DynId(62), 3);
    let j = Index::new(DynId(63), 2);

    let a = make_test_tensor_from_data(
        &[2, 2, 3],
        vec![batch.clone(), i.clone(), k.clone()],
        vec![1.0; 12],
    );
    let b = make_test_tensor_from_data(
        &[2, 3, 2],
        vec![batch.clone(), k.clone(), j.clone()],
        vec![1.0; 12],
    );

    let result = contract_pair_with_options(
        &a,
        &b,
        ContractionOptions::new().with_retain_indices(std::slice::from_ref(&batch)),
    )
    .unwrap();

    assert_eq!(result.indices(), &[batch, i, j]);
    assert_eq!(result.dims(), vec![2, 2, 2]);
    assert_eq!(result.to_vec::<f64>().unwrap(), vec![3.0; 8]);
}

#[test]
fn test_contract_retains_exact_same_id_prime_index() {
    let batch = Index::new(DynId(64), 2);
    let batch_prime = batch.prime();
    let k = Index::new(DynId(65), 3);

    let a = make_test_tensor_from_data(
        &[2, 2, 3],
        vec![batch.clone(), batch_prime.clone(), k.clone()],
        vec![1.0; 12],
    );
    let b = make_test_tensor_from_data(
        &[2, 2, 3],
        vec![batch.clone(), batch_prime.clone(), k.clone()],
        vec![1.0; 12],
    );

    let retain_indices = [batch_prime.clone()];
    let options = ContractionOptions::new().with_retain_indices(&retain_indices);
    let result = contract_with_options(&[&a, &b], options).unwrap();

    assert_eq!(result.indices(), &[batch_prime]);
    assert_eq!(result.dims(), vec![2]);
    assert_eq!(result.to_vec::<f64>().unwrap(), vec![6.0, 6.0]);
}

#[test]
fn test_contract_with_options_supports_three_way_retained_label() {
    let batch = Index::new(DynId(20), 2);
    let i = Index::new(DynId(21), 2);
    let j = Index::new(DynId(22), 3);
    let k = Index::new(DynId(23), 2);

    let a = make_test_tensor_from_data(
        &[2, 2],
        vec![batch.clone(), i.clone()],
        vec![1.0, 2.0, 3.0, 4.0],
    );
    let b = make_test_tensor_from_data(
        &[2, 3],
        vec![batch.clone(), j.clone()],
        vec![5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    );
    let c = make_test_tensor_from_data(
        &[2, 2],
        vec![batch.clone(), k.clone()],
        vec![11.0, 12.0, 13.0, 14.0],
    );

    let retain_indices = [batch.clone()];
    let options = ContractionOptions::new().with_retain_indices(&retain_indices);
    let result = contract_with_options(&[&a, &b, &c], options).unwrap();

    assert_eq!(
        result.indices(),
        &[batch.clone(), i.clone(), j.clone(), k.clone()]
    );
    assert_eq!(result.dims(), vec![2, 2, 3, 2]);

    let a_data = a.to_vec::<f64>().unwrap();
    let b_data = b.to_vec::<f64>().unwrap();
    let c_data = c.to_vec::<f64>().unwrap();
    let a_dims = a.dims();
    let b_dims = b.dims();
    let c_dims = c.dims();
    let mut expected = Vec::new();
    for k_idx in 0..2 {
        for j_idx in 0..3 {
            for i_idx in 0..2 {
                for batch_idx in 0..2 {
                    let a_offset = col_major_offset(&[batch_idx, i_idx], &a_dims);
                    let b_offset = col_major_offset(&[batch_idx, j_idx], &b_dims);
                    let c_offset = col_major_offset(&[batch_idx, k_idx], &c_dims);
                    expected.push(a_data[a_offset] * b_data[b_offset] * c_data[c_offset]);
                }
            }
        }
    }

    assert_eq!(result.to_vec::<f64>().unwrap(), expected);
}

#[test]
fn test_contract_with_options_errors_for_missing_retained_index() {
    let i = Index::new(DynId(30), 2);
    let j = Index::new(DynId(31), 3);
    let missing = Index::new(DynId(32), 2);

    let a = make_test_tensor_from_data(&[2], vec![i.clone()], vec![1.0, 2.0]);
    let b = make_test_tensor_from_data(&[3], vec![j.clone()], vec![3.0, 4.0, 5.0]);

    let retain_indices = [missing];
    let options = ContractionOptions::new().with_retain_indices(&retain_indices);
    let result = contract_with_options(&[&a, &b], options);

    assert!(result.is_err());
}

#[test]
fn test_contract_with_options_retained_index_connects_components() {
    let batch = Index::new(DynId(40), 2);
    let i = Index::new(DynId(41), 2);
    let j = Index::new(DynId(42), 3);

    let a = make_test_tensor_from_data(
        &[2, 2],
        vec![batch.clone(), i.clone()],
        vec![1.0, 2.0, 3.0, 4.0],
    );
    let b = make_test_tensor_from_data(
        &[2, 3],
        vec![batch.clone(), j.clone()],
        vec![5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    );

    let retain_indices = [batch.clone()];
    let options = ContractionOptions::new().with_retain_indices(&retain_indices);
    let result = contract_with_options(&[&a, &b], options).unwrap();

    assert_eq!(result.indices(), &[batch.clone(), i.clone(), j.clone()]);
    assert_eq!(result.dims(), vec![2, 2, 3]);

    let a_data = a.to_vec::<f64>().unwrap();
    let b_data = b.to_vec::<f64>().unwrap();
    let a_dims = a.dims();
    let b_dims = b.dims();
    let mut expected = Vec::new();
    for j_idx in 0..3 {
        for i_idx in 0..2 {
            for batch_idx in 0..2 {
                let a_offset = col_major_offset(&[batch_idx, i_idx], &a_dims);
                let b_offset = col_major_offset(&[batch_idx, j_idx], &b_dims);
                expected.push(a_data[a_offset] * b_data[b_offset]);
            }
        }
    }

    assert_eq!(result.to_vec::<f64>().unwrap(), expected);
}

#[test]
fn test_contract_owned_with_options_matches_borrowed_with_options() {
    let batch = Index::new(DynId(70), 2);
    let i = Index::new(DynId(71), 2);
    let k = Index::new(DynId(72), 3);
    let j = Index::new(DynId(73), 2);

    let a = make_test_tensor_from_data(
        &[2, 2, 3],
        vec![batch.clone(), i.clone(), k.clone()],
        (1..=12).map(|x| x as f64).collect(),
    );
    let b = make_test_tensor_from_data(
        &[2, 3, 2],
        vec![batch.clone(), k.clone(), j.clone()],
        (1..=12).map(|x| (x as f64) * 0.5).collect(),
    );

    let retain_indices = [batch.clone()];
    let options = ContractionOptions::new().with_retain_indices(&retain_indices);
    let owned = contract_owned_with_options(vec![a.clone(), b.clone()], options).unwrap();
    let borrowed = contract_with_options(&[&a, &b], options).unwrap();

    assert_eq!(owned.indices(), borrowed.indices());
    assert_eq!(owned.dims(), borrowed.dims());
    assert_eq!(
        owned.to_vec::<f64>().unwrap(),
        borrowed.to_vec::<f64>().unwrap()
    );
}

#[test]
fn test_contract_owned_with_options_single_matches_borrowed() {
    let a = make_test_tensor_from_data(
        &[2, 3],
        vec![Index::new(DynId(74), 2), Index::new(DynId(75), 3)],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    );
    let options = ContractionOptions::new();

    let owned = contract_owned_with_options(vec![a.clone()], options).unwrap();
    let borrowed = contract_with_options(&[&a], options).unwrap();

    assert_eq!(owned.indices(), borrowed.indices());
    assert_eq!(
        owned.to_vec::<f64>().unwrap(),
        borrowed.to_vec::<f64>().unwrap()
    );
}

#[test]
fn test_contract_owned_with_options_falls_back_for_structured_storage() {
    let i = Index::new(DynId(76), 3);
    let j = Index::new(DynId(77), 3);
    let k = Index::new(DynId(78), 3);

    let a = TensorDynLen::from_diag(vec![i.clone(), j.clone()], vec![1.0_f64, 2.0, 3.0]).unwrap();
    let b = TensorDynLen::from_diag(vec![j, k.clone()], vec![4.0_f64, 5.0, 6.0]).unwrap();
    let options = ContractionOptions::new();

    let owned = contract_owned_with_options(vec![a.clone(), b.clone()], options).unwrap();
    let borrowed = contract_with_options(&[&a, &b], options).unwrap();

    assert_eq!(owned.indices(), borrowed.indices());
    assert_eq!(
        owned.storage().unwrap().storage_kind(),
        StorageKind::Diagonal
    );
    assert!(owned.isapprox(&borrowed, 1e-12, 0.0).unwrap());
    let expected = TensorDynLen::from_diag(vec![i, k], vec![4.0_f64, 10.0, 18.0]).unwrap();
    assert!(owned.isapprox(&expected, 1e-12, 0.0).unwrap());
}

#[test]
fn test_contract_owned_with_options_supports_three_way_retained_label() {
    let batch = Index::new(DynId(80), 2);
    let i = Index::new(DynId(81), 2);
    let j = Index::new(DynId(82), 3);
    let k = Index::new(DynId(83), 2);

    let a = make_test_tensor_from_data(
        &[2, 2],
        vec![batch.clone(), i.clone()],
        vec![1.0, 2.0, 3.0, 4.0],
    );
    let b = make_test_tensor_from_data(
        &[2, 3],
        vec![batch.clone(), j.clone()],
        vec![5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    );
    let c = make_test_tensor_from_data(
        &[2, 2],
        vec![batch.clone(), k.clone()],
        vec![11.0, 12.0, 13.0, 14.0],
    );

    let retain_indices = [batch.clone()];
    let options = ContractionOptions::new().with_retain_indices(&retain_indices);
    let owned =
        contract_owned_with_options(vec![a.clone(), b.clone(), c.clone()], options).unwrap();
    let borrowed = contract_with_options(&[&a, &b, &c], options).unwrap();

    assert_eq!(
        owned.indices(),
        &[batch.clone(), i.clone(), j.clone(), k.clone()]
    );
    assert_eq!(owned.dims(), vec![2, 2, 3, 2]);
    assert_eq!(owned.indices(), borrowed.indices());
    assert_eq!(
        owned.to_vec::<f64>().unwrap(),
        borrowed.to_vec::<f64>().unwrap()
    );
}

#[test]
fn test_contract_owned_with_options_falls_back_to_borrowed_for_grad_tensors() {
    let batch = Index::new(DynId(90), 2);
    let i = Index::new(DynId(91), 2);
    let k = Index::new(DynId(92), 3);
    let j = Index::new(DynId(93), 2);

    let x = make_test_tensor_from_data(
        &[2, 2, 3],
        vec![batch.clone(), i.clone(), k.clone()],
        (1..=12).map(|value| value as f64).collect(),
    )
    .enable_grad()
    .unwrap();
    let y = make_test_tensor_from_data(
        &[2, 3, 2],
        vec![batch.clone(), k.clone(), j.clone()],
        vec![1.0; 12],
    );

    let retain_indices = [batch.clone()];
    let options = ContractionOptions::new().with_retain_indices(&retain_indices);
    let owned = contract_owned_with_options(vec![x.clone(), y], options).unwrap();

    let ones = TensorDynLen::from_dense(owned.indices().to_vec(), vec![1.0; 8]).unwrap();
    let loss = contract(&[&owned, &ones]).unwrap();
    loss.backward().unwrap();

    let grad = x.grad().unwrap().unwrap();
    assert_eq!(grad.dims(), vec![2, 2, 3]);
    assert_eq!(grad.to_vec::<f64>().unwrap(), vec![2.0; 12]);
}

#[test]
fn test_find_tensor_connected_components_trivial_cases() {
    let empty: Vec<&TensorDynLen> = Vec::new();
    assert!(find_tensor_connected_components(&empty).is_empty());

    let a = make_test_tensor(&[2, 3], &[1, 2]);
    assert_eq!(find_tensor_connected_components(&[&a]), vec![vec![0]]);
}

#[test]
fn test_find_tensor_connected_components_multiple_components() {
    let a = make_test_tensor(&[2, 3], &[1, 2]);
    let b = make_test_tensor(&[3, 4], &[2, 3]);
    let c = make_test_tensor(&[5, 6], &[4, 5]);

    assert_eq!(
        find_tensor_connected_components(&[&a, &b, &c]),
        vec![vec![0, 1], vec![2]]
    );
}

#[test]
fn test_union_find_remap_and_tensor_id_helpers() {
    let i = DynId(1);
    let j = DynId(2);
    let k = DynId(3);

    let mut uf = AxisUnionFind::default();
    uf.union(i, j);

    assert_eq!(uf.remap(i), uf.remap(j));
    assert_ne!(uf.remap(i), uf.remap(k));
    assert_eq!(
        uf.remap_ids(&[i, j, k]),
        vec![uf.remap(i), uf.remap(j), uf.remap(k)]
    );

    let a = make_test_tensor(&[2, 3], &[1, 2]);
    let b = make_test_tensor(&[3, 4], &[2, 3]);
    let remapped = remap_tensor_ids(&[&a, &b], &mut uf);
    assert_eq!(remapped[0].len(), 2);
    assert_eq!(remapped[1].len(), 2);

    let output = vec![Index::new(DynId(1), 2), Index::new(DynId(3), 4)];
    let remapped_output = remap_output_ids(&output, &mut uf);
    assert_eq!(remapped_output.len(), 2);

    let sizes = collect_sizes(&[&a, &b], &mut uf);
    assert_eq!(sizes[&uf.remap(DynId(1))], 2);
    assert_eq!(sizes[&uf.remap(DynId(2))], 2);
    assert_eq!(sizes[&uf.remap(DynId(3))], 4);
}

#[test]
fn test_contract_profile_helpers() {
    let _profile_env = ScopedEnvVar::set("T4A_PROFILE_CONTRACT", "1");
    reset_contract_profile();

    let signature = ContractSignature {
        operands: vec![ContractOperandSignature {
            dims: vec![2, 3],
            ids: vec![1, 2],
            is_diag: false,
        }],
        output_ids: vec![1],
        output_dims: vec![2],
    };
    record_contract_profile(signature, Duration::from_micros(10));
    print_and_reset_contract_profile();
    reset_contract_profile();
}

// ========================================================================
// Connected contraction semantics
// ========================================================================

#[test]
fn test_contract_all_pairs() {
    // A[i,j], B[j,k], C[i,l] - tensors 0, 1, 2
    let a = make_test_tensor(&[2, 3], &[1, 2]); // i=1, j=2
    let b = make_test_tensor(&[3, 4], &[2, 3]); // j=2, k=3
    let c = make_test_tensor(&[2, 5], &[1, 4]); // i=1, l=4
    let result = contract(&[&a, &b, &c]).unwrap();
    let mut sorted_dims = result.dims();
    sorted_dims.sort();
    assert_eq!(sorted_dims, vec![4, 5]); // k=4, l=5
}

#[test]
fn test_contract_disconnected_component_error() {
    let a = make_test_tensor(&[2, 3], &[1, 2]); // i=1, j=2
    let b = make_test_tensor(&[3, 4], &[2, 3]); // j=2, k=3
    let c = make_test_tensor(&[6, 5], &[5, 4]); // m=5, l=4 (no common with B)
    let result = contract(&[&a, &b, &c]);
    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string().to_lowercase();
    assert!(
        err_msg.contains("disconnected") || err_msg.contains("no contractable indices"),
        "Expected error about disconnected tensors or missing contractable indices, got: {err_msg}"
    );
}

#[test]
fn test_contract_components_then_explicit_outer_product() {
    let a = make_test_tensor(&[2, 3], &[1, 2]); // i=1, j=2
    let b = make_test_tensor(&[3, 4], &[2, 3]); // j=2, k=3
    let c = make_test_tensor(&[4, 5], &[4, 5]); // m=4, n=5
    let d = make_test_tensor(&[5, 6], &[5, 6]); // n=5, p=6
    let left = contract(&[&a, &b]).unwrap();
    let right = contract(&[&c, &d]).unwrap();
    let result = outer_product(&left, &right).unwrap();
    assert_eq!(result.dims().len(), 4);
    let mut sorted_dims = result.dims();
    sorted_dims.sort();
    assert_eq!(sorted_dims, vec![2, 4, 4, 6]);
}

// ========================================================================
// Union-Find tests
// ========================================================================

#[test]
fn test_union_find_basic() {
    let mut uf = AxisUnionFind::new();

    let a = DynId(1);
    let b = DynId(2);
    let c = DynId(3);

    uf.make_set(a);
    uf.make_set(b);
    uf.make_set(c);

    assert_ne!(uf.find(a), uf.find(b));
    assert_ne!(uf.find(b), uf.find(c));

    uf.union(a, b);
    assert_eq!(uf.find(a), uf.find(b));
    assert_ne!(uf.find(a), uf.find(c));

    uf.union(b, c);
    assert_eq!(uf.find(a), uf.find(b));
    assert_eq!(uf.find(b), uf.find(c));
    assert_eq!(uf.find(a), uf.find(c));
}

#[test]
fn test_union_find_chain() {
    let mut uf = AxisUnionFind::new();

    let i = DynId(1);
    let j = DynId(2);
    let k = DynId(3);
    let l = DynId(4);

    uf.union(i, j);
    uf.union(j, k);
    uf.union(k, l);

    let rep = uf.find(i);
    assert_eq!(uf.find(j), rep);
    assert_eq!(uf.find(k), rep);
    assert_eq!(uf.find(l), rep);
}

#[test]
fn test_remap_ids() {
    let mut uf = AxisUnionFind::new();

    let i = DynId(1);
    let j = DynId(2);
    let k = DynId(3);

    uf.union(i, j);

    let ids = vec![i, j, k];
    let remapped = uf.remap_ids(&ids);

    assert_eq!(remapped[0], remapped[1]);
    assert_ne!(remapped[0], remapped[2]);
}

#[test]
fn test_three_diag_chain() {
    let mut uf = AxisUnionFind::new();

    let i = DynId(1);
    let j = DynId(2);
    let k = DynId(3);
    let l = DynId(4);

    uf.union(i, j);
    uf.union(j, k);
    uf.union(k, l);

    let rep = uf.find(i);
    assert_eq!(uf.find(j), rep);
    assert_eq!(uf.find(k), rep);
    assert_eq!(uf.find(l), rep);
}

#[test]
fn test_three_diag_star() {
    let mut uf = AxisUnionFind::new();

    let a = DynId(1);
    let b = DynId(2);
    let c = DynId(3);
    let d = DynId(4);

    uf.union(a, b);
    uf.union(a, c);
    uf.union(a, d);

    let rep = uf.find(a);
    assert_eq!(uf.find(b), rep);
    assert_eq!(uf.find(c), rep);
    assert_eq!(uf.find(d), rep);
}

#[test]
fn test_diag_with_three_axes() {
    let mut uf = AxisUnionFind::new();

    let i = DynId(1);
    let j = DynId(2);
    let k = DynId(3);
    let l = DynId(4);

    uf.union(i, j);
    uf.union(j, k);

    let rep = uf.find(i);
    assert_eq!(uf.find(j), rep);
    assert_eq!(uf.find(k), rep);
    assert_ne!(uf.find(l), rep);
}

#[test]
fn test_two_separate_diag_groups() {
    let mut uf = AxisUnionFind::new();

    let a = DynId(1);
    let b = DynId(2);
    let c = DynId(3);
    let d = DynId(4);

    uf.union(a, b);
    uf.union(c, d);

    assert_eq!(uf.find(a), uf.find(b));
    assert_eq!(uf.find(c), uf.find(d));
    assert_ne!(uf.find(a), uf.find(c));
}

#[test]
fn test_diag_and_dense_mixed() {
    let mut uf = AxisUnionFind::new();

    let i = DynId(1);
    let j = DynId(2);
    let k = DynId(3);

    uf.union(i, j);
    uf.make_set(k);

    assert_eq!(uf.find(i), uf.find(j));
    assert_ne!(uf.find(j), uf.find(k));
}

#[test]
fn test_complex_network() {
    let mut uf = AxisUnionFind::new();

    let a = DynId(1);
    let b = DynId(2);
    let c = DynId(3);
    let d = DynId(4);
    let e = DynId(5);
    let f = DynId(6);

    uf.union(a, b);
    uf.union(b, c);
    uf.make_set(d);
    uf.union(d, e);
    uf.union(e, f);

    let rep1 = uf.find(a);
    assert_eq!(uf.find(b), rep1);
    assert_eq!(uf.find(c), rep1);

    let rep2 = uf.find(d);
    assert_eq!(uf.find(e), rep2);
    assert_eq!(uf.find(f), rep2);

    assert_ne!(rep1, rep2);
}

#[test]
fn test_single_diag_tensor() {
    let mut uf = AxisUnionFind::new();

    let i = DynId(1);
    let j = DynId(2);

    uf.union(i, j);

    assert_eq!(uf.find(i), uf.find(j));
}

#[test]
fn test_empty_union_find() {
    let mut uf = AxisUnionFind::new();

    let x = DynId(42);
    uf.make_set(x);
    assert_eq!(uf.find(x), x);
}

#[test]
fn test_idempotent_union() {
    let mut uf = AxisUnionFind::new();

    let a = DynId(1);
    let b = DynId(2);

    uf.union(a, b);
    let rep1 = uf.find(a);

    uf.union(a, b);
    let rep2 = uf.find(a);

    uf.union(b, a);
    let rep3 = uf.find(a);

    assert_eq!(rep1, rep2);
    assert_eq!(rep2, rep3);
}

#[test]
fn test_self_union() {
    let mut uf = AxisUnionFind::new();

    let a = DynId(1);
    uf.union(a, a);

    assert_eq!(uf.find(a), a);
}

#[test]
fn test_four_diag_tensors_chain() {
    let mut uf = AxisUnionFind::new();

    let i = DynId(1);
    let j = DynId(2);
    let k = DynId(3);
    let l = DynId(4);
    let m = DynId(5);

    uf.union(i, j);
    uf.union(j, k);
    uf.union(k, l);
    uf.union(l, m);

    let rep = uf.find(i);
    assert_eq!(uf.find(j), rep);
    assert_eq!(uf.find(k), rep);
    assert_eq!(uf.find(l), rep);
    assert_eq!(uf.find(m), rep);
}

#[test]
fn test_diag_tensors_merge_two_chains() {
    let mut uf = AxisUnionFind::new();

    let a = DynId(1);
    let b = DynId(2);
    let c = DynId(3);
    let d = DynId(4);
    let e = DynId(5);

    uf.union(a, b);
    uf.union(b, c);
    uf.union(d, e);
    uf.union(e, c);

    let rep = uf.find(a);
    assert_eq!(uf.find(b), rep);
    assert_eq!(uf.find(c), rep);
    assert_eq!(uf.find(d), rep);
    assert_eq!(uf.find(e), rep);
}

#[test]
fn test_remap_preserves_order() {
    let mut uf = AxisUnionFind::new();

    let i = DynId(1);
    let j = DynId(2);
    let k = DynId(3);
    let l = DynId(4);

    uf.union(i, j);
    uf.union(k, l);

    let ids = vec![i, j, k, l, i, k];
    let remapped = uf.remap_ids(&ids);

    assert_eq!(remapped.len(), 6);
    assert_eq!(remapped[0], remapped[1]);
    assert_eq!(remapped[2], remapped[3]);
    assert_ne!(remapped[0], remapped[2]);
    assert_eq!(remapped[0], remapped[4]);
    assert_eq!(remapped[2], remapped[5]);
}

// ========================================================================
// dense contract tests
// ========================================================================

fn make_dense_tensor(shape: &[usize], ids: &[u64]) -> TensorDynLen {
    let indices: Vec<DynIndex> = ids
        .iter()
        .zip(shape.iter())
        .map(|(&id, &dim)| Index::new(DynId(id), dim))
        .collect();
    let total_size: usize = shape.iter().product();
    let data: Vec<Complex64> = (0..total_size)
        .map(|i| Complex64::new((i + 1) as f64, 0.0))
        .collect();
    TensorDynLen::from_dense(indices, data).unwrap()
}

#[test]
fn test_contract_empty_dense_entry() {
    let tensors: Vec<&TensorDynLen> = vec![];
    let result = contract(&tensors);
    assert!(result.is_err());
}

#[test]
fn test_contract_single_dense_entry() {
    let tensor = make_dense_tensor(&[2, 3], &[1, 2]);
    let result = contract(&[&tensor]).unwrap();
    assert_eq!(result.dims(), tensor.dims());
}

#[test]
fn test_contract_pair_dense() {
    // A[i,j] * B[j,k] -> C[i,k]
    let a = make_dense_tensor(&[2, 3], &[1, 2]); // i=1, j=2
    let b = make_dense_tensor(&[3, 4], &[2, 3]); // j=2, k=3
    let result = contract(&[&a, &b]).unwrap();
    assert_eq!(result.dims(), vec![2, 4]); // i, k
}

#[test]
fn test_contract_three_dense() {
    // A[i,j] * B[j,k] * C[k,l] -> D[i,l]
    let a = make_dense_tensor(&[2, 3], &[1, 2]); // i=1, j=2
    let b = make_dense_tensor(&[3, 4], &[2, 3]); // j=2, k=3
    let c = make_dense_tensor(&[4, 5], &[3, 4]); // k=3, l=4
    let result = contract(&[&a, &b, &c]).unwrap();
    let mut sorted_dims = result.dims();
    sorted_dims.sort();
    assert_eq!(sorted_dims, vec![2, 5]); // i=2, l=5
}
