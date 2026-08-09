use num_complex::Complex64;
use tensor4all_core::index::DefaultIndex as Index;
use tensor4all_core::index_ops::common_inds;
use tensor4all_core::{
    contract_pair, contract_pair_with_operand_options, tensordot, DynIndex,
    PairwiseContractionOptions, TensorContractionLike, TensorDynLen,
};
use tensor4all_tensorbackend::{Storage, StorageKind};

fn dense_f64(indices: Vec<DynIndex>, data: Vec<f64>) -> TensorDynLen {
    TensorDynLen::from_dense(indices, data).unwrap()
}

fn dense_c64(indices: Vec<DynIndex>, data: Vec<Complex64>) -> TensorDynLen {
    TensorDynLen::from_dense(indices, data).unwrap()
}

fn assert_all_f64(tensor: &TensorDynLen, expected_len: usize, expected_value: f64) {
    let data = tensor.to_vec::<f64>().unwrap();
    assert_eq!(data.len(), expected_len);
    for value in data {
        assert_eq!(value, expected_value);
    }
}

#[test]
fn test_common_inds() {
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(3);
    let k = Index::new_dyn(4);

    let indices_a = vec![i.clone(), j.clone()];
    let indices_b = vec![j.clone(), k.clone()];

    let common = common_inds(&indices_a, &indices_b);
    assert_eq!(common.len(), 1);
    assert_eq!(common[0].id, j.id);
}

#[test]
fn test_contract_dyn_len_matrix_multiplication() {
    // Create two matrices: A[i, j] and B[j, k]
    // Result should be C[i, k] = A[i, j] * B[j, k]
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(3);
    let k = Index::new_dyn(4);

    // Create tensor A[i, j] with all ones
    let tensor_a = dense_f64(vec![i.clone(), j.clone()], vec![1.0; 6]);

    // Create tensor B[j, k] with all ones
    let tensor_b = dense_f64(vec![j.clone(), k.clone()], vec![1.0; 12]);

    // Contract along j: result should be C[i, k] with all 3.0 (since each element is sum of 3 ones)
    let result = tensor_a.contract_pair(&tensor_b).unwrap();
    assert_eq!(result.dims(), vec![2, 4]);
    assert_eq!(result.indices.len(), 2);
    assert_eq!(result.indices[0].id, i.id);
    assert_eq!(result.indices[1].id, k.id);

    assert_all_f64(&result, 8, 3.0);
}

#[test]
fn test_mul_operator_contraction() {
    // Test that the * operator performs tensor contraction
    // Create two matrices: A[i, j] and B[j, k]
    // Result should be C[i, k] = A[i, j] * B[j, k]
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(3);
    let k = Index::new_dyn(4);

    // Create tensor A[i, j] with all ones
    let tensor_a = dense_f64(vec![i.clone(), j.clone()], vec![1.0; 6]);

    // Create tensor B[j, k] with all ones
    let tensor_b = dense_f64(vec![j.clone(), k.clone()], vec![1.0; 12]);

    // Contract along j using * operator: result should be C[i, k] with all 3.0
    let result = tensor_a.contract_pair(&tensor_b).unwrap();
    assert_eq!(result.dims(), vec![2, 4]);
    assert_eq!(result.indices.len(), 2);
    assert_eq!(result.indices[0].id, i.id);
    assert_eq!(result.indices[1].id, k.id);

    assert_all_f64(&result, 8, 3.0);
}

#[test]
fn test_mul_operator_owned() {
    // Test * operator with owned tensors
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(3);
    let k = Index::new_dyn(4);

    let tensor_a = dense_f64(vec![i.clone(), j.clone()], vec![1.0; 6]);

    let tensor_b = dense_f64(vec![j.clone(), k.clone()], vec![1.0; 12]);

    // Use * operator with owned tensors
    let result = tensor_a.contract_pair(&tensor_b).unwrap();
    assert_eq!(result.dims(), vec![2, 4]);
    assert_eq!(result.indices.len(), 2);
}

#[test]
fn test_contract_no_common_indices_gives_outer_product() {
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(3);
    let k = Index::new_dyn(4);

    let tensor_a = TensorDynLen::zeros::<f64>(vec![i.clone(), j.clone()]).unwrap();

    let tensor_b = TensorDynLen::zeros::<f64>(vec![k.clone()]).unwrap();

    // No common indices → outer product
    let result = tensor_a.contract_pair(&tensor_b).unwrap();
    assert_eq!(result.dims(), vec![2, 3, 4]);
    assert_eq!(result.indices.len(), 3);
}

#[test]
fn test_contract_no_common_indices_preserves_left_then_right_index_order_and_values() {
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(3);

    let tensor_a = TensorDynLen::from_dense(vec![i.clone()], vec![2.0, -1.0]).unwrap();
    let tensor_b = TensorDynLen::from_dense(vec![j.clone()], vec![3.0, 4.0, -2.0]).unwrap();

    let result = tensor_a.contract_pair(&tensor_b).unwrap();

    assert_eq!(result.indices, vec![i, j]);
    let expected = TensorDynLen::from_dense(
        result.indices.clone(),
        vec![
            6.0, -3.0, 8.0, //
            -4.0, -4.0, 2.0,
        ],
    )
    .unwrap();
    assert!(result.isapprox(&expected, 1e-12, 0.0));
}

#[test]
fn structured_tensor_contract_materializes_to_correct_dense_result() {
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(2);
    let k = Index::new_dyn(2);
    let diag = TensorDynLen::from_diag(vec![i.clone(), j.clone()], vec![2.0_f64, 3.0]).unwrap();
    assert!(diag.is_diag());
    let dense = TensorDynLen::from_dense(vec![j, k.clone()], vec![5.0, 7.0, 11.0, 13.0]).unwrap();

    let result = diag.contract_pair(&dense).unwrap();

    let expected = TensorDynLen::from_dense(vec![i, k], vec![10.0, 21.0, 22.0, 39.0]).unwrap();
    assert!(result.sub(&expected).unwrap().maxabs() < 1e-12);
}

#[test]
fn general_structured_contract_preserves_output_axis_classes() {
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(3);
    let k = Index::new_dyn(2);
    let l = Index::new_dyn(2);
    let structured = TensorDynLen::from_storage(
        vec![i.clone(), j.clone(), k.clone()],
        Storage::new_structured(
            vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![2, 3],
            vec![1, 2],
            vec![0, 1, 0],
        )
        .map(std::sync::Arc::new)
        .unwrap(),
    )
    .unwrap();
    let dense = TensorDynLen::from_dense(
        vec![j, l.clone()],
        vec![10.0, 20.0, 30.0, 100.0, 200.0, 300.0],
    )
    .unwrap();

    let result = structured.contract_pair(&dense).unwrap();

    assert_eq!(result.indices, vec![i, k, l]);
    assert_eq!(
        result.storage().unwrap().storage_kind(),
        StorageKind::Structured
    );
    assert_eq!(result.storage().unwrap().axis_classes(), &[0, 0, 1]);
    assert_eq!(
        result
            .storage()
            .unwrap()
            .payload_f64_col_major_vec()
            .unwrap(),
        vec![220.0, 280.0, 2200.0, 2800.0]
    );
    assert_eq!(
        result.to_vec::<f64>().unwrap(),
        vec![220.0, 0.0, 0.0, 280.0, 2200.0, 0.0, 0.0, 2800.0]
    );
}

#[test]
fn test_contract_three_indices() {
    // Create A[i, j, k] and B[j, k, l]
    // Contract along j and k: result should be C[i, l]
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(3);
    let k = Index::new_dyn(4);
    let l = Index::new_dyn(5);

    // Create tensor A[i, j, k] with all ones
    let tensor_a = dense_f64(vec![i.clone(), j.clone(), k.clone()], vec![1.0; 24]);

    // Create tensor B[j, k, l] with all ones
    let tensor_b = dense_f64(vec![j.clone(), k.clone(), l.clone()], vec![1.0; 60]);

    // Contract along j and k: result should be C[i, l] with all 12.0 (3 * 4 = 12)
    let result = tensor_a.contract_pair(&tensor_b).unwrap();
    assert_eq!(result.dims(), vec![2, 5]);
    assert_eq!(result.indices.len(), 2);
    assert_eq!(result.indices[0].id, i.id);
    assert_eq!(result.indices[1].id, l.id);

    assert_all_f64(&result, 10, 12.0);
}

#[test]
fn test_contract_mixed_f64_c64() {
    // Test contraction between f64 and Complex64 tensors
    // A[i, j] (f64) × B[j, k] (Complex64) = C[i, k] (Complex64)
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(2);
    let k = Index::new_dyn(2);

    // Create tensor A[i, j] with all 1.0 (f64)
    let tensor_a = dense_f64(vec![i.clone(), j.clone()], vec![1.0; 4]);

    // Create tensor B[j, k] with complex values: [[1+2i, 3+4i], [5+6i, 7+8i]]
    let tensor_b = dense_c64(
        vec![j.clone(), k.clone()],
        vec![
            Complex64::new(1.0, 2.0),
            Complex64::new(5.0, 6.0),
            Complex64::new(3.0, 4.0),
            Complex64::new(7.0, 8.0),
        ],
    );

    // Contract along j: result should be C[i, k] (Complex64)
    // Expected result: [[1+2i + 5+6i, 3+4i + 7+8i], [1+2i + 5+6i, 3+4i + 7+8i]]
    //                  = [[6+8i, 10+12i], [6+8i, 10+12i]]
    let result = tensor_a.contract_pair(&tensor_b).unwrap();
    assert_eq!(result.dims(), vec![2, 2]);
    assert_eq!(result.indices.len(), 2);
    assert_eq!(result.indices[0].id, i.id);
    assert_eq!(result.indices[1].id, k.id);

    assert_eq!(
        result.to_vec::<Complex64>().unwrap(),
        vec![
            Complex64::new(6.0, 8.0),
            Complex64::new(6.0, 8.0),
            Complex64::new(10.0, 12.0),
            Complex64::new(10.0, 12.0),
        ]
    );
}

#[test]
fn test_contract_mixed_c64_f64() {
    // Test contraction between Complex64 and f64 tensors (reverse order)
    // A[i, j] (Complex64) × B[j, k] (f64) = C[i, k] (Complex64)
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(2);
    let k = Index::new_dyn(2);

    // Create tensor A[i, j] with complex values
    let tensor_a = dense_c64(
        vec![i.clone(), j.clone()],
        vec![
            Complex64::new(1.0, 2.0),
            Complex64::new(5.0, 6.0),
            Complex64::new(3.0, 4.0),
            Complex64::new(7.0, 8.0),
        ],
    );

    // Create tensor B[j, k] with all 1.0 (f64)
    let tensor_b = dense_f64(vec![j.clone(), k.clone()], vec![1.0; 4]);

    // Contract along j: result should be C[i, k] (Complex64)
    // For A[i,j] * B[j,k] where A is complex and B is real:
    // C[i,k] = sum_j A[i,j] * B[j,k]
    // With A = [[1+2i, 3+4i], [5+6i, 7+8i]] and B = [[1, 1], [1, 1]]
    // C[0,0] = (1+2i)*1 + (3+4i)*1 = 4+6i
    // C[0,1] = (1+2i)*1 + (3+4i)*1 = 4+6i
    // C[1,0] = (5+6i)*1 + (7+8i)*1 = 12+14i
    // C[1,1] = (5+6i)*1 + (7+8i)*1 = 12+14i
    let result = tensor_a.contract_pair(&tensor_b).unwrap();
    assert_eq!(result.dims(), vec![2, 2]);

    assert_eq!(
        result.to_vec::<Complex64>().unwrap(),
        vec![
            Complex64::new(4.0, 6.0),
            Complex64::new(12.0, 14.0),
            Complex64::new(4.0, 6.0),
            Complex64::new(12.0, 14.0),
        ]
    );
}

#[test]
fn test_contract_pair_with_lhs_conj_matches_materialized_conj() {
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(2);
    let k = Index::new_dyn(2);

    let lhs = dense_c64(
        vec![i.clone(), j.clone()],
        vec![
            Complex64::new(1.0, 2.0),
            Complex64::new(3.0, -1.0),
            Complex64::new(-2.0, 0.5),
            Complex64::new(0.25, 4.0),
        ],
    );
    let rhs = dense_c64(
        vec![j, k.clone()],
        vec![
            Complex64::new(0.5, -1.0),
            Complex64::new(2.0, 3.0),
            Complex64::new(-1.5, 0.25),
            Complex64::new(4.0, -0.5),
        ],
    );

    let flagged = contract_pair_with_operand_options(
        &lhs,
        &rhs,
        PairwiseContractionOptions::new().with_lhs_conj(true),
    )
    .unwrap();
    let materialized = contract_pair(&lhs.conj(), &rhs).unwrap();

    assert!(flagged.isapprox(&materialized, 1e-12, 0.0));
    assert_eq!(flagged.indices[0].id, i.id);
    assert_eq!(flagged.indices[1].id, k.id);
}

#[test]
fn test_contract_pair_with_rhs_conj_matches_materialized_conj() {
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(2);
    let k = Index::new_dyn(2);

    let lhs = dense_c64(
        vec![i.clone(), j.clone()],
        vec![
            Complex64::new(1.0, -1.0),
            Complex64::new(0.0, 2.0),
            Complex64::new(3.0, 0.5),
            Complex64::new(-2.0, 1.5),
        ],
    );
    let rhs = dense_c64(
        vec![j, k.clone()],
        vec![
            Complex64::new(2.0, 1.0),
            Complex64::new(-3.0, 0.25),
            Complex64::new(1.5, -2.0),
            Complex64::new(0.5, 4.0),
        ],
    );

    let flagged = contract_pair_with_operand_options(
        &lhs,
        &rhs,
        PairwiseContractionOptions::new().with_rhs_conj(true),
    )
    .unwrap();
    let materialized = contract_pair(&lhs, &rhs.conj()).unwrap();

    assert!(flagged.isapprox(&materialized, 1e-12, 0.0));
    assert_eq!(flagged.indices[0].id, i.id);
    assert_eq!(flagged.indices[1].id, k.id);
}

#[test]
fn test_tensordot_different_ids() {
    // Test tensordot with indices that have different IDs but same dimensions
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(3);
    let k = Index::new_dyn(3); // Same dimension as j, but different ID
    let l = Index::new_dyn(4);

    // Create tensor A[i, j]
    let tensor_a = dense_f64(vec![i.clone(), j.clone()], vec![1.0; 6]);

    // Create tensor B[k, l] where k has same dimension as j but different ID
    let tensor_b = dense_f64(vec![k.clone(), l.clone()], vec![1.0; 12]);

    // Contract j (from A) with k (from B): result should be C[i, l] with all 3.0
    let result = tensordot(&tensor_a, &tensor_b, &[(j.clone(), k.clone())]).unwrap();
    assert_eq!(result.dims(), vec![2, 4]);
    assert_eq!(result.indices.len(), 2);
    assert_eq!(result.indices[0].id, i.id);
    assert_eq!(result.indices[1].id, l.id);

    assert_all_f64(&result, 8, 3.0);
}

#[test]
fn test_tensordot_dimension_mismatch() {
    // Test that dimension mismatch returns an error
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(3);
    let k = Index::new_dyn(5); // Different dimension from j

    let tensor_a = TensorDynLen::zeros::<f64>(vec![i.clone(), j.clone()]).unwrap();

    let tensor_b = TensorDynLen::zeros::<f64>(vec![k.clone()]).unwrap();

    let result = tensordot(&tensor_a, &tensor_b, &[(j.clone(), k.clone())]);
    assert!(result.is_err());
    if let Err(e) = result {
        let err_msg = format!("{}", e);
        assert!(
            err_msg.contains("Dimension") || err_msg.contains("mismatch"),
            "Expected dimension mismatch error, got: {}",
            err_msg
        );
    }
}

#[test]
fn test_tensordot_index_not_found() {
    // Test that specifying a non-existent index returns an error
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(3);
    let k = Index::new_dyn(3);
    let nonexistent = Index::new_dyn(3);

    let tensor_a = TensorDynLen::zeros::<f64>(vec![i.clone(), j.clone()]).unwrap();

    let tensor_b = TensorDynLen::zeros::<f64>(vec![k.clone()]).unwrap();

    // Try to contract with a non-existent index from tensor_a
    let result = tensordot(&tensor_a, &tensor_b, &[(nonexistent.clone(), k.clone())]);
    assert!(result.is_err());
    if let Err(e) = result {
        let err_msg = format!("{}", e);
        assert!(
            err_msg.contains("not found") || err_msg.contains("Index"),
            "Expected index not found error, got: {}",
            err_msg
        );
    }
}

#[test]
fn test_tensordot_duplicate_axis() {
    // Test that specifying the same axis twice returns an error
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(3);
    let k = Index::new_dyn(3);
    let l = Index::new_dyn(4);

    let tensor_a = TensorDynLen::zeros::<f64>(vec![i.clone(), j.clone()]).unwrap();

    let tensor_b = TensorDynLen::zeros::<f64>(vec![k.clone(), l.clone()]).unwrap();

    // Try to contract j twice (duplicate axis in self)
    let result = tensordot(
        &tensor_a,
        &tensor_b,
        &[
            (j.clone(), k.clone()),
            (j.clone(), l.clone()), // j is used twice
        ],
    );
    // Just verify it's an error - duplicate axes should be detected
    assert!(result.is_err());
}

#[test]
fn test_tensordot_empty_pairs() {
    // Test that empty pairs returns an error
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(3);

    let tensor_a = TensorDynLen::zeros::<f64>(vec![i.clone(), j.clone()]).unwrap();

    let tensor_b = TensorDynLen::zeros::<f64>(vec![j.clone()]).unwrap();

    let result = tensordot(&tensor_a, &tensor_b, &[]);
    assert!(result.is_err());
    if let Err(e) = result {
        let err_msg = format!("{}", e);
        assert!(
            err_msg.contains("No pairs")
                || err_msg.contains("empty")
                || err_msg.contains("specified")
                || err_msg.contains("NoCommon"),
            "Expected empty pairs error, got: {}",
            err_msg
        );
    }
}

#[test]
fn test_tensordot_common_index_not_in_pairs() {
    // Test that having a common index (same ID) not in the contraction pairs returns an error
    // This is the "batch contraction not yet implemented" case
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(3); // This will be a common index (batch dimension)
    let k = Index::new_dyn(4);
    let l = Index::new_dyn(5);

    // Create tensor A[i, j, k]
    let tensor_a = TensorDynLen::zeros::<f64>(vec![i.clone(), j.clone(), k.clone()]).unwrap();

    // Create tensor B[j, l] where j is a common index with A
    let tensor_b = TensorDynLen::zeros::<f64>(vec![j.clone(), l.clone()]).unwrap();

    // Try to contract only k with l, leaving j as a "batch" dimension
    // This should fail because batch contraction is not yet implemented
    let result = tensordot(&tensor_a, &tensor_b, &[(k.clone(), l.clone())]);
    assert!(result.is_err());
    if let Err(e) = result {
        let err_msg = format!("{}", e);
        assert!(
            err_msg.contains("batch")
                || err_msg.contains("not yet implemented")
                || err_msg.contains("Common index"),
            "Expected batch contraction error, got: {}",
            err_msg
        );
    }
}

#[test]
fn test_tensordot_common_index_in_pairs_ok() {
    // Test that having a common index that IS in the contraction pairs works fine
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(3); // This is a common index, but we will contract it
    let k = Index::new_dyn(4);

    // Create tensor A[i, j]
    let tensor_a = dense_f64(vec![i.clone(), j.clone()], vec![1.0; 6]);

    // Create tensor B[j, k] where j is a common index with A
    let tensor_b = dense_f64(vec![j.clone(), k.clone()], vec![1.0; 12]);

    // Contract j with j - this should work because the common index is in pairs
    let result = tensordot(&tensor_a, &tensor_b, &[(j.clone(), j.clone())]);
    assert!(result.is_ok());
    let result = result.unwrap();
    assert_eq!(result.dims(), vec![2, 4]);
}

// --- Scalar (identity) tensor contraction tests ---

#[test]
fn test_scalar_times_tensor() {
    // scalar_one() * tensor = tensor
    let scalar = TensorDynLen::scalar_one().unwrap();
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(3);
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let tensor = TensorDynLen::from_dense(vec![i.clone(), j.clone()], data.clone()).unwrap();

    let result = scalar.contract_pair(&tensor).unwrap();
    assert_eq!(result.dims(), vec![2, 3]);
    assert_eq!(result.to_vec::<f64>().unwrap(), data);
}

#[test]
fn test_tensor_times_scalar() {
    // tensor * scalar_one() = tensor
    let scalar = TensorDynLen::scalar_one().unwrap();
    let i = Index::new_dyn(2);
    let data = vec![10.0, 20.0];
    let tensor = TensorDynLen::from_dense(vec![i.clone()], data.clone()).unwrap();

    let result = tensor.contract_pair(&scalar).unwrap();
    assert_eq!(result.dims(), vec![2]);
    assert_eq!(result.to_vec::<f64>().unwrap(), data);
}

#[test]
fn test_scalar_times_scalar() {
    let s1 = TensorDynLen::scalar(3.0).unwrap();
    let s2 = TensorDynLen::scalar(5.0).unwrap();

    let result = s1.contract_pair(&s2).unwrap();
    assert_eq!(result.dims().len(), 0);
    let val = result.to_vec::<f64>().unwrap();
    assert_eq!(val.len(), 1);
    assert!((val[0] - 15.0).abs() < 1e-10);
}

#[test]
fn test_mul_operator_scalar_times_tensor() {
    // &scalar * &tensor via Mul trait
    let scalar = TensorDynLen::scalar_one().unwrap();
    let i = Index::new_dyn(3);
    let data = vec![1.0, 2.0, 3.0];
    let tensor = TensorDynLen::from_dense(vec![i.clone()], data.clone()).unwrap();

    let result = scalar.contract_pair(&tensor).unwrap();
    assert_eq!(result.dims(), vec![3]);
    assert_eq!(result.to_vec::<f64>().unwrap(), data);
}

#[test]
fn test_foldl_sequential_contraction() {
    // Simulate foldl-style: acc = scalar_one; acc = acc * a; acc = acc * b;
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(3);
    let a = TensorDynLen::from_dense(vec![i.clone(), j.clone()], vec![1.0; 6]).unwrap();
    let b = TensorDynLen::from_dense(vec![j.clone(), i.clone()], vec![2.0; 6]).unwrap();

    let mut acc = TensorDynLen::scalar_one().unwrap();
    acc = acc.contract_pair(&a).unwrap(); // acc = a (outer product with scalar)
    acc = acc.contract_pair(&b).unwrap(); // acc = contract(a, b) over i and j

    // a[i,j] * b[j,i] = sum_j(a[i,j]*b[j,i]) summed over both → scalar
    assert_eq!(acc.dims().len(), 0);
    let val = acc.to_vec::<f64>().unwrap();
    // All elements are 1.0*2.0 = 2.0, summed over 2*3 = 6 elements → 12.0
    assert!((val[0] - 12.0).abs() < 1e-10);
}
