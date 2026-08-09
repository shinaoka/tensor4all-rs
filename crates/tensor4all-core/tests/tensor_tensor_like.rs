//! Tests for TensorLike trait implementation.

use tensor4all_core::index::{DynId, Index};
use tensor4all_core::DynIndex;
use tensor4all_core::{
    outer_product, TensorConstructionLike, TensorContractionLike, TensorDynLen, TensorIndex,
    TensorVectorSpace,
};

/// Helper to create a simple tensor with given dimensions
fn make_tensor(dims: &[usize]) -> TensorDynLen {
    let indices: Vec<DynIndex> = dims.iter().map(|&d| Index::new_dyn(d)).collect();
    let total_size: usize = dims.iter().product();
    let data: Vec<f64> = (0..total_size).map(|i| i as f64).collect();
    TensorDynLen::from_dense(indices, data).unwrap()
}

#[test]
fn test_tensor_like_external_indices() {
    let tensor = make_tensor(&[2, 3, 4]);

    // Use TensorLike trait
    let external_indices = tensor.external_indices();
    assert_eq!(external_indices.len(), 3);

    // Check dimensions through the indices
    use tensor4all_core::index_like::IndexLike;
    assert_eq!(external_indices[0].dim(), 2);
    assert_eq!(external_indices[1].dim(), 3);
    assert_eq!(external_indices[2].dim(), 4);
}

#[test]
fn test_tensor_like_num_external_indices() {
    let tensor = make_tensor(&[5, 6]);

    assert_eq!(tensor.num_external_indices(), 2);
}

#[test]
fn test_tensor_like_contract_basic() {
    // Create two tensors: A(i,j) and B(j,k)
    // Contract over j to get C(i,k)
    let i = Index::<DynId>::new_dyn(2);
    let j = Index::<DynId>::new_dyn(3);
    let k = Index::<DynId>::new_dyn(4);

    // Tensor A: 2x3 matrix
    let a_data: Vec<f64> = (0..6).map(|x| x as f64).collect();
    let a = TensorDynLen::from_dense(vec![i.clone(), j.clone()], a_data).unwrap();

    // Tensor B: 3x4 matrix (use a copy of j with same id)
    let j_copy = Index::new(j.id, j.dim);
    let b_data: Vec<f64> = (0..12).map(|x| x as f64).collect();
    let b = TensorDynLen::from_dense(vec![j_copy.clone(), k.clone()], b_data).unwrap();

    // Use TensorContractionLike::contract - auto-detects contractable pairs via is_contractable
    let c = <TensorDynLen as TensorContractionLike>::contract(&[&a, &b])
        .expect("contract should succeed");

    // Result should be 2x4
    assert_eq!(c.dims(), vec![2, 4]);
}

#[test]
fn tensor_vector_space_default_methods_cover_common_paths() {
    let i = Index::<DynId>::new_dyn(3);
    let a = TensorDynLen::from_dense(vec![i.clone()], vec![1.0, -2.0, 3.0]).unwrap();
    let b = TensorDynLen::from_dense(vec![i.clone()], vec![1.0, -2.0, 3.0 + 1.0e-13]).unwrap();

    let neg = TensorVectorSpace::neg(&a).unwrap();
    assert_eq!(neg.to_vec::<f64>().unwrap(), vec![-1.0, 2.0, -3.0]);
    assert!(a.isapprox(&b, 1.0e-12, 0.0).unwrap());

    let j = Index::<DynId>::new_dyn(2);
    let incompatible = TensorDynLen::from_dense(vec![j], vec![1.0, 2.0]).unwrap();
    assert!(a.isapprox(&incompatible, 1.0e-12, 0.0).is_err());
}

#[test]
fn tensor_contraction_and_construction_default_methods_cover_paths() {
    let i = Index::<DynId>::new_dyn(2);
    let j = Index::<DynId>::new_dyn(3);
    let a = TensorDynLen::from_dense(vec![i.clone()], vec![2.0, 5.0]).unwrap();
    let b = TensorDynLen::from_dense(vec![i.clone(), j.clone()], vec![1.0; 6]).unwrap();

    let contracted = a.contract_pair(&b).unwrap();
    assert_eq!(contracted.indices, vec![j.clone()]);
    assert_eq!(contracted.to_vec::<f64>().unwrap(), vec![7.0, 7.0, 7.0]);
    TensorContractionLike::validate(&contracted).unwrap();

    let unchanged = TensorConstructionLike::select_indices(&b, &[], &[]).unwrap();
    assert!(unchanged.isapprox(&b, 0.0, 0.0).unwrap());

    let selected =
        TensorConstructionLike::select_indices(&b, std::slice::from_ref(&i), &[1]).unwrap();
    assert_eq!(selected.indices, vec![j.clone()]);
    assert_eq!(selected.to_vec::<f64>().unwrap(), vec![1.0, 1.0, 1.0]);

    let err =
        TensorConstructionLike::select_indices(&b, std::slice::from_ref(&i), &[2]).unwrap_err();
    assert!(err.to_string().contains("out of range"));
}

#[test]
fn test_contract_three_tensor_chain() {
    // Create three tensors: A(i,j), B(j,k), C(k,l)
    // j is shared between A and B, k is shared between B and C.
    let i = Index::<DynId>::new_dyn(2);
    let j = Index::<DynId>::new_dyn(3);
    let k = Index::<DynId>::new_dyn(4);
    let l = Index::<DynId>::new_dyn(5);

    // Tensor A: 2x3 matrix (i, j)
    let a_data: Vec<f64> = (0..6).map(|x| x as f64).collect();
    let a = TensorDynLen::from_dense(vec![i.clone(), j.clone()], a_data).unwrap();

    // Tensor B: 3x4 matrix (j, k) - j has same id as A's j
    let j_copy = Index::new(j.id, j.dim);
    let b_data: Vec<f64> = (0..12).map(|x| x as f64).collect();
    let b = TensorDynLen::from_dense(vec![j_copy.clone(), k.clone()], b_data).unwrap();

    // Tensor C: 4x5 matrix (k, l) - k has same id as B's k
    let k_copy = Index::new(k.id, k.dim);
    let c_data: Vec<f64> = (0..20).map(|x| x as f64).collect();
    let c = TensorDynLen::from_dense(vec![k_copy.clone(), l.clone()], c_data).unwrap();

    let result = <TensorDynLen as TensorContractionLike>::contract(&[&a, &b, &c])
        .expect("contract should succeed");

    // Result should have: i (from A, dim=2), l (from C, dim=5)
    // j and k are contracted
    let mut sorted_dims = result.dims();
    assert_eq!(sorted_dims.len(), 2);
    sorted_dims.sort();
    assert_eq!(sorted_dims, vec![2, 5]);
}

#[test]
fn test_outer_product_with_common_indices_errors() {
    let i = Index::<DynId>::new_dyn(2);
    let j = Index::<DynId>::new_dyn(3);

    // Tensor A: 2x3 matrix
    let a_data: Vec<f64> = (0..6).map(|x| x as f64).collect();
    let a = TensorDynLen::from_dense(vec![i.clone(), j.clone()], a_data).unwrap();

    // Tensor B: 2x3 matrix (use copies of i and j with same ids)
    let i_copy = Index::new(i.id, i.dim);
    let j_copy = Index::new(j.id, j.dim);
    let b_data: Vec<f64> = (0..6).map(|x| x as f64).collect();
    let b = TensorDynLen::from_dense(vec![i_copy.clone(), j_copy.clone()], b_data).unwrap();

    let result = outer_product(&a, &b);

    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string().to_lowercase();
    assert!(err_msg.contains("common indices"));
}

#[test]
fn test_outer_product_disconnected_tensors() {
    // Disconnected inputs require an explicit outer product.
    let i = Index::<DynId>::new_dyn(2);
    let j = Index::<DynId>::new_dyn(3);
    let k = Index::<DynId>::new_dyn(4);
    let l = Index::<DynId>::new_dyn(5);

    // Tensor A: 2x3 matrix with indices (i, j)
    let a_data: Vec<f64> = (0..6).map(|x| x as f64).collect();
    let a = TensorDynLen::from_dense(vec![i.clone(), j.clone()], a_data).unwrap();

    // Tensor B: 4x5 matrix with indices (k, l) - different from a
    let b_data: Vec<f64> = (0..20).map(|x| x as f64).collect();
    let b = TensorDynLen::from_dense(vec![k.clone(), l.clone()], b_data).unwrap();

    let result = outer_product(&a, &b).unwrap();

    // Result should have 4 indices (i, j, k, l)
    let mut sorted_dims = result.dims();
    assert_eq!(sorted_dims.len(), 4);
    sorted_dims.sort();
    assert_eq!(sorted_dims, vec![2, 3, 4, 5]);
}

#[test]
fn test_outer_product_preserves_input_component_order() {
    let i = Index::<DynId>::new_dyn(2);
    let j = Index::<DynId>::new_dyn(3);

    let a = TensorDynLen::from_dense(vec![i.clone()], vec![2.0, -1.0]).unwrap();
    let b = TensorDynLen::from_dense(vec![j.clone()], vec![3.0, 4.0, -2.0]).unwrap();

    let result = outer_product(&a, &b).unwrap();

    assert_eq!(result.indices, vec![i, j]);
    let expected = TensorDynLen::from_dense(
        result.indices.clone(),
        vec![
            6.0, -3.0, 8.0, //
            -4.0, -4.0, 2.0,
        ],
    )
    .unwrap();
    assert!(result.isapprox(&expected, 1e-12, 0.0).unwrap());
}

#[test]
fn test_contract_components_then_outer_product() {
    let i = Index::<DynId>::new_dyn(2);
    let j = Index::<DynId>::new_dyn(3);

    let a = TensorDynLen::from_dense(vec![i.clone()], vec![1.0, 2.0]).unwrap();
    let i_copy = Index::new(i.id, i.dim);
    let b = TensorDynLen::from_dense(vec![i_copy.clone()], vec![3.0, 4.0]).unwrap();
    let c = TensorDynLen::from_dense(vec![j.clone()], vec![5.0, 6.0, 7.0]).unwrap();
    let j_copy = Index::new(j.id, j.dim);
    let d = TensorDynLen::from_dense(vec![j_copy.clone()], vec![8.0, 9.0, 10.0]).unwrap();

    let left = <TensorDynLen as TensorContractionLike>::contract(&[&a, &b]).unwrap();
    let right = <TensorDynLen as TensorContractionLike>::contract(&[&c, &d]).unwrap();
    let result = outer_product(&left, &right).unwrap();

    // A(i) * B(i) contracts to scalar (dim 0)
    // C(j) * D(j) contracts to scalar (dim 0)
    // Outer product of two scalars is a scalar
    assert_eq!(result.dims().len(), 0);
}

// ============================================================================
// onehot tests
// ============================================================================

#[test]
fn test_onehot_1d() {
    let i = Index::new_dyn(3);
    let t = TensorDynLen::onehot(&[(i.clone(), 0)]).unwrap();
    assert_eq!(t.dims(), vec![3]);
    let data = t.to_vec::<f64>().unwrap();
    assert_eq!(data, vec![1.0, 0.0, 0.0]);
}

#[test]
fn test_onehot_2d() {
    let i = Index::new_dyn(3);
    let j = Index::new_dyn(4);
    let t = TensorDynLen::onehot(&[(i.clone(), 1), (j.clone(), 2)]).unwrap();
    assert_eq!(t.dims(), vec![3, 4]);
    let data = t.to_vec::<f64>().unwrap();
    // Column-major: position (1,2) in 3×4 = 1 + 3*2 = 7
    let mut expected = vec![0.0; 12];
    expected[7] = 1.0;
    assert_eq!(data, expected);
}

#[test]
fn test_onehot_boundary() {
    let i = Index::new_dyn(3);
    let j = Index::new_dyn(4);
    // Last position in the first dimension, nontrivial column in the second.
    let t = TensorDynLen::onehot(&[(i.clone(), 2), (j.clone(), 1)]).unwrap();
    let data = t.to_vec::<f64>().unwrap();
    // Column-major: position (2,1) in 3×4 = 2 + 3*1 = 5
    let mut expected = vec![0.0; 12];
    expected[5] = 1.0;
    assert_eq!(data, expected);
}

#[test]
fn test_onehot_error_out_of_bounds() {
    let i = Index::new_dyn(3);
    let result = TensorDynLen::onehot(&[(i.clone(), 3)]);
    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();
    assert!(err_msg.contains("onehot"));
}

#[test]
fn test_onehot_empty() {
    // Empty input should return scalar 1.0
    let t = TensorDynLen::onehot(&[]).unwrap();
    assert_eq!(t.dims().len(), 0);
}

#[test]
fn test_onehot_contraction() {
    // Create a tensor A(i,j) and a onehot V(i)
    let i = Index::new_dyn(3);
    let j = Index::new_dyn(4);
    let a = TensorDynLen::from_dense(
        vec![i.clone(), j.clone()],
        (0..12).map(|x| x as f64).collect(),
    )
    .unwrap();

    // onehot selecting i=1
    let v = TensorDynLen::onehot(&[(i.clone(), 1)]).unwrap();

    // Contract: V(i) * A(i,j) = A[1,:]
    let result = <TensorDynLen as TensorContractionLike>::contract(&[&v, &a]).unwrap();
    assert_eq!(result.dims(), vec![4]);
    let data = result.to_vec::<f64>().unwrap();
    // Row i=1 of the column-major 3×4 matrix: [1, 4, 7, 10]
    assert_eq!(data, vec![1.0, 4.0, 7.0, 10.0]);
}

// Note: trait object tests removed - TensorLike is now fully generic and does not support dyn
