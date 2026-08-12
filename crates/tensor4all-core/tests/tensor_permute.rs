use num_complex::Complex64;
use tensor4all_core::index::DefaultIndex as Index;
use tensor4all_core::{compute_permutation_from_indices, DynIndex, IdxTensor};

fn dense_f64(indices: Vec<DynIndex>, data: Vec<f64>) -> IdxTensor {
    IdxTensor::from_dense(indices, data).unwrap()
}

fn dense_c64(indices: Vec<DynIndex>, data: Vec<Complex64>) -> IdxTensor {
    IdxTensor::from_dense(indices, data).unwrap()
}

fn col_major_multi_index(mut flat: usize, dims: &[usize]) -> Vec<usize> {
    dims.iter()
        .map(|&dim| {
            let idx = flat % dim;
            flat /= dim;
            idx
        })
        .collect()
}

fn col_major_offset(dims: &[usize], indices: &[usize]) -> usize {
    let mut stride = 1;
    let mut offset = 0;
    for (&idx, &dim) in indices.iter().zip(dims.iter()) {
        offset += idx * stride;
        stride *= dim;
    }
    offset
}

fn permute_col_major<T: Clone>(data: &[T], dims: &[usize], perm: &[usize]) -> Vec<T> {
    let permuted_dims: Vec<usize> = perm.iter().map(|&axis| dims[axis]).collect();
    (0..data.len())
        .map(|flat| {
            let permuted_index = col_major_multi_index(flat, &permuted_dims);
            let mut source_index = vec![0; dims.len()];
            for (new_axis, &old_axis) in perm.iter().enumerate() {
                source_index[old_axis] = permuted_index[new_axis];
            }
            data[col_major_offset(dims, &source_index)].clone()
        })
        .collect()
}

#[test]
fn test_compute_permutation_from_indices() {
    // Test the independent permutation computation function
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(3);
    let k = Index::new_dyn(4);

    let original = vec![i.clone(), j.clone(), k.clone()];

    // Test identity permutation
    let new_order1 = vec![i.clone(), j.clone(), k.clone()];
    let perm1 = compute_permutation_from_indices(&original, &new_order1).unwrap();
    assert_eq!(perm1, vec![0, 1, 2]);

    // Test swap first two
    let new_order2 = vec![j.clone(), i.clone(), k.clone()];
    let perm2 = compute_permutation_from_indices(&original, &new_order2).unwrap();
    assert_eq!(perm2, vec![1, 0, 2]);

    // Test reverse
    let new_order3 = vec![k.clone(), j.clone(), i.clone()];
    let perm3 = compute_permutation_from_indices(&original, &new_order3).unwrap();
    assert_eq!(perm3, vec![2, 1, 0]);

    // Test rotation
    let new_order4 = vec![j.clone(), k.clone(), i.clone()];
    let perm4 = compute_permutation_from_indices(&original, &new_order4).unwrap();
    assert_eq!(perm4, vec![1, 2, 0]);
}

#[test]
fn test_compute_permutation_from_indices_invalid() {
    // Test with invalid index (ID doesn't match)
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(3);
    let invalid = Index::new_dyn(5); // Different ID

    let original = vec![i.clone(), j.clone()];
    let new_order = vec![i.clone(), invalid];

    let err = compute_permutation_from_indices(&original, &new_order).unwrap_err();
    assert!(err.to_string().contains("permutation"));
}

#[test]
fn test_compute_permutation_from_indices_duplicate() {
    // Test with duplicate indices
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(3);

    let original = vec![i.clone(), j.clone()];
    let new_order = vec![i.clone(), i.clone()]; // Duplicate

    let err = compute_permutation_from_indices(&original, &new_order).unwrap_err();
    assert!(err.to_string().contains("duplicate"));
}

#[test]
fn test_permute_dyn_f64_2d() {
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(3);
    let indices = vec![i.clone(), j.clone()];
    let dims = vec![2, 3];
    let data = vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
    let tensor = dense_f64(indices, data.clone());

    let permuted = tensor.permute(&[1, 0]).unwrap();

    assert_eq!(permuted.dims(), vec![3, 2]);
    assert_eq!(permuted.indices[0].id, j.id);
    assert_eq!(permuted.indices[1].id, i.id);
    assert_eq!(
        permuted.to_vec::<f64>().unwrap(),
        permute_col_major(&data, &dims, &[1, 0])
    );
}

#[test]
fn test_permute_dyn_c64_2d() {
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(3);
    let indices = vec![i.clone(), j.clone()];
    let dims = vec![2, 3];
    let data = vec![
        Complex64::new(1.0, 0.0),
        Complex64::new(4.0, 0.0),
        Complex64::new(2.0, 0.0),
        Complex64::new(5.0, 0.0),
        Complex64::new(3.0, 0.0),
        Complex64::new(6.0, 0.0),
    ];
    let tensor = dense_c64(indices, data.clone());

    let permuted = tensor.permute(&[1, 0]).unwrap();

    assert_eq!(permuted.dims(), vec![3, 2]);
    assert_eq!(permuted.indices[0].id, j.id);
    assert_eq!(permuted.indices[1].id, i.id);
    assert_eq!(
        permuted.to_vec::<Complex64>().unwrap(),
        permute_col_major(&data, &dims, &[1, 0])
    );
}

#[test]
fn test_permute_dyn_f64_3d() {
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(3);
    let k = Index::new_dyn(4);
    let indices = vec![i.clone(), j.clone(), k.clone()];
    let dims = vec![2, 3, 4];

    let data: Vec<f64> = (1..=24).map(|i| i as f64).collect();
    let tensor = dense_f64(indices, data.clone());

    let permuted = tensor.permute(&[2, 0, 1]).unwrap();

    assert_eq!(permuted.dims(), vec![4, 2, 3]);
    assert_eq!(permuted.indices[0].id, k.id);
    assert_eq!(permuted.indices[1].id, i.id);
    assert_eq!(permuted.indices[2].id, j.id);
    assert_eq!(
        permuted.to_vec::<f64>().unwrap(),
        permute_col_major(&data, &dims, &[2, 0, 1])
    );
}

#[test]
fn test_permute_identity() {
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(3);
    let indices = vec![i.clone(), j.clone()];
    let data = vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
    let tensor = dense_f64(indices, data.clone());

    let permuted = tensor.permute(&[0, 1]).unwrap();

    assert_eq!(permuted.dims(), vec![2, 3]);
    assert_eq!(permuted.indices[0].id, i.id);
    assert_eq!(permuted.indices[1].id, j.id);
    assert_eq!(permuted.to_vec::<f64>().unwrap(), data);
}

#[test]
fn test_permute_indices_dyn_f64_2d() {
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(3);
    let indices = vec![i.clone(), j.clone()];
    let dims = vec![2, 3];
    let data = vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
    let tensor = dense_f64(indices, data.clone());

    let permuted = tensor.permute_indices(&[j.clone(), i.clone()]).unwrap();

    assert_eq!(permuted.dims(), vec![3, 2]);
    assert_eq!(permuted.indices[0].id, j.id);
    assert_eq!(permuted.indices[1].id, i.id);
    assert_eq!(
        permuted.to_vec::<f64>().unwrap(),
        permute_col_major(&data, &dims, &[1, 0])
    );
}

#[test]
fn test_permute_indices_c64() {
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(3);
    let indices = vec![i.clone(), j.clone()];
    let dims = vec![2, 3];
    let data = vec![
        Complex64::new(1.0, 0.0),
        Complex64::new(4.0, 0.0),
        Complex64::new(2.0, 0.0),
        Complex64::new(5.0, 0.0),
        Complex64::new(3.0, 0.0),
        Complex64::new(6.0, 0.0),
    ];
    let tensor = dense_c64(indices, data.clone());

    let permuted = tensor.permute_indices(&[j.clone(), i.clone()]).unwrap();

    assert_eq!(permuted.dims(), vec![3, 2]);
    assert_eq!(permuted.indices[0].id, j.id);
    assert_eq!(permuted.indices[1].id, i.id);
    assert_eq!(
        permuted.to_vec::<Complex64>().unwrap(),
        permute_col_major(&data, &dims, &[1, 0])
    );
}
