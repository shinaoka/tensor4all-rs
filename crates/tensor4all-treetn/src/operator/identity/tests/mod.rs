use super::*;
use tensor4all_core::index::Index;

fn make_index(dim: usize) -> DynIndex {
    Index::new_dyn(dim)
}

fn get_f64_data(tensor: &IdxTensor) -> Vec<f64> {
    tensor.to_vec::<f64>().expect("Expected DenseF64 storage")
}

#[test]
fn test_identity_single_site() {
    let s_in = make_index(2);
    let s_out = make_index(2);

    let tensor =
        build_identity_operator_tensor(std::slice::from_ref(&s_in), std::slice::from_ref(&s_out))
            .unwrap();

    assert_eq!(tensor.indices.len(), 2);
    assert_eq!(tensor.dims(), vec![2, 2]);

    // Check diagonal elements are 1
    let data = get_f64_data(&tensor);
    // Layout: [s_in, s_out] in column-major
    // (0,0) -> idx 0, (1,0) -> idx 1, (0,1) -> idx 2, (1,1) -> idx 3
    assert_eq!(data[0], 1.0); // (0,0)
    assert_eq!(data[1], 0.0); // (1,0)
    assert_eq!(data[2], 0.0); // (0,1)
    assert_eq!(data[3], 1.0); // (1,1)
}

#[test]
fn test_identity_two_sites() {
    let s1_in = make_index(2);
    let s1_out = make_index(2);
    let s2_in = make_index(3);
    let s2_out = make_index(3);

    let tensor = build_identity_operator_tensor(
        &[s1_in.clone(), s2_in.clone()],
        &[s1_out.clone(), s2_out.clone()],
    )
    .unwrap();

    assert_eq!(tensor.indices.len(), 4);
    assert_eq!(tensor.dims(), vec![2, 2, 3, 3]);

    let data = get_f64_data(&tensor);
    let dims = tensor.dims();
    let total_size: usize = dims.iter().product();
    assert_eq!(data.len(), total_size);

    // Count non-zero elements: should be 2*3 = 6 (diagonal)
    let nonzero_count = data.iter().filter(|&&x| x != 0.0).count();
    assert_eq!(nonzero_count, 6);

    // All non-zero should be 1.0
    for &val in data.iter() {
        assert!(val == 0.0 || val == 1.0);
    }
}

#[test]
fn test_identity_dimension_mismatch() {
    let s_in = make_index(2);
    let s_out = make_index(3); // Different dimension

    let result = build_identity_operator_tensor(&[s_in], &[s_out]);
    assert!(result.is_err());
}

#[test]
fn test_identity_empty() {
    let tensor = build_identity_operator_tensor(&[], &[]).unwrap();

    assert_eq!(tensor.indices.len(), 0);
    assert_eq!(tensor.dims().len(), 0);

    let data = get_f64_data(&tensor);
    assert_eq!(data.len(), 1);
    assert_eq!(data[0], 1.0);
}

#[test]
fn test_identity_single_site_values() {
    // The identity operator uses delta() which produces f64 tensors.
    // Verify the values match the expected identity matrix pattern.
    let s_in = make_index(2);
    let s_out = make_index(2);

    let tensor =
        build_identity_operator_tensor(std::slice::from_ref(&s_in), std::slice::from_ref(&s_out))
            .unwrap();

    let data = get_f64_data(&tensor);
    assert_eq!(data[0], 1.0);
    assert_eq!(data[1], 0.0);
    assert_eq!(data[2], 0.0);
    assert_eq!(data[3], 1.0);
}

#[test]
fn test_identity_multi_site_values() {
    // Test with multiple sites to cover the delta() behavior for multi-site identity.
    let s1_in = make_index(2);
    let s1_out = make_index(2);
    let s2_in = make_index(2);
    let s2_out = make_index(2);

    let tensor = build_identity_operator_tensor(&[s1_in, s2_in], &[s1_out, s2_out]).unwrap();

    // Shape should be [2, 2, 2, 2] = 16 elements
    assert_eq!(tensor.dims(), vec![2, 2, 2, 2]);
    let data = get_f64_data(&tensor);
    assert_eq!(data.len(), 16);

    // Diagonal elements should be 1, others 0
    // In identity operator, data[i, i, j, j] = 1 for all i, j
    // Linear index: (i1, o1, i2, o2) -> i1*8 + o1*4 + i2*2 + o2
    // (0,0,0,0)=0, (0,0,1,1)=3, (1,1,0,0)=12, (1,1,1,1)=15
    assert_eq!(data[0], 1.0);
    assert_eq!(data[3], 1.0);
    assert_eq!(data[12], 1.0);
    assert_eq!(data[15], 1.0);
}
