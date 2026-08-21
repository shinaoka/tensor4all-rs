use super::*;

#[test]
fn matrix_shape_product_overflow_is_reported() {
    let error = checked_matrix_len(usize::MAX, 2, "test").unwrap_err();
    let message = error.to_string();
    assert!(message.contains("overflows usize"), "{message}");
}

#[test]
fn matrix_materialization_rejects_overflow_before_length_check() {
    let error = matrix_from_col_major_values::<f64>(Vec::new(), usize::MAX, 2, "test").unwrap_err();
    let message = error.to_string();
    assert!(message.contains("overflows usize"), "{message}");
}

#[test]
fn matrix_luci_factors_convert_to_indexed_column_major_factors() {
    let left_index = DynIndex::new_dyn(2);
    let right_index = DynIndex::new_dyn(3);
    let factors = crate::MatrixLuciFactors {
        row_indices: vec![0, 1],
        col_indices: vec![0, 1],
        pivot_errors: Vec::new(),
        rank: 2,
        left: tensor4all_tensorbackend::Matrix::from_col_major_vec(2, 2, vec![1.0, 0.0, 0.0, 1.0]),
        right: tensor4all_tensorbackend::Matrix::from_col_major_vec(
            2,
            3,
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        ),
    };

    let (left, right, bond_index) = matrix_luci_factors_to_idx_tensors(
        factors,
        std::slice::from_ref(&left_index),
        std::slice::from_ref(&right_index),
    )
    .unwrap();

    assert_eq!(left.dims(), vec![2, 2]);
    assert_eq!(right.dims(), vec![2, 3]);
    assert_eq!(left.indices(), &[left_index, bond_index.clone()]);
    assert_eq!(right.indices(), &[bond_index, right_index]);
    assert_eq!(left.to_vec::<f64>().unwrap(), vec![1.0, 0.0, 0.0, 1.0]);
    assert_eq!(
        right.to_vec::<f64>().unwrap(),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    );
}
