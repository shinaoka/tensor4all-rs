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
