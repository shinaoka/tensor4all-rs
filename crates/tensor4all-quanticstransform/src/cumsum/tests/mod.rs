use super::*;
use tensor4all_simplett::AbstractTensorTrain;

#[test]
fn test_upper_triangle_tensor() {
    let t = upper_triangle_tensor();

    // State 0 -> State 0: diagonal
    assert_eq!(t[0][0][0][0], Complex64::one());
    assert_eq!(t[0][0][1][1], Complex64::one());

    // State 0 -> State 1: y > x
    assert_eq!(t[0][1][1][0], Complex64::one());

    // State 0 -> nowhere: y < x
    assert_eq!(t[0][0][0][1], Complex64::zero());
    assert_eq!(t[0][1][0][1], Complex64::zero());

    // State 1 -> State 1: all ones
    assert_eq!(t[1][1][0][0], Complex64::one());
    assert_eq!(t[1][1][0][1], Complex64::one());
    assert_eq!(t[1][1][1][0], Complex64::one());
    assert_eq!(t[1][1][1][1], Complex64::one());
}

#[test]
fn test_cumsum_mpo_structure() {
    let mpo = cumsum_mpo(4).unwrap();
    assert_eq!(mpo.len(), 4);

    // First tensor: shape (1, 4, 2)
    assert_eq!(mpo.site_tensor(0).left_dim(), 1);
    assert_eq!(mpo.site_tensor(0).site_dim(), 4);
    assert_eq!(mpo.site_tensor(0).right_dim(), 2);

    // Middle tensors: shape (2, 4, 2)
    assert_eq!(mpo.site_tensor(1).left_dim(), 2);
    assert_eq!(mpo.site_tensor(1).site_dim(), 4);
    assert_eq!(mpo.site_tensor(1).right_dim(), 2);

    // Last tensor: shape (2, 4, 1)
    assert_eq!(mpo.site_tensor(3).left_dim(), 2);
    assert_eq!(mpo.site_tensor(3).site_dim(), 4);
    assert_eq!(mpo.site_tensor(3).right_dim(), 1);
}

#[test]
fn test_cumsum_operator_creation() {
    let op = cumsum_operator(4);
    assert!(op.is_ok());
}

#[test]
fn test_cumsum_error_zero_sites() {
    let result = cumsum_operator(0);
    assert!(result.is_err());
}

#[test]
fn test_cumsum_error_one_site() {
    let result = cumsum_operator(1);
    assert!(result.is_err());
    let msg = result.unwrap_err().to_string();
    assert!(msg.contains("at least 2"), "unexpected error: {msg}");
}

#[test]
fn test_cumsum_error_oversized_site_list() {
    let error = cumsum_operator(usize::MAX).unwrap_err();
    assert!(error.to_string().contains("site list"));
}

#[test]
fn test_triangle_error_oversized_site_list() {
    let error = triangle_operator(usize::MAX, TriangleType::Lower).unwrap_err();
    assert!(error.to_string().contains("site list"));
}

#[test]
fn test_triangle_error_one_site() {
    let result = triangle_operator(1, TriangleType::Lower);
    assert!(result.is_err());
    let msg = result.unwrap_err().to_string();
    assert!(msg.contains("at least 2"), "unexpected error: {msg}");
}
