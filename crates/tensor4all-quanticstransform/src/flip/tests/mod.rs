use super::*;
use tensor4all_simplett::AbstractTensorTrain;

#[test]
fn test_single_tensor_flip() {
    let t = single_tensor_flip();

    // Verify non-zero entries using formula: out = -a + cval[cin]
    // cval = [-1, 0], so cval[0] = -1, cval[1] = 0
    //
    // tensor[cin][cout][a][b] = 1 when input a produces output b

    // When cin=1 (carry=0), a=0: out = -0 + 0 = 0, cout=1 (out >= 0), b=0 mod 2 = 0
    assert_eq!(t[1][1][0][0], Complex64::one());

    // When cin=1 (carry=0), a=1: out = -1 + 0 = -1, cout=0 (out < 0), b=(-1) mod 2 = 1
    assert_eq!(t[1][0][1][1], Complex64::one());

    // When cin=0 (carry=-1), a=0: out = -0 + (-1) = -1, cout=0 (out < 0), b=(-1) mod 2 = 1
    assert_eq!(t[0][0][0][1], Complex64::one());

    // When cin=0 (carry=-1), a=1: out = -1 + (-1) = -2, cout=0 (out < 0), b=(-2) mod 2 = 0
    assert_eq!(t[0][0][1][0], Complex64::one());
}

#[test]
fn test_flip_mpo_structure() {
    let mpo = flip_mpo(4, BoundaryCondition::Periodic).unwrap();
    assert_eq!(mpo.len(), 4);

    // Big-endian convention:
    // First tensor (site 0 = MSB): BC applied on left, cin from right
    // Shape (1, 4, 2)
    assert_eq!(mpo.site_tensor(0).left_dim(), 1);
    assert_eq!(mpo.site_tensor(0).site_dim(), 4);
    assert_eq!(mpo.site_tensor(0).right_dim(), 2);

    // Middle tensors: shape (2, 4, 2)
    assert_eq!(mpo.site_tensor(1).left_dim(), 2);
    assert_eq!(mpo.site_tensor(1).site_dim(), 4);
    assert_eq!(mpo.site_tensor(1).right_dim(), 2);

    // Last tensor (site R-1 = LSB): cout to left, initial cin on right
    // Shape (2, 4, 1)
    assert_eq!(mpo.site_tensor(3).left_dim(), 2);
    assert_eq!(mpo.site_tensor(3).site_dim(), 4);
    assert_eq!(mpo.site_tensor(3).right_dim(), 1);
}

#[test]
fn test_flip_operator_creation() {
    let op = flip_operator(4, BoundaryCondition::Periodic);
    assert!(op.is_ok());
}

#[test]
fn test_flip_error_single_site() {
    let result = flip_operator(1, BoundaryCondition::Periodic);
    assert!(result.is_err());
}

#[test]
fn test_flip_error_zero_sites() {
    let result = flip_operator(0, BoundaryCondition::Periodic);
    assert!(result.is_err());
}

#[test]
fn test_flip_error_oversized_site_list() {
    let error = flip_operator(usize::MAX, BoundaryCondition::Periodic).unwrap_err();
    assert!(error.to_string().contains("site list"));
}
