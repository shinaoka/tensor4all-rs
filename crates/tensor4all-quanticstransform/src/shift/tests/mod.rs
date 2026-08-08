use super::*;
use crate::{flip_operator_multivar, phase_rotation_operator_multivar};
use tensor4all_simplett::AbstractTensorTrain;

#[test]
fn test_shift_mpo_structure() {
    let mpo = shift_mpo(4, 5, BoundaryCondition::Periodic).unwrap();
    assert_eq!(mpo.len(), 4);

    // Big-endian convention:
    // First tensor (site 0 = MSB): BC applied on left, carry_in from right
    // Shape (1, 4, 2)
    assert_eq!(mpo.site_tensor(0).left_dim(), 1);
    assert_eq!(mpo.site_tensor(0).site_dim(), 4);
    assert_eq!(mpo.site_tensor(0).right_dim(), 2);

    // Middle tensors: shape (2, 4, 2)
    assert_eq!(mpo.site_tensor(1).left_dim(), 2);
    assert_eq!(mpo.site_tensor(1).site_dim(), 4);
    assert_eq!(mpo.site_tensor(1).right_dim(), 2);

    // Last tensor (site R-1 = LSB): carry_out to left, no input on right
    // Shape (2, 4, 1)
    assert_eq!(mpo.site_tensor(3).left_dim(), 2);
    assert_eq!(mpo.site_tensor(3).site_dim(), 4);
    assert_eq!(mpo.site_tensor(3).right_dim(), 1);
}

#[test]
fn test_shift_zero() {
    // Shift by 0 should be identity-like
    let mpo = shift_mpo(4, 0, BoundaryCondition::Periodic).unwrap();
    assert_eq!(mpo.len(), 4);

    // For offset=0, all offset_bits are 0
    // Site 0 (MSB): receives carry from right
    // For identity, with carry_in=0: x_bit=0 -> out=0, x_bit=1 -> out=1
    let t0 = mpo.site_tensor(0);
    // s = out_bit * 2 + x_bit
    // For x_bit=0, offset_bit=0, carry_in=0: sum=0, out_bit=0 -> s=0
    assert_eq!(*t0.get3(0, 0, 0), Complex64::one());
    // For x_bit=1, offset_bit=0, carry_in=0: sum=1, out_bit=1 -> s=3
    assert_eq!(*t0.get3(0, 3, 0), Complex64::one());
}

#[test]
fn test_shift_operator_creation() {
    let op = shift_operator(4, 3, BoundaryCondition::Periodic);
    assert!(op.is_ok());
}

#[test]
fn test_shift_negative() {
    // Negative shift should work with modular arithmetic
    let mpo = shift_mpo(4, -1, BoundaryCondition::Periodic).unwrap();
    assert_eq!(mpo.len(), 4);
    // -1 mod 16 = 15 = 1111 in binary
}

#[test]
fn test_shift_single_site() {
    let mpo = shift_mpo(1, 1, BoundaryCondition::Periodic).unwrap();
    assert_eq!(mpo.len(), 1);
    assert_eq!(mpo.site_tensor(0).left_dim(), 1);
    assert_eq!(mpo.site_tensor(0).site_dim(), 4);
    assert_eq!(mpo.site_tensor(0).right_dim(), 1);
}

#[test]
fn test_shift_error_zero_sites() {
    let result = shift_operator(0, 0, BoundaryCondition::Periodic);
    assert!(result.is_err());
}

#[test]
fn test_shift_error_r_ge_64() {
    // r >= 64 would overflow 1i64 << r
    let result = shift_operator(64, 1, BoundaryCondition::Periodic);
    assert!(result.is_err());
    let msg = result.unwrap_err().to_string();
    assert!(msg.contains("at most 63"), "unexpected error: {msg}");
}

#[test]
fn test_shift_r_63_ok() {
    // r = 63 should be the max valid value
    let result = shift_operator(63, 1, BoundaryCondition::Periodic);
    assert!(result.is_ok());
}

#[test]
fn shift_multivar_rejects_usize_shift_width() {
    let error = shift_operator_multivar(1, 0, BoundaryCondition::Periodic, usize::BITS as usize, 0)
        .unwrap_err();
    assert!(error.to_string().contains("nvariables"));
}

#[test]
fn multivar_dims_reject_squared_site_dimension_overflow() {
    let nvariables = usize::BITS as usize / 2;
    assert!(checked_multivar_dims(nvariables)
        .unwrap_err()
        .to_string()
        .contains("site dimension"));
}

#[test]
fn sibling_multivar_operators_reject_usize_shift_width() {
    let nvariables = usize::BITS as usize;

    let flip_error =
        flip_operator_multivar(2, BoundaryCondition::Periodic, nvariables, 0).unwrap_err();
    assert!(flip_error.to_string().contains("nvariables"));

    let phase_error = phase_rotation_operator_multivar(1, 0.0, nvariables, 0).unwrap_err();
    assert!(phase_error.to_string().contains("nvariables"));
}
