use super::*;

#[test]
fn test_boundary_condition_default() {
    assert_eq!(BoundaryCondition::default(), BoundaryCondition::Periodic);
}

#[test]
fn test_carry_direction_default() {
    assert_eq!(CarryDirection::default(), CarryDirection::LeftToRight);
}

#[test]
fn test_identity_mpo() {
    let mpo = identity_mpo(4).unwrap();
    assert_eq!(mpo.len(), 4);

    // Check that it's an identity operator
    for i in 0..4 {
        let t = mpo.site_tensor(i);
        assert_eq!(t.left_dim(), 1);
        assert_eq!(t.site_dim(), 4);
        assert_eq!(t.right_dim(), 1);
    }
}

#[test]
fn checked_allocation_len_rejects_complex_byte_limit_before_allocation() {
    let elements = isize::MAX as usize / std::mem::size_of::<Complex64>() + 1;
    let error = checked_allocation_len::<Complex64>(&[elements], "complex tensor").unwrap_err();
    assert!(error.to_string().contains("byte length"));
}

#[test]
fn checked_allocation_len_covers_product_and_zero_size_branches() {
    let error = checked_allocation_len::<u8>(&[usize::MAX, 2], "overflow").unwrap_err();
    assert!(error.to_string().contains("element count"));

    assert_eq!(
        checked_allocation_len::<u8>(&[0, usize::MAX], "zero").unwrap(),
        0
    );
    assert_eq!(checked_allocation_len::<u8>(&[], "empty").unwrap(), 1);
    assert_eq!(
        checked_allocation_len::<()>(&[usize::MAX], "zst").unwrap(),
        usize::MAX
    );
}

#[test]
fn identity_mpo_rejects_oversized_site_list() {
    let error = identity_mpo(usize::MAX).unwrap_err();
    assert!(error.to_string().contains("site list"));
}

#[test]
fn try_vec_with_capacity_reports_capacity_overflow_context() {
    let error = try_vec_with_capacity::<u8>("test site list", usize::MAX).unwrap_err();
    assert!(error.to_string().contains("test site list"));
}
