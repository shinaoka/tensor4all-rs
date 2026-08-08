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
