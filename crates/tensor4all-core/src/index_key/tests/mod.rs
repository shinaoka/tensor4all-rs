use super::*;

#[test]
fn dimension_bits_is_ceil_log2() {
    assert_eq!(dimension_bits(1).unwrap(), 0);
    assert_eq!(dimension_bits(2).unwrap(), 1);
    assert_eq!(dimension_bits(3).unwrap(), 2);
    assert_eq!(dimension_bits(4).unwrap(), 2);
    assert_eq!(dimension_bits(5).unwrap(), 3);
    assert_eq!(dimension_bits(255).unwrap(), 8);
    assert_eq!(dimension_bits(256).unwrap(), 8);
    assert_eq!(dimension_bits(257).unwrap(), 9);
}

#[test]
fn dimension_zero_is_rejected() {
    assert!(matches!(
        dimension_bits(0),
        Err(IndexKeyError::ZeroDimension { position: 0 })
    ));
}

#[test]
fn total_bits_sums_and_reports_the_offending_position() {
    assert_eq!(total_bits(&[2, 2, 2]).unwrap(), 3);
    assert_eq!(total_bits(&[4, 3, 1]).unwrap(), 4);
    assert_eq!(total_bits(&[]).unwrap(), 0);
    assert!(matches!(
        total_bits(&[2, 0, 2]),
        Err(IndexKeyError::ZeroDimension { position: 1 })
    ));
}
