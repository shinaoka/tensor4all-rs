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
fn encoding_is_injective_over_a_small_space() {
    let dims = [3usize, 4, 2];
    let indexer = FlatIndexer::try_new(&dims).unwrap();
    assert_eq!(indexer.width_bits(), 2 + 2 + 1);

    let mut seen = std::collections::HashSet::new();
    for a in 0..dims[0] {
        for b in 0..dims[1] {
            for c in 0..dims[2] {
                let key = indexer.encode(&[a, b, c]).unwrap();
                assert!(seen.insert(key), "collision at {a},{b},{c}");
            }
        }
    }
    assert_eq!(seen.len(), dims[0] * dims[1] * dims[2]);
}

#[test]
fn encoding_rejects_bad_input_instead_of_wrapping() {
    let indexer = FlatIndexer::try_new(&[3, 4]).unwrap();
    assert!(matches!(
        indexer.encode(&[0]),
        Err(IndexKeyError::LengthMismatch {
            expected: 2,
            actual: 1
        })
    ));
    assert!(matches!(
        indexer.encode(&[3, 0]),
        Err(IndexKeyError::IndexOutOfRange {
            position: 0,
            value: 3,
            dim: 3
        })
    ));
    assert!(matches!(
        indexer.encode(&[0, 4]),
        Err(IndexKeyError::IndexOutOfRange {
            position: 1,
            value: 4,
            dim: 4
        })
    ));
}

#[test]
fn width_selects_u64_then_u128() {
    let narrow = FlatIndexer::try_new(&[2; 64]).unwrap();
    assert!(matches!(narrow.encode(&[0; 64]).unwrap(), IndexKey::U64(_)));
    let wide = FlatIndexer::try_new(&[2; 65]).unwrap();
    assert!(matches!(wide.encode(&[0; 65]).unwrap(), IndexKey::U128(_)));
}

#[test]
fn wide_arms_are_selected_by_width() {
    for (bits, want_u256, want_u512, want_u1024) in [
        (129usize, true, false, false),
        (257, false, true, false),
        (513, false, false, true),
    ] {
        let indexer = FlatIndexer::try_new(&vec![2usize; bits]).unwrap();
        let key = indexer.encode(&vec![0usize; bits]).unwrap();
        assert_eq!(matches!(key, IndexKey::U256(_)), want_u256, "{bits} bits");
        assert_eq!(matches!(key, IndexKey::U512(_)), want_u512, "{bits} bits");
        assert_eq!(matches!(key, IndexKey::U1024(_)), want_u1024, "{bits} bits");
    }
}

/// #628: "the exact layout should not inflate every fixed-width key merely
/// because the enum has a large inline variant".
///
/// 32 bytes is the floor while `U128(u128)` is inline: `u128` forces 16-byte
/// alignment, so the enum rounds up to 32 once a discriminant is added. The
/// bound therefore checks what it is meant to check — an inlined `U1024` would
/// make this 128 bytes on its own, and boxing the wide arms is what keeps a
/// `u64` key from paying for them.
#[test]
fn the_key_enum_stays_small() {
    assert!(
        std::mem::size_of::<IndexKey>() <= 32,
        "IndexKey is {} bytes; wide arms must be boxed",
        std::mem::size_of::<IndexKey>()
    );
}

#[test]
fn wide_encoding_is_injective_on_the_high_bits() {
    let bits = 300usize;
    let indexer = FlatIndexer::try_new(&vec![2usize; bits]).unwrap();
    let mut low = vec![0usize; bits];
    low[0] = 1;
    let mut high = vec![0usize; bits];
    high[bits - 1] = 1;
    assert_ne!(
        indexer.encode(&low).unwrap(),
        indexer.encode(&high).unwrap()
    );
}

#[test]
fn an_empty_index_space_encodes_to_zero() {
    let indexer = FlatIndexer::try_new(&[]).unwrap();
    assert_eq!(indexer.width_bits(), 0);
    assert!(indexer.is_empty());
    assert_eq!(indexer.encode(&[]).unwrap(), IndexKey::U64(0));
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
