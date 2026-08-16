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
///
/// Measured, not predicted: the `SmallVec<[u64; 2]>` limb arm fits inside the
/// space `u128`'s alignment already reserved, so it did not move this bound.
/// If a future arm does move it, box that arm rather than raising the number.
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
fn widths_beyond_1024_bits_use_limbs_and_stay_injective() {
    let bits = 2048usize;
    let indexer = FlatIndexer::try_new(&vec![2usize; bits]).unwrap();
    assert_eq!(indexer.width_bits(), bits as u64);

    let zero = indexer.encode(&vec![0usize; bits]).unwrap();
    assert!(matches!(zero, IndexKey::Limbs(_)));

    let mut seen = std::collections::HashSet::new();
    seen.insert(zero);
    for position in [0usize, 63, 64, 65, 1023, 1024, 1025, 2047] {
        let mut idx = vec![0usize; bits];
        idx[position] = 1;
        let key = indexer.encode(&idx).unwrap();
        assert!(
            seen.insert(key),
            "collision with a single bit set at {position}"
        );
    }
}

#[test]
fn a_value_straddling_a_limb_boundary_round_trips() {
    // A radix-64 dimension needs 6 bits, so packing 40 of them puts several
    // values across a 64-bit limb boundary.
    let dims = vec![64usize; 40];
    let indexer = FlatIndexer::try_new(&dims).unwrap();
    let mut a = vec![0usize; 40];
    let mut b = vec![0usize; 40];
    a[10] = 63;
    b[10] = 62;
    assert_ne!(indexer.encode(&a).unwrap(), indexer.encode(&b).unwrap());
}

#[test]
fn every_single_bit_position_is_distinct_across_limbs() {
    let bits = 1500usize;
    let indexer = FlatIndexer::try_new(&vec![2usize; bits]).unwrap();
    let mut seen = std::collections::HashSet::new();
    for position in 0..bits {
        let mut idx = vec![0usize; bits];
        idx[position] = 1;
        assert!(
            seen.insert(indexer.encode(&idx).unwrap()),
            "collision at bit {position}"
        );
    }
    assert_eq!(seen.len(), bits);
}

#[test]
fn composition_matches_encoding_the_concatenated_multi_index() {
    let local = FlatIndexer::try_new(&[3, 2]).unwrap();
    let child = FlatIndexer::try_new(&[4, 5]).unwrap();
    let whole = FlatIndexer::try_new(&[3, 2, 4, 5]).unwrap();

    for a in 0..3 {
        for b in 0..2 {
            for c in 0..4 {
                for d in 0..5 {
                    let mut builder = KeyBuilder::with_capacity_bits(whole.width_bits()).unwrap();
                    builder
                        .push(&local.encode(&[a, b]).unwrap(), local.width_bits())
                        .unwrap();
                    builder
                        .push(&child.encode(&[c, d]).unwrap(), child.width_bits())
                        .unwrap();
                    assert_eq!(
                        builder.finish(),
                        whole.encode(&[a, b, c, d]).unwrap(),
                        "composition disagreed at {a},{b},{c},{d}"
                    );
                }
            }
        }
    }
}

#[test]
fn composition_is_injective_across_the_limb_boundary() {
    let child = FlatIndexer::try_new(&[2; 40]).unwrap();
    let zero = child.encode(&vec![0usize; 40]).unwrap();
    let one = {
        let mut v = vec![0usize; 40];
        v[0] = 1;
        child.encode(&v).unwrap()
    };

    let mut first = KeyBuilder::with_capacity_bits(80).unwrap();
    first.push(&zero, 40).unwrap();
    first.push(&one, 40).unwrap();

    let mut second = KeyBuilder::with_capacity_bits(80).unwrap();
    second.push(&one, 40).unwrap();
    second.push(&zero, 40).unwrap();

    assert_ne!(first.finish(), second.finish());
}

#[test]
fn composition_spans_the_fixed_to_dynamic_boundary() {
    let part = FlatIndexer::try_new(&[2; 600]).unwrap();
    let whole = FlatIndexer::try_new(&[2; 1200]).unwrap();
    let mut low = vec![0usize; 600];
    low[0] = 1;
    let mut high = vec![0usize; 600];
    high[599] = 1;

    let mut builder = KeyBuilder::with_capacity_bits(1200).unwrap();
    builder.push(&part.encode(&low).unwrap(), 600).unwrap();
    builder.push(&part.encode(&high).unwrap(), 600).unwrap();

    let mut expected = vec![0usize; 1200];
    expected[0] = 1;
    expected[1199] = 1;
    assert_eq!(builder.finish(), whole.encode(&expected).unwrap());
}

#[test]
fn pushing_past_the_declared_capacity_is_an_error() {
    let indexer = FlatIndexer::try_new(&[2, 2]).unwrap();
    let mut builder = KeyBuilder::with_capacity_bits(2).unwrap();
    builder.push(&indexer.encode(&[1, 1]).unwrap(), 2).unwrap();
    assert!(matches!(
        builder.push(&indexer.encode(&[1, 1]).unwrap(), 2),
        Err(IndexKeyError::WidthOverflow { .. })
    ));
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
