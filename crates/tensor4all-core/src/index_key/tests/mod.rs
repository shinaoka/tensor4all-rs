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
    assert!(matches!(
        narrow.encode(&[0; 64]).unwrap().repr(),
        Repr::U64(_)
    ));
    let wide = FlatIndexer::try_new(&[2; 65]).unwrap();
    assert!(matches!(
        wide.encode(&[0; 65]).unwrap().repr(),
        Repr::U128(_)
    ));
}

/// The fixed-width ladder stops at 512 bits; 513 and above go to limbs.
///
/// A `U1024` arm was measured at 6.06 ns/dimension against 2.51 ns for the limb
/// path at the same width, so it was removed rather than kept as a nominal
/// "fast path".
#[test]
fn wide_arms_are_selected_by_width() {
    for (bits, want_u256, want_u512, want_limbs) in [
        (129usize, true, false, false),
        (257, false, true, false),
        (513, false, false, true),
        (1024, false, false, true),
    ] {
        let indexer = FlatIndexer::try_new(&vec![2usize; bits]).unwrap();
        let key = indexer.encode(&vec![0usize; bits]).unwrap();
        assert_eq!(
            matches!(key.repr(), Repr::U256(_)),
            want_u256,
            "{bits} bits"
        );
        assert_eq!(
            matches!(key.repr(), Repr::U512(_)),
            want_u512,
            "{bits} bits"
        );
        assert_eq!(
            matches!(key.repr(), Repr::Limbs(_)),
            want_limbs,
            "{bits} bits"
        );
    }
}

/// #628: "the exact layout should not inflate every fixed-width key merely
/// because the enum has a large inline variant".
///
/// 48 bytes is the floor for a key that carries its own width while `U128`
/// stays inline. `Repr` is 32 bytes — `u128` forces 16-byte alignment, so the
/// enum rounds up to 32 once a discriminant is added — and the `width_bits`
/// field rounds the struct up one further alignment step.
///
/// Carrying the width is what makes `KeyBuilder::push` unable to place a
/// sub-key under a width that disagrees with its contents, and what lets
/// `finish` pick the same arm `encode` would. Two ways back to 32 were measured
/// and rejected: narrowing `width_bits` to `u32` changes nothing, because the
/// padding is set by `Repr`'s alignment rather than the field's size; boxing
/// the `U128` arm reaches 40, but puts a heap allocation on the 65–128 bit
/// path, which is a common width.
///
/// The bound still checks what it is meant to check — an inlined `U512` would
/// make `Repr` 64 bytes on its own, and boxing the wide arms is what keeps a
/// `u64` key from paying for them. Measured, not predicted: the
/// `SmallVec<[u64; 2]>` limb arm fits inside the space `u128`'s alignment
/// already reserved, so it did not move this bound. If a future arm does move
/// it, box that arm rather than raising the number.
#[test]
fn the_key_enum_stays_small() {
    assert!(
        std::mem::size_of::<IndexKey>() <= 48,
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
fn widths_beyond_the_fixed_ladder_use_limbs_and_stay_injective() {
    let bits = 2048usize;
    let indexer = FlatIndexer::try_new(&vec![2usize; bits]).unwrap();
    assert_eq!(indexer.width_bits(), bits as u64);

    let zero = indexer.encode(&vec![0usize; bits]).unwrap();
    assert!(matches!(zero.repr(), Repr::Limbs(_)));

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
                    builder.push(&local.encode(&[a, b]).unwrap()).unwrap();
                    builder.push(&child.encode(&[c, d]).unwrap()).unwrap();
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
    first.push(&zero).unwrap();
    first.push(&one).unwrap();

    let mut second = KeyBuilder::with_capacity_bits(80).unwrap();
    second.push(&one).unwrap();
    second.push(&zero).unwrap();

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
    builder.push(&part.encode(&low).unwrap()).unwrap();
    builder.push(&part.encode(&high).unwrap()).unwrap();

    let mut expected = vec![0usize; 1200];
    expected[0] = 1;
    expected[1199] = 1;
    assert_eq!(builder.finish(), whole.encode(&expected).unwrap());
}

#[test]
fn pushing_past_the_declared_capacity_is_an_error() {
    let indexer = FlatIndexer::try_new(&[2, 2]).unwrap();
    let mut builder = KeyBuilder::with_capacity_bits(2).unwrap();
    builder.push(&indexer.encode(&[1, 1]).unwrap()).unwrap();
    assert!(matches!(
        builder.push(&indexer.encode(&[1, 1]).unwrap()),
        Err(IndexKeyError::WidthOverflow { .. })
    ));
}

#[test]
fn an_empty_index_space_encodes_to_zero() {
    let indexer = FlatIndexer::try_new(&[]).unwrap();
    assert_eq!(indexer.width_bits(), 0);
    assert!(indexer.is_empty());
    let empty = indexer.encode(&[]).unwrap();
    assert_eq!(empty.width_bits(), 0);
    assert!(matches!(empty.repr(), Repr::U64(0)));
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

/// A sub-key can only be appended under its own width, so it cannot overwrite
/// the field that follows it.
///
/// The earlier API took the width as a separate argument to `push`, which let a
/// caller declare a width narrower than the key's contents; the surplus high
/// bits then landed in the next field and two distinct multi-indices could
/// compose to the same key. The width now travels with the key, so the mistake
/// is unrepresentable rather than rejected.
#[test]
fn composition_cannot_overwrite_the_following_field() {
    let two_bit = FlatIndexer::try_new(&[4]).unwrap();
    let one_bit = FlatIndexer::try_new(&[2]).unwrap();
    assert_eq!(two_bit.width_bits(), 2);
    assert_eq!(one_bit.width_bits(), 1);

    // `0b11` occupies both of its bits; appending a 1-bit key after it must not
    // collide with appending a 1-bit key after `0b01`.
    let mut wide_then_one = KeyBuilder::with_capacity_bits(3).unwrap();
    wide_then_one.push(&two_bit.encode(&[3]).unwrap()).unwrap();
    wide_then_one.push(&one_bit.encode(&[0]).unwrap()).unwrap();

    let mut narrow_then_one = KeyBuilder::with_capacity_bits(3).unwrap();
    narrow_then_one
        .push(&two_bit.encode(&[1]).unwrap())
        .unwrap();
    narrow_then_one
        .push(&one_bit.encode(&[1]).unwrap())
        .unwrap();

    assert_ne!(wide_then_one.finish(), narrow_then_one.finish());
}

/// `finish` reports the width actually appended, so an over-declared capacity
/// does not change the composed key.
///
/// Selecting the arm from `capacity_bits` instead made a builder with capacity
/// 600 return the limb arm for a 100-bit key, while `FlatIndexer::encode` of
/// the same 100 bits returned the `U128` arm — two unequal keys for one value,
/// contradicting the documented interchangeability.
#[test]
fn an_over_declared_capacity_does_not_change_the_key() {
    let dims = vec![2usize; 100];
    let indexer = FlatIndexer::try_new(&dims).unwrap();
    assert_eq!(indexer.width_bits(), 100);
    let mut idx = vec![0usize; 100];
    idx[0] = 1;
    idx[99] = 1;
    let direct = indexer.encode(&idx).unwrap();

    for capacity in [100u64, 600, 4096] {
        let mut builder = KeyBuilder::with_capacity_bits(capacity).unwrap();
        builder.push(&direct).unwrap();
        let composed = builder.finish();
        assert_eq!(
            composed, direct,
            "capacity {capacity} must not change the composed key"
        );
        assert_eq!(composed.width_bits(), 100);
    }
}

/// An empty builder yields a zero-width key, not a key of its declared
/// capacity.
#[test]
fn an_unfilled_builder_reports_the_width_it_holds() {
    let builder = KeyBuilder::with_capacity_bits(4096).unwrap();
    let key = builder.finish();
    assert_eq!(key.width_bits(), 0);
    assert!(matches!(key.repr(), Repr::U64(0)));
}
