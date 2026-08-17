use super::*;

fn projector(pairs: impl IntoIterator<Item = (DynIndex, usize)>) -> Projector {
    Projector::from_pairs(pairs).unwrap()
}
use std::hash::{Hash, Hasher};
use tensor4all_core::index::Index;
use tensor4all_core::TagSet;

fn make_index(size: usize) -> DynIndex {
    Index::new_dyn(size)
}

#[test]
fn test_projector_new() {
    let p = Projector::new();
    assert!(p.is_empty());
    assert_eq!(p.len(), 0);
}

#[test]
fn test_projector_from_pairs() {
    let idx0 = make_index(2);
    let idx1 = make_index(3);
    let idx2 = make_index(4);

    let p = projector([(idx0.clone(), 1), (idx2.clone(), 3)]);
    assert_eq!(p.len(), 2);
    assert!(p.is_projected_at(&idx0));
    assert!(!p.is_projected_at(&idx1));
    assert!(p.is_projected_at(&idx2));
    assert_eq!(p.get(&idx0), Some(1));
    assert_eq!(p.get(&idx1), None);
    assert_eq!(p.get(&idx2), Some(3));
}

#[test]
fn projector_accepts_zero_and_last_coordinates() {
    let zero = make_index(2);
    let last = make_index(3);
    let projector = Projector::from_pairs([(zero.clone(), 0), (last.clone(), 2)]).unwrap();

    assert_eq!(projector.get(&zero), Some(0));
    assert_eq!(projector.get(&last), Some(2));
}

#[test]
fn projector_from_pairs_rejects_out_of_range_coordinate() {
    let index = make_index(2);
    let error = Projector::from_pairs([(index.clone(), 2)]).unwrap_err();

    assert!(matches!(
        error,
        PartitionedTTError::ProjectorCoordinateOutOfBounds {
            index: error_index,
            value: 2,
            dim: 2,
        } if error_index == index
    ));
}

#[test]
fn test_projector_intersection_compatible() {
    let idx0 = make_index(2);
    let idx1 = make_index(2);
    let idx2 = make_index(2);

    let a = projector([(idx0.clone(), 1), (idx1.clone(), 0)]);
    let b = projector([(idx1.clone(), 0), (idx2.clone(), 1)]);

    let merged = a.intersection(&b).unwrap();
    assert_eq!(merged.len(), 3);
    assert_eq!(merged.get(&idx0), Some(1));
    assert_eq!(merged.get(&idx1), Some(0));
    assert_eq!(merged.get(&idx2), Some(1));
}

#[test]
fn test_projector_intersection_conflict() {
    let idx0 = make_index(2);
    let idx1 = make_index(2);

    let a = projector([(idx0.clone(), 1), (idx1.clone(), 0)]);
    let b = projector([(idx1.clone(), 1)]); // Conflict at idx1

    assert!(a.intersection(&b).is_none());
}

#[test]
fn test_projector_common_restriction() {
    let idx0 = make_index(2);
    let idx1 = make_index(2);
    let idx2 = make_index(2);

    let a = projector([(idx0.clone(), 1), (idx1.clone(), 0)]);
    let b = projector([(idx1.clone(), 0), (idx2.clone(), 1)]);

    // Only idx1 is in both with the same value
    let common = a.common_restriction(&b);
    assert_eq!(common.len(), 1);
    assert!(!common.is_projected_at(&idx0));
    assert!(common.is_projected_at(&idx1));
    assert_eq!(common.get(&idx1), Some(0));
    assert!(!common.is_projected_at(&idx2));
}

#[test]
fn test_projector_is_compatible_with() {
    let idx0 = make_index(2);
    let idx1 = make_index(2);

    let a = projector([(idx0.clone(), 1)]);
    let b = projector([(idx1.clone(), 0)]);
    let c = projector([(idx0.clone(), 0)]); // Different value at same index

    assert!(a.is_compatible_with(&b)); // No common indices, compatible
    assert!(!a.is_compatible_with(&c)); // Same index, different values
}

#[test]
fn test_projector_is_subset_of() {
    let idx0 = make_index(2);
    let idx1 = make_index(2);
    let idx2 = make_index(2);

    let a = projector([(idx0.clone(), 1), (idx1.clone(), 0), (idx2.clone(), 1)]);
    let b = projector([(idx0.clone(), 1), (idx1.clone(), 0)]);
    let c = projector([(idx0.clone(), 1)]);

    assert!(a.is_subset_of(&b)); // a projects more indices
    assert!(a.is_subset_of(&c));
    assert!(b.is_subset_of(&c));
    assert!(!b.is_subset_of(&a));
    assert!(!c.is_subset_of(&a));
}

#[test]
fn test_projector_are_disjoint() {
    let idx0 = make_index(2);

    // Disjoint projectors: different values at same index
    let p1 = projector([(idx0.clone(), 0)]);
    let p2 = projector([(idx0.clone(), 1)]);

    assert!(Projector::are_disjoint(&[p1.clone(), p2.clone()]));

    // Non-disjoint: same projection
    let p3 = projector([(idx0.clone(), 0)]);
    assert!(!Projector::are_disjoint(&[p1, p3]));
}

#[test]
fn test_projector_partial_ord() {
    let idx0 = make_index(2);
    let idx1 = make_index(2);

    let a = projector([(idx0.clone(), 1), (idx1.clone(), 0)]);
    let b = projector([(idx0.clone(), 1)]);
    let c = projector([(idx0.clone(), 0)]); // Incompatible with a and b

    assert!(a < b);
    assert!(b > a);
    assert_eq!(a.partial_cmp(&c), None);
    assert_eq!(b.partial_cmp(&c), None);
}

#[test]
fn test_projector_iteration() {
    let idx0 = make_index(2);
    let idx1 = make_index(3);

    let p = projector([(idx0.clone(), 1), (idx1.clone(), 2)]);

    let pairs: Vec<_> = p.into_iter().collect();
    assert_eq!(pairs.len(), 2);
}

#[test]
fn test_projector_equality_and_hash() {
    use std::collections::HashSet;

    let idx0 = make_index(2);
    let idx1 = make_index(2);

    let a = projector([(idx0.clone(), 1), (idx1.clone(), 0)]);
    let b = projector([(idx1.clone(), 0), (idx0.clone(), 1)]); // Same content
    let c = projector([(idx0.clone(), 1)]);

    assert_eq!(a, b);
    assert_ne!(a, c);

    let mut set = HashSet::new();
    set.insert(a.clone());
    assert!(set.contains(&b));
    assert!(!set.contains(&c));
}

fn hash_projector(projector: &Projector) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    projector.hash(&mut hasher);
    hasher.finish()
}

#[test]
fn projector_hash_uses_canonical_full_identity_order() {
    let base = make_index(2);
    let pairs: Vec<_> = (0..32)
        .map(|value| {
            let tags = TagSet::from_str(&format!("Tag{value:02}")).unwrap();
            (Index::new_with_tags(base.id, 2, tags), value % 2)
        })
        .collect();
    let mut reversed = pairs.clone();
    reversed.reverse();

    let first = projector(pairs);
    let second = projector(reversed);

    assert_eq!(first, second);
    assert_eq!(hash_projector(&first), hash_projector(&second));
}

#[test]
fn projector_multi_tag_identity_has_canonical_hash_and_order() {
    let base = make_index(2);
    let first_index = Index::new_with_tags(
        base.id,
        base.dim,
        TagSet::from_str("Site,Auxiliary").unwrap(),
    );
    let second_index = Index::new_with_tags(
        base.id,
        base.dim,
        TagSet::from_str("Auxiliary,Site").unwrap(),
    );
    let first = projector([(first_index, 1)]);
    let second = projector([(second_index, 1)]);

    assert_eq!(first, second);
    assert_eq!(hash_projector(&first), hash_projector(&second));
    assert_eq!(first.canonical_cmp(&second), std::cmp::Ordering::Equal);
}

#[test]
fn projector_insert_rejects_out_of_range_coordinate_without_mutating() {
    let index = make_index(2);
    let mut projector = projector([(index.clone(), 0)]);
    let before = projector.clone();

    for value in [index.dim, index.dim + 1] {
        let error = projector.insert(index.clone(), value).unwrap_err();
        assert!(matches!(
            error,
            PartitionedTTError::ProjectorCoordinateOutOfBounds {
                index: error_index,
                value: error_value,
                dim,
            } if error_index == index && error_value == value && dim == index.dim
        ));
        assert_eq!(projector, before);
    }
}

#[test]
fn projector_duplicate_identity_replaces_key_and_value_together() {
    let index = make_index(2);
    let replacement = Index::new_with_tags(index.id, 5, index.tags.clone());
    let mut projector = projector([(index, 1)]);

    projector.insert(replacement.clone(), 3).unwrap();

    let (stored_index, stored_value) = projector.iter().next().unwrap();
    assert_eq!(stored_index, &replacement);
    assert_eq!(*stored_value, 3);
}

#[test]
fn test_projector_filter_indices() {
    let idx0 = make_index(2);
    let idx1 = make_index(2);
    let idx2 = make_index(2);

    let p = projector([(idx0.clone(), 1), (idx1.clone(), 0), (idx2.clone(), 1)]);

    let filtered = p.filter_indices(&[idx0.clone(), idx2.clone()]);
    assert_eq!(filtered.len(), 2);
    assert!(filtered.is_projected_at(&idx0));
    assert!(!filtered.is_projected_at(&idx1));
    assert!(filtered.is_projected_at(&idx2));
}
