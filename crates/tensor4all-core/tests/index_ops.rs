use tensor4all_core::index::DefaultIndex as Index;
use tensor4all_core::index_ops::{
    sim, sim_owned, replaceinds, replaceinds_in_place, ReplaceIndsError, unique_inds,
    noncommon_inds, union_inds, hasind, hasinds, hascommoninds, common_inds,
};

#[test]
fn test_sim_preserves_symm_and_tags() {
    let mut idx = Index::new_dyn(8);
    idx.tags_mut().add_tag("site").unwrap();
    idx.tags_mut().add_tag("up").unwrap();

    let similar = sim(&idx);

    // Same size and tags, but different ID
    assert_eq!(similar.size(), idx.size());
    assert_eq!(similar.tags().len(), idx.tags().len());
    assert!(similar.tags().has_tag("site"));
    assert!(similar.tags().has_tag("up"));
    assert_ne!(similar.id, idx.id);
}

#[test]
fn test_sim_owned_consumes_input() {
    let idx = Index::new_dyn(8);
    let original_id = idx.id;

    let similar = sim_owned(idx); // idx is consumed

    // Same size, but different ID
    assert_eq!(similar.size(), 8);
    assert_ne!(similar.id, original_id);
}

#[test]
fn test_replaceinds_success() {
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(3);
    let k = Index::new_dyn(4);
    let new_j = Index::new_dyn(3); // Same size as j

    let indices = vec![i.clone(), j.clone(), k.clone()];
    let replacements = vec![(j.clone(), new_j.clone())];

    let replaced = replaceinds(indices, &replacements).unwrap();
    assert_eq!(replaced.len(), 3);
    assert_eq!(replaced[0].id, i.id); // i unchanged
    assert_eq!(replaced[1].id, new_j.id); // j replaced
    assert_eq!(replaced[2].id, k.id); // k unchanged
}

#[test]
fn test_replaceinds_space_mismatch() {
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(3);
    let wrong_size = Index::new_dyn(5); // Different size

    let indices = vec![i.clone(), j.clone()];
    let replacements = vec![(j.clone(), wrong_size.clone())];

    let result = replaceinds(indices, &replacements);
    assert!(result.is_err());
    match result.unwrap_err() {
        ReplaceIndsError::SpaceMismatch { from_dim, to_dim } => {
            assert_eq!(from_dim, 3);
            assert_eq!(to_dim, 5);
        }
        ReplaceIndsError::DuplicateIndices { .. } => {
            panic!("Expected SpaceMismatch error, got DuplicateIndices");
        }
    }
}

#[test]
fn test_replaceinds_in_place_success() {
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(3);
    let k = Index::new_dyn(4);
    let new_j = Index::new_dyn(3);

    let mut indices = vec![i.clone(), j.clone(), k.clone()];
    let replacements = vec![(j.clone(), new_j.clone())];

    replaceinds_in_place(&mut indices, &replacements).unwrap();
    assert_eq!(indices[0].id, i.id);
    assert_eq!(indices[1].id, new_j.id);
    assert_eq!(indices[2].id, k.id);
}

#[test]
fn test_replaceinds_in_place_space_mismatch() {
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(3);
    let wrong_size = Index::new_dyn(5);

    let mut indices = vec![i.clone(), j.clone()];
    let replacements = vec![(j.clone(), wrong_size.clone())];

    let result = replaceinds_in_place(&mut indices, &replacements);
    assert!(result.is_err());
}

#[test]
fn test_unique_inds() {
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(3);
    let k = Index::new_dyn(4);

    let indices_a = vec![i.clone(), j.clone()];
    let indices_b = vec![j.clone(), k.clone()];

    let unique = unique_inds(&indices_a, &indices_b);
    assert_eq!(unique.len(), 1);
    assert_eq!(unique[0].id, i.id);
}

#[test]
fn test_noncommon_inds() {
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(3);
    let k = Index::new_dyn(4);

    let indices_a = vec![i.clone(), j.clone()];
    let indices_b = vec![j.clone(), k.clone()];

    let noncommon = noncommon_inds(&indices_a, &indices_b);
    assert_eq!(noncommon.len(), 2); // i and k
    let noncommon_ids: std::collections::HashSet<_> =
        noncommon.iter().map(|idx| idx.id).collect();
    assert!(noncommon_ids.contains(&i.id));
    assert!(noncommon_ids.contains(&k.id));
    assert!(!noncommon_ids.contains(&j.id));
}

#[test]
fn test_union_inds() {
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(3);
    let k = Index::new_dyn(4);

    let indices_a = vec![i.clone(), j.clone()];
    let indices_b = vec![j.clone(), k.clone()];

    let union = union_inds(&indices_a, &indices_b);
    assert_eq!(union.len(), 3); // i, j, k (j appears once)
    let union_ids: std::collections::HashSet<_> = union.iter().map(|idx| idx.id).collect();
    assert!(union_ids.contains(&i.id));
    assert!(union_ids.contains(&j.id));
    assert!(union_ids.contains(&k.id));
}

#[test]
fn test_hasind() {
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(3);
    let k = Index::new_dyn(4);

    let indices = vec![i.clone(), j.clone()];

    assert!(hasind(&indices, &i));
    assert!(hasind(&indices, &j));
    assert!(!hasind(&indices, &k));
}

#[test]
fn test_hasinds() {
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(3);
    let k = Index::new_dyn(4);
    let l = Index::new_dyn(5);

    let indices = vec![i.clone(), j.clone(), k.clone()];

    assert!(hasinds(&indices, &[i.clone(), j.clone()]));
    assert!(hasinds(&indices, &[i.clone(), j.clone(), k.clone()]));
    assert!(!hasinds(&indices, &[i.clone(), l.clone()]));
    assert!(!hasinds(&indices, &[l.clone()]));
}

#[test]
fn test_hascommoninds() {
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(3);
    let k = Index::new_dyn(4);
    let l = Index::new_dyn(5);

    let indices_a = vec![i.clone(), j.clone()];
    let indices_b = vec![j.clone(), k.clone()];
    let indices_c = vec![k.clone(), l.clone()];

    assert!(hascommoninds(&indices_a, &indices_b)); // j is common
    assert!(!hascommoninds(&indices_a, &indices_c)); // no common
}

#[test]
fn test_common_inds_integration() {
    // Test that common_inds works with the new functions
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(3);
    let k = Index::new_dyn(4);

    let indices_a = vec![i.clone(), j.clone()];
    let indices_b = vec![j.clone(), k.clone()];

    let common = common_inds(&indices_a, &indices_b);
    assert_eq!(common.len(), 1);
    assert_eq!(common[0].id, j.id);

    // Verify hascommoninds matches
    assert!(hascommoninds(&indices_a, &indices_b));
}

#[test]
fn test_replaceinds_multiple_replacements() {
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(3);
    let k = Index::new_dyn(4);
    let new_i = Index::new_dyn(2);
    let new_k = Index::new_dyn(4);

    let indices = vec![i.clone(), j.clone(), k.clone()];
    let replacements = vec![
        (i.clone(), new_i.clone()),
        (k.clone(), new_k.clone()),
    ];

    let replaced = replaceinds(indices, &replacements).unwrap();
    assert_eq!(replaced.len(), 3);
    assert_eq!(replaced[0].id, new_i.id);
    assert_eq!(replaced[1].id, j.id); // unchanged
    assert_eq!(replaced[2].id, new_k.id);
}

#[test]
fn test_replaceinds_no_match() {
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(3);
    let k = Index::new_dyn(4);
    let other = Index::new_dyn(4); // Same size as k for valid replacement

    let indices = vec![i.clone(), j.clone()];
    let replacements = vec![(k.clone(), other.clone())]; // k not in indices, but valid replacement

    let replaced = replaceinds(indices, &replacements).unwrap();
    // No replacements should occur since k is not in indices
    assert_eq!(replaced.len(), 2);
    assert_eq!(replaced[0].id, i.id);
    assert_eq!(replaced[1].id, j.id);
}

