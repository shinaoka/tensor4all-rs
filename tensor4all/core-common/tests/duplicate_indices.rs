use tensor4all_core_common::index::DefaultIndex as Index;
use tensor4all_core_common::index_ops::{check_unique_indices, replaceinds, ReplaceIndsError};

#[test]
fn test_check_unique_indices_success() {
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(3);
    let k = Index::new_dyn(4);
    let indices = vec![i, j, k];
    assert!(check_unique_indices(&indices).is_ok());
}

#[test]
fn test_check_unique_indices_duplicate() {
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(3);
    let indices = vec![i.clone(), j.clone(), i.clone()];
    let result = check_unique_indices(&indices);
    assert!(result.is_err());
    if let Err(ReplaceIndsError::DuplicateIndices { first_pos, duplicate_pos }) = result {
        assert_eq!(first_pos, 0);
        assert_eq!(duplicate_pos, 2);
    } else {
        panic!("Expected DuplicateIndices error");
    }
}

#[test]
fn test_replaceinds_duplicate_input() {
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(3);
    let _k = Index::new_dyn(4);
    let new_j = Index::new_dyn(3);
    
    // Input has duplicates
    let indices = vec![i.clone(), j.clone(), j.clone()];
    let replacements = vec![(j.clone(), new_j.clone())];
    let result = replaceinds(indices, &replacements);
    assert!(result.is_err());
    if let Err(ReplaceIndsError::DuplicateIndices { .. }) = result {
        // Expected
    } else {
        panic!("Expected DuplicateIndices error");
    }
}

#[test]
fn test_replaceinds_duplicate_result() {
    let i = Index::new_dyn(3);
    let j = Index::new_dyn(3);
    let k = Index::new_dyn(3);
    // Create a new index to replace both i and k with the same index
    let new_idx = Index::new_dyn(3);
    
    // Replacement creates duplicates: both i and k are replaced with the same index
    let indices = vec![i.clone(), j.clone(), k.clone()];
    // Replace both i and k with new_idx, creating duplicates in the result
    let replacements = vec![(i.clone(), new_idx.clone()), (k.clone(), new_idx.clone())];
    let result = replaceinds(indices, &replacements);
    assert!(result.is_err(), "Expected error but got: {:?}", result);
    if let Err(ReplaceIndsError::DuplicateIndices { .. }) = result {
        // Expected
    } else {
        panic!("Expected DuplicateIndices error, got: {:?}", result);
    }
}

