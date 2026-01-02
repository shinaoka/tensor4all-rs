use tensor4all_core::index::DefaultIndex as Index;
use tensor4all_core::tagset::DefaultTagSet as TagSet;

#[test]
fn test_index_with_tags() {
    let mut idx = Index::new_dyn(8);
    assert_eq!(idx.tags().len(), 0);
    
    // Add tags
    idx.tags_mut().add_tag("site").unwrap();
    idx.tags_mut().add_tag("up").unwrap();
    
    assert_eq!(idx.tags().len(), 2);
    assert!(idx.tags().has_tag("site"));
    assert!(idx.tags().has_tag("up"));
}

#[test]
fn test_index_equality_by_id_only() {
    let mut idx1 = Index::new_dyn(8);
    idx1.tags_mut().add_tag("site").unwrap();
    
    let mut idx2 = Index::new_dyn(8);
    idx2.tags_mut().add_tag("bond").unwrap();
    
    // Different IDs, so not equal
    assert_ne!(idx1, idx2);
    
    // Same ID, should be equal even with different tags
    let id = idx1.id;
    let idx3 = Index {
        id,
        symm: idx1.symm,  // Automatically copied because Copy
        tags: TagSet::from_str("bond").unwrap(),
    };
    
    // Now they should be equal because IDs match (tags are ignored)
    assert_eq!(idx1, idx3);
}

#[test]
fn test_index_new_with_tags() {
    let tags = TagSet::from_str("site,up").unwrap();
    let idx = Index::new_dyn_with_tags(8, tags);
    
    assert_eq!(idx.tags().len(), 2);
    assert!(idx.tags().has_tag("site"));
    assert!(idx.tags().has_tag("up"));
}

#[test]
fn test_index_hash_by_id_only() {
    use std::collections::HashSet;
    
    let mut idx1 = Index::new_dyn(8);
    idx1.tags_mut().add_tag("site").unwrap();
    
    let mut idx2 = Index::new_dyn(8);
    idx2.tags_mut().add_tag("bond").unwrap();
    
    let mut set = HashSet::new();
    set.insert(idx1);  // Automatically copied because Copy
    set.insert(idx2);
    
    // Should have 2 entries (different IDs)
    assert_eq!(set.len(), 2);
    
    // Create another index with same ID as idx1 but different tags
    let idx3 = Index {
        id: idx1.id,
        symm: idx1.symm,  // Automatically copied because Copy
        tags: TagSet::from_str("bond").unwrap(),
    };
    
    // Should not add because ID matches idx1
    set.insert(idx3);
    assert_eq!(set.len(), 2);
}

#[test]
fn test_index_tags_mutability() {
    let mut idx = Index::new_dyn(8);
    
    // Initially empty
    assert_eq!(idx.tags().len(), 0);
    
    // Add tags via mutable reference
    idx.tags_mut().add_tag("site").unwrap();
    idx.tags_mut().add_tag("up").unwrap();
    
    // Verify tags were added
    assert_eq!(idx.tags().len(), 2);
    assert!(idx.tags().has_tag("site"));
    assert!(idx.tags().has_tag("up"));
    
    // Remove a tag
    idx.tags_mut().remove_tag("site");
    assert_eq!(idx.tags().len(), 1);
    assert!(!idx.tags().has_tag("site"));
    assert!(idx.tags().has_tag("up"));
}

