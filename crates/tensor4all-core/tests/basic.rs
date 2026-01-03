use tensor4all_core::index::DefaultIndex as Index;

#[test]
fn test_index_dyn() {
    let idx = Index::new_dyn(8);
    assert_eq!(idx.size(), 8);
    assert!(idx.id.0 > 0);
}

#[test]
fn test_index_with_custom_id() {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    struct StaticId;
    
    let idx = Index::new_with_size(StaticId, 16);
    assert_eq!(idx.size(), 16);
    assert_eq!(idx.id, StaticId);
}

