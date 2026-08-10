use super::*;
use tensor4all_core::DynIndex;

#[test]
fn test_basic_operations() {
    let mut network: LinkIndexNetwork<DynIndex> = LinkIndexNetwork::new();

    let idx1 = DynIndex::new_dyn(4);
    let idx2 = DynIndex::new_dyn(4);
    let edge1 = EdgeIndex::new(0);
    let edge2 = EdgeIndex::new(1);

    network.insert(edge1, &idx1);
    network.insert(edge2, &idx2);

    assert!(network.contains(&idx1));
    assert!(network.contains(&idx2));
    assert_eq!(network.find_edge(&idx1), Some(edge1));
    assert_eq!(network.find_edge(&idx2), Some(edge2));
    assert_eq!(network.len(), 2);
}

#[test]
fn test_replace_index() {
    let mut network: LinkIndexNetwork<DynIndex> = LinkIndexNetwork::new();

    let old_idx = DynIndex::new_dyn(4);
    let new_idx = DynIndex::new_dyn(4);
    let edge = EdgeIndex::new(0);

    network.insert(edge, &old_idx);
    assert!(network.contains(&old_idx));

    network.replace_index(&old_idx, &new_idx, edge).unwrap();

    assert!(!network.contains(&old_idx));
    assert!(network.contains(&new_idx));
    assert_eq!(network.find_edge(&new_idx), Some(edge));
}

#[test]
fn test_remove() {
    let mut network: LinkIndexNetwork<DynIndex> = LinkIndexNetwork::new();

    let idx = DynIndex::new_dyn(4);
    let edge = EdgeIndex::new(0);

    network.insert(edge, &idx);
    assert_eq!(network.remove(&idx), Some(edge));
    assert!(!network.contains(&idx));
}

#[test]
fn test_with_capacity() {
    let network: LinkIndexNetwork<DynIndex> = LinkIndexNetwork::with_capacity(10);
    assert!(network.is_empty());
    assert_eq!(network.len(), 0);
}

#[test]
fn test_clear() {
    let mut network: LinkIndexNetwork<DynIndex> = LinkIndexNetwork::new();

    let idx1 = DynIndex::new_dyn(4);
    let idx2 = DynIndex::new_dyn(4);
    let edge1 = EdgeIndex::new(0);
    let edge2 = EdgeIndex::new(1);

    network.insert(edge1, &idx1);
    network.insert(edge2, &idx2);
    assert_eq!(network.len(), 2);

    network.clear();
    assert!(network.is_empty());
    assert_eq!(network.len(), 0);
    assert!(!network.contains(&idx1));
    assert!(!network.contains(&idx2));
}

#[test]
fn test_iter() {
    let mut network: LinkIndexNetwork<DynIndex> = LinkIndexNetwork::new();

    let idx1 = DynIndex::new_dyn(4);
    let idx2 = DynIndex::new_dyn(4);
    let edge1 = EdgeIndex::new(0);
    let edge2 = EdgeIndex::new(1);

    network.insert(edge1, &idx1);
    network.insert(edge2, &idx2);

    let items: Vec<_> = network.iter().collect();
    assert_eq!(items.len(), 2);

    // Verify both entries are present
    let has_idx1 = items.iter().any(|(idx, &e)| *idx == &idx1 && e == edge1);
    let has_idx2 = items.iter().any(|(idx, &e)| *idx == &idx2 && e == edge2);
    assert!(has_idx1);
    assert!(has_idx2);
}

#[test]
fn test_default() {
    let network: LinkIndexNetwork<DynIndex> = LinkIndexNetwork::default();
    assert!(network.is_empty());
    assert_eq!(network.len(), 0);
}

#[test]
fn test_replace_index_edge_mismatch() {
    let mut network: LinkIndexNetwork<DynIndex> = LinkIndexNetwork::new();

    let old_idx = DynIndex::new_dyn(4);
    let new_idx = DynIndex::new_dyn(4);
    let edge_correct = EdgeIndex::new(0);
    let edge_wrong = EdgeIndex::new(1);

    network.insert(edge_correct, &old_idx);

    // Try to replace with wrong edge
    let result = network.replace_index(&old_idx, &new_idx, edge_wrong);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("Edge mismatch"));

    // Old index should still be there (restored on error)
    assert!(network.contains(&old_idx));
    assert_eq!(network.find_edge(&old_idx), Some(edge_correct));
}

#[test]
fn test_replace_index_not_found() {
    let mut network: LinkIndexNetwork<DynIndex> = LinkIndexNetwork::new();

    let old_idx = DynIndex::new_dyn(4);
    let new_idx = DynIndex::new_dyn(4);
    let edge = EdgeIndex::new(0);

    // Try to replace an index that doesn't exist
    let result = network.replace_index(&old_idx, &new_idx, edge);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("not found"));
}
