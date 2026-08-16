use super::*;

#[test]
fn test_indexset_basic() {
    let mut set: IndexSet<Vec<usize>> = IndexSet::new();
    set.push(vec![1, 2, 3]);
    set.push(vec![4, 5, 6]);

    assert_eq!(set.len(), 2);
    assert_eq!(set.get(0), Some(&vec![1, 2, 3]));
    assert_eq!(set.pos(&vec![1, 2, 3]), Some(0));
    assert_eq!(set.pos(&vec![4, 5, 6]), Some(1));
    assert_eq!(set.pos(&vec![7, 8, 9]), None);
}

#[test]
fn test_indexset_from_vec() {
    let set = IndexSet::from_vec(vec![vec![1], vec![2], vec![3]]);
    assert_eq!(set.len(), 3);
    assert_eq!(set[0], vec![1]);
    assert_eq!(set[2], vec![3]);
}

#[test]
fn test_indexset_contains() {
    let set = IndexSet::from_vec(vec![1, 2, 3]);
    assert!(set.contains(&1));
    assert!(set.contains(&3));
    assert!(!set.contains(&4));
}

#[test]
fn test_indexset_iter() {
    let set = IndexSet::from_vec(vec![10, 20, 30]);
    let collected: Vec<_> = set.iter().copied().collect();
    assert_eq!(collected, vec![10, 20, 30]);
}

#[test]
fn test_indexset_push_duplicate_is_noop() {
    let mut set: IndexSet<i32> = IndexSet::new();
    set.push(1);
    set.push(2);
    set.push(1); // duplicate - should be no-op

    assert_eq!(set.len(), 2);
    assert_eq!(set.pos(&1), Some(0)); // original index preserved
    assert_eq!(set.pos(&2), Some(1));
    assert_eq!(set.get(0), Some(&1));
    assert_eq!(set.get(1), Some(&2));
}

#[test]
fn test_indexset_from_vec_with_duplicates() {
    let set = IndexSet::from_vec(vec![1, 2, 3, 2, 1]);

    assert_eq!(set.len(), 3); // only unique values
    assert_eq!(set.pos(&1), Some(0)); // first occurrence index
    assert_eq!(set.pos(&2), Some(1));
    assert_eq!(set.pos(&3), Some(2));
    assert_eq!(set.get(0), Some(&1));
    assert_eq!(set.get(1), Some(&2));
    assert_eq!(set.get(2), Some(&3));
}

#[test]
fn test_indexset_default() {
    let set: IndexSet<i32> = IndexSet::default();
    assert!(set.is_empty());
    assert_eq!(set.len(), 0);
}

#[test]
fn test_indexset_positions() {
    let set = IndexSet::from_vec(vec![10, 20, 30, 40]);
    assert_eq!(set.positions(&[20, 40]), Some(vec![1, 3]));
    assert_eq!(set.positions(&[10, 99]), None); // 99 not found
}

#[test]
fn test_indexset_values() {
    let set = IndexSet::from_vec(vec![5, 10, 15]);
    assert_eq!(set.values(), &[5, 10, 15]);
}

#[test]
fn test_indexset_into_iter_owned() {
    let set = IndexSet::from_vec(vec![1, 2, 3]);
    let collected: Vec<_> = set.into_iter().collect();
    assert_eq!(collected, vec![1, 2, 3]);
}

#[test]
fn test_indexset_into_iter_ref() {
    let set = IndexSet::from_vec(vec![1, 2, 3]);
    let collected: Vec<_> = (&set).into_iter().copied().collect();
    assert_eq!(collected, vec![1, 2, 3]);
}
