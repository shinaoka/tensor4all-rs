use super::*;

#[test]
fn test_tagset_to_string() {
    let tags = TagSet::from_str("Site,n=1").unwrap();
    let s = tagset_to_string(&tags);
    // Tags are sorted, so order may differ
    assert!(s.contains("Site"));
    assert!(s.contains("n=1"));
}
