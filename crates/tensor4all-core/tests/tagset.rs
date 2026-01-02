use tensor4all_core::smallstring::{SmallString, SmallStringError};
use tensor4all_core::tagset::{DefaultTagSet, Tag, TagSet, TagSetError};

#[test]
fn test_smallstring_new() {
    let s = SmallString::<16>::new();
    assert_eq!(s.len(), 0);
    assert!(s.is_empty());
    assert_eq!(s.as_str(), "");
}

#[test]
fn test_smallstring_from_str() {
    let s = SmallString::<16>::from_str("hello").unwrap();
    assert_eq!(s.len(), 5);
    assert!(!s.is_empty());
    assert_eq!(s.as_str(), "hello");
}

#[test]
fn test_smallstring_too_long() {
    let result = SmallString::<5>::from_str("hello world");
    assert!(result.is_err());
    match result.unwrap_err() {
        SmallStringError::TooLong { actual, max } => {
            assert_eq!(actual, 11);
            assert_eq!(max, 5);
        }
    }
}

#[test]
fn test_smallstring_equality() {
    let s1 = SmallString::<16>::from_str("hello").unwrap();
    let s2 = SmallString::<16>::from_str("hello").unwrap();
    let s3 = SmallString::<16>::from_str("world").unwrap();
    
    assert_eq!(s1, s2);
    assert_ne!(s1, s3);
}

#[test]
fn test_smallstring_ordering() {
    let s1 = SmallString::<16>::from_str("apple").unwrap();
    let s2 = SmallString::<16>::from_str("banana").unwrap();
    
    assert!(s1 < s2);
    assert!(s2 > s1);
}

#[test]
fn test_smallstring_unicode() {
    let s = SmallString::<16>::from_str("αβγ").unwrap();
    assert_eq!(s.len(), 3);
    assert_eq!(s.as_str(), "αβγ");
}

#[test]
fn test_tagset_new() {
    let ts = TagSet::<4, 16>::new();
    assert_eq!(ts.len(), 0);
    assert_eq!(ts.capacity(), 4);
}

#[test]
fn test_tagset_from_str() {
    let ts = TagSet::<4, 16>::from_str("t1,t2,t3").unwrap();
    assert_eq!(ts.len(), 3);
    
    // Tags should be sorted
    assert_eq!(ts.get(0).unwrap().as_str(), "t1");
    assert_eq!(ts.get(1).unwrap().as_str(), "t2");
    assert_eq!(ts.get(2).unwrap().as_str(), "t3");
}

#[test]
fn test_tagset_sorted_order() {
    // Input order should not matter
    let ts = TagSet::<4, 16>::from_str("t3,t2,t1").unwrap();
    assert_eq!(ts.len(), 3);
    
    // Should be sorted
    assert_eq!(ts.get(0).unwrap().as_str(), "t1");
    assert_eq!(ts.get(1).unwrap().as_str(), "t2");
    assert_eq!(ts.get(2).unwrap().as_str(), "t3");
}

#[test]
fn test_tagset_whitespace_ignored() {
    let ts = TagSet::<4, 16>::from_str(" aaa , bb bb  , ccc    ").unwrap();
    assert_eq!(ts.len(), 3);
    
    // Whitespace should be removed
    assert!(ts.has_tag("aaa"));
    assert!(ts.has_tag("bbbb"));
    assert!(ts.has_tag("ccc"));
}

#[test]
fn test_tagset_has_tag() {
    let ts = TagSet::<4, 16>::from_str("t1,t2,t3").unwrap();
    assert!(ts.has_tag("t1"));
    assert!(ts.has_tag("t2"));
    assert!(ts.has_tag("t3"));
    assert!(!ts.has_tag("t4"));
}

#[test]
fn test_tagset_add_tag() {
    let mut ts = TagSet::<4, 16>::new();
    ts.add_tag("t2").unwrap();
    ts.add_tag("t1").unwrap();
    ts.add_tag("t3").unwrap();
    
    // Should be sorted
    assert_eq!(ts.get(0).unwrap().as_str(), "t1");
    assert_eq!(ts.get(1).unwrap().as_str(), "t2");
    assert_eq!(ts.get(2).unwrap().as_str(), "t3");
}

#[test]
fn test_tagset_remove_tag() {
    let mut ts = TagSet::<4, 16>::from_str("t1,t2,t3").unwrap();
    assert_eq!(ts.len(), 3);
    
    assert!(ts.remove_tag("t2"));
    assert_eq!(ts.len(), 2);
    assert!(!ts.has_tag("t2"));
    assert!(ts.has_tag("t1"));
    assert!(ts.has_tag("t3"));
}

#[test]
fn test_tagset_common_tags() {
    let ts1 = TagSet::<4, 16>::from_str("t1,t2,t3").unwrap();
    let ts2 = TagSet::<4, 16>::from_str("t2,t3,t4").unwrap();
    
    let common = ts1.common_tags(&ts2);
    assert_eq!(common.len(), 2);
    assert!(common.has_tag("t2"));
    assert!(common.has_tag("t3"));
    assert!(!common.has_tag("t1"));
    assert!(!common.has_tag("t4"));
}

#[test]
fn test_tagset_too_many_tags() {
    let mut ts = TagSet::<2, 16>::new();
    ts.add_tag("t1").unwrap();
    ts.add_tag("t2").unwrap();
    
    let result = ts.add_tag("t3");
    assert!(result.is_err());
    match result.unwrap_err() {
        TagSetError::TooManyTags { actual, max } => {
            assert_eq!(actual, 3);
            assert_eq!(max, 2);
        }
        _ => panic!("Expected TooManyTags error"),
    }
}

#[test]
fn test_default_types() {
    let tag: Tag = SmallString::<16>::from_str("test").unwrap();
    assert_eq!(tag.as_str(), "test");
    
    let ts: DefaultTagSet = TagSet::<4, 16>::from_str("t1,t2").unwrap();
    assert_eq!(ts.len(), 2);
}

