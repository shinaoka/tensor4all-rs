use tensor4all_core::smallstring::{SmallString, SmallStringError};
use tensor4all_core::tagset::{DefaultTagSet, Tag, TagSet, TagSetError, TagSetLike};

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

#[test]
fn test_tagset_like_trait() {
    // Test that TagSet implements TagSetLike
    let mut ts1: TagSet<4, 16> = <TagSet<4, 16> as TagSetLike>::from_str("t1,t2,t3").unwrap();
    let ts2: TagSet<4, 16> = <TagSet<4, 16> as TagSetLike>::from_str("t2,t3").unwrap();
    
    // Test trait methods
    assert_eq!(TagSetLike::len(&ts1), 3);
    assert!(!TagSetLike::is_empty(&ts1));
    assert_eq!(TagSetLike::capacity(&ts1), 4);
    
    // Test get() returns String
    assert_eq!(TagSetLike::get(&ts1, 0), Some("t1".to_string()));
    assert_eq!(TagSetLike::get(&ts1, 1), Some("t2".to_string()));
    assert_eq!(TagSetLike::get(&ts1, 2), Some("t3".to_string()));
    assert_eq!(TagSetLike::get(&ts1, 3), None);
    
    // Test iteration
    let tags: Vec<String> = TagSetLike::iter(&ts1).collect();
    assert_eq!(tags, vec!["t1".to_string(), "t2".to_string(), "t3".to_string()]);
    
    // Test has_tag
    assert!(TagSetLike::has_tag(&ts1, "t1"));
    assert!(TagSetLike::has_tag(&ts1, "t2"));
    assert!(!TagSetLike::has_tag(&ts1, "t4"));
    
    // Test has_tags with same type
    assert!(TagSetLike::has_tags(&ts1, &ts2));
    
    // Test add_tag via trait
    TagSetLike::add_tag(&mut ts1, "t4").unwrap();
    assert_eq!(TagSetLike::len(&ts1), 4);
    assert!(TagSetLike::has_tag(&ts1, "t4"));
    
    // Test remove_tag via trait
    assert!(TagSetLike::remove_tag(&mut ts1, "t2"));
    assert_eq!(TagSetLike::len(&ts1), 3);
    assert!(!TagSetLike::has_tag(&ts1, "t2"));
    
    // Test common_tags via trait
    // ts1 now has: t1, t3, t4 (after removing t2)
    let ts3: TagSet<4, 16> = <TagSet<4, 16> as TagSetLike>::from_str("t1,t3,t4").unwrap();
    let common = TagSetLike::common_tags(&ts1, &ts3);
    assert_eq!(TagSetLike::len(&common), 3);
    assert!(TagSetLike::has_tag(&common, "t1"));
    assert!(TagSetLike::has_tag(&common, "t3"));
    assert!(TagSetLike::has_tag(&common, "t4"));
    assert!(!TagSetLike::has_tag(&common, "t2"));
}

