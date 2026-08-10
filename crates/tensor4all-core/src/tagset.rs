use crate::smallstring::{SmallChar, SmallString, SmallStringError};
use std::cmp::Ordering;

/// Trait for tag set implementations.
///
/// This trait allows different tag set implementations to be used interchangeably,
/// enabling flexibility in storage strategies (e.g., fixed-size arrays, vectors, hash sets).
///
/// # Design Principles
///
/// - **String-based interface**: All tag operations use `&str` to hide implementation details
/// - **Sorted order**: Tags are maintained in sorted order (similar to ITensors.jl)
/// - **Error handling**: Operations that can fail return `Result<(), TagSetError>`
/// - **Iteration**: Provides iteration over tags as `&str` slices
pub trait TagSetLike: Default + Clone + PartialEq + Eq {
    /// Get the number of tags.
    fn len(&self) -> usize;

    /// Check if the tag set is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get the maximum capacity (if applicable).
    ///
    /// For unbounded implementations, this may return `usize::MAX` or a reasonable upper bound.
    fn capacity(&self) -> usize;

    /// Get a tag at the given index as a string.
    ///
    /// Returns `None` if the index is out of bounds.
    fn get(&self, index: usize) -> Option<String>;

    /// Iterate over tags as strings.
    fn iter(&self) -> TagSetIterator<'_>;

    /// Check if a tag is present.
    fn has_tag(&self, tag: &str) -> bool;

    /// Check if all tags in another tag set are present.
    ///
    /// This allows comparing different tag set implementations.
    fn has_tags<T: TagSetLike>(&self, other: &T) -> bool {
        for i in 0..other.len() {
            if let Some(tag) = other.get(i) {
                if !self.has_tag(&tag) {
                    return false;
                }
            }
        }
        true
    }

    /// Add a tag (maintains sorted order).
    ///
    /// Returns an error if the tag cannot be added (e.g., capacity exceeded, invalid tag).
    /// # Errors
    ///
    /// Returns an error when the tag is invalid (an invalid-input failure).
    ///
    fn add_tag(&mut self, tag: &str) -> Result<(), TagSetError>;

    /// Remove a tag.
    ///
    /// Returns `true` if the tag was present and removed, `false` otherwise.
    fn remove_tag(&mut self, tag: &str) -> bool;

    /// Get common tags between this tag set and another.
    ///
    /// Returns a new tag set containing only tags present in both.
    fn common_tags<T: TagSetLike>(&self, other: &T) -> Self {
        let mut result = Self::default();
        for i in 0..self.len() {
            if let Some(tag) = self.get(i) {
                if other.has_tag(&tag) {
                    result.add_tag(&tag).ok();
                }
            }
        }
        result
    }

    /// Create a tag set from a comma-separated string.
    ///
    /// ASCII spaces are ignored (matching ITensors.jl).
    /// Tags are automatically sorted.
    /// # Errors
    ///
    /// Returns an error when the string cannot be parsed as a tag or tag set (an
    /// /// invalid-input failure).
    ///
    fn from_str(s: &str) -> Result<Self, TagSetError> {
        let mut tagset = Self::default();

        // Parse comma-separated tags, ignoring ASCII spaces like ITensors.jl.
        for tag in s.split(',') {
            let trimmed: String = tag.chars().filter(|c| *c != ' ').collect();
            if !trimmed.is_empty() {
                tagset.add_tag(&trimmed)?;
            }
        }

        Ok(tagset)
    }
}

/// Iterator over tags in a tag set.
///
/// This is a type alias to allow different implementations to provide their own iterator types.
/// The iterator yields `String` values representing each tag.
pub type TagSetIterator<'a> = Box<dyn Iterator<Item = String> + 'a>;

/// A set of tags with fixed capacity, stored in sorted order.
///
/// Tags are always maintained in sorted order, regardless of insertion order,
/// similar to ITensors.jl's `TagSet`.
///
/// # Type Parameters
/// - `MAX_TAGS`: Maximum number of tags (default: 4, matching ITensors.jl)
/// - `MAX_TAG_LEN`: Maximum length of each tag (default: 16, matching ITensors.jl)
/// - `C`: Character type for tags (default: `u16` for ITensors.jl compatibility)
///
/// # Example
/// ```
/// use tensor4all_core::tagset::TagSet;
///
/// // Default: u16 characters (ITensors.jl compatible)
/// let mut ts = TagSet::<4, 16>::new();
/// ts.add_tag("site").unwrap();
///
/// // Explicit char type for full Unicode support
/// let mut ts_unicode = TagSet::<4, 16, char>::new();
/// ts_unicode.add_tag("日本語").unwrap();
/// ```
#[derive(Debug, Clone, Copy)]
pub struct TagSet<const MAX_TAGS: usize, const MAX_TAG_LEN: usize, C: SmallChar = u16> {
    tags: [SmallString<MAX_TAG_LEN, C>; MAX_TAGS],
    length: usize, // Actual number of tags (0 ≤ length ≤ MAX_TAGS)
}

/// Error type for TagSet operations.
#[derive(thiserror::Error, Debug, Clone, PartialEq, Eq)]
pub enum TagSetError {
    /// Too many tags in the tag set (capacity exceeded).
    #[error("Too many tags ({actual} > {max})")]
    TooManyTags {
        /// The actual number of tags attempted.
        actual: usize,
        /// The maximum allowed tags.
        max: usize,
    },
    /// A tag is too long.
    #[error("Tag too long ({actual} > {max})")]
    TagTooLong {
        /// The actual length of the tag.
        actual: usize,
        /// The maximum allowed length.
        max: usize,
    },
    /// A tag contains invalid characters.
    #[error("Invalid tag: {0}")]
    InvalidTag(SmallStringError),
    /// Tag contains a comma, which is reserved as a separator in from_str.
    #[error("Tag contains comma: {tag:?}")]
    TagContainsComma {
        /// The tag that contains a comma.
        tag: String,
    },
}

impl<const MAX_TAGS: usize, const MAX_TAG_LEN: usize, C: SmallChar>
    TagSet<MAX_TAGS, MAX_TAG_LEN, C>
{
    /// Create an empty TagSet.
    pub fn new() -> Self {
        Self {
            tags: [SmallString::new(); MAX_TAGS],
            length: 0,
        }
    }

    /// Create a TagSet from a comma-separated string.
    ///
    /// ASCII spaces are ignored (matching ITensors.jl).
    /// Tags are automatically sorted.
    #[allow(clippy::should_implement_trait)]
    /// # Errors
    ///
    /// Returns an error when the string cannot be parsed as a tag or tag set (an
    /// /// invalid-input failure).
    ///
    pub fn from_str(s: &str) -> Result<Self, TagSetError> {
        <Self as TagSetLike>::from_str(s)
    }

    /// Get the number of tags.
    pub fn len(&self) -> usize {
        self.length
    }

    /// Check if the tag set is empty.
    pub fn is_empty(&self) -> bool {
        self.length == 0
    }

    /// Get the maximum capacity.
    pub fn capacity(&self) -> usize {
        MAX_TAGS
    }

    /// Get a tag at the given index.
    pub fn get(&self, index: usize) -> Option<&SmallString<MAX_TAG_LEN, C>> {
        if index < self.length {
            Some(&self.tags[index])
        } else {
            None
        }
    }

    /// Iterate over tags.
    pub fn iter(&self) -> impl Iterator<Item = &SmallString<MAX_TAG_LEN, C>> {
        self.tags[..self.length].iter()
    }

    /// Check if a tag is present.
    pub fn has_tag(&self, tag: &str) -> bool {
        <Self as TagSetLike>::has_tag(self, tag)
    }

    /// Check if all tags in another TagSet are present.
    pub fn has_tags(&self, tags: &TagSet<MAX_TAGS, MAX_TAG_LEN, C>) -> bool {
        <Self as TagSetLike>::has_tags(self, tags)
    }

    /// Add a tag (maintains sorted order).
    /// # Errors
    ///
    /// Returns an error when the tag is invalid (an invalid-input failure).
    ///
    pub fn add_tag(&mut self, tag: &str) -> Result<(), TagSetError> {
        <Self as TagSetLike>::add_tag(self, tag)
    }

    /// Remove a tag.
    pub fn remove_tag(&mut self, tag: &str) -> bool {
        <Self as TagSetLike>::remove_tag(self, tag)
    }

    /// Get common tags between two TagSets.
    pub fn common_tags(&self, other: &Self) -> Self {
        <Self as TagSetLike>::common_tags(self, other)
    }
}

impl<const MAX_TAGS: usize, const MAX_TAG_LEN: usize, C: SmallChar> TagSetLike
    for TagSet<MAX_TAGS, MAX_TAG_LEN, C>
{
    fn len(&self) -> usize {
        self.length
    }

    fn capacity(&self) -> usize {
        MAX_TAGS
    }

    fn get(&self, index: usize) -> Option<String> {
        if index < self.length {
            Some(self.tags[index].as_str())
        } else {
            None
        }
    }

    fn iter(&self) -> TagSetIterator<'_> {
        Box::new(self.tags[..self.length].iter().map(|s| s.as_str()))
    }

    fn has_tag(&self, tag: &str) -> bool {
        let tag_str = match SmallString::<MAX_TAG_LEN, C>::from_str(tag) {
            Ok(s) => s,
            Err(_) => return false,
        };
        self._has_tag(&tag_str)
    }

    /// # Errors
    ///
    /// Returns an error when the tag is invalid (an invalid-input failure).
    ///
    fn add_tag(&mut self, tag: &str) -> Result<(), TagSetError> {
        let tag_str =
            SmallString::<MAX_TAG_LEN, C>::from_str(tag).map_err(TagSetError::InvalidTag)?;
        self._add_tag_ordered(tag_str)
    }

    fn remove_tag(&mut self, tag: &str) -> bool {
        let tag_str = match SmallString::<MAX_TAG_LEN, C>::from_str(tag) {
            Ok(s) => s,
            Err(_) => return false,
        };

        if let Some(pos) = self.tags[..self.length].iter().position(|t| *t == tag_str) {
            // Shift remaining tags left
            for i in pos..self.length - 1 {
                self.tags[i] = self.tags[i + 1];
            }
            self.length -= 1;
            true
        } else {
            false
        }
    }
}

impl<const MAX_TAGS: usize, const MAX_TAG_LEN: usize, C: SmallChar>
    TagSet<MAX_TAGS, MAX_TAG_LEN, C>
{
    /// Internal: Add a tag in sorted order (similar to ITensors.jl's `_addtag_ordered!`).
    fn _add_tag_ordered(&mut self, tag: SmallString<MAX_TAG_LEN, C>) -> Result<(), TagSetError> {
        // Check for duplicates
        if self._has_tag(&tag) {
            return Ok(()); // Already present, no error
        }

        // Check capacity
        if self.length >= MAX_TAGS {
            return Err(TagSetError::TooManyTags {
                actual: self.length + 1,
                max: MAX_TAGS,
            });
        }

        // Find insertion position (binary search for sorted insertion)
        let pos = self.tags[..self.length]
            .binary_search(&tag)
            .unwrap_or_else(|pos| pos);

        // Shift tags right to make room
        for i in (pos..self.length).rev() {
            self.tags[i + 1] = self.tags[i];
        }

        // Insert the new tag
        self.tags[pos] = tag;
        self.length += 1;

        Ok(())
    }

    /// Internal: Check if a tag is present (binary search).
    fn _has_tag(&self, tag: &SmallString<MAX_TAG_LEN, C>) -> bool {
        self.tags[..self.length].binary_search(tag).is_ok()
    }
}

impl<const MAX_TAGS: usize, const MAX_TAG_LEN: usize, C: SmallChar> Default
    for TagSet<MAX_TAGS, MAX_TAG_LEN, C>
{
    fn default() -> Self {
        Self::new()
    }
}

impl<const MAX_TAGS: usize, const MAX_TAG_LEN: usize, C: SmallChar> PartialEq
    for TagSet<MAX_TAGS, MAX_TAG_LEN, C>
{
    fn eq(&self, other: &Self) -> bool {
        if self.length != other.length {
            return false;
        }
        self.tags[..self.length] == other.tags[..other.length]
    }
}

impl<const MAX_TAGS: usize, const MAX_TAG_LEN: usize, C: SmallChar> Eq
    for TagSet<MAX_TAGS, MAX_TAG_LEN, C>
{
}

impl<const MAX_TAGS: usize, const MAX_TAG_LEN: usize, C: SmallChar> std::hash::Hash
    for TagSet<MAX_TAGS, MAX_TAG_LEN, C>
{
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.tags[..self.length].hash(state);
    }
}

/// A sorted tag set backed by owned strings.
///
/// Use this when tags must be preserved losslessly across language boundaries
/// or when a fixed inline string capacity would reject valid user-facing tags.
/// Tags are sorted, duplicate insertion is a no-op, and commas are rejected
/// because comma is the textual separator used by [`TagSetLike::from_str`].
///
/// # Examples
///
/// ```
/// use tensor4all_core::tagset::{DynamicTagSet, TagSetLike};
///
/// let mut tags = DynamicTagSet::from_str("Site3_Site4_Site5,Link")?;
/// tags.add_tag("LongDescriptiveTagName")?;
///
/// assert!(tags.has_tag("Site3_Site4_Site5"));
/// assert!(tags.has_tag("LongDescriptiveTagName"));
/// assert_eq!(tags.len(), 3);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct DynamicTagSet {
    tags: Vec<String>,
}

impl DynamicTagSet {
    /// Create an empty dynamic tag set.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::tagset::{DynamicTagSet, TagSetLike};
    ///
    /// let tags = DynamicTagSet::new();
    /// assert!(tags.is_empty());
    /// assert_eq!(tags.capacity(), usize::MAX);
    /// ```
    pub fn new() -> Self {
        Self { tags: Vec::new() }
    }

    /// Create a dynamic tag set from a comma-separated string.
    ///
    /// ASCII spaces are ignored, matching [`TagSetLike::from_str`].
    ///
    /// # Errors
    /// Returns an error when a parsed tag violates dynamic tag-set rules (an
    /// invalid-input failure).
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::tagset::{DynamicTagSet, TagSetLike};
    ///
    /// let tags = DynamicTagSet::from_str("Site3_Site4_Site5, Link")?;
    /// assert!(tags.has_tag("Site3_Site4_Site5"));
    /// assert!(tags.has_tag("Link"));
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[allow(clippy::should_implement_trait)]
    /// # Errors
    ///
    /// Returns an error when the string cannot be parsed as a tag or tag set (an
    /// /// invalid-input failure).
    ///
    pub fn from_str(s: &str) -> Result<Self, TagSetError> {
        <Self as TagSetLike>::from_str(s)
    }
}

impl TagSetLike for DynamicTagSet {
    fn len(&self) -> usize {
        self.tags.len()
    }

    fn capacity(&self) -> usize {
        usize::MAX
    }

    fn get(&self, index: usize) -> Option<String> {
        self.tags.get(index).cloned()
    }

    fn iter(&self) -> TagSetIterator<'_> {
        Box::new(self.tags.iter().cloned())
    }

    fn has_tag(&self, tag: &str) -> bool {
        self.tags
            .binary_search_by(|existing| existing.as_str().cmp(tag))
            .is_ok()
    }

    /// # Errors
    ///
    /// Returns an error when the tag is invalid (an invalid-input failure).
    ///
    fn add_tag(&mut self, tag: &str) -> Result<(), TagSetError> {
        if tag.contains(',') {
            return Err(TagSetError::TagContainsComma {
                tag: tag.to_string(),
            });
        }

        match self
            .tags
            .binary_search_by(|existing| existing.as_str().cmp(tag))
        {
            Ok(_) => Ok(()),
            Err(position) => {
                self.tags.insert(position, tag.to_string());
                Ok(())
            }
        }
    }

    fn remove_tag(&mut self, tag: &str) -> bool {
        match self
            .tags
            .binary_search_by(|existing| existing.as_str().cmp(tag))
        {
            Ok(position) => {
                self.tags.remove(position);
                true
            }
            Err(_) => false,
        }
    }
}

impl PartialOrd for DynamicTagSet {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for DynamicTagSet {
    fn cmp(&self, other: &Self) -> Ordering {
        self.tags.cmp(&other.tags)
    }
}

/// Default tag type (max 16 characters, u16 storage, matching ITensors.jl's `SmallString`).
pub type Tag = SmallString<16>;

/// Default tag set for public dynamic indices.
///
/// This is string-backed so FFI and language bindings preserve descriptive
/// tags losslessly instead of truncating or rejecting tags that exceed the
/// fixed inline [`Tag`] capacity. Use [`TagSet`] directly when a fixed inline
/// capacity is required.
pub type DefaultTagSet = DynamicTagSet;
