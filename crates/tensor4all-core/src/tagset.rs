use crate::smallstring::{SmallString, SmallStringError};

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
    /// Whitespace is ignored (similar to ITensors.jl).
    /// Tags are automatically sorted.
    fn from_str(s: &str) -> Result<Self, TagSetError> {
        let mut tagset = Self::default();
        
        // Parse comma-separated tags, ignoring whitespace
        let mut current_tag = String::new();
        for ch in s.chars() {
            if ch == ',' {
                if !current_tag.is_empty() {
                    let trimmed: String = current_tag.chars().filter(|c| !c.is_whitespace()).collect();
                    if !trimmed.is_empty() {
                        tagset.add_tag(&trimmed)?;
                    }
                    current_tag.clear();
                }
            } else {
                current_tag.push(ch);
            }
        }
        
        // Handle the last tag
        if !current_tag.is_empty() {
            let trimmed: String = current_tag.chars().filter(|c| !c.is_whitespace()).collect();
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
#[derive(Debug, Clone, Copy)]
pub struct TagSet<const MAX_TAGS: usize, const MAX_TAG_LEN: usize> {
    tags: [SmallString<MAX_TAG_LEN>; MAX_TAGS],
    length: usize, // Actual number of tags (0 ≤ length ≤ MAX_TAGS)
}

/// Error type for TagSet operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TagSetError {
    TooManyTags { actual: usize, max: usize },
    TagTooLong { actual: usize, max: usize },
    InvalidTag(SmallStringError),
}

impl<const MAX_TAGS: usize, const MAX_TAG_LEN: usize> TagSet<MAX_TAGS, MAX_TAG_LEN> {
    /// Create an empty TagSet.
    pub fn new() -> Self {
        Self {
            tags: [SmallString::new(); MAX_TAGS],
            length: 0,
        }
    }

    /// Create a TagSet from a comma-separated string.
    ///
    /// Whitespace is ignored (similar to ITensors.jl).
    /// Tags are automatically sorted.
    pub fn from_str(s: &str) -> Result<Self, TagSetError> {
        <Self as TagSetLike>::from_str(s)
    }

    /// Get the number of tags.
    pub fn len(&self) -> usize {
        self.length
    }

    /// Get the maximum capacity.
    pub fn capacity(&self) -> usize {
        MAX_TAGS
    }

    /// Get a tag at the given index.
    pub fn get(&self, index: usize) -> Option<&SmallString<MAX_TAG_LEN>> {
        if index < self.length {
            Some(&self.tags[index])
        } else {
            None
        }
    }

    /// Iterate over tags.
    pub fn iter(&self) -> impl Iterator<Item = &SmallString<MAX_TAG_LEN>> {
        self.tags[..self.length].iter()
    }

    /// Check if a tag is present.
    pub fn has_tag(&self, tag: &str) -> bool {
        <Self as TagSetLike>::has_tag(self, tag)
    }

    /// Check if all tags in another TagSet are present.
    pub fn has_tags(&self, tags: &TagSet<MAX_TAGS, MAX_TAG_LEN>) -> bool {
        <Self as TagSetLike>::has_tags(self, tags)
    }

    /// Add a tag (maintains sorted order).
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

impl<const MAX_TAGS: usize, const MAX_TAG_LEN: usize> TagSetLike for TagSet<MAX_TAGS, MAX_TAG_LEN> {
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
        let tag_str = match SmallString::<MAX_TAG_LEN>::from_str(tag) {
            Ok(s) => s,
            Err(_) => return false,
        };
        self._has_tag(&tag_str)
    }

    fn add_tag(&mut self, tag: &str) -> Result<(), TagSetError> {
        let tag_str = SmallString::<MAX_TAG_LEN>::from_str(tag)
            .map_err(|e| TagSetError::InvalidTag(e))?;
        self._add_tag_ordered(tag_str)
    }

    fn remove_tag(&mut self, tag: &str) -> bool {
        let tag_str = match SmallString::<MAX_TAG_LEN>::from_str(tag) {
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

impl<const MAX_TAGS: usize, const MAX_TAG_LEN: usize> TagSet<MAX_TAGS, MAX_TAG_LEN> {
    /// Internal: Add a tag in sorted order (similar to ITensors.jl's `_addtag_ordered!`).
    fn _add_tag_ordered(&mut self, tag: SmallString<MAX_TAG_LEN>) -> Result<(), TagSetError> {
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
    fn _has_tag(&self, tag: &SmallString<MAX_TAG_LEN>) -> bool {
        self.tags[..self.length].binary_search(tag).is_ok()
    }
}

impl<const MAX_TAGS: usize, const MAX_TAG_LEN: usize> Default for TagSet<MAX_TAGS, MAX_TAG_LEN> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const MAX_TAGS: usize, const MAX_TAG_LEN: usize> PartialEq for TagSet<MAX_TAGS, MAX_TAG_LEN> {
    fn eq(&self, other: &Self) -> bool {
        if self.length != other.length {
            return false;
        }
        self.tags[..self.length] == other.tags[..other.length]
    }
}

impl<const MAX_TAGS: usize, const MAX_TAG_LEN: usize> Eq for TagSet<MAX_TAGS, MAX_TAG_LEN> {}

/// Default tag type (max 16 characters, matching ITensors.jl's `SmallString`).
pub type Tag = SmallString<16>;

/// Default TagSet (max 4 tags, each tag max 16 characters, matching ITensors.jl's `TagSet`).
pub type DefaultTagSet = TagSet<4, 16>;

