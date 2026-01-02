use crate::smallstring::{SmallString, SmallStringError};

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
        let mut tagset = Self::new();
        
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
        let tag_str = match SmallString::<MAX_TAG_LEN>::from_str(tag) {
            Ok(s) => s,
            Err(_) => return false,
        };
        self._has_tag(&tag_str)
    }

    /// Check if all tags in another TagSet are present.
    pub fn has_tags(&self, tags: &TagSet<MAX_TAGS, MAX_TAG_LEN>) -> bool {
        for tag in tags.iter() {
            if !self._has_tag(tag) {
                return false;
            }
        }
        true
    }

    /// Add a tag (maintains sorted order).
    pub fn add_tag(&mut self, tag: &str) -> Result<(), TagSetError> {
        let tag_str = SmallString::<MAX_TAG_LEN>::from_str(tag)
            .map_err(|e| TagSetError::InvalidTag(e))?;
        self._add_tag_ordered(tag_str)
    }

    /// Remove a tag.
    pub fn remove_tag(&mut self, tag: &str) -> bool {
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

    /// Get common tags between two TagSets.
    pub fn common_tags(&self, other: &Self) -> Self {
        let mut result = Self::new();
        for tag in self.iter() {
            if other._has_tag(tag) {
                // Safe to unwrap because we know the tag fits
                result._add_tag_ordered(*tag).ok();
            }
        }
        result
    }

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

