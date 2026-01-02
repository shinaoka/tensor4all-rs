/// A stack-allocated fixed-capacity string with explicit length.
///
/// This type stores characters in a fixed-size array and maintains
/// an explicit length field, similar to ITensors.jl's `SmallString`.
#[derive(Debug, Clone, Copy)]
pub struct SmallString<const MAX_LEN: usize> {
    data: [char; MAX_LEN],
    len: usize, // Explicit length (0 ≤ len ≤ MAX_LEN)
}

/// Error type for SmallString operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SmallStringError {
    TooLong { actual: usize, max: usize },
}

impl<const MAX_LEN: usize> SmallString<MAX_LEN> {
    /// Create an empty SmallString.
    pub fn new() -> Self {
        Self {
            data: ['\0'; MAX_LEN],
            len: 0,
        }
    }

    /// Create a SmallString from a string slice.
    ///
    /// Returns an error if the string is longer than MAX_LEN.
    pub fn from_str(s: &str) -> Result<Self, SmallStringError> {
        let chars: Vec<char> = s.chars().collect();
        if chars.len() > MAX_LEN {
            return Err(SmallStringError::TooLong {
                actual: chars.len(),
                max: MAX_LEN,
            });
        }

        let mut data = ['\0'; MAX_LEN];
        for (i, &ch) in chars.iter().enumerate() {
            data[i] = ch;
        }

        Ok(Self {
            data,
            len: chars.len(),
        })
    }

    /// Convert to a String.
    pub fn as_str(&self) -> String {
        self.data[..self.len].iter().collect()
    }

    /// Check if the string is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Get the length of the string.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Get the maximum capacity.
    pub fn capacity(&self) -> usize {
        MAX_LEN
    }

    /// Get a character at the given index.
    pub fn get(&self, index: usize) -> Option<char> {
        if index < self.len {
            Some(self.data[index])
        } else {
            None
        }
    }

    /// Get a reference to the internal data slice.
    pub fn as_slice(&self) -> &[char] {
        &self.data[..self.len]
    }
}

impl<const MAX_LEN: usize> Default for SmallString<MAX_LEN> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const MAX_LEN: usize> PartialEq for SmallString<MAX_LEN> {
    fn eq(&self, other: &Self) -> bool {
        self.len == other.len && self.data[..self.len] == other.data[..other.len]
    }
}

impl<const MAX_LEN: usize> Eq for SmallString<MAX_LEN> {}

impl<const MAX_LEN: usize> PartialOrd for SmallString<MAX_LEN> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<const MAX_LEN: usize> Ord for SmallString<MAX_LEN> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.data[..self.len].cmp(&other.data[..other.len])
    }
}

impl<const MAX_LEN: usize> std::fmt::Display for SmallString<MAX_LEN> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

