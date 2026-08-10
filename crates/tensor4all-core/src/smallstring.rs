/// Trait for character types that can be stored in SmallString.
///
/// This trait abstracts over different character representations:
/// - `u16`: UTF-16 code unit (2 bytes), compatible with ITensors.jl's SmallString
/// - `char`: Full Unicode code point (4 bytes), supports all Unicode characters
///
/// Default is `u16` for memory efficiency and ITensors.jl compatibility.
pub trait SmallChar: Copy + Default + Ord + Eq + std::hash::Hash + std::fmt::Debug {
    /// The null/zero value for this character type.
    const ZERO: Self;

    /// Convert from a Rust char.
    /// Returns None if the character cannot be represented in this type.
    fn from_char(c: char) -> Option<Self>;

    /// Convert to a Rust char.
    fn to_char(self) -> char;
}

impl SmallChar for u16 {
    const ZERO: Self = 0;

    fn from_char(c: char) -> Option<Self> {
        // u16 can represent BMP characters (U+0000 to U+FFFF)
        let code = c as u32;
        if code <= 0xFFFF {
            Some(code as u16)
        } else {
            None // Surrogate pairs not supported
        }
    }

    fn to_char(self) -> char {
        // Safe because we only store valid BMP characters
        char::from_u32(self as u32).unwrap_or('\u{FFFD}')
    }
}

impl SmallChar for char {
    const ZERO: Self = '\0';

    fn from_char(c: char) -> Option<Self> {
        Some(c)
    }

    fn to_char(self) -> char {
        self
    }
}

/// A stack-allocated fixed-capacity string with explicit length.
///
/// This type stores characters in a fixed-size array and maintains
/// an explicit length field, similar to ITensors.jl's `SmallString`.
///
/// # Type Parameters
/// - `MAX_LEN`: Maximum number of characters (default: 16, matching ITensors.jl)
/// - `C`: Character type (default: `u16` for ITensors.jl compatibility)
///
/// # Character Type Options
/// - `u16` (default): 2 bytes per character, supports BMP (Basic Multilingual Plane)
///   - Covers ASCII, Japanese, Chinese, Korean, and most practical characters
///   - Does NOT support emoji or rare characters outside BMP
/// - `char`: 4 bytes per character, full Unicode support
///
/// # Example
/// ```
/// use tensor4all_core::smallstring::SmallString;
///
/// // Default: u16 characters (ITensors.jl compatible)
/// let s1 = SmallString::<16>::from_str("hello").unwrap();
///
/// // Explicit char type for full Unicode support
/// let s2 = SmallString::<16, char>::from_str("hello 😀").unwrap();
/// ```
#[derive(Debug, Clone, Copy)]
pub struct SmallString<const MAX_LEN: usize, C: SmallChar = u16> {
    data: [C; MAX_LEN],
    len: usize, // Explicit length (0 ≤ len ≤ MAX_LEN)
}

/// Error type for SmallString operations.
#[derive(thiserror::Error, Debug, Clone, Copy, PartialEq, Eq)]
pub enum SmallStringError {
    /// The string exceeds the maximum length.
    #[error("String too long ({actual} > {max})")]
    TooLong {
        /// The actual length of the string.
        actual: usize,
        /// The maximum allowed length.
        max: usize,
    },
    /// A character cannot be represented in the target character type.
    #[error("Invalid character: {char_value:?}")]
    InvalidChar {
        /// The character that could not be converted.
        char_value: char,
    },
}

impl<const MAX_LEN: usize, C: SmallChar> SmallString<MAX_LEN, C> {
    /// Create an empty SmallString.
    pub fn new() -> Self {
        Self {
            data: [C::ZERO; MAX_LEN],
            len: 0,
        }
    }

    /// Create a SmallString from a string slice.
    ///
    /// Returns an error if:
    /// - The string is longer than MAX_LEN characters
    /// - Any character cannot be represented in the character type C
    ///
    /// This function is allocation-free (no heap allocation).
    #[allow(clippy::should_implement_trait)]
    /// # Errors
    ///
    /// Returns an error when the operation fails (a shape or index mismatch, or
    /// /// a backend failure).
    ///
    pub fn from_str(s: &str) -> Result<Self, SmallStringError> {
        let mut data = [C::ZERO; MAX_LEN];
        let mut len = 0;

        for ch in s.chars() {
            if len >= MAX_LEN {
                // Count total characters for error message
                let actual = len + 1 + s.chars().skip(len + 1).count();
                return Err(SmallStringError::TooLong {
                    actual,
                    max: MAX_LEN,
                });
            }
            data[len] = C::from_char(ch).ok_or(SmallStringError::InvalidChar { char_value: ch })?;
            len += 1;
        }

        Ok(Self { data, len })
    }

    /// Convert to a String.
    pub fn as_str(&self) -> String {
        self.data[..self.len].iter().map(|c| c.to_char()).collect()
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
            Some(self.data[index].to_char())
        } else {
            None
        }
    }

    /// Get a reference to the internal data slice.
    pub fn as_slice(&self) -> &[C] {
        &self.data[..self.len]
    }
}

impl<const MAX_LEN: usize, C: SmallChar> Default for SmallString<MAX_LEN, C> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const MAX_LEN: usize, C: SmallChar> PartialEq for SmallString<MAX_LEN, C> {
    fn eq(&self, other: &Self) -> bool {
        self.len == other.len && self.data[..self.len] == other.data[..other.len]
    }
}

impl<const MAX_LEN: usize, C: SmallChar> Eq for SmallString<MAX_LEN, C> {}

impl<const MAX_LEN: usize, C: SmallChar> std::hash::Hash for SmallString<MAX_LEN, C> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.data[..self.len].hash(state);
    }
}

impl<const MAX_LEN: usize, C: SmallChar> PartialOrd for SmallString<MAX_LEN, C> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<const MAX_LEN: usize, C: SmallChar> Ord for SmallString<MAX_LEN, C> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.data[..self.len].cmp(&other.data[..other.len])
    }
}

impl<const MAX_LEN: usize, C: SmallChar> std::fmt::Display for SmallString<MAX_LEN, C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

#[cfg(test)]
mod tests;
