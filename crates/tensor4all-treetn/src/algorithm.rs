//! Algorithm selection types for tensor train operations.
//!
//! This module provides enums for selecting algorithms in various tensor train operations.
//! These types are designed to be FFI-friendly (representable as C integers).
//!
//! # Design Philosophy
//!
//! Following ITensors.jl's pattern, algorithms are represented as static types.
//! Unlike ITensors.jl which uses symbol-based dispatch (`Algorithm"svd"`),
//! we use Rust enums for:
//! - Compile-time exhaustiveness checking
//! - Easy FFI mapping to C integers
//! - Clear documentation of available algorithms
//!
//! # Example
//!
//! ```
//! use tensor4all_treetn::{ContractionAlgorithm, CanonicalForm};
//!
//! // Select contraction algorithm
//! let alg = ContractionAlgorithm::ZipUp;
//!
//! // Select canonical form
//! let form = CanonicalForm::Unitary;
//! ```

/// Algorithm for tensor train contraction (TT-TT or MPO-MPO).
///
/// These algorithms contract two tensor trains and produce a new tensor train,
/// optionally with compression/truncation.
///
/// # C API Representation
/// - `T4A_CONTRACT_NAIVE` = 0
/// - `T4A_CONTRACT_ZIPUP` = 1
/// - `T4A_CONTRACT_FIT` = 2
/// - `T4A_CONTRACT_SRC` = 3
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(i32)]
pub enum ContractionAlgorithm {
    /// Naive contraction followed by compression.
    ///
    /// Contracts all site tensors first (producing large intermediate tensors),
    /// then compresses the result. Exact up to compression tolerance.
    ///
    /// Memory: O(D^4) where D is the bond dimension.
    #[default]
    Naive = 0,

    /// Zip-up contraction with on-the-fly compression.
    ///
    /// Contracts and compresses site-by-site, keeping bond dimensions small.
    /// More memory efficient than naive, but may introduce additional error.
    ///
    /// Memory: O(D^2)
    ZipUp = 1,

    /// Variational fitting algorithm.
    ///
    /// Optimizes the result tensor train to minimize the distance to the
    /// exact contraction. Uses sweeping optimization.
    ///
    /// Best for cases where target bond dimension is much smaller than
    /// the exact result.
    Fit = 2,

    /// Successive randomized compression using sketching and QR projection.
    Src = 3,
}

impl ContractionAlgorithm {
    /// Create from C API integer representation.
    ///
    /// Returns `None` for invalid values.
    pub fn from_i32(value: i32) -> Option<Self> {
        match value {
            0 => Some(Self::Naive),
            1 => Some(Self::ZipUp),
            2 => Some(Self::Fit),
            3 => Some(Self::Src),
            _ => None,
        }
    }

    /// Convert to C API integer representation.
    pub fn to_i32(self) -> i32 {
        self as i32
    }

    /// Get algorithm name as string.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Naive => "naive",
            Self::ZipUp => "zipup",
            Self::Fit => "fit",
            Self::Src => "src",
        }
    }
}

/// Canonical form for tensor train / tree tensor network.
///
/// Specifies the mathematical form of the canonical representation.
/// Each form uses a specific factorization algorithm internally.
///
/// # C API Representation
/// - `T4A_CANONICAL_UNITARY` = 0
/// - `T4A_CANONICAL_LU` = 1
/// - `T4A_CANONICAL_CI` = 2
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(i32)]
pub enum CanonicalForm {
    /// Unitary (orthogonal/isometric) canonical form.
    ///
    /// Each tensor is isometric towards the orthogonality center.
    /// Uses QR decomposition internally.
    /// Properties:
    /// - Numerically stable
    /// - Easy norm computation
    /// - Standard canonical form for DMRG
    #[default]
    Unitary = 0,

    /// LU-based canonical form.
    ///
    /// Uses rank-revealing LU decomposition.
    /// Properties:
    /// - Faster than QR
    /// - One factor has unit diagonal
    LU = 1,

    /// Cross Interpolation (CI) canonical form.
    ///
    /// Uses CI/skeleton decomposition.
    /// Properties:
    /// - Adaptive rank selection
    /// - Efficient for low-rank structures
    CI = 2,
}

impl CanonicalForm {
    /// Create from C API integer representation.
    ///
    /// Returns `None` for invalid values.
    pub fn from_i32(value: i32) -> Option<Self> {
        match value {
            0 => Some(Self::Unitary),
            1 => Some(Self::LU),
            2 => Some(Self::CI),
            _ => None,
        }
    }

    /// Convert to C API integer representation.
    pub fn to_i32(self) -> i32 {
        self as i32
    }

    /// Get form name as string.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Unitary => "unitary",
            Self::LU => "lu",
            Self::CI => "ci",
        }
    }
}

/// Algorithm for tensor train compression.
///
/// These algorithms compress a tensor train to reduce bond dimensions.
///
/// # C API Representation
/// - `T4A_COMPRESS_SVD` = 0
/// - `T4A_COMPRESS_LU` = 1
/// - `T4A_COMPRESS_CI` = 2
/// - `T4A_COMPRESS_VARIATIONAL` = 3
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(i32)]
pub enum CompressionAlgorithm {
    /// SVD-based compression (orthogonalization + truncation).
    ///
    /// Sweeps through the tensor train, applying SVD at each bond.
    /// Optimal truncation for given tolerance.
    #[default]
    SVD = 0,

    /// LU-based compression.
    ///
    /// Uses LU decomposition instead of SVD. Faster but may not give
    /// optimal truncation.
    LU = 1,

    /// Cross Interpolation based compression.
    ///
    /// Uses CI/skeleton decomposition for compression.
    CI = 2,

    /// Variational compression.
    ///
    /// Optimizes the compressed tensor train using sweeping.
    /// Useful when target bond dimension is known.
    Variational = 3,
}

impl CompressionAlgorithm {
    /// Create from C API integer representation.
    ///
    /// Returns `None` for invalid values.
    pub fn from_i32(value: i32) -> Option<Self> {
        match value {
            0 => Some(Self::SVD),
            1 => Some(Self::LU),
            2 => Some(Self::CI),
            3 => Some(Self::Variational),
            _ => None,
        }
    }

    /// Convert to C API integer representation.
    pub fn to_i32(self) -> i32 {
        self as i32
    }

    /// Get algorithm name as string.
    pub fn name(&self) -> &'static str {
        match self {
            Self::SVD => "svd",
            Self::LU => "lu",
            Self::CI => "ci",
            Self::Variational => "variational",
        }
    }
}

#[cfg(test)]
mod tests;
