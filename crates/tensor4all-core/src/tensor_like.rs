//! TensorLike trait for unifying tensor types.
//!
//! This module provides a fully generic trait for tensor-like objects that expose
//! external indices and support contraction operations.
//!
//! # Design
//!
//! The trait is **fully generic** (monomorphic), meaning:
//! - No trait objects (`dyn TensorLike`)
//! - Uses associated type for `Index`
//! - All methods return `Self` instead of concrete types
//!
//! For heterogeneous tensor collections, use an enum wrapper.

use crate::any_scalar::AnyScalar;
use crate::index_like::IndexLike;
use crate::tensor_index::TensorIndex;
use crate::truncation::SvdTruncationPolicy;
use anyhow::Result;
use std::collections::HashSet;
use std::fmt::Debug;
use std::sync::Arc;

// ============================================================================
// Factorization types (non-generic, algorithm-specific)
// ============================================================================

use thiserror::Error;

/// Error type for factorize operations.
///
/// # Examples
///
/// ```
/// use tensor4all_core::{
///     factorize, Canonical, DynIndex, FactorizeOptions, TensorContractionLike, TensorDynLen,
/// };
///
/// let i = DynIndex::new_dyn(3);
/// let j = DynIndex::new_dyn(3);
/// let data: Vec<f64> = (0..9).map(|x| x as f64).collect();
/// let tensor = TensorDynLen::from_dense(vec![i.clone(), j.clone()], data).unwrap();
///
/// // QR with Canonical::Right is not supported
/// let result = factorize(
///     &tensor,
///     &[i],
///     &FactorizeOptions::qr().with_canonical(Canonical::Right),
/// );
/// assert!(result.is_err());
/// ```
#[derive(Debug, Error)]
pub enum FactorizeError {
    /// Factorization computation failed.
    #[error("Factorization failed: {0}")]
    ComputationError(
        /// The underlying error
        #[from]
        anyhow::Error,
    ),
    /// Invalid relative tolerance value (must be finite and non-negative).
    #[error("Invalid rtol value: {0}. rtol must be finite and non-negative.")]
    InvalidRtol(
        /// The invalid rtol value
        f64,
    ),
    /// Invalid algorithm-specific option combination.
    #[error("Invalid factorize options: {0}")]
    InvalidOptions(
        /// Description of the invalid option combination
        &'static str,
    ),
    /// The storage type is not supported for this operation.
    #[error("Unsupported storage type: {0}")]
    UnsupportedStorage(
        /// Description of the unsupported storage type
        &'static str,
    ),
    /// The canonical direction is not supported for this algorithm.
    #[error("Unsupported canonical direction for this algorithm: {0}")]
    UnsupportedCanonical(
        /// Description of the unsupported canonical direction
        &'static str,
    ),
    /// Error from SVD operation.
    #[error("SVD error: {0}")]
    SvdError(
        /// The underlying SVD error
        #[from]
        crate::svd::SvdError,
    ),
    /// Error from QR operation.
    #[error("QR error: {0}")]
    QrError(
        /// The underlying QR error
        #[from]
        crate::qr::QrError,
    ),
    /// Error from matrix CI operation.
    #[error("Matrix CI error: {0}")]
    MatrixCIError(
        /// The underlying matrix CI error
        #[from]
        tensor4all_tcicore::MatrixCIError,
    ),
}

/// Factorization algorithm.
///
/// Determines which matrix decomposition is used by [`TensorFactorizationLike::factorize`].
///
/// # Examples
///
/// ```
/// use tensor4all_core::FactorizeAlg;
///
/// // Default is SVD
/// assert_eq!(FactorizeAlg::default(), FactorizeAlg::SVD);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FactorizeAlg {
    /// Singular Value Decomposition.
    #[default]
    SVD,
    /// QR decomposition.
    QR,
    /// Rank-revealing LU decomposition.
    LU,
    /// Cross Interpolation (LU-based).
    CI,
}

/// Canonical direction for factorization.
///
/// This determines which factor is "canonical" (orthogonal for SVD/QR,
/// or unit-diagonal for LU/CI).
///
/// # Examples
///
/// ```
/// use tensor4all_core::{
///     factorize, Canonical, DynIndex, FactorizeOptions, TensorContractionLike, TensorDynLen,
/// };
///
/// let i = DynIndex::new_dyn(3);
/// let j = DynIndex::new_dyn(3);
/// let data: Vec<f64> = (0..9).map(|x| x as f64).collect();
/// let tensor = TensorDynLen::from_dense(vec![i.clone(), j.clone()], data).unwrap();
///
/// // Left canonical: left factor has orthonormal columns
/// let left_result = factorize(
///     &tensor,
///     &[i.clone()],
///     &FactorizeOptions::svd().with_canonical(Canonical::Left),
/// ).unwrap();
///
/// // Right canonical: right factor has orthonormal rows
/// let right_result = factorize(
///     &tensor,
///     &[i.clone()],
///     &FactorizeOptions::svd().with_canonical(Canonical::Right),
/// ).unwrap();
///
/// // Both recover the same tensor
/// let recovered_left = left_result.left.contract_pair(&left_result.right).unwrap();
/// let recovered_right = right_result.left.contract_pair(&right_result.right).unwrap();
/// assert!(recovered_left.distance(&recovered_right).unwrap() < 1e-12);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Canonical {
    /// Left factor is canonical.
    /// - SVD: L=U (orthogonal), R=S*V
    /// - QR: L=Q (orthogonal), R=R
    /// - LU/CI: L has unit diagonal
    #[default]
    Left,
    /// Right factor is canonical.
    /// - SVD: L=U*S, R=V (orthogonal)
    /// - QR: Not supported (would need LQ)
    /// - LU/CI: U has unit diagonal
    Right,
}

/// Options for tensor factorization.
///
/// Controls the algorithm, canonical direction, and truncation parameters
/// for [`TensorFactorizationLike::factorize`].
///
/// # Defaults
///
/// - Algorithm: SVD
/// - Canonical: Left (left factor is orthogonal)
/// - max_rank: `None` (no rank limit)
/// - svd_policy: `None` (uses the SVD global default policy)
/// - qr_rtol: `None` (uses the QR global default tolerance)
///
/// # Field Interactions
///
/// - `svd_policy` is only valid for `FactorizeAlg::SVD`.
/// - `qr_rtol` is only valid for `FactorizeAlg::QR`.
/// - `max_rank` is independent of the algorithm-specific tolerance settings.
///
/// # Examples
///
/// ```
/// use tensor4all_core::{
///     factorize, Canonical, DynIndex, FactorizeOptions, TensorDynLen,
/// };
///
/// let i = DynIndex::new_dyn(4);
/// let j = DynIndex::new_dyn(4);
/// let mut data = vec![0.0_f64; 16];
/// data[0] = 1.0;  // rank-1 matrix
/// let tensor = TensorDynLen::from_dense(vec![i.clone(), j.clone()], data).unwrap();
///
/// // SVD with an explicit policy
/// let opts = FactorizeOptions::svd()
///     .with_svd_policy(tensor4all_core::SvdTruncationPolicy::new(1e-10));
/// let result = factorize(&tensor, &[i.clone()], &opts).unwrap();
/// assert_eq!(result.rank, 1);
///
/// // QR with max-rank truncation
/// let opts = FactorizeOptions::qr().with_qr_rtol(1e-8).with_max_rank(2);
/// let result = factorize(&tensor, &[i.clone()], &opts).unwrap();
/// assert!(result.rank <= 2);
/// ```
#[derive(Debug, Clone)]
pub struct FactorizeOptions {
    /// Factorization algorithm to use.
    pub alg: FactorizeAlg,
    /// Canonical direction.
    pub canonical: Canonical,
    /// Maximum rank for truncation.
    /// If `None`, no rank limit is applied.
    pub max_rank: Option<usize>,
    /// SVD truncation policy.
    /// If `None`, uses the SVD global default.
    pub svd_policy: Option<SvdTruncationPolicy>,
    /// QR-specific relative tolerance.
    /// If `None`, uses the QR global default.
    pub qr_rtol: Option<f64>,
}

impl Default for FactorizeOptions {
    fn default() -> Self {
        Self {
            alg: FactorizeAlg::SVD,
            canonical: Canonical::Left,
            max_rank: None,
            svd_policy: None,
            qr_rtol: None,
        }
    }
}

impl FactorizeOptions {
    /// Create options for SVD factorization.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::{FactorizeAlg, FactorizeOptions};
    ///
    /// let opts = FactorizeOptions::svd();
    /// assert_eq!(opts.alg, FactorizeAlg::SVD);
    /// ```
    pub fn svd() -> Self {
        Self {
            alg: FactorizeAlg::SVD,
            ..Default::default()
        }
    }

    /// Create options for QR factorization.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::{FactorizeAlg, FactorizeOptions};
    ///
    /// let opts = FactorizeOptions::qr();
    /// assert_eq!(opts.alg, FactorizeAlg::QR);
    /// ```
    pub fn qr() -> Self {
        Self {
            alg: FactorizeAlg::QR,
            ..Default::default()
        }
    }

    /// Create options for LU factorization.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::{FactorizeAlg, FactorizeOptions};
    ///
    /// let opts = FactorizeOptions::lu();
    /// assert_eq!(opts.alg, FactorizeAlg::LU);
    /// ```
    pub fn lu() -> Self {
        Self {
            alg: FactorizeAlg::LU,
            ..Default::default()
        }
    }

    /// Create options for CI factorization.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::{FactorizeAlg, FactorizeOptions};
    ///
    /// let opts = FactorizeOptions::ci();
    /// assert_eq!(opts.alg, FactorizeAlg::CI);
    /// ```
    pub fn ci() -> Self {
        Self {
            alg: FactorizeAlg::CI,
            ..Default::default()
        }
    }

    /// Set canonical direction.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::{Canonical, FactorizeOptions};
    ///
    /// let opts = FactorizeOptions::svd().with_canonical(Canonical::Right);
    /// assert_eq!(opts.canonical, Canonical::Right);
    /// ```
    pub fn with_canonical(mut self, canonical: Canonical) -> Self {
        self.canonical = canonical;
        self
    }

    /// Set the SVD truncation policy.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::{FactorizeOptions, SvdTruncationPolicy};
    ///
    /// let opts = FactorizeOptions::svd().with_svd_policy(SvdTruncationPolicy::new(1e-8));
    /// assert_eq!(opts.svd_policy, Some(SvdTruncationPolicy::new(1e-8)));
    /// ```
    pub fn with_svd_policy(mut self, policy: SvdTruncationPolicy) -> Self {
        self.svd_policy = Some(policy);
        self
    }

    /// Set the QR-specific relative tolerance.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::FactorizeOptions;
    ///
    /// let opts = FactorizeOptions::qr().with_qr_rtol(1e-8);
    /// assert_eq!(opts.qr_rtol, Some(1e-8));
    /// ```
    pub fn with_qr_rtol(mut self, rtol: f64) -> Self {
        self.qr_rtol = Some(rtol);
        self
    }

    /// Set maximum rank.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::FactorizeOptions;
    ///
    /// let opts = FactorizeOptions::svd().with_max_rank(10);
    /// assert_eq!(opts.max_rank, Some(10));
    /// ```
    pub fn with_max_rank(mut self, max_rank: usize) -> Self {
        self.max_rank = Some(max_rank);
        self
    }

    /// Validate that the selected fields make sense for the chosen algorithm.
    ///
    /// # Errors
    ///
    /// Returns [`FactorizeError::InvalidOptions`] if an algorithm is paired with
    /// unsupported algorithm-specific truncation settings.
    pub fn validate(&self) -> std::result::Result<(), FactorizeError> {
        match self.alg {
            FactorizeAlg::SVD => {
                if self.qr_rtol.is_some() {
                    return Err(FactorizeError::InvalidOptions(
                        "SVD factorization does not accept qr_rtol",
                    ));
                }
            }
            FactorizeAlg::QR => {
                if self.svd_policy.is_some() {
                    return Err(FactorizeError::InvalidOptions(
                        "QR factorization does not accept svd_policy",
                    ));
                }
            }
            FactorizeAlg::LU | FactorizeAlg::CI => {
                if self.svd_policy.is_some() {
                    return Err(FactorizeError::InvalidOptions(
                        "LU/CI factorization does not accept svd_policy",
                    ));
                }
                if self.qr_rtol.is_some() {
                    return Err(FactorizeError::InvalidOptions(
                        "LU/CI factorization does not accept qr_rtol",
                    ));
                }
            }
        }

        Ok(())
    }
}

/// Result of tensor factorization.
///
/// Contains the two factors, the bond index connecting them, and metadata
/// about the decomposition. The original tensor can be recovered (up to
/// truncation error) by contracting `left` and `right` along `bond_index`.
///
/// # Examples
///
/// ```
/// use tensor4all_core::{
///     factorize, DynIndex, FactorizeOptions, TensorContractionLike, TensorDynLen,
/// };
///
/// let i = DynIndex::new_dyn(3);
/// let j = DynIndex::new_dyn(4);
/// let data: Vec<f64> = (0..12).map(|x| x as f64).collect();
/// let tensor = TensorDynLen::from_dense(vec![i.clone(), j.clone()], data).unwrap();
///
/// let result = factorize(&tensor, &[i.clone()], &FactorizeOptions::svd()).unwrap();
///
/// // Contracting left * right recovers the original tensor
/// let recovered = result.left.contract_pair(&result.right).unwrap();
/// assert!(tensor.distance(&recovered).unwrap() < 1e-12);
///
/// // SVD provides singular values
/// assert!(result.singular_values.is_some());
/// assert_eq!(result.singular_values.as_ref().unwrap().len(), result.rank);
/// ```
#[derive(Debug, Clone)]
pub struct FactorizeResult<T: TensorIndex> {
    /// Left factor tensor.
    pub left: T,
    /// Right factor tensor.
    pub right: T,
    /// Bond index connecting left and right factors.
    pub bond_index: T::Index,
    /// Singular values (only for SVD).
    pub singular_values: Option<Vec<f64>>,
    /// Rank of the factorization.
    pub rank: usize,
}

// ============================================================================
// Contraction types
// ============================================================================

/// Linearization order used when fusing or unfusing multiple logical indices
/// into one physical index.
///
/// This matters for exact reshape-style operations such as replacing one fused
/// index with several unfused indices.
///
/// # Examples
///
/// ```
/// use tensor4all_core::LinearizationOrder;
///
/// assert_eq!(LinearizationOrder::ColumnMajor.as_str(), "column-major");
/// assert_eq!(LinearizationOrder::RowMajor.as_str(), "row-major");
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinearizationOrder {
    /// First index changes fastest.
    ColumnMajor,
    /// Last index changes fastest.
    RowMajor,
}

impl LinearizationOrder {
    /// Return a short human-readable description.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::LinearizationOrder;
    ///
    /// assert_eq!(LinearizationOrder::ColumnMajor.as_str(), "column-major");
    /// ```
    pub fn as_str(self) -> &'static str {
        match self {
            Self::ColumnMajor => "column-major",
            Self::RowMajor => "row-major",
        }
    }
}

// ============================================================================
// Capability traits (fully generic)
// ============================================================================

/// Generic adapter error used by vector-space implementations whose concrete
/// backend error is not part of their public API.
///
/// # Examples
///
/// ```
/// use tensor4all_core::TensorVectorSpaceError;
///
/// let error = TensorVectorSpaceError::from(anyhow::anyhow!("backend failed"));
/// assert!(error.to_string().contains("backend failed"));
/// ```
#[derive(Debug, thiserror::Error)]
#[error("tensor vector-space operation failed: {source}")]
pub struct TensorVectorSpaceError {
    #[source]
    source: Arc<dyn std::error::Error + Send + Sync + 'static>,
}

impl From<anyhow::Error> for TensorVectorSpaceError {
    fn from(source: anyhow::Error) -> Self {
        Self {
            source: Arc::from(source.into_boxed_dyn_error()),
        }
    }
}

/// Vector-space operations for iterative linear algebra over tensor-like values.
///
/// This trait intentionally does not require tensor contraction/einsum,
/// factorization, or tensor-network construction. Krylov solvers should depend
/// on this trait instead of [`TensorLike`] so block vectors and other abstract
/// state types do not have to provide unrelated tensor-network operations.
pub trait TensorVectorSpace: TensorIndex {
    /// Compute the squared Frobenius norm of the tensor.
    ///
    /// # Errors
    /// Returns backend or storage diagnostics when the norm cannot be
    /// evaluated.
    ///
    /// # Examples
    /// ```
    /// # fn main() -> anyhow::Result<()> {
    /// use tensor4all_core::{DynIndex, TensorDynLen, TensorVectorSpace};
    ///
    /// let index = DynIndex::new_dyn(2);
    /// let tensor = TensorDynLen::from_dense(vec![index], vec![3.0_f64, 4.0])?;
    /// assert!((TensorVectorSpace::norm_squared(&tensor)? - 25.0).abs() < 1e-12);
    /// # Ok(())
    /// # }
    /// ```
    fn norm_squared(&self) -> std::result::Result<f64, Self::Error>;

    /// Compute a linear combination: `a * self + b * other`.
    ///
    /// # Errors
    ///
    /// Returns `Self::Error` when the operands have incompatible index spaces
    /// or the underlying arithmetic/backend computation fails.
    fn axpby(
        &self,
        a: AnyScalar,
        other: &Self,
        b: AnyScalar,
    ) -> std::result::Result<Self, Self::Error>;

    /// Scalar multiplication.
    ///
    /// # Errors
    ///
    /// Returns `Self::Error` when the underlying arithmetic/backend
    /// computation fails.
    fn scale(&self, scalar: AnyScalar) -> std::result::Result<Self, Self::Error>;

    /// Inner product (dot product) of two tensors.
    ///
    /// Computes `⟨self, other⟩ = Σ conj(self)_i * other_i`.
    ///
    /// # Errors
    ///
    /// Returns `Self::Error` when the operands have incompatible index spaces
    /// or the underlying contraction/backend computation fails.
    fn inner_product(&self, other: &Self) -> std::result::Result<AnyScalar, Self::Error>;

    /// Compute the Frobenius norm of the tensor.
    ///
    /// # Errors
    /// Propagates failures from [`Self::norm_squared`].
    ///
    /// # Examples
    /// ```
    /// # fn main() -> anyhow::Result<()> {
    /// use tensor4all_core::{DynIndex, TensorDynLen, TensorVectorSpace};
    ///
    /// let index = DynIndex::new_dyn(2);
    /// let tensor = TensorDynLen::from_dense(vec![index], vec![3.0_f64, 4.0])?;
    /// assert!((TensorVectorSpace::norm(&tensor)? - 5.0).abs() < 1e-12);
    /// # Ok(())
    /// # }
    /// ```
    fn norm(&self) -> std::result::Result<f64, Self::Error> {
        Ok(self.norm_squared()?.sqrt())
    }

    /// Compute the maximum absolute value of all tensor elements.
    ///
    /// # Errors
    /// Returns backend or storage diagnostics when the maximum cannot be
    /// evaluated.
    ///
    /// # Examples
    /// ```
    /// # fn main() -> anyhow::Result<()> {
    /// use tensor4all_core::{DynIndex, TensorDynLen, TensorVectorSpace};
    ///
    /// let index = DynIndex::new_dyn(2);
    /// let tensor = TensorDynLen::from_dense(vec![index], vec![-3.0_f64, 2.0])?;
    /// assert!((TensorVectorSpace::maxabs(&tensor)? - 3.0).abs() < 1e-12);
    /// # Ok(())
    /// # }
    /// ```
    fn maxabs(&self) -> std::result::Result<f64, Self::Error>;

    /// Element-wise subtraction: `self - other`.
    ///
    /// # Errors
    ///
    /// Propagates failures from [`Self::axpby`].
    fn sub(&self, other: &Self) -> std::result::Result<Self, Self::Error> {
        self.axpby(AnyScalar::new_real(1.0), other, AnyScalar::new_real(-1.0))
    }

    /// Negate all elements: `-self`.
    ///
    /// # Errors
    ///
    /// Propagates failures from [`Self::scale`].
    fn neg(&self) -> std::result::Result<Self, Self::Error> {
        self.scale(AnyScalar::new_real(-1.0))
    }

    /// Approximate equality check (Julia `isapprox` semantics).
    ///
    /// # Errors
    /// Propagates failures from subtraction or norm evaluation.
    fn isapprox(
        &self,
        other: &Self,
        atol: f64,
        rtol: f64,
    ) -> std::result::Result<bool, Self::Error> {
        let diff = self.sub(other)?;
        let diff_norm = diff.norm()?;
        let self_norm = self.norm()?;
        let other_norm = other.norm()?;
        Ok(diff_norm <= atol.max(rtol * self_norm.max(other_norm)))
    }

    /// Validate structural consistency of this tensor-like vector.
    ///
    /// # Errors
    ///
    /// Returns `Self::Error` when the tensor's index space or shape is
    /// structurally inconsistent (for example index-space or dimension
    /// mismatches, or an invalid internal state). The default implementation
    /// always returns `Ok(())`.
    fn validate(&self) -> std::result::Result<(), Self::Error> {
        Ok(())
    }
}

/// Contraction/einsum-style operations for tensor-like values.
///
/// Types that only need vector-space algebra should not implement or require
/// this trait. Tree tensor-network algorithms should use this trait when they
/// truly need index-based contraction.
pub trait TensorContractionLike: TensorIndex {
    /// Tensor conjugate operation.
    fn conj(&self) -> Self;

    /// Direct sum of two tensors along specified index pairs.
    fn direct_sum(
        &self,
        other: &Self,
        pairs: &[(<Self as TensorIndex>::Index, <Self as TensorIndex>::Index)],
    ) -> Result<DirectSumResult<Self>>;

    /// Outer product (tensor product) of two tensors.
    fn outer_product(&self, other: &Self) -> Result<Self>;

    /// Permute tensor indices to match the specified order.
    fn permuteinds(&self, new_order: &[<Self as TensorIndex>::Index]) -> Result<Self>;

    /// Fuse local tensor indices into one replacement index.
    fn fuse_indices(
        &self,
        old_indices: &[<Self as TensorIndex>::Index],
        new_index: <Self as TensorIndex>::Index,
        order: LinearizationOrder,
    ) -> Result<Self>;

    /// Contract a connected tensor network over its contractable indices.
    fn contract(tensors: &[&Self]) -> Result<Self>;

    /// Contract this tensor with one other tensor using default pairwise semantics.
    fn contract_pair(&self, other: &Self) -> Result<Self> {
        Self::contract(&[self, other])
    }

    /// Validate structural consistency of this tensor.
    fn validate(&self) -> Result<()> {
        Ok(())
    }
}

/// Factorization operations for tensor-like values.
pub trait TensorFactorizationLike: TensorIndex {
    /// Factorize this tensor into left and right factors.
    fn factorize(
        &self,
        left_inds: &[<Self as TensorIndex>::Index],
        options: &FactorizeOptions,
    ) -> std::result::Result<FactorizeResult<Self>, FactorizeError>;

    /// Factorize this tensor without applying truncation controls.
    fn factorize_full_rank(
        &self,
        left_inds: &[<Self as TensorIndex>::Index],
        alg: FactorizeAlg,
        canonical: Canonical,
    ) -> std::result::Result<FactorizeResult<Self>, FactorizeError>;
}

/// Constructors and selection helpers for index-labelled tensors.
pub trait TensorConstructionLike: TensorContractionLike {
    /// Create a diagonal (Kronecker delta) tensor for a single index pair.
    fn diagonal(
        input_index: &<Self as TensorIndex>::Index,
        output_index: &<Self as TensorIndex>::Index,
    ) -> Result<Self>;

    /// Create a delta (identity) tensor as outer product of diagonals.
    fn delta(
        input_indices: &[<Self as TensorIndex>::Index],
        output_indices: &[<Self as TensorIndex>::Index],
    ) -> Result<Self> {
        if input_indices.len() != output_indices.len() {
            return Err(anyhow::anyhow!(
                "Number of input indices ({}) must match output indices ({})",
                input_indices.len(),
                output_indices.len()
            ));
        }

        if input_indices.is_empty() {
            return Self::scalar_one();
        }

        let mut result = Self::diagonal(&input_indices[0], &output_indices[0])?;
        for (inp, out) in input_indices[1..].iter().zip(output_indices[1..].iter()) {
            let diag = Self::diagonal(inp, out)?;
            result = result.outer_product(&diag)?;
        }
        Ok(result)
    }

    /// Create a scalar tensor with value 1.0.
    fn scalar_one() -> Result<Self>;

    /// Create a tensor filled with 1.0 for the given indices.
    fn ones(indices: &[<Self as TensorIndex>::Index]) -> Result<Self>;

    /// Select fixed coordinates for a subset of this tensor's external indices.
    fn select_indices(
        &self,
        selected_indices: &[<Self as TensorIndex>::Index],
        positions: &[usize],
    ) -> Result<Self> {
        anyhow::ensure!(
            selected_indices.len() == positions.len(),
            "selected_indices length {} does not match positions length {}",
            selected_indices.len(),
            positions.len()
        );
        if selected_indices.is_empty() {
            return Ok(self.clone());
        }

        let mut seen = HashSet::with_capacity(selected_indices.len());
        for (index, &position) in selected_indices.iter().zip(positions.iter()) {
            anyhow::ensure!(
                seen.insert(index.clone()),
                "selected index appears more than once"
            );
            anyhow::ensure!(
                position < index.dim(),
                "selected coordinate {} is out of range for index {:?} with dim {}",
                position,
                index,
                index.dim()
            );
        }

        let index_vals = selected_indices
            .iter()
            .cloned()
            .zip(positions.iter().copied())
            .collect::<Vec<_>>();
        let onehot = Self::onehot(&index_vals)?;
        Self::contract(&[self, &onehot])
    }

    /// Create a one-hot tensor with value 1.0 at the specified index positions.
    fn onehot(index_vals: &[(<Self as TensorIndex>::Index, usize)]) -> Result<Self>;
}

// ============================================================================
// TensorLike trait (fully generic composite)
// ============================================================================

/// Trait for tensor-like objects that expose external indices and support contraction.
///
/// This trait is **fully generic** (monomorphic), meaning it does not support
/// trait objects (`dyn TensorLike`). For heterogeneous tensor collections,
/// use an enum wrapper instead.
///
/// # Design Principles
///
/// - **Capability composition**: combines vector-space, factorization, construction, and contraction traits
/// - **Fully generic**: Uses associated type for `Index`, returns `Self`
/// - **Stable ordering**: `external_indices()` returns indices in deterministic order
/// - **No trait objects**: Requires `Sized`, cannot use `dyn TensorLike`
///
/// # Example
///
/// ```
/// use tensor4all_core::{DynIndex, TensorContractionLike, TensorDynLen};
///
/// fn contract_pair(a: &TensorDynLen, b: &TensorDynLen) -> anyhow::Result<TensorDynLen> {
///     Ok(<TensorDynLen as TensorContractionLike>::contract(&[a, b])?)
/// }
///
/// # fn main() -> anyhow::Result<()> {
/// let i = DynIndex::new_dyn(2);
/// let j = DynIndex::new_dyn(2);
/// let a = TensorDynLen::from_dense(
///     vec![i.clone(), j.clone()],
///     vec![1.0, 0.0, 0.0, 1.0],
/// )?;
/// let b = TensorDynLen::from_dense(vec![j.clone()], vec![2.0, 3.0])?;
///
/// let result = contract_pair(&a, &b)?;
/// assert_eq!(result.to_vec::<f64>()?, vec![2.0, 3.0]);
/// # Ok(())
/// # }
/// ```
///
/// # Heterogeneous Collections
///
/// For mixing different tensor types, define an enum:
///
/// ```
/// use tensor4all_core::{block_tensor::BlockTensor, DynIndex, TensorDynLen};
///
/// let i = DynIndex::new_dyn(2);
/// let dense = TensorDynLen::from_dense(vec![i.clone()], vec![1.0, 2.0]).unwrap();
/// let block = BlockTensor::new(vec![dense.clone()], (1, 1)).unwrap();
///
/// enum TensorNetwork {
///     Dense(TensorDynLen),
///     Block(BlockTensor<TensorDynLen>),
/// }
///
/// let network = TensorNetwork::Block(block);
/// assert!(matches!(network, TensorNetwork::Block(_)));
/// ```
///
/// # Supertrait
///
/// `TensorLike` extends several capability traits. Through those traits it provides:
/// - `external_indices()` - Get all external indices
/// - `num_external_indices()` - Count external indices
/// - `replaceind()` / `replaceinds()` - Replace indices
/// - vector-space operations such as `axpby`, `inner_product`, and `norm`
/// - tensor-network operations such as contraction, construction, and factorization
///
/// Use narrower traits such as [`TensorVectorSpace`] or
/// [`TensorContractionLike`] when an algorithm does not need the full surface.
pub trait TensorLike: TensorVectorSpace + TensorFactorizationLike + TensorConstructionLike {}

impl<T> TensorLike for T where
    T: TensorVectorSpace + TensorFactorizationLike + TensorConstructionLike
{
}

/// Result of direct sum operation.
///
/// Contains the resulting tensor and the new indices created for the summed
/// dimensions (one new index per pair in the input).
///
/// # Examples
///
/// ```
/// use tensor4all_core::{DynIndex, IndexLike, TensorContractionLike, TensorDynLen};
///
/// let i = DynIndex::new_dyn(2);
/// let j = DynIndex::new_dyn(3);
///
/// let a = TensorDynLen::from_dense(vec![i.clone()], vec![1.0, 2.0]).unwrap();
/// let b = TensorDynLen::from_dense(vec![j.clone()], vec![3.0, 4.0, 5.0]).unwrap();
///
/// let result = a.direct_sum(&b, &[(i.clone(), j.clone())]).unwrap();
///
/// // New index has dimension = 2 + 3 = 5
/// assert_eq!(result.new_indices.len(), 1);
/// assert_eq!(result.new_indices[0].dim(), 5);
/// assert_eq!(result.tensor.to_vec::<f64>().unwrap(), vec![1.0, 2.0, 3.0, 4.0, 5.0]);
/// ```
#[derive(Debug, Clone)]
pub struct DirectSumResult<T: TensorIndex> {
    /// The resulting tensor from direct sum.
    pub tensor: T,
    /// New indices created for the summed dimensions (one per pair).
    pub new_indices: Vec<T::Index>,
}

#[cfg(test)]
mod tests;
