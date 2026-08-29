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
use crate::truncation::{
    validate_svd_truncation_options, SvdTruncationOptionsError, SvdTruncationPolicy,
};
use num_complex::Complex64;
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
///     factorize, Canonical, DynIndex, FactorizeOptions, TensorContractionLike, IdxTensor,
/// };
///
/// let i = DynIndex::new_dyn(3);
/// let j = DynIndex::new_dyn(3);
/// let data: Vec<f64> = (0..9).map(|x| x as f64).collect();
/// let tensor = IdxTensor::from_dense(vec![i.clone(), j.clone()], data).unwrap();
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
        crate::MatrixCIError,
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
///     factorize, Canonical, DynIndex, FactorizeOptions, TensorContractionLike, IdxTensor,
/// };
///
/// let i = DynIndex::new_dyn(3);
/// let j = DynIndex::new_dyn(3);
/// let data: Vec<f64> = (0..9).map(|x| x as f64).collect();
/// let tensor = IdxTensor::from_dense(vec![i.clone(), j.clone()], data).unwrap();
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
/// - max_bond_dim: `None` (no rank limit)
/// - svd_policy: `None` (uses the SVD global default policy)
/// - qr_rtol: `None` (uses the QR global default tolerance)
///
/// # Field Interactions
///
/// - `svd_policy` is only valid for `FactorizeAlg::SVD`.
/// - `qr_rtol` is only valid for `FactorizeAlg::QR`.
/// - `max_bond_dim` is independent of the algorithm-specific tolerance settings.
///
/// # Examples
///
/// ```
/// use tensor4all_core::{
///     factorize, Canonical, DynIndex, FactorizeOptions, IdxTensor,
/// };
///
/// let i = DynIndex::new_dyn(4);
/// let j = DynIndex::new_dyn(4);
/// let mut data = vec![0.0_f64; 16];
/// data[0] = 1.0;  // rank-1 matrix
/// let tensor = IdxTensor::from_dense(vec![i.clone(), j.clone()], data).unwrap();
///
/// // SVD with an explicit policy
/// let opts = FactorizeOptions::svd()
///     .with_svd_policy(tensor4all_core::SvdTruncationPolicy::new(1e-10));
/// let result = factorize(&tensor, &[i.clone()], &opts).unwrap();
/// assert_eq!(result.rank, 1);
///
/// // QR with max-rank truncation
/// let opts = FactorizeOptions::qr().with_qr_rtol(1e-8).with_max_bond_dim(2);
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
    pub max_bond_dim: Option<usize>,
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
            max_bond_dim: None,
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
    /// let opts = FactorizeOptions::svd().with_max_bond_dim(10);
    /// assert_eq!(opts.max_bond_dim, Some(10));
    /// ```
    pub fn with_max_bond_dim(mut self, max_bond_dim: usize) -> Self {
        self.max_bond_dim = Some(max_bond_dim);
        self
    }

    /// Validate that the selected fields make sense for the chosen algorithm.
    ///
    /// Validates max bond dimension and explicit SVD policy thresholds through
    /// the shared [`validate_svd_truncation_options`] seam, then checks the
    /// algorithm/option compatibility rules.
    ///
    /// # Errors
    ///
    /// Returns [`FactorizeError::InvalidOptions`] if an algorithm is paired with
    /// unsupported algorithm-specific truncation settings, or if `max_bond_dim`
    /// is zero or an SVD policy threshold is non-finite/negative.
    pub fn validate(&self) -> std::result::Result<(), FactorizeError> {
        validate_svd_truncation_options(self.max_bond_dim, self.svd_policy).map_err(|error| {
            FactorizeError::InvalidOptions(match error {
                SvdTruncationOptionsError::ZeroMaxBondDim => "max_bond_dim must be at least 1",
                SvdTruncationOptionsError::InvalidThreshold(_) => {
                    "SVD truncation threshold must be finite and non-negative"
                }
            })
        })?;

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
///     factorize, DynIndex, FactorizeOptions, TensorContractionLike, IdxTensor,
/// };
///
/// let i = DynIndex::new_dyn(3);
/// let j = DynIndex::new_dyn(4);
/// let data: Vec<f64> = (0..12).map(|x| x as f64).collect();
/// let tensor = IdxTensor::from_dense(vec![i.clone(), j.clone()], data).unwrap();
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
    incremental_qr_state: Option<IncrementalQrState>,
}

#[derive(Debug, Clone)]
pub(crate) enum IncrementalQrState {
    F64(tensor4all_tensorbackend::IncrementalQr<f64>),
    C64(tensor4all_tensorbackend::IncrementalQr<Complex64>),
}

impl<T: TensorIndex> FactorizeResult<T> {
    /// Construct a factorization result without an algorithm-private update
    /// state.
    ///
    /// # Arguments
    /// * `left` - Left factor tensor.
    /// * `right` - Right factor tensor.
    /// * `bond_index` - Index connecting the two factors.
    /// * `singular_values` - Singular values when the result came from SVD.
    /// * `rank` - Number of columns in the factorization.
    ///
    /// # Returns
    /// A factorization result suitable for public factorization APIs. Native
    /// incremental QR callers attach their private update state afterward.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::{DynIndex, FactorizeResult, IdxTensor};
    ///
    /// let left_index = DynIndex::new_dyn(1);
    /// let bond = DynIndex::new_bond(1).unwrap();
    /// let left = IdxTensor::from_dense(vec![left_index.clone(), bond.clone()], vec![1.0]).unwrap();
    /// let right = IdxTensor::from_dense(vec![bond.clone()], vec![2.0]).unwrap();
    /// let result = FactorizeResult::new(left, right, bond, None, 1);
    /// assert_eq!(result.rank, 1);
    /// ```
    pub fn new(
        left: T,
        right: T,
        bond_index: T::Index,
        singular_values: Option<Vec<f64>>,
        rank: usize,
    ) -> Self {
        Self {
            left,
            right,
            bond_index,
            singular_values,
            rank,
            incremental_qr_state: None,
        }
    }

    pub(crate) fn with_incremental_qr_state(mut self, state: IncrementalQrState) -> Self {
        self.incremental_qr_state = Some(state);
        self
    }

    pub(crate) fn incremental_qr_state(&self) -> Option<&IncrementalQrState> {
        self.incremental_qr_state.as_ref()
    }
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
    ///
    /// Returns `Self::Error` when the norm cannot be evaluated (a materialization
    /// /// or backend failure).
    ///
    /// # Examples
    /// ```
    /// # fn main() -> anyhow::Result<()> {
    /// use tensor4all_core::{DynIndex, IdxTensor, TensorVectorSpace};
    ///
    /// let index = DynIndex::new_dyn(2);
    /// let tensor = IdxTensor::from_dense(vec![index], vec![3.0_f64, 4.0])?;
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
    /// (an index-space mismatch) or the underlying arithmetic or backend
    /// computation reports a failure.
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
    /// Returns `Self::Error` when the scalar coefficient is invalid for the
    /// tensor's scalar type or the underlying backend computation reports a
    /// failure.
    fn scale(&self, scalar: AnyScalar) -> std::result::Result<Self, Self::Error>;

    /// Inner product (dot product) of two tensors.
    ///
    /// Computes `⟨self, other⟩ = Σ conj(self)_i * other_i`.
    ///
    /// # Errors
    ///
    /// Returns `Self::Error` when the operands have incompatible index spaces
    /// (an index-space mismatch) or the underlying contraction or backend
    /// computation reports a failure.
    fn inner_product(&self, other: &Self) -> std::result::Result<AnyScalar, Self::Error>;

    /// Compute the Frobenius norm of the tensor.
    ///
    /// # Errors
    ///
    /// Returns `Self::Error` when the norm cannot be evaluated (a materialization
    /// /// or backend failure).
    ///
    /// # Examples
    /// ```
    /// # fn main() -> anyhow::Result<()> {
    /// use tensor4all_core::{DynIndex, IdxTensor, TensorVectorSpace};
    ///
    /// let index = DynIndex::new_dyn(2);
    /// let tensor = IdxTensor::from_dense(vec![index], vec![3.0_f64, 4.0])?;
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
    ///
    /// Returns `Self::Error` when the maximum cannot be evaluated (a
    /// /// materialization or backend failure).
    ///
    /// # Examples
    /// ```
    /// # fn main() -> anyhow::Result<()> {
    /// use tensor4all_core::{DynIndex, IdxTensor, TensorVectorSpace};
    ///
    /// let index = DynIndex::new_dyn(2);
    /// let tensor = IdxTensor::from_dense(vec![index], vec![-3.0_f64, 2.0])?;
    /// assert!((TensorVectorSpace::maxabs(&tensor)? - 3.0).abs() < 1e-12);
    /// # Ok(())
    /// # }
    /// ```
    fn maxabs(&self) -> std::result::Result<f64, Self::Error>;

    /// Element-wise subtraction: `self - other`.
    ///
    /// # Errors
    ///
    /// Returns `Self::Error` when the operand index spaces are incompatible
    /// (an index-space mismatch) or the underlying arithmetic or backend
    /// computation reports a failure; propagates failures from [`Self::axpby`].
    fn sub(&self, other: &Self) -> std::result::Result<Self, Self::Error> {
        self.axpby(AnyScalar::new_real(1.0), other, AnyScalar::new_real(-1.0))
    }

    /// Negate all elements: `-self`.
    ///
    /// # Errors
    ///
    /// Returns `Self::Error` when scaling reports a failure (an invalid
    /// scalar coefficient or a backend computation failure); propagates
    /// failures from [`Self::scale`].
    fn neg(&self) -> std::result::Result<Self, Self::Error> {
        self.scale(AnyScalar::new_real(-1.0))
    }

    /// Approximate equality check (Julia `isapprox` semantics).
    ///
    /// # Errors
    /// Returns `Self::Error` when the operands have incompatible index spaces
    /// (an index-space mismatch) or when norm evaluation reports a failure;
    /// propagates failures from subtraction and norm evaluation.
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
    ///
    /// # Errors
    ///
    /// Returns `Self::Error` when the summed index spaces are incompatible
    /// (an index-space mismatch) or the underlying construction reports a
    /// failure.
    fn direct_sum(
        &self,
        other: &Self,
        pairs: &[(<Self as TensorIndex>::Index, <Self as TensorIndex>::Index)],
    ) -> std::result::Result<DirectSumResult<Self>, Self::Error>;

    /// Outer product (tensor product) of two tensors.
    ///
    /// # Errors
    ///
    /// Returns `Self::Error` when the two tensors share contractable indices
    /// (a shared-index mismatch) or the underlying construction reports a
    /// failure.
    fn outer_product(&self, other: &Self) -> std::result::Result<Self, Self::Error>;

    /// Permute tensor indices to match the specified order.
    ///
    /// # Errors
    ///
    /// Returns `Self::Error` when `new_order` does not contain exactly the
    /// tensor's external indices (an index-set mismatch or a missing-index
    /// failure).
    fn permuteinds(
        &self,
        new_order: &[<Self as TensorIndex>::Index],
    ) -> std::result::Result<Self, Self::Error>;

    /// Fuse local tensor indices into one replacement index.
    ///
    /// # Errors
    ///
    /// Returns `Self::Error` when `old_indices` is not a subset of the
    /// tensor's external indices (a missing-index failure) or the fused
    /// dimension product overflows (an overflow failure).
    fn fuse_indices(
        &self,
        old_indices: &[<Self as TensorIndex>::Index],
        new_index: <Self as TensorIndex>::Index,
        order: LinearizationOrder,
    ) -> std::result::Result<Self, Self::Error>;

    /// Contract a connected tensor network over its contractable indices.
    ///
    /// # Errors
    ///
    /// Returns `Self::Error` when the network is disconnected (a
    /// disconnected-network failure), when indices are incompatible (an
    /// index-space mismatch), or when the underlying contraction reports a
    /// failure.
    fn contract(tensors: &[&Self]) -> std::result::Result<Self, Self::Error>;

    /// Contract tensors while preserving selected shared indices as batch axes.
    ///
    /// This seam was introduced for the batched SRC sketch. The paper basis is
    /// Algorithm 1's independent probe columns; the retained-index execution
    /// and fallback contract are tensor4all-specific `[AI-Supplied]` plumbing.
    ///
    /// A retained index is matched elementwise across operands instead of being
    /// summed. This is useful for evaluating several independent contractions
    /// in one backend call, such as a batch of SRC probe columns. Implementations
    /// that do not support retained batch axes may return an operation error.
    ///
    /// # Arguments
    /// * `tensors` - Connected tensor operands to contract.
    /// * `retained_indices` - Shared indices to preserve as output batch axes.
    ///
    /// # Returns
    /// The contracted tensor with each retained index present once in its output
    /// index list. An empty `retained_indices` list has the same meaning as
    /// [`Self::contract`].
    ///
    /// # Errors
    ///
    /// Returns `Self::Error` when the operands are disconnected, an operand
    /// index is missing or incompatible, or the retained-index operation is
    /// unsupported by the tensor type.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::{DynIndex, IdxTensor, TensorContractionLike};
    ///
    /// let batch = DynIndex::new_dyn(2);
    /// let contracted = DynIndex::new_dyn(2);
    /// let left = IdxTensor::from_dense(
    ///     vec![batch.clone(), contracted.clone()],
    ///     vec![1.0_f64, 2.0, 3.0, 4.0],
    /// )
    /// .unwrap();
    /// let right = IdxTensor::from_dense(
    ///     vec![batch.clone(), contracted],
    ///     vec![2.0_f64, 3.0, 4.0, 5.0],
    /// )
    /// .unwrap();
    /// let result = IdxTensor::contract_retaining_indices(&[&left, &right], &[batch]).unwrap();
    /// assert_eq!(result.dims(), vec![2]);
    /// assert_eq!(result.to_vec::<f64>().unwrap(), vec![14.0, 26.0]);
    /// ```
    fn contract_retaining_indices(
        tensors: &[&Self],
        retained_indices: &[<Self as TensorIndex>::Index],
    ) -> std::result::Result<Self, Self::Error> {
        if retained_indices.is_empty() {
            return Self::contract(tensors);
        }
        Err(anyhow::anyhow!(
            "{} does not support retained contraction indices",
            std::any::type_name::<Self>()
        )
        .into())
    }

    /// Contract this tensor with one other tensor using default pairwise semantics.
    ///
    /// # Errors
    ///
    /// Returns `Self::Error` when the pair is disconnected or has
    /// incompatible indices (an index-space mismatch); propagates failures
    /// from [`Self::contract`].
    fn contract_pair(&self, other: &Self) -> std::result::Result<Self, Self::Error> {
        Self::contract(&[self, other])
    }

    /// Validate structural consistency of this tensor.
    ///
    /// # Errors
    ///
    /// Returns `Self::Error` when the tensor's index space or shape is
    /// structurally inconsistent (an index-space mismatch or an invalid
    /// internal state). The default implementation always returns `Ok(())`.
    fn validate(&self) -> std::result::Result<(), Self::Error> {
        Ok(())
    }
}

/// Factorization operations for tensor-like values.
pub trait TensorFactorizationLike: TensorIndex {
    /// Factorize this tensor into left and right factors.
    /// # Errors
    ///
    /// Returns `FactorizeError` when the factorization fails (a non-convergence,
    /// /// singular, or unsupported-storage failure).
    ///
    fn factorize(
        &self,
        left_inds: &[<Self as TensorIndex>::Index],
        options: &FactorizeOptions,
    ) -> std::result::Result<FactorizeResult<Self>, FactorizeError>;

    /// Factorize this tensor using policy-aware automatic SVD/eigen selection.
    ///
    /// Implementations may use Hermitian Gram eigendecomposition when the
    /// requested SVD policy is numerically suitable. The default delegates to
    /// [`Self::factorize`], preserving the existing behavior for tensor types
    /// without an automatic eigendecomposition path.
    ///
    /// # Arguments
    /// * `left_inds` - Indices to place on the left side of the split.
    /// * `options` - Ordinary SVD factorization options, including canonical
    ///   direction, truncation policy, and maximum bond dimension.
    ///
    /// # Returns
    /// The same factorization result as [`Self::factorize`], with singular
    /// values populated for SVD-compatible implementations.
    ///
    /// # Errors
    /// Returns [`FactorizeError::InvalidOptions`] when `options.alg` is not
    /// [`FactorizeAlg::SVD`], or another [`FactorizeError`] when factorization
    /// fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::{
    ///     DynIndex, FactorizeOptions, IdxTensor, SvdTruncationPolicy,
    ///     TensorFactorizationLike,
    /// };
    ///
    /// let left = DynIndex::new_dyn(2);
    /// let right = DynIndex::new_dyn(2);
    /// let tensor = IdxTensor::from_dense(
    ///     vec![left.clone(), right],
    ///     vec![1.0_f64, 0.0, 0.0, 1.0e-3],
    /// )?;
    /// let options = FactorizeOptions::svd()
    ///     .with_svd_policy(SvdTruncationPolicy::new(1.0e-2));
    /// let result = tensor.factorize_auto(&[left], &options)?;
    /// assert_eq!(result.rank, 1);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    fn factorize_auto(
        &self,
        left_inds: &[<Self as TensorIndex>::Index],
        options: &FactorizeOptions,
    ) -> std::result::Result<FactorizeResult<Self>, FactorizeError> {
        if options.alg != FactorizeAlg::SVD {
            return Err(FactorizeError::InvalidOptions(
                "automatic factorization only supports SVD options",
            ));
        }
        self.factorize(left_inds, options)
    }

    /// Factorize this tensor without applying truncation controls.
    /// # Errors
    ///
    /// Returns `FactorizeError` when the factorization fails (a non-convergence,
    /// /// singular, or unsupported-storage failure).
    ///
    fn factorize_full_rank(
        &self,
        left_inds: &[<Self as TensorIndex>::Index],
        alg: FactorizeAlg,
        canonical: Canonical,
    ) -> std::result::Result<FactorizeResult<Self>, FactorizeError>;

    /// Factorize a probe prefix, optionally extending an existing QR prefix.
    ///
    /// The append semantics are based on Appendix C of
    /// Camaño--Epperly--Tropp and the author's `IncrementalQR.append`; the
    /// generic trait/default implementation is repository plumbing and is
    /// labelled `[AI-Supplied]` in the audit.
    ///
    /// `all_columns` contains the complete prefix requested by the caller;
    /// `appended_columns` contains only the newly added columns when
    /// `previous` is present. Native dense implementations may use the
    /// previous factors and append only the new block. The default is a
    /// correctness fallback that factorizes the complete prefix from scratch.
    ///
    /// # Arguments
    /// * `previous` - Factors for the preceding prefix, or `None` for the first
    ///   prefix.
    /// * `all_columns` - All probe-column tensors in the new prefix.
    /// * `appended_columns` - Only the columns appended since `previous`.
    /// * `left_inds` - Tensor indices that form the matrix rows.
    ///
    /// # Returns
    /// A full-rank left-canonical QR factorization of the requested prefix.
    ///
    /// # Errors
    /// Returns [`FactorizeError`] when the prefix cannot be stacked or QR
    /// factorized.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::{DynIndex, IdxTensor, TensorFactorizationLike};
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let row = DynIndex::new_dyn(2);
    /// let first = IdxTensor::from_dense(
    ///     vec![row.clone()],
    ///     vec![1.0, 0.0],
    /// )?;
    /// let second = IdxTensor::from_dense(
    ///     vec![row.clone()],
    ///     vec![0.0, 1.0],
    /// )?;
    /// let result = <IdxTensor as TensorFactorizationLike>::factorize_probe_columns_incremental(
    ///     None,
    ///     &[&first, &second],
    ///     &[&first, &second],
    ///     &[row],
    /// )?;
    /// assert_eq!(result.rank, 2);
    /// assert_eq!(result.left.indices().len(), 2);
    /// # Ok(())
    /// # }
    /// ```
    fn factorize_probe_columns_incremental(
        _previous: Option<&FactorizeResult<Self>>,
        all_columns: &[&Self],
        _appended_columns: &[&Self],
        left_inds: &[<Self as TensorIndex>::Index],
    ) -> std::result::Result<FactorizeResult<Self>, FactorizeError>
    where
        Self: TensorVectorSpace + TensorConstructionLike,
    {
        let batch = <Self as TensorIndex>::Index::new_link(all_columns.len())
            .map_err(FactorizeError::ComputationError)?;
        let sketch = Self::stack_along_new_index(all_columns, batch, -1)
            .map_err(|error| FactorizeError::ComputationError(anyhow::Error::new(error)))?;
        sketch.factorize_full_rank(left_inds, FactorizeAlg::QR, Canonical::Left)
    }

    /// Evaluate the SRC adaptive stopping estimator for a small QR factor.
    ///
    /// The tensor must represent a square upper-triangular `R` matrix in
    /// column-major order. Implementations that do not expose their small
    /// matrix storage return an unsupported-storage error.
    ///
    /// # Errors
    ///
    /// Returns [`FactorizeError`] when the tensor is not a supported QR factor,
    /// when its storage cannot be read as a dense matrix, or when the backend
    /// estimator rejects the matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::{DynIndex, IdxTensor, TensorFactorizationLike};
    ///
    /// let row = DynIndex::new_dyn(2);
    /// let column = DynIndex::new_dyn(2);
    /// let r = IdxTensor::from_dense(
    ///     vec![row, column],
    ///     vec![2.0_f64, 0.0, 1.0, 3.0],
    /// ).unwrap();
    /// let estimate = r.src_error_estimate().unwrap();
    /// assert!(estimate.error.is_finite());
    /// assert!(estimate.norm.is_finite());
    /// ```
    fn src_error_estimate(
        &self,
    ) -> std::result::Result<tensor4all_tensorbackend::SrcErrorEstimate, FactorizeError> {
        Err(FactorizeError::UnsupportedStorage(
            "SRC adaptive estimation is not supported for this tensor type",
        ))
    }
}

/// Constructors and selection helpers for index-labelled tensors.
pub trait TensorConstructionLike: TensorContractionLike {
    /// Create a diagonal (Kronecker delta) tensor for a single index pair.
    ///
    /// # Errors
    ///
    /// Returns `Self::Error` when the input and output indices have unequal
    /// dimensions (a shape mismatch) or the underlying construction
    /// reports a failure.
    fn diagonal(
        input_index: &<Self as TensorIndex>::Index,
        output_index: &<Self as TensorIndex>::Index,
    ) -> std::result::Result<Self, Self::Error>;

    /// Create a delta (identity) tensor as outer product of diagonals.
    ///
    /// # Errors
    ///
    /// Returns `Self::Error` when the input and output index lists differ in
    /// length (a length mismatch) or when a constituent diagonal or outer
    /// product reports a failure; propagates failures from [`Self::diagonal`],
    /// [`Self::scalar_one`], and [`Self::outer_product`].
    fn delta(
        input_indices: &[<Self as TensorIndex>::Index],
        output_indices: &[<Self as TensorIndex>::Index],
    ) -> std::result::Result<Self, Self::Error> {
        if input_indices.len() != output_indices.len() {
            return Err(anyhow::anyhow!(
                "Number of input indices ({}) must match output indices ({})",
                input_indices.len(),
                output_indices.len()
            )
            .into());
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
    ///
    /// # Errors
    ///
    /// Returns `Self::Error` when the scalar type does not support the
    /// required construction (an invalid scalar dtype or a backend
    /// construction failure).
    fn scalar_one() -> std::result::Result<Self, Self::Error>;

    /// Create a tensor filled with 1.0 for the given indices.
    ///
    /// # Errors
    ///
    /// Returns `Self::Error` when an index dimension product overflows (an
    /// overflow failure) or the underlying construction reports a failure.
    fn ones(indices: &[<Self as TensorIndex>::Index]) -> std::result::Result<Self, Self::Error>;

    /// Construct a tensor from a column-major dense payload.
    ///
    /// Implementations with a native dense storage path should override this
    /// method. The default preserves compatibility for tensor types that only
    /// expose one-hot construction, at the cost of constructing a sparse sum
    /// of one-hot tensors.
    ///
    /// # Arguments
    /// * `indices` - External indices in the intended column-major axis order.
    /// * `data` - Dense values in column-major order; its length must equal the
    ///   product of the index dimensions.
    ///
    /// # Errors
    ///
    /// Returns `Self::Error` when the input payload length does not match the
    /// product of the index dimensions, that index-dimension product would
    /// overflow `usize`, or an underlying tensor construction operation fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::{AnyScalar, DynIndex, IdxTensor, TensorConstructionLike};
    ///
    /// let index = DynIndex::new_dyn(2);
    /// let tensor = <IdxTensor as TensorConstructionLike>::from_dense_any(
    ///     vec![index],
    ///     vec![AnyScalar::new_real(2.0), AnyScalar::new_real(3.0)],
    /// )
    /// .unwrap();
    /// assert_eq!(tensor.to_vec::<f64>().unwrap(), vec![2.0, 3.0]);
    /// ```
    fn from_dense_any(
        indices: Vec<<Self as TensorIndex>::Index>,
        data: Vec<AnyScalar>,
    ) -> std::result::Result<Self, Self::Error>
    where
        Self: TensorVectorSpace,
    {
        let expected_len = indices.iter().try_fold(1usize, |size, index| {
            size.checked_mul(index.dim())
                .ok_or_else(|| anyhow::anyhow!("dense tensor shape product overflows usize"))
        })?;
        if data.len() != expected_len {
            return Err(anyhow::anyhow!(
                "dense tensor payload has length {}, expected {}",
                data.len(),
                expected_len
            )
            .into());
        }

        let mut result = Self::ones(&indices)?.scale(AnyScalar::new_real(0.0))?;
        let one = AnyScalar::new_real(1.0);
        for (linear, value) in data.into_iter().enumerate() {
            if value.is_zero() {
                continue;
            }
            let mut remainder = linear;
            let mut index_vals = Vec::with_capacity(indices.len());
            for index in &indices {
                let dim = index.dim();
                let position = remainder % dim;
                remainder /= dim;
                index_vals.push((index.clone(), position));
            }
            let basis = Self::onehot(&index_vals)?;
            let term = basis.scale(value)?;
            result = result.axpby(one.clone(), &term, one.clone())?;
        }
        Ok(result)
    }

    /// Stack tensors along a newly created batch index.
    ///
    /// Implementations with a native batch stack should override this method.
    /// The default constructs the batch by outer products with one-hot batch
    /// vectors, which is correct but intended only as a compatibility path.
    ///
    /// # Arguments
    /// * `tensors` - Non-empty tensors with identical external index order.
    /// * `new_index` - Fresh index whose dimension equals `tensors.len()`.
    /// * `axis` - Insertion axis; negative axes count from the end, so `-1`
    ///   appends the batch axis.
    ///
    /// # Errors
    /// Returns `Self::Error` when tensors are empty, their index orders differ,
    /// the batch dimension is wrong, the axis is invalid, or construction
    /// fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::{DynIndex, IdxTensor, TensorConstructionLike};
    ///
    /// let index = DynIndex::new_dyn(2);
    /// let batch = DynIndex::new_dyn(2);
    /// let first = IdxTensor::from_dense(vec![index.clone()], vec![1.0, 2.0]).unwrap();
    /// let second = IdxTensor::from_dense(vec![index.clone()], vec![3.0, 4.0]).unwrap();
    /// let stacked = <IdxTensor as TensorConstructionLike>::stack_along_new_index(
    ///     &[&first, &second],
    ///     batch.clone(),
    ///     -1,
    /// )
    /// .unwrap();
    /// assert_eq!(stacked.indices(), &[index, batch]);
    /// assert_eq!(stacked.to_vec::<f64>().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
    /// ```
    fn stack_along_new_index(
        tensors: &[&Self],
        new_index: <Self as TensorIndex>::Index,
        axis: isize,
    ) -> std::result::Result<Self, Self::Error>
    where
        Self: TensorVectorSpace,
    {
        let first = tensors
            .first()
            .ok_or_else(|| anyhow::anyhow!("stack_along_new_index requires at least one tensor"))?;
        if new_index.dim() != tensors.len() {
            return Err(anyhow::anyhow!(
                "stack_along_new_index batch dimension {} does not match tensor count {}",
                new_index.dim(),
                tensors.len()
            )
            .into());
        }
        let base_indices = first.external_indices();
        for tensor in tensors.iter().skip(1) {
            if tensor.external_indices() != base_indices {
                return Err(anyhow::anyhow!(
                    "stack_along_new_index tensors must have identical index order"
                )
                .into());
            }
        }

        let result_rank = base_indices.len() + 1;
        let insertion_axis = if axis < 0 {
            result_rank as isize + axis
        } else {
            axis
        };
        if !(0..=base_indices.len() as isize).contains(&insertion_axis) {
            return Err(anyhow::anyhow!(
                "stack_along_new_index axis {} is invalid for rank {}",
                axis,
                base_indices.len()
            )
            .into());
        }

        let batch_one = AnyScalar::new_real(1.0);
        let mut result: Option<Self> = None;
        for (position, tensor) in tensors.iter().enumerate() {
            let batch = Self::onehot(&[(new_index.clone(), position)])?;
            let term = tensor.outer_product(&batch)?;
            result = Some(match result {
                Some(current) => current.axpby(batch_one.clone(), &term, batch_one.clone())?,
                None => term,
            });
        }
        let mut desired = base_indices;
        desired.insert(insertion_axis as usize, new_index);
        let result = result
            .ok_or_else(|| anyhow::anyhow!("stack_along_new_index requires at least one tensor"))?;
        if result.external_indices() == desired {
            Ok(result)
        } else {
            result.permuteinds(&desired)
        }
    }

    /// Concatenate tensors whose selected axes are replaced by one new index.
    ///
    /// The tensors must have the same index order away from the selected axis.
    /// Each tensor may use a distinct source index at that axis; the source
    /// axes are copied in tensor order into `new_index`. This is the batched
    /// counterpart to appending column blocks without recomputing the old
    /// columns.
    ///
    /// # Arguments
    /// * `tensors` - Non-empty tensors with matching non-concatenated axes.
    /// * `source_indices` - One axis to concatenate for each tensor.
    /// * `new_index` - Fresh output axis whose dimension is the sum of source
    ///   dimensions.
    ///
    /// # Errors
    ///
    /// Returns `Self::Error` when the input list is empty; the tensor and
    /// source-index counts do not match; a source index is missing; source axes
    /// occupy incompatible positions; non-concatenated indices are
    /// incompatible; the source-dimension sum overflows; or the new index
    /// dimension does not match that sum.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::{DynIndex, IdxTensor, TensorConstructionLike};
    ///
    /// let row = DynIndex::new_dyn(2);
    /// let first_batch = DynIndex::new_link(1).unwrap();
    /// let second_batch = DynIndex::new_link(2).unwrap();
    /// let combined = DynIndex::new_link(3).unwrap();
    /// let first = IdxTensor::from_dense(
    ///     vec![row.clone(), first_batch.clone()],
    ///     vec![1.0_f64, 2.0],
    /// ).unwrap();
    /// let second = IdxTensor::from_dense(
    ///     vec![row.clone(), second_batch.clone()],
    ///     vec![3.0, 4.0, 5.0, 6.0],
    /// ).unwrap();
    /// let result = <IdxTensor as TensorConstructionLike>::concatenate_along_new_index(
    ///     &[&first, &second],
    ///     &[first_batch, second_batch],
    ///     combined.clone(),
    /// ).unwrap();
    /// assert_eq!(result.indices(), &[row, combined]);
    /// assert_eq!(result.to_vec::<f64>().unwrap(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    /// ```
    fn concatenate_along_new_index(
        tensors: &[&Self],
        source_indices: &[<Self as TensorIndex>::Index],
        new_index: <Self as TensorIndex>::Index,
    ) -> std::result::Result<Self, Self::Error>
    where
        Self: TensorVectorSpace,
    {
        let first = tensors
            .first()
            .ok_or_else(|| anyhow::anyhow!("concatenate requires at least one tensor"))?;
        if tensors.len() != source_indices.len() {
            return Err(anyhow::anyhow!(
                "concatenate tensor count {} does not match source-index count {}",
                tensors.len(),
                source_indices.len()
            )
            .into());
        }
        let first_axis = first
            .external_indices()
            .iter()
            .position(|index| index == &source_indices[0])
            .ok_or_else(|| anyhow::anyhow!("concatenate source index is not present"))?;
        let mut total_dim = 0usize;
        for (tensor, source) in tensors.iter().zip(source_indices) {
            let axis = tensor
                .external_indices()
                .iter()
                .position(|index| index == source)
                .ok_or_else(|| anyhow::anyhow!("concatenate source index is not present"))?;
            if axis != first_axis {
                return Err(
                    anyhow::anyhow!("concatenate source axes must have the same position").into(),
                );
            }
            let indices = tensor.external_indices();
            let first_indices = first.external_indices();
            if indices.len() != first_indices.len()
                || indices.iter().zip(first_indices).enumerate().any(
                    |(position, (actual, expected))| position != first_axis && *actual != expected,
                )
            {
                return Err(anyhow::anyhow!(
                    "concatenate tensors must match away from the source axis"
                )
                .into());
            }
            total_dim = total_dim
                .checked_add(source.dim())
                .ok_or_else(|| anyhow::anyhow!("concatenate dimension overflow"))?;
        }
        if new_index.dim() != total_dim {
            return Err(anyhow::anyhow!(
                "concatenate output dimension {} does not match source dimension sum {}",
                new_index.dim(),
                total_dim
            )
            .into());
        }

        let mut slices = Vec::with_capacity(total_dim);
        for (tensor, source) in tensors.iter().zip(source_indices) {
            for position in 0..source.dim() {
                slices.push(tensor.select_indices(std::slice::from_ref(source), &[position])?);
            }
        }
        let slice_refs = slices.iter().collect::<Vec<_>>();
        Self::stack_along_new_index(&slice_refs, new_index, first_axis as isize)
    }

    /// Select fixed coordinates for a subset of this tensor's external indices.
    ///
    /// # Errors
    ///
    /// Returns `Self::Error` when `selected_indices` and `positions` differ in
    /// length (a length mismatch), when an index is selected more than once
    /// (a duplicate-index failure), when a coordinate is out of range (an
    /// out of bounds failure), or when the underlying one-hot construction or
    /// contraction reports a failure; propagates failures from
    /// [`Self::onehot`] and [`Self::contract`].
    fn select_indices(
        &self,
        selected_indices: &[<Self as TensorIndex>::Index],
        positions: &[usize],
    ) -> std::result::Result<Self, Self::Error> {
        if selected_indices.len() != positions.len() {
            return Err(anyhow::anyhow!(
                "selected_indices length {} does not match positions length {}",
                selected_indices.len(),
                positions.len()
            )
            .into());
        }
        if selected_indices.is_empty() {
            return Ok(self.clone());
        }

        let mut seen = HashSet::with_capacity(selected_indices.len());
        for (index, &position) in selected_indices.iter().zip(positions.iter()) {
            if !seen.insert(index.clone()) {
                return Err(anyhow::anyhow!("selected index appears more than once").into());
            }
            if position >= index.dim() {
                return Err(anyhow::anyhow!(
                    "selected coordinate {} is out of range for index {:?} with dim {}",
                    position,
                    index,
                    index.dim()
                )
                .into());
            }
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
    ///
    /// # Errors
    ///
    /// Returns `Self::Error` when a position is out of range for its index
    /// (an out of bounds failure) or the underlying construction reports a
    /// failure.
    fn onehot(
        index_vals: &[(<Self as TensorIndex>::Index, usize)],
    ) -> std::result::Result<Self, Self::Error>;
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
/// use tensor4all_core::{DynIndex, TensorContractionLike, IdxTensor};
///
/// fn contract_pair(a: &IdxTensor, b: &IdxTensor) -> anyhow::Result<IdxTensor> {
///     Ok(<IdxTensor as TensorContractionLike>::contract(&[a, b])?)
/// }
///
/// # fn main() -> anyhow::Result<()> {
/// let i = DynIndex::new_dyn(2);
/// let j = DynIndex::new_dyn(2);
/// let a = IdxTensor::from_dense(
///     vec![i.clone(), j.clone()],
///     vec![1.0, 0.0, 0.0, 1.0],
/// )?;
/// let b = IdxTensor::from_dense(vec![j.clone()], vec![2.0, 3.0])?;
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
/// use tensor4all_core::{block_tensor::BlockTensor, DynIndex, IdxTensor};
///
/// let i = DynIndex::new_dyn(2);
/// let dense = IdxTensor::from_dense(vec![i.clone()], vec![1.0, 2.0]).unwrap();
/// let block = BlockTensor::new(vec![dense.clone()], (1, 1)).unwrap();
///
/// enum TensorNetwork {
///     Dense(IdxTensor),
///     Block(BlockTensor<IdxTensor>),
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
/// - `replaceind()` / `replace_indices()` - Replace indices
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
/// use tensor4all_core::{DynIndex, IndexLike, TensorContractionLike, IdxTensor};
///
/// let i = DynIndex::new_dyn(2);
/// let j = DynIndex::new_dyn(3);
///
/// let a = IdxTensor::from_dense(vec![i.clone()], vec![1.0, 2.0]).unwrap();
/// let b = IdxTensor::from_dense(vec![j.clone()], vec![3.0, 4.0, 5.0]).unwrap();
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
