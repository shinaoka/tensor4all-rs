//! Truncation policy types for decomposition algorithms.
//!
//! This module keeps algorithm selection separate from algorithm-specific
//! truncation semantics. SVD-based routines use [`SvdTruncationPolicy`],
//! while QR and other decompositions keep their own option types.

use thiserror::Error;

/// Decomposition/factorization algorithm.
///
/// This enum unifies the algorithm choices across different crates
/// (`tensor4all-core`, `tensor4all-treetn`, etc.).
///
/// # Examples
///
/// ```
/// use tensor4all_core::DecompositionAlg;
///
/// assert_eq!(DecompositionAlg::default(), DecompositionAlg::SVD);
/// assert!(DecompositionAlg::SVD.is_svd_based());
/// assert!(DecompositionAlg::SVD.is_orthogonal());
/// assert!(!DecompositionAlg::LU.is_orthogonal());
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DecompositionAlg {
    /// Singular Value Decomposition (optimal truncation).
    #[default]
    SVD,
    /// Randomized SVD (faster for large matrices).
    RSVD,
    /// QR decomposition.
    QR,
    /// Rank-revealing LU decomposition.
    LU,
    /// Cross Interpolation.
    CI,
}

impl DecompositionAlg {
    /// Check if this algorithm is SVD-based (SVD or RSVD).
    #[must_use]
    pub fn is_svd_based(&self) -> bool {
        matches!(self, Self::SVD | Self::RSVD)
    }

    /// Check if this algorithm provides orthogonal factors.
    #[must_use]
    pub fn is_orthogonal(&self) -> bool {
        matches!(self, Self::SVD | Self::RSVD | Self::QR)
    }
}

/// Threshold scaling for SVD truncation.
///
/// Relative thresholds compare against a scale derived from the singular values.
/// Absolute thresholds compare directly against the configured cutoff.
///
/// # Examples
///
/// ```
/// use tensor4all_core::ThresholdScale;
///
/// assert_eq!(ThresholdScale::default(), ThresholdScale::Relative);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ThresholdScale {
    /// Compare against a singular-value-derived reference scale.
    #[default]
    Relative,
    /// Compare directly against the configured threshold.
    Absolute,
}

/// Singular-value-derived quantity used for truncation.
///
/// # Examples
///
/// ```
/// use tensor4all_core::SingularValueMeasure;
///
/// assert_eq!(SingularValueMeasure::default(), SingularValueMeasure::Value);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SingularValueMeasure {
    /// Compare using singular values `σ_i`.
    #[default]
    Value,
    /// Compare using squared singular values `σ_i²`.
    SquaredValue,
}

/// Rule used to map singular values to a retained rank.
///
/// # Examples
///
/// ```
/// use tensor4all_core::TruncationRule;
///
/// assert_eq!(TruncationRule::default(), TruncationRule::PerValue);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TruncationRule {
    /// Keep values whose individual measure exceeds the threshold rule.
    #[default]
    PerValue,
    /// Discard a suffix while the cumulative discarded measure stays below
    /// the threshold rule.
    DiscardedTailSum,
}

/// Explicit truncation policy for SVD-based decompositions.
///
/// Use this type when you need to describe how singular values are measured,
/// scaled, and turned into a retained rank. [`SvdOptions`](crate::SvdOptions)
/// carries this policy plus an independent `max_bond_dim` cap.
///
/// # Examples
///
/// ```
/// use tensor4all_core::{
///     SingularValueMeasure, SvdTruncationPolicy, ThresholdScale, TruncationRule,
/// };
///
/// let policy = SvdTruncationPolicy::new(1e-12);
/// assert_eq!(policy.scale, ThresholdScale::Relative);
/// assert_eq!(policy.measure, SingularValueMeasure::Value);
/// assert_eq!(policy.rule, TruncationRule::PerValue);
///
/// let tail_policy = SvdTruncationPolicy::new(1e-8)
///     .with_absolute()
///     .with_squared_values()
///     .with_discarded_tail_sum();
/// assert_eq!(tail_policy.scale, ThresholdScale::Absolute);
/// assert_eq!(tail_policy.measure, SingularValueMeasure::SquaredValue);
/// assert_eq!(tail_policy.rule, TruncationRule::DiscardedTailSum);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SvdTruncationPolicy {
    /// Threshold value used by the selected scale/rule combination.
    pub threshold: f64,
    /// Whether the threshold is interpreted relatively or absolutely.
    pub scale: ThresholdScale,
    /// Whether the policy measures singular values or squared singular values.
    pub measure: SingularValueMeasure,
    /// Whether truncation is per value or based on a discarded tail sum.
    pub rule: TruncationRule,
}

impl SvdTruncationPolicy {
    /// Create a policy with the default semantics:
    /// relative threshold, singular values, and per-value truncation.
    #[must_use]
    pub const fn new(threshold: f64) -> Self {
        Self {
            threshold,
            scale: ThresholdScale::Relative,
            measure: SingularValueMeasure::Value,
            rule: TruncationRule::PerValue,
        }
    }

    /// Use relative threshold scaling.
    #[must_use]
    pub const fn with_relative(mut self) -> Self {
        self.scale = ThresholdScale::Relative;
        self
    }

    /// Use absolute threshold scaling.
    #[must_use]
    pub const fn with_absolute(mut self) -> Self {
        self.scale = ThresholdScale::Absolute;
        self
    }

    /// Measure singular values directly.
    #[must_use]
    pub const fn with_values(mut self) -> Self {
        self.measure = SingularValueMeasure::Value;
        self
    }

    /// Measure squared singular values.
    #[must_use]
    pub const fn with_squared_values(mut self) -> Self {
        self.measure = SingularValueMeasure::SquaredValue;
        self
    }

    /// Apply the threshold independently to each singular value.
    #[must_use]
    pub const fn with_per_value(mut self) -> Self {
        self.rule = TruncationRule::PerValue;
        self
    }

    /// Apply the threshold to the cumulative discarded tail.
    #[must_use]
    pub const fn with_discarded_tail_sum(mut self) -> Self {
        self.rule = TruncationRule::DiscardedTailSum;
        self
    }
}

/// Error for invalid SVD truncation thresholds.
#[derive(Debug, Error, Clone, Copy, PartialEq)]
#[error("Invalid SVD truncation threshold: {0}. Threshold must be finite and non-negative.")]
pub struct InvalidThresholdError(pub f64);

/// Validate one threshold value.
pub(crate) fn validate_threshold_value(threshold: f64) -> Result<(), InvalidThresholdError> {
    if !threshold.is_finite() || threshold < 0.0 {
        return Err(InvalidThresholdError(threshold));
    }
    Ok(())
}

/// Validate one full SVD truncation policy.
pub(crate) fn validate_svd_truncation_policy(
    policy: SvdTruncationPolicy,
) -> Result<(), InvalidThresholdError> {
    validate_threshold_value(policy.threshold)
}

/// Error from validating optional SVD truncation options.
///
/// This is the shared typed error for the neutral validation seam reused by
/// TreeTN truncation/contraction/fit, partition algebra and adaptive patching,
/// and the low-level SVD entry points.
///
/// # Examples
///
/// ```
/// use tensor4all_core::SvdTruncationOptionsError;
///
/// assert_eq!(
///     SvdTruncationOptionsError::ZeroMaxBondDim.to_string(),
///     "max_bond_dim must be at least 1 when set"
/// );
/// ```
#[derive(Debug, Error, Clone, Copy, PartialEq)]
pub enum SvdTruncationOptionsError {
    /// `max_bond_dim` is present but zero.
    #[error("max_bond_dim must be at least 1 when set")]
    ZeroMaxBondDim,
    /// The SVD cutoff threshold is not finite or is negative.
    #[error("invalid SVD truncation threshold: {0}; threshold must be finite and non-negative")]
    InvalidThreshold(f64),
}

/// Validate optional SVD truncation options shared by all callers.
///
/// `max_bond_dim` must be `None` or at least one, and an explicit SVD policy
/// threshold must be finite and non-negative. Call this at every public
/// truncation, contraction, fit, and adaptive entry point **before** any
/// mutation or no-op shortcut (empty center, single node, zero sweep, disjoint
/// partition addition, or `Naive`/`Zipup` dispatch), so invalid options are
/// rejected with the same typed error on every path.
///
/// # Arguments
///
/// * `max_bond_dim` - Optional maximum retained bond dimension; `Some(0)` is
///   rejected because a rank cap of zero would silently collapse.
/// * `policy` - Optional explicit [`SvdTruncationPolicy`]; when present its
///   threshold must be finite and non-negative.
///
/// # Returns
///
/// `Ok(())` when both optional fields are valid, otherwise the typed
/// error naming the failing field.
///
/// # Errors
///
/// Returns [`SvdTruncationOptionsError::ZeroMaxBondDim`] when `max_bond_dim`
/// is `Some(0)`, or [`SvdTruncationOptionsError::InvalidThreshold`] when the
/// explicit policy threshold is non-finite or negative.
///
/// # Examples
///
/// ```
/// use tensor4all_core::{
///     validate_svd_truncation_options, SvdTruncationOptionsError,
///     SvdTruncationPolicy,
/// };
///
/// assert!(validate_svd_truncation_options(None, None).is_ok());
/// assert!(validate_svd_truncation_options(
///     Some(64),
///     Some(SvdTruncationPolicy::new(1e-8)),
/// )
/// .is_ok());
/// assert_eq!(
///     validate_svd_truncation_options(Some(0), None),
///     Err(SvdTruncationOptionsError::ZeroMaxBondDim)
/// );
/// assert!(matches!(
///     validate_svd_truncation_options(
///         None,
///         Some(SvdTruncationPolicy::new(f64::NAN))
///     ),
///     Err(SvdTruncationOptionsError::InvalidThreshold(_))
/// ));
/// ```
pub fn validate_svd_truncation_options(
    max_bond_dim: Option<usize>,
    policy: Option<SvdTruncationPolicy>,
) -> Result<(), SvdTruncationOptionsError> {
    if max_bond_dim == Some(0) {
        return Err(SvdTruncationOptionsError::ZeroMaxBondDim);
    }
    if let Some(policy) = policy {
        validate_svd_truncation_policy(policy)
            .map_err(|error| SvdTruncationOptionsError::InvalidThreshold(error.0))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests;
