//! Options and traits for TreeTN operations.
//!
//! Provides:
//! - [`CanonicalizationOptions`]: Options for canonicalization
//! - [`TruncationOptions`]: Options for truncation
//! - [`SplitOptions`]: Options for split operations
//! - [`RestructureOptions`]: Options for multi-phase restructure operations

use crate::algorithm::CanonicalForm;
use crate::treetn::SwapOptions;
use tensor4all_core::SvdTruncationPolicy;

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub(crate) struct FactorizationToleranceOptions {
    pub max_rank: Option<usize>,
    pub svd_policy: Option<SvdTruncationPolicy>,
    pub qr_rtol: Option<f64>,
}

/// Options for canonicalization operations.
///
/// # Builder Pattern
///
/// ```
/// use tensor4all_treetn::{CanonicalForm, CanonicalizationOptions};
///
/// let options = CanonicalizationOptions::default()
///     .with_form(CanonicalForm::LU)
///     .force();
///
/// assert!(matches!(options.form, CanonicalForm::LU));
/// assert!(options.force);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct CanonicalizationOptions {
    /// Canonical form to use (QR, LU, or CI)
    pub form: CanonicalForm,
    /// If true, always performs full canonicalization.
    /// If false, checks current state and may skip or optimize.
    pub force: bool,
}

impl Default for CanonicalizationOptions {
    fn default() -> Self {
        Self {
            form: CanonicalForm::Unitary,
            force: false,
        }
    }
}

impl CanonicalizationOptions {
    /// Create options with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create options that force full canonicalization.
    pub fn forced() -> Self {
        Self {
            form: CanonicalForm::Unitary,
            force: true,
        }
    }

    /// Set the canonical form.
    pub fn with_form(mut self, form: CanonicalForm) -> Self {
        self.form = form;
        self
    }

    /// Set force mode (always perform full canonicalization).
    pub fn force(mut self) -> Self {
        self.force = true;
        self
    }

    /// Disable force mode (check current state before canonicalizing).
    pub fn smart(mut self) -> Self {
        self.force = false;
        self
    }
}

/// Options for truncation operations.
///
/// # Builder Pattern
///
/// ```
/// use tensor4all_core::SvdTruncationPolicy;
/// use tensor4all_treetn::TruncationOptions;
///
/// let options = TruncationOptions::default()
///     .with_max_rank(50)
///     .with_svd_policy(SvdTruncationPolicy::new(1e-10));
///
/// assert_eq!(options.max_rank(), Some(50));
/// assert_eq!(options.svd_policy(), Some(SvdTruncationPolicy::new(1e-10)));
/// ```
#[derive(Debug, Clone, Copy)]
pub struct TruncationOptions {
    #[allow(dead_code)]
    pub(crate) form: CanonicalForm,
    pub(crate) truncation: FactorizationToleranceOptions,
}

impl Default for TruncationOptions {
    fn default() -> Self {
        Self {
            form: CanonicalForm::Unitary,
            truncation: FactorizationToleranceOptions::default(),
        }
    }
}

impl TruncationOptions {
    /// Create options with default settings (no truncation limits).
    pub fn new() -> Self {
        Self::default()
    }

    /// Create options with a maximum rank.
    pub fn with_max_rank(mut self, rank: usize) -> Self {
        self.truncation.max_rank = Some(rank);
        self
    }

    /// Set the SVD truncation policy used during the truncation sweep.
    pub fn with_svd_policy(mut self, policy: SvdTruncationPolicy) -> Self {
        self.truncation.svd_policy = Some(policy);
        self
    }

    /// Get the SVD truncation policy.
    pub fn svd_policy(&self) -> Option<SvdTruncationPolicy> {
        self.truncation.svd_policy
    }

    /// Get max_rank.
    pub fn max_rank(&self) -> Option<usize> {
        self.truncation.max_rank
    }
}

/// Options for split operations.
///
/// # Builder Pattern
///
/// ```
/// use tensor4all_core::SvdTruncationPolicy;
/// use tensor4all_treetn::{CanonicalForm, SplitOptions};
///
/// let options = SplitOptions::default()
///     .with_max_rank(50)
///     .with_svd_policy(SvdTruncationPolicy::new(1e-10))
///     .with_qr_rtol(1e-12)
///     .with_final_sweep(true);
///
/// assert!(matches!(options.form, CanonicalForm::Unitary));
/// assert_eq!(options.max_rank(), Some(50));
/// assert_eq!(options.svd_policy(), Some(SvdTruncationPolicy::new(1e-10)));
/// assert_eq!(options.qr_rtol(), Some(1e-12));
/// assert!(options.final_sweep);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct SplitOptions {
    /// Canonical form / algorithm to use (SVD, QR, etc.)
    pub form: CanonicalForm,
    /// Algorithm-aware factorization tolerances.
    pub(crate) truncation: FactorizationToleranceOptions,
    /// Whether to perform a final sweep for global bond dimension optimization
    pub final_sweep: bool,
}

impl Default for SplitOptions {
    fn default() -> Self {
        Self {
            form: CanonicalForm::Unitary,
            truncation: FactorizationToleranceOptions::default(),
            final_sweep: false,
        }
    }
}

impl SplitOptions {
    /// Create options with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create options with a maximum rank.
    pub fn with_max_rank(mut self, rank: usize) -> Self {
        self.truncation.max_rank = Some(rank);
        self
    }

    /// Set the SVD truncation policy used when `form` is unitary/SVD-based.
    pub fn with_svd_policy(mut self, policy: SvdTruncationPolicy) -> Self {
        self.truncation.svd_policy = Some(policy);
        self
    }

    /// Set the QR-specific relative tolerance used when `form` is QR-based.
    pub fn with_qr_rtol(mut self, rtol: f64) -> Self {
        self.truncation.qr_rtol = Some(rtol);
        self
    }

    /// Set the canonical form / algorithm.
    pub fn with_form(mut self, form: CanonicalForm) -> Self {
        self.form = form;
        self
    }

    /// Enable or disable final sweep for global optimization.
    pub fn with_final_sweep(mut self, final_sweep: bool) -> Self {
        self.final_sweep = final_sweep;
        self
    }

    /// Get the SVD truncation policy.
    pub fn svd_policy(&self) -> Option<SvdTruncationPolicy> {
        self.truncation.svd_policy
    }

    /// Get the QR-specific relative tolerance.
    pub fn qr_rtol(&self) -> Option<f64> {
        self.truncation.qr_rtol
    }

    /// Get max_rank.
    pub fn max_rank(&self) -> Option<usize> {
        self.truncation.max_rank
    }
}

/// Options for `TreeTN::restructure_to` multi-phase restructures.
///
/// `RestructureOptions` combines the three phases in the approved B2a design:
/// a split/refinement phase, a site-transport phase, and an optional final
/// truncation sweep after the target structure has been assembled.
///
/// Related types:
/// - [`SplitOptions`] controls exact splitting plus any optional final sweep
///
///   inside the split primitive.
/// - [`SwapOptions`] controls bond truncation during site-index transport.
/// - [`TruncationOptions`] can be applied once at the end of the full
///
///   restructure to clean up bond dimensions on the final topology.
///
/// When in doubt, start with `RestructureOptions::default()`: exact splitting,
/// exact transport, and no extra final truncation sweep.
///
/// # Examples
///
/// ```
/// use tensor4all_treetn::{
///     RestructureOptions, SplitOptions, SwapOptions, TruncationOptions,
/// };
/// use tensor4all_core::SvdTruncationPolicy;
///
/// let options = RestructureOptions::new()
///     .with_split(SplitOptions::new().with_max_rank(32))
///     .with_swap(SwapOptions {
///         max_rank: Some(16),
///         rtol: Some(1e-10),
///     })
///     .with_final_truncation(
///         TruncationOptions::new().with_svd_policy(SvdTruncationPolicy::new(1e-12)),
///     );
///
/// assert_eq!(options.split.max_rank(), Some(32));
/// assert!(!options.split.final_sweep);
/// assert_eq!(options.swap.max_rank, Some(16));
/// assert_eq!(options.swap.rtol, Some(1e-10));
/// assert_eq!(
///     options
///         .final_truncation
///         .as_ref()
///         .and_then(TruncationOptions::svd_policy),
///     Some(SvdTruncationPolicy::new(1e-12))
/// );
/// ```
#[derive(Debug, Clone, Default)]
pub struct RestructureOptions {
    /// Options for the split/refinement phase.
    ///
    /// These settings matter when a current node must be factored into multiple
    /// fragments before any fragment movement can happen. Higher `max_rank`,
    /// stricter `svd_policy`, and smaller `qr_rtol` preserve more fidelity but
    /// can increase intermediate bond dimensions. `final_sweep` should usually
    /// remain `false` here unless a split-only workflow is being optimized in
    /// isolation.
    pub split: SplitOptions,
    /// Options for the site-transport / swap phase.
    ///
    /// This phase only moves already planned fragments across existing edges.
    /// Leaving both fields unset keeps swaps exact. Setting `max_rank` or
    /// `rtol` can control intermediate rank growth, but may introduce
    /// approximation earlier than the optional final truncation sweep.
    pub swap: SwapOptions,
    /// Optional final truncation sweep on the fully restructured network.
    ///
    /// Use this when the split and swap phases should remain as exact as
    /// possible, and compression should happen only after the target topology
    /// and grouping have been reached. `None` disables this cleanup pass.
    pub final_truncation: Option<TruncationOptions>,
}

impl RestructureOptions {
    /// Create options with exact split/swap phases and no final cleanup sweep.
    ///
    /// # Returns
    /// A `RestructureOptions` value equivalent to [`Default::default`].
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_treetn::RestructureOptions;
    ///
    /// let options = RestructureOptions::new();
    ///
    /// assert!(!options.split.final_sweep);
    /// assert_eq!(options.swap.max_rank, None);
    /// assert_eq!(options.swap.rtol, None);
    /// assert!(options.final_truncation.is_none());
    /// ```
    pub fn new() -> Self {
        Self::default()
    }

    /// Replace the split-phase options.
    ///
    /// # Arguments
    /// * `split` - Split/refinement settings used before fragment transport.
    ///
    /// # Returns
    /// Updated restructure options using the provided split settings.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_treetn::{RestructureOptions, SplitOptions};
    ///
    /// let options = RestructureOptions::new()
    ///     .with_split(SplitOptions::new().with_max_rank(24).with_final_sweep(true));
    ///
    /// assert_eq!(options.split.max_rank(), Some(24));
    /// assert!(options.split.final_sweep);
    /// ```
    pub fn with_split(mut self, split: SplitOptions) -> Self {
        self.split = split;
        self
    }

    /// Replace the swap/transport options.
    ///
    /// # Arguments
    /// * `swap` - Truncation settings applied during fragment movement.
    ///
    /// # Returns
    /// Updated restructure options using the provided swap settings.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_treetn::{RestructureOptions, SwapOptions};
    ///
    /// let options = RestructureOptions::new().with_swap(SwapOptions {
    ///     max_rank: Some(12),
    ///     rtol: Some(1e-8),
    /// });
    ///
    /// assert_eq!(options.swap.max_rank, Some(12));
    /// assert_eq!(options.swap.rtol, Some(1e-8));
    /// ```
    pub fn with_swap(mut self, swap: SwapOptions) -> Self {
        self.swap = swap;
        self
    }

    /// Set the optional final truncation sweep.
    ///
    /// # Arguments
    /// * `final_truncation` - Final cleanup sweep to run on the target
    ///
    ///   topology.
    ///
    /// # Returns
    /// Updated restructure options using the provided final sweep settings.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_treetn::{RestructureOptions, TruncationOptions};
    /// use tensor4all_core::SvdTruncationPolicy;
    ///
    /// let options = RestructureOptions::new()
    ///     .with_final_truncation(
    ///         TruncationOptions::new()
    ///             .with_max_rank(10)
    ///             .with_svd_policy(SvdTruncationPolicy::new(1e-10)),
    ///     );
    ///
    /// assert_eq!(
    ///     options
    ///         .final_truncation
    ///         .as_ref()
    ///         .and_then(TruncationOptions::max_rank),
    ///     Some(10)
    /// );
    /// ```
    pub fn with_final_truncation(mut self, final_truncation: TruncationOptions) -> Self {
        self.final_truncation = Some(final_truncation);
        self
    }
}

#[cfg(test)]
mod tests;
