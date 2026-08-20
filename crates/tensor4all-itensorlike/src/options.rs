//! Configuration options for tensor train operations.

use tensor4all_core::{AnyScalar, SvdTruncationPolicy};

use crate::error::{Result, TensorTrainError};

// Re-export CanonicalForm from treetn for convenience.
pub use tensor4all_treetn::CanonicalForm;

pub(crate) fn validate_svd_truncation_options(
    max_bond_dim: Option<usize>,
    svd_policy: Option<SvdTruncationPolicy>,
) -> Result<()> {
    if let Some(policy) = svd_policy {
        if !policy.threshold.is_finite() || policy.threshold < 0.0 {
            return Err(TensorTrainError::OperationError {
                message: format!(
                    "svd_policy.threshold must be finite and >= 0, got {}",
                    policy.threshold
                ),
            });
        }
    }

    if let Some(max_bond_dim) = max_bond_dim {
        if max_bond_dim == 0 {
            return Err(TensorTrainError::OperationError {
                message: "max_bond_dim/maxdim must be >= 1".to_string(),
            });
        }
    }

    Ok(())
}

/// Options for tensor train truncation.
///
/// Truncation is explicitly SVD-based. Canonicalization remains the API for
/// LU/CI-style forms; truncate itself only accepts SVD truncation controls.
///
/// # Examples
///
/// ```
/// use tensor4all_core::SvdTruncationPolicy;
/// use tensor4all_itensorlike::TruncateOptions;
///
/// let opts = TruncateOptions::svd()
///     .with_svd_policy(SvdTruncationPolicy::new(1e-10))
///     .with_max_bond_dim(20);
///
/// assert_eq!(opts.svd_policy(), Some(SvdTruncationPolicy::new(1e-10)));
/// assert_eq!(opts.max_bond_dim(), Some(20));
/// ```
#[derive(Debug, Clone, Default)]
pub struct TruncateOptions {
    max_bond_dim: Option<usize>,
    svd_policy: Option<SvdTruncationPolicy>,
}

impl TruncateOptions {
    /// Create options for SVD-based truncation.
    pub fn svd() -> Self {
        Self::default()
    }

    /// Set the explicit SVD truncation policy.
    pub fn with_svd_policy(mut self, policy: SvdTruncationPolicy) -> Self {
        self.svd_policy = Some(policy);
        self
    }

    /// Set the maximum retained bond dimension.
    pub fn with_max_bond_dim(mut self, max_bond_dim: usize) -> Self {
        self.max_bond_dim = Some(max_bond_dim);
        self
    }

    /// Get the SVD truncation policy override.
    #[inline]
    pub fn svd_policy(&self) -> Option<SvdTruncationPolicy> {
        self.svd_policy
    }

    /// Get the maximum retained bond dimension.
    #[inline]
    pub fn max_bond_dim(&self) -> Option<usize> {
        self.max_bond_dim
    }
}

/// Contraction method for tensor train operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ContractMethod {
    /// Zip-up contraction (faster, one-pass).
    #[default]
    Zipup,
    /// Fit/variational contraction (iterative optimization).
    Fit,
    /// Dense/reference contraction: contract to a full tensor, then decompose back.
    /// Useful only for small debugging and testing cases; memory scales as the
    /// product of external dimensions.
    Naive,
}

/// Options for tensor train contraction.
///
/// # Examples
///
/// ```
/// use tensor4all_core::SvdTruncationPolicy;
/// use tensor4all_itensorlike::ContractOptions;
///
/// let opts = ContractOptions::fit()
///     .with_svd_policy(SvdTruncationPolicy::new(1e-8))
///     .with_max_bond_dim(50)
///     .with_nsweeps(3);
///
/// assert_eq!(opts.max_bond_dim(), Some(50));
/// assert_eq!(opts.svd_policy(), Some(SvdTruncationPolicy::new(1e-8)));
/// assert_eq!(opts.nhalfsweeps(), 6);
/// ```
#[derive(Debug, Clone)]
pub struct ContractOptions {
    method: ContractMethod,
    max_bond_dim: Option<usize>,
    svd_policy: Option<SvdTruncationPolicy>,
    nhalfsweeps: usize,
    dense_reference_limit: Option<usize>,
    fit_initializer: FitInitializer,
}

impl Default for ContractOptions {
    fn default() -> Self {
        Self {
            method: ContractMethod::default(),
            max_bond_dim: None,
            svd_policy: None,
            nhalfsweeps: 2,
            dense_reference_limit: None,
            fit_initializer: FitInitializer::default(),
        }
    }
}

/// Initialization strategy for fit (`ContractMethod::Fit`) contraction.
///
/// The initializer determines the variational starting state `C₀ ≈ A·B` that
/// the two-site sweeps refine.
///
/// # Examples
///
/// ```
/// use tensor4all_itensorlike::{ContractOptions, FitInitializer};
///
/// // The default already starts small (bond 1, deterministic seed) and grows
/// // adaptively when a tolerance is supplied with no `max_bond_dim`.
/// let opts = ContractOptions::fit().with_svd_policy(
///     tensor4all_core::SvdTruncationPolicy::new(1e-10),
/// );
/// assert_eq!(
///     opts.initializer(),
///     FitInitializer::LowRankRandom {
///         bond_dim: 1,
///         seed: None
///     }
/// );
///
/// // Opt back in to the exact zip-up start.
/// let zipup = ContractOptions::fit().with_initializer(FitInitializer::ZipUp);
/// assert_eq!(zipup.initializer(), FitInitializer::ZipUp);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FitInitializer {
    /// Start `C₀` from the SVD-based zip-up contraction of `A·B`, preserving
    /// the input topology.
    ///
    /// Zip-up materializes the full product bond (up to `χ_A·χ_B` per edge)
    /// when no truncation tolerance is supplied, so it is the compatibility
    /// choice rather than the memory-scalable one.
    ZipUp,
    /// Start `C₀` from a deterministic small-rank random network carrying the
    /// surviving output site indices.
    ///
    /// No term of dimension χ_A·χ_B is ever formed: each output tensor is an
    /// independent random tensor of bond dimension [`bond_dim`](Self::LowRankRandom::bond_dim).
    /// With `max_bond_dim = None` and an SVD truncation tolerance, the sweeps
    /// grow the ranks adaptively as required.
    LowRankRandom {
        /// Initial bond dimension of every edge (`1` = start as small as
        /// possible and let the sweeps grow it).
        bond_dim: usize,
        /// RNG seed for reproducibility; `None` uses a fixed deterministic
        /// default seed.
        seed: Option<u64>,
    },
}

impl Default for FitInitializer {
    fn default() -> Self {
        Self::LowRankRandom {
            bond_dim: 1,
            seed: None,
        }
    }
}

/// Default seed used when [`FitInitializer::LowRankRandom`] is constructed
/// without an explicit seed.
pub const DEFAULT_FIT_INITIALIZER_SEED: u64 = 0x005e_ed5e_edc0_ffee;

impl ContractOptions {
    /// Create options for zipup contraction.
    pub fn zipup() -> Self {
        Self {
            method: ContractMethod::Zipup,
            ..Default::default()
        }
    }

    /// Create options for fit contraction.
    pub fn fit() -> Self {
        Self {
            method: ContractMethod::Fit,
            ..Default::default()
        }
    }

    /// Create options for naive contraction.
    ///
    /// Naive contraction is a dense/reference path. Call
    /// [`ContractOptions::with_dense_reference_limit`] before use to bound full
    /// dense materialization.
    ///
    /// # Returns
    /// Options configured for the dense/reference contraction method.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_itensorlike::{ContractMethod, ContractOptions};
    ///
    /// let opts = ContractOptions::naive().with_dense_reference_limit(16);
    ///
    /// assert_eq!(opts.method(), ContractMethod::Naive);
    /// assert_eq!(opts.dense_reference_limit(), Some(16));
    /// ```
    pub fn naive() -> Self {
        Self {
            method: ContractMethod::Naive,
            ..Default::default()
        }
    }

    /// Set the maximum retained bond dimension.
    pub fn with_max_bond_dim(mut self, max_bond_dim: usize) -> Self {
        self.max_bond_dim = Some(max_bond_dim);
        self
    }

    /// Set the explicit SVD truncation policy.
    pub fn with_svd_policy(mut self, policy: SvdTruncationPolicy) -> Self {
        self.svd_policy = Some(policy);
        self
    }

    /// Set the fit initializer strategy.
    ///
    /// Ignored for `Zipup` and `Naive` methods. The default is
    /// [`FitInitializer::LowRankRandom`] with `bond_dim = 1` and the
    /// deterministic default seed: fit contraction therefore starts from a
    /// small random state and (with a truncation tolerance and no
    /// `max_bond_dim`) grows ranks adaptively without constructing the exact
    /// product bond. Pass [`FitInitializer::ZipUp`] to restore the previous
    /// zip-up-initialized behavior.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_itensorlike::{ContractOptions, FitInitializer};
    ///
    /// let opts = ContractOptions::fit()
    ///     .with_initializer(FitInitializer::LowRankRandom {
    ///         bond_dim: 4,
    ///         seed: Some(99),
    ///     });
    /// assert_eq!(
    ///     opts.initializer(),
    ///     FitInitializer::LowRankRandom {
    ///         bond_dim: 4,
    ///         seed: Some(99)
    ///     }
    /// );
    /// ```
    pub fn with_initializer(mut self, initializer: FitInitializer) -> Self {
        self.fit_initializer = initializer;
        self
    }

    /// Set number of half-sweeps for fit contraction.
    pub fn with_nhalfsweeps(mut self, nhalfsweeps: usize) -> Self {
        self.nhalfsweeps = nhalfsweeps;
        self
    }

    /// Set number of full sweeps.
    ///
    /// A full sweep is two half-sweeps. Values that would overflow the
    /// half-sweep counter saturate at `usize::MAX`.
    pub fn with_nsweeps(mut self, nsweeps: usize) -> Self {
        self.nhalfsweeps = nsweeps.saturating_mul(2);
        self
    }

    /// Set the maximum dense elements allowed for naive dense/reference contraction.
    ///
    /// # Arguments
    /// * `max_elements` - Maximum element count allowed for each dense input
    ///
    ///   and output tensor materialized by the reference path. Use small,
    ///   test-sized values unless you have explicitly budgeted memory.
    ///
    /// # Returns
    /// Updated options with the dense/reference limit enabled.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_itensorlike::ContractOptions;
    ///
    /// let opts = ContractOptions::naive().with_dense_reference_limit(32);
    ///
    /// assert_eq!(opts.dense_reference_limit(), Some(32));
    /// ```
    pub fn with_dense_reference_limit(mut self, max_elements: usize) -> Self {
        self.dense_reference_limit = Some(max_elements);
        self
    }

    /// Get the contraction method.
    #[inline]
    pub fn method(&self) -> ContractMethod {
        self.method
    }

    /// Get the maximum retained bond dimension.
    #[inline]
    pub fn max_bond_dim(&self) -> Option<usize> {
        self.max_bond_dim
    }

    /// Get the SVD truncation policy override.
    #[inline]
    pub fn svd_policy(&self) -> Option<SvdTruncationPolicy> {
        self.svd_policy
    }

    /// Get the fit initializer strategy used for [`ContractMethod::Fit`].
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_itensorlike::{ContractOptions, FitInitializer};
    ///
    /// let opts = ContractOptions::fit();
    /// assert!(matches!(opts.initializer(), FitInitializer::LowRankRandom { .. }));
    /// ```
    #[inline]
    pub fn initializer(&self) -> FitInitializer {
        self.fit_initializer
    }

    /// Get number of half-sweeps.
    #[inline]
    pub fn nhalfsweeps(&self) -> usize {
        self.nhalfsweeps
    }

    /// Get the dense/reference element limit for naive contraction.
    ///
    /// # Returns
    /// `Some(max_elements)` when a dense/reference limit has been configured,
    /// otherwise `None`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_itensorlike::ContractOptions;
    ///
    /// let opts = ContractOptions::naive().with_dense_reference_limit(8);
    ///
    /// assert_eq!(opts.dense_reference_limit(), Some(8));
    /// ```
    #[inline]
    pub fn dense_reference_limit(&self) -> Option<usize> {
        self.dense_reference_limit
    }
}

/// Options for the linear solver.
///
/// Solves `(a₀ + a₁ * A) * x = b` using DMRG-like sweeps with local GMRES.
///
/// # Examples
///
/// ```
/// use tensor4all_core::SvdTruncationPolicy;
/// use tensor4all_itensorlike::LinsolveOptions;
///
/// let opts = LinsolveOptions::new(5)
///     .with_svd_policy(SvdTruncationPolicy::new(1e-10))
///     .with_max_bond_dim(64)
///     .with_gmres_tol(1e-8)
///     .with_coefficients(1.0, -1.0);
///
/// assert_eq!(opts.max_bond_dim(), Some(64));
/// assert_eq!(opts.svd_policy(), Some(SvdTruncationPolicy::new(1e-10)));
/// assert_eq!(opts.nhalfsweeps(), 10);
/// ```
#[derive(Debug, Clone)]
pub struct LinsolveOptions {
    nhalfsweeps: usize,
    max_bond_dim: Option<usize>,
    svd_policy: Option<SvdTruncationPolicy>,
    gmres_tol: f64,
    gmres_max_restarts: usize,
    gmres_restart_dim: usize,
    a0: AnyScalar,
    a1: AnyScalar,
    convergence_tol: Option<f64>,
    check_residual: bool,
}

impl Default for LinsolveOptions {
    fn default() -> Self {
        Self {
            nhalfsweeps: 10,
            max_bond_dim: None,
            svd_policy: None,
            gmres_tol: 1e-10,
            gmres_max_restarts: 100,
            gmres_restart_dim: 30,
            a0: AnyScalar::new_real(0.0),
            a1: AnyScalar::new_real(1.0),
            convergence_tol: None,
            check_residual: true,
        }
    }
}

impl LinsolveOptions {
    /// Create options with the specified number of full sweeps.
    ///
    /// Values that would overflow the half-sweep counter saturate at
    /// `usize::MAX`.
    pub fn new(nsweeps: usize) -> Self {
        Self {
            nhalfsweeps: nsweeps.saturating_mul(2),
            ..Default::default()
        }
    }

    /// Set the explicit SVD truncation policy.
    pub fn with_svd_policy(mut self, policy: SvdTruncationPolicy) -> Self {
        self.svd_policy = Some(policy);
        self
    }

    /// Set the maximum retained bond dimension.
    pub fn with_max_bond_dim(mut self, max_bond_dim: usize) -> Self {
        self.max_bond_dim = Some(max_bond_dim);
        self
    }

    /// Set number of half-sweeps.
    pub fn with_nhalfsweeps(mut self, nhalfsweeps: usize) -> Self {
        self.nhalfsweeps = nhalfsweeps;
        self
    }

    /// Set number of full sweeps. Values that would overflow the
    /// half-sweep counter saturate at `usize::MAX`.
    pub fn with_nsweeps(mut self, nsweeps: usize) -> Self {
        self.nhalfsweeps = nsweeps.saturating_mul(2);
        self
    }

    /// Set GMRES tolerance.
    pub fn with_gmres_tol(mut self, tol: f64) -> Self {
        self.gmres_tol = tol;
        self
    }

    /// Set maximum number of GMRES restart cycles per local solve.
    ///
    /// This matches KrylovKit's `maxiter` convention. The maximum number of
    /// operator expansion steps is roughly `gmres_max_restarts * gmres_restart_dim`.
    pub fn with_gmres_max_restarts(mut self, max_restarts: usize) -> Self {
        self.gmres_max_restarts = max_restarts;
        self
    }

    /// Set GMRES restart cycle length.
    pub fn with_gmres_restart_dim(mut self, dim: usize) -> Self {
        self.gmres_restart_dim = dim;
        self
    }

    /// Set coefficients `a₀` and `a₁` in `(a₀ + a₁ * A) * x = b`.
    pub fn with_coefficients<A0, A1>(mut self, a0: A0, a1: A1) -> Self
    where
        A0: Into<AnyScalar>,
        A1: Into<AnyScalar>,
    {
        self.a0 = a0.into();
        self.a1 = a1.into();
        self
    }

    /// Set convergence tolerance for early termination.
    pub fn with_convergence_tol(mut self, tol: f64) -> Self {
        self.convergence_tol = Some(tol);
        self
    }

    /// Set whether to compute the final true residual after the sweep.
    pub fn with_residual_check(mut self, check_residual: bool) -> Self {
        self.check_residual = check_residual;
        self
    }

    /// Get the maximum retained bond dimension.
    #[inline]
    pub fn max_bond_dim(&self) -> Option<usize> {
        self.max_bond_dim
    }

    /// Get the SVD truncation policy override.
    #[inline]
    pub fn svd_policy(&self) -> Option<SvdTruncationPolicy> {
        self.svd_policy
    }

    /// Get number of half-sweeps.
    #[inline]
    pub fn nhalfsweeps(&self) -> usize {
        self.nhalfsweeps
    }

    /// Get GMRES tolerance.
    #[inline]
    pub fn gmres_tol(&self) -> f64 {
        self.gmres_tol
    }

    /// Get maximum number of GMRES restart cycles per local solve.
    #[inline]
    pub fn gmres_max_restarts(&self) -> usize {
        self.gmres_max_restarts
    }

    /// Get GMRES restart cycle length.
    #[inline]
    pub fn gmres_restart_dim(&self) -> usize {
        self.gmres_restart_dim
    }

    /// Get coefficients `(a0, a1)`.
    #[inline]
    pub fn coefficients(&self) -> (AnyScalar, AnyScalar) {
        (self.a0.clone(), self.a1.clone())
    }

    /// Get convergence tolerance.
    #[inline]
    pub fn convergence_tol(&self) -> Option<f64> {
        self.convergence_tol
    }

    /// Get whether the final true residual is computed after the sweep.
    #[inline]
    pub fn check_residual(&self) -> bool {
        self.check_residual
    }
}

#[cfg(test)]
mod tests;
