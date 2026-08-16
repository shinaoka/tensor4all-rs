//! Options for Quantics TCI interpolation.

use quanticsgrids::UnfoldingScheme;
use tensor4all_treetci::TreeTciOptions;

/// Options for Quantics TCI interpolation.
///
/// Controls convergence criteria, bond dimension limits, pivot search,
/// and quantics-specific settings. Use the builder methods to customize.
///
/// # Quick reference
///
/// | Field | Default | Typical range | Purpose |
/// |---|---|---|---|
/// | `tolerance` | `1e-8` | `1e-6` .. `1e-12` | Relative convergence threshold |
/// | `max_bond_dim` | `None` (unlimited) | `50` .. `500` | Cap on bond dimension |
/// | `max_iter` | `200` | `20` .. `500` | Maximum half-sweep iterations |
/// | `n_random_init_pivot` | `5` | `3` .. `20` | Random initial pivots added |
/// | `unfolding_scheme` | `Interleaved` | — | How quantics bits are arranged |
/// | `normalize_error` | `true` | — | Normalize error by max sample value |
/// | `verbosity` | `0` | `0` .. `2` | Logging verbosity |
/// | `nsearchglobalpivot` | `5` | `1` .. `20` | Global pivots tested per iteration |
/// | `nsearch` | `100` | `10` .. `1000` | Random candidates for global pivot search |
///
/// # Examples
///
/// ```
/// use tensor4all_quanticstci::QtciOptions;
///
/// // Default options
/// let opts = QtciOptions::default();
/// assert!((opts.tolerance - 1e-8).abs() < 1e-15);
/// assert_eq!(opts.max_bond_dim, None);
/// assert_eq!(opts.max_iter, 200);
/// assert_eq!(opts.n_random_init_pivot, 5);
/// assert_eq!(opts.verbosity, 0);
/// assert!(opts.normalize_error);
///
/// // Builder-style customization
/// let custom = QtciOptions::default()
///     .with_tolerance(1e-10)
///     .with_max_bond_dim(50)
///     .with_maxiter(100)
///     .with_nrandominitpivot(10)
///     .with_verbosity(1);
///
/// assert!((custom.tolerance - 1e-10).abs() < 1e-18);
/// assert_eq!(custom.max_bond_dim, Some(50));
/// assert_eq!(custom.max_iter, 100);
/// assert_eq!(custom.n_random_init_pivot, 10);
/// assert_eq!(custom.verbosity, 1);
/// ```
#[derive(Debug, Clone)]
pub struct QtciOptions {
    /// Relative convergence tolerance.
    ///
    /// The algorithm stops when the bond error falls below this threshold
    /// for several consecutive iterations. When `normalize_error` is `true`
    /// (the default), the error is divided by the maximum sampled value,
    /// making this a relative tolerance.
    ///
    /// - Use `1e-6` for quick exploration.
    /// - Use `1e-10` .. `1e-12` for high-accuracy work.
    ///
    /// Default: `1e-8`.
    pub tolerance: f64,

    /// Maximum bond dimension (rank) of the tensor train.
    ///
    /// `None` means unlimited. Set to `50`--`500` when the function is
    /// expensive to evaluate, to prevent runaway computation.
    ///
    /// Default: `None`.
    pub max_bond_dim: Option<usize>,

    /// Maximum number of half-sweep iterations.
    ///
    /// The algorithm terminates after this many sweeps even if
    /// convergence has not been reached. Increase to `500` for difficult
    /// functions that need more sweeps.
    ///
    /// Default: `200`.
    #[doc(alias = "maxiter")]
    pub max_iter: usize,

    /// Number of random initial pivots to add.
    ///
    /// These pivots seed the TCI algorithm in addition to any
    /// user-supplied pivots. More pivots improve robustness for
    /// functions with multiple separated features, at the cost of
    /// extra initial evaluations. Typical values: `3`--`20`.
    ///
    /// Default: `5`.
    #[doc(alias = "nrandominitpivot")]
    pub n_random_init_pivot: usize,

    /// Unfolding scheme for the quantics tensor train.
    ///
    /// `Interleaved` interleaves bits from different dimensions across
    /// sites. `Fused` groups all bits of one dimension together. For
    /// most applications, `Interleaved` gives better compression.
    ///
    /// Default: [`UnfoldingScheme::Interleaved`].
    #[doc(alias = "unfoldingscheme")]
    pub unfolding_scheme: UnfoldingScheme,

    /// Whether to normalize the convergence error by the maximum
    /// sampled function value.
    ///
    /// When `true`, `tolerance` acts as a relative threshold. Set to
    /// `false` for an absolute tolerance.
    ///
    /// Default: `true`.
    pub normalize_error: bool,

    /// Verbosity level. `0` = silent, `1` = progress summary,
    /// `2` = per-sweep details.
    ///
    /// Default: `0`.
    pub verbosity: usize,

    /// Number of global pivot candidates to accept per iteration.
    ///
    /// Each iteration searches for up to this many new global pivots
    /// to improve the approximation. Increasing this can help
    /// difficult functions but costs more evaluations.
    ///
    /// Default: `5`.
    pub nsearchglobalpivot: usize,

    /// Number of random candidates sampled when searching for global
    /// pivots.
    ///
    /// Controls how thoroughly the index space is explored for global
    /// pivot candidates. Increase to `500`--`1000` for high-dimensional
    /// problems.
    ///
    /// Default: `100`.
    pub nsearch: usize,
}

impl Default for QtciOptions {
    fn default() -> Self {
        Self {
            tolerance: 1e-8,
            max_bond_dim: None,
            max_iter: 200,
            n_random_init_pivot: 5,
            unfolding_scheme: UnfoldingScheme::Interleaved,
            normalize_error: true,
            verbosity: 0,
            nsearchglobalpivot: 5,
            nsearch: 100,
        }
    }
}

impl QtciOptions {
    /// Set the convergence tolerance.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_quanticstci::QtciOptions;
    /// let opts = QtciOptions::default().with_tolerance(1e-12);
    /// assert!((opts.tolerance - 1e-12).abs() < 1e-18);
    /// ```
    pub fn with_tolerance(mut self, tolerance: f64) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Set the maximum bond dimension.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_quanticstci::QtciOptions;
    /// let opts = QtciOptions::default().with_max_bond_dim(64);
    /// assert_eq!(opts.max_bond_dim, Some(64));
    /// ```
    pub fn with_max_bond_dim(mut self, max_bond_dim: usize) -> Self {
        self.max_bond_dim = Some(max_bond_dim);
        self
    }

    /// Set the maximum number of half-sweep iterations.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_quanticstci::QtciOptions;
    /// let opts = QtciOptions::default().with_maxiter(500);
    /// assert_eq!(opts.max_iter, 500);
    /// ```
    pub fn with_maxiter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set the number of random initial pivots.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_quanticstci::QtciOptions;
    /// let opts = QtciOptions::default().with_nrandominitpivot(10);
    /// assert_eq!(opts.n_random_init_pivot, 10);
    /// ```
    pub fn with_nrandominitpivot(mut self, n: usize) -> Self {
        self.n_random_init_pivot = n;
        self
    }

    /// Set the unfolding scheme.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_quanticstci::{QtciOptions, UnfoldingScheme};
    /// let opts = QtciOptions::default().with_unfoldingscheme(UnfoldingScheme::Fused);
    /// assert_eq!(opts.unfolding_scheme, UnfoldingScheme::Fused);
    /// ```
    pub fn with_unfoldingscheme(mut self, scheme: UnfoldingScheme) -> Self {
        self.unfolding_scheme = scheme;
        self
    }

    /// Set the verbosity level.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_quanticstci::QtciOptions;
    /// let opts = QtciOptions::default().with_verbosity(2);
    /// assert_eq!(opts.verbosity, 2);
    /// ```
    pub fn with_verbosity(mut self, verbosity: usize) -> Self {
        self.verbosity = verbosity;
        self
    }

    /// Set the number of global pivot candidates to accept per iteration.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_quanticstci::QtciOptions;
    /// let opts = QtciOptions::default().with_nsearchglobalpivot(10);
    /// assert_eq!(opts.nsearchglobalpivot, 10);
    /// ```
    pub fn with_nsearchglobalpivot(mut self, n: usize) -> Self {
        self.nsearchglobalpivot = n;
        self
    }

    /// Set the number of random candidates for global pivot search.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_quanticstci::QtciOptions;
    /// let opts = QtciOptions::default().with_nsearch(500);
    /// assert_eq!(opts.nsearch, 500);
    /// ```
    pub fn with_nsearch(mut self, n: usize) -> Self {
        self.nsearch = n;
        self
    }

    /// Convert to [`TreeTciOptions`] for the underlying algorithm.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_quanticstci::QtciOptions;
    /// let opts = QtciOptions::default()
    ///     .with_tolerance(1e-10)
    ///     .with_max_bond_dim(64)
    ///     .with_maxiter(100);
    /// let tree_opts = opts.to_treetci_options();
    /// assert!((tree_opts.tolerance - 1e-10).abs() < 1e-18);
    /// assert_eq!(tree_opts.max_bond_dim, Some(64));
    /// assert_eq!(tree_opts.max_iter, 100);
    /// ```
    pub fn to_treetci_options(&self) -> TreeTciOptions {
        TreeTciOptions {
            tolerance: self.tolerance,
            max_iter: self.max_iter,
            max_bond_dim: self.max_bond_dim,
            normalize_error: self.normalize_error,
            // Global pivot search stays opt-in; the legacy `nsearchglobalpivot`
            // and `nsearch` fields are not wired through yet.
            enable_global_pivots: false,
            nsearch: self.nsearch,
            max_nglobal_pivot: self.nsearchglobalpivot,
            tol_margin_global_search: 10.0,
            seed: None,
        }
    }
}

#[cfg(test)]
mod tests;
