//! Configuration for Alternating Cross Interpolation.

use tensor4all_simplett::{SimpleTensorTrain, TTScalar};

/// Options controlling an Alternating Cross Interpolation run.
///
/// Use this type to choose iteration limits, truncation pressure, stopping
/// tolerance, and an optional initial tensor-train guess. The default values are
/// conservative: they run at least two sweeps, allow up to twenty sweeps, do not
/// cap bond dimensions, and use a tolerance of `1e-12` relative to the sampled
/// operator-output magnitude (see [`scale_tolerance`](Self::scale_tolerance)).
///
/// Related types: [`AciResult`](crate::AciResult) stores the tensor train and
/// convergence history produced by an ACI run, while [`ElementwiseBatch`](crate::ElementwiseBatch)
/// describes batched column-major operator inputs.
///
/// # Examples
///
/// ```
/// use tensor4all_aci::AciOptions;
///
/// let options = AciOptions::<f64>::default();
/// assert_eq!(options.max_iters, 20);
/// assert_eq!(options.min_iters, 2);
/// assert_eq!(options.max_bond_dim, None);
/// assert!((options.tolerance - 1e-12).abs() < 1e-15);
/// assert!(options.scale_tolerance);
/// assert!(options.initial_guess.is_none());
/// assert_eq!(options.rng_seed, 0);
/// assert!(options.enable_global_guard);
/// assert_eq!(options.nsearch_global_pivots, 5);
/// assert_eq!(options.max_nglobal_pivot, 5);
/// assert_eq!(options.nsweeps_global_search, 100);
/// assert!((options.tol_margin_global_search - 10.0).abs() < 1e-15);
/// ```
#[derive(Debug, Clone)]
pub struct AciOptions<T: TTScalar> {
    /// Maximum number of ACI sweeps to run.
    ///
    /// The default is `20`, which is usually enough for small and medium
    /// problems while still preventing runaway iteration. Increase this when
    /// convergence is steady but the requested [`tolerance`](Self::tolerance)
    /// has not been reached.
    pub max_iters: usize,

    /// Minimum number of ACI sweeps to run before convergence checks may stop.
    ///
    /// The default is `2`, which gives the interpolation pivots at least one
    /// forward and backward refinement opportunity. Keep this at least `1` and
    /// below or equal to [`max_iters`](Self::max_iters).
    pub min_iters: usize,

    /// Maximum allowed tensor-train bond dimension.
    ///
    /// The default is `None`, meaning no explicit cap. Lower values
    /// reduce memory and runtime but may prevent the approximation from
    /// reaching the requested [`tolerance`](Self::tolerance).
    pub max_bond_dim: Option<usize>,

    /// Requested stopping tolerance for the ACI residual estimate.
    ///
    /// The default is `1e-12`. When [`scale_tolerance`](Self::scale_tolerance)
    /// is `true` (the default), the public sweep APIs compare this value to the
    /// largest per-bond relative metric, obtained by dividing each bond's pivot
    /// error by that bond's largest sampled operator-output magnitude from the
    /// completed sweep. When `scale_tolerance` is `false`, this is interpreted
    /// as an absolute tolerance on the same pivot errors.
    pub tolerance: f64,

    /// Whether to scale [`tolerance`](Self::tolerance) by the output magnitude.
    ///
    /// The default is `true`, giving relative stopping behavior: each bond's
    /// pivot error is compared against `tolerance` times the largest operator
    /// output sampled at that bond. This matches the reference contract of
    /// `TensorCrossInterpolation.jl` (`normalizeerror = true`), where the
    /// tolerance is normalized by the largest sampled value.
    ///
    /// Set this to `false` only when an absolute error budget is genuinely
    /// wanted, and keep in mind what an absolute budget asks for: a tolerance
    /// far below the accuracy of the input tensor trains makes the sweep
    /// interpolate their own approximation error, which is not low rank. The
    /// pivot count then keeps rising with the number of sites at fixed
    /// tolerance instead of saturating with the rank of the function (issue
    /// #618). Scale-blind tolerances are also easy to misjudge: for an operator
    /// output of magnitude `1e3`, an absolute `1e-12` is a relative `1e-15`,
    /// close to double-precision rounding noise.
    ///
    /// When in doubt, keep the default `true`.
    pub scale_tolerance: bool,

    /// Optional tensor train used to initialize interpolation pivots and ranks.
    ///
    /// The default is `None`, so ACI chooses its own starting state. Provide a
    /// guess when a nearby solution is available; it must have site dimensions
    /// compatible with the ACI problem.
    pub initial_guess: Option<SimpleTensorTrain<T>>,

    /// Seed for randomized choices made by ACI.
    ///
    /// The default is `0`, giving deterministic behavior for repeated runs with
    /// the same inputs and options. Change this to sample a different initial
    /// pivot path when convergence depends on random choices.
    pub rng_seed: u64,

    /// Whether to run the global pivot search guard before accepting convergence.
    ///
    /// The local sweep estimates error from bond-local 2-site blocks only, so a
    /// feature outside the sampled crosses (e.g. a near-degenerate second peak
    /// far from the initial pivots) is invisible to the stopping rule. When
    /// enabled, the optimizer runs a global pivot search after every sweep,
    /// injects any significantly-wrong points as pivots, and only accepts
    /// convergence when the search has found nothing for [`min_iters`](Self::min_iters)
    /// consecutive sweeps. Default: `true`.
    pub enable_global_guard: bool,

    /// Number of random starting points for the global pivot search guard.
    ///
    /// Each starting point launches a floating-zone walk (greedy
    /// coordinate-descent; see [`floating_zone_walk`](tensor4all_core::floating_zone_walk))
    /// that moves one site coordinate at a time toward the largest
    /// interpolation error. Larger values explore the index space more
    /// thoroughly at the cost of more evaluations. Ignored when
    /// `enable_global_guard` is `false`; `0` disables the search. Default: `5`.
    pub nsearch_global_pivots: usize,

    /// Maximum number of global pivots injected per guard run.
    ///
    /// Ignored when `enable_global_guard` is `false`; `0` disables the search.
    /// Default: `5`.
    pub max_nglobal_pivot: usize,

    /// Upper bound on coordinate sweeps per floating-zone walk.
    ///
    /// A walk stops early once the maximum error stops improving or exceeds
    /// `abs_tol * tol_margin_global_search`, so this bound rarely binds; it is
    /// a safety cap for pathological error landscapes. Ignored when
    /// `enable_global_guard` is `false`; `0` disables the search. Default: `100`.
    pub nsweeps_global_search: usize,

    /// Tolerance margin for the global pivot search guard.
    ///
    /// A candidate point is injected when its interpolation error exceeds
    /// `abs_tol * tol_margin_global_search`, where `abs_tol` is the configured
    /// tolerance (scaled by the maximum sampled operator magnitude when
    /// `scale_tolerance` is enabled). The value is always validated (finite and
    /// nonnegative) but only consulted when `enable_global_guard` is `true`.
    /// Default: `10.0`.
    pub tol_margin_global_search: f64,
}

impl<T: TTScalar> Default for AciOptions<T> {
    fn default() -> Self {
        Self {
            max_iters: 20,
            min_iters: 2,
            max_bond_dim: None,
            tolerance: 1e-12,
            scale_tolerance: true,
            initial_guess: None,
            rng_seed: 0,
            enable_global_guard: true,
            nsearch_global_pivots: 5,
            max_nglobal_pivot: 5,
            nsweeps_global_search: 100,
            tol_margin_global_search: 10.0,
        }
    }
}
