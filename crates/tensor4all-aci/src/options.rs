//! Configuration for Alternating Cross Interpolation.

use tensor4all_simplett::{SimpleTensorTrain, TTScalar};

/// Options controlling an Alternating Cross Interpolation run.
///
/// Use this type to choose iteration limits, truncation pressure, stopping
/// tolerance, and an optional initial tensor-train guess. The default values are
/// conservative: they run at least two sweeps, allow up to twenty sweeps, do not
/// cap bond dimensions, and use an absolute tolerance of `1e-12`.
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
/// assert!(!options.scale_tolerance);
/// assert!(options.initial_guess.is_none());
/// assert_eq!(options.rng_seed, 0);
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
    /// The default is [`usize::MAX`], meaning no explicit cap. Lower values
    /// reduce memory and runtime but may prevent the approximation from
    /// reaching the requested [`tolerance`](Self::tolerance).
    pub max_bond_dim: Option<usize>,

    /// Requested stopping tolerance for the ACI residual estimate.
    ///
    /// The default is `1e-12`. When [`scale_tolerance`](Self::scale_tolerance)
    /// is `false`, this is interpreted as an absolute tolerance. When
    /// `scale_tolerance` is `true`, the public sweep APIs compare this value to
    /// the largest per-bond relative metric, obtained by dividing each bond's
    /// pivot error by that bond's largest sampled operator-output magnitude
    /// from the completed sweep.
    pub tolerance: f64,

    /// Whether to scale [`tolerance`](Self::tolerance) by the output magnitude.
    ///
    /// The default is `false`, giving absolute tolerance behavior. Set this to
    /// `true` when outputs have problem-dependent scales and relative stopping
    /// behavior is more appropriate. When in doubt, keep the default `false`
    /// for absolute tolerance behavior.
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
}

impl<T: TTScalar> Default for AciOptions<T> {
    fn default() -> Self {
        Self {
            max_iters: 20,
            min_iters: 2,
            max_bond_dim: None,
            tolerance: 1e-12,
            scale_tolerance: false,
            initial_guess: None,
            rng_seed: 0,
        }
    }
}
