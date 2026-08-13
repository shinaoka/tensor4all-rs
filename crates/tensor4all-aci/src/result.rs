//! Result types returned by Alternating Cross Interpolation.

use tensor4all_simplett::{SimpleTensorTrain, TTScalar};

/// Reason an [`AciResult`] run stopped.
///
/// Related types: [`AciResult::termination`] reports which exit fired.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AciTermination {
    /// The sweep reached the requested tolerance with stable rank and the
    /// global pivot search found nothing over the convergence window.
    Converged,
    /// The rank stayed at [`AciOptions::max_bond_dim`](crate::AciOptions::max_bond_dim)
    /// for the whole convergence window, so the tolerance is not reachable
    /// under the bond cap. This is by design, not a failure; the final
    /// [`AciResult::errors`] entry is expected to remain above tolerance.
    RankLimited,
    /// The run consumed all [`AciOptions::max_iters`](crate::AciOptions::max_iters)
    /// sweeps without satisfying either exit. Increase `max_iters` or loosen
    /// the tolerance/bond cap if convergence is expected.
    MaxIterations,
}

/// Output of an Alternating Cross Interpolation run.
///
/// The result contains the approximating tensor train plus convergence metadata
/// collected during the run. Use [`tensor_train`](Self::tensor_train) for
/// subsequent tensor-train operations, and inspect [`ranks`](Self::ranks),
/// [`errors`](Self::errors), and [`termination`](Self::termination) to diagnose
/// how and why the run stopped.
///
/// Related types: [`AciOptions`](crate::AciOptions) configures the run that
/// produces this value; [`SimpleTensorTrain`] stores the approximating tensor.
///
/// # Examples
///
/// ```
/// use tensor4all_aci::{AciResult, AciTermination};
/// use tensor4all_simplett::{AbstractTensorTrain, SimpleTensorTrain};
///
/// let tensor_train = SimpleTensorTrain::<f64>::constant(&[2, 3], 4.0);
/// let result = AciResult {
///     tensor_train,
///     ranks: vec![1, 2],
///     errors: vec![1e-3, 0.0],
///     nglobal_pivots: vec![0, 0],
///     termination: AciTermination::Converged,
/// };
///
/// assert_eq!(result.tensor_train.site_dims(), vec![2, 3]);
/// assert_eq!(result.ranks, vec![1, 2]);
/// assert_eq!(result.errors, vec![1e-3, 0.0]);
/// assert_eq!(result.nglobal_pivots, vec![0, 0]);
/// assert_eq!(result.termination, AciTermination::Converged);
/// assert!((result.tensor_train.evaluate(&[1, 2]).unwrap() - 4.0).abs() < 1e-12);
/// ```
#[derive(Debug, Clone)]
pub struct AciResult<T: TTScalar> {
    /// Tensor-train approximation produced by ACI.
    pub tensor_train: SimpleTensorTrain<T>,

    /// Maximum bond dimension after each completed sweep.
    ///
    /// Entries are stored in completed-sweep order so callers can compare rank
    /// growth against [`AciOptions::max_bond_dim`](crate::AciOptions::max_bond_dim).
    pub ranks: Vec<usize>,

    /// Maximum pivot error after each completed sweep.
    ///
    /// Entries are stored in completed-sweep order and use the same scaling
    /// convention as [`AciOptions::tolerance`](crate::AciOptions::tolerance).
    ///
    /// The last entry is not guaranteed to be at or below the requested
    /// tolerance; see [`termination`](Self::termination) for the stopping
    /// reason. A rank-limited run finishes above tolerance by design.
    pub errors: Vec<f64>,

    /// Global pivot search findings after each completed sweep, one entry per
    /// sweep in the same order as [`ranks`](Self::ranks) and [`errors`](Self::errors).
    ///
    /// Each entry is the number of global pivots the guard found in that
    /// sweep (not the number injected — already-represented pivots are
    /// skipped on injection). Convergence is only accepted once the trailing
    /// [`AciOptions::min_iters`](crate::AciOptions::min_iters) entries are all
    /// zero, so a nonzero trailing entry means the run stopped for another
    /// reason.
    pub nglobal_pivots: Vec<usize>,

    /// Why the run stopped.
    pub termination: AciTermination,
}
