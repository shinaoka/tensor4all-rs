//! Result types returned by Alternating Cross Interpolation.

use tensor4all_simplett::{SimpleTensorTrain, TTScalar};

/// Output of an Alternating Cross Interpolation run.
///
/// The result contains the approximating tensor train plus convergence metadata
/// collected during the run. Use [`tensor_train`](Self::tensor_train) for
/// subsequent tensor-train operations, and inspect [`ranks`](Self::ranks) and
/// [`errors`](Self::errors) to diagnose sweep-by-sweep convergence behavior.
///
/// Related types: [`AciOptions`](crate::AciOptions) configures the run that
/// produces this value; [`SimpleTensorTrain`] stores the approximating tensor.
///
/// # Examples
///
/// ```
/// use tensor4all_aci::AciResult;
/// use tensor4all_simplett::{AbstractTensorTrain, SimpleTensorTrain};
///
/// let tensor_train = SimpleTensorTrain::<f64>::constant(&[2, 3], 4.0);
/// let result = AciResult {
///     tensor_train,
///     ranks: vec![1, 2],
///     errors: vec![1e-3, 0.0],
/// };
///
/// assert_eq!(result.tensor_train.site_dims(), vec![2, 3]);
/// assert_eq!(result.ranks, vec![1, 2]);
/// assert_eq!(result.errors, vec![1e-3, 0.0]);
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
    /// tolerance. A run under a binding
    /// [`AciOptions::max_bond_dim`](crate::AciOptions::max_bond_dim) stops once
    /// the rank has sat at the cap, because the sweep can no longer add pivots
    /// and the tolerance can never be met; such a run finishes above tolerance
    /// by design. Compare the last [`ranks`](Self::ranks) entry against
    /// `max_bond_dim` to tell a rank-limited run from a converged one.
    pub errors: Vec<f64>,
}
