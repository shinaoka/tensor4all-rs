//! Post-hoc error estimation for tensor train approximations.
//!
//! After running [`crossinterpolate2`](crate::crossinterpolate2),
//! [`estimate_true_error`] can verify the approximation quality by
//! searching for multi-indices with large interpolation error via
//! [`floating_zone`] optimization.

use rand::Rng;
use tensor4all_core::{floating_zone_walk, MultiIndex, Scalar};
use tensor4all_simplett::{
    AbstractTensorTrain, EinsumScalar, SimpleTensorTrain, TTCache, TTScalar, Tensor3Ops,
};

use crate::error::{Result, TCIError};

/// Estimate the true interpolation error by searching for worst-case indices.
///
/// Launches [`floating_zone`] from `nsearch` random starting points (or
/// from explicit `initial_points`), returning all found (pivot, error)
/// pairs sorted by descending error.
///
/// This is useful as a post-hoc check: if the largest returned error is
/// below your tolerance, you can be more confident in the approximation.
///
/// # Examples
///
/// ```
/// use tensor4all_tensorci::estimate_true_error;
/// use tensor4all_simplett::SimpleTensorTrain;
///
/// // Build a constant TT (value = 1.0) on a 4×4 grid
/// let tt = SimpleTensorTrain::<f64>::constant(&[4, 4], 1.0);
///
/// // Exact function differs from the constant
/// let f = |idx: &Vec<usize>| (idx[0] * idx[1]) as f64;
/// let mut rng = rand::rng();
///
/// let errors = estimate_true_error(&tt, &f, 10, None, &mut rng).unwrap();
///
/// // Results are sorted by descending error
/// for w in errors.windows(2) {
///     assert!(w[0].1 >= w[1].1, "must be sorted descending");
/// }
///
/// // The worst-case error for |i*j - 1| on [0..4]x[0..4] is at (3,3): |9-1|=8
/// let (best_pivot, max_err) = &errors[0];
/// assert_eq!(*best_pivot, vec![3, 3]);
/// assert!((max_err - 8.0).abs() < 1e-10);
/// ```
///
/// # Arguments
///
/// * `tt` -- the tensor train approximation
/// * `f` -- the exact function
/// * `nsearch` -- number of random starting points (ignored when
///
///   `initial_points` is `Some`)
/// * `initial_points` -- explicit starting points for the search
/// * `rng` -- random number generator
///
/// # Returns
///
/// `Vec<(MultiIndex, f64)>` sorted by descending error, with duplicate
/// pivots removed.
///
/// # Errors
/// Returns [`TCIError::InvalidConfiguration`] for an invalid site dimension,
/// [`TCIError::InvalidPivot`] for an invalid starting point, or
/// [`TCIError::SimpleTensorTrain`] when tensor-train evaluation fails.
pub fn estimate_true_error<T, F>(
    tt: &SimpleTensorTrain<T>,
    f: &F,
    nsearch: usize,
    initial_points: Option<Vec<MultiIndex>>,
    rng: &mut impl Rng,
) -> Result<Vec<(MultiIndex, f64)>>
where
    T: Scalar + TTScalar + EinsumScalar,
    F: Fn(&MultiIndex) -> T,
{
    let site_dims: Vec<usize> = (0..tt.len())
        .map(|i| tt.site_tensor(i).site_dim())
        .collect();
    if site_dims.contains(&0) {
        return Err(TCIError::InvalidConfiguration {
            message: "tensor train contains a zero-dimensional site".to_string(),
        });
    }

    let points = if let Some(pts) = initial_points {
        pts
    } else {
        (0..nsearch)
            .map(|_| {
                site_dims
                    .iter()
                    .map(|&d| rng.random_range(0..d))
                    .collect::<MultiIndex>()
            })
            .collect()
    };

    let mut pivot_errors: Vec<(MultiIndex, f64)> = points
        .into_iter()
        .map(|init_p| floating_zone(tt, f, &site_dims, Some(&init_p), f64::MAX))
        .collect::<Result<_>>()?;

    // Sort by descending error
    pivot_errors.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Remove duplicates (same pivot)
    pivot_errors.dedup_by(|a, b| a.0 == b.0);

    Ok(pivot_errors)
}

/// Local search for the multi-index with the largest interpolation error.
///
/// Starting from `init_p`, sweeps through each site position, evaluating
/// all local indices while fixing the others, and picks the index with
/// the maximum error `|f(idx) - tt(idx)|`. Repeats until the error
/// stops increasing or `early_stop_tol` is exceeded.
///
/// # Examples
///
/// ```
/// use tensor4all_tensorci::floating_zone;
/// use tensor4all_simplett::SimpleTensorTrain;
///
/// // Constant TT (value = 0.0) on a 4×4 grid
/// let tt = SimpleTensorTrain::<f64>::constant(&[4, 4], 0.0);
///
/// // f(i,j) = i * j, so TT error = |i*j|
/// let f = |idx: &Vec<usize>| (idx[0] * idx[1]) as f64;
/// let local_dims = vec![4, 4];
///
/// // Search from (2, 2) without early stopping
/// let (pivot, error) = floating_zone(&tt, &f, &local_dims, Some(&vec![2, 2]), f64::MAX).unwrap();
///
/// // Should find maximum error at (3, 3): |3*3 - 0| = 9
/// assert_eq!(pivot, vec![3, 3]);
/// assert!((error - 9.0).abs() < 1e-10);
/// ```
///
/// # Arguments
///
/// * `tt` -- the tensor train approximation
/// * `f` -- the exact function
/// * `local_dims` -- number of values each index can take
/// * `init_p` -- starting point (`None` draws a random starting point)
/// * `early_stop_tol` -- stop early once the error exceeds this value
///
///   (use `f64::MAX` to search exhaustively)
///
/// # Returns
///
/// `(pivot, max_error)` -- the best multi-index found and its error.
///
/// # Errors
/// Returns [`TCIError::InvalidConfiguration`] for invalid dimensions,
/// [`TCIError::InvalidPivot`] for invalid coordinates, or
/// [`TCIError::SimpleTensorTrain`] when tensor-train evaluation fails.
pub fn floating_zone<T, F>(
    tt: &SimpleTensorTrain<T>,
    f: &F,
    local_dims: &[usize],
    init_p: Option<&MultiIndex>,
    early_stop_tol: f64,
) -> Result<(MultiIndex, f64)>
where
    T: Scalar + TTScalar + EinsumScalar,
    F: Fn(&MultiIndex) -> T,
{
    if local_dims.len() != tt.len() {
        return Err(TCIError::InvalidConfiguration {
            message: format!(
                "local_dims length {} does not match tensor train length {}",
                local_dims.len(),
                tt.len()
            ),
        });
    }
    if local_dims.contains(&0) {
        return Err(TCIError::InvalidConfiguration {
            message: "local_dims must contain only positive dimensions".to_string(),
        });
    }

    // Julia's `_floatingzone` draws a random starting point when none is
    // given; match that instead of fixing the all-zeros index.
    let init_p = match init_p {
        Some(p) => {
            if p.len() != local_dims.len() || p.iter().zip(local_dims).any(|(&i, &d)| i >= d) {
                return Err(TCIError::InvalidPivot {
                    message: format!("initial pivot {p:?} does not fit local_dims {local_dims:?}"),
                });
            }
            p.clone()
        }
        None => local_dims
            .iter()
            .map(|&d| rand::rng().random_range(0..d))
            .collect(),
    };

    let max_sweeps =
        local_dims
            .len()
            .checked_mul(10)
            .ok_or_else(|| TCIError::InvalidConfiguration {
                message: "local_dims sweep count overflowed usize".to_string(),
            })?;
    let mut tt_cache = TTCache::new(tt);
    let (pivot, error) = floating_zone_walk(
        local_dims,
        &init_p,
        max_sweeps,
        early_stop_tol,
        |points: &[MultiIndex]| {
            // Batch-evaluate the tensor train once per site scan; `f` is
            // evaluated pointwise (no batch API in this signature).
            let tt_values = tt_cache
                .evaluate_many(points, None)
                .map_err(TCIError::SimpleTensorTrain)?;
            let errors = points
                .iter()
                .zip(tt_values.iter())
                .map(|(point, &tt_value)| {
                    let diff = f(point) - tt_value;
                    f64::sqrt(Scalar::abs_sq(diff))
                })
                .collect::<Vec<f64>>();
            Ok::<Vec<f64>, TCIError>(errors)
        },
    )?;

    Ok((pivot, error))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_floating_zone_finds_error() {
        use tensor4all_simplett::Tensor3Ops;

        // Build a rank-1 TT approximation of constant 1.0
        let mut t0 = tensor4all_simplett::tensor3_zeros(1, 4, 1);
        let mut t1 = tensor4all_simplett::tensor3_zeros(1, 4, 1);
        for s in 0..4 {
            t0.set3(0, s, 0, 1.0);
            t1.set3(0, s, 0, 1.0);
        }
        let tt = SimpleTensorTrain::new(vec![t0, t1]).unwrap();

        // Verify TT evaluates to 1.0 everywhere
        assert!((tt.evaluate(&[0, 0]).unwrap() - 1.0).abs() < 1e-14);
        assert!((tt.evaluate(&[3, 3]).unwrap() - 1.0).abs() < 1e-14);

        // Exact function has non-constant behavior
        let f = |idx: &MultiIndex| (idx[0] * idx[1]) as f64;
        let local_dims = vec![4, 4];

        // Start from (1, 1) so initial error is not zero: |1*1 - 1| = 0
        // Actually start from (2, 2) so initial error is |4 - 1| = 3
        let (pivot, error) =
            floating_zone(&tt, &f, &local_dims, Some(&vec![2, 2]), f64::MAX).unwrap();

        // Error should be > 0 since tt=1 but f(i,j)=i*j varies
        // The maximum error should be at (3,3): |9-1|=8
        assert!(error > 0.0, "Error should be positive, got {}", error);
        assert_eq!(pivot, vec![3, 3], "Should find max error at (3,3)");
        assert!(
            (error - 8.0).abs() < 1e-10,
            "Error should be 8.0, got {}",
            error
        );
    }

    #[test]
    fn test_estimate_true_error_sorted() {
        // Build a constant TT (all 0)
        let t0 = tensor4all_simplett::tensor3_zeros(1, 4, 1);
        let t1 = tensor4all_simplett::tensor3_zeros(1, 4, 1);
        let tt = SimpleTensorTrain::new(vec![t0, t1]).unwrap();

        let f = |idx: &MultiIndex| (idx[0] + idx[1]) as f64;
        let mut rng = rand::rng();

        let errors = estimate_true_error(&tt, &f, 10, None, &mut rng).unwrap();

        // Verify sorted in descending order
        for i in 0..errors.len().saturating_sub(1) {
            assert!(
                errors[i].1 >= errors[i + 1].1,
                "Errors should be sorted descending: {} < {}",
                errors[i].1,
                errors[i + 1].1
            );
        }
    }
}
