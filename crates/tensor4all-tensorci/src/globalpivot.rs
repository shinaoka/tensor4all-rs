//! Global pivot finder for the TCI2 algorithm.
//!
//! After each two-site sweep, the TCI2 algorithm calls a
//! [`GlobalPivotFinder`] to locate regions of high interpolation error
//! that the local sweeps may have missed. The default implementation
//! ([`DefaultGlobalPivotFinder`]) uses random starting points with local
//! optimization.

use rand::Rng;
use tensor4all_core::{MultiIndex, Scalar};
use tensor4all_simplett::{AbstractTensorTrain, SimpleTensorTrain, TTScalar};

/// Snapshot of the current TCI state, passed to [`GlobalPivotFinder`].
pub struct GlobalPivotSearchInput<T: Scalar + TTScalar> {
    /// Local dimensions of each tensor index.
    pub local_dims: Vec<usize>,
    /// Current tensor train approximation.
    pub current_tt: SimpleTensorTrain<T>,
    /// Maximum absolute function value encountered so far.
    pub max_sample_value: f64,
    /// Left index sets (I) for each site.
    pub i_set: Vec<Vec<MultiIndex>>,
    /// Right index sets (J) for each site.
    pub j_set: Vec<Vec<MultiIndex>>,
}

/// Trait for global pivot finders.
///
/// Implementors search for multi-indices where the interpolation error
/// `|f(idx) - tt(idx)|` is large. Found pivots are added to the TCI state
/// to improve the approximation in the next sweep.
///
/// The default implementation is [`DefaultGlobalPivotFinder`]. Implement
/// this trait to supply domain-specific search strategies.
///
/// # Examples
///
/// Using the default implementation via [`DefaultGlobalPivotFinder`]:
///
/// ```
/// use tensor4all_tensorci::{DefaultGlobalPivotFinder, GlobalPivotFinder,
///     GlobalPivotSearchInput};
/// use tensor4all_simplett::SimpleTensorTrain;
///
/// // Constant-zero TT on a 4×4 grid
/// let tt = SimpleTensorTrain::<f64>::constant(&[4, 4], 0.0);
///
/// let input = GlobalPivotSearchInput {
///     local_dims: vec![4, 4],
///     current_tt: tt,
///     max_sample_value: 9.0,
///     i_set: vec![vec![vec![]], vec![vec![0]]],
///     j_set: vec![vec![vec![0]], vec![vec![]]],
/// };
///
/// let finder = DefaultGlobalPivotFinder::default();
/// let mut rng = rand::rng();
///
/// // f(i,j) = i*j has nonzero values, so pivots should be found
/// let pivots = finder.find_global_pivots(
///     &input,
///     &|idx: &Vec<usize>| (idx[0] * idx[1]) as f64,
///     0.1,
///     &mut rng,
/// );
///
/// // All returned pivots must have valid indices
/// for p in &pivots {
///     assert_eq!(p.len(), 2);
///     assert!(p[0] < 4 && p[1] < 4);
/// }
/// ```
pub trait GlobalPivotFinder {
    /// Find multi-indices with high interpolation error.
    ///
    /// # Arguments
    ///
    /// * `input` -- current TCI state (tensor train, index sets, etc.)
    /// * `f` -- the function being interpolated
    /// * `abs_tol` -- absolute tolerance; pivots with error above this
    ///
    ///   threshold (times `tol_margin`) are interesting
    /// * `rng` -- random number generator for stochastic search
    ///
    /// # Returns
    ///
    /// Multi-indices where the interpolation error is large, up to the
    /// implementation's maximum count.
    fn find_global_pivots<T, F>(
        &self,
        input: &GlobalPivotSearchInput<T>,
        f: &F,
        abs_tol: f64,
        rng: &mut impl Rng,
    ) -> Vec<MultiIndex>
    where
        T: Scalar + TTScalar,
        F: Fn(&MultiIndex) -> T;
}

/// Default global pivot finder using random search with local optimization.
///
/// Algorithm:
///
/// 1. Generate `nsearch` random initial points.
/// 2. For each point, sweep all dimensions and pick the index with the
///    largest interpolation error at each position.
/// 3. Keep points where the error exceeds `abs_tol * tol_margin`.
/// 4. Return at most `max_nglobal_pivot` results.
///
/// # Examples
///
/// ```
/// use tensor4all_tensorci::DefaultGlobalPivotFinder;
///
/// // Default configuration
/// let finder = DefaultGlobalPivotFinder::default();
/// assert_eq!(finder.nsearch, 5);
/// assert_eq!(finder.max_nglobal_pivot, 5);
/// assert!((finder.tol_margin - 10.0).abs() < 1e-15);
///
/// // Custom configuration
/// let custom = DefaultGlobalPivotFinder::new(20, 10, 5.0);
/// assert_eq!(custom.nsearch, 20);
/// assert_eq!(custom.max_nglobal_pivot, 10);
/// assert!((custom.tol_margin - 5.0).abs() < 1e-15);
/// ```
#[derive(Debug, Clone)]
pub struct DefaultGlobalPivotFinder {
    /// Number of random initial points to search from
    pub nsearch: usize,
    /// Maximum number of pivots to add per iteration
    pub max_nglobal_pivot: usize,
    /// Search for pivots with error > abs_tol × tol_margin
    pub tol_margin: f64,
}

impl Default for DefaultGlobalPivotFinder {
    fn default() -> Self {
        Self {
            nsearch: 5,
            max_nglobal_pivot: 5,
            tol_margin: 10.0,
        }
    }
}

impl DefaultGlobalPivotFinder {
    /// Create a new DefaultGlobalPivotFinder with the given parameters.
    pub fn new(nsearch: usize, max_nglobal_pivot: usize, tol_margin: f64) -> Self {
        Self {
            nsearch,
            max_nglobal_pivot,
            tol_margin,
        }
    }
}

impl GlobalPivotFinder for DefaultGlobalPivotFinder {
    fn find_global_pivots<T, F>(
        &self,
        input: &GlobalPivotSearchInput<T>,
        f: &F,
        abs_tol: f64,
        rng: &mut impl Rng,
    ) -> Vec<MultiIndex>
    where
        T: Scalar + TTScalar,
        F: Fn(&MultiIndex) -> T,
    {
        let n = input.local_dims.len();

        // Generate random initial points
        let initial_points: Vec<MultiIndex> = (0..self.nsearch)
            .map(|_| {
                (0..n)
                    .map(|p| rng.random_range(0..input.local_dims[p]))
                    .collect()
            })
            .collect();

        let mut found_pivots: Vec<MultiIndex> = Vec::new();

        for point in &initial_points {
            let mut current_point = point.clone();
            let mut best_error = 0.0f64;
            let mut best_point = point.clone();

            // Local search: sweep all dimensions
            for p in 0..n {
                let original = current_point[p];
                for v in 0..input.local_dims[p] {
                    current_point[p] = v;
                    let f_val = f(&current_point);
                    let tt_val = input
                        .current_tt
                        .evaluate(&current_point)
                        .unwrap_or(T::zero());
                    let diff = f_val - tt_val;
                    let error = f64::sqrt(Scalar::abs_sq(diff));
                    if error > best_error {
                        best_error = error;
                        best_point = current_point.clone();
                    }
                }
                current_point[p] = original; // Reset to original for next dimension
            }

            // Add point if error exceeds threshold
            if best_error > abs_tol * self.tol_margin {
                found_pivots.push(best_point);
            }
        }

        // Limit number of pivots
        found_pivots.truncate(self.max_nglobal_pivot);

        found_pivots
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_global_pivot_finder() {
        // Simple function: f(i, j) = i * j
        let f = |idx: &MultiIndex| (idx[0] * idx[1]) as f64;
        let local_dims = vec![4, 4];

        // Build a deliberately bad TT (constant 0) to ensure large errors
        let tensors = vec![
            tensor4all_simplett::tensor3_zeros(1, 4, 1),
            tensor4all_simplett::tensor3_zeros(1, 4, 1),
        ];
        let tt = SimpleTensorTrain::new(tensors).unwrap();

        let input = GlobalPivotSearchInput {
            local_dims: local_dims.clone(),
            current_tt: tt,
            max_sample_value: 9.0,
            i_set: vec![vec![vec![]], vec![vec![0]]],
            j_set: vec![vec![vec![0]], vec![vec![]]],
        };

        let finder = DefaultGlobalPivotFinder::new(10, 3, 1.0);
        let mut rng = rand::rng();

        let pivots = finder.find_global_pivots(&input, &f, 0.1, &mut rng);

        // Should find some pivots since the TT is zero but f is not
        // (except at i=0 or j=0)
        assert!(
            pivots.len() <= 3,
            "Should limit to max_nglobal_pivot=3, got {}",
            pivots.len()
        );
    }

    #[test]
    fn test_custom_global_pivot_finder() {
        struct FixedPivotFinder;

        impl GlobalPivotFinder for FixedPivotFinder {
            fn find_global_pivots<T, F>(
                &self,
                _input: &GlobalPivotSearchInput<T>,
                _f: &F,
                _abs_tol: f64,
                _rng: &mut impl Rng,
            ) -> Vec<MultiIndex>
            where
                T: Scalar + TTScalar,
                F: Fn(&MultiIndex) -> T,
            {
                // Always return a fixed pivot
                vec![vec![1, 2]]
            }
        }

        let finder = FixedPivotFinder;
        let f = |_: &MultiIndex| 1.0f64;
        let tensors = vec![
            tensor4all_simplett::tensor3_zeros(1, 3, 1),
            tensor4all_simplett::tensor3_zeros(1, 3, 1),
        ];
        let tt = SimpleTensorTrain::new(tensors).unwrap();

        let input = GlobalPivotSearchInput {
            local_dims: vec![3, 3],
            current_tt: tt,
            max_sample_value: 1.0,
            i_set: vec![],
            j_set: vec![],
        };

        let mut rng = rand::rng();
        let pivots = finder.find_global_pivots(&input, &f, 0.0, &mut rng);
        assert_eq!(pivots, vec![vec![1, 2]]);
    }
}
