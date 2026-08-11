//! Options for interpolative QTT construction.

/// Options controlling SVD compression of constructed tensor trains.
///
/// `InterpolativeQttOptions` mirrors the Julia keyword arguments
/// `tolerance` and `max_bond_dim`. When in doubt, use the default: it keeps
/// near machine precision while allowing the bond dimension to grow as needed.
///
/// # Fields
///
/// - `tolerance`: relative SVD truncation threshold. Smaller values preserve
///
///   more accuracy and usually produce larger bonds.
/// - `max_bond_dim`: hard upper bound on every TT bond. Use `usize::MAX` for
///
///   no explicit cap.
///
/// # Examples
///
/// ```
/// use tensor4all_interpolativeqtt::InterpolativeQttOptions;
///
/// let opts = InterpolativeQttOptions::default()
///     .with_tolerance(1e-10)
///     .with_max_bond_dim(64);
///
/// assert!((opts.tolerance - 1e-10).abs() < 1e-15);
/// assert_eq!(opts.max_bond_dim, 64);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InterpolativeQttOptions {
    /// Relative SVD truncation threshold.
    pub tolerance: f64,
    /// Maximum allowed TT bond dimension.
    pub max_bond_dim: usize,
}

impl Default for InterpolativeQttOptions {
    fn default() -> Self {
        Self {
            tolerance: 1.0e-12,
            max_bond_dim: usize::MAX,
        }
    }
}

impl InterpolativeQttOptions {
    /// Return a copy with a different compression tolerance.
    ///
    /// `tolerance` is compared to the largest singular value at each
    /// compression step. Values around `1e-12` are nearly lossless; values
    /// around `1e-8` to `1e-6` are useful for exploratory work.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_interpolativeqtt::InterpolativeQttOptions;
    ///
    /// let opts = InterpolativeQttOptions::default().with_tolerance(0.0);
    /// assert_eq!(opts.tolerance, 0.0);
    /// ```
    pub fn with_tolerance(mut self, tolerance: f64) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Return a copy with a different maximum bond dimension.
    ///
    /// Use `usize::MAX` for no explicit cap. Smaller caps force more
    /// aggressive compression and may increase approximation error.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_interpolativeqtt::InterpolativeQttOptions;
    ///
    /// let opts = InterpolativeQttOptions::default().with_max_bond_dim(8);
    /// assert_eq!(opts.max_bond_dim, 8);
    /// ```
    pub fn with_max_bond_dim(mut self, max_bond_dim: usize) -> Self {
        self.max_bond_dim = max_bond_dim;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_matches_julia_style_unbounded_compression() {
        let opts = InterpolativeQttOptions::default();

        assert_eq!(opts.tolerance, 1.0e-12);
        assert_eq!(opts.max_bond_dim, usize::MAX);
    }

    #[test]
    fn builder_methods_update_only_requested_fields() {
        let opts = InterpolativeQttOptions::default().with_tolerance(1.0e-8);

        assert_eq!(opts.tolerance, 1.0e-8);
        assert_eq!(opts.max_bond_dim, usize::MAX);

        let capped = opts.with_max_bond_dim(12);

        assert_eq!(capped.tolerance, 1.0e-8);
        assert_eq!(capped.max_bond_dim, 12);
    }
}
