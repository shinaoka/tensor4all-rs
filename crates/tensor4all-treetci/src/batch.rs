use crate::error::Result as TreeTciResult;

/// Borrowed view of a global site-order batch.
///
/// The data is stored in column-major layout with shape `(n_sites, n_points)`.
/// Each column is one multi-index point, with `data[site + n_sites * point]`
/// giving the local index at `site` for `point`.
///
/// This type is the main interface for the batch evaluator closure passed to
/// [`crossinterpolate2`](crate::crossinterpolate2).
///
/// # Examples
///
/// ```
/// use tensor4all_treetci::GlobalIndexBatch;
///
/// // 2 sites, 3 points: column-major layout
/// // point 0: (0, 1), point 1: (1, 0), point 2: (0, 0)
/// let data = vec![0, 1, 1, 0, 0, 0];
/// let batch = GlobalIndexBatch::new(&data, 2, 3).unwrap();
///
/// assert_eq!(batch.n_sites(), 2);
/// assert_eq!(batch.n_points(), 3);
/// assert_eq!(batch.get(0, 0), Some(0)); // site 0, point 0
/// assert_eq!(batch.get(1, 0), Some(1)); // site 1, point 0
/// assert_eq!(batch.get(0, 1), Some(1)); // site 0, point 1
/// assert_eq!(batch.get(0, 5), None);    // out of bounds
/// ```
#[derive(Clone, Copy, Debug)]
pub struct GlobalIndexBatch<'a> {
    data: &'a [usize],
    n_sites: usize,
    n_points: usize,
}

impl<'a> GlobalIndexBatch<'a> {
    /// Create a borrowed batch view with column-major `(n_sites, n_points)` storage.
    ///
    /// Returns an error if `data.len() != n_sites * n_points`.
    ///
    /// # Errors
    ///
    /// Returns an error when the construction or conversion fails (a shape or
    /// /// index mismatch, or a backend failure).
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_treetci::GlobalIndexBatch;
    ///
    /// let data = vec![0, 1, 2, 3];
    /// let batch = GlobalIndexBatch::new(&data, 2, 2).unwrap();
    /// assert_eq!(batch.n_sites(), 2);
    /// assert_eq!(batch.n_points(), 2);
    ///
    /// // Wrong length produces an error
    /// assert!(GlobalIndexBatch::new(&data, 3, 2).is_err());
    /// ```
    pub fn new(data: &'a [usize], n_sites: usize, n_points: usize) -> TreeTciResult<Self> {
        let expected = n_sites
            .checked_mul(n_points)
            .ok_or_else(|| anyhow::anyhow!("global index batch shape product overflowed usize"))?;
        if data.len() != expected {
            return Err(anyhow::anyhow!(
                "global index batch has length {}, expected {}",
                data.len(),
                expected
            )
            .into());
        };
        Ok(Self {
            data,
            n_sites,
            n_points,
        })
    }

    /// Borrow the raw column-major backing storage.
    pub fn data(&self) -> &'a [usize] {
        self.data
    }

    /// Number of sites per point.
    pub fn n_sites(&self) -> usize {
        self.n_sites
    }

    /// Number of points in the batch.
    pub fn n_points(&self) -> usize {
        self.n_points
    }

    /// Get one value from `(site, point)` coordinates.
    ///
    /// Returns `None` if either index is out of bounds.
    pub fn get(&self, site: usize, point: usize) -> Option<usize> {
        (site < self.n_sites && point < self.n_points)
            .then(|| self.data[site + self.n_sites * point])
    }
}

/// Owned column-major batch buffer for global site-order evaluation.
///
/// Same layout as [`GlobalIndexBatch`] but owns its data. Useful for
/// constructing batches programmatically.
///
/// # Examples
///
/// ```
/// use tensor4all_treetci::OwnedGlobalIndexBatch;
///
/// let batch = OwnedGlobalIndexBatch::new(vec![0, 1, 1, 0], 2, 2).unwrap();
/// let view = batch.as_view();
/// assert_eq!(view.get(0, 0), Some(0));
/// assert_eq!(view.get(1, 0), Some(1));
///
/// let raw = batch.into_vec();
/// assert_eq!(raw, vec![0, 1, 1, 0]);
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OwnedGlobalIndexBatch {
    data: Vec<usize>,
    n_sites: usize,
    n_points: usize,
}

impl OwnedGlobalIndexBatch {
    /// Create an owned batch buffer with column-major `(n_sites, n_points)` storage.
    ///
    /// Returns an error if `data.len() != n_sites * n_points`.
    ///
    /// # Errors
    ///
    /// Returns an error when the construction or conversion fails (a shape or
    /// /// index mismatch, or a backend failure).
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_treetci::OwnedGlobalIndexBatch;
    ///
    /// let batch = OwnedGlobalIndexBatch::new(vec![10, 20, 30, 40], 2, 2).unwrap();
    /// assert_eq!(batch.as_view().n_sites(), 2);
    /// assert_eq!(batch.as_view().n_points(), 2);
    ///
    /// // Wrong length is an error
    /// assert!(OwnedGlobalIndexBatch::new(vec![1, 2, 3], 2, 2).is_err());
    /// ```
    pub fn new(data: Vec<usize>, n_sites: usize, n_points: usize) -> TreeTciResult<Self> {
        GlobalIndexBatch::new(&data, n_sites, n_points)?;
        Ok(Self {
            data,
            n_sites,
            n_points,
        })
    }

    /// Borrow this batch as a view.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_treetci::OwnedGlobalIndexBatch;
    ///
    /// let batch = OwnedGlobalIndexBatch::new(vec![0, 1, 2, 3], 2, 2).unwrap();
    /// let view = batch.as_view();
    /// assert_eq!(view.get(0, 0), Some(0));
    /// assert_eq!(view.get(1, 1), Some(3));
    /// ```
    pub fn as_view(&self) -> GlobalIndexBatch<'_> {
        GlobalIndexBatch {
            data: &self.data,
            n_sites: self.n_sites,
            n_points: self.n_points,
        }
    }

    /// Consume the batch and return the raw backing storage.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_treetci::OwnedGlobalIndexBatch;
    ///
    /// let batch = OwnedGlobalIndexBatch::new(vec![5, 6, 7, 8], 2, 2).unwrap();
    /// let raw = batch.into_vec();
    /// assert_eq!(raw, vec![5, 6, 7, 8]);
    /// ```
    pub fn into_vec(self) -> Vec<usize> {
        self.data
    }
}
