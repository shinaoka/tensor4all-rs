//! SimpleTensorTrain implementation

use crate::error::{Result, TensorTrainError};
use crate::traits::{AbstractTensorTrain, TTScalar};
use crate::types::{tensor3_zeros, Tensor3, Tensor3Ops};

/// Tensor Train (Matrix Product State) representation.
///
/// A tensor train decomposes a high-dimensional tensor `T[i0, i1, ..., i_{L-1}]`
/// into a chain of rank-3 core tensors:
///
/// ```text
/// T[i0, i1, ..., i_{L-1}] = A0[i0] * A1[i1] * ... * A_{L-1}[i_{L-1}]
/// ```
///
/// where each core `Ak` has shape `(r_{k-1}, d_k, r_k)` with:
/// - `r_k` = bond dimension (link between site `k` and `k+1`),
/// - `d_k` = physical (site) dimension at site `k`,
/// - `r_{-1} = r_{L-1} = 1` (boundary condition).
///
/// # Construction
///
/// - [`SimpleTensorTrain::constant`] -- all entries equal to a given value
/// - [`SimpleTensorTrain::zeros`] -- all entries zero
/// - [`SimpleTensorTrain::new`] -- from explicit rank-3 core tensors
///
/// # Related types
///
/// - [`CompressionOptions`](crate::CompressionOptions) -- configure compression
/// - [`TTCache`](crate::TTCache) -- cached evaluation
/// - [`SiteTensorTrain`](crate::SiteTensorTrain) -- center-canonical form
/// - [`VidalTensorTrain`](crate::VidalTensorTrain) -- Vidal canonical form
///
/// # Examples
///
/// ```
/// use tensor4all_simplett::{SimpleTensorTrain, AbstractTensorTrain};
///
/// // Create a constant tensor train: T[i,j,k] = 3.0 for all i,j,k
/// let tt = SimpleTensorTrain::<f64>::constant(&[2, 3, 4], 3.0);
///
/// assert_eq!(tt.len(), 3);
/// assert_eq!(tt.site_dims(), vec![2, 3, 4]);
/// assert_eq!(tt.link_dims(), vec![1, 1]); // bond dim = 1 for constant
///
/// // Evaluate at a specific index
/// let val = tt.evaluate(&[0, 1, 2]).unwrap();
/// assert!((val - 3.0).abs() < 1e-12);
///
/// // Sum over all indices: 3.0 * 2 * 3 * 4 = 72.0
/// let s = tt.sum();
/// assert!((s - 72.0).abs() < 1e-10);
/// ```
#[derive(Debug, Clone)]
pub struct SimpleTensorTrain<T: TTScalar> {
    /// The tensors that make up the tensor train
    /// Each tensor has shape (left_bond, site_dim, right_bond)
    tensors: Vec<Tensor3<T>>,
}

impl<T: TTScalar> SimpleTensorTrain<T> {
    /// Create a new tensor train from a list of rank-3 core tensors.
    ///
    /// Each tensor must have shape `(left_bond, site_dim, right_bond)` where
    /// the `right_bond` of tensor `i` equals the `left_bond` of tensor `i+1`.
    /// The first tensor must have `left_bond = 1` and the last must have
    /// `right_bond = 1`.
    ///
    /// # Errors
    ///
    /// Returns [`TensorTrainError::DimensionMismatch`] if adjacent bond
    /// dimensions do not match, or [`TensorTrainError::InvalidOperation`] if
    /// boundary dimensions are not 1.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_simplett::{SimpleTensorTrain, AbstractTensorTrain, Tensor3Ops, tensor3_zeros};
    ///
    /// // Build a 2-site TT with bond dimension 1 and site dimensions [2, 3]
    /// let mut t0 = tensor3_zeros::<f64>(1, 2, 1);
    /// t0.set3(0, 0, 0, 1.0);
    /// t0.set3(0, 1, 0, 2.0);
    ///
    /// let mut t1 = tensor3_zeros::<f64>(1, 3, 1);
    /// t1.set3(0, 0, 0, 10.0);
    /// t1.set3(0, 1, 0, 20.0);
    /// t1.set3(0, 2, 0, 30.0);
    ///
    /// let tt = SimpleTensorTrain::new(vec![t0, t1]).unwrap();
    /// assert_eq!(tt.len(), 2);
    ///
    /// // T[0, 2] = 1.0 * 30.0 = 30.0
    /// let val = tt.evaluate(&[0, 2]).unwrap();
    /// assert!((val - 30.0).abs() < 1e-12);
    /// ```
    pub fn new(tensors: Vec<Tensor3<T>>) -> Result<Self> {
        // Validate dimensions
        for i in 0..tensors.len().saturating_sub(1) {
            if tensors[i].right_dim() != tensors[i + 1].left_dim() {
                return Err(TensorTrainError::DimensionMismatch { site: i });
            }
        }

        // First tensor should have left_dim = 1
        if !tensors.is_empty() && tensors[0].left_dim() != 1 {
            return Err(TensorTrainError::InvalidOperation {
                message: "First tensor must have left dimension 1".to_string(),
            });
        }

        // Last tensor should have right_dim = 1
        if let Some(last) = tensors.last() {
            if last.right_dim() != 1 {
                return Err(TensorTrainError::InvalidOperation {
                    message: "Last tensor must have right dimension 1".to_string(),
                });
            }
        }

        Ok(Self { tensors })
    }

    /// Create a tensor train from tensors without dimension validation
    /// (for internal use when dimensions are known to be correct)
    pub(crate) fn from_tensors_unchecked(tensors: Vec<Tensor3<T>>) -> Self {
        Self { tensors }
    }

    /// Create a tensor train where every entry is zero.
    ///
    /// The resulting TT has bond dimension 1 at every link.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_simplett::{SimpleTensorTrain, AbstractTensorTrain};
    ///
    /// let tt = SimpleTensorTrain::<f64>::zeros(&[2, 3]);
    /// assert!((tt.evaluate(&[1, 2]).unwrap()).abs() < 1e-14);
    /// assert!((tt.sum()).abs() < 1e-14);
    /// ```
    pub fn zeros(site_dims: &[usize]) -> Self {
        let tensors: Vec<Tensor3<T>> = site_dims.iter().map(|&d| tensor3_zeros(1, d, 1)).collect();
        Self { tensors }
    }

    /// Create a tensor train where every entry equals `value`.
    ///
    /// The resulting TT has bond dimension 1 at every link.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_simplett::{SimpleTensorTrain, AbstractTensorTrain};
    ///
    /// let tt = SimpleTensorTrain::<f64>::constant(&[2, 3, 4], 5.0);
    ///
    /// // Every entry is 5.0
    /// assert!((tt.evaluate(&[0, 0, 0]).unwrap() - 5.0).abs() < 1e-12);
    /// assert!((tt.evaluate(&[1, 2, 3]).unwrap() - 5.0).abs() < 1e-12);
    ///
    /// // Sum = 5.0 * 2 * 3 * 4 = 120.0
    /// assert!((tt.sum() - 120.0).abs() < 1e-10);
    /// ```
    pub fn constant(site_dims: &[usize], value: T) -> Self {
        if site_dims.is_empty() {
            return Self {
                tensors: Vec::new(),
            };
        }

        let n = site_dims.len();
        let mut tensors = Vec::with_capacity(n);

        // First tensor: all ones
        let mut first = tensor3_zeros(1, site_dims[0], 1);
        for s in 0..site_dims[0] {
            first.set3(0, s, 0, T::one());
        }
        tensors.push(first);

        // Middle tensors: all ones (only if n > 2)
        if n > 2 {
            for &d in &site_dims[1..n - 1] {
                let mut tensor = tensor3_zeros(1, d, 1);
                for s in 0..d {
                    tensor.set3(0, s, 0, T::one());
                }
                tensors.push(tensor);
            }
        }

        // Last tensor: multiply by value
        if n > 1 {
            let mut last = tensor3_zeros(1, site_dims[n - 1], 1);
            for s in 0..site_dims[n - 1] {
                last.set3(0, s, 0, value);
            }
            tensors.push(last);
        } else {
            // Single site: multiply the first (and only) tensor by value
            for s in 0..site_dims[0] {
                let current = *tensors[0].get3(0, s, 0);
                tensors[0].set3(0, s, 0, current * value);
            }
        }

        Self { tensors }
    }

    /// Get mutable access to the site tensors
    pub fn site_tensors_mut(&mut self) -> &mut [Tensor3<T>] {
        &mut self.tensors
    }

    /// Consume this tensor train and return its rank-3 core tensors.
    ///
    /// Use this when another representation needs to take ownership of the
    /// tensor train cores without cloning them. The returned tensors keep the
    /// same site order and bond dimensions as the original tensor train.
    ///
    /// # Returns
    ///
    /// The owned rank-3 site tensors in left-to-right site order.
    ///
    /// # Panics
    ///
    /// This method does not panic.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_simplett::{AbstractTensorTrain, SimpleTensorTrain};
    ///
    /// let tt = SimpleTensorTrain::<f64>::constant(&[2, 3], 4.0);
    /// let link_dims = tt.link_dims();
    /// let cores = tt.into_site_tensors();
    ///
    /// assert_eq!(cores.len(), 2);
    /// assert_eq!(link_dims, vec![1]);
    /// assert_eq!(cores[0].dim(1), 2);
    /// assert_eq!(cores[1].dim(1), 3);
    /// ```
    pub fn into_site_tensors(self) -> Vec<Tensor3<T>> {
        self.tensors
    }

    /// Multiply every entry of the tensor train by `factor` in place.
    ///
    /// Only the last core tensor is rescaled, so this is an O(d * r^2) operation
    /// where `d` is the site dimension and `r` the bond dimension of the last site.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_simplett::{SimpleTensorTrain, AbstractTensorTrain};
    ///
    /// let mut tt = SimpleTensorTrain::<f64>::constant(&[2, 3], 1.0);
    /// tt.scale_mut(3.0);
    /// assert!((tt.evaluate(&[0, 0]).unwrap() - 3.0).abs() < 1e-12);
    /// assert!((tt.sum() - 18.0).abs() < 1e-10);
    /// ```
    pub fn scale_mut(&mut self, factor: T) {
        if !self.tensors.is_empty() {
            let last = self.tensors.len() - 1;
            let tensor = &mut self.tensors[last];
            for l in 0..tensor.left_dim() {
                for s in 0..tensor.site_dim() {
                    for r in 0..tensor.right_dim() {
                        let val = *tensor.get3(l, s, r);
                        tensor.set3(l, s, r, val * factor);
                    }
                }
            }
        }
    }

    /// Return a new tensor train with every entry multiplied by `factor`.
    ///
    /// This is the non-mutating version of [`scale`](Self::scale).
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_simplett::{SimpleTensorTrain, AbstractTensorTrain};
    ///
    /// let tt = SimpleTensorTrain::<f64>::constant(&[2, 3], 1.0);
    /// let tt2 = tt.scale(4.0);
    /// // Original is unchanged
    /// assert!((tt.evaluate(&[0, 0]).unwrap() - 1.0).abs() < 1e-12);
    /// // Scaled copy
    /// assert!((tt2.evaluate(&[0, 0]).unwrap() - 4.0).abs() < 1e-12);
    /// ```
    pub fn scale(&self, factor: T) -> Self {
        let mut result = self.clone();
        result.scale_mut(factor);
        result
    }

    /// Reverse the order of sites in the tensor train.
    ///
    /// The reversed TT satisfies `reversed.evaluate(&[i_{L-1}, ..., i_0]) ==
    /// original.evaluate(&[i_0, ..., i_{L-1}])`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_simplett::{SimpleTensorTrain, AbstractTensorTrain, Tensor3Ops, tensor3_zeros};
    ///
    /// let mut t0 = tensor3_zeros::<f64>(1, 2, 1);
    /// t0.set3(0, 0, 0, 1.0);
    /// t0.set3(0, 1, 0, 2.0);
    /// let mut t1 = tensor3_zeros::<f64>(1, 3, 1);
    /// t1.set3(0, 0, 0, 10.0);
    /// t1.set3(0, 1, 0, 20.0);
    /// t1.set3(0, 2, 0, 30.0);
    /// let tt = SimpleTensorTrain::new(vec![t0, t1]).unwrap();
    ///
    /// let rev = tt.reverse();
    /// assert_eq!(rev.site_dims(), vec![3, 2]);
    /// // T[0, 1] = 1.0 * 10.0 = 10.0, reversed: T_rev[0, 1] should also be 10.0 (site 0->10, site 1->2)
    /// // Original: T[1, 0] = 2.0 * 10.0 = 20.0
    /// // Reversed: T_rev[0, 1] = 20.0
    /// assert!((rev.evaluate(&[0, 1]).unwrap() - tt.evaluate(&[1, 0]).unwrap()).abs() < 1e-12);
    /// ```
    pub fn reverse(&self) -> Self {
        let mut new_tensors = Vec::with_capacity(self.tensors.len());
        for tensor in self.tensors.iter().rev() {
            // Swap left and right dimensions
            let mut new_tensor =
                tensor3_zeros(tensor.right_dim(), tensor.site_dim(), tensor.left_dim());
            for l in 0..tensor.left_dim() {
                for s in 0..tensor.site_dim() {
                    for r in 0..tensor.right_dim() {
                        new_tensor.set3(r, s, l, *tensor.get3(l, s, r));
                    }
                }
            }
            new_tensors.push(new_tensor);
        }
        Self {
            tensors: new_tensors,
        }
    }
}

impl<T: TTScalar> SimpleTensorTrain<T> {
    /// Materialize the tensor train as a full dense tensor.
    ///
    /// Returns `(data, shape)` where `data` is a flat vector in **column-major**
    /// order and `shape` is the site dimensions. The total number of elements
    /// is `prod(shape)`.
    ///
    /// **Warning:** The full tensor can be extremely large for high-dimensional
    /// problems. Only use this for small tensors or debugging.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_simplett::{SimpleTensorTrain, AbstractTensorTrain};
    ///
    /// let tt = SimpleTensorTrain::<f64>::constant(&[2, 3], 7.0);
    /// let (data, shape) = tt.full_tensor();
    /// assert_eq!(shape, vec![2, 3]);
    /// assert_eq!(data.len(), 6);
    /// // Every element should be 7.0
    /// assert!(data.iter().all(|&v| (v - 7.0).abs() < 1e-12));
    /// ```
    pub fn full_tensor(&self) -> (Vec<T>, Vec<usize>) {
        if self.is_empty() {
            return (Vec::new(), Vec::new());
        }

        let site_dims: Vec<usize> = self.site_dims();
        let total_size: usize = site_dims.iter().product();

        if total_size == 0 {
            return (Vec::new(), site_dims);
        }

        // Build full tensor by iterating over all indices
        let mut result = Vec::with_capacity(total_size);
        let mut indices = vec![0usize; site_dims.len()];

        loop {
            // Evaluate at current indices
            if let Ok(val) = self.evaluate(&indices) {
                result.push(val);
            } else {
                result.push(T::zero());
            }

            // Increment indices in column-major order, leftmost index fastest.
            let mut carry = true;
            for i in 0..site_dims.len() {
                if carry {
                    indices[i] += 1;
                    if indices[i] >= site_dims[i] {
                        indices[i] = 0;
                    } else {
                        carry = false;
                    }
                }
            }

            if carry {
                break; // All indices wrapped around
            }
        }

        (result, site_dims)
    }
}

impl<T: TTScalar> SimpleTensorTrain<T> {
    /// Sum (trace out) selected site dimensions, returning a lower-order TT.
    ///
    /// `dims` is a slice of 0-indexed site positions to sum over. The
    /// remaining sites keep their original order. If *all* dimensions are
    /// summed, the result is a 1-site TT wrapping the scalar total.
    ///
    /// # Errors
    ///
    /// Returns an error when the contraction or operation fails (a shape or
    /// /// index mismatch, or a backend failure).
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_simplett::{SimpleTensorTrain, AbstractTensorTrain};
    ///
    /// // 3-site constant TT: T[i,j,k] = 1.0, dims = [2, 3, 4]
    /// let tt = SimpleTensorTrain::<f64>::constant(&[2, 3, 4], 1.0);
    ///
    /// // Sum over the middle site (index 1): result has dims [2, 4]
    /// let summed = tt.partial_sum(&[1]).unwrap();
    /// assert_eq!(summed.site_dims(), vec![2, 4]);
    ///
    /// // Each remaining entry = 1.0 * 3 (summed over dim=3)
    /// let val = summed.evaluate(&[0, 0]).unwrap();
    /// assert!((val - 3.0).abs() < 1e-12);
    /// ```
    pub fn partial_sum(&self, dims: &[usize]) -> Result<SimpleTensorTrain<T>> {
        use tensor4all_tensorbackend::{mat_mul, Matrix};

        let n = self.len();
        if n == 0 {
            return Ok(Self {
                tensors: Vec::new(),
            });
        }

        // Validate dims
        for &d in dims {
            if d >= n {
                return Err(TensorTrainError::InvalidOperation {
                    message: format!("Dimension {} out of range (0..{})", d, n),
                });
            }
        }

        let mut result_tensors: Vec<Tensor3<T>> = Vec::new();

        // Tprod: accumulator matrix, starts as 1x1 identity
        let mut tprod = Matrix::zeros(1, 1);
        tprod[[0, 0]] = T::one();

        for site in 0..n {
            let t = self.site_tensor(site);
            let left_dim = t.left_dim();
            let site_dim = t.site_dim();
            let right_dim = t.right_dim();

            if dims.contains(&site) {
                // Sum over site index: result is (left_dim, right_dim) matrix
                // sum(T, dims=2)[:, 1, :] in Julia
                let mut site_sum = Matrix::zeros(left_dim, right_dim);
                for l in 0..left_dim {
                    for r in 0..right_dim {
                        let mut acc = T::zero();
                        for s in 0..site_dim {
                            acc = acc + *t.get3(l, s, r);
                        }
                        site_sum[[l, r]] = acc;
                    }
                }
                // Tprod = Tprod * site_sum
                tprod = mat_mul(&tprod, &site_sum).map_err(|err| {
                    TensorTrainError::InvalidOperation {
                        message: format!("dimension summation matrix multiply failed: {err}"),
                    }
                })?;
            } else {
                // Keep this dimension: multiply Tprod into the site tensor
                // Tprod (tprod_rows, left_dim) * T reshaped to (left_dim, site_dim * right_dim)
                let tprod_rows = tprod.nrows();
                let mut t_reshaped = Matrix::zeros(left_dim, site_dim * right_dim);
                for l in 0..left_dim {
                    for s in 0..site_dim {
                        for r in 0..right_dim {
                            t_reshaped[[l, s * right_dim + r]] = *t.get3(l, s, r);
                        }
                    }
                }
                let product = mat_mul(&tprod, &t_reshaped).map_err(|err| {
                    TensorTrainError::InvalidOperation {
                        message: format!("dimension retention matrix multiply failed: {err}"),
                    }
                })?;

                // Reshape product (tprod_rows, site_dim * right_dim)
                // into tensor (tprod_rows, site_dim, right_dim)
                let mut new_tensor = tensor3_zeros(tprod_rows, site_dim, right_dim);
                for l in 0..tprod_rows {
                    for s in 0..site_dim {
                        for r in 0..right_dim {
                            new_tensor.set3(l, s, r, product[[l, s * right_dim + r]]);
                        }
                    }
                }
                result_tensors.push(new_tensor);

                // Reset Tprod to identity of size right_dim
                tprod = Matrix::zeros(right_dim, right_dim);
                for i in 0..right_dim {
                    tprod[[i, i]] = T::one();
                }
            }
        }

        if result_tensors.is_empty() {
            // All dims summed → return 1-site TT wrapping scalar
            // tprod should be 1×1
            let scalar = tprod[[0, 0]];
            let mut t = tensor3_zeros(1, 1, 1);
            t.set3(0, 0, 0, scalar);
            return SimpleTensorTrain::new(vec![t]);
        }

        // Contract final Tprod into last result tensor
        let last = result_tensors.last().ok_or(TensorTrainError::Empty)?;
        let last_left = last.left_dim();
        let last_site = last.site_dim();
        let last_right = last.right_dim();
        let tprod_cols = tprod.ncols();

        // Reshape last tensor to (last_left * last_site, last_right)
        let mut last_mat = Matrix::zeros(last_left * last_site, last_right);
        for l in 0..last_left {
            for s in 0..last_site {
                for r in 0..last_right {
                    last_mat[[l * last_site + s, r]] = *last.get3(l, s, r);
                }
            }
        }

        // Multiply: last_mat * Tprod → (last_left * last_site, tprod_cols)
        let contracted =
            mat_mul(&last_mat, &tprod).map_err(|err| TensorTrainError::InvalidOperation {
                message: format!("final dimension summation multiply failed: {err}"),
            })?;

        // Reshape back to tensor (last_left, last_site, tprod_cols)
        let mut new_last = tensor3_zeros(last_left, last_site, tprod_cols);
        for l in 0..last_left {
            for s in 0..last_site {
                for r in 0..tprod_cols {
                    new_last.set3(l, s, r, contracted[[l * last_site + s, r]]);
                }
            }
        }
        *result_tensors.last_mut().ok_or(TensorTrainError::Empty)? = new_last;

        SimpleTensorTrain::new(result_tensors)
    }
}

impl<T: TTScalar> AbstractTensorTrain<T> for SimpleTensorTrain<T> {
    fn len(&self) -> usize {
        self.tensors.len()
    }

    fn site_tensor(&self, i: usize) -> &Tensor3<T> {
        &self.tensors[i]
    }

    fn site_tensors(&self) -> &[Tensor3<T>] {
        &self.tensors
    }
}

#[cfg(test)]
mod tests;
