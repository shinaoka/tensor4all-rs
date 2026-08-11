//! Compression algorithms for tensor trains

use crate::einsum_helper::{tensor_to_col_major_vec, typed_tensor_from_col_major_slice};
use crate::error::{Result, TensorTrainError};
use crate::tensortrain::SimpleTensorTrain;
use crate::traits::{AbstractTensorTrain, TTScalar};
use crate::types::{tensor3_zeros, Tensor3, Tensor3Ops};
use tenferro_tensor::TensorScalar;
use tensor4all_tcicore::Scalar;
use tensor4all_tcicore::{rrlu, AbstractMatrixCI, MatrixLUCI, RrLUOptions};
use tensor4all_tensorbackend::{mat_mul, BackendLinalgScalar, Matrix};

/// Matrix decomposition method used during TT compression.
///
/// The method controls how each bond is factored during the right-to-left
/// truncation sweep.
///
/// # Examples
///
/// ```
/// use tensor4all_simplett::CompressionMethod;
///
/// let method = CompressionMethod::default();
/// assert_eq!(method, CompressionMethod::LU);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CompressionMethod {
    /// Rank-revealing LU decomposition (default).
    ///
    /// Fast and robust for most use cases.
    #[default]
    LU,
    /// Cross-interpolation (CI) decomposition.
    ///
    /// Can be faster than LU for very large bond dimensions because it
    /// avoids dense matrix operations.
    CI,
    /// Singular value decomposition (SVD).
    ///
    /// Gives the optimal low-rank approximation (Eckart--Young theorem)
    /// but is more expensive than LU or CI.
    SVD,
}

/// Configuration for tensor train compression.
///
/// Controls the accuracy-vs-cost trade-off when reducing bond dimensions
/// via [`SimpleTensorTrain::compress`] or [`SimpleTensorTrain::compressed`].
///
/// # Fields
///
/// | Field | Default | Meaning |
/// |-------|---------|---------|
/// | `method` | `LU` | Decomposition algorithm (see [`CompressionMethod`]) |
/// | `tolerance` | `1e-12` | Relative truncation threshold per bond |
/// | `max_bond_dim` | `usize::MAX` | Hard upper bound on any bond dimension |
/// | `normalize_error` | `true` | Whether error is measured relative to the norm |
///
/// # Choosing `tolerance`
///
/// - `1e-12` (default): near machine precision, almost lossless.
/// - `1e-8` to `1e-6`: good for most scientific applications.
/// - `1e-4` to `1e-2`: aggressive compression, useful for exploratory work.
///
/// Tighter tolerances produce larger bond dimensions and slower evaluation.
///
/// # Examples
///
/// ```
/// use tensor4all_simplett::{CompressionOptions, CompressionMethod};
///
/// // Default: LU with tolerance 1e-12
/// let opts = CompressionOptions::default();
/// assert_eq!(opts.method, CompressionMethod::LU);
/// assert!((opts.tolerance - 1e-12).abs() < 1e-15);
/// assert_eq!(opts.max_bond_dim, usize::MAX);
///
/// // Custom: SVD with moderate tolerance and bond dim cap
/// let opts = CompressionOptions {
///     method: CompressionMethod::SVD,
///     tolerance: 1e-6,
///     max_bond_dim: 50,
///     ..Default::default()
/// };
/// assert_eq!(opts.method, CompressionMethod::SVD);
/// ```
#[derive(Debug, Clone)]
pub struct CompressionOptions {
    /// Decomposition method (LU, CI, or SVD).
    pub method: CompressionMethod,
    /// Relative truncation tolerance per bond.
    ///
    /// Singular values (or pivots) smaller than `tolerance * sigma_max` are
    /// discarded. Smaller values preserve more accuracy but produce larger
    /// bond dimensions.
    pub tolerance: f64,
    /// Hard upper bound on any bond dimension.
    ///
    /// Even if the tolerance would allow a larger rank, the bond dimension
    /// is capped at this value. Set to `usize::MAX` (default) for no limit.
    pub max_bond_dim: usize,
    /// Whether to normalize the truncation error by the tensor norm.
    pub normalize_error: bool,
}

impl Default for CompressionOptions {
    fn default() -> Self {
        Self {
            method: CompressionMethod::LU,
            tolerance: 1e-12,
            max_bond_dim: usize::MAX,
            normalize_error: true,
        }
    }
}

/// Convert Tensor3 to Matrix for factorization (left matrix view)
fn tensor3_to_left_matrix<T: TTScalar + Scalar + Default>(tensor: &Tensor3<T>) -> Matrix<T> {
    let left_dim = tensor.left_dim();
    let site_dim = tensor.site_dim();
    let right_dim = tensor.right_dim();
    let rows = left_dim * site_dim;
    let cols = right_dim;

    let mut mat = Matrix::zeros(rows, cols);
    for l in 0..left_dim {
        for s in 0..site_dim {
            for r in 0..right_dim {
                mat[[l * site_dim + s, r]] = *tensor.get3(l, s, r);
            }
        }
    }
    mat
}

/// Convert Tensor3 to Matrix for factorization (right matrix view)
fn tensor3_to_right_matrix<T: TTScalar + Scalar + Default>(tensor: &Tensor3<T>) -> Matrix<T> {
    let left_dim = tensor.left_dim();
    let site_dim = tensor.site_dim();
    let right_dim = tensor.right_dim();
    let rows = left_dim;
    let cols = site_dim * right_dim;

    let mut mat = Matrix::zeros(rows, cols);
    for l in 0..left_dim {
        for s in 0..site_dim {
            for r in 0..right_dim {
                mat[[l, s * right_dim + r]] = *tensor.get3(l, s, r);
            }
        }
    }
    mat
}

/// Factorize a matrix into left and right factors
fn factorize<T>(
    matrix: &Matrix<T>,
    method: CompressionMethod,
    tolerance: f64,
    max_bond_dim: usize,
    left_orthogonal: bool,
) -> crate::error::Result<(Matrix<T>, Matrix<T>, usize)>
where
    T: TTScalar
        + Scalar
        + tensor4all_tcicore::MatrixLuciScalar
        + Default
        + Copy
        + BackendLinalgScalar
        + TensorScalar,
    f64: From<<T as TensorScalar>::Real>,
{
    let reltol = if tolerance > 0.0 { tolerance } else { 1e-14 };
    let abstol = 0.0;

    let options = RrLUOptions {
        max_rank: max_bond_dim,
        rel_tol: reltol,
        abs_tol: abstol,
        left_orthogonal,
    };

    match method {
        CompressionMethod::LU => {
            let lu = rrlu(matrix, Some(options))?;
            let left = lu.left(true); // permuted
            let right = lu.right(true); // permuted
            let npivots = lu.npivots();
            Ok((left, right, npivots))
        }
        CompressionMethod::CI => {
            let luci = MatrixLUCI::from_matrix(matrix, Some(options))?;
            let left = luci.left();
            let right = luci.right();
            let npivots = luci.rank();
            Ok((left, right, npivots))
        }
        CompressionMethod::SVD => factorize_svd(matrix, tolerance, max_bond_dim, left_orthogonal),
    }
}

/// SVD-based factorization for TT compression
fn factorize_svd<T>(
    matrix: &Matrix<T>,
    tolerance: f64,
    max_bond_dim: usize,
    left_orthogonal: bool,
) -> crate::error::Result<(Matrix<T>, Matrix<T>, usize)>
where
    T: TTScalar + Scalar + Default + Copy + BackendLinalgScalar + TensorScalar,
    f64: From<<T as TensorScalar>::Real>,
{
    let m = matrix.nrows();
    let n = matrix.ncols();

    if m == 0 || n == 0 {
        return Err(crate::error::TensorTrainError::InvalidOperation {
            message: "Cannot factorize empty matrix".to_string(),
        });
    }

    let a_tensor = typed_tensor_from_col_major_slice(matrix.as_col_major_slice(), &[m, n])
        .map_err(|err| TensorTrainError::InvalidOperation {
            message: format!("Failed to prepare SVD input matrix: {err}"),
        })?;

    let svd_result = tensor4all_tensorbackend::svd_backend(&a_tensor).map_err(|e| {
        crate::error::TensorTrainError::InvalidOperation {
            message: format!("SVD computation failed: {e:?}"),
        }
    })?;

    let u = Matrix::from_col_major_vec(
        svd_result.u.shape()[0],
        svd_result.u.shape()[1],
        tensor_to_col_major_vec(&svd_result.u),
    );
    let s_data: Vec<f64> = tensor_to_col_major_vec(&svd_result.s)
        .into_iter()
        .map(f64::from)
        .collect();
    let vt = Matrix::from_col_major_vec(
        svd_result.vt.shape()[0],
        svd_result.vt.shape()[1],
        tensor_to_col_major_vec(&svd_result.vt),
    );

    // Determine rank based on tolerance and max_bond_dim
    let min_dim = m.min(n);
    let s_max = if !s_data.is_empty() { s_data[0] } else { 0.0 };

    let mut rank = 0;
    for &singular_value in s_data.iter().take(min_dim) {
        if rank >= max_bond_dim {
            break;
        }
        if singular_value < tolerance * s_max {
            break;
        }
        rank += 1;
    }
    rank = rank.max(1);

    // Build result matrices
    let mut left = Matrix::zeros(m, rank);
    let mut right = Matrix::zeros(rank, n);

    if left_orthogonal {
        // Left = U[:, :rank], Right = diag(S[:rank]) * Vt[:rank, :]
        for i in 0..m {
            for j in 0..rank {
                left[[i, j]] = u[[i, j]];
            }
        }
        for i in 0..rank {
            let sv = T::from_f64(s_data[i]);
            for j in 0..n {
                right[[i, j]] = sv * vt[[i, j]];
            }
        }
    } else {
        // Left = U[:, :rank] * diag(S[:rank]), Right = Vt[:rank, :]
        for i in 0..m {
            for j in 0..rank {
                let sv = T::from_f64(s_data[j]);
                left[[i, j]] = u[[i, j]] * sv;
            }
        }
        for i in 0..rank {
            for j in 0..n {
                right[[i, j]] = vt[[i, j]];
            }
        }
    }

    Ok((left, right, rank))
}

impl<T: TTScalar + Scalar + Default> SimpleTensorTrain<T> {
    /// Compress the tensor train **in place**, reducing bond dimensions.
    ///
    /// The algorithm performs two sweeps:
    /// 1. **Left-to-right**: orthogonalize each bond without truncation.
    /// 2. **Right-to-left**: truncate each bond according to `options`.
    ///
    /// After compression, the tensor train approximates the original within
    /// the specified tolerance while using smaller bond dimensions.
    ///
    /// # Errors
    ///
    /// Returns an error when the operation fails (a shape or index mismatch, an
    /// /// SVD or non-convergence failure, or a backend failure).
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_simplett::{SimpleTensorTrain, AbstractTensorTrain, CompressionOptions};
    ///
    /// // Add two constant TTs to get bond dim 2, then compress back to 1
    /// let a = SimpleTensorTrain::<f64>::constant(&[2, 3, 4], 1.0);
    /// let b = SimpleTensorTrain::<f64>::constant(&[2, 3, 4], 2.0);
    /// let mut sum = a.add(&b).unwrap(); // bond dim = 2
    /// assert_eq!(sum.rank(), 2);
    ///
    /// sum.compress(&CompressionOptions::default()).unwrap();
    /// assert_eq!(sum.rank(), 1); // compressed back to optimal
    ///
    /// // Values are preserved: 1.0 + 2.0 = 3.0
    /// assert!((sum.evaluate(&[0, 0, 0]).unwrap() - 3.0).abs() < 1e-10);
    /// ```
    pub fn compress(&mut self, options: &CompressionOptions) -> Result<()>
    where
        T: tensor4all_tcicore::MatrixLuciScalar + Copy + BackendLinalgScalar + TensorScalar,
        f64: From<<T as TensorScalar>::Real>,
    {
        let n = self.len();
        if n <= 1 {
            return Ok(());
        }

        let tensors = self.site_tensors_mut();

        // Left-to-right sweep: make left-orthogonal without truncation
        for ell in 0..n - 1 {
            let left_dim = tensors[ell].left_dim();
            let site_dim = tensors[ell].site_dim();

            // Reshape to matrix: (left_dim * site_dim, right_dim)
            let mat = tensor3_to_left_matrix(&tensors[ell]);

            // Factorize without truncation
            let (left_factor, right_factor, new_bond_dim) = factorize(
                &mat,
                options.method,
                0.0,        // No truncation in left sweep
                usize::MAX, // No max bond dim in left sweep
                true,       // left orthogonal
            )?;

            // Update current tensor
            let mut new_tensor = tensor3_zeros(left_dim, site_dim, new_bond_dim);
            for l in 0..left_dim {
                for s in 0..site_dim {
                    for r in 0..new_bond_dim {
                        let row = l * site_dim + s;
                        if row < left_factor.nrows() && r < left_factor.ncols() {
                            new_tensor.set3(l, s, r, left_factor[[row, r]]);
                        }
                    }
                }
            }
            tensors[ell] = new_tensor;

            // Contract right_factor with next tensor
            let next_site_dim = tensors[ell + 1].site_dim();
            let next_right_dim = tensors[ell + 1].right_dim();

            // Build next tensor as matrix (old_left_dim, site_dim * right_dim)
            let next_mat = tensor3_to_right_matrix(&tensors[ell + 1]);

            // Multiply: right_factor * next_mat
            let contracted = mat_mul(&right_factor, &next_mat).map_err(|err| {
                TensorTrainError::InvalidOperation {
                    message: format!("left-to-right compression multiply failed: {err}"),
                }
            })?;

            // Update next tensor
            let mut new_next_tensor = tensor3_zeros(new_bond_dim, next_site_dim, next_right_dim);
            for l in 0..new_bond_dim {
                for s in 0..next_site_dim {
                    for r in 0..next_right_dim {
                        new_next_tensor.set3(l, s, r, contracted[[l, s * next_right_dim + r]]);
                    }
                }
            }
            tensors[ell + 1] = new_next_tensor;
        }

        // Right-to-left sweep: truncate
        for ell in (1..n).rev() {
            let site_dim = tensors[ell].site_dim();
            let right_dim = tensors[ell].right_dim();

            // Reshape to matrix: (left_dim, site_dim * right_dim)
            let mat = tensor3_to_right_matrix(&tensors[ell]);

            // Factorize with truncation
            let (left_factor, right_factor, new_bond_dim) = factorize(
                &mat,
                options.method,
                options.tolerance,
                options.max_bond_dim,
                false, // right orthogonal
            )?;

            // Update current tensor from right_factor
            let mut new_tensor = tensor3_zeros(new_bond_dim, site_dim, right_dim);
            for l in 0..new_bond_dim {
                for s in 0..site_dim {
                    for r in 0..right_dim {
                        new_tensor.set3(l, s, r, right_factor[[l, s * right_dim + r]]);
                    }
                }
            }
            tensors[ell] = new_tensor;

            // Contract previous tensor with left_factor
            let prev_left_dim = tensors[ell - 1].left_dim();
            let prev_site_dim = tensors[ell - 1].site_dim();

            // Build prev tensor as matrix (left_dim * site_dim, old_right_dim)
            let prev_mat = tensor3_to_left_matrix(&tensors[ell - 1]);

            // Multiply: prev_mat * left_factor
            let contracted = mat_mul(&prev_mat, &left_factor).map_err(|err| {
                TensorTrainError::InvalidOperation {
                    message: format!("right-to-left compression multiply failed: {err}"),
                }
            })?;

            // Update prev tensor
            let mut new_prev_tensor = tensor3_zeros(prev_left_dim, prev_site_dim, new_bond_dim);
            for l in 0..prev_left_dim {
                for s in 0..prev_site_dim {
                    for r in 0..new_bond_dim {
                        new_prev_tensor.set3(l, s, r, contracted[[l * prev_site_dim + s, r]]);
                    }
                }
            }
            tensors[ell - 1] = new_prev_tensor;
        }

        Ok(())
    }

    /// Return a compressed copy of the tensor train (non-mutating).
    ///
    /// Equivalent to cloning and calling [`compress`](Self::compress).
    ///
    /// # Errors
    ///
    /// Returns an error when the operation fails (a shape or index mismatch, an
    /// /// SVD or non-convergence failure, or a backend failure).
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_simplett::{SimpleTensorTrain, AbstractTensorTrain, CompressionOptions};
    ///
    /// // A tensor train with redundant bond dimension can be compressed
    /// let tt = SimpleTensorTrain::<f64>::constant(&[2, 3, 2], 1.0);
    ///
    /// let opts = CompressionOptions::default();
    /// let compressed = tt.compressed(&opts).unwrap();
    ///
    /// // The compressed TT has the same number of sites
    /// assert_eq!(compressed.len(), tt.len());
    ///
    /// // Evaluations agree
    /// let val_orig = tt.evaluate(&[0, 1, 0]).unwrap();
    /// let val_comp = compressed.evaluate(&[0, 1, 0]).unwrap();
    /// assert!((val_orig - val_comp).abs() < 1e-10);
    /// ```
    pub fn compressed(&self, options: &CompressionOptions) -> Result<Self>
    where
        T: tensor4all_tcicore::MatrixLuciScalar + Copy + BackendLinalgScalar + TensorScalar,
        f64: From<<T as TensorScalar>::Real>,
    {
        let mut result = self.clone();
        result.compress(options)?;
        Ok(result)
    }
}

#[cfg(test)]
mod tests;
