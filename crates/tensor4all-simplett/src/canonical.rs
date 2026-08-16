//! Canonical forms for tensor trains
//!
//! This module provides tensor train representations with canonical forms:
//! - `SiteTensorTrain`: Center-canonical form where tensors left of center are
//!   left-orthogonal and tensors right of center are right-orthogonal.

use std::ops::Range;

use crate::error::{Result, TensorTrainError};
use crate::tensortrain::SimpleTensorTrain;
use crate::traits::{AbstractTensorTrain, TTScalar};
use crate::types::{tensor3_zeros, Tensor3, Tensor3Ops};
use tensor4all_core::Scalar;
use tensor4all_core::{rrlu, RrLUOptions};
use tensor4all_tensorbackend::{mat_mul, transpose, Matrix};

/// Compute QR decomposition using rank-revealing LU with left-orthogonal output
fn qr_decomp<T: TTScalar + Scalar>(matrix: &Matrix<T>) -> Result<(Matrix<T>, Matrix<T>)> {
    let options = RrLUOptions {
        max_bond_dim: matrix.ncols().min(matrix.nrows()),
        rel_tol: 0.0, // No truncation
        abs_tol: 0.0,
        left_orthogonal: true,
    };
    let lu = rrlu(matrix, Some(options)).map_err(|err| TensorTrainError::InvalidOperation {
        message: format!("QR decomposition failed: {err}"),
    })?;
    Ok((lu.left(true), lu.right(true)))
}

/// Compute LQ decomposition (transpose, QR, transpose)
fn lq_decomp<T: TTScalar + Scalar>(matrix: &Matrix<T>) -> Result<(Matrix<T>, Matrix<T>)> {
    let at = transpose(matrix);
    let (qt, lt) = qr_decomp(&at)?;
    Ok((transpose(&lt), transpose(&qt)))
}

/// Convert Tensor3 to Matrix with left dimensions flattened
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

/// Convert Tensor3 to Matrix with right dimensions flattened
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

/// Site Tensor Train with center canonical form
///
/// A tensor train where:
/// - Tensors at indices < center are left-orthogonal
/// - Tensors at indices > center are right-orthogonal
/// - The tensor at the center index is the "center" tensor
///
/// # Examples
///
/// ```
/// use tensor4all_simplett::{SimpleTensorTrain, AbstractTensorTrain, SiteTensorTrain};
///
/// // Build a constant tensor train and convert to center-canonical form.
/// let tt = SimpleTensorTrain::<f64>::constant(&[2, 3, 4], 1.0);
/// let stt = SiteTensorTrain::from_tensor_train(&tt, 1).unwrap();
///
/// // The center is at site 1.
/// assert_eq!(stt.center(), 1);
/// assert_eq!(stt.len(), 3);
///
/// // Converting back preserves tensor train values.
/// let tt2 = stt.to_tensor_train();
/// let val = tt2.evaluate(&[0, 1, 2]).unwrap();
/// assert!((val - 1.0).abs() < 1e-12);
/// ```
#[derive(Debug, Clone)]
pub struct SiteTensorTrain<T: TTScalar> {
    /// Site tensors
    tensors: Vec<Tensor3<T>>,
    /// Center index (0-based)
    center: usize,
    /// Active partition range
    partition: Range<usize>,
}

impl<T: TTScalar + Scalar + Default> SiteTensorTrain<T> {
    /// Create a new SiteTensorTrain from tensors with specified center
    /// # Errors
    ///
    /// Returns an error when the tensor train is empty or structurally invalid (an
    /// /// invalid-state or shape mismatch failure).
    ///
    pub fn new(tensors: Vec<Tensor3<T>>, center: usize) -> Result<Self> {
        let n = tensors.len();
        if n == 0 {
            return Err(TensorTrainError::Empty);
        }
        if center >= n {
            return Err(TensorTrainError::InvalidOperation {
                message: format!("Center {} is out of range for {} tensors", center, n),
            });
        }

        // Validate dimensions
        for i in 0..n.saturating_sub(1) {
            if tensors[i].right_dim() != tensors[i + 1].left_dim() {
                return Err(TensorTrainError::DimensionMismatch { site: i });
            }
        }

        let mut result = Self {
            tensors,
            center,
            partition: 0..n,
        };
        result.canonicalize()?;
        Ok(result)
    }

    /// Create from SimpleTensorTrain with specified center
    /// # Errors
    ///
    /// Returns an error when the tensor train is structurally invalid (an
    /// /// invalid-state or shape mismatch failure).
    ///
    pub fn from_tensor_train(tt: &SimpleTensorTrain<T>, center: usize) -> Result<Self> {
        let tensors = tt.site_tensors().to_vec();
        Self::new(tensors, center)
    }

    /// Get the center index
    pub fn center(&self) -> usize {
        self.center
    }

    /// Get the partition range
    pub fn partition(&self) -> &Range<usize> {
        &self.partition
    }

    /// Get mutable access to site tensors
    pub fn site_tensors_mut(&mut self) -> &mut [Tensor3<T>] {
        &mut self.tensors
    }

    /// Canonicalize the tensor train around the center
    fn canonicalize(&mut self) -> Result<()> {
        let n = self.len();
        if n <= 1 {
            return Ok(());
        }

        // Left sweep: make tensors [0..center) left-orthogonal
        for i in 0..self.center {
            self.make_left_orthogonal(i)?;
        }

        // Right sweep: make tensors (center..n] right-orthogonal
        for i in (self.center + 1..n).rev() {
            self.make_right_orthogonal(i)?;
        }
        Ok(())
    }

    /// Make tensor at site i left-orthogonal, pushing R to site i+1
    fn make_left_orthogonal(&mut self, i: usize) -> Result<()> {
        if i >= self.len() - 1 {
            return Ok(());
        }

        let left_dim = self.tensors[i].left_dim();
        let site_dim = self.tensors[i].site_dim();

        // Reshape to (left_dim * site_dim, right_dim)
        let mat = tensor3_to_left_matrix(&self.tensors[i]);
        let (q, r) = qr_decomp(&mat)?;

        let new_bond_dim = q.ncols();

        // Update current tensor with Q
        let mut new_tensor = tensor3_zeros(left_dim, site_dim, new_bond_dim);
        for l in 0..left_dim {
            for s in 0..site_dim {
                for b in 0..new_bond_dim {
                    let row = l * site_dim + s;
                    if row < q.nrows() && b < q.ncols() {
                        new_tensor.set3(l, s, b, q[[row, b]]);
                    }
                }
            }
        }
        self.tensors[i] = new_tensor;

        // Contract R with next tensor
        let next_site_dim = self.tensors[i + 1].site_dim();
        let next_right_dim = self.tensors[i + 1].right_dim();
        let next_mat = tensor3_to_right_matrix(&self.tensors[i + 1]);

        // R * next_mat
        let contracted =
            mat_mul(&r, &next_mat).map_err(|err| TensorTrainError::InvalidOperation {
                message: format!("left-orthogonalization matrix multiply failed: {err}"),
            })?;

        // Update next tensor
        let mut new_next_tensor = tensor3_zeros(new_bond_dim, next_site_dim, next_right_dim);
        for l in 0..new_bond_dim {
            for s in 0..next_site_dim {
                for r_idx in 0..next_right_dim {
                    new_next_tensor.set3(l, s, r_idx, contracted[[l, s * next_right_dim + r_idx]]);
                }
            }
        }
        self.tensors[i + 1] = new_next_tensor;
        Ok(())
    }

    /// Make tensor at site i right-orthogonal, pushing L to site i-1
    fn make_right_orthogonal(&mut self, i: usize) -> Result<()> {
        if i == 0 {
            return Ok(());
        }

        let site_dim = self.tensors[i].site_dim();
        let right_dim = self.tensors[i].right_dim();

        // Reshape to (left_dim, site_dim * right_dim)
        let mat = tensor3_to_right_matrix(&self.tensors[i]);
        let (l_mat, q) = lq_decomp(&mat)?;

        let new_bond_dim = q.nrows();

        // Update current tensor with Q
        let mut new_tensor = tensor3_zeros(new_bond_dim, site_dim, right_dim);
        for l in 0..new_bond_dim {
            for s in 0..site_dim {
                for r in 0..right_dim {
                    new_tensor.set3(l, s, r, q[[l, s * right_dim + r]]);
                }
            }
        }
        self.tensors[i] = new_tensor;

        // Contract previous tensor with L
        let prev_left_dim = self.tensors[i - 1].left_dim();
        let prev_site_dim = self.tensors[i - 1].site_dim();
        let prev_mat = tensor3_to_left_matrix(&self.tensors[i - 1]);

        // prev_mat * L
        let contracted =
            mat_mul(&prev_mat, &l_mat).map_err(|err| TensorTrainError::InvalidOperation {
                message: format!("right-orthogonalization matrix multiply failed: {err}"),
            })?;

        // Update previous tensor
        let mut new_prev_tensor = tensor3_zeros(prev_left_dim, prev_site_dim, new_bond_dim);
        for l in 0..prev_left_dim {
            for s in 0..prev_site_dim {
                for r in 0..new_bond_dim {
                    new_prev_tensor.set3(l, s, r, contracted[[l * prev_site_dim + s, r]]);
                }
            }
        }
        self.tensors[i - 1] = new_prev_tensor;
        Ok(())
    }

    /// Move the center one position to the right
    /// # Errors
    ///
    /// Returns an error when the center cannot be moved (a shape or index mismatch,
    /// /// or a backend failure).
    ///
    pub fn move_center_right(&mut self) -> Result<()> {
        if self.center >= self.len() - 1 {
            return Err(TensorTrainError::InvalidOperation {
                message: "Cannot move center right: already at rightmost position".to_string(),
            });
        }

        self.make_left_orthogonal(self.center)?;
        self.center += 1;
        Ok(())
    }

    /// Move the center one position to the left
    /// # Errors
    ///
    /// Returns an error when the center cannot be moved (a shape or index mismatch,
    /// /// or a backend failure).
    ///
    pub fn move_center_left(&mut self) -> Result<()> {
        if self.center == 0 {
            return Err(TensorTrainError::InvalidOperation {
                message: "Cannot move center left: already at leftmost position".to_string(),
            });
        }

        self.make_right_orthogonal(self.center)?;
        self.center -= 1;
        Ok(())
    }

    /// Move the center to a specific position
    /// # Errors
    ///
    /// Returns an error when the site is out of bounds or the center cannot be
    /// /// set (a shape or index mismatch failure).
    ///
    pub fn set_center(&mut self, new_center: usize) -> Result<()> {
        if new_center >= self.len() {
            return Err(TensorTrainError::InvalidOperation {
                message: format!(
                    "New center {} is out of range for {} tensors",
                    new_center,
                    self.len()
                ),
            });
        }

        while self.center < new_center {
            self.move_center_right()?;
        }
        while self.center > new_center {
            self.move_center_left()?;
        }
        Ok(())
    }

    /// Convert to a regular SimpleTensorTrain
    pub fn to_tensor_train(&self) -> SimpleTensorTrain<T> {
        SimpleTensorTrain::from_tensors_unchecked(self.tensors.clone())
    }

    /// Set the tensor at a specific site
    ///
    /// Note: This may invalidate the canonical form. Use with caution.
    pub fn set_site_tensor(&mut self, i: usize, tensor: Tensor3<T>) {
        self.tensors[i] = tensor;
    }

    /// Set two adjacent site tensors (useful for TEBD-like algorithms)
    /// # Errors
    ///
    /// Returns an error when the two-site tensors are incompatible (a shape
    /// /// mismatch) or the merge fails (a backend failure).
    ///
    pub fn set_two_site_tensors(
        &mut self,
        i: usize,
        tensor1: Tensor3<T>,
        tensor2: Tensor3<T>,
    ) -> Result<()> {
        if i >= self.len() - 1 {
            return Err(TensorTrainError::InvalidOperation {
                message: format!(
                    "Cannot set two-site tensors at site {} (max {})",
                    i,
                    self.len() - 2
                ),
            });
        }

        self.tensors[i] = tensor1;
        self.tensors[i + 1] = tensor2;
        Ok(())
    }
}

impl<T: TTScalar + Scalar + Default> AbstractTensorTrain<T> for SiteTensorTrain<T> {
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

/// Center canonicalize a vector of tensors in place
///
/// After this call, tensors at indices `< center` are left-orthogonal and
/// tensors at indices `> center` are right-orthogonal. The overall tensor
/// train represented by the slice is unchanged.
///
/// # Examples
///
/// ```
/// use tensor4all_simplett::{SimpleTensorTrain, AbstractTensorTrain, tensor3_zeros, Tensor3Ops};
/// use tensor4all_simplett::canonical::center_canonicalize;
///
/// // Start with a simple 3-site constant TT.
/// let tt = SimpleTensorTrain::<f64>::constant(&[2, 3, 2], 1.0);
/// let mut tensors: Vec<_> = tt.site_tensors().to_vec();
///
/// // Canonicalize around site 1.
/// center_canonicalize(&mut tensors, 1).unwrap();
///
/// // Rebuild TT from the canonicalized tensors; values are preserved.
/// let tt2 = SimpleTensorTrain::new(tensors).unwrap();
/// let val = tt2.evaluate(&[0, 1, 0]).unwrap();
/// assert!((val - 1.0).abs() < 1e-12);
/// ```
///
/// # Errors
///
/// Returns an error when the canonicalization fails (a shape or index
/// /// mismatch, or a backend failure).
///
pub fn center_canonicalize<T: TTScalar + Scalar + Default>(
    tensors: &mut [Tensor3<T>],
    center: usize,
) -> Result<()> {
    let n = tensors.len();
    if n <= 1 || center >= n {
        return Ok(());
    }

    // Left sweep: make tensors [0..center) left-orthogonal
    for i in 0..center {
        let left_dim = tensors[i].left_dim();
        let site_dim = tensors[i].site_dim();

        let mat = tensor3_to_left_matrix(&tensors[i]);
        let (q, r) = qr_decomp(&mat)?;

        let new_bond_dim = q.ncols();

        // Update current tensor
        let mut new_tensor = tensor3_zeros(left_dim, site_dim, new_bond_dim);
        for l in 0..left_dim {
            for s in 0..site_dim {
                for b in 0..new_bond_dim {
                    let row = l * site_dim + s;
                    if row < q.nrows() && b < q.ncols() {
                        new_tensor.set3(l, s, b, q[[row, b]]);
                    }
                }
            }
        }
        tensors[i] = new_tensor;

        // Contract R with next tensor
        if i + 1 < n {
            let next_site_dim = tensors[i + 1].site_dim();
            let next_right_dim = tensors[i + 1].right_dim();
            let next_mat = tensor3_to_right_matrix(&tensors[i + 1]);

            let contracted =
                mat_mul(&r, &next_mat).map_err(|err| TensorTrainError::InvalidOperation {
                    message: format!("left sweep matrix multiply failed: {err}"),
                })?;

            let mut new_next_tensor = tensor3_zeros(new_bond_dim, next_site_dim, next_right_dim);
            for l in 0..new_bond_dim {
                for s in 0..next_site_dim {
                    for r_idx in 0..next_right_dim {
                        new_next_tensor.set3(
                            l,
                            s,
                            r_idx,
                            contracted[[l, s * next_right_dim + r_idx]],
                        );
                    }
                }
            }
            tensors[i + 1] = new_next_tensor;
        }
    }

    // Right sweep: make tensors (center..n] right-orthogonal
    for i in (center + 1..n).rev() {
        let site_dim = tensors[i].site_dim();
        let right_dim = tensors[i].right_dim();

        let mat = tensor3_to_right_matrix(&tensors[i]);
        let (l_mat, q) = lq_decomp(&mat)?;

        let new_bond_dim = q.nrows();

        // Update current tensor
        let mut new_tensor = tensor3_zeros(new_bond_dim, site_dim, right_dim);
        for l in 0..new_bond_dim {
            for s in 0..site_dim {
                for r in 0..right_dim {
                    new_tensor.set3(l, s, r, q[[l, s * right_dim + r]]);
                }
            }
        }
        tensors[i] = new_tensor;

        // Contract L with previous tensor
        if i > 0 {
            let prev_left_dim = tensors[i - 1].left_dim();
            let prev_site_dim = tensors[i - 1].site_dim();
            let prev_mat = tensor3_to_left_matrix(&tensors[i - 1]);

            let contracted =
                mat_mul(&prev_mat, &l_mat).map_err(|err| TensorTrainError::InvalidOperation {
                    message: format!("right sweep matrix multiply failed: {err}"),
                })?;

            let mut new_prev_tensor = tensor3_zeros(prev_left_dim, prev_site_dim, new_bond_dim);
            for l in 0..prev_left_dim {
                for s in 0..prev_site_dim {
                    for r in 0..new_bond_dim {
                        new_prev_tensor.set3(l, s, r, contracted[[l * prev_site_dim + s, r]]);
                    }
                }
            }
            tensors[i - 1] = new_prev_tensor;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
