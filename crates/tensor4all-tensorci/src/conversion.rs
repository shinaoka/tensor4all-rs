//! Conversion constructors for TensorCI state objects.

use crate::error::{Result, TCIError};
use crate::TensorCI2;
use tensor4all_core::{
    matrix_luci_factors_from_matrix, MatrixLuciScalar, MultiIndex, RrLUOptions, Scalar,
};
use tensor4all_simplett::{AbstractTensorTrain, SimpleTensorTrain, TTScalar, Tensor3, Tensor3Ops};
use tensor4all_tensorbackend::{mat_mul_owned, BlasMul, Matrix};

/// Options for constructing [`TensorCI2`] from a tensor train.
///
/// Use this with [`TensorCI2::from_tensor_train`] when converting an existing
/// tensor train into TensorCI2 index sets without materializing the full dense
/// tensor. [`crate::TCI2Options`] controls subsequent TCI2 optimization; this
/// type only controls the one-site LU extraction used during conversion.
///
/// When in doubt, use [`Default::default`]. Lower `tolerance` values and larger
/// `max_bond_dim` values preserve more of the input tensor train rank, while
/// smaller rank caps give cheaper downstream TensorCI2 states.
///
/// # Examples
///
/// ```
/// use tensor4all_tensorci::TensorCI2FromTensorTrainOptions;
///
/// let options = TensorCI2FromTensorTrainOptions {
///     tolerance: 1e-14,
///     max_bond_dim: Some(8),
///     max_iter: 4,
/// };
///
/// assert!((options.tolerance - 1e-14).abs() < 1e-20);
/// assert_eq!(options.max_bond_dim, Some(8));
/// assert_eq!(options.max_iter, 4);
/// ```
#[derive(Debug, Clone)]
pub struct TensorCI2FromTensorTrainOptions {
    /// Relative tolerance used during one-site index extraction.
    ///
    /// The default is `1e-12`. Smaller values preserve more pivots; larger
    /// values can reduce the extracted rank.
    pub tolerance: f64,
    /// Maximum bond dimension retained during index extraction.
    ///
    /// The default is `None`, meaning no explicit cap. Set this to the
    /// largest acceptable converted TCI rank for expensive downstream use.
    pub max_bond_dim: Option<usize>,
    /// Maximum number of alternating one-site index extraction sweeps.
    ///
    /// The default is `3`, matching the legacy Julia conversion constructor.
    /// At least the initial forward and backward sweeps are required.
    pub max_iter: usize,
}

impl Default for TensorCI2FromTensorTrainOptions {
    fn default() -> Self {
        Self {
            tolerance: 1e-12,
            max_bond_dim: None,
            max_iter: 3,
        }
    }
}

pub(crate) fn tensorci2_from_tensor_train<T>(
    mut tt: SimpleTensorTrain<T>,
    options: TensorCI2FromTensorTrainOptions,
) -> Result<TensorCI2<T>>
where
    T: Scalar + TTScalar + Default + MatrixLuciScalar + BlasMul,
{
    validate_options(&options)?;
    if tt.len() < 2 {
        return Err(TCIError::DimensionMismatch {
            message: "TensorCI2 conversion requires at least 2 tensor-train sites".to_string(),
        });
    }

    let local_dims = tt.site_dims();
    let (mut i_set, _) = sweep1site_get_indices(&mut tt, true, None, &options)?;
    let (mut j_set, mut pivot_errors) = sweep1site_get_indices(&mut tt, false, None, &options)?;

    for iter in 3..=options.max_iter {
        if iter % 2 == 1 {
            let mut filtered_j_set = j_set.clone();
            let (new_i_set, errors) =
                sweep1site_get_indices(&mut tt, true, Some(&mut filtered_j_set), &options)?;
            j_set = filtered_j_set;
            pivot_errors = errors;
            if new_i_set == i_set {
                break;
            }
            i_set = new_i_set;
        } else {
            let mut filtered_i_set = i_set.clone();
            let (new_j_set, errors) =
                sweep1site_get_indices(&mut tt, false, Some(&mut filtered_i_set), &options)?;
            i_set = filtered_i_set;
            pivot_errors = errors;
            if new_j_set == j_set {
                break;
            }
            j_set = new_j_set;
        }
    }

    let site_tensors = tt.into_site_tensors();
    let max_sample_value = max_site_tensor_abs(&site_tensors);
    let bond_errors = vec![0.0; local_dims.len().saturating_sub(1)];

    TensorCI2::from_parts_for_conversion(
        local_dims,
        i_set,
        j_set,
        site_tensors,
        pivot_errors,
        bond_errors,
        max_sample_value,
    )
}

fn validate_options(options: &TensorCI2FromTensorTrainOptions) -> Result<()> {
    if !options.tolerance.is_finite() || options.tolerance < 0.0 {
        return Err(TCIError::InvalidOperation {
            message: "TensorCI2 conversion tolerance must be finite and nonnegative".to_string(),
        });
    }
    if options.max_bond_dim == Some(0) {
        return Err(TCIError::InvalidOperation {
            message: "TensorCI2 conversion max_bond_dim must be nonzero".to_string(),
        });
    }
    if options.max_iter < 2 {
        return Err(TCIError::InvalidOperation {
            message: "TensorCI2 conversion max_iter must be at least 2".to_string(),
        });
    }
    Ok(())
}

fn sweep1site_get_indices<T>(
    tt: &mut SimpleTensorTrain<T>,
    forward: bool,
    spectator_indices: Option<&mut [Vec<MultiIndex>]>,
    options: &TensorCI2FromTensorTrainOptions,
) -> Result<(Vec<Vec<MultiIndex>>, Vec<f64>)>
where
    T: Scalar + TTScalar + Default + MatrixLuciScalar + BlasMul,
{
    let n = tt.len();
    let mut index_set = vec![vec![Vec::new()]];
    let mut pivot_errors = vec![0.0; tt.rank() + 1];
    let mut spectator_indices = spectator_indices;

    for step in 0..n - 1 {
        let site = if forward { step } else { n - step - 1 };
        let next_site = if forward { site + 1 } else { site - 1 };

        let tensors = tt.site_tensors_mut();
        if forward {
            let (left, right) = tensors.split_at_mut(next_site);
            let current = &mut left[site];
            let next = &mut right[0];
            sweep_pair(
                current,
                next,
                site,
                forward,
                &mut index_set,
                &mut pivot_errors,
                spectator_indices.as_deref_mut(),
                options,
            )?;
        } else {
            let (left, right) = tensors.split_at_mut(site);
            let next = &mut left[next_site];
            let current = &mut right[0];
            sweep_pair(
                current,
                next,
                site,
                forward,
                &mut index_set,
                &mut pivot_errors,
                spectator_indices.as_deref_mut(),
                options,
            )?;
        }
    }

    if forward {
        Ok((index_set, pivot_errors))
    } else {
        index_set.reverse();
        Ok((index_set, pivot_errors))
    }
}

#[allow(clippy::too_many_arguments)]
fn sweep_pair<T>(
    current: &mut Tensor3<T>,
    next: &mut Tensor3<T>,
    site: usize,
    forward: bool,
    index_set: &mut Vec<Vec<MultiIndex>>,
    pivot_errors: &mut Vec<f64>,
    spectator_indices: Option<&mut [Vec<MultiIndex>]>,
    options: &TensorCI2FromTensorTrainOptions,
) -> Result<()>
where
    T: Scalar + TTScalar + Default + MatrixLuciScalar + BlasMul,
{
    let current_shape = tensor_shape(current);
    let next_shape = tensor_shape(next);
    let current_matrix = group_indices(current, forward, false);
    let lu_options = RrLUOptions {
        max_bond_dim: options.max_bond_dim.unwrap_or(usize::MAX),
        rel_tol: options.tolerance,
        abs_tol: 0.0,
        left_orthogonal: forward,
    };
    let factors = matrix_luci_factors_from_matrix(&current_matrix, Some(lu_options))?;
    let rank = factors.rank;

    if forward {
        let base_indices = index_set.last().ok_or_else(|| TCIError::InvalidOperation {
            message: "TensorCI2 conversion index set is unexpectedly empty".to_string(),
        })?;
        let candidates = kronecker_append(base_indices, current_shape.1);
        index_set.push(select_multi_indices(&candidates, &factors.row_indices)?);
        if let Some(spectators) = spectator_indices {
            spectators[site] = select_multi_indices(&spectators[site], &factors.col_indices)?;
        }

        let next_matrix = group_indices(next, forward, true);
        let updated_next = mat_mul_owned(factors.right, next_matrix).map_err(|err| {
            TCIError::InvalidOperation {
                message: format!("TensorCI2 conversion forward transfer failed: {err}"),
            }
        })?;
        *current = split_indices(factors.left, current_shape, rank, forward, false)?;
        *next = split_indices(updated_next, next_shape, rank, forward, true)?;
    } else {
        let base_indices = index_set.last().ok_or_else(|| TCIError::InvalidOperation {
            message: "TensorCI2 conversion index set is unexpectedly empty".to_string(),
        })?;
        let candidates = kronecker_prepend(current_shape.1, base_indices);
        index_set.push(select_multi_indices(&candidates, &factors.col_indices)?);
        if let Some(spectators) = spectator_indices {
            spectators[site] = select_multi_indices(&spectators[site], &factors.row_indices)?;
        }

        let next_matrix = group_indices(next, forward, true);
        let updated_next =
            mat_mul_owned(next_matrix, factors.left).map_err(|err| TCIError::InvalidOperation {
                message: format!("TensorCI2 conversion backward transfer failed: {err}"),
            })?;
        *current = split_indices(factors.right, current_shape, rank, forward, false)?;
        *next = split_indices(updated_next, next_shape, rank, forward, true)?;
    }

    merge_pivot_errors(pivot_errors, &factors.pivot_errors);
    Ok(())
}

fn group_indices<T>(tensor: &Tensor3<T>, forward: bool, next: bool) -> Matrix<T>
where
    T: TTScalar + Default,
{
    let (data, nrows, ncols) = if forward != next {
        tensor.as_left_matrix()
    } else {
        tensor.as_right_matrix()
    };
    Matrix::from_col_major_vec(nrows, ncols, data)
}

fn split_indices<T>(
    matrix: Matrix<T>,
    shape: (usize, usize, usize),
    new_bond_dim: usize,
    forward: bool,
    next: bool,
) -> Result<Tensor3<T>>
where
    T: TTScalar + Default,
{
    let (left_dim, site_dim, right_dim) = shape;
    if forward != next {
        if matrix.nrows() != left_dim * site_dim || matrix.ncols() != new_bond_dim {
            return Err(TCIError::DimensionMismatch {
                message: format!(
                    "cannot reshape conversion matrix {}x{} into ({left_dim}, {site_dim}, {new_bond_dim})",
                    matrix.nrows(),
                    matrix.ncols()
                ),
            });
        }
        let mut tensor = tensor4all_simplett::tensor3_zeros(left_dim, site_dim, new_bond_dim);
        for r in 0..new_bond_dim {
            for l in 0..left_dim {
                for s in 0..site_dim {
                    tensor.set3(l, s, r, matrix[[l * site_dim + s, r]]);
                }
            }
        }
        Ok(tensor)
    } else {
        if matrix.nrows() != new_bond_dim || matrix.ncols() != site_dim * right_dim {
            return Err(TCIError::DimensionMismatch {
                message: format!(
                    "cannot reshape conversion matrix {}x{} into ({new_bond_dim}, {site_dim}, {right_dim})",
                    matrix.nrows(),
                    matrix.ncols()
                ),
            });
        }
        let mut tensor = tensor4all_simplett::tensor3_zeros(new_bond_dim, site_dim, right_dim);
        for l in 0..new_bond_dim {
            for s in 0..site_dim {
                for r in 0..right_dim {
                    tensor.set3(l, s, r, matrix[[l, s * right_dim + r]]);
                }
            }
        }
        Ok(tensor)
    }
}

fn tensor_shape<T>(tensor: &Tensor3<T>) -> (usize, usize, usize)
where
    T: TTScalar + Default,
{
    (tensor.left_dim(), tensor.site_dim(), tensor.right_dim())
}

fn kronecker_append(indices: &[MultiIndex], local_dim: usize) -> Vec<MultiIndex> {
    let mut result = Vec::with_capacity(indices.len() * local_dim);
    for index in indices {
        for local in 0..local_dim {
            let mut next = index.clone();
            next.push(local);
            result.push(next);
        }
    }
    result
}

fn kronecker_prepend(local_dim: usize, indices: &[MultiIndex]) -> Vec<MultiIndex> {
    let mut result = Vec::with_capacity(indices.len() * local_dim);
    for local in 0..local_dim {
        for index in indices {
            let mut next = Vec::with_capacity(index.len() + 1);
            next.push(local);
            next.extend(index.iter().copied());
            result.push(next);
        }
    }
    result
}

fn select_multi_indices(indices: &[MultiIndex], positions: &[usize]) -> Result<Vec<MultiIndex>> {
    let mut result = Vec::with_capacity(positions.len());
    for &position in positions {
        let index = indices
            .get(position)
            .ok_or_else(|| TCIError::IndexOutOfBounds {
                message: format!(
                    "conversion selected index {position} from set of length {}",
                    indices.len()
                ),
            })?;
        result.push(index.clone());
    }
    Ok(result)
}

fn merge_pivot_errors(target: &mut Vec<f64>, errors: &[f64]) {
    if target.len() < errors.len() {
        target.resize(errors.len(), 0.0);
    }
    for (target_error, &error) in target.iter_mut().zip(errors) {
        *target_error = target_error.max(error);
    }
}

fn max_site_tensor_abs<T>(site_tensors: &[Tensor3<T>]) -> f64
where
    T: Scalar + TTScalar + Default,
{
    site_tensors
        .iter()
        .flat_map(|tensor| tensor.iter())
        .map(|&value| Scalar::abs_sq(value).sqrt())
        .fold(0.0_f64, f64::max)
}

#[cfg(test)]
mod tests;
