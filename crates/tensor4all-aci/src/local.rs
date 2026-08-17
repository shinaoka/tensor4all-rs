use std::cell::RefCell;
use std::collections::HashMap;
#[cfg(test)]
use std::time::{Duration, Instant};

use crate::scalar::AciScalar;
use crate::{AciError, ElementwiseBatch, ElementwiseProblem, Result};
use tensor4all_simplett::{AbstractTensorTrain, Tensor3Ops};
use tensor4all_tensorbackend::{batched_mat_mul_same_shape_owned, mat_mul, mat_mul_owned, Matrix};

type LocalOperator<'a, T> =
    dyn for<'batch> Fn(ElementwiseBatch<'batch, T>, &mut [T]) -> Result<()> + 'a;

fn checked_mul(lhs: usize, rhs: usize, what: &str) -> Result<usize> {
    lhs.checked_mul(rhs)
        .ok_or_else(|| AciError::InvalidOptions {
            message: format!("{what} overflowed usize"),
        })
}

#[cfg(test)]
fn local_setup_batching_enabled() -> bool {
    std::env::var("T4A_ACI_DISABLE_BATCHED_LOCAL_SETUP").as_deref() != Ok("1")
}

#[cfg(not(test))]
fn local_setup_batching_enabled() -> bool {
    true
}

#[cfg(test)]
fn local_materialize_batching_enabled() -> bool {
    std::env::var("T4A_ACI_DISABLE_BATCHED_MATERIALIZE").as_deref() != Ok("1")
}

#[cfg(not(test))]
fn local_materialize_batching_enabled() -> bool {
    true
}

pub(crate) struct LocalBlockEvaluator<'a, T: AciScalar> {
    problem: &'a ElementwiseProblem<T>,
    operator: &'a LocalOperator<'a, T>,
    bond: usize,
    nrows: usize,
    ncols: usize,
    input_factors: Vec<LocalInputFactors<T>>,
    cache: RefCell<HashMap<usize, T>>,
    first_error: RefCell<Option<AciError>>,
    max_output_abs: RefCell<f64>,
}

#[cfg(test)]
#[derive(Clone, Copy, Default)]
pub(crate) struct LocalInputSetupTiming {
    pub(crate) shape_validation: Duration,
    pub(crate) dims: Duration,
    pub(crate) left_factor: Duration,
    pub(crate) right_factor: Duration,
}

#[derive(Clone)]
pub(crate) struct LocalInputFactors<T: AciScalar> {
    nrows: usize,
    ncols: usize,
    left_rows: usize,
    site_dim_left: usize,
    middle_dim: usize,
    site_dim_right: usize,
    right_cols: usize,
    left_values: Vec<T>,
    right_values: Vec<T>,
}

impl<T: AciScalar> LocalInputFactors<T> {
    pub(crate) fn value(&self, row: usize, col: usize) -> Result<T> {
        if row >= self.nrows {
            return Err(AciError::InvalidInitialGuess {
                message: format!(
                    "local row index {row} out of bounds for length {}",
                    self.nrows
                ),
            });
        }
        if col >= self.ncols {
            return Err(AciError::InvalidInitialGuess {
                message: format!(
                    "local column index {col} out of bounds for length {}",
                    self.ncols
                ),
            });
        }

        let left_pivot = row % self.left_rows;
        let site_left = row / self.left_rows;
        let site_right = col % self.site_dim_right;
        let right_pivot = col / self.site_dim_right;
        let mut value = T::zero();
        for middle in 0..self.middle_dim {
            value = value
                + self.left_values[self.left_offset(left_pivot, site_left, middle)?]
                    * self.right_values[self.right_offset(middle, site_right, right_pivot)?];
        }
        Ok(value)
    }

    fn materialize_values(&self) -> Result<Vec<T>> {
        let left =
            Matrix::from_col_major_vec(self.nrows, self.middle_dim, self.left_values.clone());
        let right =
            Matrix::from_col_major_vec(self.middle_dim, self.ncols, self.right_values.clone());
        let product = mat_mul_owned(left, right)
            .map_err(|err| local_factor_error("local input materialization matmul", err))?;
        Ok(product.into_col_major_vec())
    }

    fn left_offset(&self, left_pivot: usize, site_left: usize, middle: usize) -> Result<usize> {
        let site_offset = checked_local_add(
            site_left,
            checked_local_mul(self.site_dim_left, middle, "left factor site offset")?,
            "left factor site offset",
        )?;
        checked_local_add(
            left_pivot,
            checked_local_mul(self.left_rows, site_offset, "left factor offset")?,
            "left factor offset",
        )
    }

    fn right_offset(&self, middle: usize, site_right: usize, right_pivot: usize) -> Result<usize> {
        let site_offset = checked_local_add(
            site_right,
            checked_local_mul(self.site_dim_right, right_pivot, "right factor site offset")?,
            "right factor site offset",
        )?;
        checked_local_add(
            middle,
            checked_local_mul(self.middle_dim, site_offset, "right factor offset")?,
            "right factor offset",
        )
    }
}

impl<'a, T: AciScalar> LocalBlockEvaluator<'a, T> {
    pub(crate) fn new<F>(
        problem: &'a ElementwiseProblem<T>,
        bond: usize,
        operator: &'a F,
    ) -> Result<Self>
    where
        F: for<'batch> Fn(ElementwiseBatch<'batch, T>, &mut [T]) -> Result<()> + 'a,
    {
        if problem.n_inputs() == 0 {
            return Err(AciError::EmptyInputs);
        }

        let (nrows, ncols) = validate_local_shapes(problem, bond)?;
        let input_factors = local_input_factors_for_problem(problem, bond)?;

        Ok(Self {
            problem,
            operator,
            bond,
            nrows,
            ncols,
            input_factors,
            cache: RefCell::new(HashMap::new()),
            first_error: RefCell::new(None),
            max_output_abs: RefCell::new(0.0),
        })
    }

    #[cfg(test)]
    pub(crate) fn new_with_setup_timing<F>(
        problem: &'a ElementwiseProblem<T>,
        bond: usize,
        operator: &'a F,
        timing: &mut LocalInputSetupTiming,
    ) -> Result<Self>
    where
        F: for<'batch> Fn(ElementwiseBatch<'batch, T>, &mut [T]) -> Result<()> + 'a,
    {
        if problem.n_inputs() == 0 {
            return Err(AciError::EmptyInputs);
        }

        let start = Instant::now();
        let (nrows, ncols) = validate_local_shapes(problem, bond)?;
        timing.shape_validation += start.elapsed();

        let input_factors = local_input_factors_for_problem_with_timing(problem, bond, timing)?;

        Ok(Self {
            problem,
            operator,
            bond,
            nrows,
            ncols,
            input_factors,
            cache: RefCell::new(HashMap::new()),
            first_error: RefCell::new(None),
            max_output_abs: RefCell::new(0.0),
        })
    }

    pub(crate) fn nrows(&self) -> usize {
        self.nrows
    }

    pub(crate) fn ncols(&self) -> usize {
        self.ncols
    }

    pub(crate) fn fill_local_block(
        &self,
        rows: &[usize],
        cols: &[usize],
        out: &mut [T],
    ) -> Result<()> {
        let n_points = checked_local_mul(rows.len(), cols.len(), "local block point count")?;
        if out.len() != n_points {
            return Err(AciError::LengthMismatch {
                expected: n_points,
                got: out.len(),
            });
        }
        validate_indices("row", rows, self.nrows)?;
        validate_indices("column", cols, self.ncols)?;
        if n_points == 0 {
            return Ok(());
        }

        let mut point_keys = Vec::with_capacity(n_points);
        let mut missing_by_key = HashMap::new();
        let mut missing_rows = Vec::new();
        let mut missing_cols = Vec::new();

        {
            let cache = self.cache.borrow();
            for (col_position, &col) in cols.iter().enumerate() {
                for (row_position, &row) in rows.iter().enumerate() {
                    let point = row_position + rows.len() * col_position;
                    let key = self.entry_key(row, col)?;
                    point_keys.push(key);
                    if let Some(&value) = cache.get(&key) {
                        out[point] = value;
                    } else if let std::collections::hash_map::Entry::Vacant(entry) =
                        missing_by_key.entry(key)
                    {
                        entry.insert(missing_rows.len());
                        missing_rows.push(row);
                        missing_cols.push(col);
                    }
                }
            }
        }

        if missing_rows.is_empty() {
            return Ok(());
        }

        let n_inputs = self.problem.n_inputs();
        let n_missing = missing_rows.len();
        let input_value_count =
            checked_local_mul(n_inputs, n_missing, "local batch input value count")?;
        let mut input_values = vec![T::zero(); input_value_count];
        for (point, (&row, &col)) in missing_rows.iter().zip(&missing_cols).enumerate() {
            for input in 0..n_inputs {
                input_values[input + n_inputs * point] =
                    self.input_factors[input].value(row, col)?;
            }
        }

        let batch = ElementwiseBatch::new(&input_values, n_inputs, n_missing)?;
        let mut missing_output = vec![T::zero(); n_missing];
        (self.operator)(batch, &mut missing_output)?;
        self.record_output_scale(&missing_output);

        {
            let mut cache = self.cache.borrow_mut();
            for (&key, &missing_index) in &missing_by_key {
                cache.insert(key, missing_output[missing_index]);
            }
        }

        for (point, &key) in point_keys.iter().enumerate() {
            if let Some(&missing_index) = missing_by_key.get(&key) {
                out[point] = missing_output[missing_index];
            }
        }
        Ok(())
    }

    pub(crate) fn materialize_local_matrix(&self) -> Result<Matrix<T>> {
        let input_values = self.materialize_input_values()?;
        self.apply_operator_to_input_values(&input_values)
    }

    pub(crate) fn materialize_input_values(&self) -> Result<Vec<T>> {
        let n_points = checked_local_mul(self.nrows, self.ncols, "local matrix entry count")?;
        let n_inputs = self.problem.n_inputs();
        let input_value_count =
            checked_local_mul(n_inputs, n_points, "local matrix input value count")?;
        let mut input_values = vec![T::zero(); input_value_count];

        if local_materialize_batching_enabled() {
            if let Some(middle_dim) = self.shared_middle_dim() {
                let left_batch_len = checked_local_mul(
                    checked_local_mul(n_inputs, self.nrows, "local left batch input count")?,
                    middle_dim,
                    "local left batch size",
                )?;
                let right_batch_len = checked_local_mul(
                    checked_local_mul(n_inputs, middle_dim, "local right batch input count")?,
                    self.ncols,
                    "local right batch size",
                )?;
                let mut left_batch = Vec::with_capacity(left_batch_len);
                let mut right_batch = Vec::with_capacity(right_batch_len);
                for factors in &self.input_factors {
                    left_batch.extend_from_slice(&factors.left_values);
                    right_batch.extend_from_slice(&factors.right_values);
                }
                let values = batched_mat_mul_same_shape_owned(
                    n_inputs,
                    self.nrows,
                    middle_dim,
                    self.ncols,
                    left_batch,
                    right_batch,
                )
                .map_err(|err| local_factor_error("batched local input materialization", err))?;
                for input in 0..n_inputs {
                    let offset = input * n_points;
                    for point in 0..n_points {
                        input_values[input + n_inputs * point] = values[offset + point];
                    }
                }
                return Ok(input_values);
            }
        }

        for input in 0..n_inputs {
            let values = self.input_factors[input].materialize_values()?;
            for point in 0..n_points {
                input_values[input + n_inputs * point] = values[point];
            }
        }
        Ok(input_values)
    }

    fn shared_middle_dim(&self) -> Option<usize> {
        let first = self.input_factors.first()?;
        if self.input_factors.iter().all(|factors| {
            factors.nrows == self.nrows
                && factors.ncols == self.ncols
                && factors.middle_dim == first.middle_dim
        }) {
            Some(first.middle_dim)
        } else {
            None
        }
    }

    pub(crate) fn apply_operator_to_input_values(&self, input_values: &[T]) -> Result<Matrix<T>> {
        let n_points = checked_local_mul(self.nrows, self.ncols, "local matrix entry count")?;
        let n_inputs = self.problem.n_inputs();
        let batch = ElementwiseBatch::new(input_values, n_inputs, n_points)?;
        let mut output = vec![T::zero(); n_points];
        (self.operator)(batch, &mut output)?;
        self.record_output_scale(&output);
        Ok(Matrix::from_col_major_vec(self.nrows, self.ncols, output))
    }

    pub(crate) fn fill_local_block_or_zero(&self, rows: &[usize], cols: &[usize], out: &mut [T]) {
        if let Err(err) = self.fill_local_block(rows, cols, out) {
            self.record_error(err);
            out.fill(T::zero());
        }
    }

    pub(crate) fn take_error(&self) -> Option<AciError> {
        self.first_error.borrow_mut().take()
    }

    pub(crate) fn clear_cache(&self) {
        self.cache.borrow_mut().clear();
    }

    pub(crate) fn max_output_abs(&self) -> f64 {
        *self.max_output_abs.borrow()
    }

    fn record_error(&self, err: AciError) {
        let mut first_error = self.first_error.borrow_mut();
        if first_error.is_none() {
            *first_error = Some(err);
        }
    }

    fn entry_key(&self, row: usize, col: usize) -> Result<usize> {
        let col_offset = checked_local_mul(self.nrows, col, "local block cache key")?;
        col_offset
            .checked_add(row)
            .ok_or_else(|| AciError::InvalidInitialGuess {
                message: "local block cache key overflows usize".to_string(),
            })
    }

    fn record_output_scale(&self, values: &[T]) {
        let mut max_output_abs = self.max_output_abs.borrow_mut();
        for &value in values {
            *max_output_abs = max_output_abs.max(tensor4all_core::MatrixLuciScalar::abs_val(value));
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct LocalInputFactorDims {
    left_rows: usize,
    input_left_dim: usize,
    site_dim_left: usize,
    middle_dim: usize,
    site_dim_right: usize,
    input_right_dim: usize,
    right_cols: usize,
}

pub(crate) fn local_input_factors_for_problem<T: AciScalar>(
    problem: &ElementwiseProblem<T>,
    bond: usize,
) -> Result<Vec<LocalInputFactors<T>>> {
    let dims = local_input_factor_dims_for_problem(problem, bond)?;
    build_local_input_factors_from_dims(problem, bond, &dims)
}

#[cfg(test)]
pub(crate) fn local_input_factors_for_problem_with_timing<T: AciScalar>(
    problem: &ElementwiseProblem<T>,
    bond: usize,
    timing: &mut LocalInputSetupTiming,
) -> Result<Vec<LocalInputFactors<T>>> {
    let start = Instant::now();
    let dims = local_input_factor_dims_for_problem(problem, bond)?;
    timing.dims += start.elapsed();
    build_local_input_factors_from_dims_with_timing(problem, bond, &dims, timing)
}

fn local_input_factor_dims_for_problem<T: AciScalar>(
    problem: &ElementwiseProblem<T>,
    bond: usize,
) -> Result<Vec<LocalInputFactorDims>> {
    (0..problem.n_inputs())
        .map(|input| local_input_factor_dims(problem, input, bond))
        .collect()
}

fn local_input_factor_dims<T: AciScalar>(
    problem: &ElementwiseProblem<T>,
    input: usize,
    bond: usize,
) -> Result<LocalInputFactorDims> {
    let _ = problem.local_input_shape(input, bond)?;
    let left_frame = local_left_frame(problem, input, bond)?;
    let right_frame = local_right_frame(problem, input, bond)?;
    let left_core = problem.inputs[input].site_tensor(bond);
    let right_core = problem.inputs[input].site_tensor(bond + 1);

    Ok(LocalInputFactorDims {
        left_rows: left_frame.nrows(),
        input_left_dim: left_core.left_dim(),
        site_dim_left: left_core.site_dim(),
        middle_dim: left_core.right_dim(),
        site_dim_right: right_core.site_dim(),
        input_right_dim: right_core.right_dim(),
        right_cols: right_frame.ncols(),
    })
}

#[cfg(test)]
fn build_local_input_factors_from_dims_with_timing<T: AciScalar>(
    problem: &ElementwiseProblem<T>,
    bond: usize,
    dims: &[LocalInputFactorDims],
    timing: &mut LocalInputSetupTiming,
) -> Result<Vec<LocalInputFactors<T>>> {
    let start = Instant::now();
    let left_values = build_left_factors(problem, bond, dims)?;
    timing.left_factor += start.elapsed();

    let start = Instant::now();
    let right_values = build_right_factors(problem, bond, dims)?;
    timing.right_factor += start.elapsed();

    assemble_local_input_factors(dims, left_values, right_values)
}

fn build_local_input_factors_from_dims<T: AciScalar>(
    problem: &ElementwiseProblem<T>,
    bond: usize,
    dims: &[LocalInputFactorDims],
) -> Result<Vec<LocalInputFactors<T>>> {
    let left_values = build_left_factors(problem, bond, dims)?;
    let right_values = build_right_factors(problem, bond, dims)?;
    assemble_local_input_factors(dims, left_values, right_values)
}

fn assemble_local_input_factors<T: AciScalar>(
    dims: &[LocalInputFactorDims],
    left_values: Vec<Vec<T>>,
    right_values: Vec<Vec<T>>,
) -> Result<Vec<LocalInputFactors<T>>> {
    dims.iter()
        .zip(left_values)
        .zip(right_values)
        .map(|((dims, left_values), right_values)| {
            local_input_factors_from_values(*dims, left_values, right_values)
        })
        .collect()
}

fn local_input_factors_from_values<T: AciScalar>(
    dims: LocalInputFactorDims,
    left_values: Vec<T>,
    right_values: Vec<T>,
) -> Result<LocalInputFactors<T>> {
    let nrows = checked_local_mul(dims.left_rows, dims.site_dim_left, "local block row count")?;
    let ncols = checked_local_mul(
        dims.site_dim_right,
        dims.right_cols,
        "local block column count",
    )?;
    Ok(LocalInputFactors {
        nrows,
        ncols,
        left_rows: dims.left_rows,
        site_dim_left: dims.site_dim_left,
        middle_dim: dims.middle_dim,
        site_dim_right: dims.site_dim_right,
        right_cols: dims.right_cols,
        left_values,
        right_values,
    })
}

fn shared_input_factor_dims(dims: &[LocalInputFactorDims]) -> Option<LocalInputFactorDims> {
    let first = *dims.first()?;
    if dims.iter().all(|dims| *dims == first) {
        Some(first)
    } else {
        None
    }
}

fn build_left_factors<T: AciScalar>(
    problem: &ElementwiseProblem<T>,
    bond: usize,
    dims: &[LocalInputFactorDims],
) -> Result<Vec<Vec<T>>> {
    if local_setup_batching_enabled() {
        if let Some(shared) = shared_input_factor_dims(dims) {
            let n_inputs = dims.len();
            let left_cols =
                checked_mul(shared.site_dim_left, shared.middle_dim, "ACI left columns")?;
            let frame_batch_len = checked_mul(
                checked_mul(n_inputs, shared.left_rows, "ACI left frame batch")?,
                shared.input_left_dim,
                "ACI left frame batch",
            )?;
            let core_batch_len = checked_mul(
                checked_mul(n_inputs, shared.input_left_dim, "ACI left core batch")?,
                left_cols,
                "ACI left core batch",
            )?;
            let mut frame_batch = Vec::with_capacity(frame_batch_len);
            let mut core_batch = Vec::with_capacity(core_batch_len);
            for input in 0..n_inputs {
                frame_batch.extend_from_slice(
                    local_left_frame(problem, input, bond)?.as_col_major_slice(),
                );
                core_batch.extend_from_slice(
                    problem
                        .input_core_left_matrix(input, bond)
                        .as_col_major_slice(),
                );
            }
            let values = batched_mat_mul_same_shape_owned(
                n_inputs,
                shared.left_rows,
                shared.input_left_dim,
                left_cols,
                frame_batch,
                core_batch,
            )
            .map_err(|err| local_factor_error("batched left factor matmul", err))?;
            let item_len = checked_mul(shared.left_rows, left_cols, "ACI left factor item")?;
            return Ok((0..n_inputs)
                .map(|input| values[input * item_len..(input + 1) * item_len].to_vec())
                .collect());
        }
    }

    (0..dims.len())
        .map(|input| build_left_factor(problem, input, bond))
        .collect()
}

fn build_right_factors<T: AciScalar>(
    problem: &ElementwiseProblem<T>,
    bond: usize,
    dims: &[LocalInputFactorDims],
) -> Result<Vec<Vec<T>>> {
    if local_setup_batching_enabled() {
        if let Some(shared) = shared_input_factor_dims(dims) {
            let n_inputs = dims.len();
            let right_rows =
                checked_mul(shared.middle_dim, shared.site_dim_right, "ACI right rows")?;
            let core_batch_len = checked_mul(
                checked_mul(n_inputs, right_rows, "ACI right core batch")?,
                shared.input_right_dim,
                "ACI right core batch",
            )?;
            let frame_batch_len = checked_mul(
                checked_mul(n_inputs, shared.input_right_dim, "ACI right frame batch")?,
                shared.right_cols,
                "ACI right frame batch",
            )?;
            let mut core_batch = Vec::with_capacity(core_batch_len);
            let mut frame_batch = Vec::with_capacity(frame_batch_len);
            for input in 0..n_inputs {
                core_batch.extend_from_slice(
                    problem
                        .input_core_right_matrix(input, bond + 1)
                        .as_col_major_slice(),
                );
                frame_batch.extend_from_slice(
                    local_right_frame(problem, input, bond)?.as_col_major_slice(),
                );
            }
            let values = batched_mat_mul_same_shape_owned(
                n_inputs,
                right_rows,
                shared.input_right_dim,
                shared.right_cols,
                core_batch,
                frame_batch,
            )
            .map_err(|err| local_factor_error("batched right factor matmul", err))?;
            let item_len = checked_mul(right_rows, shared.right_cols, "ACI right factor item")?;
            return Ok((0..n_inputs)
                .map(|input| values[input * item_len..(input + 1) * item_len].to_vec())
                .collect());
        }
    }

    (0..dims.len())
        .map(|input| build_right_factor(problem, input, bond))
        .collect()
}

fn build_left_factor<T: AciScalar>(
    problem: &ElementwiseProblem<T>,
    input: usize,
    bond: usize,
) -> Result<Vec<T>> {
    let frame = local_left_frame(problem, input, bond)?;
    let core_matrix = problem.input_core_left_matrix(input, bond);
    let product =
        mat_mul(frame, core_matrix).map_err(|err| local_factor_error("left factor matmul", err))?;
    Ok(product.as_col_major_slice().to_vec())
}

fn build_right_factor<T: AciScalar>(
    problem: &ElementwiseProblem<T>,
    input: usize,
    bond: usize,
) -> Result<Vec<T>> {
    let frame = local_right_frame(problem, input, bond)?;
    let core_matrix = problem.input_core_right_matrix(input, bond + 1);
    let product = mat_mul(core_matrix, frame)
        .map_err(|err| local_factor_error("right factor matmul", err))?;
    Ok(product.as_col_major_slice().to_vec())
}

fn validate_local_shapes<T: AciScalar>(
    problem: &ElementwiseProblem<T>,
    bond: usize,
) -> Result<(usize, usize)> {
    let (nrows, ncols) = problem.local_input_shape(0, bond)?;
    for input in 1..problem.n_inputs() {
        let (input_nrows, input_ncols) = problem.local_input_shape(input, bond)?;
        if input_nrows != nrows || input_ncols != ncols {
            return Err(AciError::InvalidInitialGuess {
                message: format!(
                    "local block shape mismatch at input {input}, bond {bond}: \
                     expected ({nrows}, {ncols}), got ({input_nrows}, {input_ncols})"
                ),
            });
        }
    }
    Ok((nrows, ncols))
}

fn local_left_frame<T: AciScalar>(
    problem: &ElementwiseProblem<T>,
    input: usize,
    bond: usize,
) -> Result<&tensor4all_tensorbackend::Matrix<T>> {
    problem
        .left_frames
        .get(input)
        .and_then(|frames| frames.get(bond))
        .and_then(|frame| frame.as_ref())
        .ok_or_else(|| AciError::InvalidInitialGuess {
            message: format!("missing left frame at input {input}, site {bond}"),
        })
}

fn local_right_frame<T: AciScalar>(
    problem: &ElementwiseProblem<T>,
    input: usize,
    bond: usize,
) -> Result<&tensor4all_tensorbackend::Matrix<T>> {
    let site = bond + 2;
    problem
        .right_frames
        .get(input)
        .and_then(|frames| frames.get(site))
        .and_then(|frame| frame.as_ref())
        .ok_or_else(|| AciError::InvalidInitialGuess {
            message: format!("missing right frame at input {input}, site {site}"),
        })
}

fn local_factor_error(context: &str, err: impl std::fmt::Display) -> AciError {
    AciError::InvalidInitialGuess {
        message: format!("{context} failed: {err}"),
    }
}

fn validate_indices(kind: &'static str, indices: &[usize], len: usize) -> Result<()> {
    for &index in indices {
        if index >= len {
            return Err(AciError::InvalidInitialGuess {
                message: format!("local {kind} index {index} out of bounds for length {len}"),
            });
        }
    }
    Ok(())
}

fn checked_local_mul(lhs: usize, rhs: usize, description: &str) -> Result<usize> {
    lhs.checked_mul(rhs)
        .ok_or_else(|| AciError::InvalidInitialGuess {
            message: format!("{description} overflows usize"),
        })
}

fn checked_local_add(lhs: usize, rhs: usize, description: &str) -> Result<usize> {
    lhs.checked_add(rhs)
        .ok_or_else(|| AciError::InvalidInitialGuess {
            message: format!("{description} overflows usize"),
        })
}
