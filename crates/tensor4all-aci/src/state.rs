use std::cell::RefCell;

use crate::scalar::AciScalar;
use crate::{initial_guess, AciError, AciOptions, ElementwiseBatch, LocalBlockEvaluator, Result};
use tensor4all_simplett::{
    tensor3_from_data, AbstractTensorTrain, SimpleTensorTrain, Tensor3, Tensor3Ops,
};
use tensor4all_tcicore::{
    matrix_luci_factors_from_matrix, matrix_luci_factors_from_matrix_owned, RrLUOptions,
};
use tensor4all_tensorbackend::{batched_mat_mul_same_shape_owned, mat_mul, mat_mul_owned, Matrix};

#[cfg(test)]
fn frame_batching_enabled() -> bool {
    std::env::var("T4A_ACI_DISABLE_BATCHED_FRAME").as_deref() != Ok("1")
}

#[cfg(not(test))]
fn frame_batching_enabled() -> bool {
    true
}

pub(crate) struct ElementwiseProblem<T: AciScalar> {
    pub(crate) inputs: Vec<SimpleTensorTrain<T>>,
    pub(crate) solution: SimpleTensorTrain<T>,
    input_core_matrices: Vec<Vec<InputCoreMatrices<T>>>,
    pub(crate) left_frames: Vec<Vec<Option<Matrix<T>>>>,
    pub(crate) right_frames: Vec<Vec<Option<Matrix<T>>>>,
    pub(crate) pivot_errors: Vec<f64>,
    pub(crate) pivot_scales: Vec<f64>,
}

#[derive(Clone)]
struct InputCoreMatrices<T> {
    left_grouped: Matrix<T>,
    right_grouped: Matrix<T>,
}

impl<T: AciScalar> InputCoreMatrices<T> {
    fn from_core(core: &Tensor3<T>) -> Self {
        let data = core.to_col_major_vec();
        Self {
            left_grouped: Matrix::from_col_major_vec(
                core.left_dim(),
                core.site_dim() * core.right_dim(),
                data.clone(),
            ),
            right_grouped: Matrix::from_col_major_vec(
                core.left_dim() * core.site_dim(),
                core.right_dim(),
                data,
            ),
        }
    }
}

impl<T: AciScalar> ElementwiseProblem<T> {
    pub(crate) fn new(inputs: Vec<SimpleTensorTrain<T>>, options: AciOptions<T>) -> Result<Self> {
        let solution = initial_guess(&inputs, &options)?;
        let n = solution.len();
        let n_inputs = inputs.len();
        let input_core_matrices = inputs
            .iter()
            .map(|input| {
                input
                    .site_tensors()
                    .iter()
                    .map(InputCoreMatrices::from_core)
                    .collect()
            })
            .collect();
        let mut left_frames = vec![vec![None; n + 1]; n_inputs];
        let mut right_frames = vec![vec![None; n + 1]; n_inputs];

        for input in 0..n_inputs {
            left_frames[input][0] = Some(unit_frame());
            right_frames[input][n] = Some(unit_frame());
        }

        let mut problem = Self {
            inputs,
            solution,
            input_core_matrices,
            left_frames,
            right_frames,
            pivot_errors: vec![0.0; n.saturating_sub(1)],
            pivot_scales: vec![0.0; n.saturating_sub(1)],
        };
        problem.initialize_right_frames()?;
        Ok(problem)
    }

    pub(crate) fn len(&self) -> usize {
        self.solution.len()
    }

    pub(crate) fn n_inputs(&self) -> usize {
        self.inputs.len()
    }

    pub(crate) fn input_core_left_matrix(&self, input: usize, site: usize) -> &Matrix<T> {
        &self.input_core_matrices[input][site].left_grouped
    }

    pub(crate) fn input_core_right_matrix(&self, input: usize, site: usize) -> &Matrix<T> {
        &self.input_core_matrices[input][site].right_grouped
    }

    pub(crate) fn left_frame_shape(&self, input: usize, site: usize) -> Option<(usize, usize)> {
        self.left_frames
            .get(input)
            .and_then(|frames| frames.get(site))
            .and_then(|frame| frame.as_ref())
            .map(|frame| (frame.nrows(), frame.ncols()))
    }

    pub(crate) fn right_frame_shape(&self, input: usize, site: usize) -> Option<(usize, usize)> {
        self.right_frames
            .get(input)
            .and_then(|frames| frames.get(site))
            .and_then(|frame| frame.as_ref())
            .map(|frame| (frame.nrows(), frame.ncols()))
    }

    pub(crate) fn left_frame_value(
        &self,
        input: usize,
        site: usize,
        row: usize,
        col: usize,
    ) -> Option<T> {
        frame_value(&self.left_frames, input, site, row, col)
    }

    pub(crate) fn right_frame_value(
        &self,
        input: usize,
        site: usize,
        row: usize,
        col: usize,
    ) -> Option<T> {
        frame_value(&self.right_frames, input, site, row, col)
    }

    pub(crate) fn local_input_shape(&self, input: usize, bond: usize) -> Result<(usize, usize)> {
        let context = self.local_input_context(input, bond)?;
        Ok((context.nrows, context.ncols))
    }

    pub(crate) fn local_input_value(
        &self,
        input: usize,
        bond: usize,
        row: usize,
        col: usize,
    ) -> Result<T> {
        let context = self.local_input_context(input, bond)?;
        if row >= context.nrows {
            return Err(AciError::InvalidInitialGuess {
                message: format!(
                    "local row index {row} out of bounds for bond {bond} with {} rows",
                    context.nrows
                ),
            });
        }
        if col >= context.ncols {
            return Err(AciError::InvalidInitialGuess {
                message: format!(
                    "local column index {col} out of bounds for bond {bond} with {} columns",
                    context.ncols
                ),
            });
        }

        let r_left = context.left_frame.nrows();
        let d_right = context.right_core.site_dim();
        let left_pivot = row % r_left;
        let site_left = row / r_left;
        let site_right = col % d_right;
        let right_pivot = col / d_right;

        let mut sum = T::zero();
        for a in 0..context.left_core.left_dim() {
            for m in 0..context.left_core.right_dim() {
                for b in 0..context.right_core.right_dim() {
                    sum = sum
                        + context.left_frame[[left_pivot, a]]
                            * *context.left_core.get3(a, site_left, m)
                            * *context.right_core.get3(m, site_right, b)
                            * context.right_frame[[b, right_pivot]];
                }
            }
        }
        Ok(sum)
    }

    pub(crate) fn update_left_frame(
        &mut self,
        input: usize,
        site: usize,
        row_indices: &[usize],
    ) -> Result<()> {
        let n = self.len();
        validate_input_site(input, site, self.n_inputs(), n)?;
        let source = self.left_frames[input][site]
            .as_ref()
            .ok_or_else(|| missing_frame("left", input, site))?
            .clone();
        let core = self.inputs[input].site_tensor(site);

        if source.ncols() != core.left_dim() {
            return Err(AciError::InvalidInitialGuess {
                message: format!(
                    "left frame/input bond mismatch at input {input}, site {site}: \
                     frame has {} columns, core left bond is {}",
                    source.ncols(),
                    core.left_dim()
                ),
            });
        }

        let full_rows = checked_frame_mul(source.nrows(), core.site_dim(), "left frame row count")?;
        validate_selection("row", row_indices, full_rows)?;

        let core_matrix = self.input_core_left_matrix(input, site);
        let full_frame = frame_matmul_checked(&source, core_matrix, "left", input, site)?;
        let full_data = full_frame.as_col_major_slice();
        let mut selected = Matrix::zeros(row_indices.len(), core.right_dim());
        for (selected_row, &full_row) in row_indices.iter().enumerate() {
            for right in 0..core.right_dim() {
                selected[[selected_row, right]] = full_data[full_row + full_rows * right];
            }
        }

        self.left_frames[input][site + 1] = Some(selected);
        Ok(())
    }

    pub(crate) fn update_right_frame(
        &mut self,
        input: usize,
        site: usize,
        col_indices: &[usize],
    ) -> Result<()> {
        let n = self.len();
        validate_input_site(input, site, self.n_inputs(), n)?;
        let source_site = site + 1;
        let source = self.right_frames[input][source_site]
            .as_ref()
            .ok_or_else(|| missing_frame("right", input, source_site))?
            .clone();
        let core = self.inputs[input].site_tensor(site);

        if core.right_dim() != source.nrows() {
            return Err(AciError::InvalidInitialGuess {
                message: format!(
                    "right frame/input bond mismatch at input {input}, site {site}: \
                     core right bond is {}, frame has {} rows",
                    core.right_dim(),
                    source.nrows()
                ),
            });
        }

        let full_cols =
            checked_frame_mul(core.site_dim(), source.ncols(), "right frame column count")?;
        validate_selection("column", col_indices, full_cols)?;

        let core_matrix = self.input_core_right_matrix(input, site);
        let full_frame = frame_matmul_checked(core_matrix, &source, "right", input, site)?;
        let full_data = full_frame.as_col_major_slice();
        let mut selected = Matrix::zeros(core.left_dim(), col_indices.len());
        for (selected_col, &full_col) in col_indices.iter().enumerate() {
            for left in 0..core.left_dim() {
                selected[[left, selected_col]] = full_data[left + core.left_dim() * full_col];
            }
        }

        self.right_frames[input][site] = Some(selected);
        Ok(())
    }

    pub(crate) fn update_left_frames(&mut self, site: usize, row_indices: &[usize]) -> Result<()> {
        if frame_batching_enabled() {
            if let Some(frames) = self.batched_left_frame_updates(site, row_indices)? {
                for (input, frame) in frames.into_iter().enumerate() {
                    self.left_frames[input][site + 1] = Some(frame);
                }
                return Ok(());
            }
        }

        for input in 0..self.n_inputs() {
            self.update_left_frame(input, site, row_indices)?;
        }
        Ok(())
    }

    pub(crate) fn update_right_frames(&mut self, site: usize, col_indices: &[usize]) -> Result<()> {
        if frame_batching_enabled() {
            if let Some(frames) = self.batched_right_frame_updates(site, col_indices)? {
                for (input, frame) in frames.into_iter().enumerate() {
                    self.right_frames[input][site] = Some(frame);
                }
                return Ok(());
            }
        }

        for input in 0..self.n_inputs() {
            self.update_right_frame(input, site, col_indices)?;
        }
        Ok(())
    }

    fn batched_left_frame_updates(
        &self,
        site: usize,
        row_indices: &[usize],
    ) -> Result<Option<Vec<Matrix<T>>>> {
        let n = self.len();
        let n_inputs = self.n_inputs();
        if n_inputs <= 1 {
            return Ok(None);
        }

        let mut shared: Option<(usize, usize, usize, usize, usize)> = None;
        let mut frame_batch = Vec::new();
        let mut core_batch = Vec::new();

        for input in 0..n_inputs {
            validate_input_site(input, site, n_inputs, n)?;
            let source = self.left_frames[input][site]
                .as_ref()
                .ok_or_else(|| missing_frame("left", input, site))?;
            let core = self.inputs[input].site_tensor(site);

            if source.ncols() != core.left_dim() {
                return Err(AciError::InvalidInitialGuess {
                    message: format!(
                        "left frame/input bond mismatch at input {input}, site {site}: \
                         frame has {} columns, core left bond is {}",
                        source.ncols(),
                        core.left_dim()
                    ),
                });
            }

            let full_rows =
                checked_frame_mul(source.nrows(), core.site_dim(), "left frame row count")?;
            validate_selection("row", row_indices, full_rows)?;

            let dims = (
                source.nrows(),
                source.ncols(),
                core.site_dim(),
                core.right_dim(),
                full_rows,
            );
            if let Some(shared) = shared {
                if shared != dims {
                    return Ok(None);
                }
            } else {
                shared = Some(dims);
            }

            frame_batch.extend_from_slice(source.as_col_major_slice());
            core_batch.extend_from_slice(
                self.input_core_left_matrix(input, site)
                    .as_col_major_slice(),
            );
        }

        let Some((source_rows, source_cols, site_dim, right_dim, full_rows)) = shared else {
            return Ok(None);
        };
        let output_cols = site_dim * right_dim;
        let values = batched_mat_mul_same_shape_owned(
            n_inputs,
            source_rows,
            source_cols,
            output_cols,
            frame_batch,
            core_batch,
        )
        .map_err(|err| frame_matmul_error("batched left", err))?;
        let item_len = source_rows * output_cols;

        let frames = (0..n_inputs)
            .map(|input| {
                let offset = input * item_len;
                let full_data = &values[offset..offset + item_len];
                let mut selected = Matrix::zeros(row_indices.len(), right_dim);
                for (selected_row, &full_row) in row_indices.iter().enumerate() {
                    for right in 0..right_dim {
                        selected[[selected_row, right]] = full_data[full_row + full_rows * right];
                    }
                }
                selected
            })
            .collect();
        Ok(Some(frames))
    }

    fn batched_right_frame_updates(
        &self,
        site: usize,
        col_indices: &[usize],
    ) -> Result<Option<Vec<Matrix<T>>>> {
        let n = self.len();
        let n_inputs = self.n_inputs();
        if n_inputs <= 1 {
            return Ok(None);
        }

        let mut shared: Option<(usize, usize, usize, usize, usize)> = None;
        let mut core_batch = Vec::new();
        let mut frame_batch = Vec::new();
        let source_site = site + 1;

        for input in 0..n_inputs {
            validate_input_site(input, site, n_inputs, n)?;
            let source = self.right_frames[input][source_site]
                .as_ref()
                .ok_or_else(|| missing_frame("right", input, source_site))?;
            let core = self.inputs[input].site_tensor(site);

            if core.right_dim() != source.nrows() {
                return Err(AciError::InvalidInitialGuess {
                    message: format!(
                        "right frame/input bond mismatch at input {input}, site {site}: \
                         core right bond is {}, frame has {} rows",
                        core.right_dim(),
                        source.nrows()
                    ),
                });
            }

            let full_cols =
                checked_frame_mul(core.site_dim(), source.ncols(), "right frame column count")?;
            validate_selection("column", col_indices, full_cols)?;

            let dims = (
                core.left_dim(),
                core.site_dim(),
                core.right_dim(),
                source.ncols(),
                full_cols,
            );
            if let Some(shared) = shared {
                if shared != dims {
                    return Ok(None);
                }
            } else {
                shared = Some(dims);
            }

            core_batch.extend_from_slice(
                self.input_core_right_matrix(input, site)
                    .as_col_major_slice(),
            );
            frame_batch.extend_from_slice(source.as_col_major_slice());
        }

        let Some((left_dim, site_dim, right_dim, source_cols, _full_cols)) = shared else {
            return Ok(None);
        };
        let core_rows = left_dim * site_dim;
        let values = batched_mat_mul_same_shape_owned(
            n_inputs,
            core_rows,
            right_dim,
            source_cols,
            core_batch,
            frame_batch,
        )
        .map_err(|err| frame_matmul_error("batched right", err))?;
        let item_len = core_rows * source_cols;

        let frames = (0..n_inputs)
            .map(|input| {
                let offset = input * item_len;
                let full_data = &values[offset..offset + item_len];
                let mut selected = Matrix::zeros(left_dim, col_indices.len());
                for (selected_col, &full_col) in col_indices.iter().enumerate() {
                    for left in 0..left_dim {
                        selected[[left, selected_col]] = full_data[left + left_dim * full_col];
                    }
                }
                selected
            })
            .collect();
        Ok(Some(frames))
    }

    /// Inject global pivots into the frame index sets and the solution rank.
    ///
    /// Internal bond `k` (between sites `k - 1` and `k`) is sampled through two
    /// frames of equal size: `left_frames[j][k]`, whose rows are left
    /// environments `L_j(x[..k])`, and `right_frames[j][k]`, whose columns are
    /// right environments `R_j(x[k..])`. A point `x` becomes visible to the
    /// next sweep's local block at bond `b` once `left_frames[j][b]` holds the
    /// row for `x[..b]` and `right_frames[j][b + 2]` the column for
    /// `x[b + 2..]`, so a pivot is injected by growing every internal bond by
    /// one row and one column. Because ACI couples the frame sizes to the
    /// solution rank (every `local_update` cross-checks them), the solution's
    /// internal bond dimensions are zero-padded by the same per-bond amount.
    ///
    /// Growth is skipped for a bond that cannot absorb it, so no frame is ever
    /// padded with a provably linearly dependent row or column:
    ///
    /// * **Duplicate.** Both the row and the column of this pivot are already
    ///   present in every input's frames at that bond.
    /// * **Index-space saturation.** The bond dimension already equals the
    ///   number of distinct multi-indices on the shorter side of the cut
    ///   (`prod(site_dims[..k])` or `prod(site_dims[k..])`). One of the two
    ///   frames then already enumerates its whole index space, so an appended
    ///   row (or column) can only repeat one that is there, and the local
    ///   block at that bond already interpolates exactly.
    ///
    /// Without these guards a run can carry bond dimensions above the algebraic
    /// maximum of the site space (issue #618 observed a transient rank of 1044
    /// on an eleven-site dimension-four train, whose bound is 1024), which
    /// costs full-rank sweep work for pivots that carry no information.
    ///
    /// Returns the number of pivots that grew at least one bond (0 when every
    /// found pivot was already represented in the frames, or when no bond could
    /// absorb it).
    pub(crate) fn add_global_pivots(&mut self, pivots: &[Vec<usize>]) -> Result<usize>
    where
        T: PartialEq,
    {
        let n = self.len();
        let mut new_pivots = 0usize;
        let mut growth = vec![0usize; n + 1];
        let bounds = self.algebraic_bond_bounds();
        let mut dims = vec![0usize; n + 1];
        for (bond, dim) in self.solution.link_dims().iter().enumerate() {
            dims[bond + 1] = *dim;
        }

        for pivot in pivots {
            if pivot.len() != n {
                return Err(AciError::InvalidInitialGuess {
                    message: format!(
                        "global pivot length {} must match the number of sites {n}",
                        pivot.len()
                    ),
                });
            }

            let mut injected = false;
            for bond in 1..n {
                if dims[bond] >= bounds[bond] {
                    continue;
                }
                // The row lives in `left_frames[j][bond]`, which exists for
                // bonds up to `n - 2`; the column lives in
                // `right_frames[j][bond]`, which exists from bond 2 on.
                let needs_row = bond + 1 < n;
                let needs_col = bond >= 2;
                let mut duplicate = true;
                for input in 0..self.n_inputs() {
                    if needs_row {
                        let row = left_environment(&self.inputs[input], &pivot[..bond])?;
                        if !frame_has_row(self.left_frames[input][bond].as_ref(), &row)? {
                            duplicate = false;
                        }
                    }
                    if needs_col {
                        let col = right_environment(&self.inputs[input], &pivot[bond..])?;
                        if !frame_has_col(self.right_frames[input][bond].as_ref(), &col)? {
                            duplicate = false;
                        }
                    }
                }
                if duplicate {
                    continue;
                }

                for input in 0..self.n_inputs() {
                    if needs_row {
                        let row = left_environment(&self.inputs[input], &pivot[..bond])?;
                        append_row(
                            self.left_frames[input][bond].as_mut().ok_or_else(|| {
                                AciError::InvalidInitialGuess {
                                    message: format!(
                                        "missing left frame at input {input}, bond {bond}"
                                    ),
                                }
                            })?,
                            &row,
                        )?;
                    }
                    if needs_col {
                        let col = right_environment(&self.inputs[input], &pivot[bond..])?;
                        append_col(
                            self.right_frames[input][bond].as_mut().ok_or_else(|| {
                                AciError::InvalidInitialGuess {
                                    message: format!(
                                        "missing right frame at input {input}, bond {bond}"
                                    ),
                                }
                            })?,
                            &col,
                        )?;
                    }
                }
                growth[bond] += 1;
                dims[bond] += 1;
                injected = true;
            }

            if injected {
                new_pivots += 1;
            }
        }
        if new_pivots > 0 {
            self.pad_solution_internal_bonds(&growth)?;
        }
        Ok(new_pivots)
    }

    /// Number of distinct multi-indices on the shorter side of each bond cut.
    ///
    /// Entry `k` refers to the bond between sites `k - 1` and `k` and is the
    /// largest number of linearly independent rows (or columns) its frames can
    /// hold; entries `0` and `n` are the trivial boundary bonds and are `1`.
    /// Products are computed with saturating arithmetic, so a site space too
    /// large to count simply never reports saturation.
    fn algebraic_bond_bounds(&self) -> Vec<usize> {
        let n = self.len();
        let dims: Vec<usize> = (0..n).map(|site| self.solution.site_dim(site)).collect();
        let mut left_space = vec![1usize; n + 1];
        for site in 0..n {
            left_space[site + 1] = left_space[site].saturating_mul(dims[site]);
        }
        let mut right_space = vec![1usize; n + 1];
        for site in (0..n).rev() {
            right_space[site] = right_space[site + 1].saturating_mul(dims[site]);
        }

        (0..=n)
            .map(|bond| left_space[bond].min(right_space[bond]))
            .collect()
    }

    /// Grow internal bond `k` of the solution by `growth[k]` with zero padding.
    ///
    /// ACI couples the solution rank to the frame sizes, so injecting pivots
    /// into the frames requires growing the solution's internal bond
    /// dimensions by the same per-bond amount to keep `local_update`'s shape
    /// checks consistent. The zero-padded entries preserve the current solution
    /// values; the following sweep's LU fills them from the new crosses.
    /// `growth` has `len() + 1` entries; the boundary entries `0` and `n` are
    /// ignored because those bonds are trivial.
    fn pad_solution_internal_bonds(&mut self, growth: &[usize]) -> Result<()> {
        let n = self.len();
        if growth.iter().all(|&g| g == 0) {
            return Ok(());
        }
        if growth.len() != n + 1 {
            return Err(AciError::InvalidInitialGuess {
                message: format!(
                    "bond growth has {} entries, expected {} for {n} sites",
                    growth.len(),
                    n + 1
                ),
            });
        }
        let cores = self.solution.site_tensors().to_vec();
        let mut new_cores = Vec::with_capacity(n);
        for (site, core) in cores.iter().enumerate() {
            let site_dim = core.site_dim();
            let left_dim = core.left_dim();
            let right_dim = core.right_dim();
            let new_left = if site == 0 {
                left_dim
            } else {
                left_dim + growth[site]
            };
            let new_right = if site == n - 1 {
                right_dim
            } else {
                right_dim + growth[site + 1]
            };
            let mut data = vec![T::zero(); new_left * site_dim * new_right];
            for r in 0..right_dim {
                for s in 0..site_dim {
                    for l in 0..left_dim {
                        data[l + new_left * (s + site_dim * r)] = *core.get3(l, s, r);
                    }
                }
            }
            new_cores.push(tensor3_from_data(data, new_left, site_dim, new_right)?);
        }
        self.solution = SimpleTensorTrain::new(new_cores)?;
        Ok(())
    }

    pub(crate) fn local_update<F>(
        &mut self,
        bond: usize,
        left_orthogonal: bool,
        options: &AciOptions<T>,
        op: &mut F,
    ) -> Result<()>
    where
        F: for<'batch> FnMut(ElementwiseBatch<'batch, T>, &mut [T]) -> Result<()>,
    {
        let n = self.len();
        if bond >= n.saturating_sub(1) {
            return Err(AciError::InvalidInitialGuess {
                message: format!("bond index {bond} out of bounds for tensor train length {n}"),
            });
        }

        let left_core = self.solution.site_tensor(bond);
        let right_core = self.solution.site_tensor(bond + 1);
        if left_core.right_dim() != right_core.left_dim() {
            return Err(AciError::InvalidInitialGuess {
                message: format!(
                    "adjacent solution core bond mismatch at bond {bond}: \
                     left core right bond is {}, right core left bond is {}",
                    left_core.right_dim(),
                    right_core.left_dim()
                ),
            });
        }

        let left_solution_rank = left_core.left_dim();
        let site_dim_left = left_core.site_dim();
        let site_dim_right = right_core.site_dim();
        let right_solution_rank = right_core.right_dim();
        let nrows = checked_frame_mul(
            left_solution_rank,
            site_dim_left,
            "solution local block row count",
        )?;
        let ncols = checked_frame_mul(
            site_dim_right,
            right_solution_rank,
            "solution local block column count",
        )?;

        let (factors, sampled_scale) = {
            let op_cell = RefCell::new(op);
            let operator = |batch: ElementwiseBatch<'_, T>, output: &mut [T]| {
                let mut op_ref = op_cell.borrow_mut();
                (*op_ref)(batch, output)
            };
            let evaluator = LocalBlockEvaluator::new(self, bond, &operator)?;
            if evaluator.nrows() != nrows || evaluator.ncols() != ncols {
                return Err(AciError::InvalidInitialGuess {
                    message: format!(
                        "local block shape mismatch at bond {bond}: solution implies \
                         ({nrows}, {ncols}), evaluator has ({}, {})",
                        evaluator.nrows(),
                        evaluator.ncols()
                    ),
                });
            }

            let local_matrix = evaluator.materialize_local_matrix()?;
            let factors_result = matrix_luci_factors_from_matrix_owned(
                local_matrix,
                Some(RrLUOptions {
                    max_bond_dim: options.max_bond_dim.unwrap_or(usize::MAX),
                    rel_tol: if options.scale_tolerance {
                        options.tolerance
                    } else {
                        0.0
                    },
                    abs_tol: if options.scale_tolerance {
                        0.0
                    } else {
                        options.tolerance
                    },
                    left_orthogonal,
                }),
            );
            (factors_result?, evaluator.max_output_abs())
        };

        let pivot_error = match factors.pivot_errors.last() {
            Some(&error) => error,
            None => 0.0,
        };
        let (new_rank, left_factor, right_factor, row_indices, col_indices) = if factors.rank == 0 {
            if nrows == 0 || ncols == 0 {
                return Err(AciError::InvalidInitialGuess {
                    message: format!(
                        "zero-rank local update at bond {bond} requires positive local block \
                             shape, got ({nrows}, {ncols})"
                    ),
                });
            }
            (
                1,
                Matrix::zeros(nrows, 1),
                Matrix::zeros(1, ncols),
                vec![0],
                vec![0],
            )
        } else {
            (
                factors.rank,
                factors.left,
                factors.right,
                factors.row_indices,
                factors.col_indices,
            )
        };

        let new_left_core =
            matrix_into_tensor3(left_factor, left_solution_rank, site_dim_left, new_rank)?;
        let new_right_core =
            right_factor_into_tensor3(right_factor, new_rank, site_dim_right, right_solution_rank)?;
        let solution_cores = self.solution.site_tensors_mut();
        solution_cores[bond] = new_left_core;
        solution_cores[bond + 1] = new_right_core;

        if left_orthogonal {
            self.update_left_frames(bond, &row_indices)?;
        } else {
            self.update_right_frames(bond + 1, &col_indices)?;
        }
        self.pivot_errors[bond] = pivot_error;
        self.pivot_scales[bond] = sampled_scale;

        Ok(())
    }

    fn initialize_right_frames(&mut self) -> Result<()> {
        let n = self.solution.len();
        let mut solution_cores = self.solution.site_tensors().to_vec();

        for site in (1..n).rev() {
            let current = &solution_cores[site];
            let site_dim = current.site_dim();
            let right_dim = current.right_dim();
            let matrix = right_matrix_julia_order(current);
            let factors = matrix_luci_factors_from_matrix(
                &matrix,
                Some(RrLUOptions {
                    left_orthogonal: false,
                    rel_tol: 0.0,
                    abs_tol: 0.0,
                    max_bond_dim: usize::MAX,
                }),
            )?;

            let ncols =
                checked_frame_mul(site_dim, right_dim, "solution right matrix column count")?;
            let (new_rank, left_factor, right_factor, col_indices) = if factors.rank == 0 {
                if matrix.nrows() == 0 || ncols == 0 {
                    return Err(AciError::InvalidInitialGuess {
                        message: format!(
                            "zero-rank right-frame initialization at site {site} requires positive \
                             matrix shape, got ({}, {ncols})",
                            matrix.nrows()
                        ),
                    });
                }
                (
                    1,
                    Matrix::zeros(matrix.nrows(), 1),
                    Matrix::zeros(1, ncols),
                    vec![0],
                )
            } else {
                (
                    factors.rank,
                    factors.left,
                    factors.right,
                    factors.col_indices,
                )
            };

            validate_selection("column", &col_indices, ncols)?;
            solution_cores[site] =
                right_factor_into_tensor3(right_factor, new_rank, site_dim, right_dim)?;

            let previous = &solution_cores[site - 1];
            let previous_left_dim = previous.left_dim();
            let previous_site_dim = previous.site_dim();
            let previous_matrix = left_matrix_julia_order(previous);
            let product = matmul_checked_owned(previous_matrix, left_factor, site - 1)?;
            solution_cores[site - 1] =
                matrix_into_tensor3(product, previous_left_dim, previous_site_dim, new_rank)?;

            self.update_right_frames(site, &col_indices)?;
        }

        self.solution = SimpleTensorTrain::new(solution_cores)?;
        Ok(())
    }

    fn local_input_context(&self, input: usize, bond: usize) -> Result<LocalInputContext<'_, T>> {
        let n = self.len();
        if input >= self.n_inputs() {
            return Err(AciError::InvalidInitialGuess {
                message: format!(
                    "input index {input} out of bounds for {} inputs",
                    self.n_inputs()
                ),
            });
        }
        if bond >= n.saturating_sub(1) {
            return Err(AciError::InvalidInitialGuess {
                message: format!("bond index {bond} out of bounds for tensor train length {n}"),
            });
        }

        let right_frame_site = bond + 2;
        let left_frame = self
            .left_frames
            .get(input)
            .and_then(|frames| frames.get(bond))
            .and_then(|frame| frame.as_ref())
            .ok_or_else(|| missing_frame("left", input, bond))?;
        let right_frame = self
            .right_frames
            .get(input)
            .and_then(|frames| frames.get(right_frame_site))
            .and_then(|frame| frame.as_ref())
            .ok_or_else(|| missing_frame("right", input, right_frame_site))?;
        let left_core = self.inputs[input].site_tensor(bond);
        let right_core = self.inputs[input].site_tensor(bond + 1);

        if left_frame.nrows() == 0
            || left_core.site_dim() == 0
            || right_core.site_dim() == 0
            || right_frame.ncols() == 0
        {
            return Err(AciError::InvalidInitialGuess {
                message: format!("local block at input {input}, bond {bond} has a zero dimension"),
            });
        }
        if left_frame.ncols() != left_core.left_dim() {
            return Err(AciError::InvalidInitialGuess {
                message: format!(
                    "left frame/input bond mismatch at input {input}, bond {bond}: \
                     frame has {} columns, left core left bond is {}",
                    left_frame.ncols(),
                    left_core.left_dim()
                ),
            });
        }
        if left_core.right_dim() != right_core.left_dim() {
            return Err(AciError::InvalidInitialGuess {
                message: format!(
                    "adjacent input core bond mismatch at input {input}, bond {bond}: \
                     left core right bond is {}, right core left bond is {}",
                    left_core.right_dim(),
                    right_core.left_dim()
                ),
            });
        }
        if right_core.right_dim() != right_frame.nrows() {
            return Err(AciError::InvalidInitialGuess {
                message: format!(
                    "right frame/input bond mismatch at input {input}, bond {bond}: \
                     right core right bond is {}, frame has {} rows",
                    right_core.right_dim(),
                    right_frame.nrows()
                ),
            });
        }

        let nrows = checked_frame_mul(
            left_frame.nrows(),
            left_core.site_dim(),
            "local block row count",
        )?;
        let ncols = checked_frame_mul(
            right_core.site_dim(),
            right_frame.ncols(),
            "local block column count",
        )?;

        Ok(LocalInputContext {
            left_frame,
            right_frame,
            left_core,
            right_core,
            nrows,
            ncols,
        })
    }
}

struct LocalInputContext<'a, T: AciScalar> {
    left_frame: &'a Matrix<T>,
    right_frame: &'a Matrix<T>,
    left_core: &'a Tensor3<T>,
    right_core: &'a Tensor3<T>,
    nrows: usize,
    ncols: usize,
}

fn unit_frame<T: AciScalar>() -> Matrix<T> {
    Matrix::from_col_major_vec(1, 1, vec![T::one()])
}

fn frame_value<T: AciScalar>(
    frames: &[Vec<Option<Matrix<T>>>],
    input: usize,
    site: usize,
    row: usize,
    col: usize,
) -> Option<T> {
    let frame = frames.get(input)?.get(site)?.as_ref()?;
    if row >= frame.nrows() || col >= frame.ncols() {
        return None;
    }
    Some(frame[[row, col]])
}

fn validate_input_site(input: usize, site: usize, n_inputs: usize, n_sites: usize) -> Result<()> {
    if input >= n_inputs {
        return Err(AciError::InvalidInitialGuess {
            message: format!("input index {input} out of bounds for {n_inputs} inputs"),
        });
    }
    if site >= n_sites {
        return Err(AciError::InvalidInitialGuess {
            message: format!("site index {site} out of bounds for tensor train length {n_sites}"),
        });
    }
    Ok(())
}

fn validate_selection(kind: &'static str, indices: &[usize], len: usize) -> Result<()> {
    for &index in indices {
        if index >= len {
            return Err(AciError::InvalidInitialGuess {
                message: format!("{kind} index {index} out of bounds for length {len}"),
            });
        }
    }
    Ok(())
}

fn checked_frame_mul(lhs: usize, rhs: usize, description: &str) -> Result<usize> {
    lhs.checked_mul(rhs)
        .ok_or_else(|| AciError::InvalidInitialGuess {
            message: format!("{description} overflows usize"),
        })
}

fn missing_frame(kind: &'static str, input: usize, site: usize) -> AciError {
    AciError::InvalidInitialGuess {
        message: format!("{kind} frame for input {input}, site {site} is not initialized"),
    }
}

fn right_matrix_julia_order<T: AciScalar>(core: &Tensor3<T>) -> Matrix<T> {
    let left_dim = core.left_dim();
    let site_dim = core.site_dim();
    let right_dim = core.right_dim();
    let mut data = Vec::with_capacity(left_dim * site_dim * right_dim);
    for right in 0..right_dim {
        for site in 0..site_dim {
            for left in 0..left_dim {
                data.push(*core.get3(left, site, right));
            }
        }
    }
    Matrix::from_col_major_vec(left_dim, site_dim * right_dim, data)
}

fn left_matrix_julia_order<T: AciScalar>(core: &Tensor3<T>) -> Matrix<T> {
    let left_dim = core.left_dim();
    let site_dim = core.site_dim();
    let right_dim = core.right_dim();
    let mut data = Vec::with_capacity(left_dim * site_dim * right_dim);
    for right in 0..right_dim {
        for site in 0..site_dim {
            for left in 0..left_dim {
                data.push(*core.get3(left, site, right));
            }
        }
    }
    Matrix::from_col_major_vec(left_dim * site_dim, right_dim, data)
}

pub(crate) fn matrix_to_tensor3<T: AciScalar>(
    matrix: &Matrix<T>,
    left_dim: usize,
    site_dim: usize,
    right_dim: usize,
) -> Result<Tensor3<T>> {
    if matrix.nrows() != left_dim * site_dim || matrix.ncols() != right_dim {
        return Err(AciError::InvalidInitialGuess {
            message: format!(
                "cannot reshape matrix of shape ({}, {}) to tensor core ({left_dim}, \
                 {site_dim}, {right_dim})",
                matrix.nrows(),
                matrix.ncols()
            ),
        });
    }
    Ok(tensor3_from_data(
        matrix.as_col_major_slice().to_vec(),
        left_dim,
        site_dim,
        right_dim,
    )?)
}

pub(crate) fn matrix_into_tensor3<T: AciScalar>(
    matrix: Matrix<T>,
    left_dim: usize,
    site_dim: usize,
    right_dim: usize,
) -> Result<Tensor3<T>> {
    if matrix.nrows() != left_dim * site_dim || matrix.ncols() != right_dim {
        return Err(AciError::InvalidInitialGuess {
            message: format!(
                "cannot reshape matrix of shape ({}, {}) to tensor core ({left_dim}, \
                 {site_dim}, {right_dim})",
                matrix.nrows(),
                matrix.ncols()
            ),
        });
    }
    Ok(tensor3_from_data(
        matrix.into_col_major_vec(),
        left_dim,
        site_dim,
        right_dim,
    )?)
}

pub(crate) fn right_factor_to_tensor3<T: AciScalar>(
    matrix: &Matrix<T>,
    left_dim: usize,
    site_dim: usize,
    right_dim: usize,
) -> Result<Tensor3<T>> {
    if matrix.nrows() != left_dim || matrix.ncols() != site_dim * right_dim {
        return Err(AciError::InvalidInitialGuess {
            message: format!(
                "cannot reshape right factor of shape ({}, {}) to tensor core ({left_dim}, \
                 {site_dim}, {right_dim})",
                matrix.nrows(),
                matrix.ncols()
            ),
        });
    }
    Ok(tensor3_from_data(
        matrix.as_col_major_slice().to_vec(),
        left_dim,
        site_dim,
        right_dim,
    )?)
}

pub(crate) fn right_factor_into_tensor3<T: AciScalar>(
    matrix: Matrix<T>,
    left_dim: usize,
    site_dim: usize,
    right_dim: usize,
) -> Result<Tensor3<T>> {
    if matrix.nrows() != left_dim || matrix.ncols() != site_dim * right_dim {
        return Err(AciError::InvalidInitialGuess {
            message: format!(
                "cannot reshape right factor of shape ({}, {}) to tensor core ({left_dim}, \
                 {site_dim}, {right_dim})",
                matrix.nrows(),
                matrix.ncols()
            ),
        });
    }
    Ok(tensor3_from_data(
        matrix.into_col_major_vec(),
        left_dim,
        site_dim,
        right_dim,
    )?)
}

fn matmul_checked_owned<T: AciScalar>(
    left: Matrix<T>,
    right: Matrix<T>,
    site: usize,
) -> Result<Matrix<T>> {
    if left.ncols() != right.nrows() {
        return Err(AciError::InvalidInitialGuess {
            message: format!(
                "solution core update at site {site} has incompatible matrix shapes: \
                 left ({}, {}), right ({}, {})",
                left.nrows(),
                left.ncols(),
                right.nrows(),
                right.ncols()
            ),
        });
    }

    mat_mul_owned(left, right).map_err(|err| AciError::InvalidInitialGuess {
        message: format!("solution core update at site {site} failed: {err}"),
    })
}

fn frame_matmul_checked<T: AciScalar>(
    left: &Matrix<T>,
    right: &Matrix<T>,
    direction: &'static str,
    input: usize,
    site: usize,
) -> Result<Matrix<T>> {
    if left.ncols() != right.nrows() {
        return Err(AciError::InvalidInitialGuess {
            message: format!(
                "{direction} frame update at input {input}, site {site} has incompatible matrix \
                 shapes: left ({}, {}), right ({}, {})",
                left.nrows(),
                left.ncols(),
                right.nrows(),
                right.ncols()
            ),
        });
    }

    mat_mul(left, right).map_err(|err| AciError::InvalidInitialGuess {
        message: format!("{direction} frame update at input {input}, site {site} failed: {err}"),
    })
}

fn frame_matmul_error(direction: &'static str, err: impl std::fmt::Display) -> AciError {
    AciError::InvalidInitialGuess {
        message: format!("{direction} frame update failed: {err}"),
    }
}

/// Left environment of `tt` at a prefix: the contraction of cores
/// `0..prefix.len()` at the given index values, as a row vector of length
/// `site_tensor(prefix.len()).left_dim()`.
fn left_environment<T: AciScalar>(tt: &SimpleTensorTrain<T>, prefix: &[usize]) -> Result<Vec<T>> {
    let mut env = Matrix::from_col_major_vec(1, 1, vec![T::one()]);
    for (site, &index) in prefix.iter().enumerate() {
        let core = tt.site_tensor(site);
        let slice =
            Matrix::from_col_major_vec(core.left_dim(), core.right_dim(), core.slice_site(index));
        env = mat_mul(&env, &slice).map_err(|err| AciError::InvalidInitialGuess {
            message: format!("left environment contraction failed: {err}"),
        })?;
    }
    Ok(env.as_col_major_slice().to_vec())
}

/// Right environment of `tt` at a suffix: the contraction of the last
/// `suffix.len()` cores at the given index values, as a column vector of
/// length `site_tensor(n - suffix.len()).left_dim()`.
fn right_environment<T: AciScalar>(tt: &SimpleTensorTrain<T>, suffix: &[usize]) -> Result<Vec<T>> {
    let start = tt.len() - suffix.len();
    let mut env = Matrix::from_col_major_vec(1, 1, vec![T::one()]);
    for offset in (0..suffix.len()).rev() {
        let site = start + offset;
        let core = tt.site_tensor(site);
        let slice = Matrix::from_col_major_vec(
            core.left_dim(),
            core.right_dim(),
            core.slice_site(suffix[offset]),
        );
        env = mat_mul(&slice, &env).map_err(|err| AciError::InvalidInitialGuess {
            message: format!("right environment contraction failed: {err}"),
        })?;
    }
    Ok(env.as_col_major_slice().to_vec())
}

/// Whether `frame` contains a row equal to `row`.
fn frame_has_row<T: AciScalar + PartialEq>(frame: Option<&Matrix<T>>, row: &[T]) -> Result<bool> {
    let Some(frame) = frame else {
        return Ok(false);
    };
    if frame.ncols() != row.len() {
        return Err(AciError::InvalidInitialGuess {
            message: format!(
                "cannot match a row of length {} against a frame with {} columns",
                row.len(),
                frame.ncols()
            ),
        });
    }
    for r in 0..frame.nrows() {
        if (0..frame.ncols()).all(|c| frame[[r, c]] == row[c]) {
            return Ok(true);
        }
    }
    Ok(false)
}

/// Append `row` to `frame`. `row` must have length `frame.ncols()`.
fn append_row<T: AciScalar>(frame: &mut Matrix<T>, row: &[T]) -> Result<()> {
    if frame.ncols() != row.len() {
        return Err(AciError::InvalidInitialGuess {
            message: format!(
                "cannot append a row of length {} to a frame with {} columns",
                row.len(),
                frame.ncols()
            ),
        });
    }
    let mut data = frame.as_col_major_slice().to_vec();
    data.extend_from_slice(row);
    *frame = Matrix::from_col_major_vec(frame.nrows() + 1, frame.ncols(), data);
    Ok(())
}

/// Whether `frame` contains a column equal to `col`.
fn frame_has_col<T: AciScalar + PartialEq>(frame: Option<&Matrix<T>>, col: &[T]) -> Result<bool> {
    let Some(frame) = frame else {
        return Ok(false);
    };
    if frame.nrows() != col.len() {
        return Err(AciError::InvalidInitialGuess {
            message: format!(
                "cannot match a column of length {} against a frame with {} rows",
                col.len(),
                frame.nrows()
            ),
        });
    }
    for c in 0..frame.ncols() {
        if (0..frame.nrows()).all(|r| frame[[r, c]] == col[r]) {
            return Ok(true);
        }
    }
    Ok(false)
}

/// Append `col` to `frame`. `col` must have length `frame.nrows()`.
fn append_col<T: AciScalar>(frame: &mut Matrix<T>, col: &[T]) -> Result<()> {
    if frame.nrows() != col.len() {
        return Err(AciError::InvalidInitialGuess {
            message: format!(
                "cannot append a column of length {} to a frame with {} rows",
                col.len(),
                frame.nrows()
            ),
        });
    }
    let mut data = frame.as_col_major_slice().to_vec();
    for value in col {
        data.push(*value);
    }
    *frame = Matrix::from_col_major_vec(frame.nrows(), frame.ncols() + 1, data);
    Ok(())
}
