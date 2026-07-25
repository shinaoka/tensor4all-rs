//! Adaptive tensor cross interpolation over disjoint projected patches.
//!
//! The patch queue, convergence/splitting flow, and diagonal-pivot recycling are
//! derived from `adaptiveinterpolate`, `createpatch`, and `_globalpivots` in
//! TCIAlgorithms.jl at commit e501032278c9dd41b46c5851d8238169c8d178c5
//! (MIT license; Copyright 2023 Ritter.Marc and contributors).

use std::collections::{HashSet, VecDeque};

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use tensor4all_core::{DynIndex, TensorDynLen, TensorElement};
use tensor4all_itensorlike::TensorTrain;
use tensor4all_simplett::{tensor3_from_data, AbstractTensorTrain, TTScalar};
use tensor4all_tcicore::{MatrixLuciScalar, MultiIndex, Scalar};
use tensor4all_tensorbackend::StorageScalar;
use tensor4all_tensorci::{
    crossinterpolate2, TCI2OptimizationResult, TCI2Options, TCI2Termination, TensorCI2,
};

use crate::{PartitionedTT, PartitionedTTError, Projector, Result, SubDomainTT};

const ZERO_SAMPLE_THRESHOLD: f64 = 1.0e-30;

/// Options controlling adaptive interpolation and patch subdivision.
///
/// `AdaptiveInterpolateOptions` augments [`TCI2Options`] with a deterministic
/// patch order and initial-pivot policy. Use [`TCI2Options`] directly when no
/// domain subdivision is needed.
///
/// # Examples
///
/// ```
/// use tensor4all_partitionedtt::AdaptiveInterpolateOptions;
///
/// let options = AdaptiveInterpolateOptions::default();
/// assert_eq!(options.n_initial_pivots, 5);
/// assert!(!options.recycle_pivots);
/// assert!(options.patch_order.is_empty());
/// assert!((options.tci_options.tolerance - 1.0e-8).abs() < 1.0e-16);
/// ```
#[derive(Debug, Clone)]
pub struct AdaptiveInterpolateOptions {
    /// TCI2 sweep, tolerance, rank-cap, and random-search options.
    ///
    /// A patch is accepted when its final normalized or absolute bond error,
    /// according to `normalize_error`, is at most `tolerance`. Otherwise it is
    /// split. When in doubt, use [`TCI2Options::default`].
    pub tci_options: TCI2Options,

    /// Complete order in which site indices are fixed when patches split.
    ///
    /// An empty vector uses the order of `site_indices`. A nonempty vector must
    /// be an exact permutation of `site_indices`, including index identity,
    /// tags, prime level, and dimension.
    pub patch_order: Vec<DynIndex>,

    /// Target number of distinct initial pivot candidates per patch.
    ///
    /// Compatible user and recycled pivots are retained, then deterministic
    /// random candidates are added until this target is reached (or the patch
    /// contains fewer points). The recommended and default value is `5`.
    pub n_initial_pivots: usize,

    /// Whether a nonconverged parent TCI's diagonal pivots seed its child patches.
    ///
    /// Recycling is opt-in because it retains more pivot state. Incompatible
    /// pivots are discarded, and every child is replenished to
    /// `n_initial_pivots`; a child is never classified as zero merely because
    /// no recycled pivot is compatible.
    pub recycle_pivots: bool,
}

impl Default for AdaptiveInterpolateOptions {
    fn default() -> Self {
        Self {
            tci_options: TCI2Options::default(),
            patch_order: Vec::new(),
            n_initial_pivots: 5,
            recycle_pivots: false,
        }
    }
}

#[derive(Debug)]
struct PendingPatch {
    projector: Projector,
    recycled_pivots: Vec<MultiIndex>,
}

/// Adaptively interpolate a discrete function as a partitioned tensor train.
///
/// Each attempted patch runs TCI2 on the sites not fixed by its projector. A
/// converged patch is retained; a patch whose final error exceeds
/// `options.tci_options.tolerance` is split at the next index in
/// `options.patch_order`. Patches with zero or one active site are evaluated
/// exactly because TCI2 requires at least two sites.
///
/// Initial pivots use zero-based coordinates and span the full `site_indices`
/// domain. Compatible supplied and recycled pivots are supplemented with
/// seeded random pivots. If every candidate evaluates below `1e-30`, the patch
/// is represented as zero without further sampling; sparse functions should
/// therefore provide pivots in known nonzero regions.
///
/// # Arguments
///
/// - `f`: scalar evaluator receiving one full, zero-based multi-index.
/// - `batched_f`: optional batch evaluator receiving full multi-indices and
///   returning values in the same order. Use `None` when batching is unavailable.
/// - `site_indices`: one distinct [`DynIndex`] per TCI site, in evaluator order.
/// - `initial_pivots`: full-domain, zero-based pivots. Empty input is allowed.
/// - `options`: TCI2, patch-order, pivot-count, and recycling settings.
///
/// # Returns
///
/// A [`PartitionedTT`] whose mutually disjoint patches cover the full domain.
/// Calling [`PartitionedTT::to_tensor_train`] combines the patches into one TT.
///
/// # Errors
///
/// Returns [`PartitionedTTError::InvalidAdaptiveInterpolationInput`] for empty,
/// duplicate, zero-dimensional, or inconsistently ordered site indices; invalid
/// pivots; a zero pivot target; or invalid TCI tolerances/rank limits. It also
/// forwards TCI2 and tensor-train construction failures.
///
/// # Examples
///
/// ```
/// use tensor4all_core::contract;
/// use tensor4all_partitionedtt::{
///     adaptiveinterpolate, AdaptiveInterpolateOptions, DynIndex, MultiIndex,
/// };
///
/// let sites = vec![DynIndex::new_dyn(2), DynIndex::new_dyn(2)];
/// let f = |idx: &MultiIndex| ((idx[0] + 1) * (idx[1] + 1)) as f64;
/// let result = adaptiveinterpolate::<f64, _, fn(&[MultiIndex]) -> Vec<f64>>(
///     f,
///     None,
///     sites,
///     vec![vec![1, 1]],
///     AdaptiveInterpolateOptions::default(),
/// )
/// .unwrap();
///
/// let tt = result.to_tensor_train().unwrap();
/// let dense = contract(&[tt.tensor(0).unwrap(), tt.tensor(1).unwrap()]).unwrap();
/// assert_eq!(dense.to_vec::<f64>().unwrap(), vec![1.0, 2.0, 2.0, 4.0]);
/// ```
pub fn adaptiveinterpolate<T, F, B>(
    f: F,
    batched_f: Option<B>,
    site_indices: Vec<DynIndex>,
    initial_pivots: Vec<MultiIndex>,
    options: AdaptiveInterpolateOptions,
) -> Result<PartitionedTT>
where
    T: Scalar + TTScalar + MatrixLuciScalar + TensorElement + StorageScalar + Default + Copy,
    F: Fn(&MultiIndex) -> T,
    B: Fn(&[MultiIndex]) -> Vec<T>,
{
    let patch_order = validate_inputs(&site_indices, &initial_pivots, &options)?;
    let mut rng = StdRng::seed_from_u64(options.tci_options.seed.unwrap_or(0));
    let mut pending = VecDeque::from([PendingPatch {
        projector: Projector::new(),
        recycled_pivots: Vec::new(),
    }]);
    let mut accepted = Vec::new();

    while let Some(patch) = pending.pop_front() {
        let active_positions = active_positions(&site_indices, &patch.projector);

        if active_positions.is_empty() {
            let value = f(&expand_pivot(
                &Vec::new(),
                &active_positions,
                &patch.projector,
                &site_indices,
            ));
            let tt = rank_one_full_tt(&site_indices, &patch.projector, value)?;
            accepted.push(SubDomainTT::new(tt, patch.projector));
            continue;
        }

        if active_positions.len() == 1 {
            let dim = site_indices[active_positions[0]].dim;
            let data = (0..dim)
                .map(|value| {
                    f(&expand_pivot(
                        &vec![value],
                        &active_positions,
                        &patch.projector,
                        &site_indices,
                    ))
                })
                .collect();
            let core = tensor3_from_data(data, 1, dim, 1)
                .map_err(|error| PartitionedTTError::TensorTrainError(error.to_string()))?;
            let exact = tensor4all_simplett::TensorTrain::new(vec![core])
                .map_err(|error| PartitionedTTError::TensorTrainError(error.to_string()))?;
            let tt = embed_active_tt(exact, &site_indices, &active_positions, &patch.projector)?;
            accepted.push(SubDomainTT::new(tt, patch.projector));
            continue;
        }

        let candidate_pivots = patch_candidates(
            &site_indices,
            &active_positions,
            &patch.projector,
            &initial_pivots,
            &patch.recycled_pivots,
            options.n_initial_pivots,
            &mut rng,
        )?;
        let candidate_values: Vec<T> = candidate_pivots
            .iter()
            .map(|pivot| {
                f(&expand_pivot(
                    pivot,
                    &active_positions,
                    &patch.projector,
                    &site_indices,
                ))
            })
            .collect();

        if candidate_values
            .iter()
            .all(|value| Scalar::abs_val(*value) < ZERO_SAMPLE_THRESHOLD)
        {
            let tt = rank_one_full_tt(&site_indices, &patch.projector, T::zero())?;
            accepted.push(SubDomainTT::new(tt, patch.projector));
            continue;
        }

        let local_dims = active_positions
            .iter()
            .map(|&position| site_indices[position].dim)
            .collect();
        let local_f = |pivot: &MultiIndex| {
            f(&expand_pivot(
                pivot,
                &active_positions,
                &patch.projector,
                &site_indices,
            ))
        };
        let local_batch = batched_f.as_ref().map(|batch| {
            |pivots: &[MultiIndex]| {
                let full_pivots: Vec<_> = pivots
                    .iter()
                    .map(|pivot| {
                        expand_pivot(pivot, &active_positions, &patch.projector, &site_indices)
                    })
                    .collect();
                batch(&full_pivots)
            }
        });
        let TCI2OptimizationResult {
            tci,
            errors,
            termination,
            ..
        } = crossinterpolate2(
            local_f,
            local_batch,
            local_dims,
            candidate_pivots,
            options.tci_options.clone(),
        )?;
        let normalization = if options.tci_options.normalize_error && tci.max_sample_value() > 0.0 {
            tci.max_sample_value()
        } else {
            1.0
        };
        let final_error = errors
            .last()
            .copied()
            .unwrap_or_else(|| tci.max_bond_error() / normalization);

        if patch_is_accepted(termination, final_error, options.tci_options.tolerance) {
            let simple_tt = tci.to_tensor_train()?;
            let tt = embed_active_tt(
                simple_tt,
                &site_indices,
                &active_positions,
                &patch.projector,
            )?;
            accepted.push(SubDomainTT::new(tt, patch.projector));
            continue;
        }

        let split_index = patch_order
            .iter()
            .find(|index| !patch.projector.is_projected_at(index))
            .ok_or_else(|| {
                PartitionedTTError::InvalidAdaptiveInterpolationInput(
                    "a nonconverged patch has no remaining split index".to_string(),
                )
            })?
            .clone();
        let recycled_pivots = if options.recycle_pivots {
            global_diagonal_pivots(&tci, &active_positions, &patch.projector, &site_indices)
        } else {
            Vec::new()
        };
        for value in 0..split_index.dim {
            let mut child_projector = patch.projector.clone();
            child_projector.insert(split_index.clone(), value);
            pending.push_back(PendingPatch {
                projector: child_projector,
                recycled_pivots: recycled_pivots.clone(),
            });
        }
    }

    PartitionedTT::from_subdomains(accepted)
}

fn validate_inputs(
    site_indices: &[DynIndex],
    initial_pivots: &[MultiIndex],
    options: &AdaptiveInterpolateOptions,
) -> Result<Vec<DynIndex>> {
    if site_indices.is_empty() {
        return Err(PartitionedTTError::InvalidAdaptiveInterpolationInput(
            "site_indices must not be empty".to_string(),
        ));
    }
    if site_indices.iter().any(|index| index.dim == 0) {
        return Err(PartitionedTTError::InvalidAdaptiveInterpolationInput(
            "site indices must have positive dimensions".to_string(),
        ));
    }
    let unique_sites: HashSet<_> = site_indices.iter().cloned().collect();
    if unique_sites.len() != site_indices.len() {
        return Err(PartitionedTTError::InvalidAdaptiveInterpolationInput(
            "site_indices contains duplicate indices".to_string(),
        ));
    }
    if options.n_initial_pivots == 0 {
        return Err(PartitionedTTError::InvalidAdaptiveInterpolationInput(
            "n_initial_pivots must be positive".to_string(),
        ));
    }
    if !options.tci_options.tolerance.is_finite() || options.tci_options.tolerance < 0.0 {
        return Err(PartitionedTTError::InvalidAdaptiveInterpolationInput(
            "TCI tolerance must be finite and nonnegative".to_string(),
        ));
    }
    if options.tci_options.max_iter == 0 {
        return Err(PartitionedTTError::InvalidAdaptiveInterpolationInput(
            "TCI max_iter must be positive".to_string(),
        ));
    }
    if options.tci_options.max_bond_dim == 0 {
        return Err(PartitionedTTError::InvalidAdaptiveInterpolationInput(
            "TCI max_bond_dim must be positive".to_string(),
        ));
    }
    if options.tci_options.ncheck_history == 0 {
        return Err(PartitionedTTError::InvalidAdaptiveInterpolationInput(
            "TCI ncheck_history must be positive".to_string(),
        ));
    }
    if !options.tci_options.tol_margin_global_search.is_finite()
        || options.tci_options.tol_margin_global_search < 0.0
    {
        return Err(PartitionedTTError::InvalidAdaptiveInterpolationInput(
            "TCI tol_margin_global_search must be finite and nonnegative".to_string(),
        ));
    }
    for pivot in initial_pivots {
        if pivot.len() != site_indices.len() {
            return Err(PartitionedTTError::InvalidAdaptiveInterpolationInput(
                "every initial pivot must have one coordinate per site".to_string(),
            ));
        }
        if pivot
            .iter()
            .zip(site_indices)
            .any(|(&value, index)| value >= index.dim)
        {
            return Err(PartitionedTTError::InvalidAdaptiveInterpolationInput(
                "an initial pivot coordinate is outside its site dimension".to_string(),
            ));
        }
    }

    let patch_order = if options.patch_order.is_empty() {
        site_indices.to_vec()
    } else {
        options.patch_order.clone()
    };
    let unique_order: HashSet<_> = patch_order.iter().cloned().collect();
    if patch_order.len() != site_indices.len()
        || unique_order.len() != patch_order.len()
        || unique_order != unique_sites
    {
        return Err(PartitionedTTError::InvalidAdaptiveInterpolationInput(
            "patch_order must be an exact permutation of site_indices".to_string(),
        ));
    }
    Ok(patch_order)
}

fn patch_is_accepted(termination: TCI2Termination, final_error: f64, tolerance: f64) -> bool {
    termination == TCI2Termination::Converged && final_error <= tolerance
}

fn active_positions(site_indices: &[DynIndex], projector: &Projector) -> Vec<usize> {
    site_indices
        .iter()
        .enumerate()
        .filter_map(|(position, index)| (!projector.is_projected_at(index)).then_some(position))
        .collect()
}

fn patch_candidates(
    site_indices: &[DynIndex],
    active_positions: &[usize],
    projector: &Projector,
    initial_pivots: &[MultiIndex],
    recycled_pivots: &[MultiIndex],
    target: usize,
    rng: &mut StdRng,
) -> Result<Vec<MultiIndex>> {
    let mut candidates = Vec::new();
    let mut seen = HashSet::new();
    for full_pivot in initial_pivots.iter().chain(recycled_pivots) {
        if is_compatible_pivot(full_pivot, site_indices, projector) {
            let local = active_positions
                .iter()
                .map(|&position| full_pivot[position])
                .collect::<Vec<_>>();
            if seen.insert(local.clone()) {
                candidates.push(local);
            }
        }
    }

    let local_dims: Vec<_> = active_positions
        .iter()
        .map(|&position| site_indices[position].dim)
        .collect();
    let point_count = local_dims.iter().try_fold(1usize, |count, &dim| {
        count.checked_mul(dim).ok_or_else(|| {
            PartitionedTTError::InvalidAdaptiveInterpolationInput(
                "active patch point count exceeds usize".to_string(),
            )
        })
    })?;
    let desired = target.max(candidates.len()).min(point_count);
    let random_attempts = desired
        .checked_mul(20)
        .and_then(|attempts| attempts.checked_add(100))
        .ok_or_else(|| {
            PartitionedTTError::InvalidAdaptiveInterpolationInput(
                "initial-pivot search attempt count exceeds usize".to_string(),
            )
        })?;
    for _ in 0..random_attempts {
        if candidates.len() >= desired {
            break;
        }
        let pivot: Vec<_> = local_dims
            .iter()
            .map(|&dim| rng.random_range(0..dim))
            .collect();
        if seen.insert(pivot.clone()) {
            candidates.push(pivot);
        }
    }
    for flat in 0..point_count {
        if candidates.len() >= desired {
            break;
        }
        let pivot = decode_col_major(flat, &local_dims);
        if seen.insert(pivot.clone()) {
            candidates.push(pivot);
        }
    }
    Ok(candidates)
}

fn is_compatible_pivot(
    pivot: &MultiIndex,
    site_indices: &[DynIndex],
    projector: &Projector,
) -> bool {
    pivot.len() == site_indices.len()
        && site_indices.iter().enumerate().all(|(position, index)| {
            projector
                .get(index)
                .is_none_or(|value| pivot[position] == value)
        })
}

fn decode_col_major(mut flat: usize, dims: &[usize]) -> MultiIndex {
    dims.iter()
        .map(|&dim| {
            let value = flat % dim;
            flat /= dim;
            value
        })
        .collect()
}

fn expand_pivot(
    local_pivot: &MultiIndex,
    active_positions: &[usize],
    projector: &Projector,
    site_indices: &[DynIndex],
) -> MultiIndex {
    let mut full = vec![0; site_indices.len()];
    for (&position, &value) in active_positions.iter().zip(local_pivot) {
        full[position] = value;
    }
    for (position, index) in site_indices.iter().enumerate() {
        if let Some(value) = projector.get(index) {
            full[position] = value;
        }
    }
    full
}

fn global_diagonal_pivots<T>(
    tci: &TensorCI2<T>,
    active_positions: &[usize],
    projector: &Projector,
    site_indices: &[DynIndex],
) -> Vec<MultiIndex>
where
    T: Scalar + TTScalar + MatrixLuciScalar + Default,
{
    let mut result = Vec::new();
    let mut seen = HashSet::new();
    for bond in 0..active_positions.len() - 1 {
        for (left, right) in tci.i_set(bond + 1).iter().zip(tci.j_set(bond)) {
            let mut local = left.clone();
            local.extend(right);
            if local.len() == active_positions.len() {
                let full = expand_pivot(&local, active_positions, projector, site_indices);
                if seen.insert(full.clone()) {
                    result.push(full);
                }
            }
        }
    }
    result
}

fn embed_active_tt<T>(
    active_tt: tensor4all_simplett::TensorTrain<T>,
    site_indices: &[DynIndex],
    active_positions: &[usize],
    projector: &Projector,
) -> Result<TensorTrain>
where
    T: Scalar + TTScalar + TensorElement + StorageScalar + Default + Copy,
{
    let active_count = active_positions.len();
    let link_dims = active_tt.link_dims();
    let cores = active_tt.into_site_tensors();
    let mut edge_dims = Vec::with_capacity(site_indices.len().saturating_sub(1));
    for edge in 0..site_indices.len().saturating_sub(1) {
        let active_left = active_positions
            .iter()
            .filter(|&&position| position <= edge)
            .count();
        edge_dims.push(if active_left == 0 || active_left == active_count {
            1
        } else {
            link_dims[active_left - 1]
        });
    }
    let edge_indices: Vec<_> = edge_dims
        .iter()
        .map(|&dimension| DynIndex::new_dyn(dimension))
        .collect();

    let mut tensors = Vec::with_capacity(site_indices.len());
    let mut next_active = 0;
    for (position, site_index) in site_indices.iter().enumerate() {
        let left = position.checked_sub(1).map(|edge| &edge_indices[edge]);
        let right = edge_indices.get(position);
        if active_positions.get(next_active) == Some(&position) {
            let core = &cores[next_active];
            let mut indices = Vec::with_capacity(3);
            if let Some(index) = left {
                indices.push(index.clone());
            }
            indices.push(site_index.clone());
            if let Some(index) = right {
                indices.push(index.clone());
            }
            tensors.push(
                TensorDynLen::from_dense(indices, core.to_col_major_vec())
                    .map_err(|error| PartitionedTTError::TensorTrainError(error.to_string()))?,
            );
            next_active += 1;
        } else {
            let value = projector.get(site_index).ok_or_else(|| {
                PartitionedTTError::InvalidAdaptiveInterpolationInput(
                    "an embedded inactive site is missing from its projector".to_string(),
                )
            })?;
            tensors.push(projected_site_tensor::<T>(
                left,
                site_index,
                right,
                value,
                T::one(),
            )?);
        }
    }
    TensorTrain::new(tensors)
        .map_err(|error| PartitionedTTError::TensorTrainError(error.to_string()))
}

fn projected_site_tensor<T>(
    left: Option<&DynIndex>,
    site: &DynIndex,
    right: Option<&DynIndex>,
    value: usize,
    scale: T,
) -> Result<TensorDynLen>
where
    T: Scalar + TensorElement + StorageScalar + Default + Copy,
{
    match (left, right) {
        (Some(left), Some(right)) => TensorDynLen::from_copy_selector(
            left.clone(),
            site.clone(),
            right.clone(),
            value,
            scale,
        )
        .map_err(|error| PartitionedTTError::TensorTrainError(error.to_string())),
        (None, Some(right)) => {
            if right.dim != 1 {
                return Err(PartitionedTTError::TensorTrainError(format!(
                    "projected first site requires a unit right bond, got {}",
                    right.dim
                )));
            }
            let mut data = vec![T::zero(); site.dim];
            data[value] = scale;
            TensorDynLen::from_dense(vec![site.clone(), right.clone()], data)
                .map_err(|error| PartitionedTTError::TensorTrainError(error.to_string()))
        }
        (Some(left), None) => {
            if left.dim != 1 {
                return Err(PartitionedTTError::TensorTrainError(format!(
                    "projected last site requires a unit left bond, got {}",
                    left.dim
                )));
            }
            let mut data = vec![T::zero(); site.dim];
            data[value] = scale;
            TensorDynLen::from_dense(vec![left.clone(), site.clone()], data)
                .map_err(|error| PartitionedTTError::TensorTrainError(error.to_string()))
        }
        (None, None) => {
            let mut data = vec![T::zero(); site.dim];
            data[value] = scale;
            TensorDynLen::from_dense(vec![site.clone()], data)
                .map_err(|error| PartitionedTTError::TensorTrainError(error.to_string()))
        }
    }
}

fn rank_one_full_tt<T>(
    site_indices: &[DynIndex],
    projector: &Projector,
    scale: T,
) -> Result<TensorTrain>
where
    T: Scalar + TensorElement + StorageScalar + Default + Copy,
{
    let edge_indices: Vec<_> = (0..site_indices.len().saturating_sub(1))
        .map(|_| DynIndex::new_dyn(1))
        .collect();
    let mut tensors = Vec::with_capacity(site_indices.len());
    for (position, site) in site_indices.iter().enumerate() {
        let left = position.checked_sub(1).map(|edge| &edge_indices[edge]);
        let right = edge_indices.get(position);
        let local_scale = if position == 0 { scale } else { T::one() };
        if let Some(value) = projector.get(site) {
            tensors.push(projected_site_tensor(
                left,
                site,
                right,
                value,
                local_scale,
            )?);
        } else {
            let mut indices = Vec::with_capacity(3);
            if let Some(index) = left {
                indices.push(index.clone());
            }
            indices.push(site.clone());
            if let Some(index) = right {
                indices.push(index.clone());
            }
            tensors.push(
                TensorDynLen::from_dense(indices, vec![local_scale; site.dim])
                    .map_err(|error| PartitionedTTError::TensorTrainError(error.to_string()))?,
            );
        }
    }
    TensorTrain::new(tensors)
        .map_err(|error| PartitionedTTError::TensorTrainError(error.to_string()))
}

#[cfg(test)]
mod tests;
