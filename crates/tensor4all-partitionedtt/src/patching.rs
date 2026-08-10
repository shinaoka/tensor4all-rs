//! Adaptive patching algorithms for PartitionedTT
//!
//! This module provides functions for adding SubDomainTTs with automatic
//! splitting when bond dimensions exceed limits.
//!
//! **Note**: These functions are experimental and may change or fail for
//! complex use cases. The core functionality relies on TT addition which
//! is now implemented.

use crate::error::{PartitionedTTError, Result};
use crate::partitioned_tt::PartitionedTT;
use crate::projector::Projector;
use crate::subdomain_tt::SubDomainTT;
use tensor4all_core::{DynIndex, SvdTruncationPolicy};
use tensor4all_itensorlike::{ContractOptions, TruncateOptions};

#[derive(Debug, Clone)]
struct PatchStats {
    subdomain: SubDomainTT,
    volume: usize,
    norm_squared: f64,
}

/// Strategy used to choose the next projected index for patch splitting.
///
/// # Examples
///
/// ```
/// use tensor4all_partitionedtt::PatchSplitStrategy;
///
/// assert_eq!(
///     PatchSplitStrategy::default(),
///     PatchSplitStrategy::ExactParameterGain
/// );
/// ```
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum PatchSplitStrategy {
    /// Split the first available index from `PatchingOptions::patch_order`.
    Sequential,
    /// Choose the candidate with the smallest exact post-split TT parameter count.
    #[default]
    ExactParameterGain,
}

/// Options for adaptive patching and truncation.
///
/// Use this when constructing a [`PartitionedTT`] from subdomains whose
/// internal bond dimensions may exceed the target cap. The global tolerance is
/// converted into per-patch absolute squared-norm budgets proportional to each
/// patch volume.
///
/// # Examples
///
/// ```
/// use tensor4all_partitionedtt::PatchingOptions;
///
/// let options = PatchingOptions::default();
/// assert_eq!(options.rtol, 1e-12);
/// assert_eq!(options.max_bond_dim, 100);
/// assert!(options.patch_order.is_empty());
/// assert_eq!(options.split_strategy, tensor4all_partitionedtt::PatchSplitStrategy::ExactParameterGain);
/// ```
#[derive(Debug, Clone)]
pub struct PatchingOptions {
    /// Global relative tolerance used to compute the total squared-error budget.
    ///
    /// Each patch receives `rtol^2 * ||F||^2 * patch_volume / total_volume`.
    /// Use smaller values for stricter truncation. The value must be finite and
    /// non-negative.
    pub rtol: f64,

    /// Maximum retained bond dimension after truncation.
    ///
    /// Values above this cap trigger splitting when `patch_order` contains an
    /// unprojected index for the patch. The value must be at least 1.
    pub max_bond_dim: usize,

    /// Static fallback order for patch splitting.
    ///
    /// The first unprojected index from this list that belongs to an over-cap
    /// subdomain is considered for the next split. An empty order lets the
    /// selected strategy consider all currently unprojected site indices.
    pub patch_order: Vec<DynIndex>,

    /// Strategy used to choose from the available split candidates.
    ///
    /// `Sequential` preserves the explicit order in `patch_order`.
    /// `ExactParameterGain` evaluates every candidate by actually forming,
    /// budget-truncating, and counting the child TT cores.
    pub split_strategy: PatchSplitStrategy,
}

impl Default for PatchingOptions {
    fn default() -> Self {
        Self {
            rtol: 1e-12,
            max_bond_dim: 100,
            patch_order: Vec::new(),
            split_strategy: PatchSplitStrategy::default(),
        }
    }
}

/// Add subdomains with automatic order-driven patching.
///
/// Creates a [`PartitionedTT`] from the supplied subdomains. Subdomains whose
/// budget-truncated bond dimension remains above `options.max_bond_dim` are
/// split along an index chosen by `options.split_strategy`; the result is then
/// truncated using volume-proportional absolute budgets.
///
/// # Errors
///
/// Returns an error when the operation fails (a shape or index mismatch, or
/// /// a backend failure).
///
/// # Examples
///
/// ```
/// use tensor4all_core::index::Index;
/// use tensor4all_partitionedtt::{
///     add_with_patching, PatchingOptions, SubDomainTT, TensorDynLen, TensorTrain,
/// };
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let s0 = Index::new_dyn(2);
/// let bond = Index::new_dyn(3);
/// let s1 = Index::new_dyn(2);
/// let t0 = TensorDynLen::from_dense(
///     vec![s0.clone(), bond.clone()],
///     (1..=6).map(f64::from).collect(),
/// )?;
/// let t1 = TensorDynLen::from_dense(vec![bond, s1], (1..=6).map(f64::from).collect())?;
/// let tt = TensorTrain::new(vec![t0, t1])?;
/// let options = PatchingOptions {
///     rtol: 1e-12,
///     max_bond_dim: 1,
///     patch_order: vec![s0],
///     split_strategy: tensor4all_partitionedtt::PatchSplitStrategy::Sequential,
/// };
///
/// let patched = add_with_patching(vec![SubDomainTT::from_tt(tt)], &options)?;
///
/// assert_eq!(patched.len(), 2);
/// assert!(patched.values().all(|patch| patch.max_bond_dim() <= 1));
/// # Ok(())
/// # }
/// ```
///
pub fn add_with_patching(
    subdomains: Vec<SubDomainTT>,
    options: &PatchingOptions,
) -> Result<PartitionedTT> {
    validate_patching_options(options)?;

    let mut working = subdomains;

    loop {
        working = assign_volume_budgets(working, options.rtol)?;
        working = budget_truncate_for_split_decision(working)?;
        let over_budget = working
            .iter()
            .any(|subdomain| subdomain.max_bond_dim() > options.max_bond_dim);
        if !over_budget {
            let partitioned = PartitionedTT::from_subdomains(working)?;
            return truncate_adaptive(&partitioned, options.rtol, options.max_bond_dim);
        }

        let mut next = Vec::new();
        let mut split_any = false;
        for subdomain in working {
            if subdomain.max_bond_dim() > options.max_bond_dim {
                if let Some(children) = split_subdomain_by_patch_order(&subdomain, options)? {
                    split_any = true;
                    next.extend(children);
                    continue;
                }
            }
            next.push(subdomain);
        }

        if !split_any {
            let partitioned = PartitionedTT::from_subdomains(next)?;
            return truncate_adaptive(&partitioned, options.rtol, options.max_bond_dim);
        }

        working = next;
    }
}

/// Contract two partitioned tensor trains and adaptively truncate the output.
///
/// The contraction first uses the supplied [`ContractOptions`]. The resulting
/// partitioned tensor is then re-truncated with [`truncate_adaptive`], so the
/// final per-patch budgets are computed from the corrected output norm rather
/// than from an input-side estimate.
///
/// # Arguments
///
/// * `left` - Left partitioned tensor train.
/// * `right` - Right partitioned tensor train.
/// * `contract_options` - Contraction method, sweep count, and provisional
///
///   truncation settings used by the underlying TT contraction.
/// * `patching_options` - Global output tolerance and output rank cap. Split
///
///   fields are validated but not used by this post-contraction truncation
///   pass.
///
/// # Returns
///
/// A partitioned tensor train containing the compatible pairwise contractions,
/// with each output patch adaptively truncated against the final output norm.
///
/// # Errors
///
/// Returns an error when the contraction or operation fails (a shape or
/// /// index mismatch, or a backend failure).
///
/// # Examples
///
/// ```
/// use tensor4all_core::index::Index;
/// use tensor4all_partitionedtt::{
///     contract_adaptive, ContractOptions, PartitionedTT, PatchSplitStrategy, PatchingOptions,
///     SubDomainTT, TensorDynLen, TensorTrain,
/// };
///
/// # fn three_site_tt(
/// #     s0: &tensor4all_partitionedtt::DynIndex,
/// #     l01: &tensor4all_partitionedtt::DynIndex,
/// #     s1: &tensor4all_partitionedtt::DynIndex,
/// #     l12: &tensor4all_partitionedtt::DynIndex,
/// #     s2: &tensor4all_partitionedtt::DynIndex,
/// # ) -> Result<TensorTrain, Box<dyn std::error::Error>> {
/// #     let t0 = TensorDynLen::from_dense(vec![s0.clone(), l01.clone()], vec![1.0; 6])?;
/// #     let t1 = TensorDynLen::from_dense(
/// #         vec![l01.clone(), s1.clone(), l12.clone()],
/// #         vec![1.0; 18],
/// #     )?;
/// #     let t2 = TensorDynLen::from_dense(vec![l12.clone(), s2.clone()], vec![1.0; 6])?;
/// #     Ok(TensorTrain::new(vec![t0, t1, t2])?)
/// # }
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let s0 = Index::new_dyn(2);
/// let s1 = Index::new_dyn(2);
/// let s2 = Index::new_dyn(2);
/// let left_l01 = Index::new_dyn(3);
/// let left_l12 = Index::new_dyn(3);
/// let right_l01 = Index::new_dyn(3);
/// let right_l12 = Index::new_dyn(3);
/// let left = PartitionedTT::from_subdomain(SubDomainTT::from_tt(three_site_tt(
///     &s0, &left_l01, &s1, &left_l12, &s2,
/// )?));
/// let right = PartitionedTT::from_subdomain(SubDomainTT::from_tt(three_site_tt(
///     &s0, &right_l01, &s1, &right_l12, &s2,
/// )?));
/// let patching = PatchingOptions {
///     rtol: 0.0,
///     max_bond_dim: 1,
///     patch_order: vec![s0],
///     split_strategy: PatchSplitStrategy::Sequential,
/// };
///
/// let contracted =
///     contract_adaptive(&left, &right, &ContractOptions::default(), &patching)?;
///
/// assert_eq!(contracted.len(), 1);
/// assert!(contracted.values().all(|patch| patch.max_bond_dim() <= 1));
/// # Ok(())
/// # }
/// ```
pub fn contract_adaptive(
    left: &PartitionedTT,
    right: &PartitionedTT,
    contract_options: &ContractOptions,
    patching_options: &PatchingOptions,
) -> Result<PartitionedTT> {
    validate_patching_options(patching_options)?;
    let contracted = left.contract(right, contract_options)?;
    truncate_adaptive(
        &contracted,
        patching_options.rtol,
        patching_options.max_bond_dim,
    )
}

/// Truncate a partitioned tensor train with volume-proportional absolute budgets.
///
/// The total squared-error budget is `rtol^2 * ||F||^2`, where `||F||^2` is the
/// sum of projected patch norms. Each patch receives a share proportional to
/// its unprojected grid volume. Patches whose projected squared norm is at or
/// below their budget are dropped entirely.
///
/// # Arguments
///
/// * `partitioned` - Partitioned tensor train with mutually disjoint patches.
/// * `rtol` - Global relative tolerance; must be finite and non-negative.
/// * `max_bond_dim` - Maximum retained bond dimension; must be at least 1.
///
/// # Returns
///
/// A new [`PartitionedTT`] containing retained patches with projected data and
/// bond dimensions capped by `max_bond_dim` where truncation succeeds.
///
/// # Errors
///
/// Returns an error if the tolerance or rank cap is invalid, if patch volume
/// arithmetic overflows, or if tensor train truncation fails.
///
/// # Examples
///
/// ```
/// use tensor4all_core::index::Index;
/// use tensor4all_partitionedtt::{
///     truncate_adaptive, PartitionedTT, Projector, SubDomainTT, TensorDynLen, TensorTrain,
/// };
///
/// # fn rank_one_tt(
/// #     s0: &tensor4all_partitionedtt::DynIndex,
/// #     s1: &tensor4all_partitionedtt::DynIndex,
/// #     scale: f64,
/// # ) -> Result<TensorTrain, Box<dyn std::error::Error>> {
/// #     let bond = Index::new_dyn(1);
/// #     let t0 = TensorDynLen::from_dense(vec![s0.clone(), bond.clone()], vec![scale; s0.dim])?;
/// #     let t1 = TensorDynLen::from_dense(vec![bond, s1.clone()], vec![1.0; s1.dim])?;
/// #     Ok(TensorTrain::new(vec![t0, t1])?)
/// # }
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let s0 = Index::new_dyn(2);
/// let s1 = Index::new_dyn(2);
/// let high_proj = Projector::from_pairs([(s0.clone(), 0)]);
/// let low_proj = Projector::from_pairs([(s0.clone(), 1)]);
/// let high = SubDomainTT::new(rank_one_tt(&s0, &s1, 10.0)?, high_proj.clone());
/// let low = SubDomainTT::new(rank_one_tt(&s0, &s1, 0.01)?, low_proj.clone());
/// let partitioned = PartitionedTT::from_subdomains(vec![high, low])?;
///
/// let truncated = truncate_adaptive(&partitioned, 0.01, 4)?;
///
/// assert_eq!(truncated.len(), 1);
/// assert!(truncated.contains(&high_proj));
/// assert!(!truncated.contains(&low_proj));
/// # Ok(())
/// # }
/// ```
pub fn truncate_adaptive(
    partitioned: &PartitionedTT,
    rtol: f64,
    max_bond_dim: usize,
) -> Result<PartitionedTT> {
    validate_truncation_options(rtol, max_bond_dim)?;

    if partitioned.is_empty() {
        return Ok(PartitionedTT::new());
    }

    let stats = patch_stats(partitioned)?;
    let total_volume = stats.iter().try_fold(0usize, |acc, stat| {
        acc.checked_add(stat.volume).ok_or_else(|| {
            PartitionedTTError::InvalidOptions("partition volume overflowed usize".to_string())
        })
    })?;
    if total_volume == 0 {
        return Ok(PartitionedTT::new());
    }

    let total_norm_squared: f64 = stats.iter().map(|stat| stat.norm_squared).sum();
    if !total_norm_squared.is_finite() {
        return Err(PartitionedTTError::tensor_train_operation(
            "partitioned norm is not finite".to_string(),
        ));
    }
    let global_budget_squared = rtol * rtol * total_norm_squared;
    if !global_budget_squared.is_finite() {
        return Err(PartitionedTTError::InvalidOptions(
            "rtol produced a non-finite truncation budget".to_string(),
        ));
    }

    let mut retained = Vec::new();
    for stat in stats {
        let patch_budget_squared =
            global_budget_squared * (stat.volume as f64 / total_volume as f64);
        if stat.norm_squared <= patch_budget_squared {
            continue;
        }

        let mut subdomain = stat.subdomain;
        let _unused_budget_squared =
            truncate_subdomain_with_budget(&mut subdomain, patch_budget_squared, max_bond_dim)?;
        retained.push(subdomain.with_budget_squared(patch_budget_squared));
    }

    PartitionedTT::from_subdomains(retained)
}

fn validate_patching_options(options: &PatchingOptions) -> Result<()> {
    validate_truncation_options(options.rtol, options.max_bond_dim)?;
    for index in &options.patch_order {
        if index.dim == 0 {
            return Err(PartitionedTTError::InvalidOptions(
                "patch_order cannot contain zero-dimensional indices".to_string(),
            ));
        }
    }
    Ok(())
}

fn assign_volume_budgets(subdomains: Vec<SubDomainTT>, rtol: f64) -> Result<Vec<SubDomainTT>> {
    if subdomains.is_empty() {
        return Ok(subdomains);
    }

    let partitioned = PartitionedTT::from_subdomains(subdomains)?;
    let stats = patch_stats(&partitioned)?;
    let total_volume = stats.iter().try_fold(0usize, |acc, stat| {
        acc.checked_add(stat.volume).ok_or_else(|| {
            PartitionedTTError::InvalidOptions("partition volume overflowed usize".to_string())
        })
    })?;
    if total_volume == 0 {
        return Ok(Vec::new());
    }
    let total_norm_squared: f64 = stats.iter().map(|stat| stat.norm_squared).sum();
    if !total_norm_squared.is_finite() {
        return Err(PartitionedTTError::tensor_train_operation(
            "partitioned norm is not finite".to_string(),
        ));
    }
    let global_budget_squared = rtol * rtol * total_norm_squared;
    if !global_budget_squared.is_finite() {
        return Err(PartitionedTTError::InvalidOptions(
            "rtol produced a non-finite truncation budget".to_string(),
        ));
    }

    stats
        .into_iter()
        .map(|stat| {
            let patch_budget_squared =
                global_budget_squared * (stat.volume as f64 / total_volume as f64);
            Ok(stat.subdomain.with_budget_squared(patch_budget_squared))
        })
        .collect()
}

fn budget_truncate_for_split_decision(subdomains: Vec<SubDomainTT>) -> Result<Vec<SubDomainTT>> {
    let mut retained = Vec::new();
    for mut subdomain in subdomains {
        let budget_squared = subdomain.budget_squared().ok_or_else(|| {
            PartitionedTTError::InvalidOptions(
                "subdomain was missing a split-decision budget".to_string(),
            )
        })?;
        if subdomain.norm_squared()? <= budget_squared {
            continue;
        }
        truncate_subdomain_with_budget_only(&mut subdomain, budget_squared)?;
        retained.push(subdomain);
    }
    Ok(retained)
}

fn validate_truncation_options(rtol: f64, max_bond_dim: usize) -> Result<()> {
    if !rtol.is_finite() || rtol < 0.0 {
        return Err(PartitionedTTError::InvalidOptions(
            "rtol must be finite and non-negative".to_string(),
        ));
    }
    if max_bond_dim == 0 {
        return Err(PartitionedTTError::InvalidOptions(
            "max_bond_dim must be at least 1".to_string(),
        ));
    }
    Ok(())
}

fn patch_stats(partitioned: &PartitionedTT) -> Result<Vec<PatchStats>> {
    let mut stats = Vec::with_capacity(partitioned.len());
    for subdomain in partitioned.values() {
        let projected = projected_subdomain(subdomain)?;
        let volume = subdomain_volume(&projected)?;
        let norm_squared = projected.norm_squared()?;
        stats.push(PatchStats {
            subdomain: projected,
            volume,
            norm_squared,
        });
    }
    Ok(stats)
}

fn projected_subdomain(subdomain: &SubDomainTT) -> Result<SubDomainTT> {
    if subdomain.projector().is_empty() {
        return Ok(subdomain.clone());
    }
    subdomain.project(subdomain.projector())?.ok_or_else(|| {
        PartitionedTTError::IncompatibleProjectors(
            "subdomain projector was incompatible with itself".to_string(),
        )
    })
}

fn subdomain_volume(subdomain: &SubDomainTT) -> Result<usize> {
    subdomain
        .all_indices()
        .into_iter()
        .try_fold(1usize, |volume, index| {
            let factor = if subdomain.projector().is_projected_at(&index) {
                1
            } else {
                index.dim
            };
            volume.checked_mul(factor).ok_or_else(|| {
                PartitionedTTError::InvalidOptions("subdomain volume overflowed usize".to_string())
            })
        })
}

fn truncate_subdomain_with_budget(
    subdomain: &mut SubDomainTT,
    budget_squared: f64,
    max_bond_dim: usize,
) -> Result<f64> {
    truncate_subdomain_with_optional_max_rank(subdomain, budget_squared, Some(max_bond_dim))
}

fn truncate_subdomain_with_budget_only(
    subdomain: &mut SubDomainTT,
    budget_squared: f64,
) -> Result<f64> {
    truncate_subdomain_with_optional_max_rank(subdomain, budget_squared, None)
}

fn truncate_subdomain_with_optional_max_rank(
    subdomain: &mut SubDomainTT,
    budget_squared: f64,
    max_bond_dim: Option<usize>,
) -> Result<f64> {
    let before = subdomain.norm_squared()?;
    let policy = SvdTruncationPolicy::new(budget_squared)
        .with_absolute()
        .with_squared_values()
        .with_discarded_tail_sum();
    let mut options = TruncateOptions::svd().with_svd_policy(policy);
    if let Some(max_bond_dim) = max_bond_dim {
        options = options.with_max_rank(max_bond_dim);
    }
    subdomain.truncate(&options)?;
    let after = subdomain.norm_squared()?;
    let used = (before - after).max(0.0);
    Ok((budget_squared - used).max(0.0))
}

fn split_subdomain_by_patch_order(
    subdomain: &SubDomainTT,
    options: &PatchingOptions,
) -> Result<Option<Vec<SubDomainTT>>> {
    choose_split_index(subdomain, options)?
        .map(|index| split_subdomain(subdomain, &index))
        .transpose()
}

fn choose_split_index(
    subdomain: &SubDomainTT,
    options: &PatchingOptions,
) -> Result<Option<DynIndex>> {
    let candidates = split_candidates(subdomain, options);
    if candidates.is_empty() {
        return Ok(None);
    }

    match options.split_strategy {
        PatchSplitStrategy::Sequential => Ok(candidates.into_iter().next()),
        PatchSplitStrategy::ExactParameterGain => {
            choose_exact_parameter_gain_split(subdomain, options, candidates)
        }
    }
}

fn split_candidates(subdomain: &SubDomainTT, options: &PatchingOptions) -> Vec<DynIndex> {
    let all_indices = subdomain.all_indices();
    let raw_candidates: Vec<DynIndex> = if options.patch_order.is_empty() {
        all_indices.clone()
    } else {
        options.patch_order.clone()
    };

    let mut candidates = Vec::new();
    for index in raw_candidates {
        if subdomain.is_projected_at(&index) {
            continue;
        }
        if !all_indices.iter().any(|candidate| candidate == &index) {
            continue;
        }
        if candidates.iter().any(|candidate| candidate == &index) {
            continue;
        }
        candidates.push(index);
    }
    candidates
}

fn choose_exact_parameter_gain_split(
    subdomain: &SubDomainTT,
    options: &PatchingOptions,
    candidates: Vec<DynIndex>,
) -> Result<Option<DynIndex>> {
    let mut best: Option<(usize, DynIndex)> = None;
    for candidate in candidates {
        let candidate_count = split_child_parameter_count(subdomain, &candidate, options)?;
        if best
            .as_ref()
            .is_none_or(|(best_count, _)| candidate_count < *best_count)
        {
            best = Some((candidate_count, candidate));
        }
    }
    Ok(best.map(|(_, index)| index))
}

fn split_child_parameter_count(
    subdomain: &SubDomainTT,
    index: &DynIndex,
    options: &PatchingOptions,
) -> Result<usize> {
    let children = split_subdomain(subdomain, index)?;
    children.into_iter().try_fold(0usize, |total, mut child| {
        let budget_squared = child.budget_squared().unwrap_or(0.0);
        let projected = projected_subdomain(&child)?;
        if projected.norm_squared()? <= budget_squared {
            return Ok(total);
        }
        truncate_subdomain_with_budget(&mut child, budget_squared, options.max_bond_dim)?;
        let child_count = subdomain_parameter_count(&child)?;
        total.checked_add(child_count).ok_or_else(|| {
            PartitionedTTError::InvalidOptions(
                "split child parameter count overflowed usize".to_string(),
            )
        })
    })
}

fn subdomain_parameter_count(subdomain: &SubDomainTT) -> Result<usize> {
    subdomain
        .data()
        .tensors()
        .into_iter()
        .try_fold(0usize, |total, tensor| {
            let tensor_count = tensor.dims().into_iter().try_fold(1usize, |acc, dim| {
                acc.checked_mul(dim).ok_or_else(|| {
                    PartitionedTTError::InvalidOptions(
                        "tensor parameter count overflowed usize".to_string(),
                    )
                })
            })?;
            total.checked_add(tensor_count).ok_or_else(|| {
                PartitionedTTError::InvalidOptions(
                    "subdomain parameter count overflowed usize".to_string(),
                )
            })
        })
}

fn split_subdomain(subdomain: &SubDomainTT, index: &DynIndex) -> Result<Vec<SubDomainTT>> {
    if index.dim == 0 {
        return Err(PartitionedTTError::InvalidOptions(
            "cannot split along a zero-dimensional index".to_string(),
        ));
    }

    let child_budget_squared = subdomain
        .budget_squared()
        .map(|budget_squared| budget_squared / index.dim as f64);
    let mut children = Vec::with_capacity(index.dim);
    for value in 0..index.dim {
        let projector = Projector::from_pairs([(index.clone(), value)]);
        let child = subdomain.project(&projector)?.ok_or_else(|| {
            PartitionedTTError::IncompatibleProjectors(
                "split projector was incompatible with subdomain projector".to_string(),
            )
        })?;
        children.push(match child_budget_squared {
            Some(budget_squared) => child.with_budget_squared(budget_squared),
            None => child,
        });
    }
    Ok(children)
}

#[cfg(test)]
mod tests;
