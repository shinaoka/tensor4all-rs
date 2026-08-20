//! TreeTN-general adaptive patching.
//!
//! The partition representation follows the independent Rust successor's
//! relationship to [PartitionedMPSs.jl](https://github.com/tensor4all/PartitionedMPSs.jl).
//! The volume-proportional absolute squared-tail policy follows the published
//! method “Adaptive Patching for Tensor Train Computations”
//! ([arXiv:2602.22372](https://arxiv.org/abs/2602.22372)). This module does not
//! implement adaptive interpolation or copy TCIAlgorithms.jl code.

use std::collections::HashMap;
use std::fmt::Debug;
use std::hash::Hash;

use crate::error::{PartitionedTreeTNError, Result};
use crate::partitioned_tree_tn::PartitionedTreeTN;
use crate::projector::{canonical_index_cmp, Projector};
use crate::subdomain_tree_tn::{
    ensure_center, validate_contraction_options, ScalarKind, SubDomainTreeTN,
};
use tensor4all_core::{validate_svd_truncation_options, SvdTruncationPolicy};
use tensor4all_treetn::{contraction::ContractionOptions, TruncationOptions};

type ContractGroup<V> = (
    Vec<(SubDomainTreeTN<V>, SubDomainTreeTN<V>)>,
    Vec<SubDomainTreeTN<V>>,
);

#[derive(Debug)]
struct PatchStats<V>
where
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug,
{
    subdomain: SubDomainTreeTN<V>,
    volume: usize,
    norm_squared: f64,
}

/// Strategy used to choose the next full site index for patch splitting.
///
/// `Sequential` takes the first available index in `PatchingOptions::patch_order`.
/// `ExactParameterGain` forms, budget-truncates, and logically counts every
/// candidate's children, selecting the candidate with the smallest total.
///
/// # Examples
///
/// ```
/// use tensor4all_partitionedtreetn::PatchSplitStrategy;
///
/// assert_eq!(PatchSplitStrategy::default(), PatchSplitStrategy::ExactParameterGain);
/// assert_ne!(PatchSplitStrategy::Sequential, PatchSplitStrategy::ExactParameterGain);
/// ```
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum PatchSplitStrategy {
    /// Split the first available index from the explicit patch order.
    Sequential,
    /// Select the split with the smallest checked logical child parameter count.
    #[default]
    ExactParameterGain,
}

/// Options for TreeTN-general adaptive patching and volume-proportional truncation.
///
/// The scalar truncation surface follows the ITensorMPS `cutoff` convention: a
/// dimensionless discarded-**weight** cutoff applied at every local SVD as an
/// absolute squared-tail budget proportional to the patch's share of the total
/// volume (`local_cutoff_p = cutoff * ||F||^2 * volume_p / total_volume`). It is
/// best effort for the final whole-network error. `max_bond_dim` is the
/// post-truncation hard cap; `None` leaves the bond dimension uncapped. An
/// empty `patch_order` lets adaptive patching consider all unprojected site
/// indices in deterministic full-index order. A nonempty order may be partial,
/// but every entry must be an exact full site-index identity of the input
/// TreeTN with a compatible dimension.
///
/// # Examples
///
/// ```
/// use tensor4all_partitionedtreetn::{PatchSplitStrategy, PatchingOptions};
///
/// let options = PatchingOptions::default();
/// assert_eq!(options.cutoff, 1.0e-24);
/// assert_eq!(options.max_bond_dim, Some(100));
/// assert!(options.patch_order.is_empty());
/// assert_eq!(options.split_strategy, PatchSplitStrategy::ExactParameterGain);
/// ```
#[derive(Debug, Clone)]
pub struct PatchingOptions {
    /// Dimensionless discarded-weight cutoff used to derive each patch's
    /// absolute squared-weight budget.
    ///
    /// One absolute local threshold
    /// `cutoff * ||F||^2 * volume_p / total_volume` is derived per operation
    /// from the operation's input norm `||F||`, and that whole threshold is
    /// applied independently at every local SVD of the patch. Smaller values
    /// retain more information. `cutoff = 0` disables threshold truncation,
    /// leaving only the hard `max_bond_dim` cap. This is best effort for the
    /// final whole-network error: reuse across several bonds or repeated
    /// truncation stages does not provide a global error bound.
    pub cutoff: f64,

    /// Optional maximum retained bond dimension. `Some(0)` is invalid.
    ///
    /// This is a hard cap and takes precedence over `cutoff`. When a
    /// budget-truncated patch still exceeds this cap, adaptive patching
    /// splits it if an unprojected index is available.
    pub max_bond_dim: Option<usize>,

    /// Full site-index order used by [`PatchSplitStrategy::Sequential`].
    ///
    /// Entries are matched by complete index identity, including tags and
    /// prime level; an entry absent from the TreeTN is rejected.
    pub patch_order: Vec<tensor4all_core::DynIndex>,

    /// Candidate-selection strategy used for over-cap patches.
    pub split_strategy: PatchSplitStrategy,
}

impl Default for PatchingOptions {
    fn default() -> Self {
        Self {
            // Behavior-parity translation of the superseded global `rtol =
            // 1e-12` via `cutoff = old_rtol^2`; see the design record.
            cutoff: 1.0e-24,
            max_bond_dim: Some(100),
            patch_order: Vec::new(),
            split_strategy: PatchSplitStrategy::default(),
        }
    }
}

/// Add subdomains with automatic TreeTN patch splitting.
///
/// Input patches that share an equal projector key are first summed by strict
/// subdomain addition, then the result is validated as one homogeneous,
/// exact-topology partition. Over-cap patches are budget-truncated, split
/// along full site indices, and retried until no permitted split remains. The
/// explicit `center` is used for every TreeTN truncation and is never stored in
/// the result.
///
/// # Arguments
///
/// * `subdomains` - Eagerly masked patches with mutually disjoint projectors.
/// * `center` - Existing named TreeTN node used as every truncation center.
/// * `options` - Discarded-weight cutoff, bond cap, split order, and split
///   strategy.
///
/// # Returns
///
/// A new partition containing the retained, budget-truncated patches. The
/// returned partition may still contain a patch above the cap when no
/// unprojected split candidate exists. The whole-network error is best effort
/// and is not bounded by `options.cutoff`.
///
/// # Errors
///
/// Returns [`PartitionedTreeTNError::InvalidOptions`] for invalid cutoff,
/// cap, or split-order options; [`PartitionedTreeTNError::Empty`] only when a
/// nonempty operand is required by a downstream TreeTN operation;
/// [`PartitionedTreeTNError::InvalidCenter`] when `center` is absent;
/// [`PartitionedTreeTNError::TopologyMismatch`],
/// [`PartitionedTreeTNError::SiteIndexMismatch`], or
/// [`PartitionedTreeTNError::DTypeMismatch`] for inconsistent inputs;
/// [`PartitionedTreeTNError::VolumeOverflow`] or
/// [`PartitionedTreeTNError::LogicalParameterCountOverflow`] for checked
/// arithmetic overflow; and typed TreeTN/backend errors for masking or
/// truncation failures. No input patch is mutated.
///
/// # Examples
///
/// ```
/// use tensor4all_core::{DynIndex, IdxTensor};
/// use tensor4all_partitionedtreetn::{
///     add_with_patching, PatchSplitStrategy, PatchingOptions, SubDomainTreeTN,
/// };
/// use tensor4all_treetn::TreeTN;
///
/// let site0 = DynIndex::new_dyn(2);
/// let bond = DynIndex::new_dyn(2);
/// let site1 = DynIndex::new_dyn(2);
/// let left = IdxTensor::from_dense(
///     vec![site0.clone(), bond.clone()],
///     vec![1.0_f64, 0.0, 0.0, 1.0],
/// )?;
/// let right = IdxTensor::from_dense(
///     vec![bond, site1],
///     vec![1.0_f64, 0.0, 0.0, 1.0],
/// )?;
/// let tree = TreeTN::from_tensors(vec![left, right], vec![0usize, 1])?;
/// let patch = SubDomainTreeTN::from_treetn(tree)?;
/// let result = add_with_patching(
///     vec![patch],
///     &0,
///     &PatchingOptions {
///         cutoff: 0.0,
///         max_bond_dim: Some(1),
///         patch_order: vec![site0],
///         split_strategy: PatchSplitStrategy::Sequential,
///     },
/// )?;
/// assert_eq!(result.len(), 2);
/// assert!(result.values().all(|patch| patch.max_bond_dim() <= 1));
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn add_with_patching<V>(
    subdomains: Vec<SubDomainTreeTN<V>>,
    center: &V,
    options: &PatchingOptions,
) -> Result<PartitionedTreeTN<V>>
where
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug,
{
    validate_patching_options(options)?;
    let subdomains = combine_equal_key_subdomains(subdomains)?;
    let partitioned = PartitionedTreeTN::from_subdomains(subdomains)?;
    if partitioned.is_empty() {
        return Ok(partitioned);
    }

    let template = partitioned
        .values()
        .next()
        .ok_or(PartitionedTreeTNError::Empty)?;
    ensure_center(template.data(), center)?;
    validate_patch_order(template, options)?;

    // Pin the reference squared norm and total volume once from the combined
    // input partition. Split and re-truncation stages reuse these values so
    // `cutoff` keeps meaning "fraction of the original weight" throughout the
    // operation; child patches inherit volume-proportional cutoffs.
    let (_, total_volume, reference_norm_squared) = patch_stats_and_totals(&partitioned)?;

    let mut working = partitioned
        .into_iter()
        .map(|(_, subdomain)| subdomain)
        .collect::<Vec<_>>();

    loop {
        working = assign_volume_budgets(
            working,
            options.cutoff,
            reference_norm_squared,
            total_volume,
        )?;
        working = budget_truncate_for_split_decision(working, center)?;
        let over_cap = working.iter().any(|subdomain| {
            options
                .max_bond_dim
                .is_some_and(|cap| subdomain.max_bond_dim() > cap)
        });
        if !over_cap {
            let partitioned = PartitionedTreeTN::from_subdomains(working)?;
            return truncate_adaptive(&partitioned, center, options.cutoff, options.max_bond_dim);
        }

        let mut next = Vec::new();
        let mut split_any = false;
        for subdomain in working {
            let over_cap = options
                .max_bond_dim
                .is_some_and(|cap| subdomain.max_bond_dim() > cap);
            if over_cap {
                if let Some(children) = split_subdomain_by_options(&subdomain, center, options)? {
                    split_any = true;
                    next.extend(children);
                    continue;
                }
            }
            next.push(subdomain);
        }

        if !split_any {
            let partitioned = PartitionedTreeTN::from_subdomains(next)?;
            return truncate_adaptive(&partitioned, center, options.cutoff, options.max_bond_dim);
        }
        working = next;
    }
}

/// Contract two partitions and adaptively truncate the output.
///
/// Contraction uses the already eager-masked patches and the supplied TreeTN
/// contraction options. Its output is then retruncated with cutoffs computed
/// from the corrected output norm. The split fields of `patching_options` are
/// retained for API parity but are not used by this post-contraction pass.
///
/// # Arguments
///
/// * `left` - First validated partition.
/// * `right` - Second validated partition.
/// * `center` - Existing node used by contraction and adaptive truncation.
/// * `contract_options` - Non-dense TreeTN contraction method and rank policy.
/// * `patching_options` - Output discarded-weight cutoff and bond cap.
///
/// # Returns
///
/// A new partition containing compatible pairwise contractions and their
/// volume-proportionally truncated output patches. The whole-network error is
/// best effort and is not bounded by `patching_options.cutoff`.
///
/// # Errors
///
/// Returns the typed validation errors from partition contraction, including
/// invalid options, empty operands, topology, site assignment, dtype, and
/// center errors. It also returns the checked volume or logical-count overflow
/// errors and typed TreeTN/backend failures from output retruncation. Neither
/// input is mutated.
///
/// # Examples
///
/// ```
/// use tensor4all_core::{DynIndex, IdxTensor};
/// use tensor4all_partitionedtreetn::{
///     contract_adaptive, PatchingOptions, PartitionedTreeTN, SubDomainTreeTN,
/// };
/// use tensor4all_treetn::{contraction::ContractionOptions, TreeTN};
///
/// let site = DynIndex::new_dyn(2);
/// let make = |values| -> Result<_, Box<dyn std::error::Error>> {
///     let tensor = IdxTensor::from_dense(vec![site.clone()], values)?;
///     Ok(SubDomainTreeTN::from_treetn(TreeTN::from_tensors(
///         vec![tensor],
///         vec![0usize],
///     )?)?)
/// };
/// let left = PartitionedTreeTN::from_subdomain(make(vec![1.0_f64, 2.0])?)?;
/// let right = PartitionedTreeTN::from_subdomain(make(vec![3.0_f64, 4.0])?)?;
/// let result = contract_adaptive(
///     &left,
///     &right,
///     &0,
///     &ContractionOptions::default(),
///     &PatchingOptions {
///         cutoff: 0.0,
///         max_bond_dim: Some(1),
///         ..PatchingOptions::default()
///     },
/// )?;
/// assert_eq!(result.len(), 1);
/// assert!(result.values().next().ok_or("missing result")?.all_indices().is_empty());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn contract_adaptive<V>(
    left: &PartitionedTreeTN<V>,
    right: &PartitionedTreeTN<V>,
    center: &V,
    contract_options: &ContractionOptions,
    patching_options: &PatchingOptions,
) -> Result<PartitionedTreeTN<V>>
where
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug,
{
    // Reject invalid contraction options before the empty-operand shortcut so
    // a bad policy or `max_bond_dim == 0` cannot pass through a no-op path.
    validate_contraction_options(contract_options)?;
    validate_patching_options(patching_options)?;
    let mut left_patches: Vec<_> = left.values().collect();
    let mut right_patches: Vec<_> = right.values().collect();
    left_patches.sort_by(|a, b| a.projector().canonical_cmp(b.projector()));
    right_patches.sort_by(|a, b| a.projector().canonical_cmp(b.projector()));

    let mut groups: HashMap<Projector, ContractGroup<V>> = HashMap::new();
    for left_patch in left_patches {
        for right_patch in &right_patches {
            if let Some(contribution) =
                left_patch.contract(right_patch, center, contract_options.clone())?
            {
                let entry = groups.entry(contribution.projector().clone()).or_default();
                entry.0.push((left_patch.clone(), (*right_patch).clone()));
                entry.1.push(contribution);
            }
        }
    }

    let mut grouped: Vec<_> = groups.into_iter().collect();
    grouped.sort_by(|(a, _), (b, _)| a.canonical_cmp(b));
    let mut subdomains = Vec::new();
    for (_, (pairs, contributions)) in grouped {
        subdomains.extend(contract_group_project_first(
            pairs,
            Some(contributions),
            center,
            contract_options,
            patching_options,
        )?);
    }
    let contracted = PartitionedTreeTN::from_subdomains(subdomains)?;
    truncate_adaptive(
        &contracted,
        center,
        patching_options.cutoff,
        patching_options.max_bond_dim,
    )
}

fn contract_group_project_first<V>(
    pairs: Vec<(SubDomainTreeTN<V>, SubDomainTreeTN<V>)>,
    precomputed: Option<Vec<SubDomainTreeTN<V>>>,
    center: &V,
    contract_options: &ContractionOptions,
    patching_options: &PatchingOptions,
) -> Result<Vec<SubDomainTreeTN<V>>>
where
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug,
{
    let contributions = if let Some(contributions) = precomputed {
        contributions
    } else {
        let mut contributions = Vec::with_capacity(pairs.len());
        for (left, right) in &pairs {
            if let Some(contribution) = left.contract(right, center, contract_options.clone())? {
                contributions.push(contribution);
            }
        }
        contributions
    };
    let mut iter = contributions.iter();
    let Some(first) = iter.next() else {
        return Ok(Vec::new());
    };
    // Exact-add the whole group without intermediate truncation (strict TreeTN
    // addition), then truncate the completed group exactly once. Repeated
    // add-and-truncate was hash-order dependent and re-approximated the sum at
    // every contribution; the single post-add truncation caps the group once.
    let mut probe = first.clone();
    for contribution in iter {
        probe = probe.add(contribution)?;
    }
    let truncation = truncation_options_from_contract(contract_options);
    probe.truncate(center, truncation)?;

    let Some(cap) = patching_options.max_bond_dim else {
        return Ok(vec![probe]);
    };
    let saturated = probe.max_bond_dim() >= cap
        || contributions
            .iter()
            .any(|contribution| contribution.max_bond_dim() >= cap);
    if !saturated {
        return Ok(vec![probe]);
    }
    let probe_norm = probe.norm_squared()?;
    let probe_volume = subdomain_volume(&probe)?;
    let probe = assign_volume_budgets(
        vec![probe],
        patching_options.cutoff,
        probe_norm,
        probe_volume,
    )?
    .pop()
    .ok_or(PartitionedTreeTNError::Empty)?;
    let Some(index) = choose_split_index(&probe, center, patching_options)? else {
        return Ok(vec![probe]);
    };

    let mut children = Vec::with_capacity(index.dim);
    for value in 0..index.dim {
        let mut child_pairs = Vec::with_capacity(pairs.len());
        let mut projected_left = HashMap::new();
        let mut projected_right = HashMap::new();
        for (left, right) in &pairs {
            let left_key = left.projector().clone();
            if !projected_left.contains_key(&left_key) {
                projected_left.insert(
                    left_key.clone(),
                    project_if_present(left, &index, value, center)?,
                );
            }
            let right_key = right.projector().clone();
            if !projected_right.contains_key(&right_key) {
                projected_right.insert(
                    right_key.clone(),
                    project_if_present(right, &index, value, center)?,
                );
            }
            if let (Some(child_left), Some(child_right)) = (
                projected_left.get(&left_key).cloned().flatten(),
                projected_right.get(&right_key).cloned().flatten(),
            ) {
                child_pairs.push((child_left, child_right));
            }
        }
        children.extend(contract_group_project_first(
            child_pairs,
            None,
            center,
            contract_options,
            patching_options,
        )?);
    }
    Ok(children)
}

#[cfg(test)]
fn add_project_first<V>(
    contributions: Vec<SubDomainTreeTN<V>>,
    center: &V,
    contract_options: &ContractionOptions,
    patching_options: &PatchingOptions,
) -> Result<Vec<SubDomainTreeTN<V>>>
where
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug,
{
    let mut iter = contributions.iter();
    let Some(first) = iter.next() else {
        return Ok(Vec::new());
    };
    // Same collect-group-exact-add-then-truncate-once pattern as the
    // contraction helper: no hash-order-dependent intermediate truncation.
    let mut probe = first.clone();
    let truncation = truncation_options_from_contract(contract_options);
    for contribution in iter {
        probe = probe.add(contribution)?;
    }
    probe.truncate(center, truncation)?;

    let Some(cap) = patching_options.max_bond_dim else {
        return Ok(vec![probe]);
    };
    if probe.max_bond_dim() < cap {
        return Ok(vec![probe]);
    }
    let probe_norm = probe.norm_squared()?;
    let probe_volume = subdomain_volume(&probe)?;
    let probe = assign_volume_budgets(
        vec![probe],
        patching_options.cutoff,
        probe_norm,
        probe_volume,
    )?
    .pop()
    .ok_or(PartitionedTreeTNError::Empty)?;
    let Some(index) = choose_split_index(&probe, center, patching_options)? else {
        return Ok(vec![probe]);
    };

    let mut children = Vec::new();
    for value in 0..index.dim {
        let mut projected = Vec::with_capacity(contributions.len());
        for contribution in &contributions {
            if let Some(child) = project_if_present(contribution, &index, value, center)? {
                projected.push(child);
            }
        }
        children.extend(add_project_first(
            projected,
            center,
            contract_options,
            patching_options,
        )?);
    }
    Ok(children)
}

fn project_if_present<V>(
    subdomain: &SubDomainTreeTN<V>,
    index: &tensor4all_core::DynIndex,
    value: usize,
    center: &V,
) -> Result<Option<SubDomainTreeTN<V>>>
where
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug,
{
    if !subdomain
        .all_indices()
        .iter()
        .any(|candidate| candidate == index)
    {
        return Ok(Some(subdomain.clone()));
    }
    let Some(mut projected) =
        subdomain.project(&Projector::from_pairs([(index.clone(), value)])?)?
    else {
        return Ok(None);
    };
    let threshold = match projected.scalar_kind()? {
        Some(ScalarKind::F32 | ScalarKind::C32) => 64.0 * f32::EPSILON as f64,
        Some(ScalarKind::F64 | ScalarKind::C64) | None => 64.0 * f64::EPSILON,
    };
    projected.truncate(
        center,
        TruncationOptions::new().with_svd_policy(SvdTruncationPolicy::new(threshold)),
    )?;
    Ok(Some(projected))
}

fn truncation_options_from_contract(options: &ContractionOptions) -> TruncationOptions {
    let mut truncation = TruncationOptions::default();
    if let Some(policy) = options.svd_policy {
        truncation = truncation.with_svd_policy(policy);
    }
    if let Some(max_bond_dim) = options.max_bond_dim {
        truncation = truncation.with_max_bond_dim(max_bond_dim);
    }
    truncation
}

/// Truncate a partition with volume-proportional absolute discarded-weight cutoffs.
///
/// The reference squared norm `||F||^2` and total patch volume are measured
/// once from the eager stored patches. A patch receives the absolute local
/// threshold `local_cutoff_p = cutoff * ||F||^2 * volume_p / total_volume`;
/// its unprojected site dimensions define `volume`, while projected dimensions
/// contribute one. That whole threshold is applied independently at every
/// local SVD truncation of the patch. Patches whose norm is at most their
/// local cutoff are dropped.
///
/// # Best effort, not a global bound
///
/// `cutoff` bounds the discarded-squared-weight of one local factorization.
/// Reuse of the same local cutoff across several bonds or repeated truncation
/// stages does not provide a global error bound; the final whole-network
/// relative error is best effort and may exceed `cutoff`. Use `max_bond_dim`
/// when a hard retained-rank bound is required.
///
/// # Arguments
///
/// * `partitioned` - Homogeneous partition with eagerly masked patches.
/// * `center` - Existing TreeTN node used for every local truncation.
/// * `cutoff` - Finite non-negative dimensionless discarded-weight cutoff;
///   `cutoff = 0` disables threshold truncation.
/// * `max_bond_dim` - Optional positive maximum retained bond dimension.
///
/// # Returns
///
/// A new partition with the retained patches and their assigned absolute
/// squared-cutoff metadata. The input partition is unchanged, including when
/// truncation fails.
///
/// # Errors
///
/// Returns [`PartitionedTreeTNError::InvalidOptions`] for invalid tolerance or
/// rank cap, [`PartitionedTreeTNError::InvalidCenter`] for an absent center,
/// [`PartitionedTreeTNError::VolumeOverflow`] for checked volume arithmetic,
/// [`PartitionedTreeTNError::NonFiniteAdaptiveValue`] for a non-finite norm or
/// budget, and typed TreeTN/backend errors for truncation failures. Stored
/// topology, site assignment, and dtype mismatches are rejected before a patch
/// can be dropped as a shortcut.
///
/// # Examples
///
/// ```
/// use tensor4all_core::{DynIndex, IdxTensor};
/// use tensor4all_partitionedtreetn::{
///     truncate_adaptive, PartitionedTreeTN, Projector, SubDomainTreeTN,
/// };
/// use tensor4all_treetn::TreeTN;
///
/// let site = DynIndex::new_dyn(2);
/// let make = |values, coordinate| -> Result<_, Box<dyn std::error::Error>> {
///     let tensor = IdxTensor::from_dense(vec![site.clone()], values)?;
///     let tree = TreeTN::from_tensors(vec![tensor], vec![0usize])?;
///     Ok(SubDomainTreeTN::new(
///         tree,
///         Projector::from_pairs([(site.clone(), coordinate)])?,
///     )?)
/// };
/// let partition = PartitionedTreeTN::from_subdomains(vec![
///     make(vec![10.0_f64, 1.0e12], 0)?,
///     make(vec![1.0e12, 0.01], 1)?,
/// ])?;
/// let result = truncate_adaptive(&partition, &0, 0.1, Some(2))?;
/// assert_eq!(result.len(), 1);
/// assert!(result.contains(&Projector::from_pairs([(site, 0)])?));
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn truncate_adaptive<V>(
    partitioned: &PartitionedTreeTN<V>,
    center: &V,
    cutoff: f64,
    max_bond_dim: Option<usize>,
) -> Result<PartitionedTreeTN<V>>
where
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug,
{
    validate_adaptive_truncation_options(cutoff, max_bond_dim)?;
    partitioned.validate_contents()?;
    if partitioned.is_empty() {
        return Ok(PartitionedTreeTN::new());
    }

    let template = partitioned
        .values()
        .next()
        .ok_or(PartitionedTreeTNError::Empty)?;
    ensure_center(template.data(), center)?;

    let (stats, total_volume, total_norm_squared) = patch_stats_and_totals(partitioned)?;
    if total_volume == 0 {
        return Ok(PartitionedTreeTN::new());
    }

    let global_cutoff_squared = cutoff * total_norm_squared;
    if !global_cutoff_squared.is_finite() {
        return Err(PartitionedTreeTNError::NonFiniteAdaptiveValue);
    }

    let mut retained = Vec::with_capacity(stats.len());
    for stat in stats {
        let local_cutoff_squared =
            global_cutoff_squared * (stat.volume as f64 / total_volume as f64);
        if !local_cutoff_squared.is_finite() {
            return Err(PartitionedTreeTNError::NonFiniteAdaptiveValue);
        }
        if stat.norm_squared <= local_cutoff_squared {
            continue;
        }

        let mut subdomain = stat.subdomain;
        truncate_subdomain_with_cutoff(&mut subdomain, center, local_cutoff_squared, max_bond_dim)?;
        retained.push(subdomain.with_budget_squared(local_cutoff_squared)?);
    }

    PartitionedTreeTN::from_subdomains(retained)
}

/// Sum input patches that share an equal projector key, then return the list.
///
/// `add_with_patching` is named for addition, so two patches that describe the
/// same projector region (for example two unprojected patches over one site)
/// must be added by strict subdomain addition rather than silently replacing
/// each other through `from_subdomains`' last-write-wins key semantics.
fn combine_equal_key_subdomains<V>(
    subdomains: Vec<SubDomainTreeTN<V>>,
) -> Result<Vec<SubDomainTreeTN<V>>>
where
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug,
{
    let mut combined: HashMap<Projector, SubDomainTreeTN<V>> = HashMap::new();
    for subdomain in subdomains {
        let projector = subdomain.projector().clone();
        match combined.remove(&projector) {
            Some(existing) => {
                let summed = existing.add(&subdomain)?;
                combined.insert(projector, summed);
            }
            None => {
                combined.insert(projector, subdomain);
            }
        }
    }
    Ok(combined.into_values().collect())
}

fn validate_patching_options(options: &PatchingOptions) -> Result<()> {
    validate_adaptive_truncation_options(options.cutoff, options.max_bond_dim)?;
    for (position, index) in options.patch_order.iter().enumerate() {
        if index.dim == 0 {
            return Err(PartitionedTreeTNError::InvalidOptions {
                operation: "patching",
                reason: "patch_order cannot contain zero-dimensional indices",
            });
        }
        if options.patch_order[..position]
            .iter()
            .any(|previous| previous == index)
        {
            return Err(PartitionedTreeTNError::InvalidOptions {
                operation: "patching",
                reason: "patch_order cannot contain duplicate full indices",
            });
        }
    }
    Ok(())
}

fn validate_adaptive_truncation_options(cutoff: f64, max_bond_dim: Option<usize>) -> Result<()> {
    if !cutoff.is_finite() || cutoff < 0.0 {
        return Err(PartitionedTreeTNError::InvalidOptions {
            operation: "patching",
            reason: "cutoff must be finite and non-negative",
        });
    }
    validate_svd_truncation_options(max_bond_dim, None).map_err(|_| {
        PartitionedTreeTNError::InvalidOptions {
            operation: "patching",
            reason: "max_bond_dim must be at least 1",
        }
    })?;
    Ok(())
}

fn validate_patch_order<V>(subdomain: &SubDomainTreeTN<V>, options: &PatchingOptions) -> Result<()>
where
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug,
{
    let indices = subdomain.all_indices();
    for requested in &options.patch_order {
        let Some(matched) = indices.iter().find(|candidate| *candidate == requested) else {
            return Err(PartitionedTreeTNError::InvalidOptions {
                operation: "patching",
                reason: "patch_order must contain only full TreeTN site indices",
            });
        };
        // `DynIndex` equality excludes the dimension, so an identity-matching
        // entry with a different dimension must be rejected before any split
        // uses the aliased dimension for masking or indexing.
        if requested.dim != matched.dim {
            return Err(PartitionedTreeTNError::SiteIndexMismatch);
        }
    }
    Ok(())
}

fn assign_volume_budgets<V>(
    subdomains: Vec<SubDomainTreeTN<V>>,
    cutoff: f64,
    reference_norm_squared: f64,
    total_volume: usize,
) -> Result<Vec<SubDomainTreeTN<V>>>
where
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug,
{
    if subdomains.is_empty() {
        return Ok(subdomains);
    }
    if total_volume == 0 {
        return Ok(Vec::new());
    }

    let global_cutoff_squared = cutoff * reference_norm_squared;
    if !global_cutoff_squared.is_finite() {
        return Err(PartitionedTreeTNError::NonFiniteAdaptiveValue);
    }

    subdomains
        .into_iter()
        .map(|subdomain| {
            let volume = subdomain_volume(&subdomain)?;
            let local_cutoff_squared =
                global_cutoff_squared * (volume as f64 / total_volume as f64);
            if !local_cutoff_squared.is_finite() {
                return Err(PartitionedTreeTNError::NonFiniteAdaptiveValue);
            }
            subdomain.with_budget_squared(local_cutoff_squared)
        })
        .collect()
}

fn budget_truncate_for_split_decision<V>(
    subdomains: Vec<SubDomainTreeTN<V>>,
    center: &V,
) -> Result<Vec<SubDomainTreeTN<V>>>
where
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug,
{
    let mut retained = Vec::new();
    for mut subdomain in subdomains {
        let budget_squared =
            subdomain
                .budget_squared()
                .ok_or(PartitionedTreeTNError::InvalidOptions {
                    operation: "patching",
                    reason: "adaptive split decisions require assigned patch budgets",
                })?;
        if subdomain.norm_squared()? <= budget_squared {
            continue;
        }
        truncate_subdomain_with_cutoff(&mut subdomain, center, budget_squared, None)?;
        retained.push(subdomain);
    }
    Ok(retained)
}

fn patch_stats<V>(partitioned: &PartitionedTreeTN<V>) -> Result<Vec<PatchStats<V>>>
where
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug,
{
    let mut stats = Vec::with_capacity(partitioned.len());
    for subdomain in partitioned.values() {
        let volume = subdomain_volume(subdomain)?;
        let norm_squared = subdomain.norm_squared()?;
        if !norm_squared.is_finite() {
            return Err(PartitionedTreeTNError::NonFiniteAdaptiveValue);
        }
        stats.push(PatchStats {
            subdomain: subdomain.clone(),
            volume,
            norm_squared,
        });
    }
    Ok(stats)
}

fn patch_stats_and_totals<V>(
    partitioned: &PartitionedTreeTN<V>,
) -> Result<(Vec<PatchStats<V>>, usize, f64)>
where
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug,
{
    let stats = patch_stats(partitioned)?;
    let total_volume = stats.iter().try_fold(0usize, |total, stat| {
        total
            .checked_add(stat.volume)
            .ok_or(PartitionedTreeTNError::VolumeOverflow)
    })?;
    let total_norm_squared = checked_finite_sum(stats.iter().map(|stat| stat.norm_squared))?;
    Ok((stats, total_volume, total_norm_squared))
}

fn subdomain_volume<V>(subdomain: &SubDomainTreeTN<V>) -> Result<usize>
where
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug,
{
    checked_product(
        subdomain.all_indices().into_iter().map(|index| {
            if index.dim == 0 {
                return Err(PartitionedTreeTNError::InvalidOptions {
                    operation: "patching",
                    reason: "TreeTN site indices must have positive dimensions",
                });
            }
            Ok(if subdomain.projector().is_projected_at(&index) {
                1
            } else {
                index.dim
            })
        }),
        || PartitionedTreeTNError::VolumeOverflow,
    )
}

fn checked_product(
    factors: impl IntoIterator<Item = Result<usize>>,
    overflow: fn() -> PartitionedTreeTNError,
) -> Result<usize> {
    factors.into_iter().try_fold(1usize, |product, factor| {
        product.checked_mul(factor?).ok_or_else(overflow)
    })
}

fn checked_finite_sum(values: impl IntoIterator<Item = f64>) -> Result<f64> {
    values.into_iter().try_fold(0.0, |sum, value| {
        if !value.is_finite() {
            return Err(PartitionedTreeTNError::NonFiniteAdaptiveValue);
        }
        let next = sum + value;
        if next.is_finite() {
            Ok(next)
        } else {
            Err(PartitionedTreeTNError::NonFiniteAdaptiveValue)
        }
    })
}

/// Truncate one subdomain with its whole absolute discarded-weight threshold.
///
/// The whole `local_cutoff_squared` is passed to every local SVD of the
/// truncation sweep. Reuse across several bonds or repeated truncation stages
/// is intentional and best effort; it provides no global error bound, so there
/// is no per-edge division and no consumable remaining-budget bookkeeping.
fn truncate_subdomain_with_cutoff<V>(
    subdomain: &mut SubDomainTreeTN<V>,
    center: &V,
    local_cutoff_squared: f64,
    max_bond_dim: Option<usize>,
) -> Result<()>
where
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug,
{
    if !local_cutoff_squared.is_finite() || local_cutoff_squared < 0.0 {
        return Err(PartitionedTreeTNError::NonFiniteAdaptiveValue);
    }
    if max_bond_dim == Some(0) {
        return Err(PartitionedTreeTNError::InvalidOptions {
            operation: "patching",
            reason: "max_bond_dim must be at least 1",
        });
    }

    // Absolute discarded-tail-sum cutoff on squared singular values: the local
    // factorization drops a suffix while its cumulative squared discarded
    // weight stays below `local_cutoff_squared`.
    let policy = SvdTruncationPolicy::new(local_cutoff_squared)
        .with_absolute()
        .with_squared_values()
        .with_discarded_tail_sum();
    let mut options = TruncationOptions::default().with_svd_policy(policy);
    if let Some(max_bond_dim) = max_bond_dim {
        options = options.with_max_bond_dim(max_bond_dim);
    }
    subdomain.truncate(center, options)
}

fn split_subdomain_by_options<V>(
    subdomain: &SubDomainTreeTN<V>,
    center: &V,
    options: &PatchingOptions,
) -> Result<Option<Vec<SubDomainTreeTN<V>>>>
where
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug,
{
    choose_split_index(subdomain, center, options)?
        .map(|index| split_subdomain(subdomain, &index))
        .transpose()
}

fn choose_split_index<V>(
    subdomain: &SubDomainTreeTN<V>,
    center: &V,
    options: &PatchingOptions,
) -> Result<Option<tensor4all_core::DynIndex>>
where
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug,
{
    let candidates = split_candidates(subdomain, options);
    if candidates.is_empty() {
        return Ok(None);
    }

    match options.split_strategy {
        PatchSplitStrategy::Sequential => Ok(candidates.into_iter().next()),
        PatchSplitStrategy::ExactParameterGain => {
            choose_exact_parameter_gain_split(subdomain, center, options, candidates)
        }
    }
}

fn split_candidates<V>(
    subdomain: &SubDomainTreeTN<V>,
    options: &PatchingOptions,
) -> Vec<tensor4all_core::DynIndex>
where
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug,
{
    let mut all_indices = subdomain.all_indices();
    all_indices.sort_by(canonical_index_cmp);
    let raw_candidates: Vec<_> = if options.patch_order.is_empty() {
        all_indices.clone()
    } else {
        options.patch_order.clone()
    };

    raw_candidates
        .into_iter()
        .filter(|index| !subdomain.is_projected_at(index))
        .filter(|index| all_indices.iter().any(|candidate| candidate == index))
        .fold(Vec::new(), |mut candidates, index| {
            if !candidates.iter().any(|candidate| candidate == &index) {
                candidates.push(index);
            }
            candidates
        })
}

fn choose_exact_parameter_gain_split<V>(
    subdomain: &SubDomainTreeTN<V>,
    center: &V,
    options: &PatchingOptions,
    candidates: Vec<tensor4all_core::DynIndex>,
) -> Result<Option<tensor4all_core::DynIndex>>
where
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug,
{
    let mut best: Option<(usize, tensor4all_core::DynIndex)> = None;
    for candidate in candidates {
        let candidate_count = split_child_parameter_count(subdomain, center, &candidate, options)?;
        if best
            .as_ref()
            .is_none_or(|(best_count, _)| candidate_count < *best_count)
        {
            best = Some((candidate_count, candidate));
        }
    }
    Ok(best.map(|(_, index)| index))
}

fn split_child_parameter_count<V>(
    subdomain: &SubDomainTreeTN<V>,
    center: &V,
    index: &tensor4all_core::DynIndex,
    options: &PatchingOptions,
) -> Result<usize>
where
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug,
{
    let children = split_subdomain(subdomain, index)?;
    children.into_iter().try_fold(0usize, |total, mut child| {
        let budget_squared =
            child
                .budget_squared()
                .ok_or(PartitionedTreeTNError::InvalidOptions {
                    operation: "patching",
                    reason: "adaptive split decisions require assigned patch budgets",
                })?;
        if child.norm_squared()? <= budget_squared {
            return Ok(total);
        }
        truncate_subdomain_with_cutoff(&mut child, center, budget_squared, options.max_bond_dim)?;
        let child_count = logical_parameter_count(&child)?;
        total
            .checked_add(child_count)
            .ok_or(PartitionedTreeTNError::LogicalParameterCountOverflow)
    })
}

fn logical_parameter_count<V>(subdomain: &SubDomainTreeTN<V>) -> Result<usize>
where
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug,
{
    let mut total = 0usize;
    for name in subdomain.data().node_names() {
        let node = subdomain
            .data()
            .node_index(&name)
            .ok_or_else(|| PartitionedTreeTNError::tree("TreeTN node name has no node index"))?;
        let tensor = subdomain
            .data()
            .tensor(node)
            .ok_or_else(|| PartitionedTreeTNError::tree("TreeTN node has no tensor"))?;
        let local = checked_product(tensor.dims().into_iter().map(Ok), || {
            PartitionedTreeTNError::LogicalParameterCountOverflow
        })?;
        total = total
            .checked_add(local)
            .ok_or(PartitionedTreeTNError::LogicalParameterCountOverflow)?;
    }
    Ok(total)
}

fn split_subdomain<V>(
    subdomain: &SubDomainTreeTN<V>,
    index: &tensor4all_core::DynIndex,
) -> Result<Vec<SubDomainTreeTN<V>>>
where
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug,
{
    if index.dim == 0 {
        return Err(PartitionedTreeTNError::InvalidOptions {
            operation: "patching",
            reason: "cannot split along a zero-dimensional index",
        });
    }

    let child_budget_squared = subdomain
        .budget_squared()
        .map(|budget_squared| budget_squared / index.dim as f64);
    let mut children = Vec::with_capacity(index.dim);
    for value in 0..index.dim {
        let projector = Projector::from_pairs([(index.clone(), value)])?;
        let child = subdomain
            .project(&projector)?
            .ok_or(PartitionedTreeTNError::ProjectorConflict)?;
        children.push(match child_budget_squared {
            Some(budget_squared) => child.with_budget_squared(budget_squared)?,
            None => child,
        });
    }
    Ok(children)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tensor4all_core::{DynIndex, IdxTensor};
    use tensor4all_treetn::TreeTN;

    #[test]
    fn adaptive_budgets_are_volume_proportional() {
        // One two-site node; projected sites contribute volume 1 and
        // unprojected sites contribute their dimension. Patch A fixes s0
        // (volume 1 * 3 = 3), patch B fixes s1 (volume 2 * 1 = 2).
        let site0 = DynIndex::new_dyn(2);
        let site1 = DynIndex::new_dyn(3);
        let make = |projector: Projector| {
            SubDomainTreeTN::new(
                TreeTN::from_tensors(
                    vec![IdxTensor::from_dense(
                        vec![site0.clone(), site1.clone()],
                        vec![1.0_f64; 6],
                    )
                    .unwrap()],
                    vec![0usize],
                )
                .unwrap(),
                projector,
            )
            .unwrap()
        };
        let a = make(Projector::from_pairs([(site0.clone(), 0)]).unwrap());
        let b = make(Projector::from_pairs([(site1.clone(), 1)]).unwrap());

        let budgets = assign_volume_budgets(vec![a, b], 0.1, 25.0, 5).unwrap();
        let a_budget = budgets[0].budget_squared().unwrap();
        let b_budget = budgets[1].budget_squared().unwrap();

        // local_cutoff_p = cutoff * ||F||^2 * volume_p / total_volume.
        assert!((a_budget - 0.1 * 25.0 * 3.0 / 5.0).abs() < 1.0e-12);
        assert!((b_budget - 0.1 * 25.0 * 2.0 / 5.0).abs() < 1.0e-12);
        assert!((a_budget / b_budget - 1.5).abs() < 1.0e-12);
    }

    #[test]
    fn local_cutoff_budget_of_zero_leaves_only_the_hard_cap() {
        let site = DynIndex::new_dyn(2);
        let tensor = IdxTensor::from_dense(vec![site.clone()], vec![1.0_f64, 0.5]).unwrap();
        let tree = TreeTN::from_tensors(vec![tensor], vec![0usize]).unwrap();
        let subdomain = SubDomainTreeTN::from_treetn(tree).unwrap();

        let budgets = assign_volume_budgets(vec![subdomain], 0.0, 1.25, 2).unwrap();
        assert_eq!(budgets[0].budget_squared(), Some(0.0));
    }

    #[test]
    fn checked_volume_and_parameter_count_arithmetic_is_fallible() {
        let volume = checked_product([Ok(usize::MAX), Ok(2)], || {
            PartitionedTreeTNError::VolumeOverflow
        });
        assert!(matches!(
            volume,
            Err(PartitionedTreeTNError::VolumeOverflow)
        ));

        let parameters = checked_product([Ok(usize::MAX), Ok(2)], || {
            PartitionedTreeTNError::LogicalParameterCountOverflow
        });
        assert!(matches!(
            parameters,
            Err(PartitionedTreeTNError::LogicalParameterCountOverflow)
        ));
    }

    #[test]
    fn logical_parameter_count_uses_structured_logical_dimensions() {
        let site = DynIndex::new_dyn(3);
        let auxiliary = DynIndex::new_dyn(3);
        let tensor = IdxTensor::from_diag(vec![site, auxiliary], vec![2.0, 3.0, 5.0]).unwrap();
        let tree = TreeTN::from_tensors(vec![tensor], vec![0usize]).unwrap();
        let subdomain = SubDomainTreeTN::from_treetn(tree).unwrap();

        assert_eq!(logical_parameter_count(&subdomain).unwrap(), 9);
    }

    #[test]
    fn projection_removes_exact_zero_bond_space_without_error_budget() {
        let site0 = DynIndex::new_dyn(2);
        let site1 = DynIndex::new_dyn(2);
        let bond = DynIndex::new_bond(2).unwrap();
        let left =
            IdxTensor::from_dense(vec![site0.clone(), bond.clone()], vec![1.0, 0.0, 0.0, 1.0])
                .unwrap();
        let right = IdxTensor::from_dense(vec![bond, site1], vec![1.0, 0.0, 0.0, 1.0]).unwrap();
        let subdomain = SubDomainTreeTN::from_treetn(
            TreeTN::from_tensors(vec![left, right], vec![0usize, 1]).unwrap(),
        )
        .unwrap();
        let projector = Projector::from_pairs([(site0.clone(), 0)]).unwrap();
        let projected = subdomain.project(&projector).unwrap().unwrap();
        let compressed = project_if_present(&subdomain, &site0, 0, &0)
            .unwrap()
            .unwrap();

        assert_eq!(projected.max_bond_dim(), 2);
        assert_eq!(compressed.max_bond_dim(), 1);
        assert!(
            (compressed.norm_squared().unwrap() - projected.norm_squared().unwrap()).abs()
                < 1.0e-12
        );
    }

    #[test]
    fn adaptive_add_projects_original_addends_before_retrying() {
        let site0 = DynIndex::new_dyn(2);
        let site1 = DynIndex::new_dyn(2);
        let product_patch = |left_values: Vec<f64>, right_values: Vec<f64>| {
            let bond = DynIndex::new_bond(1).unwrap();
            let left =
                IdxTensor::from_dense(vec![site0.clone(), bond.clone()], left_values).unwrap();
            let right = IdxTensor::from_dense(vec![bond, site1.clone()], right_values).unwrap();
            SubDomainTreeTN::from_treetn(
                TreeTN::from_tensors(vec![left, right], vec![0usize, 1]).unwrap(),
            )
            .unwrap()
        };
        let contributions = vec![
            product_patch(vec![1.0, 0.0], vec![1.0, 0.0]),
            product_patch(vec![0.0, 1.0], vec![0.0, 1.0]),
        ];
        let contraction = ContractionOptions::default().with_max_bond_dim(1);
        let patching = PatchingOptions {
            cutoff: 0.0,
            max_bond_dim: Some(1),
            patch_order: vec![site0.clone()],
            split_strategy: PatchSplitStrategy::Sequential,
        };

        let result = add_project_first(contributions, &0, &contraction, &patching).unwrap();

        assert_eq!(result.len(), 2);
        assert!(result.iter().all(|patch| patch.max_bond_dim() == 1));
        assert!(result
            .iter()
            .all(|patch| patch.projector().get(&site0).is_some()));
        assert!(
            (result
                .iter()
                .map(|patch| patch.norm_squared().unwrap())
                .sum::<f64>()
                - 2.0)
                .abs()
                < 1.0e-12
        );
    }

    #[test]
    fn exact_parameter_gain_can_choose_a_later_candidate() {
        let site0 = DynIndex::new_dyn(4);
        let site1 = DynIndex::new_dyn(2);
        let site2 = DynIndex::new_dyn(2);
        let bond01 = DynIndex::new_dyn(2);
        let bond12 = DynIndex::new_dyn(2);
        let left =
            IdxTensor::from_dense(vec![site0.clone(), bond01.clone()], vec![1.0_f64; 8]).unwrap();
        let mut center_values = vec![0.0_f64; 8];
        for channel in 0..2 {
            center_values[channel + 2 * channel + 4 * channel] = 1.0;
        }
        let center =
            IdxTensor::from_dense(vec![bond01, site1.clone(), bond12.clone()], center_values)
                .unwrap();
        let right = IdxTensor::from_dense(vec![bond12, site2], vec![1.0_f64; 4]).unwrap();
        let tree = TreeTN::from_tensors(vec![left, center, right], vec![0usize, 1, 2]).unwrap();
        let subdomain = SubDomainTreeTN::from_treetn(tree)
            .unwrap()
            .with_budget_squared(1.0e-20)
            .unwrap();
        let options = PatchingOptions {
            cutoff: 1.0e-24,
            max_bond_dim: Some(1),
            patch_order: vec![site0.clone(), site1.clone()],
            split_strategy: PatchSplitStrategy::ExactParameterGain,
        };

        assert_eq!(
            choose_split_index(&subdomain, &1, &options).unwrap(),
            Some(site1)
        );
        let sequential = PatchingOptions {
            split_strategy: PatchSplitStrategy::Sequential,
            ..options
        };
        assert_eq!(
            choose_split_index(&subdomain, &1, &sequential).unwrap(),
            Some(site0)
        );
    }
}
