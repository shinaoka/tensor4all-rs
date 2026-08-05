//! Adaptive physical-domain patching for tree tensor cross interpolation.
//!
//! This module adapts the patch-queue algorithm from
//! `tensor4all-partitionedtt::adaptiveinterpolate` to the tree setting. A
//! function that a single [`TreeTN`](tensor4all_treetn::TreeTN) cannot fit at
//! the requested tolerance is split by fixing one physical site at a time, and
//! each child is fit independently by ordinary
//! [`crossinterpolate2`](crate::crossinterpolate2).
//!
//! ## What patching does
//!
//! - Patching **fixes physical site values**. Each patch records, for every
//!   TreeTCI site, either `None` (active) or `Some(value)` (fixed).
//! - **All patches share one fixed [`TreeTciGraph`](crate::TreeTciGraph).** The
//!   topology never changes between patches.
//! - **Fixed sites are represented by local dimension one.** A patch keeps the
//!   same number of TreeTCI sites as the original problem; a fixed site simply
//!   has a single allowed coordinate.
//! - **Splitting is sequential.** When a patch fails to converge, the next
//!   unprojected site in `patch_order` is fixed, producing one child per
//!   original local value at that site.
//! - **Zero patches may be omitted.** A patch whose candidate initial pivots all
//!   evaluate below `1e-30` is treated as numerically zero and dropped rather
//!   than fed to TreeTCI.
//!
//! ## Not yet implemented
//!
//! Pivot recycling from parent patches, patched TreeTN arithmetic (addition,
//! contraction, truncation, conversion to one monolithic TreeTN), adaptive
//! split-site selection, and parallel patch processing are deliberately out of
//! scope for this first version.
//!
//! ## Sparse-function caveat
//!
//! Zero-patch detection uses finite random probing. A genuinely sparse function
//! whose nonzero support is missed by the random probes would be wrongly
//! classified as zero. Provide initial pivots in known nonzero regions so the
//! root patch is not discarded by accident.

use std::collections::hash_map::DefaultHasher;
use std::collections::{HashSet, VecDeque};
use std::hash::{Hash, Hasher};

use anyhow::{ensure, Result};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use tensor4all_core::{CommonScalar, TensorDynLen};
use tensor4all_treetn::TreeTN;

use crate::{
    crossinterpolate2, materialize::FullPivLuScalar, GlobalIndexBatch, MultiIndex,
    PivotCandidateProposer, TreeTciGraph, TreeTciOptions,
};

/// Magnitude below which a candidate sample is treated as numerically zero.
///
/// Matches the threshold used by `tensor4all-partitionedtt`'s adaptive
/// interpolation: if every candidate pivot of a patch evaluates below this, the
/// patch is omitted without running TreeTCI.
const ZERO_SAMPLE_THRESHOLD: f64 = 1.0e-30;

/// One accepted patch from adaptive tree TCI.
///
/// A patch is the restriction of the source function to the coordinates where
/// every `Some` site in `fixed_values` is held to its recorded value; `None`
/// sites remain active. `treetn` is the TreeTCI fit of that restriction.
#[derive(Debug)]
pub struct AdaptiveTreeTciPatch {
    /// Per-site assignment: `None` for an active site, `Some(value)` for a site
    /// fixed to `value`.
    pub fixed_values: Vec<Option<usize>>,
    /// TreeTCI fit of this patch's restriction of the source function.
    pub treetn: TreeTN<TensorDynLen, usize>,
    /// Final normalized bond error reported by TreeTCI for this patch.
    pub final_error: f64,
    /// Maximum bond dimension reached by this patch's TreeTCI fit.
    pub max_rank: usize,
}

/// Result of adaptive tree tensor cross interpolation.
///
/// `patches` are mutually disjoint and together cover every nonzero region of
/// the source function. An empty `patches` list represents the identically zero
/// function.
#[derive(Debug)]
pub struct AdaptiveTreeTciResult {
    /// Original local dimensions, unchanged across all patches.
    pub local_dims: Vec<usize>,
    /// Accepted patches in acceptance order.
    pub patches: Vec<AdaptiveTreeTciPatch>,
}

impl AdaptiveTreeTciResult {
    /// Number of accepted patches.
    pub fn len(&self) -> usize {
        self.patches.len()
    }

    /// Whether the result contains no accepted patches (the zero function).
    pub fn is_empty(&self) -> bool {
        self.patches.is_empty()
    }
}

/// Options controlling adaptive tree TCI patching.
///
/// Wraps ordinary [`TreeTciOptions`] with a deterministic patch split order and
/// initial-pivot policy.
///
/// # Examples
///
/// ```
/// use tensor4all_treetci::AdaptiveTreeTciOptions;
///
/// let options = AdaptiveTreeTciOptions::default();
/// assert_eq!(options.n_initial_pivots, 5);
/// assert!(options.patch_order.is_empty());
/// assert_eq!(options.seed, 0);
/// ```
#[derive(Clone, Debug)]
pub struct AdaptiveTreeTciOptions {
    /// Sweep, tolerance, rank-cap, and error-normalization options forwarded to
    /// every patch's [`crossinterpolate2`](crate::crossinterpolate2) call. A
    /// patch is accepted when its final reported error is at most
    /// `tci_options.tolerance`.
    pub tci_options: TreeTciOptions,

    /// Order in which sites are fixed when a patch is split.
    ///
    /// An empty vector uses `0..local_dims.len()`. A nonempty vector must be an
    /// exact permutation of every site position.
    pub patch_order: Vec<usize>,

    /// Target number of distinct initial pivot candidates per patch.
    ///
    /// Compatible input pivots are kept, then deterministic random candidates
    /// are added until this target is reached or the patch's point count is
    /// exhausted. Default: `5`.
    pub n_initial_pivots: usize,

    /// Base seed for per-pivot random generation. Each patch derives its own
    /// seed from this value and its fixed assignment, so generated pivots are
    /// deterministic and independent of queue order.
    pub seed: u64,
}

impl Default for AdaptiveTreeTciOptions {
    fn default() -> Self {
        Self {
            tci_options: TreeTciOptions::default(),
            patch_order: Vec::new(),
            n_initial_pivots: 5,
            seed: 0,
        }
    }
}

/// Adaptively cross-interpolate a function as a collection of patched
/// [`TreeTN`]s.
///
/// The algorithm starts with one unprojected root patch, runs
/// [`crossinterpolate2`] on it, and accepts it when its final reported error is
/// at most `options.tci_options.tolerance`. A nonconverged patch is split at
/// the next unprojected site in `options.patch_order`, producing one child per
/// original local value at that site. Splitting continues breadth-first until
/// every nonzero patch is accepted; numerically zero patches are omitted.
///
/// All patches share the same fixed `graph` topology. Patching only fixes
/// physical site values; it never changes the graph.
///
/// # Arguments
///
/// - `evaluate`: batch evaluator of the source function on the original full
///   discrete domain.
/// - `local_dims`: original local dimension per site.
/// - `graph`: fixed [`TreeTciGraph`] reused unchanged for every patch.
/// - `initial_pivots`: full-domain pivots. Each patch keeps the pivots
///   compatible with its fixed values.
/// - `options`: TreeTCI options, patch order, pivot-count, and seed.
/// - `center_site`: optional BFS root forwarded to every patch's
///   `crossinterpolate2` call.
/// - `proposer`: TreeTCI pivot proposer forwarded to every patch.
///
/// # Returns
///
/// An [`AdaptiveTreeTciResult`] whose mutually disjoint patches cover every
/// nonzero region of the source function. An empty patch list represents the
/// identically zero function.
///
/// # Errors
///
/// Returns an error for invalid input (empty or nonpositive dimensions,
/// dimension/graph mismatch, zero `n_initial_pivots`, invalid tolerance or
/// rank/iteration limits, out-of-range pivots, a `patch_order` that is not an
/// exact permutation, an out-of-range `center_site`), or if a nonconverged
/// patch has no remaining site to split. It also forwards TreeTCI failures.
///
/// This entry point returns `anyhow::Result` to match the existing
/// [`crossinterpolate2`](crate::crossinterpolate2) convention; the rest of the
/// crate uses the same surface.
///
/// # Examples
///
/// Forced patching on the two-site identity `f(i, j) = 1 if i == j else 0`.
/// With `max_bond_dim = 1` the identity (rank 2) cannot fit a single patch, so
/// site 0 is split and each child is rank one.
///
/// ```
/// use tensor4all_treetci::{
///     adaptive_crossinterpolate2, AdaptiveTreeTciOptions, DefaultProposer,
///     GlobalIndexBatch, TreeTciEdge, TreeTciGraph, TreeTciOptions,
/// };
/// use anyhow::Result;
///
/// let graph = TreeTciGraph::new(2, &[TreeTciEdge::new(0, 1)]).unwrap();
/// let evaluate = |batch: GlobalIndexBatch<'_>| -> Result<Vec<f64>> {
///     let mut values = Vec::with_capacity(batch.n_points());
///     for p in 0..batch.n_points() {
///         let i = batch.get(0, p).unwrap();
///         let j = batch.get(1, p).unwrap();
///         values.push(if i == j { 1.0 } else { 0.0 });
///     }
///     Ok(values)
/// };
///
/// let options = AdaptiveTreeTciOptions {
///     tci_options: TreeTciOptions {
///         tolerance: 1e-10,
///         max_iter: 5,
///         max_bond_dim: 1,
///         ..Default::default()
///     },
///     patch_order: vec![0, 1],
///     ..Default::default()
/// };
///
/// let result = adaptive_crossinterpolate2::<f64, _, _>(
///     evaluate,
///     vec![2, 2],
///     graph,
///     vec![vec![0, 0], vec![1, 1]],
///     options,
///     None,
///     &DefaultProposer,
/// )
/// .unwrap();
///
/// // The identity splits into two rank-one patches fixing site 0.
/// assert_eq!(result.patches.len(), 2);
/// let fixed0: Vec<usize> = result
///     .patches
///     .iter()
///     .map(|p| p.fixed_values[0].unwrap())
///     .collect();
/// assert_eq!(fixed0, vec![0, 1]);
/// assert!(result
///     .patches
///     .iter()
///     .all(|p| p.final_error <= 1e-10));
/// ```
#[allow(clippy::too_many_arguments)]
pub fn adaptive_crossinterpolate2<T, F, P>(
    evaluate: F,
    local_dims: Vec<usize>,
    graph: TreeTciGraph,
    initial_pivots: Vec<MultiIndex>,
    options: AdaptiveTreeTciOptions,
    center_site: Option<usize>,
    proposer: &P,
) -> Result<AdaptiveTreeTciResult>
where
    T: FullPivLuScalar + CommonScalar,
    F: Fn(GlobalIndexBatch<'_>) -> Result<Vec<T>>,
    P: PivotCandidateProposer,
{
    validate_inputs(&local_dims, &graph, &initial_pivots, &options, center_site)?;

    let n_sites = local_dims.len();
    let patch_order = if options.patch_order.is_empty() {
        (0..n_sites).collect::<Vec<_>>()
    } else {
        options.patch_order.clone()
    };

    let mut pending: VecDeque<Vec<Option<usize>>> = VecDeque::from([vec![None; n_sites]]);
    let mut patches: Vec<AdaptiveTreeTciPatch> = Vec::new();

    while let Some(fixed_values) = pending.pop_front() {
        let mut rng = StdRng::seed_from_u64(seed_for_patch(options.seed, &fixed_values));
        let candidate_pivots = patch_candidates(
            &local_dims,
            &fixed_values,
            &initial_pivots,
            options.n_initial_pivots,
            &mut rng,
        )?;

        let candidate_values = eval_patch_pivots(&evaluate, &fixed_values, &candidate_pivots)?;
        if candidate_values
            .iter()
            .all(|value| CommonScalar::abs_val(*value) < ZERO_SAMPLE_THRESHOLD)
        {
            // Numerically zero patch: omit it (absent patch == zero).
            continue;
        }

        let patch_dims: Vec<usize> = (0..n_sites)
            .map(|site| fixed_values[site].map_or(local_dims[site], |_| 1))
            .collect();

        let local_evaluate = |batch: GlobalIndexBatch<'_>| -> Result<Vec<T>> {
            let full = expand_local_batch(&fixed_values, batch)?;
            evaluate(GlobalIndexBatch::new(
                &full,
                batch.n_sites(),
                batch.n_points(),
            )?)
        };

        let (treetn, ranks, errors) = crossinterpolate2::<T, _, P>(
            local_evaluate,
            patch_dims,
            graph.clone(),
            candidate_pivots,
            options.tci_options.clone(),
            center_site,
            proposer,
        )?;

        let final_error = errors.last().copied().unwrap_or(f64::INFINITY);
        let max_rank = ranks.last().copied().unwrap_or(0);

        if final_error <= options.tci_options.tolerance {
            patches.push(AdaptiveTreeTciPatch {
                fixed_values,
                treetn,
                final_error,
                max_rank,
            });
            continue;
        }

        let split_site = patch_order
            .iter()
            .copied()
            .find(|&site| fixed_values[site].is_none())
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "nonconverged patch has no remaining site to split: \
                     fixed_values={:?}, final_error={}, tolerance={}, max_rank={}",
                    fixed_values,
                    final_error,
                    options.tci_options.tolerance,
                    max_rank
                )
            })?;

        for value in 0..local_dims[split_site] {
            let mut child = fixed_values.clone();
            child[split_site] = Some(value);
            pending.push_back(child);
        }
    }

    Ok(AdaptiveTreeTciResult {
        local_dims,
        patches,
    })
}

fn validate_inputs(
    local_dims: &[usize],
    graph: &TreeTciGraph,
    initial_pivots: &[MultiIndex],
    options: &AdaptiveTreeTciOptions,
    center_site: Option<usize>,
) -> Result<()> {
    ensure!(!local_dims.is_empty(), "local_dims must not be empty");
    ensure!(
        local_dims.iter().all(|&dim| dim > 0),
        "every local dimension must be positive"
    );
    ensure!(
        local_dims.len() == graph.n_sites(),
        "local_dims length {} must match graph site count {}",
        local_dims.len(),
        graph.n_sites()
    );
    ensure!(
        options.n_initial_pivots > 0,
        "n_initial_pivots must be positive"
    );
    let tolerance = options.tci_options.tolerance;
    ensure!(
        tolerance.is_finite() && tolerance >= 0.0,
        "TreeTCI tolerance must be finite and nonnegative, got {}",
        tolerance
    );
    ensure!(
        options.tci_options.max_iter > 0,
        "TreeTCI max_iter must be positive"
    );
    ensure!(
        options.tci_options.max_bond_dim > 0,
        "TreeTCI max_bond_dim must be positive"
    );

    for pivot in initial_pivots {
        ensure!(
            pivot.len() == local_dims.len(),
            "every initial pivot must have one coordinate per site, got length {}",
            pivot.len()
        );
        ensure!(
            pivot
                .iter()
                .zip(local_dims)
                .all(|(&value, &dim)| value < dim),
            "an initial pivot coordinate is outside its site dimension"
        );
    }

    if !options.patch_order.is_empty() {
        let n = local_dims.len();
        ensure!(
            options.patch_order.len() == n,
            "patch_order must be an exact permutation of all site positions, got length {}",
            options.patch_order.len()
        );
        let mut sorted = options.patch_order.clone();
        sorted.sort_unstable();
        let expected = (0..n).collect::<Vec<_>>();
        ensure!(
            sorted == expected,
            "patch_order must be an exact permutation of all site positions"
        );
    }

    if let Some(site) = center_site {
        ensure!(
            site < local_dims.len(),
            "center_site {} is out of range for {} sites",
            site,
            local_dims.len()
        );
    }

    Ok(())
}

/// Derive a per-patch random seed from the base seed and the fixed assignment.
///
/// Hashing the assignment makes generated pivots deterministic and independent
/// of the order in which patches are dequeued.
fn seed_for_patch(seed: u64, fixed_values: &[Option<usize>]) -> u64 {
    let mut hasher = DefaultHasher::new();
    seed.hash(&mut hasher);
    for entry in fixed_values {
        match entry {
            None => 0u64.hash(&mut hasher),
            Some(value) => {
                1u64.hash(&mut hasher);
                value.hash(&mut hasher);
            }
        }
    }
    hasher.finish()
}

/// Build the initial patch-local pivot candidates for one patch.
///
/// Keeps compatible input pivots (projected to patch-local coordinates), then
/// fills with seeded random pivots, and finally enumerates any remaining points
/// in column-major order. Deduplicates throughout.
fn patch_candidates(
    local_dims: &[usize],
    fixed_values: &[Option<usize>],
    initial_pivots: &[MultiIndex],
    target: usize,
    rng: &mut StdRng,
) -> Result<Vec<MultiIndex>> {
    let n_sites = local_dims.len();
    let patch_dims: Vec<usize> = (0..n_sites)
        .map(|site| fixed_values[site].map_or(local_dims[site], |_| 1))
        .collect();

    let mut candidates: Vec<MultiIndex> = Vec::new();
    let mut seen: HashSet<MultiIndex> = HashSet::new();

    for full_pivot in initial_pivots {
        if is_compatible(full_pivot, fixed_values) {
            let local: MultiIndex = (0..n_sites)
                .map(|site| match fixed_values[site] {
                    Some(_) => 0,
                    None => full_pivot[site],
                })
                .collect();
            if seen.insert(local.clone()) {
                candidates.push(local);
            }
        }
    }

    let point_count = patch_dims
        .iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
        .ok_or_else(|| anyhow::anyhow!("patch point count overflows usize"))?;
    let desired = target.max(candidates.len()).min(point_count);

    let random_attempts = desired.saturating_mul(20).saturating_add(100);
    for _ in 0..random_attempts {
        if candidates.len() >= desired {
            break;
        }
        let pivot: MultiIndex = (0..n_sites)
            .map(|site| rng.random_range(0..patch_dims[site]))
            .collect();
        if seen.insert(pivot.clone()) {
            candidates.push(pivot);
        }
    }
    for flat in 0..point_count {
        if candidates.len() >= desired {
            break;
        }
        let pivot = decode_col_major(flat, &patch_dims);
        if seen.insert(pivot.clone()) {
            candidates.push(pivot);
        }
    }

    Ok(candidates)
}

fn is_compatible(full_pivot: &MultiIndex, fixed_values: &[Option<usize>]) -> bool {
    full_pivot.len() == fixed_values.len()
        && fixed_values
            .iter()
            .enumerate()
            .all(|(site, fixed)| fixed.is_none_or(|value| full_pivot[site] == value))
}

/// Evaluate patch-local pivots against the source function.
///
/// Converts each patch-local pivot to a full-domain point (fixed sites take
/// their fixed value, active sites pass through) and runs the original
/// evaluator. Uses [`expand_local_batch`] for the conversion, the same helper
/// the wrapped TreeTCI evaluator uses.
fn eval_patch_pivots<T, F>(
    evaluate: &F,
    fixed_values: &[Option<usize>],
    pivots: &[MultiIndex],
) -> Result<Vec<T>>
where
    F: Fn(GlobalIndexBatch<'_>) -> Result<Vec<T>>,
{
    if pivots.is_empty() {
        return Ok(Vec::new());
    }
    let n_sites = fixed_values.len();
    let flat: Vec<usize> = pivots.iter().flat_map(|p| p.iter().copied()).collect();
    let local_batch = GlobalIndexBatch::new(&flat, n_sites, pivots.len())?;
    let full = expand_local_batch(fixed_values, local_batch)?;
    evaluate(GlobalIndexBatch::new(&full, n_sites, pivots.len())?)
}

/// Convert a patch-local batch to full-domain column-major coordinates.
///
/// For each point: active sites pass their local coordinate through unchanged;
/// fixed sites must carry local coordinate `0` and are replaced with the fixed
/// original-domain value. The output uses the same column-major
/// `(n_sites, n_points)` layout as the input.
fn expand_local_batch(
    fixed_values: &[Option<usize>],
    batch: GlobalIndexBatch<'_>,
) -> Result<Vec<usize>> {
    let n_sites = batch.n_sites();
    let n_points = batch.n_points();
    ensure!(
        n_sites == fixed_values.len(),
        "patch-local batch has {} sites but patch fixes {} sites",
        n_sites,
        fixed_values.len()
    );
    let mut full = Vec::with_capacity(n_sites * n_points);
    for point in 0..n_points {
        for (site, fixed) in fixed_values.iter().enumerate() {
            let local = batch.get(site, point).ok_or_else(|| {
                anyhow::anyhow!("missing coordinate at site {} point {}", site, point)
            })?;
            match fixed {
                Some(fixed_value) => {
                    ensure!(
                        local == 0,
                        "fixed site {} received local coordinate {} (expected 0)",
                        site,
                        local
                    );
                    full.push(*fixed_value);
                }
                None => full.push(local),
            }
        }
    }
    Ok(full)
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

#[cfg(test)]
mod tests;
