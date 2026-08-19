//! Fit algorithm for TreeTN contraction.
//!
//! This module provides the fit (variational) algorithm for contracting two TreeTNs.
//! The algorithm iteratively optimizes `C ≈ A * B` by minimizing `||A*B - C||²`.
//!
//! # Algorithm Overview
//!
//! 1. Prepare input TNs with `sim_internal_inds()` to avoid index collision
//! 2. Initialize C (using zipup result, or a caller-provided [`contract_fit_from_initial`]
//!    initial state)
//! 3. For each sweep:
//!    a. Compute/update environment tensors
//!    b. For each 2-site step:
//!       - Extract local tensors from A, B, C
//!       - Compute optimal local C tensor: L × A[i] × B[i] × A[j] × B[j] × R
//!       - Factorize and update C
//!       - Update environment cache
//!
//! # Environment Tensors
//!
//! For each edge (from, to), the environment `env[(from, to)]` represents:
//! - The contraction of the "from" side subtree of A×B with conj(C)
//! - Shape: (link_A, link_B, link_C) pointing towards "to"
//!
//! The A- and B-link indices stay **separate** inside the environment; the
//! algorithm never fuses them into a persistent output bond of dimension
//! `χ_A·χ_B`. The only place the full `χ_A·χ_B` product is materialized is
//! inside the zip-up *initializer* (see below).
//!
//! # Initialization and rank growth
//!
//! Initialization is decoupled from the sweeps:
//!
//! - [`contract_fit`] initializes `C` with a topology-preserving zip-up
//!   contraction of `A·B`. Without a truncation policy this constructs the
//!   exact product, whose bond dimension can reach `χ_A·χ_B` per edge — fine
//!   for small systems, wasteful for large QTT/MPO bonds.
//! - [`contract_fit_from_initial`] optimizes from a caller-supplied state. The
//!   typical adaptive starting state is [`low_rank_initializer_tree_tn`], which
//!   creates random tensors carrying the surviving output site indices with
//!   every bond at a small dimension (usually 1). It never forms `χ_A·χ_B`.
//!
//! Rank growth during sweeps follows [`FitContractionOptions::max_bond_dim`]:
//!
//! * `Some(cap)` → every sweep bond is capped at `cap`;
//! * `None` plus an SVD truncation policy → bonds grow adaptively up to
//!   whatever the tolerance demands (no user-supplied rank cap required);
//! * `None` and no policy → the existing bond dimensions are preserved, so the
//!   initializer decides the starting rank.
//!
//! # References
//!
//! - T4AMPOContractions.jl: `contract_fit`, `leftenvironment!`, `rightenvironment!`
//! - ITensorMPS.jl: `contract`/`apply` with `alg="fit"`, `init` and `cutoff`
//! - ITensorNetworks.jl: `contract` with fitting algorithm

use crate::error::TreeTNOperationError;
use std::collections::{HashMap, HashSet};
use std::hash::Hash;
use std::time::{Duration, Instant};

use rand::rngs::StdRng;
use rand::SeedableRng;

use num_complex::Complex64;
use tensor4all_core::{DynIndex, IdxTensor};

use anyhow::Result;

use tensor4all_core::{
    print_and_reset_contract_profile, print_and_reset_native_einsum_profile,
    reset_contract_profile, reset_native_einsum_profile, sort_indices_deterministic, Canonical,
    FactorizeAlg, FactorizeOptions, FactorizeResult, IndexLike, SvdTruncationPolicy, TensorLike,
};

use super::localupdate::{LocalUpdateStep, LocalUpdateSweepPlan, LocalUpdater};
use super::TreeTN;

#[cfg(test)]
use std::cell::Cell;

#[derive(Debug, Default, Clone)]
struct FitProfile {
    zipup_init_time: Duration,
    canonicalize_time: Duration,
    sweep_time: Duration,
    env_get_time: Duration,
    env_leaf_time: Duration,
    env_internal_time: Duration,
    left_inds_time: Duration,
    two_site_contract_time: Duration,
    factorize_time: Duration,
    replace_time: Duration,
    invalidate_time: Duration,
    env_requests: usize,
    env_hits: usize,
    env_misses: usize,
    step_count: usize,
    sweep_count: usize,
}

thread_local! {
    static FIT_PROFILE_STATE: std::cell::RefCell<Option<FitProfile>> =
        const { std::cell::RefCell::new(None) };
}

#[cfg(test)]
thread_local! {
    static FORCE_FIT_PROFILE: Cell<bool> = const { Cell::new(false) };
}

fn fit_profile_enabled() -> bool {
    #[cfg(test)]
    if FORCE_FIT_PROFILE.with(Cell::get) {
        return true;
    }
    std::env::var("T4A_PROFILE_FIT").is_ok()
}

#[cfg(test)]
fn set_fit_profile_enabled_for_tests(enabled: bool) {
    FORCE_FIT_PROFILE.with(|slot| slot.set(enabled));
}

fn fit_profile_reset() {
    if fit_profile_enabled() {
        FIT_PROFILE_STATE.with(|state| {
            *state.borrow_mut() = Some(FitProfile::default());
        });
    }
}

fn with_fit_profile(f: impl FnOnce(&mut FitProfile)) {
    if fit_profile_enabled() {
        FIT_PROFILE_STATE.with(|state| {
            if let Some(profile) = state.borrow_mut().as_mut() {
                f(profile);
            }
        });
    }
}

fn take_fit_profile() -> Option<FitProfile> {
    FIT_PROFILE_STATE.with(|state| state.borrow_mut().take())
}

fn sorted_neighbors_by_node_index<T, V>(
    tn: &TreeTN<T, V>,
    node: &V,
    excluded: Option<&V>,
    context: &str,
) -> Result<Vec<V>>
where
    T: TensorLike,
    V: Clone + Hash + Eq + Send + Sync + std::fmt::Debug,
{
    let mut neighbors = Vec::new();
    for neighbor in tn.site_index_network().neighbors(node) {
        if excluded == Some(&neighbor) {
            continue;
        }
        let node_idx = tn.node_index(&neighbor).ok_or_else(|| {
            anyhow::anyhow!(
                "{context}: neighbor {:?} of node {:?} is missing from TreeTN node map",
                neighbor,
                node
            )
        })?;
        neighbors.push((node_idx.index(), neighbor));
    }
    neighbors.sort_by_key(|(index, _node)| *index);
    Ok(neighbors.into_iter().map(|(_index, node)| node).collect())
}

fn tensor_at_node<'a, T, V>(tn: &'a TreeTN<T, V>, node: &V, tree_name: &str) -> Result<&'a T>
where
    T: TensorLike,
    V: Clone + Hash + Eq + Send + Sync + std::fmt::Debug,
{
    let node_idx = tn
        .node_index(node)
        .ok_or_else(|| anyhow::anyhow!("Node {:?} not found in {}", node, tree_name))?;
    tn.tensor(node_idx)
        .ok_or_else(|| anyhow::anyhow!("Tensor for node {:?} not found in {}", node, tree_name))
}

fn tensors_share_contractable_index<T>(left: &T, right: &T) -> bool
where
    T: TensorLike,
{
    let left_indices = left.external_indices();
    let right_indices = right.external_indices();
    left_indices.iter().any(|left_index| {
        right_indices
            .iter()
            .any(|right_index| left_index.is_contractable(right_index))
    })
}

fn tensor_connected_components<T>(tensors: &[&T]) -> Vec<Vec<usize>>
where
    T: TensorLike,
{
    let mut visited = vec![false; tensors.len()];
    let mut components = Vec::new();

    for start in 0..tensors.len() {
        if visited[start] {
            continue;
        }

        let mut component = Vec::new();
        let mut stack = vec![start];
        visited[start] = true;

        while let Some(current) = stack.pop() {
            component.push(current);
            for candidate in 0..tensors.len() {
                if visited[candidate] {
                    continue;
                }
                if tensors_share_contractable_index(tensors[current], tensors[candidate]) {
                    visited[candidate] = true;
                    stack.push(candidate);
                }
            }
        }

        component.sort_unstable();
        components.push(component);
    }

    components
}

fn contract_fit_tensor_refs<T>(tensors: &[&T]) -> Result<T>
where
    T: TensorLike,
    <T::Index as IndexLike>::Id: Clone + std::hash::Hash + Eq + Ord + std::fmt::Debug + Send + Sync,
{
    if tensors.is_empty() {
        return Err(anyhow::anyhow!(
            "fit contraction requires at least one tensor"
        ));
    }

    let components = tensor_connected_components(tensors);
    if components.len() == 1 {
        return T::contract(tensors).map_err(|e| anyhow::anyhow!("contract failed: {}", e));
    }

    let mut contracted_components = Vec::with_capacity(components.len());
    for component in components {
        let component_refs = component
            .iter()
            .map(|&tensor_index| tensors[tensor_index])
            .collect::<Vec<_>>();
        let contracted = if component_refs.len() == 1 {
            component_refs[0].clone()
        } else {
            T::contract(&component_refs)
                .map_err(|e| anyhow::anyhow!("component contract failed: {}", e))?
        };
        contracted_components.push(contracted);
    }

    let mut iter = contracted_components.into_iter();
    let mut result = iter
        .next()
        .ok_or_else(|| anyhow::anyhow!("fit contraction produced no components"))?;
    for component in iter {
        result = result
            .outer_product(&component)
            .map_err(|e| anyhow::anyhow!("component outer product failed: {}", e))?;
    }

    Ok(result)
}

// ============================================================================
// FitEnvironment: Environment tensor cache
// ============================================================================

/// Environment tensor cache for fit algorithm.
///
/// Stores environment tensors for each directed edge (from, to).
/// The environment `env[(from, to)]` represents the contraction of the
/// "from" side subtree (A×B contracted with conj(C)).
///
/// # Lazy Evaluation
///
/// The cache starts empty. When an environment is requested via `get_or_compute`,
/// it is computed recursively from the leaves and cached for future use.
///
/// # Cache Invalidation
///
/// When tensors in a region T are updated, all caches containing those tensors
/// must be invalidated. The invalidation propagates recursively from T towards
/// the leaves of the tree.
#[derive(Debug, Clone)]
pub struct FitEnvironment<T, V>
where
    T: TensorLike,
    V: Clone + Hash + Eq,
{
    /// Environment tensors: (from, to) -> tensor
    envs: HashMap<(V, V), T>,
}

impl<T, V> FitEnvironment<T, V>
where
    T: TensorLike,
    <T::Index as IndexLike>::Id: Clone + std::hash::Hash + Eq + std::fmt::Debug,
    V: Clone + Hash + Eq + Send + Sync + std::fmt::Debug,
{
    /// Create an empty environment cache.
    pub fn new() -> Self {
        Self {
            envs: HashMap::new(),
        }
    }

    /// Get the environment tensor for edge (from, to) if it exists.
    pub fn get(&self, from: &V, to: &V) -> Option<&T> {
        self.envs.get(&(from.clone(), to.clone()))
    }

    /// Insert an environment tensor for edge (from, to).
    /// This is mainly for testing; normally use `get_or_compute` for lazy evaluation.
    #[allow(dead_code)]
    pub(crate) fn insert(&mut self, from: V, to: V, env: T) {
        self.envs.insert((from, to), env);
    }

    /// Check if environment exists for edge (from, to).
    pub fn contains(&self, from: &V, to: &V) -> bool {
        self.envs.contains_key(&(from.clone(), to.clone()))
    }

    /// Get the number of cached environments.
    pub fn len(&self) -> usize {
        self.envs.len()
    }

    /// Check if the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.envs.is_empty()
    }

    /// Clear all cached environments.
    pub fn clear(&mut self) {
        self.envs.clear();
    }

    /// Get or compute the environment tensor for edge (from, to).
    ///
    /// If the environment is cached, returns it directly.
    /// Otherwise, recursively computes it from child environments (towards leaves)
    /// and caches the result.
    ///
    /// # Arguments
    /// * `from` - The node whose subtree we're computing
    /// * `to` - The direction we're looking towards
    /// * `tn_a` - First input TreeTN
    /// * `tn_b` - Second input TreeTN
    /// * `tn_c` - Current approximation TreeTN
    /// # Errors
    ///
    /// Returns an error when the cached fit value cannot be computed (a shape or
    /// /// index mismatch, or a backend failure).
    ///
    pub fn get_or_compute(
        &mut self,
        from: &V,
        to: &V,
        tn_a: &TreeTN<T, V>,
        tn_b: &TreeTN<T, V>,
        tn_c: &TreeTN<T, V>,
    ) -> std::result::Result<T, TreeTNOperationError>
    where
        <T::Index as IndexLike>::Id:
            Clone + std::hash::Hash + Eq + Ord + std::fmt::Debug + Send + Sync,
        V: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug,
    {
        let started = fit_profile_enabled().then(Instant::now);
        with_fit_profile(|profile| {
            profile.env_requests += 1;
        });

        // If already cached, return a clone
        if let Some(env) = self.envs.get(&(from.clone(), to.clone())) {
            if let Some(started) = started {
                with_fit_profile(|profile| {
                    profile.env_hits += 1;
                    profile.env_get_time += started.elapsed();
                });
            }
            return Ok(env.clone());
        }

        with_fit_profile(|profile| {
            profile.env_misses += 1;
        });

        // Get neighbors of `from` excluding `to`
        let child_neighbors =
            sorted_neighbors_by_node_index(tn_c, from, Some(to), "FitEnvironment::get_or_compute")?;

        // Recursively get or compute child environments
        let child_envs: Vec<T> = child_neighbors
            .iter()
            .map(|child| self.get_or_compute(child, from, tn_a, tn_b, tn_c))
            .collect::<std::result::Result<Vec<_>, TreeTNOperationError>>()?;

        // Compute the environment for (from, to) using child environments
        let env = compute_single_node_environment(from, to, tn_a, tn_b, tn_c, &child_envs)?;

        // Cache and return
        self.envs.insert((from.clone(), to.clone()), env.clone());
        if let Some(started) = started {
            with_fit_profile(|profile| {
                profile.env_get_time += started.elapsed();
            });
        }
        Ok(env)
    }

    /// Invalidate all caches affected by updates to tensors in region T.
    ///
    /// For each `t ∈ T`:
    /// 1. Remove all `env[(t, *)]` (0th generation)
    /// 2. Recursively remove caches propagating towards leaves
    ///
    /// # Arguments
    /// * `region` - The set of nodes whose tensors were updated
    /// * `tn_c` - The TreeTN (for topology information)
    pub fn invalidate<'a>(&mut self, region: impl IntoIterator<Item = &'a V>, tn_c: &TreeTN<T, V>)
    where
        V: 'a + Send + Sync,
    {
        let _ = self.try_invalidate(region, tn_c);
    }

    fn try_invalidate<'a>(
        &mut self,
        region: impl IntoIterator<Item = &'a V>,
        tn_c: &TreeTN<T, V>,
    ) -> Result<()>
    where
        V: 'a + Send + Sync,
    {
        for t in region {
            // Get all neighbors of t
            let neighbors =
                sorted_neighbors_by_node_index(tn_c, t, None, "FitEnvironment::invalidate")?;

            // Remove all env[(t, *)] and propagate recursively
            for neighbor in neighbors {
                self.invalidate_recursive(t, &neighbor, tn_c)?;
            }
        }
        Ok(())
    }

    /// Recursively invalidate caches starting from env[(from, to)] towards leaves.
    ///
    /// If env[(from, to)] exists, remove it and propagate to env[(to, x)] for all x ≠ from.
    fn invalidate_recursive(&mut self, from: &V, to: &V, tn_c: &TreeTN<T, V>) -> Result<()> {
        // Remove env[(from, to)] if it exists
        if self.envs.remove(&(from.clone(), to.clone())).is_some() {
            // Propagate to next generation: env[(to, x)] for all neighbors x of to, x ≠ from
            let neighbors = sorted_neighbors_by_node_index(
                tn_c,
                to,
                Some(from),
                "FitEnvironment::invalidate_recursive",
            )?;

            for neighbor in neighbors {
                self.invalidate_recursive(to, &neighbor, tn_c)?;
            }
        }
        Ok(())
    }

    /// Verify cache structural consistency.
    ///
    /// For any `env[(x, x1)]` where `x` is not a leaf (has neighbors other than `x1`),
    /// all child environments `env[(y, x)]` for neighbors `y ≠ x1` must exist.
    ///
    /// # Arguments
    /// * `tn_c` - The TreeTN (for topology information)
    ///
    /// # Returns
    /// `Ok(())` if consistent, or an error describing the inconsistency.
    /// # Errors
    ///
    /// Returns an error when the fitted structure is inconsistent (a graph
    /// /// consistency failure).
    ///
    pub fn verify_structural_consistency(
        &self,
        tn_c: &TreeTN<T, V>,
    ) -> std::result::Result<(), TreeTNOperationError>
    where
        V: Clone + Hash + Eq + std::fmt::Debug,
    {
        for (from, to) in self.envs.keys() {
            // Get neighbors of `from` excluding `to`
            let child_neighbors = sorted_neighbors_by_node_index(
                tn_c,
                from,
                Some(to),
                "FitEnvironment::verify_structural_consistency",
            )?;

            // If `from` is not a leaf, all child environments must exist
            for child in &child_neighbors {
                if !self.envs.contains_key(&(child.clone(), from.clone())) {
                    return Err(anyhow::anyhow!(
                        "Structural inconsistency: env[({:?}, {:?})] exists but child env[({:?}, {:?})] is missing",
                        from, to, child, from
                    ).into());
                }
            }
        }
        Ok(())
    }
}

impl<T, V> Default for FitEnvironment<T, V>
where
    T: TensorLike,
    <T::Index as IndexLike>::Id: Clone + std::hash::Hash + Eq + std::fmt::Debug,
    V: Clone + Hash + Eq + Send + Sync + std::fmt::Debug,
{
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Environment computation helpers
// ============================================================================

/// Compute environment for a leaf node (no children in subtree).
fn compute_leaf_environment<T, V>(
    node: &V,
    _towards: &V,
    tn_a: &TreeTN<T, V>,
    tn_b: &TreeTN<T, V>,
    tn_c: &TreeTN<T, V>,
) -> Result<T>
where
    T: TensorLike,
    <T::Index as IndexLike>::Id: Clone + std::hash::Hash + Eq + Ord + std::fmt::Debug + Send + Sync,
    V: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug,
{
    let started = fit_profile_enabled().then(Instant::now);

    // Get tensors
    let tensor_a = tensor_at_node(tn_a, node, "tn_a")?;
    let tensor_b = tensor_at_node(tn_b, node, "tn_b")?;
    let tensor_c = tensor_at_node(tn_c, node, "tn_c")?;

    // A, B, and C must form one connected local environment.
    let c_conj = tensor_c.conj();
    let env = contract_fit_tensor_refs(&[tensor_a, tensor_b, &c_conj])?;

    if let Some(started) = started {
        with_fit_profile(|profile| {
            profile.env_leaf_time += started.elapsed();
        });
    }

    Ok(env)
}

/// Compute environment for a single node using child environments.
///
/// This computes: child_envs × A[node] × B[node] × conj(C[node])
/// leaving open only the indices connecting to `towards`.
fn compute_single_node_environment<T, V>(
    node: &V,
    towards: &V,
    tn_a: &TreeTN<T, V>,
    tn_b: &TreeTN<T, V>,
    tn_c: &TreeTN<T, V>,
    child_envs: &[T],
) -> Result<T>
where
    T: TensorLike,
    <T::Index as IndexLike>::Id: Clone + std::hash::Hash + Eq + Ord + std::fmt::Debug + Send + Sync,
    V: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug,
{
    let started = fit_profile_enabled().then(Instant::now);

    // Get local tensors
    let tensor_a = tensor_at_node(tn_a, node, "tn_a")?;
    let tensor_b = tensor_at_node(tn_b, node, "tn_b")?;
    let tensor_c = tensor_at_node(tn_c, node, "tn_c")?;

    if child_envs.is_empty() {
        // Leaf node: compute from tensors directly
        return compute_leaf_environment(node, towards, tn_a, tn_b, tn_c);
    }

    // Non-leaf: all local tensors and child environments must form one
    // connected contraction graph.
    let c_conj = tensor_c.conj();
    let mut tensor_refs: Vec<&T> = vec![tensor_a, tensor_b, &c_conj];
    tensor_refs.extend(child_envs.iter());
    let result = contract_fit_tensor_refs(&tensor_refs)?;

    if let Some(started) = started {
        with_fit_profile(|profile| {
            profile.env_internal_time += started.elapsed();
        });
    }

    Ok(result)
}

// ============================================================================
// FitUpdater: LocalUpdater implementation for fit algorithm
// ============================================================================

/// Fit updater for variational contraction.
///
/// Implements the `LocalUpdater` trait to perform 2-site updates
/// that optimize `C ≈ A * B`.
#[derive(Debug)]
pub struct FitUpdater<T, V>
where
    T: TensorLike,
    V: Clone + Hash + Eq + Send + Sync + std::fmt::Debug,
{
    /// First input TreeTN (with sim'd internal indices)
    pub tn_a: TreeTN<T, V>,
    /// Second input TreeTN (with sim'd internal indices)
    pub tn_b: TreeTN<T, V>,
    /// Environment cache
    pub envs: FitEnvironment<T, V>,
    /// Maximum bond dimension
    pub max_bond_dim: Option<usize>,
    /// Legacy relative tolerance retained for same-crate tests and call chains.
    pub(crate) rtol: Option<f64>,
    /// Explicit SVD truncation policy
    pub svd_policy: Option<SvdTruncationPolicy>,
    /// QR-specific relative tolerance
    pub qr_rtol: Option<f64>,
    /// Factorization algorithm
    pub factorize_alg: FactorizeAlg,
}

impl<T, V> FitUpdater<T, V>
where
    T: TensorLike,
    <T::Index as IndexLike>::Id: Clone + std::hash::Hash + Eq + std::fmt::Debug + Send + Sync,
    V: Clone + Hash + Eq + Send + Sync + std::fmt::Debug,
{
    /// Create a new FitUpdater.
    ///
    /// # Arguments
    /// * `tn_a` - First input TreeTN
    /// * `tn_b` - Second input TreeTN
    /// * `max_bond_dim` - Maximum bond dimension for truncation
    /// * `svd_policy` - Explicit SVD truncation policy
    /// * `qr_rtol` - QR-specific relative tolerance
    ///
    /// Note: sim_internal_inds() should be called on tn_a and tn_b before passing
    /// if index collision is a concern. This is not done here because contraction
    /// module (which provides sim_internal_inds) is currently disabled.
    pub fn new(
        tn_a: TreeTN<T, V>,
        tn_b: TreeTN<T, V>,
        max_bond_dim: Option<usize>,
        rtol: Option<f64>,
    ) -> Self {
        Self {
            tn_a,
            tn_b,
            envs: FitEnvironment::new(),
            max_bond_dim,
            rtol,
            svd_policy: rtol.map(SvdTruncationPolicy::new),
            qr_rtol: None,
            factorize_alg: FactorizeAlg::SVD,
        }
    }

    /// Set the factorization algorithm.
    pub fn with_factorize_alg(mut self, alg: FactorizeAlg) -> Self {
        self.factorize_alg = alg;
        self
    }

    /// Set the SVD truncation policy used by fit sweeps.
    pub(crate) fn with_svd_policy(mut self, policy: Option<SvdTruncationPolicy>) -> Self {
        self.rtol = policy.map(|value| value.threshold);
        self.svd_policy = policy;
        self
    }

    /// Set the QR-specific relative tolerance used by fit sweeps.
    pub(crate) fn with_qr_rtol(mut self, qr_rtol: Option<f64>) -> Self {
        self.qr_rtol = qr_rtol;
        self
    }
}

impl<T, V> LocalUpdater<T, V> for FitUpdater<T, V>
where
    T: TensorLike,
    <T::Index as IndexLike>::Id: Clone + std::hash::Hash + Eq + Ord + std::fmt::Debug + Send + Sync,
    V: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug,
{
    fn update(
        &mut self,
        mut subtree: TreeTN<T, V>,
        step: &LocalUpdateStep<V>,
        full_treetn: &TreeTN<T, V>,
    ) -> std::result::Result<TreeTN<T, V>, TreeTNOperationError> {
        // FitUpdater is designed for nsite=2
        if step.nodes.len() != 2 {
            return Err(anyhow::anyhow!(
                "FitUpdater requires exactly 2 nodes, got {}",
                step.nodes.len()
            )
            .into());
        }

        let node_u = &step.nodes[0];
        let node_v = &step.nodes[1];
        with_fit_profile(|profile| {
            profile.step_count += 1;
        });

        // Get tensors
        let a_u = tensor_at_node(&self.tn_a, node_u, "tn_a")?;
        let a_v = tensor_at_node(&self.tn_a, node_v, "tn_a")?;
        let b_u = tensor_at_node(&self.tn_b, node_u, "tn_b")?;
        let b_v = tensor_at_node(&self.tn_b, node_v, "tn_b")?;

        if full_treetn.node_index(node_u).is_none() {
            return Err(anyhow::anyhow!("Node {:?} not found in full TreeTN", node_u).into());
        }
        if full_treetn.node_index(node_v).is_none() {
            return Err(anyhow::anyhow!("Node {:?} not found in full TreeTN", node_v).into());
        }
        if full_treetn.edge_between(node_u, node_v).is_none() {
            return Err(anyhow::anyhow!(
                "FitUpdater update step nodes {:?} and {:?} are not adjacent in full TreeTN",
                node_u,
                node_v
            )
            .into());
        }

        // Collect environments from neighbors (excluding the edge between u and v)
        // Uses lazy evaluation: computes and caches if not already present
        let mut env_tensors: Vec<T> = Vec::new();

        // Environments from u's neighbors (except v)
        let u_neighbors =
            sorted_neighbors_by_node_index(full_treetn, node_u, None, "FitUpdater::update")?;
        let mut left_bond_indices = Vec::new();
        for neighbor in &u_neighbors {
            if neighbor == node_v {
                continue;
            }
            let edge = full_treetn.edge_between(node_u, neighbor).ok_or_else(|| {
                anyhow::anyhow!(
                    "FitUpdater: missing edge between {:?} and {:?} in full TreeTN",
                    node_u,
                    neighbor
                )
            })?;
            let bond = full_treetn.bond_index(edge).ok_or_else(|| {
                anyhow::anyhow!(
                    "FitUpdater: missing bond index for edge between {:?} and {:?}",
                    node_u,
                    neighbor
                )
            })?;
            left_bond_indices.push(bond.clone());
            let env =
                self.envs
                    .get_or_compute(neighbor, node_u, &self.tn_a, &self.tn_b, full_treetn)?;
            env_tensors.push(env);
        }

        // Environments from v's neighbors (except u)
        let v_neighbors =
            sorted_neighbors_by_node_index(full_treetn, node_v, None, "FitUpdater::update")?;
        for neighbor in &v_neighbors {
            if neighbor == node_u {
                continue;
            }
            let env =
                self.envs
                    .get_or_compute(neighbor, node_v, &self.tn_a, &self.tn_b, full_treetn)?;
            env_tensors.push(env);
        }

        // Compute optimal 2-site tensor: env × A[u] × B[u] × A[v] × B[v] × env.
        // Collect all tensors and let contract() find the optimal contraction order.
        let contract_started = fit_profile_enabled().then(Instant::now);
        let mut tensor_refs: Vec<&T> = vec![a_u, b_u, a_v, b_v];
        tensor_refs.extend(env_tensors.iter());
        let ab_uv = contract_fit_tensor_refs(&tensor_refs)?;
        if let Some(contract_started) = contract_started {
            with_fit_profile(|profile| {
                profile.two_site_contract_time += contract_started.elapsed();
            });
        }

        // The result ab_uv is the optimal 2-site tensor
        // Factorize to get new C[u] and C[v]

        // Determine left indices (indices that will remain on u after factorization)
        let left_inds_started = fit_profile_enabled().then(Instant::now);
        let site_c_u = full_treetn.site_space(node_u).cloned().ok_or_else(|| {
            anyhow::anyhow!(
                "FitUpdater: site space for node {:?} not found in full TreeTN",
                node_u
            )
        })?;
        let ab_uv_indices = ab_uv.external_indices();
        let mut left_inds: Vec<_> = ab_uv_indices
            .iter()
            .filter(|idx| {
                // Keep site indices of u and link indices to u's other neighbors
                site_c_u.contains(*idx) || left_bond_indices.iter().any(|bond| bond == *idx)
            })
            .cloned()
            .collect();
        sort_indices_deterministic(&mut left_inds);
        if let Some(left_inds_started) = left_inds_started {
            with_fit_profile(|profile| {
                profile.left_inds_time += left_inds_started.elapsed();
            });
        }

        // Set up factorization options
        let mut options = match self.factorize_alg {
            FactorizeAlg::SVD => FactorizeOptions::svd(),
            FactorizeAlg::QR => FactorizeOptions::qr(),
            FactorizeAlg::LU => FactorizeOptions::lu(),
            FactorizeAlg::CI => FactorizeOptions::ci(),
        };
        options = options.with_canonical(Canonical::Left);

        if let Some(policy) = self.svd_policy {
            options = options.with_svd_policy(policy);
        }
        if let Some(qr_rtol) = self.qr_rtol {
            options = options.with_qr_rtol(qr_rtol);
        }
        options
            .validate()
            .map_err(|e| anyhow::anyhow!("invalid fit factorization options: {e}"))?;

        // Determine bond dimension cap for this factorization step.
        // - If max_bond_dim is explicitly specified, use it.
        // - If an algorithm-specific tolerance is specified (but max_bond_dim is not),
        //   allow bonds to grow freely and let the factorization policy decide.
        // - Otherwise, cap at the existing bond dimension to preserve the zipup
        //   initialization size.
        let bond_cap = if self.max_bond_dim.is_some() {
            self.max_bond_dim
        } else if self.svd_policy.is_some() || self.qr_rtol.is_some() {
            None
        } else {
            subtree
                .edge_between(node_u, node_v)
                .and_then(|e| subtree.bond_index(e))
                .map(|b| b.dim())
        };
        if let Some(cap) = bond_cap {
            options = options.with_max_bond_dim(cap);
        }

        // Factorize using TensorFactorizationLike::factorize
        let factorize_started = fit_profile_enabled().then(Instant::now);
        let factorize_result = if left_inds.is_empty() || left_inds.len() == ab_uv_indices.len() {
            let (dummy_left, dummy_right) = T::Index::create_dummy_link_pair();
            let dummy_left_tensor = T::ones(std::slice::from_ref(&dummy_left))
                .map_err(|e| anyhow::anyhow!("Failed to create dummy left tensor: {e}"))?;
            let dummy_right_tensor = T::ones(std::slice::from_ref(&dummy_right))
                .map_err(|e| anyhow::anyhow!("Failed to create dummy right tensor: {e}"))?;

            let (left, right) = if left_inds.is_empty() {
                let right = ab_uv
                    .outer_product(&dummy_right_tensor)
                    .map_err(|e| anyhow::anyhow!("Failed to attach dummy right bond: {e}"))?;
                (dummy_left_tensor, right)
            } else {
                let left = ab_uv
                    .outer_product(&dummy_left_tensor)
                    .map_err(|e| anyhow::anyhow!("Failed to attach dummy left bond: {e}"))?;
                (left, dummy_right_tensor)
            };

            FactorizeResult {
                left,
                right,
                bond_index: dummy_left,
                singular_values: None,
                rank: 1,
            }
        } else {
            ab_uv
                .factorize(&left_inds, &options)
                .map_err(|e| anyhow::anyhow!("Factorization failed: {}", e))?
        };
        if let Some(factorize_started) = factorize_started {
            with_fit_profile(|profile| {
                profile.factorize_time += factorize_started.elapsed();
            });
        }

        let new_tensor_u = factorize_result.left;
        let new_tensor_v = factorize_result.right;
        let new_bond = factorize_result.bond_index;

        // Get edge between u and v
        let edge_uv = subtree.edge_between(node_u, node_v).ok_or_else(|| {
            anyhow::anyhow!(
                "FitUpdater: subtree is missing edge between {:?} and {:?}",
                node_u,
                node_v
            )
        })?;

        // Update subtree with new bond and tensors
        let idx_u_sub = subtree.node_index(node_u).ok_or_else(|| {
            anyhow::anyhow!("FitUpdater: node {:?} not found in update subtree", node_u)
        })?;
        let idx_v_sub = subtree.node_index(node_v).ok_or_else(|| {
            anyhow::anyhow!("FitUpdater: node {:?} not found in update subtree", node_v)
        })?;

        let replace_started = fit_profile_enabled().then(Instant::now);
        subtree.replace_edge_bond(edge_uv, new_bond.clone())?;
        subtree
            .replace_tensor(idx_u_sub, new_tensor_u)?
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "FitUpdater: node {:?} disappeared before tensor replacement",
                    node_u
                )
            })?;
        subtree
            .replace_tensor(idx_v_sub, new_tensor_v)?
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "FitUpdater: node {:?} disappeared before tensor replacement",
                    node_v
                )
            })?;

        // Set ortho_towards
        subtree.set_ortho_towards(&new_bond, Some(step.new_center.clone()));
        subtree.set_canonical_region([step.new_center.clone()])?;
        if let Some(replace_started) = replace_started {
            with_fit_profile(|profile| {
                profile.replace_time += replace_started.elapsed();
            });
        }

        Ok(subtree)
    }

    fn after_step(
        &mut self,
        step: &LocalUpdateStep<V>,
        full_treetn_after: &TreeTN<T, V>,
    ) -> std::result::Result<(), TreeTNOperationError> {
        // Invalidate all caches affected by the updated region
        let started = fit_profile_enabled().then(Instant::now);
        self.envs.try_invalidate(&step.nodes, full_treetn_after)?;
        if let Some(started) = started {
            with_fit_profile(|profile| {
                profile.invalidate_time += started.elapsed();
            });
        }
        Ok(())
    }
}

// ============================================================================
// High-level API: contract_fit
// ============================================================================

/// Options for fit contraction.
#[derive(Debug, Clone)]
pub struct FitContractionOptions {
    /// Number of full sweeps to perform.
    ///
    /// A full sweep visits each edge twice (forward and backward) using an Euler tour.
    pub nfullsweeps: usize,
    /// Maximum bond dimension.
    pub max_bond_dim: Option<usize>,
    /// Legacy relative tolerance retained for same-crate tests and call chains.
    pub(crate) rtol: Option<f64>,
    /// Explicit SVD truncation policy.
    pub svd_policy: Option<SvdTruncationPolicy>,
    /// QR-specific relative tolerance.
    pub qr_rtol: Option<f64>,
    /// Factorization algorithm.
    pub factorize_alg: FactorizeAlg,
    /// Tolerance for early termination based on relative change.
    /// If `None`, run exactly `nfullsweeps` sweeps.
    /// If `Some(tol)`, stop early if `||C_{i+1} - C_i|| / ||C_i|| < tol`.
    pub convergence_tol: Option<f64>,
}

impl Default for FitContractionOptions {
    fn default() -> Self {
        Self {
            nfullsweeps: 1,
            max_bond_dim: None,
            rtol: None,
            svd_policy: None,
            qr_rtol: None,
            factorize_alg: FactorizeAlg::SVD,
            convergence_tol: None,
        }
    }
}

impl FitContractionOptions {
    /// Create new options with specified number of full sweeps.
    pub fn new(nfullsweeps: usize) -> Self {
        Self {
            nfullsweeps,
            ..Default::default()
        }
    }

    /// Set maximum bond dimension.
    pub fn with_max_bond_dim(mut self, max_bond_dim: usize) -> Self {
        self.max_bond_dim = Some(max_bond_dim);
        self
    }

    /// Set the SVD truncation policy.
    pub fn with_svd_policy(mut self, policy: SvdTruncationPolicy) -> Self {
        self.rtol = Some(policy.threshold);
        self.svd_policy = Some(policy);
        self
    }

    /// Set the QR-specific relative tolerance.
    pub fn with_qr_rtol(mut self, rtol: f64) -> Self {
        self.qr_rtol = Some(rtol);
        self
    }

    /// Set relative tolerance as a per-value relative SVD policy.
    pub(crate) fn with_rtol(self, rtol: f64) -> Self {
        self.with_svd_policy(SvdTruncationPolicy::new(rtol))
    }

    /// Get the legacy SVD threshold value when represented as an rtol.
    pub(crate) fn rtol(&self) -> Option<f64> {
        self.svd_policy.map(|policy| policy.threshold)
    }

    /// Set factorization algorithm.
    pub fn with_factorize_alg(mut self, alg: FactorizeAlg) -> Self {
        self.factorize_alg = alg;
        self
    }

    /// Set convergence tolerance for early termination.
    pub fn with_convergence_tol(mut self, tol: f64) -> Self {
        self.convergence_tol = Some(tol);
        self
    }
}

/// Contract two TreeTNs using the fit (variational) algorithm.
///
/// This algorithm minimizes `||A*B - C||²` iteratively by optimizing
/// each local tensor of C while keeping others fixed, starting from an
/// initial state `tn_c`.
///
/// # Argument semantics
///
/// * `tn_a`, `tn_b` - The two input TreeTNs with the same topology. Their
///   shared site indices are contracted; `C` approximates the result.
/// * `center` - Node to use as canonical center.
/// * `options` - Fit algorithm options (sweep count, rank/truncation policy).
/// * `tn_c` - The initial variational state. It must have the same topology
///   as `tn_a`/`tn_b` and carry the surviving (non-contracted) site indices
///   that appear in `tn_a` and `tn_b`. It is updated in place.
///
/// # Rank growth semantics
///
/// * `max_bond_dim = Some(cap)` caps every sweep bond at `cap`.
/// * `max_bond_dim = None` with an SVD truncation policy allows bonds to grow
///   adaptively up to whatever the policy demands.
/// * Neither a cap nor a policy → bonds keep the dimension they already have
///   in `tn_c` (the initializer decides the starting rank).
///
/// Because the initialization is decoupled, a caller can start from a small
/// rank-1 state (see [`low_rank_initializer_tree_tn`]) without ever forming
/// the exact A·B product bond.
///
/// # Returns
/// A new TreeTN representing the contracted result.
/// # Errors
///
/// Returns an error when the fit contraction fails (a shape or index
/// mismatch, or a backend failure).
///
/// # Examples
///
/// Fit a two-node contraction from a low-rank random start, growing the bond
/// adaptively from tolerance only (`max_bond_dim = None`):
/// ```
/// use tensor4all_treetn::{
///     contract_fit_from_initial, low_rank_initializer_tree_tn, FitContractionOptions,
/// };
/// use tensor4all_core::{DynIndex, IdxTensor, SvdTruncationPolicy};
/// use tensor4all_treetn::TreeTN;
///
/// let s0 = DynIndex::new_dyn(2);
/// let s1 = DynIndex::new_dyn(2);
/// let a0 = DynIndex::new_dyn(2);
/// let a1 = DynIndex::new_dyn(2);
/// let b0 = DynIndex::new_dyn(2);
/// let b1 = DynIndex::new_dyn(2);
/// let ba = DynIndex::new_dyn(2);
/// let bb = DynIndex::new_dyn(2);
/// let ta = IdxTensor::from_dense(vec![s0.clone(), a0.clone(), ba.clone()],
///     (1..=8).map(|v| v as f64 / 8.0).collect()).unwrap();
/// let ta2 = IdxTensor::from_dense(vec![ba, s1.clone(), a1], vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.5, 0.5]).unwrap();
/// let tb = IdxTensor::from_dense(vec![s0, b0.clone(), bb.clone()],
///     (1..=8).map(|v| (v as f64 - 1.0) / 9.0).collect()).unwrap();
/// let tb2 = IdxTensor::from_dense(vec![bb, s1, b1], vec![1.0, 1.0, 0.0, 0.0, 0.25, 0.5, 0.25, 0.25]).unwrap();
/// let tn_a = TreeTN::from_tensors(vec![ta, ta2], vec![0usize, 1]).unwrap();
/// let tn_b = TreeTN::from_tensors(vec![tb, tb2], vec![0usize, 1]).unwrap();
/// let center = 1usize;
/// let init = low_rank_initializer_tree_tn(&tn_a, &tn_b, &center, 1, 7).unwrap();
/// let c = contract_fit_from_initial(
///     &tn_a,
///     &tn_b,
///     &center,
///     FitContractionOptions::new(2).with_svd_policy(SvdTruncationPolicy::new(1e-8)),
///     init,
/// )
/// .unwrap();
/// assert_eq!(c.node_count(), 2);
/// assert!(c.to_dense().unwrap().distance(&tn_a.contract_naive(&tn_b).unwrap()).unwrap() < 1e-6);
/// ```
pub fn contract_fit_from_initial<T, V>(
    tn_a: &TreeTN<T, V>,
    tn_b: &TreeTN<T, V>,
    center: &V,
    options: FitContractionOptions,
    mut tn_c: TreeTN<T, V>,
) -> std::result::Result<TreeTN<T, V>, TreeTNOperationError>
where
    T: TensorLike,
    <T::Index as IndexLike>::Id: Clone + std::hash::Hash + Eq + Ord + std::fmt::Debug + Send + Sync,
    V: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug,
{
    use super::localupdate::apply_local_update_sweep;
    let profile_enabled = fit_profile_enabled();

    // Validate topologies match
    if !tn_a.same_topology(tn_b) {
        return Err(
            anyhow::anyhow!("TreeTNs must have the same topology for fit contraction").into(),
        );
    }
    if !tn_a.same_topology(&tn_c) {
        return Err(anyhow::anyhow!(
            "fit initial TreeTN must have the same topology as the input TreeTNs"
        )
        .into());
    }

    // The sweep machinery requires the network to carry a single canonical
    // center at the requested node.
    tn_c.set_canonical_region([center.clone()])?;

    // Zero sweeps means "use the initial state as-is".
    if options.nfullsweeps == 0 {
        return Ok(tn_c);
    }

    // Create FitUpdater (environments are computed lazily)
    let mut updater = FitUpdater::new(tn_a.clone(), tn_b.clone(), options.max_bond_dim, None)
        .with_svd_policy(options.svd_policy)
        .with_qr_rtol(options.qr_rtol)
        .with_factorize_alg(options.factorize_alg);

    // Create sweep plan
    let plan = LocalUpdateSweepPlan::from_treetn(&tn_c, center, 2)
        .ok_or_else(|| anyhow::anyhow!("Failed to create sweep plan"))?;

    // Perform sweeps
    for _sweep in 0..options.nfullsweeps {
        with_fit_profile(|profile| {
            profile.sweep_count += 1;
        });
        let log_norm_before = if options.convergence_tol.is_some() {
            Some(tn_c.log_norm()?)
        } else {
            None
        };

        let sweep_started = profile_enabled.then(Instant::now);
        apply_local_update_sweep(&mut tn_c, &plan, &mut updater)?;
        if let Some(sweep_started) = sweep_started {
            with_fit_profile(|profile| {
                profile.sweep_time += sweep_started.elapsed();
            });
        }

        // Check convergence using log_norm
        if let (Some(tol), Some(log_norm_before)) = (options.convergence_tol, log_norm_before) {
            let log_norm_after = tn_c.log_norm()?;
            // relative change in norm: |exp(log_after) - exp(log_before)| / exp(log_before)
            // = |exp(log_after - log_before) - 1|
            let relative_change = (f64::exp(log_norm_after - log_norm_before) - 1.0).abs();
            if relative_change < tol {
                break;
            }
        }
    }

    if let Some(profile) = take_fit_profile() {
        eprintln!("=== contract_fit Profiling ===");
        eprintln!("zipup init:        {:?}", profile.zipup_init_time);
        eprintln!("canonicalize:      {:?}", profile.canonicalize_time);
        eprintln!("sweeps total:      {:?}", profile.sweep_time);
        eprintln!("steps:             {}", profile.step_count);
        eprintln!("sweeps:            {}", profile.sweep_count);
        eprintln!(
            "env get:           {:?} (requests={}, hits={}, misses={})",
            profile.env_get_time, profile.env_requests, profile.env_hits, profile.env_misses
        );
        eprintln!("env leaf compute:  {:?}", profile.env_leaf_time);
        eprintln!("env node compute:  {:?}", profile.env_internal_time);
        eprintln!("2-site contract:   {:?}", profile.two_site_contract_time);
        eprintln!("left_inds:         {:?}", profile.left_inds_time);
        eprintln!("factorize:         {:?}", profile.factorize_time);
        eprintln!("replace/update:    {:?}", profile.replace_time);
        eprintln!("invalidate:        {:?}", profile.invalidate_time);
    }
    // These are tensor4all-owned profiling hooks, intentionally kept during the
    // migration so fit profiling still works without the removed tenferro
    // runtime APIs.
    print_and_reset_contract_profile();
    print_and_reset_native_einsum_profile();

    Ok(tn_c)
}

/// Contract two TreeTNs using the fit (variational) algorithm.
///
/// Convenience wrapper over [`contract_fit_from_initial`] that initializes
/// `C` with the SVD-based zip-up contraction of `A·B` while preserving the
/// input topology.
///
/// # Note on the exact product bond
///
/// Zip-up initialization *materializes* the full product bond (up to
/// χ_A·χ_B per edge) when no truncation policy is supplied. If that is not
/// desired (large QTT/MPO bonds), construct a small-rank starting state with
/// [`low_rank_initializer_tree_tn`] and pass it to
/// [`contract_fit_from_initial`] instead.
///
/// # Arguments
/// * `tn_a` - First TreeTN
/// * `tn_b` - Second TreeTN
/// * `center` - Node to use as canonical center
/// * `options` - Fit algorithm options
///
/// # Returns
/// A new TreeTN representing the contracted result.
/// # Errors
///
/// Returns an error when the fit contraction fails (a shape or index
/// mismatch, or a backend failure).
///
pub fn contract_fit<T, V>(
    tn_a: &TreeTN<T, V>,
    tn_b: &TreeTN<T, V>,
    center: &V,
    options: FitContractionOptions,
) -> std::result::Result<TreeTN<T, V>, TreeTNOperationError>
where
    T: TensorLike,
    <T::Index as IndexLike>::Id: Clone + std::hash::Hash + Eq + Ord + std::fmt::Debug + Send + Sync,
    V: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug,
{
    use crate::CanonicalForm;
    let profile_enabled = fit_profile_enabled();
    if profile_enabled {
        fit_profile_reset();
    }
    reset_contract_profile();
    reset_native_einsum_profile();

    // Validate topologies match
    if !tn_a.same_topology(tn_b) {
        return Err(
            anyhow::anyhow!("TreeTNs must have the same topology for fit contraction").into(),
        );
    }

    // Initialize C using the SVD-based zipup contraction while preserving
    // the input topology required by variational sweeps.
    let zipup_started = profile_enabled.then(Instant::now);
    let tn_c = tn_a.contract_zipup_preserving_topology_with(
        tn_b,
        center,
        CanonicalForm::Unitary,
        options.svd_policy,
        options.max_bond_dim,
    )?;
    if let Some(zipup_started) = zipup_started {
        with_fit_profile(|profile| {
            profile.zipup_init_time += zipup_started.elapsed();
        });
    }

    contract_fit_from_initial(tn_a, tn_b, center, options, tn_c)
}

/// Build a low-rank, topology-compatible initial state `C₀` for fit contraction.
///
/// The returned network has the *same topology* as `tn_a`/`tn_b` and carries
/// the surviving site indices of the product `A·B` (the site indices not
/// shared between `tn_a` and `tn_b` at each node), with every bond set to
/// dimension `bond_dim` and filled deterministically from `seed`.
///
/// # Why this avoids the product bond
///
/// The construction never contracts `A` and `B`: each output tensor is an
/// independent random tensor over the output site indices plus its small
/// bonds. No tensor of dimension proportional to χ_A·χ_B is ever formed, so
/// the subsequent two-site fit sweeps can grow ranks adaptively (with an SVD
/// truncation policy and `max_bond_dim = None`) without first building the
/// exact product.
///
/// # Arguments
/// * `tn_a` - First input TreeTN (real or complex `IdxTensor` tensors).
/// * `tn_b` - Second input TreeTN.
/// * `center` - Canonical center for the returned network.
/// * `bond_dim` - Initial bond dimension of every edge (≥ 1; use 1 for the
///   "start small and grow" behavior).
/// * `seed` - Deterministic RNG seed. A fixed seed reproduces the exact same
///   initializer across runs.
///
/// # Errors
///
/// Returns an error when the input topologies differ, `bond_dim` is zero, the
/// input tensors use a dtype other than `f64`/`Complex64`, or metadata
/// construction fails.
///
/// # Examples
///
/// ```
/// use tensor4all_treetn::{
///     low_rank_initializer_tree_tn, TreeTN,
/// };
/// use tensor4all_core::{DynIndex, IdxTensor, IndexLike};
///
/// // Two-node chains: A = (s0,a0)-(s1,a1), B = (s0,b0)-(s1,b1). The `s` sites
/// // are shared (contracted in A·B); `a`/`b` survive into the output.
/// let s0 = DynIndex::new_dyn(2);
/// let s1 = DynIndex::new_dyn(2);
/// let a0 = DynIndex::new_dyn(2);
/// let a1 = DynIndex::new_dyn(2);
/// let b0 = DynIndex::new_dyn(2);
/// let b1 = DynIndex::new_dyn(2);
/// let ba = DynIndex::new_dyn(2);
/// let bb = DynIndex::new_dyn(2);
/// let ta0 = IdxTensor::from_dense(vec![s0.clone(), a0.clone(), ba.clone()], vec![1.0; 8]).unwrap();
/// let ta1 = IdxTensor::from_dense(vec![ba, s1.clone(), a1], vec![1.0; 8]).unwrap();
/// let tb0 = IdxTensor::from_dense(vec![s0.clone(), b0.clone(), bb.clone()], vec![1.0; 8]).unwrap();
/// let tb1 = IdxTensor::from_dense(vec![bb, s1.clone(), b1], vec![1.0; 8]).unwrap();
/// let tn_a = TreeTN::from_tensors(vec![ta0, ta1], vec![0usize, 1]).unwrap();
/// let tn_b = TreeTN::from_tensors(vec![tb0, tb1], vec![0usize, 1]).unwrap();
///
/// let init = low_rank_initializer_tree_tn(&tn_a, &tn_b, &1usize, 1, 42).unwrap();
/// // Node set is preserved and the single bond starts at dimension 1,
/// // not chi_a*chi_b.
/// assert_eq!(init.node_count(), 2);
/// let bond = init
///     .edge_between(&0usize, &1usize)
///     .and_then(|e| init.bond_index(e))
///     .unwrap()
///     .dim();
/// assert_eq!(bond, 1);
/// assert_eq!(init.site_space(&0).unwrap().len(), 2); // {a0, b0} survive
/// ```
pub fn low_rank_initializer_tree_tn<V>(
    tn_a: &TreeTN<IdxTensor, V>,
    tn_b: &TreeTN<IdxTensor, V>,
    center: &V,
    bond_dim: usize,
    seed: u64,
) -> std::result::Result<TreeTN<IdxTensor, V>, TreeTNOperationError>
where
    V: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug,
{
    if bond_dim == 0 {
        return Err(anyhow::anyhow!("fit low-rank initializer requires bond_dim >= 1").into());
    }
    if !tn_a.same_topology(tn_b) {
        return Err(
            anyhow::anyhow!("TreeTNs must have the same topology for fit contraction").into(),
        );
    }

    // Dtype of the random fill follows the input networks.
    let complex = network_tensors_complex(tn_a) || network_tensors_complex(tn_b);
    let mut rng = StdRng::seed_from_u64(seed);

    // 1. Fresh bond indices for each edge, with the requested dimension.
    let mut edges: Vec<(V, V)> = tn_a.site_index_network().edges().collect();
    edges.sort();
    let mut edge_bond: HashMap<(V, V), DynIndex> = HashMap::with_capacity(edges.len());
    for (a, b) in &edges {
        let key = if a < b {
            (a.clone(), b.clone())
        } else {
            (b.clone(), a.clone())
        };
        edge_bond.insert(key, DynIndex::new_dyn(bond_dim));
    }

    // 2. Per-node tensor: surviving site indices (symmetric difference of the
    //    per-node site spaces of A and B) plus the incident bonds.
    let mut node_names: Vec<V> = tn_a
        .site_index_network()
        .node_names()
        .into_iter()
        .cloned()
        .collect();
    node_names.sort();

    let mut tensors: Vec<IdxTensor> = Vec::with_capacity(node_names.len());
    for node in &node_names {
        let a_sites: HashSet<DynIndex> = tn_a
            .site_index_network()
            .site_space(node)
            .cloned()
            .unwrap_or_default();
        let b_sites: HashSet<DynIndex> = tn_b
            .site_index_network()
            .site_space(node)
            .cloned()
            .unwrap_or_default();
        let shared: HashSet<DynIndex> = a_sites.intersection(&b_sites).cloned().collect();

        let mut indices: Vec<DynIndex> = Vec::new();
        indices.extend(a_sites.iter().filter(|s| !shared.contains(*s)).cloned());
        indices.extend(b_sites.iter().filter(|s| !shared.contains(*s)).cloned());
        sort_indices_deterministic(&mut indices);

        let mut neighbors: Vec<V> = tn_a.site_index_network().neighbors(node).collect();
        neighbors.sort();
        for neighbor in &neighbors {
            let key = if node < neighbor {
                (node.clone(), neighbor.clone())
            } else {
                (neighbor.clone(), node.clone())
            };
            if let Some(bond) = edge_bond.get(&key) {
                indices.push(bond.clone());
            }
        }

        let tensor = if complex {
            IdxTensor::random::<Complex64, _>(&mut rng, indices.clone())
                .map_err(TreeTNOperationError::from)?
        } else {
            IdxTensor::random::<f64, _>(&mut rng, indices.clone())
                .map_err(TreeTNOperationError::from)?
        };
        tensors.push(tensor);
    }

    let mut tn_c = TreeTN::from_tensors(tensors, node_names)?;
    tn_c.set_canonical_region([center.clone()])?;
    Ok(tn_c)
}

fn network_tensors_complex<V>(tn: &TreeTN<IdxTensor, V>) -> bool
where
    V: Clone + Hash + Eq + Send + Sync + std::fmt::Debug,
{
    let mut complex = false;
    for node in tn.site_index_network().node_names() {
        let node = node.clone();
        if let Some(idx) = tn.node_index(&node) {
            if let Some(t) = tn.tensor(idx) {
                if t.is_complex() {
                    complex = true;
                    break;
                }
            }
        }
    }
    complex
}

#[cfg(test)]
mod tests;
