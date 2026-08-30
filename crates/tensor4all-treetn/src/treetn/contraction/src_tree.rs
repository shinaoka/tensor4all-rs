//! Paper-faithful tree extension of successive randomized compression.
//!
//! Provenance: the per-edge sketch/QR/projection pattern is derived from
//! Algorithm 1 and Sections 3.1-3.5 of Camaño--Epperly--Tropp,
//! [arXiv:2504.06475](https://arxiv.org/abs/2504.06475). The author repository
//! `chriscamano/RandomMPOMPS` contains no tree implementation; its chain
//! reference is `code/tensornetwork/contraction.py::random_contraction_inc`
//! (lines 405--593). Rooting, directed message passing, complement environments,
//! and the center-directed tree assembly are therefore a manual tensor-network
//! derivation and are explicitly marked `[AI-Supplied]` in the audit worklog.

use anyhow::Result;
use std::collections::{HashMap, HashSet};
use std::hash::Hash;

use tensor4all_core::{IndexLike, SvdTruncationPolicy, TensorLike};

use super::src_probe::{
    connect_result_edge, contract_retaining, factorize_probe_columns, initial_width,
    local_output_indices, local_site_pairs, mark_result_canonical, maximum_site_width,
    probed_site_pair, probed_site_pair_batch_range, product_dim, ProbeBank,
};
use super::{SrcOptions, TreeTN};

type DirectedEnvironment<T, V> = HashMap<(V, V), T>;
type BatchedEnvironment<T, V> = (
    <T as tensor4all_core::TensorIndex>::Index,
    DirectedEnvironment<T, V>,
);
/// One growth call's stored result: `(start, width, batch index, per-edge
/// environment)`, covering probe columns `[start, start + width)`.
type EnvironmentSegment<T, V> = (
    usize,
    usize,
    <T as tensor4all_core::TensorIndex>::Index,
    DirectedEnvironment<T, V>,
);

/// Execute successive randomized compression on a chain or a rooted tree.
pub(super) fn contract<T, V>(
    tn_a: &TreeTN<T, V>,
    tn_b: &TreeTN<T, V>,
    center: &V,
    svd_policy: Option<SvdTruncationPolicy>,
    max_bond_dim: usize,
    src_options: &SrcOptions,
) -> Result<TreeTN<T, V>>
where
    T: TensorLike,
    <T::Index as IndexLike>::Id: Clone + Hash + Eq + Ord + std::fmt::Debug + Send + Sync,
    V: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug,
{
    if let Some(chain) = tn_a.chain_order(center) {
        // The chain recurrence produces a left-canonical sweep whose center
        // is the final site. An interior requested center needs the rooted
        // tree recurrence so both sides are compressed towards that center.
        if chain.last() == Some(center) {
            return super::src_chain::contract(
                tn_a,
                tn_b,
                center,
                svd_policy,
                max_bond_dim,
                src_options,
            );
        }
    }
    if !tn_a.same_topology(tn_b) {
        anyhow::bail!("contract_src: networks have incompatible topologies");
    }
    if tn_a.node_index(center).is_none() {
        anyhow::bail!("contract_src: center node is missing");
    }

    let mut nodes = tn_a.node_names();
    if nodes.is_empty() {
        anyhow::bail!("contract_src: empty tree");
    }
    nodes.sort();
    let edges = tn_a
        .edges_to_canonicalize_by_names(center)
        .ok_or_else(|| anyhow::anyhow!("contract_src: cannot root tree at center"))?;
    if edges.len() + 1 != nodes.len() {
        anyhow::bail!(
            "contract_src: expected a connected tree, got {} nodes and {} rooted edges",
            nodes.len(),
            edges.len()
        );
    }

    let tn_a_sim = tn_a.sim_internal_inds();
    let tn_b_sim = tn_b.sim_internal_inds();
    let sketch_options = src_options.sketch_options(svd_policy.is_some());
    let local_values = local_site_pairs(&tn_a_sim, &tn_b_sim, &nodes)?;
    let local = nodes
        .iter()
        .cloned()
        .zip(local_values)
        .collect::<HashMap<_, _>>();
    let outputs = nodes
        .iter()
        .map(|node| {
            local
                .get(node)
                .ok_or_else(|| anyhow::anyhow!("contract_src: local tensors are missing"))?;
            Ok((
                node.clone(),
                local_output_indices(&tn_a_sim, &tn_b_sim, node)?,
            ))
        })
        .collect::<Result<HashMap<_, _>>>()?;

    // Iterate `nodes` (already sorted, so process-stable) rather than
    // `outputs.values()`: `HashMap` iteration order is randomized per
    // process, and the flattened order here is consumed sequentially by
    // `ProbeBank`'s seeded RNG (see `local_output_indices`'s doc comment in
    // `src_probe.rs` for the matching per-node instability this mirrors at
    // the whole-tree level).
    let probe_indices = nodes
        .iter()
        .flat_map(|node| {
            outputs
                .get(node)
                .into_iter()
                .flat_map(|site| site.iter().cloned())
        })
        .collect::<Vec<_>>();
    if outputs
        .values()
        .map(|site| product_dim(site))
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .any(|dimension| dimension == 0)
    {
        anyhow::bail!("contract_src: tree output space has zero dimension");
    }
    let mut probes = ProbeBank::new(probe_indices, 1, src_options.seed)?;
    let mut environment_cache =
        EnvironmentCache::new(&tn_a_sim, &edges, &nodes, &local, &outputs, &mut probes);

    let mut result_tensors = HashMap::with_capacity(nodes.len());
    let mut projected_children: HashMap<V, Vec<T>> = HashMap::new();

    // Rooted edges are in child-to-parent postorder. Every projected child
    // bridge therefore exists before its parent source is assembled.
    for (child, parent) in &edges {
        let (tensor_a, tensor_b) = local
            .get(child)
            .copied()
            .ok_or_else(|| anyhow::anyhow!("contract_src: child local tensors are missing"))?;
        let source_factors = site_factors(
            tensor_a,
            tensor_b,
            projected_children.get(child).map(Vec::as_slice),
        );
        let (parent_bond_a, parent_bond_b) = edge_bonds(&tn_a_sim, &tn_b_sim, child, parent)?;
        let cut_dimension = parent_bond_a
            .dim()
            .checked_mul(parent_bond_b.dim())
            .ok_or_else(|| anyhow::anyhow!("contract_src: tree cut dimension overflow"))?;
        let parent_bonds = HashSet::from([parent_bond_a, parent_bond_b]);
        let left_indices = uncontracted_indices(&source_factors, &parent_bonds);

        let (factor, projected) = if left_indices.is_empty() {
            // A scalar-only subtree still needs a structural bridge. Its
            // scalar value stays in the projected source absorbed by parent.
            let cap = T::Index::new_link(1)?;
            let factor = T::ones(std::slice::from_ref(&cap)).map_err(|error| {
                anyhow::anyhow!("contract_src: scalar tree cap construction failed: {error}")
            })?;
            let source = contract_factors(&source_factors, "contract_src: scalar tree source")?;
            let projected = source.outer_product(&factor).map_err(|error| {
                anyhow::anyhow!("contract_src: scalar tree projection failed: {error}")
            })?;
            (factor, projected)
        } else {
            let row_dim = product_dim(&left_indices)?;
            let site_max_width =
                maximum_site_width(max_bond_dim, row_dim, cut_dimension, &sketch_options);
            if site_max_width == 0 {
                anyhow::bail!("contract_src: tree sketch row space is empty");
            }
            let site_initial_width = if sketch_options.rtol.is_some() {
                initial_width(site_max_width, &sketch_options)
            } else {
                site_max_width
            };
            let label = format!("tree edge {:?}->{:?}", child, parent);
            let (factor, _cap) = if sketch_options.rtol.is_none() {
                let (environment, batch) =
                    environment_cache.batch(parent, child, site_max_width)?;
                let mut sketch_factors = source_factors.clone();
                sketch_factors.push(&environment);
                let sketch = contract_retaining(&sketch_factors, &batch).map_err(|error| {
                    anyhow::anyhow!(
                        "contract_src: tree batched sketch for {:?}->{:?} failed: {error}",
                        child,
                        parent
                    )
                })?;
                let factorized = sketch
                    .factorize_full_rank(
                        &left_indices,
                        tensor4all_core::FactorizeAlg::QR,
                        tensor4all_core::Canonical::Left,
                    )
                    .map_err(|error| {
                        anyhow::anyhow!(
                            "contract_src: {label} QR failed: {error}; sketch indices={:?}, left indices={left_indices:?}",
                            sketch.external_indices()
                        )
                    })?;
                (factorized.left, factorized.bond_index)
            } else {
                factorize_probe_columns(
                    &left_indices,
                    site_initial_width,
                    site_max_width,
                    &sketch_options,
                    &label,
                    |column| {
                        let environment = environment_cache.column(parent, child, column)?;
                        let mut factors = source_factors.clone();
                        factors.push(&environment);
                        T::contract(&factors).map_err(|error| {
                            anyhow::anyhow!(
                                "contract_src: tree sketch for {:?}->{:?} failed: {error}",
                                child,
                                parent
                            )
                        })
                    },
                )?
            };
            let factor_conj = factor.conj();
            let mut projection_factors = Vec::with_capacity(source_factors.len() + 1);
            projection_factors.push(&factor_conj);
            projection_factors.extend(source_factors.iter().copied());
            let projected = T::contract(&projection_factors).map_err(|error| {
                anyhow::anyhow!(
                    "contract_src: tree projection for {:?}->{:?} failed: {error}",
                    child,
                    parent
                )
            })?;
            (factor, projected)
        };

        result_tensors.insert(child.clone(), factor);
        projected_children
            .entry(parent.clone())
            .or_default()
            .push(projected);
    }

    let (root_a, root_b) = local
        .get(center)
        .copied()
        .ok_or_else(|| anyhow::anyhow!("contract_src: root local tensors are missing"))?;
    let root_tensor = merge_projected(
        root_a,
        root_b,
        projected_children.get(center).map(Vec::as_slice),
    )?;
    result_tensors.insert(center.clone(), root_tensor);

    let mut result = TreeTN::new();
    for node in &nodes {
        let tensor = result_tensors
            .remove(node)
            .ok_or_else(|| anyhow::anyhow!("contract_src: result tensor is missing"))?;
        result.add_tensor(node.clone(), tensor)?;
    }
    for (child, parent) in &edges {
        connect_result_edge(&mut result, child, parent)?;
    }

    if src_options.final_svd {
        result.truncate_impl(
            [center.clone()],
            svd_policy,
            Some(max_bond_dim),
            "contract_src: tree final truncate",
        )?;
    } else {
        mark_result_canonical(&mut result, center, &edges)?;
    }
    Ok(result)
}

struct EnvironmentCache<'a, T, V>
where
    T: TensorLike,
    V: Clone + Hash + Eq + Send + Sync + std::fmt::Debug,
{
    tn: &'a TreeTN<T, V>,
    edges: &'a [(V, V)],
    nodes: &'a [V],
    local: &'a HashMap<V, (&'a T, &'a T)>,
    outputs: &'a HashMap<V, Vec<T::Index>>,
    probes: &'a mut ProbeBank<T::Index>,
    environments: Vec<HashMap<(V, V), T>>,
    batched_environments: HashMap<usize, BatchedEnvironment<T, V>>,
    segments: Vec<EnvironmentSegment<T, V>>,
    segment_total_width: usize,
}

impl<'a, T, V> EnvironmentCache<'a, T, V>
where
    T: TensorLike,
    T::Index: IndexLike + Clone + Hash + Eq,
    V: Clone + Hash + Eq + Send + Sync + std::fmt::Debug,
{
    fn new(
        tn: &'a TreeTN<T, V>,
        edges: &'a [(V, V)],
        nodes: &'a [V],
        local: &'a HashMap<V, (&'a T, &'a T)>,
        outputs: &'a HashMap<V, Vec<T::Index>>,
        probes: &'a mut ProbeBank<T::Index>,
    ) -> Self {
        Self {
            tn,
            edges,
            nodes,
            local,
            outputs,
            probes,
            environments: Vec::new(),
            batched_environments: HashMap::new(),
            segments: Vec::new(),
            segment_total_width: 0,
        }
    }

    fn ensure_width(&mut self, width: usize) -> Result<()> {
        self.probes.extend_to(width)?;
        for column in self.environments.len()..width {
            let probed = self
                .nodes
                .iter()
                .map(|node| {
                    let (tensor_a, tensor_b) =
                        self.local.get(node).copied().ok_or_else(|| {
                            anyhow::anyhow!("contract_src: local tensor is missing")
                        })?;
                    let site_outputs = self
                        .outputs
                        .get(node)
                        .ok_or_else(|| anyhow::anyhow!("contract_src: output list is missing"))?;
                    Ok((
                        node.clone(),
                        probed_site_pair(tensor_a, tensor_b, site_outputs, self.probes, column)?,
                    ))
                })
                .collect::<Result<HashMap<_, _>>>()?;
            let mut directed = directed_messages(self.tn, self.edges, &probed)?;
            let mut selected = HashMap::with_capacity(self.edges.len());
            for (child, parent) in self.edges {
                let message = directed
                    .remove(&(parent.clone(), child.clone()))
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "contract_src: complement environment is missing for {:?}->{:?}",
                            parent,
                            child
                        )
                    })?;
                selected.insert((parent.clone(), child.clone()), message);
            }
            self.environments.push(selected);
        }
        Ok(())
    }

    fn batch(&mut self, parent: &V, child: &V, width: usize) -> Result<(T, T::Index)> {
        if width == 0 {
            anyhow::bail!("contract_src: tree environment batch must be non-empty");
        }
        if let Some((batch, environments)) = self.batched_environments.get(&width) {
            let environment = environments
                .get(&(parent.clone(), child.clone()))
                .cloned()
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "contract_src: cached batched complement environment is missing for {:?}->{:?}",
                        parent,
                        child
                    )
                })?;
            return Ok((environment, batch.clone()));
        }
        self.probes.extend_to(width)?;
        let batch = T::Index::new_link(width)?;
        let probed =
            self.nodes
                .iter()
                .map(|node| {
                    let (tensor_a, tensor_b) =
                        self.local.get(node).copied().ok_or_else(|| {
                            anyhow::anyhow!("contract_src: local tensor is missing")
                        })?;
                    let site_outputs = self
                        .outputs
                        .get(node)
                        .ok_or_else(|| anyhow::anyhow!("contract_src: output list is missing"))?;
                    Ok((
                        node.clone(),
                        probed_site_pair_batch_range(
                            tensor_a,
                            tensor_b,
                            site_outputs,
                            self.probes,
                            0,
                            width,
                            &batch,
                        )?,
                    ))
                })
                .collect::<Result<HashMap<_, _>>>()?;
        let directed = directed_messages_batched(self.tn, self.edges, &probed, &batch)?;
        let environment = directed
            .get(&(parent.clone(), child.clone()))
            .cloned()
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "contract_src: batched complement environment is missing for {:?}->{:?}",
                    parent,
                    child
                )
            })?;
        self.batched_environments
            .insert(width, (batch.clone(), directed));
        Ok((environment, batch))
    }

    fn column(&mut self, parent: &V, child: &V, column: usize) -> Result<T> {
        self.ensure_width(column + 1)?;
        self.environments
            .get(column)
            .and_then(|messages| messages.get(&(parent.clone(), child.clone())))
            .cloned()
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "contract_src: complement environment is missing for {:?}->{:?}",
                    parent,
                    child
                )
            })
    }

    /// Compute one new segment covering `[start, start + width)`'s
    /// per-edge environment tensors, without ever materializing a
    /// per-column (`width == 1`) representation.
    fn grow_segment(&mut self, start: usize, width: usize) -> Result<()> {
        self.probes.extend_to(start + width)?;
        let batch = T::Index::new_link(width)?;
        let probed =
            self.nodes
                .iter()
                .map(|node| {
                    let (tensor_a, tensor_b) =
                        self.local.get(node).copied().ok_or_else(|| {
                            anyhow::anyhow!("contract_src: local tensor is missing")
                        })?;
                    let site_outputs = self
                        .outputs
                        .get(node)
                        .ok_or_else(|| anyhow::anyhow!("contract_src: output list is missing"))?;
                    Ok((
                        node.clone(),
                        probed_site_pair_batch_range(
                            tensor_a,
                            tensor_b,
                            site_outputs,
                            self.probes,
                            start,
                            width,
                            &batch,
                        )?,
                    ))
                })
                .collect::<Result<HashMap<_, _>>>()?;
        // `directed_messages_batched` contracts every message via
        // `contract_retaining(&factors, batch)`, which explicitly keeps
        // `batch` in the result -- every tensor in `directed` carries this
        // exact `batch` index, so it is safe to store and hand back later
        // without re-deriving it from any one tensor's own index list.
        let directed = directed_messages_batched(self.tn, self.edges, &probed, &batch)?;
        self.segments.push((start, width, batch, directed));
        self.segment_total_width += width;
        Ok(())
    }

    /// Return the environment tensor for `(parent, child)` covering
    /// `[start, start + width)`, growing new segments first if needed. See
    /// `PrefixCache::request` (`src_chain.rs`) for the equivalent
    /// chain-path method and the same aligned-vs-misaligned reasoning.
    fn request(
        &mut self,
        parent: &V,
        child: &V,
        start: usize,
        width: usize,
    ) -> Result<(T, T::Index)> {
        while self.segment_total_width < start + width {
            let next_start = self.segment_total_width;
            self.grow_segment(next_start, width.min(start + width - next_start))?;
        }
        for (segment_start, segment_width, batch_index, environments) in &self.segments {
            if *segment_start == start && *segment_width == width {
                let environment = environments
                    .get(&(parent.clone(), child.clone()))
                    .cloned()
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "contract_src: cached segment environment is missing for {:?}->{:?}",
                            parent,
                            child
                        )
                    })?;
                return Ok((environment, batch_index.clone()));
            }
        }
        anyhow::bail!(
            "contract_src: no segment covers [{start}, {}) for {:?}->{:?} after growth",
            start + width,
            parent,
            child
        )
    }
}

fn merge_projected<T>(tensor_a: &T, tensor_b: &T, projected: Option<&[T]>) -> Result<T>
where
    T: TensorLike,
{
    let mut factors = Vec::with_capacity(projected.map_or(2, |items| items.len() + 2));
    factors.push(tensor_a);
    factors.push(tensor_b);
    if let Some(projected) = projected {
        factors.extend(projected.iter());
    }
    contract_factors(&factors, "contract_src: projected local merge")
}

fn site_factors<'a, T>(tensor_a: &'a T, tensor_b: &'a T, projected: Option<&'a [T]>) -> Vec<&'a T>
where
    T: TensorLike,
{
    let mut factors = Vec::with_capacity(projected.map_or(2, |items| items.len() + 2));
    factors.push(tensor_a);
    factors.push(tensor_b);
    if let Some(projected) = projected {
        factors.extend(projected.iter());
    }
    factors
}

fn uncontracted_indices<T>(factors: &[&T], excluded: &HashSet<T::Index>) -> Vec<T::Index>
where
    T: TensorLike,
    T::Index: IndexLike + Clone + Hash + Eq,
    <T::Index as IndexLike>::Id: Ord,
{
    // Order follows each factor's own `external_indices()`, in the order
    // `factors` is given -- not `HashMap`/`sort_indices_deterministic`
    // iteration order. `contract()` re-mints every bond index's random id on
    // every call (`sim_internal_inds()`), so when a physical output index
    // and a same-dimension bond share a tie under `sort_indices_deterministic`,
    // the row order fed to the QR factorization below used to change from
    // one `contract()` call to the next -- even within one process, on the
    // exact same input tensors -- producing small (row-permutation-induced)
    // floating-point rounding differences. See
    // docs/worklogs/2026-08-30-src-probe-order-nondeterminism.md.
    let mut counts: HashMap<T::Index, usize> = HashMap::new();
    for factor in factors {
        for index in factor.external_indices() {
            *counts.entry(index).or_insert(0usize) += 1;
        }
    }
    let mut seen = HashSet::new();
    let mut result = Vec::new();
    for factor in factors {
        for index in factor.external_indices() {
            if counts.get(&index) == Some(&1)
                && !excluded.contains(&index)
                && seen.insert(index.clone())
            {
                result.push(index);
            }
        }
    }
    result
}

fn contract_factors<T>(factors: &[&T], context: &str) -> Result<T>
where
    T: TensorLike,
{
    match factors {
        [] => anyhow::bail!("{context}: no tensors to contract"),
        [single] => Ok((*single).clone()),
        _ => T::contract(factors).map_err(|error| anyhow::anyhow!("{context}: {error}")),
    }
}

fn edge_bonds<T, V>(
    tn_a: &TreeTN<T, V>,
    tn_b: &TreeTN<T, V>,
    child: &V,
    parent: &V,
) -> Result<(T::Index, T::Index)>
where
    T: TensorLike,
    T::Index: IndexLike + Clone + Hash + Eq,
    V: Clone + Hash + Eq + Send + Sync + std::fmt::Debug,
{
    let edge_a = tn_a
        .edge_between(child, parent)
        .ok_or_else(|| anyhow::anyhow!("contract_src: A parent edge is missing"))?;
    let edge_b = tn_b
        .edge_between(child, parent)
        .ok_or_else(|| anyhow::anyhow!("contract_src: B parent edge is missing"))?;
    let bond_a = tn_a
        .bond_index(edge_a)
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("contract_src: A parent bond is missing"))?;
    let bond_b = tn_b
        .bond_index(edge_b)
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("contract_src: B parent bond is missing"))?;
    Ok((bond_a, bond_b))
}

fn directed_messages<T, V>(
    tn: &TreeTN<T, V>,
    edges: &[(V, V)],
    probed: &HashMap<V, T>,
) -> Result<HashMap<(V, V), T>>
where
    T: TensorLike,
    V: Clone + Hash + Eq + Send + Sync + std::fmt::Debug,
{
    let mut messages = HashMap::with_capacity(edges.len() * 2);

    // Upward pass: each child message contains that whole child subtree,
    // with the physical outputs already contracted against this probe column.
    for (child, parent) in edges {
        let mut factors = vec![probed
            .get(child)
            .ok_or_else(|| anyhow::anyhow!("contract_src: probed child tensor is missing"))?];
        for neighbor in tn.site_index_network().neighbors(child) {
            if &neighbor == parent {
                continue;
            }
            let message = messages
                .get(&(neighbor.clone(), child.clone()))
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "contract_src: upward message is missing for {:?}->{:?}",
                        neighbor,
                        child
                    )
                })?;
            factors.push(message);
        }
        let message = contract_factors(&factors, "contract_src: upward message")?;
        messages.insert((child.clone(), parent.clone()), message);
    }

    // Downward pass: the reverse postorder guarantees that the parent-side
    // message is available before the next child is visited.
    for (child, parent) in edges.iter().rev() {
        let mut factors = vec![probed
            .get(parent)
            .ok_or_else(|| anyhow::anyhow!("contract_src: probed parent tensor is missing"))?];
        for neighbor in tn.site_index_network().neighbors(parent) {
            if &neighbor == child {
                continue;
            }
            let message = messages
                .get(&(neighbor.clone(), parent.clone()))
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "contract_src: side message is missing for {:?} around {:?}",
                        neighbor,
                        parent
                    )
                })?;
            factors.push(message);
        }
        let message = contract_factors(&factors, "contract_src: downward message")?;
        messages.insert((parent.clone(), child.clone()), message);
    }
    Ok(messages)
}

fn directed_messages_batched<T, V>(
    tn: &TreeTN<T, V>,
    edges: &[(V, V)],
    probed: &HashMap<V, T>,
    batch: &T::Index,
) -> Result<HashMap<(V, V), T>>
where
    T: TensorLike,
    T::Index: IndexLike + Clone + Hash + Eq,
    V: Clone + Hash + Eq + Send + Sync + std::fmt::Debug,
{
    let mut messages = HashMap::with_capacity(edges.len() * 2);

    for (child, parent) in edges {
        let mut factors = vec![probed
            .get(child)
            .ok_or_else(|| anyhow::anyhow!("contract_src: probed child tensor is missing"))?];
        for neighbor in tn.site_index_network().neighbors(child) {
            if &neighbor == parent {
                continue;
            }
            let message = messages
                .get(&(neighbor.clone(), child.clone()))
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "contract_src: upward batched message is missing for {:?}->{:?}",
                        neighbor,
                        child
                    )
                })?;
            factors.push(message);
        }
        let message = contract_retaining(&factors, batch)?;
        messages.insert((child.clone(), parent.clone()), message);
    }

    for (child, parent) in edges.iter().rev() {
        let mut factors = vec![probed
            .get(parent)
            .ok_or_else(|| anyhow::anyhow!("contract_src: probed parent tensor is missing"))?];
        for neighbor in tn.site_index_network().neighbors(parent) {
            if &neighbor == child {
                continue;
            }
            let message = messages
                .get(&(neighbor.clone(), parent.clone()))
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "contract_src: side batched message is missing for {:?} around {:?}",
                        neighbor,
                        parent
                    )
                })?;
            factors.push(message);
        }
        let message = contract_retaining(&factors, batch)?;
        messages.insert((parent.clone(), child.clone()), message);
    }

    Ok(messages)
}

#[cfg(test)]
mod tests {
    use super::super::src_probe::{local_output_indices, local_site_pairs, ProbeBank};
    use super::{uncontracted_indices, EnvironmentCache};
    use crate::treetn::TreeTN;
    use std::collections::HashSet;
    use tensor4all_core::{DynIndex, IdxTensor, IndexLike};

    /// `uncontracted_indices` used to order same-dimension survivors by
    /// `Index::id()` (via `HashMap` + `sort_indices_deterministic`), and
    /// `id()` is drawn from a per-process, unseeded RNG that `contract()`
    /// re-invokes on every call (`sim_internal_inds()` mints fresh bond ids
    /// each time). Rebuilding `out_a`/`out_b` fresh each iteration mimics
    /// that: under the old implementation this had roughly a 50% chance per
    /// iteration of returning `[out_b, out_a]` instead, so 200 iterations
    /// would fail with probability `1 - 2^-200`. Under the fix, order always
    /// follows `factors`' own order, regardless of the ids involved.
    #[test]
    fn uncontracted_indices_orders_same_dimension_survivors_by_factor_order_not_random_id() {
        for _ in 0..200 {
            let shared = DynIndex::new_dyn(2);
            let out_a = DynIndex::new_dyn(3);
            let out_b = DynIndex::new_dyn(3);

            let factor_1 =
                IdxTensor::from_dense(vec![shared.clone(), out_a.clone()], vec![0.0; 2 * 3])
                    .unwrap();
            let factor_2 =
                IdxTensor::from_dense(vec![shared.clone(), out_b.clone()], vec![0.0; 2 * 3])
                    .unwrap();

            let result = uncontracted_indices(&[&factor_1, &factor_2], &HashSet::new());
            assert_eq!(result, vec![out_a.clone(), out_b.clone()]);
        }
    }

    fn three_node_path(offset: f64) -> TreeTN<IdxTensor, String> {
        let dim = 3usize;
        let ab = DynIndex::new_dyn(dim);
        let bc = DynIndex::new_dyn(dim);
        let a_out = DynIndex::new_dyn(dim);
        let b_out = DynIndex::new_dyn(dim);
        let c_out = DynIndex::new_dyn(dim);
        let a = IdxTensor::from_dense(
            vec![a_out, ab.clone()],
            (0..dim * dim)
                .map(|i| offset + f64::from(i as i32) / 10.0)
                .collect(),
        )
        .unwrap();
        let b = IdxTensor::from_dense(
            vec![ab, b_out, bc.clone()],
            (0..dim * dim * dim)
                .map(|i| offset + f64::from(i as i32) / 11.0)
                .collect(),
        )
        .unwrap();
        let c = IdxTensor::from_dense(
            vec![bc, c_out],
            (0..dim * dim)
                .map(|i| offset + f64::from(i as i32) / 12.0)
                .collect(),
        )
        .unwrap();
        TreeTN::from_tensors(vec![a, b, c], vec!["A".into(), "B".into(), "C".into()]).unwrap()
    }

    #[test]
    fn request_grows_a_fresh_segment_and_re_reads_an_aligned_one() {
        let tn_a = three_node_path(1.0);
        let tn_b = three_node_path(2.0);
        let mut nodes = tn_a.node_names();
        nodes.sort();
        let edges = tn_a
            .edges_to_canonicalize_by_names(&"B".to_string())
            .unwrap();

        let local_values = local_site_pairs(&tn_a, &tn_b, &nodes).unwrap();
        let local = nodes
            .iter()
            .cloned()
            .zip(local_values)
            .collect::<std::collections::HashMap<_, _>>();
        let outputs = nodes
            .iter()
            .map(|node| {
                (
                    node.clone(),
                    local_output_indices(&tn_a, &tn_b, node).unwrap(),
                )
            })
            .collect::<std::collections::HashMap<_, _>>();
        let mut probe_indices = nodes
            .iter()
            .flat_map(|node| outputs[node].iter().cloned())
            .collect::<Vec<_>>();
        let mut probes = ProbeBank::new(std::mem::take(&mut probe_indices), 1, 99).unwrap();
        let mut cache = EnvironmentCache::new(&tn_a, &edges, &nodes, &local, &outputs, &mut probes);

        let (first, first_batch) = cache
            .request(&"B".to_string(), &"A".to_string(), 0, 3)
            .unwrap();
        assert_eq!(first_batch.dim(), 3);

        let (again, again_batch) = cache
            .request(&"B".to_string(), &"A".to_string(), 0, 3)
            .unwrap();
        assert_eq!(again_batch.dim(), first_batch.dim());
        let difference = again.sub(&first).unwrap().maxabs().unwrap();
        assert!(
            difference < 1e-12,
            "re-read segment tensor should exactly match the grown one, got difference {difference}"
        );
        assert_eq!(
            cache.segments.len(),
            1,
            "re-reading an aligned segment must not grow a new one"
        );
    }
}
