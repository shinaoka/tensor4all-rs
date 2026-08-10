use crate::error::Result as TreeTciResult;
use crate::{assemble::MultiIndex, column_2d, ncols_2d, SubtreeKey, TreeTCI2, TreeTciEdge};
use anyhow::{ensure, Result};
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use tensor4all_core::ColMajorArray;

/// Generates candidate pivot sets for one edge bipartition.
///
/// Implementors produce candidate multi-indices for both sides of an edge
/// bipartition. The optimizer evaluates the function at these candidates to
/// select new pivots.
///
/// Built-in proposers:
/// - [`DefaultProposer`] -- neighbor-product candidates (recommended default)
/// - [`SimpleProposer`] -- random candidates with deterministic seed
/// - [`TruncatedDefaultProposer`] -- truncated default candidates with random sampling
pub trait PivotCandidateProposer {
    /// Return `(I_candidates, J_candidates)` for the requested edge.
    ///
    /// `I_candidates` are multi-indices for the left (u-side) subtree,
    /// `J_candidates` for the right (v-side) subtree.
    /// # Errors
    ///
    /// Returns an error when the construction or conversion fails (a shape or
    /// /// index mismatch, or a backend failure).
    ///
    fn candidates<T>(
        &self,
        state: &TreeTCI2<T>,
        edge: TreeTciEdge,
    ) -> TreeTciResult<(Vec<MultiIndex>, Vec<MultiIndex>)>;
}

/// Default neighbor-product proposer that mirrors `TreeTCI.jl`.
///
/// Generates candidates by combining existing pivots from adjacent
/// subtrees with a Kronecker expansion over the local dimension of
/// the edge endpoints. This is the recommended proposer for most use cases.
///
/// # Examples
///
/// ```
/// use tensor4all_treetci::DefaultProposer;
///
/// let proposer = DefaultProposer;
/// // Typically used with crossinterpolate2 or optimize_with_proposer
/// ```
#[derive(Clone, Copy, Debug, Default)]
pub struct DefaultProposer;

impl PivotCandidateProposer for DefaultProposer {
    /// # Errors
    ///
    /// Returns an error when the construction or conversion fails (a shape or
    /// /// index mismatch, or a backend failure).
    ///
    fn candidates<T>(
        &self,
        state: &TreeTCI2<T>,
        edge: TreeTciEdge,
    ) -> TreeTciResult<(Vec<MultiIndex>, Vec<MultiIndex>)> {
        let (vp, vq) = state.graph.separate_vertices(edge)?;
        let (ikey, jkey) = state.graph.subregion_vertices(edge)?;

        let adjacent_vp = state.graph.adjacent_edges(vp, &[edge]);
        let in_ikeys = state.graph.edge_in_ij_keys(vp, &adjacent_vp)?;
        let ipivots = pivot_set(&state.ijset, &in_ikeys, &ikey)?;
        let isite_index = subtree_position(&ikey, vp)?;
        let iset = kronecker(&ipivots, isite_index, state.local_dims[vp]);

        let adjacent_vq = state.graph.adjacent_edges(vq, &[edge]);
        let in_jkeys = state.graph.edge_in_ij_keys(vq, &adjacent_vq)?;
        let jpivots = pivot_set(&state.ijset, &in_jkeys, &jkey)?;
        let jsite_index = subtree_position(&jkey, vq)?;
        let jset = kronecker(&jpivots, jsite_index, state.local_dims[vq]);

        let history = state.ijset_history.last();
        let icombined = union_with_history(iset, history, &ikey)?;
        let jcombined = union_with_history(jset, history, &jkey)?;
        Ok((icombined, jcombined))
    }
}

/// Simple random proposer that mirrors `TreeTCI.jl`'s
/// `SimplePivotCandidateProposer`.
///
/// Generates random candidate multi-indices using a deterministic seed.
/// Useful for reproducible benchmarking or when the default proposer
/// produces too many candidates.
///
/// # Examples
///
/// ```
/// use tensor4all_treetci::SimpleProposer;
///
/// // Default seed (0)
/// let p = SimpleProposer::default();
///
/// // Deterministic seed for reproducibility
/// let p = SimpleProposer::seeded(42);
/// ```
#[derive(Clone, Copy, Debug)]
pub struct SimpleProposer {
    seed: u64,
}

impl SimpleProposer {
    /// Construct a proposer with a deterministic base seed.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_treetci::SimpleProposer;
    ///
    /// let proposer = SimpleProposer::seeded(123);
    /// ```
    pub const fn seeded(seed: u64) -> Self {
        Self { seed }
    }
}

impl Default for SimpleProposer {
    fn default() -> Self {
        Self::seeded(0)
    }
}

impl PivotCandidateProposer for SimpleProposer {
    /// # Errors
    ///
    /// Returns an error when the construction or conversion fails (a shape or
    /// /// index mismatch, or a backend failure).
    ///
    fn candidates<T>(
        &self,
        state: &TreeTCI2<T>,
        edge: TreeTciEdge,
    ) -> TreeTciResult<(Vec<MultiIndex>, Vec<MultiIndex>)> {
        let (vp, vq) = state.graph.separate_vertices(edge)?;
        let (ikey, jkey) = state.graph.subregion_vertices(edge)?;
        let mut rng = rng_for_edge(state, edge, self.seed, "simple")?;

        let ichi =
            state.local_dims[vp]
                * ncols_2d(state.ijset.get(&ikey).ok_or_else(|| {
                    anyhow::anyhow!("missing pivot set for subtree key {:?}", ikey)
                })?)?;
        let jchi =
            state.local_dims[vq]
                * ncols_2d(state.ijset.get(&jkey).ok_or_else(|| {
                    anyhow::anyhow!("missing pivot set for subtree key {:?}", jkey)
                })?)?;

        let iset = random_candidates(&mut rng, state.local_dims.as_slice(), &ikey, ichi);
        let jset = random_candidates(&mut rng, state.local_dims.as_slice(), &jkey, jchi);

        let history = state.ijset_history.last();
        let icombined = union_with_history(iset, history, &ikey)?;
        let jcombined = union_with_history(jset, history, &jkey)?;
        Ok((icombined, jcombined))
    }
}

/// Truncated default proposer that samples an ordered subset from the default
/// candidate set, mirroring `TreeTCI.jl`'s
/// `TruncatedDefaultPivotCandidateProposer`.
///
/// Starts from the [`DefaultProposer`] candidates but truncates them to a
/// bounded size using random sampling. Useful for large problems where the
/// default candidate set would be prohibitively large.
///
/// # Examples
///
/// ```
/// use tensor4all_treetci::TruncatedDefaultProposer;
///
/// let proposer = TruncatedDefaultProposer::seeded(42);
/// ```
#[derive(Clone, Copy, Debug)]
pub struct TruncatedDefaultProposer {
    seed: u64,
}

impl TruncatedDefaultProposer {
    /// Construct a proposer with a deterministic base seed.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_treetci::TruncatedDefaultProposer;
    ///
    /// let proposer = TruncatedDefaultProposer::seeded(99);
    /// ```
    pub const fn seeded(seed: u64) -> Self {
        Self { seed }
    }
}

impl Default for TruncatedDefaultProposer {
    fn default() -> Self {
        Self::seeded(0)
    }
}

impl PivotCandidateProposer for TruncatedDefaultProposer {
    /// # Errors
    ///
    /// Returns an error when the construction or conversion fails (a shape or
    /// /// index mismatch, or a backend failure).
    ///
    fn candidates<T>(
        &self,
        state: &TreeTCI2<T>,
        edge: TreeTciEdge,
    ) -> TreeTciResult<(Vec<MultiIndex>, Vec<MultiIndex>)> {
        let (vp, vq) = state.graph.separate_vertices(edge)?;
        let (ikey, jkey) = state.graph.subregion_vertices(edge)?;
        let (default_i, default_j) = DefaultProposer.candidates(state, edge)?;
        let mut rng = rng_for_edge(state, edge, self.seed, "truncated_default")?;

        let ichi =
            state.local_dims[vp]
                * ncols_2d(state.ijset.get(&ikey).ok_or_else(|| {
                    anyhow::anyhow!("missing pivot set for subtree key {:?}", ikey)
                })?)?;
        let jchi =
            state.local_dims[vq]
                * ncols_2d(state.ijset.get(&jkey).ok_or_else(|| {
                    anyhow::anyhow!("missing pivot set for subtree key {:?}", jkey)
                })?)?;

        Ok((
            sample_ordered_candidates(&default_i, ichi, &mut rng),
            sample_ordered_candidates(&default_j, jchi, &mut rng),
        ))
    }
}

fn subtree_position(key: &SubtreeKey, site: usize) -> Result<usize> {
    key.as_slice()
        .iter()
        .position(|&value| value == site)
        .ok_or_else(|| anyhow::anyhow!("site {} not found in subtree key {:?}", site, key))
}

fn union_with_history(
    values: Vec<MultiIndex>,
    history: Option<&HashMap<SubtreeKey, ColMajorArray<usize>>>,
    key: &SubtreeKey,
) -> Result<Vec<MultiIndex>> {
    let mut unique = Vec::with_capacity(values.len());
    for candidate in values {
        if !unique.contains(&candidate) {
            unique.push(candidate);
        }
    }
    if let Some(arr) = history.and_then(|history| history.get(key)) {
        for j in 0..ncols_2d(arr)? {
            let col = column_2d(arr, j)?.to_vec();
            if !unique.contains(&col) {
                unique.push(col);
            }
        }
    }
    Ok(unique)
}

fn pivot_set(
    ijset: &HashMap<SubtreeKey, ColMajorArray<usize>>,
    in_keys: &[SubtreeKey],
    out_key: &SubtreeKey,
) -> Result<Vec<MultiIndex>> {
    let out_len = out_key.as_slice().len();
    let mut pivots = vec![vec![0; out_len]];

    for in_key in in_keys {
        let incoming = ijset
            .get(in_key)
            .ok_or_else(|| anyhow::anyhow!("missing pivot set for subtree key {:?}", in_key))?;
        let mut next = Vec::new();
        for base in &pivots {
            for j in 0..ncols_2d(incoming)? {
                let index = column_2d(incoming, j)?;
                ensure!(
                    index.len() == in_key.as_slice().len(),
                    "pivot length {} does not match subtree key length {}",
                    index.len(),
                    in_key.as_slice().len()
                );
                let mut merged = base.clone();
                for (&site, &value) in in_key.as_slice().iter().zip(index.iter()) {
                    let out_pos = subtree_position(out_key, site)?;
                    merged[out_pos] = value;
                }
                next.push(merged);
            }
        }
        pivots = next;
    }

    Ok(pivots)
}

fn kronecker(pivots: &[MultiIndex], site_index: usize, local_dim: usize) -> Vec<MultiIndex> {
    let mut result = Vec::with_capacity(pivots.len() * local_dim);
    for pivot in pivots {
        for value in 0..local_dim {
            let mut candidate = pivot.clone();
            candidate[site_index] = value;
            result.push(candidate);
        }
    }
    result
}

fn random_candidates(
    rng: &mut SmallRng,
    local_dims: &[usize],
    key: &SubtreeKey,
    size: usize,
) -> Vec<MultiIndex> {
    (0..size)
        .map(|_| {
            key.as_slice()
                .iter()
                .map(|&site| rng.random_range(0..local_dims[site]))
                .collect()
        })
        .collect()
}

fn rng_for_edge<T>(
    state: &TreeTCI2<T>,
    edge: TreeTciEdge,
    seed: u64,
    tag: &str,
) -> Result<SmallRng> {
    let (ikey, jkey) = state.graph.subregion_vertices(edge)?;
    let mut hasher = DefaultHasher::new();
    seed.hash(&mut hasher);
    tag.hash(&mut hasher);
    edge.hash(&mut hasher);
    state.ijset_history.len().hash(&mut hasher);
    state
        .ijset
        .get(&ikey)
        .map(ncols_2d)
        .transpose()?
        .unwrap_or(0)
        .hash(&mut hasher);
    state
        .ijset
        .get(&jkey)
        .map(ncols_2d)
        .transpose()?
        .unwrap_or(0)
        .hash(&mut hasher);
    Ok(SmallRng::seed_from_u64(hasher.finish()))
}

fn sample_ordered_candidates(
    candidates: &[MultiIndex],
    max_size: usize,
    rng: &mut SmallRng,
) -> Vec<MultiIndex> {
    if candidates.len() <= max_size {
        return candidates.to_vec();
    }

    let mut selected_indices = (0..candidates.len()).collect::<Vec<_>>();
    selected_indices.shuffle(rng);
    selected_indices.truncate(max_size);
    selected_indices.sort_unstable();
    selected_indices
        .into_iter()
        .map(|index| candidates[index].clone())
        .collect()
}

#[cfg(test)]
mod tests;
