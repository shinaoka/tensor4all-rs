//! Gaussian probe generation and column-major probe batches for SRC.
//!
//! Provenance: the Khatri--Rao construction and the `A * Omega` sketch are
//! derived from Algorithm 1 and Section 2.2 of Camaño--Epperly--Tropp,
//! [arXiv:2504.06475](https://arxiv.org/abs/2504.06475), especially the
//! `Omega^(1) odot ... odot Omega^(n)` definition. The incremental column and
//! environment conventions were cross-checked against
//! `chriscamano/RandomMPOMPS/code/tensornetwork/contraction.py`,
//! `random_contraction` (lines 82--353) and `random_contraction_inc`
//! (lines 357--593). The separate A/B probe partition is required by
//! tensor4all-rs issue #563 comment 5396107820. Batch retention, deterministic
//! seeding, and error handling are repository engineering choices; the audit
//! marks unsupported choices `[AI-Supplied]`.

use anyhow::Result;
use rand::Rng;
#[cfg(test)]
use rand::SeedableRng;
#[cfg(test)]
use rand_chacha::ChaCha8Rng;
use std::collections::{HashMap, HashSet};
use std::hash::Hash;
use tensor4all_core::{IndexLike, TensorLike};

use super::{SrcOptions, TreeTN};
use crate::algorithm::CanonicalForm;

/// Reusable Gaussian probe columns indexed by physical tensor legs.
///
/// For each index, coefficients are stored as a column-major `dim × width`
/// matrix. Extending the bank advances one persistent RNG and appends columns,
/// so an adaptive run observes exactly the same prefix as a fixed-width run
/// with the same seed and index order.
pub(super) struct ProbeBank<I, R> {
    indices: Vec<I>,
    coefficients: HashMap<I, Vec<f64>>,
    rng: R,
    width: usize,
}

#[cfg(test)]
impl<I> ProbeBank<I, ChaCha8Rng>
where
    I: IndexLike,
{
    #[cfg(test)]
    pub(super) fn from_seed(indices: Vec<I>, width: usize, seed: u64) -> Result<Self> {
        Self::new(indices, width, ChaCha8Rng::seed_from_u64(seed))
    }
}

impl<I, R> ProbeBank<I, R>
where
    I: IndexLike,
    R: Rng,
{
    /// Construct a bank with `width` Gaussian columns from `rng`.
    pub(super) fn new(indices: Vec<I>, width: usize, rng: R) -> Result<Self> {
        if width == 0 {
            anyhow::bail!("SRC probe bank width must be at least 1");
        }
        let mut seen = HashSet::with_capacity(indices.len());
        let mut coefficients = HashMap::with_capacity(indices.len());
        for index in &indices {
            if index.dim() == 0 {
                anyhow::bail!("SRC probe index {:?} has zero dimension", index);
            }
            if !seen.insert(index.clone()) {
                anyhow::bail!("SRC probe index {:?} occurs more than once", index);
            }
            let capacity = index
                .dim()
                .checked_mul(width)
                .ok_or_else(|| anyhow::anyhow!("SRC probe bank size overflow"))?;
            coefficients.insert(index.clone(), Vec::with_capacity(capacity));
        }

        let mut bank = Self {
            indices,
            coefficients,
            rng,
            width: 0,
        };
        bank.extend_to(width)?;
        Ok(bank)
    }
}

impl<I, R> ProbeBank<I, R>
where
    I: IndexLike,
{
    /// Return the number of columns currently stored in the bank.
    pub(super) fn width(&self) -> usize {
        self.width
    }

    /// Return the column-major coefficient matrix for one physical index.
    pub(super) fn coefficients(&self, index: &I) -> Option<&[f64]> {
        self.coefficients.get(index).map(Vec::as_slice)
    }

    /// Return one Gaussian probe column for an index.
    pub(super) fn column(&self, index: &I, column: usize) -> Result<&[f64]> {
        if column >= self.width {
            anyhow::bail!(
                "SRC probe column {} is outside width {}",
                column,
                self.width
            );
        }
        let values = self
            .coefficients
            .get(index)
            .ok_or_else(|| anyhow::anyhow!("SRC probe index is missing from bank"))?;
        let start = column
            .checked_mul(index.dim())
            .ok_or_else(|| anyhow::anyhow!("SRC probe column offset overflow"))?;
        let end = start
            .checked_add(index.dim())
            .ok_or_else(|| anyhow::anyhow!("SRC probe column end overflow"))?;
        values
            .get(start..end)
            .ok_or_else(|| anyhow::anyhow!("SRC probe column storage is inconsistent"))
    }
}

impl<I, R> ProbeBank<I, R>
where
    I: IndexLike,
    R: Rng,
{
    /// Append Gaussian columns until the bank reaches `target_width`.
    pub(super) fn extend_to(&mut self, target_width: usize) -> Result<()> {
        if target_width <= self.width {
            return Ok(());
        }

        for _ in self.width..target_width {
            for index in &self.indices {
                let values = self
                    .coefficients
                    .get_mut(index)
                    .ok_or_else(|| anyhow::anyhow!("SRC probe index is missing from bank"))?;
                values.extend((0..index.dim()).map(|_| standard_normal(&mut self.rng)));
            }
        }
        self.width = target_width;
        Ok(())
    }
}

fn standard_normal<R: Rng + ?Sized>(rng: &mut R) -> f64 {
    let u1 = rng.random::<f64>().max(f64::MIN_POSITIVE);
    let u2 = rng.random::<f64>();
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

fn single_probe<T, R>(index: &T::Index, probes: &ProbeBank<T::Index, R>, column: usize) -> Result<T>
where
    T: TensorLike,
    T::Index: IndexLike + Clone + Hash + Eq,
{
    let data = probes.column(index, column)?.to_vec();
    T::from_dense(vec![index.clone()], data)
        .map_err(|error| anyhow::anyhow!("contract_src: probe construction failed: {error}"))
}

fn single_probe_batch<T, R>(
    index: &T::Index,
    probes: &ProbeBank<T::Index, R>,
    first_column: usize,
    width: usize,
    batch: &T::Index,
) -> Result<T>
where
    T: TensorLike,
    T::Index: IndexLike + Clone + Hash + Eq,
{
    let capacity = index
        .dim()
        .checked_mul(width)
        .ok_or_else(|| anyhow::anyhow!("contract_src: probe batch size overflow"))?;
    let end_column = first_column
        .checked_add(width)
        .ok_or_else(|| anyhow::anyhow!("contract_src: probe batch column range overflow"))?;
    let mut data = Vec::with_capacity(capacity);
    for column in first_column..end_column {
        data.extend_from_slice(probes.column(index, column)?);
    }
    T::from_dense(vec![index.clone(), batch.clone()], data)
        .map_err(|error| anyhow::anyhow!("contract_src: probe batch construction failed: {error}"))
}

/// Build the factorized probe tensors for one batch of SRC columns.
///
/// Each returned tensor carries one physical output index and the shared
/// `batch` index. Keeping these tensors separate lets a tree-message
/// contraction eliminate incoming branch bonds before joining the two local
/// operands, instead of materializing their full virtual-bond product.
pub(super) fn probe_batch_tensors<T, R>(
    outputs: &[T::Index],
    probes: &ProbeBank<T::Index, R>,
    first_column: usize,
    width: usize,
    batch: &T::Index,
) -> Result<Vec<T>>
where
    T: TensorLike,
    T::Index: IndexLike + Clone + Hash + Eq,
{
    if width == 0 || batch.dim() != width {
        anyhow::bail!(
            "contract_src: probe batch dimension {} does not match width {}",
            batch.dim(),
            width
        );
    }
    outputs
        .iter()
        .map(|index| single_probe_batch::<T, R>(index, probes, first_column, width, batch))
        .collect()
}

/// Return the checked product of a list of index dimensions.
pub(super) fn product_dim<I: IndexLike>(indices: &[I]) -> Result<usize> {
    indices.iter().try_fold(1usize, |size, index| {
        size.checked_mul(index.dim())
            .ok_or_else(|| anyhow::anyhow!("contract_src: output dimension overflow"))
    })
}

/// Contract two local operands with one independent probe per surviving leg.
///
/// Keeping the probes as separate one-index tensors is the factorized
/// MPO--MPO path from the paper: the shared physical leg is contracted between
/// the operands without first constructing a fused `d^2` local product.
pub(super) fn probed_site_pair<T, R>(
    tensor_a: &T,
    tensor_b: &T,
    outputs: &[T::Index],
    probes: &ProbeBank<T::Index, R>,
    column: usize,
) -> Result<T>
where
    T: TensorLike,
    T::Index: IndexLike + Clone + Hash + Eq,
{
    let probe_tensors = outputs
        .iter()
        .map(|index| single_probe::<T, R>(index, probes, column))
        .collect::<Result<Vec<_>>>()?;
    let (a_probes, b_probes) = partition_probes(tensor_a, tensor_b, outputs, &probe_tensors)?;
    let probed_a = contract_operand_with_probes(tensor_a, &a_probes, None)?;
    let probed_b = contract_operand_with_probes(tensor_b, &b_probes, None)?;
    T::contract(&[&probed_a, &probed_b]).map_err(|error| {
        anyhow::anyhow!("contract_src: factorized probe contraction failed: {error}")
    })
}

/// Contract two local operands with a block of factorized probes.
pub(super) fn probed_site_pair_batch_range<T, R>(
    tensor_a: &T,
    tensor_b: &T,
    outputs: &[T::Index],
    probes: &ProbeBank<T::Index, R>,
    first_column: usize,
    width: usize,
    batch: &T::Index,
) -> Result<T>
where
    T: TensorLike,
    T::Index: IndexLike + Clone + Hash + Eq,
{
    if width == 0 || batch.dim() != width {
        anyhow::bail!(
            "contract_src: probe batch dimension {} does not match width {}",
            batch.dim(),
            width
        );
    }
    if outputs.is_empty() {
        let local = T::contract(&[tensor_a, tensor_b]).map_err(|error| {
            anyhow::anyhow!("contract_src: scalar local pair contraction failed: {error}")
        })?;
        let batch_values = T::ones(std::slice::from_ref(batch)).map_err(|error| {
            anyhow::anyhow!("contract_src: scalar probe batch construction failed: {error}")
        })?;
        return local.outer_product(&batch_values).map_err(|error| {
            anyhow::anyhow!("contract_src: scalar probe batch broadcast failed: {error}")
        });
    }
    let probe_tensors = probe_batch_tensors(outputs, probes, first_column, width, batch)?;
    let (a_probes, b_probes) = partition_probes(tensor_a, tensor_b, outputs, &probe_tensors)?;
    let probed_a = contract_operand_with_probes(tensor_a, &a_probes, Some(batch))?;
    let probed_b = contract_operand_with_probes(tensor_b, &b_probes, Some(batch))?;
    contract_retaining(&[&probed_a, &probed_b], batch)
}

/// Contract a left environment into an MPO-MPO site pair before joining the
/// two local operands.
// INVARIANT: this is the unbatched counterpart of
// `contract_prefix_with_probed_site_pair_batch_range`; the ordering mirrors
// the reference `env @ psi[j]` then `H[j] @ ...` contractions and avoids
// materializing both local virtual bonds before the left environment reduces
// them.
pub(super) fn contract_prefix_with_site_pair<T>(prefix: &T, tensor_a: &T, tensor_b: &T) -> Result<T>
where
    T: TensorLike,
{
    let after_a = T::contract(&[prefix, tensor_a])
        .map_err(|error| anyhow::anyhow!("contract_src: prefix-A contraction failed: {error}"))?;
    T::contract(&[&after_a, tensor_b])
        .map_err(|error| anyhow::anyhow!("contract_src: prefix-B contraction failed: {error}"))
}

/// Contract an unbatched left environment into a probed MPO-MPO site pair.
// INVARIANT: this is the single-column counterpart of
// `contract_prefix_with_probed_site_pair_batch_range`. It follows the same
// environment-first ordering while retaining the independent probe vectors
// on the local output legs.
pub(super) fn contract_prefix_with_probed_site_pair<T, R>(
    prefix: &T,
    tensor_a: &T,
    tensor_b: &T,
    outputs: &[T::Index],
    probes: &ProbeBank<T::Index, R>,
    column: usize,
) -> Result<T>
where
    T: TensorLike,
    T::Index: IndexLike + Clone + Hash + Eq,
{
    let probe_tensors = outputs
        .iter()
        .map(|index| single_probe::<T, R>(index, probes, column))
        .collect::<Result<Vec<_>>>()?;
    let (a_probes, b_probes) = partition_probes(tensor_a, tensor_b, outputs, &probe_tensors)?;
    let mut result = T::contract(&[prefix, tensor_a])
        .map_err(|error| anyhow::anyhow!("contract_src: prefix-A contraction failed: {error}"))?;
    result = contract_operand_with_probes(&result, &a_probes, None)?;
    result = T::contract(&[&result, tensor_b])
        .map_err(|error| anyhow::anyhow!("contract_src: prefix-B contraction failed: {error}"))?;
    contract_operand_with_probes(&result, &b_probes, None)
}

/// Contract a factorized local pair with an incoming batched prefix.
// INVARIANT: `prefix`, the two local operands, the probe bank, and the batch
// range are all one contraction step; keeping them explicit prevents the
// factorized MPO--MPO path from hiding a fused physical product.
#[allow(clippy::too_many_arguments)]
pub(super) fn contract_prefix_with_probed_site_pair_batch_range<T, R>(
    prefix: &T,
    tensor_a: &T,
    tensor_b: &T,
    outputs: &[T::Index],
    probes: &ProbeBank<T::Index, R>,
    first_column: usize,
    width: usize,
    batch: &T::Index,
) -> Result<T>
where
    T: TensorLike,
    T::Index: IndexLike + Clone + Hash + Eq,
{
    if width == 0 || batch.dim() != width {
        anyhow::bail!(
            "contract_src: probe batch dimension {} does not match width {}",
            batch.dim(),
            width
        );
    }
    let probe_tensors = probe_batch_tensors(outputs, probes, first_column, width, batch)?;
    let (a_probes, b_probes) = partition_probes(tensor_a, tensor_b, outputs, &probe_tensors)?;

    let mut a_factors = Vec::with_capacity(a_probes.len() + 2);
    a_factors.push(tensor_a);
    a_factors.push(prefix);
    a_factors.extend(a_probes.iter().copied());
    let result = contract_retaining(&a_factors, batch)
        .map_err(|error| anyhow::anyhow!("contract_src: prefix-A contraction failed: {error}"))?;

    let mut b_factors = Vec::with_capacity(b_probes.len() + 2);
    b_factors.push(tensor_b);
    b_factors.push(&result);
    b_factors.extend(b_probes.iter().copied());
    contract_retaining(&b_factors, batch)
        .map_err(|error| anyhow::anyhow!("contract_src: prefix-B contraction failed: {error}"))
}

/// Partition local probes by the MPO operand carrying their external index.
///
/// This explicit partition is important for MPO--MPO SRC. It makes the
/// factorized contraction order visible: each probe is contracted into its own
/// operand before the two operands are contracted along their shared physical
/// and virtual legs.
fn partition_probes<'a, T>(
    tensor_a: &T,
    tensor_b: &T,
    outputs: &[T::Index],
    probes: &'a [T],
) -> Result<(Vec<&'a T>, Vec<&'a T>)>
where
    T: TensorLike,
    T::Index: IndexLike + Clone + Hash + Eq,
{
    if outputs.len() != probes.len() {
        anyhow::bail!(
            "contract_src: {} local outputs but {} local probes",
            outputs.len(),
            probes.len()
        );
    }
    let a_indices = tensor_a.external_indices();
    let b_indices = tensor_b.external_indices();
    let mut a_probes = Vec::new();
    let mut b_probes = Vec::new();
    for (index, probe) in outputs.iter().zip(probes) {
        let in_a = a_indices.iter().any(|candidate| candidate == index);
        let in_b = b_indices.iter().any(|candidate| candidate == index);
        match (in_a, in_b) {
            (true, false) => a_probes.push(probe),
            (false, true) => b_probes.push(probe),
            (false, false) => anyhow::bail!(
                "contract_src: local output index {:?} is absent from both MPO operands",
                index
            ),
            (true, true) => anyhow::bail!(
                "contract_src: local output index {:?} is shared by both MPO operands",
                index
            ),
        }
    }
    Ok((a_probes, b_probes))
}

/// Contract one MPO operand with its probes while retaining a shared batch
/// index when batched probes are used.
fn contract_operand_with_probes<T>(
    operand: &T,
    probes: &[&T],
    batch: Option<&T::Index>,
) -> Result<T>
where
    T: TensorLike,
    T::Index: IndexLike + Clone + Hash + Eq,
{
    let Some(batch) = batch else {
        if probes.is_empty() {
            return Ok(operand.clone());
        }
        let mut factors = Vec::with_capacity(probes.len() + 1);
        factors.push(operand);
        factors.extend(probes.iter().copied());
        return T::contract(&factors).map_err(|error| {
            anyhow::anyhow!("contract_src: operand probe contraction failed: {error}")
        });
    };

    let mut result = operand.clone();
    for probe in probes {
        result = contract_retaining(&[&result, *probe], batch)?;
    }
    Ok(result)
}

/// Contract two local operands with additional tensor-network factors.
pub(super) fn contract_site_pair<T>(tensor_a: &T, tensor_b: &T, extra: &[&T]) -> Result<T>
where
    T: TensorLike,
{
    let mut factors = Vec::with_capacity(2 + extra.len());
    factors.push(tensor_a);
    factors.push(tensor_b);
    factors.extend(extra.iter().copied());
    T::contract(&factors)
        .map_err(|error| anyhow::anyhow!("contract_src: local pair contraction failed: {error}"))
}

/// Contract tensors while preserving a shared batch index.
pub(super) fn contract_retaining<T>(tensors: &[&T], batch: &T::Index) -> Result<T>
where
    T: TensorLike,
    T::Index: IndexLike + Clone + Hash + Eq,
{
    let result = T::contract_retaining_indices(tensors, std::slice::from_ref(batch))
        .map_err(|error| anyhow::anyhow!("contract_src: batched contraction failed: {error}"))?;
    let indices = result.external_indices();
    if indices.last() == Some(batch) {
        return Ok(result);
    }
    let mut trailing = indices
        .into_iter()
        .filter(|index| index != batch)
        .collect::<Vec<_>>();
    trailing.push(batch.clone());
    result
        .permuteinds(&trailing)
        .map_err(|error| anyhow::anyhow!("contract_src: batch-axis permutation failed: {error}"))
}

/// Contract corresponding tensors at every named site once.
pub(super) fn site_operands<'a, T, V>(
    tn_a: &'a TreeTN<T, V>,
    tn_b: &'a TreeTN<T, V>,
    node: &V,
) -> Result<(&'a T, &'a T)>
where
    T: TensorLike,
    V: Clone + Hash + Eq + Send + Sync + std::fmt::Debug,
{
    let node_a = tn_a
        .node_index(node)
        .ok_or_else(|| anyhow::anyhow!("contract_src: node is missing in A"))?;
    let node_b = tn_b
        .node_index(node)
        .ok_or_else(|| anyhow::anyhow!("contract_src: node is missing in B"))?;
    let tensor_a = tn_a
        .tensor(node_a)
        .ok_or_else(|| anyhow::anyhow!("contract_src: tensor is missing in A"))?;
    let tensor_b = tn_b
        .tensor(node_b)
        .ok_or_else(|| anyhow::anyhow!("contract_src: tensor is missing in B"))?;
    Ok((tensor_a, tensor_b))
}

pub(super) fn local_site_pairs<'a, T, V>(
    tn_a: &'a TreeTN<T, V>,
    tn_b: &'a TreeTN<T, V>,
    nodes: &[V],
) -> Result<Vec<(&'a T, &'a T)>>
where
    T: TensorLike,
    V: Clone + Hash + Eq + Send + Sync + std::fmt::Debug,
{
    nodes
        .iter()
        .map(|node| site_operands(tn_a, tn_b, node))
        .collect()
}

/// Return the physical output indices of one local product tensor.
pub(super) fn local_output_indices<T, V>(
    tn_a: &TreeTN<T, V>,
    tn_b: &TreeTN<T, V>,
    node: &V,
) -> Result<Vec<T::Index>>
where
    T: TensorLike,
    T::Index: IndexLike + Clone + Hash + Eq,
    <T::Index as IndexLike>::Id: Ord,
    V: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug,
{
    let mut bonds = HashSet::new();
    for neighbor in tn_a.site_index_network().neighbors(node) {
        let edge = tn_a
            .edge_between(node, &neighbor)
            .ok_or_else(|| anyhow::anyhow!("contract_src: A edge is missing"))?;
        let bond = tn_a
            .bond_index(edge)
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("contract_src: A bond is missing"))?;
        bonds.insert(bond);
    }
    for neighbor in tn_b.site_index_network().neighbors(node) {
        let edge = tn_b
            .edge_between(node, &neighbor)
            .ok_or_else(|| anyhow::anyhow!("contract_src: B edge is missing"))?;
        let bond = tn_b
            .bond_index(edge)
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("contract_src: B bond is missing"))?;
        bonds.insert(bond);
    }
    let (tensor_a, tensor_b) = site_operands(tn_a, tn_b, node)?;
    // Order comes from each tensor's own `external_indices()` (fixed at
    // construction), not from `HashSet`/`sort_indices_deterministic`
    // iteration order: same-dimension legs are common, and
    // `sort_indices_deterministic` tie-breaks those by `Index::id()`, which
    // is drawn from a per-process, unseeded RNG (`generate_id()` in
    // tensor4all-core). That made this function's output order -- and thus
    // which physical leg `ProbeBank` assigns which Gaussian sketch column to
    // -- non-deterministic across process runs despite a fixed SRC seed. See
    // docs/worklogs/2026-08-30-src-probe-order-nondeterminism.md. `HashSet`s
    // below are only used for O(1) `contains` probes.
    let a_indices = tensor_a.external_indices();
    let b_indices = tensor_b.external_indices();
    let a_external: HashSet<_> = a_indices
        .iter()
        .filter(|index| !bonds.contains(*index))
        .cloned()
        .collect();
    let b_external: HashSet<_> = b_indices
        .iter()
        .filter(|index| !bonds.contains(*index))
        .cloned()
        .collect();
    let outputs = a_indices
        .into_iter()
        .filter(|index| a_external.contains(index) && !b_external.contains(index))
        .chain(
            b_indices
                .into_iter()
                .filter(|index| b_external.contains(index) && !a_external.contains(index)),
        )
        .collect::<Vec<_>>();
    Ok(outputs)
}

/// Select the fixed-width probe count used by the chain/tree schedule.
pub(super) fn fixed_probe_width(
    target_rank: usize,
    largest_output_dim: usize,
    final_svd: bool,
) -> usize {
    let requested = if final_svd {
        target_rank
            .saturating_mul(3)
            .div_ceil(2)
            .max(target_rank.saturating_add(10))
    } else {
        target_rank
    };
    requested.min(largest_output_dim)
}

/// Select the maximum sketch width for one local row space.
pub(super) fn maximum_site_width(
    target_rank: usize,
    row_dimension: usize,
    cut_dimension: usize,
    src_options: &SrcOptions,
) -> usize {
    if let Some(max_rank) = src_options.max_rank {
        max_rank.min(row_dimension).min(cut_dimension)
    } else {
        fixed_probe_width(target_rank, row_dimension, src_options.final_svd).min(cut_dimension)
    }
}

/// Return the initial adaptive probe width for a local sketch.
pub(super) fn initial_width(maximum_width: usize, src_options: &SrcOptions) -> usize {
    src_options.min_rank.min(maximum_width).max(1)
}

/// Batch-native adaptive sketch growth loop: `make_batch` receives
/// `(start, width)` and returns one batch-indexed tensor covering exactly
/// `[start, start + width)`, instead of being asked for individual columns
/// one at a time. See
/// `docs/superpowers/specs/2026-08-30-src-adaptive-batch-probe-columns-design.md`.
pub(super) fn factorize_probe_batches<T, F>(
    left_indices: &[T::Index],
    initial_width: usize,
    maximum_width: usize,
    src_options: &SrcOptions,
    label: &str,
    mut make_batch: F,
) -> Result<(T, T::Index)>
where
    T: TensorLike,
    T::Index: IndexLike + Clone + Hash + Eq,
    F: FnMut(usize, usize) -> Result<(T, T::Index)>,
{
    if maximum_width == 0 {
        anyhow::bail!("contract_src: {label} has no usable probe columns");
    }
    let mut width = initial_width.min(maximum_width).max(1);
    let mut previous_width = 0;
    let mut previous = None;
    loop {
        let (batch_tensor, batch_index) = make_batch(previous_width, width - previous_width)?;
        let factorized = T::factorize_probe_batch_incremental(
            previous.as_ref(),
            &batch_tensor,
            &batch_index,
            left_indices,
        )
        .map_err(|error| anyhow::anyhow!("contract_src: {label} QR failed: {error}"))?;
        let saturated = factorized.rank < width || width == maximum_width;
        let stop = if src_options.rtol.is_none() || saturated {
            true
        } else {
            match factorized.right.src_error_estimate() {
                Ok(estimate) => {
                    estimate.error
                        <= src_options.atol + src_options.rtol.unwrap_or(0.0) * estimate.norm
                }
                Err(error) => {
                    return Err(anyhow::anyhow!(
                        "contract_src: {label} adaptive estimator unavailable: {error}"
                    ));
                }
            }
        };
        if !stop {
            previous_width = width;
            previous = Some(factorized);
            width = width
                .saturating_add(src_options.rank_increment)
                .min(maximum_width);
            continue;
        }
        return Ok((factorized.left, factorized.bond_index));
    }
}

/// Connect two result tensors through their single shared SRC cap index.
pub(super) fn connect_result_edge<T, V>(
    result: &mut TreeTN<T, V>,
    left: &V,
    right: &V,
) -> Result<()>
where
    T: TensorLike,
    T::Index: IndexLike + Clone + Hash + Eq,
    V: Clone + Hash + Eq + Send + Sync + std::fmt::Debug,
{
    let left_node = result
        .node_index(left)
        .ok_or_else(|| anyhow::anyhow!("contract_src: result left node is missing"))?;
    let right_node = result
        .node_index(right)
        .ok_or_else(|| anyhow::anyhow!("contract_src: result right node is missing"))?;
    let left_tensor = result
        .tensor(left_node)
        .ok_or_else(|| anyhow::anyhow!("contract_src: result left tensor is missing"))?;
    let right_tensor = result
        .tensor(right_node)
        .ok_or_else(|| anyhow::anyhow!("contract_src: result right tensor is missing"))?;
    let common = tensor4all_core::index_ops::common_inds::<T::Index>(
        &left_tensor.external_indices(),
        &right_tensor.external_indices(),
    );
    let bond = common
        .first()
        .ok_or_else(|| anyhow::anyhow!("contract_src: result edge has no connecting bond"))?;
    result.connect_internal(left_node, bond, right_node, bond)?;
    Ok(())
}

/// Record the canonical metadata for factors already produced by SRC QR.
///
/// SRC constructs every non-center factor with a left-canonical QR. Re-running
/// the generic TreeTN canonicalization sweep would perform another QR of the
/// same factors, so this helper only records the verified orientation.
pub(super) fn mark_result_canonical<T, V>(
    result: &mut TreeTN<T, V>,
    center: &V,
    edges: &[(V, V)],
) -> Result<()>
where
    T: TensorLike,
    T::Index: IndexLike + Clone + Hash + Eq,
    V: Clone + Hash + Eq + Send + Sync + std::fmt::Debug,
{
    result
        .set_canonical_region([center.clone()])
        .map_err(|error| anyhow::anyhow!("contract_src: set canonical center failed: {error}"))?;
    for (child, parent) in edges {
        let edge = result
            .edge_between(child, parent)
            .ok_or_else(|| anyhow::anyhow!("contract_src: result canonical edge is missing"))?;
        result
            .set_edge_ortho_towards(edge, Some(parent.clone()))
            .map_err(|error| {
                anyhow::anyhow!("contract_src: set canonical edge direction failed: {error}")
            })?;
    }
    result.canonical_form = Some(CanonicalForm::Unitary);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        contract_prefix_with_probed_site_pair, contract_prefix_with_probed_site_pair_batch_range,
        contract_prefix_with_site_pair, contract_retaining, contract_site_pair,
        local_output_indices, maximum_site_width, probed_site_pair, probed_site_pair_batch_range,
        ProbeBank, TreeTN,
    };
    use crate::treetn::contraction::SrcOptions;
    use tensor4all_core::{DynIndex, IdxTensor, IndexLike};

    /// `local_output_indices` used to order same-dimension output legs by
    /// `Index::id()` (via `HashSet` + `sort_indices_deterministic`), and
    /// `id()` is drawn from a per-process, unseeded RNG -- so on a real
    /// process restart, freshly created `out_x`/`out_y` below would have a
    /// coin-flip chance of swapping places. This test can't literally
    /// restart the process, but it rebuilds `out_x`/`out_y` (and therefore
    /// their random ids) fresh on every iteration: under the old
    /// implementation that gave roughly a 50% chance per iteration of
    /// observing `[out_y, out_x]`, so 200 iterations would fail with
    /// probability `1 - 2^-200`. Under the fix, order always follows
    /// `tensor_a`'s own construction order, regardless of the ids involved.
    #[test]
    fn local_output_indices_orders_same_dimension_legs_by_construction_not_by_random_id() {
        for _ in 0..200 {
            let in_idx = DynIndex::new_dyn(2);
            let out_x = DynIndex::new_dyn(3);
            let out_y = DynIndex::new_dyn(3);

            let tensor_a = IdxTensor::from_dense(
                vec![in_idx.clone(), out_x.clone(), out_y.clone()],
                vec![0.0; 2 * 3 * 3],
            )
            .unwrap();
            let tensor_b = IdxTensor::from_dense(vec![in_idx.clone()], vec![0.0; 2]).unwrap();

            let tn_a = TreeTN::from_tensors(vec![tensor_a], vec!["X".to_string()]).unwrap();
            let tn_b = TreeTN::from_tensors(vec![tensor_b], vec!["X".to_string()]).unwrap();

            let outputs = local_output_indices(&tn_a, &tn_b, &"X".to_string()).unwrap();
            assert_eq!(outputs, vec![out_x, out_y]);
        }
    }

    #[test]
    fn probe_bank_extension_preserves_the_existing_prefix() {
        let first = DynIndex::new_dyn(3);
        let second = DynIndex::new_dyn(2);
        let indices = vec![first.clone(), second.clone()];

        let mut extended = ProbeBank::from_seed(indices.clone(), 2, 17).unwrap();
        let prefix_first = extended.coefficients(&first).unwrap().to_vec();
        let prefix_second = extended.coefficients(&second).unwrap().to_vec();
        extended.extend_to(5).unwrap();

        let reference = ProbeBank::from_seed(indices, 5, 17).unwrap();
        assert_eq!(extended.width(), 5);
        assert_eq!(
            &extended.coefficients(&first).unwrap()[..6],
            &prefix_first[..]
        );
        assert_eq!(
            &extended.coefficients(&second).unwrap()[..4],
            &prefix_second[..]
        );
        assert_eq!(
            extended.coefficients(&first).unwrap(),
            reference.coefficients(&first).unwrap()
        );
        assert_eq!(
            extended.coefficients(&second).unwrap(),
            reference.coefficients(&second).unwrap()
        );
        assert_eq!(first.dim(), 3);
        assert_eq!(second.dim(), 2);
    }

    #[test]
    fn probe_bank_rejects_zero_width_and_zero_dimensional_indices() {
        let index = DynIndex::new_dyn(2);
        assert!(ProbeBank::from_seed(vec![index.clone()], 0, 0).is_err());

        let zero_dimensional = DynIndex::new_dyn(0);
        assert!(ProbeBank::from_seed(vec![zero_dimensional], 1, 0).is_err());
    }

    #[test]
    fn probed_site_pair_contracts_mpo_mpo_outputs_before_pairing_the_physical_leg() {
        let shared = DynIndex::new_dyn(2);
        let output_a = DynIndex::new_dyn(2);
        let output_b = DynIndex::new_dyn(2);
        let tensor_a = IdxTensor::from_dense(
            vec![output_a.clone(), shared.clone()],
            vec![1.0, 2.0, 3.0, 4.0],
        )
        .unwrap();
        let tensor_b =
            IdxTensor::from_dense(vec![shared, output_b.clone()], vec![5.0, 6.0, 7.0, 8.0])
                .unwrap();
        let probes = ProbeBank::from_seed(vec![output_a.clone(), output_b.clone()], 1, 23).unwrap();
        let x = probes.column(&output_a, 0).unwrap();
        let y = probes.column(&output_b, 0).unwrap();
        let a_values = tensor_a.to_vec::<f64>().unwrap();
        let b_values = tensor_b.to_vec::<f64>().unwrap();
        let expected = (0..output_a.dim())
            .map(|s| {
                (0..output_b.dim())
                    .map(|t| {
                        (0..2)
                            .map(|u| {
                                a_values[s + output_a.dim() * u] * b_values[u + 2 * t] * x[s] * y[t]
                            })
                            .sum::<f64>()
                    })
                    .sum::<f64>()
            })
            .sum::<f64>();

        let actual =
            probed_site_pair(&tensor_a, &tensor_b, &[output_a, output_b], &probes, 0).unwrap();
        let value = actual.to_vec::<f64>().unwrap();
        assert_eq!(value.len(), 1);
        assert!((value[0] - expected).abs() < 1.0e-12);
    }

    #[test]
    fn batched_probed_site_pair_keeps_independent_mpo_probes_paired() {
        let shared = DynIndex::new_dyn(2);
        let output_a = DynIndex::new_dyn(2);
        let output_b = DynIndex::new_dyn(2);
        let batch = DynIndex::new_link(2).unwrap();
        let tensor_a = IdxTensor::from_dense(
            vec![output_a.clone(), shared.clone()],
            vec![1.0, 2.0, 3.0, 4.0],
        )
        .unwrap();
        let tensor_b =
            IdxTensor::from_dense(vec![shared, output_b.clone()], vec![5.0, 6.0, 7.0, 8.0])
                .unwrap();
        let probes = ProbeBank::from_seed(vec![output_a.clone(), output_b.clone()], 2, 23).unwrap();
        let actual = probed_site_pair_batch_range(
            &tensor_a,
            &tensor_b,
            &[output_a.clone(), output_b.clone()],
            &probes,
            0,
            2,
            &batch,
        )
        .unwrap();
        assert_eq!(actual.indices(), std::slice::from_ref(&batch));

        let a_values = tensor_a.to_vec::<f64>().unwrap();
        let b_values = tensor_b.to_vec::<f64>().unwrap();
        let expected = (0..2)
            .map(|column| {
                let x = probes.column(&output_a, column).unwrap();
                let y = probes.column(&output_b, column).unwrap();
                (0..output_a.dim())
                    .map(|s| {
                        (0..output_b.dim())
                            .map(|t| {
                                (0..2)
                                    .map(|u| {
                                        a_values[s + output_a.dim() * u]
                                            * b_values[u + 2 * t]
                                            * x[s]
                                            * y[t]
                                    })
                                    .sum::<f64>()
                            })
                            .sum::<f64>()
                    })
                    .sum::<f64>()
            })
            .collect::<Vec<_>>();
        let actual = actual.to_vec::<f64>().unwrap();
        assert_eq!(actual.len(), expected.len());
        assert!(actual
            .iter()
            .zip(expected)
            .all(|(actual, expected)| (actual - expected).abs() < 1.0e-12));
    }

    #[test]
    fn prefix_probe_contraction_matches_local_product_with_optimized_order() {
        let left_a = DynIndex::new_dyn(2);
        let left_b = DynIndex::new_dyn(3);
        let right_a = DynIndex::new_dyn(2);
        let right_b = DynIndex::new_dyn(2);
        let shared = DynIndex::new_dyn(2);
        let output_a = DynIndex::new_dyn(2);
        let output_b = DynIndex::new_dyn(2);
        let batch = DynIndex::new_link(2).unwrap();
        let prefix = IdxTensor::from_dense(
            vec![left_a.clone(), left_b.clone(), batch.clone()],
            (0..12).map(|value| value as f64 + 1.0).collect(),
        )
        .unwrap();
        let tensor_a = IdxTensor::from_dense(
            vec![
                left_a.clone(),
                right_a.clone(),
                shared.clone(),
                output_a.clone(),
            ],
            (0..16).map(|value| value as f64 + 1.0).collect(),
        )
        .unwrap();
        let tensor_b = IdxTensor::from_dense(
            vec![
                left_b.clone(),
                right_b.clone(),
                shared.clone(),
                output_b.clone(),
            ],
            (0..24).map(|value| value as f64 + 2.0).collect(),
        )
        .unwrap();
        let probes = ProbeBank::from_seed(vec![output_a.clone(), output_b.clone()], 2, 31).unwrap();
        let local = probed_site_pair_batch_range(
            &tensor_a,
            &tensor_b,
            &[output_a.clone(), output_b.clone()],
            &probes,
            0,
            2,
            &batch,
        )
        .unwrap();
        let reference = contract_retaining(&[&prefix, &local], &batch).unwrap();
        let actual = contract_prefix_with_probed_site_pair_batch_range(
            &prefix,
            &tensor_a,
            &tensor_b,
            &[output_a.clone(), output_b.clone()],
            &probes,
            0,
            2,
            &batch,
        )
        .unwrap();
        let error = actual.distance(&reference).unwrap();
        assert!(
            error < 1.0e-10,
            "batched prefix contraction error is {error}"
        );

        let prefix_without_batch = prefix.select_indices(&[batch], &[0]).unwrap();
        let reference = contract_site_pair(&tensor_a, &tensor_b, &[&prefix_without_batch]).unwrap();
        let actual =
            contract_prefix_with_site_pair(&prefix_without_batch, &tensor_a, &tensor_b).unwrap();
        assert_eq!(actual.indices(), reference.indices());
        let actual_values = actual.to_vec::<f64>().unwrap();
        let reference_values = reference.to_vec::<f64>().unwrap();
        assert_eq!(actual_values.len(), reference_values.len());
        assert!(actual_values
            .iter()
            .zip(reference_values)
            .all(|(actual, expected)| (actual - expected).abs() < 1.0e-10));

        let local = probed_site_pair(
            &tensor_a,
            &tensor_b,
            &[output_a.clone(), output_b.clone()],
            &probes,
            0,
        )
        .unwrap();
        let reference = contract_site_pair(&prefix_without_batch, &local, &[]).unwrap();
        let actual = contract_prefix_with_probed_site_pair(
            &prefix_without_batch,
            &tensor_a,
            &tensor_b,
            &[output_a, output_b],
            &probes,
            0,
        )
        .unwrap();
        assert_eq!(actual.indices(), reference.indices());
        let actual_values = actual.to_vec::<f64>().unwrap();
        let reference_values = reference.to_vec::<f64>().unwrap();
        assert_eq!(actual_values.len(), reference_values.len());
        assert!(actual_values
            .iter()
            .zip(reference_values)
            .all(|(actual, expected)| (actual - expected).abs() < 1.0e-10));
    }

    #[test]
    fn scalar_probed_site_pair_batch_range_broadcasts_over_the_batch_axis() {
        let batch = DynIndex::new_dyn(3);
        let bank = ProbeBank::from_seed(vec![], 3, 17).unwrap();

        let left = DynIndex::new_dyn(2);
        let shared = DynIndex::new_dyn(1);
        let local =
            IdxTensor::from_dense(vec![left.clone(), shared.clone()], vec![2.0, 3.0]).unwrap();
        let other = IdxTensor::from_dense(vec![shared], vec![4.0]).unwrap();
        let pair = probed_site_pair_batch_range(&local, &other, &[], &bank, 0, 3, &batch).unwrap();
        assert_eq!(pair.indices(), &[left, batch]);
        assert_eq!(
            pair.to_vec::<f64>().unwrap(),
            vec![8.0, 12.0, 8.0, 12.0, 8.0, 12.0]
        );
    }

    #[test]
    fn maximum_probe_width_respects_rank_row_and_cut_bounds() {
        let fixed = SrcOptions::fixed();
        assert_eq!(maximum_site_width(8, 16, 12, &fixed), 8);
        assert_eq!(maximum_site_width(12, 16, 12, &fixed), 12);
        assert_eq!(maximum_site_width(32, 16, 12, &fixed), 12);

        let oversampled = SrcOptions::fixed().with_final_svd(true);
        assert_eq!(maximum_site_width(8, 16, 12, &oversampled), 12);

        let adaptive = SrcOptions::adaptive(1.0e-6, 4096);
        assert_eq!(maximum_site_width(4096, 1024, 256, &adaptive), 256);
    }

    #[test]
    fn final_svd_adaptive_sketch_uses_the_paper_safety_factor() {
        let adaptive = SrcOptions::adaptive(1.0e-6, 64).with_final_svd(true);
        assert_eq!(adaptive.sketch_options(true).rtol, Some(1.0e-7));
        assert_eq!(adaptive.sketch_options(true).atol, adaptive.atol);

        let fixed = SrcOptions::fixed().with_final_svd(true);
        assert_eq!(fixed.sketch_options(true), fixed);
    }

    #[test]
    fn final_svd_without_a_tolerance_policy_keeps_the_requested_sketch_tolerance() {
        let adaptive = SrcOptions::adaptive(1.0e-6, 64).with_final_svd(true);

        assert_eq!(adaptive.sketch_options(false).rtol, Some(1.0e-6));
    }

    #[test]
    fn adaptive_src_defaults_to_the_requested_tolerance_without_final_round() {
        let adaptive = SrcOptions::adaptive(1.0e-6, 64);

        assert!(!adaptive.final_svd);
        assert_eq!(adaptive.sketch_options(false).rtol, Some(1.0e-6));
    }

    #[test]
    fn factorize_probe_batches_grows_by_rank_increment_and_stops_on_error_estimate() {
        use super::factorize_probe_batches;

        let row = DynIndex::new_dyn(4);
        // A 4x4 identity-ish sketch (well-conditioned, rank 4) split into two
        // width-2 batches, so a rank_increment of 2 should need exactly two
        // `make_batch` calls to reach the full rank-4 estimate below tolerance.
        let batch0 = DynIndex::new_dyn(2);
        let batch1 = DynIndex::new_dyn(2);
        let full = IdxTensor::from_dense(
            vec![row.clone(), DynIndex::new_dyn(4)],
            vec![
                1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
            ],
        )
        .unwrap();
        let column_index = full.indices()[1].clone();
        let col = |position: usize| {
            full.select_indices(std::slice::from_ref(&column_index), &[position])
                .unwrap()
        };
        let first_two =
            IdxTensor::stack_along_new_index(&[&col(0), &col(1)], batch0.clone(), -1).unwrap();
        let last_two =
            IdxTensor::stack_along_new_index(&[&col(2), &col(3)], batch1.clone(), -1).unwrap();

        let mut calls = Vec::new();
        let src_options = SrcOptions::adaptive(1.0e-10, 4)
            .with_min_rank(1)
            .with_rank_increment(2);
        let (result, result_batch) = factorize_probe_batches::<IdxTensor, _>(
            &[row],
            2,
            4,
            &src_options,
            "test",
            |start, width| {
                calls.push((start, width));
                if start == 0 {
                    Ok((first_two.clone(), batch0.clone()))
                } else {
                    Ok((last_two.clone(), batch1.clone()))
                }
            },
        )
        .unwrap();

        assert_eq!(calls, vec![(0, 2), (2, 2)]);
        assert_eq!(result_batch.dim(), 4);
        let values = result.to_vec::<f64>().unwrap();
        assert_eq!(values.len(), 16);
    }
}
