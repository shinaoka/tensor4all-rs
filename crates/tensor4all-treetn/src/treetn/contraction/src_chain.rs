//! Paper-faithful chain specialization of successive randomized compression.
//!
//! Provenance: `contract` implements the right-to-left schedule in Algorithm 1
//! and Sections 3.1-3.5 of Camaño--Epperly--Tropp,
//! [arXiv:2504.06475](https://arxiv.org/abs/2504.06475). Its local contraction
//! ordering and adaptive loop were cross-checked against
//! `chriscamano/RandomMPOMPS/code/tensornetwork/contraction.py`,
//! `random_contraction` (lines 133--353) and `random_contraction_inc`
//! (lines 405--593). Prefix batching and the Q-column reuse optimization are
//! derived implementation choices, not claims that the author code contains
//! the same Rust abstractions; they are labelled `[AI-Supplied]` in the audit.

use anyhow::Result;
use rand::Rng;
use std::hash::Hash;

use tensor4all_core::{
    Canonical as FactorizeCanonical, FactorizeAlg, IndexLike, SvdTruncationPolicy, TensorLike,
};
use tensor4all_tensorbackend::ExecutionContext;

use super::src_probe::{
    connect_result_edge, contract_prefix_with_probed_site_pair_batch_range, contract_retaining,
    contract_site_pair, factorize_probe_batches, initial_width, local_output_indices,
    local_site_pairs, mark_result_canonical, maximum_site_width, probed_site_pair_batch_range,
    product_dim, ProbeBank,
};
use super::{SrcOptions, TreeTN};
use crate::algorithm::CanonicalForm;

#[allow(clippy::too_many_arguments)]
/// Execute the paper's successive randomized compression schedule on a chain.
pub(super) fn contract<T, V, R>(
    tn_a: &TreeTN<T, V>,
    tn_b: &TreeTN<T, V>,
    center: &V,
    svd_policy: Option<SvdTruncationPolicy>,
    max_bond_dim: usize,
    src_options: &SrcOptions,
    rng: R,
    context: &ExecutionContext,
) -> Result<TreeTN<T, V>>
where
    T: TensorLike,
    T::Index: IndexLike + Clone + Hash + Eq,
    <T::Index as IndexLike>::Id: Clone + Hash + Eq + Ord + std::fmt::Debug + Send + Sync,
    V: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug,
    R: Rng,
{
    let chain = tn_a
        .chain_order(center)
        .ok_or_else(|| anyhow::anyhow!("contract_src: expected a chain containing center"))?;
    if !tn_a.same_topology(tn_b) {
        anyhow::bail!("contract_src: networks have incompatible topologies");
    }
    if chain.is_empty() {
        anyhow::bail!("contract_src: empty chain");
    }

    let tn_a = tn_a.sim_internal_inds();
    let tn_b = tn_b.sim_internal_inds();
    let local = local_site_pairs(&tn_a, &tn_b, &chain)?;
    if chain.len() == 1 {
        let mut result = TreeTN::new();
        let tensor = contract_site_pair(local[0].0, local[0].1, &[])?;
        result.add_tensor(chain[0].clone(), tensor)?;
        result.canonicalize_impl_in(
            [center.clone()],
            CanonicalForm::Unitary,
            "contract_src: single-site canonicalization",
            context,
        )?;
        return Ok(result);
    }

    let outputs = local
        .iter()
        .enumerate()
        .map(|(site, _)| local_output_indices(&tn_a, &tn_b, &chain[site]))
        .collect::<Result<Vec<_>>>()?;
    let cut_dimensions = chain_cut_dimensions(&tn_a, &tn_b, &chain)?;
    let probe_indices = outputs[..outputs.len() - 1]
        .iter()
        .flat_map(|site| site.iter().cloned())
        .collect::<Vec<_>>();
    let last_output_dim = product_dim(&outputs[outputs.len() - 1])?;
    let last_maximum_width = maximum_site_width(
        max_bond_dim,
        last_output_dim,
        *cut_dimensions
            .last()
            .ok_or_else(|| anyhow::anyhow!("contract_src: chain has no internal cut"))?,
        src_options,
    );
    if last_maximum_width == 0 {
        anyhow::bail!("contract_src: last-site output space has zero dimension");
    }
    let mut probes = ProbeBank::new(probe_indices, 1, rng)?;
    if src_options.rtol.is_none() {
        return contract_fixed(FixedContractionRequest {
            center,
            svd_policy,
            max_bond_dim,
            chain: &chain,
            local: &local,
            outputs: &outputs,
            cut_dimensions: &cut_dimensions,
            probes: &mut probes,
            final_svd: src_options.final_svd,
            context,
        });
    }
    let sketch_options = src_options.sketch_options(svd_policy.is_some());
    let mut prefixes = PrefixCache::new(
        &local,
        &outputs,
        &mut probes,
        sketch_options.rank_increment,
        context,
    );

    let last = chain.len() - 1;
    let mut factors: Vec<Option<T>> = (0..chain.len()).map(|_| None).collect();
    let mut caps: Vec<Option<T::Index>> = (0..chain.len()).map(|_| None).collect();

    let last_initial_width = if sketch_options.rtol.is_some() {
        initial_width(last_maximum_width, &sketch_options)
    } else {
        last_maximum_width
    };
    let (last_factor, last_cap, mut cap_environment) = if outputs[last].is_empty() {
        let cap = T::Index::new_link(1)?;
        let factor = T::ones_in(context, std::slice::from_ref(&cap)).map_err(|error| {
            anyhow::anyhow!("contract_src: scalar last-site cap construction failed: {error}")
        })?;
        let local_product = contract_site_pair(local[last].0, local[last].1, &[])?;
        let environment = local_product.outer_product(&factor).map_err(|error| {
            anyhow::anyhow!(
                "contract_src: scalar last-site environment construction failed: {error}"
            )
        })?;
        (factor, cap, environment)
    } else {
        factorize_site_adaptive(
            FactorizeSiteRequest {
                outputs: &outputs[last],
                right_cap: None,
                operands: local[last],
                right_environment: None,
                initial_width: last_initial_width,
                maximum_width: last_maximum_width,
                src_options: &sketch_options,
                label: "last-site",
                context,
            },
            |start, width| {
                let (prefix, batch_index) = prefixes.request(last - 1, start, width)?;
                let after_a = contract_retaining(&[&prefix, local[last].0], &batch_index).map_err(
                    |error| {
                        anyhow::anyhow!(
                            "contract_src: last-site prefix-A contraction failed: {error}"
                        )
                    },
                )?;
                contract_retaining(&[&after_a, local[last].1], &batch_index)
                    .map(|tensor| (tensor, batch_index))
                    .map_err(|error| {
                        anyhow::anyhow!("contract_src: last-site sketch failed: {error}")
                    })
            },
        )?
    };
    caps[last] = Some(last_cap);
    factors[last] = Some(last_factor);

    for site in (1..last).rev() {
        let right_environment = cap_environment;
        let right_cap = caps[site + 1]
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("contract_src: missing right cap at site {site}"))?;
        let row_dim = product_dim(&outputs[site])?
            .checked_mul(right_cap.dim())
            .ok_or_else(|| {
                anyhow::anyhow!("contract_src: site {site} sketch row dimension overflow")
            })?;
        let cut_dimension = cut_dimensions[site - 1].max(cut_dimensions[site]);
        let site_max_width =
            maximum_site_width(max_bond_dim, row_dim, cut_dimension, &sketch_options);
        if site_max_width == 0 {
            anyhow::bail!("contract_src: site {site} sketch row space is empty");
        }
        let site_initial_width = if sketch_options.rtol.is_some() {
            initial_width(site_max_width, &sketch_options)
        } else {
            site_max_width
        };
        let label = format!("site {site}");
        let (factor, cap, next_environment) = factorize_site_adaptive(
            FactorizeSiteRequest {
                outputs: &outputs[site],
                right_cap: Some(right_cap),
                operands: local[site],
                right_environment: Some(&right_environment),
                initial_width: site_initial_width,
                maximum_width: site_max_width,
                src_options: &sketch_options,
                label: &label,
                context,
            },
            |start, width| {
                let (stacked, batch_index) = prefixes.request(site - 1, start, width)?;
                let after_a = contract_retaining(&[&stacked, local[site].0], &batch_index)
                    .map_err(|error| {
                        anyhow::anyhow!(
                            "contract_src: site {site} prefix-A contraction failed: {error}"
                        )
                    })?;
                let after_b = contract_retaining(&[&after_a, local[site].1], &batch_index)
                    .map_err(|error| {
                        anyhow::anyhow!(
                            "contract_src: site {site} prefix-B contraction failed: {error}"
                        )
                    })?;
                contract_retaining(&[&after_b, &right_environment], &batch_index)
                    .map(|tensor| (tensor, batch_index))
                    .map_err(|error| {
                        anyhow::anyhow!("contract_src: site {site} sketch failed: {error}")
                    })
            },
        )?;
        caps[site] = Some(cap);
        factors[site] = Some(factor);
        cap_environment = next_environment;
    }

    let first = contract_site_pair(local[0].0, local[0].1, &[&cap_environment])
        .map_err(|error| anyhow::anyhow!("contract_src: first-site contraction failed: {error}"))?;
    factors[0] = Some(first);

    let mut result = TreeTN::new();
    for (site, node) in chain.iter().enumerate() {
        let tensor = factors[site]
            .take()
            .ok_or_else(|| anyhow::anyhow!("contract_src: missing result tensor at site {site}"))?;
        result.add_tensor(node.clone(), tensor)?;
    }
    for site in 1..chain.len() {
        connect_result_edge(&mut result, &chain[site - 1], &chain[site])?;
    }

    if src_options.final_svd {
        result.truncate_impl(
            [center.clone()],
            svd_policy,
            Some(max_bond_dim),
            "contract_src: final truncate",
        )?;
    } else {
        let rooted_edges = chain
            .windows(2)
            .map(|sites| (sites[0].clone(), sites[1].clone()))
            .collect::<Vec<_>>();
        mark_result_canonical(&mut result, center, &rooted_edges)?;
    }
    Ok(result)
}

struct FixedContractionRequest<'a, T, V, R>
where
    T: TensorLike,
{
    center: &'a V,
    svd_policy: Option<SvdTruncationPolicy>,
    max_bond_dim: usize,
    chain: &'a [V],
    local: &'a [(&'a T, &'a T)],
    outputs: &'a [Vec<T::Index>],
    cut_dimensions: &'a [usize],
    probes: &'a mut ProbeBank<T::Index, R>,
    final_svd: bool,
    context: &'a ExecutionContext,
}

fn contract_fixed<T, V, R>(request: FixedContractionRequest<'_, T, V, R>) -> Result<TreeTN<T, V>>
where
    T: TensorLike,
    T::Index: IndexLike + Clone + Hash + Eq,
    <T::Index as IndexLike>::Id: Clone + Hash + Eq + Ord + std::fmt::Debug + Send + Sync,
    V: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug,
    R: Rng,
{
    let FixedContractionRequest {
        center,
        svd_policy,
        max_bond_dim,
        chain,
        local,
        outputs,
        cut_dimensions,
        probes,
        final_svd,
        context,
    } = request;
    let last = chain.len() - 1;
    let fixed_options = SrcOptions::fixed().with_final_svd(final_svd);
    let last_maximum_width = maximum_site_width(
        max_bond_dim,
        product_dim(&outputs[last])?,
        *cut_dimensions
            .last()
            .ok_or_else(|| anyhow::anyhow!("contract_src: chain has no internal cut"))?,
        &fixed_options,
    );
    if last_maximum_width == 0 {
        anyhow::bail!("contract_src: last-site output space has zero dimension");
    }
    let mut prefixes = BatchedPrefixCache::new(local, outputs, probes, context);
    let mut factors: Vec<Option<T>> = (0..chain.len()).map(|_| None).collect();
    let mut caps: Vec<Option<T::Index>> = (0..chain.len()).map(|_| None).collect();

    let (last_factor, last_cap, mut cap_environment) = if outputs[last].is_empty() {
        let cap = T::Index::new_link(1)?;
        let factor = T::ones_in(context, std::slice::from_ref(&cap)).map_err(|error| {
            anyhow::anyhow!("contract_src: scalar last-site cap construction failed: {error}")
        })?;
        let local_product = contract_site_pair(local[last].0, local[last].1, &[])?;
        let environment = local_product.outer_product(&factor).map_err(|error| {
            anyhow::anyhow!(
                "contract_src: scalar last-site environment construction failed: {error}"
            )
        })?;
        (factor, cap, environment)
    } else {
        let (prefix, batch) = prefixes.batch(last - 1, last_maximum_width)?;
        let sketch = contract_prefix_with_probed_site_pair_batch_range(
            &prefix,
            local[last].0,
            local[last].1,
            &[],
            &*prefixes.probes,
            0,
            last_maximum_width,
            &batch,
            context,
        )
        .map_err(|error| anyhow::anyhow!("contract_src: last-site sketch failed: {error}"))?;
        let (factor, cap) = factorize_fixed_batch(&sketch, &outputs[last], "last-site", context)?;
        let factor_conj = factor.conj();
        let environment = contract_site_pair(local[last].0, local[last].1, &[&factor_conj])
            .map_err(|error| {
                anyhow::anyhow!("contract_src: last-site environment failed: {error}")
            })?;
        (factor, cap, environment)
    };
    caps[last] = Some(last_cap);
    factors[last] = Some(last_factor);

    for site in (1..last).rev() {
        let right_environment = cap_environment;
        let right_cap = caps[site + 1]
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("contract_src: missing right cap at site {site}"))?;
        let row_dim = product_dim(&outputs[site])?
            .checked_mul(right_cap.dim())
            .ok_or_else(|| {
                anyhow::anyhow!("contract_src: site {site} sketch row dimension overflow")
            })?;
        let cut_dimension = cut_dimensions[site - 1].max(cut_dimensions[site]);
        let site_max_width =
            maximum_site_width(max_bond_dim, row_dim, cut_dimension, &fixed_options);
        if site_max_width == 0 {
            anyhow::bail!("contract_src: site {site} sketch row space is empty");
        }
        let mut left_indices = outputs[site].clone();
        left_indices.push(right_cap.clone());
        let (prefix, batch) = prefixes.batch(site - 1, site_max_width)?;
        let prefix_local = contract_prefix_with_probed_site_pair_batch_range(
            &prefix,
            local[site].0,
            local[site].1,
            &[],
            &*prefixes.probes,
            0,
            site_max_width,
            &batch,
            context,
        )
        .map_err(|error| {
            anyhow::anyhow!("contract_src: site {site} prefix contraction failed: {error}")
        })?;
        let sketch = T::contract(&[&prefix_local, &right_environment])
            .map_err(|error| anyhow::anyhow!("contract_src: site {site} sketch failed: {error}"))?;
        let (factor, cap) =
            factorize_fixed_batch(&sketch, &left_indices, &format!("site {site}"), context)?;
        let factor_conj = factor.conj();
        let next_environment = contract_site_pair(
            local[site].0,
            local[site].1,
            &[&factor_conj, &right_environment],
        )
        .map_err(|error| {
            anyhow::anyhow!("contract_src: site {site} environment failed: {error}")
        })?;
        caps[site] = Some(cap);
        factors[site] = Some(factor);
        cap_environment = next_environment;
    }

    let first = contract_site_pair(local[0].0, local[0].1, &[&cap_environment])
        .map_err(|error| anyhow::anyhow!("contract_src: first-site contraction failed: {error}"))?;
    factors[0] = Some(first);

    let mut result = TreeTN::new();
    for (site, node) in chain.iter().enumerate() {
        let tensor = factors[site]
            .take()
            .ok_or_else(|| anyhow::anyhow!("contract_src: missing result tensor at site {site}"))?;
        result.add_tensor(node.clone(), tensor)?;
    }
    for site in 1..chain.len() {
        connect_result_edge(&mut result, &chain[site - 1], &chain[site])?;
    }
    if final_svd {
        result.truncate_impl_in(
            [center.clone()],
            svd_policy,
            Some(max_bond_dim),
            "contract_src: final truncate",
            context,
        )?;
    } else {
        let rooted_edges = chain
            .windows(2)
            .map(|sites| (sites[0].clone(), sites[1].clone()))
            .collect::<Vec<_>>();
        mark_result_canonical(&mut result, center, &rooted_edges)?;
    }
    Ok(result)
}

fn chain_cut_dimensions<T, V>(
    tn_a: &TreeTN<T, V>,
    tn_b: &TreeTN<T, V>,
    chain: &[V],
) -> Result<Vec<usize>>
where
    T: TensorLike,
    T::Index: IndexLike + Clone + Hash + Eq,
    V: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug,
{
    chain
        .windows(2)
        .map(|sites| {
            let left = &sites[0];
            let right = &sites[1];
            let edge_a = tn_a
                .edge_between(left, right)
                .ok_or_else(|| anyhow::anyhow!("contract_src: A chain edge is missing"))?;
            let edge_b = tn_b
                .edge_between(left, right)
                .ok_or_else(|| anyhow::anyhow!("contract_src: B chain edge is missing"))?;
            let dim_a = tn_a
                .bond_index(edge_a)
                .ok_or_else(|| anyhow::anyhow!("contract_src: A chain bond is missing"))?
                .dim();
            let dim_b = tn_b
                .bond_index(edge_b)
                .ok_or_else(|| anyhow::anyhow!("contract_src: B chain bond is missing"))?
                .dim();
            dim_a
                .checked_mul(dim_b)
                .ok_or_else(|| anyhow::anyhow!("contract_src: chain cut dimension overflow"))
        })
        .collect()
}

fn factorize_fixed_batch<T>(
    sketch: &T,
    left_indices: &[T::Index],
    label: &str,
    context: &ExecutionContext,
) -> Result<(T, T::Index)>
where
    T: TensorLike,
    T::Index: IndexLike + Clone + Hash + Eq,
{
    let factorized = sketch
        .factorize_full_rank_in(
            left_indices,
            FactorizeAlg::QR,
            FactorizeCanonical::Left,
            context,
        )
        .map_err(|error| {
            anyhow::anyhow!(
                "contract_src: {label} QR failed: {error}; sketch indices={:?}, left indices={:?}",
                sketch.external_indices(),
                left_indices
            )
        })?;
    Ok((factorized.left, factorized.bond_index))
}

struct PrefixCache<'a, T, R>
where
    T: TensorLike,
{
    local: &'a [(&'a T, &'a T)],
    outputs: &'a [Vec<T::Index>],
    probes: &'a mut ProbeBank<T::Index, R>,
    context: &'a ExecutionContext,
    batch_size: usize,
    // Per-site list of (batch tensor, batch index, width) segments, storing
    // whatever chunk each `grow_segment` call actually produced. The first
    // segment may be narrower than `batch_size`; subsequent segments stay on
    // the full rank-increment grid.
    segments: Vec<Vec<(T, T::Index, usize)>>,
    segment_total_width: usize,
}

/// Batched (whole-width-at-once) prefix cache for the fixed-rank path.
///
/// Unlike [`PrefixCache`] (used by the adaptive path, which needs individual
/// per-column tensors for its incremental QR), `contract_fixed` always wants
/// one `width`-wide batch-indexed tensor per site. Grows `combined` by
/// concatenating (`concatenate_along_new_index`) exactly one new segment onto
/// the existing combined tensor per width-growth step -- O(1) amortized
/// concatenate calls per site, not the O(segments-seen-so-far) full
/// re-concatenation the earlier version of this cache did, and not the
/// N-reshape-then-concatenate cost `stack_along_new_index` pays per
/// individual column (`EagerTensor::stack` reshapes every input before
/// concatenating; `concatenate_along_new_index` reshapes none, since the
/// segments already carry a batch axis from their own construction).
/// Mirrors how the reference Python implementation's `envs` cache needs no
/// re-concatenation once a column is computed for a site.
struct BatchedPrefixCache<'a, T, R>
where
    T: TensorLike,
{
    local: &'a [(&'a T, &'a T)],
    outputs: &'a [Vec<T::Index>],
    probes: &'a mut ProbeBank<T::Index, R>,
    combined: Vec<T>,
    combined_batch: Option<T::Index>,
    generated_width: usize,
    context: &'a ExecutionContext,
}

impl<'a, T, R> BatchedPrefixCache<'a, T, R>
where
    T: TensorLike,
    T::Index: IndexLike + Clone + Hash + Eq,
    R: Rng,
{
    fn new(
        local: &'a [(&'a T, &'a T)],
        outputs: &'a [Vec<T::Index>],
        probes: &'a mut ProbeBank<T::Index, R>,
        context: &'a ExecutionContext,
    ) -> Self {
        Self {
            local,
            outputs,
            probes,
            context,
            combined: Vec::new(),
            combined_batch: None,
            generated_width: 0,
        }
    }

    fn batch(&mut self, site: usize, width: usize) -> Result<(T, T::Index)> {
        if width < self.generated_width {
            // The chain schedule normally grows the requested width while
            // sweeping right-to-left. Reset on a decrease so growth stays a
            // simple monotonic append rather than needing to shrink a
            // combined tensor back down.
            self.combined = Vec::new();
            self.combined_batch = None;
            self.generated_width = 0;
        }
        if width > self.generated_width {
            self.probes.extend_to(width)?;
            let start = self.generated_width;
            let segment_width = width - start;
            let segment_batch = T::Index::new_link(segment_width)?;
            let mut prefix = probed_site_pair_batch_range(
                self.local[0].0,
                self.local[0].1,
                &self.outputs[0],
                self.probes,
                start,
                segment_width,
                &segment_batch,
                self.context,
            )?;
            let mut segment_prefixes = vec![prefix.clone()];
            for prefix_site in 1..self.local.len() - 1 {
                prefix = contract_prefix_with_probed_site_pair_batch_range(
                    &prefix,
                    self.local[prefix_site].0,
                    self.local[prefix_site].1,
                    &self.outputs[prefix_site],
                    self.probes,
                    start,
                    segment_width,
                    &segment_batch,
                    self.context,
                )?;
                segment_prefixes.push(prefix.clone());
            }
            if start == 0 {
                // The first segment already spans the whole requested width,
                // so it IS the combined tensor -- its own `segment_batch` is
                // the combined batch index, not a fresh one (the tensors in
                // `segment_prefixes` don't carry any other batch index, so
                // claiming a different one here would make every downstream
                // `contract_retaining` call fail to find it).
                self.combined = segment_prefixes;
                self.combined_batch = Some(segment_batch);
            } else {
                let previous_batch = self
                    .combined_batch
                    .clone()
                    .ok_or_else(|| anyhow::anyhow!("contract_src: prefix batch is missing"))?;
                let combined_batch = T::Index::new_link(width)?;
                for (prefix_site, segment) in segment_prefixes.into_iter().enumerate() {
                    self.combined[prefix_site] = T::concatenate_along_new_index(
                        &[&self.combined[prefix_site], &segment],
                        &[previous_batch.clone(), segment_batch.clone()],
                        combined_batch.clone(),
                    )?;
                }
                self.combined_batch = Some(combined_batch);
            }
            self.generated_width = width;
        }
        let batch = self
            .combined_batch
            .clone()
            .ok_or_else(|| anyhow::anyhow!("contract_src: prefix batch is missing"))?;
        let result = self
            .combined
            .get(site)
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("contract_src: prefix batch is missing"))?;
        Ok((result, batch))
    }
}

impl<'a, T, R> PrefixCache<'a, T, R>
where
    T: TensorLike,
    T::Index: IndexLike + Clone + Hash + Eq,
    R: Rng,
{
    fn new(
        local: &'a [(&'a T, &'a T)],
        outputs: &'a [Vec<T::Index>],
        probes: &'a mut ProbeBank<T::Index, R>,
        batch_size: usize,
        context: &'a ExecutionContext,
    ) -> Self {
        Self {
            local,
            outputs,
            probes,
            context,
            batch_size: batch_size.max(1),
            segments: (0..local.len() - 1).map(|_| Vec::new()).collect(),
            segment_total_width: 0,
        }
    }

    /// Compute one new segment covering `[start, start + width)` at site 0.
    /// [`Self::extend_segments_to`] propagates it farther only when requested.
    fn grow_segment(&mut self, start: usize, width: usize) -> Result<()> {
        let batch = T::Index::new_link(width)?;
        let prefix = probed_site_pair_batch_range(
            self.local[0].0,
            self.local[0].1,
            &self.outputs[0],
            self.probes,
            start,
            width,
            &batch,
            self.context,
        )?;
        self.segments[0].push((prefix, batch, width));
        self.segment_total_width += width;
        Ok(())
    }

    /// Extend cached segments just far enough to serve `site` through `end`.
    fn extend_segments_to(&mut self, site: usize, end: usize) -> Result<()> {
        let mut covered = 0;
        for segment in 0..self.segments[0].len() {
            if covered >= end {
                break;
            }
            let width = self.segments[0][segment].2;
            while self.segments[site].len() <= segment {
                let next_site = (1..=site)
                    .find(|&candidate| self.segments[candidate].len() <= segment)
                    .ok_or_else(|| anyhow::anyhow!("contract_src: missing prefix segment"))?;
                let (prefix, batch, _) = self.segments[next_site - 1][segment].clone();
                let extended = contract_prefix_with_probed_site_pair_batch_range(
                    &prefix,
                    self.local[next_site].0,
                    self.local[next_site].1,
                    &self.outputs[next_site],
                    self.probes,
                    covered,
                    width,
                    &batch,
                    self.context,
                )?;
                debug_assert_eq!(self.segments[next_site].len(), segment);
                self.segments[next_site].push((extended, batch, width));
            }
            covered += width;
        }
        Ok(())
    }

    /// Return `[start, start + width)` as one batch-indexed tensor for
    /// `site`, growing new segments first if the requested range extends
    /// past what is cached.
    ///
    /// The common case (the request aligns exactly with one already-grown or
    /// newly-grown segment's boundaries) returns that segment directly, with
    /// no `select_indices`/`stack_along_new_index` at all. A request that
    /// only partially overlaps a segment boundary falls back to splitting and
    /// re-stacking the covering segments; this is expected to be rare, not
    /// routine. Only the initial segment may be narrower than `batch_size`;
    /// later growth deliberately computes a full segment so rank increments
    /// remain aligned across sites.
    fn request(&mut self, site: usize, start: usize, width: usize) -> Result<(T, T::Index)> {
        let requested_end = start
            .checked_add(width)
            .ok_or_else(|| anyhow::anyhow!("contract_src: requested probe range overflows"))?;
        while self.segment_total_width < requested_end {
            let next_start = self.segment_total_width;
            let next_width = if next_start == 0 {
                self.batch_size.min(requested_end)
            } else {
                self.batch_size
            };
            let next_end = next_start
                .checked_add(next_width)
                .ok_or_else(|| anyhow::anyhow!("contract_src: cached probe range overflows"))?;
            self.probes.extend_to(next_end)?;
            self.grow_segment(next_start, next_width)?;
        }
        self.extend_segments_to(site, requested_end)?;

        let site_segments = &self.segments[site];
        let mut cursor = 0usize;
        for (tensor, batch_index, segment_width) in site_segments {
            if cursor == start && *segment_width == width {
                return Ok((tensor.clone(), batch_index.clone()));
            }
            cursor += segment_width;
        }

        // Misaligned fallback: split the covering segment(s) into
        // individual columns via `select_indices` and re-stack the
        // requested range. Reachable only when `[start, start+width)`
        // doesn't align with a single stored segment's boundaries -- see
        // this method's doc comment.
        let mut collected = Vec::with_capacity(width);
        let mut cursor = 0usize;
        for (tensor, batch_index, segment_width) in &self.segments[site] {
            let segment_start = cursor;
            let segment_end = cursor + segment_width;
            cursor = segment_end;
            let overlap_start = segment_start.max(start);
            let overlap_end = segment_end.min(requested_end);
            if overlap_start >= overlap_end {
                continue;
            }
            for position in (overlap_start - segment_start)..(overlap_end - segment_start) {
                collected.push(tensor.select_indices(std::slice::from_ref(batch_index), &[position]).map_err(
                    |error| {
                        anyhow::anyhow!(
                            "contract_src: site {site} misaligned segment split at position {position} failed: {error}"
                        )
                    },
                )?);
            }
        }
        anyhow::ensure!(
            collected.len() == width,
            "contract_src: site {site} misaligned segment request [{start}, {}) only found {} of {} columns",
            requested_end,
            collected.len(),
            width
        );
        let collected_refs = collected.iter().collect::<Vec<_>>();
        let batch_index = T::Index::new_link(width)?;
        let stacked = T::stack_along_new_index(&collected_refs, batch_index.clone(), -1).map_err(|error| {
            anyhow::anyhow!(
                "contract_src: site {site} misaligned segment request [{start}, {}) failed: {error}",
                start + width
            )
        })?;
        Ok((stacked, batch_index))
    }
}

struct FactorizeSiteRequest<'a, T>
where
    T: TensorLike,
{
    outputs: &'a [T::Index],
    right_cap: Option<&'a T::Index>,
    operands: (&'a T, &'a T),
    right_environment: Option<&'a T>,
    initial_width: usize,
    maximum_width: usize,
    src_options: &'a SrcOptions,
    label: &'a str,
    context: &'a ExecutionContext,
}

fn factorize_site_adaptive<T, F>(
    request: FactorizeSiteRequest<'_, T>,
    make_batch: F,
) -> Result<(T, T::Index, T)>
where
    T: TensorLike,
    T::Index: IndexLike + Clone + Hash + Eq,
    F: FnMut(usize, usize) -> Result<(T, T::Index)>,
{
    let FactorizeSiteRequest {
        outputs,
        right_cap,
        operands,
        right_environment,
        initial_width,
        maximum_width,
        src_options,
        label,
        context,
    } = request;
    let mut left = outputs.to_vec();
    if let Some(right_cap) = right_cap {
        left.push(right_cap.clone());
    }
    let (factor, cap) = factorize_probe_batches(
        &left,
        initial_width,
        maximum_width,
        src_options,
        label,
        make_batch,
        context,
    )?;
    let factor_conj = factor.conj();
    let environment = if let Some(right_environment) = right_environment {
        contract_site_pair(operands.0, operands.1, &[&factor_conj, right_environment])
    } else {
        contract_site_pair(operands.0, operands.1, &[&factor_conj])
    }
    .map_err(|error| anyhow::anyhow!("contract_src: {label} environment failed: {error}"))?;
    Ok((factor, cap, environment))
}

#[cfg(test)]
mod tests {
    use super::PrefixCache;
    use crate::treetn::contraction::src_probe::ProbeBank;
    use tensor4all_core::{DynIndex, IdxTensor, IndexLike};

    fn two_site_local(dim: usize) -> (IdxTensor, IdxTensor, IdxTensor, IdxTensor) {
        let s0_out = DynIndex::new_dyn(dim);
        let s0_in = DynIndex::new_dyn(dim);
        let s1_out = DynIndex::new_dyn(dim);
        let elements = dim * dim;
        let a0 = IdxTensor::from_dense(vec![s0_out, s0_in.clone()], vec![0.1; elements]).unwrap();
        let b0 = IdxTensor::from_dense(vec![s0_in], vec![0.2; dim]).unwrap();
        let a1 = IdxTensor::from_dense(vec![s1_out.clone()], vec![0.3; dim]).unwrap();
        let b1 = IdxTensor::from_dense(vec![s1_out], vec![0.4; dim]).unwrap();
        (a0, b0, a1, b1)
    }

    #[test]
    fn request_grows_a_fresh_segment_and_reuses_a_previously_cached_one() {
        let context = tensor4all_tensorbackend::ExecutionContext::Cpu(
            tensor4all_tensorbackend::default_cpu_execution_context(),
        );
        let (a0, b0, a1, b1) = two_site_local(3);
        let local = vec![(&a0, &b0), (&a1, &b1)];
        let outputs = vec![vec![a0.indices()[0].clone()], vec![a1.indices()[0].clone()]];
        let mut probes = ProbeBank::from_seed(
            outputs.iter().flat_map(|o| o.iter().cloned()).collect(),
            1,
            42,
        )
        .unwrap();
        let mut cache = PrefixCache::new(&local, &outputs, &mut probes, 3, &context);

        let (first, first_batch) = cache.request(0, 0, 3).unwrap();
        assert_eq!(first_batch.dim(), 3);
        // Site 0's own output index (`s0_out`) is one of `outputs[0]`, so it
        // is contracted away against the probe columns rather than kept --
        // like `s0_in` (the a0/b0 shared bond), it does not survive into the
        // prefix. Only the batch axis remains external.
        assert_eq!(first.dims().len(), 1); // [batch]

        // Requesting the same already-cached range again must not recompute
        // it -- assert the returned tensor is bit-identical (a fresh
        // recomputation of the same probes would also be numerically
        // identical here since ProbeBank is deterministic, so this alone
        // isn't proof of reuse; the ragged-boundary test below adds a
        // segment-count assertion that actually proves it).
        let (again, again_batch) = cache.request(0, 0, 3).unwrap();
        assert_eq!(again_batch.dim(), first_batch.dim());
        assert_eq!(
            again.to_vec::<f64>().unwrap(),
            first.to_vec::<f64>().unwrap()
        );
    }

    #[test]
    fn request_overgenerates_a_full_post_initial_segment_for_reuse() {
        let context = tensor4all_tensorbackend::ExecutionContext::Cpu(
            tensor4all_tensorbackend::default_cpu_execution_context(),
        );
        let (a0, b0, a1, b1) = two_site_local(3);
        let local = vec![(&a0, &b0), (&a1, &b1)];
        let outputs = vec![vec![a0.indices()[0].clone()], vec![a1.indices()[0].clone()]];
        let mut probes = ProbeBank::from_seed(
            outputs.iter().flat_map(|o| o.iter().cloned()).collect(),
            1,
            42,
        )
        .unwrap();
        // batch_size 3, first caller only needs width 4. The cache computes
        // a full second increment so a later width-3 request can reuse it.
        let mut cache = PrefixCache::new(&local, &outputs, &mut probes, 3, &context);
        let (_first, _) = cache.request(0, 0, 4).unwrap();
        assert_eq!(cache.segments[0].len(), 2, "expected two width-3 segments");
        assert_eq!(
            cache.segments[0][1].2, 3,
            "post-initial growth should use the full rank increment"
        );

        // A later caller can reuse that full segment directly.
        let (_second, second_batch) = cache.request(0, 3, 3).unwrap();
        assert_eq!(second_batch.dim(), 3);
        assert_eq!(
            cache.segments[0].len(),
            2,
            "re-reading an existing aligned segment must not grow a new one"
        );
    }

    #[test]
    fn request_extends_only_needed_segments_to_requested_site() {
        let context = tensor4all_tensorbackend::ExecutionContext::Cpu(
            tensor4all_tensorbackend::default_cpu_execution_context(),
        );
        let outputs = (0..3).map(|_| DynIndex::new_dyn(3)).collect::<Vec<_>>();
        let inner = (0..3).map(|_| DynIndex::new_dyn(3)).collect::<Vec<_>>();
        let tensors = (0..3)
            .map(|site| {
                (
                    IdxTensor::from_dense(
                        vec![outputs[site].clone(), inner[site].clone()],
                        vec![0.1; 9],
                    )
                    .unwrap(),
                    IdxTensor::from_dense(vec![inner[site].clone()], vec![0.2; 3]).unwrap(),
                )
            })
            .collect::<Vec<_>>();
        let local = tensors.iter().map(|(a, b)| (a, b)).collect::<Vec<_>>();
        let site_outputs = outputs
            .iter()
            .cloned()
            .map(|output| vec![output])
            .collect::<Vec<_>>();
        let mut probes = ProbeBank::from_seed(outputs.clone(), 1, 42).unwrap();
        let mut cache = PrefixCache::new(&local, &site_outputs, &mut probes, 2, &context);

        cache.request(0, 0, 2).unwrap();
        cache.request(0, 2, 2).unwrap();
        assert_eq!(cache.segments[0].len(), 2);
        assert!(
            cache.segments[1].is_empty(),
            "site 1 must stay lazy while only site 0 is requested"
        );

        let (lazy, _) = cache.request(1, 0, 2).unwrap();
        assert_eq!(
            cache.segments[1].len(),
            1,
            "requesting the first range must not extend a later segment"
        );

        let mut direct_probes = ProbeBank::from_seed(outputs, 1, 42).unwrap();
        let mut direct = PrefixCache::new(&local, &site_outputs, &mut direct_probes, 2, &context);
        let (expected, _) = direct.request(1, 0, 2).unwrap();
        assert_eq!(
            lazy.to_vec::<f64>().unwrap(),
            expected.to_vec::<f64>().unwrap()
        );

        let (later, _) = cache.request(1, 4, 2).unwrap();
        assert_eq!(
            cache.segments[1].len(),
            3,
            "growing while ragged must fill the missing segment before appending"
        );
        let (middle, _) = cache.request(1, 2, 2).unwrap();

        let (expected_middle, _) = direct.request(1, 2, 2).unwrap();
        let (expected_later, _) = direct.request(1, 4, 2).unwrap();
        assert_eq!(
            middle.to_vec::<f64>().unwrap(),
            expected_middle.to_vec::<f64>().unwrap()
        );
        assert_eq!(
            later.to_vec::<f64>().unwrap(),
            expected_later.to_vec::<f64>().unwrap()
        );
    }

    #[test]
    fn request_misaligned_range_spanning_a_segment_boundary_matches_a_direct_reference() {
        let context = tensor4all_tensorbackend::ExecutionContext::Cpu(
            tensor4all_tensorbackend::default_cpu_execution_context(),
        );
        let (a0, b0, a1, b1) = two_site_local(3);
        let local = vec![(&a0, &b0), (&a1, &b1)];
        let outputs = vec![vec![a0.indices()[0].clone()], vec![a1.indices()[0].clone()]];
        let index_list = outputs
            .iter()
            .flat_map(|o| o.iter().cloned())
            .collect::<Vec<_>>();
        let mut probes = ProbeBank::from_seed(index_list.clone(), 1, 42).unwrap();
        // batch_size 2: the first request exactly fills one aligned segment
        // [0,2); the second request [1,3) straddles that segment's right
        // boundary and the freshly grown [2,3) segment, so neither stored
        // segment alone covers [1,3) -- this must hit the misaligned
        // fallback (unlike the ragged-reuse test above, whose second
        // request re-reads an existing segment exactly and so never
        // exercises this branch).
        let mut cache = PrefixCache::new(&local, &outputs, &mut probes, 2, &context);
        let (_first, _) = cache.request(0, 0, 2).unwrap();
        assert_eq!(
            cache.segments[0].len(),
            1,
            "first request should be a single aligned segment"
        );

        let (misaligned, misaligned_batch) = cache.request(0, 1, 2).unwrap();
        assert_eq!(misaligned_batch.dim(), 2);
        assert_eq!(
            cache.segments[0].len(),
            2,
            "the misaligned request should grow a second segment covering [2,3) \
             but still needs to fall back to splitting/restacking, since [1,3) \
             does not align with either stored segment's boundaries"
        );

        // Ground truth: an independent probe bank/prefix computation for the
        // same [1, 3) range, built without going through `PrefixCache` at
        // all.
        let mut reference_probes = ProbeBank::from_seed(index_list, 1, 42).unwrap();
        reference_probes.extend_to(3).unwrap();
        let reference_batch = DynIndex::new_link(2).unwrap();
        let context = tensor4all_tensorbackend::ExecutionContext::Cpu(
            tensor4all_tensorbackend::default_cpu_execution_context(),
        );
        let reference = crate::treetn::contraction::src_probe::probed_site_pair_batch_range(
            &a0,
            &b0,
            &outputs[0],
            &reference_probes,
            1,
            2,
            &reference_batch,
            &context,
        )
        .unwrap();

        assert_eq!(
            misaligned.to_vec::<f64>().unwrap(),
            reference.to_vec::<f64>().unwrap(),
            "misaligned fallback result must match a directly computed reference"
        );
    }
}
