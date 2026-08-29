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
use std::hash::Hash;

use tensor4all_core::{
    Canonical as FactorizeCanonical, FactorizeAlg, IndexLike, SvdTruncationPolicy, TensorLike,
};

use super::src_probe::{
    connect_result_edge, contract_prefix_with_probed_site_pair_batch_range,
    contract_prefix_with_site_pair, contract_retaining, contract_site_pair,
    factorize_probe_columns, initial_width, local_output_indices, local_site_pairs,
    mark_result_canonical, maximum_site_width, probed_site_pair_batch_range, product_dim,
    ProbeBank,
};
use super::{SrcOptions, TreeTN};
use crate::algorithm::CanonicalForm;

/// Execute the paper's successive randomized compression schedule on a chain.
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
    T::Index: IndexLike + Clone + Hash + Eq,
    <T::Index as IndexLike>::Id: Clone + Hash + Eq + Ord + std::fmt::Debug + Send + Sync,
    V: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug,
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
        result.canonicalize_impl(
            [center.clone()],
            CanonicalForm::Unitary,
            "contract_src: single-site canonicalization",
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
    let mut probes = ProbeBank::new(probe_indices, 1, src_options.seed)?;
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
        });
    }
    let sketch_options = src_options.sketch_options(svd_policy.is_some());
    let mut prefixes =
        PrefixCache::new(&local, &outputs, &mut probes, sketch_options.rank_increment);

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
        let factor = T::ones(std::slice::from_ref(&cap)).map_err(|error| {
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
            },
            |column| {
                let prefix = prefixes.column(last - 1, column)?;
                contract_prefix_with_site_pair(&prefix, local[last].0, local[last].1).map_err(
                    |error| anyhow::anyhow!("contract_src: last-site sketch failed: {error}"),
                )
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
        // `tensor_a`/`tensor_b`/`right_environment` are the same for every
        // probe column at this site -- only `prefix` (fetched from
        // `prefixes`, one column at a time) varies. Rather than repeat the
        // 3-tensor contraction chain once per single column (as many times
        // as `site_max_width` requires), fetch and stack a whole
        // `rank_increment`-sized lookahead block of prefixes into one
        // `batch`-indexed tensor via `stack_along_new_index`, contract that
        // block through `tensor_a`/`tensor_b`/`right_environment` ONCE with
        // `contract_retaining` (mirroring `contract_fixed`'s own batched
        // interior-site step just above, which already avoids this
        // per-column repetition), then split the block back into individual
        // columns via `select_indices` for `factorize_probe_columns`'s
        // per-column QR interface. `factorize_probe_columns` always requests
        // columns in strictly increasing order starting at 0, so a small
        // lookahead queue is sufficient -- no need to support random access.
        let lookahead_width = sketch_options.rank_increment.max(1);
        let mut pending_columns: std::collections::VecDeque<T> = std::collections::VecDeque::new();
        let mut next_column = 0usize;
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
            },
            |column| {
                debug_assert_eq!(
                    column, next_column,
                    "contract_src: site {site} probe columns must be requested in order"
                );
                if pending_columns.is_empty() {
                    let width = lookahead_width.min(site_max_width - next_column);
                    // `fresh_segment` returns the batch-indexed tensor
                    // `PrefixCache` already builds internally before
                    // splitting it into individual columns, whenever this
                    // request is exactly its next growth step (the common
                    // case here, since `lookahead_width == batch_size`) --
                    // avoiding the fetch-individual-columns-then-
                    // `stack_along_new_index` round trip below, which
                    // `EagerTensor::stack` (tenferro-ad) pays for by
                    // reshaping every individual input tensor before
                    // concatenating. Falls back to the fetch+stack path
                    // (still correct, just not the fast path) whenever the
                    // request isn't a fresh aligned segment -- e.g. this
                    // site's first block, requested after a different site
                    // already grew the shared cache further.
                    let (stacked, batch_index) = match prefixes.fresh_segment(next_column, width)? {
                        Some((mut segment_prefixes, batch_index)) => {
                            (segment_prefixes.swap_remove(site - 1), batch_index)
                        }
                        None => {
                            let block = (0..width)
                                .map(|offset| prefixes.column(site - 1, next_column + offset))
                                .collect::<Result<Vec<_>>>()?;
                            let block_refs = block.iter().collect::<Vec<_>>();
                            let batch_index = T::Index::new_link(width)?;
                            let stacked =
                                T::stack_along_new_index(&block_refs, batch_index.clone(), -1)
                                    .map_err(|error| {
                                        anyhow::anyhow!(
                                            "contract_src: site {site} probe batch stacking \
                                                 failed: {error}"
                                        )
                                    })?;
                            (stacked, batch_index)
                        }
                    };
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
                    let after_env =
                        contract_retaining(&[&after_b, &right_environment], &batch_index).map_err(
                            |error| {
                                anyhow::anyhow!("contract_src: site {site} sketch failed: {error}")
                            },
                        )?;
                    for position in 0..width {
                        let single = after_env
                            .select_indices(std::slice::from_ref(&batch_index), &[position])
                            .map_err(|error| {
                                anyhow::anyhow!(
                                    "contract_src: site {site} probe batch split failed: {error}"
                                )
                            })?;
                        pending_columns.push_back(single);
                    }
                }
                next_column += 1;
                pending_columns.pop_front().ok_or_else(|| {
                    anyhow::anyhow!("contract_src: site {site} probe batch underflow")
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

struct FixedContractionRequest<'a, T, V>
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
    probes: &'a mut ProbeBank<T::Index>,
    final_svd: bool,
}

fn contract_fixed<T, V>(request: FixedContractionRequest<'_, T, V>) -> Result<TreeTN<T, V>>
where
    T: TensorLike,
    T::Index: IndexLike + Clone + Hash + Eq,
    <T::Index as IndexLike>::Id: Clone + Hash + Eq + Ord + std::fmt::Debug + Send + Sync,
    V: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug,
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
    let mut prefixes = BatchedPrefixCache::new(local, outputs, probes);
    let mut factors: Vec<Option<T>> = (0..chain.len()).map(|_| None).collect();
    let mut caps: Vec<Option<T::Index>> = (0..chain.len()).map(|_| None).collect();

    let (last_factor, last_cap, mut cap_environment) = if outputs[last].is_empty() {
        let cap = T::Index::new_link(1)?;
        let factor = T::ones(std::slice::from_ref(&cap)).map_err(|error| {
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
        )
        .map_err(|error| anyhow::anyhow!("contract_src: last-site sketch failed: {error}"))?;
        let (factor, cap) = factorize_fixed_batch(&sketch, &outputs[last], "last-site")?;
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
        )
        .map_err(|error| {
            anyhow::anyhow!("contract_src: site {site} prefix contraction failed: {error}")
        })?;
        let sketch = prefix_local
            .contract_pair(&right_environment)
            .map_err(|error| anyhow::anyhow!("contract_src: site {site} sketch failed: {error}"))?;
        let (factor, cap) = factorize_fixed_batch(&sketch, &left_indices, &format!("site {site}"))?;
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
) -> Result<(T, T::Index)>
where
    T: TensorLike,
    T::Index: IndexLike + Clone + Hash + Eq,
{
    let factorized = sketch
        .factorize_full_rank(left_indices, FactorizeAlg::QR, FactorizeCanonical::Left)
        .map_err(|error| {
            anyhow::anyhow!(
                "contract_src: {label} QR failed: {error}; sketch indices={:?}, left indices={:?}",
                sketch.external_indices(),
                left_indices
            )
        })?;
    Ok((factorized.left, factorized.bond_index))
}

struct PrefixCache<'a, T>
where
    T: TensorLike,
{
    local: &'a [(&'a T, &'a T)],
    outputs: &'a [Vec<T::Index>],
    probes: &'a mut ProbeBank<T::Index>,
    prefixes: Vec<Vec<T>>,
    batch_size: usize,
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
struct BatchedPrefixCache<'a, T>
where
    T: TensorLike,
{
    local: &'a [(&'a T, &'a T)],
    outputs: &'a [Vec<T::Index>],
    probes: &'a mut ProbeBank<T::Index>,
    combined: Vec<T>,
    combined_batch: Option<T::Index>,
    generated_width: usize,
}

impl<'a, T> BatchedPrefixCache<'a, T>
where
    T: TensorLike,
    T::Index: IndexLike + Clone + Hash + Eq,
{
    fn new(
        local: &'a [(&'a T, &'a T)],
        outputs: &'a [Vec<T::Index>],
        probes: &'a mut ProbeBank<T::Index>,
    ) -> Self {
        Self {
            local,
            outputs,
            probes,
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

impl<'a, T> PrefixCache<'a, T>
where
    T: TensorLike,
    T::Index: IndexLike + Clone + Hash + Eq,
{
    fn new(
        local: &'a [(&'a T, &'a T)],
        outputs: &'a [Vec<T::Index>],
        probes: &'a mut ProbeBank<T::Index>,
        batch_size: usize,
    ) -> Self {
        Self {
            local,
            outputs,
            probes,
            prefixes: (0..local.len() - 1).map(|_| Vec::new()).collect(),
            batch_size: batch_size.max(1),
        }
    }

    /// Compute one new segment covering `[start, start + segment_width)` for
    /// every site, splitting it into `self.prefixes`'s individual per-column
    /// storage (for [`Self::column`]) and returning the pre-split
    /// batch-indexed tensors too, so callers that want a whole fresh segment
    /// as one tensor ([`Self::fresh_segment`]) don't have to re-fetch and
    /// re-stack the columns this just split apart.
    fn grow_one_segment(
        &mut self,
        start: usize,
        segment_width: usize,
    ) -> Result<(Vec<T>, T::Index)> {
        let batch = T::Index::new_link(segment_width)?;
        let mut prefix = probed_site_pair_batch_range(
            self.local[0].0,
            self.local[0].1,
            &self.outputs[0],
            self.probes,
            start,
            segment_width,
            &batch,
        )?;
        let mut segment_prefixes = vec![prefix.clone()];
        for site in 1..self.local.len() - 1 {
            prefix = contract_prefix_with_probed_site_pair_batch_range(
                &prefix,
                self.local[site].0,
                self.local[site].1,
                &self.outputs[site],
                self.probes,
                start,
                segment_width,
                &batch,
            )?;
            segment_prefixes.push(prefix.clone());
        }
        for (site, segment) in segment_prefixes.iter().enumerate() {
            for position in 0..segment_width {
                self.prefixes[site].push(
                    segment
                        .select_indices(std::slice::from_ref(&batch), &[position])
                        .map_err(|error| {
                            anyhow::anyhow!(
                                "contract_src: prefix batch split at site {site} failed: {error}"
                            )
                        })?,
                );
            }
        }
        Ok((segment_prefixes, batch))
    }

    fn ensure_width(&mut self, width: usize) -> Result<()> {
        let current_width = self.prefixes.first().map_or(0, Vec::len);
        self.probes.extend_to(width)?;
        let mut start = current_width;
        while start < width {
            let segment_width = self.batch_size.min(width - start);
            self.grow_one_segment(start, segment_width)?;
            start += segment_width;
        }
        Ok(())
    }

    /// Return `[first_column, first_column + width)` as one batch-indexed
    /// tensor per site, directly from a freshly grown segment, when this
    /// request is exactly the cache's next growth step (`first_column`
    /// picks up where the cache left off, and `width` fits in one
    /// `batch_size`-sized chunk -- the shape lookahead-batched callers
    /// naturally produce when `width == batch_size`). Falls back to `Ok(None)`
    /// (after still ensuring the range via [`Self::ensure_width`], so
    /// [`Self::column`] stays correct either way) for any other request --
    /// already-cached range, a gap spanning multiple segments, or a request
    /// wider than one chunk -- since a single fresh segment can't represent
    /// those without re-splitting and re-combining, which is exactly the
    /// round trip this method exists to avoid paying when it isn't needed.
    fn fresh_segment(
        &mut self,
        first_column: usize,
        width: usize,
    ) -> Result<Option<(Vec<T>, T::Index)>> {
        let current_width = self.prefixes.first().map_or(0, Vec::len);
        if first_column != current_width || width == 0 || width > self.batch_size {
            self.ensure_width(first_column + width)?;
            return Ok(None);
        }
        self.probes.extend_to(first_column + width)?;
        let (segment_prefixes, batch) = self.grow_one_segment(first_column, width)?;
        Ok(Some((segment_prefixes, batch)))
    }

    fn column(&mut self, site: usize, column: usize) -> Result<T> {
        let next_batch = column
            .checked_div(self.batch_size)
            .and_then(|block| block.checked_add(1))
            .and_then(|block| block.checked_mul(self.batch_size))
            .ok_or_else(|| anyhow::anyhow!("contract_src: prefix batch width overflow"))?;
        self.ensure_width(next_batch.max(column + 1))?;
        self.prefixes
            .get(site)
            .and_then(|prefixes| prefixes.get(column))
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("contract_src: prefix column is missing"))
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
}

fn factorize_site_adaptive<T, F>(
    request: FactorizeSiteRequest<'_, T>,
    make_column: F,
) -> Result<(T, T::Index, T)>
where
    T: TensorLike,
    T::Index: IndexLike + Clone + Hash + Eq,
    F: FnMut(usize) -> Result<T>,
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
    } = request;
    let mut left = outputs.to_vec();
    if let Some(right_cap) = right_cap {
        left.push(right_cap.clone());
    }
    let (factor, cap) = factorize_probe_columns(
        &left,
        initial_width,
        maximum_width,
        src_options,
        label,
        make_column,
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
