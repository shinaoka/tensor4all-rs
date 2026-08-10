//! Partial site contraction for TreeTN.
//!
//! Provides [`partial_contract`] for selecting which site indices should be
//! contracted and which should be linked through explicit diagonal/copy
//! structure before calling the existing TreeTN contraction pipeline.

use std::collections::{HashMap, HashSet};
use std::fmt::Debug;
use std::hash::Hash;

use anyhow::{anyhow, bail, Context, Result};

use super::contraction::{contract, ContractionOptions};
use super::decompose::{factorize_tensor_to_treetn_with, TreeTopology};
use super::swap::SwapOptions;
use super::TreeTN;
use crate::error::{format_anyhow_error, SelectedIndexContractionError};
use crate::options::RestructureOptions;
use crate::site_index_network::SiteIndexNetwork;
use tensor4all_core::{
    tensordot, AnyScalar, DynIndex, FactorizeAlg, FactorizeOptions, IndexLike,
    TensorConstructionLike, TensorContractionLike, TensorDynLen, TensorIndex, TensorLike,
};

type DiagonalPairApplication<V> = (
    TreeTN<TensorDynLen, V>,
    TreeTN<TensorDynLen, V>,
    Vec<DynIndex>,
    Vec<DynIndex>,
);

/// Specification for partial site contraction between two TreeTNs.
///
/// - `contract_pairs`: Site index pairs to sum over and remove from the result.
/// - `diagonal_pairs`: Site index pairs to identify through diagonal/copy
///   structure while keeping the left-hand site leg in the result.
/// - Remaining (unmentioned) site indices pass through as external legs.
///
/// Uses `Index` objects directly (not raw IDs), following Julia ITensor-style
/// conventions.
///
/// # Examples
///
/// ```
/// use tensor4all_core::DynIndex;
/// use tensor4all_treetn::PartialContractionSpec;
///
/// let idx_a = DynIndex::new_dyn(4);
/// let idx_b = DynIndex::new_dyn(4);
/// let idx_c = DynIndex::new_dyn(3);
/// let idx_d = DynIndex::new_dyn(3);
///
/// let spec = PartialContractionSpec {
///     contract_pairs: vec![(idx_a.clone(), idx_b.clone())],
///     diagonal_pairs: vec![(idx_c.clone(), idx_d.clone())],
///     output_order: None,
/// };
///
/// assert_eq!(spec.contract_pairs.len(), 1);
/// assert_eq!(spec.diagonal_pairs.len(), 1);
/// ```
#[derive(Debug, Clone)]
pub struct PartialContractionSpec<I: IndexLike> {
    /// Site index pairs to contract (summed over, removed from result).
    pub contract_pairs: Vec<(I, I)>,
    /// Site index pairs to link through diagonal/copy structure while keeping
    /// the left-hand site index in the result.
    pub diagonal_pairs: Vec<(I, I)>,
    /// Optional order for the surviving external site indices in the result.
    ///
    /// The indices must refer to the final result indices after applying
    /// `contract_pairs` and `diagonal_pairs`. When provided, the result is
    /// post-processed so that these indices appear in the requested order.
    ///
    /// Current implementation requires that each surviving site index occupies a
    /// distinct node in the result.
    pub output_order: Option<Vec<I>>,
}

fn validate_partial_contraction_spec<T, V>(
    a: &TreeTN<T, V>,
    b: &TreeTN<T, V>,
    spec: &PartialContractionSpec<T::Index>,
) -> Result<()>
where
    T: TensorLike,
    V: Clone + Hash + Eq + Send + Sync + Debug + Ord,
    <T::Index as IndexLike>::Id: Clone + Hash + Eq + Debug + Send + Sync + Ord,
{
    let a_external_indices: HashSet<_> = a.external_indices().into_iter().collect();
    let b_external_indices: HashSet<_> = b.external_indices().into_iter().collect();

    let mut seen_a_indices = HashSet::new();
    let mut seen_b_indices = HashSet::new();

    for (kind, pairs) in [
        ("contract_pairs", &spec.contract_pairs),
        ("diagonal_pairs", &spec.diagonal_pairs),
    ] {
        for (idx_a, idx_b) in pairs {
            if idx_a.dim() != idx_b.dim() {
                bail!(
                    "partial_contract: {} index shape mismatch: {} != {}",
                    kind,
                    idx_a.dim(),
                    idx_b.dim()
                );
            }

            if !a_external_indices.contains(idx_a) {
                bail!(
                    "partial_contract: {:?} from {} not found in first TreeTN external indices",
                    idx_a,
                    kind
                );
            }
            if !b_external_indices.contains(idx_b) {
                bail!(
                    "partial_contract: {:?} from {} not found in second TreeTN external indices",
                    idx_b,
                    kind
                );
            }

            if !seen_a_indices.insert(idx_a.clone()) {
                bail!(
                    "partial_contract: first TreeTN index {:?} appears in multiple pairs",
                    idx_a
                );
            }
            if !seen_b_indices.insert(idx_b.clone()) {
                bail!(
                    "partial_contract: second TreeTN index {:?} appears in multiple pairs",
                    idx_b
                );
            }
        }
    }

    Ok(())
}

fn canonical_edge<V>(left: &V, right: &V) -> (V, V)
where
    V: Clone + Ord,
{
    if left <= right {
        (left.clone(), right.clone())
    } else {
        (right.clone(), left.clone())
    }
}

fn sorted_edge_set<V>(tn: &TreeTN<TensorDynLen, V>) -> Vec<(V, V)>
where
    V: Clone + Hash + Eq + Send + Sync + Debug + Ord,
{
    let mut edges: Vec<_> = tn
        .site_index_network()
        .edges()
        .map(|(u, v)| canonical_edge(&u, &v))
        .collect();
    edges.sort();
    edges.dedup();
    edges
}

fn compatible_union_node_names<V>(
    a: &TreeTN<TensorDynLen, V>,
    b: &TreeTN<TensorDynLen, V>,
) -> Vec<V>
where
    V: Clone + Hash + Eq + Send + Sync + Debug + Ord,
{
    let mut names: Vec<_> = a.node_names();
    names.extend(b.node_names());
    names.sort();
    names.dedup();
    names
}

fn validate_union_topology<V>(node_names: &[V], edges: &[(V, V)]) -> Result<()>
where
    V: Clone + Hash + Eq + Send + Sync + Debug + Ord,
{
    if node_names.is_empty() {
        bail!("partial_contract: networks must contain at least one node");
    }

    if edges.len() + 1 != node_names.len() {
        bail!("partial_contract: networks have incompatible topologies");
    }

    let mut adjacency: HashMap<V, Vec<V>> = node_names
        .iter()
        .cloned()
        .map(|name| (name, Vec::new()))
        .collect();
    for (u, v) in edges {
        let Some(neighbors_u) = adjacency.get_mut(u) else {
            bail!("partial_contract: union topology references unknown node");
        };
        neighbors_u.push(v.clone());
        let Some(neighbors_v) = adjacency.get_mut(v) else {
            bail!("partial_contract: union topology references unknown node");
        };
        neighbors_v.push(u.clone());
    }

    let mut seen = HashSet::new();
    let mut stack = vec![node_names[0].clone()];
    while let Some(node) = stack.pop() {
        if !seen.insert(node.clone()) {
            continue;
        }
        if let Some(neighbors) = adjacency.get(&node) {
            stack.extend(neighbors.iter().cloned());
        }
    }

    if seen.len() != node_names.len() {
        bail!("partial_contract: networks have incompatible topologies");
    }

    Ok(())
}

fn factorize_options_from_contraction_options(
    options: &ContractionOptions,
) -> Result<FactorizeOptions> {
    let mut factorize_options = match options.factorize_alg {
        FactorizeAlg::SVD => FactorizeOptions::svd(),
        FactorizeAlg::QR => FactorizeOptions::qr(),
        FactorizeAlg::LU => FactorizeOptions::lu(),
        FactorizeAlg::CI => FactorizeOptions::ci(),
    };
    if let Some(policy) = options.svd_policy {
        factorize_options = factorize_options.with_svd_policy(policy);
    }
    if let Some(rtol) = options.qr_rtol {
        factorize_options = factorize_options.with_qr_rtol(rtol);
    }
    if let Some(max_rank) = options.max_rank {
        factorize_options = factorize_options.with_max_rank(max_rank);
    }
    factorize_options.validate().map_err(|err| {
        anyhow!("partial_contract: invalid contraction factorization options: {err}")
    })?;
    Ok(factorize_options)
}

fn union_result_topology<V>(
    a: &TreeTN<TensorDynLen, V>,
    b: &TreeTN<TensorDynLen, V>,
    contracted_tensor: &TensorDynLen,
) -> Result<TreeTopology<V, DynIndex>>
where
    V: Clone + Hash + Eq + Send + Sync + Debug + Ord,
    <DynIndex as IndexLike>::Id: Clone + Hash + Eq + Ord + Debug + Send + Sync,
{
    let node_names = compatible_union_node_names(a, b);
    let mut union_edges = sorted_edge_set(a);
    union_edges.extend(sorted_edge_set(b));
    union_edges.sort();
    union_edges.dedup();
    validate_union_topology(&node_names, &union_edges)?;

    let surviving_indices: HashSet<_> = contracted_tensor.external_indices().into_iter().collect();

    let mut nodes = HashMap::new();
    for node_name in &node_names {
        let mut indices = Vec::new();

        if let Some(site_space_a) = a.site_index_network().site_space(node_name) {
            for site_idx in site_space_a {
                if surviving_indices.contains(site_idx) {
                    indices.push(site_idx.clone());
                }
            }
        }

        if let Some(site_space_b) = b.site_index_network().site_space(node_name) {
            for site_idx in site_space_b {
                if surviving_indices.contains(site_idx) && !indices.contains(site_idx) {
                    indices.push(site_idx.clone());
                }
            }
        }

        nodes.insert(node_name.clone(), indices);
    }

    Ok(TreeTopology::new(nodes, union_edges))
}

fn align_to_union_topology<V>(
    tn: &TreeTN<TensorDynLen, V>,
    node_names: &[V],
    union_edges: &[(V, V)],
) -> Result<TreeTN<TensorDynLen, V>>
where
    V: Clone + Hash + Eq + Send + Sync + Debug + Ord,
    <DynIndex as IndexLike>::Id: Clone + Hash + Eq + Ord + Debug + Send + Sync,
{
    let existing_nodes: HashSet<_> = tn.node_names().into_iter().collect();
    let existing_edges: HashSet<_> = sorted_edge_set(tn).into_iter().collect();
    let mut structural_links = HashMap::<V, Vec<DynIndex>>::new();

    for (u, v) in union_edges {
        if existing_edges.contains(&(u.clone(), v.clone())) {
            continue;
        }

        let link = DynIndex::new_dyn(1);
        structural_links
            .entry(u.clone())
            .or_default()
            .push(link.clone());
        structural_links.entry(v.clone()).or_default().push(link);
    }

    let mut tensors = Vec::with_capacity(node_names.len());
    let mut names = Vec::with_capacity(node_names.len());

    for node_name in node_names {
        let links = structural_links.remove(node_name).unwrap_or_default();
        let tensor = if existing_nodes.contains(node_name) {
            let node = tn.node_index(node_name).ok_or_else(|| {
                anyhow!(
                    "partial_contract: missing node {:?} while aligning topology",
                    node_name
                )
            })?;
            let mut tensor = tn.tensor(node).cloned().ok_or_else(|| {
                anyhow!(
                    "partial_contract: missing tensor for node {:?} while aligning topology",
                    node_name
                )
            })?;
            if !links.is_empty() {
                let link_tensor = <TensorDynLen as TensorConstructionLike>::ones(&links)
                    .context("partial_contract: failed to build dimension-1 structural links")?;
                tensor = tensor
                    .outer_product(&link_tensor)
                    .context("partial_contract: failed to attach dimension-1 structural links")?;
            }
            tensor
        } else {
            <TensorDynLen as TensorConstructionLike>::ones(&links)
                .context("partial_contract: failed to build missing-node scalar tensor")?
        };

        tensors.push(tensor);
        names.push(node_name.clone());
    }

    TreeTN::from_tensors(tensors, names)
        .context("partial_contract: failed to align TreeTN to union topology")
}

fn dense_element_count(indices: &[DynIndex]) -> Result<usize> {
    indices.iter().try_fold(1usize, |acc, index| {
        acc.checked_mul(index.dim()).ok_or_else(|| {
            anyhow!(
                "partial_contract: dense/reference element count overflowed for mismatched-topology fallback"
            )
        })
    })
}

fn dense_contract_output_indices(a_indices: &[DynIndex], b_indices: &[DynIndex]) -> Vec<DynIndex> {
    let mut b_remaining = b_indices.to_vec();
    let mut output = Vec::new();

    for idx_a in a_indices {
        if let Some(position) = b_remaining
            .iter()
            .position(|idx_b| idx_a.is_contractable(idx_b))
        {
            b_remaining.remove(position);
        } else {
            output.push(idx_a.clone());
        }
    }

    output.extend(b_remaining);
    output
}

fn ensure_dense_reference_limit(
    label: &str,
    indices: &[DynIndex],
    max_elements: usize,
) -> Result<()> {
    let elements = dense_element_count(indices)?;
    if elements > max_elements {
        bail!(
            "partial_contract: mismatched-topology dense/reference fallback would materialize {label} with {elements} elements, exceeding limit {max_elements}"
        );
    }
    Ok(())
}

fn validate_mismatched_dense_reference_fallback<V>(
    a: &TreeTN<TensorDynLen, V>,
    b: &TreeTN<TensorDynLen, V>,
    options: &ContractionOptions,
) -> Result<()>
where
    V: Clone + Hash + Eq + Send + Sync + Debug + Ord,
{
    let Some(max_elements) = options.mismatched_topology_dense_limit else {
        bail!(
            "partial_contract: mismatched topologies require an explicit dense/reference limit; call ContractionOptions::with_mismatched_topology_dense_limit(max_elements) for small reference cases"
        );
    };

    let a_external = a.external_indices();
    let b_external = b.external_indices();
    let output = dense_contract_output_indices(&a_external, &b_external);

    ensure_dense_reference_limit("first TreeTN", &a_external, max_elements)?;
    ensure_dense_reference_limit("second TreeTN", &b_external, max_elements)?;
    ensure_dense_reference_limit("contracted result", &output, max_elements)
}

fn contract_mismatched_topologies<V>(
    a: &TreeTN<TensorDynLen, V>,
    b: &TreeTN<TensorDynLen, V>,
    center: &V,
    options: ContractionOptions,
) -> Result<TreeTN<TensorDynLen, V>>
where
    V: Clone + Hash + Eq + Send + Sync + Debug + Ord,
    <DynIndex as IndexLike>::Id: Clone + Hash + Eq + Ord + Debug + Send + Sync,
{
    let node_names = compatible_union_node_names(a, b);
    let mut union_edges = sorted_edge_set(a);
    union_edges.extend(sorted_edge_set(b));
    union_edges.sort();
    union_edges.dedup();
    validate_union_topology(&node_names, &union_edges)?;

    let structural_result = (|| {
        let mut aligned_a = align_to_union_topology(a, &node_names, &union_edges)?;
        let mut aligned_b = align_to_union_topology(b, &node_names, &union_edges)?;
        connect_nodewise_outer_product_spectators(&mut aligned_a, &mut aligned_b)?;
        contract(&aligned_a, &aligned_b, center, options.clone())
            .context("partial_contract: failed contraction after aligning mismatched topologies")
    })();

    match structural_result {
        Ok(result) => return Ok(result),
        Err(err) if options.mismatched_topology_dense_limit.is_none() => {
            return Err(err);
        }
        Err(_) => {}
    }

    validate_mismatched_dense_reference_fallback(a, b, &options)?;

    let a_dense = a
        .sim_internal_inds()
        .contract_to_tensor()
        .context("partial_contract: failed to contract first mismatched-topology TreeTN")?;
    let b_dense = b
        .sim_internal_inds()
        .contract_to_tensor()
        .context("partial_contract: failed to contract second mismatched-topology TreeTN")?;
    let contracted_tensor = a_dense
        .contract_pair(&b_dense)
        .context("partial_contract: failed dense contraction for mismatched topologies")?;

    if contracted_tensor.external_indices().is_empty() {
        let mut result = TreeTN::<TensorDynLen, V>::new();
        result
            .add_tensor(center.clone(), contracted_tensor)
            .context("partial_contract: failed to wrap scalar mismatched-topology result")?;
        result
            .set_canonical_region([center.clone()])
            .context("partial_contract: failed to set canonical region for scalar result")?;
        return Ok(result);
    }

    let topology = union_result_topology(a, b, &contracted_tensor)?;
    let factorize_options = factorize_options_from_contraction_options(&options)?;
    factorize_tensor_to_treetn_with(&contracted_tensor, &topology, factorize_options, center)
        .context("partial_contract: failed to factorize mismatched-topology dense result")
}

fn apply_output_order<T, V>(result: TreeTN<T, V>, output_order: &[T::Index]) -> Result<TreeTN<T, V>>
where
    T: TensorLike,
    V: Clone + Hash + Eq + Send + Sync + Debug + Ord,
    T::Index: Clone + Hash + Eq,
    <T::Index as IndexLike>::Id: Clone + Hash + Eq + Ord + Debug + Send + Sync,
{
    let (current_indices, _) = result.all_site_indices()?;
    if output_order.len() != current_indices.len() {
        bail!(
            "partial_contract: output_order length {} does not match surviving external index count {}",
            output_order.len(),
            current_indices.len()
        );
    }

    let current_index_set: HashSet<_> = current_indices.iter().cloned().collect();
    let requested_index_set: HashSet<_> = output_order.iter().cloned().collect();
    if current_index_set != requested_index_set {
        bail!("partial_contract: output_order must contain exactly the surviving external indices");
    }

    let mut current_nodes = Vec::with_capacity(current_indices.len());
    for index in &current_indices {
        let node = result.site_index_network().find_node_by_index(index).ok_or_else(|| {
            anyhow!(
                "partial_contract: current result index {:?} is not present in the site index network",
                index
            )
        })?;
        current_nodes.push(node.clone());
    }

    let unique_current_nodes: HashSet<_> = current_nodes.iter().cloned().collect();
    if unique_current_nodes.len() != current_nodes.len() {
        bail!(
            "partial_contract: output_order currently requires at most one surviving site index per node; use partial_contract_to_site_network with an explicit target network to split surviving indices across nodes"
        );
    }

    let mut seen_requested = HashSet::new();
    let mut ordered_nodes = Vec::with_capacity(result.node_count());
    let mut ordered_node_set = HashSet::new();

    for index in output_order {
        if !seen_requested.insert(index.clone()) {
            bail!("partial_contract: output_order contains duplicate indices");
        }
        let current_node = result
            .site_index_network()
            .find_node_by_index(index)
            .ok_or_else(|| {
                anyhow!(
                    "partial_contract: output_order index {:?} is not present in the result",
                    index
                )
            })?;
        if !ordered_node_set.insert(current_node.clone()) {
            bail!(
                "partial_contract: output_order currently requires each requested index to occupy a distinct node"
            );
        }
        ordered_nodes.push(current_node.clone());
    }

    for node_name in result.node_names() {
        if ordered_node_set.insert(node_name.clone()) {
            ordered_nodes.push(node_name);
        }
    }

    let tensors = ordered_nodes
        .iter()
        .map(|node_name| {
            let node_idx = result.node_index(node_name).ok_or_else(|| {
                anyhow!(
                    "partial_contract: output_order node {:?} is not present in the result",
                    node_name
                )
            })?;
            result.tensor(node_idx).cloned().ok_or_else(|| {
                anyhow!(
                    "partial_contract: tensor for output_order node {:?} is missing",
                    node_name
                )
            })
        })
        .collect::<Result<Vec<_>>>()?;

    let mut reordered = TreeTN::from_tensors(tensors, ordered_nodes)
        .context("partial_contract: failed to rebuild result in requested output order")?;
    reordered.canonical_region = result.canonical_region.clone();
    reordered.canonical_form = result.canonical_form;
    reordered.ortho_towards = result.ortho_towards.clone();
    Ok(reordered)
}

fn diagonal_copy_value(tensor: &TensorDynLen) -> AnyScalar {
    if tensor.is_complex() {
        AnyScalar::new_complex(1.0, 0.0)
    } else {
        AnyScalar::new_real(1.0)
    }
}

fn apply_diagonal_pairs<V>(
    a: &TreeTN<TensorDynLen, V>,
    b: &TreeTN<TensorDynLen, V>,
    diagonal_pairs: &[(DynIndex, DynIndex)],
) -> Result<DiagonalPairApplication<V>>
where
    V: Clone + Hash + Eq + Send + Sync + Debug + Ord,
    <DynIndex as IndexLike>::Id: Clone + Hash + Eq + Ord + Debug + Send + Sync,
{
    let mut a_modified = a.clone();
    let mut b_modified = b.clone();
    let mut restore_from = Vec::with_capacity(diagonal_pairs.len());
    let mut restore_to = Vec::with_capacity(diagonal_pairs.len());

    for (idx_a, idx_b) in diagonal_pairs {
        let node_name = a_modified
            .site_index_network()
            .find_node_by_index(idx_a)
            .cloned()
            .ok_or_else(|| {
                anyhow!(
                    "partial_contract: diagonal pair left index {:?} is not a site index of the first TreeTN",
                    idx_a.id()
                )
            })?;
        let node_idx = a_modified.node_index(&node_name).ok_or_else(|| {
            anyhow!(
                "partial_contract: node {:?} for left diagonal index {:?} not found",
                node_name,
                idx_a.id()
            )
        })?;
        let local_tensor = a_modified.tensor(node_idx).cloned().ok_or_else(|| {
            anyhow!(
                "partial_contract: tensor for node {:?} not found while processing diagonal pair {:?}",
                node_name,
                idx_a.id()
            )
        })?;

        let aux_index = idx_a.sim();
        let kept_index = idx_a.sim();
        let copy_tensor = TensorDynLen::copy_tensor(
            vec![idx_a.clone(), aux_index.clone(), kept_index.clone()],
            diagonal_copy_value(&local_tensor),
        )
        .with_context(|| {
            format!(
                "partial_contract: failed to build copy tensor for diagonal pair {:?} <- {:?}",
                idx_a.id(),
                idx_b.id()
            )
        })?;
        let expanded_tensor = tensordot(
            &local_tensor,
            &copy_tensor,
            &[(idx_a.clone(), idx_a.clone())],
        )
        .with_context(|| {
            format!(
                "partial_contract: failed to apply diagonal structure for pair {:?} <- {:?}",
                idx_a.id(),
                idx_b.id()
            )
        })?;
        a_modified
            .replace_tensor(node_idx, expanded_tensor)
            .with_context(|| {
                format!(
                    "partial_contract: failed to replace tensor at node {:?} for diagonal pair {:?}",
                    node_name,
                    idx_a.id()
                )
            })?
            .ok_or_else(|| {
                anyhow!(
                    "partial_contract: node {:?} disappeared while processing diagonal pair {:?}",
                    node_name,
                    idx_a.id()
                )
            })?;

        b_modified = b_modified.replaceind(idx_b, &aux_index).with_context(|| {
            format!(
                "partial_contract: failed to align diagonal pair {:?} <- {:?}",
                idx_a.id(),
                idx_b.id()
            )
        })?;

        restore_from.push(kept_index);
        restore_to.push(idx_a.clone());
    }

    Ok((a_modified, b_modified, restore_from, restore_to))
}

fn align_contract_pair_site_nodes<V>(
    a: &TreeTN<TensorDynLen, V>,
    b: &mut TreeTN<TensorDynLen, V>,
    contract_pairs: &[(DynIndex, DynIndex)],
) -> Result<()>
where
    V: Clone + Hash + Eq + Send + Sync + Debug + Ord,
    <DynIndex as IndexLike>::Id: Clone + Hash + Eq + Ord + Debug + Send + Sync,
{
    let mut target_assignment = HashMap::new();

    for (idx_a, _) in contract_pairs {
        let left_node = a
            .site_index_network()
            .find_node_by_index(idx_a)
            .cloned()
            .ok_or_else(|| {
                anyhow!(
                    "partial_contract: contract pair left index {:?} is not a site index of the first TreeTN",
                    idx_a.id()
                )
            })?;
        let right_node = b
            .site_index_network()
            .find_node_by_index(idx_a)
            .cloned()
            .ok_or_else(|| {
                anyhow!(
                    "partial_contract: aligned contract index {:?} is not a site index of the second TreeTN",
                    idx_a.id()
                )
            })?;

        if left_node != right_node {
            target_assignment.insert(idx_a.clone(), left_node);
        }
    }

    b.swap_site_indices(&target_assignment, &SwapOptions::default())
        .context("partial_contract: failed to move aligned contract indices to matching nodes")?;
    Ok(())
}

fn has_contractable_index_pair(left: &[DynIndex], right: &[DynIndex]) -> bool {
    left.iter().any(|idx_left| {
        right
            .iter()
            .any(|idx_right| idx_left.is_contractable(idx_right))
    })
}

fn connect_nodewise_outer_product_spectators<V>(
    a: &mut TreeTN<TensorDynLen, V>,
    b: &mut TreeTN<TensorDynLen, V>,
) -> Result<()>
where
    V: Clone + Hash + Eq + Send + Sync + Debug + Ord,
    <DynIndex as IndexLike>::Id: Clone + Hash + Eq + Ord + Debug + Send + Sync,
{
    let mut node_names = a.node_names();
    node_names.retain(|node_name| b.node_index(node_name).is_some());
    node_names.sort();

    for node_name in node_names {
        let node_a = a
            .node_index(&node_name)
            .context("partial_contract: node disappeared while adding dummy contraction links")?;
        let node_b = b.node_index(&node_name).context(
            "partial_contract: matching node disappeared while adding dummy contraction links",
        )?;

        let tensor_a = a
            .tensor(node_a)
            .cloned()
            .context("partial_contract: missing tensor while adding dummy contraction links")?;
        let tensor_b = b.tensor(node_b).cloned().context(
            "partial_contract: missing matching tensor while adding dummy contraction links",
        )?;

        if has_contractable_index_pair(&tensor_a.external_indices(), &tensor_b.external_indices()) {
            continue;
        }

        let (dummy_a, dummy_b) = DynIndex::create_dummy_link_pair();
        let dummy_tensor_a =
            <TensorDynLen as TensorConstructionLike>::ones(std::slice::from_ref(&dummy_a))
                .context("partial_contract: failed to build dummy contraction link")?;
        let dummy_tensor_b =
            <TensorDynLen as TensorConstructionLike>::ones(std::slice::from_ref(&dummy_b))
                .context("partial_contract: failed to build matching dummy contraction link")?;
        let expanded_a = tensor_a
            .outer_product(&dummy_tensor_a)
            .context("partial_contract: failed to attach dummy contraction link")?;
        let expanded_b = tensor_b
            .outer_product(&dummy_tensor_b)
            .context("partial_contract: failed to attach matching dummy contraction link")?;

        a.replace_tensor(node_a, expanded_a)
            .context(
                "partial_contract: failed to replace tensor after adding dummy contraction link",
            )?
            .context("partial_contract: node disappeared after adding dummy contraction link")?;
        b.replace_tensor(node_b, expanded_b)
            .context(
                "partial_contract: failed to replace matching tensor after adding dummy contraction link",
            )?
            .context(
                "partial_contract: matching node disappeared after adding dummy contraction link",
            )?;
    }

    Ok(())
}

/// Partially contract two TreeTNs according to the given specification.
///
/// # Arguments
/// * `a` - First tensor network
/// * `b` - Second tensor network
/// * `spec` - Which site indices to contract versus link through diagonal
///   structure
/// * `center` - Canonical center node for the result
/// * `options` - Contraction algorithm options
///
/// # Index handling
///
/// - **contract_pairs**: Both indices are traced over (inner product).
///   Neither appears in the result.
/// - **diagonal_pairs**: The two indices are linked through explicit diagonal
///   structure so that only matching values contribute, while the left-hand site
///   index remains in the result.
/// - **Unmentioned indices**: Pass through unchanged as external legs.
///
/// # Examples
///
/// ```
/// use tensor4all_core::{DynIndex, TensorDynLen};
/// use tensor4all_treetn::{
///     contraction::ContractionOptions,
///     partial_contract,
///     PartialContractionSpec,
///     TreeTN,
/// };
///
/// let idx_a = DynIndex::new_dyn(2);
/// let idx_b = DynIndex::new_dyn(2);
/// let a = TreeTN::<TensorDynLen, usize>::from_tensors(
///     vec![TensorDynLen::from_dense(vec![idx_a.clone()], vec![1.0, 2.0]).unwrap()],
///     vec![0],
/// ).unwrap();
/// let b = TreeTN::<TensorDynLen, usize>::from_tensors(
///     vec![TensorDynLen::from_dense(vec![idx_b.clone()], vec![3.0, 4.0]).unwrap()],
///     vec![0],
/// ).unwrap();
///
/// let spec = PartialContractionSpec {
///     contract_pairs: vec![(idx_a.clone(), idx_b.clone())],
///     diagonal_pairs: vec![],
///     output_order: None,
/// };
///
/// let result = partial_contract(&a, &b, &spec, &0usize, ContractionOptions::default()).unwrap();
/// let scalar = result.contract_to_tensor().unwrap().only().unwrap().real();
/// assert!((scalar - 11.0).abs() < 1.0e-12);
/// ```
pub fn partial_contract<V>(
    a: &TreeTN<TensorDynLen, V>,
    b: &TreeTN<TensorDynLen, V>,
    spec: &PartialContractionSpec<DynIndex>,
    center: &V,
    options: ContractionOptions,
) -> Result<TreeTN<TensorDynLen, V>>
where
    V: Clone + Hash + Eq + Send + Sync + Debug + Ord,
    <DynIndex as IndexLike>::Id: Clone + Hash + Eq + Ord + Debug + Send + Sync,
{
    validate_partial_contraction_spec(a, b, spec)?;

    let (mut a_modified, mut b_modified, restore_from, restore_to) =
        apply_diagonal_pairs(a, b, &spec.diagonal_pairs)?;

    for (idx_a, idx_b) in &spec.contract_pairs {
        b_modified = b_modified.replaceind(idx_b, idx_a).with_context(|| {
            format!(
                "partial_contract: failed to align contract pair {:?} <- {:?}",
                idx_a.id(),
                idx_b.id()
            )
        })?;
    }

    let mut result = if a_modified.same_topology(&b_modified) {
        align_contract_pair_site_nodes(&a_modified, &mut b_modified, &spec.contract_pairs)?;
        connect_nodewise_outer_product_spectators(&mut a_modified, &mut b_modified)?;
        contract(&a_modified, &b_modified, center, options)
            .context("partial_contract: contraction failed")?
    } else {
        contract_mismatched_topologies(&a_modified, &b_modified, center, options)?
    };

    if !restore_from.is_empty() {
        result = result.replaceinds(&restore_from, &restore_to).context(
            "partial_contract: failed to restore surviving left-hand indices after diagonal pairing",
        )?;
    }

    if let Some(output_order) = &spec.output_order {
        apply_output_order(result, output_order)
    } else {
        Ok(result)
    }
}

/// Partially contract two TreeTNs and restructure the result to a target site network.
///
/// Use this when the surviving site indices need a specific output topology,
/// including cases where several surviving indices initially occupy the same
/// result node. The contraction itself is performed by [`partial_contract`]
/// without `output_order`; the returned TreeTN is then transformed with
/// [`TreeTN::restructure_to`].
///
/// # Arguments
/// * `a` - First tensor network. Left indices in `spec` must be site indices of this network.
/// * `b` - Second tensor network. Right indices in `spec` must be site indices of this network.
/// * `spec` - Site-index contraction and diagonal-pair specification. `output_order`
///   must be `None` because `target` supplies the output layout.
/// * `center` - Canonical center node used for the intermediate contraction.
/// * `target` - Target site-index network containing exactly the surviving result
///   indices, assigned to the desired output nodes and topology.
/// * `options` - Contraction algorithm options.
/// * `restructure_options` - Split, swap, and optional final truncation settings
///   used when transforming the intermediate result to `target`.
///
/// # Returns
/// A TreeTN with node names and site-index assignment matching `target`.
///
/// # Errors
/// Returns an error if `spec.output_order` is set, if the partial contraction
/// fails, or if the contracted result cannot be restructured to `target`.
///
/// # Examples
///
/// ```
/// use std::collections::HashSet;
///
/// use tensor4all_core::{DynIndex, TensorDynLen, TensorIndex};
/// use tensor4all_treetn::{
///     contraction::ContractionOptions,
///     partial_contract_to_site_network,
///     PartialContractionSpec,
///     RestructureOptions,
///     SiteIndexNetwork,
///     TreeTN,
/// };
///
/// let i = DynIndex::new_dyn(2);
/// let k_left = DynIndex::new_dyn(2);
/// let k_right = DynIndex::new_dyn(2);
/// let j = DynIndex::new_dyn(2);
///
/// let a = TreeTN::<TensorDynLen, &str>::from_tensors(
///     vec![TensorDynLen::from_dense(
///         vec![i.clone(), k_left.clone()],
///         vec![1.0, 2.0, 3.0, 4.0],
///     ).unwrap()],
///     vec!["center"],
/// ).unwrap();
/// let b = TreeTN::<TensorDynLen, &str>::from_tensors(
///     vec![TensorDynLen::from_dense(
///         vec![k_right.clone(), j.clone()],
///         vec![5.0, 6.0, 7.0, 8.0],
///     ).unwrap()],
///     vec!["center"],
/// ).unwrap();
///
/// let spec = PartialContractionSpec {
///     contract_pairs: vec![(k_left, k_right)],
///     diagonal_pairs: vec![],
///     output_order: None,
/// };
///
/// let mut target = SiteIndexNetwork::new();
/// target.add_node("0_row", HashSet::from([i.clone()])).unwrap();
/// target.add_node("1_col", HashSet::from([j.clone()])).unwrap();
/// target.add_edge(&"0_row", &"1_col").unwrap();
///
/// let result = partial_contract_to_site_network(
///     &a,
///     &b,
///     &spec,
///     &"center",
///     &target,
///     ContractionOptions::default(),
///     &RestructureOptions::default(),
/// ).unwrap();
/// let dense = result.to_dense().unwrap();
///
/// assert_eq!(dense.external_indices(), vec![i.clone(), j.clone()]);
/// let expected = vec![23.0, 34.0, 31.0, 46.0];
/// for (actual, expected) in dense.to_vec::<f64>().unwrap().into_iter().zip(expected) {
///     assert!((actual - expected).abs() < 1e-12);
/// }
/// assert_eq!(result.site_index_network().find_node_by_index(&i), Some(&"0_row"));
/// assert_eq!(result.site_index_network().find_node_by_index(&j), Some(&"1_col"));
/// ```
pub fn partial_contract_to_site_network<V, TargetV>(
    a: &TreeTN<TensorDynLen, V>,
    b: &TreeTN<TensorDynLen, V>,
    spec: &PartialContractionSpec<DynIndex>,
    center: &V,
    target: &SiteIndexNetwork<TargetV, DynIndex>,
    options: ContractionOptions,
    restructure_options: &RestructureOptions,
) -> Result<TreeTN<TensorDynLen, TargetV>>
where
    V: Clone + Hash + Eq + Send + Sync + Debug + Ord,
    TargetV: Clone + Hash + Eq + Send + Sync + Debug + Ord,
    <DynIndex as IndexLike>::Id: Clone + Hash + Eq + Ord + Debug + Send + Sync,
{
    if spec.output_order.is_some() {
        bail!(
            "partial_contract_to_site_network: spec.output_order must be None because the target site network defines the output layout"
        );
    }

    let result = partial_contract(a, b, spec, center, options)?;
    result.restructure_to(target, restructure_options).context(
        "partial_contract_to_site_network: failed to restructure result to target site network",
    )
}

/// Multiply two TreeTNs elementwise along selected external index pairs.
///
/// This is a convenience wrapper around [`partial_contract`] using diagonal
/// pairs. For each `(left_index, right_index)` pair, only matching coordinates
/// contribute and the left index remains in the result.
///
/// # Arguments
/// * `left` - Left operand. Paired left indices are preserved in the output.
/// * `right` - Right operand. Paired right indices are matched diagonally.
/// * `index_pairs` - External index pairs with equal dimensions.
/// * `center` - Canonical center node for the result.
/// * `options` - Contraction algorithm options.
///
/// # Returns
/// A TreeTN representing the selected-index Hadamard product.
///
/// # Errors
/// Returns an error if a pair has mismatched dimensions, references a
/// non-external index, or the underlying contraction fails.
///
/// # Examples
/// ```
/// use tensor4all_core::{DynIndex, TensorDynLen, TensorIndex};
/// use tensor4all_treetn::{contraction::ContractionOptions, hadamard, TreeTN};
///
/// let i = DynIndex::new_dyn(2);
/// let j = DynIndex::new_dyn(2);
/// let left = TreeTN::<TensorDynLen, usize>::from_tensors(
///     vec![TensorDynLen::from_dense(vec![i.clone()], vec![2.0, 3.0]).unwrap()],
///     vec![0],
/// ).unwrap();
/// let right = TreeTN::<TensorDynLen, usize>::from_tensors(
///     vec![TensorDynLen::from_dense(vec![j.clone()], vec![5.0, 7.0]).unwrap()],
///     vec![0],
/// ).unwrap();
///
/// let result = hadamard(&left, &right, &[(i.clone(), j)], &0, ContractionOptions::default()).unwrap();
/// let dense = result.to_dense().unwrap();
/// assert_eq!(dense.external_indices(), vec![i]);
/// assert_eq!(dense.to_vec::<f64>().unwrap(), vec![10.0, 21.0]);
/// ```
pub fn hadamard<V>(
    left: &TreeTN<TensorDynLen, V>,
    right: &TreeTN<TensorDynLen, V>,
    index_pairs: &[(DynIndex, DynIndex)],
    center: &V,
    options: ContractionOptions,
) -> std::result::Result<TreeTN<TensorDynLen, V>, SelectedIndexContractionError>
where
    V: Clone + Hash + Eq + Send + Sync + Debug + Ord,
    <DynIndex as IndexLike>::Id: Clone + Hash + Eq + Ord + Debug + Send + Sync,
{
    let spec = PartialContractionSpec {
        contract_pairs: Vec::new(),
        diagonal_pairs: index_pairs.to_vec(),
        output_order: None,
    };
    partial_contract(left, right, &spec, center, options).map_err(|error| {
        SelectedIndexContractionError::PartialContractFailed {
            message: format_anyhow_error(error),
        }
    })
}

/// Sum over selected external index pairs with a weight TreeTN.
///
/// This contracts each `(state_index, weight_index)` pair and leaves all
/// unpaired state and weight indices external.
///
/// # Arguments
/// * `state` - Left operand whose selected indices are summed over.
/// * `weights` - Right operand providing weights for the selected coordinates.
/// * `index_pairs` - External index pairs to contract.
/// * `center` - Canonical center node for the result.
/// * `options` - Contraction algorithm options.
///
/// # Returns
/// A TreeTN after summing over all selected pairs.
///
/// # Errors
/// Returns an error if a pair has mismatched dimensions, references a
/// non-external index, or the underlying contraction fails.
///
/// # Examples
/// ```
/// use tensor4all_core::{DynIndex, TensorDynLen, TensorIndex};
/// use tensor4all_treetn::{
///     contraction::ContractionOptions,
///     weighted_sum_over_index_pairs,
///     TreeTN,
/// };
///
/// let x = DynIndex::new_dyn(2);
/// let z = DynIndex::new_dyn(2);
/// let wz = DynIndex::new_dyn(2);
/// let state = TreeTN::<TensorDynLen, usize>::from_tensors(
///     vec![TensorDynLen::from_dense(vec![x.clone(), z.clone()], vec![1.0, 2.0, 3.0, 4.0]).unwrap()],
///     vec![0],
/// ).unwrap();
/// let weights = TreeTN::<TensorDynLen, usize>::from_tensors(
///     vec![TensorDynLen::from_dense(vec![wz.clone()], vec![10.0, 100.0]).unwrap()],
///     vec![0],
/// ).unwrap();
///
/// let result = weighted_sum_over_index_pairs(
///     &state,
///     &weights,
///     &[(z, wz)],
///     &0,
///     ContractionOptions::default(),
/// ).unwrap();
/// let dense = result.to_dense().unwrap();
/// assert_eq!(dense.external_indices(), vec![x]);
/// assert_eq!(dense.to_vec::<f64>().unwrap(), vec![310.0, 420.0]);
/// ```
pub fn weighted_sum_over_index_pairs<V>(
    state: &TreeTN<TensorDynLen, V>,
    weights: &TreeTN<TensorDynLen, V>,
    index_pairs: &[(DynIndex, DynIndex)],
    center: &V,
    options: ContractionOptions,
) -> std::result::Result<TreeTN<TensorDynLen, V>, SelectedIndexContractionError>
where
    V: Clone + Hash + Eq + Send + Sync + Debug + Ord,
    <DynIndex as IndexLike>::Id: Clone + Hash + Eq + Ord + Debug + Send + Sync,
{
    let spec = PartialContractionSpec {
        contract_pairs: index_pairs.to_vec(),
        diagonal_pairs: Vec::new(),
        output_order: None,
    };
    partial_contract(state, weights, &spec, center, options).map_err(|error| {
        SelectedIndexContractionError::PartialContractFailed {
            message: format_anyhow_error(error),
        }
    })
}

/// Sum a TreeTN over selected external indices using factorized unit weights.
///
/// This constructs an all-ones weight TreeTN on the same topology as `state`
/// and delegates to [`weighted_sum_over_index_pairs`].
///
/// # Arguments
/// * `state` - Input TreeTN.
/// * `sum_indices` - External indices of `state` to sum over. Each index must
///   be present exactly once in the list.
/// * `center` - Canonical center node for the result.
/// * `options` - Contraction algorithm options.
///
/// # Returns
/// A TreeTN with `sum_indices` contracted away. If `sum_indices` is empty, this
/// returns `state.clone()`.
///
/// # Errors
/// Returns an error if an index is duplicated, is not external to `state`, or
/// the underlying contraction fails.
///
/// # Examples
/// ```
/// use tensor4all_core::{DynIndex, TensorDynLen, TensorIndex};
/// use tensor4all_treetn::{contraction::ContractionOptions, sum_over_indices, TreeTN};
///
/// let x = DynIndex::new_dyn(2);
/// let z = DynIndex::new_dyn(2);
/// let state = TreeTN::<TensorDynLen, usize>::from_tensors(
///     vec![TensorDynLen::from_dense(vec![x.clone(), z.clone()], vec![1.0, 2.0, 3.0, 4.0]).unwrap()],
///     vec![0],
/// ).unwrap();
///
/// let result = sum_over_indices(&state, &[z], &0, ContractionOptions::default()).unwrap();
/// let dense = result.to_dense().unwrap();
/// assert_eq!(dense.external_indices(), vec![x]);
/// assert_eq!(dense.to_vec::<f64>().unwrap(), vec![4.0, 6.0]);
/// ```
pub fn sum_over_indices<V>(
    state: &TreeTN<TensorDynLen, V>,
    sum_indices: &[DynIndex],
    center: &V,
    options: ContractionOptions,
) -> std::result::Result<TreeTN<TensorDynLen, V>, SelectedIndexContractionError>
where
    V: Clone + Hash + Eq + Send + Sync + Debug + Ord,
    <DynIndex as IndexLike>::Id: Clone + Hash + Eq + Ord + Debug + Send + Sync,
{
    if sum_indices.is_empty() {
        return Ok(state.clone());
    }

    let mut seen = HashSet::new();
    for index in sum_indices {
        if !seen.insert(index.clone()) {
            return Err(SelectedIndexContractionError::DuplicateIndex {
                index: format!("{index:?}"),
            });
        }
    }

    let mut node_names = state.node_names();
    node_names.sort();

    let mut state_index_to_node: HashMap<DynIndex, V> = HashMap::new();
    for node in &node_names {
        let Some(site_space) = state.site_space(node) else {
            continue;
        };
        for index in site_space {
            if seen.contains(index)
                && state_index_to_node
                    .insert(index.clone(), node.clone())
                    .is_some()
            {
                return Err(SelectedIndexContractionError::IndexInMultipleSiteSpaces {
                    index: format!("{index:?}"),
                });
            }
        }
    }

    let mut weight_indices_by_node: HashMap<V, Vec<DynIndex>> = HashMap::new();
    let mut index_pairs = Vec::with_capacity(sum_indices.len());
    for index in sum_indices {
        let node = state_index_to_node.get(index).ok_or_else(|| {
            SelectedIndexContractionError::IndexNotFound {
                index: format!("{index:?}"),
            }
        })?;
        let weight_index = index.sim();
        weight_indices_by_node
            .entry(node.clone())
            .or_default()
            .push(weight_index.clone());
        index_pairs.push((index.clone(), weight_index));
    }

    let mut link_indices_by_node: HashMap<V, Vec<DynIndex>> = HashMap::new();
    let mut edges: Vec<_> = state.site_index_network().edges().collect();
    edges.sort();
    for (left, right) in edges {
        let link = DynIndex::new_dyn(1);
        link_indices_by_node
            .entry(left.clone())
            .or_default()
            .push(link.clone());
        link_indices_by_node.entry(right).or_default().push(link);
    }

    let mut tensors = Vec::with_capacity(node_names.len());
    for node in &node_names {
        let mut indices = weight_indices_by_node.remove(node).unwrap_or_default();
        if let Some(mut links) = link_indices_by_node.remove(node) {
            indices.append(&mut links);
        }
        tensors.push(
            <TensorDynLen as TensorConstructionLike>::ones(&indices).map_err(|error| {
                SelectedIndexContractionError::BuildOnesTensor {
                    node: format!("{node:?}"),
                    message: format_anyhow_error(anyhow::Error::new(error)),
                }
            })?,
        );
    }

    let weights = TreeTN::from_tensors(tensors, node_names).map_err(|error| {
        SelectedIndexContractionError::BuildWeightsTree {
            message: format_anyhow_error(error),
        }
    })?;
    weighted_sum_over_index_pairs(state, &weights, &index_pairs, center, options)
}

#[cfg(test)]
mod tests;
