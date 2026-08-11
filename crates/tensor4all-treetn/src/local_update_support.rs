//! Shared support routines for local sweep algorithms.
//!
//! These helpers are intentionally crate-private. They encode the local-update
//! conventions shared by linsolve, DMRG, and future TDVP-style algorithms:
//! contract only the active region, pass multi-tensor contractions to
//! `T::contract(&refs)`, and never materialize the full TreeTN in production
//! update paths.

use std::collections::HashMap;
use std::hash::Hash;

use anyhow::Result;
use tensor4all_core::{IndexLike, TensorLike};
use thiserror::Error;

use crate::operator::{IndexMapping, LinearOperator};
use crate::treetn::{get_boundary_edges, LocalUpdateStep, TreeTN, TreeTopology};

pub(crate) type SiteMappings<V, I> = (HashMap<V, IndexMapping<I>>, HashMap<V, IndexMapping<I>>);

#[derive(Debug, Error)]
pub(crate) enum SquareSiteMappingError {
    #[error(
        "square local update requires exactly one state site index at node {node}, found {count}"
    )]
    UnsupportedStateSiteCount { node: String, count: usize },
    #[error(
        "square local update requires exactly one {role} mapping at node {node}, found {count}"
    )]
    UnsupportedMultipleSiteMappings {
        node: String,
        role: &'static str,
        count: usize,
    },
    #[error("square local update missing {role} mapping at node {node}")]
    MissingMapping { node: String, role: &'static str },
    #[error("invalid square local update mapping at node {node}: {reason}")]
    InvalidMapping { node: String, reason: String },
}

pub(crate) fn single_site_square_mappings<T, V>(
    operator: &LinearOperator<T, V>,
    state: &TreeTN<T, V>,
) -> Result<SiteMappings<V, T::Index>, SquareSiteMappingError>
where
    T: TensorLike,
    T::Index: IndexLike,
    <T::Index as IndexLike>::Id: Clone + std::hash::Hash + Eq + Ord + std::fmt::Debug + Send + Sync,
    V: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug,
{
    let mut input = HashMap::new();
    let mut output = HashMap::new();

    for node in state.node_names() {
        let node_name = format!("{node:?}");
        let state_sites = state.site_space(&node).ok_or_else(|| {
            SquareSiteMappingError::UnsupportedStateSiteCount {
                node: node_name.clone(),
                count: 0,
            }
        })?;
        if state_sites.len() != 1 {
            return Err(SquareSiteMappingError::UnsupportedStateSiteCount {
                node: node_name,
                count: state_sites.len(),
            });
        }
        let state_site = state_sites.iter().next().ok_or_else(|| {
            SquareSiteMappingError::UnsupportedStateSiteCount {
                node: node_name.clone(),
                count: 0,
            }
        })?;

        let in_mapping = single_mapping(
            operator.input_mappings().get(&node).map(Vec::as_slice),
            &node,
            "input",
        )?;
        let out_mapping = single_mapping(
            operator.output_mappings().get(&node).map(Vec::as_slice),
            &node,
            "output",
        )?;

        if &in_mapping.true_index != state_site {
            return Err(SquareSiteMappingError::InvalidMapping {
                node: format!("{node:?}"),
                reason: "input true index does not match the state site index".to_string(),
            });
        }
        if &out_mapping.true_index != state_site {
            return Err(SquareSiteMappingError::InvalidMapping {
                node: format!("{node:?}"),
                reason:
                    "output true index must equal the state site index for square local updates"
                        .to_string(),
            });
        }
        if in_mapping.internal_index.dim() != state_site.dim()
            || out_mapping.internal_index.dim() != state_site.dim()
        {
            return Err(SquareSiteMappingError::InvalidMapping {
                node: format!("{node:?}"),
                reason: "operator internal mapping dimensions must match the state site dimension"
                    .to_string(),
            });
        }

        input.insert(node.clone(), in_mapping.clone());
        output.insert(node, out_mapping.clone());
    }

    Ok((input, output))
}

fn single_mapping<'a, I, V>(
    mappings: Option<&'a [IndexMapping<I>]>,
    node: &V,
    role: &'static str,
) -> Result<&'a IndexMapping<I>, SquareSiteMappingError>
where
    I: IndexLike,
    V: std::fmt::Debug,
{
    let mappings = mappings.ok_or_else(|| SquareSiteMappingError::MissingMapping {
        node: format!("{node:?}"),
        role,
    })?;
    if mappings.len() != 1 {
        return Err(SquareSiteMappingError::UnsupportedMultipleSiteMappings {
            node: format!("{node:?}"),
            role,
            count: mappings.len(),
        });
    }
    Ok(&mappings[0])
}

/// Initialize a reference state by cloning the ket state and relabeling links.
pub(crate) fn initialize_reference_state_if_empty<T, V>(
    reference_state: &mut TreeTN<T, V>,
    ket_state: &TreeTN<T, V>,
) -> Result<()>
where
    T: TensorLike,
    T::Index: IndexLike,
    <T::Index as IndexLike>::Id: Clone + std::hash::Hash + Eq + Ord + std::fmt::Debug + Send + Sync,
    V: Clone + Hash + Eq + Send + Sync + std::fmt::Debug,
{
    if !reference_state.node_names().is_empty() {
        return Ok(());
    }

    let mut initialized = ket_state.clone();
    initialized.sim_link_indices_mut()?;
    *reference_state = initialized;
    Ok(())
}

/// Contract all tensors in a local region into one tensor.
pub(crate) fn contract_region<T, V>(subtree: &TreeTN<T, V>, region: &[V]) -> Result<T>
where
    T: TensorLike,
    V: Clone + Hash + Eq + Send + Sync + std::fmt::Debug,
{
    if region.is_empty() {
        return Err(anyhow::anyhow!("Region cannot be empty"));
    }

    let tensors: Vec<T> = region
        .iter()
        .map(|node| {
            let idx = subtree
                .node_index(node)
                .ok_or_else(|| anyhow::anyhow!("Node {:?} not found in subtree", node))?;
            let tensor = subtree
                .tensor(idx)
                .ok_or_else(|| anyhow::anyhow!("Tensor not found for node {:?}", node))?;
            Ok(tensor.clone())
        })
        .collect::<Result<_>>()?;

    let tensor_refs: Vec<&T> = tensors.iter().collect();
    T::contract(&tensor_refs).map_err(anyhow::Error::new)
}

/// Build a decomposition topology for a solved local tensor.
pub(crate) fn build_subtree_topology<T, V>(
    solved_tensor: &T,
    region: &[V],
    full_treetn: &TreeTN<T, V>,
) -> Result<TreeTopology<V, T::Index>>
where
    T: TensorLike,
    T::Index: IndexLike,
    <T::Index as IndexLike>::Id: Clone + std::hash::Hash + Eq + Ord + std::fmt::Debug + Send + Sync,
    V: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug,
{
    let mut nodes: HashMap<V, Vec<T::Index>> = HashMap::new();
    let mut edges: Vec<(V, V)> = Vec::new();

    let solved_indices = solved_tensor.external_indices();

    for node in region {
        let mut indices = Vec::new();

        if let Some(site_indices) = full_treetn.site_space(node) {
            for site_idx in site_indices {
                if solved_indices.iter().any(|idx| idx == site_idx) {
                    indices.push(site_idx.clone());
                }
            }
        }

        for neighbor in full_treetn.site_index_network().neighbors(node) {
            if !region.contains(&neighbor) {
                if let Some(edge) = full_treetn.edge_between(node, &neighbor) {
                    if let Some(bond) = full_treetn.bond_index(edge) {
                        if solved_indices.iter().any(|idx| idx == bond) {
                            indices.push(bond.clone());
                        }
                    }
                }
            }
        }

        nodes.insert(node.clone(), indices);
    }

    for (i, node_a) in region.iter().enumerate() {
        for node_b in region.iter().skip(i + 1) {
            if full_treetn.edge_between(node_a, node_b).is_some() {
                edges.push((node_a.clone(), node_b.clone()));
            }
        }
    }

    Ok(TreeTopology::new(nodes, edges))
}

/// Copy a decomposed local tensor network back into the extracted subtree.
pub(crate) fn copy_decomposed_to_subtree<T, V>(
    subtree: &mut TreeTN<T, V>,
    decomposed: &TreeTN<T, V>,
    region: &[V],
    full_treetn: &TreeTN<T, V>,
) -> Result<()>
where
    T: TensorLike,
    T::Index: IndexLike,
    <T::Index as IndexLike>::Id: Clone + std::hash::Hash + Eq + Ord + std::fmt::Debug + Send + Sync,
    V: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug,
{
    let mut bond_mapping: HashMap<T::Index, T::Index> = HashMap::new();

    for (i, node_a) in region.iter().enumerate() {
        for node_b in region.iter().skip(i + 1) {
            if let Some(decomp_edge) = decomposed.edge_between(node_a, node_b) {
                if let Some(decomp_bond) = decomposed.bond_index(decomp_edge) {
                    if let Some(orig_edge) = subtree.edge_between(node_a, node_b) {
                        let new_bond = decomp_bond.sim();
                        bond_mapping.insert(decomp_bond.clone(), new_bond.clone());
                        subtree.replace_edge_bond(orig_edge, new_bond)?;
                    }
                }
            }
        }
    }

    for node in region {
        let decomp_idx = decomposed
            .node_index(node)
            .ok_or_else(|| anyhow::anyhow!("Node {:?} not found in decomposed TreeTN", node))?;
        let mut new_tensor = decomposed
            .tensor(decomp_idx)
            .ok_or_else(|| anyhow::anyhow!("Tensor not found for node {:?}", node))?
            .clone();

        for neighbor in full_treetn.site_index_network().neighbors(node) {
            if region.contains(&neighbor) {
                if let Some(decomp_edge) = decomposed.edge_between(node, &neighbor) {
                    if let Some(decomp_bond) = decomposed.bond_index(decomp_edge) {
                        if let Some(new_bond) = bond_mapping.get(decomp_bond) {
                            new_tensor = new_tensor.replaceind(decomp_bond, new_bond)?;
                        }
                    }
                }
            }
        }

        let subtree_idx = subtree
            .node_index(node)
            .ok_or_else(|| anyhow::anyhow!("Node {:?} not found in subtree", node))?;
        subtree.replace_tensor(subtree_idx, new_tensor)?;
    }

    Ok(())
}

/// Synchronize a separate reference state after a local update.
pub(crate) fn sync_reference_state_region<T, V>(
    reference_state: &mut TreeTN<T, V>,
    boundary_bond_map: Option<&mut HashMap<(V, V), T::Index>>,
    step: &LocalUpdateStep<V>,
    ket_state: &TreeTN<T, V>,
) -> Result<()>
where
    T: TensorLike,
    T::Index: IndexLike,
    <T::Index as IndexLike>::Id: Clone + std::hash::Hash + Eq + Ord + std::fmt::Debug + Send + Sync,
    V: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug,
{
    let ket_region = ket_state.extract_subtree(&step.nodes)?;
    let mut ket_to_ref_bond_map: HashMap<T::Index, T::Index> = HashMap::new();

    let region_nodes: std::collections::HashSet<_> = step.nodes.iter().collect();
    for node in &step.nodes {
        for neighbor in ket_state.site_index_network().neighbors(node) {
            let ket_edge = match ket_state.edge_between(node, &neighbor) {
                Some(edge) => edge,
                None => continue,
            };
            let ket_bond = match ket_state.bond_index(ket_edge) {
                Some(bond) => bond,
                None => continue,
            };

            let ref_bond = if region_nodes.contains(&neighbor) {
                ket_bond.sim()
            } else {
                let ref_edge = match reference_state.edge_between(node, &neighbor) {
                    Some(edge) => edge,
                    None => continue,
                };
                match reference_state.bond_index(ref_edge) {
                    Some(bond) => bond.clone(),
                    None => continue,
                }
            };
            ket_to_ref_bond_map.insert(ket_bond.clone(), ref_bond);
        }
    }

    if let Some(boundary_bond_map) = boundary_bond_map {
        for boundary_edge in get_boundary_edges(ket_state, &step.nodes)? {
            if let Some(edge) = reference_state.edge_between(
                &boundary_edge.node_in_region,
                &boundary_edge.neighbor_outside,
            ) {
                if let Some(ref_bond) = reference_state.bond_index(edge) {
                    boundary_bond_map.insert(
                        (
                            boundary_edge.node_in_region.clone(),
                            boundary_edge.neighbor_outside.clone(),
                        ),
                        ref_bond.clone(),
                    );
                }
            }
        }
    }

    let mut ref_region = ket_region.clone();

    let mut edges_to_update: Vec<(V, V, T::Index)> = Vec::new();
    for node in &step.nodes {
        let neighbors: Vec<V> = ref_region.site_index_network().neighbors(node).collect();
        for neighbor in neighbors {
            if let Some(edge) = ref_region.edge_between(node, &neighbor) {
                if let Some(bond) = ref_region.bond_index(edge) {
                    if let Some(new_bond) = ket_to_ref_bond_map.get(bond) {
                        edges_to_update.push((node.clone(), neighbor, new_bond.clone()));
                    }
                }
            }
        }
    }

    for (node, neighbor, new_bond) in edges_to_update {
        if let Some(edge) = ref_region.edge_between(&node, &neighbor) {
            ref_region.replace_edge_bond(edge, new_bond)?;
        }
    }

    for node in &step.nodes {
        if let Some(node_idx) = ref_region.node_index(node) {
            if let Some(tensor) = ref_region.tensor(node_idx) {
                let mut new_tensor = tensor.clone();
                let tensor_indices = tensor.external_indices();

                for ket_idx in &tensor_indices {
                    if let Some(ref_bond) = ket_to_ref_bond_map.get(ket_idx) {
                        new_tensor = new_tensor.replaceind(ket_idx, ref_bond)?;
                    }
                }

                ref_region.replace_tensor(node_idx, new_tensor)?;
            }
        }
    }

    reference_state.replace_subtree(&step.nodes, &ref_region)?;

    Ok(())
}
