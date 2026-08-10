//! Composition of exclusive (non-overlapping) operators.
//!
//! This module provides functions to compose multiple operators that act on
//! non-overlapping regions into a single operator on the full target space.

use std::collections::{HashMap, HashSet};
use std::fmt::Debug;
use std::hash::Hash;

use anyhow::{Context, Result};
use petgraph::stable_graph::NodeIndex;

use tensor4all_core::{IndexLike, TensorLike};

use super::index_mapping::IndexMapping;
use super::linear_operator::LinearOperator;
use super::Operator;
use crate::site_index_network::SiteIndexNetwork;
use crate::treetn::TreeTN;

/// Check if a set of operators are exclusive (non-overlapping) on the target network.
/// Operators are exclusive if:
/// 1. **Vertex-disjoint**: No two operators share a node
/// 2. **Connected subtrees**: Each operator's nodes form a connected subtree
/// 3. **Path-exclusive**: Paths between different operators don't cross other operators
/// # Arguments
/// * `target` - The target site index network (full space)
/// * `operators` - The operators to check
/// # Returns
/// `true` if operators are exclusive, `false` otherwise.
pub fn are_exclusive_operators<T, V, O>(
    target: &SiteIndexNetwork<V, T::Index>,
    operators: &[&O],
) -> bool
where
    T: TensorLike,
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug,
    O: Operator<T, V>,
{
    // Collect node sets for each operator
    let node_sets: Vec<HashSet<V>> = operators.iter().map(|op| op.node_names()).collect();

    // 1. Check vertex-disjoint
    for i in 0..node_sets.len() {
        for j in (i + 1)..node_sets.len() {
            if !node_sets[i].is_disjoint(&node_sets[j]) {
                return false;
            }
        }
    }

    // 2. Check each operator's nodes form a connected subtree in target
    for node_set in &node_sets {
        if node_set.is_empty() {
            continue;
        }

        // Convert to NodeIndex set
        let node_indices: HashSet<NodeIndex> = node_set
            .iter()
            .filter_map(|name| target.node_index(name))
            .collect();

        if node_indices.len() != node_set.len() {
            // Some nodes don't exist in target
            return false;
        }

        if !target.is_connected_subset(&node_indices) {
            return false;
        }
    }

    // 3. Path-exclusive check: paths between operators should not cross other operators
    for i in 0..node_sets.len() {
        for j in (i + 1)..node_sets.len() {
            if !check_path_exclusive::<T, V>(target, &node_sets[i], &node_sets[j], &node_sets) {
                return false;
            }
        }
    }

    true
}

/// Check if paths between two operator regions don't cross other operators.
fn check_path_exclusive<T, V>(
    target: &SiteIndexNetwork<V, T::Index>,
    set_a: &HashSet<V>,
    set_b: &HashSet<V>,
    all_sets: &[HashSet<V>],
) -> bool
where
    T: TensorLike,
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug,
{
    // Find a node from each set
    let node_a = match set_a.iter().next() {
        Some(n) => n,
        None => return true, // Empty set
    };
    let node_b = match set_b.iter().next() {
        Some(n) => n,
        None => return true,
    };

    // Get path between them
    let idx_a = match target.node_index(node_a) {
        Some(idx) => idx,
        None => return false,
    };
    let idx_b = match target.node_index(node_b) {
        Some(idx) => idx,
        None => return false,
    };

    let path = match target.path_between(idx_a, idx_b) {
        Some(p) => p,
        None => return false, // No path means disconnected, which is fine for exclusivity
    };

    // Check that path nodes (excluding endpoints) don't belong to other operators
    let other_operator_nodes: HashSet<&V> = all_sets
        .iter()
        .filter(|s| *s != set_a && *s != set_b)
        .flat_map(|s| s.iter())
        .collect();

    for node_idx in &path[1..path.len().saturating_sub(1)] {
        if let Some(name) = target.node_name(*node_idx) {
            if other_operator_nodes.contains(name) {
                return false;
            }
        }
    }

    true
}

/// Compose exclusive LinearOperators into a single LinearOperator.
/// This function takes multiple non-overlapping operators and combines them into
/// a single operator that acts on the full target space. Gap positions (nodes not
/// covered by any operator) are filled with identity operators using `T::delta()`.
/// # Arguments
/// * `target` - The full site index network (defines the output structure)
/// * `operators` - Non-overlapping LinearOperators to compose
/// * `gap_site_indices` - Site indices for gap nodes: node_name -> [(input_index, output_index), ...]
/// # Returns
/// A LinearOperator representing the composed operator on the full target space.
/// # Errors
/// Returns an error when the operators overlap (an invalid-configuration
/// failure), an operator node is missing (a missing-index failure), or the
/// gap node site indices are not provided (a missing-index failure).
#[allow(clippy::type_complexity)]
/// # Errors
///
/// Returns an error when the operators overlap (an invalid-configuration
///     /// failure), a node is missing (a missing-index failure), or the
///     /// composition fails (a backend failure).
///
pub fn compose_exclusive_linear_operators<T, V>(
    target: &SiteIndexNetwork<V, T::Index>,
    operators: &[&LinearOperator<T, V>],
    gap_site_indices: &HashMap<V, Vec<(T::Index, T::Index)>>,
) -> Result<LinearOperator<T, V>>
where
    T: TensorLike,
    T::Index: IndexLike + Clone + Hash + Eq + Debug,
    <T::Index as IndexLike>::Id: Clone + Hash + Eq + Ord + Debug + Send + Sync,
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug,
{
    compose_exclusive_linear_operators_inner(target, operators, gap_site_indices, true)
}

#[allow(clippy::type_complexity)]
#[allow(dead_code)]
pub(crate) fn compose_exclusive_linear_operators_unchecked<T, V>(
    target: &SiteIndexNetwork<V, T::Index>,
    operators: &[&LinearOperator<T, V>],
    gap_site_indices: &HashMap<V, Vec<(T::Index, T::Index)>>,
) -> Result<LinearOperator<T, V>>
where
    T: TensorLike,
    T::Index: IndexLike + Clone + Hash + Eq + Debug,
    <T::Index as IndexLike>::Id: Clone + Hash + Eq + Ord + Debug + Send + Sync,
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug,
{
    compose_exclusive_linear_operators_inner(target, operators, gap_site_indices, false)
}

#[allow(clippy::type_complexity)]
fn compose_exclusive_linear_operators_inner<T, V>(
    target: &SiteIndexNetwork<V, T::Index>,
    operators: &[&LinearOperator<T, V>],
    gap_site_indices: &HashMap<V, Vec<(T::Index, T::Index)>>,
    validate_exclusivity: bool,
) -> Result<LinearOperator<T, V>>
where
    T: TensorLike,
    T::Index: IndexLike + Clone + Hash + Eq + Debug,
    <T::Index as IndexLike>::Id: Clone + Hash + Eq + Ord + Debug + Send + Sync,
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug,
{
    // 1. Validate exclusivity
    if validate_exclusivity && !are_exclusive_operators::<T, V, _>(target, operators) {
        return Err(anyhow::anyhow!(
            "Operators are not exclusive: they may overlap or not form connected subtrees"
        ))
        .context("compose_exclusive_linear_operators: operators must be exclusive");
    }

    // 2. Collect covered nodes and build node-to-operator map
    let covered: HashSet<V> = operators.iter().flat_map(|op| op.node_names()).collect();

    let mut node_to_operator: HashMap<V, usize> = HashMap::new();
    for (op_idx, op) in operators.iter().enumerate() {
        for name in op.node_names() {
            node_to_operator.insert(name, op_idx);
        }
    }

    // 3. Identify gap nodes
    let all_target_nodes: HashSet<V> = target.node_names().into_iter().cloned().collect();
    let gaps: Vec<V> = all_target_nodes.difference(&covered).cloned().collect();
    let gap_set: HashSet<V> = gaps.iter().cloned().collect();

    // 4. Identify cross-component edges and create dummy link pairs
    // An edge is cross-component if endpoints are in different operators or one is a gap
    let mut dummy_links_for_node: HashMap<V, Vec<T::Index>> = HashMap::new();

    for (node_a, node_b) in target.edges() {
        let comp_a = node_to_operator.get(&node_a);
        let comp_b = node_to_operator.get(&node_b);
        let is_gap_a = gap_set.contains(&node_a);
        let is_gap_b = gap_set.contains(&node_b);

        let is_cross = match (comp_a, comp_b, is_gap_a, is_gap_b) {
            (Some(a), Some(b), false, false) => a != b,
            _ => true,
        };

        if is_cross {
            let (link_a, link_b) = T::Index::create_dummy_link_pair();
            dummy_links_for_node
                .entry(node_a.clone())
                .or_default()
                .push(link_a);
            dummy_links_for_node
                .entry(node_b.clone())
                .or_default()
                .push(link_b);
        }
    }

    // 5. Build tensors and mappings
    let mut tensors: Vec<T> = Vec::new();
    let mut result_node_names: Vec<V> = Vec::new();
    let mut combined_input_mapping: HashMap<V, Vec<IndexMapping<T::Index>>> = HashMap::new();
    let mut combined_output_mapping: HashMap<V, Vec<IndexMapping<T::Index>>> = HashMap::new();

    // 5a. Add tensors from operators (with dummy links added via outer product)
    for op in operators {
        for name in op.node_names() {
            let node_idx = op
                .mpo()
                .node_index(&name)
                .ok_or_else(|| anyhow::anyhow!("Node {:?} not found in operator", name))?;
            let mut tensor = op
                .mpo()
                .tensor(node_idx)
                .ok_or_else(|| anyhow::anyhow!("Tensor not found for node {:?}", name))?
                .clone();

            // Add dummy links via outer product with ones tensor
            if let Some(links) = dummy_links_for_node.get(&name) {
                for link in links {
                    let ones = T::ones(std::slice::from_ref(link)).with_context(|| {
                        format!("Failed to create ones tensor for dummy link at {:?}", name)
                    })?;
                    tensor = tensor.outer_product(&ones).with_context(|| {
                        format!("Failed to add dummy link to tensor at {:?}", name)
                    })?;
                }
            }

            tensors.push(tensor);
            result_node_names.push(name.clone());

            // Copy mappings
            if let Some(input_maps) = op.get_input_mappings(&name) {
                combined_input_mapping.insert(name.clone(), input_maps.to_vec());
            }
            if let Some(output_maps) = op.get_output_mappings(&name) {
                combined_output_mapping.insert(name.clone(), output_maps.to_vec());
            }
        }
    }

    // 5b. Add identity tensors at gaps (with dummy links added via outer product)
    for gap_name in gaps {
        let index_pairs = gap_site_indices.get(&gap_name).ok_or_else(|| {
            anyhow::anyhow!("Site indices not provided for gap node {:?}", gap_name)
        })?;

        // Collect site indices for delta
        let input_indices: Vec<T::Index> = index_pairs.iter().map(|(i, _)| i.clone()).collect();
        let output_indices: Vec<T::Index> = index_pairs.iter().map(|(_, o)| o.clone()).collect();

        if !index_pairs.is_empty() {
            combined_input_mapping
                .entry(gap_name.clone())
                .or_insert_with(Vec::new)
                .extend(input_indices.iter().map(|index| IndexMapping {
                    true_index: index.clone(),
                    internal_index: index.clone(),
                }));
            combined_output_mapping
                .entry(gap_name.clone())
                .or_insert_with(Vec::new)
                .extend(output_indices.iter().map(|index| IndexMapping {
                    true_index: index.clone(),
                    internal_index: index.clone(),
                }));
        }

        // Create delta tensor for site indices
        let mut identity_tensor = if input_indices.is_empty() {
            T::delta(&[], &[]).context("Failed to create scalar identity tensor")?
        } else {
            T::delta(&input_indices, &output_indices).with_context(|| {
                format!("Failed to build identity tensor for gap {:?}", gap_name)
            })?
        };

        // Add dummy links via outer product (bond indices, not site indices)
        if let Some(links) = dummy_links_for_node.get(&gap_name) {
            for link in links {
                let ones = T::ones(std::slice::from_ref(link)).with_context(|| {
                    format!(
                        "Failed to create ones tensor for dummy link at gap {:?}",
                        gap_name
                    )
                })?;
                identity_tensor = identity_tensor.outer_product(&ones).with_context(|| {
                    format!("Failed to add dummy link to gap tensor {:?}", gap_name)
                })?;
            }
        }

        tensors.push(identity_tensor);
        result_node_names.push(gap_name);
    }

    // 6. Create TreeTN from tensors
    let mpo = TreeTN::from_tensors(tensors, result_node_names)
        .context("compose_exclusive_linear_operators: failed to create TreeTN")?;

    Ok(LinearOperator::new_multi(
        mpo,
        combined_input_mapping,
        combined_output_mapping,
    ))
}

#[cfg(test)]
mod tests;
