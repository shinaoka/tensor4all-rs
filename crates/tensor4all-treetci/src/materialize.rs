use crate::error::{Result as TreeTciResult, TreeTciError};
use crate::{
    assemble::{assemble_points_column_major, MultiIndex},
    assemble_global_point, column_2d, ncols_2d, GlobalIndexBatch, SubtreeKey, TreeTCI2,
    TreeTciEdge,
};
use anyhow::Result;

use std::collections::HashMap;
use tensor4all_core::{ColMajorArray, DynIndex, TensorDynLen};
use tensor4all_tcicore::MatrixLuciScalar as Scalar;
use tensor4all_tensorbackend::FullPivLuScalar;
use tensor4all_treetn::TreeTN;

/// Materialize a converged TreeTCI state as a `TreeTN`.
///
/// Converts the pivot sets stored in a [`TreeTCI2`] into site tensors
/// of a [`TreeTN`]. The `evaluate` closure is called to fill tensor
/// entries at the selected pivot points.
///
/// `center_site` selects the BFS root for the tree decomposition
/// (default: site 0).
///
/// This function is called internally by [`crossinterpolate2`](crate::crossinterpolate2).
/// # Errors
///
/// Returns an error when the operation fails (a shape or index mismatch, or
/// /// a backend failure).
///
pub fn to_treetn<T, F>(
    state: &TreeTCI2<T>,
    evaluate: F,
    center_site: Option<usize>,
) -> TreeTciResult<TreeTN<TensorDynLen, usize>>
where
    T: FullPivLuScalar + tensor4all_tcicore::MatrixLuciScalar + tensor4all_core::TensorElement,
    F: Fn(GlobalIndexBatch<'_>) -> Result<Vec<T>>,
{
    let root = center_site.unwrap_or(0);
    let (parents, distances) = state.graph.bfs_tree(root)?;

    let mut bond_indices = HashMap::new();
    for edge in state.graph.edges() {
        let (left_key, right_key) = state.graph.subregion_vertices(edge)?;
        let left_rank = state
            .ijset
            .get(&left_key)
            .map(ncols_2d)
            .transpose()?
            .unwrap_or(0);
        let right_rank = state
            .ijset
            .get(&right_key)
            .map(ncols_2d)
            .transpose()?
            .unwrap_or(0);
        if !(left_rank == right_rank) {
            return Err(anyhow::anyhow!(
                "bond ranks disagree across edge {:?}: left {}, right {}",
                edge,
                left_rank,
                right_rank
            )
            .into());
        };
        bond_indices.insert(edge, DynIndex::new_dyn(left_rank.max(1)));
    }

    let mut sites = (0..state.graph.n_sites()).collect::<Vec<_>>();
    sites.sort_by_key(|&site| (distances[site], site));

    let mut tensors = Vec::with_capacity(sites.len());
    let mut node_names = Vec::with_capacity(sites.len());
    for site in sites {
        let parent_edge = parents[site]
            .map(|parent| state.graph.edge_between(site, parent))
            .transpose()?;
        let incoming_edges = match parent_edge {
            Some(edge) => state.graph.adjacent_edges(site, &[edge]),
            None => state.graph.adjacent_edges(site, &[]),
        };
        let in_keys = state.graph.edge_in_ij_keys(site, &incoming_edges)?;
        let out_edges = parent_edge.into_iter().collect::<Vec<_>>();
        let out_keys = state.graph.edge_in_ij_keys(site, &out_edges)?;

        let data = if out_edges.is_empty() {
            fill_tensor_values(state, &in_keys, &out_keys, &[site], &evaluate)?
        } else {
            site_tensor_with_parent(state, site, out_edges[0], &in_keys, &out_keys, &evaluate)?
        };

        let mut indices = Vec::with_capacity(1 + incoming_edges.len() + out_edges.len());
        indices.push(DynIndex::new_dyn(state.local_dims[site]));
        for edge in &incoming_edges {
            indices.push(
                bond_indices
                    .get(edge)
                    .cloned()
                    .ok_or_else(|| anyhow::anyhow!("missing bond index for edge {:?}", edge))?,
            );
        }
        for edge in &out_edges {
            indices.push(
                bond_indices
                    .get(edge)
                    .cloned()
                    .ok_or_else(|| anyhow::anyhow!("missing bond index for edge {:?}", edge))?,
            );
        }

        tensors.push(TensorDynLen::from_dense(indices, data)?);
        node_names.push(site);
    }

    TreeTN::from_tensors(tensors, node_names).map_err(TreeTciError::from)
}

fn site_tensor_with_parent<T, F>(
    state: &TreeTCI2<T>,
    site: usize,
    parent_edge: TreeTciEdge,
    in_keys: &[SubtreeKey],
    out_keys: &[SubtreeKey],
    evaluate: &F,
) -> Result<Vec<T>>
where
    T: FullPivLuScalar + tensor4all_tcicore::MatrixLuciScalar + tensor4all_core::TensorElement,
    F: Fn(GlobalIndexBatch<'_>) -> Result<Vec<T>>,
{
    if !(out_keys.len() == 1) {
        return Err(anyhow::anyhow!(
            "MVP TreeTCI materialization expects exactly one outgoing key per non-root site"
        ));
    };

    let pi1_values = fill_tensor_values(state, in_keys, out_keys, &[site], evaluate)?;
    let rows = state.local_dims[site] * product_pivot_dims(state, in_keys)?;
    let cols = product_pivot_dims(state, out_keys)?;

    let site_side_key = site_side_key(state, site, parent_edge)?;
    let p_values = fill_tensor_values(
        state,
        std::slice::from_ref(&site_side_key),
        out_keys,
        &[],
        evaluate,
    )?;
    let p_rows = state
        .ijset
        .get(&site_side_key)
        .ok_or_else(|| anyhow::anyhow!("missing pivot set for subtree key {:?}", site_side_key))
        .and_then(ncols_2d)?;
    if !(p_rows == cols) {
        return Err(anyhow::anyhow!(
            "pivot matrix for site {} is not square: {} x {}",
            site,
            p_rows,
            cols
        ));
    };

    T::solve_right_full_piv_lu(&pi1_values, rows, cols, &p_values, p_rows, cols)
        .map_err(anyhow::Error::from)
}

fn site_side_key<T>(state: &TreeTCI2<T>, site: usize, edge: TreeTciEdge) -> Result<SubtreeKey> {
    let (left_key, right_key) = state.graph.subregion_vertices(edge)?;
    if left_key.as_slice().contains(&site) {
        Ok(left_key)
    } else if right_key.as_slice().contains(&site) {
        Ok(right_key)
    } else {
        Err(anyhow::anyhow!(
            "site {} does not appear in either side of edge {:?}",
            site,
            edge
        ))
    }
}

fn product_pivot_dims<T>(state: &TreeTCI2<T>, keys: &[SubtreeKey]) -> Result<usize> {
    let mut product = 1usize;
    for key in keys {
        let dim = state
            .ijset
            .get(key)
            .ok_or_else(|| anyhow::anyhow!("missing pivot set for subtree key {:?}", key))
            .and_then(ncols_2d)?;
        product = product.saturating_mul(dim.max(1));
    }
    Ok(product)
}

fn fill_tensor_values<T, F>(
    state: &TreeTCI2<T>,
    in_keys: &[SubtreeKey],
    out_keys: &[SubtreeKey],
    central_sites: &[usize],
    evaluate: &F,
) -> Result<Vec<T>>
where
    T: Scalar,
    F: Fn(GlobalIndexBatch<'_>) -> Result<Vec<T>>,
{
    let in_combos = cartesian_entries(&state.ijset, in_keys)?;
    let out_combos = cartesian_entries(&state.ijset, out_keys)?;
    let central_combos = central_assignments(&state.local_dims, central_sites);
    let mut points =
        Vec::with_capacity(in_combos.len() * out_combos.len() * central_combos.len().max(1));

    for out_combo in &out_combos {
        for in_combo in &in_combos {
            for central in &central_combos {
                let mut assignments = Vec::with_capacity(in_keys.len() + out_keys.len());
                assignments.extend(in_keys.iter().zip(in_combo.iter()));
                assignments.extend(out_keys.iter().zip(out_combo.iter()));
                points.push(assemble_global_point(
                    state.local_dims.len(),
                    &assignments,
                    central,
                )?);
            }
        }
    }

    let batch = assemble_points_column_major(&points)?;
    let values = evaluate(batch.as_view())?;
    if !(values.len() == points.len()) {
        return Err(anyhow::anyhow!(
            "batch evaluator returned {} values for {} fill-tensor points",
            values.len(),
            points.len()
        ));
    };
    Ok(values)
}

/// Extract columns from ColMajorArray ijset entries and produce cartesian products.
///
/// Returns Vec<Vec<MultiIndex>> where each inner Vec has one MultiIndex per key.
fn cartesian_entries(
    ijset: &HashMap<SubtreeKey, ColMajorArray<usize>>,
    keys: &[SubtreeKey],
) -> Result<Vec<Vec<MultiIndex>>> {
    if keys.is_empty() {
        return Ok(vec![Vec::new()]);
    }

    // Convert each ColMajorArray to Vec<MultiIndex> (columns as Vecs)
    let entry_sets = keys
        .iter()
        .map(|key| {
            let arr = ijset
                .get(key)
                .ok_or_else(|| anyhow::anyhow!("missing pivot set for subtree key {:?}", key))?;
            let columns: Vec<MultiIndex> = (0..ncols_2d(arr)?)
                .map(|j| column_2d(arr, j).map(|column| column.to_vec()))
                .collect::<Result<_>>()?;
            Ok(columns)
        })
        .collect::<Result<Vec<_>>>()?;

    let mut current = vec![Vec::new(); keys.len()];
    let mut combos = Vec::new();
    cartesian_entries_recursive(&entry_sets, keys.len(), &mut current, &mut combos);
    Ok(combos)
}

fn cartesian_entries_recursive(
    entry_sets: &[Vec<MultiIndex>],
    remaining: usize,
    current: &mut [MultiIndex],
    out: &mut Vec<Vec<MultiIndex>>,
) {
    if remaining == 0 {
        out.push(current.to_vec());
        return;
    }

    let level = remaining - 1;
    for entry in &entry_sets[level] {
        current[level] = entry.clone();
        cartesian_entries_recursive(entry_sets, level, current, out);
    }
}

fn central_assignments(local_dims: &[usize], central_sites: &[usize]) -> Vec<Vec<(usize, usize)>> {
    let mut combos = vec![Vec::new()];
    for &site in central_sites {
        let mut next = Vec::new();
        for combo in &combos {
            for value in 0..local_dims[site] {
                let mut extended = combo.clone();
                extended.push((site, value));
                next.push(extended);
            }
        }
        combos = next;
    }
    if central_sites.is_empty() {
        vec![Vec::new()]
    } else {
        combos
    }
}

#[cfg(test)]
mod tests;
