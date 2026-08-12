//! Direct sum operations for tensors.
//!
//! This module provides functionality to compute the direct sum of tensors
//! along specified index pairs. The direct sum concatenates tensor data along
//! the paired indices, creating new indices with combined dimensions.
//!
//! This module works with concrete types (`DynIndex`, `IdxTensor`) only.

use crate::defaults::DynIndex;
use crate::defaults::IdxTensorError;
use crate::index_like::IndexLike;
use crate::tensor::IdxTensor;
use anyhow::Result;
use num_traits::Zero;
use tensor4all_tensorbackend::TensorElement;

/// Compute the direct sum of two tensors along specified index pairs.
///
/// For tensors A and B with indices to be summed specified as pairs,
/// creates a new tensor C where each paired index has dimension = dim_A + dim_B.
/// Non-paired indices must match exactly between A and B.
///
/// # Arguments
///
/// * `a` - First tensor
/// * `b` - Second tensor
/// * `pairs` - Pairs of (a_index, b_index) to be summed. Each pair creates
///
///   a new index in the result with dimension = dim(a_index) + dim(b_index).
///
/// # Returns
///
/// A tuple of:
/// - The direct sum tensor
/// - The new indices created for the summed dimensions (one per pair)
///
/// # Errors
///
/// Returns an error when the operation fails (a shape or index mismatch, or
/// /// a backend failure).
///
/// # Example
///
/// ```
/// use tensor4all_core::{direct_sum, DynIndex, IdxTensor};
///
/// # fn main() -> anyhow::Result<()> {
/// let j = DynIndex::new_dyn(2);
/// let k = DynIndex::new_dyn(3);
///
/// let a = IdxTensor::from_dense(vec![j.clone()], vec![1.0, 2.0])?;
/// let b = IdxTensor::from_dense(vec![k.clone()], vec![3.0, 4.0, 5.0])?;
/// let (result, new_indices) = direct_sum(&a, &b, &[(j.clone(), k.clone())])?;
///
/// assert_eq!(new_indices.len(), 1);
/// assert_eq!(result.to_vec::<f64>()?, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
/// # Ok(())
/// # }
/// ```
pub fn direct_sum(
    a: &IdxTensor,
    b: &IdxTensor,
    pairs: &[(DynIndex, DynIndex)],
) -> std::result::Result<(IdxTensor, Vec<DynIndex>), IdxTensorError> {
    if a.is_f64() && b.is_f64() {
        direct_sum_typed::<f64>(a, b, pairs).map_err(IdxTensorError::from)
    } else if a.is_complex() && b.is_complex() {
        direct_sum_typed::<num_complex::Complex64>(a, b, pairs).map_err(IdxTensorError::from)
    } else {
        Err(IdxTensorError::Operation {
            operation: "direct_sum",
            source: std::sync::Arc::new(std::io::Error::other(
                "direct_sum requires both tensors to have the same dense scalar type (f64 or Complex64)",
            )),
        })
    }
}

/// Setup data for direct sum computation
#[allow(dead_code)]
struct DirectSumSetup {
    common_a_positions: Vec<usize>,
    common_b_positions: Vec<usize>,
    paired_a_positions: Vec<usize>,
    paired_b_positions: Vec<usize>,
    paired_dims_a: Vec<usize>,
    paired_dims_b: Vec<usize>,
    result_indices: Vec<DynIndex>,
    result_dims: Vec<usize>,
    result_strides: Vec<usize>,
    result_total: usize,
    a_strides: Vec<usize>,
    b_strides: Vec<usize>,
    new_indices: Vec<DynIndex>,
    n_common: usize,
}

fn setup_direct_sum(
    a: &IdxTensor,
    b: &IdxTensor,
    pairs: &[(DynIndex, DynIndex)],
) -> Result<DirectSumSetup> {
    use std::collections::HashMap;

    if pairs.is_empty() {
        return Err(anyhow::anyhow!(
            "direct_sum requires at least one index pair"
        ));
    }

    // Build maps from full index metadata to positions. Same-id indices that
    // differ by prime level or tags are distinct axes.
    let a_idx_map: HashMap<&DynIndex, usize> = a
        .indices
        .iter()
        .enumerate()
        .map(|(i, idx)| (idx, i))
        .collect();
    let b_idx_map: HashMap<&DynIndex, usize> = b
        .indices
        .iter()
        .enumerate()
        .map(|(i, idx)| (idx, i))
        .collect();

    // Build set of paired indices.
    let mut a_paired_indices: std::collections::HashSet<&DynIndex> =
        std::collections::HashSet::new();
    let mut b_paired_indices: std::collections::HashSet<&DynIndex> =
        std::collections::HashSet::new();

    for (a_idx, b_idx) in pairs {
        if !a_idx_map.contains_key(a_idx) {
            return Err(anyhow::anyhow!(
                "Index not found in first tensor (direct_sum)"
            ));
        }
        if !b_idx_map.contains_key(b_idx) {
            return Err(anyhow::anyhow!(
                "Index not found in second tensor (direct_sum)"
            ));
        }
        a_paired_indices.insert(a_idx);
        b_paired_indices.insert(b_idx);
    }

    // Identify common (non-paired) indices
    // Iterate over a.indices (Vec) to preserve deterministic order,
    // using b_idx_map only for lookups.
    let mut common_a_positions: Vec<usize> = Vec::new();
    let mut common_b_positions: Vec<usize> = Vec::new();
    for (a_pos, a_idx) in a.indices.iter().enumerate() {
        if a_paired_indices.contains(a_idx) {
            continue;
        }
        if let Some(&b_pos) = b_idx_map.get(a_idx) {
            if b_paired_indices.contains(a_idx) {
                continue;
            }
            let a_dims = a.dims();
            let b_dims = b.dims();
            if a_dims[a_pos] != b_dims[b_pos] {
                return Err(anyhow::anyhow!(
                    "Dimension mismatch for common index: {} vs {}",
                    a_dims[a_pos],
                    b_dims[b_pos]
                ));
            }
            common_a_positions.push(a_pos);
            common_b_positions.push(b_pos);
        }
    }

    // Build paired positions and dimensions
    let paired_a_positions: Vec<usize> = pairs.iter().map(|(a_idx, _)| a_idx_map[a_idx]).collect();
    let paired_b_positions: Vec<usize> = pairs.iter().map(|(_, b_idx)| b_idx_map[b_idx]).collect();
    let a_dims = a.dims();
    let b_dims = b.dims();
    let paired_dims_a: Vec<usize> = paired_a_positions.iter().map(|&p| a_dims[p]).collect();
    let paired_dims_b: Vec<usize> = paired_b_positions.iter().map(|&p| b_dims[p]).collect();

    // Create new indices for paired dimensions
    let mut new_indices: Vec<DynIndex> = Vec::new();
    for (&dim_a, &dim_b) in paired_dims_a.iter().zip(&paired_dims_b) {
        let new_dim = dim_a + dim_b;
        let new_index = DynIndex::new_link(new_dim)
            .map_err(|e| anyhow::anyhow!("Failed to create index: {:?}", e))?;
        new_indices.push(new_index);
    }

    // Build result indices and dimensions
    let mut result_indices: Vec<DynIndex> = Vec::new();
    let mut result_dims: Vec<usize> = Vec::new();

    // Common indices first.
    let a_dims = a.dims();
    for &a_pos in &common_a_positions {
        result_indices.push(a.indices[a_pos].clone());
        result_dims.push(a_dims[a_pos]);
    }

    // New indices from pairs preserve the full index identity semantics (id + plev + tags).
    for new_idx in &new_indices {
        result_indices.push(new_idx.clone());
        result_dims.push(new_idx.dim());
    }

    // Compute column-major strides (leftmost index fastest).
    let result_total: usize = result_dims.iter().product();
    let mut result_strides: Vec<usize> = vec![1; result_dims.len()];
    for i in 1..result_dims.len() {
        result_strides[i] = result_strides[i - 1] * result_dims[i - 1];
    }

    let a_dims = a.dims();
    let b_dims = b.dims();
    let mut a_strides: Vec<usize> = vec![1; a_dims.len()];
    for i in 1..a_dims.len() {
        a_strides[i] = a_strides[i - 1] * a_dims[i - 1];
    }

    let mut b_strides: Vec<usize> = vec![1; b_dims.len()];
    for i in 1..b_dims.len() {
        b_strides[i] = b_strides[i - 1] * b_dims[i - 1];
    }

    let n_common = common_a_positions.len();

    Ok(DirectSumSetup {
        common_a_positions,
        common_b_positions,
        paired_a_positions,
        paired_b_positions,
        paired_dims_a,
        paired_dims_b,
        result_indices,
        result_dims,
        result_strides,
        result_total,
        a_strides,
        b_strides,
        new_indices,
        n_common,
    })
}

fn linear_to_multi(linear: usize, dims: &[usize]) -> Vec<usize> {
    let mut multi = vec![0; dims.len()];
    let mut remaining = linear;
    for i in 0..dims.len() {
        multi[i] = remaining % dims[i];
        remaining /= dims[i];
    }
    multi
}

fn multi_to_linear(multi: &[usize], strides: &[usize]) -> usize {
    multi.iter().zip(strides).map(|(&m, &s)| m * s).sum()
}

fn direct_sum_typed<T: TensorElement + Zero>(
    a: &IdxTensor,
    b: &IdxTensor,
    pairs: &[(DynIndex, DynIndex)],
) -> Result<(IdxTensor, Vec<DynIndex>)> {
    let setup = setup_direct_sum(a, b, pairs)?;
    let a_data = a.to_vec::<T>()?;
    let b_data = b.to_vec::<T>()?;

    let mut result_data: Vec<T> = vec![T::zero(); setup.result_total];

    #[allow(clippy::needless_range_loop)]
    for result_linear in 0..setup.result_total {
        let result_multi = linear_to_multi(result_linear, &setup.result_dims);
        let common_multi: Vec<usize> = result_multi[..setup.n_common].to_vec();
        let paired_multi: Vec<usize> = result_multi[setup.n_common..].to_vec();

        let all_from_a = paired_multi
            .iter()
            .enumerate()
            .all(|(i, &pm)| pm < setup.paired_dims_a[i]);
        let all_from_b = paired_multi
            .iter()
            .enumerate()
            .all(|(i, &pm)| pm >= setup.paired_dims_a[i]);

        if all_from_a {
            let a_dims = a.dims();
            let mut a_multi = vec![0usize; a_dims.len()];
            for (i, &cp) in setup.common_a_positions.iter().enumerate() {
                a_multi[cp] = common_multi[i];
            }
            for (i, &pp) in setup.paired_a_positions.iter().enumerate() {
                a_multi[pp] = paired_multi[i];
            }
            let a_linear = multi_to_linear(&a_multi, &setup.a_strides);
            result_data[result_linear] = a_data[a_linear];
        } else if all_from_b {
            let b_dims = b.dims();
            let mut b_multi = vec![0usize; b_dims.len()];
            for (i, &cp) in setup.common_b_positions.iter().enumerate() {
                b_multi[cp] = common_multi[i];
            }
            for (i, &pp) in setup.paired_b_positions.iter().enumerate() {
                b_multi[pp] = paired_multi[i] - setup.paired_dims_a[i];
            }
            let b_linear = multi_to_linear(&b_multi, &setup.b_strides);
            result_data[result_linear] = b_data[b_linear];
        }
        // else: mixed case stays T::zero()
    }

    let result = IdxTensor::from_dense(setup.result_indices, result_data)?;
    Ok((result, setup.new_indices))
}

#[cfg(test)]
mod tests;
