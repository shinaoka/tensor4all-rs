//! Identity tensor construction for operator composition.
//!
//! When composing exclusive operators, gap positions (nodes not covered by any operator)
//! need identity tensors that pass information through unchanged.
//!
//! This module provides convenience wrappers around `TensorConstructionLike::delta()`.

use anyhow::Result;

use tensor4all_core::{DynIndex, TensorConstructionLike, TensorDynLen};

/// Build an identity operator tensor for a gap node.
///
/// For a node with site indices `{s1, s2, ...}` and bond indices `{l1, l2, ...}`,
/// this creates an identity tensor where:
/// - Each site index `s` gets a primed version `s'` (output index)
/// - The tensor is diagonal: `T[s1, s1', s2, s2', ...] = δ_{s1,s1'} × δ_{s2,s2'} × ...`
///
/// This is a convenience wrapper around `TensorConstructionLike::delta()`.
///
/// # Arguments
///
/// * `site_indices` - The site indices at this node
/// * `output_site_indices` - The output (primed) site indices, must have same dimensions
///
/// # Returns
///
/// A tensor representing the identity operator on the given site space.
///
/// # Errors
///
/// Returns an error when the operator construction fails (an overflow or
/// /// invalid-configuration failure, or a shape mismatch).
///
/// # Example
///
/// For a single site index of dimension 2:
/// ```text
/// T[s, s'] = δ_{s,s'} = [[1, 0], [0, 1]]
/// ```
pub fn build_identity_operator_tensor(
    site_indices: &[DynIndex],
    output_site_indices: &[DynIndex],
) -> Result<TensorDynLen> {
    <TensorDynLen as TensorConstructionLike>::delta(site_indices, output_site_indices)
        .map_err(anyhow::Error::new)
}

#[cfg(test)]
mod tests;
