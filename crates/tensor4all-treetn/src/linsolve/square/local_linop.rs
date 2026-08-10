//! LocalLinOp: Linear operator wrapper for local GMRES solving (square case).
//!
//! This module provides a wrapper that applies the local projected operator
//! to tensors, enabling GMRES solving via `tensor4all_core::krylov::gmres`.

use std::hash::Hash;
use std::sync::{Arc, RwLock};

use anyhow::{Context, Result};
use tensor4all_core::{IndexLike, TensorLike};

use crate::linsolve::common::{NetworkTopology, ProjectedOperator};
use crate::treetn::TreeTN;

/// LocalLinOp: Wraps the projected operator for local GMRES solving.
///
/// This applies only the projected local operator `H * x`.
/// Affine coefficients such as `a₀ I + a₁ H` belong to the Krylov solver
/// so the Arnoldi basis is built from the same unshifted operator as KrylovKit.
///
/// This is the V_in = V_out specialized version that maintains a separate
/// reference state for stable environment computation.
pub struct LocalLinOp<'a, T, V>
where
    T: TensorLike + 'static,
    <T::Index as IndexLike>::Id:
        Clone + std::hash::Hash + Eq + Ord + std::fmt::Debug + Send + Sync + 'static,
    V: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug + 'static,
{
    /// The projected operator (shared, mutable for environment caching)
    pub projected_operator: Arc<RwLock<ProjectedOperator<T, V>>>,
    /// The region being updated
    pub region: Vec<V>,
    /// Current state for ket in environment computation
    pub state: &'a TreeTN<T, V>,
    /// Reference state for bra in environment computation
    /// Uses separate bond indices to prevent unintended contractions
    pub reference_state: &'a TreeTN<T, V>,
    /// Expected local vector-space indices for inputs to this projected operator.
    expected_input_indices: Vec<T::Index>,
}

impl<'a, T, V> LocalLinOp<'a, T, V>
where
    T: TensorLike + 'static,
    T::Index: IndexLike,
    <T::Index as IndexLike>::Id:
        Clone + std::hash::Hash + Eq + Ord + std::fmt::Debug + Send + Sync + 'static,
    V: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug + 'static,
{
    /// Create a new LocalLinOp with explicit reference state.
    ///
    /// The reference_state should have separate bond indices from state
    /// to prevent unintended bra↔ket contractions in environment computation.
    pub(crate) fn with_expected_input_indices(
        projected_operator: Arc<RwLock<ProjectedOperator<T, V>>>,
        region: Vec<V>,
        state: &'a TreeTN<T, V>,
        reference_state: &'a TreeTN<T, V>,
        expected_input_indices: Vec<T::Index>,
    ) -> Self {
        Self {
            projected_operator,
            region,
            state,
            reference_state,
            expected_input_indices,
        }
    }

    fn same_index_set(left: &[T::Index], right: &[T::Index]) -> bool {
        left.len() == right.len()
            && left.iter().all(|left_idx| {
                right
                    .iter()
                    .any(|right_idx| left_idx == right_idx && left_idx.dim() == right_idx.dim())
            })
    }

    pub(crate) fn apply_projected_with_operator<NT: NetworkTopology<V>>(
        &self,
        x: &T,
        proj_op: &mut ProjectedOperator<T, V>,
        topology: &NT,
    ) -> Result<T> {
        let x_indices = x.external_indices();
        if !Self::same_index_set(&x_indices, &self.expected_input_indices) {
            return Err(anyhow::anyhow!(
                "LocalLinOp::apply_projected: index structure mismatch between input (x) and the local state vector space:\n  x has {} indices: {:?}\n  expected {} indices: {:?}",
                x_indices.len(),
                x_indices.iter().map(|i| format!("{:?}:{}", i.id(), i.dim())).collect::<Vec<_>>(),
                self.expected_input_indices.len(),
                self.expected_input_indices.iter().map(|i| format!("{:?}:{}", i.id(), i.dim())).collect::<Vec<_>>(),
            ));
        }

        let hx = proj_op.apply(x, &self.region, self.state, self.reference_state, topology)?;

        let hx_indices = hx.external_indices();
        let indices_match = x_indices.len() == hx_indices.len()
            && x_indices
                .iter()
                .zip(hx_indices.iter())
                .all(|(x_idx, hx_idx)| x_idx == hx_idx && x_idx.dim() == hx_idx.dim());
        if !indices_match {
            return Err(anyhow::anyhow!(
                "LocalLinOp::apply_projected: index structure mismatch between operator output (hx) and input (x):\n  x has {} indices: {:?}\n  hx has {} indices: {:?}\n\nProjectedOperator::apply is expected to return output in the same local vector space and index order as its input.",
                x_indices.len(),
                x_indices.iter().map(|i| format!("{:?}:{}", i.id(), i.dim())).collect::<Vec<_>>(),
                hx_indices.len(),
                hx_indices.iter().map(|i| format!("{:?}:{}", i.id(), i.dim())).collect::<Vec<_>>(),
            ));
        }

        Ok(hx)
    }

    /// Apply the projected local operator: `y = H * x`.
    /// # Errors
    ///
    /// Returns an error when the contraction or operation fails (a shape or
    /// /// index mismatch, or a backend failure).
    ///
    pub fn apply_projected(&self, x: &T) -> Result<T> {
        let mut proj_op = self
            .projected_operator
            .write()
            .map_err(|e| {
                anyhow::anyhow!("Failed to acquire write lock on projected_operator: {}", e)
            })
            .context("LocalLinOp::apply: lock poisoned")?;

        self.apply_projected_with_operator(x, &mut proj_op, self.state.site_index_network())
    }
}

#[cfg(test)]
mod tests;
