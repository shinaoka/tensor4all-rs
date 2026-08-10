//! Truncation methods for TreeTN.
//!
//! This module provides methods for truncating tree tensor networks.
//!
//! The truncation algorithm uses a two-site sweep (nsite=2) based on Euler tour traversal:
//! 1. First, canonicalize the network towards the specified center
//! 2. Generate a sweep plan that visits each edge twice (forward and backward)
//! 3. For each step: extract two adjacent nodes, perform SVD-based truncation, replace
//! 4. The canonical center moves along the sweep path
//!
//! This ensures all bonds are optimally truncated in both directions.

use std::collections::HashSet;
use std::hash::Hash;

use anyhow::{Context, Result};

use tensor4all_core::SvdTruncationPolicy;
use tensor4all_core::{IndexLike, TensorLike};

use super::localupdate::{apply_local_update_sweep, LocalUpdateSweepPlan, TruncateUpdater};
use super::TreeTN;
use crate::options::{CanonicalizationOptions, TruncationOptions};
use crate::CanonicalForm;

impl<T, V> TreeTN<T, V>
where
    T: TensorLike,
    V: Clone + Hash + Eq + Send + Sync + std::fmt::Debug,
{
    /// Truncate the network towards the specified center using options.
    ///
    /// This is the recommended unified API for truncation. It accepts:
    /// - Center nodes specified by their node names (V)
    /// - [`TruncationOptions`] to control the SVD policy and max_rank
    ///
    /// # Algorithm
    /// 1. Canonicalize the network towards the center (required for truncation)
    /// 2. Generate a two-site sweep plan using Euler tour traversal
    /// 3. Apply SVD-based truncation at each step, visiting each edge twice
    ///
    /// # Errors
    ///
    /// Returns an error when the operation fails (a shape or index mismatch, an
    /// /// SVD or non-convergence failure, or a backend failure).
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_treetn::{TreeTN, TruncationOptions};
    /// use tensor4all_core::{DynIndex, TensorDynLen, TensorLike};
    ///
    /// // Build a 2-node chain
    /// let s0 = DynIndex::new_dyn(2);
    /// let bond = DynIndex::new_dyn(3);
    /// let s1 = DynIndex::new_dyn(2);
    ///
    /// let t0 = TensorDynLen::from_dense(
    ///     vec![s0.clone(), bond.clone()],
    ///     vec![1.0_f64, 0.0, 0.0, 1.0, 0.0, 0.0],
    /// ).unwrap();
    /// let t1 = TensorDynLen::from_dense(
    ///     vec![bond.clone(), s1.clone()],
    ///     vec![1.0_f64, 0.0, 0.0, 1.0, 0.0, 0.0],
    /// ).unwrap();
    ///
    /// let tn = TreeTN::<_, String>::from_tensors(
    ///     vec![t0, t1],
    ///     vec!["A".to_string(), "B".to_string()],
    /// ).unwrap();
    ///
    /// // Truncate with max rank 2 towards node "A"
    /// let tn = tn.truncate(
    ///     ["A".to_string()],
    ///     TruncationOptions::default().with_max_rank(2),
    /// ).unwrap();
    ///
    /// // Bond dimension is now at most 2
    /// assert_eq!(tn.node_count(), 2);
    /// ```
    pub fn truncate(
        mut self,
        canonical_region: impl IntoIterator<Item = V>,
        options: TruncationOptions,
    ) -> Result<Self>
    where
        V: Ord,
        <T::Index as IndexLike>::Id: Ord,
    {
        self.truncate_impl(
            canonical_region,
            options.svd_policy(),
            options.truncation.max_rank,
            "truncate",
        )?;
        Ok(self)
    }

    /// Truncate the network in-place towards the specified center using options.
    ///
    /// This is the `&mut self` version of [`Self::truncate`].
    /// # Errors
    ///
    /// Returns an error when the operation fails (a shape or index mismatch, an
    /// /// SVD or non-convergence failure, or a backend failure).
    ///
    pub fn truncate_mut(
        &mut self,
        canonical_region: impl IntoIterator<Item = V>,
        options: TruncationOptions,
    ) -> Result<()>
    where
        V: Ord,
        <T::Index as IndexLike>::Id: Ord,
    {
        self.truncate_impl(
            canonical_region,
            options.svd_policy(),
            options.truncation.max_rank,
            "truncate_mut",
        )
    }

    /// Internal implementation for truncation.
    ///
    /// Uses LocalUpdateSweepPlan with TruncateUpdater for full two-site sweeps.
    pub(crate) fn truncate_impl(
        &mut self,
        canonical_region: impl IntoIterator<Item = V>,
        svd_policy: Option<SvdTruncationPolicy>,
        max_rank: Option<usize>,
        context_name: &str,
    ) -> Result<()>
    where
        V: Ord,
        <T::Index as IndexLike>::Id: Ord,
    {
        // Collect center nodes
        let center_nodes: HashSet<V> = canonical_region.into_iter().collect();

        if center_nodes.is_empty() {
            return Ok(()); // Nothing to do
        }

        // Currently only single-node center is supported for truncation
        if center_nodes.len() != 1 {
            return Err(anyhow::anyhow!(
                "truncate currently requires a single-node center, got {} nodes",
                center_nodes.len()
            ))
            .context(format!("{}: multi-node center not supported", context_name));
        }

        let center_node = center_nodes
            .iter()
            .next()
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("truncate requires a center node"))
            .context(format!("{}: missing center node", context_name))?;

        // Step 1: Canonicalize towards the center (required before truncation sweep)
        let canonicalize_options =
            CanonicalizationOptions::default().with_form(CanonicalForm::Unitary);
        self.canonicalize_impl(
            [center_node.clone()],
            canonicalize_options.form,
            &format!("{}: pre-canonicalize", context_name),
        )?;

        // Step 2: Generate sweep plan (nsite=2 for two-site truncation)
        let plan = LocalUpdateSweepPlan::from_treetn(self, &center_node, 2)
            .ok_or_else(|| {
                anyhow::anyhow!("Failed to create sweep plan from center {:?}", center_node)
            })
            .context(format!("{}: sweep plan creation failed", context_name))?;

        // If no steps (single node network), nothing more to do
        if plan.is_empty() {
            return Ok(());
        }

        // Step 3: Apply truncation sweep
        let mut updater = TruncateUpdater::new(max_rank, svd_policy);
        apply_local_update_sweep(self, &plan, &mut updater)
            .context(format!("{}: truncation sweep failed", context_name))?;

        // The canonical form is maintained by the sweep
        self.canonical_form = Some(CanonicalForm::Unitary);

        Ok(())
    }
}
