//! ProjectedState: 2-chain environment for RHS computation (square case).
//!
//! Computes the local RHS term consistent with the square linsolve setup:
//! - ProjectedOperator builds `<ref|H|x>`
//! - ProjectedState builds `<ref|b>`
//!
//! This returns a local tensor with open indices aligned to the current solution's
//! local tensor (up to permutation), so GMRES can operate in the same vector space.
//!
//! This is the V_in = V_out specialized version.

use std::hash::Hash;

use anyhow::Result;

use tensor4all_core::{IndexLike, TensorLike};

use crate::linsolve::common::{EnvironmentCache, NetworkTopology};
use crate::treetn::TreeTN;

/// ProjectedState: Manages 2-chain environments for RHS computation.
///
/// This computes `<ref|b>` (conjugated reference contracted with the RHS) for
/// each local region during the sweep.
///
/// For Tree Tensor Networks, the environment is computed by contracting
/// all tensors outside the "open region" into environment tensors.
/// The open region consists of nodes being updated in the current sweep step.
///
/// # Structure
///
/// For each edge (from, to) pointing towards the open region, we cache:
/// ```text
/// env[(from, to)] = contraction of:
///   - bra tensor at `from` (conjugated reference state)
///   - ket tensor at `from` (RHS)
///   - all child environments (edges pointing away from `to`)
/// ```
///
/// This forms a "2-chain" overlap: `<ref|b>` contracted over
/// all nodes except the open region.
pub struct ProjectedState<T, V>
where
    T: TensorLike,
    V: Clone + Hash + Eq + Send + Sync + std::fmt::Debug,
{
    /// The RHS state |b⟩
    pub rhs: TreeTN<T, V>,
    /// Environment cache
    pub envs: EnvironmentCache<T, V>,
}

impl<T, V> ProjectedState<T, V>
where
    T: TensorLike,
    <T::Index as IndexLike>::Id:
        Clone + std::hash::Hash + Eq + Ord + std::fmt::Debug + Send + Sync + 'static,
    V: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug,
{
    /// Create a new ProjectedState.
    pub fn new(rhs: TreeTN<T, V>) -> Self {
        Self {
            rhs,
            envs: EnvironmentCache::new(),
        }
    }

    /// Compute the local constant term (local RHS) for the given region.
    ///
    /// This returns the local RHS tensors contracted with environments.
    ///
    /// For the square case, the `reference_state` is used as the bra (conjugated),
    /// and `rhs` is used as the ket, i.e. environments are constructed for `<ref|b>`.
    /// If the reference and RHS networks share link indices, the reference links
    /// are relabeled for this local contraction so bra/ket boundary links are
    /// not accidentally traced out.
    ///
    /// # Arguments
    /// * `region` - The nodes in the local update region
    /// * `reference_state` - The current solution state (used as reference for environments)
    /// * `topology` - The network topology
    pub fn local_constant_term<NT: NetworkTopology<V>>(
        &mut self,
        region: &[V],
        reference_state: &TreeTN<T, V>,
        topology: &NT,
    ) -> Result<T> {
        let reference_storage;
        let reference_state = if self.has_link_collision_with_rhs(reference_state, topology)? {
            // A bra/ket overlap needs two independent copies of every open link:
            // the RHS-side link is contracted with the local RHS tensor, while
            // the reference-side link remains as the output local vector space.
            self.envs.clear();
            reference_storage = Some(reference_state.sim_linkinds()?);
            reference_storage
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("simulated reference storage was not initialized"))?
        } else {
            reference_state
        };

        // Ensure environments are computed
        self.ensure_environments(region, reference_state, topology)?;

        // Collect all tensors to contract: local RHS tensors + environments
        let mut all_tensors: Vec<T> = Vec::new();

        // Collect local RHS tensors (ket side; do NOT conjugate)
        for node in region {
            let node_idx = self
                .rhs
                .node_index(node)
                .ok_or_else(|| anyhow::anyhow!("Node {:?} not found in RHS", node))?;
            let tensor = self
                .rhs
                .tensor(node_idx)
                .ok_or_else(|| anyhow::anyhow!("Tensor not found in RHS"))?
                .clone();
            all_tensors.push(tensor);
        }

        // Collect environments from neighbors outside the region
        for node in region {
            for neighbor in topology.neighbors(node) {
                if !region.contains(&neighbor) {
                    if let Some(env) = self.envs.get(&neighbor, node) {
                        all_tensors.push(env.clone());
                    }
                }
            }
        }

        // Use T::contract for optimal contraction ordering
        let tensor_refs: Vec<&T> = all_tensors.iter().collect();
        T::contract(&tensor_refs)
    }

    fn has_link_collision_with_rhs<NT: NetworkTopology<V>>(
        &self,
        reference_state: &TreeTN<T, V>,
        topology: &NT,
    ) -> Result<bool> {
        for node in reference_state.node_names() {
            for neighbor in topology.neighbors(&node) {
                if node > neighbor {
                    continue;
                }

                let Some(ref_edge) = reference_state.edge_between(&node, &neighbor) else {
                    continue;
                };
                let Some(rhs_edge) = self.rhs.edge_between(&node, &neighbor) else {
                    continue;
                };
                let ref_bond = reference_state.bond_index(ref_edge).ok_or_else(|| {
                    anyhow::anyhow!(
                        "Reference bond index not found for edge {:?}-{:?}",
                        node,
                        neighbor
                    )
                })?;
                let rhs_bond = self.rhs.bond_index(rhs_edge).ok_or_else(|| {
                    anyhow::anyhow!(
                        "RHS bond index not found for edge {:?}-{:?}",
                        node,
                        neighbor
                    )
                })?;

                if ref_bond == rhs_bond {
                    return Ok(true);
                }
            }
        }
        Ok(false)
    }

    /// Ensure environments are computed for neighbors of the region.
    fn ensure_environments<NT: NetworkTopology<V>>(
        &mut self,
        region: &[V],
        reference_state: &TreeTN<T, V>,
        topology: &NT,
    ) -> Result<()> {
        for node in region {
            for neighbor in topology.neighbors(node) {
                if !region.contains(&neighbor) && !self.envs.contains(&neighbor, node) {
                    let env =
                        self.compute_environment(&neighbor, node, reference_state, topology)?;
                    self.envs.insert(neighbor.clone(), node.clone(), env);
                }
            }
        }
        Ok(())
    }

    /// Recursively compute environment for edge (from, to).
    ///
    /// Computes `<ref|b>` partial contraction at node `from`.
    fn compute_environment<NT: NetworkTopology<V>>(
        &mut self,
        from: &V,
        to: &V,
        reference_state: &TreeTN<T, V>,
        topology: &NT,
    ) -> Result<T> {
        // First, ensure child environments are computed
        let child_neighbors: Vec<V> = topology.neighbors(from).filter(|n| n != to).collect();

        for child in &child_neighbors {
            if !self.envs.contains(child, from) {
                let child_env = self.compute_environment(child, from, reference_state, topology)?;
                self.envs.insert(child.clone(), from.clone(), child_env);
            }
        }

        // Collect child environments
        let child_envs: Vec<&T> = child_neighbors
            .iter()
            .filter_map(|child| self.envs.get(child, from))
            .collect();

        // Contract bra (reference_state) with ket (RHS) at this node
        let node_idx_ref = reference_state
            .node_index(from)
            .ok_or_else(|| anyhow::anyhow!("Node {:?} not found in reference_state", from))?;
        let node_idx_b = self
            .rhs
            .node_index(from)
            .ok_or_else(|| anyhow::anyhow!("Node {:?} not found in RHS", from))?;

        let tensor_ref = reference_state
            .tensor(node_idx_ref)
            .ok_or_else(|| anyhow::anyhow!("Tensor not found in reference_state"))?;
        let tensor_b = self
            .rhs
            .tensor(node_idx_b)
            .ok_or_else(|| anyhow::anyhow!("Tensor not found in RHS"))?;

        let bra_conj = tensor_ref.conj();

        // Contract bra and ket - T::contract auto-detects contractable pairs
        let bra_ket = T::contract(&[&bra_conj, tensor_b])?;

        // Contract bra*ket with child environments using T::contract
        if child_envs.is_empty() {
            Ok(bra_ket)
        } else {
            let mut all_tensors: Vec<&T> = vec![&bra_ket];
            all_tensors.extend(child_envs);
            T::contract(&all_tensors)
        }
    }

    /// Invalidate caches affected by updates to the given region.
    pub fn invalidate<NT: NetworkTopology<V>>(&mut self, region: &[V], topology: &NT) {
        self.envs.invalidate(region, topology);
    }
}
