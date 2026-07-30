//! Site index swap: reorder which node holds which site index.
//!
//! Implements swapping site indices between adjacent nodes along the tree
//! so that the network reaches a target assignment (index -> node name).

use std::collections::{HashMap, HashSet, VecDeque};
use std::hash::Hash;

use anyhow::{Context, Result};
use petgraph::stable_graph::NodeIndex;

use tensor4all_core::{FactorizeOptions, FactorizeResult, IndexLike, TensorLike};

use crate::node_name_network::NodeNameNetwork;

use super::{localupdate::LocalUpdateSweepPlan, TreeTN};

// ============================================================================
// Factorize with trivial-bond handling
// ============================================================================

/// Factorize a tensor into left and right parts connected by a bond index.
///
/// Extends [`TensorFactorizationLike::factorize`](tensor4all_core::TensorFactorizationLike::factorize)
/// to handle degenerate cases where all
/// indices go to one side (empty `left_inds` or `left_inds == all_inds`).
/// For these cases a dimension-1 trivial bond is created so that
/// `contract(left, right)` recovers the input tensor exactly.
///
/// With `Canonical::Left` (the only mode used by swap):
/// - **Normal case**: delegates to `TensorFactorizationLike::factorize`.
/// - **Empty `left_inds`**: `left = [1]` (dim-1 scalar isometry),
///   `right = tensor ⊗ [1]` (acquires the trivial bond).
/// - **Full `left_inds`**: `left = (tensor ⊗ [1]) / ‖tensor‖`,
///   `right = [‖tensor‖]` (norm on the right side, left is isometric).
pub(crate) fn factorize_or_trivial<T>(
    tensor: &T,
    left_inds: &[T::Index],
    all_inds: &[T::Index],
    factorize_options: &FactorizeOptions,
) -> anyhow::Result<FactorizeResult<T>>
where
    T: TensorLike,
    <T::Index as IndexLike>::Id: Clone + Hash + Eq + std::fmt::Debug + Send + Sync,
{
    if left_inds.is_empty() {
        // All indices go to the right side.
        let bond = <T::Index as IndexLike>::create_dummy_link_pair().0;
        let left = T::onehot(&[(bond.clone(), 0)])
            .map_err(|e| anyhow::anyhow!("factorize_or_trivial: left onehot: {}", e))?;
        let right_bond = T::onehot(&[(bond.clone(), 0)])
            .map_err(|e| anyhow::anyhow!("factorize_or_trivial: right onehot: {}", e))?;
        let right = tensor
            .outer_product(&right_bond)
            .context("factorize_or_trivial: right outer_product")?;
        return Ok(FactorizeResult {
            left,
            right,
            bond_index: bond,
            singular_values: None,
            rank: 1,
        });
    }

    if left_inds.len() == all_inds.len() {
        // All indices go to the left side.
        let bond = <T::Index as IndexLike>::create_dummy_link_pair().0;
        let left_bond = T::onehot(&[(bond.clone(), 0)])
            .map_err(|e| anyhow::anyhow!("factorize_or_trivial: left onehot: {}", e))?;
        let mut left = tensor
            .outer_product(&left_bond)
            .context("factorize_or_trivial: left outer_product")?;
        let mut right = T::onehot(&[(bond.clone(), 0)])
            .map_err(|e| anyhow::anyhow!("factorize_or_trivial: right onehot: {}", e))?;
        let left_norm = left.norm();
        if left_norm > 0.0 {
            left = left
                .scale(tensor4all_core::AnyScalar::new_real(1.0 / left_norm))
                .context("factorize_or_trivial: normalize left")?;
            right = right
                .scale(tensor4all_core::AnyScalar::new_real(left_norm))
                .context("factorize_or_trivial: scale right")?;
        }
        return Ok(FactorizeResult {
            left,
            right,
            bond_index: bond,
            singular_values: None,
            rank: 1,
        });
    }

    // Normal case: delegate to TensorFactorizationLike::factorize
    tensor
        .factorize(left_inds, factorize_options)
        .map_err(|e| anyhow::anyhow!("factorize_or_trivial: factorize: {}", e))
}

// ============================================================================
// SwapOptions
// ============================================================================

/// Options for site index swap (truncation during SVD).
///
/// When `max_rank` or `rtol` are set, the swap may introduce approximation error
/// by truncating bond dimension. Default (both `None`) allows rank growth to preserve
/// the tensor exactly: `None` is an explicit "no truncation" request and does
/// *not* fall back to the process-global SVD truncation policy.
#[derive(Debug, Clone, Default)]
pub struct SwapOptions {
    /// Maximum bond dimension after each SVD (None = no limit).
    pub max_rank: Option<usize>,
    /// Relative tolerance for singular value truncation (None = no truncation,
    /// equivalent to `Some(0.0)`).
    pub rtol: Option<f64>,
}

// ============================================================================
// ScheduledSwapStep
// ============================================================================

/// A single two-site update step in a pre-computed swap schedule.
///
/// Use this to inspect exactly which edge is updated, whether the canonical
/// center must be transported first, and which site indices must end up on
/// each side of the edge after the local factorization.
///
/// Related types:
/// - [`SwapSchedule`] stores the full ordered sequence of these steps.
/// - [`SwapOptions`] controls truncation only during execution, not schedule construction.
///
/// # Examples
///
/// ```
/// use std::collections::HashSet;
///
/// use tensor4all_treetn::ScheduledSwapStep;
///
/// let step = ScheduledSwapStep {
///     transport_path: vec!["L0".to_string(), "C".to_string()],
///     node_a: "C".to_string(),
///     node_b: "L1".to_string(),
///     a_side_sites: HashSet::from(["s1".to_string()]),
///     b_side_sites: HashSet::from(["s0".to_string()]),
/// };
///
/// assert_eq!(step.transport_path, vec!["L0".to_string(), "C".to_string()]);
/// assert!(step.a_side_sites.contains("s1"));
/// assert!(step.b_side_sites.contains("s0"));
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScheduledSwapStep<V, K>
where
    K: Eq + Hash,
{
    /// Path to transport the canonical center before the swap.
    ///
    /// Empty when the center is already at `node_a` or `node_b`.
    /// Otherwise this is `[current_center, ..., node_a]`.
    pub transport_path: Vec<V>,
    /// The first node in the directed sweep edge.
    pub node_a: V,
    /// The second node in the directed sweep edge.
    pub node_b: V,
    /// Site index keys that should live on `node_a`'s side after this step.
    pub a_side_sites: HashSet<K>,
    /// Site index keys that should live on `node_b`'s side after this step.
    pub b_side_sites: HashSet<K>,
}

// ============================================================================
// SwapSchedule
// ============================================================================

/// Pre-computed swap schedule for `swap_site_indices`.
///
/// The schedule is derived purely from graph structure plus current and target
/// site assignments. It contains no tensor data and can therefore be built,
/// inspected, and unit-tested without performing any tensor contractions.
///
/// Related types:
/// - [`ScheduledSwapStep`] is one local two-site update in this schedule.
/// - [`SwapOptions`] affects execution of the schedule, but not its contents.
///
/// # Examples
///
/// ```
/// use std::collections::HashMap;
///
/// use tensor4all_treetn::{NodeNameNetwork, SwapSchedule};
///
/// let mut topology = NodeNameNetwork::new();
/// topology.add_node("A".to_string()).unwrap();
/// topology.add_node("B".to_string()).unwrap();
/// topology.add_edge(&"A".to_string(), &"B".to_string()).unwrap();
///
/// let current = HashMap::from([("s0".to_string(), "A".to_string())]);
/// let target = HashMap::from([("s0".to_string(), "B".to_string())]);
/// let root = "A".to_string();
///
/// let schedule = SwapSchedule::build(&topology, &current, &target, &root).unwrap();
///
/// assert_eq!(schedule.root, "A");
/// assert_eq!(schedule.steps.len(), 1);
/// assert_eq!(schedule.steps[0].node_a, "A");
/// assert_eq!(schedule.steps[0].node_b, "B");
/// ```
#[derive(Debug, Clone)]
pub struct SwapSchedule<V, K>
where
    K: Eq + Hash,
{
    /// Root used for the base Euler sweep and initial canonicalization.
    pub root: V,
    /// Fully expanded sequence of swap steps.
    pub steps: Vec<ScheduledSwapStep<V, K>>,
}

impl<V, K> SwapSchedule<V, K>
where
    V: Clone + Hash + Eq + std::fmt::Debug + Send + Sync,
    K: Clone + Hash + Eq + std::fmt::Debug,
{
    /// Build a swap schedule from topology plus current and target assignments.
    ///
    /// The returned schedule is a pure graph computation. It simulates site
    /// positions through repeated Euler-tour sweeps, emits only edges where at
    /// least one targeted site index crosses, and records any required
    /// canonical-center transport between non-adjacent emitted swap steps.
    ///
    /// # Arguments
    /// * `topology` - Tree topology whose nodes are named by `V`.
    /// * `current_assignment` - Current node for every site index key in the network.
    /// * `target_assignment` - Partial target map; indices not listed keep their current side.
    /// * `root` - Sweep root and assumed initial canonical center.
    ///
    /// # Returns
    /// A [`SwapSchedule`] containing the ordered local updates needed to realize `target_assignment`.
    ///
    /// # Errors
    /// Returns an error if `root` is missing, an index key in `target_assignment`
    /// is unknown, a referenced node is missing from `topology`, no tree path
    /// exists between required nodes, or the simulated sweeps fail to satisfy
    /// the requested target assignment within the tree-diameter pass bound.
    pub fn build(
        topology: &NodeNameNetwork<V>,
        current_assignment: &HashMap<K, V>,
        target_assignment: &HashMap<K, V>,
        root: &V,
    ) -> Result<Self> {
        if !topology.has_node(root) {
            return Err(anyhow::anyhow!(
                "SwapSchedule::build: root {:?} not in topology",
                root
            ));
        }

        for (index, current_node) in current_assignment {
            if !topology.has_node(current_node) {
                return Err(anyhow::anyhow!(
                    "SwapSchedule::build: current node {:?} for index {:?} is not in the topology",
                    current_node,
                    index
                ));
            }
        }

        for (index, target_node) in target_assignment {
            if !current_assignment.contains_key(index) {
                return Err(anyhow::anyhow!(
                    "SwapSchedule::build: target_assignment contains index {:?} which is not in the network",
                    index
                ));
            }
            if !topology.has_node(target_node) {
                return Err(anyhow::anyhow!(
                    "SwapSchedule::build: target node {:?} for index {:?} is not in the topology",
                    target_node,
                    index
                ));
            }
        }

        let oracle = SubtreeOracle::new(topology, root)?;
        let base_sweep = LocalUpdateSweepPlan::new(topology, root, 2)
            .ok_or_else(|| anyhow::anyhow!("SwapSchedule::build: failed to build 2-site sweep"))?;
        let max_passes = tree_diameter(topology)?;

        let mut position = current_assignment.clone();
        let mut center = root.clone();
        let mut steps = Vec::new();

        for _pass in 0..max_passes {
            if positions_satisfy_targets(&position, target_assignment) {
                break;
            }

            let mut any_moved_this_pass = false;

            for sweep_step in base_sweep.iter() {
                if sweep_step.nodes.len() != 2 {
                    continue;
                }

                let node_a = sweep_step.nodes[0].clone();
                let node_b = sweep_step.nodes[1].clone();

                let mut a_side_sites = HashSet::new();
                let mut b_side_sites = HashSet::new();
                let mut any_crossing = false;
                let mut any_site_on_edge = false;

                for (index, current_node) in &position {
                    if current_node != &node_a && current_node != &node_b {
                        continue;
                    }

                    any_site_on_edge = true;

                    if let Some(target_node) = target_assignment.get(index) {
                        if oracle.is_target_on_a_side(&node_a, &node_b, target_node) {
                            a_side_sites.insert(index.clone());
                            if current_node == &node_b {
                                any_crossing = true;
                            }
                        } else {
                            b_side_sites.insert(index.clone());
                            if current_node == &node_a {
                                any_crossing = true;
                            }
                        }
                    } else if current_node == &node_a {
                        a_side_sites.insert(index.clone());
                    } else {
                        b_side_sites.insert(index.clone());
                    }
                }

                if !any_site_on_edge || !any_crossing {
                    continue;
                }

                let transport_path = if center == node_a || center == node_b {
                    Vec::new()
                } else {
                    tree_path(topology, &center, &node_a)?
                };

                steps.push(ScheduledSwapStep {
                    transport_path,
                    node_a: node_a.clone(),
                    node_b: node_b.clone(),
                    a_side_sites: a_side_sites.clone(),
                    b_side_sites: b_side_sites.clone(),
                });

                for index in &a_side_sites {
                    position.insert(index.clone(), node_a.clone());
                }
                for index in &b_side_sites {
                    position.insert(index.clone(), node_b.clone());
                }

                center = node_b;
                any_moved_this_pass = true;
            }

            if !any_moved_this_pass {
                break;
            }
        }

        if !positions_satisfy_targets(&position, target_assignment) {
            return Err(anyhow::anyhow!(
                "SwapSchedule::build: did not converge within {} passes",
                max_passes
            ));
        }

        Ok(Self {
            root: root.clone(),
            steps,
        })
    }
}

fn positions_satisfy_targets<V, K>(
    position: &HashMap<K, V>,
    target_assignment: &HashMap<K, V>,
) -> bool
where
    V: Eq,
    K: Hash + Eq,
{
    target_assignment
        .iter()
        .all(|(index, target_node)| position.get(index).is_some_and(|node| node == target_node))
}

fn tree_path<V>(topology: &NodeNameNetwork<V>, from: &V, to: &V) -> Result<Vec<V>>
where
    V: Clone + Hash + Eq + std::fmt::Debug + Send + Sync,
{
    let from_idx = topology
        .node_index(from)
        .ok_or_else(|| anyhow::anyhow!("tree_path: node {:?} not found", from))?;
    let to_idx = topology
        .node_index(to)
        .ok_or_else(|| anyhow::anyhow!("tree_path: node {:?} not found", to))?;

    topology
        .path_between(from_idx, to_idx)
        .ok_or_else(|| anyhow::anyhow!("tree_path: no path between {:?} and {:?}", from, to))?
        .into_iter()
        .map(|node_idx| {
            topology
                .node_name(node_idx)
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("tree_path: node name not found"))
        })
        .collect()
}

fn tree_diameter<V>(topology: &NodeNameNetwork<V>) -> Result<usize>
where
    V: Clone + Hash + Eq + std::fmt::Debug + Send + Sync,
{
    let mut node_indices = topology.graph().node_indices();
    let Some(start) = node_indices.next() else {
        return Ok(0);
    };

    let (farthest, _) = farthest_node(topology, start)?;
    let (_, diameter) = farthest_node(topology, farthest)?;
    Ok(diameter)
}

fn farthest_node<V>(topology: &NodeNameNetwork<V>, start: NodeIndex) -> Result<(NodeIndex, usize)>
where
    V: Clone + Hash + Eq + std::fmt::Debug + Send + Sync,
{
    let graph = topology.graph();
    let mut visited = HashSet::new();
    let mut queue = VecDeque::from([(start, 0usize)]);
    let mut farthest = (start, 0usize);

    visited.insert(start);

    while let Some((node, distance)) = queue.pop_front() {
        if distance > farthest.1 {
            farthest = (node, distance);
        }

        for neighbor in graph.neighbors(node) {
            if visited.insert(neighbor) {
                queue.push_back((neighbor, distance + 1));
            }
        }
    }

    if visited.len() != graph.node_count() {
        return Err(anyhow::anyhow!(
            "SwapSchedule::build: topology must be connected"
        ));
    }

    Ok(farthest)
}

// ============================================================================
// Helpers: current assignment
// ============================================================================

/// Build full site index -> node name from a TreeTN (all site indices).
pub(crate) fn current_site_assignment<T, V>(treetn: &TreeTN<T, V>) -> HashMap<T::Index, V>
where
    T: TensorLike,
    T::Index: Hash + Eq,
    V: Clone + Hash + Eq + Send + Sync + std::fmt::Debug,
{
    let mut out: HashMap<T::Index, V> = HashMap::new();
    for node_name in treetn.node_names() {
        if let Some(site_space) = treetn.site_space(&node_name) {
            for idx in site_space {
                out.insert(idx.clone(), node_name.clone());
            }
        }
    }
    out
}

// ============================================================================
// SubtreeOracle
// ============================================================================

/// Pre-computed DFS timestamps enabling O(1) "which side of edge?" queries.
///
/// For an edge (A, B): `is_target_on_a_side(A, B, target)` returns true iff
/// `target` is in the component containing A when the edge is removed.
pub(crate) struct SubtreeOracle<V> {
    in_time: HashMap<V, usize>,
    out_time: HashMap<V, usize>,
}

impl<V> SubtreeOracle<V>
where
    V: Clone + Hash + Eq + std::fmt::Debug + Send + Sync,
{
    /// Build from a tree topology rooted at `root`.
    /// DFS entry/exit timestamps are computed iteratively.
    pub(crate) fn new(topology: &NodeNameNetwork<V>, root: &V) -> Result<Self> {
        let root_idx = topology
            .node_index(root)
            .ok_or_else(|| anyhow::anyhow!("SubtreeOracle: root {:?} not in topology", root))?;

        let mut in_time: HashMap<V, usize> = HashMap::new();
        let mut out_time: HashMap<V, usize> = HashMap::new();
        let mut timer = 0usize;

        // Stack: (node_idx, parent_idx, is_exit)
        let mut stack: Vec<(NodeIndex, Option<NodeIndex>, bool)> = vec![(root_idx, None, false)];

        while let Some((node_idx, parent_idx, is_exit)) = stack.pop() {
            let name = topology
                .node_name(node_idx)
                .ok_or_else(|| anyhow::anyhow!("SubtreeOracle: node name not found"))?
                .clone();
            if is_exit {
                out_time.insert(name, timer);
                timer += 1;
            } else {
                in_time.insert(name, timer);
                timer += 1;
                stack.push((node_idx, parent_idx, true));
                let graph = topology.graph();
                for neighbor in graph.neighbors(node_idx) {
                    if Some(neighbor) != parent_idx {
                        stack.push((neighbor, Some(node_idx), false));
                    }
                }
            }
        }

        Ok(Self { in_time, out_time })
    }

    /// Returns `true` iff `target` is on the A-side of edge (A, B).
    ///
    /// A-side = the connected component containing A after the (A,B) edge is removed.
    pub(crate) fn is_target_on_a_side(&self, node_a: &V, node_b: &V, target: &V) -> bool {
        if target == node_a {
            return true;
        }
        if target == node_b {
            return false;
        }
        let in_a = match self.in_time.get(node_a) {
            Some(&t) => t,
            None => return false,
        };
        let out_a = match self.out_time.get(node_a) {
            Some(&t) => t,
            None => return false,
        };
        let in_b = match self.in_time.get(node_b) {
            Some(&t) => t,
            None => return false,
        };
        let out_b = match self.out_time.get(node_b) {
            Some(&t) => t,
            None => return false,
        };
        let in_t = match self.in_time.get(target) {
            Some(&t) => t,
            None => return false,
        };
        let out_t = match self.out_time.get(target) {
            Some(&t) => t,
            None => return false,
        };

        if in_a <= in_b && out_b <= out_a {
            !(in_b <= in_t && out_t <= out_b)
        } else {
            in_a <= in_t && out_t <= out_a
        }
    }
}

#[cfg(test)]
mod tests;
