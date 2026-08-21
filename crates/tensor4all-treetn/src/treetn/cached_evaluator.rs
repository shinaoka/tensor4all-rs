//! Cached batch evaluation for tree tensor networks.

use crate::error::TreeTNOperationError;
use std::collections::{HashMap, HashSet};
use std::fmt::Debug;
use std::hash::Hash;
use std::sync::Arc;

/// Temporary phase-timing counters for root-cause investigation into why the
/// message cache does not deliver a net speedup despite a high hit rate. Not
/// on the hot path in non-test builds.
#[cfg(test)]
mod phase_timing {
    use std::sync::atomic::{AtomicU64, Ordering};

    pub static KEY_AND_LOOKUP_NS: AtomicU64 = AtomicU64::new(0);
    pub static CONTRACT_NS: AtomicU64 = AtomicU64::new(0);
    pub static TENSOR_VALUES_NS: AtomicU64 = AtomicU64::new(0);
    pub static RECONSTRUCT_NS: AtomicU64 = AtomicU64::new(0);
    pub static INSERT_NS: AtomicU64 = AtomicU64::new(0);

    pub fn add(counter: &AtomicU64, elapsed: std::time::Duration) {
        counter.fetch_add(elapsed.as_nanos() as u64, Ordering::Relaxed);
    }

    pub fn reset_all() {
        KEY_AND_LOOKUP_NS.store(0, Ordering::Relaxed);
        CONTRACT_NS.store(0, Ordering::Relaxed);
        TENSOR_VALUES_NS.store(0, Ordering::Relaxed);
        RECONSTRUCT_NS.store(0, Ordering::Relaxed);
        INSERT_NS.store(0, Ordering::Relaxed);
    }
}

use anyhow::{bail, ensure, Context, Result};
use num_complex::Complex64;
use tensor4all_core::{
    contract_with_options,
    index_key::{FlatIndexer, IndexKey},
    AnyScalar, ColMajorArrayRef, ContractionOptions, DynIndex, IdxTensor, IndexLike,
    TensorContractionLike, TensorIndex, TensorLike,
};
use tensor4all_tensorbackend::{mat_mul_owned, BlasMul, Matrix};

use super::TreeTN;

type KeyId = usize;
type EnvironmentCache<V> = HashMap<V, StackedMessage>;
type CacheBuildResult<V> = (Vec<ComponentBatch<V>>, EnvironmentCache<V>);
type ParentMap<V> = HashMap<V, Option<V>>;

/// Minimum scalar multiply count before the backend setup cost is amortized
/// by the grouped chain kernel. Smaller contractions keep the existing scalar
/// loop, which is faster for the tiny messages common at low bond dimension.
const CHAIN_BLAS_WORK_THRESHOLD: usize = 4096;
/// Minimum number of columns in every physical-value group before paying for
/// a backend matrix multiplication. This avoids turning the common one- or
/// two-point floating-zone callback into a collection of matrix-vector calls.
const CHAIN_BLAS_MIN_GROUP_POINTS: usize = 4;

#[derive(Clone, Copy, Debug)]
struct ChainContractionSpec {
    strides: [usize; 3],
    physical_axis: usize,
    parent_axis: usize,
    child_axis: usize,
    parent_dim: usize,
    child_dim: usize,
}

#[derive(Clone, Debug)]
struct SiteEntry {
    index: DynIndex,
    input_position: usize,
    local_axis: usize,
}

#[derive(Clone, Debug)]
struct MessageCacheLayout {
    input_positions: Vec<usize>,
    indexer: FlatIndexer,
}

#[derive(Clone, Debug)]
struct EvaluatorLayout<V> {
    entries_by_node: HashMap<V, Vec<SiteEntry>>,
    n_indices: usize,
}

#[derive(Default)]
struct KeyInterner<T>
where
    T: Clone + Eq + Hash,
{
    ids: HashMap<T, KeyId>,
}

impl<T> KeyInterner<T>
where
    T: Clone + Eq + Hash,
{
    fn intern(&mut self, key: T) -> KeyId {
        let next = self.ids.len();
        *self.ids.entry(key).or_insert(next)
    }
}

#[derive(Clone, Debug)]
struct AssignmentBatch {
    point_to_assignment: Vec<usize>,
    first_points: Vec<usize>,
}

#[derive(Clone, Debug)]
struct ComponentBatch<V> {
    neighbor: V,
    point_to_assignment: Vec<usize>,
}

#[derive(Clone, Debug)]
struct StackedMessage {
    assignment_index: DynIndex,
    tensor: Option<IdxTensor>,
    raw_values: Option<Vec<CachedScalar>>,
}

#[derive(Clone, Copy, Debug)]
enum CachedScalar {
    Real(f64),
    Complex(Complex64),
}

impl CachedScalar {
    fn from_any(value: AnyScalar) -> Self {
        value
            .as_c64()
            .map(Self::Complex)
            .unwrap_or_else(|| Self::Real(value.real()))
    }

    fn into_any(self) -> AnyScalar {
        match self {
            Self::Real(value) => AnyScalar::new_real(value),
            Self::Complex(value) => AnyScalar::new_complex(value.re, value.im),
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct CachedEvaluationStats {
    subtree_environment_count: usize,
    directed_message_count: usize,
    batched_message_contract_count: usize,
    batched_center_contract_count: usize,
    message_cache_hits: usize,
    message_cache_misses: usize,
}

#[derive(Clone, Debug)]
struct RootedMessagePlan<V> {
    children: HashMap<V, Vec<V>>,
    subtree_nodes: HashMap<V, Vec<V>>,
    postorder: Vec<V>,
    parent: ParentMap<V>,
}

#[derive(Debug)]
struct ComponentCostIndex<V> {
    neighbors: HashMap<V, Vec<V>>,
    directed_counts: HashMap<(V, V), usize>,
    node_costs: Option<HashMap<V, usize>>,
}

impl<V> ComponentCostIndex<V>
where
    V: Clone + Eq + Hash + Ord + Debug + Send + Sync,
{
    /// # Errors
    ///
    /// Returns an error when the construction or conversion fails (a shape or
    /// /// index mismatch, or a backend failure).
    ///
    fn new(
        tree: &TreeTN<IdxTensor, V>,
        indices: &[DynIndex],
        values: ColMajorArrayRef<'_, usize>,
    ) -> Result<Self> {
        let layout = build_layout(tree, indices)?;
        Self::from_layout(tree, &layout, values)
    }

    fn from_layout(
        tree: &TreeTN<IdxTensor, V>,
        layout: &EvaluatorLayout<V>,
        values: ColMajorArrayRef<'_, usize>,
    ) -> Result<Self> {
        validate_values_shape(values, layout.n_indices, "ComponentCostIndex::new")?;
        let n_points = values.shape()[1];

        let neighbors = sorted_neighbors(tree);
        if neighbors.is_empty() {
            return Ok(Self {
                neighbors,
                directed_counts: HashMap::new(),
                node_costs: None,
            });
        }

        let mut node_names: Vec<V> = neighbors.keys().cloned().collect();
        node_names.sort();
        let root = node_names[0].clone();

        let (parent, order) = rooted_tree(&neighbors, &root)?;
        let mut local_interner = KeyInterner::<Vec<usize>>::default();
        let mut local_keys: HashMap<V, Vec<KeyId>> = HashMap::with_capacity(node_names.len());
        for node in &node_names {
            let entries = layout
                .entries_by_node
                .get(node)
                .map(Vec::as_slice)
                .unwrap_or(&[]);
            let mut keys = Vec::with_capacity(n_points);
            for point in 0..n_points {
                let key = entries
                    .iter()
                    .map(|entry| {
                        value_at(
                            values,
                            entry.input_position,
                            point,
                            "ComponentCostIndex::new",
                        )
                    })
                    .collect::<Result<Vec<_>>>()?;
                validate_entry_values(entries, &key, "ComponentCostIndex::new")?;
                keys.push(local_interner.intern(key));
            }
            local_keys.insert(node.clone(), keys);
        }

        let mut component_interner = KeyInterner::<Vec<KeyId>>::default();
        let mut directed_keys: HashMap<(V, V), Vec<KeyId>> =
            HashMap::with_capacity(tree.edge_count() * 2);

        for node in order.iter().rev() {
            let Some(parent_node) = parent.get(node).and_then(Clone::clone) else {
                continue;
            };
            let node_neighbors = neighbors.get(node).ok_or_else(|| {
                anyhow::anyhow!("ComponentCostIndex::new: missing neighbors for {:?}", node)
            })?;
            let incoming = node_neighbors
                .iter()
                .filter(|neighbor| *neighbor != &parent_node)
                .map(|neighbor| {
                    directed_keys
                        .get(&(neighbor.clone(), node.clone()))
                        .with_context(|| {
                            format!(
                                "ComponentCostIndex::new: missing child key {:?}->{:?}",
                                neighbor, node
                            )
                        })
                })
                .collect::<Result<Vec<_>>>()?;
            let node_local_keys = local_keys.get(node).ok_or_else(|| {
                anyhow::anyhow!("ComponentCostIndex::new: missing local keys for {:?}", node)
            })?;
            let keys = intern_component_keys(
                node_local_keys,
                &incoming,
                n_points,
                &mut component_interner,
            );
            directed_keys.insert((node.clone(), parent_node), keys);
        }

        for node in &order {
            let node_neighbors = neighbors.get(node).ok_or_else(|| {
                anyhow::anyhow!("ComponentCostIndex::new: missing neighbors for {:?}", node)
            })?;
            for child in node_neighbors.iter().filter(|neighbor| {
                parent.get(*neighbor).and_then(Clone::clone) == Some(node.clone())
            }) {
                let incoming = node_neighbors
                    .iter()
                    .filter(|neighbor| *neighbor != child)
                    .map(|neighbor| {
                        directed_keys
                            .get(&(neighbor.clone(), node.clone()))
                            .with_context(|| {
                                format!(
                                    "ComponentCostIndex::new: missing incoming key {:?}->{:?}",
                                    neighbor, node
                                )
                            })
                    })
                    .collect::<Result<Vec<_>>>()?;
                let node_local_keys = local_keys.get(node).ok_or_else(|| {
                    anyhow::anyhow!("ComponentCostIndex::new: missing local keys for {:?}", node)
                })?;
                let keys = intern_component_keys(
                    node_local_keys,
                    &incoming,
                    n_points,
                    &mut component_interner,
                );
                directed_keys.insert((node.clone(), child.clone()), keys);
            }
        }

        let directed_counts = directed_keys
            .into_iter()
            .map(|(edge, keys)| {
                let count = keys.into_iter().collect::<HashSet<_>>().len();
                (edge, count)
            })
            .collect();

        Ok(Self {
            neighbors,
            directed_counts,
            node_costs: None,
        })
    }

    fn all_nodes(&self) -> Vec<V> {
        let mut nodes: Vec<V> = self.neighbors.keys().cloned().collect();
        nodes.sort();
        nodes
    }

    fn component_count(&self, edge: &(V, V)) -> Option<usize> {
        self.directed_counts.get(edge).copied()
    }

    fn center_cost(&self, center: &V) -> Result<usize> {
        if let Some(node_costs) = &self.node_costs {
            return node_costs.get(center).copied().ok_or_else(|| {
                anyhow::anyhow!("center {:?} is not present in cost index", center)
            });
        }
        let neighbors = self
            .neighbors
            .get(center)
            .ok_or_else(|| anyhow::anyhow!("center {:?} is not present in cost index", center))?;
        neighbors.iter().try_fold(0usize, |acc, neighbor| {
            self.component_count(&(neighbor.clone(), center.clone()))
                .map(|count| acc + count)
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "missing component cost for directed edge {:?}->{:?}",
                        neighbor,
                        center
                    )
                })
        })
    }

    #[cfg(test)]
    fn from_parts_for_test(
        mut neighbors: HashMap<V, Vec<V>>,
        node_costs: HashMap<V, usize>,
    ) -> Self {
        for neighbor_list in neighbors.values_mut() {
            neighbor_list.sort();
        }
        Self {
            neighbors,
            directed_counts: HashMap::new(),
            node_costs: Some(node_costs),
        }
    }
}

fn intern_component_keys(
    local_keys: &[KeyId],
    incoming: &[&Vec<KeyId>],
    n_points: usize,
    interner: &mut KeyInterner<Vec<KeyId>>,
) -> Vec<KeyId> {
    let mut keys = Vec::with_capacity(n_points);
    for point in 0..n_points {
        let mut tuple = Vec::with_capacity(1 + incoming.len());
        tuple.push(local_keys[point]);
        for incoming_keys in incoming {
            tuple.push(incoming_keys[point]);
        }
        keys.push(interner.intern(tuple));
    }
    keys
}

impl<V> RootedMessagePlan<V>
where
    V: Clone + Eq + Hash + Ord + Debug + Send + Sync,
{
    /// # Errors
    ///
    /// Returns an error when the construction or conversion fails (a shape or
    /// /// index mismatch, or a backend failure).
    ///
    fn new(tree: &TreeTN<IdxTensor, V>, center: &V) -> Result<Self> {
        let neighbors = sorted_neighbors(tree);
        let (parent, order) = rooted_tree(&neighbors, center)?;

        let mut children = HashMap::<V, Vec<V>>::new();
        for node in neighbors.keys() {
            children.insert(node.clone(), Vec::new());
        }
        for (node, parent_node) in &parent {
            if let Some(parent_node) = parent_node {
                children
                    .get_mut(parent_node)
                    .ok_or_else(|| anyhow::anyhow!("missing rooted parent {:?}", parent_node))?
                    .push(node.clone());
            }
        }
        for node_children in children.values_mut() {
            node_children.sort();
        }

        let mut subtree_nodes = HashMap::<V, Vec<V>>::with_capacity(order.len());
        for node in order.iter().rev() {
            let mut subtree = vec![node.clone()];
            for child in children.get(node).map(Vec::as_slice).unwrap_or(&[]) {
                let child_subtree = subtree_nodes.get(child).ok_or_else(|| {
                    anyhow::anyhow!("missing rooted subtree for child {:?}", child)
                })?;
                subtree.extend(child_subtree.iter().cloned());
            }
            subtree_nodes.insert(node.clone(), subtree);
        }

        let postorder = order
            .into_iter()
            .rev()
            .filter(|node| node != center)
            .collect::<Vec<_>>();

        Ok(Self {
            children,
            subtree_nodes,
            postorder,
            parent,
        })
    }
}

/// Options controlling cached batch evaluation for [`TreeTN`].
///
/// Use this to pin the contraction center or to configure the greedy automatic
/// center search. When in doubt, leave all fields at their defaults.
///
/// # Examples
///
/// ```
/// use tensor4all_treetn::CachedEvaluatorOptions;
///
/// let options = CachedEvaluatorOptions::<usize>::default();
/// assert!(options.center.is_none());
/// assert!(options.initial_centers.is_empty());
/// assert!(options.max_greedy_steps_per_start.is_none());
/// assert_eq!(options.message_cache_max_bytes, usize::MAX);
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CachedEvaluatorOptions<V> {
    /// Fixed center node for evaluation.
    ///
    /// When set, greedy center search is skipped. Use this when the caller
    /// already knows where repeated batch structure is concentrated.
    pub center: Option<V>,
    /// Candidate starting centers for greedy automatic center search.
    ///
    /// Empty means all nodes are eligible as starts. Supplying a short list can
    /// reduce center-search overhead for large trees.
    pub initial_centers: Vec<V>,
    /// Maximum number of greedy moves from each initial center.
    ///
    /// `None` means no explicit step limit; the search stops at a local minimum.
    pub max_greedy_steps_per_start: Option<usize>,
    /// Maximum logical payload bytes retained by the persistent message cache.
    ///
    /// A value of `0` disables retention while preserving the same evaluation
    /// results. The default is `usize::MAX`, which preserves the historical
    /// unbounded cache policy; callers that evaluate many changing batches
    /// should set an explicit finite budget or `0`.
    pub message_cache_max_bytes: usize,
}

impl<V> Default for CachedEvaluatorOptions<V> {
    fn default() -> Self {
        Self {
            center: None,
            initial_centers: Vec::new(),
            max_greedy_steps_per_start: None,
            message_cache_max_bytes: usize::MAX,
        }
    }
}

/// Per-call knowledge a caller can supply to `evaluate_batched_with_hint`.
///
/// Non-exhaustive so later hints can be added without breaking callers.
///
/// # Examples
///
/// ```
/// use tensor4all_treetn::EvaluationHint;
///
/// assert!(EvaluationHint::<usize>::default().center.is_none());
/// assert_eq!(EvaluationHint::around(3usize).center, Some(3));
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct EvaluationHint<V> {
    /// The node this batch varies around, if the caller knows it.
    ///
    /// `None` keeps the evaluator's existing centre, chosen from options or by
    /// greedy search on the first batch.
    pub center: Option<V>,
}

impl<V> Default for EvaluationHint<V> {
    /// An empty hint, which leaves the evaluator's centre selection unchanged.
    ///
    /// Written out rather than derived: `derive(Default)` would demand
    /// `V: Default`, but a hint that names no node needs nothing of `V`.
    fn default() -> Self {
        Self { center: None }
    }
}

impl<V> EvaluationHint<V> {
    /// A hint naming the node this batch varies around.
    pub fn around(center: V) -> Self {
        Self {
            center: Some(center),
        }
    }
}

/// Result of greedy center search for cached TreeTN evaluation.
///
/// The result records the selected node, its estimated cost, and the path taken
/// by greedy descent from the chosen start.
///
/// # Examples
///
/// ```
/// use tensor4all_treetn::CenterSearchResult;
///
/// let result = CenterSearchResult {
///     center: 2_usize,
///     cost: 7,
///     path: vec![0, 1, 2],
/// };
/// assert_eq!(result.center, 2);
/// assert_eq!(result.cost, 7);
/// assert_eq!(result.path.last(), Some(&2));
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CenterSearchResult<V> {
    /// Selected center node.
    pub center: V,
    /// Estimated cache cost at `center`.
    pub cost: usize,
    /// Greedy descent path that produced `center`.
    pub path: Vec<V>,
}

/// Greedy local search for TreeTN cached-evaluation centers.
///
/// This type is intentionally separate from [`TreeTNCachedEvaluator`] so future
/// center-selection algorithms can share the same cost model.
///
/// # Examples
///
/// ```
/// use tensor4all_treetn::GreedyCenterSearch;
///
/// let search = GreedyCenterSearch::<usize>::default();
/// assert!(search.max_steps().is_none());
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GreedyCenterSearch<V> {
    max_steps: Option<usize>,
    _marker: std::marker::PhantomData<V>,
}

impl<V> Default for GreedyCenterSearch<V> {
    fn default() -> Self {
        Self {
            max_steps: None,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<V> GreedyCenterSearch<V> {
    /// Creates a greedy center search with an optional step limit.
    ///
    /// `max_steps` limits the number of edge moves from each start. `None`
    /// searches until no neighbor has lower cost.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_treetn::GreedyCenterSearch;
    ///
    /// let search = GreedyCenterSearch::<usize>::with_max_steps(Some(3));
    /// assert_eq!(search.max_steps(), Some(3));
    /// ```
    pub fn with_max_steps(max_steps: Option<usize>) -> Self {
        Self {
            max_steps,
            _marker: std::marker::PhantomData,
        }
    }

    /// Returns the optional greedy-step limit.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_treetn::GreedyCenterSearch;
    ///
    /// let search = GreedyCenterSearch::<usize>::with_max_steps(None);
    /// assert_eq!(search.max_steps(), None);
    /// ```
    pub fn max_steps(&self) -> Option<usize> {
        self.max_steps
    }
}

impl<V> GreedyCenterSearch<V>
where
    V: Clone + Eq + Hash + Ord + Debug + Send + Sync,
{
    fn search(
        &self,
        cost_index: &ComponentCostIndex<V>,
        starts: &[V],
    ) -> Result<CenterSearchResult<V>> {
        let owned_starts;
        let starts = if starts.is_empty() {
            owned_starts = cost_index.all_nodes();
            owned_starts.as_slice()
        } else {
            starts
        };
        if starts.is_empty() {
            bail!("GreedyCenterSearch::search: cost index has no nodes");
        }

        let mut best: Option<CenterSearchResult<V>> = None;
        for start in starts {
            if !cost_index.neighbors.contains_key(start) {
                bail!(
                    "GreedyCenterSearch::search: initial center {:?} is not present in TreeTN",
                    start
                );
            }
            let result = self.descend_from(cost_index, start)?;
            match &best {
                None => best = Some(result),
                Some(current) => {
                    if (result.cost, result.center.clone()) < (current.cost, current.center.clone())
                    {
                        best = Some(result);
                    }
                }
            }
        }

        best.ok_or_else(|| anyhow::anyhow!("GreedyCenterSearch::search: no start centers"))
    }

    fn descend_from(
        &self,
        cost_index: &ComponentCostIndex<V>,
        start: &V,
    ) -> Result<CenterSearchResult<V>> {
        let mut center = start.clone();
        let mut cost = cost_index.center_cost(&center)?;
        let mut path = vec![center.clone()];
        let mut steps = 0usize;

        loop {
            if self.max_steps.is_some_and(|max_steps| steps >= max_steps) {
                break;
            }

            let mut candidates = Vec::new();
            let neighbors = cost_index.neighbors.get(&center).ok_or_else(|| {
                anyhow::anyhow!(
                    "GreedyCenterSearch::descend_from: center {:?} is not present in cost index",
                    center
                )
            })?;
            for neighbor in neighbors {
                candidates.push((cost_index.center_cost(neighbor)?, neighbor.clone()));
            }
            candidates.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
            let Some((next_cost, next_center)) = candidates.into_iter().next() else {
                break;
            };
            if next_cost >= cost {
                break;
            }

            center = next_center;
            cost = next_cost;
            path.push(center.clone());
            steps += 1;
        }

        Ok(CenterSearchResult { center, cost, path })
    }
}

/// Cached batch evaluator for [`TreeTN`].
///
/// Use this when many batch points share repeated assignments on subtrees. It
/// chooses a center node and caches contractions from neighboring components
/// into that center.
///
/// # Examples
///
/// ```
/// use tensor4all_core::{ColMajorArrayRef, DynIndex, IdxTensor};
/// use tensor4all_treetn::{CachedEvaluatorOptions, TreeTN, TreeTNCachedEvaluator};
///
/// let s = DynIndex::new_dyn(2);
/// let tensor = IdxTensor::from_dense(vec![s.clone()], vec![4.0_f64, 6.0])?;
/// let tree = TreeTN::<_, usize>::from_tensors(vec![tensor], vec![0])?;
/// let values = [0usize, 1usize];
/// let shape = [1usize, 2usize];
/// let points = ColMajorArrayRef::new(&values, &shape).unwrap();
///
/// let mut evaluator = TreeTNCachedEvaluator::new(
///     &tree,
///     &[s],
///     CachedEvaluatorOptions::<usize>::default(),
/// )?;
/// let result = evaluator.evaluate_batched(points)?;
/// assert_eq!(result.len(), 2);
/// assert_eq!(result[0].real(), 4.0);
/// assert_eq!(result[1].real(), 6.0);
/// assert_eq!(evaluator.center(), Some(&0));
/// # Ok::<(), anyhow::Error>(())
/// ```
pub struct TreeTNCachedEvaluator<'a, V>
where
    V: Clone + Eq + Hash + Ord + Debug + Send + Sync,
{
    tree: &'a TreeTN<IdxTensor, V>,
    layout: EvaluatorLayout<V>,
    options: CachedEvaluatorOptions<V>,
    center: Option<V>,
    last_stats: CachedEvaluationStats,
    /// Run-scoped, per-directed-edge persistent message cache. Lives as long
    /// as this evaluator: an input-tree evaluator lives for a whole TreeACI
    /// run, so its cache does too; an output-tree evaluator that must be
    /// dropped when pivot injection changes the output is a caller-level
    /// concern (drop this evaluator and build a new one), not this field's.
    /// Keyed by the physical assignments in the node's rooted subtree. This
    /// is the minimal assignment set on which a directed message depends;
    /// changing a site in another component must not invalidate this entry.
    message_caches: HashMap<V, PackedMessageCache<IndexKey, CachedScalar>>,
    /// Each node's parent-bond `DynIndex`, memoized: it never changes for a
    /// fixed rooting, but `TreeTN::edge_between`/`bond_index` are graph
    /// lookups that cost real time if repeated on every call.
    parent_bond_indices: HashMap<V, DynIndex>,
    /// The centre `message_caches`/`parent_bond_indices` were built for.
    ///
    /// `evaluate_batched_with_hint` can pass a *different* centre on every
    /// call (`EvaluationHint::around`, used by `global_guard.rs` to pin the
    /// contraction centre to whichever site a batch varies). A node's
    /// "message toward its parent" means a different neighbour under a
    /// different rooting, so a cache built under one centre is not just
    /// stale but wrong under another -- both caches are cleared whenever the
    /// centre actually used changes.
    rooted_for_center: Option<V>,
    /// Rooting and cache layouts depend only on the fixed centre and tree
    /// topology. Keep them across batches; rebuilding them for every small
    /// floating-zone callback otherwise adds an O(nodes) allocation tax.
    rooted_plan: Option<Arc<RootedMessagePlan<V>>>,
    message_cache_layouts: Option<Arc<HashMap<V, MessageCacheLayout>>>,
    raw_chain_messages: bool,
}

impl<'a, V> TreeTNCachedEvaluator<'a, V>
where
    V: Clone + Eq + Hash + Ord + Debug + Send + Sync,
{
    /// Creates a cached evaluator for `tree` and the requested physical indices.
    ///
    /// If `options.center` is set, that node is used directly. Otherwise, the
    /// first call to [`Self::evaluate_batched`] chooses a center with greedy search
    /// using that batch's repeated-subtree structure.
    ///
    /// # Errors
    ///
    /// Returns an error when the construction or conversion fails (a shape or
    /// /// index mismatch, or a backend failure).
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::{DynIndex, IdxTensor};
    /// use tensor4all_treetn::{CachedEvaluatorOptions, TreeTN, TreeTNCachedEvaluator};
    ///
    /// let s = DynIndex::new_dyn(2);
    /// let tensor = IdxTensor::from_dense(vec![s.clone()], vec![1.0_f64, 2.0])?;
    /// let tree = TreeTN::<_, usize>::from_tensors(vec![tensor], vec![5])?;
    /// let evaluator = TreeTNCachedEvaluator::new(
    ///     &tree,
    ///     &[s],
    ///     CachedEvaluatorOptions { center: Some(5), ..Default::default() },
    /// )?;
    /// assert_eq!(evaluator.center(), Some(&5));
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    pub fn new(
        tree: &'a TreeTN<IdxTensor, V>,
        indices: &[DynIndex],
        options: CachedEvaluatorOptions<V>,
    ) -> std::result::Result<Self, TreeTNOperationError> {
        let layout = build_layout(tree, indices)?;
        if let Some(center) = &options.center {
            ensure_node_exists(tree, center, "TreeTNCachedEvaluator::new: center")?;
        }
        for initial_center in &options.initial_centers {
            ensure_node_exists(
                tree,
                initial_center,
                "TreeTNCachedEvaluator::new: initial center",
            )?;
        }
        let center = options.center.clone();
        Ok(Self {
            tree,
            layout,
            options,
            center,
            last_stats: CachedEvaluationStats::default(),
            message_caches: HashMap::new(),
            parent_bond_indices: HashMap::new(),
            rooted_for_center: None,
            rooted_plan: None,
            message_cache_layouts: None,
            raw_chain_messages: false,
        })
    }

    /// Returns the selected center node, if one has been selected.
    ///
    /// A fixed center is available immediately after [`Self::new`]. An automatic
    /// center is selected during the first [`Self::evaluate_batched`] call.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::{ColMajorArrayRef, DynIndex, IdxTensor};
    /// use tensor4all_treetn::{CachedEvaluatorOptions, TreeTN, TreeTNCachedEvaluator};
    ///
    /// let s = DynIndex::new_dyn(2);
    /// let tensor = IdxTensor::from_dense(vec![s.clone()], vec![1.0_f64, 2.0])?;
    /// let tree = TreeTN::<_, usize>::from_tensors(vec![tensor], vec![0])?;
    /// let mut evaluator = TreeTNCachedEvaluator::new(
    ///     &tree,
    ///     &[s],
    ///     CachedEvaluatorOptions::<usize>::default(),
    /// )?;
    /// assert_eq!(evaluator.center(), None);
    /// let values = [0usize];
    /// let shape = [1usize, 1usize];
    /// let _ = evaluator.evaluate_batched(ColMajorArrayRef::new(&values, &shape).unwrap())?;
    /// assert_eq!(evaluator.center(), Some(&0));
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    pub fn center(&self) -> Option<&V> {
        self.center.as_ref()
    }

    /// Evaluates all batch points using cached subtree environments.
    ///
    /// `values` must have shape `[indices.len(), n_points]` in column-major
    /// layout. The returned vector contains one scalar per column.
    ///
    /// # Errors
    ///
    /// Returns an error when the operation fails (a shape or index mismatch, or
    /// /// a backend failure).
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::{ColMajorArrayRef, DynIndex, IdxTensor};
    /// use tensor4all_treetn::{CachedEvaluatorOptions, TreeTN, TreeTNCachedEvaluator};
    ///
    /// let s = DynIndex::new_dyn(2);
    /// let tensor = IdxTensor::from_dense(vec![s.clone()], vec![4.0_f64, 6.0])?;
    /// let tree = TreeTN::<_, usize>::from_tensors(vec![tensor], vec![0])?;
    /// let values = [0usize, 1usize];
    /// let shape = [1usize, 2usize];
    /// let mut evaluator = TreeTNCachedEvaluator::new(
    ///     &tree,
    ///     &[s],
    ///     CachedEvaluatorOptions::<usize>::default(),
    /// )?;
    /// let result = evaluator.evaluate_batched(ColMajorArrayRef::new(&values, &shape).unwrap())?;
    /// assert_eq!(result.len(), 2);
    /// assert_eq!(result[0].real(), 4.0);
    /// assert_eq!(result[1].real(), 6.0);
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    pub fn evaluate_batched(
        &mut self,
        values: ColMajorArrayRef<'_, usize>,
    ) -> std::result::Result<Vec<AnyScalar>, TreeTNOperationError> {
        self.evaluate_batched_with_hint(values, EvaluationHint::default())
    }

    /// Evaluates a batch, optionally naming the node this batch varies around.
    ///
    /// A caller that scans one site while holding the rest fixed knows which
    /// node that is. Contracting around it makes every incoming message
    /// constant across the batch, so each is contracted once; contracting
    /// around any other node recontracts the messages on the path between them
    /// once per scanned value.
    ///
    /// The hint applies to this call only and does not replace a centre already
    /// chosen by [`CachedEvaluatorOptions::center`] or by greedy search, so a
    /// caller that knows nothing keeps the existing behaviour.
    ///
    /// # Errors
    ///
    /// Returns an error when `values` has the wrong shape for this evaluator's
    /// index list (a shape mismatch), when the hinted node is not in the tree
    /// (a missing-node failure), or when a contraction fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::{ColMajorArrayRef, DynIndex, IdxTensor};
    /// use tensor4all_treetn::{
    ///     CachedEvaluatorOptions, EvaluationHint, TreeTN, TreeTNCachedEvaluator,
    /// };
    ///
    /// let a = DynIndex::new_dyn(2);
    /// let b = DynIndex::new_dyn(2);
    /// let bond = DynIndex::new_dyn(1);
    /// let left = IdxTensor::from_dense(vec![a.clone(), bond.clone()], vec![1.0_f64, 2.0])?;
    /// let right = IdxTensor::from_dense(vec![bond, b.clone()], vec![1.0_f64, 10.0])?;
    /// let tree = TreeTN::from_tensors(vec![left, right], vec![0usize, 1])?;
    ///
    /// let mut evaluator = TreeTNCachedEvaluator::new(
    ///     &tree,
    ///     &[a, b],
    ///     CachedEvaluatorOptions::<usize>::default(),
    /// )?;
    ///
    /// // Scan site 1 with site 0 held fixed.
    /// let values = [0usize, 0, 0, 1];
    /// let points = ColMajorArrayRef::new(&values, &[2, 2])?;
    /// let hinted = evaluator.evaluate_batched_with_hint(points, EvaluationHint::around(1))?;
    ///
    /// assert_eq!(hinted.len(), 2);
    /// assert_eq!(hinted[0].real(), 1.0);
    /// assert_eq!(hinted[1].real(), 10.0);
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    pub fn evaluate_batched_with_hint(
        &mut self,
        values: ColMajorArrayRef<'_, usize>,
        hint: EvaluationHint<V>,
    ) -> std::result::Result<Vec<AnyScalar>, TreeTNOperationError> {
        validate_values_shape(
            values,
            self.layout.n_indices,
            "TreeTNCachedEvaluator::evaluate_batched",
        )?;
        if values.shape()[1] == 0 {
            self.last_stats = CachedEvaluationStats::default();
            return Ok(Vec::new());
        }
        let center = match hint.center {
            Some(node) => {
                if self.tree.node_index(&node).is_none() {
                    return Err(TreeTNOperationError::from(anyhow::anyhow!(
                        "TreeTNCachedEvaluator: hinted centre {:?} is not a node of this tree",
                        node
                    )));
                }
                node
            }
            None => self.ensure_center(values)?.clone(),
        };
        let (component_batches, environment_cache) =
            self.build_environment_cache(&center, values)?;
        let results = self.contract_center_for_points(
            &center,
            values,
            &component_batches,
            &environment_cache,
        )?;
        self.last_stats.batched_center_contract_count = 1;
        Ok(results)
    }

    fn ensure_center(&mut self, values: ColMajorArrayRef<'_, usize>) -> Result<&V> {
        if self.center.is_none() {
            let cost_index = ComponentCostIndex::from_layout(self.tree, &self.layout, values)?;
            let search =
                GreedyCenterSearch::<V>::with_max_steps(self.options.max_greedy_steps_per_start);
            let result = search.search(&cost_index, &self.options.initial_centers)?;
            self.center = Some(result.center);
        }
        self.center.as_ref().ok_or_else(|| {
            anyhow::anyhow!("TreeTNCachedEvaluator::ensure_center: no center selected")
        })
    }

    fn build_environment_cache(
        &mut self,
        center: &V,
        values: ColMajorArrayRef<'_, usize>,
    ) -> Result<CacheBuildResult<V>> {
        if self.rooted_for_center.as_ref() != Some(center) {
            // A node's cached "message toward parent" means a different
            // neighbour under a different rooting, so a cache built for one
            // centre is wrong, not merely stale, once the centre changes.
            self.message_caches.clear();
            self.parent_bond_indices.clear();
            self.rooted_plan = None;
            self.message_cache_layouts = None;
            self.rooted_for_center = Some(center.clone());
        }
        self.last_stats = CachedEvaluationStats::default();
        if self.rooted_plan.is_none() {
            let plan = Arc::new(RootedMessagePlan::new(self.tree, center)?);
            let layouts = Arc::new(self.build_message_cache_layouts(&plan)?);
            self.rooted_plan = Some(plan);
            self.message_cache_layouts = Some(layouts);
        }
        let plan = Arc::clone(
            self.rooted_plan
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("missing rooted message plan"))?,
        );
        let message_cache_layouts = Arc::clone(
            self.message_cache_layouts
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("missing message cache layouts"))?,
        );
        let raw_chain = self.can_use_raw_chain_leaf_center(center)?;
        self.raw_chain_messages = raw_chain;
        let assignment_batches = self.build_message_assignment_batches(&plan, values)?;

        let mut messages = HashMap::<V, StackedMessage>::new();
        let mut directed_message_count = 0usize;
        let mut batched_message_contract_count = 0usize;
        for node in &plan.postorder {
            let assignment_batch = assignment_batches
                .get(node)
                .ok_or_else(|| anyhow::anyhow!("missing assignment batch for node {:?}", node))?;
            directed_message_count += assignment_batch.first_points.len();
            let node_message = self.get_or_compute_node_message(
                node,
                values,
                &plan,
                &message_cache_layouts,
                &assignment_batches,
                &messages,
            )?;
            batched_message_contract_count += 1;
            messages.insert(node.clone(), node_message);
        }

        let mut component_batches = Vec::new();
        let mut cache = HashMap::new();
        let mut subtree_environment_count = 0usize;
        for neighbor in plan.children.get(center).cloned().unwrap_or_default() {
            let assignment_batch = assignment_batches.get(&neighbor).ok_or_else(|| {
                anyhow::anyhow!(
                    "TreeTNCachedEvaluator::evaluate_batched: missing assignments for neighbor {:?}",
                    neighbor
                )
            })?;
            let environment = messages.remove(&neighbor).ok_or_else(|| {
                anyhow::anyhow!(
                    "TreeTNCachedEvaluator::evaluate_batched: missing messages for neighbor {:?}",
                    neighbor
                )
            })?;
            subtree_environment_count += assignment_batch.first_points.len();
            cache.insert(neighbor.clone(), environment);
            component_batches.push(ComponentBatch {
                neighbor,
                point_to_assignment: assignment_batch.point_to_assignment.clone(),
            });
        }
        self.last_stats.subtree_environment_count = subtree_environment_count;
        self.last_stats.directed_message_count = directed_message_count;
        self.last_stats.batched_message_contract_count = batched_message_contract_count;
        Ok((component_batches, cache))
    }

    fn build_message_cache_layouts(
        &self,
        plan: &RootedMessagePlan<V>,
    ) -> Result<HashMap<V, MessageCacheLayout>> {
        let mut layouts = HashMap::with_capacity(plan.subtree_nodes.len());
        for (node, subtree_nodes) in &plan.subtree_nodes {
            let mut entries = subtree_nodes
                .iter()
                .flat_map(|subtree_node| {
                    self.layout
                        .entries_by_node
                        .get(subtree_node)
                        .into_iter()
                        .flatten()
                })
                .collect::<Vec<_>>();
            entries.sort_by_key(|entry| entry.input_position);
            let input_positions = entries
                .iter()
                .map(|entry| entry.input_position)
                .collect::<Vec<_>>();
            let dimensions = entries
                .iter()
                .map(|entry| entry.index.dim())
                .collect::<Vec<_>>();
            let indexer = FlatIndexer::try_new(&dimensions).map_err(anyhow::Error::from)?;
            layouts.insert(
                node.clone(),
                MessageCacheLayout {
                    input_positions,
                    indexer,
                },
            );
        }
        Ok(layouts)
    }

    fn build_message_assignment_batches(
        &self,
        plan: &RootedMessagePlan<V>,
        values: ColMajorArrayRef<'_, usize>,
    ) -> Result<HashMap<V, AssignmentBatch>> {
        let n_points = values.shape()[1];
        let mut local_interner = KeyInterner::<Vec<usize>>::default();
        let mut local_keys = HashMap::<V, Vec<KeyId>>::new();

        let mut node_names = self.tree.node_names();
        node_names.sort();
        for node in &node_names {
            let entries = self
                .layout
                .entries_by_node
                .get(node)
                .map(Vec::as_slice)
                .unwrap_or(&[]);
            let mut keys = Vec::with_capacity(n_points);
            for point in 0..n_points {
                let key = entries
                    .iter()
                    .map(|entry| {
                        value_at(
                            values,
                            entry.input_position,
                            point,
                            "TreeTNCachedEvaluator::evaluate_batched",
                        )
                    })
                    .collect::<Result<Vec<_>>>()?;
                validate_entry_values(entries, &key, "TreeTNCachedEvaluator::evaluate_batched")?;
                keys.push(local_interner.intern(key));
            }
            local_keys.insert(node.clone(), keys);
        }

        let mut assignment_batches = HashMap::<V, AssignmentBatch>::new();
        for node in &plan.postorder {
            let local_keys = local_keys.get(node).ok_or_else(|| {
                anyhow::anyhow!(
                    "TreeTNCachedEvaluator::evaluate_batched: missing local keys for {:?}",
                    node
                )
            })?;
            let children = plan.children.get(node).map(Vec::as_slice).unwrap_or(&[]);
            let child_batches = children
                .iter()
                .map(|child| {
                    assignment_batches.get(child).ok_or_else(|| {
                        anyhow::anyhow!(
                            "TreeTNCachedEvaluator::evaluate_batched: missing child assignments for {:?}",
                            child
                        )
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            let assignment_batch =
                build_compact_assignment_batch(local_keys, &child_batches, n_points);
            assignment_batches.insert(node.clone(), assignment_batch);
        }

        Ok(assignment_batches)
    }

    fn can_use_raw_chain_leaf_center(&self, center: &V) -> Result<bool> {
        let neighbors = sorted_neighbors(self.tree);
        if neighbors.get(center).map(Vec::len) != Some(1) {
            return Ok(false);
        }
        if neighbors.values().any(|neighbors| neighbors.len() > 2) {
            return Ok(false);
        }
        if self
            .layout
            .entries_by_node
            .values()
            .any(|entries| entries.len() != 1)
        {
            return Ok(false);
        }
        let mut scalar_is_complex = None;
        for (node, node_neighbors) in &neighbors {
            let tensor = tensor_for_node(self.tree, node)?;
            if tensor.indices().len() != node_neighbors.len() + 1 {
                return Ok(false);
            }
            let is_complex = tensor.is_complex();
            if let Some(previous) = scalar_is_complex {
                if previous != is_complex {
                    return Ok(false);
                }
            } else {
                scalar_is_complex = Some(is_complex);
            }
        }
        Ok(true)
    }

    fn message_cache_key(
        &self,
        values: ColMajorArrayRef<'_, usize>,
        point: usize,
        cache_layout: &MessageCacheLayout,
    ) -> Result<IndexKey> {
        let raw = cache_layout
            .input_positions
            .iter()
            .map(|&row| {
                value_at(
                    values,
                    row,
                    point,
                    "TreeTNCachedEvaluator::evaluate_batched",
                )
            })
            .collect::<Result<Vec<_>>>()?;
        cache_layout
            .indexer
            .encode(&raw)
            .map_err(anyhow::Error::from)
    }

    /// Computes a leaf node's message directly from the tree tensor's raw
    /// data, bypassing `contract_with_options`/`IdxTensor` entirely.
    ///
    /// Root cause (see `docs/worklogs/2026-08-18-treeaci-message-cache-prototype.md`):
    /// a contraction result is backend-resident/non-contiguous, so reading it
    /// back out via `IdxTensor::to_vec` falls through to an expensive
    /// session-based materialization (measured at 71.3% of miss-path time,
    /// 3x the contraction itself). A leaf node has no contraction to do at
    /// all -- its message is just its own tensor with the physical index
    /// fixed -- so this reads the tensor's already-host-resident data once
    /// and slices it directly, producing a plain `Vec<f64>` with no backend
    /// round-trip.
    ///
    /// Returns `Ok(None)` when `node` is not eligible for this fast path
    /// (more than one physical index, a complex-valued tensor, or a tensor
    /// shape other than the expected 2 axes for a leaf), so the caller can
    /// fall back to [`Self::compute_stacked_message`]. Ineligibility is not
    /// an error: most of this crate's existing tests exercise multi-physical
    /// or branched nodes this first slice does not yet cover.
    fn try_compute_leaf_message_raw(
        &self,
        node: &V,
        values: ColMajorArrayRef<'_, usize>,
        points: &[usize],
    ) -> Result<Option<Vec<f64>>> {
        let entries = self
            .layout
            .entries_by_node
            .get(node)
            .map(Vec::as_slice)
            .unwrap_or(&[]);
        let [entry] = entries else {
            return Ok(None);
        };

        let tensor = tensor_for_node(self.tree, node)?;
        if tensor.is_complex() {
            return Ok(None);
        }
        let tensor_indices = tensor.indices();
        if tensor_indices.len() != 2 {
            return Ok(None);
        }
        let Some(physical_axis) = tensor_indices.iter().position(|idx| idx == &entry.index) else {
            return Ok(None);
        };
        let parent_axis = 1 - physical_axis;

        let dims = tensor.dims();
        let parent_dim = dims[parent_axis];
        // Column-major: stride of axis k is the product of the dims before it.
        let strides = [1usize, dims[0]];

        let raw = tensor.to_vec::<f64>()?;

        let mut out = Vec::with_capacity(parent_dim * points.len());
        for &point in points {
            let physical_value = value_at(
                values,
                entry.input_position,
                point,
                "TreeTNCachedEvaluator::try_compute_leaf_message_raw",
            )?;
            for parent_value in 0..parent_dim {
                let mut axis_values = [0usize; 2];
                axis_values[physical_axis] = physical_value;
                axis_values[parent_axis] = parent_value;
                let flat = axis_values[0] * strides[0] + axis_values[1] * strides[1];
                out.push(raw[flat]);
            }
        }
        Ok(Some(out))
    }

    /// Complex-valued counterpart of [`Self::try_compute_leaf_message_raw`].
    ///
    /// Keeping this path separate from the real-valued helper avoids changing
    /// the established f64 path while allowing SGW's complex tensors to skip
    /// the generic contraction/materialization fallback as well.
    fn try_compute_leaf_message_complex_raw(
        &self,
        node: &V,
        values: ColMajorArrayRef<'_, usize>,
        points: &[usize],
    ) -> Result<Option<Vec<Complex64>>> {
        let entries = self
            .layout
            .entries_by_node
            .get(node)
            .map(Vec::as_slice)
            .unwrap_or(&[]);
        let [entry] = entries else {
            return Ok(None);
        };

        let tensor = tensor_for_node(self.tree, node)?;
        if !tensor.is_complex() {
            return Ok(None);
        }
        let tensor_indices = tensor.indices();
        if tensor_indices.len() != 2 {
            return Ok(None);
        }
        let Some(physical_axis) = tensor_indices.iter().position(|idx| idx == &entry.index) else {
            return Ok(None);
        };
        let parent_axis = 1 - physical_axis;

        let dims = tensor.dims();
        let parent_dim = dims[parent_axis];
        let strides = [1usize, dims[0]];
        let raw = tensor.to_vec::<Complex64>()?;

        let mut out = Vec::with_capacity(parent_dim * points.len());
        for &point in points {
            let physical_value = value_at(
                values,
                entry.input_position,
                point,
                "TreeTNCachedEvaluator::try_compute_leaf_message_complex_raw",
            )?;
            for parent_value in 0..parent_dim {
                let mut axis_values = [0usize; 2];
                axis_values[physical_axis] = physical_value;
                axis_values[parent_axis] = parent_value;
                let flat = axis_values[0] * strides[0] + axis_values[1] * strides[1];
                out.push(raw[flat]);
            }
        }
        Ok(Some(out))
    }

    /// Contracts the parent-message matrices for one chain node in groups of
    /// equal physical values. Large groups use the tensorbackend matrix
    /// multiply; small groups retain the scalar implementation so backend
    /// setup does not dominate low-rank calls.
    fn grouped_chain_message_contraction<T>(
        spec: ChainContractionSpec,
        raw: &[T],
        physical_values: &[usize],
        child_columns: &[T],
    ) -> Result<Vec<T>>
    where
        T: BlasMul + Copy + Default + std::ops::AddAssign + std::ops::Mul<Output = T>,
    {
        let ChainContractionSpec {
            strides,
            physical_axis,
            parent_axis,
            child_axis,
            parent_dim,
            child_dim,
        } = spec;
        let point_count = physical_values.len();
        let expected_child_values = point_count
            .checked_mul(child_dim)
            .ok_or_else(|| anyhow::anyhow!("chain child-message shape overflows usize"))?;
        anyhow::ensure!(
            child_columns.len() == expected_child_values,
            "chain child-message length {} does not match {} points x {} child values",
            child_columns.len(),
            point_count,
            child_dim
        );
        let output_len = point_count
            .checked_mul(parent_dim)
            .ok_or_else(|| anyhow::anyhow!("chain parent-message shape overflows usize"))?;

        if point_count < 2 * CHAIN_BLAS_MIN_GROUP_POINTS {
            return scalar_chain_message_contraction(spec, raw, physical_values, child_columns);
        }

        let mut groups = HashMap::<usize, Vec<usize>>::new();
        for (point, &physical_value) in physical_values.iter().enumerate() {
            groups.entry(physical_value).or_default().push(point);
        }
        let scalar_work = parent_dim
            .checked_mul(child_dim)
            .and_then(|work| work.checked_mul(point_count))
            .ok_or_else(|| anyhow::anyhow!("chain contraction work estimate overflows usize"))?;
        if scalar_work < CHAIN_BLAS_WORK_THRESHOLD
            || groups.len() > 8
            || groups
                .values()
                .any(|points| points.len() < CHAIN_BLAS_MIN_GROUP_POINTS)
        {
            return scalar_chain_message_contraction(spec, raw, physical_values, child_columns);
        }

        let matrix_len = parent_dim
            .checked_mul(child_dim)
            .ok_or_else(|| anyhow::anyhow!("chain matrix shape overflows usize"))?;
        let mut output = vec![T::default(); output_len];
        for (physical_value, points) in groups {
            let mut left = vec![T::default(); matrix_len];
            for child_value in 0..child_dim {
                for parent_value in 0..parent_dim {
                    let mut axis_values = [0usize; 3];
                    axis_values[physical_axis] = physical_value;
                    axis_values[parent_axis] = parent_value;
                    axis_values[child_axis] = child_value;
                    let flat = axis_values[0]
                        .checked_mul(strides[0])
                        .and_then(|value| {
                            value.checked_add(axis_values[1].checked_mul(strides[1])?)
                        })
                        .and_then(|value| {
                            value.checked_add(axis_values[2].checked_mul(strides[2])?)
                        })
                        .ok_or_else(|| anyhow::anyhow!("chain tensor offset overflows usize"))?;
                    let left_offset = parent_dim
                        .checked_mul(child_value)
                        .and_then(|value| value.checked_add(parent_value))
                        .ok_or_else(|| anyhow::anyhow!("chain matrix offset overflows usize"))?;
                    left[left_offset] = *raw.get(flat).ok_or_else(|| {
                        anyhow::anyhow!("chain tensor offset {flat} is out of bounds")
                    })?;
                }
            }

            let right_len = child_dim
                .checked_mul(points.len())
                .ok_or_else(|| anyhow::anyhow!("chain right matrix shape overflows usize"))?;
            let mut right = Vec::with_capacity(right_len);
            for &point in &points {
                let start = point
                    .checked_mul(child_dim)
                    .ok_or_else(|| anyhow::anyhow!("chain child column offset overflows usize"))?;
                let end = start
                    .checked_add(child_dim)
                    .ok_or_else(|| anyhow::anyhow!("chain child column end overflows usize"))?;
                right.extend_from_slice(child_columns.get(start..end).ok_or_else(|| {
                    anyhow::anyhow!("chain child column {start}..{end} is out of bounds")
                })?);
            }

            let product = mat_mul_owned(
                Matrix::from_col_major_vec(parent_dim, child_dim, left),
                Matrix::from_col_major_vec(child_dim, points.len(), right),
            )
            .map_err(anyhow::Error::from)?;
            for (column, &point) in points.iter().enumerate() {
                let destination = point
                    .checked_mul(parent_dim)
                    .ok_or_else(|| anyhow::anyhow!("chain output offset overflows usize"))?;
                let source = column
                    .checked_mul(parent_dim)
                    .ok_or_else(|| anyhow::anyhow!("chain product offset overflows usize"))?;
                let source_end = source
                    .checked_add(parent_dim)
                    .ok_or_else(|| anyhow::anyhow!("chain product end overflows usize"))?;
                let destination_end = destination
                    .checked_add(parent_dim)
                    .ok_or_else(|| anyhow::anyhow!("chain output end overflows usize"))?;
                output[destination..destination_end]
                    .copy_from_slice(&product.as_col_major_slice()[source..source_end]);
            }
        }
        Ok(output)
    }

    /// Computes an interior chain node's (exactly one child) message directly
    /// from raw data, generalizing [`Self::try_compute_leaf_message_raw`] the
    /// way `row_vector_times_matrix`
    /// (`crates/tensor4all-simplett/src/einsum_helper.rs`) generalizes a bare
    /// slice: contract the node's own raw tensor data against the child's
    /// already-computed message column without constructing an `IdxTensor`.
    ///
    /// The child's message must already be present in `messages` (true for
    /// any node reached in postorder) and must itself be real-valued and
    /// `IdxTensor`-backed by already-host-resident data -- true for every
    /// `StackedMessage` this evaluator produces, since both
    /// `IdxTensor::from_dense_any` (the cache-hit path) and a leaf's own
    /// slice-only tensor are host-resident by construction, unlike a
    /// `contract_with_options` result.
    ///
    /// Returns `Ok(None)` when `node` is not eligible (not exactly one
    /// physical index and one child, a complex-valued tensor, or an
    /// unexpected axis count), so the caller falls back to
    /// [`Self::compute_stacked_message`].
    fn try_compute_chain_message_raw(
        &self,
        node: &V,
        values: ColMajorArrayRef<'_, usize>,
        points: &[usize],
        plan: &RootedMessagePlan<V>,
        assignment_batches: &HashMap<V, AssignmentBatch>,
        messages: &HashMap<V, StackedMessage>,
    ) -> Result<Option<Vec<f64>>> {
        let entries = self
            .layout
            .entries_by_node
            .get(node)
            .map(Vec::as_slice)
            .unwrap_or(&[]);
        let [entry] = entries else {
            return Ok(None);
        };
        let children = plan.children.get(node).map(Vec::as_slice).unwrap_or(&[]);
        let [child] = children else {
            return Ok(None);
        };

        let tensor = tensor_for_node(self.tree, node)?;
        if tensor.is_complex() {
            return Ok(None);
        }
        let tensor_indices = tensor.indices();
        if tensor_indices.len() != 3 {
            return Ok(None);
        }
        let Some(physical_axis) = tensor_indices.iter().position(|idx| idx == &entry.index) else {
            return Ok(None);
        };
        let Some(child_edge) = self.tree.edge_between(node, child) else {
            return Ok(None);
        };
        let Some(child_bond_index) = self.tree.bond_index(child_edge) else {
            return Ok(None);
        };
        let Some(child_axis) = tensor_indices
            .iter()
            .position(|idx| idx == child_bond_index)
        else {
            return Ok(None);
        };
        let Some(parent_axis) = (0..3).find(|&axis| axis != physical_axis && axis != child_axis)
        else {
            return Ok(None);
        };

        let child_message = messages.get(child).ok_or_else(|| {
            anyhow::anyhow!(
                "TreeTNCachedEvaluator::try_compute_chain_message_raw: missing message for child {:?}",
                child
            )
        })?;
        let child_values = if let Some(raw_values) = &child_message.raw_values {
            let mut values = Vec::with_capacity(raw_values.len());
            for value in raw_values {
                let CachedScalar::Real(value) = value else {
                    return Ok(None);
                };
                values.push(*value);
            }
            values
        } else {
            let Some(tensor) = child_message.tensor.as_ref() else {
                return Ok(None);
            };
            if tensor.is_complex() {
                return Ok(None);
            }
            tensor.to_vec::<f64>()?
        };
        let child_assignment_batch = assignment_batches.get(child).ok_or_else(|| {
            anyhow::anyhow!(
                "TreeTNCachedEvaluator::try_compute_chain_message_raw: missing assignment batch for child {:?}",
                child
            )
        })?;

        let dims = tensor.dims();
        let parent_dim = dims[parent_axis];
        let child_dim = dims[child_axis];
        let strides = [
            1usize,
            dims[0],
            dims[0]
                .checked_mul(dims[1])
                .ok_or_else(|| anyhow::anyhow!("chain tensor strides overflow usize"))?,
        ];
        let spec = ChainContractionSpec {
            strides,
            physical_axis,
            parent_axis,
            child_axis,
            parent_dim,
            child_dim,
        };
        let raw = tensor.to_vec::<f64>()?;

        let mut physical_values = Vec::with_capacity(points.len());
        let mut child_columns = Vec::with_capacity(child_dim * points.len());
        for &point in points {
            let physical_value = value_at(
                values,
                entry.input_position,
                point,
                "TreeTNCachedEvaluator::try_compute_chain_message_raw",
            )?;
            physical_values.push(physical_value);
            let child_assignment = child_assignment_batch
                .point_to_assignment
                .get(point)
                .copied()
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "TreeTNCachedEvaluator::try_compute_chain_message_raw: missing child assignment for point {point}"
                    )
                })?;
            let child_start = child_assignment
                .checked_mul(child_dim)
                .ok_or_else(|| anyhow::anyhow!("chain child assignment offset overflows usize"))?;
            let child_end = child_start
                .checked_add(child_dim)
                .ok_or_else(|| anyhow::anyhow!("chain child assignment end overflows usize"))?;
            child_columns.extend_from_slice(
                child_values
                    .get(child_start..child_end)
                    .ok_or_else(|| anyhow::anyhow!("chain child assignment is out of bounds"))?,
            );
        }
        let result =
            Self::grouped_chain_message_contraction(spec, &raw, &physical_values, &child_columns)?;
        Ok(Some(result))
    }

    /// Complex-valued counterpart of [`Self::try_compute_chain_message_raw`].
    ///
    /// This preserves the same postorder message and assignment-batch
    /// semantics as the real-valued helper, but performs the contraction with
    /// `Complex64` values directly in host memory.
    fn try_compute_chain_message_complex_raw(
        &self,
        node: &V,
        values: ColMajorArrayRef<'_, usize>,
        points: &[usize],
        plan: &RootedMessagePlan<V>,
        assignment_batches: &HashMap<V, AssignmentBatch>,
        messages: &HashMap<V, StackedMessage>,
    ) -> Result<Option<Vec<Complex64>>> {
        let entries = self
            .layout
            .entries_by_node
            .get(node)
            .map(Vec::as_slice)
            .unwrap_or(&[]);
        let [entry] = entries else {
            return Ok(None);
        };
        let children = plan.children.get(node).map(Vec::as_slice).unwrap_or(&[]);
        let [child] = children else {
            return Ok(None);
        };

        let tensor = tensor_for_node(self.tree, node)?;
        if !tensor.is_complex() {
            return Ok(None);
        }
        let tensor_indices = tensor.indices();
        if tensor_indices.len() != 3 {
            return Ok(None);
        }
        let Some(physical_axis) = tensor_indices.iter().position(|idx| idx == &entry.index) else {
            return Ok(None);
        };
        let Some(child_edge) = self.tree.edge_between(node, child) else {
            return Ok(None);
        };
        let Some(child_bond_index) = self.tree.bond_index(child_edge) else {
            return Ok(None);
        };
        let Some(child_axis) = tensor_indices
            .iter()
            .position(|idx| idx == child_bond_index)
        else {
            return Ok(None);
        };
        let Some(parent_axis) = (0..3).find(|&axis| axis != physical_axis && axis != child_axis)
        else {
            return Ok(None);
        };

        let child_message = messages.get(child).ok_or_else(|| {
            anyhow::anyhow!(
                "TreeTNCachedEvaluator::try_compute_chain_message_complex_raw: missing message for child {:?}",
                child
            )
        })?;
        let child_values = if let Some(raw_values) = &child_message.raw_values {
            let mut values = Vec::with_capacity(raw_values.len());
            for value in raw_values {
                let CachedScalar::Complex(value) = value else {
                    return Ok(None);
                };
                values.push(*value);
            }
            values
        } else {
            let Some(tensor) = child_message.tensor.as_ref() else {
                return Ok(None);
            };
            if !tensor.is_complex() {
                return Ok(None);
            }
            tensor.to_vec::<Complex64>()?
        };
        let child_assignment_batch = assignment_batches.get(child).ok_or_else(|| {
            anyhow::anyhow!(
                "TreeTNCachedEvaluator::try_compute_chain_message_complex_raw: missing assignment batch for child {:?}",
                child
            )
        })?;

        let dims = tensor.dims();
        let parent_dim = dims[parent_axis];
        let child_dim = dims[child_axis];
        let strides = [
            1usize,
            dims[0],
            dims[0]
                .checked_mul(dims[1])
                .ok_or_else(|| anyhow::anyhow!("chain tensor strides overflow usize"))?,
        ];
        let spec = ChainContractionSpec {
            strides,
            physical_axis,
            parent_axis,
            child_axis,
            parent_dim,
            child_dim,
        };
        let raw = tensor.to_vec::<Complex64>()?;

        let mut physical_values = Vec::with_capacity(points.len());
        let mut child_columns = Vec::with_capacity(child_dim * points.len());
        for &point in points {
            let physical_value = value_at(
                values,
                entry.input_position,
                point,
                "TreeTNCachedEvaluator::try_compute_chain_message_complex_raw",
            )?;
            physical_values.push(physical_value);
            let child_assignment = child_assignment_batch
                .point_to_assignment
                .get(point)
                .copied()
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "TreeTNCachedEvaluator::try_compute_chain_message_complex_raw: missing child assignment for point {point}"
                    )
                })?;
            let child_start = child_assignment
                .checked_mul(child_dim)
                .ok_or_else(|| anyhow::anyhow!("chain child assignment offset overflows usize"))?;
            let child_end = child_start
                .checked_add(child_dim)
                .ok_or_else(|| anyhow::anyhow!("chain child assignment end overflows usize"))?;
            child_columns.extend_from_slice(
                child_values
                    .get(child_start..child_end)
                    .ok_or_else(|| anyhow::anyhow!("chain child assignment is out of bounds"))?,
            );
        }
        let result =
            Self::grouped_chain_message_contraction(spec, &raw, &physical_values, &child_columns)?;
        Ok(Some(result))
    }

    /// Computes `node`'s directed message toward its parent, consulting the
    /// per-node persistent cache first.
    ///
    /// Only the assignments genuinely missing from the cache are recomputed
    /// -- via a smaller call to [`Self::compute_stacked_message`] scoped to
    /// just those points -- and the result is merged with the cached columns
    /// for everything else, in the caller's original point order. A node
    /// whose whole batch is already cached skips computation entirely.
    ///
    /// Cache keys contain only the physical assignments in `node`'s rooted
    /// subtree. Sites in another component cannot affect this directed
    /// message and therefore do not belong in its cache key.
    fn get_or_compute_node_message(
        &mut self,
        node: &V,
        values: ColMajorArrayRef<'_, usize>,
        plan: &RootedMessagePlan<V>,
        message_cache_layouts: &HashMap<V, MessageCacheLayout>,
        assignment_batches: &HashMap<V, AssignmentBatch>,
        messages: &HashMap<V, StackedMessage>,
    ) -> Result<StackedMessage> {
        let assignment_batch = assignment_batches.get(node).ok_or_else(|| {
            anyhow::anyhow!(
                "TreeTNCachedEvaluator::evaluate_batched: missing assignments for {:?}",
                node
            )
        })?;
        #[cfg(test)]
        let phase_start = std::time::Instant::now();
        let points = assignment_batch.first_points.clone();
        let cache_layout = message_cache_layouts.get(node).ok_or_else(|| {
            anyhow::anyhow!(
                "TreeTNCachedEvaluator::evaluate_batched: missing message cache layout for {:?}",
                node
            )
        })?;
        let keys = points
            .iter()
            .map(|&point| self.message_cache_key(values, point, cache_layout))
            .collect::<Result<Vec<_>>>()?;

        let Some(Some(parent)) = plan.parent.get(node) else {
            // No parent under this rooting: `node` is not the fixed centre but
            // has none, which should not happen for a postorder entry. Fall
            // back to the uncached path rather than fail the whole call.
            return self.compute_stacked_message(
                node,
                values,
                &points,
                plan,
                assignment_batches,
                messages,
            );
        };
        let parent = parent.clone();
        let bond_index = match self.parent_bond_indices.get(node) {
            Some(index) => index.clone(),
            None => {
                let edge = self.tree.edge_between(node, &parent).ok_or_else(|| {
                    anyhow::anyhow!(
                        "TreeTNCachedEvaluator::evaluate_batched: no edge between {:?} and {:?}",
                        node,
                        parent
                    )
                })?;
                let index = self
                    .tree
                    .bond_index(edge)
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "TreeTNCachedEvaluator::evaluate_batched: missing bond index between {:?} and {:?}",
                            node,
                            parent
                        )
                    })?
                    .clone();
                self.parent_bond_indices.insert(node.clone(), index.clone());
                index
            }
        };
        let bond_dim = bond_index.dim();
        let assignment_index = DynIndex::new_dyn(keys.len());
        let message_cache_max_bytes = self.options.message_cache_max_bytes;

        // Split into hits and misses without computing anything yet. The
        // cache borrow must not outlive this block: `compute_stacked_message`
        // below needs `&self`, which conflicts with a live `&mut
        // self.message_caches` entry.
        let (hit_keys, missing_indices) = {
            let cache = self
                .message_caches
                .entry(node.clone())
                .or_insert_with(|| PackedMessageCache::new(bond_dim, message_cache_max_bytes));
            if let Some(positions) = cache.get_all_cached(&keys) {
                #[cfg(test)]
                phase_timing::add(&phase_timing::KEY_AND_LOOKUP_NS, phase_start.elapsed());
                #[cfg(test)]
                let reconstruct_start = std::time::Instant::now();
                let mut data = Vec::with_capacity(bond_dim * keys.len());
                for position in positions {
                    data.extend_from_slice(cache.column(position));
                }
                let (tensor, raw_values) = if self.raw_chain_messages {
                    (None, Some(data))
                } else {
                    (
                        Some(tensor_from_cached_values(
                            vec![bond_index, assignment_index.clone()],
                            data,
                        )?),
                        None,
                    )
                };
                #[cfg(test)]
                phase_timing::add(&phase_timing::RECONSTRUCT_NS, reconstruct_start.elapsed());
                self.last_stats.message_cache_hits += keys.len();
                return Ok(StackedMessage {
                    assignment_index,
                    tensor,
                    raw_values,
                });
            }
            let mut hit_keys = Vec::new();
            let mut missing_indices = Vec::new();
            for (i, key) in keys.iter().enumerate() {
                if cache.contains(key) {
                    hit_keys.push(key.clone());
                } else {
                    missing_indices.push(i);
                }
            }
            (hit_keys, missing_indices)
        };
        #[cfg(test)]
        phase_timing::add(&phase_timing::KEY_AND_LOOKUP_NS, phase_start.elapsed());
        self.last_stats.message_cache_hits += hit_keys.len();
        self.last_stats.message_cache_misses += missing_indices.len();
        if !hit_keys.is_empty() {
            // `entry().or_insert_with()` rather than `get_mut().expect(...)`:
            // the entry for `node` was inserted above and nothing removes
            // entries from `message_caches` (the only other mutator is
            // `Self::message_caches.clear()`, gated to run once at the top
            // of `build_environment_cache`, before this function's only
            // caller). `or_insert_with`'s closure is never invoked here, so
            // this cannot fail rather than merely being checked not to.
            let cache = self
                .message_caches
                .entry(node.clone())
                .or_insert_with(|| PackedMessageCache::new(bond_dim, message_cache_max_bytes));
            cache.get_all_cached(&hit_keys); // counted above; discard positions here
        }

        // Compute only the missing points, as a batch of just that size.
        let missing_points = missing_indices
            .iter()
            .map(|&i| points[i])
            .collect::<Vec<_>>();
        let missing_keys = missing_indices
            .iter()
            .map(|&i| keys[i].clone())
            .collect::<Vec<_>>();
        #[cfg(test)]
        let contract_start = std::time::Instant::now();
        let tensor_is_complex = tensor_for_node(self.tree, node)?.is_complex();
        let missing_values: Vec<CachedScalar> = if tensor_is_complex {
            let leaf = self.try_compute_leaf_message_complex_raw(node, values, &missing_points)?;
            let raw_missing_values = if leaf.is_some() {
                leaf
            } else {
                self.try_compute_chain_message_complex_raw(
                    node,
                    values,
                    &missing_points,
                    plan,
                    assignment_batches,
                    messages,
                )?
            };
            match raw_missing_values {
                Some(raw) => raw.into_iter().map(CachedScalar::Complex).collect(),
                None => {
                    let missing_message = self.compute_stacked_message(
                        node,
                        values,
                        &missing_points,
                        plan,
                        assignment_batches,
                        messages,
                    )?;
                    tensor_values_any(missing_message.tensor.as_ref().ok_or_else(|| {
                        anyhow::anyhow!("generic message did not materialize a tensor")
                    })?)?
                    .into_iter()
                    .map(CachedScalar::from_any)
                    .collect()
                }
            }
        } else {
            let leaf = self.try_compute_leaf_message_raw(node, values, &missing_points)?;
            let raw_missing_values = if leaf.is_some() {
                leaf
            } else {
                self.try_compute_chain_message_raw(
                    node,
                    values,
                    &missing_points,
                    plan,
                    assignment_batches,
                    messages,
                )?
            };
            match raw_missing_values {
                Some(raw) => raw.into_iter().map(CachedScalar::Real).collect(),
                None => {
                    let missing_message = self.compute_stacked_message(
                        node,
                        values,
                        &missing_points,
                        plan,
                        assignment_batches,
                        messages,
                    )?;
                    tensor_values_any(missing_message.tensor.as_ref().ok_or_else(|| {
                        anyhow::anyhow!("generic message did not materialize a tensor")
                    })?)?
                    .into_iter()
                    .map(CachedScalar::from_any)
                    .collect()
                }
            }
        };
        #[cfg(test)]
        phase_timing::add(&phase_timing::CONTRACT_NS, contract_start.elapsed());

        #[cfg(test)]
        let insert_start = std::time::Instant::now();
        // Same reasoning as the `get_all_cached` call above: the entry for
        // `node` was inserted at the top of this function and nothing
        // between there and here removes it, so `entry().or_insert_with()`
        // cannot fail rather than merely being checked not to.
        let cache = self
            .message_caches
            .entry(node.clone())
            .or_insert_with(|| PackedMessageCache::new(bond_dim, message_cache_max_bytes));
        let missing_slots = cache.get_or_compute_batch(&missing_keys, |request_keys| {
            request_keys
                .iter()
                .map(|request_key| {
                    let index = missing_keys.iter().position(|key| key == request_key).ok_or_else(|| {
                        anyhow::anyhow!(
                            "TreeTNCachedEvaluator::evaluate_batched: missing key not found in this call's batch"
                        )
                    })?;
                    Ok(missing_values[index * bond_dim..(index + 1) * bond_dim].to_vec())
                })
                .collect::<Result<Vec<_>>>()
        })?;
        #[cfg(test)]
        phase_timing::add(&phase_timing::INSERT_NS, insert_start.elapsed());

        #[cfg(test)]
        let reconstruct_start = std::time::Instant::now();
        // Merge cached and uncached columns in the original point order. A
        // finite budget may return `CacheSlot::Uncached`; those values are
        // still valid for this call and must not be looked up again.
        let mut data = Vec::with_capacity(bond_dim * keys.len());
        let mut missing_slot_iter = missing_slots.into_iter();
        let mut missing_index_iter = missing_indices.into_iter().peekable();
        for (point_index, key) in keys.iter().enumerate() {
            if missing_index_iter.peek() == Some(&point_index) {
                missing_index_iter.next();
                let slot = missing_slot_iter.next().ok_or_else(|| {
                    anyhow::anyhow!(
                        "TreeTNCachedEvaluator::evaluate_batched: missing cache slot for missing key"
                    )
                })?;
                match slot {
                    CacheSlot::Cached(position) => data.extend_from_slice(cache.column(position)),
                    CacheSlot::Uncached(column) => {
                        ensure!(
                            column.len() == bond_dim,
                            "TreeTNCachedEvaluator::evaluate_batched: uncached message column has length {}, expected {bond_dim}",
                            column.len()
                        );
                        data.extend_from_slice(&column);
                    }
                }
            } else {
                let position = cache.position(key).ok_or_else(|| {
                    anyhow::anyhow!(
                        "TreeTNCachedEvaluator::evaluate_batched: cached key missing during merge"
                    )
                })?;
                data.extend_from_slice(cache.column(position));
            }
        }
        ensure!(
            missing_index_iter.next().is_none(),
            "TreeTNCachedEvaluator::evaluate_batched: missing point index after merge"
        );
        ensure!(
            missing_slot_iter.next().is_none(),
            "TreeTNCachedEvaluator::evaluate_batched: extra cache slots after merge"
        );
        let (tensor, raw_values) = if self.raw_chain_messages {
            (None, Some(data))
        } else {
            (
                Some(tensor_from_cached_values(
                    vec![bond_index, assignment_index.clone()],
                    data,
                )?),
                None,
            )
        };
        #[cfg(test)]
        phase_timing::add(&phase_timing::RECONSTRUCT_NS, reconstruct_start.elapsed());
        Ok(StackedMessage {
            assignment_index,
            tensor,
            raw_values,
        })
    }

    /// Computes `node`'s directed message for exactly `points` (global point
    /// indices into `values`, in the order the result's assignment axis
    /// should carry) -- not necessarily the node's whole assignment batch, so
    /// a caller with a persistent cache can pass only the points it still
    /// needs to compute.
    fn compute_stacked_message(
        &self,
        node: &V,
        values: ColMajorArrayRef<'_, usize>,
        points: &[usize],
        plan: &RootedMessagePlan<V>,
        assignment_batches: &HashMap<V, AssignmentBatch>,
        messages: &HashMap<V, StackedMessage>,
    ) -> Result<StackedMessage> {
        let assignment_index = DynIndex::new_dyn(points.len());
        let tensor = tensor_for_node(self.tree, node)?;
        let mut local_slices = Vec::with_capacity(points.len());
        for point in points.iter().copied() {
            let index_vals = self.index_vals_for_point(node, values, point)?;
            local_slices.push(slice_tensor(tensor, &index_vals).with_context(|| {
                format!(
                    "TreeTNCachedEvaluator::evaluate_batched: failed to slice message node {:?}",
                    node
                )
            })?);
        }
        let local_message = stack_tensors_with_assignment_index(&assignment_index, &local_slices)
            .with_context(|| {
            format!(
                "TreeTNCachedEvaluator::evaluate_batched: failed to stack message node {:?}",
                node
            )
        })?;

        let children = plan.children.get(node).map(Vec::as_slice).unwrap_or(&[]);
        if children.is_empty() {
            return Ok(StackedMessage {
                assignment_index,
                tensor: Some(local_message),
                raw_values: None,
            });
        }

        let mut operands = Vec::with_capacity(1 + children.len());
        operands.push(local_message);
        for child in children {
            let child_assignment_batch = assignment_batches.get(child).ok_or_else(|| {
                anyhow::anyhow!(
                    "TreeTNCachedEvaluator::evaluate_batched: missing child assignments for {:?}",
                    child
                )
            })?;
            let selected_assignments = points
                .iter()
                .map(|&point| {
                    child_assignment_batch
                        .point_to_assignment
                        .get(point)
                        .copied()
                        .ok_or_else(|| {
                            anyhow::anyhow!("missing child assignment for point {point}")
                        })
                })
                .collect::<Result<Vec<_>>>()?;
            let child_message = messages.get(child).ok_or_else(|| {
                anyhow::anyhow!(
                    "TreeTNCachedEvaluator::evaluate_batched: missing child message for {:?}",
                    child
                )
            })?;
            operands.push(gather_stacked_tensor(
                child_message
                    .tensor
                    .as_ref()
                    .ok_or_else(|| anyhow::anyhow!("child message did not materialize a tensor"))?,
                &child_message.assignment_index,
                &assignment_index,
                &selected_assignments,
            )?);
        }

        let retain = [assignment_index.clone()];
        let options = ContractionOptions::new().with_retain_indices(&retain);
        let operand_refs = operands.iter().collect::<Vec<_>>();
        let tensor = contract_with_options(&operand_refs, options).context(
            "TreeTNCachedEvaluator::evaluate_batched: failed to contract batched directed message",
        )?;
        let tensor = ensure_assignment_axis_last(tensor, &assignment_index)?;
        Ok(StackedMessage {
            assignment_index,
            tensor: Some(tensor),
            raw_values: None,
        })
    }

    fn try_contract_leaf_center_from_raw(
        &self,
        center: &V,
        values: ColMajorArrayRef<'_, usize>,
        component: &ComponentBatch<V>,
        environment: &StackedMessage,
    ) -> Result<Option<Vec<AnyScalar>>> {
        let Some(raw_values) = environment.raw_values.as_ref() else {
            return Ok(None);
        };
        let entries = self
            .layout
            .entries_by_node
            .get(center)
            .map(Vec::as_slice)
            .unwrap_or(&[]);
        let [entry] = entries else {
            return Ok(None);
        };
        let center_tensor = tensor_for_node(self.tree, center)?;
        let center_indices = center_tensor.indices();
        if center_indices.len() != 2 {
            return Ok(None);
        }
        let Some(physical_axis) = center_indices
            .iter()
            .position(|index| index == &entry.index)
        else {
            return Ok(None);
        };
        let bond_axis = 1 - physical_axis;
        let bond_dim = center_tensor.dims()[bond_axis];
        if bond_dim == 0 || raw_values.len() % bond_dim != 0 {
            return Ok(None);
        }
        let assignment_dim = raw_values.len() / bond_dim;
        let center_dims = center_tensor.dims();
        let center_strides = [1usize, center_dims[0]];
        let n_points = values.shape()[1];

        if center_tensor.is_complex() {
            let center_raw = center_tensor.to_vec::<Complex64>()?;
            let mut result = Vec::with_capacity(n_points);
            for point in 0..n_points {
                let physical_value = value_at(
                    values,
                    entry.input_position,
                    point,
                    "TreeTNCachedEvaluator::try_contract_leaf_center_from_raw",
                )?;
                let assignment = component
                    .point_to_assignment
                    .get(point)
                    .copied()
                    .ok_or_else(|| {
                        anyhow::anyhow!("missing centre assignment for point {point}")
                    })?;
                ensure!(
                    assignment < assignment_dim,
                    "centre assignment {assignment} is out of bounds for dimension {assignment_dim}"
                );
                let mut sum = Complex64::new(0.0, 0.0);
                for bond in 0..bond_dim {
                    let center_offset = physical_value * center_strides[physical_axis]
                        + bond * center_strides[bond_axis];
                    let environment_offset = assignment * bond_dim + bond;
                    let CachedScalar::Complex(environment_value) = raw_values[environment_offset]
                    else {
                        return Ok(None);
                    };
                    sum += center_raw[center_offset] * environment_value;
                }
                result.push(AnyScalar::new_complex(sum.re, sum.im));
            }
            return Ok(Some(result));
        }

        let center_raw = center_tensor.to_vec::<f64>()?;
        let mut result = Vec::with_capacity(n_points);
        for point in 0..n_points {
            let physical_value = value_at(
                values,
                entry.input_position,
                point,
                "TreeTNCachedEvaluator::try_contract_leaf_center_from_raw",
            )?;
            let assignment = component
                .point_to_assignment
                .get(point)
                .copied()
                .ok_or_else(|| anyhow::anyhow!("missing centre assignment for point {point}"))?;
            ensure!(
                assignment < assignment_dim,
                "centre assignment {assignment} is out of bounds for dimension {assignment_dim}"
            );
            let mut sum = 0.0;
            for bond in 0..bond_dim {
                let center_offset = physical_value * center_strides[physical_axis]
                    + bond * center_strides[bond_axis];
                let environment_offset = assignment * bond_dim + bond;
                let CachedScalar::Real(environment_value) = raw_values[environment_offset] else {
                    return Ok(None);
                };
                sum += center_raw[center_offset] * environment_value;
            }
            result.push(AnyScalar::new_real(sum));
        }
        Ok(Some(result))
    }

    /// Evaluates a one-component centre on a path without constructing a
    /// backend contraction. A leaf centre has one physical axis and one bond
    /// axis, so each point is just a dot product of the sliced centre row with
    /// the cached incoming message column. Branching centres and nonstandard
    /// tensor layouts return `None` and retain the generic contraction path.
    fn try_contract_leaf_center_raw(
        &self,
        center: &V,
        values: ColMajorArrayRef<'_, usize>,
        component_batches: &[ComponentBatch<V>],
        environment_cache: &EnvironmentCache<V>,
    ) -> Result<Option<Vec<AnyScalar>>> {
        let [component] = component_batches else {
            return Ok(None);
        };
        let entries = self
            .layout
            .entries_by_node
            .get(center)
            .map(Vec::as_slice)
            .unwrap_or(&[]);
        let [entry] = entries else {
            return Ok(None);
        };
        let center_tensor = tensor_for_node(self.tree, center)?;
        let environment = environment_cache.get(&component.neighbor).ok_or_else(|| {
            anyhow::anyhow!("TreeTNCachedEvaluator::evaluate_batched: missing cached environment")
        })?;
        if environment.tensor.is_none() {
            return self.try_contract_leaf_center_from_raw(center, values, component, environment);
        }
        let center_indices = center_tensor.indices();
        let Some(environment_tensor) = environment.tensor.as_ref() else {
            return Ok(None);
        };
        let environment_indices = environment_tensor.indices();
        if center_indices.len() != 2 || environment_indices.len() != 2 {
            return Ok(None);
        }
        let Some(physical_axis) = center_indices
            .iter()
            .position(|index| index == &entry.index)
        else {
            return Ok(None);
        };
        let bond_axis = 1 - physical_axis;
        let bond_index = &center_indices[bond_axis];
        let Some(environment_bond_axis) = environment_indices
            .iter()
            .position(|index| index == bond_index)
        else {
            return Ok(None);
        };
        let assignment_axis = 1 - environment_bond_axis;
        let center_dims = center_tensor.dims();
        let environment_dims = environment_tensor.dims();
        let bond_dim = center_dims[bond_axis];
        if environment_dims[environment_bond_axis] != bond_dim {
            return Ok(None);
        }
        let assignment_dim = environment_dims[assignment_axis];
        let n_points = values.shape()[1];
        let center_strides = [1usize, center_dims[0]];
        let environment_strides = [1usize, environment_dims[0]];
        let physical_position = entry.input_position;

        if center_tensor.is_complex() != environment_tensor.is_complex() {
            return Ok(None);
        }
        if center_tensor.is_complex() {
            let center_raw = center_tensor.to_vec::<Complex64>()?;
            let environment_raw = environment_tensor.to_vec::<Complex64>()?;
            let mut result = Vec::with_capacity(n_points);
            for point in 0..n_points {
                let physical_value = value_at(
                    values,
                    physical_position,
                    point,
                    "TreeTNCachedEvaluator::try_contract_leaf_center_raw",
                )?;
                let assignment = component
                    .point_to_assignment
                    .get(point)
                    .copied()
                    .ok_or_else(|| {
                        anyhow::anyhow!("missing centre assignment for point {point}")
                    })?;
                ensure!(
                    assignment < assignment_dim,
                    "centre assignment {assignment} is out of bounds for dimension {assignment_dim}"
                );
                let mut sum = Complex64::new(0.0, 0.0);
                for bond in 0..bond_dim {
                    let center_offset = physical_value * center_strides[physical_axis]
                        + bond * center_strides[bond_axis];
                    let environment_offset = bond * environment_strides[environment_bond_axis]
                        + assignment * environment_strides[assignment_axis];
                    sum += center_raw[center_offset] * environment_raw[environment_offset];
                }
                result.push(AnyScalar::new_complex(sum.re, sum.im));
            }
            return Ok(Some(result));
        }

        let center_raw = center_tensor.to_vec::<f64>()?;
        let environment_raw = environment_tensor.to_vec::<f64>()?;
        let mut result = Vec::with_capacity(n_points);
        for point in 0..n_points {
            let physical_value = value_at(
                values,
                physical_position,
                point,
                "TreeTNCachedEvaluator::try_contract_leaf_center_raw",
            )?;
            let assignment = component
                .point_to_assignment
                .get(point)
                .copied()
                .ok_or_else(|| anyhow::anyhow!("missing centre assignment for point {point}"))?;
            ensure!(
                assignment < assignment_dim,
                "centre assignment {assignment} is out of bounds for dimension {assignment_dim}"
            );
            let mut sum = 0.0;
            for bond in 0..bond_dim {
                let center_offset = physical_value * center_strides[physical_axis]
                    + bond * center_strides[bond_axis];
                let environment_offset = bond * environment_strides[environment_bond_axis]
                    + assignment * environment_strides[assignment_axis];
                sum += center_raw[center_offset] * environment_raw[environment_offset];
            }
            result.push(AnyScalar::new_real(sum));
        }
        Ok(Some(result))
    }

    fn contract_center_for_points(
        &self,
        center: &V,
        values: ColMajorArrayRef<'_, usize>,
        component_batches: &[ComponentBatch<V>],
        environment_cache: &EnvironmentCache<V>,
    ) -> Result<Vec<AnyScalar>> {
        let n_points = values.shape()[1];
        if n_points == 0 {
            return Ok(Vec::new());
        }
        if let Some(result) =
            self.try_contract_leaf_center_raw(center, values, component_batches, environment_cache)?
        {
            return Ok(result);
        }
        let center_entries = self
            .layout
            .entries_by_node
            .get(center)
            .map(Vec::as_slice)
            .unwrap_or(&[]);
        let center_tensor = tensor_for_node(self.tree, center)?;
        let point_index = DynIndex::new_dyn(n_points);
        let mut center_slices = Vec::with_capacity(n_points);
        for point in 0..n_points {
            let center_index_vals = center_entries
                .iter()
                .map(|entry| {
                    let value = value_at(
                        values,
                        entry.input_position,
                        point,
                        "TreeTNCachedEvaluator::evaluate_batched",
                    )?;
                    Ok((entry.index.clone(), value))
                })
                .collect::<Result<Vec<_>>>()?;
            validate_index_vals(
                &center_index_vals,
                "TreeTNCachedEvaluator::evaluate_batched",
            )?;
            center_slices.push(slice_tensor(center_tensor, &center_index_vals).context(
                "TreeTNCachedEvaluator::evaluate_batched: failed to slice center tensor",
            )?);
        }

        let mut operands = Vec::with_capacity(1 + component_batches.len());
        operands.push(
            stack_tensors_with_assignment_index(&point_index, &center_slices).context(
                "TreeTNCachedEvaluator::evaluate_batched: failed to stack center tensor",
            )?,
        );

        for batch in component_batches {
            let environment = environment_cache.get(&batch.neighbor).ok_or_else(|| {
                anyhow::anyhow!(
                    "TreeTNCachedEvaluator::evaluate_batched: missing cached environment"
                )
            })?;
            operands.push(
                gather_stacked_tensor(
                    environment.tensor.as_ref().ok_or_else(|| {
                        anyhow::anyhow!(
                            "TreeTNCachedEvaluator::evaluate_batched: environment did not materialize a tensor"
                        )
                    })?,
                    &environment.assignment_index,
                    &point_index,
                    &batch.point_to_assignment,
                )
                .context(
                    "TreeTNCachedEvaluator::evaluate_batched: failed to gather center environment",
                )?,
            );
        }

        let result_tensor = if operands.len() == 1 {
            operands.remove(0)
        } else {
            let retain = [point_index.clone()];
            let options = ContractionOptions::new().with_retain_indices(&retain);
            let operand_refs = operands.iter().collect::<Vec<_>>();
            contract_with_options(&operand_refs, options).context(
                "TreeTNCachedEvaluator::evaluate_batched: failed to contract center batch",
            )?
        };
        let result_tensor = ensure_assignment_axis_last(result_tensor, &point_index)?;
        anyhow::ensure!(
            result_tensor.indices() == std::slice::from_ref(&point_index),
            "TreeTNCachedEvaluator::evaluate_batched: center contraction left non-scalar indices {:?}",
            result_tensor.indices()
        );

        tensor_values_any(&result_tensor)
    }

    fn index_vals_for_point(
        &self,
        node: &V,
        values: ColMajorArrayRef<'_, usize>,
        point: usize,
    ) -> Result<Vec<(DynIndex, usize)>> {
        let Some(entries) = self.layout.entries_by_node.get(node) else {
            return Ok(Vec::new());
        };
        entries
            .iter()
            .map(|entry| {
                let value = value_at(
                    values,
                    entry.input_position,
                    point,
                    "TreeTNCachedEvaluator::evaluate_batched",
                )?;
                Ok((entry.index.clone(), value))
            })
            .collect()
    }

    #[cfg(test)]
    fn stats_for_test(&self) -> CachedEvaluationStats {
        self.last_stats.clone()
    }
}

fn tensor_from_cached_values(
    indices: Vec<DynIndex>,
    values: Vec<CachedScalar>,
) -> Result<IdxTensor> {
    if let Some(data) = values
        .iter()
        .map(|value| match value {
            CachedScalar::Real(value) => Some(*value),
            CachedScalar::Complex(_) => None,
        })
        .collect::<Option<Vec<_>>>()
    {
        return Ok(IdxTensor::from_dense(indices, data)?);
    }
    if let Some(data) = values
        .iter()
        .map(|value| match value {
            CachedScalar::Complex(value) => Some(*value),
            CachedScalar::Real(_) => None,
        })
        .collect::<Option<Vec<_>>>()
    {
        return Ok(IdxTensor::from_dense(indices, data)?);
    }
    Ok(IdxTensor::from_dense_any(
        indices,
        values.into_iter().map(CachedScalar::into_any).collect(),
    )?)
}

fn scalar_chain_message_contraction<T>(
    spec: ChainContractionSpec,
    raw: &[T],
    physical_values: &[usize],
    child_columns: &[T],
) -> Result<Vec<T>>
where
    T: Copy + Default + std::ops::AddAssign + std::ops::Mul<Output = T>,
{
    let ChainContractionSpec {
        strides,
        physical_axis,
        parent_axis,
        child_axis,
        parent_dim,
        child_dim,
    } = spec;
    let point_count = physical_values.len();
    let output_len = point_count
        .checked_mul(parent_dim)
        .ok_or_else(|| anyhow::anyhow!("chain parent-message shape overflows usize"))?;
    let mut output = vec![T::default(); output_len];
    for (point, &physical_value) in physical_values.iter().enumerate() {
        for parent_value in 0..parent_dim {
            let mut sum = T::default();
            for child_value in 0..child_dim {
                let mut axis_values = [0usize; 3];
                axis_values[physical_axis] = physical_value;
                axis_values[parent_axis] = parent_value;
                axis_values[child_axis] = child_value;
                let flat = axis_values[0]
                    .checked_mul(strides[0])
                    .and_then(|value| value.checked_add(axis_values[1].checked_mul(strides[1])?))
                    .and_then(|value| value.checked_add(axis_values[2].checked_mul(strides[2])?))
                    .ok_or_else(|| anyhow::anyhow!("chain tensor offset overflows usize"))?;
                let child_offset = point
                    .checked_mul(child_dim)
                    .and_then(|value| value.checked_add(child_value))
                    .ok_or_else(|| anyhow::anyhow!("chain child offset overflows usize"))?;
                sum += *raw
                    .get(flat)
                    .ok_or_else(|| anyhow::anyhow!("chain tensor offset is out of bounds"))?
                    * *child_columns.get(child_offset).ok_or_else(|| {
                        anyhow::anyhow!("chain child column offset is out of bounds")
                    })?;
            }
            let destination = point
                .checked_mul(parent_dim)
                .and_then(|value| value.checked_add(parent_value))
                .ok_or_else(|| anyhow::anyhow!("chain output offset overflows usize"))?;
            output[destination] = sum;
        }
    }
    Ok(output)
}

fn build_layout<V>(tree: &TreeTN<IdxTensor, V>, indices: &[DynIndex]) -> Result<EvaluatorLayout<V>>
where
    V: Clone + Eq + Hash + Ord + Debug + Send + Sync,
{
    if tree.node_count() == 0 {
        bail!("TreeTNCachedEvaluator::new: network must have at least one node");
    }

    let total_site_indices = tree.site_index_network().site_index_count();
    anyhow::ensure!(
        indices.len() == total_site_indices,
        "TreeTNCachedEvaluator::new: indices.len() ({}) != total site indices ({})",
        indices.len(),
        total_site_indices
    );

    let mut seen = HashSet::with_capacity(indices.len());
    for index in indices {
        anyhow::ensure!(
            seen.insert(index.clone()),
            "TreeTNCachedEvaluator::new: duplicate index {:?}",
            index
        );
    }

    let mut entries_by_node: HashMap<V, Vec<SiteEntry>> = HashMap::new();
    let mut tensor_indices_by_node: HashMap<V, Vec<DynIndex>> = HashMap::new();
    for (input_position, index) in indices.iter().enumerate() {
        let node_name = tree
            .site_index_network()
            .find_node_by_index(index)
            .ok_or_else(|| {
                anyhow::anyhow!("TreeTNCachedEvaluator::new: unknown index {:?}", index)
            })?
            .clone();
        let tensor = tensor_for_node(tree, &node_name)?;
        let tensor_indices = tensor_indices_by_node
            .entry(node_name.clone())
            .or_insert_with(|| tensor.external_indices());
        let local_axis = tensor_indices
            .iter()
            .position(|tensor_index| tensor_index == index)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "TreeTNCachedEvaluator::new: site index {:?} is registered on node {:?} but not present in its tensor",
                    index,
                    node_name
                )
            })?;
        entries_by_node
            .entry(node_name)
            .or_default()
            .push(SiteEntry {
                index: index.clone(),
                input_position,
                local_axis,
            });
    }

    for entries in entries_by_node.values_mut() {
        entries.sort_by_key(|entry| entry.local_axis);
    }

    Ok(EvaluatorLayout {
        entries_by_node,
        n_indices: indices.len(),
    })
}

fn build_compact_assignment_batch(
    local_keys: &[KeyId],
    child_batches: &[&AssignmentBatch],
    n_points: usize,
) -> AssignmentBatch {
    let mut assignment_ids = HashMap::<Vec<KeyId>, usize>::new();
    let mut first_points = Vec::<usize>::new();
    let mut point_to_assignment = Vec::with_capacity(n_points);
    for (point, &local_key) in local_keys.iter().enumerate().take(n_points) {
        let mut assignment = Vec::with_capacity(1 + child_batches.len());
        assignment.push(local_key);
        for child_batch in child_batches {
            assignment.push(child_batch.point_to_assignment[point]);
        }

        let assignment_id = if let Some(&assignment_id) = assignment_ids.get(&assignment) {
            assignment_id
        } else {
            let assignment_id = assignment_ids.len();
            assignment_ids.insert(assignment, assignment_id);
            first_points.push(point);
            assignment_id
        };
        point_to_assignment.push(assignment_id);
    }

    AssignmentBatch {
        point_to_assignment,
        first_points,
    }
}

fn validate_values_shape(
    values: ColMajorArrayRef<'_, usize>,
    n_indices: usize,
    context: &str,
) -> Result<()> {
    anyhow::ensure!(
        values.shape().len() == 2,
        "{context}: values must be 2D, got {}D",
        values.shape().len()
    );
    anyhow::ensure!(
        values.shape()[0] == n_indices,
        "{context}: row count {} does not match indices.len() {}",
        values.shape()[0],
        n_indices
    );
    Ok(())
}

fn value_at(
    values: ColMajorArrayRef<'_, usize>,
    input_position: usize,
    point: usize,
    context: &str,
) -> Result<usize> {
    values
        .get(&[input_position, point])
        .copied()
        .ok_or_else(|| {
            anyhow::anyhow!(
                "{context}: missing coordinate at row {} point {} for shape {:?}",
                input_position,
                point,
                values.shape()
            )
        })
}

fn validate_entry_values(entries: &[SiteEntry], values: &[usize], context: &str) -> Result<()> {
    let index_vals = entries
        .iter()
        .zip(values.iter().copied())
        .map(|(entry, value)| (entry.index.clone(), value))
        .collect::<Vec<_>>();
    validate_index_vals(&index_vals, context)
}

fn validate_index_vals<I>(index_vals: &[(I, usize)], context: &str) -> Result<()>
where
    I: IndexLike,
{
    for (index, value) in index_vals {
        anyhow::ensure!(
            *value < index.dim(),
            "{context}: coordinate {} is out of range for index {:?} with dim {}",
            value,
            index,
            index.dim()
        );
    }
    Ok(())
}

fn ensure_node_exists<V>(tree: &TreeTN<IdxTensor, V>, node: &V, context: &str) -> Result<()>
where
    V: Clone + Eq + Hash + Debug + Send + Sync,
{
    if tree.node_index(node).is_none() {
        bail!("{context} {:?} is not present in TreeTN", node);
    }
    Ok(())
}

fn tensor_for_node<'a, V>(tree: &'a TreeTN<IdxTensor, V>, node: &V) -> Result<&'a IdxTensor>
where
    V: Clone + Eq + Hash + Debug + Send + Sync,
{
    let node_idx = tree
        .node_index(node)
        .ok_or_else(|| anyhow::anyhow!("node {:?} is not present in TreeTN", node))?;
    tree.tensor(node_idx)
        .ok_or_else(|| anyhow::anyhow!("tensor for node {:?} is not present", node))
}

fn slice_tensor(tensor: &IdxTensor, index_vals: &[(DynIndex, usize)]) -> Result<IdxTensor> {
    if index_vals.is_empty() {
        return Ok(tensor.clone());
    }
    validate_index_vals(index_vals, "slice_tensor")?;
    let selected_indices = index_vals
        .iter()
        .map(|(index, _)| index.clone())
        .collect::<Vec<_>>();
    let positions = index_vals
        .iter()
        .map(|(_, position)| *position)
        .collect::<Vec<_>>();
    tensor
        .select_indices(&selected_indices, &positions)
        .map_err(anyhow::Error::from)
}

fn tensor_values_any(tensor: &IdxTensor) -> Result<Vec<AnyScalar>> {
    if tensor.is_complex() {
        tensor
            .to_vec::<Complex64>()
            .map(|values| {
                values
                    .into_iter()
                    .map(|value| AnyScalar::new_complex(value.re, value.im))
                    .collect()
            })
            .map_err(anyhow::Error::from)
    } else {
        tensor
            .to_vec::<f64>()
            .map(|values| values.into_iter().map(AnyScalar::new_real).collect())
            .map_err(anyhow::Error::from)
    }
}

fn stack_tensors_with_assignment_index(
    assignment_index: &DynIndex,
    tensors: &[IdxTensor],
) -> Result<IdxTensor> {
    anyhow::ensure!(
        !tensors.is_empty(),
        "stack_tensors_with_assignment_index requires at least one tensor"
    );
    anyhow::ensure!(
        assignment_index.dim() == tensors.len(),
        "assignment index dim {} does not match tensor count {}",
        assignment_index.dim(),
        tensors.len()
    );

    let tensor_refs = tensors.iter().collect::<Vec<_>>();
    IdxTensor::stack_along_new_index(&tensor_refs, assignment_index.clone(), -1)
        .map_err(anyhow::Error::from)
}

fn gather_stacked_tensor(
    stacked: &IdxTensor,
    source_assignment_index: &DynIndex,
    target_assignment_index: &DynIndex,
    selected_assignments: &[usize],
) -> Result<IdxTensor> {
    anyhow::ensure!(
        stacked.indices().last() == Some(source_assignment_index),
        "source assignment index must be the last stacked axis"
    );
    anyhow::ensure!(
        selected_assignments.len() == target_assignment_index.dim(),
        "selected assignment count {} does not match target assignment dim {}",
        selected_assignments.len(),
        target_assignment_index.dim()
    );

    stacked
        .index_select(
            source_assignment_index,
            target_assignment_index.clone(),
            selected_assignments,
        )
        .map_err(anyhow::Error::from)
}

fn ensure_assignment_axis_last(
    tensor: IdxTensor,
    assignment_index: &DynIndex,
) -> Result<IdxTensor> {
    if tensor.indices().last() == Some(assignment_index) {
        return Ok(tensor);
    }
    anyhow::ensure!(
        tensor.indices().contains(assignment_index),
        "batched contraction result is missing assignment index {:?}",
        assignment_index
    );
    let mut new_order = Vec::with_capacity(tensor.indices().len());
    new_order.extend(
        tensor
            .indices()
            .iter()
            .filter(|index| *index != assignment_index)
            .cloned(),
    );
    new_order.push(assignment_index.clone());
    tensor.permuteinds(&new_order).map_err(anyhow::Error::new)
}

fn sorted_neighbors<T, V>(tree: &TreeTN<T, V>) -> HashMap<V, Vec<V>>
where
    T: TensorLike,
    V: Clone + Eq + Hash + Ord + Debug + Send + Sync,
{
    let mut map = HashMap::new();
    let mut node_names = tree.node_names();
    node_names.sort();
    for node in node_names {
        let mut neighbors: Vec<V> = tree.site_index_network().neighbors(&node).collect();
        neighbors.sort();
        map.insert(node, neighbors);
    }
    map
}

fn rooted_tree<V>(neighbors: &HashMap<V, Vec<V>>, root: &V) -> Result<(ParentMap<V>, Vec<V>)>
where
    V: Clone + Eq + Hash + Ord + Debug,
{
    let mut parent = HashMap::<V, Option<V>>::new();
    let mut order = Vec::<V>::new();
    let mut stack = vec![(root.clone(), None)];
    while let Some((node, parent_node)) = stack.pop() {
        if parent.contains_key(&node) {
            continue;
        }
        parent.insert(node.clone(), parent_node.clone());
        order.push(node.clone());
        let mut children = neighbors
            .get(&node)
            .ok_or_else(|| anyhow::anyhow!("node {:?} is missing from neighbor map", node))?
            .iter()
            .filter(|neighbor| Some(*neighbor) != parent_node.as_ref())
            .cloned()
            .collect::<Vec<_>>();
        children.sort_by(|a, b| b.cmp(a));
        for child in children {
            stack.push((child, Some(node.clone())));
        }
    }

    anyhow::ensure!(
        parent.len() == neighbors.len(),
        "TreeTN topology is disconnected: reached {} of {} nodes",
        parent.len(),
        neighbors.len()
    );
    Ok((parent, order))
}

/// A run-scoped, append-only cache of packed message columns for one
/// directed edge.
///
/// Per Hiroshi's #646 review design: entries are never evicted individually
/// and there is no public `clear` -- the cache lives no longer than the
/// evaluator that owns it, so nothing outside can hold a handle that would
/// need clearing. Columns are stored contiguously in a single flat buffer
/// (column-major: column `i` occupies `columns[i * bond_dim .. (i+1) *
/// bond_dim]`) instead of one heap allocation per message.
struct PackedMessageCache<K, T> {
    bond_dim: usize,
    max_bytes: usize,
    positions: HashMap<K, usize>,
    columns: Vec<T>,
    hits: usize,
    misses: usize,
}

/// Where one requested key's column ended up.
///
/// `Uncached` carries the computed values directly rather than a position,
/// since an over-budget entry is never written into `columns` -- there is
/// nothing in the packed buffer to point at.
#[derive(Debug, Clone, PartialEq)]
enum CacheSlot<T> {
    Cached(usize),
    Uncached(Vec<T>),
}

impl<K, T> PackedMessageCache<K, T>
where
    K: Eq + Hash + Clone,
    T: Clone,
{
    fn new(bond_dim: usize, max_bytes: usize) -> Self {
        Self {
            bond_dim,
            max_bytes,
            positions: HashMap::new(),
            columns: Vec::new(),
            hits: 0,
            misses: 0,
        }
    }

    fn column(&self, position: usize) -> &[T] {
        &self.columns[position * self.bond_dim..(position + 1) * self.bond_dim]
    }

    /// Looks up every key without computing anything.
    ///
    /// Returns `Some(positions)`, one per key in order, and counts each as a
    /// hit, only if every key is already cached; otherwise returns `None`
    /// without touching the hit/miss counters, so a caller that falls back to
    /// [`Self::get_or_compute_batch`] on a partial hit does not double-count.
    fn get_all_cached(&mut self, keys: &[K]) -> Option<Vec<usize>> {
        let positions = keys
            .iter()
            .map(|key| self.positions.get(key).copied())
            .collect::<Option<Vec<_>>>()?;
        self.hits += keys.len();
        Some(positions)
    }

    fn contains(&self, key: &K) -> bool {
        self.positions.contains_key(key)
    }

    /// Looks up one key's column position without touching the hit/miss
    /// counters -- for reassembling a merged result after the counted
    /// lookup/insert calls that drove the merge have already run.
    fn position(&self, key: &K) -> Option<usize> {
        self.positions.get(key).copied()
    }

    fn retained_bytes(&self) -> usize {
        self.columns.len() * std::mem::size_of::<T>()
    }

    fn hits(&self) -> usize {
        self.hits
    }

    fn misses(&self) -> usize {
        self.misses
    }

    /// Returns where each requested key's column ended up, in order.
    ///
    /// Follows Hiroshi's seven-step batch protocol: look up hits, dedupe
    /// misses, compute the missing columns as one batch via
    /// `compute_missing`, then append and commit as many as the byte budget
    /// admits together, so the cache stays consistent even if computation
    /// fails partway. Once the budget is exhausted, further misses are still
    /// computed and returned (`CacheSlot::Uncached`) but not retained --
    /// matching Hiroshi's "continue evaluating new messages without caching
    /// them" rather than evicting an already-cached entry to make room.
    fn get_or_compute_batch<F, E>(
        &mut self,
        keys: &[K],
        compute_missing: F,
    ) -> std::result::Result<Vec<CacheSlot<T>>, E>
    where
        F: FnOnce(&[K]) -> std::result::Result<Vec<Vec<T>>, E>,
        E: From<anyhow::Error>,
    {
        let mut missing_keys = Vec::new();
        let mut missing_seen = HashSet::new();
        for key in keys {
            if self.positions.contains_key(key) {
                self.hits += 1;
            } else {
                self.misses += 1;
                if missing_seen.insert(key.clone()) {
                    missing_keys.push(key.clone());
                }
            }
        }

        let mut computed_values: HashMap<K, Vec<T>> = HashMap::new();
        if !missing_keys.is_empty() {
            let computed = compute_missing(&missing_keys)?;
            let bytes_per_column = self.bond_dim * std::mem::size_of::<T>();
            for (key, column) in missing_keys.into_iter().zip(computed) {
                let would_retain_bytes = self.retained_bytes() + bytes_per_column;
                if would_retain_bytes <= self.max_bytes {
                    let position = self.columns.len() / self.bond_dim;
                    self.columns.extend(column);
                    self.positions.insert(key, position);
                } else {
                    computed_values.insert(key, column);
                }
            }
        }

        // Every key is either cached above or, when the byte budget refused
        // it, present in `computed_values` -- unless `compute_missing`
        // returned fewer columns than the `missing_keys` it was asked for,
        // which is a caller contract violation this type cannot prevent, so
        // it is reported as an error rather than assumed impossible.
        keys.iter()
            .map(|key| match self.positions.get(key) {
                Some(&position) => Ok(CacheSlot::Cached(position)),
                None => computed_values
                    .get(key)
                    .cloned()
                    .map(CacheSlot::Uncached)
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "PackedMessageCache::get_or_compute_batch: compute_missing returned \
                             fewer columns than requested keys"
                        )
                        .into()
                    }),
            })
            .collect::<std::result::Result<Vec<_>, E>>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tensor4all_core::{ColMajorArrayRef, DynIndex, IdxTensor};

    #[test]
    fn tensor_from_cached_values_preserves_each_scalar_storage_kind() {
        let real_index = DynIndex::new_dyn(2);
        let real = tensor_from_cached_values(
            vec![real_index],
            vec![CachedScalar::Real(1.0), CachedScalar::Real(2.0)],
        )
        .unwrap();
        assert_eq!(real.to_vec::<f64>().unwrap(), vec![1.0, 2.0]);

        let complex_index = DynIndex::new_dyn(2);
        let complex = tensor_from_cached_values(
            vec![complex_index],
            vec![
                CachedScalar::Complex(Complex64::new(1.0, -2.0)),
                CachedScalar::Complex(Complex64::new(3.0, -4.0)),
            ],
        )
        .unwrap();
        assert_eq!(
            complex.to_vec::<Complex64>().unwrap(),
            vec![Complex64::new(1.0, -2.0), Complex64::new(3.0, -4.0)]
        );

        let mixed_index = DynIndex::new_dyn(2);
        let mixed = tensor_from_cached_values(
            vec![mixed_index],
            vec![
                CachedScalar::Real(5.0),
                CachedScalar::Complex(Complex64::new(6.0, 7.0)),
            ],
        )
        .unwrap();
        let mixed_values = tensor_values_any(&mixed).unwrap();
        assert_eq!(mixed_values[0].real(), 5.0);
        assert_eq!(mixed_values[0].imag(), 0.0);
        assert_eq!(mixed_values[1].real(), 6.0);
        assert_eq!(mixed_values[1].imag(), 7.0);
    }

    #[test]
    fn packed_message_cache_computes_and_stores_new_keys() {
        let mut cache = PackedMessageCache::<u32, f64>::new(2, usize::MAX);
        let mut compute_calls = 0usize;

        let slots = cache
            .get_or_compute_batch(&[1u32, 2u32], |missing: &[u32]| {
                compute_calls += 1;
                Ok::<_, anyhow::Error>(
                    missing
                        .iter()
                        .map(|k| vec![*k as f64, (*k as f64) * 10.0])
                        .collect(),
                )
            })
            .unwrap();

        assert_eq!(compute_calls, 1);
        let CacheSlot::Cached(p0) = slots[0] else {
            panic!("expected a cached slot: {:?}", slots[0])
        };
        let CacheSlot::Cached(p1) = slots[1] else {
            panic!("expected a cached slot: {:?}", slots[1])
        };
        assert_eq!(cache.column(p0), &[1.0, 10.0]);
        assert_eq!(cache.column(p1), &[2.0, 20.0]);
    }

    #[test]
    fn packed_message_cache_reuses_columns_across_calls() {
        let mut cache = PackedMessageCache::<u32, f64>::new(2, usize::MAX);
        let mut compute_calls = 0usize;
        let compute = |missing: &[u32]| -> std::result::Result<Vec<Vec<f64>>, anyhow::Error> {
            Ok(missing.iter().map(|k| vec![*k as f64, 0.0]).collect())
        };

        let first = cache
            .get_or_compute_batch(&[1u32, 2u32], |missing| {
                compute_calls += 1;
                compute(missing)
            })
            .unwrap();

        // A later call across a batch that repeats key 1 and adds new key 3
        // must recompute only the miss (3), not the already-cached hit (1).
        let second = cache
            .get_or_compute_batch(&[1u32, 3u32], |missing| {
                compute_calls += 1;
                assert_eq!(missing, &[3u32], "must not recompute an already-cached key");
                compute(missing)
            })
            .unwrap();

        assert_eq!(compute_calls, 2);
        assert_eq!(second[0], first[0], "key 1 must resolve to the same column");
        let CacheSlot::Cached(p3) = second[1] else {
            panic!("expected a cached slot: {:?}", second[1])
        };
        assert_eq!(cache.column(p3), &[3.0, 0.0]);
    }

    #[test]
    fn packed_message_cache_reports_cumulative_hits_and_misses() {
        let mut cache = PackedMessageCache::<u32, f64>::new(2, usize::MAX);
        let compute = |missing: &[u32]| -> std::result::Result<Vec<Vec<f64>>, anyhow::Error> {
            Ok(missing.iter().map(|k| vec![*k as f64, 0.0]).collect())
        };

        cache.get_or_compute_batch(&[1u32, 2u32], compute).unwrap();
        assert_eq!((cache.hits(), cache.misses()), (0, 2));

        // key 1 repeats (a hit), key 3 is new (a miss); duplicate key 1 within
        // the same batch counts as two hits, not one.
        cache
            .get_or_compute_batch(&[1u32, 1u32, 3u32], compute)
            .unwrap();
        assert_eq!((cache.hits(), cache.misses()), (2, 3));
    }

    #[test]
    fn packed_message_cache_get_all_cached_returns_none_on_any_miss_without_counting() {
        let mut cache = PackedMessageCache::<u32, f64>::new(2, usize::MAX);
        cache
            .get_or_compute_batch(&[1u32], |missing| {
                Ok::<_, anyhow::Error>(missing.iter().map(|k| vec![*k as f64, 0.0]).collect())
            })
            .unwrap();

        assert!(cache.get_all_cached(&[1u32, 2u32]).is_none());
        assert_eq!(
            (cache.hits(), cache.misses()),
            (0, 1),
            "a partial-miss lookup must not touch the counters"
        );

        let positions = cache.get_all_cached(&[1u32]).unwrap();
        assert_eq!(cache.column(positions[0]), &[1.0, 0.0]);
        assert_eq!((cache.hits(), cache.misses()), (1, 1));
    }

    #[test]
    fn packed_message_cache_stops_retaining_once_budget_is_exhausted() {
        // bond_dim=2, f64 -> 16 bytes per column. Budget fits exactly one.
        let mut cache = PackedMessageCache::<u32, f64>::new(2, 16);

        let slots = cache
            .get_or_compute_batch(&[1u32, 2u32], |missing| {
                Ok::<_, anyhow::Error>(missing.iter().map(|k| vec![*k as f64, 0.0]).collect())
            })
            .unwrap();

        let CacheSlot::Cached(position) = slots[0] else {
            panic!(
                "first key should fit the budget and be cached: {:?}",
                slots[0]
            );
        };
        assert_eq!(cache.column(position), &[1.0, 0.0]);

        let CacheSlot::Uncached(ref values) = slots[1] else {
            panic!(
                "second key exceeds the budget and must not be retained: {:?}",
                slots[1]
            );
        };
        assert_eq!(values, &[2.0, 0.0]);
        assert!(
            !cache.contains(&2u32),
            "an over-budget key must not be recorded as cached"
        );

        // Recomputed on a later call, since it was never retained.
        let mut recompute_calls = 0usize;
        let slots_again = cache
            .get_or_compute_batch(&[2u32], |missing| {
                recompute_calls += 1;
                Ok::<_, anyhow::Error>(missing.iter().map(|k| vec![*k as f64, 0.0]).collect())
            })
            .unwrap();
        assert_eq!(recompute_calls, 1);
        assert!(matches!(slots_again[0], CacheSlot::Uncached(_)));
    }

    /// `get_or_compute_batch` trusts `compute_missing` to return exactly one
    /// column per requested missing key. If a caller's closure violates that
    /// contract (returns fewer), the call must report a descriptive error
    /// rather than panicking.
    #[test]
    fn packed_message_cache_reports_an_error_when_compute_missing_returns_too_few_columns() {
        let mut cache = PackedMessageCache::<u32, f64>::new(2, usize::MAX);

        let result = cache.get_or_compute_batch(&[1u32, 2u32], |missing| {
            // Only ever returns a column for the first requested key,
            // regardless of how many were actually missing.
            Ok::<_, anyhow::Error>(vec![vec![missing[0] as f64, 0.0]])
        });

        let error = result.expect_err("a short compute_missing result must not panic");
        assert!(
            error.to_string().contains("fewer columns"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn grouped_chain_contraction_matches_scalar_reference_for_real_values() {
        let raw = vec![
            1.0, 2.0, 3.0, 4.0, // physical value 0
            5.0, 6.0, 7.0, 8.0, // physical value 1
        ];
        let physical_values = [0usize, 1, 0, 1];
        let child_columns = [
            0.5, 1.5, // point 0
            2.0, 3.0, // point 1
            4.0, 5.0, // point 2
            6.0, 7.0, // point 3
        ];

        let spec = ChainContractionSpec {
            strides: [1, 2, 4],
            physical_axis: 0,
            parent_axis: 1,
            child_axis: 2,
            parent_dim: 2,
            child_dim: 2,
        };
        let actual = TreeTNCachedEvaluator::<usize>::grouped_chain_message_contraction(
            spec,
            &raw,
            &physical_values,
            &child_columns,
        )
        .unwrap();

        let expected = vec![
            8.0, 12.0, // physical 0, child column [0.5, 1.5]
            22.0, 32.0, // physical 1, child column [2.0, 3.0]
            29.0, 47.0, // physical 0, child column [4.0, 5.0]
            54.0, 80.0, // physical 1, child column [6.0, 7.0]
        ];
        assert_eq!(actual, expected);
    }

    #[test]
    fn grouped_chain_contraction_matches_scalar_reference_for_complex_values() {
        let raw = vec![
            Complex64::new(1.0, 0.5),
            Complex64::new(2.0, -1.0),
            Complex64::new(3.0, 1.5),
            Complex64::new(4.0, -2.0),
            Complex64::new(5.0, 2.5),
            Complex64::new(6.0, -3.0),
            Complex64::new(7.0, 3.5),
            Complex64::new(8.0, -4.0),
        ];
        let physical_values = [0usize, 1, 0, 1];
        let child_columns = [
            Complex64::new(0.5, -0.5),
            Complex64::new(1.5, 0.25),
            Complex64::new(2.0, 1.0),
            Complex64::new(3.0, -0.75),
            Complex64::new(4.0, -1.5),
            Complex64::new(5.0, 0.5),
            Complex64::new(6.0, 2.0),
            Complex64::new(7.0, -1.25),
        ];

        let spec = ChainContractionSpec {
            strides: [1, 2, 4],
            physical_axis: 0,
            parent_axis: 1,
            child_axis: 2,
            parent_dim: 2,
            child_dim: 2,
        };
        let actual = TreeTNCachedEvaluator::<usize>::grouped_chain_message_contraction(
            spec,
            &raw,
            &physical_values,
            &child_columns,
        )
        .unwrap();

        let expected = scalar_grouped_chain_reference(spec, &raw, &physical_values, &child_columns);
        for (actual, expected) in actual.iter().zip(expected) {
            assert!((actual - expected).norm() < 1.0e-10);
        }
    }

    #[test]
    fn grouped_chain_contraction_large_real_groups_match_scalar_reference() {
        let parent_dim = 64;
        let child_dim = 64;
        let point_count = 16;
        let spec = ChainContractionSpec {
            strides: [1, 2, 2 * parent_dim],
            physical_axis: 0,
            parent_axis: 1,
            child_axis: 2,
            parent_dim,
            child_dim,
        };
        let raw: Vec<f64> = (0..2 * parent_dim * child_dim)
            .map(|value| (value % 19) as f64 - 9.0)
            .collect();
        let physical_values: Vec<usize> = (0..point_count).map(|point| point % 2).collect();
        let child_columns: Vec<f64> = (0..point_count * child_dim)
            .map(|value| (value % 13) as f64 - 6.0)
            .collect();

        let actual = TreeTNCachedEvaluator::<usize>::grouped_chain_message_contraction(
            spec,
            &raw,
            &physical_values,
            &child_columns,
        )
        .unwrap();
        let expected = scalar_grouped_chain_reference(spec, &raw, &physical_values, &child_columns);

        assert!(actual
            .iter()
            .zip(expected)
            .all(|(actual, expected)| (actual - expected).abs() < 1.0e-8));
    }

    #[test]
    fn grouped_chain_contraction_large_complex_groups_match_scalar_reference() {
        let parent_dim = 64;
        let child_dim = 64;
        let point_count = 16;
        let spec = ChainContractionSpec {
            strides: [1, 2, 2 * parent_dim],
            physical_axis: 0,
            parent_axis: 1,
            child_axis: 2,
            parent_dim,
            child_dim,
        };
        let raw: Vec<Complex64> = (0..2 * parent_dim * child_dim)
            .map(|value| Complex64::new((value % 19) as f64 - 9.0, (value % 11) as f64 - 5.0))
            .collect();
        let physical_values: Vec<usize> = (0..point_count).map(|point| point % 2).collect();
        let child_columns: Vec<Complex64> = (0..point_count * child_dim)
            .map(|value| Complex64::new((value % 13) as f64 - 6.0, (value % 7) as f64 - 3.0))
            .collect();

        let actual = TreeTNCachedEvaluator::<usize>::grouped_chain_message_contraction(
            spec,
            &raw,
            &physical_values,
            &child_columns,
        )
        .unwrap();
        let expected = scalar_grouped_chain_reference(spec, &raw, &physical_values, &child_columns);

        assert!(actual
            .iter()
            .zip(expected)
            .all(|(actual, expected)| (*actual - expected).norm() < 1.0e-8));
    }

    fn scalar_grouped_chain_reference<
        T: Copy + Default + std::ops::AddAssign + std::ops::Mul<Output = T>,
    >(
        spec: ChainContractionSpec,
        raw: &[T],
        physical_values: &[usize],
        child_columns: &[T],
    ) -> Vec<T> {
        let ChainContractionSpec {
            strides,
            physical_axis,
            parent_axis,
            child_axis,
            parent_dim,
            child_dim,
        } = spec;
        let mut output = vec![T::default(); parent_dim * physical_values.len()];
        for (point, &physical_value) in physical_values.iter().enumerate() {
            for parent_value in 0..parent_dim {
                let mut sum = T::default();
                for child_value in 0..child_dim {
                    let mut axis_values = [0usize; 3];
                    axis_values[physical_axis] = physical_value;
                    axis_values[parent_axis] = parent_value;
                    axis_values[child_axis] = child_value;
                    let flat = axis_values[0] * strides[0]
                        + axis_values[1] * strides[1]
                        + axis_values[2] * strides[2];
                    sum += raw[flat] * child_columns[point * child_dim + child_value];
                }
                output[point * parent_dim + parent_value] = sum;
            }
        }
        output
    }

    fn varied_three_node_chain() -> (TreeTN<IdxTensor, usize>, Vec<DynIndex>) {
        let s0 = DynIndex::new_dyn(2);
        let b01 = DynIndex::new_dyn(2);
        let s1 = DynIndex::new_dyn(2);
        let b12 = DynIndex::new_dyn(2);
        let s2 = DynIndex::new_dyn(2);

        let t0 = IdxTensor::from_dense(vec![s0.clone(), b01.clone()], vec![1.0_f64, 2.0, 3.0, 4.0])
            .unwrap();
        let t1 = IdxTensor::from_dense(
            vec![b01, s1.clone(), b12.clone()],
            (0..8).map(|i| i as f64 + 1.0).collect(),
        )
        .unwrap();
        let t2 =
            IdxTensor::from_dense(vec![b12, s2.clone()], vec![0.5_f64, 1.5, 2.5, 3.5]).unwrap();
        let tree = TreeTN::<_, usize>::from_tensors(vec![t0, t1, t2], vec![0, 1, 2]).unwrap();
        (tree, vec![s0, s1, s2])
    }

    /// A persistent per-edge message cache must not change results and must
    /// actually get used across separate `evaluate_batched` calls on the same
    /// evaluator -- the whole point of #626/#646's review-requested cache.
    #[test]
    fn evaluate_batched_reuses_persistent_message_cache_across_calls() {
        let (tree, indices) = varied_three_node_chain();
        let mut evaluator = TreeTNCachedEvaluator::new(
            &tree,
            &indices,
            CachedEvaluatorOptions {
                center: Some(0),
                ..Default::default()
            },
        )
        .unwrap();
        let shape = [3usize, 2usize];

        let values1 = [0usize, 0, 0, 0, 0, 1];
        let points1 = ColMajorArrayRef::new(&values1, &shape).unwrap();
        let actual1 = evaluator.evaluate_batched(points1).unwrap();
        let expected1 = tree.evaluate(&indices, points1).unwrap();
        assert_scalars_close(&actual1, &expected1);

        // Repeats point0=(0,0,0) from the first call; point1=(1,0,0) is new.
        let values2 = [0usize, 0, 0, 1, 0, 0];
        let points2 = ColMajorArrayRef::new(&values2, &shape).unwrap();
        let actual2 = evaluator.evaluate_batched(points2).unwrap();
        let expected2 = tree.evaluate(&indices, points2).unwrap();
        assert_scalars_close(&actual2, &expected2);

        let stats = evaluator.stats_for_test();
        assert!(
            stats.message_cache_hits > 0,
            "expected at least one message cache hit on the second call: {stats:?}"
        );
    }

    #[test]
    fn complex_cached_evaluator_preserves_values_across_cache_reuse() {
        let (tree, indices) = complex_three_node_chain();
        let mut evaluator = TreeTNCachedEvaluator::new(
            &tree,
            &indices,
            CachedEvaluatorOptions {
                center: Some(0),
                ..Default::default()
            },
        )
        .unwrap();
        let shape = [3usize, 2usize];
        let values1 = [0usize, 0, 0, 0, 0, 1];
        let points1 = ColMajorArrayRef::new(&values1, &shape).unwrap();
        let actual1 = evaluator.evaluate_batched(points1).unwrap();
        let expected1 = tree.evaluate(&indices, points1).unwrap();
        for (actual, expected) in actual1.iter().zip(expected1.iter()) {
            assert!((actual.real() - expected.real()).abs() < 1.0e-12);
            assert!((actual.imag() - expected.imag()).abs() < 1.0e-12);
        }

        let values2 = [0usize, 0, 0, 1, 0, 0];
        let points2 = ColMajorArrayRef::new(&values2, &shape).unwrap();
        let actual2 = evaluator.evaluate_batched(points2).unwrap();
        let expected2 = tree.evaluate(&indices, points2).unwrap();
        for (actual, expected) in actual2.iter().zip(expected2.iter()) {
            assert!((actual.real() - expected.real()).abs() < 1.0e-12);
            assert!((actual.imag() - expected.imag()).abs() < 1.0e-12);
        }
        assert!(evaluator.stats_for_test().message_cache_hits > 0);
    }

    #[test]
    fn message_cache_reuses_a_subtree_when_another_site_changes() {
        let (tree, indices) = varied_three_node_chain();
        let shape = [3usize, 1usize];
        let first_values = [0usize, 0, 0];
        let second_values = [0usize, 0, 1];
        let first = ColMajorArrayRef::new(&first_values, &shape).unwrap();
        let second = ColMajorArrayRef::new(&second_values, &shape).unwrap();
        let mut evaluator = TreeTNCachedEvaluator::new(
            &tree,
            &indices,
            CachedEvaluatorOptions {
                center: Some(1),
                ..Default::default()
            },
        )
        .unwrap();

        let actual_first = evaluator.evaluate_batched(first).unwrap();
        let actual_second = evaluator.evaluate_batched(second).unwrap();
        let expected_first = tree.evaluate(&indices, first).unwrap();
        let expected_second = tree.evaluate(&indices, second).unwrap();
        assert_scalars_close(&actual_first, &expected_first);
        assert_scalars_close(&actual_second, &expected_second);
        assert!(
            evaluator.stats_for_test().message_cache_hits > 0,
            "changing a site outside node 0's subtree should reuse its cached message: {:?}",
            evaluator.stats_for_test()
        );
    }

    #[test]
    fn zero_message_cache_budget_preserves_results_without_retaining_payload() {
        let (tree, indices) = varied_three_node_chain();
        let values = [0usize, 0, 0, 1, 0, 0];
        let shape = [3usize, 2usize];
        let points = ColMajorArrayRef::new(&values, &shape).unwrap();
        let expected = tree.evaluate(&indices, points).unwrap();
        let mut evaluator = TreeTNCachedEvaluator::new(
            &tree,
            &indices,
            CachedEvaluatorOptions {
                center: Some(0),
                message_cache_max_bytes: 0,
                ..Default::default()
            },
        )
        .unwrap();

        let actual = evaluator.evaluate_batched(points).unwrap();

        assert_scalars_close(&actual, &expected);
        assert!(!evaluator.message_caches.is_empty());
        assert!(evaluator
            .message_caches
            .values()
            .all(|cache| cache.retained_bytes() == 0));
        assert_eq!(evaluator.stats_for_test().message_cache_hits, 0);
    }

    /// Root cause of the cache slowdown (see the message-cache-prototype
    /// worklog) is `IdxTensor::to_vec` on a `contract_with_options` result
    /// hitting an expensive non-contiguous/backend-resident fallback. This
    /// tests the fix's first slice: a leaf node's message computed directly
    /// from the tree tensor's raw data, with no `contract_with_options` and
    /// no intermediate `IdxTensor` at all, must match the existing generic
    /// path exactly.
    #[test]
    fn raw_leaf_message_matches_generic_contraction() {
        let (tree, indices) = varied_three_node_chain();
        let mut evaluator = TreeTNCachedEvaluator::new(
            &tree,
            &indices,
            CachedEvaluatorOptions {
                center: Some(0),
                ..Default::default()
            },
        )
        .unwrap();
        let shape = [3usize, 2usize];
        let values = [0usize, 0, 0, 0, 0, 1];
        let points = ColMajorArrayRef::new(&values, &shape).unwrap();

        // Force layout/center bookkeeping the same way evaluate_batched would.
        evaluator.evaluate_batched(points).unwrap();

        let plan = RootedMessagePlan::new(&tree, &0).unwrap();
        let assignment_batches = evaluator
            .build_message_assignment_batches(&plan, points)
            .unwrap();
        let leaf_points = assignment_batches.get(&2).unwrap().first_points.clone();

        let expected_message = evaluator
            .compute_stacked_message(
                &2,
                points,
                &leaf_points,
                &plan,
                &assignment_batches,
                &HashMap::new(),
            )
            .unwrap();
        let expected = tensor_values_any(expected_message.tensor.as_ref().unwrap()).unwrap();

        let actual = evaluator
            .try_compute_leaf_message_raw(&2, points, &leaf_points)
            .unwrap()
            .expect("leaf node with one physical index and a real tensor must be eligible");

        assert_eq!(actual.len(), expected.len());
        for (a, e) in actual.iter().zip(expected.iter()) {
            assert!(
                (a - e.real()).abs() < 1.0e-12,
                "raw={a} generic={}",
                e.real()
            );
        }
    }

    #[test]
    fn raw_complex_leaf_message_matches_generic_contraction() {
        let (tree, indices) = complex_three_node_chain();
        let mut evaluator = TreeTNCachedEvaluator::new(
            &tree,
            &indices,
            CachedEvaluatorOptions {
                center: Some(0),
                ..Default::default()
            },
        )
        .unwrap();
        let shape = [3usize, 2usize];
        let values = [0usize, 0, 0, 0, 0, 1];
        let points = ColMajorArrayRef::new(&values, &shape).unwrap();

        evaluator.evaluate_batched(points).unwrap();

        let plan = RootedMessagePlan::new(&tree, &0).unwrap();
        let assignment_batches = evaluator
            .build_message_assignment_batches(&plan, points)
            .unwrap();
        let leaf_points = assignment_batches.get(&2).unwrap().first_points.clone();
        let expected_message = evaluator
            .compute_stacked_message(
                &2,
                points,
                &leaf_points,
                &plan,
                &assignment_batches,
                &HashMap::new(),
            )
            .unwrap();
        let expected = tensor_values_any(expected_message.tensor.as_ref().unwrap()).unwrap();

        let actual = evaluator
            .try_compute_leaf_message_complex_raw(&2, points, &leaf_points)
            .unwrap()
            .expect("complex leaf message should use the raw path");

        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected.iter()) {
            assert!((actual.re - expected.real()).abs() < 1.0e-12);
            assert!((actual.im - expected.imag()).abs() < 1.0e-12);
        }
    }

    #[test]
    fn raw_chain_message_matches_generic_contraction() {
        let (tree, indices) = varied_three_node_chain();
        let mut evaluator = TreeTNCachedEvaluator::new(
            &tree,
            &indices,
            CachedEvaluatorOptions {
                center: Some(0),
                ..Default::default()
            },
        )
        .unwrap();
        let shape = [3usize, 2usize];
        let values = [0usize, 0, 0, 0, 0, 1];
        let points = ColMajorArrayRef::new(&values, &shape).unwrap();
        evaluator.evaluate_batched(points).unwrap();

        let plan = RootedMessagePlan::new(&tree, &0).unwrap();
        let assignment_batches = evaluator
            .build_message_assignment_batches(&plan, points)
            .unwrap();

        // Node 2 (leaf, child of node 1) via the generic oracle path.
        let leaf_points = assignment_batches.get(&2).unwrap().first_points.clone();
        let node2_message = evaluator
            .compute_stacked_message(
                &2,
                points,
                &leaf_points,
                &plan,
                &assignment_batches,
                &HashMap::new(),
            )
            .unwrap();
        let mut messages = HashMap::new();
        messages.insert(2usize, node2_message);

        let node1_points = assignment_batches.get(&1).unwrap().first_points.clone();
        let expected_message = evaluator
            .compute_stacked_message(
                &1,
                points,
                &node1_points,
                &plan,
                &assignment_batches,
                &messages,
            )
            .unwrap();
        let expected = tensor_values_any(expected_message.tensor.as_ref().unwrap()).unwrap();

        let actual = evaluator
            .try_compute_chain_message_raw(
                &1,
                points,
                &node1_points,
                &plan,
                &assignment_batches,
                &messages,
            )
            .unwrap()
            .expect("interior node with one child and one physical index must be eligible");

        assert_eq!(actual.len(), expected.len());
        for (a, e) in actual.iter().zip(expected.iter()) {
            assert!(
                (a - e.real()).abs() < 1.0e-10,
                "raw={a} generic={}",
                e.real()
            );
        }
    }

    #[test]
    fn raw_complex_chain_message_matches_generic_contraction() {
        let (tree, indices) = complex_three_node_chain();
        let mut evaluator = TreeTNCachedEvaluator::new(
            &tree,
            &indices,
            CachedEvaluatorOptions {
                center: Some(0),
                ..Default::default()
            },
        )
        .unwrap();
        let shape = [3usize, 2usize];
        let values = [0usize, 0, 0, 0, 0, 1];
        let points = ColMajorArrayRef::new(&values, &shape).unwrap();
        evaluator.evaluate_batched(points).unwrap();

        let plan = RootedMessagePlan::new(&tree, &0).unwrap();
        let assignment_batches = evaluator
            .build_message_assignment_batches(&plan, points)
            .unwrap();

        let leaf_points = assignment_batches.get(&2).unwrap().first_points.clone();
        let node2_message = evaluator
            .compute_stacked_message(
                &2,
                points,
                &leaf_points,
                &plan,
                &assignment_batches,
                &HashMap::new(),
            )
            .unwrap();
        let mut messages = HashMap::new();
        messages.insert(2usize, node2_message);

        let node1_points = assignment_batches.get(&1).unwrap().first_points.clone();
        let expected_message = evaluator
            .compute_stacked_message(
                &1,
                points,
                &node1_points,
                &plan,
                &assignment_batches,
                &messages,
            )
            .unwrap();
        let expected = tensor_values_any(expected_message.tensor.as_ref().unwrap()).unwrap();

        let actual = evaluator
            .try_compute_chain_message_complex_raw(
                &1,
                points,
                &node1_points,
                &plan,
                &assignment_batches,
                &messages,
            )
            .unwrap()
            .expect("complex chain message should use the raw path");

        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected.iter()) {
            assert!((actual.re - expected.real()).abs() < 1.0e-10);
            assert!((actual.im - expected.imag()).abs() < 1.0e-10);
        }
    }

    #[test]
    fn fixed_center_and_scan_hint_evaluation_match() {
        let (tree, indices) = complex_three_node_chain();
        let shape = [3usize, 2usize];
        let values = [0usize, 0, 0, 0, 0, 1];
        let points = ColMajorArrayRef::new(&values, &shape).unwrap();

        let mut hinted =
            TreeTNCachedEvaluator::new(&tree, &indices, CachedEvaluatorOptions::<usize>::default())
                .unwrap();
        let mut fixed_center =
            TreeTNCachedEvaluator::new(&tree, &indices, CachedEvaluatorOptions::<usize>::default())
                .unwrap();

        let hinted_values = hinted
            .evaluate_batched_with_hint(points, EvaluationHint::around(1))
            .unwrap();
        let fixed_center_values = fixed_center.evaluate_batched(points).unwrap();

        assert_scalars_close(&hinted_values, &fixed_center_values);
    }

    fn assert_scalars_close(actual: &[AnyScalar], expected: &[AnyScalar]) {
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected.iter()) {
            assert!(
                (actual.real() - expected.real()).abs() < 1.0e-12,
                "actual={} expected={}",
                actual.real(),
                expected.real()
            );
        }
    }

    #[test]
    fn stack_tensors_adds_trailing_assignment_axis_in_column_major_order() {
        let batch = DynIndex::new_dyn(2);
        let i = DynIndex::new_dyn(2);
        let a = IdxTensor::from_dense(vec![i.clone()], vec![1.0_f64, 2.0]).unwrap();
        let b = IdxTensor::from_dense(vec![i.clone()], vec![3.0_f64, 4.0]).unwrap();

        let stacked = stack_tensors_with_assignment_index(&batch, &[a, b]).unwrap();

        assert_eq!(stacked.indices(), &[i, batch]);
        assert_eq!(stacked.to_vec::<f64>().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn gather_stacked_tensor_remaps_trailing_assignment_axis() {
        let source_batch = DynIndex::new_dyn(3);
        let target_batch = DynIndex::new_dyn(4);
        let i = DynIndex::new_dyn(2);
        let stacked = IdxTensor::from_dense(
            vec![i.clone(), source_batch.clone()],
            vec![10.0_f64, 11.0, 20.0, 21.0, 30.0, 31.0],
        )
        .unwrap();

        let gathered =
            gather_stacked_tensor(&stacked, &source_batch, &target_batch, &[2, 0, 2, 1]).unwrap();

        assert_eq!(gathered.indices(), &[i, target_batch]);
        assert_eq!(
            gathered.to_vec::<f64>().unwrap(),
            vec![30.0, 31.0, 10.0, 11.0, 30.0, 31.0, 20.0, 21.0]
        );
    }

    fn two_node_tree() -> (TreeTN<IdxTensor, usize>, Vec<DynIndex>) {
        let s0 = DynIndex::new_dyn(2);
        let bond = DynIndex::new_dyn(2);
        let s1 = DynIndex::new_dyn(2);

        let t0 =
            IdxTensor::from_dense(vec![s0.clone(), bond.clone()], vec![1.0_f64, 2.0, 3.0, 4.0])
                .unwrap();
        let t1 =
            IdxTensor::from_dense(vec![bond, s1.clone()], vec![0.5_f64, 1.5, 2.5, 3.5]).unwrap();

        let tree = TreeTN::<_, usize>::from_tensors(vec![t0, t1], vec![0, 1]).unwrap();
        (tree, vec![s0, s1])
    }

    fn three_node_chain() -> (TreeTN<IdxTensor, usize>, Vec<DynIndex>) {
        let s0 = DynIndex::new_dyn(2);
        let b01 = DynIndex::new_dyn(2);
        let s1 = DynIndex::new_dyn(2);
        let b12 = DynIndex::new_dyn(2);
        let s2 = DynIndex::new_dyn(2);

        let t0 = IdxTensor::from_dense(vec![s0.clone(), b01.clone()], vec![1.0_f64; 4]).unwrap();
        let t1 =
            IdxTensor::from_dense(vec![b01, s1.clone(), b12.clone()], vec![1.0_f64; 8]).unwrap();
        let t2 = IdxTensor::from_dense(vec![b12, s2.clone()], vec![1.0_f64; 4]).unwrap();
        let tree = TreeTN::<_, usize>::from_tensors(vec![t0, t1, t2], vec![0, 1, 2]).unwrap();
        (tree, vec![s0, s1, s2])
    }

    fn complex_three_node_chain() -> (TreeTN<IdxTensor, usize>, Vec<DynIndex>) {
        let s0 = DynIndex::new_dyn(2);
        let b01 = DynIndex::new_dyn(2);
        let s1 = DynIndex::new_dyn(2);
        let b12 = DynIndex::new_dyn(2);
        let s2 = DynIndex::new_dyn(2);

        let t0 = IdxTensor::from_dense(
            vec![s0.clone(), b01.clone()],
            vec![
                Complex64::new(1.0, 0.5),
                Complex64::new(-0.25, 1.5),
                Complex64::new(2.0, -0.75),
                Complex64::new(0.5, -1.0),
            ],
        )
        .unwrap();
        let t1 = IdxTensor::from_dense(
            vec![b01, s1.clone(), b12.clone()],
            (0..8)
                .map(|value| Complex64::new(value as f64 + 0.5, -(value as f64) * 0.25))
                .collect(),
        )
        .unwrap();
        let t2 = IdxTensor::from_dense(
            vec![b12, s2.clone()],
            vec![
                Complex64::new(0.75, -0.5),
                Complex64::new(1.25, 0.25),
                Complex64::new(-1.0, 0.75),
                Complex64::new(2.5, -1.25),
            ],
        )
        .unwrap();
        let tree = TreeTN::<_, usize>::from_tensors(vec![t0, t1, t2], vec![0, 1, 2]).unwrap();
        (tree, vec![s0, s1, s2])
    }

    fn five_node_chain() -> (TreeTN<IdxTensor, usize>, Vec<DynIndex>) {
        let sites: Vec<DynIndex> = (0..5).map(|_| DynIndex::new_dyn(2)).collect();
        let bonds: Vec<DynIndex> = (0..4).map(|_| DynIndex::new_dyn(2)).collect();

        let t0 = IdxTensor::from_dense(vec![sites[0].clone(), bonds[0].clone()], vec![1.0_f64; 4])
            .unwrap();
        let t1 = IdxTensor::from_dense(
            vec![bonds[0].clone(), sites[1].clone(), bonds[1].clone()],
            vec![1.0_f64; 8],
        )
        .unwrap();
        let t2 = IdxTensor::from_dense(
            vec![bonds[1].clone(), sites[2].clone(), bonds[2].clone()],
            vec![1.0_f64; 8],
        )
        .unwrap();
        let t3 = IdxTensor::from_dense(
            vec![bonds[2].clone(), sites[3].clone(), bonds[3].clone()],
            vec![1.0_f64; 8],
        )
        .unwrap();
        let t4 = IdxTensor::from_dense(vec![bonds[3].clone(), sites[4].clone()], vec![1.0_f64; 4])
            .unwrap();

        let tree = TreeTN::<_, usize>::from_tensors(vec![t0, t1, t2, t3, t4], vec![0, 1, 2, 3, 4])
            .unwrap();
        (tree, sites)
    }

    fn star_tree() -> (TreeTN<IdxTensor, usize>, Vec<DynIndex>) {
        let sc = DynIndex::new_dyn(2);
        let s0 = DynIndex::new_dyn(2);
        let s1 = DynIndex::new_dyn(2);
        let s2 = DynIndex::new_dyn(2);
        let b0 = DynIndex::new_dyn(2);
        let b1 = DynIndex::new_dyn(2);
        let b2 = DynIndex::new_dyn(2);
        let center_data: Vec<f64> = (0..16).map(|value| value as f64 + 1.0).collect();
        let center = IdxTensor::from_dense(
            vec![sc.clone(), b0.clone(), b1.clone(), b2.clone()],
            center_data,
        )
        .unwrap();
        let leaf0 =
            IdxTensor::from_dense(vec![b0, s0.clone()], vec![1.0_f64, 0.5, 1.5, 2.0]).unwrap();
        let leaf1 =
            IdxTensor::from_dense(vec![b1, s1.clone()], vec![0.25_f64, 1.0, 1.25, 2.0]).unwrap();
        let leaf2 =
            IdxTensor::from_dense(vec![b2, s2.clone()], vec![2.0_f64, 1.0, 0.75, 1.5]).unwrap();
        let tree =
            TreeTN::<_, usize>::from_tensors(vec![center, leaf0, leaf1, leaf2], vec![0, 1, 2, 3])
                .unwrap();
        (tree, vec![sc, s0, s1, s2])
    }

    #[test]
    fn cached_evaluator_matches_tree_evaluate_on_two_node_chain() {
        let (tree, indices) = two_node_tree();
        let values = vec![0, 0, 1, 0, 0, 1, 1, 1];
        let shape = [2, 4];
        let points = ColMajorArrayRef::new(&values, &shape).unwrap();

        let expected = tree.evaluate(&indices, points).unwrap();
        let options = CachedEvaluatorOptions {
            center: Some(0),
            ..CachedEvaluatorOptions::default()
        };
        let mut evaluator = TreeTNCachedEvaluator::new(&tree, &indices, options).unwrap();
        let actual = evaluator.evaluate_batched(points).unwrap();

        assert_scalars_close(&actual, &expected);
        assert_eq!(evaluator.center(), Some(&0));
    }

    #[test]
    fn component_cost_index_counts_unique_directed_components() {
        let (tree, indices) = three_node_chain();
        let values = vec![0, 0, 0, 0, 1, 1, 1, 1, 1];
        let shape = [3, 3];
        let points = ColMajorArrayRef::new(&values, &shape).unwrap();

        let cost_index = ComponentCostIndex::new(&tree, &indices, points).unwrap();

        assert_eq!(cost_index.component_count(&(0, 1)).unwrap(), 2);
        assert_eq!(cost_index.component_count(&(1, 0)).unwrap(), 2);
        assert_eq!(cost_index.component_count(&(1, 2)).unwrap(), 3);
        assert_eq!(cost_index.component_count(&(2, 1)).unwrap(), 2);
        assert_eq!(cost_index.center_cost(&0).unwrap(), 2);
        assert_eq!(cost_index.center_cost(&1).unwrap(), 4);
        assert_eq!(cost_index.center_cost(&2).unwrap(), 3);
    }

    #[test]
    fn component_cost_index_rejects_wrong_batch_row_count() {
        let (tree, indices) = three_node_chain();
        let values = vec![0, 1, 1, 0];
        let shape = [2, 2];
        let points = ColMajorArrayRef::new(&values, &shape).unwrap();

        let err = ComponentCostIndex::new(&tree, &indices, points)
            .err()
            .unwrap();
        assert!(err.to_string().contains("row count"));
    }

    #[test]
    fn greedy_center_search_descends_to_lower_cost_neighbor() {
        let cost_index = ComponentCostIndex::from_parts_for_test(
            HashMap::from([(0, vec![1]), (1, vec![0, 2]), (2, vec![1, 3]), (3, vec![2])]),
            HashMap::from([(0, 40), (1, 20), (2, 10), (3, 15)]),
        );

        let result = GreedyCenterSearch::<usize>::default()
            .search(&cost_index, &[0])
            .unwrap();

        assert_eq!(result.center, 2);
        assert_eq!(result.cost, 10);
        assert_eq!(result.path, vec![0, 1, 2]);
    }

    #[test]
    fn greedy_center_search_uses_best_of_multiple_starts() {
        let cost_index = ComponentCostIndex::from_parts_for_test(
            HashMap::from([
                ("a", vec!["b"]),
                ("b", vec!["a", "c"]),
                ("c", vec!["b", "d"]),
                ("d", vec!["c"]),
            ]),
            HashMap::from([("a", 8), ("b", 6), ("c", 5), ("d", 2)]),
        );

        let result = GreedyCenterSearch::<&str>::with_max_steps(Some(1))
            .search(&cost_index, &["a", "d"])
            .unwrap();

        assert_eq!(result.center, "d");
        assert_eq!(result.cost, 2);
    }

    #[test]
    fn cached_evaluator_selects_greedy_center_when_center_is_not_fixed() {
        let (tree, indices) = three_node_chain();
        let values = vec![0, 0, 0, 0, 1, 1, 1, 1, 1];
        let shape = [3, 3];
        let points = ColMajorArrayRef::new(&values, &shape).unwrap();

        let expected = tree.evaluate(&indices, points).unwrap();
        let mut evaluator = TreeTNCachedEvaluator::new(
            &tree,
            &indices,
            CachedEvaluatorOptions {
                initial_centers: vec![1],
                ..Default::default()
            },
        )
        .unwrap();

        let actual = evaluator.evaluate_batched(points).unwrap();

        assert_eq!(evaluator.center(), Some(&0));
        assert_scalars_close(&actual, &expected);
    }

    #[test]
    fn cached_evaluator_rejects_unknown_initial_center() {
        let (tree, indices) = three_node_chain();
        let err = TreeTNCachedEvaluator::new(
            &tree,
            &indices,
            CachedEvaluatorOptions {
                initial_centers: vec![99],
                ..Default::default()
            },
        )
        .err()
        .unwrap();

        assert!(err.to_string().contains("initial center"));
    }

    #[test]
    fn cached_evaluator_computes_one_environment_per_unique_subtree_assignment() {
        let (tree, indices) = three_node_chain();
        let values = vec![0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1];
        let shape = [3, 4];
        let points = ColMajorArrayRef::new(&values, &shape).unwrap();

        let mut evaluator = TreeTNCachedEvaluator::new(
            &tree,
            &indices,
            CachedEvaluatorOptions {
                center: Some(1),
                ..Default::default()
            },
        )
        .unwrap();
        let expected = tree.evaluate(&indices, points).unwrap();
        let actual = evaluator.evaluate_batched(points).unwrap();

        assert_scalars_close(&actual, &expected);
        assert_eq!(evaluator.stats_for_test().subtree_environment_count, 4);
    }

    #[test]
    fn cached_evaluator_reuses_directed_messages_inside_components() {
        let (tree, indices) = five_node_chain();
        let values = vec![
            0, 0, 0, 0, 0, //
            0, 1, 0, 0, 1, //
            1, 0, 1, 1, 0, //
            1, 1, 1, 1, 1,
        ];
        let shape = [5, 4];
        let points = ColMajorArrayRef::new(&values, &shape).unwrap();

        let expected = tree.evaluate(&indices, points).unwrap();
        let mut evaluator = TreeTNCachedEvaluator::new(
            &tree,
            &indices,
            CachedEvaluatorOptions {
                center: Some(2),
                ..Default::default()
            },
        )
        .unwrap();
        let actual = evaluator.evaluate_batched(points).unwrap();

        assert_scalars_close(&actual, &expected);
        assert_eq!(evaluator.stats_for_test().directed_message_count, 12);
    }

    #[test]
    fn cached_evaluator_batches_directed_messages() {
        let (tree, indices) = five_node_chain();
        let values = vec![
            0, 0, 0, 0, 0, //
            0, 1, 0, 0, 1, //
            1, 0, 1, 1, 0, //
            1, 1, 1, 1, 1,
        ];
        let shape = [5, 4];
        let points = ColMajorArrayRef::new(&values, &shape).unwrap();

        let expected = tree.evaluate(&indices, points).unwrap();
        let mut evaluator = TreeTNCachedEvaluator::new(
            &tree,
            &indices,
            CachedEvaluatorOptions {
                center: Some(2),
                ..Default::default()
            },
        )
        .unwrap();
        let actual = evaluator.evaluate_batched(points).unwrap();
        let stats = evaluator.stats_for_test();

        assert_scalars_close(&actual, &expected);
        assert!(stats.batched_message_contract_count < stats.directed_message_count);
    }

    #[test]
    fn cached_evaluator_rejects_wrong_value_row_count() {
        let (tree, indices) = two_node_tree();
        let values = vec![0, 1, 1];
        let shape = [1, 3];
        let points = ColMajorArrayRef::new(&values, &shape).unwrap();
        let mut evaluator =
            TreeTNCachedEvaluator::new(&tree, &indices, CachedEvaluatorOptions::default()).unwrap();

        let err = evaluator.evaluate_batched(points).unwrap_err();
        assert!(err.to_string().contains("row count"));
    }

    #[test]
    fn cached_evaluator_rejects_out_of_range_site_value() {
        let (tree, indices) = two_node_tree();
        let values = vec![0, 2];
        let shape = [2, 1];
        let points = ColMajorArrayRef::new(&values, &shape).unwrap();
        let mut evaluator =
            TreeTNCachedEvaluator::new(&tree, &indices, CachedEvaluatorOptions::default()).unwrap();

        let err = evaluator.evaluate_batched(points).unwrap_err();
        assert!(err.to_string().contains("out of range"));
    }

    #[test]
    fn cached_evaluator_handles_repeated_points_without_changing_order() {
        let (tree, indices) = two_node_tree();
        let values = vec![0, 0, 1, 1, 0, 0, 1, 1];
        let shape = [2, 4];
        let points = ColMajorArrayRef::new(&values, &shape).unwrap();
        let expected = tree.evaluate(&indices, points).unwrap();
        let mut evaluator =
            TreeTNCachedEvaluator::new(&tree, &indices, CachedEvaluatorOptions::default()).unwrap();

        let actual = evaluator.evaluate_batched(points).unwrap();

        assert_scalars_close(&actual, &expected);
        assert_eq!(actual[0].real(), actual[2].real());
        assert_eq!(actual[1].real(), actual[3].real());
    }

    #[test]
    fn cached_evaluator_matches_tree_evaluate_on_star_tree() {
        let (tree, indices) = star_tree();
        let values = vec![
            0, 0, 0, 0, //
            1, 0, 1, 0, //
            0, 1, 0, 1, //
            1, 1, 1, 1,
        ];
        let shape = [4, 4];
        let points = ColMajorArrayRef::new(&values, &shape).unwrap();
        let expected = tree.evaluate(&indices, points).unwrap();
        let mut evaluator = TreeTNCachedEvaluator::new(
            &tree,
            &indices,
            CachedEvaluatorOptions {
                center: Some(0),
                ..Default::default()
            },
        )
        .unwrap();

        let actual = evaluator.evaluate_batched(points).unwrap();

        assert_scalars_close(&actual, &expected);
    }

    #[test]
    fn cached_evaluator_batches_center_contraction() {
        let (tree, indices) = star_tree();
        let values = vec![
            0, 0, 0, 0, //
            1, 0, 1, 0, //
            0, 1, 0, 1, //
            1, 1, 1, 1,
        ];
        let shape = [4, 4];
        let points = ColMajorArrayRef::new(&values, &shape).unwrap();
        let expected = tree.evaluate(&indices, points).unwrap();
        let mut evaluator = TreeTNCachedEvaluator::new(
            &tree,
            &indices,
            CachedEvaluatorOptions {
                center: Some(0),
                ..Default::default()
            },
        )
        .unwrap();

        let actual = evaluator.evaluate_batched(points).unwrap();

        assert_scalars_close(&actual, &expected);
        assert_eq!(evaluator.stats_for_test().batched_center_contract_count, 1);
    }

    /// Uncached-path duplicate of `evaluate_batched_with_hint`, calling
    /// `compute_stacked_message` directly instead of
    /// `get_or_compute_node_message`. Kept as measurement tooling (see
    /// `message_cache_wall_time_on_realistic_floating_zone_walk`): times the
    /// same floating-zone walk with and without the persistent message
    /// cache, using one evaluator so construction overhead is identical
    /// between the two conditions.
    fn evaluate_batched_uncached(
        evaluator: &mut TreeTNCachedEvaluator<'_, usize>,
        values: ColMajorArrayRef<'_, usize>,
        center: &usize,
    ) -> Result<Vec<AnyScalar>> {
        let plan = RootedMessagePlan::new(evaluator.tree, center)?;
        let assignment_batches = evaluator.build_message_assignment_batches(&plan, values)?;
        let mut messages = HashMap::<usize, StackedMessage>::new();
        for node in &plan.postorder {
            let points = assignment_batches.get(node).unwrap().first_points.clone();
            let node_message = evaluator.compute_stacked_message(
                node,
                values,
                &points,
                &plan,
                &assignment_batches,
                &messages,
            )?;
            messages.insert(*node, node_message);
        }
        let mut component_batches = Vec::new();
        let mut environment_cache = HashMap::new();
        for neighbor in plan.children.get(center).cloned().unwrap_or_default() {
            let assignment_batch = assignment_batches.get(&neighbor).unwrap();
            let environment = messages.remove(&neighbor).unwrap();
            environment_cache.insert(neighbor, environment);
            component_batches.push(ComponentBatch {
                neighbor,
                point_to_assignment: assignment_batch.point_to_assignment.clone(),
            });
        }
        evaluator.contract_center_for_points(center, values, &component_batches, &environment_cache)
    }

    /// Measurement, not a regression test: how much does the persistent
    /// message cache save on the real `find_global_pivots` call pattern?
    /// Drives `floating_zone_walk` with the same defaults
    /// (`nsearch_global_pivots = 5`, `nsweeps_global_search = 100`) against a
    /// 16-site chain at bond 128, once through the normal (now cached) path
    /// and once through the direct, uncached path, on the same evaluator so
    /// construction cost cancels out. Prints wall time and message counts for
    /// both; asserts nothing about the ratio, since wall-clock numbers are
    /// not a reproducible pass/fail condition.
    ///
    /// The cached path is expected to be faster on this workload because the
    /// raw chain path avoids constructing an `IdxTensor` per message and the
    /// persistent cache avoids recomputing unchanged directed messages. The
    /// test remains measurement tooling rather than a wall-clock regression
    /// assertion; see `docs/worklogs/2026-08-18-treeaci-message-cache-prototype.md`.
    #[test]
    fn message_cache_wall_time_on_realistic_floating_zone_walk() {
        use rand::{Rng, SeedableRng};
        use rand_chacha::ChaCha8Rng;
        use std::time::Instant;
        use tensor4all_core::floating_zone_walk;
        use tensor4all_simplett::{tensor3_zeros, SimpleTensorTrain, Tensor3, Tensor3Ops};

        const N_SITES: usize = 16;
        const LOCAL_DIM: usize = 2;
        const BOND_DIM: usize = 128;
        const NSEARCH: usize = 5;
        const MAX_SWEEPS: usize = 100;

        fn build_tree(seed: u64) -> (TreeTN<IdxTensor, usize>, Vec<DynIndex>) {
            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            let mut tensors: Vec<Tensor3<f64>> = Vec::with_capacity(N_SITES);
            for site in 0..N_SITES {
                let left_dim = if site == 0 { 1 } else { BOND_DIM };
                let right_dim = if site == N_SITES - 1 { 1 } else { BOND_DIM };
                let mut tensor = tensor3_zeros(left_dim, LOCAL_DIM, right_dim);
                for l in 0..left_dim {
                    for s in 0..LOCAL_DIM {
                        for r in 0..right_dim {
                            tensor.set3(l, s, r, rng.random::<f64>());
                        }
                    }
                }
                tensors.push(tensor);
            }
            let tt = SimpleTensorTrain::new(tensors).unwrap();
            crate::tensor_train_to_treetn(&tt).unwrap()
        }

        fn starts() -> Vec<Vec<usize>> {
            let mut rng = ChaCha8Rng::seed_from_u64(11);
            (0..NSEARCH)
                .map(|_| {
                    (0..N_SITES)
                        .map(|_| rng.random_range(0..LOCAL_DIM))
                        .collect()
                })
                .collect()
        }
        let site_dims = vec![LOCAL_DIM; N_SITES];

        // Cached path: the real evaluate_batched, as find_global_pivots uses it.
        let (tree, site_indices) = build_tree(7);
        let mut evaluator =
            TreeTNCachedEvaluator::new(&tree, &site_indices, CachedEvaluatorOptions::default())
                .unwrap();
        // Force a centre outside the timed region, matching the uncached arm.
        let warm_values0 = vec![0usize; N_SITES];
        let warm_shape0 = [N_SITES, 1];
        let warm_points0 = ColMajorArrayRef::new(&warm_values0, &warm_shape0).unwrap();
        evaluator.evaluate_batched(warm_points0).unwrap();
        let mut total_hits = 0usize;
        let mut total_misses = 0usize;
        let mut total_calls = 0usize;
        let cached_start = Instant::now();
        for start in &starts() {
            floating_zone_walk(
                &site_dims,
                start,
                MAX_SWEEPS,
                f64::INFINITY,
                |points: &[Vec<usize>]| -> Result<Vec<f64>> {
                    let mut values = vec![0usize; N_SITES * points.len()];
                    for (p, point) in points.iter().enumerate() {
                        for (site, &v) in point.iter().enumerate() {
                            values[site + N_SITES * p] = v;
                        }
                    }
                    let shape = [N_SITES, points.len()];
                    let arr = ColMajorArrayRef::new(&values, &shape).unwrap();
                    let out = evaluator.evaluate_batched(arr)?;
                    let stats = evaluator.stats_for_test();
                    total_hits += stats.message_cache_hits;
                    total_misses += stats.message_cache_misses;
                    total_calls += 1;
                    Ok(out.iter().map(|v| v.real().abs()).collect())
                },
            )
            .unwrap();
        }
        let cached_elapsed = cached_start.elapsed();

        // Uncached path: identical tree, identical walk, direct compute.
        let (tree2, site_indices2) = build_tree(7);
        let mut evaluator2 =
            TreeTNCachedEvaluator::new(&tree2, &site_indices2, CachedEvaluatorOptions::default())
                .unwrap();
        // Force a centre exactly as the cached run's first call would.
        let warm_values = vec![0usize; N_SITES];
        let warm_shape = [N_SITES, 1];
        let warm_points = ColMajorArrayRef::new(&warm_values, &warm_shape).unwrap();
        evaluator2.evaluate_batched(warm_points).unwrap();
        let center = *evaluator2.center().unwrap();
        let uncached_start = Instant::now();
        for start in &starts() {
            floating_zone_walk(
                &site_dims,
                start,
                MAX_SWEEPS,
                f64::INFINITY,
                |points: &[Vec<usize>]| -> Result<Vec<f64>> {
                    let mut values = vec![0usize; N_SITES * points.len()];
                    for (p, point) in points.iter().enumerate() {
                        for (site, &v) in point.iter().enumerate() {
                            values[site + N_SITES * p] = v;
                        }
                    }
                    let shape = [N_SITES, points.len()];
                    let arr = ColMajorArrayRef::new(&values, &shape).unwrap();
                    let out = evaluate_batched_uncached(&mut evaluator2, arr, &center)?;
                    Ok(out.iter().map(|v| v.real().abs()).collect())
                },
            )
            .unwrap();
        }
        let uncached_elapsed = uncached_start.elapsed();

        println!(
            "floating-zone walk at bond={BOND_DIM}: cached={cached_elapsed:?} uncached={uncached_elapsed:?} speedup={:.2}x total_calls={total_calls} total_hits={total_hits} total_misses={total_misses} node_hit_rate={:.3}",
            uncached_elapsed.as_secs_f64() / cached_elapsed.as_secs_f64(),
            total_hits as f64 / (total_hits + total_misses) as f64,
        );
        assert!(cached_elapsed.as_nanos() > 0);
    }

    /// Root-cause investigation for the slowdown found by
    /// `message_cache_wall_time_on_realistic_floating_zone_walk`: which phase
    /// inside `get_or_compute_node_message` actually accounts for the time,
    /// on the real call pattern -- not inferred from the old bond=2 primitive
    /// breakdown in `2026-08-17-treeaci-per-evaluation-cost.md`. Reuses the
    /// same 16-site, bond=128 walk as the wall-time measurement.
    #[test]
    fn message_cache_phase_breakdown_on_realistic_floating_zone_walk() {
        use rand::{Rng, SeedableRng};
        use rand_chacha::ChaCha8Rng;
        use tensor4all_core::floating_zone_walk;
        use tensor4all_simplett::{tensor3_zeros, SimpleTensorTrain, Tensor3, Tensor3Ops};

        const N_SITES: usize = 16;
        const LOCAL_DIM: usize = 2;
        const BOND_DIM: usize = 128;
        const NSEARCH: usize = 5;
        const MAX_SWEEPS: usize = 100;

        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let mut tensors: Vec<Tensor3<f64>> = Vec::with_capacity(N_SITES);
        for site in 0..N_SITES {
            let left_dim = if site == 0 { 1 } else { BOND_DIM };
            let right_dim = if site == N_SITES - 1 { 1 } else { BOND_DIM };
            let mut tensor = tensor3_zeros(left_dim, LOCAL_DIM, right_dim);
            for l in 0..left_dim {
                for s in 0..LOCAL_DIM {
                    for r in 0..right_dim {
                        tensor.set3(l, s, r, rng.random::<f64>());
                    }
                }
            }
            tensors.push(tensor);
        }
        let tt = SimpleTensorTrain::new(tensors).unwrap();
        let (tree, site_indices) = crate::tensor_train_to_treetn(&tt).unwrap();
        let mut evaluator =
            TreeTNCachedEvaluator::new(&tree, &site_indices, CachedEvaluatorOptions::default())
                .unwrap();
        let warm_values = vec![0usize; N_SITES];
        let warm_shape = [N_SITES, 1];
        evaluator
            .evaluate_batched(ColMajorArrayRef::new(&warm_values, &warm_shape).unwrap())
            .unwrap();

        phase_timing::reset_all();

        let site_dims = vec![LOCAL_DIM; N_SITES];
        let mut start_rng = ChaCha8Rng::seed_from_u64(11);
        let starts: Vec<Vec<usize>> = (0..NSEARCH)
            .map(|_| {
                (0..N_SITES)
                    .map(|_| start_rng.random_range(0..LOCAL_DIM))
                    .collect()
            })
            .collect();
        for start in &starts {
            floating_zone_walk(
                &site_dims,
                start,
                MAX_SWEEPS,
                f64::INFINITY,
                |points: &[Vec<usize>]| -> Result<Vec<f64>> {
                    let mut values = vec![0usize; N_SITES * points.len()];
                    for (p, point) in points.iter().enumerate() {
                        for (site, &v) in point.iter().enumerate() {
                            values[site + N_SITES * p] = v;
                        }
                    }
                    let shape = [N_SITES, points.len()];
                    let arr = ColMajorArrayRef::new(&values, &shape).unwrap();
                    let out = evaluator.evaluate_batched(arr)?;
                    Ok(out.iter().map(|v| v.real().abs()).collect())
                },
            )
            .unwrap();
        }

        use std::sync::atomic::Ordering;
        let key_and_lookup = phase_timing::KEY_AND_LOOKUP_NS.load(Ordering::Relaxed);
        let contract = phase_timing::CONTRACT_NS.load(Ordering::Relaxed);
        let tensor_values = phase_timing::TENSOR_VALUES_NS.load(Ordering::Relaxed);
        let insert = phase_timing::INSERT_NS.load(Ordering::Relaxed);
        let reconstruct = phase_timing::RECONSTRUCT_NS.load(Ordering::Relaxed);
        let total = key_and_lookup + contract + tensor_values + insert + reconstruct;
        println!(
            "phase breakdown at bond={BOND_DIM}: key_and_lookup={:.1}ms ({:.1}%) contract={:.1}ms ({:.1}%) tensor_values={:.1}ms ({:.1}%) insert={:.1}ms ({:.1}%) reconstruct={:.1}ms ({:.1}%) total={:.1}ms",
            key_and_lookup as f64 / 1e6, 100.0 * key_and_lookup as f64 / total as f64,
            contract as f64 / 1e6, 100.0 * contract as f64 / total as f64,
            tensor_values as f64 / 1e6, 100.0 * tensor_values as f64 / total as f64,
            insert as f64 / 1e6, 100.0 * insert as f64 / total as f64,
            reconstruct as f64 / 1e6, 100.0 * reconstruct as f64 / total as f64,
            total as f64 / 1e6,
        );
        assert!(total > 0);
    }
}
