use crate::error::Result as TreeTciResult;
use crate::SubtreeKey;
use anyhow::anyhow;
use petgraph::algo::connected_components;
use petgraph::graph::{NodeIndex, UnGraph};
use petgraph::visit::EdgeRef;
use std::collections::{BTreeMap, BTreeSet, VecDeque};

/// Canonical undirected edge used by TreeTCI.
/// Endpoints are stored in canonical order (`u <= v`).
/// # Examples
/// ```
/// use tensor4all_treetci::TreeTciEdge;
/// let e = TreeTciEdge::new(3, 1);
/// // canonical ordering: lower endpoint first
/// assert_eq!(e.u(), 1);
/// assert_eq!(e.v(), 3);
/// // same edge regardless of argument order
/// assert_eq!(TreeTciEdge::new(1, 3), TreeTciEdge::new(3, 1));
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TreeTciEdge {
    u: usize,
    v: usize,
}

impl TreeTciEdge {
    /// Construct an undirected edge with canonical endpoint ordering.
    pub fn new(a: usize, b: usize) -> Self {
        if a <= b {
            Self { u: a, v: b }
        } else {
            Self { u: b, v: a }
        }
    }

    /// Return the lower endpoint.
    pub fn u(self) -> usize {
        self.u
    }

    /// Return the higher endpoint.
    pub fn v(self) -> usize {
        self.v
    }
}

/// Tree graph metadata for TreeTCI.
/// A tree graph is a connected, acyclic undirected graph. Each site corresponds
/// to a node and each bond is an undirected edge.
/// # Examples
/// ```
/// use tensor4all_treetci::{TreeTciEdge, TreeTciGraph};
/// // Linear chain: 0 -- 1 -- 2
/// let graph = TreeTciGraph::new(3, &[
///     TreeTciEdge::new(0, 1),
///     TreeTciEdge::new(1, 2),
/// ]).unwrap();
/// assert_eq!(graph.n_sites(), 3);
/// assert_eq!(graph.edges().len(), 2);
/// // Neighbors of site 1 are sites 0 and 2
/// let mut neighbors = graph.neighbors(1).unwrap();
/// neighbors.sort();
/// assert_eq!(neighbors, vec![0, 2]);
/// ```
#[derive(Clone, Debug)]
pub struct TreeTciGraph {
    n_sites: usize,
    graph: UnGraph<(), ()>,
}

impl TreeTciGraph {
    /// Construct a tree graph from the site count and edge list.
    /// # Errors
    /// Returns an error when the graph is invalid (an invalid-topology failure).
    ///
    pub fn new(n_sites: usize, edges: &[TreeTciEdge]) -> TreeTciResult<Self> {
        if !(n_sites > 0) {
            return Err(anyhow::anyhow!("TreeTCI graph must contain at least one site").into());
        };

        let mut graph = UnGraph::<(), ()>::default();
        for _ in 0..n_sites {
            graph.add_node(());
        }

        let mut seen = BTreeSet::new();
        for &edge in edges {
            if !(edge.u != edge.v) {
                return Err(anyhow::anyhow!("self-loops are not allowed in TreeTCI graphs").into());
            };
            if !(edge.v < n_sites) {
                return Err(anyhow::anyhow!(
                    "edge endpoint {} is out of bounds for {} sites",
                    edge.v,
                    n_sites
                )
                .into());
            };
            if !(seen.insert(edge)) {
                return Err(anyhow::anyhow!("duplicate edge ({}, {})", edge.u, edge.v).into());
            };
            graph.add_edge(NodeIndex::new(edge.u), NodeIndex::new(edge.v), ());
        }

        if !(graph.edge_count() + 1 == n_sites) {
            return Err(anyhow::anyhow!(
                "TreeTCI graph must be a tree: expected {} edges for {} sites, got {}",
                n_sites.saturating_sub(1),
                n_sites,
                graph.edge_count()
            )
            .into());
        };
        if !(connected_components(&graph) == 1) {
            return Err(anyhow::anyhow!("TreeTCI graph must be connected").into());
        };

        Ok(Self { n_sites, graph })
    }

    /// Return the canonicalized edge if present in the graph.
    /// # Errors
    /// Returns an error when the graph is invalid (an invalid-topology failure).
    ///
    pub fn separate_vertices(&self, edge: TreeTciEdge) -> TreeTciResult<(usize, usize)> {
        if self.has_edge(edge) {
            Ok((edge.u, edge.v))
        } else {
            Err(anyhow!("edge ({}, {}) is not in the graph", edge.u, edge.v).into())
        }
    }

    /// Return the sorted subtree sites when traversing away from `parent`.
    /// # Errors
    /// Returns an error when the graph is invalid (an invalid-topology failure).
    ///
    pub fn subtree_vertices(&self, parent: usize, children: &[usize]) -> TreeTciResult<SubtreeKey> {
        if !(parent < self.n_sites) {
            return Err(anyhow::anyhow!("parent site {} is out of bounds", parent).into());
        };

        let mut sites = Vec::new();
        let mut seen = vec![false; self.n_sites];
        for &child in children {
            if !(child < self.n_sites) {
                return Err(anyhow::anyhow!("child site {} is out of bounds", child).into());
            };
            if !(self.has_edge(TreeTciEdge::new(parent, child))) {
                return Err(
                    anyhow::anyhow!("sites {} and {} are not adjacent", parent, child).into(),
                );
            };
            self.collect_subtree_sites(parent, child, &mut seen, &mut sites);
        }
        sites.sort_unstable();
        Ok(SubtreeKey::new(sites))
    }

    /// Return the two sides of the edge bipartition.
    ///
    /// Removing an edge from a tree splits the sites into two disjoint sets.
    /// Returns `(left_key, right_key)` where left contains the subtree on the
    /// `u` side and right contains the subtree on the `v` side.
    ///
    /// # Errors
    /// Returns an error when the graph is invalid (an invalid-topology failure).
    /// # Examples
    ///
    /// ```
    /// use tensor4all_treetci::{TreeTciEdge, TreeTciGraph};
    ///
    /// // Tree: 0 -- 1 -- 2
    /// let graph = TreeTciGraph::linear_chain(3).unwrap();
    /// let edge = TreeTciEdge::new(0, 1);
    /// let (left, right) = graph.subregion_vertices(edge).unwrap();
    ///
    /// // Removing edge (0,1) splits into {0} and {1,2}
    /// assert_eq!(left.as_slice(), &[0]);
    /// assert_eq!(right.as_slice(), &[1, 2]);
    /// ```
    pub fn subregion_vertices(&self, edge: TreeTciEdge) -> TreeTciResult<(SubtreeKey, SubtreeKey)> {
        let (u, v) = self.separate_vertices(edge)?;
        Ok((
            self.subtree_vertices(v, &[u])?,
            self.subtree_vertices(u, &[v])?,
        ))
    }

    /// Return edges adjacent to a site, excluding any explicitly combined edges.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_treetci::{TreeTciEdge, TreeTciGraph};
    ///
    /// // Star: 0 -- 1, 0 -- 2, 0 -- 3
    /// let graph = TreeTciGraph::new(4, &[
    ///     TreeTciEdge::new(0, 1),
    ///     TreeTciEdge::new(0, 2),
    ///     TreeTciEdge::new(0, 3),
    /// ]).unwrap();
    ///
    /// // All edges adjacent to site 0
    /// let all = graph.adjacent_edges(0, &[]);
    /// assert_eq!(all.len(), 3);
    ///
    /// // Exclude one edge
    /// let filtered = graph.adjacent_edges(0, &[TreeTciEdge::new(0, 2)]);
    /// assert_eq!(filtered.len(), 2);
    /// assert!(!filtered.contains(&TreeTciEdge::new(0, 2)));
    /// ```
    pub fn adjacent_edges(&self, site: usize, combined_edges: &[TreeTciEdge]) -> Vec<TreeTciEdge> {
        if site >= self.n_sites {
            return Vec::new();
        }

        let excluded = combined_edges.iter().copied().collect::<BTreeSet<_>>();
        let mut edges = self
            .graph
            .edges(NodeIndex::new(site))
            .map(|edge_ref| TreeTciEdge::new(edge_ref.source().index(), edge_ref.target().index()))
            .filter(|edge| !excluded.contains(edge))
            .collect::<Vec<_>>();
        edges.sort_unstable();
        edges
    }

    /// Return the candidate edges adjacent to either endpoint of `edge`, excluding `edge`.
    /// # Errors
    /// Returns an error when the graph is invalid (an invalid-topology failure).
    ///
    pub fn candidate_edges(&self, edge: TreeTciEdge) -> TreeTciResult<Vec<TreeTciEdge>> {
        let (u, v) = self.separate_vertices(edge)?;
        let candidates = self
            .adjacent_edges(u, &[edge])
            .into_iter()
            .chain(self.adjacent_edges(v, &[edge]))
            .collect::<BTreeSet<_>>();
        Ok(candidates.into_iter().collect())
    }

    /// Return graph distances from `edge` to every edge in the tree.
    /// # Errors
    /// Returns an error when the graph is invalid (an invalid-topology failure).
    ///
    pub fn distance_edges(&self, edge: TreeTciEdge) -> TreeTciResult<BTreeMap<TreeTciEdge, usize>> {
        let (u, v) = self.separate_vertices(edge)?;
        let mut distances = BTreeMap::new();
        self.collect_edge_distances_from_root(u, v, &mut distances);
        self.collect_edge_distances_from_root(v, u, &mut distances);
        distances.insert(edge, 0);
        Ok(distances)
    }

    /// Return all canonical edges in the tree in sorted order.
    pub fn edges(&self) -> Vec<TreeTciEdge> {
        let mut edges = self
            .graph
            .edge_references()
            .map(|edge_ref| TreeTciEdge::new(edge_ref.source().index(), edge_ref.target().index()))
            .collect::<Vec<_>>();
        edges.sort_unstable();
        edges
    }

    /// Number of sites in the tree.
    pub fn n_sites(&self) -> usize {
        self.n_sites
    }

    /// Return the neighbors of a site in ascending order.
    /// # Errors
    /// Returns an error when the graph is invalid (an invalid-topology failure).
    ///
    pub fn neighbors(&self, site: usize) -> TreeTciResult<Vec<usize>> {
        if !(site < self.n_sites) {
            return Err(anyhow::anyhow!("site {} is out of bounds", site).into());
        };
        let mut neighbors = self
            .graph
            .neighbors(NodeIndex::new(site))
            .map(|neighbor| neighbor.index())
            .collect::<Vec<_>>();
        neighbors.sort_unstable();
        Ok(neighbors)
    }

    /// Return the canonical edge between two adjacent sites.
    ///
    /// # Errors
    /// Returns an error when the graph is invalid (an invalid-topology failure).
    /// # Examples
    ///
    /// ```
    /// use tensor4all_treetci::{TreeTciEdge, TreeTciGraph};
    ///
    /// let graph = TreeTciGraph::linear_chain(3).unwrap();
    /// let edge = graph.edge_between(2, 1).unwrap();
    /// assert_eq!(edge, TreeTciEdge::new(1, 2));
    ///
    /// // Non-adjacent sites produce an error
    /// assert!(graph.edge_between(0, 2).is_err());
    /// ```
    pub fn edge_between(&self, a: usize, b: usize) -> TreeTciResult<TreeTciEdge> {
        let edge = TreeTciEdge::new(a, b);
        self.separate_vertices(edge)?;
        Ok(edge)
    }

    /// Return `(parents, distances)` from a BFS rooted at `root`.
    ///
    /// `parents[i]` is `Some(parent)` for all nodes except the root.
    /// `distances[i]` is the graph distance from `root` to node `i`.
    ///
    /// # Errors
    /// Returns an error when the graph is invalid (an invalid-topology failure).
    /// # Examples
    ///
    /// ```
    /// use tensor4all_treetci::TreeTciGraph;
    ///
    /// // Chain: 0 -- 1 -- 2 -- 3
    /// let graph = TreeTciGraph::linear_chain(4).unwrap();
    /// let (parents, distances) = graph.bfs_tree(0).unwrap();
    ///
    /// assert_eq!(parents[0], None);       // root has no parent
    /// assert_eq!(parents[1], Some(0));
    /// assert_eq!(parents[2], Some(1));
    /// assert_eq!(parents[3], Some(2));
    /// assert_eq!(distances, vec![0, 1, 2, 3]);
    /// ```
    pub fn bfs_tree(&self, root: usize) -> TreeTciResult<(Vec<Option<usize>>, Vec<usize>)> {
        if !(root < self.n_sites) {
            return Err(anyhow::anyhow!("root site {} is out of bounds", root).into());
        };
        let mut parents = vec![None; self.n_sites];
        let mut distances = vec![usize::MAX; self.n_sites];
        let mut queue = VecDeque::from([root]);
        distances[root] = 0;

        while let Some(current) = queue.pop_front() {
            for neighbor in self.neighbors(current)? {
                if distances[neighbor] == usize::MAX {
                    parents[neighbor] = Some(current);
                    distances[neighbor] = distances[current] + 1;
                    queue.push_back(neighbor);
                }
            }
        }

        Ok((parents, distances))
    }

    /// Return subtree keys for the incoming direction at `site` from the given edges.
    /// # Errors
    /// Returns an error when the graph is invalid (an invalid-topology failure).
    ///
    pub fn edge_in_ij_keys(
        &self,
        site: usize,
        edges: &[TreeTciEdge],
    ) -> TreeTciResult<Vec<SubtreeKey>> {
        if !(site < self.n_sites) {
            return Err(anyhow::anyhow!("site {} is out of bounds", site).into());
        };

        edges
            .iter()
            .map(|&edge| {
                let (u, v) = self.separate_vertices(edge)?;
                if u == site {
                    self.subtree_vertices(u, &[v])
                } else if v == site {
                    self.subtree_vertices(v, &[u])
                } else {
                    Err(anyhow!("edge ({}, {}) is not adjacent to site {}", u, v, site).into())
                }
            })
            .collect()
    }

    fn has_edge(&self, edge: TreeTciEdge) -> bool {
        self.graph
            .find_edge(NodeIndex::new(edge.u), NodeIndex::new(edge.v))
            .is_some()
    }

    fn collect_subtree_sites(
        &self,
        parent: usize,
        child: usize,
        seen: &mut [bool],
        out: &mut Vec<usize>,
    ) {
        let mut stack = vec![(parent, child)];
        while let Some((previous, current)) = stack.pop() {
            if seen[current] {
                continue;
            }
            seen[current] = true;
            out.push(current);
            for neighbor in self.graph.neighbors(NodeIndex::new(current)) {
                let neighbor = neighbor.index();
                if neighbor != previous {
                    stack.push((current, neighbor));
                }
            }
        }
    }

    /// Create a linear chain graph: 0--1--2--...--(*n*-1).
    ///
    /// This is a convenience constructor for the most common topology.
    ///
    /// # Errors
    /// Returns an error when the graph is invalid (an invalid-topology failure).
    /// # Examples
    ///
    /// ```
    /// use tensor4all_treetci::TreeTciGraph;
    ///
    /// let chain = TreeTciGraph::linear_chain(4).unwrap();
    /// assert_eq!(chain.n_sites(), 4);
    /// assert_eq!(chain.edges().len(), 3);
    /// assert_eq!(chain.neighbors(1).unwrap(), vec![0, 2]);
    ///
    /// // Single site has no edges
    /// let single = TreeTciGraph::linear_chain(1).unwrap();
    /// assert_eq!(single.n_sites(), 1);
    /// assert_eq!(single.edges().len(), 0);
    ///
    /// // Zero sites is an error
    /// assert!(TreeTciGraph::linear_chain(0).is_err());
    /// ```
    pub fn linear_chain(n_sites: usize) -> TreeTciResult<Self> {
        if n_sites == 0 {
            return Err(anyhow::anyhow!("linear_chain requires at least 1 site").into());
        }
        let edges: Vec<TreeTciEdge> = (0..n_sites.saturating_sub(1))
            .map(|i| TreeTciEdge::new(i, i + 1))
            .collect();
        Self::new(n_sites, &edges)
    }

    fn collect_edge_distances_from_root(
        &self,
        root: usize,
        blocked: usize,
        distances: &mut BTreeMap<TreeTciEdge, usize>,
    ) {
        let mut queue = VecDeque::from([(root, root, 0usize)]);
        let mut seen = vec![false; self.n_sites];
        seen[blocked] = true;

        while let Some((parent, current, dist)) = queue.pop_front() {
            if seen[current] {
                continue;
            }
            seen[current] = true;
            if current != root {
                distances.insert(TreeTciEdge::new(parent, current), dist);
            }
            for neighbor in self.graph.neighbors(NodeIndex::new(current)) {
                let neighbor = neighbor.index();
                if !seen[neighbor] {
                    queue.push_back((current, neighbor, dist + 1));
                }
            }
        }
    }
}

#[cfg(test)]
mod tests;
