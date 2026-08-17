//! Deterministic continuous minimum-retracing walks for tree sweeps.

// INVARIANT: this module is the tested scheduler seam consumed by the next
// prepared-problem implementation phase; no public entry point exists yet.
#![allow(dead_code)]

use std::collections::HashSet;

use thiserror::Error;

use crate::TreeAciTraversalStrategy;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct OrientedEdgeStep {
    pub(crate) edge: usize,
    pub(crate) from: usize,
    pub(crate) to: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct OrientedPath {
    pub(crate) steps: Vec<OrientedEdgeStep>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct PathPhase {
    pub(crate) paths: Vec<OrientedPath>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct SweepPlan {
    pub(crate) start: usize,
    pub(crate) forward: Vec<PathPhase>,
    pub(crate) reverse: Vec<PathPhase>,
}

#[derive(Debug, Error)]
pub(crate) enum PathPlanError {
    #[error("a tree scheduler requires at least one node")]
    Empty,
    #[error("edge {edge} has an endpoint outside 0..{node_count}")]
    InvalidEndpoint { edge: usize, node_count: usize },
    #[error("edge {edge} is a self-loop")]
    SelfLoop { edge: usize },
    #[error("the supplied topology is not a connected acyclic tree")]
    NotTree,
    #[error("tree-walk construction violated its connectivity invariant")]
    InvalidCover,
    #[error("invalid sweep plan: {0}")]
    InvalidPlan(&'static str),
}

impl SweepPlan {
    pub(crate) fn for_strategy(
        strategy: TreeAciTraversalStrategy,
        node_count: usize,
        edges: &[(usize, usize)],
        requested_start: Option<usize>,
    ) -> Result<Self, PathPlanError> {
        match strategy {
            TreeAciTraversalStrategy::MinimumRetracingWalk => {
                Self::from_minimum_retracing_walk(node_count, edges, requested_start)
            }
        }
    }

    fn from_minimum_retracing_walk(
        node_count: usize,
        edges: &[(usize, usize)],
        requested_start: Option<usize>,
    ) -> Result<Self, PathPlanError> {
        validate_tree(node_count, edges)?;
        if edges.is_empty() {
            return Ok(Self {
                start: 0,
                forward: Vec::new(),
                reverse: Vec::new(),
            });
        }
        let adjacency = adjacency(node_count, edges);
        let start = match requested_start {
            Some(start) if start < node_count => start,
            Some(_) => {
                return Err(PathPlanError::InvalidEndpoint {
                    edge: 0,
                    node_count,
                })
            }
            None => farthest_node(0, &adjacency).0,
        };
        let end = farthest_node(start, &adjacency).0;
        let spine = path_nodes(start, end, &adjacency)?;
        let mut spine_next = vec![None; node_count];
        for pair in spine.windows(2) {
            spine_next[pair[0]] = Some(pair[1]);
        }
        let steps = continuous_walk(start, &adjacency, &spine_next)?;
        let forward = vec![PathPhase {
            paths: vec![OrientedPath { steps }],
        }];
        let reverse = reverse_pass(&forward);
        let plan = Self {
            start,
            forward,
            reverse,
        };
        plan.validate(node_count, edges)?;
        Ok(plan)
    }

    fn validate(&self, node_count: usize, edges: &[(usize, usize)]) -> Result<(), PathPlanError> {
        validate_tree(node_count, edges)?;
        validate_pass(&self.forward, edges)?;
        validate_pass(&self.reverse, edges)?;
        if self.reverse != reverse_pass(&self.forward) {
            return Err(PathPlanError::InvalidPlan(
                "reverse pass is not the exact inverse of the forward pass",
            ));
        }
        Ok(())
    }

    #[cfg(test)]
    fn path_count(&self) -> usize {
        self.forward.iter().map(|phase| phase.paths.len()).sum()
    }
}

fn adjacency(node_count: usize, edges: &[(usize, usize)]) -> Vec<Vec<(usize, usize)>> {
    let mut adjacency = vec![Vec::new(); node_count];
    for (edge, &(left, right)) in edges.iter().enumerate() {
        adjacency[left].push((right, edge));
        adjacency[right].push((left, edge));
    }
    for neighbors in &mut adjacency {
        neighbors.sort_unstable();
    }
    adjacency
}

fn farthest_node(start: usize, adjacency: &[Vec<(usize, usize)>]) -> (usize, usize) {
    let mut best = (start, 0usize);
    let mut stack = vec![(start, usize::MAX, 0usize)];
    while let Some((node, parent, distance)) = stack.pop() {
        if distance > best.1 || (distance == best.1 && node < best.0) {
            best = (node, distance);
        }
        for &(neighbor, _) in adjacency[node].iter().rev() {
            if neighbor != parent {
                stack.push((neighbor, node, distance + 1));
            }
        }
    }
    best
}

fn path_nodes(
    start: usize,
    end: usize,
    adjacency: &[Vec<(usize, usize)>],
) -> Result<Vec<usize>, PathPlanError> {
    let mut parent = vec![None; adjacency.len()];
    let mut stack = vec![start];
    parent[start] = Some(start);
    while let Some(node) = stack.pop() {
        if node == end {
            break;
        }
        for &(neighbor, _) in adjacency[node].iter().rev() {
            if parent[neighbor].is_none() {
                parent[neighbor] = Some(node);
                stack.push(neighbor);
            }
        }
    }
    if parent[end].is_none() {
        return Err(PathPlanError::NotTree);
    }
    let mut path = vec![end];
    while *path.last().ok_or(PathPlanError::InvalidCover)? != start {
        let node = *path.last().ok_or(PathPlanError::InvalidCover)?;
        path.push(parent[node].ok_or(PathPlanError::InvalidCover)?);
    }
    path.reverse();
    Ok(path)
}

#[derive(Clone, Copy)]
enum WalkAction {
    Explore { node: usize, parent: Option<usize> },
    Move(OrientedEdgeStep),
}

fn continuous_walk(
    start: usize,
    adjacency: &[Vec<(usize, usize)>],
    spine_next: &[Option<usize>],
) -> Result<Vec<OrientedEdgeStep>, PathPlanError> {
    let mut result = Vec::new();
    let mut stack = vec![WalkAction::Explore {
        node: start,
        parent: None,
    }];
    while let Some(action) = stack.pop() {
        match action {
            WalkAction::Move(step) => result.push(step),
            WalkAction::Explore { node, parent } => {
                let designated = spine_next[node];
                let mut actions = Vec::new();
                for &(neighbor, edge) in &adjacency[node] {
                    if Some(neighbor) == parent || Some(neighbor) == designated {
                        continue;
                    }
                    actions.push(WalkAction::Move(OrientedEdgeStep {
                        edge,
                        from: node,
                        to: neighbor,
                    }));
                    actions.push(WalkAction::Explore {
                        node: neighbor,
                        parent: Some(node),
                    });
                    actions.push(WalkAction::Move(OrientedEdgeStep {
                        edge,
                        from: neighbor,
                        to: node,
                    }));
                }
                if let Some(next) = designated {
                    let edge = adjacency[node]
                        .iter()
                        .find_map(|&(neighbor, edge)| (neighbor == next).then_some(edge))
                        .ok_or(PathPlanError::InvalidCover)?;
                    actions.push(WalkAction::Move(OrientedEdgeStep {
                        edge,
                        from: node,
                        to: next,
                    }));
                    actions.push(WalkAction::Explore {
                        node: next,
                        parent: Some(node),
                    });
                }
                stack.extend(actions.into_iter().rev());
            }
        }
    }
    Ok(result)
}

fn reverse_pass(forward: &[PathPhase]) -> Vec<PathPhase> {
    forward
        .iter()
        .rev()
        .map(|phase| PathPhase {
            paths: phase
                .paths
                .iter()
                .rev()
                .map(|path| OrientedPath {
                    steps: path
                        .steps
                        .iter()
                        .rev()
                        .map(|step| OrientedEdgeStep {
                            edge: step.edge,
                            from: step.to,
                            to: step.from,
                        })
                        .collect(),
                })
                .collect(),
        })
        .collect()
}

fn validate_pass(phases: &[PathPhase], edges: &[(usize, usize)]) -> Result<(), PathPlanError> {
    let mut edge_visits = vec![0usize; edges.len()];
    for phase in phases {
        let mut phase_vertices = HashSet::new();
        for path in &phase.paths {
            if path.steps.is_empty() {
                return Err(PathPlanError::InvalidPlan("a path is empty"));
            }
            let mut path_vertices = HashSet::with_capacity(path.steps.len() + 1);
            for (position, step) in path.steps.iter().enumerate() {
                let Some(&(left, right)) = edges.get(step.edge) else {
                    return Err(PathPlanError::InvalidPlan(
                        "a step references an unknown edge",
                    ));
                };
                if !((step.from == left && step.to == right)
                    || (step.from == right && step.to == left))
                {
                    return Err(PathPlanError::InvalidPlan(
                        "a step orientation does not match its edge",
                    ));
                }
                if position > 0 && path.steps[position - 1].to != step.from {
                    return Err(PathPlanError::InvalidPlan(
                        "consecutive path steps are disconnected",
                    ));
                }
                path_vertices.insert(step.from);
                path_vertices.insert(step.to);
                edge_visits[step.edge] = edge_visits[step.edge]
                    .checked_add(1)
                    .ok_or(PathPlanError::InvalidPlan("edge visit count overflowed"))?;
            }
            if !phase_vertices.is_disjoint(&path_vertices) {
                return Err(PathPlanError::InvalidPlan(
                    "paths in one phase share a vertex",
                ));
            }
            phase_vertices.extend(path_vertices);
        }
    }
    if edge_visits.contains(&0) {
        return Err(PathPlanError::InvalidPlan(
            "a directional pass must visit every edge at least once",
        ));
    }
    Ok(())
}

fn validate_tree(node_count: usize, edges: &[(usize, usize)]) -> Result<(), PathPlanError> {
    if node_count == 0 {
        return Err(PathPlanError::Empty);
    }
    if edges.len() != node_count - 1 {
        return Err(PathPlanError::NotTree);
    }
    let mut union_find = UnionFind::new(node_count);
    for (edge, &(left, right)) in edges.iter().enumerate() {
        if left >= node_count || right >= node_count {
            return Err(PathPlanError::InvalidEndpoint { edge, node_count });
        }
        if left == right {
            return Err(PathPlanError::SelfLoop { edge });
        }
        if !union_find.union(left, right) {
            return Err(PathPlanError::NotTree);
        }
    }
    let root = union_find.find(0);
    if (1..node_count).any(|node| union_find.find(node) != root) {
        return Err(PathPlanError::NotTree);
    }
    Ok(())
}

struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<u8>,
}

impl UnionFind {
    fn new(len: usize) -> Self {
        Self {
            parent: (0..len).collect(),
            rank: vec![0; len],
        }
    }

    fn find(&mut self, node: usize) -> usize {
        if self.parent[node] != node {
            self.parent[node] = self.find(self.parent[node]);
        }
        self.parent[node]
    }

    fn union(&mut self, left: usize, right: usize) -> bool {
        let mut left_root = self.find(left);
        let mut right_root = self.find(right);
        if left_root == right_root {
            return false;
        }
        if self.rank[left_root] < self.rank[right_root] {
            std::mem::swap(&mut left_root, &mut right_root);
        }
        self.parent[right_root] = left_root;
        if self.rank[left_root] == self.rank[right_root] {
            self.rank[left_root] += 1;
        }
        true
    }
}

#[cfg(test)]
mod tests;
