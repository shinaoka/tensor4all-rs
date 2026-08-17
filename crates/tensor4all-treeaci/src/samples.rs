//! Immutable recursive component samples for directed tree cuts.

// INVARIANT: this staged internal state becomes live through Task 5 frame
// construction and Task 7 initialization; until then only its unit tests use it.
#![allow(dead_code)]

use std::{collections::HashMap, mem::size_of};

use crate::{
    problem::{DirectedEdgeId, PreparedTreeProblem},
    Result, TreeAciError, TreeAciNode,
};

pub(crate) type SampleId = usize;

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct ComponentSampleKey {
    local_coordinate: usize,
    incoming: Vec<(DirectedEdgeId, SampleId)>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ComponentSample {
    pub(crate) local_coordinate: usize,
    pub(crate) incoming: Vec<(DirectedEdgeId, SampleId)>,
}

#[derive(Clone, Debug, Default)]
struct DirectedSampleArena {
    records: Vec<ComponentSample>,
    dedup: HashMap<ComponentSampleKey, SampleId>,
}

#[derive(Clone, Debug)]
pub(crate) struct SampleArena {
    directed: Vec<DirectedSampleArena>,
    retained_bytes: usize,
    max_retained_bytes: usize,
}

/// Candidate component samples per directed cut.
///
/// These feed the candidate row and column spaces of *neighbouring* edges. They
/// are replaced when their own edge is updated and appended to by global pivot
/// injection. They are deliberately not the same thing as the pivot pairs that
/// set a bond's rank: a duplicate here is a harmless redundant candidate, while
/// a duplicate in a pivot list makes `P_e` singular.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct CandidateSets {
    pub(crate) generation: u64,
    pub(crate) ids: Vec<Vec<SampleId>>,
}

impl CandidateSets {
    pub(crate) fn new(directed_edge_count: usize) -> Self {
        Self {
            generation: 0,
            ids: vec![Vec::new(); directed_edge_count],
        }
    }

    /// Appends `id` to `edge` unless it is already present.
    ///
    /// Returns `true` when the id was appended.
    pub(crate) fn push_unique(&mut self, edge: DirectedEdgeId, id: SampleId) -> bool {
        let ids = &mut self.ids[edge];
        if ids.contains(&id) {
            return false;
        }
        ids.push(id);
        true
    }
}

/// Selected cross pivots per undirected edge.
///
/// Entry `k` of edge `e` is the pair of component samples whose intersection is
/// the `k`-th pivot of `P_e`. The forward and reverse projections therefore have
/// equal length by construction, and neither may contain a repeat.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct PivotPairs {
    pub(crate) per_edge: Vec<Vec<(SampleId, SampleId)>>,
}

impl PivotPairs {
    pub(crate) fn new(edge_count: usize) -> Self {
        Self {
            per_edge: vec![Vec::new(); edge_count],
        }
    }

    pub(crate) fn rank(&self, edge_number: usize) -> usize {
        self.per_edge[edge_number].len()
    }

    pub(crate) fn set(&mut self, edge_number: usize, pairs: Vec<(SampleId, SampleId)>) {
        self.per_edge[edge_number] = pairs;
    }

    pub(crate) fn forward_ids(&self, edge_number: usize) -> Vec<SampleId> {
        self.per_edge[edge_number]
            .iter()
            .map(|(forward, _)| *forward)
            .collect()
    }

    pub(crate) fn reverse_ids(&self, edge_number: usize) -> Vec<SampleId> {
        self.per_edge[edge_number]
            .iter()
            .map(|(_, reverse)| *reverse)
            .collect()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct InjectionReport {
    pub(crate) added_by_edge: Vec<usize>,
    pub(crate) total_added: usize,
}

impl SampleArena {
    pub(crate) fn from_global_seeds<V: TreeAciNode>(
        problem: &PreparedTreeProblem<V>,
        seeds: &[Vec<usize>],
    ) -> Result<(Self, CandidateSets)> {
        let deterministic_seed;
        let seeds = if seeds.is_empty() {
            deterministic_seed = vec![vec![0; problem.node_order.len()]];
            deterministic_seed.as_slice()
        } else {
            seeds
        };
        let mut arena = Self {
            directed: vec![DirectedSampleArena::default(); problem.directed_edges.len()],
            retained_bytes: 0,
            max_retained_bytes: problem.max_sample_arena_bytes,
        };
        let mut candidates = CandidateSets::new(problem.directed_edges.len());
        for point in seeds {
            arena.validate_point(problem, point)?;
            for directed_edge in 0..problem.directed_edges.len() {
                let id = arena.project_component(problem, directed_edge, point)?;
                candidates.push_unique(directed_edge, id);
            }
        }
        Ok((arena, candidates))
    }

    pub(crate) fn inject_global_point<V: TreeAciNode>(
        &mut self,
        candidates: &mut CandidateSets,
        problem: &PreparedTreeProblem<V>,
        point: &[usize],
    ) -> Result<InjectionReport> {
        self.inject_global_point_impl(
            candidates,
            problem,
            point,
            &vec![true; problem.directed_edges.len()],
            false,
        )
    }

    pub(crate) fn inject_global_point_masked<V: TreeAciNode>(
        &mut self,
        candidates: &mut CandidateSets,
        problem: &PreparedTreeProblem<V>,
        point: &[usize],
        activate_directed_cut: &[bool],
    ) -> Result<InjectionReport> {
        self.inject_global_point_impl(candidates, problem, point, activate_directed_cut, true)
    }

    fn inject_global_point_impl<V: TreeAciNode>(
        &mut self,
        candidates: &mut CandidateSets,
        problem: &PreparedTreeProblem<V>,
        point: &[usize],
        activate_directed_cut: &[bool],
        pair_opposite_cuts: bool,
    ) -> Result<InjectionReport> {
        self.validate_point(problem, point)?;
        if candidates.ids.len() != problem.directed_edges.len() {
            return Err(TreeAciError::InternalInvariant {
                message: "candidate-set count differs from directed edge count",
            });
        }
        if activate_directed_cut.len() != problem.directed_edges.len() {
            return Err(TreeAciError::InternalInvariant {
                message: "global-pivot activation mask differs from directed edge count",
            });
        }

        // INVARIANT: failed projection must not leave records that are invisible
        // to the caller's generation. Clone-and-commit is intentionally simple
        // in phase one; later profiling may replace it with a journal rollback.
        let mut proposed = self.clone();
        let projected = (0..problem.directed_edges.len())
            .map(|edge| proposed.project_component(problem, edge, point))
            .collect::<Result<Vec<_>>>()?;
        let mut added_by_edge = vec![0; projected.len()];
        if pair_opposite_cuts {
            for forward in (0..projected.len()).step_by(2) {
                let reverse = problem.directed_edges[forward].reverse;
                if activate_directed_cut[forward]
                    && activate_directed_cut[reverse]
                    && (!candidates.ids[forward].contains(&projected[forward])
                        || !candidates.ids[reverse].contains(&projected[reverse]))
                {
                    added_by_edge[forward] = 1;
                    added_by_edge[reverse] = 1;
                }
            }
        } else {
            for (edge, id) in projected.iter().copied().enumerate() {
                if activate_directed_cut[edge] && !candidates.ids[edge].contains(&id) {
                    added_by_edge[edge] = 1;
                }
            }
        }
        let total_added = added_by_edge.iter().sum();
        if total_added > 0 {
            let next_generation =
                candidates
                    .generation
                    .checked_add(1)
                    .ok_or(TreeAciError::SizeOverflow {
                        context: "sample generation",
                    })?;
            for (edge, id) in projected.into_iter().enumerate() {
                if added_by_edge[edge] == 1 {
                    candidates.ids[edge].push(id);
                }
            }
            candidates.generation = next_generation;
            *self = proposed;
        }
        Ok(InjectionReport {
            added_by_edge,
            total_added,
        })
    }

    pub(crate) fn materialize_global_point<V: TreeAciNode>(
        &self,
        problem: &PreparedTreeProblem<V>,
        forward: DirectedEdgeId,
        left: SampleId,
        right: SampleId,
    ) -> Result<Vec<usize>> {
        let edge = problem
            .directed_edges
            .get(forward)
            .ok_or(TreeAciError::InternalInvariant {
                message: "materialization references an unknown directed edge",
            })?;
        let mut coordinates = vec![0; problem.node_order.len()];
        let mut visited = vec![false; problem.node_order.len()];
        self.write_component(problem, forward, left, &mut coordinates, &mut visited)?;
        self.write_component(problem, edge.reverse, right, &mut coordinates, &mut visited)?;
        if visited.iter().any(|is_visited| !is_visited) {
            return Err(TreeAciError::InternalInvariant {
                message: "two component samples omitted a physical node",
            });
        }
        Ok(coordinates)
    }

    pub(crate) fn record_count(&self) -> usize {
        self.directed.iter().map(|arena| arena.records.len()).sum()
    }

    pub(crate) fn retained_bytes(&self) -> usize {
        self.retained_bytes
    }

    pub(crate) fn directed_record_count(&self, directed_edge: DirectedEdgeId) -> Result<usize> {
        self.directed
            .get(directed_edge)
            .map(|arena| arena.records.len())
            .ok_or(TreeAciError::InternalInvariant {
                message: "sample count requested for an unknown directed edge",
            })
    }

    pub(crate) fn record(
        &self,
        directed_edge: DirectedEdgeId,
        sample: SampleId,
    ) -> Result<&ComponentSample> {
        self.directed
            .get(directed_edge)
            .and_then(|arena| arena.records.get(sample))
            .ok_or(TreeAciError::InternalInvariant {
                message: "component sample ID is not retained by its directed arena",
            })
    }

    pub(crate) fn intern_component<V: TreeAciNode>(
        &mut self,
        problem: &PreparedTreeProblem<V>,
        directed_edge: DirectedEdgeId,
        sample: ComponentSample,
    ) -> Result<SampleId> {
        let edge =
            problem
                .directed_edges
                .get(directed_edge)
                .ok_or(TreeAciError::InternalInvariant {
                    message: "sample insertion references an unknown directed edge",
                })?;
        if sample
            .incoming
            .iter()
            .map(|(incoming, _)| *incoming)
            .ne(edge.incoming_to_from.iter().copied())
        {
            return Err(TreeAciError::InternalInvariant {
                message: "sample insertion has the wrong ordered incoming branches",
            });
        }
        for &(incoming, id) in &sample.incoming {
            self.record(incoming, id)?;
        }
        let key = ComponentSampleKey {
            local_coordinate: sample.local_coordinate,
            incoming: sample.incoming,
        };
        self.intern_key(directed_edge, key)
    }

    fn validate_point<V: TreeAciNode>(
        &self,
        problem: &PreparedTreeProblem<V>,
        point: &[usize],
    ) -> Result<()> {
        if point.len() != problem.node_order.len() {
            return Err(TreeAciError::PointLengthMismatch {
                expected: problem.node_order.len(),
                actual: point.len(),
            });
        }
        for (node, (&coordinate, physical)) in point.iter().zip(&problem.physical).enumerate() {
            if coordinate >= physical.local_dim {
                return Err(TreeAciError::PhysicalCoordinateOutOfBounds {
                    node,
                    coordinate,
                    local_dim: physical.local_dim,
                });
            }
        }
        Ok(())
    }

    fn project_component<V: TreeAciNode>(
        &mut self,
        problem: &PreparedTreeProblem<V>,
        directed_edge: DirectedEdgeId,
        point: &[usize],
    ) -> Result<SampleId> {
        let edge =
            problem
                .directed_edges
                .get(directed_edge)
                .ok_or(TreeAciError::InternalInvariant {
                    message: "component projection references an unknown directed edge",
                })?;
        let node =
            *problem
                .node_positions
                .get(&edge.from)
                .ok_or(TreeAciError::InternalInvariant {
                    message: "directed edge source has no prepared node position",
                })?;
        let incoming_edges = edge.incoming_to_from.clone();
        let mut incoming = Vec::with_capacity(incoming_edges.len());
        for child_edge in incoming_edges {
            let child_id = self.project_component(problem, child_edge, point)?;
            incoming.push((child_edge, child_id));
        }
        let key = ComponentSampleKey {
            local_coordinate: point[node],
            incoming,
        };
        self.intern_key(directed_edge, key)
    }

    fn intern_key(
        &mut self,
        directed_edge: DirectedEdgeId,
        key: ComponentSampleKey,
    ) -> Result<SampleId> {
        if let Some(id) = self.directed[directed_edge].dedup.get(&key) {
            return Ok(*id);
        }
        let added_bytes = logical_record_bytes(key.incoming.len())?;
        let requested =
            self.retained_bytes
                .checked_add(added_bytes)
                .ok_or(TreeAciError::SizeOverflow {
                    context: "sample arena bytes",
                })?;
        if requested > self.max_retained_bytes {
            return Err(TreeAciError::ResourceLimit {
                resource: "sample arena bytes",
                requested,
                limit: self.max_retained_bytes,
            });
        }
        let arena = &mut self.directed[directed_edge];
        let id = arena.records.len();
        arena.records.push(ComponentSample {
            local_coordinate: key.local_coordinate,
            incoming: key.incoming.clone(),
        });
        arena.dedup.insert(key, id);
        self.retained_bytes = requested;
        Ok(id)
    }

    fn write_component<V: TreeAciNode>(
        &self,
        problem: &PreparedTreeProblem<V>,
        directed_edge: DirectedEdgeId,
        sample: SampleId,
        coordinates: &mut [usize],
        visited: &mut [bool],
    ) -> Result<()> {
        let edge =
            problem
                .directed_edges
                .get(directed_edge)
                .ok_or(TreeAciError::InternalInvariant {
                    message: "component record references an unknown directed edge",
                })?;
        let node =
            *problem
                .node_positions
                .get(&edge.from)
                .ok_or(TreeAciError::InternalInvariant {
                    message: "component record source has no node position",
                })?;
        if visited[node] {
            return Err(TreeAciError::InternalInvariant {
                message: "component samples overlap at a physical node",
            });
        }
        let record = self.directed[directed_edge].records.get(sample).ok_or(
            TreeAciError::InternalInvariant {
                message: "component sample ID is not retained by its directed arena",
            },
        )?;
        visited[node] = true;
        coordinates[node] = record.local_coordinate;
        for &(incoming_edge, incoming_sample) in &record.incoming {
            self.write_component(
                problem,
                incoming_edge,
                incoming_sample,
                coordinates,
                visited,
            )?;
        }
        Ok(())
    }
}

fn logical_record_bytes(incoming_count: usize) -> Result<usize> {
    let pair_bytes = incoming_count
        .checked_mul(size_of::<(DirectedEdgeId, SampleId)>())
        .and_then(|bytes| bytes.checked_mul(2))
        .ok_or(TreeAciError::SizeOverflow {
            context: "sample record bytes",
        })?;
    size_of::<ComponentSample>()
        .checked_add(size_of::<ComponentSampleKey>())
        .and_then(|bytes| bytes.checked_add(pair_bytes))
        .ok_or(TreeAciError::SizeOverflow {
            context: "sample record bytes",
        })
}

#[cfg(test)]
mod tests;
