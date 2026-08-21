//! Validated topology and physical-index preparation for tree ACI.

#![allow(dead_code)]

use std::collections::HashMap;

use tensor4all_core::{DynIndex, IdxTensor, IndexLike};
use tensor4all_treetn::TreeTN;

use crate::{path_cover::SweepPlan, Result, TreeAciError, TreeAciNode, TreeAciOptions};

pub(crate) type DirectedEdgeId = usize;

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct DirectedEdge<V> {
    pub(crate) id: DirectedEdgeId,
    pub(crate) from: V,
    pub(crate) to: V,
    pub(crate) reverse: DirectedEdgeId,
    pub(crate) incoming_to_from: Vec<DirectedEdgeId>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct LocalPhysicalPlan {
    pub(crate) indices: Vec<DynIndex>,
    pub(crate) dims: Vec<usize>,
    pub(crate) strides: Vec<usize>,
    pub(crate) local_dim: usize,
}

#[derive(Clone, Debug)]
pub(crate) struct PreparedTreeProblem<V> {
    pub(crate) node_order: Vec<V>,
    pub(crate) node_positions: HashMap<V, usize>,
    pub(crate) directed_edges: Vec<DirectedEdge<V>>,
    pub(crate) physical: Vec<LocalPhysicalPlan>,
    pub(crate) root: V,
    pub(crate) schedule: SweepPlan,
    pub(crate) is_path: bool,
    pub(crate) max_sample_arena_bytes: usize,
    pub(crate) max_frame_elements: usize,
    pub(crate) max_frame_bytes: usize,
    pub(crate) max_core_elements: usize,
    pub(crate) max_working_bytes: usize,
}

pub(crate) fn prepare_problem<V: TreeAciNode>(
    inputs: &[TreeTN<IdxTensor, V>],
    options: &TreeAciOptions<V>,
) -> Result<PreparedTreeProblem<V>> {
    validate_options(options)?;
    let reference = inputs.first().ok_or(TreeAciError::NoInputs)?;
    if reference.node_count() == 0 {
        return Err(TreeAciError::EmptyTree { input: 0 });
    }
    reference.validate_tree()?;
    for (input, tree) in inputs.iter().enumerate().skip(1) {
        if tree.node_count() == 0 {
            return Err(TreeAciError::EmptyTree { input });
        }
        tree.validate_tree()?;
        if !reference.same_topology(tree) {
            return Err(TreeAciError::TopologyMismatch { input });
        }
    }

    let mut node_order = reference.node_names();
    node_order.sort();
    let node_positions = node_order
        .iter()
        .cloned()
        .enumerate()
        .map(|(position, node)| (node, position))
        .collect::<HashMap<_, _>>();

    let requested_root = match &options.root {
        Some(root) if node_positions.contains_key(root) => Some(root.clone()),
        Some(_) => {
            return Err(TreeAciError::InvalidOption {
                option: "root",
                message: "the requested root is not a node in the input tree",
            });
        }
        None => None,
    };

    let mut physical = Vec::with_capacity(node_order.len());
    for node in &node_order {
        let reference_space =
            reference
                .site_space(node)
                .ok_or(TreeAciError::InternalInvariant {
                    message: "a validated node is missing its physical-index space",
                })?;
        for (input, tree) in inputs.iter().enumerate().skip(1) {
            if tree.site_space(node) != Some(reference_space) {
                return Err(TreeAciError::PhysicalIndexMismatch {
                    input,
                    node: format!("{node:?}"),
                });
            }
        }

        let node_index = reference
            .node_index(node)
            .ok_or(TreeAciError::InternalInvariant {
                message: "a validated node name has no graph index",
            })?;
        let tensor = reference
            .tensor(node_index)
            .ok_or(TreeAciError::InternalInvariant {
                message: "a validated node has no tensor",
            })?;
        let indices = tensor
            .indices()
            .iter()
            .filter(|index| reference_space.contains(*index))
            .cloned()
            .collect::<Vec<_>>();
        if indices.len() != reference_space.len() {
            return Err(TreeAciError::InternalInvariant {
                message: "physical-index metadata disagrees with tensor axes",
            });
        }
        let dims = indices.iter().map(IndexLike::dim).collect::<Vec<_>>();
        if dims.contains(&0) {
            return Err(TreeAciError::PhysicalIndexMismatch {
                input: 0,
                node: format!("{node:?}"),
            });
        }
        let mut strides = Vec::with_capacity(dims.len());
        let mut local_dim = 1usize;
        for dim in &dims {
            strides.push(local_dim);
            local_dim = local_dim
                .checked_mul(*dim)
                .ok_or(TreeAciError::SizeOverflow {
                    context: "local physical dimension",
                })?;
        }
        enforce_limit("candidate rows", local_dim, options.max_candidate_rows)?;
        enforce_limit("candidate columns", local_dim, options.max_candidate_cols)?;
        physical.push(LocalPhysicalPlan {
            indices,
            dims,
            strides,
            local_dim,
        });
    }

    for tree in inputs {
        for node in &node_order {
            let node_index = tree
                .node_index(node)
                .ok_or(TreeAciError::InternalInvariant {
                    message: "topology-compatible input is missing a node",
                })?;
            let tensor = tree
                .tensor(node_index)
                .ok_or(TreeAciError::InternalInvariant {
                    message: "topology-compatible input is missing a tensor",
                })?;
            let elements = checked_product(tensor.indices().iter().map(IndexLike::dim), "core")?;
            enforce_limit("core elements", elements, options.max_core_elements)?;
        }
    }

    let mut edges = Vec::with_capacity(reference.edge_count());
    for (from_position, from) in node_order.iter().enumerate() {
        let from_index = reference
            .node_index(from)
            .ok_or(TreeAciError::InternalInvariant {
                message: "a validated node name has no graph index",
            })?;
        for (_, neighbor_index) in reference.edges_for_node(from_index) {
            let neighbor = reference
                .site_index_network()
                .node_name(neighbor_index)
                .ok_or(TreeAciError::InternalInvariant {
                    message: "a validated edge endpoint has no node name",
                })?;
            let to_position =
                *node_positions
                    .get(neighbor)
                    .ok_or(TreeAciError::InternalInvariant {
                        message: "a validated edge endpoint has no stable position",
                    })?;
            if from_position < to_position {
                edges.push((from_position, to_position));
            }
        }
    }
    edges.sort_unstable();

    for &(left, right) in &edges {
        let elements = physical[left]
            .local_dim
            .checked_mul(physical[right].local_dim)
            .ok_or(TreeAciError::SizeOverflow {
                context: "minimum local matrix",
            })?;
        enforce_limit(
            "local matrix elements",
            elements,
            options.max_local_matrix_elements,
        )?;
        let bytes = elements
            .checked_mul(std::mem::size_of::<num_complex::Complex64>())
            .ok_or(TreeAciError::SizeOverflow {
                context: "minimum local matrix bytes",
            })?;
        enforce_limit("working bytes", bytes, options.max_working_bytes)?;
    }

    let requested_start = requested_root
        .as_ref()
        .and_then(|root| node_positions.get(root).copied());
    let schedule = SweepPlan::for_strategy(
        options.traversal_strategy,
        node_order.len(),
        &edges,
        requested_start,
    )
    .map_err(|_| TreeAciError::InternalInvariant {
        message: "validated topology could not be lowered to a sweep plan",
    })?;
    let root = node_order[schedule.start].clone();
    let directed_edges = build_directed_edges(&node_order, &edges);
    let mut degrees = vec![0usize; node_order.len()];
    for &(left, right) in &edges {
        degrees[left] += 1;
        degrees[right] += 1;
    }

    Ok(PreparedTreeProblem {
        node_order,
        node_positions,
        directed_edges,
        physical,
        root,
        schedule,
        is_path: degrees.into_iter().all(|degree| degree <= 2),
        max_sample_arena_bytes: options.max_sample_arena_bytes,
        max_frame_elements: options.max_frame_elements,
        max_frame_bytes: options.max_frame_bytes,
        max_core_elements: options.max_core_elements,
        max_working_bytes: options.max_working_bytes,
    })
}

fn validate_options<V: TreeAciNode>(options: &TreeAciOptions<V>) -> Result<()> {
    if options.min_sweeps == 0 || options.max_sweeps < options.min_sweeps {
        return Err(TreeAciError::InvalidOption {
            option: "min_sweeps/max_sweeps",
            message: "sweep bounds must be nonzero and ordered",
        });
    }
    if !options.tolerance.is_finite() || options.tolerance <= 0.0 {
        return Err(TreeAciError::InvalidOption {
            option: "tolerance",
            message: "tolerance must be finite and positive",
        });
    }
    if !options.global_tolerance_margin.is_finite() || options.global_tolerance_margin < 0.0 {
        return Err(TreeAciError::InvalidOption {
            option: "global_tolerance_margin",
            message: "the margin must be finite and nonnegative",
        });
    }
    if options.max_bond_dim == Some(0) {
        return Err(TreeAciError::InvalidOption {
            option: "max_bond_dim",
            message: "a configured bond limit must be positive",
        });
    }
    for (name, limit) in [
        ("max_candidate_rows", options.max_candidate_rows),
        ("max_candidate_cols", options.max_candidate_cols),
        (
            "max_local_matrix_elements",
            options.max_local_matrix_elements,
        ),
        ("max_core_elements", options.max_core_elements),
        ("max_frame_elements", options.max_frame_elements),
        ("max_frame_bytes", options.max_frame_bytes),
        ("max_sample_arena_bytes", options.max_sample_arena_bytes),
        ("max_working_bytes", options.max_working_bytes),
    ] {
        if limit == 0 {
            return Err(TreeAciError::InvalidOption {
                option: name,
                message: "resource limits must be positive",
            });
        }
    }
    Ok(())
}

/// Rejects a request that exceeds its configured ceiling.
///
/// Shared so every allocation site reports the same `ResourceLimit` shape, and
/// so a new site cannot quietly grow its own ad hoc check.
pub(crate) fn enforce_limit(resource: &'static str, requested: usize, limit: usize) -> Result<()> {
    if requested > limit {
        return Err(TreeAciError::ResourceLimit {
            resource,
            requested,
            limit,
        });
    }
    Ok(())
}

fn checked_product(
    values: impl IntoIterator<Item = usize>,
    context: &'static str,
) -> Result<usize> {
    values.into_iter().try_fold(1usize, |product, value| {
        product
            .checked_mul(value)
            .ok_or(TreeAciError::SizeOverflow { context })
    })
}

fn build_directed_edges<V: TreeAciNode>(
    nodes: &[V],
    edges: &[(usize, usize)],
) -> Vec<DirectedEdge<V>> {
    let mut arcs = Vec::with_capacity(2 * edges.len());
    let mut endpoints = Vec::with_capacity(2 * edges.len());
    let mut incoming_by_target = vec![Vec::new(); nodes.len()];
    for (edge, &(left, right)) in edges.iter().enumerate() {
        let forward = 2 * edge;
        let reverse = forward + 1;
        endpoints.push((left, right));
        endpoints.push((right, left));
        incoming_by_target[right].push(forward);
        incoming_by_target[left].push(reverse);
        arcs.push(DirectedEdge {
            id: forward,
            from: nodes[left].clone(),
            to: nodes[right].clone(),
            reverse,
            incoming_to_from: Vec::new(),
        });
        arcs.push(DirectedEdge {
            id: reverse,
            from: nodes[right].clone(),
            to: nodes[left].clone(),
            reverse: forward,
            incoming_to_from: Vec::new(),
        });
    }
    for (id, arc) in arcs.iter_mut().enumerate() {
        let (from, to) = endpoints[id];
        let mut incoming = incoming_by_target[from]
            .iter()
            .copied()
            .filter(|incoming| endpoints[*incoming].0 != to)
            .collect::<Vec<_>>();
        incoming.sort_unstable();
        arc.incoming_to_from = incoming;
    }
    arcs
}

#[cfg(test)]
mod tests {
    use tensor4all_core::{DynIndex, IdxTensor, IndexLike};
    use tensor4all_treetn::TreeTN;

    use super::prepare_problem;
    use crate::{TreeAciError, TreeAciOptions};

    fn make_tree(
        edges: &[(usize, usize)],
        physical: &[Vec<DynIndex>],
        bond_dim: usize,
    ) -> TreeTN<IdxTensor, usize> {
        let bonds = edges
            .iter()
            .map(|_| DynIndex::new_dyn(bond_dim))
            .collect::<Vec<_>>();
        let tensors = (0..physical.len())
            .map(|node| {
                let mut indices = physical[node].clone();
                for (edge, &(left, right)) in edges.iter().enumerate() {
                    if left == node || right == node {
                        indices.push(bonds[edge].clone());
                    }
                }
                let len = indices.iter().map(IndexLike::dim).product();
                IdxTensor::from_dense(indices, vec![1.0; len]).unwrap()
            })
            .collect();
        TreeTN::from_tensors(tensors, (0..physical.len()).collect()).unwrap()
    }

    fn one_site_indices(node_count: usize) -> Vec<Vec<DynIndex>> {
        (0..node_count)
            .map(|_| vec![DynIndex::new_dyn(2)])
            .collect()
    }

    #[test]
    fn accepts_path_y_and_degree_four_topologies() {
        for edges in [
            vec![(0, 1), (1, 2), (2, 3)],
            vec![(0, 1), (0, 2), (0, 3)],
            vec![(0, 1), (0, 2), (0, 3), (0, 4)],
        ] {
            let node_count = edges.len() + 1;
            let tree = make_tree(&edges, &one_site_indices(node_count), 2);
            let prepared = prepare_problem(&[tree], &TreeAciOptions::default()).unwrap();
            assert_eq!(prepared.node_order.len(), node_count);
            assert_eq!(prepared.directed_edges.len(), 2 * edges.len());
        }
    }

    #[test]
    fn directed_edges_record_reverse_and_incoming_branches() {
        let tree = make_tree(&[(0, 1), (0, 2), (0, 3)], &one_site_indices(4), 2);
        let prepared = prepare_problem(&[tree], &TreeAciOptions::default()).unwrap();

        for arc in &prepared.directed_edges {
            let reverse = &prepared.directed_edges[arc.reverse];
            assert_eq!(reverse.from, arc.to);
            assert_eq!(reverse.to, arc.from);
            assert_eq!(reverse.reverse, arc.id);
            assert!(arc
                .incoming_to_from
                .iter()
                .all(|incoming| prepared.directed_edges[*incoming].to == arc.from));
            assert!(arc
                .incoming_to_from
                .iter()
                .all(|incoming| prepared.directed_edges[*incoming].from != arc.to));
        }
        let away_from_center = prepared
            .directed_edges
            .iter()
            .filter(|arc| arc.from == 0)
            .collect::<Vec<_>>();
        assert!(away_from_center
            .iter()
            .all(|arc| arc.incoming_to_from.len() == 2));
        assert!(!prepared.is_path);
    }

    #[test]
    fn physical_layout_follows_reference_tensor_order_and_allows_zero_legs() {
        let scalar_tree = make_tree(&[], &[vec![]], 1);
        let scalar = prepare_problem(&[scalar_tree], &TreeAciOptions::default()).unwrap();
        assert_eq!(scalar.physical[0].local_dim, 1);
        assert!(scalar.physical[0].strides.is_empty());

        let first = DynIndex::new_dyn(2);
        let second = DynIndex::new_dyn(3);
        let tree = make_tree(&[], &[vec![second.clone(), first.clone()]], 1);
        let prepared = prepare_problem(&[tree], &TreeAciOptions::default()).unwrap();
        assert_eq!(prepared.physical[0].indices, vec![second, first]);
        assert_eq!(prepared.physical[0].dims, vec![3, 2]);
        assert_eq!(prepared.physical[0].strides, vec![1, 3]);
        assert_eq!(prepared.physical[0].local_dim, 6);
    }

    #[test]
    fn full_physical_index_equality_is_required() {
        let site = DynIndex::new_dyn(2);
        let reference = make_tree(&[], &[vec![site.clone()]], 1);
        let primed = make_tree(&[], &[vec![site.prime()]], 1);

        assert!(matches!(
            prepare_problem(&[reference, primed], &TreeAciOptions::default()),
            Err(TreeAciError::PhysicalIndexMismatch { input: 1, .. })
        ));
    }

    #[test]
    fn input_bond_dimensions_may_differ() {
        let physical = one_site_indices(2);
        let rank_two = make_tree(&[(0, 1)], &physical, 2);
        let rank_five = make_tree(&[(0, 1)], &physical, 5);
        assert!(prepare_problem(&[rank_two, rank_five], &TreeAciOptions::default()).is_ok());
    }

    #[test]
    fn rejects_missing_empty_mismatched_and_oversized_inputs() {
        assert!(matches!(
            prepare_problem::<usize>(&[], &TreeAciOptions::default()),
            Err(TreeAciError::NoInputs)
        ));
        assert!(matches!(
            prepare_problem::<usize>(&[TreeTN::new()], &TreeAciOptions::default()),
            Err(TreeAciError::EmptyTree { input: 0 })
        ));

        let sites = one_site_indices(3);
        let path = make_tree(&[(0, 1), (1, 2)], &sites, 2);
        let star = make_tree(&[(0, 1), (0, 2)], &sites, 2);
        assert!(matches!(
            prepare_problem(&[path, star], &TreeAciOptions::default()),
            Err(TreeAciError::TopologyMismatch { input: 1 })
        ));

        let tree = make_tree(&[], &[vec![DynIndex::new_dyn(8)]], 1);
        let options = TreeAciOptions {
            max_core_elements: 7,
            ..TreeAciOptions::default()
        };
        assert!(matches!(
            prepare_problem(&[tree], &options),
            Err(TreeAciError::ResourceLimit {
                resource: "core elements",
                requested: 8,
                limit: 7
            })
        ));
    }

    #[test]
    fn rejects_cycles_zero_dimensions_and_unknown_roots() {
        let b01 = DynIndex::new_dyn(2);
        let b12 = DynIndex::new_dyn(2);
        let b20 = DynIndex::new_dyn(2);
        let tensors = vec![
            IdxTensor::from_dense(vec![b01.clone(), b20.clone()], vec![1.0; 4]).unwrap(),
            IdxTensor::from_dense(vec![b01.clone(), b12.clone()], vec![1.0; 4]).unwrap(),
            IdxTensor::from_dense(vec![b12.clone(), b20.clone()], vec![1.0; 4]).unwrap(),
        ];
        let mut cycle = TreeTN::new();
        let nodes = tensors
            .into_iter()
            .enumerate()
            .map(|(name, tensor)| cycle.add_tensor(name, tensor).unwrap())
            .collect::<Vec<_>>();
        cycle.connect(nodes[0], &b01, nodes[1], &b01).unwrap();
        cycle.connect(nodes[1], &b12, nodes[2], &b12).unwrap();
        cycle.connect(nodes[2], &b20, nodes[0], &b20).unwrap();
        assert!(matches!(
            prepare_problem(&[cycle], &TreeAciOptions::default()),
            Err(TreeAciError::TreeTN(_))
        ));

        let zero = DynIndex::new_dyn(0);
        assert!(IdxTensor::from_dense(vec![zero], Vec::<f64>::new()).is_err());

        let tree = make_tree(&[(0, 1)], &one_site_indices(2), 2);
        let options = TreeAciOptions {
            root: Some(9),
            ..TreeAciOptions::default()
        };
        assert!(matches!(
            prepare_problem(&[tree], &options),
            Err(TreeAciError::InvalidOption { option: "root", .. })
        ));
    }
}
