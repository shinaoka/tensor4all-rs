//! Native TreeTN initialization for tree ACI state.

use rand::{rngs::StdRng, SeedableRng};
use rand_distr::{Distribution, StandardNormal};
use tensor4all_core::{DynIndex, IdxTensor, IndexLike};
use tensor4all_treetn::TreeTN;

use crate::{
    problem::{DirectedEdgeId, PreparedTreeProblem},
    samples::{CandidateSets, PivotPairs, SampleArena},
    Result, TreeAciError, TreeAciNode, TreeAciOptions, TreeAciScalar,
};

pub(crate) fn initial_edge_ranks<V: TreeAciNode>(
    inputs: &[TreeTN<IdxTensor, V>],
    problem: &PreparedTreeProblem<V>,
    options: &TreeAciOptions<V>,
) -> Result<Vec<usize>> {
    let algebraic_bounds = algebraic_edge_bounds(problem)?;
    let mut ranks = Vec::with_capacity(problem.directed_edges.len() / 2);
    for forward in (0..problem.directed_edges.len()).step_by(2) {
        let edge = &problem.directed_edges[forward];
        let algebraic = algebraic_bounds[forward / 2];
        let rank = if let Some(guess) = &options.initial_guess {
            let graph_edge = guess.edge_between(&edge.from, &edge.to).ok_or_else(|| {
                TreeAciError::InvalidInitialGuess {
                    message: format!("missing edge between {:?} and {:?}", edge.from, edge.to),
                }
            })?;
            guess
                .bond_index(graph_edge)
                .ok_or_else(|| TreeAciError::InvalidInitialGuess {
                    message: "an initial-guess edge has no bond index".into(),
                })?
                .dim()
        } else {
            inputs.iter().try_fold(usize::MAX, |minimum, input| {
                let graph_edge = input.edge_between(&edge.from, &edge.to).ok_or(
                    TreeAciError::InternalInvariant {
                        message: "prepared input is missing an output edge",
                    },
                )?;
                let rank = input
                    .bond_index(graph_edge)
                    .ok_or(TreeAciError::InternalInvariant {
                        message: "prepared input edge has no bond index",
                    })?
                    .dim();
                Ok::<_, TreeAciError>(minimum.min(rank))
            })?
        };
        let configured = options.max_bond_dim.unwrap_or(usize::MAX);
        if options.initial_guess.is_some() && (rank > algebraic || rank > configured) {
            return Err(TreeAciError::InvalidInitialGuess {
                message: format!(
                    "edge {:?}--{:?} rank {rank} exceeds algebraic bound {algebraic} or configured bound {configured}",
                    edge.from, edge.to
                ),
            });
        }
        ranks.push(rank.min(algebraic).min(configured).max(1));
    }
    Ok(ranks)
}

pub(crate) fn algebraic_edge_bounds<V: TreeAciNode>(
    problem: &PreparedTreeProblem<V>,
) -> Result<Vec<usize>> {
    (0..problem.directed_edges.len())
        .step_by(2)
        .map(|forward| {
            let reverse = problem.directed_edges[forward].reverse;
            Ok(component_dimension(problem, forward)?
                .min(component_dimension(problem, reverse)?)
                .max(1))
        })
        .collect()
}

pub(crate) fn validate_initial_guess<T: TreeAciScalar, V: TreeAciNode>(
    guess: &TreeTN<IdxTensor, V>,
    reference: &TreeTN<IdxTensor, V>,
    problem: &PreparedTreeProblem<V>,
    options: &TreeAciOptions<V>,
) -> Result<()> {
    guess
        .validate_tree()
        .map_err(|error| TreeAciError::InvalidInitialGuess {
            message: error.to_string(),
        })?;
    if !reference.same_topology(guess) {
        return Err(TreeAciError::InvalidInitialGuess {
            message: "labeled topology differs from the input topology".into(),
        });
    }
    for node in &problem.node_order {
        if reference.site_space(node) != guess.site_space(node) {
            return Err(TreeAciError::InvalidInitialGuess {
                message: format!("physical indices differ at node {node:?}"),
            });
        }
        let node_index =
            guess
                .node_index(node)
                .ok_or_else(|| TreeAciError::InvalidInitialGuess {
                    message: format!("missing node {node:?}"),
                })?;
        let tensor = guess
            .tensor(node_index)
            .ok_or_else(|| TreeAciError::InvalidInitialGuess {
                message: format!("missing tensor at node {node:?}"),
            })?;
        let elements = checked_product(tensor.indices().iter().map(IndexLike::dim), "guess core")?;
        enforce_limit("core elements", elements, options.max_core_elements)?;
        tensor
            .to_vec::<T>()
            .map_err(|error| TreeAciError::InvalidInitialGuess {
                message: error.to_string(),
            })?;
    }
    guess
        .verify_internal_consistency()
        .map_err(|error| TreeAciError::InvalidInitialGuess {
            message: error.to_string(),
        })
}

pub(crate) fn build_random_output<T: TreeAciScalar, V: TreeAciNode>(
    reference: &TreeTN<IdxTensor, V>,
    problem: &PreparedTreeProblem<V>,
    ranks: &[usize],
    options: &TreeAciOptions<V>,
) -> Result<TreeTN<IdxTensor, V>> {
    let output_bonds = ranks
        .iter()
        .map(|rank| DynIndex::new_dyn(*rank))
        .collect::<Vec<_>>();
    let mut rng = StdRng::seed_from_u64(options.rng_seed);
    let mut tensors = Vec::with_capacity(problem.node_order.len());
    for node in &problem.node_order {
        let node_index = reference
            .node_index(node)
            .ok_or(TreeAciError::InternalInvariant {
                message: "output reference is missing a prepared node",
            })?;
        let reference_tensor =
            reference
                .tensor(node_index)
                .ok_or(TreeAciError::InternalInvariant {
                    message: "output reference is missing a prepared tensor",
                })?;
        let site_space = reference
            .site_space(node)
            .ok_or(TreeAciError::InternalInvariant {
                message: "output reference is missing a physical-index space",
            })?;
        let mut indices = Vec::with_capacity(reference_tensor.indices().len());
        for index in reference_tensor.indices() {
            if site_space.contains(index) {
                indices.push(index.clone());
                continue;
            }
            let replacement = (0..ranks.len()).find_map(|edge_number| {
                let edge = &problem.directed_edges[2 * edge_number];
                if &edge.from != node && &edge.to != node {
                    return None;
                }
                let graph_edge = reference.edge_between(&edge.from, &edge.to)?;
                (reference.bond_index(graph_edge)? == index)
                    .then(|| output_bonds[edge_number].clone())
            });
            indices.push(replacement.ok_or(TreeAciError::InternalInvariant {
                message: "reference tensor has a nonphysical axis with no tree edge",
            })?);
        }
        let elements = checked_product(indices.iter().map(IndexLike::dim), "output core")?;
        enforce_limit("core elements", elements, options.max_core_elements)?;
        let values = (0..elements)
            .map(|_| {
                let value: f64 = StandardNormal.sample(&mut rng);
                tensor4all_core::Scalar::from_f64(value)
            })
            .collect::<Vec<T>>();
        tensors.push(IdxTensor::from_dense(indices, values).map_err(|error| {
            TreeAciError::Numerical {
                message: error.to_string(),
            }
        })?);
    }
    let output = TreeTN::from_tensors(tensors, problem.node_order.clone())?;
    output.verify_internal_consistency()?;
    Ok(output)
}

pub(crate) fn bootstrap_samples<V: TreeAciNode>(
    problem: &PreparedTreeProblem<V>,
    edge_ranks: &[usize],
) -> Result<(SampleArena, CandidateSets, PivotPairs)> {
    let (mut arena, mut candidates) = SampleArena::from_global_seeds(problem, &[])?;
    let targets = edge_ranks
        .iter()
        .flat_map(|rank| [*rank, *rank])
        .collect::<Vec<_>>();
    for edge in 0..problem.directed_edges.len() {
        if targets.get(edge).is_none() {
            return Err(TreeAciError::InternalInvariant {
                message: "initial edge-rank count differs from tree edge count",
            });
        }
        let nodes = component_nodes(problem, edge)?;
        let space = checked_product(
            nodes.iter().map(|node| problem.physical[*node].local_dim),
            "component sample space",
        )?;
        let mut ordinal = 1usize;
        while candidates.ids[edge].len() < targets[edge] && ordinal < space {
            let mut point = vec![0; problem.node_order.len()];
            let mut encoded = ordinal;
            for node in &nodes {
                let dim = problem.physical[*node].local_dim;
                point[*node] = encoded % dim;
                encoded /= dim;
            }
            let id = arena.project_point_onto_edge(problem, edge, &point)?;
            candidates.push_unique(edge, id);
            ordinal += 1;
        }
        if candidates.ids[edge].len() < targets[edge] {
            return Err(TreeAciError::InternalInvariant {
                message: "component sample bootstrap could not reach its algebraic rank",
            });
        }
    }
    for (ids, target) in candidates.ids.iter_mut().zip(targets) {
        ids.truncate(target);
    }
    // INVARIANT: the pivot pairs must be built after truncation, so that
    // `PivotPairs::rank` agrees with `edge_ranks` and with the initialized
    // output bond dimensions. Building them from the untruncated candidate sets
    // would overcount on every edge whose bootstrap overshot its target.
    let mut pivots = PivotPairs::new(edge_ranks.len());
    for edge_number in 0..edge_ranks.len() {
        let forward = &candidates.ids[2 * edge_number];
        let reverse = &candidates.ids[2 * edge_number + 1];
        let rank = forward.len().min(reverse.len());
        pivots.set(
            edge_number,
            (0..rank).map(|k| (forward[k], reverse[k])).collect(),
        );
    }
    Ok((arena, candidates, pivots))
}

fn component_dimension<V: TreeAciNode>(
    problem: &PreparedTreeProblem<V>,
    edge: DirectedEdgeId,
) -> Result<usize> {
    checked_product(
        component_nodes(problem, edge)?
            .into_iter()
            .map(|node| problem.physical[node].local_dim),
        "algebraic cut dimension",
    )
}

fn component_nodes<V: TreeAciNode>(
    problem: &PreparedTreeProblem<V>,
    edge: DirectedEdgeId,
) -> Result<Vec<usize>> {
    fn visit<V: TreeAciNode>(
        problem: &PreparedTreeProblem<V>,
        edge: DirectedEdgeId,
        nodes: &mut Vec<usize>,
    ) -> Result<()> {
        let directed = &problem.directed_edges[edge];
        let node =
            *problem
                .node_positions
                .get(&directed.from)
                .ok_or(TreeAciError::InternalInvariant {
                    message: "component edge source has no prepared node position",
                })?;
        nodes.push(node);
        for incoming in &directed.incoming_to_from {
            visit(problem, *incoming, nodes)?;
        }
        Ok(())
    }
    let mut nodes = Vec::new();
    visit(problem, edge, &mut nodes)?;
    nodes.sort_unstable();
    Ok(nodes)
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

fn enforce_limit(resource: &'static str, requested: usize, limit: usize) -> Result<()> {
    if requested > limit {
        return Err(TreeAciError::ResourceLimit {
            resource,
            requested,
            limit,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests;
