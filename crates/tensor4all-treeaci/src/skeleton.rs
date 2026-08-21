use tensor4all_tensorbackend::{solve_matrix_owned, Matrix};

use crate::{
    problem::{DirectedEdgeId, PreparedTreeProblem},
    samples::{PivotPairs, SampleArena, SampleId},
    Result, TreeAciError, TreeAciNode, TreeAciScalar,
};

#[derive(Clone, Debug)]
pub(crate) struct SkeletonTensors<T> {
    pub(crate) node: Vec<Vec<T>>,
    pub(crate) node_shape: Vec<Vec<usize>>,
    pub(crate) gauge: Vec<Matrix<T>>,
}

pub(crate) fn skeleton_tensors<T, V, O>(
    problem: &PreparedTreeProblem<V>,
    arena: &SampleArena,
    pivots: &PivotPairs,
    oracle: &mut O,
) -> Result<SkeletonTensors<T>>
where
    T: TreeAciScalar,
    V: TreeAciNode,
    O: FnMut(&[usize]) -> Result<T>,
{
    let edge_count = problem.directed_edges.len() / 2;
    if pivots.per_edge.len() != edge_count {
        return Err(TreeAciError::InternalInvariant {
            message: "skeleton pivot count differs from prepared edge count",
        });
    }

    let mut node = Vec::with_capacity(problem.node_order.len());
    let mut node_shape = Vec::with_capacity(problem.node_order.len());
    for node_position in 0..problem.node_order.len() {
        let incidents = incident_edges(problem, node_position)?;
        let mut shape = vec![problem.physical[node_position].local_dim];
        for &(edge_number, incoming) in &incidents {
            shape.push(oriented_pivot_ids(pivots, edge_number, incoming)?.len());
        }
        let element_count = checked_product(&shape)?;
        let mut values = vec![T::zero(); element_count];
        for (flat, value) in values.iter_mut().enumerate() {
            let coordinates = decode_mixed_radix(flat, &shape)?;
            let mut incoming_samples = Vec::with_capacity(incidents.len());
            for (axis, &(edge_number, incoming)) in incidents.iter().enumerate() {
                let ids = oriented_pivot_ids(pivots, edge_number, incoming)?;
                incoming_samples.push((incoming, ids[coordinates[axis + 1]]));
            }
            let point = materialize_node_point(
                problem,
                arena,
                node_position,
                coordinates[0],
                &incoming_samples,
            )?;
            *value = oracle(&point)?;
        }
        node.push(values);
        node_shape.push(shape);
    }

    let mut gauge = Vec::with_capacity(edge_count);
    for edge_number in 0..edge_count {
        let pairs = pivots
            .per_edge
            .get(edge_number)
            .ok_or(TreeAciError::InternalInvariant {
                message: "skeleton edge has no pivot pair list",
            })?;
        let rank = pairs.len();
        if rank == 0 {
            return Err(TreeAciError::InternalInvariant {
                message: "skeleton cannot invert an empty pivot block",
            });
        }
        let forward = 2 * edge_number;
        let mut cross = Matrix::zeros(rank, rank);
        for (row, &(left, _)) in pairs.iter().enumerate() {
            for (column, &(_, right)) in pairs.iter().enumerate() {
                let point = arena.materialize_global_point(problem, forward, left, right)?;
                cross[[row, column]] = oracle(&point)?;
            }
        }
        let mut identity = Matrix::zeros(rank, rank);
        for diagonal in 0..rank {
            identity[[diagonal, diagonal]] = T::one();
        }
        let inverse =
            solve_matrix_owned(cross, identity).map_err(|error| TreeAciError::Numerical {
                message: format!("skeleton pivot block solve failed: {error}"),
            })?;
        gauge.push(inverse);
    }

    Ok(SkeletonTensors {
        node,
        node_shape,
        gauge,
    })
}

pub(crate) fn skeleton_evaluate<T, V>(
    tensors: &SkeletonTensors<T>,
    problem: &PreparedTreeProblem<V>,
    sigma: &[usize],
) -> Result<T>
where
    T: TreeAciScalar,
    V: TreeAciNode,
{
    if sigma.len() != problem.node_order.len() || tensors.node.len() != sigma.len() {
        return Err(TreeAciError::PointLengthMismatch {
            expected: problem.node_order.len(),
            actual: sigma.len(),
        });
    }
    for (node, (&coordinate, physical)) in sigma.iter().zip(&problem.physical).enumerate() {
        if coordinate >= physical.local_dim {
            return Err(TreeAciError::PhysicalCoordinateOutOfBounds {
                node,
                coordinate,
                local_dim: physical.local_dim,
            });
        }
    }
    if tensors.gauge.len() * 2 != problem.directed_edges.len() {
        return Err(TreeAciError::InternalInvariant {
            message: "skeleton gauge count differs from prepared edge count",
        });
    }
    let mut forward_states = vec![0; tensors.gauge.len()];
    let mut reverse_states = vec![0; tensors.gauge.len()];
    evaluate_edge_assignments(
        tensors,
        problem,
        sigma,
        0,
        &mut forward_states,
        &mut reverse_states,
    )
}

fn evaluate_edge_assignments<T, V>(
    tensors: &SkeletonTensors<T>,
    problem: &PreparedTreeProblem<V>,
    sigma: &[usize],
    edge_number: usize,
    forward_states: &mut [usize],
    reverse_states: &mut [usize],
) -> Result<T>
where
    T: TreeAciScalar,
    V: TreeAciNode,
{
    if edge_number == tensors.gauge.len() {
        let mut value = T::one();
        for (node_position, &coordinate) in sigma.iter().enumerate() {
            let shape =
                tensors
                    .node_shape
                    .get(node_position)
                    .ok_or(TreeAciError::InternalInvariant {
                        message: "skeleton node shape is missing",
                    })?;
            let incidents = incident_edges(problem, node_position)?;
            if shape.len() != incidents.len() + 1 {
                return Err(TreeAciError::InternalInvariant {
                    message: "skeleton node shape has the wrong incident-edge count",
                });
            }
            let mut flat = coordinate;
            let mut stride = shape[0];
            for (axis, &(edge, _incoming)) in incidents.iter().enumerate() {
                let directed = &problem.directed_edges[2 * edge];
                let state = if problem.node_order[node_position] == directed.from {
                    reverse_states[edge]
                } else {
                    forward_states[edge]
                };
                if state >= shape[axis + 1] {
                    return Err(TreeAciError::InternalInvariant {
                        message: "skeleton bond state exceeds node axis dimension",
                    });
                }
                flat =
                    flat.checked_add(stride.checked_mul(state).ok_or(
                        TreeAciError::SizeOverflow {
                            context: "skeleton node offset",
                        },
                    )?)
                    .ok_or(TreeAciError::SizeOverflow {
                        context: "skeleton node offset",
                    })?;
                stride = stride
                    .checked_mul(shape[axis + 1])
                    .ok_or(TreeAciError::SizeOverflow {
                        context: "skeleton node stride",
                    })?;
            }
            let tensor =
                tensors
                    .node
                    .get(node_position)
                    .ok_or(TreeAciError::InternalInvariant {
                        message: "skeleton node tensor is missing",
                    })?;
            value = value
                * *tensor.get(flat).ok_or(TreeAciError::InternalInvariant {
                    message: "skeleton node offset exceeds tensor storage",
                })?;
        }
        for edge in 0..tensors.gauge.len() {
            value = value * tensors.gauge[edge][[reverse_states[edge], forward_states[edge]]];
        }
        return Ok(value);
    }

    let rank = tensors.gauge[edge_number].nrows();
    let mut value = T::zero();
    for forward in 0..rank {
        for reverse in 0..rank {
            forward_states[edge_number] = forward;
            reverse_states[edge_number] = reverse;
            value = value
                + evaluate_edge_assignments(
                    tensors,
                    problem,
                    sigma,
                    edge_number + 1,
                    forward_states,
                    reverse_states,
                )?;
        }
    }
    Ok(value)
}

fn incident_edges<V: TreeAciNode>(
    problem: &PreparedTreeProblem<V>,
    node_position: usize,
) -> Result<Vec<(usize, DirectedEdgeId)>> {
    let node = problem
        .node_order
        .get(node_position)
        .ok_or(TreeAciError::InternalInvariant {
            message: "skeleton references an unknown node position",
        })?;
    let mut result = Vec::new();
    for edge_number in 0..problem.directed_edges.len() / 2 {
        let forward = &problem.directed_edges[2 * edge_number];
        if &forward.from == node {
            result.push((edge_number, forward.reverse));
        } else if &forward.to == node {
            result.push((edge_number, forward.id));
        }
    }
    Ok(result)
}

fn oriented_pivot_ids(
    pivots: &PivotPairs,
    edge_number: usize,
    directed: DirectedEdgeId,
) -> Result<Vec<SampleId>> {
    let forward = 2 * edge_number;
    if directed == forward {
        Ok(pivots.forward_ids(edge_number))
    } else if directed == forward + 1 {
        Ok(pivots.reverse_ids(edge_number))
    } else {
        Err(TreeAciError::InternalInvariant {
            message: "skeleton directed edge is not an orientation of its edge pair",
        })
    }
}

fn materialize_node_point<V: TreeAciNode>(
    problem: &PreparedTreeProblem<V>,
    arena: &SampleArena,
    node_position: usize,
    local_coordinate: usize,
    incoming: &[(DirectedEdgeId, SampleId)],
) -> Result<Vec<usize>> {
    let physical = problem
        .physical
        .get(node_position)
        .ok_or(TreeAciError::InternalInvariant {
            message: "skeleton node has no physical plan",
        })?;
    if local_coordinate >= physical.local_dim {
        return Err(TreeAciError::PhysicalCoordinateOutOfBounds {
            node: node_position,
            coordinate: local_coordinate,
            local_dim: physical.local_dim,
        });
    }
    let mut point = vec![0; problem.node_order.len()];
    let mut visited = vec![false; problem.node_order.len()];
    point[node_position] = local_coordinate;
    visited[node_position] = true;
    for &(directed, sample) in incoming {
        write_component(problem, arena, directed, sample, &mut point, &mut visited)?;
    }
    if visited.iter().any(|visited| !visited) {
        return Err(TreeAciError::InternalInvariant {
            message: "skeleton node samples do not cover the full tree",
        });
    }
    Ok(point)
}

fn write_component<V: TreeAciNode>(
    problem: &PreparedTreeProblem<V>,
    arena: &SampleArena,
    directed: DirectedEdgeId,
    sample: SampleId,
    point: &mut [usize],
    visited: &mut [bool],
) -> Result<()> {
    let edge = problem
        .directed_edges
        .get(directed)
        .ok_or(TreeAciError::InternalInvariant {
            message: "skeleton component references an unknown directed edge",
        })?;
    let node =
        problem
            .node_positions
            .get(&edge.from)
            .copied()
            .ok_or(TreeAciError::InternalInvariant {
                message: "skeleton component source has no node position",
            })?;
    if visited[node] {
        return Err(TreeAciError::InternalInvariant {
            message: "skeleton component samples overlap",
        });
    }
    let record = arena.record(directed, sample)?;
    visited[node] = true;
    point[node] = record.local_coordinate;
    for &(incoming, child_sample) in &record.incoming {
        write_component(problem, arena, incoming, child_sample, point, visited)?;
    }
    Ok(())
}

fn checked_product(shape: &[usize]) -> Result<usize> {
    shape.iter().try_fold(1usize, |product, &dimension| {
        product
            .checked_mul(dimension)
            .ok_or(TreeAciError::SizeOverflow {
                context: "skeleton tensor elements",
            })
    })
}

fn decode_mixed_radix(mut flat: usize, shape: &[usize]) -> Result<Vec<usize>> {
    let mut coordinates = Vec::with_capacity(shape.len());
    for &dimension in shape {
        if dimension == 0 {
            return Err(TreeAciError::InternalInvariant {
                message: "skeleton tensor has a zero-sized axis",
            });
        }
        coordinates.push(flat % dimension);
        flat /= dimension;
    }
    if flat != 0 {
        return Err(TreeAciError::InternalInvariant {
            message: "skeleton mixed-radix decode overflowed its shape",
        });
    }
    Ok(coordinates)
}

#[cfg(test)]
mod tests;
