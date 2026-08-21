use num_complex::Complex64;
use tensor4all_core::{DynIndex, IdxTensor, IndexLike};
use tensor4all_treetn::TreeTN;

use super::{tree_elementwise, tree_elementwise_batched};
use crate::{hadamard_many, TreeAciOptions, TreeAciTermination};

fn product_tree<T: crate::TreeAciScalar>(
    edges: &[(usize, usize)],
    physical: &[DynIndex],
    offset: f64,
) -> TreeTN<IdxTensor, usize> {
    let bonds = edges
        .iter()
        .map(|_| DynIndex::new_dyn(1))
        .collect::<Vec<_>>();
    let tensors = physical
        .iter()
        .enumerate()
        .map(|(node, site)| {
            let mut indices = vec![site.clone()];
            for (edge, &(left, right)) in edges.iter().enumerate() {
                if left == node || right == node {
                    indices.push(bonds[edge].clone());
                }
            }
            let len = indices.iter().map(IndexLike::dim).product();
            let values = (0..len)
                .map(|coordinate| {
                    <T as tensor4all_core::Scalar>::from_f64(
                        offset + node as f64 + coordinate as f64,
                    )
                })
                .collect();
            IdxTensor::from_dense(indices, values).unwrap()
        })
        .collect();
    TreeTN::from_tensors(tensors, (0..physical.len()).collect()).unwrap()
}

fn aligned_values<T: crate::TreeAciScalar>(
    tree: &TreeTN<IdxTensor, usize>,
    indices: &[DynIndex],
) -> Vec<T> {
    tree.to_dense()
        .unwrap()
        .permute_indices(indices)
        .unwrap()
        .to_vec::<T>()
        .unwrap()
}

fn gauss(peak: usize, value: usize) -> f64 {
    let distance = value as f64 - peak as f64;
    (-0.5 * distance * distance).exp()
}

fn separated_two_peak_tree(n: usize) -> (TreeTN<IdxTensor, usize>, Vec<DynIndex>) {
    let dimension = 4;
    let physical = (0..n)
        .map(|_| DynIndex::new_dyn(dimension))
        .collect::<Vec<_>>();
    let bonds = (0..n - 1).map(|_| DynIndex::new_dyn(2)).collect::<Vec<_>>();
    let mut tensors = Vec::with_capacity(n);
    for site in 0..n {
        let (left_rank, right_rank) = if site == 0 {
            (1, 2)
        } else if site == n - 1 {
            (2, 1)
        } else {
            (2, 2)
        };
        let mut values = vec![0.0; left_rank * dimension * right_rank];
        for right in 0..right_rank {
            for coordinate in 0..dimension {
                for left in 0..left_rank {
                    values[left + left_rank * (coordinate + dimension * right)] =
                        match (site, left, right) {
                            (0, 0, 0) => 3.0 * gauss(0, coordinate),
                            (0, 0, 1) => 2.0 * gauss(3, coordinate),
                            (last, 1, 0) if last == n - 1 => gauss(3, coordinate),
                            (_, 0, 0) => gauss(0, coordinate),
                            (_, 1, 1) => gauss(3, coordinate),
                            _ => 0.0,
                        };
                }
            }
        }
        let mut indices = Vec::with_capacity(3);
        if site > 0 {
            indices.push(bonds[site - 1].clone());
        }
        indices.push(physical[site].clone());
        if site + 1 < n {
            indices.push(bonds[site].clone());
        }
        tensors.push(IdxTensor::from_dense(indices, values).unwrap());
    }
    (
        TreeTN::from_tensors(tensors, (0..n).collect()).unwrap(),
        physical,
    )
}

#[test]
fn native_elementwise_addition_runs_on_path_y_binary_and_degree_four() {
    for edges in [
        vec![(0, 1), (1, 2), (2, 3)],
        vec![(0, 1), (0, 2), (0, 3)],
        vec![(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)],
        vec![(0, 1), (0, 2), (0, 3), (0, 4)],
    ] {
        let physical = (0..=edges.len())
            .map(|_| DynIndex::new_dyn(2))
            .collect::<Vec<_>>();
        let a = product_tree::<f64>(&edges, &physical, 1.0);
        let b = product_tree::<f64>(&edges, &physical, 2.0);
        let reference = a.to_dense().unwrap();
        let indices = reference.indices().to_vec();
        let av = aligned_values::<f64>(&a, &indices);
        let bv = aligned_values::<f64>(&b, &indices);
        let expected = av
            .into_iter()
            .zip(bv)
            .map(|(left, right)| left + right)
            .collect::<Vec<_>>();
        let options = TreeAciOptions {
            enable_global_guard: false,
            ..TreeAciOptions::default()
        };

        let result =
            tree_elementwise(|values: &[f64]| values[0] + values[1], &[a, b], &options).unwrap();
        let actual = aligned_values::<f64>(&result.tree, &indices);
        assert_eq!(actual.len(), expected.len());
        assert!(
            actual
                .iter()
                .zip(&expected)
                .all(|(actual, expected)| (actual - expected).abs() < 1.0e-9),
            "topology={edges:?}, actual={actual:?}, expected={expected:?}, ranks={:?}, errors={:?}",
            result.max_ranks,
            result.max_errors
        );
        assert_eq!(result.diagnostics.edge_ranks.len(), edges.len());
    }
}

#[test]
fn chain_spine_only_return_preserves_values_without_rank_equality() {
    let edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)];
    let physical = (0..6).map(|_| DynIndex::new_dyn(2)).collect::<Vec<_>>();
    let a = product_tree::<f64>(&edges, &physical, 1.0);
    let b = product_tree::<f64>(&edges, &physical, 2.0);
    let reference = a.to_dense().unwrap();
    let indices = reference.indices().to_vec();
    let expected = aligned_values::<f64>(&a, &indices)
        .into_iter()
        .zip(aligned_values::<f64>(&b, &indices))
        .map(|(left, right)| left + right)
        .collect::<Vec<_>>();
    let options = TreeAciOptions {
        enable_global_guard: false,
        ..TreeAciOptions::default()
    };

    let result =
        tree_elementwise(|values: &[f64]| values[0] + values[1], &[a, b], &options).unwrap();
    let actual = aligned_values::<f64>(&result.tree, &indices);
    let max_error = actual
        .iter()
        .zip(&expected)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0, f64::max);

    assert!(max_error < 1.0e-9, "chain max error was {max_error:e}");
    assert_eq!(result.max_ranks.len(), result.max_errors.len());
    assert!(matches!(
        result.termination,
        TreeAciTermination::Converged
            | TreeAciTermination::RankLimited
            | TreeAciTermination::MaxSweeps
    ));
}

#[test]
fn complex_batched_operator_preserves_complex_values() {
    let physical = vec![DynIndex::new_dyn(2), DynIndex::new_dyn(2)];
    let input = product_tree::<Complex64>(&[(0, 1)], &physical, 1.0);
    let options = TreeAciOptions {
        enable_global_guard: false,
        ..TreeAciOptions::default()
    };
    let result = tree_elementwise_batched::<Complex64, _, _>(
        |batch, output| {
            for (point, value) in output.iter_mut().enumerate() {
                *value = batch.get(0, point)? * Complex64::new(0.0, 2.0);
            }
            Ok(())
        },
        std::slice::from_ref(&input),
        &options,
    )
    .unwrap();
    let indices = input.to_dense().unwrap().indices().to_vec();
    let expected = aligned_values::<Complex64>(&input, &indices)
        .into_iter()
        .map(|value| value * Complex64::new(0.0, 2.0))
        .collect::<Vec<_>>();
    let actual = aligned_values::<Complex64>(&result.tree, &indices);
    assert!(actual
        .into_iter()
        .zip(expected)
        .all(|(actual, expected)| (actual - expected).norm() < 1.0e-9));
}

#[test]
fn hadamard_many_handles_one_two_four_and_eight_inputs_in_one_run() {
    let edges = [(0, 1), (0, 2), (0, 3)];
    let physical = (0..4).map(|_| DynIndex::new_dyn(2)).collect::<Vec<_>>();
    let options = TreeAciOptions {
        enable_global_guard: false,
        ..TreeAciOptions::default()
    };
    for count in [1, 2, 4, 8] {
        let inputs = (0..count)
            .map(|input| product_tree::<f64>(&edges, &physical, 1.0 + input as f64))
            .collect::<Vec<_>>();
        let reference = inputs[0].to_dense().unwrap();
        let indices = reference.indices().to_vec();
        let mut expected = vec![1.0; reference.to_vec::<f64>().unwrap().len()];
        for input in &inputs {
            for (product, value) in expected
                .iter_mut()
                .zip(aligned_values::<f64>(input, &indices))
            {
                *product *= value;
            }
        }

        let result = hadamard_many::<f64, _>(&inputs, &options).unwrap();
        let actual = aligned_values::<f64>(&result.tree, &indices);
        assert!(actual.iter().zip(expected).all(|(actual, expected)| {
            let scale = expected.abs().max(1.0);
            (actual - expected).abs() / scale < 1.0e-9
        }));
        assert_eq!(result.max_ranks.len(), result.max_errors.len());
    }
}

#[test]
fn diagnostics_report_candidate_set_sizes_for_every_directed_cut() {
    let edges = [(0, 1), (0, 2), (0, 3)];
    let physical = (0..4).map(|_| DynIndex::new_dyn(2)).collect::<Vec<_>>();
    let options = TreeAciOptions {
        enable_global_guard: false,
        ..TreeAciOptions::default()
    };
    let input = product_tree::<f64>(&edges, &physical, 1.0);
    let result = tree_elementwise(
        |values: &[f64]| values[0],
        std::slice::from_ref(&input),
        &options,
    )
    .unwrap();

    let sizes = &result.diagnostics.candidate_set_sizes;
    assert_eq!(
        sizes.len(),
        2 * edges.len(),
        "one entry per directed cut on a three-leaf star"
    );
    assert!(
        sizes.iter().all(|(_, _, len)| *len > 0),
        "every directed cut must retain at least one candidate"
    );
    // The commit rule replaces a candidate set with the selected pivots, so the
    // two orientations of a cut stay in step with its bond dimension.
    for (edge_number, rank) in result.diagnostics.edge_ranks.iter().enumerate() {
        assert_eq!(sizes[2 * edge_number].2, rank.2);
        assert_eq!(sizes[2 * edge_number + 1].2, rank.2);
    }
}

#[test]
fn global_guard_recovers_a_separated_feature_end_to_end() {
    let (input, physical) = separated_two_peak_tree(10);
    let run = |enable_global_guard, nsearch_global_pivots| {
        tree_elementwise(
            |values: &[f64]| values[0],
            std::slice::from_ref(&input),
            &TreeAciOptions {
                rng_seed: 0,
                enable_global_guard,
                nsearch_global_pivots,
                tolerance: 1.0e-4,
                ..TreeAciOptions::default()
            },
        )
        .unwrap()
    };
    let without_guard = run(false, 5);
    let with_guard = run(true, 30);
    let far_peak = vec![3; physical.len()];
    let near_peak = vec![0; physical.len()];
    let error_without_guard = (without_guard
        .tree
        .evaluate_point(&physical, &far_peak)
        .unwrap()
        .real()
        - 2.0)
        .abs();
    let error_with_guard = (with_guard
        .tree
        .evaluate_point(&physical, &far_peak)
        .unwrap()
        .real()
        - 2.0)
        .abs();
    let near_error_with_guard = (with_guard
        .tree
        .evaluate_point(&physical, &near_peak)
        .unwrap()
        .real()
        - 3.0)
        .abs();

    assert!(error_without_guard > 1.0);
    assert!(error_with_guard < 1.0e-6);
    assert!(near_error_with_guard < 1.0e-6);
    assert!(with_guard
        .global_pivots_found
        .iter()
        .any(|&count| count > 0));
}

/// The guard's recovery guarantee at the **default** search count.
///
/// `enable_global_guard` defaults to `true` and `nsearch_global_pivots` to 5, so
/// this is the configuration an ordinary caller receives. The test above pins
/// `nsearch = 30`, which no default caller ever gets, leaving the default path
/// unexercised. This test deliberately does not override the search count, so
/// that changing the default breaks it.
#[test]
fn global_guard_recovers_a_separated_feature_at_the_default_search_count() {
    let (input, physical) = separated_two_peak_tree(10);
    let options = TreeAciOptions {
        rng_seed: 0,
        tolerance: 1.0e-4,
        ..TreeAciOptions::default()
    };
    assert_eq!(
        options.nsearch_global_pivots, 5,
        "this test must exercise the default search count"
    );
    assert!(options.enable_global_guard);

    let result = tree_elementwise(
        |values: &[f64]| values[0],
        std::slice::from_ref(&input),
        &options,
    )
    .unwrap();

    let far_peak = vec![3; physical.len()];
    let near_peak = vec![0; physical.len()];
    let far_error = (result
        .tree
        .evaluate_point(&physical, &far_peak)
        .unwrap()
        .real()
        - 2.0)
        .abs();
    let near_error = (result
        .tree
        .evaluate_point(&physical, &near_peak)
        .unwrap()
        .real()
        - 3.0)
        .abs();

    assert!(
        far_error < 1.0e-6,
        "far peak not recovered at the default search count: {far_error}"
    );
    assert!(near_error < 1.0e-6, "near peak lost: {near_error}");
    assert!(result.global_pivots_found.iter().any(|&count| count > 0));
}
