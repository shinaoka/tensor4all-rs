//! Tests for adaptive tree TCI patching.

use std::collections::HashSet;

use anyhow::Result;
use tensor4all_core::TensorDynLen;
use tensor4all_treetn::TreeTN;

use super::{adaptive_crossinterpolate2, AdaptiveTreeTciOptions, AdaptiveTreeTciResult};
use crate::{GlobalIndexBatch, TreeTciEdge, TreeTciGraph, TreeTciOptions};

/// Batch evaluator wrapping a scalar point evaluator.
fn batch_from_point<T: Copy>(
    point_eval: impl Fn(&[usize]) -> T,
) -> impl Fn(GlobalIndexBatch<'_>) -> Result<Vec<T>> {
    move |batch: GlobalIndexBatch<'_>| {
        let mut values = Vec::with_capacity(batch.n_points());
        let mut point = vec![0usize; batch.n_sites()];
        for p in 0..batch.n_points() {
            for (s, slot) in point.iter_mut().enumerate() {
                *slot = batch.get(s, p).unwrap();
            }
            values.push(point_eval(&point));
        }
        Ok(values)
    }
}

/// Evaluate an [`AdaptiveTreeTciResult`] at one full-domain point.
///
/// Returns zero when no patch matches (an omitted zero patch). Fails if more
/// than one patch matches.
fn evaluate_result(result: &AdaptiveTreeTciResult, point: &[usize]) -> f64 {
    let n_sites = result.local_dims.len();
    let matching: Vec<_> = result
        .patches
        .iter()
        .filter(|patch| {
            point.len() == n_sites
                && patch
                    .fixed_values
                    .iter()
                    .enumerate()
                    .all(|(site, fixed)| fixed.is_none_or(|value| point[site] == value))
        })
        .collect();
    assert!(
        matching.len() <= 1,
        "more than one patch matches point {:?}",
        point
    );
    let Some(patch) = matching.first() else {
        return 0.0;
    };

    // Patch-local coordinates: fixed sites -> 0, active sites -> original value.
    let local: Vec<usize> = (0..n_sites)
        .map(|site| match patch.fixed_values[site] {
            Some(_) => 0,
            None => point[site],
        })
        .collect();

    evaluate_treetn(&patch.treetn, &local)
}

/// Evaluate a `TreeTN` whose node names are the TreeTCI site numbers 0..n.
fn evaluate_treetn(treetn: &TreeTN<TensorDynLen, usize>, local: &[usize]) -> f64 {
    let (indices, vertices) = treetn.all_site_indices().unwrap();
    let pos: Vec<usize> = (0..local.len())
        .map(|site| vertices.iter().position(|&name| name == site).unwrap())
        .collect();
    let mut data = vec![0usize; indices.len()];
    for (site, &value) in local.iter().enumerate() {
        data[pos[site]] = value;
    }
    treetn.evaluate_point(&indices, &data).unwrap().real()
}

/// Exhaustively check that the patched result reproduces a scalar source function.
fn assert_reproduces(result: &AdaptiveTreeTciResult, f: &impl Fn(&[usize]) -> f64, tol: f64) {
    let dims = &result.local_dims;
    let mut point = vec![0usize; dims.len()];
    let mut max_diff = 0.0_f64;
    loop {
        let expected = f(&point);
        let got = evaluate_result(result, &point);
        max_diff = max_diff.max((got - expected).abs());
        assert!(
            (got - expected).abs() <= tol * expected.abs().max(1.0),
            "mismatch at {:?}: got {}, expected {}",
            point,
            got,
            expected
        );
        // Increment the multi-index (column-major, last site fastest).
        let mut site = dims.len();
        while site > 0 {
            site -= 1;
            point[site] += 1;
            if point[site] < dims[site] {
                break;
            }
            point[site] = 0;
            if site == 0 {
                let _ = max_diff; // exhaustively covered
                return;
            }
        }
    }
}

fn star3_graph() -> TreeTciGraph {
    TreeTciGraph::new(3, &[TreeTciEdge::new(0, 1), TreeTciEdge::new(0, 2)]).unwrap()
}

// ---------------------------------------------------------------------------
// 1. Input validation
// ---------------------------------------------------------------------------

#[test]
fn rejects_invalid_patch_order() {
    let graph = TreeTciGraph::new(3, &[TreeTciEdge::new(0, 1), TreeTciEdge::new(0, 2)]).unwrap();
    let options = AdaptiveTreeTciOptions {
        tci_options: TreeTciOptions {
            tolerance: 1e-8,
            max_bond_dim: 2,
            ..Default::default()
        },
        patch_order: vec![0, 1], // missing site 2
        ..Default::default()
    };
    let err = adaptive_crossinterpolate2::<f64, _, _>(
        batch_from_point(|_| 1.0),
        vec![2, 2, 2],
        graph,
        vec![],
        options,
        None,
        &crate::DefaultProposer,
    )
    .unwrap_err();
    assert!(
        err.to_string().contains("exact permutation"),
        "unexpected error: {err}"
    );
}

#[test]
fn rejects_out_of_range_pivot() {
    let graph = TreeTciGraph::new(2, &[TreeTciEdge::new(0, 1)]).unwrap();
    let options = AdaptiveTreeTciOptions::default();
    let err = adaptive_crossinterpolate2::<f64, _, _>(
        batch_from_point(|_| 1.0),
        vec![2, 2],
        graph,
        vec![vec![0, 5]], // j out of range
        options,
        None,
        &crate::DefaultProposer,
    )
    .unwrap_err();
    assert!(
        err.to_string().contains("outside its site dimension"),
        "unexpected error: {err}"
    );
}

#[test]
fn rejects_zero_n_initial_pivots() {
    let graph = TreeTciGraph::new(2, &[TreeTciEdge::new(0, 1)]).unwrap();
    let options = AdaptiveTreeTciOptions {
        n_initial_pivots: 0,
        ..Default::default()
    };
    let err = adaptive_crossinterpolate2::<f64, _, _>(
        batch_from_point(|_| 1.0),
        vec![2, 2],
        graph,
        vec![],
        options,
        None,
        &crate::DefaultProposer,
    )
    .unwrap_err();
    assert!(
        err.to_string().contains("n_initial_pivots"),
        "unexpected error: {err}"
    );
}

#[test]
fn rejects_graph_dimension_mismatch() {
    let graph = TreeTciGraph::new(2, &[TreeTciEdge::new(0, 1)]).unwrap();
    let options = AdaptiveTreeTciOptions::default();
    let err = adaptive_crossinterpolate2::<f64, _, _>(
        batch_from_point(|_| 1.0),
        vec![2, 2, 2], // 3 dims but 2-site graph
        graph,
        vec![],
        options,
        None,
        &crate::DefaultProposer,
    )
    .unwrap_err();
    assert!(
        err.to_string().contains("graph site count"),
        "unexpected error: {err}"
    );
}

// ---------------------------------------------------------------------------
// 2. No patching required
// ---------------------------------------------------------------------------

#[test]
fn low_rank_function_needs_one_patch() {
    // Product function on the 3-site star: separable, rank one, no patching.
    let f = |idx: &[usize]| idx.iter().fold(1.0, |acc, &x| acc * (x as f64 + 1.0));

    let options = AdaptiveTreeTciOptions {
        tci_options: TreeTciOptions {
            tolerance: 1e-12,
            max_iter: 8,
            max_bond_dim: 2,
            normalize_error: true,
        },
        ..Default::default()
    };

    let result = adaptive_crossinterpolate2::<f64, _, _>(
        batch_from_point(f),
        vec![2, 2, 2],
        star3_graph(),
        vec![vec![0, 0, 0]],
        options,
        Some(0),
        &crate::DefaultProposer,
    )
    .unwrap();

    assert_eq!(result.patches.len(), 1, "expected a single accepted patch");
    let patch = &result.patches[0];
    assert!(
        patch.fixed_values.iter().all(|v| v.is_none()),
        "the single patch must fix no sites"
    );
    assert!(
        patch.final_error <= 1e-12,
        "final error {} must satisfy tolerance",
        patch.final_error
    );
    assert_reproduces(&result, &f, 1e-10);
}

// ---------------------------------------------------------------------------
// 3. Forced chain patching
// ---------------------------------------------------------------------------

#[test]
fn two_site_identity_forces_chain_patching() {
    // f(i, j) = 1 if i == j else 0. The 2x2 identity has rank 2, so max_bond 1
    // cannot fit it and site 0 must split into two rank-one children.
    let f = |idx: &[usize]| if idx[0] == idx[1] { 1.0 } else { 0.0 };

    let graph = TreeTciGraph::new(2, &[TreeTciEdge::new(0, 1)]).unwrap();
    let options = AdaptiveTreeTciOptions {
        tci_options: TreeTciOptions {
            tolerance: 1e-10,
            max_iter: 5,
            max_bond_dim: 1,
            normalize_error: true,
        },
        patch_order: vec![0, 1],
        ..Default::default()
    };

    let result = adaptive_crossinterpolate2::<f64, _, _>(
        batch_from_point(f),
        vec![2, 2],
        graph,
        vec![vec![0, 0], vec![1, 1]],
        options,
        None,
        &crate::DefaultProposer,
    )
    .unwrap();

    assert_eq!(result.patches.len(), 2, "expected two patches");
    for patch in &result.patches {
        assert!(
            patch.fixed_values[0].is_some(),
            "each patch must fix site 0"
        );
        assert!(
            patch.final_error <= 1e-10,
            "patch must converge: error {}",
            patch.final_error
        );
    }
    // The two patches fix site 0 to distinct values (mutually disjoint).
    let fixed0: HashSet<usize> = result
        .patches
        .iter()
        .map(|p| p.fixed_values[0].unwrap())
        .collect();
    assert_eq!(fixed0, HashSet::from([0, 1]));

    assert_reproduces(&result, &f, 1e-8);
}

// ---------------------------------------------------------------------------
// 4. Actual branching tree
// ---------------------------------------------------------------------------

#[test]
fn branching_star_forces_patching() {
    // 3-site star: 0--1, 0--2. f = delta(x0,x1)*delta(x0,x2). Not rank one on the
    // tree, so the root must split. Fixing x0 yields rank-one children.
    let f = |idx: &[usize]| {
        if idx[0] == idx[1] && idx[0] == idx[2] {
            1.0
        } else {
            0.0
        }
    };

    let options = AdaptiveTreeTciOptions {
        tci_options: TreeTciOptions {
            tolerance: 1e-10,
            max_iter: 5,
            max_bond_dim: 1,
            normalize_error: true,
        },
        patch_order: vec![0, 1, 2],
        ..Default::default()
    };

    let result = adaptive_crossinterpolate2::<f64, _, _>(
        batch_from_point(f),
        vec![2, 2, 2],
        star3_graph(),
        vec![vec![0, 0, 0], vec![1, 1, 1]],
        options,
        Some(0),
        &crate::DefaultProposer,
    )
    .unwrap();

    assert!(
        result.patches.len() >= 2,
        "expected patching to occur on a branching tree"
    );
    // Splitting x0 should produce rank-one children fixing site 0.
    assert!(
        result.patches.iter().all(|p| p.fixed_values[0].is_some()),
        "every patch should fix site 0"
    );
    for patch in &result.patches {
        assert!(
            patch.final_error <= 1e-10,
            "patch must converge: error {}",
            patch.final_error
        );
    }
    assert_reproduces(&result, &f, 1e-8);
}

// ---------------------------------------------------------------------------
// 5. Zero child
// ---------------------------------------------------------------------------

#[test]
fn zero_split_child_is_omitted() {
    // 2-site chain, x0 in {0,1,2}, x1 in {0,1}. f matrix:
    //   row 0: [1, 0]
    //   row 1: [0, 1]
    //   row 2: [0, 0]   <- identically zero split child
    // Rank 2, so max_bond 1 forces patching; the x0=2 child is zero and omitted.
    let f = |idx: &[usize]| {
        if idx[0] == 0 {
            if idx[1] == 0 {
                1.0
            } else {
                0.0
            }
        } else if idx[0] == 1 {
            if idx[1] == 1 {
                1.0
            } else {
                0.0
            }
        } else {
            0.0
        }
    };

    let graph = TreeTciGraph::new(2, &[TreeTciEdge::new(0, 1)]).unwrap();
    let options = AdaptiveTreeTciOptions {
        tci_options: TreeTciOptions {
            tolerance: 1e-10,
            max_iter: 5,
            max_bond_dim: 1,
            normalize_error: true,
        },
        patch_order: vec![0, 1],
        n_initial_pivots: 6,
        ..Default::default()
    };

    let result = adaptive_crossinterpolate2::<f64, _, _>(
        batch_from_point(f),
        vec![3, 2],
        graph,
        vec![vec![0, 0], vec![1, 1]],
        options,
        None,
        &crate::DefaultProposer,
    )
    .unwrap();

    // Two nonzero patches (x0=0, x0=1); the x0=2 zero child is omitted.
    assert_eq!(result.patches.len(), 2, "expected the zero child omitted");
    let fixed0: HashSet<usize> = result
        .patches
        .iter()
        .map(|p| p.fixed_values[0].unwrap())
        .collect();
    assert_eq!(fixed0, HashSet::from([0, 1]));

    assert_reproduces(&result, &f, 1e-8);
}

#[test]
fn entirely_zero_function_yields_empty_result() {
    let f = |_: &[usize]| 0.0;
    let graph = TreeTciGraph::new(2, &[TreeTciEdge::new(0, 1)]).unwrap();
    let options = AdaptiveTreeTciOptions {
        tci_options: TreeTciOptions {
            tolerance: 1e-10,
            max_iter: 3,
            max_bond_dim: 2,
            ..Default::default()
        },
        ..Default::default()
    };

    let result = adaptive_crossinterpolate2::<f64, _, _>(
        batch_from_point(f),
        vec![2, 2],
        graph,
        vec![vec![0, 0]],
        options,
        None,
        &crate::DefaultProposer,
    )
    .unwrap();

    assert!(result.is_empty(), "zero function should yield no patches");
}
