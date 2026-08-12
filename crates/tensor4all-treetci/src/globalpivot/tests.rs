use super::find_global_pivots;
use crate::{
    materialize::to_treetn, optimize_with_proposer, DefaultProposer, GlobalIndexBatch, TreeTCI2,
    TreeTciEdge, TreeTciGraph, TreeTciOptions,
};
use anyhow::Result;
use num_complex::Complex64;
use tensor4all_core::TensorDynLen;
use tensor4all_treetn::TreeTN;

/// Batch evaluator for a two-peak function on a 10-site chain.
///
/// f(idx) = 3 * exp(-beta * d2(idx, A)) + 2 * exp(-beta * d2(idx, B)) with
/// d2(x, y) = |x - y|^2. Each single Gaussian is exactly rank 1, so the
/// full function needs rank 2 — but only if both peaks are sampled. A sweep
/// seeded only at A walks toward B one coordinate at a time, and the walk
/// stalls once intermediate |f| drops below the absolute tolerance, so the
/// near-degenerate second peak is silently lost (the gw-rs defect).
const N: usize = 10;
const PEAK_A: [usize; N] = [0; N];
const PEAK_B: [usize; N] = [3; N];
const BETA: f64 = 0.5;

fn peak_value(index: &[usize]) -> f64 {
    let d2 = |peak: &[usize; N]| -> f64 {
        (0..N)
            .map(|i| {
                let d = index[i] as f64 - peak[i] as f64;
                d * d
            })
            .sum()
    };
    3.0 * (-BETA * d2(&PEAK_A)).exp() + 2.0 * (-BETA * d2(&PEAK_B)).exp()
}

fn evaluate(batch: GlobalIndexBatch<'_>) -> Result<Vec<f64>> {
    let mut values = Vec::with_capacity(batch.n_points());
    for point in 0..batch.n_points() {
        let mut index = [0usize; N];
        for (site, slot) in index.iter_mut().enumerate() {
            *slot = batch.get(site, point).unwrap();
        }
        values.push(peak_value(&index));
    }
    Ok(values)
}

fn chain_graph() -> TreeTciGraph {
    let edges: Vec<TreeTciEdge> = (0..N - 1).map(|i| TreeTciEdge::new(i, i + 1)).collect();
    TreeTciGraph::new(N, &edges).unwrap()
}

fn options(enable_global_pivots: bool) -> TreeTciOptions {
    TreeTciOptions {
        tolerance: 1e-8,
        max_iter: 30,
        max_bond_dim: None,
        normalize_error: true,
        enable_global_pivots,
        nsearch: 10,
        max_nglobal_pivot: 5,
        tol_margin_global_search: 1.0,
        seed: Some(42),
    }
}

fn seeded_state() -> TreeTCI2<f64> {
    let mut tci = TreeTCI2::<f64>::new(vec![4; N], chain_graph()).unwrap();
    // Seed the sweep in the first basin only.
    tci.add_global_pivots(&[PEAK_A.to_vec()]).unwrap();
    let flat: Vec<usize> = PEAK_A.to_vec();
    let init_batch = GlobalIndexBatch::new(&flat, N, 1).unwrap();
    let init_values = evaluate(init_batch).unwrap();
    tci.max_sample_value = init_values.iter().copied().fold(0.0f64, f64::max);
    tci
}

/// Evaluate the materialized tree at a full-site index.
fn tree_value(tree: &TreeTN<TensorDynLen, usize>, index: &[usize]) -> f64 {
    let mut site_indices = Vec::with_capacity(index.len());
    for site in 0..index.len() {
        let node = tree.node_index(&site).unwrap();
        let tensor = tree.tensor(node).unwrap();
        site_indices.push(tensor.indices()[0].clone());
    }
    tree.evaluate_point(&site_indices, index).unwrap().real()
}

fn run(enable_global_pivots: bool) -> (Vec<usize>, Vec<f64>, f64, f64) {
    let mut tci = seeded_state();
    let opts = options(enable_global_pivots);
    let (ranks, errors) =
        optimize_with_proposer(&mut tci, evaluate, &opts, &DefaultProposer).unwrap();
    let tree = to_treetn(&tci, evaluate, None).unwrap();
    (
        ranks,
        errors,
        tree_value(&tree, &PEAK_A),
        tree_value(&tree, &PEAK_B),
    )
}

#[test]
fn global_pivot_search_finds_pivots_with_large_error() {
    // Direct finder check: with a rank-1 state seeded only at peak A, the
    // search must escape the A basin and return points near peak B.
    let tci = seeded_state();
    let pivots =
        find_global_pivots(&tci, evaluate, 10, 5, 1.0, 1e-8 * tci.max_sample_value, 42).unwrap();

    let closer_to_b = |pivot: &Vec<usize>| {
        let d2 = |peak: &[usize; N]| {
            (0..N)
                .map(|i| {
                    let d = pivot[i] as i64 - peak[i] as i64;
                    d * d
                })
                .sum::<i64>()
        };
        d2(&PEAK_B) < d2(&PEAK_A)
    };
    assert!(
        !pivots.is_empty() && pivots.iter().all(closer_to_b),
        "global pivot search must escape the seeded basin, got {pivots:?}"
    );
}

#[test]
fn global_pivots_capture_both_near_degenerate_basins() {
    // Regression for the gw-rs defect (lingrui96/gw-rs#9): with initial
    // pivots confined to one basin, the sweep self-reports convergence while
    // a whole near-degenerate peak silently vanishes. Enabling the global
    // pivot search must recover both peaks.
    let (ranks, errors, value_a, value_b) = run(true);

    assert!(
        errors.last().copied().unwrap_or(1.0) < 1e-6,
        "expected convergence with global pivots, errors={errors:?}"
    );
    assert!(
        (value_a - peak_value(&PEAK_A)).abs() < 1e-6,
        "first basin lost: tree value at A is {value_a}, expected {}",
        peak_value(&PEAK_A)
    );
    assert!(
        (value_b - peak_value(&PEAK_B)).abs() < 1e-6,
        "second basin lost: tree value at B is {value_b}, expected {}",
        peak_value(&PEAK_B)
    );
    assert!(*ranks.last().unwrap() >= 2, "both peaks need rank >= 2");
}

#[test]
fn without_global_pivots_second_basin_is_lost() {
    // Documents why the option exists: the same instance without the global
    // pivot search converges (error estimate near peak A) but the materialized
    // tree evaluates to ~zero at peak B, where f(B) = 2.
    let (_, errors, value_a, value_b) = run(false);

    assert!(errors.last().copied().unwrap_or(1.0) < 1e-6);
    assert!((value_a - peak_value(&PEAK_A)).abs() < 1e-6);
    assert!(
        (value_b - peak_value(&PEAK_B)).abs() > 1.0,
        "expected the second basin to be missed without global pivots, tree value at B is {value_b}"
    );
}

#[test]
fn find_global_pivots_rejects_invalid_parameters() {
    let tci = seeded_state();
    // Non-finite or negative tolerances are rejected.
    assert!(find_global_pivots(&tci, evaluate, 5, 5, 1.0, f64::NAN, 1).is_err());
    assert!(find_global_pivots(&tci, evaluate, 5, 5, -1.0, 1e-8, 1).is_err());
    assert!(find_global_pivots(&tci, evaluate, 5, 5, 1.0, -1.0, 1).is_err());
    // A disabled search (no starting points or no pivot budget) is a no-op.
    assert!(find_global_pivots(&tci, evaluate, 0, 5, 1.0, 1e-8, 1)
        .unwrap()
        .is_empty());
    assert!(find_global_pivots(&tci, evaluate, 5, 0, 1.0, 1e-8, 1)
        .unwrap()
        .is_empty());
}

#[test]
fn find_global_pivots_rejects_bad_batch_length() {
    let tci = seeded_state();
    // Evaluator returns one value regardless of the requested batch size.
    let bad = |_batch: GlobalIndexBatch<'_>| -> Result<Vec<f64>> { Ok(vec![0.0]) };
    assert!(find_global_pivots(&tci, bad, 5, 5, 1.0, 1e-8, 1).is_err());
}

#[test]
fn find_global_pivots_respects_threshold_and_limit() {
    let tci = seeded_state();
    // A huge absolute tolerance rejects every candidate.
    let none = find_global_pivots(&tci, evaluate, 10, 5, 1.0, 1e10, 1).unwrap();
    assert!(none.is_empty());
    // A strict pivot budget truncates the found candidates.
    let few =
        find_global_pivots(&tci, evaluate, 10, 2, 1.0, 1e-8 * tci.max_sample_value, 1).unwrap();
    assert!(!few.is_empty());
    assert!(few.len() <= 2);
}

#[test]
fn find_global_pivots_supports_complex_scalars() {
    let graph = chain_graph();
    let mut tci = TreeTCI2::<Complex64>::new(vec![4; N], graph).unwrap();
    tci.add_global_pivots(&[PEAK_A.to_vec()]).unwrap();

    // Nonzero complex function: mixture + i * mixture / 2.
    let complex_eval = |batch: GlobalIndexBatch<'_>| -> Result<Vec<Complex64>> {
        let mut values = Vec::with_capacity(batch.n_points());
        for point in 0..batch.n_points() {
            let mut index = [0usize; N];
            for (site, slot) in index.iter_mut().enumerate() {
                *slot = batch.get(site, point).unwrap();
            }
            let v = peak_value(&index);
            values.push(Complex64::new(v, 0.5 * v));
        }
        Ok(values)
    };

    let pivots = find_global_pivots(&tci, complex_eval, 10, 5, 1.0, 1e-8, 1).unwrap();
    assert!(!pivots.is_empty());
}
