use super::{optimize_default, TreeTciOptions};
use crate::test_support::assert_scalar_close;
use crate::{GlobalIndexBatch, TreeTCI2, TreeTciEdge, TreeTciGraph};
use anyhow::Result;

fn two_site_graph() -> TreeTciGraph {
    TreeTciGraph::new(2, &[TreeTciEdge::new(0, 1)]).unwrap()
}

#[test]
fn optimize_default_converges_on_two_site_identity() {
    let mut tci = TreeTCI2::<f64>::new(vec![2, 2], two_site_graph()).unwrap();
    tci.add_global_pivots(&[vec![0, 0]]).unwrap();

    let batch_eval = |batch: GlobalIndexBatch<'_>| -> Result<Vec<f64>> {
        let mut values = Vec::with_capacity(batch.n_points());
        for point in 0..batch.n_points() {
            let i = batch.get(0, point).unwrap();
            let j = batch.get(1, point).unwrap();
            values.push(if i == j { 1.0 } else { 0.0 });
        }
        Ok(values)
    };

    let (ranks, errors) = optimize_default(
        &mut tci,
        batch_eval,
        &TreeTciOptions {
            tolerance: 1e-12,
            max_iter: 4,
            max_bond_dim: usize::MAX,
            normalize_error: true,
        },
    )
    .unwrap();

    assert_eq!(ranks.last().copied(), Some(2));
    assert_scalar_close(
        errors.last().copied().unwrap_or(f64::NAN),
        0.0,
        tci.max_sample_value,
        1e-12,
    );
    assert_eq!(tci.max_rank(), 2);
}

// Previously named `optimize_default_runs_all_iterations_like_upstream_tree_tci`
// and asserted `ranks.len() == 4` / `errors.len() == 4` (max_iter), pinning
// parity with upstream TreeTCI.jl's sweep loop, which has no early-convergence
// break at all. That upstream behavior is a known bug: see
// ~/gw/CombTCI/COMPATIBILITY.md's `treetci-fix-convergence-criterion.patch`
// (a *different*, scale-mismatch bug in the same convergence-check area,
// found and locally patched by the same user, not yet upstreamed) and
// ~/tensor4all-rust/treetci-optimize-no-early-stop-bug.md for this crate's
// specific issue (no break at all, not just a wrong-scale comparison).
// Renamed and flipped to assert the fixed (early-stopping) behavior instead.
#[test]
fn optimize_default_stops_early_once_converged() {
    let mut tci = TreeTCI2::<f64>::new(vec![2, 2], two_site_graph()).unwrap();
    tci.add_global_pivots(&[vec![0, 0]]).unwrap();

    let batch_eval = |batch: GlobalIndexBatch<'_>| -> Result<Vec<f64>> {
        let mut values = Vec::with_capacity(batch.n_points());
        for point in 0..batch.n_points() {
            let i = batch.get(0, point).unwrap();
            let j = batch.get(1, point).unwrap();
            values.push(if i == j { 1.0 } else { 0.0 });
        }
        Ok(values)
    };

    let (ranks, errors) = optimize_default(
        &mut tci,
        batch_eval,
        &TreeTciOptions {
            tolerance: 1e-12,
            max_iter: 4,
            max_bond_dim: usize::MAX,
            normalize_error: true,
        },
    )
    .unwrap();

    // The 2x2 identity function is exactly rank 2 and converges on the first
    // sweep; the loop must not keep going through the remaining max_iter-1
    // sweeps once the error is already below tolerance.
    assert!(ranks.len() < 4);
    assert_eq!(ranks.len(), errors.len());
    assert_eq!(ranks.last().copied(), Some(2));
}

// Mirrors `TreeTCI.jl`'s `convergencecriterion` third disjunct
// (`all(lastranks .>= maxbonddim)`, branch `local-fix-convergence`, commit
// 06563dd): once the rank has saturated at `max_bond_dim` for the trailing
// window, further sweeps cannot lower the error, so the loop should stop
// even though the error never crosses `tolerance`.
#[test]
fn optimize_default_stops_early_when_bond_dim_saturated() {
    let mut tci = TreeTCI2::<f64>::new(vec![3, 3], two_site_graph()).unwrap();
    tci.add_global_pivots(&[vec![0, 0]]).unwrap();

    let batch_eval = |batch: GlobalIndexBatch<'_>| -> Result<Vec<f64>> {
        let mut values = Vec::with_capacity(batch.n_points());
        for point in 0..batch.n_points() {
            let i = batch.get(0, point).unwrap();
            let j = batch.get(1, point).unwrap();
            values.push(if i == j { 1.0 } else { 0.0 });
        }
        Ok(values)
    };

    let (ranks, errors) = optimize_default(
        &mut tci,
        batch_eval,
        &TreeTciOptions {
            tolerance: 1e-12,
            max_iter: 10,
            max_bond_dim: 1,
            normalize_error: true,
        },
    )
    .unwrap();

    // Rank-3 identity capped at max_bond_dim = 1 can never reach the 1e-12
    // tolerance; without the bond-dim-saturation criterion this would run
    // all 10 sweeps.
    assert!(ranks.len() < 10);
    assert!(ranks.iter().all(|&r| r <= 1));
    assert!(errors.last().copied().unwrap_or(0.0) > 1e-12);
}
