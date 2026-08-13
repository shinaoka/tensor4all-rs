use super::find_global_pivots;
use crate::{elementwise, AciOptions, ElementwiseProblem};
use tensor4all_simplett::{tensor3_from_data, AbstractTensorTrain, SimpleTensorTrain};

const D: usize = 4;

fn gauss(peak: usize, value: usize) -> f64 {
    let d = value as f64 - peak as f64;
    (-0.5 * d * d).exp()
}

/// f(idx) = 3*exp(-0.5*d2(idx, A)) + 2*exp(-0.5*d2(idx, B)) as a rank-2 TT
/// with peaks A = (0, .., 0) and B = (3, .., 3) on an `n`-site chain.
fn two_peak_tt(n: usize) -> SimpleTensorTrain<f64> {
    let mut cores = Vec::new();
    for site in 0..n {
        let (l, r) = if site == 0 {
            (1, 2)
        } else if site == n - 1 {
            (2, 1)
        } else {
            (2, 2)
        };
        let mut data = vec![0.0; l * D * r];
        for r_idx in 0..r {
            for i in 0..D {
                for l_idx in 0..l {
                    let value = match (site, l_idx, r_idx) {
                        (0, 0, 0) => 3.0 * gauss(0, i),
                        (0, 0, 1) => 2.0 * gauss(3, i),
                        (s, 1, 0) if s == n - 1 => gauss(3, i),
                        (_, 0, 0) => gauss(0, i),
                        (_, 1, 1) => gauss(3, i),
                        _ => 0.0,
                    };
                    data[l_idx + l * (i + D * r_idx)] = value;
                }
            }
        }
        cores.push(tensor3_from_data(data, l, D, r).unwrap());
    }
    SimpleTensorTrain::new(cores).unwrap()
}

fn run_case(seed: u64, guard: bool, nsearch: usize) -> SimpleTensorTrain<f64> {
    let input = two_peak_tt(10);
    let options = AciOptions {
        rng_seed: seed,
        enable_global_guard: guard,
        nsearch_global_pivots: nsearch,
        tolerance: 1e-4,
        ..AciOptions::default()
    };
    elementwise(|xs: &[f64]| xs[0], &[input], &options)
        .unwrap()
        .tensor_train
}

#[test]
fn global_guard_recovers_missed_near_degenerate_feature() {
    // The two near-degenerate peaks are far apart, so the local sweeps can
    // converge on the first basin only while the second peak silently
    // vanishes: the guard-off run evaluates to ~0 at B (f(B) = 2). The
    // global pivot search guard finds the missed point, injects it, and the
    // guard-on run captures both peaks.
    let fb = 2.0;
    let off = run_case(0, false, 5);
    let on = run_case(0, true, 30);

    let err_off = (off.evaluate(&[3; 10]).unwrap() - fb).abs();
    let err_on = (on.evaluate(&[3; 10]).unwrap() - fb).abs();
    let err_on_a = (on.evaluate(&[0; 10]).unwrap() - 3.0).abs();

    assert!(
        err_off > 1.0,
        "expected the guard-off run to miss the second peak, err at B = {err_off}"
    );
    assert!(
        err_on < 1e-6,
        "expected the guard to recover the second peak, err at B = {err_on}"
    );
    assert!(
        err_on_a < 1e-6,
        "the guard must not damage the first peak, err at A = {err_on_a}"
    );
}

#[test]
fn global_guard_disabled_by_zero_search_budget() {
    // nsearch_global_pivots = 0 disables the search: the run behaves exactly
    // like the guard-off run (second peak missed).
    let fb = 2.0;
    let input = two_peak_tt(10);
    let options = AciOptions {
        rng_seed: 0,
        enable_global_guard: true,
        nsearch_global_pivots: 0,
        tolerance: 1e-4,
        ..AciOptions::default()
    };
    let result = elementwise(|xs: &[f64]| xs[0], &[input], &options).unwrap();
    let err = (result.tensor_train.evaluate(&[3; 10]).unwrap() - fb).abs();
    assert!(
        err > 1.0,
        "zero search budget must behave like the guard is off"
    );
}

#[test]
fn find_global_pivots_noop_without_search_budget() {
    // Direct finder check: a zero budget returns nothing without touching
    // the problem.
    let input = two_peak_tt(10);
    let mut problem = ElementwiseProblem::new(vec![input], AciOptions::default()).unwrap();
    let options = AciOptions {
        nsearch_global_pivots: 0,
        max_nglobal_pivot: 0,
        ..AciOptions::default()
    };
    let mut op = |_batch: crate::ElementwiseBatch<'_, f64>, output: &mut [f64]| {
        output.fill(0.0);
        Ok(())
    };
    let pivots = find_global_pivots(&mut problem, &mut op, &options, 0).unwrap();
    assert!(pivots.is_empty());
}

#[test]
fn global_guard_terminates_when_bond_dimension_capped() {
    // With max_bond_dim = 1 the two-peak output cannot be represented, so the
    // guard keeps finding the same missed point and must terminate cleanly
    // (the pivot is already in the frames -> nothing new injected -> stop)
    // instead of burning iterations or failing.
    let input = two_peak_tt(10);
    let options = AciOptions {
        rng_seed: 0,
        enable_global_guard: true,
        nsearch_global_pivots: 30,
        tolerance: 1e-4,
        max_bond_dim: Some(1),
        ..AciOptions::default()
    };
    let result = elementwise(|xs: &[f64]| xs[0], &[input], &options).unwrap();
    assert!(result.tensor_train.rank() <= 1);
    assert!(result.ranks.len() <= options.max_iters);
}
