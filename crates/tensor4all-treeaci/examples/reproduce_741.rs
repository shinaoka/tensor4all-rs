//! Independent low-temperature slice check for issue #741.
#[path = "../tests/support/g0.rs"]
mod g0;

use num_complex::Complex64;
use tensor4all_core::ColMajorArrayRef;
use tensor4all_treeaci::tree_elementwise_batched;
use tensor4all_treetn::{CachedEvaluatorOptions, EvaluationHint, TreeTNCachedEvaluator};

fn main() -> g0::TestResult<()> {
    let args: Vec<_> = std::env::args().collect();
    let r: usize = args.get(1).map_or(Ok(4), |v| v.parse())?;
    assert!((2..=10).contains(&r));
    let mode = args.get(2).map_or("cttn", String::as_str);
    let (sites, inputs) = g0::fixture(r, mode)?;
    let mut options = g0::options();
    if let Some(value) = args.get(3) {
        options.max_sweeps = value.parse()?;
        options.min_sweeps = options.max_sweeps;
    }
    let started = std::time::Instant::now();
    let result = tree_elementwise_batched(g0::operator, &inputs, &options)?;
    eprintln!(
        "r={r} mode={mode} seconds={} points={} termination={:?} ranks={:?} errors={:?} guard={:?}",
        started.elapsed().as_secs_f64(),
        result.diagnostics.evaluated_points,
        result.termination,
        result.max_ranks,
        result.max_errors,
        result.global_pivots_found
    );
    let mut evaluator =
        TreeTNCachedEvaluator::new(&result.tree, &sites, CachedEvaluatorOptions::default())?;
    let size = 1usize << r;
    let mut max_error = 0.0_f64;
    let mut worst = (0, 0, 0);
    for n in [size / 2 - 1, size / 2] {
        for y in 0..size {
            let mut coords = Vec::with_capacity(3 * r * size);
            for x in 0..size {
                for bit in 0..r {
                    for value in [x, y, n] {
                        coords.push((value >> (r - 1 - bit)) & 1);
                    }
                }
            }
            let values = evaluator.evaluate_batched_typed::<Complex64>(
                ColMajorArrayRef::new(&coords, &[3 * r, size])?,
                EvaluationHint::default(),
            )?;
            for (x, value) in values.into_iter().enumerate() {
                let error = (value - g0::exact(r, x, y, n)).norm();
                if error > max_error {
                    max_error = error;
                    worst = (x, y, n);
                }
            }
        }
    }
    println!(
        "r={r} mode={mode} max_abs={max_error:e} normalized={:e} worst={worst:?}",
        max_error / 31.830555038989377
    );
    assert!(max_error <= options.tolerance * options.global_tolerance_margin);
    Ok(())
}
