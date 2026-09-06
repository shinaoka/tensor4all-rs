#[path = "support/g0.rs"]
mod g0;

use num_complex::Complex64;
use tensor4all_core::{ColMajorArrayRef, IdxTensor};
use tensor4all_treeaci::{tree_elementwise_batched, TreeAciTermination};
use tensor4all_treetn::{CachedEvaluatorOptions, EvaluationHint, TreeTNCachedEvaluator};

#[test]
fn low_temperature_g0_full_grid() -> g0::TestResult<()> {
    for r in [3, 4, 5] {
        for mode in ["cttn", "swap", "nblock"] {
            let (sites, inputs) = g0::fixture(r, mode)?;
            let expected_values: Vec<_> = (0..1usize << (3 * r))
                .map(|flat| {
                    let mut point = [0; 3];
                    for bit in 0..r {
                        for (var, x) in point.iter_mut().enumerate() {
                            *x = 2 * *x + ((flat >> (3 * bit + var)) & 1);
                        }
                    }
                    g0::exact(r, point[0], point[1], point[2])
                })
                .collect();
            let expected = IdxTensor::from_dense(sites.clone(), expected_values)?;
            let dense_inputs: Vec<Vec<Complex64>> = inputs
                .iter()
                .map(|tree| {
                    Ok(tree
                        .to_dense()?
                        .permute_indices(&sites)?
                        .to_vec::<Complex64>()?)
                })
                .collect::<g0::TestResult<_>>()?;
            let direct = IdxTensor::from_dense(
                sites.clone(),
                (0..dense_inputs[0].len())
                    .map(|i| {
                        1.0 / (Complex64::new(0.5, 0.0)
                            + 2.0 * dense_inputs[0][i]
                            + 2.0 * dense_inputs[1][i]
                            + Complex64::i() * dense_inputs[2][i]
                            - dense_inputs[3][i])
                    })
                    .collect::<Vec<_>>(),
            )?;
            assert!(direct.sub(&expected)?.maxabs()? < 1e-10);
            let options = g0::options();
            let result = tree_elementwise_batched(g0::operator, &inputs, &options)?;
            let error = result.tree.to_dense()?.sub(&expected)?.maxabs()?;
            assert!(
                error <= options.tolerance * options.global_tolerance_margin,
                "r={r}, mode={mode}, max absolute residual={error:e}"
            );
        }
    }
    Ok(())
}

#[test]
fn low_temperature_branch_convergence_does_not_hide_growth_on_smaller_cuts() -> g0::TestResult<()> {
    let r = 9;
    let (sites, inputs) = g0::fixture(r, "cttn")?;
    let options = g0::options();
    let result = tree_elementwise_batched(g0::operator, &inputs, &options)?;
    assert_eq!(result.termination, TreeAciTermination::Converged);
    // Independent witnesses from #741 and their reflection partners. Before
    // the fix the first point has absolute error 0.247, despite Converged.
    let points = [
        (292, 70, 256),
        (292, 442, 256),
        (242, 70, 256),
        (230, 442, 256),
        (243, 454, 256),
    ];
    let mut coordinates = Vec::with_capacity(points.len() * 3 * r);
    for &(x, y, n) in &points {
        for bit in 0..r {
            for value in [x, y, n] {
                coordinates.push((value >> (r - 1 - bit)) & 1);
            }
        }
    }
    let mut evaluator =
        TreeTNCachedEvaluator::new(&result.tree, &sites, CachedEvaluatorOptions::default())?;
    let actual = evaluator.evaluate_batched_typed::<Complex64>(
        ColMajorArrayRef::new(&coordinates, &[3 * r, points.len()])?,
        EvaluationHint::default(),
    )?;
    let error = actual
        .iter()
        .zip(&points)
        .map(|(&value, &(x, y, n))| (value - g0::exact(r, x, y, n)).norm())
        .fold(0.0_f64, f64::max);
    assert!(
        error <= options.tolerance * options.global_tolerance_margin,
        "max absolute witness residual={error:e}"
    );
    Ok(())
}
