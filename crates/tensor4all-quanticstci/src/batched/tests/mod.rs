use super::*;
use crate::QtciOptions;
use quanticsgrids::DiscretizedGrid;
use tensor4all_simplett::AbstractTensorTrain;

#[test]
fn test_batched_tci_2component_1d() {
    let grid = DiscretizedGrid::builder(&[6])
        .with_variable_names(&["x"])
        .with_lower_bound(&[0.0])
        .with_upper_bound(&[1.0])
        .build()
        .unwrap();

    let options = QtciOptions::default().with_tolerance(1e-8);

    // Use sin(x)+1 and cos(x) so all components are non-zero at x=0
    // (the default initial pivot).
    let (result, _, _) = quanticscrossinterpolate_batched::<f64, _>(
        &grid,
        |x: &[f64]| {
            vec![
                (2.0 * std::f64::consts::PI * x[0]).sin() + 1.0,
                (2.0 * std::f64::consts::PI * x[0]).cos(),
            ]
        },
        &[2],
        None,
        options,
    )
    .unwrap();

    // 6 grid sites + 1 component site = 7 sites
    assert_eq!(result.tensor_train().len(), 7);
    assert_eq!(result.output_dims(), &[2]);

    // Verify values at all grid points.
    let n_grid_points = 1usize << 6; // 2^6 = 64
    for i in 0..n_grid_points {
        let grid_idx = vec![i]; // 0-indexed
        let quantics = grid.grididx_to_quantics(&grid_idx).unwrap();
        let coord = grid.quantics_to_origcoord(&quantics).unwrap();
        let x = coord[0];

        // Evaluate component 0 (sin + 1)
        let mut indices0 = quantics.clone();
        indices0.push(0);
        let val0 = result.tensor_train().evaluate(&indices0).unwrap();
        let expected0 = (2.0 * std::f64::consts::PI * x).sin() + 1.0;
        assert!(
            (val0 - expected0).abs() < 1e-6,
            "comp 0 mismatch at x={}: got {}, expected {}",
            x,
            val0,
            expected0
        );

        // Evaluate component 1 (cos)
        let mut indices1 = quantics.clone();
        indices1.push(1);
        let val1 = result.tensor_train().evaluate(&indices1).unwrap();
        let expected1 = (2.0 * std::f64::consts::PI * x).cos();
        assert!(
            (val1 - expected1).abs() < 1e-6,
            "comp 1 mismatch at x={}: got {}, expected {}",
            x,
            val1,
            expected1
        );
    }
}

#[test]
fn test_batched_tci_scalar_equivalent() {
    // 1-component batched should give same result as scalar TCI
    let grid = DiscretizedGrid::builder(&[4])
        .with_variable_names(&["x"])
        .with_lower_bound(&[0.0])
        .with_upper_bound(&[1.0])
        .build()
        .unwrap();

    let options = QtciOptions::default().with_tolerance(1e-8);

    let (scalar_result, _, _) = crate::quanticscrossinterpolate::<f64, _>(
        &grid,
        |x: &[f64]| x[0] * x[0],
        None,
        options.clone(),
    )
    .unwrap();

    let (batched_result, _, _) = quanticscrossinterpolate_batched::<f64, _>(
        &grid,
        |x: &[f64]| vec![x[0] * x[0]],
        &[1],
        None,
        options,
    )
    .unwrap();

    // Scalar has 4 sites, batched has 4 + 1 = 5 sites
    assert_eq!(scalar_result.tensor_train().len(), 4);
    assert_eq!(batched_result.tensor_train().len(), 5); // 4 + 1 component

    // Both should produce the same values at grid points.
    let n_grid_points = 1usize << 4; // 16
    for i in 0..n_grid_points {
        let grid_idx = vec![i];
        let quantics = grid.grididx_to_quantics(&grid_idx).unwrap();

        let scalar_val = scalar_result.evaluate(&grid_idx).unwrap();

        let mut batched_indices = quantics;
        batched_indices.push(0); // single component
        let batched_val = batched_result
            .tensor_train()
            .evaluate(&batched_indices)
            .unwrap();

        assert!(
            (scalar_val - batched_val).abs() < 1e-10,
            "mismatch at grid_idx={}: scalar={}, batched={}",
            i,
            scalar_val,
            batched_val
        );
    }
}

#[test]
fn test_batched_tci_empty_output_error() {
    let grid = DiscretizedGrid::builder(&[4])
        .with_variable_names(&["x"])
        .with_lower_bound(&[0.0])
        .with_upper_bound(&[1.0])
        .build()
        .unwrap();

    let options = QtciOptions::default();

    // Empty output_dims should fail
    let result = quanticscrossinterpolate_batched::<f64, _>(
        &grid,
        |_: &[f64]| vec![],
        &[],
        None,
        options.clone(),
    );
    assert!(result.is_err());

    // Zero in output_dims should fail
    let result =
        quanticscrossinterpolate_batched::<f64, _>(&grid, |_: &[f64]| vec![], &[0], None, options);
    assert!(result.is_err());
}

#[test]
fn test_batched_tci_matrix_valued() {
    // Test with a 2x2 matrix-valued function (4 components).
    let grid = DiscretizedGrid::builder(&[4])
        .with_variable_names(&["x"])
        .with_lower_bound(&[0.0])
        .with_upper_bound(&[1.0])
        .build()
        .unwrap();

    let options = QtciOptions::default().with_tolerance(1e-8);

    // f(x) returns a 2x2 matrix flattened as [a00, a01, a10, a11]
    // Use x+1 so f(0) != 0 (default initial pivot is at x=0).
    let (result, _, _) = quanticscrossinterpolate_batched::<f64, _>(
        &grid,
        |x: &[f64]| {
            let v = x[0] + 1.0;
            vec![v, 2.0 * v, 3.0 * v, 4.0 * v]
        },
        &[2, 2],
        None,
        options,
    )
    .unwrap();

    assert_eq!(result.output_dims(), &[2, 2]);
    // 4 grid sites + 1 component site = 5
    assert_eq!(result.tensor_train().len(), 5);

    // Verify values at several grid points.
    let n_grid_points = 1usize << 4;
    for i in 0..n_grid_points {
        let grid_idx = vec![i];
        let quantics = grid.grididx_to_quantics(&grid_idx).unwrap();
        let coord = grid.quantics_to_origcoord(&quantics).unwrap();
        let x = coord[0];

        for comp in 0..4 {
            let mut indices = quantics.clone();
            indices.push(comp);
            let val = result.tensor_train().evaluate(&indices).unwrap();
            let expected = (comp as f64 + 1.0) * (x + 1.0);
            assert!(
                (val - expected).abs() < 1e-8,
                "mismatch at x={}, comp={}: got {}, expected {}",
                x,
                comp,
                val,
                expected
            );
        }
    }
}

#[test]
fn test_batched_tci_caching_reduces_evaluations() {
    // Verify that the shared cache reduces function evaluations.
    // Use Arc<Mutex<usize>> counter to track how many times f is actually called.
    use std::sync::atomic::{AtomicUsize, Ordering};

    let call_count = Arc::new(AtomicUsize::new(0));
    let call_count_clone = call_count.clone();

    let grid = DiscretizedGrid::builder(&[3])
        .with_variable_names(&["x"])
        .with_lower_bound(&[0.0])
        .with_upper_bound(&[1.0])
        .build()
        .unwrap();

    let options = QtciOptions::default()
        .with_tolerance(1e-8)
        .with_nrandominitpivot(0); // no random pivots for determinism

    // Use functions that are non-zero at x=0 (the default initial pivot).
    let (result, _, _) = quanticscrossinterpolate_batched::<f64, _>(
        &grid,
        move |x: &[f64]| {
            call_count_clone.fetch_add(1, Ordering::Relaxed);
            vec![x[0] + 1.0, x[0] * x[0] + 1.0]
        },
        &[2],
        None,
        options,
    )
    .unwrap();

    assert_eq!(result.tensor_train().len(), 4); // 3 grid sites + 1 component

    // The number of function calls should be less than 2 * (max possible unique points)
    // because the cache is shared across components.
    let total_calls = call_count.load(Ordering::Relaxed);
    let max_grid_points = 1usize << 3; // 8

    // Without caching, we'd expect up to 2 * max_grid_points calls (one per component).
    // With caching, we should see at most max_grid_points calls.
    // (In practice it could be less if TCI doesn't sample all points.)
    assert!(
        total_calls <= max_grid_points,
        "expected at most {} calls with caching, got {}",
        max_grid_points,
        total_calls
    );
}

#[test]
fn test_combine_component_tts_basic() {
    // Test the combine function directly with known simple TTs.
    // Create two rank-1 TTs: constant 2.0 and constant 3.0 on a 2-site, dim-2 grid.
    let tt1 = SimpleTensorTrain::<f64>::constant(&[2, 2], 2.0);
    let tt2 = SimpleTensorTrain::<f64>::constant(&[2, 2], 3.0);

    let combined = combine_component_tts(&[tt1, tt2]).unwrap();

    // Should have 3 sites: 2 grid sites + 1 selector
    assert_eq!(combined.len(), 3);

    // Evaluate: component 0 should give 2.0, component 1 should give 3.0
    for i in 0..2 {
        for j in 0..2 {
            let val0 = combined.evaluate(&[i, j, 0]).unwrap();
            assert!(
                (val0 - 2.0).abs() < 1e-12,
                "comp 0 at ({},{}): got {}",
                i,
                j,
                val0
            );

            let val1 = combined.evaluate(&[i, j, 1]).unwrap();
            assert!(
                (val1 - 3.0).abs() < 1e-12,
                "comp 1 at ({},{}): got {}",
                i,
                j,
                val1
            );
        }
    }
}
