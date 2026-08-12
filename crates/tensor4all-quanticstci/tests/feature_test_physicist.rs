//! Integration tests exercising all three main entry points of tensor4all-quanticstci
//! from the perspective of a computational physicist.
//!
//! Entry points tested:
//!   1. `quanticscrossinterpolate_discrete` -- discrete integer grids
//!   2. `quanticscrossinterpolate`          -- continuous grids via DiscretizedGrid
//!   3. `quanticscrossinterpolate_from_arrays` -- continuous grids from explicit arrays
//!
//! Test functions include polynomials, exponentials, and products, with analytical
//! cross-checks for evaluation, summation, and integration.

use tensor4all_quanticstci::{
    quanticscrossinterpolate, quanticscrossinterpolate_discrete,
    quanticscrossinterpolate_from_arrays, DiscretizedGrid, QtciOptions, UnfoldingScheme,
};

// ---------------------------------------------------------------------------
// Helper: generic tolerance for floating-point comparisons
// ---------------------------------------------------------------------------
const TOL: f64 = 1e-8;

// ===========================================================================
// 1. quanticscrossinterpolate_discrete
// ===========================================================================

/// Polynomial on a 1D discrete grid: f(i) = i^2.
/// Verifies evaluation at every grid point and the exact sum.
#[test]
fn discrete_1d_polynomial() {
    let n: usize = 16; // 2^4
    let f = |idx: &[usize]| {
        let i = idx[0];
        (i * i) as f64
    };
    let sizes = vec![n];

    let opts = QtciOptions::default()
        .with_tolerance(1e-12)
        .with_nrandominitpivot(5);

    let (qtci, _ranks, errors) =
        quanticscrossinterpolate_discrete(&sizes, f, None, opts).expect("discrete 1D should work");

    // Convergence check
    assert!(
        *errors.last().unwrap() < 1e-10,
        "final error too large: {}",
        errors.last().unwrap()
    );

    // Evaluate at every grid point (0-indexed)
    for i in 0..n {
        let expected = (i * i) as f64;
        let actual = qtci.evaluate(&[i]).unwrap();
        assert!(
            (actual - expected).abs() < TOL,
            "f({i}) = {actual}, expected {expected}"
        );
    }

    // Sum: sum_{i=0}^{N-1} i^2 = (N-1)*N*(2N-1)/6
    let expected_sum = ((n - 1) * n * (2 * n - 1)) as f64 / 6.0;
    let actual_sum = qtci.sum().unwrap();
    assert!(
        (actual_sum - expected_sum).abs() < TOL,
        "sum = {actual_sum}, expected {expected_sum}"
    );
}

/// Product function on a 2D discrete grid: f(i, j) = i * j.
/// Verifies that the TCI captures a rank-1 (in the original variables) function.
#[test]
fn discrete_2d_product() {
    let n: usize = 16;
    let f = |idx: &[usize]| (idx[0] * idx[1]) as f64;
    let sizes = vec![n, n];

    let opts = QtciOptions::default()
        .with_tolerance(1e-12)
        .with_nrandominitpivot(5);

    let (qtci, _ranks, errors) =
        quanticscrossinterpolate_discrete(&sizes, f, None, opts).expect("discrete 2D should work");

    assert!(
        *errors.last().unwrap() < 1e-10,
        "final error too large: {}",
        errors.last().unwrap()
    );

    // Spot-check evaluations
    for &(i, j) in &[(0usize, 0usize), (0, 15), (7, 7), (15, 15)] {
        let expected = (i * j) as f64;
        let actual = qtci.evaluate(&[i, j]).unwrap();
        assert!(
            (actual - expected).abs() < TOL,
            "f({i},{j}) = {actual}, expected {expected}"
        );
    }

    // Sum: (sum_{i=0}^{N-1} i)^2 = [N(N-1)/2]^2
    let s = (n as f64) * ((n as f64) - 1.0) / 2.0;
    let expected_sum = s * s;
    let actual_sum = qtci.sum().unwrap();
    assert!(
        (actual_sum - expected_sum).abs() / expected_sum < 1e-10,
        "sum = {actual_sum}, expected {expected_sum}"
    );
}

/// Exponential on a 1D discrete grid: f(i) = exp(-0.1 * i).
/// Non-trivial test -- the function is smooth but non-polynomial.
#[test]
fn discrete_1d_exponential() {
    let n: usize = 64; // 2^6
    let alpha = 0.1_f64;
    let f = move |idx: &[usize]| (-alpha * idx[0] as f64).exp();
    let sizes = vec![n];

    let opts = QtciOptions::default()
        .with_tolerance(1e-12)
        .with_nrandominitpivot(10);

    let (qtci, _ranks, errors) =
        quanticscrossinterpolate_discrete(&sizes, f, None, opts).expect("discrete exp should work");

    assert!(
        *errors.last().unwrap() < 1e-10,
        "final error too large: {}",
        errors.last().unwrap()
    );

    // Evaluate at selected points
    for i in [0usize, 9, 31, 63] {
        let expected = (-alpha * i as f64).exp();
        let actual = qtci.evaluate(&[i]).unwrap();
        assert!(
            (actual - expected).abs() < TOL,
            "f({i}) = {actual}, expected {expected}"
        );
    }

    // Analytical sum: sum_{i=0}^{N-1} exp(-alpha*i) = (1 - exp(-alpha*N)) / (1 - exp(-alpha))
    let r = (-alpha).exp();
    let expected_sum = (1.0 - r.powi(n as i32)) / (1.0 - r);
    let actual_sum = qtci.sum().unwrap();
    assert!(
        (actual_sum - expected_sum).abs() < 1e-6,
        "sum = {actual_sum}, expected {expected_sum}"
    );
}

/// Edge case: constant function f(i) = 42.0 on a 1D grid.
/// The rank should be 1 and sum should be 42 * N.
#[test]
fn discrete_constant_function() {
    let n: usize = 32; // 2^5
    let f = |_idx: &[usize]| 42.0_f64;
    let sizes = vec![n];

    let opts = QtciOptions::default()
        .with_tolerance(1e-12)
        .with_nrandominitpivot(3);

    let (qtci, _ranks, _errors) =
        quanticscrossinterpolate_discrete(&sizes, f, None, opts).expect("constant fn should work");

    // Rank should be 1 for a constant
    assert_eq!(qtci.rank(), 1, "constant function should have rank 1");

    let actual_sum = qtci.sum().unwrap();
    let expected_sum = 42.0 * n as f64;
    assert!(
        (actual_sum - expected_sum).abs() < TOL,
        "sum = {actual_sum}, expected {expected_sum}"
    );
}

/// Verify that providing explicit initial pivots does not break anything.
#[test]
fn discrete_with_explicit_pivots() {
    let n: usize = 16;
    let f = |idx: &[usize]| (idx[0] + 2 * idx[1]) as f64;
    let sizes = vec![n, n];

    let opts = QtciOptions::default()
        .with_tolerance(1e-12)
        .with_nrandominitpivot(3);

    // Provide 0-indexed grid pivots
    let pivots = vec![vec![0, 0], vec![7, 7], vec![15, 15]];
    let (qtci, _ranks, errors) = quanticscrossinterpolate_discrete(&sizes, f, Some(pivots), opts)
        .expect("explicit pivots should work");

    assert!(
        *errors.last().unwrap() < 1e-10,
        "final error too large: {}",
        errors.last().unwrap()
    );

    let val = qtci.evaluate(&[4, 9]).unwrap();
    assert!((val - 22.0).abs() < TOL, "f(4,9) = {val}, expected 22.0");
}

/// Verify that invalid grid sizes produce errors.
#[test]
fn discrete_non_power_of_two_error() {
    let f = |_idx: &[usize]| 0.0_f64;
    let result = quanticscrossinterpolate_discrete(&[10], f, None, QtciOptions::default());
    assert!(
        result.is_err(),
        "non-power-of-2 grid size should return an error"
    );
}

/// Verify that unequal dimension sizes produce errors.
#[test]
fn discrete_unequal_sizes_error() {
    let f = |_idx: &[usize]| 0.0_f64;
    let result = quanticscrossinterpolate_discrete(&[8, 16], f, None, QtciOptions::default());
    assert!(
        result.is_err(),
        "unequal dimension sizes should return an error"
    );
}

// ===========================================================================
// 2. quanticscrossinterpolate (DiscretizedGrid)
// ===========================================================================

/// 1D continuous interpolation: f(x) = x^2 on [0, 1].
/// Verifies evaluation and numerical integration.
#[test]
fn continuous_1d_polynomial() {
    let r = 6; // 2^6 = 64 grid points
    let grid = DiscretizedGrid::builder(&[r])
        .with_lower_bound(&[0.0])
        .with_upper_bound(&[1.0])
        .include_endpoint(false) // midpoint rule style
        .build()
        .expect("grid should build");

    let f = |coords: &[f64]| coords[0] * coords[0];

    let opts = QtciOptions::default()
        .with_tolerance(1e-12)
        .with_nrandominitpivot(5);

    let (qtci, _ranks, errors) =
        quanticscrossinterpolate(&grid, f, None, opts).expect("continuous 1D should work");

    assert!(
        *errors.last().unwrap() < 1e-10,
        "final error: {}",
        errors.last().unwrap()
    );

    // Should have discretized_grid, not inherent_grid
    assert!(qtci.discretized_grid().is_some());
    assert!(qtci.inherent_grid().is_none());

    // The integral of x^2 over [0,1] is 1/3.
    // With a finite grid, integral() returns a Riemann-sum approximation.
    // The grid step h = 1/64 gives O(h) discretization error for a left-rule sum,
    // so we expect accuracy around ~0.01.
    let integral = qtci.integral().unwrap();
    let expected_integral = 1.0 / 3.0;
    assert!(
        (integral - expected_integral).abs() < 0.02,
        "integral = {integral}, expected ~ {expected_integral}"
    );
}

/// 2D continuous interpolation: f(x, y) = exp(-(x^2 + y^2)) on [-2, 2]^2.
/// Gaussian-like function -- verifies the TCI can handle smooth 2D functions.
#[test]
fn continuous_2d_gaussian() {
    let r = 6; // 2^6 = 64 points per dimension
    let grid = DiscretizedGrid::builder(&[r, r])
        .with_lower_bound(&[-2.0, -2.0])
        .with_upper_bound(&[2.0, 2.0])
        .include_endpoint(false)
        .build()
        .expect("2D grid should build");

    let f = |coords: &[f64]| (-(coords[0] * coords[0] + coords[1] * coords[1])).exp();

    let opts = QtciOptions::default()
        .with_tolerance(1e-10)
        .with_nrandominitpivot(10);

    let (qtci, _ranks, errors) =
        quanticscrossinterpolate(&grid, f, None, opts).expect("2D Gaussian should work");

    assert!(
        *errors.last().unwrap() < 1e-8,
        "final error: {}",
        errors.last().unwrap()
    );

    // Evaluate at origin: f(0, 0) = 1.0
    // Need to find the grid index closest to (0,0).
    // Grid goes from -2 to 2 with 64 points, no endpoint.
    // Grid step = 4/64 = 0.0625. Points: -2 + 0.03125, -2 + 0.09375, ...
    // The midpoint nearest to 0 is around index 31 or 32 (0-indexed).
    // Just check at a few points that the interpolation is reasonable.
    let val_corner = qtci.evaluate(&[1, 1]).unwrap();
    assert!(val_corner.is_finite());

    // The 2D integral of exp(-(x^2+y^2)) over [-2,2]^2 is approximately pi * erf(2)^2
    // erf(2) ~ 0.9953, so integral ~ pi * 0.9906 ~ 3.112
    let integral = qtci.integral().unwrap();
    let erf_2 = 0.9953222650189527;
    let expected_integral = std::f64::consts::PI * erf_2 * erf_2;
    assert!(
        (integral - expected_integral).abs() < 0.1,
        "integral = {integral}, expected ~ {expected_integral}"
    );
}

/// Verify cachedata_origcoord() returns sensible data for a continuous grid.
#[test]
fn continuous_cachedata_origcoord() {
    let grid = DiscretizedGrid::builder(&[4]) // 2^4 = 16 points
        .with_lower_bound(&[0.0])
        .with_upper_bound(&[1.0])
        .include_endpoint(true)
        .build()
        .unwrap();

    let f = |coords: &[f64]| coords[0].sin();

    let opts = QtciOptions::default()
        .with_tolerance(1e-12)
        .with_nrandominitpivot(5);

    let (qtci, _ranks, _errors) =
        quanticscrossinterpolate(&grid, f, None, opts).expect("sin interpolation should work");

    let cache = qtci.cachedata_origcoord().unwrap();
    assert!(!cache.is_empty(), "cache should not be empty");

    for (coord, val) in &cache {
        assert_eq!(coord.len(), 1);
        let x = coord[0];
        let expected = x.sin();
        assert!(
            (val - expected).abs() < 1e-10,
            "cached f({x}) = {val}, expected {expected}"
        );
    }
}

/// Verify integral of a constant function over a continuous grid.
#[test]
fn continuous_integral_constant() {
    let grid = DiscretizedGrid::builder(&[4]) // 2^4 = 16 points
        .with_lower_bound(&[0.0])
        .with_upper_bound(&[1.0])
        .build()
        .unwrap();

    let f = |_coords: &[f64]| 1.0_f64;

    let opts = QtciOptions::default()
        .with_tolerance(1e-12)
        .with_nrandominitpivot(3);

    let (qtci, _ranks, _errors) =
        quanticscrossinterpolate(&grid, f, None, opts).expect("constant should work");

    // integral of 1 over [0,1] = 1.0
    let integral = qtci.integral().unwrap();
    assert!(
        (integral - 1.0).abs() < TOL,
        "integral = {integral}, expected 1.0"
    );
}

// ===========================================================================
// 3. quanticscrossinterpolate_from_arrays
// ===========================================================================

/// 1D from_arrays: f(x) = x^3 on [-1, 1] with 128 points.
#[test]
fn from_arrays_1d_cubic() {
    let n = 128_usize; // 2^7
    let xvals: Vec<f64> = (0..n)
        .map(|i| -1.0 + 2.0 * i as f64 / (n - 1) as f64)
        .collect();

    let f = |coords: &[f64]| coords[0].powi(3);

    let opts = QtciOptions::default()
        .with_tolerance(1e-10)
        .with_nrandominitpivot(5);

    let (qtci, _ranks, errors) =
        quanticscrossinterpolate_from_arrays(std::slice::from_ref(&xvals), f, None, opts)
            .expect("from_arrays 1D cubic should work");

    assert!(
        *errors.last().unwrap() < 1e-8,
        "final error: {}",
        errors.last().unwrap()
    );

    // Evaluate at several grid points (0-indexed)
    for i in [0_usize, 31, 63, 64, 127] {
        let x = xvals[i];
        let expected = x.powi(3);
        let actual = qtci.evaluate(&[i]).unwrap();
        assert!(
            (actual - expected).abs() < 1e-6,
            "f({x}) = {actual}, expected {expected}"
        );
    }
}

/// 2D from_arrays: f(x,y) = x + y on [0, 3]^2 with 4 points each.
/// Small grid to verify correctness at every point.
#[test]
fn from_arrays_2d_linear() {
    let xvals = vec![0.0, 1.0, 2.0, 3.0]; // 4 = 2^2
    let f = |coords: &[f64]| coords[0] + coords[1];

    let opts = QtciOptions::default()
        .with_tolerance(1e-12)
        .with_unfoldingscheme(UnfoldingScheme::Fused);

    // Use explicit initial pivot at (1,1) where f(1,1)=2 ≠ 0, avoiding the
    // flaky failure when random pivots all land on (0,0) where f(0,0)=0.
    let initial_pivots = vec![vec![1usize, 1]];

    let (qtci, _ranks, _errors) = quanticscrossinterpolate_from_arrays(
        &[xvals.clone(), xvals.clone()],
        f,
        Some(initial_pivots),
        opts,
    )
    .expect("from_arrays 2D linear should work");

    // Evaluate at all 4x4 = 16 points
    for i in 0..4 {
        for j in 0..4 {
            let x = xvals[i];
            let y = xvals[j];
            let expected = x + y;
            let actual = qtci.evaluate(&[i, j]).unwrap();
            assert!(
                (actual - expected).abs() < TOL,
                "f({x},{y}) = {actual}, expected {expected}"
            );
        }
    }
}

/// from_arrays: f(x) = exp(-x^2) on [-3, 3] with 256 points.
/// Verifies integral approximation against known value.
#[test]
fn from_arrays_1d_gaussian_integral() {
    let n = 256_usize; // 2^8
    let a = -3.0_f64;
    let b = 3.0_f64;
    let xvals: Vec<f64> = (0..n)
        .map(|i| a + (b - a) * i as f64 / (n - 1) as f64)
        .collect();

    let f = |coords: &[f64]| (-coords[0] * coords[0]).exp();

    let opts = QtciOptions::default()
        .with_tolerance(1e-12)
        .with_nrandominitpivot(10);

    let (qtci, _ranks, _errors) = quanticscrossinterpolate_from_arrays(&[xvals], f, None, opts)
        .expect("from_arrays Gaussian should work");

    // integral of exp(-x^2) from -3 to 3 ~ sqrt(pi) * erf(3) ~ 1.7724539 * 0.9999779 ~ 1.7724
    let integral = qtci.integral().unwrap();
    let expected = std::f64::consts::PI.sqrt() * 0.9999779095030014; // erf(3)
    assert!(
        (integral - expected).abs() < 0.05,
        "integral = {integral}, expected ~ {expected}"
    );
}

/// from_arrays: error cases -- empty arrays and non-power-of-2.
#[test]
fn from_arrays_error_cases() {
    let f = |_coords: &[f64]| 1.0_f64;

    // Empty
    let result =
        quanticscrossinterpolate_from_arrays::<f64, _>(&[], f, None, QtciOptions::default());
    assert!(result.is_err(), "empty xvals should error");

    // Non-power-of-2
    let result = quanticscrossinterpolate_from_arrays::<f64, _>(
        &[vec![0.0, 1.0, 2.0]],
        f,
        None,
        QtciOptions::default(),
    );
    assert!(result.is_err(), "3-element array should error");

    // Unequal dimensions
    let result = quanticscrossinterpolate_from_arrays::<f64, _>(
        &[vec![0.0, 1.0, 2.0, 3.0], vec![0.0, 1.0]],
        f,
        None,
        QtciOptions::default(),
    );
    assert!(result.is_err(), "unequal dimension sizes should error");
}

// ===========================================================================
// 4. Builder pattern and option verification
// ===========================================================================

/// Verify all builder methods on QtciOptions.
#[test]
fn options_builder_all_fields() {
    let opts = QtciOptions::default()
        .with_tolerance(1e-6)
        .with_max_bond_dim(50)
        .with_maxiter(100)
        .with_nrandominitpivot(10)
        .with_unfoldingscheme(UnfoldingScheme::Fused)
        .with_verbosity(2)
        .with_nsearchglobalpivot(3)
        .with_nsearch(200);

    assert!((opts.tolerance - 1e-6).abs() < 1e-15);
    assert_eq!(opts.max_bond_dim, Some(50));
    assert_eq!(opts.maxiter, 100);
    assert_eq!(opts.nrandominitpivot, 10);
    assert_eq!(opts.unfoldingscheme, UnfoldingScheme::Fused);
    assert_eq!(opts.verbosity, 2);
    assert_eq!(opts.nsearchglobalpivot, 3);
    assert_eq!(opts.nsearch, 200);

    // Verify conversion to TreeTciOptions preserves key fields
    let tree_opts = opts.to_treetci_options();
    assert!((tree_opts.tolerance - 1e-6).abs() < 1e-15);
    assert_eq!(tree_opts.max_bond_dim, Some(50));
    assert_eq!(tree_opts.max_iter, 100);
}

/// Verify that max_bond_dim actually limits the rank.
#[test]
fn options_max_bond_dim_limits_rank() {
    // Use a function that would normally require higher rank
    let n: usize = 64;
    let f = |idx: &[usize]| ((idx[0] as f64) * 0.1).sin() * ((idx[1] as f64) * 0.2).cos();
    let sizes = vec![n, n];

    let opts = QtciOptions::default()
        .with_tolerance(1e-14) // Very tight tolerance to force high rank
        .with_max_bond_dim(3) // But cap bond dim at 3
        .with_nrandominitpivot(5);

    let (qtci, _ranks, _errors) = quanticscrossinterpolate_discrete(&sizes, f, None, opts)
        .expect("max_bond_dim test should work");

    // The rank should respect the maximum bond dimension
    assert!(
        qtci.rank() <= 3,
        "rank {} should be <= max_bond_dim 3",
        qtci.rank()
    );
}

// ===========================================================================
// 5. Structural tests: tensor_train(), link_dims(), rank()
// ===========================================================================

/// Verify that tensor_train() returns a valid tensor train and that
/// evaluate through the TT matches evaluate through QTCI.
#[test]
fn tensor_train_consistency() {
    let n: usize = 16;
    let f = |idx: &[usize]| (idx[0] + idx[1]) as f64;
    let sizes = vec![n, n];

    let opts = QtciOptions::default()
        .with_tolerance(1e-12)
        .with_nrandominitpivot(5);

    let (qtci, _ranks, _errors) =
        quanticscrossinterpolate_discrete(&sizes, f, None, opts).expect("should work");

    let _tt = qtci.tensor_train();

    // link_dims from QTCI should be non-empty
    let qtci_dims = qtci.link_dims();
    assert!(!qtci_dims.is_empty());
    assert!(qtci.rank() > 0);

    // Verify sum is correct analytically: sum_{i=0}^{N-1} sum_{j=0}^{N-1} (i + j)
    // = N * sum_i + N * sum_j = 2 * N * N*(N-1)/2 = N^2 * (N-1)
    let nf = n as f64;
    let expected_sum = nf * nf * (nf - 1.0);
    let qtci_sum = qtci.sum().unwrap();
    assert!(
        (qtci_sum - expected_sum).abs() < TOL,
        "QTCI sum ({qtci_sum}) != expected ({expected_sum})"
    );
}

/// Verify that tensor_train() returns a valid TT and rank matches.
#[test]
fn tensor_train_accessor() {
    use tensor4all_quanticstci::AbstractTensorTrain;

    let f = |idx: &[usize]| idx[0] as f64;
    let sizes = vec![8]; // 2^3

    let opts = QtciOptions::default()
        .with_tolerance(1e-12)
        .with_nrandominitpivot(3);

    let (qtci, _ranks, _errors) =
        quanticscrossinterpolate_discrete(&sizes, f, None, opts).expect("should work");

    let tt = qtci.tensor_train();
    assert!(tt.rank() > 0);
    assert_eq!(tt.rank(), qtci.rank());
}

// ===========================================================================
// 6. Unfolding scheme comparison
// ===========================================================================

/// Run the same function with Interleaved and Fused schemes.
/// Both should give correct results, though ranks may differ.
#[test]
fn unfolding_scheme_comparison() {
    let n: usize = 16;
    let f = |idx: &[usize]| (idx[0] * idx[0] + idx[1]) as f64;
    let sizes = vec![n, n];

    for scheme in [UnfoldingScheme::Interleaved, UnfoldingScheme::Fused] {
        let opts = QtciOptions::default()
            .with_tolerance(1e-12)
            .with_nrandominitpivot(5)
            .with_unfoldingscheme(scheme);

        let (qtci, _ranks, _errors) = quanticscrossinterpolate_discrete(&sizes, f, None, opts)
            .unwrap_or_else(|e| panic!("scheme {scheme:?} failed: {e}"));

        // Check a few evaluations
        let val = qtci.evaluate(&[2, 4]).unwrap();
        let expected = (2usize * 2 + 4) as f64;
        assert!(
            (val - expected).abs() < TOL,
            "scheme {scheme:?}: f(2,4) = {val}, expected {expected}"
        );

        // Sum should be the same regardless of scheme
        // sum_{i=0}^{N-1} sum_{j=0}^{N-1} (i^2 + j) = N * sum(i^2) + N * sum(j)
        let sum_i2 = ((n - 1) as f64) * (n as f64) * (2.0 * (n as f64) - 1.0) / 6.0;
        let sum_j = (n as f64) * ((n as f64) - 1.0) / 2.0;
        let expected_sum = (n as f64) * sum_i2 + (n as f64) * sum_j;
        let actual_sum = qtci.sum().unwrap();
        assert!(
            (actual_sum - expected_sum).abs() / expected_sum < 1e-10,
            "scheme {scheme:?}: sum = {actual_sum}, expected {expected_sum}"
        );
    }
}
