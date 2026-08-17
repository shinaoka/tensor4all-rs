use super::*;
use num_complex::Complex64;
use tensor4all_tensorbackend::{from_vec2d, mat_mul, transpose, Matrix};

fn identity(n: usize) -> Matrix<f64> {
    let mut m = Matrix::zeros(n, n);
    for i in 0..n {
        m[[i, i]] = 1.0;
    }
    m
}

#[test]
fn triangular_helpers_validate_shapes_and_pivots() {
    let p = from_vec2d(vec![vec![1.0_f64, 0.0], vec![0.0, 0.0]]);
    let mut columns = from_vec2d(vec![vec![1.0_f64], vec![2.0]]);
    assert!(cols_to_l_matrix(&mut columns, &p, true).is_err());

    let p = from_vec2d(vec![vec![1.0_f64, 0.0], vec![0.0, 1.0]]);
    let mut too_narrow = from_vec2d(vec![vec![1.0_f64], vec![2.0]]);
    assert!(cols_to_l_matrix(&mut too_narrow, &p, true).is_err());

    let mut rows = from_vec2d(vec![vec![1.0_f64, 2.0]]);
    assert!(rows_to_u_matrix(&mut rows, &p, true).is_err());
}

#[test]
fn test_rrlu_identity() {
    let m = identity(3);
    let lu = rrlu(&m, None).unwrap();

    assert_eq!(lu.npivots(), 3);
    assert!(lu.error.abs() < 1e-10 || lu.error == 0.0);
}

#[test]
fn test_rrlu_rank_deficient() {
    // Rank-1 matrix
    let m = from_vec2d(vec![
        vec![1.0, 2.0, 3.0],
        vec![2.0, 4.0, 6.0],
        vec![3.0, 6.0, 9.0],
    ]);

    let lu = rrlu(&m, None).unwrap();

    // Should detect rank-1
    assert_eq!(lu.npivots(), 1);
}

#[test]
fn test_rrlu_full_rank() {
    let m = from_vec2d(vec![
        vec![1.0, 2.0, 3.0],
        vec![4.0, 5.0, 6.0],
        vec![7.0, 8.0, 10.0], // Not rank-deficient
    ]);

    let lu = rrlu(&m, None).unwrap();

    assert_eq!(lu.npivots(), 3);
}

#[test]
fn test_rrlu_reconstruct() {
    let m = from_vec2d(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);

    let lu = rrlu(&m, None).unwrap();
    let l = lu.left(true);
    let u = lu.right(true);

    // Reconstruct: L * U should approximate original matrix
    let reconstructed = mat_mul(&l, &u).unwrap();

    for i in 0..2 {
        for j in 0..2 {
            let diff = (m[[i, j]] - reconstructed[[i, j]]).abs();
            assert!(
                diff < 1e-10,
                "Reconstruction error at ({}, {}): {}",
                i,
                j,
                diff
            );
        }
    }
}

#[test]
fn rrlu_shape_validation_rejects_overflow_and_length_mismatch() {
    assert!(validate_col_major_matrix_len(usize::MAX, 2, 0)
        .unwrap_err()
        .to_string()
        .contains("overflows"));
    assert!(validate_col_major_matrix_len(2, 2, 3)
        .unwrap_err()
        .to_string()
        .contains("expected 4"));
    validate_col_major_matrix_len(0, usize::MAX, 0).unwrap();
}

#[test]
fn test_rrlu_complex64_reconstruct() {
    let m = from_vec2d(vec![
        vec![Complex64::new(1.0, 1.0), Complex64::new(2.0, -1.0)],
        vec![Complex64::new(3.0, -2.0), Complex64::new(5.0, 0.5)],
    ]);

    let lu = rrlu(&m, None).unwrap();
    assert_eq!(lu.npivots(), 2);

    let reconstructed = mat_mul(&lu.left(true), &lu.right(true)).unwrap();
    for i in 0..m.nrows() {
        for j in 0..m.ncols() {
            let residual = (m[[i, j]] - reconstructed[[i, j]]).norm();
            assert!(
                residual < 1e-10,
                "Complex64 reconstruction residual at ({i}, {j}): {residual}"
            );
        }
    }
}

/// Regression test for issue #227:
/// crossinterpolate2 panics with NaN in LU for oscillatory functions (e.g. sin)
#[test]
fn test_rrlu_no_nan_panic() {
    // Matrix that could produce near-zero pivots during elimination
    let m = from_vec2d(vec![
        vec![1e-20, 1.0, 0.0],
        vec![1.0, 1e-20, 0.0],
        vec![0.0, 0.0, 1e-20],
    ]);

    // Should return Ok, not panic
    let result = rrlu(&m, None);
    assert!(result.is_ok());
}

#[test]
fn test_rrlu_zero_matrix() {
    let m = from_vec2d(vec![
        vec![0.0, 0.0, 0.0],
        vec![0.0, 0.0, 0.0],
        vec![0.0, 0.0, 0.0],
    ]);

    let lu = rrlu(&m, None).unwrap();
    // Zero matrix should have 0 pivots
    assert_eq!(lu.npivots(), 0);
}

/// Julia: "Implementation of rank-revealing LU" (4x4 matrix)
#[test]
fn test_rrlu_4x4_reconstruct() {
    let m = from_vec2d(vec![
        vec![0.711002, 0.724557, 0.789335, 0.382373],
        vec![0.910429, 0.726781, 0.719957, 0.486302],
        vec![0.632716, 0.39967, 0.571809, 0.0803125],
        vec![0.885709, 0.531645, 0.569399, 0.481214],
    ]);

    let lu = rrlu(&m, None).unwrap();
    assert_eq!(lu.nrows(), 4);
    assert_eq!(lu.ncols(), 4);

    // L (unpermuted) should be unit lower triangular
    let l = lu.left(false);
    for i in 0..l.nrows() {
        assert!((l[[i, i]] - 1.0).abs() < 1e-14, "L diagonal should be 1");
        for j in (i + 1)..l.ncols() {
            assert!(l[[i, j]].abs() < 1e-14, "L should be lower triangular");
        }
    }

    // U (unpermuted) should be upper triangular
    let u = lu.right(false);
    for i in 0..u.nrows() {
        for j in 0..i {
            assert!(u[[i, j]].abs() < 1e-14, "U should be upper triangular");
        }
    }

    // L * U (permuted) should reconstruct original matrix
    let l_perm = lu.left(true);
    let u_perm = lu.right(true);
    let reconstructed = mat_mul(&l_perm, &u_perm).unwrap();
    for i in 0..4 {
        for j in 0..4 {
            assert!(
                (m[[i, j]] - reconstructed[[i, j]]).abs() < 1e-10,
                "Reconstruction error at ({}, {})",
                i,
                j,
            );
        }
    }
}

/// Julia: "Truncated rank-revealing LU" (max_bond_dim stopping)
#[test]
fn test_rrlu_max_bond_dim() {
    let m = from_vec2d(vec![
        vec![0.684025, 0.784249, 0.826742, 0.054321, 0.0234695, 0.467096],
        vec![0.73928, 0.295516, 0.877126, 0.111711, 0.103509, 0.653785],
        vec![0.394016, 0.753239, 0.889128, 0.291669, 0.873509, 0.0965536],
        vec![0.378539, 0.0123737, 0.20112, 0.758088, 0.973042, 0.308372],
        vec![0.235156, 0.51939, 0.788184, 0.363171, 0.230001, 0.984971],
        vec![0.893223, 0.220834, 0.18001, 0.258537, 0.396583, 0.142105],
        vec![0.0417881, 0.890706, 0.328631, 0.279332, 0.963188, 0.706944],
        vec![0.914298, 0.792345, 0.311083, 0.129653, 0.350062, 0.683966],
    ]);

    let opts = RrLUOptions {
        max_bond_dim: 4,
        ..Default::default()
    };
    let lu = rrlu(&m, Some(opts)).unwrap();
    assert_eq!(lu.row_indices().len(), 4);
    assert_eq!(lu.col_indices().len(), 4);

    let l = lu.left(false);
    assert_eq!(l.nrows(), 8);
    assert_eq!(l.ncols(), 4);

    let u = lu.right(false);
    assert_eq!(u.nrows(), 4);
    assert_eq!(u.ncols(), 6);
}

/// Julia: "rrLU for exact low-rank matrix" (rank-3 from outer product)
#[test]
fn test_rrlu_exact_low_rank() {
    // p: 10x3, q: 3x10 => A = p*q is rank-3
    let p = from_vec2d(vec![
        vec![0.284975, 0.505168, 0.570921],
        vec![0.302884, 0.475901, 0.645776],
        vec![0.622955, 0.361755, 0.99539],
        vec![0.748447, 0.354849, 0.431366],
        vec![0.28338, 0.0378148, 0.994162],
        vec![0.643177, 0.74173, 0.802733],
        vec![0.58113, 0.526715, 0.879048],
        vec![0.238002, 0.557812, 0.251512],
        vec![0.458861, 0.141355, 0.0306212],
        vec![0.490269, 0.810266, 0.7946],
    ]);
    let q = from_vec2d(vec![
        vec![
            0.239552, 0.306094, 0.299063, 0.0382492, 0.185462, 0.0334971, 0.697561, 0.389596,
            0.105665, 0.0912763,
        ],
        vec![
            0.0570609, 0.56623, 0.97183, 0.994184, 0.371695, 0.284437, 0.993251, 0.902347,
            0.572944, 0.0531369,
        ],
        vec![
            0.45002, 0.461168, 0.6086, 0.613702, 0.543997, 0.759954, 0.0959818, 0.638499, 0.407382,
            0.482592,
        ],
    ]);

    let a = mat_mul(&p, &q).unwrap();
    let lu = rrlu(&a, None).unwrap();

    assert_eq!(lu.npivots(), 3);

    // Reconstruct
    let l = lu.left(true);
    let u = lu.right(true);
    let reconstructed = mat_mul(&l, &u).unwrap();
    for i in 0..a.nrows() {
        for j in 0..a.ncols() {
            assert!(
                (a[[i, j]] - reconstructed[[i, j]]).abs() < 1e-10,
                "Reconstruction error at ({}, {})",
                i,
                j,
            );
        }
    }
}

/// Julia: "lastpivoterror for full-rank matrix"
#[test]
fn test_rrlu_pivot_errors_full_rank() {
    let m = identity(2);
    let lu = rrlu(&m, None).unwrap();

    let errors = lu.pivot_errors();
    assert_eq!(errors.len(), 3); // 2 pivots + last error
    assert!((errors[0] - 1.0).abs() < 1e-14);
    assert!((errors[1] - 1.0).abs() < 1e-14);
    assert!(errors[2].abs() < 1e-14); // last_pivot_error == 0 for full rank
    assert!(lu.last_pivot_error().abs() < 1e-14);
}

/// Julia: "lastpivoterror for limited maxrank or tolerance"
#[test]
fn test_rrlu_pivot_errors_truncated() {
    let m = from_vec2d(vec![
        vec![0.433088, 0.956638, 0.0907974, 0.0447859, 0.0196053],
        vec![0.855517, 0.782503, 0.291197, 0.540828, 0.358579],
        vec![0.37455, 0.536457, 0.205479, 0.75896, 0.701206],
        vec![0.47272, 0.0172539, 0.518177, 0.242864, 0.461635],
        vec![0.0676373, 0.450878, 0.672335, 0.77726, 0.540691],
    ]);

    // max_bond_dim=2
    let opts = RrLUOptions {
        max_bond_dim: 2,
        ..Default::default()
    };
    let lu = rrlu(&m, Some(opts)).unwrap();
    assert_eq!(lu.pivot_errors().len(), 3); // 2 pivots + last error
    assert!(lu.last_pivot_error() > 0.0);

    // abstol=0.5
    let opts2 = RrLUOptions {
        abs_tol: 0.5,
        ..Default::default()
    };
    let lu2 = rrlu(&m, Some(opts2)).unwrap();
    assert!(lu2.last_pivot_error() < 0.5);

    // abstol=0.0, should compute full rank
    let opts3 = RrLUOptions {
        abs_tol: 0.0,
        ..Default::default()
    };
    let lu3 = rrlu(&m, Some(opts3)).unwrap();
    assert!(lu3.last_pivot_error().abs() < 1e-14);
}

/// Julia: "LU for matrices with very small absolute values"
#[test]
fn test_rrlu_small_values_abstol() {
    let scale = 1e-13;
    let m = from_vec2d(vec![
        vec![
            scale * 0.585383,
            scale * 0.124568,
            scale * 0.352426,
            scale * 0.573507,
        ],
        vec![
            scale * 0.865875,
            scale * 0.600153,
            scale * 0.727443,
            scale * 0.902388,
        ],
        vec![
            scale * 0.913477,
            scale * 0.954081,
            scale * 0.116965,
            scale * 0.817,
        ],
        vec![
            scale * 0.985918,
            scale * 0.516114,
            scale * 0.600366,
            scale * 0.0200085,
        ],
    ]);

    let opts = RrLUOptions {
        abs_tol: 1e-3,
        ..Default::default()
    };
    let lu = rrlu(&m, Some(opts)).unwrap();
    assert_eq!(lu.npivots(), 1);
    assert!(!lu.pivot_errors().is_empty());
    assert!(lu.last_pivot_error() > 0.0);
}

/// Julia: "transpose"
#[test]
fn test_rrlu_transpose() {
    let m = from_vec2d(vec![
        vec![1.0, 2.0, 3.0, 4.0],
        vec![5.0, 6.0, 7.0, 8.0],
        vec![9.0, 10.0, 12.0, 11.0],
    ]);

    let lu = rrlu(&m, None).unwrap();
    let tlu = lu.transpose();

    let l = tlu.left(true);
    let u = tlu.right(true);
    let reconstructed = mat_mul(&l, &u).unwrap();
    let mt = transpose(&m);

    for i in 0..mt.nrows() {
        for j in 0..mt.ncols() {
            assert!(
                (mt[[i, j]] - reconstructed[[i, j]]).abs() < 1e-10,
                "Transpose reconstruction error at ({}, {})",
                i,
                j,
            );
        }
    }
}
