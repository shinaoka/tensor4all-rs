use super::*;

#[test]
fn test_factorize_svd() {
    let mut matrix: Matrix2<f64> = matrix2_zeros(4, 3);
    for i in 0..4 {
        for j in 0..3 {
            matrix[[i, j]] = (i * 3 + j + 1) as f64;
        }
    }

    let options = FactorizeOptions {
        method: FactorizeMethod::SVD,
        tolerance: 1e-12,
        max_bond_dim: Some(10),
        left_orthogonal: true,
        ..Default::default()
    };

    let result = factorize(&matrix, &options).unwrap();
    assert!(result.rank >= 1);
    assert!(result.rank <= 3); // Max rank is min(4, 3) = 3

    // Verify reconstruction: L @ R ≈ original
    let m = 4;
    let n = 3;
    for i in 0..m {
        for j in 0..n {
            let mut reconstructed = 0.0;
            for k in 0..result.rank {
                reconstructed += result.left[[i, k]] * result.right[[k, j]];
            }
            let original = matrix[[i, j]];
            assert!(
                (reconstructed - original).abs() < 1e-10,
                "Reconstruction failed at [{}, {}]: {} vs {}",
                i,
                j,
                reconstructed,
                original
            );
        }
    }
}

#[test]
fn test_factorize_lu() {
    let mut matrix: Matrix2<f64> = matrix2_zeros(4, 3);
    for i in 0..4 {
        for j in 0..3 {
            matrix[[i, j]] = (i * 3 + j) as f64;
        }
    }

    let options = FactorizeOptions {
        method: FactorizeMethod::LU,
        tolerance: 1e-12,
        max_bond_dim: Some(10),
        left_orthogonal: true,
        ..Default::default()
    };

    let result = factorize_lu(&matrix, &options).unwrap();
    assert!(result.rank >= 1);
    assert!(result.rank <= 3); // Max rank is min(4, 3) = 3
}

#[test]
fn test_factorize_with_truncation() {
    // Create a rank-2 matrix
    let mut matrix: Matrix2<f64> = matrix2_zeros(4, 4);
    for i in 0..4 {
        for j in 0..4 {
            // Rank-2: outer product of [1,2,3,4] and [1,1,1,1] + [1,1,1,1] and [1,2,3,4]
            matrix[[i, j]] = (i + 1) as f64 + (j + 1) as f64;
        }
    }

    let options = FactorizeOptions {
        method: FactorizeMethod::SVD,
        tolerance: 1e-10,
        max_bond_dim: Some(2),
        left_orthogonal: true,
        ..Default::default()
    };

    let result = factorize(&matrix, &options).unwrap();
    assert!(result.rank <= 2);
}

/// Build a matrix with a geometrically decaying spectrum, scaled by `scale`.
///
/// Singular values are `scale * 1.0`, `scale * 1e-3`, `scale * 1e-6`,
/// `scale * 1e-9`, `scale * 1e-12` up to orthogonal factors, so a relative
/// tolerance of `1e-7` must keep exactly the first three.
fn decaying_spectrum_matrix(scale: f64) -> Matrix2<f64> {
    let n = 5;
    let sigma = [1.0, 1e-3, 1e-6, 1e-9, 1e-12];

    // Orthogonal factors from a discrete sine transform, so the constructed
    // matrix is dense but its spectrum is exactly `scale * sigma`.
    let basis = |i: usize, k: usize| -> f64 {
        (2.0 / (n as f64 + 1.0)).sqrt()
            * (std::f64::consts::PI * ((i + 1) * (k + 1)) as f64 / (n as f64 + 1.0)).sin()
    };

    let mut matrix: Matrix2<f64> = matrix2_zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            let mut value = 0.0;
            for (k, &s) in sigma.iter().enumerate() {
                value += basis(i, k) * scale * s * basis(k, j);
            }
            matrix[[i, j]] = value;
        }
    }
    matrix
}

#[test]
fn test_factorize_svd_truncation_is_scale_invariant() {
    let options = FactorizeOptions {
        method: FactorizeMethod::SVD,
        tolerance: 1e-7,
        max_bond_dim: Some(10),
        left_orthogonal: true,
        ..Default::default()
    };

    let unscaled = factorize(&decaying_spectrum_matrix(1.0), &options).unwrap();
    assert_eq!(
        unscaled.rank, 3,
        "relative tolerance 1e-7 should keep sigma >= 1e-7 * sigma_max"
    );

    // With an absolute cutoff the scaled matrix would keep extra near-noise
    // singular values (1e6 * 1e-9 and 1e6 * 1e-12 both exceed 1e-7).
    for scale in [1e6, 1e-6] {
        let scaled = factorize(&decaying_spectrum_matrix(scale), &options).unwrap();
        assert_eq!(
            scaled.rank, unscaled.rank,
            "rank must not depend on the overall scale {scale} of the matrix"
        );
    }
}

#[test]
fn test_factorize_svd_zero_matrix_returns_rank_one() {
    let matrix: Matrix2<f64> = matrix2_zeros(3, 4);
    let options = FactorizeOptions {
        method: FactorizeMethod::SVD,
        tolerance: 1e-10,
        ..Default::default()
    };

    let result = factorize(&matrix, &options).unwrap();
    assert_eq!(result.rank, 1);
    assert_eq!(result.discarded, 0.0);
    for i in 0..3 {
        for j in 0..4 {
            let value = result.left[[i, 0]] * result.right[[0, j]];
            assert!(value.abs() < 1e-15, "zero matrix reconstructed as {value}");
        }
    }
}

#[test]
fn test_factorize_svd_complex64() {
    use num_complex::Complex64;

    let mut matrix: Matrix2<Complex64> = matrix2_zeros(4, 3);
    for i in 0..4 {
        for j in 0..3 {
            // Create complex values with both real and imaginary parts
            let re = (i * 3 + j + 1) as f64;
            let im = ((i + j) % 3) as f64 * 0.5;
            matrix[[i, j]] = Complex64::new(re, im);
        }
    }

    let options = FactorizeOptions {
        method: FactorizeMethod::SVD,
        tolerance: 1e-12,
        max_bond_dim: Some(10),
        left_orthogonal: true,
        ..Default::default()
    };

    let result = factorize(&matrix, &options).unwrap();
    assert!(result.rank >= 1);
    assert!(result.rank <= 3); // Max rank is min(4, 3) = 3

    // Verify reconstruction: L @ R ≈ original
    let m = 4;
    let n = 3;
    let mut max_error: f64 = 0.0;
    for i in 0..m {
        for j in 0..n {
            let mut reconstructed = Complex64::new(0.0, 0.0);
            for k in 0..result.rank {
                reconstructed += result.left[[i, k]] * result.right[[k, j]];
            }
            let original = matrix[[i, j]];
            let error = (reconstructed - original).norm();
            max_error = max_error.max(error);
        }
    }
    assert!(
        max_error < 1e-10,
        "Reconstruction error too large: {}",
        max_error
    );
}
