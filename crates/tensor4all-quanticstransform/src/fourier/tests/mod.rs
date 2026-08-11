use super::*;
use approx::assert_relative_eq;
use tensor4all_simplett::AbstractTensorTrain;

#[test]
fn test_chebyshev_grid() {
    let (grid, weights) = chebyshev_grid(4).unwrap();

    // Check endpoints
    assert_relative_eq!(grid[0], 0.0, epsilon = 1e-10);
    assert_relative_eq!(grid[4], 1.0, epsilon = 1e-10);

    // Check symmetry around 0.5
    assert_relative_eq!(grid[1] + grid[3], 1.0, epsilon = 1e-10);
    assert_relative_eq!(grid[2], 0.5, epsilon = 1e-10);

    // Weights should be non-zero
    for w in &weights {
        assert!(w.abs() > 1e-20);
    }
}

#[test]
fn test_lagrange_polynomial_at_grid_points() {
    let (grid, weights) = chebyshev_grid(5).unwrap();
    // P_alpha(grid[alpha]) = 1
    for alpha in 0..=5 {
        let val = lagrange_polynomial(&grid, &weights, alpha, grid[alpha]);
        assert_relative_eq!(val, 1.0, epsilon = 1e-10);
    }

    // P_alpha(grid[beta]) = 0 for alpha != beta
    for alpha in 0..=5 {
        for beta in 0..=5 {
            if alpha != beta {
                let val = lagrange_polynomial(&grid, &weights, alpha, grid[beta]);
                assert_relative_eq!(val, 0.0, epsilon = 1e-10);
            }
        }
    }
}

#[test]
fn test_fourier_mpo_structure() {
    let options = FourierOptions::default();
    let mpo = quantics_fourier_mpo(4, &options).unwrap();
    assert_eq!(mpo.len(), 4);

    // Bond dimensions should be compressed from K+1
    // After compression, they should be <= max_bond_dim
    for i in 0..3 {
        assert!(options
            .max_bond_dim
            .is_none_or(|cap| mpo.site_tensor(i).right_dim() <= cap));
    }
}

#[test]
fn test_fourier_operator_creation() {
    let options = FourierOptions::default();
    let op = quantics_fourier_operator(4, options);
    assert!(op.is_ok());
}

#[test]
fn test_ftcore_creation() {
    let options = FourierOptions::default();
    let ft = FTCore::new(4, options);
    assert!(ft.is_ok());

    let ft = ft.unwrap();
    assert_eq!(ft.r(), 4);

    let forward = ft.forward();
    assert!(forward.is_ok());

    let backward = ft.backward();
    assert!(backward.is_ok());
}

#[test]
fn test_fourier_inverse_sign() {
    let forward_options = FourierOptions::forward();
    let inverse_options = FourierOptions::inverse();

    assert_eq!(forward_options.sign, -1.0);
    assert_eq!(inverse_options.sign, 1.0);
}

#[test]
fn test_fourier_error_zero_sites() {
    let options = FourierOptions::default();
    let result = quantics_fourier_operator(0, options);
    assert!(result.is_err());
}

#[test]
fn test_fourier_error_one_site() {
    let options = FourierOptions::default();
    let result = quantics_fourier_operator(1, options);
    assert!(result.is_err());
    let msg = result.unwrap_err().to_string();
    assert!(msg.contains("at least 2"), "unexpected error: {msg}");
}

#[test]
fn test_fourier_error_oversized_site_list() {
    let error = quantics_fourier_operator(usize::MAX, FourierOptions::default()).unwrap_err();
    assert!(error.to_string().contains("site list"));
}

#[test]
fn test_ftcore_error_one_site() {
    let options = FourierOptions::default();
    let result = FTCore::new(1, options);
    assert!(result.is_err());
    let msg = result.err().unwrap().to_string();
    assert!(msg.contains("at least 2"), "unexpected error: {msg}");
}

#[test]
fn test_fourier_rejects_zero_chebyshev_order() {
    let options = FourierOptions {
        k: 0,
        ..FourierOptions::default()
    };
    let error = quantics_fourier_operator(2, options).unwrap_err();
    assert!(error.to_string().contains("k must be positive"));
}

#[test]
fn test_fourier_rejects_quadratic_core_element_overflow() {
    let options = FourierOptions {
        k: usize::MAX / 4,
        ..FourierOptions::default()
    };
    let error = quantics_fourier_operator(2, options).unwrap_err();
    assert!(
        error.to_string().contains("Fourier core tensor"),
        "unexpected error: {error}"
    );
}

#[test]
fn test_fourier_grid_reports_actual_capacity_failure() {
    let error = chebyshev_grid(usize::MAX - 1).unwrap_err();
    assert!(
        error.to_string().contains("allocation failed"),
        "unexpected error: {error}"
    );
}

#[test]
fn test_fourier_tensor3_checks_byte_limit_before_wrapper() {
    let count = isize::MAX as usize / (4 * std::mem::size_of::<Complex64>()) + 1;
    let error = checked_tensor3_size(1, 4, count, "Fourier test tensor").unwrap_err();
    assert!(error.to_string().contains("byte length"));
}

#[test]
fn test_fourier_core_size_checks_product() {
    let error = checked_fourier_core_size(usize::MAX / 4).unwrap_err();
    assert!(error.to_string().contains("Fourier core tensor"));
}

#[test]
fn test_fourier_tensor4_checks_byte_limit_before_wrapper() {
    let count = isize::MAX as usize / std::mem::size_of::<Complex64>() + 1;
    let error = checked_tensor4_size([1, 1, 1, count], "Fourier test core").unwrap_err();
    assert!(error.to_string().contains("byte length"));
}
