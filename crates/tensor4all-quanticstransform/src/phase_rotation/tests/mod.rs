use super::*;
use approx::assert_relative_eq;
use tensor4all_simplett::AbstractTensorTrain;

#[test]
fn test_phase_rotation_mpo_structure() {
    let mpo = phase_rotation_mpo(4, PI / 4.0).unwrap();
    assert_eq!(mpo.len(), 4);

    // All tensors should have shape (1, 4, 1) - diagonal operator
    for i in 0..4 {
        assert_eq!(mpo.site_tensor(i).left_dim(), 1);
        assert_eq!(mpo.site_tensor(i).site_dim(), 4);
        assert_eq!(mpo.site_tensor(i).right_dim(), 1);
    }
}

#[test]
fn test_phase_rotation_zero_theta() {
    // θ = 0 should give identity
    let mpo = phase_rotation_mpo(4, 0.0).unwrap();

    for i in 0..4 {
        let t = mpo.site_tensor(i);
        // Check diagonal entries are 1
        assert_relative_eq!(t.get3(0, 0, 0).re, 1.0, epsilon = 1e-10);
        assert_relative_eq!(t.get3(0, 0, 0).im, 0.0, epsilon = 1e-10);
        assert_relative_eq!(t.get3(0, 3, 0).re, 1.0, epsilon = 1e-10);
        assert_relative_eq!(t.get3(0, 3, 0).im, 0.0, epsilon = 1e-10);
    }
}

#[test]
fn test_phase_rotation_pi() {
    // θ = π should give (-1)^x
    // With big-endian: site 0 (MSB) has phase π * 2^(R-1), site R-1 (LSB) has phase π * 2^0
    let r = 3;
    let mpo = phase_rotation_mpo(r, PI).unwrap();

    // For site 0 (MSB), phase = π * 2^2 = 4π ≡ 0 (mod 2π), exp(0) = 1
    let t_first = mpo.site_tensor(0);
    assert_relative_eq!(t_first.get3(0, 0, 0).re, 1.0, epsilon = 1e-10);
    assert_relative_eq!(t_first.get3(0, 3, 0).re, 1.0, epsilon = 1e-10);
    assert_relative_eq!(t_first.get3(0, 3, 0).im, 0.0, epsilon = 1e-10);

    // For site 2 (LSB), phase = π * 2^0 = π, exp(i*π) = -1
    let t_last = mpo.site_tensor(r - 1);
    assert_relative_eq!(t_last.get3(0, 0, 0).re, 1.0, epsilon = 1e-10);
    assert_relative_eq!(t_last.get3(0, 3, 0).re, -1.0, epsilon = 1e-10);
    assert_relative_eq!(t_last.get3(0, 3, 0).im, 0.0, epsilon = 1e-10);
}

#[test]
fn test_phase_rotation_operator_creation() {
    let op = phase_rotation_operator(4, PI / 2.0);
    assert!(op.is_ok());
}

#[test]
fn test_phase_rotation_error_zero_sites() {
    let result = phase_rotation_operator(0, PI);
    assert!(result.is_err());
}

#[test]
fn test_phase_rotation_error_oversized_site_list() {
    let error = phase_rotation_operator(usize::MAX, PI).unwrap_err();
    assert!(error.to_string().contains("site list"));
}

#[test]
fn test_phase_rotation_periodicity() {
    // θ and θ + 2π should give the same result
    let mpo1 = phase_rotation_mpo(4, PI / 3.0).unwrap();
    let mpo2 = phase_rotation_mpo(4, PI / 3.0 + 2.0 * PI).unwrap();

    for i in 0..4 {
        let t1 = mpo1.site_tensor(i);
        let t2 = mpo2.site_tensor(i);

        assert_relative_eq!(t1.get3(0, 0, 0).re, t2.get3(0, 0, 0).re, epsilon = 1e-10);
        assert_relative_eq!(t1.get3(0, 0, 0).im, t2.get3(0, 0, 0).im, epsilon = 1e-10);
        assert_relative_eq!(t1.get3(0, 3, 0).re, t2.get3(0, 3, 0).re, epsilon = 1e-10);
        assert_relative_eq!(t1.get3(0, 3, 0).im, t2.get3(0, 3, 0).im, epsilon = 1e-10);
    }
}

#[test]
fn test_phase_rotation_error_nan() {
    let result = phase_rotation_operator(4, f64::NAN);
    assert!(result.is_err());
    let msg = result.unwrap_err().to_string();
    assert!(msg.contains("finite"), "unexpected error: {msg}");
}

#[test]
fn test_phase_rotation_error_inf() {
    let result = phase_rotation_operator(4, f64::INFINITY);
    assert!(result.is_err());
    let msg = result.unwrap_err().to_string();
    assert!(msg.contains("finite"), "unexpected error: {msg}");
}

#[test]
fn test_phase_rotation_error_neg_inf() {
    let result = phase_rotation_operator(4, f64::NEG_INFINITY);
    assert!(result.is_err());
}

#[test]
fn test_phase_rotation_multivar_error_nan() {
    let result = phase_rotation_operator_multivar(4, f64::NAN, 2, 0);
    assert!(result.is_err());
}
