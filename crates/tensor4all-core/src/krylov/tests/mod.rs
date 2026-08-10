use super::*;
use crate::defaults::tensordynlen::TensorDynLen;
use crate::defaults::DynIndex;
use crate::tensor_index::TensorIndex;
use crate::{TensorVectorSpace, TensorVectorSpaceError};
use num_complex::Complex64;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

static GMRES_PROFILE_ENV_LOCK: Mutex<()> = Mutex::new(());

struct GmresProfileEnvGuard;

impl GmresProfileEnvGuard {
    fn set() -> Self {
        unsafe {
            std::env::set_var("T4A_GMRES_OP_PROFILE", "1");
        }
        Self
    }
}

impl Drop for GmresProfileEnvGuard {
    fn drop(&mut self) {
        unsafe {
            std::env::remove_var("T4A_GMRES_OP_PROFILE");
        }
    }
}

fn with_gmres_profile_env<R>(f: impl FnOnce() -> R) -> R {
    let _lock = GMRES_PROFILE_ENV_LOCK.lock().unwrap();
    let _guard = GmresProfileEnvGuard::set();
    f()
}

#[derive(Debug, Clone)]
struct PlainVector {
    data: Vec<f64>,
}

impl TensorIndex for PlainVector {
    type Index = DynIndex;

    fn external_indices(&self) -> Vec<Self::Index> {
        Vec::new()
    }

    fn replaceind(&self, _old_index: &Self::Index, _new_index: &Self::Index) -> Result<Self> {
        Ok(self.clone())
    }

    fn replaceinds(
        &self,
        _old_indices: &[Self::Index],
        _new_indices: &[Self::Index],
    ) -> Result<Self> {
        Ok(self.clone())
    }
}

impl TensorVectorSpace for PlainVector {
    type Error = TensorVectorSpaceError;

    fn norm_squared(&self) -> std::result::Result<f64, Self::Error> {
        Ok(self.data.iter().map(|x| x * x).sum())
    }

    fn axpby(&self, a: AnyScalar, other: &Self, b: AnyScalar) -> Result<Self> {
        anyhow::ensure!(
            self.data.len() == other.data.len(),
            "vector lengths must match"
        );
        anyhow::ensure!(
            a.is_real() && b.is_real(),
            "PlainVector test helper only supports real coefficients"
        );
        let data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(&x, &y)| a.real() * x + b.real() * y)
            .collect();
        Ok(Self { data })
    }

    fn scale(&self, scalar: AnyScalar) -> Result<Self> {
        anyhow::ensure!(
            scalar.is_real(),
            "PlainVector test helper only supports real coefficients"
        );
        Ok(Self {
            data: self.data.iter().map(|&x| scalar.real() * x).collect(),
        })
    }

    fn inner_product(&self, other: &Self) -> Result<AnyScalar> {
        anyhow::ensure!(
            self.data.len() == other.data.len(),
            "vector lengths must match"
        );
        Ok(AnyScalar::new_real(
            self.data
                .iter()
                .zip(other.data.iter())
                .map(|(&x, &y)| x * y)
                .sum(),
        ))
    }

    fn maxabs(&self) -> std::result::Result<f64, Self::Error> {
        Ok(self.data.iter().map(|x| x.abs()).fold(0.0, f64::max))
    }
}

#[test]
fn hermitian_lanczos_lowest_eigenpair_diagonal_plain_vector() {
    let apply = |x: &PlainVector| -> Result<PlainVector> {
        Ok(PlainVector {
            data: vec![2.0 * x.data[0], -x.data[1], 4.0 * x.data[2]],
        })
    };
    let initial = PlainVector {
        data: vec![1.0, 1.0, 1.0],
    };

    let result =
        hermitian_lanczos_lowest_eigenpair(apply, &initial, &HermitianLanczosOptions::default())
            .unwrap();

    assert!(result.converged, "residual={}", result.residual_norm);
    assert!((result.eigenvalue + 1.0).abs() < 1.0e-10);
    assert!(result.residual_norm < 1.0e-8);
    assert!((result.eigenvector.data[1].abs() - 1.0).abs() < 1.0e-8);
}

#[test]
fn hermitian_lanczos_avoids_true_residual_matvec_on_every_iteration() {
    let apply_calls = AtomicUsize::new(0);
    let apply = |x: &PlainVector| -> Result<PlainVector> {
        apply_calls.fetch_add(1, Ordering::Relaxed);
        Ok(PlainVector {
            data: vec![2.0 * x.data[0], -x.data[1], 4.0 * x.data[2]],
        })
    };
    let initial = PlainVector {
        data: vec![1.0, 1.0, 1.0],
    };

    let result = hermitian_lanczos_lowest_eigenpair(
        apply,
        &initial,
        &HermitianLanczosOptions {
            max_iter: 3,
            rtol: 1.0e-12,
            ..HermitianLanczosOptions::default()
        },
    )
    .unwrap();

    assert!(result.converged, "residual={}", result.residual_norm);
    assert_eq!(result.iterations, 3);
    assert_eq!(apply_calls.load(Ordering::Relaxed), result.iterations + 1);
}

#[test]
fn hermitian_lanczos_lowest_eigenpair_rejects_zero_initial_vector() {
    let apply = |x: &PlainVector| -> Result<PlainVector> { Ok(x.clone()) };
    let initial = PlainVector {
        data: vec![0.0, 0.0],
    };

    let err =
        hermitian_lanczos_lowest_eigenpair(apply, &initial, &HermitianLanczosOptions::default())
            .unwrap_err();

    assert!(err.to_string().contains("zero initial vector"));
}

#[test]
fn hermitian_lanczos_lowest_eigenpair_complex_pauli_y() {
    let idx = DynIndex::new_dyn(2);
    let initial = make_vector_c64(
        vec![Complex64::new(1.0, 0.0), Complex64::new(0.25, 0.5)],
        &idx,
    );
    let minus_i = Complex64::new(0.0, -1.0);
    let plus_i = Complex64::new(0.0, 1.0);
    let pauli_y = [
        Complex64::new(0.0, 0.0),
        minus_i,
        plus_i,
        Complex64::new(0.0, 0.0),
    ];

    let result = hermitian_lanczos_lowest_eigenpair(
        |x: &TensorDynLen| apply_matrix2_c64(x, &pauli_y),
        &initial,
        &HermitianLanczosOptions {
            max_iter: 4,
            rtol: 1.0e-12,
            ..HermitianLanczosOptions::default()
        },
    )
    .unwrap();

    assert!(result.converged, "residual={}", result.residual_norm);
    assert!((result.eigenvalue + 1.0).abs() < 1.0e-10);
    assert!(result.residual_norm < 1.0e-8);
}

#[test]
fn hermitian_lanczos_lowest_eigenpair_rejects_non_hermitian_projection() {
    let idx = DynIndex::new_dyn(2);
    let initial = make_vector_with_index(vec![1.0, 1.0], &idx);
    let non_hermitian = [0.0, 1.0, 0.0, 0.0];

    let err = hermitian_lanczos_lowest_eigenpair(
        |x: &TensorDynLen| apply_matrix2_f64(x, &non_hermitian),
        &initial,
        &HermitianLanczosOptions {
            max_iter: 2,
            rtol: 0.0,
            ..HermitianLanczosOptions::default()
        },
    )
    .unwrap_err();

    assert!(err.to_string().contains("not Hermitian"));
}

#[test]
fn hermitian_krylov_expm_multiply_diagonal_real_time_matches_exact() {
    let idx = DynIndex::new_dyn(2);
    let initial = make_vector_c64(
        vec![Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)],
        &idx,
    );

    let result = hermitian_krylov_expm_multiply(
        |x: &TensorDynLen| scale_vector_c64(x, &[1.0, 3.0]),
        Complex64::new(0.0, -0.25),
        &initial,
        &HermitianKrylovExpmOptions {
            max_iter: 8,
            tol: 1.0e-12,
            ..HermitianKrylovExpmOptions::default()
        },
    )
    .unwrap();

    assert!(result.converged, "error={:.3e}", result.error_estimate);
    let actual = result.output.to_vec::<Complex64>().unwrap();
    let expected0 = Complex64::new(0.25_f64.cos(), -0.25_f64.sin());
    let expected1 = Complex64::new(2.0 * 0.75_f64.cos(), -2.0 * 0.75_f64.sin());
    assert!((actual[0] - expected0).norm() < 1.0e-10);
    assert!((actual[1] - expected1).norm() < 1.0e-10);
}

#[test]
fn hermitian_krylov_expm_multiply_first_step_breakdown_reports_finite_error() {
    let idx = DynIndex::new_dyn(2);
    let initial = make_vector_c64(
        vec![Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)],
        &idx,
    );

    let result = hermitian_krylov_expm_multiply(
        |x: &TensorDynLen| scale_vector_c64(x, &[2.0, 2.0]),
        Complex64::new(0.0, -0.25),
        &initial,
        &HermitianKrylovExpmOptions::default(),
    )
    .unwrap();

    assert!(result.converged);
    assert_eq!(result.iterations, 1);
    assert!(result.error_estimate.is_finite());
    assert!(result.error_estimate <= 1.0e-15);
    let actual = result.output.to_vec::<Complex64>().unwrap();
    let phase = Complex64::new(0.5_f64.cos(), -0.5_f64.sin());
    assert!((actual[0] - phase).norm() < 1.0e-10);
    assert!((actual[1] - Complex64::new(2.0, 0.0) * phase).norm() < 1.0e-10);
}

#[test]
fn hermitian_krylov_expm_multiply_zero_exponent_returns_input() {
    let idx = DynIndex::new_dyn(2);
    let initial = make_vector_c64(
        vec![Complex64::new(2.0, 1.0), Complex64::new(-3.0, 0.5)],
        &idx,
    );

    let result = hermitian_krylov_expm_multiply(
        |_x: &TensorDynLen| panic!("zero exponent must not request a matvec"),
        Complex64::new(0.0, 0.0),
        &initial,
        &HermitianKrylovExpmOptions::default(),
    )
    .unwrap();

    assert!(result.converged);
    assert_eq!(result.matvecs, 0);
    assert_eq!(
        result.output.to_vec::<Complex64>().unwrap(),
        initial.to_vec::<Complex64>().unwrap()
    );
}

#[test]
fn hermitian_krylov_expm_multiply_zero_initial_returns_zero_without_matvecs() {
    let idx = DynIndex::new_dyn(2);
    let initial = make_vector_c64(vec![Complex64::new(0.0, 0.0); 2], &idx);

    let result = hermitian_krylov_expm_multiply(
        |_x: &TensorDynLen| panic!("zero input must not request a matvec"),
        Complex64::new(0.0, -0.25),
        &initial,
        &HermitianKrylovExpmOptions::default(),
    )
    .unwrap();

    assert!(result.converged);
    assert_eq!(result.matvecs, 0);
    assert_eq!(
        result.output.to_vec::<Complex64>().unwrap(),
        initial.to_vec::<Complex64>().unwrap()
    );
}

#[test]
fn hermitian_krylov_expm_multiply_real_tensor_promotes_complex_evolution() {
    let idx = DynIndex::new_dyn(2);
    let initial = make_vector_with_index(vec![1.0, 0.0], &idx);

    let result = hermitian_krylov_expm_multiply(
        |x: &TensorDynLen| apply_matrix2_f64(x, &[0.0, 1.0, 1.0, 0.0]),
        Complex64::new(0.0, -0.5),
        &initial,
        &HermitianKrylovExpmOptions::default(),
    )
    .unwrap();

    let actual = result.output.to_vec::<Complex64>().unwrap();
    assert!((actual[0] - Complex64::new(0.5_f64.cos(), 0.0)).norm() < 1.0e-10);
    assert!((actual[1] - Complex64::new(0.0, -0.5_f64.sin())).norm() < 1.0e-10);
}

#[test]
fn hermitian_krylov_expm_multiply_errors_when_unconverged() {
    let idx = DynIndex::new_dyn(3);
    let initial = make_vector_c64(
        vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0),
        ],
        &idx,
    );

    let err = hermitian_krylov_expm_multiply(
        |x: &TensorDynLen| scale_vector_c64(x, &[1.0, 2.0, 3.0]),
        Complex64::new(0.0, -1.0),
        &initial,
        &HermitianKrylovExpmOptions {
            max_iter: 1,
            max_time_splits: 1,
            tol: 1.0e-14,
            ..HermitianKrylovExpmOptions::default()
        },
    )
    .unwrap_err();

    assert!(err.to_string().contains("did not converge"));
}

#[test]
fn gmres_accepts_vector_space_without_tensorlike() {
    let b = PlainVector {
        data: vec![1.0, -2.0],
    };
    let x0 = PlainVector {
        data: vec![0.0, 0.0],
    };
    let result = gmres(
        |x: &PlainVector| Ok(x.clone()),
        &b,
        &x0,
        &GmresOptions::default(),
    )
    .expect("GMRES should accept TensorVectorSpace-only values");

    assert!(result.converged);
    assert!(result.solution.sub(&b).unwrap().maxabs().unwrap() < 1e-12);
}

#[test]
fn gmres_absolute_tolerance_and_total_iteration_limit_paths() {
    let idx = DynIndex::new_dyn(2);
    let b = make_vector_with_index(vec![4.0, 9.0], &idx);
    let x0 = make_vector_with_index(vec![0.0, 0.0], &idx);
    let apply_a = |x: &TensorDynLen| -> Result<TensorDynLen> { scale_vector_f64(x, &[2.0, 3.0]) };
    let options = GmresOptions {
        max_iter: 4,
        rtol: 1e-12,
        max_restarts: 2,
        verbose: false,
        check_true_residual: true,
    };

    let result = gmres_with_absolute_tolerance(apply_a, &b, &x0, &options, 1e-10).unwrap();
    assert!(result.converged);
    let expected = make_vector_with_index(vec![2.0, 3.0], &idx);
    assert!(result.solution.sub(&expected).unwrap().maxabs().unwrap() < 1e-10);

    let limited = gmres_with_total_iteration_limit(
        |x: &TensorDynLen| scale_vector_f64(x, &[2.0, 3.0]),
        &b,
        &x0,
        &options,
        0,
    )
    .unwrap();
    assert!(!limited.converged);
    assert_eq!(limited.iterations, 0);
}

#[test]
fn gmres_affine_matches_shifted_system_and_scalar_shortcuts() {
    let idx = DynIndex::new_dyn(2);
    let b = make_vector_with_index(vec![10.0, 28.0], &idx);
    let x0 = make_vector_with_index(vec![0.0, 0.0], &idx);
    let options = GmresOptions {
        max_iter: 4,
        rtol: 1e-12,
        max_restarts: 2,
        verbose: true,
        check_true_residual: true,
    };

    let result = gmres_affine_with_absolute_tolerance(
        |x: &TensorDynLen| scale_vector_f64(x, &[2.0, 5.0]),
        &b,
        &x0,
        AnyScalar::new_real(1.0),
        AnyScalar::new_real(2.0),
        &options,
        1e-10,
    )
    .unwrap();
    assert!(result.converged);
    let expected = make_vector_with_index(vec![2.0, 28.0 / 11.0], &idx);
    assert!(result.solution.sub(&expected).unwrap().maxabs().unwrap() < 1e-10);

    let scaled = gmres_affine(
        |x: &TensorDynLen| scale_vector_f64(x, &[100.0, 100.0]),
        &b,
        &x0,
        AnyScalar::new_real(2.0),
        AnyScalar::new_real(0.0),
        &options,
    )
    .unwrap();
    assert!(scaled.converged);
    assert_eq!(scaled.iterations, 0);
    assert!(
        scaled
            .solution
            .sub(&make_vector_with_index(vec![5.0, 14.0], &idx))
            .unwrap()
            .maxabs()
            .unwrap()
            < 1e-12
    );

    let err = gmres_affine(
        |x: &TensorDynLen| Ok(x.clone()),
        &b,
        &x0,
        AnyScalar::new_real(0.0),
        AnyScalar::new_real(0.0),
        &options,
    )
    .unwrap_err();
    assert!(err.to_string().contains("at least one affine coefficient"));
}

#[test]
fn gmres_affine_profile_and_zero_rhs_paths() {
    let idx = DynIndex::new_dyn(1);
    let b = make_vector_with_index(vec![0.0], &idx);
    let x0 = make_vector_with_index(vec![7.0], &idx);
    let options = GmresOptions {
        max_iter: 1,
        rtol: 1e-12,
        max_restarts: 1,
        verbose: false,
        check_true_residual: false,
    };

    let result = with_gmres_profile_env(|| {
        gmres_affine(
            |x: &TensorDynLen| Ok(x.clone()),
            &b,
            &x0,
            AnyScalar::new_real(1.0),
            AnyScalar::new_real(1.0),
            &options,
        )
        .unwrap()
    });

    assert!(result.converged);
    assert_eq!(result.iterations, 0);
    assert!(result.solution.sub(&x0).unwrap().maxabs().unwrap() < 1e-12);
}

#[test]
fn restart_gmres_options_builder_sets_all_fields() {
    let options = RestartGmresOptions::new()
        .with_max_outer_iters(3)
        .with_rtol(1e-7)
        .with_inner_max_iter(5)
        .with_inner_max_restarts(2)
        .with_min_reduction(0.8)
        .with_inner_rtol(0.25)
        .with_verbose(true);

    assert_eq!(options.max_outer_iters, 3);
    assert_eq!(options.rtol, 1e-7);
    assert_eq!(options.inner_max_iter, 5);
    assert_eq!(options.inner_max_restarts, 2);
    assert_eq!(options.min_reduction, Some(0.8));
    assert_eq!(options.inner_rtol, Some(0.25));
    assert!(options.verbose);
}

#[test]
fn gmres_affine_profile_covers_nonconverged_restart_and_final_paths() {
    let idx = DynIndex::new_dyn(2);
    let b = make_vector_with_index(vec![1.0, 1.0], &idx);
    let x0 = make_vector_with_index(vec![0.0, 0.0], &idx);
    let options = GmresOptions {
        max_iter: 1,
        rtol: 1e-30,
        max_restarts: 1,
        verbose: true,
        check_true_residual: false,
    };

    let result = with_gmres_profile_env(|| {
        gmres_affine(
            |x: &TensorDynLen| scale_vector_f64(x, &[2.0, 5.0]),
            &b,
            &x0,
            AnyScalar::new_real(0.5),
            AnyScalar::new_real(1.25),
            &options,
        )
        .unwrap()
    });

    assert!(!result.converged);
    assert_eq!(result.iterations, 1);
    assert!(result.residual_norm.is_finite());
}

#[test]
fn gmres_affine_profile_covers_scalar_shortcut() {
    let idx = DynIndex::new_dyn(2);
    let b = make_vector_with_index(vec![2.0, 4.0], &idx);
    let x0 = make_vector_with_index(vec![0.0, 0.0], &idx);
    let options = GmresOptions::default();

    let result = with_gmres_profile_env(|| {
        gmres_affine(
            |x: &TensorDynLen| Ok(x.clone()),
            &b,
            &x0,
            AnyScalar::new_real(2.0),
            AnyScalar::new_real(0.0),
            &options,
        )
        .unwrap()
    });

    assert!(result.converged);
    assert_eq!(result.iterations, 0);
    assert_eq!(result.solution.to_vec::<f64>().unwrap(), vec![1.0, 2.0]);
}

#[test]
fn gmres_lucky_breakdown_paths_are_reachable_with_zero_tolerance() {
    let idx = DynIndex::new_dyn(1);
    let b = make_vector_with_index(vec![3.0], &idx);
    let x0 = make_vector_with_index(vec![0.0], &idx);
    let options = GmresOptions {
        max_iter: 1,
        rtol: 0.0,
        max_restarts: 1,
        verbose: false,
        check_true_residual: false,
    };

    let result = gmres(|x: &TensorDynLen| Ok(x.clone()), &b, &x0, &options).unwrap();
    assert!(!result.converged);
    assert_eq!(result.iterations, 1);
    assert!(result.solution.sub(&b).unwrap().maxabs().unwrap() < 1e-12);

    let affine = gmres_affine(
        |x: &TensorDynLen| Ok(x.clone()),
        &b,
        &x0,
        AnyScalar::new_real(0.0),
        AnyScalar::new_real(1.0),
        &options,
    )
    .unwrap();
    assert!(!affine.converged);
    assert_eq!(affine.iterations, 1);
    assert!(affine.solution.sub(&b).unwrap().maxabs().unwrap() < 1e-12);

    let truncated = gmres_with_truncation(
        |x: &TensorDynLen| Ok(x.clone()),
        &b,
        &x0,
        &options,
        |_: &mut TensorDynLen| Ok(()),
    )
    .unwrap();
    assert!(!truncated.converged);
    assert_eq!(truncated.iterations, 1);
    assert!(truncated.solution.sub(&b).unwrap().maxabs().unwrap() < 1e-12);
}

#[test]
fn gmres_convergence_branches_cover_true_residual_and_affine_fast_finish() {
    let idx = DynIndex::new_dyn(1);
    let b = make_vector_with_index(vec![3.0], &idx);
    let x0 = make_vector_with_index(vec![0.0], &idx);
    let options = GmresOptions {
        max_iter: 1,
        rtol: 1e-12,
        max_restarts: 1,
        verbose: true,
        check_true_residual: true,
    };

    let checked = gmres(|x: &TensorDynLen| Ok(x.clone()), &b, &x0, &options).unwrap();
    assert!(checked.converged);
    assert!(checked.solution.sub(&b).unwrap().maxabs().unwrap() < 1e-12);

    let truncated = gmres_with_truncation(
        |x: &TensorDynLen| Ok(x.clone()),
        &b,
        &x0,
        &options,
        |_: &mut TensorDynLen| Ok(()),
    )
    .unwrap();
    assert!(truncated.converged);
    assert!(truncated.solution.sub(&b).unwrap().maxabs().unwrap() < 1e-12);

    let no_true_check = GmresOptions {
        check_true_residual: false,
        ..options
    };
    let affine = gmres_affine(
        |x: &TensorDynLen| Ok(x.clone()),
        &b,
        &x0,
        AnyScalar::new_real(0.0),
        AnyScalar::new_real(1.0),
        &no_true_check,
    )
    .unwrap();
    assert!(affine.converged);
    assert!(affine.solution.sub(&b).unwrap().maxabs().unwrap() < 1e-12);
}

#[test]
fn gmres_affine_profile_covers_true_residual_rejection_and_lucky_breakdown() {
    use std::cell::Cell;

    let idx = DynIndex::new_dyn(1);
    let b = make_vector_with_index(vec![1.0], &idx);
    let x0 = make_vector_with_index(vec![0.0], &idx);
    let checked_options = GmresOptions {
        max_iter: 1,
        rtol: 1e-12,
        max_restarts: 1,
        verbose: true,
        check_true_residual: true,
    };

    let calls = Cell::new(0usize);
    let checked = with_gmres_profile_env(|| {
        gmres_affine(
            |x: &TensorDynLen| {
                let call = calls.get();
                calls.set(call + 1);
                if call == 2 {
                    x.scale(AnyScalar::new_real(2.0))
                } else {
                    Ok(x.clone())
                }
            },
            &b,
            &x0,
            AnyScalar::new_real(0.0),
            AnyScalar::new_real(1.0),
            &checked_options,
        )
        .unwrap()
    });
    assert!(checked.converged);
    assert!(checked.solution.sub(&b).unwrap().maxabs().unwrap() < 1e-12);

    let lucky_options = GmresOptions {
        check_true_residual: false,
        rtol: 0.0,
        ..checked_options
    };
    let lucky = with_gmres_profile_env(|| {
        gmres_affine(
            |x: &TensorDynLen| Ok(x.clone()),
            &b,
            &x0,
            AnyScalar::new_real(0.0),
            AnyScalar::new_real(1.0),
            &lucky_options,
        )
        .unwrap()
    });
    assert!(!lucky.converged);
    assert_eq!(lucky.iterations, 1);
    assert!(lucky.solution.sub(&b).unwrap().maxabs().unwrap() < 1e-12);
}

#[test]
fn gmres_zero_cycle_and_restart_nonzero_update_paths() {
    let idx = DynIndex::new_dyn(1);
    let b = make_vector_with_index(vec![2.0], &idx);
    let x0 = make_vector_with_index(vec![0.0], &idx);
    let zero_cycle_options = GmresOptions {
        max_iter: 0,
        rtol: 1e-12,
        max_restarts: 1,
        verbose: false,
        check_true_residual: false,
    };

    let zero_cycle = gmres_with_total_iteration_limit(
        |x: &TensorDynLen| Ok(x.clone()),
        &b,
        &x0,
        &zero_cycle_options,
        1,
    )
    .unwrap();
    assert!(!zero_cycle.converged);
    assert_eq!(zero_cycle.iterations, 0);

    let restart_options = RestartGmresOptions {
        max_outer_iters: 2,
        rtol: 0.0,
        inner_max_iter: 1,
        inner_max_restarts: 0,
        min_reduction: None,
        inner_rtol: Some(0.0),
        verbose: true,
    };
    let restarted = restart_gmres_with_truncation(
        |x: &TensorDynLen| Ok(x.clone()),
        &b,
        None,
        &restart_options,
        |_: &mut TensorDynLen| Ok(()),
    )
    .unwrap();
    assert!(!restarted.converged);
    assert_eq!(restarted.outer_iterations, 2);
    assert!(restarted.solution.sub(&b).unwrap().maxabs().unwrap() < 1e-12);
}

#[test]
fn gmres_private_helpers_cover_edge_paths() {
    let zero = AnyScalar::new_real(0.0);
    let (c, s) = compute_givens_rotation(&zero, &zero);
    assert!(c.is_real());
    assert!(s.is_real());
    assert_eq!(c.real(), 1.0);
    assert_eq!(s.real(), 0.0);

    let empty = solve_upper_triangular(&[], &[]).unwrap();
    assert!(empty.is_empty());

    let singular = solve_upper_triangular(
        &[vec![AnyScalar::new_real(0.0)]],
        &[AnyScalar::new_real(1.0)],
    )
    .unwrap_err();
    assert!(singular.to_string().contains("Near-singular"));

    let idx = DynIndex::new_dyn(2);
    let x = make_vector_with_index(vec![1.0, 1.0], &idx);
    let v = make_vector_with_index(vec![2.0, -1.0], &idx);
    let updated = update_solution(&x, &[v], &[AnyScalar::new_real(3.0)]).unwrap();
    assert_eq!(updated.to_vec::<f64>().unwrap(), vec![7.0, -2.0]);
}

/// Helper to create a 1D tensor (vector) with given data and shared index.
fn make_vector_with_index(data: Vec<f64>, idx: &DynIndex) -> TensorDynLen {
    TensorDynLen::from_dense(vec![idx.clone()], data).unwrap()
}

fn scale_vector_f64(x: &TensorDynLen, diag: &[f64]) -> Result<TensorDynLen> {
    let x_data = x.to_vec::<f64>()?;
    let result_data: Vec<f64> = x_data
        .iter()
        .zip(diag.iter())
        .map(|(&xi, &di)| xi * di)
        .collect();
    Ok(TensorDynLen::from_dense(x.indices.clone(), result_data).unwrap())
}

fn scale_vector_c64(x: &TensorDynLen, diag: &[f64]) -> Result<TensorDynLen> {
    let x_data = x.to_vec::<Complex64>()?;
    let result_data: Vec<_> = x_data
        .iter()
        .zip(diag.iter())
        .map(|(&xi, &di)| xi * di)
        .collect();
    Ok(TensorDynLen::from_dense(x.indices.clone(), result_data).unwrap())
}

fn apply_matrix2_f64(x: &TensorDynLen, a_data: &[f64; 4]) -> Result<TensorDynLen> {
    let x_data = x.to_vec::<f64>()?;
    let result_data = vec![
        a_data[0] * x_data[0] + a_data[1] * x_data[1],
        a_data[2] * x_data[0] + a_data[3] * x_data[1],
    ];
    Ok(TensorDynLen::from_dense(x.indices.clone(), result_data).unwrap())
}

fn make_vector_c64(data: Vec<Complex64>, idx: &DynIndex) -> TensorDynLen {
    TensorDynLen::from_dense(vec![idx.clone()], data).unwrap()
}

fn apply_matrix2_c64(x: &TensorDynLen, a_data: &[Complex64; 4]) -> Result<TensorDynLen> {
    let x_data = x.to_vec::<Complex64>()?;
    let result_data = vec![
        a_data[0] * x_data[0] + a_data[1] * x_data[1],
        a_data[2] * x_data[0] + a_data[3] * x_data[1],
    ];
    Ok(TensorDynLen::from_dense(x.indices.clone(), result_data).unwrap())
}

#[test]
fn test_givens_rotation_real() {
    let a = AnyScalar::new_real(3.0);
    let b = AnyScalar::new_real(4.0);
    let (c, s) = compute_givens_rotation(&a, &b);

    // c = 3/5 = 0.6, s = 4/5 = 0.8
    assert!(!c.is_complex());
    assert!(!s.is_complex());
    assert!((c.real() - 0.6).abs() < 1e-10);
    assert!((s.real() - 0.8).abs() < 1e-10);
}

#[test]
fn test_apply_givens_rotation_real() {
    let c = AnyScalar::new_real(0.6);
    let s = AnyScalar::new_real(0.8);
    let x = AnyScalar::new_real(3.0);
    let y = AnyScalar::new_real(4.0);

    let (new_x, new_y) = apply_givens_rotation(&c, &s, &x, &y);

    // new_x = 0.6*3 + 0.8*4 = 1.8 + 3.2 = 5.0
    // new_y = -0.8*3 + 0.6*4 = -2.4 + 2.4 = 0.0
    assert!(!new_x.is_complex());
    assert!(!new_y.is_complex());
    assert!((new_x.real() - 5.0).abs() < 1e-10);
    assert!(new_y.real().abs() < 1e-10);
}

#[test]
fn test_givens_rotation_complex() {
    let a = AnyScalar::new_complex(3.0, 4.0);
    let b = AnyScalar::new_complex(1.0, -2.0);
    let (c, s) = compute_givens_rotation(&a, &b);

    assert!(c.is_real());
    assert!(s.is_complex());

    let (first, second) = apply_givens_rotation(&c, &s, &a, &b);
    let expected_norm = (3.0_f64 * 3.0 + 4.0 * 4.0 + 1.0 * 1.0 + 2.0 * 2.0).sqrt();
    assert!((first.abs() - expected_norm).abs() < 1e-12);
    assert!(second.abs() < 1e-12, "{second:?}");
    assert!(((c.abs() * c.abs() + s.abs() * s.abs()) - 1.0).abs() < 1e-12);
}

#[test]
fn test_apply_givens_rotation_complex() {
    let c = AnyScalar::new_real(0.6);
    let s = AnyScalar::new_complex(0.8, -0.2);
    let x = AnyScalar::new_complex(3.0, 1.0);
    let y = AnyScalar::new_complex(4.0, -2.0);

    let (new_x, new_y) = apply_givens_rotation(&c, &s, &x, &y);

    assert!(new_x.is_complex());
    assert!(new_y.is_complex());
    assert!(new_x.real().is_finite() && new_x.imag().is_finite());
    assert!(new_y.real().is_finite() && new_y.imag().is_finite());
}

#[test]
fn test_complex_givens_rotation_eliminates_second_component() {
    let cases = [
        (
            AnyScalar::new_complex(2.0, -3.0),
            AnyScalar::new_complex(-1.5, 0.75),
        ),
        (
            AnyScalar::new_complex(0.0, 0.0),
            AnyScalar::new_complex(1.0, -2.0),
        ),
        (
            AnyScalar::new_complex(-0.25, 0.5),
            AnyScalar::new_complex(3.0, 4.0),
        ),
    ];

    for (a, b) in cases {
        let (c, s) = compute_givens_rotation(&a, &b);
        let (_first, second) = apply_givens_rotation(&c, &s, &a, &b);
        assert!(second.abs() < 1e-12, "a={a:?}, b={b:?}, second={second:?}");
    }
}

#[test]
fn test_gmres_identity_operator() {
    // Solve A x = b where A = I (identity)
    // Solution: x = b
    let idx = DynIndex::new_dyn(3);
    let b = make_vector_with_index(vec![1.0, 2.0, 3.0], &idx);
    let x0 = make_vector_with_index(vec![0.0, 0.0, 0.0], &idx);

    // Identity operator: A x = x
    let apply_a = |x: &TensorDynLen| -> Result<TensorDynLen> { Ok(x.clone()) };

    let options = GmresOptions {
        max_iter: 10,
        rtol: 1e-10,
        max_restarts: 1,
        verbose: false,
        check_true_residual: false,
    };

    let result = gmres(apply_a, &b, &x0, &options).unwrap();

    assert!(result.converged, "GMRES should converge for identity");
    assert!(
        result.residual_norm < 1e-10,
        "Residual should be small: {}",
        result.residual_norm
    );

    // Check solution matches b
    let diff = result
        .solution
        .axpby(AnyScalar::new_real(1.0), &b, AnyScalar::new_real(-1.0))
        .unwrap();
    assert!(diff.norm().unwrap() < 1e-10, "Solution should equal b");
}

#[test]
fn test_gmres_diagonal_matrix() {
    // Solve A x = b where A = diag(2, 3, 4)
    // b = [2, 6, 12] → x = [1, 2, 3]
    let idx = DynIndex::new_dyn(3);
    let b = make_vector_with_index(vec![2.0, 6.0, 12.0], &idx);
    let x0 = make_vector_with_index(vec![0.0, 0.0, 0.0], &idx);
    let expected_x = make_vector_with_index(vec![1.0, 2.0, 3.0], &idx);

    // Diagonal scaling operator
    let diag = [2.0, 3.0, 4.0];
    let apply_a = move |x: &TensorDynLen| scale_vector_f64(x, &diag);

    let options = GmresOptions {
        max_iter: 10,
        rtol: 1e-10,
        max_restarts: 1,
        verbose: false,
        check_true_residual: false,
    };

    let result = gmres(apply_a, &b, &x0, &options).unwrap();

    assert!(result.converged, "GMRES should converge");
    assert!(
        result.residual_norm < 1e-10,
        "Residual should be small: {}",
        result.residual_norm
    );

    // Check solution
    let diff = result
        .solution
        .axpby(
            AnyScalar::new_real(1.0),
            &expected_x,
            AnyScalar::new_real(-1.0),
        )
        .unwrap();
    assert!(
        diff.norm().unwrap() < 1e-8,
        "Solution error too large: {}",
        diff.norm().unwrap()
    );
}

#[test]
fn test_gmres_complex_nonsymmetric_matrix() {
    let idx = DynIndex::new_dyn(2);
    let expected_x = make_vector_c64(
        vec![Complex64::new(1.0, -0.5), Complex64::new(-2.0, 0.75)],
        &idx,
    );
    let x0 = make_vector_c64(vec![Complex64::new(0.0, 0.0); 2], &idx);
    let a_data = [
        Complex64::new(2.0, 0.5),
        Complex64::new(-1.0, 0.25),
        Complex64::new(0.75, -0.5),
        Complex64::new(1.5, 1.0),
    ];
    let b = apply_matrix2_c64(&expected_x, &a_data).unwrap();
    let apply_a = move |x: &TensorDynLen| apply_matrix2_c64(x, &a_data);
    let options = GmresOptions {
        max_iter: 2,
        rtol: 1e-12,
        max_restarts: 1,
        verbose: false,
        check_true_residual: true,
    };

    let result = gmres(apply_a, &b, &x0, &options).unwrap();
    assert!(result.converged, "residual={}", result.residual_norm);
    let diff = result
        .solution
        .axpby(
            AnyScalar::new_real(1.0),
            &expected_x,
            AnyScalar::new_real(-1.0),
        )
        .unwrap();
    assert!(
        diff.norm().unwrap() < 1e-10,
        "solution error={}",
        diff.norm().unwrap()
    );
}

#[test]
fn test_gmres_with_total_iteration_limit_shortens_final_restart() {
    let idx = DynIndex::new_dyn(6);
    let b = make_vector_with_index(vec![1.0; 6], &idx);
    let x0 = make_vector_with_index(vec![0.0; 6], &idx);

    let diag = [1.0, 1.7, 2.3, 3.1, 4.2, 5.6];
    let apply_a = move |x: &TensorDynLen| scale_vector_f64(x, &diag);
    let options = GmresOptions {
        max_iter: 3,
        rtol: 0.0,
        max_restarts: 3,
        verbose: false,
        check_true_residual: false,
    };

    let result = gmres_with_total_iteration_limit(apply_a, &b, &x0, &options, 4).unwrap();

    assert_eq!(result.iterations, 4);
    assert!(!result.converged);
}

#[test]
fn test_gmres_with_total_iteration_limit_allows_zero_iterations() {
    let idx = DynIndex::new_dyn(3);
    let b = make_vector_with_index(vec![1.0, 2.0, 3.0], &idx);
    let x0 = make_vector_with_index(vec![0.0, 0.0, 0.0], &idx);

    let apply_a = |x: &TensorDynLen| -> Result<TensorDynLen> { Ok(x.clone()) };
    let options = GmresOptions {
        max_iter: 3,
        rtol: 1e-10,
        max_restarts: 2,
        verbose: false,
        check_true_residual: false,
    };

    let result = gmres_with_total_iteration_limit(apply_a, &b, &x0, &options, 0).unwrap();

    assert_eq!(result.iterations, 0);
    assert!(!result.converged);
    assert!(result.solution.sub(&x0).unwrap().norm().unwrap() < 1.0e-12);
}

#[test]
fn test_gmres_nonsymmetric_matrix() {
    // Solve A x = b where A is a 2x2 non-symmetric matrix
    // A = [[2, 1], [0, 3]]
    // A x = [2*1 + 1*2, 0*1 + 3*2] = [4, 6]
    // So b = [4, 6] → x = [1, 2]
    let idx = DynIndex::new_dyn(2);
    let b = make_vector_with_index(vec![4.0, 6.0], &idx);
    let x0 = make_vector_with_index(vec![0.0, 0.0], &idx);
    let expected_x = make_vector_with_index(vec![1.0, 2.0], &idx);

    // Matrix A = [[2, 1], [0, 3]]
    let a_data = [2.0, 1.0, 0.0, 3.0];

    let apply_a = move |x: &TensorDynLen| apply_matrix2_f64(x, &a_data);

    let options = GmresOptions {
        max_iter: 10,
        rtol: 1e-10,
        max_restarts: 1,
        verbose: false,
        check_true_residual: false,
    };

    let result = gmres(apply_a, &b, &x0, &options).unwrap();

    assert!(result.converged, "GMRES should converge");

    // Check solution
    let diff = result
        .solution
        .axpby(
            AnyScalar::new_real(1.0),
            &expected_x,
            AnyScalar::new_real(-1.0),
        )
        .unwrap();
    assert!(
        diff.norm().unwrap() < 1e-8,
        "Solution error too large: {}",
        diff.norm().unwrap()
    );
}

#[test]
fn test_gmres_with_good_initial_guess() {
    // If initial guess is already the solution, should converge immediately
    let idx = DynIndex::new_dyn(3);
    let b = make_vector_with_index(vec![2.0, 4.0, 6.0], &idx);
    let x0 = make_vector_with_index(vec![1.0, 2.0, 3.0], &idx); // Already the solution for A=diag(2,2,2)

    let apply_a = |x: &TensorDynLen| -> Result<TensorDynLen> {
        // A = 2*I
        x.scale(AnyScalar::new_real(2.0))
    };

    let options = GmresOptions {
        max_iter: 10,
        rtol: 1e-10,
        max_restarts: 1,
        verbose: false,
        check_true_residual: false,
    };

    let result = gmres(apply_a, &b, &x0, &options).unwrap();

    assert!(result.converged);
    assert_eq!(result.iterations, 0, "Should converge with 0 iterations");
}

/// Helper to create a 1D complex tensor (vector) with given data and shared index.
fn make_vector_c64_with_index(data: Vec<num_complex::Complex64>, idx: &DynIndex) -> TensorDynLen {
    TensorDynLen::from_dense(vec![idx.clone()], data).unwrap()
}

#[test]
fn test_gmres_identity_operator_c64() {
    // Solve A x = b where A = I (identity), b is complex
    // Solution: x = b
    use num_complex::Complex64;

    let idx = DynIndex::new_dyn(4);
    let b = make_vector_c64_with_index(
        vec![
            Complex64::new(1.0, 2.0),
            Complex64::new(-3.0, 0.5),
            Complex64::new(0.0, -1.0),
            Complex64::new(2.5, 3.5),
        ],
        &idx,
    );
    let x0 = make_vector_c64_with_index(vec![Complex64::new(0.0, 0.0); 4], &idx);

    // Identity operator: A x = x
    let apply_a = |x: &TensorDynLen| -> Result<TensorDynLen> { Ok(x.clone()) };

    let options = GmresOptions {
        max_iter: 20,
        rtol: 1e-10,
        max_restarts: 1,
        verbose: false,
        check_true_residual: false,
    };

    let result = gmres(apply_a, &b, &x0, &options).unwrap();

    // Check solution matches b
    let diff = result
        .solution
        .axpby(AnyScalar::new_real(1.0), &b, AnyScalar::new_real(-1.0))
        .unwrap();
    let err = diff.norm().unwrap();

    assert!(result.converged, "GMRES should converge for identity (c64)");
    assert!(
        result.residual_norm < 1e-10,
        "Residual should be small: {}",
        result.residual_norm
    );
    assert!(err < 1e-8, "Solution should equal b, error: {}", err);
}

#[test]
fn test_gmres_diagonal_c64() {
    // Solve A x = b where A = diag(2+i, 3-i, 1+2i, 4)
    use num_complex::Complex64;

    let idx = DynIndex::new_dyn(4);
    let diag = [
        Complex64::new(2.0, 1.0),
        Complex64::new(3.0, -1.0),
        Complex64::new(1.0, 2.0),
        Complex64::new(4.0, 0.0),
    ];
    let x_true = [
        Complex64::new(1.0, -1.0),
        Complex64::new(0.5, 2.0),
        Complex64::new(-1.0, 0.0),
        Complex64::new(0.0, 1.0),
    ];
    let b_data: Vec<Complex64> = diag.iter().zip(x_true.iter()).map(|(d, x)| d * x).collect();

    let b = make_vector_c64_with_index(b_data, &idx);
    let x0 = make_vector_c64_with_index(vec![Complex64::new(0.0, 0.0); 4], &idx);
    let expected = make_vector_c64_with_index(x_true.to_vec(), &idx);

    let apply_a = move |x: &TensorDynLen| -> Result<TensorDynLen> {
        let x_data = x.to_vec::<Complex64>()?;
        let result_data: Vec<Complex64> = x_data
            .iter()
            .zip(diag.iter())
            .map(|(&xi, &di)| di * xi)
            .collect();
        Ok(TensorDynLen::from_dense(x.indices.clone(), result_data).unwrap())
    };

    let options = GmresOptions {
        max_iter: 20,
        rtol: 1e-10,
        max_restarts: 1,
        verbose: false,
        check_true_residual: false,
    };

    let result = gmres(apply_a, &b, &x0, &options).unwrap();

    let diff = result
        .solution
        .axpby(
            AnyScalar::new_real(1.0),
            &expected,
            AnyScalar::new_real(-1.0),
        )
        .unwrap();
    let err = diff.norm().unwrap();

    assert!(result.converged, "GMRES should converge for diagonal (c64)");
    assert!(err < 1e-8, "Solution error too large: {}", err);
}

#[test]
fn test_gmres_zero_rhs() {
    // Solve A x = 0 → x = 0 (for any invertible A)
    let idx = DynIndex::new_dyn(3);
    let b = make_vector_with_index(vec![0.0, 0.0, 0.0], &idx);
    let x0 = make_vector_with_index(vec![1.0, 2.0, 3.0], &idx);

    let apply_a = |x: &TensorDynLen| -> Result<TensorDynLen> { Ok(x.clone()) };

    let options = GmresOptions::default();

    let result = gmres(apply_a, &b, &x0, &options).unwrap();

    // With zero RHS, should return initial guess immediately
    assert!(result.converged);
}

// ==========================================================================
// Tests for restart_gmres_with_truncation
// ==========================================================================

#[test]
fn test_restart_gmres_identity_operator() {
    // Solve A x = b where A = I (identity)
    let idx = DynIndex::new_dyn(3);
    let b = make_vector_with_index(vec![1.0, 2.0, 3.0], &idx);

    let apply_a = |x: &TensorDynLen| -> Result<TensorDynLen> { Ok(x.clone()) };

    // No-op truncation
    let truncate = |_x: &mut TensorDynLen| -> Result<()> { Ok(()) };

    let options = RestartGmresOptions::default();

    let result = restart_gmres_with_truncation(apply_a, &b, None, &options, truncate).unwrap();

    assert!(
        result.converged,
        "Restart GMRES should converge for identity"
    );
    assert!(
        result.residual_norm < 1e-10,
        "Residual should be small: {}",
        result.residual_norm
    );

    // Check solution matches b
    let diff = result
        .solution
        .axpby(AnyScalar::new_real(1.0), &b, AnyScalar::new_real(-1.0))
        .unwrap();
    assert!(diff.norm().unwrap() < 1e-10, "Solution should equal b");
}

#[test]
fn test_restart_gmres_diagonal_matrix() {
    // Solve A x = b where A = diag(2, 3, 4)
    // b = [2, 6, 12] → x = [1, 2, 3]
    let idx = DynIndex::new_dyn(3);
    let b = make_vector_with_index(vec![2.0, 6.0, 12.0], &idx);
    let expected_x = make_vector_with_index(vec![1.0, 2.0, 3.0], &idx);

    let diag = [2.0, 3.0, 4.0];
    let apply_a = move |x: &TensorDynLen| scale_vector_f64(x, &diag);

    // No-op truncation
    let truncate = |_x: &mut TensorDynLen| -> Result<()> { Ok(()) };

    let options = RestartGmresOptions {
        max_outer_iters: 10,
        rtol: 1e-10,
        inner_max_iter: 5,
        inner_max_restarts: 0,
        min_reduction: None,
        inner_rtol: None,
        verbose: false,
    };

    let result = restart_gmres_with_truncation(apply_a, &b, None, &options, truncate).unwrap();

    assert!(result.converged, "Restart GMRES should converge");
    assert!(
        result.residual_norm < 1e-10,
        "Residual should be small: {}",
        result.residual_norm
    );

    // Check solution
    let diff = result
        .solution
        .axpby(
            AnyScalar::new_real(1.0),
            &expected_x,
            AnyScalar::new_real(-1.0),
        )
        .unwrap();
    assert!(
        diff.norm().unwrap() < 1e-8,
        "Solution error too large: {}",
        diff.norm().unwrap()
    );
}

#[test]
fn test_restart_gmres_with_initial_guess() {
    // Solve A x = b with a good initial guess
    let idx = DynIndex::new_dyn(3);
    let b = make_vector_with_index(vec![2.0, 6.0, 12.0], &idx);
    let x0 = make_vector_with_index(vec![0.9, 1.9, 2.9], &idx); // Close to [1, 2, 3]
    let expected_x = make_vector_with_index(vec![1.0, 2.0, 3.0], &idx);

    let diag = [2.0, 3.0, 4.0];
    let apply_a = move |x: &TensorDynLen| scale_vector_f64(x, &diag);

    let truncate = |_x: &mut TensorDynLen| -> Result<()> { Ok(()) };

    let options = RestartGmresOptions::default();

    let result = restart_gmres_with_truncation(apply_a, &b, Some(&x0), &options, truncate).unwrap();

    assert!(result.converged, "Should converge with good initial guess");

    let diff = result
        .solution
        .axpby(
            AnyScalar::new_real(1.0),
            &expected_x,
            AnyScalar::new_real(-1.0),
        )
        .unwrap();
    assert!(
        diff.norm().unwrap() < 1e-8,
        "Solution error too large: {}",
        diff.norm().unwrap()
    );
}

#[test]
fn test_restart_gmres_outer_iterations_tracked() {
    // Verify that outer_iterations is tracked correctly
    let idx = DynIndex::new_dyn(3);
    let b = make_vector_with_index(vec![2.0, 6.0, 12.0], &idx);

    let diag = [2.0, 3.0, 4.0];
    let apply_a = move |x: &TensorDynLen| scale_vector_f64(x, &diag);

    let truncate = |_x: &mut TensorDynLen| -> Result<()> { Ok(()) };

    // Use small inner_max_iter to encourage multiple outer iterations
    let options = RestartGmresOptions {
        max_outer_iters: 20,
        rtol: 1e-10,
        inner_max_iter: 2,
        inner_max_restarts: 0,
        min_reduction: None,
        inner_rtol: Some(0.1),
        verbose: false,
    };

    let result = restart_gmres_with_truncation(apply_a, &b, None, &options, truncate).unwrap();

    assert!(result.converged, "Should converge");
    // Verify iteration counts are reasonable
    assert!(
        result.iterations >= 1,
        "Should have at least 1 inner iteration"
    );
    // outer_iterations can be 0 if converged at first check
    assert!(
        result.outer_iterations <= options.max_outer_iters,
        "outer_iterations should not exceed max_outer_iters"
    );
}

#[test]
fn test_restart_gmres_zero_rhs() {
    // Solve A x = 0 → x = 0
    let idx = DynIndex::new_dyn(3);
    let b = make_vector_with_index(vec![0.0, 0.0, 0.0], &idx);

    let apply_a = |x: &TensorDynLen| -> Result<TensorDynLen> { Ok(x.clone()) };
    let truncate = |_x: &mut TensorDynLen| -> Result<()> { Ok(()) };

    let options = RestartGmresOptions::default();

    let result = restart_gmres_with_truncation(apply_a, &b, None, &options, truncate).unwrap();

    assert!(result.converged, "Should converge for zero RHS");
    assert_eq!(result.iterations, 0, "Should converge immediately");
    assert_eq!(result.outer_iterations, 0, "Should have 0 outer iterations");
}

#[test]
fn test_restart_gmres_options_builder() {
    let options = RestartGmresOptions::new()
        .with_max_outer_iters(30)
        .with_rtol(1e-12)
        .with_inner_max_iter(15)
        .with_inner_max_restarts(1)
        .with_min_reduction(0.95)
        .with_inner_rtol(0.01)
        .with_verbose(true);

    assert_eq!(options.max_outer_iters, 30);
    assert!((options.rtol - 1e-12).abs() < 1e-15);
    assert_eq!(options.inner_max_iter, 15);
    assert_eq!(options.inner_max_restarts, 1);
    assert_eq!(options.min_reduction, Some(0.95));
    assert_eq!(options.inner_rtol, Some(0.01));
    assert!(options.verbose);
}

#[test]
fn test_gmres_with_truncation_check_true_residual_safe() {
    // When check_true_residual is enabled and convergence is reported,
    // the residual_norm should reflect the checked residual (not the
    // potentially inaccurate Hessenberg estimate).
    let idx = DynIndex::new_dyn(3);
    let b = make_vector_with_index(vec![2.0, 6.0, 12.0], &idx);
    let x0 = make_vector_with_index(vec![0.0, 0.0, 0.0], &idx);

    let diag = [2.0, 3.0, 4.0];
    let apply_a = move |x: &TensorDynLen| scale_vector_f64(x, &diag);

    // No-op truncation: convergence should work normally with check enabled
    let truncate = |_x: &mut TensorDynLen| -> Result<()> { Ok(()) };

    let options = GmresOptions {
        max_iter: 10,
        rtol: 1e-10,
        max_restarts: 3,
        verbose: false,
        check_true_residual: true,
    };

    let result = gmres_with_truncation(apply_a, &b, &x0, &options, truncate).unwrap();

    assert!(
        result.converged,
        "Should converge with true residual check and no-op truncation"
    );

    // Verify the reported residual is actually the checked residual
    let ax = apply_a(&result.solution).unwrap();
    let r = ax
        .axpby(AnyScalar::new_real(1.0), &b, AnyScalar::new_real(-1.0))
        .unwrap();
    let true_rel_res = r.norm().unwrap() / b.norm().unwrap();
    assert!(
        true_rel_res < 1e-8,
        "True residual should be small: {}",
        true_rel_res
    );
}

#[test]
#[allow(clippy::needless_borrows_for_generic_args)]
fn test_gmres_with_truncation_check_true_residual_consistency() {
    // Test that when check_true_residual is enabled, the reported residual_norm
    // is consistent with the actual checked (truncated) residual.
    let idx = DynIndex::new_dyn(4);
    let b = make_vector_with_index(vec![1.0, 2.0, 3.0, 4.0], &idx);
    let x0 = make_vector_with_index(vec![0.0, 0.0, 0.0, 0.0], &idx);

    // A = diag(1, 2, 3, 4)
    let diag = [1.0, 2.0, 3.0, 4.0];
    let apply_a = move |x: &TensorDynLen| scale_vector_f64(x, &diag);

    // No-op truncation: the solver should converge normally
    let truncate = |_x: &mut TensorDynLen| -> Result<()> { Ok(()) };

    // Without check: converged with Hessenberg residual
    let options_no_check = GmresOptions {
        max_iter: 10,
        rtol: 1e-10,
        max_restarts: 3,
        verbose: false,
        check_true_residual: false,
    };
    let result_no_check =
        gmres_with_truncation(&apply_a, &b, &x0, &options_no_check, &truncate).unwrap();

    // With check: converged with verified residual
    let options_check = GmresOptions {
        max_iter: 10,
        rtol: 1e-10,
        max_restarts: 3,
        verbose: false,
        check_true_residual: true,
    };
    let result_check = gmres_with_truncation(&apply_a, &b, &x0, &options_check, &truncate).unwrap();

    // Both should converge for this simple problem
    assert!(result_no_check.converged, "No-check should converge");
    assert!(result_check.converged, "With-check should converge");

    // With check, the reported residual should be the true residual
    let ax = apply_a(&result_check.solution).unwrap();
    let r = ax
        .axpby(AnyScalar::new_real(1.0), &b, AnyScalar::new_real(-1.0))
        .unwrap();
    let true_rel_res = r.norm().unwrap() / b.norm().unwrap();

    // The reported residual should match the true residual closely
    assert!(
        (result_check.residual_norm - true_rel_res).abs() < 1e-8,
        "Reported residual ({:.6e}) should match true residual ({:.6e})",
        result_check.residual_norm,
        true_rel_res
    );
}

// ==========================================================================
// Tests for uncovered code paths
// ==========================================================================

#[test]
fn test_gmres_verbose_output() {
    // Exercise verbose code paths in gmres (lines 151, 158-161, 242)
    let idx = DynIndex::new_dyn(2);
    let b = make_vector_with_index(vec![4.0, 6.0], &idx);
    let x0 = make_vector_with_index(vec![0.0, 0.0], &idx);

    let a_data = [2.0, 1.0, 0.0, 3.0];
    let apply_a = move |x: &TensorDynLen| apply_matrix2_f64(x, &a_data);

    let options = GmresOptions {
        max_iter: 10,
        rtol: 1e-10,
        max_restarts: 2,
        verbose: true,
        check_true_residual: false,
    };

    let result = gmres(apply_a, &b, &x0, &options).unwrap();
    assert!(result.converged);
}

#[test]
fn test_gmres_no_convergence_exhausts_restarts() {
    // Force gmres to exhaust all restarts without converging.
    // Use max_iter=1 and max_restarts=1 with a harder problem so it can't converge.
    let idx = DynIndex::new_dyn(3);
    let b = make_vector_with_index(vec![1.0, 2.0, 3.0], &idx);
    let x0 = make_vector_with_index(vec![0.0, 0.0, 0.0], &idx);

    let diag = [2.0, 3.0, 4.0];
    let apply_a = move |x: &TensorDynLen| scale_vector_f64(x, &diag);

    // Very few iterations and restarts, tight tolerance
    let options = GmresOptions {
        max_iter: 1,
        rtol: 1e-15,
        max_restarts: 1,
        verbose: false,
        check_true_residual: false,
    };

    let result = gmres(apply_a, &b, &x0, &options).unwrap();
    // With max_iter=1 and max_restarts=1, it may or may not converge,
    // but the end-of-restart and final-residual code paths should be hit.
    // The important thing is it runs without error.
    assert!(result.residual_norm >= 0.0);
}

#[test]
fn test_gmres_lucky_breakdown() {
    // For a 1D system, the Krylov subspace is exact after 1 iteration,
    // causing h_{j+1,j} = 0 (lucky breakdown).
    // A = [5], b = [10], x0 = [0] -> solution x = [2]
    let idx = DynIndex::new_dyn(1);
    let b = make_vector_with_index(vec![10.0], &idx);
    let x0 = make_vector_with_index(vec![0.0], &idx);

    let apply_a = |x: &TensorDynLen| -> Result<TensorDynLen> { scale_vector_f64(x, &[5.0]) };

    let options = GmresOptions {
        max_iter: 10,
        rtol: 1e-10,
        max_restarts: 1,
        verbose: false,
        check_true_residual: false,
    };

    let result = gmres(apply_a, &b, &x0, &options).unwrap();
    assert!(result.converged, "Should converge via lucky breakdown");

    let expected = make_vector_with_index(vec![2.0], &idx);
    let diff = result
        .solution
        .axpby(
            AnyScalar::new_real(1.0),
            &expected,
            AnyScalar::new_real(-1.0),
        )
        .unwrap();
    assert!(diff.norm().unwrap() < 1e-10, "Solution should be [2.0]");
}

#[test]
fn test_gmres_with_truncation_zero_rhs() {
    // Exercise the zero-b early return in gmres_with_truncation (lines 345-350)
    let idx = DynIndex::new_dyn(3);
    let b = make_vector_with_index(vec![0.0, 0.0, 0.0], &idx);
    let x0 = make_vector_with_index(vec![1.0, 2.0, 3.0], &idx);

    let apply_a = |x: &TensorDynLen| -> Result<TensorDynLen> { Ok(x.clone()) };
    let truncate = |_x: &mut TensorDynLen| -> Result<()> { Ok(()) };

    let options = GmresOptions {
        max_iter: 10,
        rtol: 1e-10,
        max_restarts: 1,
        verbose: false,
        check_true_residual: false,
    };

    let result = gmres_with_truncation(apply_a, &b, &x0, &options, truncate).unwrap();
    assert!(result.converged);
    assert_eq!(result.iterations, 0);
}

#[test]
fn test_gmres_with_truncation_already_converged() {
    // Initial guess already satisfies the equation, triggering early return (lines 375-380)
    let idx = DynIndex::new_dyn(3);
    let b = make_vector_with_index(vec![2.0, 6.0, 12.0], &idx);
    let x0 = make_vector_with_index(vec![1.0, 2.0, 3.0], &idx); // Exact solution for diag(2,3,4)

    let diag = [2.0, 3.0, 4.0];
    let apply_a = move |x: &TensorDynLen| scale_vector_f64(x, &diag);
    let truncate = |_x: &mut TensorDynLen| -> Result<()> { Ok(()) };

    let options = GmresOptions {
        max_iter: 10,
        rtol: 1e-10,
        max_restarts: 1,
        verbose: false,
        check_true_residual: false,
    };

    let result = gmres_with_truncation(apply_a, &b, &x0, &options, truncate).unwrap();
    assert!(result.converged);
    assert_eq!(result.iterations, 0);
}

#[test]
fn test_gmres_with_truncation_verbose() {
    // Exercise verbose output paths in gmres_with_truncation
    // (lines 361, 368-371, 446-457, 463, 467, 503, 517, 522-525)
    let idx = DynIndex::new_dyn(3);
    let b = make_vector_with_index(vec![2.0, 6.0, 12.0], &idx);
    let x0 = make_vector_with_index(vec![0.0, 0.0, 0.0], &idx);

    let diag = [2.0, 3.0, 4.0];
    let apply_a = move |x: &TensorDynLen| scale_vector_f64(x, &diag);
    let truncate = |_x: &mut TensorDynLen| -> Result<()> { Ok(()) };

    let options = GmresOptions {
        max_iter: 10,
        rtol: 1e-10,
        max_restarts: 2,
        verbose: true,
        check_true_residual: true,
    };

    let result = gmres_with_truncation(apply_a, &b, &x0, &options, truncate).unwrap();
    assert!(result.converged);
}

#[test]
fn test_restart_gmres_stagnation_detection() {
    // Test stagnation detection in restart_gmres_with_truncation.
    // Use a system where the exact solution has irrational-like components
    // that can't be represented after rounding, causing stagnation.
    let idx = DynIndex::new_dyn(3);
    // A = diag(3, 7, 11), b = [1, 1, 1] -> x = [1/3, 1/7, 1/11]
    // These solutions can't be exactly represented after rounding to 1 decimal place.
    let b = make_vector_with_index(vec![1.0, 1.0, 1.0], &idx);

    let diag = [3.0, 7.0, 11.0];
    let apply_a = move |x: &TensorDynLen| scale_vector_f64(x, &diag);

    // Truncation that rounds to 1 decimal place - the exact solution [1/3, 1/7, 1/11]
    // gets rounded to [0.3, 0.1, 0.1], preventing convergence below the rounding error.
    let truncate = |x: &mut TensorDynLen| -> Result<()> {
        let data = x.to_vec::<f64>()?;
        let new_data: Vec<f64> = data.iter().map(|&v| (v * 10.0).round() / 10.0).collect();
        *x = TensorDynLen::from_dense(x.indices.clone(), new_data).unwrap();
        Ok(())
    };

    let options = RestartGmresOptions {
        max_outer_iters: 30,
        rtol: 1e-12,
        inner_max_iter: 5,
        inner_max_restarts: 0,
        min_reduction: Some(0.999),
        inner_rtol: Some(0.1),
        verbose: false,
    };

    let result = restart_gmres_with_truncation(apply_a, &b, None, &options, truncate).unwrap();
    // Stagnation should be detected and converged should be false
    assert!(!result.converged, "Should detect stagnation");
}

#[test]
fn test_restart_gmres_exhausts_outer_iters() {
    // Test the path where restart_gmres exhausts all outer iterations without converging.
    let idx = DynIndex::new_dyn(3);
    let b = make_vector_with_index(vec![2.0, 6.0, 12.0], &idx);

    let diag = [2.0, 3.0, 4.0];
    let apply_a = move |x: &TensorDynLen| scale_vector_f64(x, &diag);
    let truncate = |_x: &mut TensorDynLen| -> Result<()> { Ok(()) };

    // Very few iterations and extremely tight tolerance to prevent convergence
    let options = RestartGmresOptions {
        max_outer_iters: 1,
        rtol: 1e-100, // impossibly tight
        inner_max_iter: 1,
        inner_max_restarts: 0,
        min_reduction: None,
        inner_rtol: Some(0.5),
        verbose: false,
    };

    let result = restart_gmres_with_truncation(apply_a, &b, None, &options, truncate).unwrap();
    assert!(!result.converged);
    assert_eq!(result.outer_iterations, options.max_outer_iters);
}

#[test]
fn test_restart_gmres_verbose() {
    // Exercise verbose output paths in restart_gmres_with_truncation
    let idx = DynIndex::new_dyn(3);
    let b = make_vector_with_index(vec![2.0, 6.0, 12.0], &idx);

    let diag = [2.0, 3.0, 4.0];
    let apply_a = move |x: &TensorDynLen| scale_vector_f64(x, &diag);
    let truncate = |_x: &mut TensorDynLen| -> Result<()> { Ok(()) };

    let options = RestartGmresOptions {
        max_outer_iters: 10,
        rtol: 1e-10,
        inner_max_iter: 5,
        inner_max_restarts: 0,
        min_reduction: None,
        inner_rtol: Some(0.1),
        verbose: true,
    };

    let result = restart_gmres_with_truncation(apply_a, &b, None, &options, truncate).unwrap();
    assert!(result.converged);
}

#[test]
fn test_restart_gmres_zero_rhs_with_x0() {
    // Cover the zero-b-with-x0 path in restart_gmres_with_truncation (line 779)
    let idx = DynIndex::new_dyn(3);
    let b = make_vector_with_index(vec![0.0, 0.0, 0.0], &idx);
    let x0 = make_vector_with_index(vec![1.0, 2.0, 3.0], &idx);

    let diag = [2.0, 3.0, 4.0];
    let apply_a = move |x: &TensorDynLen| scale_vector_f64(x, &diag);
    let truncate = |_x: &mut TensorDynLen| -> Result<()> { Ok(()) };

    let options = RestartGmresOptions {
        max_outer_iters: 5,
        rtol: 1e-10,
        inner_max_iter: 3,
        inner_max_restarts: 0,
        min_reduction: None,
        inner_rtol: None,
        verbose: false,
    };

    let result = restart_gmres_with_truncation(apply_a, &b, Some(&x0), &options, truncate).unwrap();
    assert!(result.converged);
    assert_eq!(result.iterations, 0);
    // Solution should be x0 when b is zero
    assert!(result.solution.distance(&x0).unwrap() < 1e-12);
}

#[test]
fn test_restart_gmres_stagnation_verbose() {
    // Cover the stagnation verbose output path (lines 847-851)
    let n = 10;
    let idx = DynIndex::new_dyn(n);
    let b_data: Vec<f64> = (1..=n).map(|i| i as f64).collect();
    let b = make_vector_with_index(b_data, &idx);

    let diag: Vec<f64> = (0..n)
        .map(|i| 10.0_f64.powf(i as f64 * 3.0 / (n as f64 - 1.0)))
        .collect();
    let apply_a = move |x: &TensorDynLen| scale_vector_f64(x, &diag);

    let truncate = |x: &mut TensorDynLen| -> Result<()> {
        let data = x.to_vec::<f64>()?;
        let new_data: Vec<f64> = data.iter().map(|&v| (v * 100.0).round() / 100.0).collect();
        *x = TensorDynLen::from_dense(x.indices.clone(), new_data).unwrap();
        Ok(())
    };

    let options = RestartGmresOptions {
        max_outer_iters: 10,
        rtol: 1e-12,
        inner_max_iter: 3,
        inner_max_restarts: 0,
        min_reduction: Some(0.999),
        inner_rtol: Some(0.1),
        verbose: true, // Cover verbose stagnation output
    };

    let result = restart_gmres_with_truncation(apply_a, &b, None, &options, truncate).unwrap();
    assert!(!result.converged);
}
