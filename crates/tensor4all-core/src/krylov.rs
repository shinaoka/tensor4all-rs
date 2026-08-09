//! Krylov subspace methods for solving linear equations with abstract tensors.
//!
//! This module provides iterative solvers that work with any type implementing [`TensorVectorSpace`],
//! enabling their use in tensor network algorithms without requiring dense vector representations.
//!
//! # Solvers
//!
//! - [`gmres`]: Generalized Minimal Residual Method (GMRES) for non-symmetric systems
//! - [`hermitian_lanczos_lowest_eigenpair`]: Matrix-free lowest eigenpair solver for Hermitian operators
//!
//! # Future Extensions
//!
//! - CG (Conjugate Gradient) for symmetric positive definite systems
//! - BiCGSTAB for non-symmetric systems with better convergence properties
//!
//! # Example
//!
//! ```
//! use tensor4all_core::{
//!     krylov::{gmres, GmresOptions},
//!     DynIndex, TensorDynLen, TensorVectorSpace,
//! };
//!
//! # fn main() -> anyhow::Result<()> {
//! let i = DynIndex::new_dyn(2);
//! let rhs = TensorDynLen::from_dense(vec![i.clone()], vec![1.0, -1.0])?;
//! let initial_guess = TensorDynLen::from_dense(vec![i.clone()], vec![0.0, 0.0])?;
//!
//! let apply_operator = |x: &TensorDynLen| Ok(x.clone());
//! let result = gmres(apply_operator, &rhs, &initial_guess, &GmresOptions::default())?;
//!
//! assert!(result.converged);
//! assert!(result.solution.sub(&rhs)?.maxabs()? < 1e-12);
//! # Ok(())
//! # }
//! ```

use crate::any_scalar::AnyScalar;
use crate::TensorVectorSpace;
use anyhow::Result;
use num_complex::Complex64;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};
use tensor4all_tensorbackend::{
    hermitian_exponential_first_column, lowest_hermitian_eigenpair, Matrix,
};

static GMRES_OP_PROFILE_COUNTER: AtomicUsize = AtomicUsize::new(0);

#[derive(Debug, Clone)]
struct GmresOpProfile {
    started: Instant,
    b_norm: Duration,
    apply: Duration,
    inner_product: Duration,
    axpby: Duration,
    norm: Duration,
    scale: Duration,
    triangular_solve: Duration,
    solution_update: Duration,
    apply_calls: usize,
    inner_product_calls: usize,
    axpby_calls: usize,
    norm_calls: usize,
    scale_calls: usize,
    triangular_solve_calls: usize,
    solution_update_calls: usize,
}

impl Default for GmresOpProfile {
    fn default() -> Self {
        Self {
            started: Instant::now(),
            b_norm: Duration::ZERO,
            apply: Duration::ZERO,
            inner_product: Duration::ZERO,
            axpby: Duration::ZERO,
            norm: Duration::ZERO,
            scale: Duration::ZERO,
            triangular_solve: Duration::ZERO,
            solution_update: Duration::ZERO,
            apply_calls: 0,
            inner_product_calls: 0,
            axpby_calls: 0,
            norm_calls: 0,
            scale_calls: 0,
            triangular_solve_calls: 0,
            solution_update_calls: 0,
        }
    }
}

impl GmresOpProfile {
    fn measured(&self) -> Duration {
        self.b_norm
            + self.apply
            + self.inner_product
            + self.axpby
            + self.norm
            + self.scale
            + self.triangular_solve
            + self.solution_update
    }

    fn print(&self, id: usize, iterations: usize, residual_norm: f64, converged: bool) {
        let total = self.started.elapsed();
        let other = total.saturating_sub(self.measured());
        eprintln!(
            "T4A gmres_op_profile #{id}: iterations={iterations} residual={residual_norm:.6e} converged={converged} total_ms={:.3} apply_ms={:.3} apply_calls={} inner_ms={:.3} inner_calls={} axpby_ms={:.3} axpby_calls={} norm_ms={:.3} norm_calls={} scale_ms={:.3} scale_calls={} update_ms={:.3} update_calls={} triangular_ms={:.3} triangular_calls={} b_norm_ms={:.3} other_ms={:.3}",
            total.as_secs_f64() * 1000.0,
            self.apply.as_secs_f64() * 1000.0,
            self.apply_calls,
            self.inner_product.as_secs_f64() * 1000.0,
            self.inner_product_calls,
            self.axpby.as_secs_f64() * 1000.0,
            self.axpby_calls,
            self.norm.as_secs_f64() * 1000.0,
            self.norm_calls,
            self.scale.as_secs_f64() * 1000.0,
            self.scale_calls,
            self.solution_update.as_secs_f64() * 1000.0,
            self.solution_update_calls,
            self.triangular_solve.as_secs_f64() * 1000.0,
            self.triangular_solve_calls,
            self.b_norm.as_secs_f64() * 1000.0,
            other.as_secs_f64() * 1000.0,
        );
    }
}

/// Options for GMRES solver.
///
/// # Examples
///
/// ```
/// use tensor4all_core::krylov::GmresOptions;
///
/// let opts = GmresOptions {
///     max_iter: 50,
///     rtol: 1e-8,
///     max_restarts: 5,
///     verbose: false,
///     check_true_residual: true,
/// };
/// assert_eq!(opts.max_iter, 50);
/// assert_eq!(opts.rtol, 1e-8);
/// ```
#[derive(Debug, Clone)]
pub struct GmresOptions {
    /// Maximum number of iterations (restart cycle length).
    /// Default: 100
    pub max_iter: usize,

    /// Convergence tolerance for relative residual norm.
    /// The solver stops when `||r|| / ||b|| < rtol`.
    /// Default: 1e-10
    pub rtol: f64,

    /// Maximum number of restarts.
    /// Total iterations = max_iter * max_restarts.
    /// Default: 10
    pub max_restarts: usize,

    /// Whether to print convergence information.
    /// Default: false
    pub verbose: bool,

    /// When true, verify convergence by computing the true residual `||b - A*x|| / ||b||`
    /// before declaring convergence. This prevents false convergence caused by
    /// truncation corrupting the Krylov basis orthogonality (see Issue #207).
    /// Costs one additional `apply_a` call when convergence is detected.
    /// Default: false
    pub check_true_residual: bool,
}

impl Default for GmresOptions {
    fn default() -> Self {
        Self {
            max_iter: 100,
            rtol: 1e-10,
            max_restarts: 10,
            verbose: false,
            check_true_residual: false,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum GmresTolerance {
    Relative(f64),
    Absolute(f64),
}

impl GmresTolerance {
    fn residual_value(self, residual_norm: f64, b_norm: f64) -> f64 {
        match self {
            Self::Relative(_) => residual_norm / b_norm,
            Self::Absolute(_) => residual_norm,
        }
    }

    fn is_converged(self, residual_norm: f64, b_norm: f64) -> bool {
        match self {
            Self::Relative(rtol) => residual_norm / b_norm < rtol,
            Self::Absolute(atol) => residual_norm < atol,
        }
    }
}

/// Result of GMRES solver.
///
/// Contains the solution, iteration count, final residual norm, and
/// convergence status.
///
/// # Examples
///
/// ```
/// use tensor4all_core::{DynIndex, TensorDynLen, TensorVectorSpace};
/// use tensor4all_core::krylov::{gmres, GmresOptions};
///
/// let i = DynIndex::new_dyn(2);
/// let b = TensorDynLen::from_dense(vec![i.clone()], vec![3.0, 7.0]).unwrap();
/// let x0 = TensorDynLen::from_dense(vec![i.clone()], vec![0.0, 0.0]).unwrap();
///
/// let result = gmres(|x: &TensorDynLen| Ok(x.clone()), &b, &x0, &GmresOptions::default()).unwrap();
/// assert!(result.converged);
/// assert!(result.residual_norm < 1e-10);
/// ```
#[derive(Debug, Clone)]
pub struct GmresResult<T> {
    /// The solution vector.
    pub solution: T,

    /// Number of iterations performed.
    pub iterations: usize,

    /// Final relative residual norm.
    pub residual_norm: f64,

    /// Whether the solver converged.
    pub converged: bool,
}

/// Options for [`hermitian_lanczos_lowest_eigenpair`].
///
/// These options control the maximum Krylov subspace dimension, convergence
/// tolerance, and validation tolerance for the small projected Hermitian matrix.
/// When in doubt, use [`Default::default`].
///
/// # Examples
///
/// ```
/// use tensor4all_core::krylov::HermitianLanczosOptions;
///
/// let opts = HermitianLanczosOptions {
///     max_iter: 32,
///     rtol: 1.0e-9,
///     ..HermitianLanczosOptions::default()
/// };
/// assert_eq!(opts.max_iter, 32);
/// assert_eq!(opts.rtol, 1.0e-9);
/// ```
#[derive(Debug, Clone)]
pub struct HermitianLanczosOptions {
    /// Maximum Krylov subspace dimension.
    ///
    /// Larger values usually improve convergence but increase memory and the
    /// cost of the small Rayleigh-Ritz eigensolve. Default: `64`.
    pub max_iter: usize,

    /// Relative residual tolerance.
    ///
    /// Convergence requires `||A v - lambda v|| <= max(atol, rtol *
    /// max(1, |lambda|))`. Default: `1e-10`.
    pub rtol: f64,

    /// Absolute residual tolerance.
    ///
    /// This is useful when targeting eigenvalues near zero. Default: `0.0`.
    pub atol: f64,

    /// Breakdown threshold for the next Krylov vector norm.
    ///
    /// If the orthogonalized vector norm is at or below this value, the current
    /// Krylov subspace is treated as invariant and iteration stops. Default:
    /// `1e-14`.
    pub breakdown_tol: f64,

    /// Hermitian validation tolerance for the projected Rayleigh-Ritz matrix.
    ///
    /// The projected operator is rejected when `V^dagger A V` is not Hermitian
    /// within this absolute tolerance. Default: `1e-10`.
    pub hermitian_tol: f64,

    /// Whether to print per-iteration residual information.
    ///
    /// Default: `false`.
    pub verbose: bool,
}

impl Default for HermitianLanczosOptions {
    fn default() -> Self {
        Self {
            max_iter: 64,
            rtol: 1.0e-10,
            atol: 0.0,
            breakdown_tol: 1.0e-14,
            hermitian_tol: 1.0e-10,
            verbose: false,
        }
    }
}

/// Result of [`hermitian_lanczos_lowest_eigenpair`].
///
/// Contains the lowest Ritz eigenpair, the final true residual norm, iteration
/// count, and convergence status.
///
/// # Examples
///
/// ```
/// use tensor4all_core::krylov::HermitianLanczosResult;
///
/// let result = HermitianLanczosResult {
///     eigenvalue: 1.0,
///     eigenvector: vec![1.0_f64],
///     iterations: 1,
///     residual_norm: 0.0,
///     converged: true,
/// };
/// assert!(result.converged);
/// assert_eq!(result.eigenvalue, 1.0);
/// ```
#[derive(Debug, Clone)]
pub struct HermitianLanczosResult<T> {
    /// Lowest Ritz eigenvalue.
    pub eigenvalue: f64,

    /// Corresponding Ritz eigenvector in the original vector space.
    pub eigenvector: T,

    /// Number of Krylov operator applications used to build the final subspace.
    pub iterations: usize,

    /// True residual norm `||A v - lambda v||`.
    pub residual_norm: f64,

    /// Whether the true residual satisfied the requested tolerance.
    pub converged: bool,
}

/// Options for [`hermitian_krylov_expm_multiply`].
///
/// The routine builds a Hermitian Lanczos basis for a matrix-free operator,
/// exponentiates the small projected Hermitian matrix, and combines the basis
/// vectors without materializing the full operator. Scalar convergence checks
/// and projected eigendecompositions are explicit non-differentiable
/// boundaries; vector-space tensor operations preserve backend metadata when
/// the underlying tensor implementation supports it.
///
/// # Examples
///
/// ```
/// use tensor4all_core::krylov::HermitianKrylovExpmOptions;
///
/// let options = HermitianKrylovExpmOptions {
///     max_iter: 32,
///     tol: 1.0e-10,
///     ..HermitianKrylovExpmOptions::default()
/// };
/// assert_eq!(options.max_iter, 32);
/// assert_eq!(options.tol, 1.0e-10);
/// ```
#[derive(Debug, Clone)]
pub struct HermitianKrylovExpmOptions {
    /// Maximum Krylov subspace dimension for one exponential application.
    pub max_iter: usize,
    /// Maximum number of equal time splits attempted after non-convergence.
    pub max_time_splits: usize,
    /// Relative convergence tolerance for the Krylov residual estimate.
    pub tol: f64,
    /// Breakdown threshold for zero vectors and invariant Krylov subspaces.
    pub breakdown_tol: f64,
    /// Hermitian validation tolerance for projected matrices.
    pub hermitian_tol: f64,
    /// Whether to print per-attempt diagnostics.
    pub verbose: bool,
}

impl Default for HermitianKrylovExpmOptions {
    fn default() -> Self {
        Self {
            max_iter: 30,
            max_time_splits: 100,
            tol: 1.0e-12,
            breakdown_tol: 1.0e-14,
            hermitian_tol: 1.0e-10,
            verbose: false,
        }
    }
}

/// Result of [`hermitian_krylov_expm_multiply`].
///
/// # Examples
///
/// ```
/// use tensor4all_core::krylov::HermitianKrylovExpmResult;
///
/// let result = HermitianKrylovExpmResult {
///     output: vec![1.0_f64, 0.0],
///     iterations: 1,
///     matvecs: 1,
///     error_estimate: 0.0,
///     converged: true,
///     time_splits: 1,
/// };
/// assert!(result.converged);
/// assert_eq!(result.output[0], 1.0);
/// ```
#[derive(Debug, Clone)]
pub struct HermitianKrylovExpmResult<T> {
    /// Approximation to `exp(exponent * A) * initial`.
    pub output: T,
    /// Total Krylov iterations over all accepted substeps.
    pub iterations: usize,
    /// Total matrix-vector products over all accepted substeps.
    pub matvecs: usize,
    /// Maximum Krylov residual estimate over accepted substeps.
    pub error_estimate: f64,
    /// Whether the requested tolerance was met.
    pub converged: bool,
    /// Number of equal time splits used.
    pub time_splits: usize,
}

#[derive(Debug, Clone)]
struct HermitianRitzState {
    eigenvalue: f64,
    coefficients: Vec<Complex64>,
    iterations: usize,
    residual_estimate: f64,
}

/// Compute the lowest eigenpair of a Hermitian matrix-free operator.
///
/// The operator is supplied as `apply_a`, which maps a vector to `A * vector`.
/// The algorithm builds an orthonormal Krylov basis using modified
/// Gram-Schmidt with a second reorthogonalization pass, forms the small
/// projected matrix `V^dagger A V`, and solves that small Hermitian problem via
/// tensorbackend. The full operator matrix is never materialized.
///
/// # Arguments
/// * `apply_a` - Matrix-free Hermitian operator application.
/// * `initial` - Nonzero initial vector that defines the starting Krylov vector.
/// * `options` - Krylov dimension, convergence tolerances, and Hermitian
///   validation settings.
///
/// # Returns
/// The lowest Ritz eigenpair and the true residual norm.
///
/// # Errors
/// Returns an error if the initial vector is zero, a vector-space operation
/// fails, `apply_a` fails, the projected operator is not Hermitian, or the small
/// projected eigensolver fails.
///
/// # Examples
///
/// ```
/// use tensor4all_core::{DynIndex, TensorDynLen, TensorVectorSpace};
/// use tensor4all_core::krylov::{hermitian_lanczos_lowest_eigenpair, HermitianLanczosOptions};
///
/// let i = DynIndex::new_dyn(2);
/// let initial = TensorDynLen::from_dense(vec![i.clone()], vec![1.0_f64, 1.0]).unwrap();
/// let result = hermitian_lanczos_lowest_eigenpair(
///     |x: &TensorDynLen| Ok(x.clone()),
///     &initial,
///     &HermitianLanczosOptions::default(),
/// ).unwrap();
///
/// assert!(result.converged);
/// assert!((result.eigenvalue - 1.0).abs() < 1.0e-12);
/// ```
pub fn hermitian_lanczos_lowest_eigenpair<T, F>(
    apply_a: F,
    initial: &T,
    options: &HermitianLanczosOptions,
) -> Result<HermitianLanczosResult<T>>
where
    T: TensorVectorSpace,
    F: Fn(&T) -> Result<T>,
{
    validate_hermitian_lanczos_options(options)?;
    initial.validate()?;

    let initial_norm = initial.norm()?;
    anyhow::ensure!(
        initial_norm > options.breakdown_tol,
        "hermitian_lanczos_lowest_eigenpair: zero initial vector"
    );

    let mut basis = Vec::with_capacity(options.max_iter + 1);
    basis.push(initial.scale(AnyScalar::new_real(1.0 / initial_norm))?);
    let mut h_cols: Vec<Vec<Complex64>> = Vec::with_capacity(options.max_iter);
    let mut last_ritz: Option<HermitianRitzState> = None;

    for j in 0..options.max_iter {
        let w = apply_a(&basis[j])?;
        if j == 0 {
            w.validate()?;
        }

        let mut h_col = Vec::with_capacity(j + 2);
        let mut w_orth = w;

        for v_i in basis.iter().take(j + 1) {
            let h_ij = v_i.inner_product(&w_orth)?;
            h_col.push(any_scalar_to_complex(&h_ij));
            let neg_h_ij = AnyScalar::new_real(0.0) - h_ij;
            w_orth = w_orth.axpby(AnyScalar::new_real(1.0), v_i, neg_h_ij)?;
        }

        for (i, v_i) in basis.iter().take(j + 1).enumerate() {
            let correction = v_i.inner_product(&w_orth)?;
            h_col[i] += any_scalar_to_complex(&correction);
            let neg_correction = AnyScalar::new_real(0.0) - correction;
            w_orth = w_orth.axpby(AnyScalar::new_real(1.0), v_i, neg_correction)?;
        }

        let beta = w_orth.norm()?;
        h_col.push(Complex64::new(beta, 0.0));
        h_cols.push(h_col);

        let subspace_dim = j + 1;
        let projected = projected_matrix_from_columns(&h_cols, subspace_dim);
        let ritz = lowest_hermitian_eigenpair(&projected, options.hermitian_tol)
            .map_err(|err| anyhow::anyhow!("projected operator is not Hermitian: {err}"))?;
        let residual_estimate = beta * ritz.eigenvector[subspace_dim - 1].norm();
        let threshold = hermitian_lanczos_threshold(ritz.eigenvalue, options);
        let estimate_converged = residual_estimate <= threshold;

        if options.verbose {
            eprintln!(
                "Hermitian Lanczos iter {}: eigenvalue={:.16e} residual_estimate={:.6e} threshold={:.6e}",
                subspace_dim, ritz.eigenvalue, residual_estimate, threshold
            );
        }

        let ritz_state = HermitianRitzState {
            eigenvalue: ritz.eigenvalue,
            iterations: subspace_dim,
            residual_estimate,
            coefficients: ritz.eigenvector,
        };
        if estimate_converged || beta <= options.breakdown_tol {
            let result = finalize_hermitian_lanczos_result(
                &apply_a,
                &basis[..subspace_dim],
                &ritz_state,
                options,
            )?;
            if result.converged || beta <= options.breakdown_tol {
                return Ok(result);
            }
        }
        last_ritz = Some(ritz_state);
        basis.push(w_orth.scale(AnyScalar::new_real(1.0 / beta))?);
    }

    let ritz_state = last_ritz.ok_or_else(|| {
        anyhow::anyhow!("hermitian_lanczos_lowest_eigenpair: max_iter must be greater than zero")
    })?;
    finalize_hermitian_lanczos_result(
        &apply_a,
        &basis[..ritz_state.iterations],
        &ritz_state,
        options,
    )
}

/// Apply `exp(exponent * A)` to a vector using a matrix-free Hermitian Krylov method.
///
/// This routine only materializes small projected Krylov matrices. It never
/// forms the full matrix for `A`, so it can be used by tensor-network local
/// solvers whose operator application is available as a closure.
///
/// # Arguments
/// * `apply_a` - Matrix-free Hermitian operator application.
/// * `exponent` - Scalar exponent, for example `-im * dt` for real-time TDVP.
/// * `initial` - Initial vector.
/// * `options` - Krylov dimension, tolerance, and Hermitian validation options.
///
/// # Returns
/// A [`HermitianKrylovExpmResult`] containing the evolved vector and diagnostics.
///
/// # Errors
/// Returns an error if an option is invalid, a vector-space operation fails,
/// the projected matrix is not Hermitian, or the requested tolerance is not met
/// after the configured time splits.
///
/// # Examples
///
/// ```
/// use num_complex::Complex64;
/// use tensor4all_core::krylov::{
///     hermitian_krylov_expm_multiply, HermitianKrylovExpmOptions,
/// };
/// use tensor4all_core::{DynIndex, TensorDynLen};
///
/// # fn main() -> anyhow::Result<()> {
/// let index = DynIndex::new_dyn(2);
/// let initial = TensorDynLen::from_dense(
///     vec![index.clone()],
///     vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
/// )?;
/// let options = HermitianKrylovExpmOptions {
///     max_iter: 4,
///     tol: 1.0e-12,
///     ..Default::default()
/// };
/// let result = hermitian_krylov_expm_multiply(
///     |x: &TensorDynLen| {
///         let data = x.to_vec::<Complex64>()?;
///         TensorDynLen::from_dense(
///             vec![index.clone()],
///             vec![data[0], Complex64::new(2.0, 0.0) * data[1]],
///         )
///     },
///     Complex64::new(0.0, -0.25),
///     &initial,
///     &options,
/// )?;
/// let evolved = result.output.to_vec::<Complex64>()?;
/// let expected = Complex64::new(0.25_f64.cos(), -0.25_f64.sin());
/// assert!((evolved[0] - expected).norm() < 1.0e-10);
/// assert!(evolved[1].norm() < 1.0e-12);
/// # Ok(())
/// # }
/// ```
pub fn hermitian_krylov_expm_multiply<T, F>(
    mut apply_a: F,
    exponent: Complex64,
    initial: &T,
    options: &HermitianKrylovExpmOptions,
) -> Result<HermitianKrylovExpmResult<T>>
where
    T: TensorVectorSpace,
    F: FnMut(&T) -> Result<T>,
{
    validate_hermitian_krylov_expm_options(options)?;
    initial.validate()?;

    if exponent == Complex64::new(0.0, 0.0) {
        return Ok(HermitianKrylovExpmResult {
            output: initial.clone(),
            iterations: 0,
            matvecs: 0,
            error_estimate: 0.0,
            converged: true,
            time_splits: 1,
        });
    }
    if initial.norm()? <= options.breakdown_tol {
        return Ok(HermitianKrylovExpmResult {
            output: initial.clone(),
            iterations: 0,
            matvecs: 0,
            error_estimate: 0.0,
            converged: true,
            time_splits: 1,
        });
    }

    let mut splits = 1usize;
    loop {
        let step_exponent = exponent / splits as f64;
        let mut output = initial.clone();
        let mut iterations = 0usize;
        let mut matvecs = 0usize;
        let mut max_error = 0.0_f64;
        let mut converged = true;

        for _ in 0..splits {
            let result =
                hermitian_krylov_expm_multiply_once(&mut apply_a, step_exponent, &output, options)?;
            iterations += result.iterations;
            matvecs += result.matvecs;
            max_error = max_error.max(result.error_estimate);
            output = result.output;
            if !result.converged {
                converged = false;
                break;
            }
        }

        if converged {
            return Ok(HermitianKrylovExpmResult {
                output,
                iterations,
                matvecs,
                error_estimate: max_error,
                converged: true,
                time_splits: splits,
            });
        }

        if options.verbose {
            eprintln!(
                "Hermitian Krylov expm did not converge with {splits} time split(s); retrying"
            );
        }
        if splits >= options.max_time_splits {
            return Err(anyhow::anyhow!(
                "hermitian_krylov_expm_multiply did not converge within max_time_splits={} and max_iter={}",
                options.max_time_splits,
                options.max_iter
            ));
        }
        splits = splits.saturating_mul(2).min(options.max_time_splits);
    }
}

fn hermitian_krylov_expm_multiply_once<T, F>(
    apply_a: &mut F,
    exponent: Complex64,
    initial: &T,
    options: &HermitianKrylovExpmOptions,
) -> Result<HermitianKrylovExpmResult<T>>
where
    T: TensorVectorSpace,
    F: FnMut(&T) -> Result<T>,
{
    let initial_norm = initial.norm()?;
    if initial_norm <= options.breakdown_tol {
        return Ok(HermitianKrylovExpmResult {
            output: initial.clone(),
            iterations: 0,
            matvecs: 0,
            error_estimate: 0.0,
            converged: true,
            time_splits: 1,
        });
    }

    let mut basis = Vec::with_capacity(options.max_iter + 1);
    basis.push(initial.scale(AnyScalar::new_real(1.0 / initial_norm))?);
    let mut h_cols: Vec<Vec<Complex64>> = Vec::with_capacity(options.max_iter);
    let mut last_coefficients: Option<Vec<Complex64>> = None;
    let mut last_error_estimate = f64::INFINITY;
    let mut last_subspace_dim = 0usize;
    let threshold = options.tol * initial_norm.max(1.0);

    for j in 0..options.max_iter {
        let w = apply_a(&basis[j])?;
        if j == 0 {
            w.validate()?;
        }
        let mut w_orth = w;
        let mut h_col = Vec::with_capacity(j + 2);

        for v_i in basis.iter().take(j + 1) {
            let h_ij = v_i.inner_product(&w_orth)?;
            h_col.push(any_scalar_to_complex(&h_ij));
            let neg_h_ij = AnyScalar::new_real(0.0) - h_ij;
            w_orth = w_orth.axpby(AnyScalar::new_real(1.0), v_i, neg_h_ij)?;
        }

        for (i, v_i) in basis.iter().take(j + 1).enumerate() {
            let correction = v_i.inner_product(&w_orth)?;
            h_col[i] += any_scalar_to_complex(&correction);
            let neg_correction = AnyScalar::new_real(0.0) - correction;
            w_orth = w_orth.axpby(AnyScalar::new_real(1.0), v_i, neg_correction)?;
        }

        let beta_next = w_orth.norm()?;
        h_col.push(Complex64::new(beta_next, 0.0));
        h_cols.push(h_col);

        let subspace_dim = j + 1;
        let projected = projected_matrix_from_columns(&h_cols, subspace_dim);
        let coefficients =
            hermitian_exponential_first_column(&projected, exponent, options.hermitian_tol)
                .map_err(|err| anyhow::anyhow!("projected operator is not Hermitian: {err}"))?;
        let error_estimate = if beta_next <= options.breakdown_tol {
            0.0
        } else {
            initial_norm * beta_next * coefficients[subspace_dim - 1].norm()
        };
        let converged = beta_next <= options.breakdown_tol || error_estimate <= threshold;

        if options.verbose {
            eprintln!(
                "Hermitian Krylov expm iter {}: residual_estimate={:.6e} threshold={:.6e}",
                subspace_dim, error_estimate, threshold
            );
        }

        if converged {
            let output = finalize_hermitian_krylov_expm_output(
                &basis[..subspace_dim],
                &coefficients,
                initial_norm,
                options.hermitian_tol,
            )?;
            return Ok(HermitianKrylovExpmResult {
                output,
                iterations: subspace_dim,
                matvecs: subspace_dim,
                error_estimate,
                converged: true,
                time_splits: 1,
            });
        }
        last_coefficients = Some(coefficients);
        last_error_estimate = error_estimate;
        last_subspace_dim = subspace_dim;
        basis.push(w_orth.scale(AnyScalar::new_real(1.0 / beta_next))?);
    }

    let coefficients = last_coefficients.ok_or_else(|| {
        anyhow::anyhow!("hermitian_krylov_expm_multiply: max_iter must be greater than zero")
    })?;
    let output = finalize_hermitian_krylov_expm_output(
        &basis[..last_subspace_dim],
        &coefficients,
        initial_norm,
        options.hermitian_tol,
    )?;
    Ok(HermitianKrylovExpmResult {
        output,
        iterations: last_subspace_dim,
        matvecs: last_subspace_dim,
        error_estimate: last_error_estimate,
        converged: false,
        time_splits: 1,
    })
}

fn finalize_hermitian_krylov_expm_output<T>(
    basis: &[T],
    coefficients: &[Complex64],
    initial_norm: f64,
    hermitian_tol: f64,
) -> Result<T>
where
    T: TensorVectorSpace,
{
    let unit_output = combine_basis_with_complex_coefficients(
        basis,
        coefficients,
        hermitian_tol,
        "hermitian_krylov_expm_multiply",
    )?;
    unit_output.scale(AnyScalar::new_real(initial_norm))
}

/// Solve `A x = b` using GMRES (Generalized Minimal Residual Method).
///
/// This implements the restarted GMRES algorithm that works with abstract tensor types
/// through the [`TensorVectorSpace`] trait's vector space operations.
///
/// # Algorithm
///
/// GMRES builds an orthonormal basis for the Krylov subspace
/// `K_m = span{r_0, A r_0, A^2 r_0, ..., A^{m-1} r_0}` and finds the
/// solution that minimizes `||b - A x||` over this subspace.
///
/// # Type Parameters
///
/// * `T` - A tensor type implementing `TensorVectorSpace`
/// * `F` - A function that applies the linear operator: `F(x) = A x`
///
/// # Arguments
///
/// * `apply_a` - Function that applies the linear operator A to a tensor
/// * `b` - Right-hand side tensor
/// * `x0` - Initial guess
/// * `options` - Solver options
///
/// # Returns
///
/// A `GmresResult` containing the solution and convergence information.
///
/// # Errors
///
/// Returns an error if:
/// - Vector space operations (add, sub, scale, inner_product) fail
/// - The linear operator application fails
pub fn gmres<T, F>(apply_a: F, b: &T, x0: &T, options: &GmresOptions) -> Result<GmresResult<T>>
where
    T: TensorVectorSpace,
    F: Fn(&T) -> Result<T>,
{
    gmres_impl(
        apply_a,
        b,
        x0,
        options,
        GmresTolerance::Relative(options.rtol),
        None,
    )
}

/// Solve `A x = b` using GMRES with an absolute residual tolerance.
///
/// This variant stops when `||b - A*x|| < atol`. The default [`gmres`] API uses
/// relative residual tolerance and is preferred for scale-independent solves.
pub fn gmres_with_absolute_tolerance<T, F>(
    apply_a: F,
    b: &T,
    x0: &T,
    options: &GmresOptions,
    atol: f64,
) -> Result<GmresResult<T>>
where
    T: TensorVectorSpace,
    F: Fn(&T) -> Result<T>,
{
    gmres_impl(
        apply_a,
        b,
        x0,
        options,
        GmresTolerance::Absolute(atol),
        None,
    )
}

/// Solve `(a0 I + a1 A) x = b` using GMRES with relative residual tolerance.
///
/// The Arnoldi basis is built from the unshifted `A` callback, while affine
/// coefficients are applied in the projected Hessenberg problem, matching
/// KrylovKit's affine linear-solve convention.
pub fn gmres_affine<T, F>(
    apply_a: F,
    b: &T,
    x0: &T,
    a0: AnyScalar,
    a1: AnyScalar,
    options: &GmresOptions,
) -> Result<GmresResult<T>>
where
    T: TensorVectorSpace,
    F: Fn(&T) -> Result<T>,
{
    gmres_affine_impl(
        apply_a,
        b,
        x0,
        a0,
        a1,
        options,
        GmresTolerance::Relative(options.rtol),
    )
}

/// Solve `(a0 I + a1 A) x = b` using GMRES with an absolute residual tolerance.
///
/// The Arnoldi basis is built from the unshifted `A` callback, while the affine
/// coefficients are applied to the small Hessenberg problem. This mirrors
/// KrylovKit's `linsolve(operator, b, a0, a1)` algorithm and avoids changing the
/// Krylov basis when affine coefficients are present.
pub fn gmres_affine_with_absolute_tolerance<T, F>(
    apply_a: F,
    b: &T,
    x0: &T,
    a0: AnyScalar,
    a1: AnyScalar,
    options: &GmresOptions,
    atol: f64,
) -> Result<GmresResult<T>>
where
    T: TensorVectorSpace,
    F: Fn(&T) -> Result<T>,
{
    gmres_affine_impl(
        apply_a,
        b,
        x0,
        a0,
        a1,
        options,
        GmresTolerance::Absolute(atol),
    )
}

fn gmres_affine_impl<T, F>(
    apply_a: F,
    b: &T,
    x0: &T,
    a0: AnyScalar,
    a1: AnyScalar,
    options: &GmresOptions,
    tolerance: GmresTolerance,
) -> Result<GmresResult<T>>
where
    T: TensorVectorSpace,
    F: Fn(&T) -> Result<T>,
{
    b.validate()?;
    x0.validate()?;

    let profile_enabled = std::env::var_os("T4A_GMRES_OP_PROFILE").is_some();
    let profile_id = if profile_enabled {
        GMRES_OP_PROFILE_COUNTER.fetch_add(1, Ordering::Relaxed)
    } else {
        0
    };
    let mut profile = GmresOpProfile::default();
    let mut total_iters = 0usize;

    macro_rules! finish {
        ($result:expr) => {{
            let result = $result;
            if profile_enabled {
                profile.print(
                    profile_id,
                    result.iterations,
                    result.residual_norm,
                    result.converged,
                );
            }
            return Ok(result);
        }};
    }

    let started = Instant::now();
    let b_norm = b.norm()?;
    if profile_enabled {
        profile.b_norm += started.elapsed();
    }
    if b_norm < 1e-15 {
        finish!(GmresResult {
            solution: x0.clone(),
            iterations: 0,
            residual_norm: 0.0,
            converged: true,
        });
    }
    if a0.is_zero() && a1.is_zero() {
        anyhow::bail!("gmres_affine: at least one affine coefficient must be nonzero");
    }
    if a1.is_zero() {
        let started = Instant::now();
        let solution = b.scale(AnyScalar::new_real(1.0) / a0)?;
        if profile_enabled {
            profile.scale += started.elapsed();
            profile.scale_calls += 1;
        }
        finish!(GmresResult {
            solution,
            iterations: 0,
            residual_norm: 0.0,
            converged: true,
        });
    }

    let mut x = x0.clone();

    for restart in 0..options.max_restarts {
        let started = Instant::now();
        let ax = apply_a(&x)?;
        if profile_enabled {
            profile.apply += started.elapsed();
            profile.apply_calls += 1;
        }
        if restart == 0 {
            ax.validate()?;
        }
        let started = Instant::now();
        let affine_x = x.axpby(a0.clone(), &ax, a1.clone())?;
        if profile_enabled {
            profile.axpby += started.elapsed();
            profile.axpby_calls += 1;
        }
        let started = Instant::now();
        let r = b.axpby(
            AnyScalar::new_real(1.0),
            &affine_x,
            AnyScalar::new_real(-1.0),
        )?;
        if profile_enabled {
            profile.axpby += started.elapsed();
            profile.axpby_calls += 1;
        }
        let started = Instant::now();
        let r_norm = r.norm()?;
        if profile_enabled {
            profile.norm += started.elapsed();
            profile.norm_calls += 1;
        }
        let residual_value = tolerance.residual_value(r_norm, b_norm);
        if options.verbose {
            eprintln!(
                "GMRES restart {}: initial residual = {:.6e}",
                restart, residual_value
            );
        }
        if tolerance.is_converged(r_norm, b_norm) {
            finish!(GmresResult {
                solution: x,
                iterations: total_iters,
                residual_norm: residual_value,
                converged: true,
            });
        }

        let cycle_max_iter = options.max_iter;
        let mut v_basis: Vec<T> = Vec::with_capacity(cycle_max_iter + 1);
        let started = Instant::now();
        v_basis.push(r.scale(AnyScalar::new_real(1.0 / r_norm))?);
        if profile_enabled {
            profile.scale += started.elapsed();
            profile.scale_calls += 1;
        }

        let mut h_matrix: Vec<Vec<AnyScalar>> = Vec::with_capacity(cycle_max_iter);
        let mut cs: Vec<AnyScalar> = Vec::with_capacity(cycle_max_iter);
        let mut sn: Vec<AnyScalar> = Vec::with_capacity(cycle_max_iter);
        let mut g: Vec<AnyScalar> = vec![AnyScalar::new_real(r_norm)];
        let mut solution_already_updated = false;

        for j in 0..cycle_max_iter {
            total_iters += 1;

            let started = Instant::now();
            let w = apply_a(&v_basis[j])?;
            if profile_enabled {
                profile.apply += started.elapsed();
                profile.apply_calls += 1;
            }
            let mut h_a_col: Vec<AnyScalar> = Vec::with_capacity(j + 2);
            let mut w_orth = w;

            for v_i in v_basis.iter().take(j + 1) {
                let started = Instant::now();
                let h_ij = v_i.inner_product(&w_orth)?;
                if profile_enabled {
                    profile.inner_product += started.elapsed();
                    profile.inner_product_calls += 1;
                }
                h_a_col.push(h_ij.clone());
                let neg_h_ij = AnyScalar::new_real(0.0) - h_ij;
                let started = Instant::now();
                w_orth = w_orth.axpby(AnyScalar::new_real(1.0), v_i, neg_h_ij)?;
                if profile_enabled {
                    profile.axpby += started.elapsed();
                    profile.axpby_calls += 1;
                }
            }
            for (i, v_i) in v_basis.iter().take(j + 1).enumerate() {
                let started = Instant::now();
                let correction = v_i.inner_product(&w_orth)?;
                if profile_enabled {
                    profile.inner_product += started.elapsed();
                    profile.inner_product_calls += 1;
                }
                h_a_col[i] = h_a_col[i].clone() + correction.clone();
                let neg_correction = AnyScalar::new_real(0.0) - correction;
                let started = Instant::now();
                w_orth = w_orth.axpby(AnyScalar::new_real(1.0), v_i, neg_correction)?;
                if profile_enabled {
                    profile.axpby += started.elapsed();
                    profile.axpby_calls += 1;
                }
            }

            let started = Instant::now();
            let h_jp1_j_real = w_orth.norm()?;
            if profile_enabled {
                profile.norm += started.elapsed();
                profile.norm_calls += 1;
            }
            h_a_col.push(AnyScalar::new_real(h_jp1_j_real));

            let mut h_col: Vec<AnyScalar> = Vec::with_capacity(j + 2);
            for h in h_a_col.iter().take(j) {
                h_col.push(a1.clone() * h.clone());
            }
            h_col.push(a0.clone() + a1.clone() * h_a_col[j].clone());
            h_col.push(a1.clone() * h_a_col[j + 1].clone());

            #[allow(clippy::needless_range_loop)]
            for i in 0..j {
                let h_i = h_col[i].clone();
                let h_ip1 = h_col[i + 1].clone();
                let (new_hi, new_hip1) = apply_givens_rotation(&cs[i], &sn[i], &h_i, &h_ip1);
                h_col[i] = new_hi;
                h_col[i + 1] = new_hip1;
            }

            let (c_j, s_j) = compute_givens_rotation(&h_col[j], &h_col[j + 1]);
            cs.push(c_j.clone());
            sn.push(s_j.clone());

            let (new_hj, _) = apply_givens_rotation(&c_j, &s_j, &h_col[j], &h_col[j + 1]);
            h_col[j] = new_hj;
            h_col[j + 1] = AnyScalar::new_real(0.0);

            let g_j = g[j].clone();
            let g_jp1 = AnyScalar::new_real(0.0);
            let (new_gj, new_gjp1) = apply_givens_rotation(&c_j, &s_j, &g_j, &g_jp1);
            g[j] = new_gj;
            let res_norm = new_gjp1.abs();
            g.push(new_gjp1);

            h_matrix.push(h_col);
            let residual_value = tolerance.residual_value(res_norm, b_norm);
            if options.verbose {
                eprintln!("GMRES iter {}: residual = {:.6e}", j + 1, residual_value);
            }

            if tolerance.is_converged(res_norm, b_norm) {
                let started = Instant::now();
                let y = solve_upper_triangular(&h_matrix, &g[..=j])?;
                if profile_enabled {
                    profile.triangular_solve += started.elapsed();
                    profile.triangular_solve_calls += 1;
                }
                let started = Instant::now();
                x = update_solution(&x, &v_basis[..=j], &y)?;
                if profile_enabled {
                    profile.solution_update += started.elapsed();
                    profile.solution_update_calls += 1;
                }
                if options.check_true_residual {
                    let started = Instant::now();
                    let ax_check = apply_a(&x)?;
                    if profile_enabled {
                        profile.apply += started.elapsed();
                        profile.apply_calls += 1;
                    }
                    let started = Instant::now();
                    let affine_check = x.axpby(a0.clone(), &ax_check, a1.clone())?;
                    if profile_enabled {
                        profile.axpby += started.elapsed();
                        profile.axpby_calls += 1;
                    }
                    let started = Instant::now();
                    let r_check = b.axpby(
                        AnyScalar::new_real(1.0),
                        &affine_check,
                        AnyScalar::new_real(-1.0),
                    )?;
                    if profile_enabled {
                        profile.axpby += started.elapsed();
                        profile.axpby_calls += 1;
                    }
                    let started = Instant::now();
                    let true_abs_res = r_check.norm()?;
                    if profile_enabled {
                        profile.norm += started.elapsed();
                        profile.norm_calls += 1;
                    }
                    let true_residual_value = tolerance.residual_value(true_abs_res, b_norm);
                    if options.verbose {
                        eprintln!(
                            "GMRES true residual check: hessenberg={:.6e}, checked={:.6e}",
                            residual_value, true_residual_value
                        );
                    }
                    if tolerance.is_converged(true_abs_res, b_norm) {
                        finish!(GmresResult {
                            solution: x,
                            iterations: total_iters,
                            residual_norm: true_residual_value,
                            converged: true,
                        });
                    }
                    solution_already_updated = true;
                    break;
                } else {
                    finish!(GmresResult {
                        solution: x,
                        iterations: total_iters,
                        residual_norm: residual_value,
                        converged: true,
                    });
                }
            }

            if h_jp1_j_real > 1e-14 {
                let started = Instant::now();
                v_basis.push(w_orth.scale(AnyScalar::new_real(1.0 / h_jp1_j_real))?);
                if profile_enabled {
                    profile.scale += started.elapsed();
                    profile.scale_calls += 1;
                }
            } else {
                let started = Instant::now();
                let y = solve_upper_triangular(&h_matrix, &g[..=j])?;
                if profile_enabled {
                    profile.triangular_solve += started.elapsed();
                    profile.triangular_solve_calls += 1;
                }
                let started = Instant::now();
                x = update_solution(&x, &v_basis[..=j], &y)?;
                if profile_enabled {
                    profile.solution_update += started.elapsed();
                    profile.solution_update_calls += 1;
                }
                let started = Instant::now();
                let ax_final = apply_a(&x)?;
                if profile_enabled {
                    profile.apply += started.elapsed();
                    profile.apply_calls += 1;
                }
                let started = Instant::now();
                let affine_final = x.axpby(a0.clone(), &ax_final, a1.clone())?;
                if profile_enabled {
                    profile.axpby += started.elapsed();
                    profile.axpby_calls += 1;
                }
                let started = Instant::now();
                let r_final = b.axpby(
                    AnyScalar::new_real(1.0),
                    &affine_final,
                    AnyScalar::new_real(-1.0),
                )?;
                if profile_enabled {
                    profile.axpby += started.elapsed();
                    profile.axpby_calls += 1;
                }
                let started = Instant::now();
                let final_abs_res = r_final.norm()?;
                if profile_enabled {
                    profile.norm += started.elapsed();
                    profile.norm_calls += 1;
                }
                let final_res = tolerance.residual_value(final_abs_res, b_norm);
                finish!(GmresResult {
                    solution: x,
                    iterations: total_iters,
                    residual_norm: final_res,
                    converged: tolerance.is_converged(final_abs_res, b_norm),
                });
            }
        }

        if !solution_already_updated {
            let actual_iters = h_matrix.len();
            let started = Instant::now();
            let y = solve_upper_triangular(&h_matrix, &g[..actual_iters])?;
            if profile_enabled {
                profile.triangular_solve += started.elapsed();
                profile.triangular_solve_calls += 1;
            }
            let started = Instant::now();
            x = update_solution(&x, &v_basis[..actual_iters], &y)?;
            if profile_enabled {
                profile.solution_update += started.elapsed();
                profile.solution_update_calls += 1;
            }
        }
    }

    let started = Instant::now();
    let ax_final = apply_a(&x)?;
    if profile_enabled {
        profile.apply += started.elapsed();
        profile.apply_calls += 1;
    }
    let started = Instant::now();
    let affine_final = x.axpby(a0, &ax_final, a1)?;
    if profile_enabled {
        profile.axpby += started.elapsed();
        profile.axpby_calls += 1;
    }
    let started = Instant::now();
    let r_final = b.axpby(
        AnyScalar::new_real(1.0),
        &affine_final,
        AnyScalar::new_real(-1.0),
    )?;
    if profile_enabled {
        profile.axpby += started.elapsed();
        profile.axpby_calls += 1;
    }
    let started = Instant::now();
    let final_abs_res = r_final.norm()?;
    if profile_enabled {
        profile.norm += started.elapsed();
        profile.norm_calls += 1;
    }
    let final_res = tolerance.residual_value(final_abs_res, b_norm);

    finish!(GmresResult {
        solution: x,
        iterations: total_iters,
        residual_norm: final_res,
        converged: tolerance.is_converged(final_abs_res, b_norm),
    })
}

/// Solve `A x = b` using GMRES while enforcing a total iteration limit.
///
/// [`GmresOptions::max_iter`] remains the restart cycle length and
/// [`GmresOptions::max_restarts`] remains the maximum number of restart cycles.
/// `max_total_iter` caps the total number of Arnoldi steps across all restart
/// cycles; the final cycle is shortened when necessary.
pub fn gmres_with_total_iteration_limit<T, F>(
    apply_a: F,
    b: &T,
    x0: &T,
    options: &GmresOptions,
    max_total_iter: usize,
) -> Result<GmresResult<T>>
where
    T: TensorVectorSpace,
    F: Fn(&T) -> Result<T>,
{
    gmres_impl(
        apply_a,
        b,
        x0,
        options,
        GmresTolerance::Relative(options.rtol),
        Some(max_total_iter),
    )
}

fn gmres_impl<T, F>(
    apply_a: F,
    b: &T,
    x0: &T,
    options: &GmresOptions,
    tolerance: GmresTolerance,
    max_total_iter: Option<usize>,
) -> Result<GmresResult<T>>
where
    T: TensorVectorSpace,
    F: Fn(&T) -> Result<T>,
{
    // Validate structural consistency of inputs
    b.validate()?;
    x0.validate()?;

    let b_norm = b.norm()?;
    if b_norm < 1e-15 {
        // b is effectively zero, return x0
        return Ok(GmresResult {
            solution: x0.clone(),
            iterations: 0,
            residual_norm: 0.0,
            converged: true,
        });
    }

    let mut x = x0.clone();
    let mut total_iters = 0;

    for _restart in 0..options.max_restarts {
        let cycle_max_iter = match max_total_iter {
            Some(limit) => {
                let remaining = limit.saturating_sub(total_iters);
                if remaining == 0 {
                    break;
                }
                options.max_iter.min(remaining)
            }
            None => options.max_iter,
        };
        if cycle_max_iter == 0 {
            break;
        }

        // Compute initial residual: r = b - A*x
        let ax = apply_a(&x)?;
        // Validate operator output on first restart
        if _restart == 0 {
            ax.validate()?;
        }
        // r = 1.0 * b + (-1.0) * ax
        let r = b.axpby(AnyScalar::new_real(1.0), &ax, AnyScalar::new_real(-1.0))?;
        let r_norm = r.norm()?;
        let residual_value = tolerance.residual_value(r_norm, b_norm);

        if options.verbose {
            eprintln!(
                "GMRES restart {}: initial residual = {:.6e}",
                _restart, residual_value
            );
        }

        if tolerance.is_converged(r_norm, b_norm) {
            return Ok(GmresResult {
                solution: x,
                iterations: total_iters,
                residual_norm: residual_value,
                converged: true,
            });
        }

        // Arnoldi process with modified Gram-Schmidt
        let mut v_basis: Vec<T> = Vec::with_capacity(cycle_max_iter + 1);
        let mut h_matrix: Vec<Vec<AnyScalar>> = Vec::with_capacity(cycle_max_iter);

        // v_0 = r / ||r||
        let v0 = r.scale(AnyScalar::new_real(1.0 / r_norm))?;
        v_basis.push(v0);

        // Initialize Givens rotation storage
        let mut cs: Vec<AnyScalar> = Vec::with_capacity(cycle_max_iter);
        let mut sn: Vec<AnyScalar> = Vec::with_capacity(cycle_max_iter);
        let mut g: Vec<AnyScalar> = vec![AnyScalar::new_real(r_norm)]; // residual in upper Hessenberg space
        let mut solution_already_updated = false;

        for j in 0..cycle_max_iter {
            total_iters += 1;

            // w = A * v_j
            let w = apply_a(&v_basis[j])?;

            // Modified Gram-Schmidt orthogonalization
            let mut h_col: Vec<AnyScalar> = Vec::with_capacity(j + 2);
            let mut w_orth = w;

            for v_i in v_basis.iter().take(j + 1) {
                let h_ij = v_i.inner_product(&w_orth)?;
                h_col.push(h_ij.clone());
                // w_orth = w_orth - h_ij * v_i = 1.0 * w_orth + (-h_ij) * v_i
                let neg_h_ij = AnyScalar::new_real(0.0) - h_ij;
                w_orth = w_orth.axpby(AnyScalar::new_real(1.0), v_i, neg_h_ij)?;
            }

            // KrylovKit's default orthogonalizer is ModifiedGramSchmidt2.
            // The second pass is important for long Krylov bases and complex
            // non-Hermitian local problems.
            for (i, v_i) in v_basis.iter().take(j + 1).enumerate() {
                let correction = v_i.inner_product(&w_orth)?;
                h_col[i] = h_col[i].clone() + correction.clone();
                let neg_correction = AnyScalar::new_real(0.0) - correction;
                w_orth = w_orth.axpby(AnyScalar::new_real(1.0), v_i, neg_correction)?;
            }

            let h_jp1_j_real = w_orth.norm()?;
            let h_jp1_j = AnyScalar::new_real(h_jp1_j_real);
            h_col.push(h_jp1_j);

            // Apply previous Givens rotations to new column
            #[allow(clippy::needless_range_loop)]
            for i in 0..j {
                let h_i = h_col[i].clone();
                let h_ip1 = h_col[i + 1].clone();
                let (new_hi, new_hip1) = apply_givens_rotation(&cs[i], &sn[i], &h_i, &h_ip1);
                h_col[i] = new_hi;
                h_col[i + 1] = new_hip1;
            }

            // Compute new Givens rotation for h_col[j] and h_col[j+1]
            let (c_j, s_j) = compute_givens_rotation(&h_col[j], &h_col[j + 1]);
            cs.push(c_j.clone());
            sn.push(s_j.clone());

            // Apply new rotation to eliminate h_col[j+1]
            let (new_hj, _) = apply_givens_rotation(&c_j, &s_j, &h_col[j], &h_col[j + 1]);
            h_col[j] = new_hj;
            h_col[j + 1] = AnyScalar::new_real(0.0);

            // Apply rotation to g
            let g_j = g[j].clone();
            let g_jp1 = AnyScalar::new_real(0.0);
            let (new_gj, new_gjp1) = apply_givens_rotation(&c_j, &s_j, &g_j, &g_jp1);
            g[j] = new_gj;
            let res_norm = new_gjp1.abs();
            g.push(new_gjp1);

            h_matrix.push(h_col);

            // Check convergence
            let residual_value = tolerance.residual_value(res_norm, b_norm);

            if options.verbose {
                eprintln!("GMRES iter {}: residual = {:.6e}", j + 1, residual_value);
            }

            if tolerance.is_converged(res_norm, b_norm) {
                // Solve upper triangular system and update x
                let y = solve_upper_triangular(&h_matrix, &g[..=j])?;
                x = update_solution(&x, &v_basis[..=j], &y)?;
                if options.check_true_residual {
                    let ax_check = apply_a(&x)?;
                    let r_check = b.axpby(
                        AnyScalar::new_real(1.0),
                        &ax_check,
                        AnyScalar::new_real(-1.0),
                    )?;
                    let true_abs_res = r_check.norm()?;
                    let true_residual_value = tolerance.residual_value(true_abs_res, b_norm);

                    if options.verbose {
                        eprintln!(
                            "GMRES true residual check: hessenberg={:.6e}, checked={:.6e}",
                            residual_value, true_residual_value
                        );
                    }

                    if tolerance.is_converged(true_abs_res, b_norm) {
                        return Ok(GmresResult {
                            solution: x,
                            iterations: total_iters,
                            residual_norm: true_residual_value,
                            converged: true,
                        });
                    }
                    solution_already_updated = true;
                    break;
                } else {
                    return Ok(GmresResult {
                        solution: x,
                        iterations: total_iters,
                        residual_norm: residual_value,
                        converged: true,
                    });
                }
            }

            // Add new basis vector (if not converged and h_jp1_j is not too small)
            if h_jp1_j_real > 1e-14 {
                let v_jp1 = w_orth.scale(AnyScalar::new_real(1.0 / h_jp1_j_real))?;
                v_basis.push(v_jp1);
            } else {
                // Lucky breakdown - we've found the exact solution in the Krylov subspace
                let y = solve_upper_triangular(&h_matrix, &g[..=j])?;
                x = update_solution(&x, &v_basis[..=j], &y)?;
                let ax_final = apply_a(&x)?;
                let r_final = b.axpby(
                    AnyScalar::new_real(1.0),
                    &ax_final,
                    AnyScalar::new_real(-1.0),
                )?;
                let final_abs_res = r_final.norm()?;
                let final_res = tolerance.residual_value(final_abs_res, b_norm);
                return Ok(GmresResult {
                    solution: x,
                    iterations: total_iters,
                    residual_norm: final_res,
                    converged: tolerance.is_converged(final_abs_res, b_norm),
                });
            }
        }

        // End of restart cycle - update x with current solution
        if !solution_already_updated {
            let actual_iters = h_matrix.len();
            let y = solve_upper_triangular(&h_matrix, &g[..actual_iters])?;
            x = update_solution(&x, &v_basis[..actual_iters], &y)?;
        }
    }

    // Compute final residual
    let ax_final = apply_a(&x)?;
    let r_final = b.axpby(
        AnyScalar::new_real(1.0),
        &ax_final,
        AnyScalar::new_real(-1.0),
    )?;
    let final_abs_res = r_final.norm()?;
    let final_res = tolerance.residual_value(final_abs_res, b_norm);

    Ok(GmresResult {
        solution: x,
        iterations: total_iters,
        residual_norm: final_res,
        converged: tolerance.is_converged(final_abs_res, b_norm),
    })
}

/// Solve `A x = b` using GMRES with optional truncation after each iteration.
///
/// This is an extension of [`gmres`] that allows truncating Krylov basis vectors
/// to control bond dimension growth in tensor network representations.
///
/// # Type Parameters
///
/// * `T` - A tensor type implementing `TensorVectorSpace`
/// * `F` - A function that applies the linear operator: `F(x) = A x`
/// * `Tr` - A function that truncates a tensor in-place: `Tr(&mut x)`
///
/// # Arguments
///
/// * `apply_a` - Function that applies the linear operator A to a tensor
/// * `b` - Right-hand side tensor
/// * `x0` - Initial guess
/// * `options` - Solver options
/// * `truncate` - Function that truncates a tensor to control bond dimension
///
/// # Note
///
/// Truncation is applied after each Gram-Schmidt orthogonalization step
/// and after the final solution update. This helps control the bond dimension
/// growth that would otherwise occur in MPS/MPO representations.
///
/// # Examples
///
/// Solve `2x = b` with a no-op truncation function:
///
/// ```
/// use tensor4all_core::{DynIndex, TensorDynLen, TensorVectorSpace, AnyScalar};
/// use tensor4all_core::krylov::{gmres_with_truncation, GmresOptions};
///
/// let i = DynIndex::new_dyn(2);
/// let b = TensorDynLen::from_dense(vec![i.clone()], vec![4.0, 6.0]).unwrap();
/// let x0 = TensorDynLen::from_dense(vec![i.clone()], vec![0.0, 0.0]).unwrap();
///
/// // Operator A = 2*I (scales input by 2)
/// let apply_a = |x: &TensorDynLen| x.scale(AnyScalar::new_real(2.0));
/// // No-op truncation
/// let truncate = |_x: &mut TensorDynLen| Ok(());
///
/// let result = gmres_with_truncation(apply_a, &b, &x0, &GmresOptions::default(), truncate).unwrap();
/// assert!(result.converged);
/// // Solution should be [2.0, 3.0]
/// let expected = TensorDynLen::from_dense(vec![i], vec![2.0, 3.0]).unwrap();
/// assert!(result.solution.sub(&expected).unwrap().maxabs().unwrap() < 1e-8);
/// ```
pub fn gmres_with_truncation<T, F, Tr>(
    apply_a: F,
    b: &T,
    x0: &T,
    options: &GmresOptions,
    truncate: Tr,
) -> Result<GmresResult<T>>
where
    T: TensorVectorSpace,
    F: Fn(&T) -> Result<T>,
    Tr: Fn(&mut T) -> Result<()>,
{
    // Validate structural consistency of inputs
    b.validate()?;
    x0.validate()?;

    let b_norm = b.norm()?;
    if b_norm < 1e-15 {
        return Ok(GmresResult {
            solution: x0.clone(),
            iterations: 0,
            residual_norm: 0.0,
            converged: true,
        });
    }

    let mut x = x0.clone();
    let mut total_iters = 0;

    for _restart in 0..options.max_restarts {
        let ax = apply_a(&x)?;
        // Validate operator output on first restart
        if _restart == 0 {
            ax.validate()?;
        }
        let mut r = b.axpby(AnyScalar::new_real(1.0), &ax, AnyScalar::new_real(-1.0))?;
        truncate(&mut r)?;
        let r_norm = r.norm()?;
        let rel_res = r_norm / b_norm;

        if options.verbose {
            eprintln!(
                "GMRES restart {}: initial residual = {:.6e}",
                _restart, rel_res
            );
        }

        if rel_res < options.rtol {
            return Ok(GmresResult {
                solution: x,
                iterations: total_iters,
                residual_norm: rel_res,
                converged: true,
            });
        }

        let mut v_basis: Vec<T> = Vec::with_capacity(options.max_iter + 1);
        let mut h_matrix: Vec<Vec<AnyScalar>> = Vec::with_capacity(options.max_iter);

        let mut v0 = r.scale(AnyScalar::new_real(1.0 / r_norm))?;
        truncate(&mut v0)?;
        // After truncation, v0 might not be unit norm and might point in a different direction.
        // We need to:
        // 1. Renormalize v0 to unit norm for numerical stability
        // 2. Recompute g[0] = <r, v0> to maintain the correct relationship
        let v0_norm = v0.norm()?;
        let effective_g0 = if v0_norm > 1e-15 {
            v0 = v0.scale(AnyScalar::new_real(1.0 / v0_norm))?;
            // g[0] should be the component of r in the direction of v0
            // Since r was truncated and v0 = truncate(r/||r||)/||truncate(r/||r||)||,
            // g[0] = <r, v0> ≈ ||r|| * ||truncate(r/||r||)|| = r_norm * v0_norm
            r_norm * v0_norm
        } else {
            r_norm
        };
        v_basis.push(v0);

        let mut cs: Vec<AnyScalar> = Vec::with_capacity(options.max_iter);
        let mut sn: Vec<AnyScalar> = Vec::with_capacity(options.max_iter);
        let mut g: Vec<AnyScalar> = vec![AnyScalar::new_real(effective_g0)];
        let mut solution_already_updated = false;

        for j in 0..options.max_iter {
            total_iters += 1;

            let w = apply_a(&v_basis[j])?;

            let mut h_col: Vec<AnyScalar> = Vec::with_capacity(j + 2);
            let mut w_orth = w;

            for v_i in v_basis.iter().take(j + 1) {
                let h_ij = v_i.inner_product(&w_orth)?;
                h_col.push(h_ij.clone());
                let neg_h_ij = AnyScalar::new_real(0.0) - h_ij;
                w_orth = w_orth.axpby(AnyScalar::new_real(1.0), v_i, neg_h_ij)?;
            }

            // Iterative reorthogonalization with truncation
            // Truncation can change the direction of w_orth, breaking orthogonality.
            // We iterate until all corrections are below a threshold to ensure
            // the Krylov basis remains orthogonal despite truncation.
            const REORTH_THRESHOLD: f64 = 1e-12;
            const MAX_REORTH_ITERS: usize = 10;

            let mut reorth_iter_count = 0;
            for reorth_iter in 0..MAX_REORTH_ITERS {
                reorth_iter_count = reorth_iter + 1;
                let norm_before_truncate = w_orth.norm()?;
                truncate(&mut w_orth)?;
                let norm_after_truncate = w_orth.norm()?;

                let mut max_correction = 0.0;
                for (i, v_i) in v_basis.iter().enumerate() {
                    let correction = v_i.inner_product(&w_orth)?;
                    let correction_abs = correction.abs();
                    if correction_abs > max_correction {
                        max_correction = correction_abs;
                    }
                    if correction_abs > REORTH_THRESHOLD {
                        let neg_correction = AnyScalar::new_real(0.0) - correction.clone();
                        w_orth = w_orth.axpby(AnyScalar::new_real(1.0), v_i, neg_correction)?;
                        // Update Hessenberg matrix entry to include correction
                        h_col[i] = h_col[i].clone() + correction;
                    }
                }

                if options.verbose {
                    eprintln!(
                        "  reorth iter {}: norm {:.6e} -> {:.6e}, max_correction = {:.6e}",
                        reorth_iter, norm_before_truncate, norm_after_truncate, max_correction
                    );
                }

                // If all corrections are small enough, we're done
                if max_correction < REORTH_THRESHOLD {
                    break;
                }
            }

            if options.verbose && reorth_iter_count > 1 {
                eprintln!("  (needed {} reorth iterations)", reorth_iter_count);
            }

            let h_jp1_j_real = w_orth.norm()?;
            let h_jp1_j = AnyScalar::new_real(h_jp1_j_real);
            h_col.push(h_jp1_j);

            #[allow(clippy::needless_range_loop)]
            for i in 0..j {
                let h_i = h_col[i].clone();
                let h_ip1 = h_col[i + 1].clone();
                let (new_hi, new_hip1) = apply_givens_rotation(&cs[i], &sn[i], &h_i, &h_ip1);
                h_col[i] = new_hi;
                h_col[i + 1] = new_hip1;
            }

            let (c_j, s_j) = compute_givens_rotation(&h_col[j], &h_col[j + 1]);
            cs.push(c_j.clone());
            sn.push(s_j.clone());

            let (new_hj, _) = apply_givens_rotation(&c_j, &s_j, &h_col[j], &h_col[j + 1]);
            h_col[j] = new_hj;
            h_col[j + 1] = AnyScalar::new_real(0.0);

            let g_j = g[j].clone();
            let g_jp1 = AnyScalar::new_real(0.0);
            let (new_gj, new_gjp1) = apply_givens_rotation(&c_j, &s_j, &g_j, &g_jp1);
            g[j] = new_gj;
            let res_norm = new_gjp1.abs();
            g.push(new_gjp1);

            h_matrix.push(h_col);

            let rel_res = res_norm / b_norm;

            if options.verbose {
                eprintln!("GMRES iter {}: residual = {:.6e}", j + 1, rel_res);
            }

            if rel_res < options.rtol {
                let y = solve_upper_triangular(&h_matrix, &g[..=j])?;
                x = update_solution_truncated(&x, &v_basis[..=j], &y, &truncate)?;

                if options.check_true_residual {
                    // Verify with true residual to prevent false convergence
                    let ax_check = apply_a(&x)?;
                    let mut r_check = b.axpby(
                        AnyScalar::new_real(1.0),
                        &ax_check,
                        AnyScalar::new_real(-1.0),
                    )?;
                    truncate(&mut r_check)?;
                    let true_rel_res = r_check.norm()? / b_norm;

                    if options.verbose {
                        eprintln!(
                            "GMRES true residual check: hessenberg={:.6e}, checked={:.6e}",
                            rel_res, true_rel_res
                        );
                    }

                    if true_rel_res < options.rtol {
                        return Ok(GmresResult {
                            solution: x,
                            iterations: total_iters,
                            residual_norm: true_rel_res,
                            converged: true,
                        });
                    }
                    // False convergence detected: x is already updated above,
                    // so skip the end-of-cycle update and go to next restart
                    solution_already_updated = true;
                    break;
                } else {
                    return Ok(GmresResult {
                        solution: x,
                        iterations: total_iters,
                        residual_norm: rel_res,
                        converged: true,
                    });
                }
            }

            if h_jp1_j_real > 1e-14 {
                // Create v_{j+1} = w_orth / ||w_orth||
                // w_orth has already been truncated twice (after orthogonalization and after reorthogonalization)
                // so we don't need to truncate again. Scale doesn't increase bond dimensions.
                let v_jp1 = w_orth.scale(AnyScalar::new_real(1.0 / h_jp1_j_real))?;
                // v_jp1 should have norm ~1.0 by construction
                // The Arnoldi relation h_{j+1,j} * v_{j+1} = w_orth is maintained exactly
                v_basis.push(v_jp1);
            } else {
                let y = solve_upper_triangular(&h_matrix, &g[..=j])?;
                x = update_solution_truncated(&x, &v_basis[..=j], &y, &truncate)?;
                let ax_final = apply_a(&x)?;
                let r_final = b.axpby(
                    AnyScalar::new_real(1.0),
                    &ax_final,
                    AnyScalar::new_real(-1.0),
                )?;
                let final_res = r_final.norm()? / b_norm;
                return Ok(GmresResult {
                    solution: x,
                    iterations: total_iters,
                    residual_norm: final_res,
                    converged: final_res < options.rtol,
                });
            }
        }

        if !solution_already_updated {
            let actual_iters = v_basis.len().min(options.max_iter);
            let y = solve_upper_triangular(&h_matrix, &g[..actual_iters])?;
            x = update_solution_truncated(&x, &v_basis[..actual_iters], &y, &truncate)?;
        }
    }

    let ax_final = apply_a(&x)?;
    let r_final = b.axpby(
        AnyScalar::new_real(1.0),
        &ax_final,
        AnyScalar::new_real(-1.0),
    )?;
    let final_res = r_final.norm()? / b_norm;

    Ok(GmresResult {
        solution: x,
        iterations: total_iters,
        residual_norm: final_res,
        converged: final_res < options.rtol,
    })
}

/// Options for restarted GMRES with truncation.
///
/// This is used by [`restart_gmres_with_truncation`] which wraps the standard GMRES
/// with an outer loop that recomputes the true residual at each restart.
///
/// # Examples
///
/// ```
/// use tensor4all_core::krylov::RestartGmresOptions;
///
/// let opts = RestartGmresOptions::new()
///     .with_max_outer_iters(10)
///     .with_rtol(1e-6)
///     .with_inner_max_iter(20)
///     .with_inner_max_restarts(2)
///     .with_min_reduction(0.99)
///     .with_inner_rtol(0.01)
///     .with_verbose(false);
///
/// assert_eq!(opts.max_outer_iters, 10);
/// assert_eq!(opts.rtol, 1e-6);
/// assert_eq!(opts.inner_max_iter, 20);
/// assert_eq!(opts.inner_max_restarts, 2);
/// assert_eq!(opts.min_reduction, Some(0.99));
/// assert_eq!(opts.inner_rtol, Some(0.01));
/// ```
#[derive(Debug, Clone)]
pub struct RestartGmresOptions {
    /// Maximum number of outer restart iterations.
    /// Default: 20
    pub max_outer_iters: usize,

    /// Convergence tolerance for relative residual norm (based on true residual).
    /// The solver stops when `||b - A*x|| / ||b|| < rtol`.
    /// Default: 1e-10
    pub rtol: f64,

    /// Maximum iterations per inner GMRES cycle.
    /// Default: 10
    pub inner_max_iter: usize,

    /// Number of restarts within each inner GMRES (usually 0).
    /// Default: 0
    pub inner_max_restarts: usize,

    /// Stagnation detection threshold.
    /// If the residual reduction ratio exceeds this value (i.e., residual doesn't decrease enough),
    /// the solver considers it stagnated.
    /// For example, 0.99 means stagnation is detected when residual decreases by less than 1%.
    /// Default: None (no stagnation detection)
    pub min_reduction: Option<f64>,

    /// Inner GMRES relative tolerance.
    /// If None, uses 0.1 (solve inner problem loosely).
    /// Default: None
    pub inner_rtol: Option<f64>,

    /// Whether to print convergence information.
    /// Default: false
    pub verbose: bool,
}

impl Default for RestartGmresOptions {
    fn default() -> Self {
        Self {
            max_outer_iters: 20,
            rtol: 1e-10,
            inner_max_iter: 10,
            inner_max_restarts: 0,
            min_reduction: None,
            inner_rtol: None,
            verbose: false,
        }
    }
}

impl RestartGmresOptions {
    /// Create new options with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set maximum number of outer iterations.
    pub fn with_max_outer_iters(mut self, max_outer_iters: usize) -> Self {
        self.max_outer_iters = max_outer_iters;
        self
    }

    /// Set convergence tolerance.
    pub fn with_rtol(mut self, rtol: f64) -> Self {
        self.rtol = rtol;
        self
    }

    /// Set maximum iterations per inner GMRES cycle.
    pub fn with_inner_max_iter(mut self, inner_max_iter: usize) -> Self {
        self.inner_max_iter = inner_max_iter;
        self
    }

    /// Set number of restarts within each inner GMRES.
    pub fn with_inner_max_restarts(mut self, inner_max_restarts: usize) -> Self {
        self.inner_max_restarts = inner_max_restarts;
        self
    }

    /// Set stagnation detection threshold.
    pub fn with_min_reduction(mut self, min_reduction: f64) -> Self {
        self.min_reduction = Some(min_reduction);
        self
    }

    /// Set inner GMRES relative tolerance.
    pub fn with_inner_rtol(mut self, inner_rtol: f64) -> Self {
        self.inner_rtol = Some(inner_rtol);
        self
    }

    /// Enable verbose output.
    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }
}

/// Result of restarted GMRES solver.
///
/// # Examples
///
/// ```
/// use tensor4all_core::{DynIndex, TensorDynLen, AnyScalar};
/// use tensor4all_core::krylov::{restart_gmres_with_truncation, RestartGmresOptions};
///
/// let i = DynIndex::new_dyn(2);
/// let b = TensorDynLen::from_dense(vec![i.clone()], vec![3.0, 5.0]).unwrap();
///
/// let apply_a = |x: &TensorDynLen| x.scale(AnyScalar::new_real(3.0));
/// let truncate = |_x: &mut TensorDynLen| Ok(());
///
/// let result = restart_gmres_with_truncation(
///     apply_a, &b, None, &RestartGmresOptions::default(), truncate,
/// ).unwrap();
///
/// assert!(result.converged);
/// assert!(result.residual_norm < 1e-10);
/// assert!(result.outer_iterations <= 20);
/// ```
#[derive(Debug, Clone)]
pub struct RestartGmresResult<T> {
    /// The solution vector.
    pub solution: T,

    /// Total number of inner GMRES iterations performed.
    pub iterations: usize,

    /// Number of outer restart iterations performed.
    pub outer_iterations: usize,

    /// Final relative residual norm (true residual).
    pub residual_norm: f64,

    /// Whether the solver converged.
    pub converged: bool,
}

/// Solve `A x = b` using restarted GMRES with truncation.
///
/// This wraps [`gmres_with_truncation`] with an outer loop that recomputes the true residual
/// at each restart. This is particularly useful for MPS/MPO computations where truncation
/// can cause the inner GMRES residual to be inaccurate.
///
/// # Algorithm
///
/// ```text
/// for outer_iter in 0..max_outer_iters:
///     r = b - A*x0          // Compute true residual
///     r = truncate(r)
///     if ||r|| / ||b|| < rtol:
///         return x0         // Converged
///     x' = gmres_with_truncation(A, r, 0, inner_options, truncate)
///     x0 = truncate(x0 + x')
/// ```
///
/// # Type Parameters
///
/// * `T` - A tensor type implementing `TensorVectorSpace`
/// * `F` - A function that applies the linear operator: `F(x) = A x`
/// * `Tr` - A function that truncates a tensor in-place: `Tr(&mut x)`
///
/// # Arguments
///
/// * `apply_a` - Function that applies the linear operator A to a tensor
/// * `b` - Right-hand side tensor
/// * `x0` - Initial guess (if None, starts from zero)
/// * `options` - Solver options
/// * `truncate` - Function that truncates a tensor to control bond dimension
///
/// # Returns
///
/// A `RestartGmresResult` containing the solution and convergence information.
///
/// # Examples
///
/// Solve `5x = b` with no truncation:
///
/// ```
/// use tensor4all_core::{DynIndex, TensorDynLen, TensorVectorSpace, AnyScalar};
/// use tensor4all_core::krylov::{restart_gmres_with_truncation, RestartGmresOptions};
///
/// let i = DynIndex::new_dyn(3);
/// let b = TensorDynLen::from_dense(vec![i.clone()], vec![5.0, 10.0, 15.0]).unwrap();
///
/// let apply_a = |x: &TensorDynLen| x.scale(AnyScalar::new_real(5.0));
/// let truncate = |_x: &mut TensorDynLen| Ok(());
///
/// let result = restart_gmres_with_truncation(
///     apply_a, &b, None, &RestartGmresOptions::default(), truncate,
/// ).unwrap();
///
/// assert!(result.converged);
/// let expected = TensorDynLen::from_dense(vec![i], vec![1.0, 2.0, 3.0]).unwrap();
/// assert!(result.solution.sub(&expected).unwrap().maxabs().unwrap() < 1e-8);
/// ```
pub fn restart_gmres_with_truncation<T, F, Tr>(
    apply_a: F,
    b: &T,
    x0: Option<&T>,
    options: &RestartGmresOptions,
    truncate: Tr,
) -> Result<RestartGmresResult<T>>
where
    T: TensorVectorSpace,
    F: Fn(&T) -> Result<T>,
    Tr: Fn(&mut T) -> Result<()>,
{
    // Validate structural consistency of inputs
    b.validate()?;
    if let Some(x) = x0 {
        x.validate()?;
    }

    let b_norm = b.norm()?;
    if b_norm < 1e-15 {
        // b is effectively zero, return x0 or zero
        let solution = match x0 {
            Some(x) => x.clone(),
            None => b.scale(AnyScalar::new_real(0.0))?,
        };
        return Ok(RestartGmresResult {
            solution,
            iterations: 0,
            outer_iterations: 0,
            residual_norm: 0.0,
            converged: true,
        });
    }

    // Initialize x: use x0 if provided, otherwise start from zero.
    // Track whether x is zero to avoid unnecessary bond dimension doubling
    // when adding the first correction via axpby.
    let mut x_is_zero = x0.is_none();
    let mut x = match x0 {
        Some(x) => x.clone(),
        None => b.scale(AnyScalar::new_real(0.0))?,
    };

    let mut total_inner_iters = 0;
    let mut prev_residual_norm = f64::INFINITY;

    // Inner GMRES options
    let inner_options = GmresOptions {
        max_iter: options.inner_max_iter,
        rtol: options.inner_rtol.unwrap_or(0.1), // Solve loosely by default
        max_restarts: options.inner_max_restarts + 1, // +1 because max_restarts=0 means 1 cycle
        verbose: options.verbose,
        check_true_residual: true, // Always check in restart context to avoid false convergence
    };

    for outer_iter in 0..options.max_outer_iters {
        // Compute true residual: r = b - A*x
        let ax = apply_a(&x)?;
        // Validate operator output on first outer iteration
        if outer_iter == 0 {
            ax.validate()?;
        }
        let mut r = b.axpby(AnyScalar::new_real(1.0), &ax, AnyScalar::new_real(-1.0))?;
        truncate(&mut r)?;

        let r_norm = r.norm()?;
        let rel_res = r_norm / b_norm;

        if options.verbose {
            eprintln!(
                "Restart GMRES outer iter {}: true residual = {:.6e}",
                outer_iter, rel_res
            );
        }

        // Check convergence
        if rel_res < options.rtol {
            return Ok(RestartGmresResult {
                solution: x,
                iterations: total_inner_iters,
                outer_iterations: outer_iter,
                residual_norm: rel_res,
                converged: true,
            });
        }

        // Check stagnation
        if let Some(min_reduction) = options.min_reduction {
            if outer_iter > 0 && rel_res > prev_residual_norm * min_reduction {
                if options.verbose {
                    eprintln!(
                        "Restart GMRES stagnated: residual ratio = {:.6e} > {:.6e}",
                        rel_res / prev_residual_norm,
                        min_reduction
                    );
                }
                return Ok(RestartGmresResult {
                    solution: x,
                    iterations: total_inner_iters,
                    outer_iterations: outer_iter,
                    residual_norm: rel_res,
                    converged: false,
                });
            }
        }
        prev_residual_norm = rel_res;

        // Solve A*x' = r using inner GMRES with zero initial guess
        // The zero initial guess is created by scaling r by 0
        let zero = r.scale(AnyScalar::new_real(0.0))?;
        let inner_result = gmres_with_truncation(&apply_a, &r, &zero, &inner_options, &truncate)?;

        total_inner_iters += inner_result.iterations;

        if options.verbose {
            eprintln!(
                "  Inner GMRES: {} iterations, residual = {:.6e}, converged = {}",
                inner_result.iterations, inner_result.residual_norm, inner_result.converged
            );
        }

        // Update solution: x = x + x'
        // When x is zero (first iteration with no initial guess), use x' directly
        // to avoid bond dimension doubling from axpby with a zero tensor.
        if x_is_zero {
            x = inner_result.solution;
            x_is_zero = false;
        } else {
            x = x.axpby(
                AnyScalar::new_real(1.0),
                &inner_result.solution,
                AnyScalar::new_real(1.0),
            )?;
        }
        truncate(&mut x)?;
    }

    // Did not converge within max_outer_iters
    // Compute final residual
    let ax = apply_a(&x)?;
    let mut r = b.axpby(AnyScalar::new_real(1.0), &ax, AnyScalar::new_real(-1.0))?;
    truncate(&mut r)?;
    let final_rel_res = r.norm()? / b_norm;

    Ok(RestartGmresResult {
        solution: x,
        iterations: total_inner_iters,
        outer_iterations: options.max_outer_iters,
        residual_norm: final_rel_res,
        converged: false,
    })
}

fn validate_hermitian_lanczos_options(options: &HermitianLanczosOptions) -> Result<()> {
    anyhow::ensure!(
        options.max_iter > 0,
        "hermitian_lanczos_lowest_eigenpair: max_iter must be greater than zero"
    );
    anyhow::ensure!(
        options.rtol.is_finite() && options.rtol >= 0.0,
        "hermitian_lanczos_lowest_eigenpair: rtol must be finite and non-negative"
    );
    anyhow::ensure!(
        options.atol.is_finite() && options.atol >= 0.0,
        "hermitian_lanczos_lowest_eigenpair: atol must be finite and non-negative"
    );
    anyhow::ensure!(
        options.breakdown_tol.is_finite() && options.breakdown_tol >= 0.0,
        "hermitian_lanczos_lowest_eigenpair: breakdown_tol must be finite and non-negative"
    );
    anyhow::ensure!(
        options.hermitian_tol.is_finite() && options.hermitian_tol >= 0.0,
        "hermitian_lanczos_lowest_eigenpair: hermitian_tol must be finite and non-negative"
    );
    Ok(())
}

fn validate_hermitian_krylov_expm_options(options: &HermitianKrylovExpmOptions) -> Result<()> {
    anyhow::ensure!(
        options.max_iter > 0,
        "hermitian_krylov_expm_multiply: max_iter must be greater than zero"
    );
    anyhow::ensure!(
        options.max_time_splits > 0,
        "hermitian_krylov_expm_multiply: max_time_splits must be greater than zero"
    );
    anyhow::ensure!(
        options.tol.is_finite() && options.tol >= 0.0,
        "hermitian_krylov_expm_multiply: tol must be finite and non-negative"
    );
    anyhow::ensure!(
        options.breakdown_tol.is_finite() && options.breakdown_tol >= 0.0,
        "hermitian_krylov_expm_multiply: breakdown_tol must be finite and non-negative"
    );
    anyhow::ensure!(
        options.hermitian_tol.is_finite() && options.hermitian_tol >= 0.0,
        "hermitian_krylov_expm_multiply: hermitian_tol must be finite and non-negative"
    );
    Ok(())
}

fn projected_matrix_from_columns(h_cols: &[Vec<Complex64>], dim: usize) -> Matrix<Complex64> {
    let mut data = vec![Complex64::new(0.0, 0.0); dim * dim];
    for col in 0..dim {
        for row in 0..dim {
            if let Some(value) = h_cols.get(col).and_then(|h_col| h_col.get(row)) {
                data[row + dim * col] = *value;
            }
        }
    }
    Matrix::from_col_major_vec(dim, dim, data)
}

fn any_scalar_to_complex(value: &AnyScalar) -> Complex64 {
    value
        .as_c64()
        .unwrap_or_else(|| Complex64::new(value.real(), 0.0))
}

fn any_scalar_from_complex(value: Complex64, tolerance: f64) -> Result<AnyScalar> {
    if value.im.abs() <= tolerance {
        Ok(AnyScalar::new_real(value.re))
    } else {
        Ok(AnyScalar::new_complex(value.re, value.im))
    }
}

fn hermitian_lanczos_threshold(eigenvalue: f64, options: &HermitianLanczosOptions) -> f64 {
    options.atol.max(options.rtol * eigenvalue.abs().max(1.0))
}

fn hermitian_true_residual_norm<T, F>(apply_a: &F, eigenvector: &T, eigenvalue: f64) -> Result<f64>
where
    T: TensorVectorSpace,
    F: Fn(&T) -> Result<T>,
{
    let av = apply_a(eigenvector)?;
    let lambda_v = eigenvector.scale(AnyScalar::new_real(eigenvalue))?;
    let residual = av.axpby(
        AnyScalar::new_real(1.0),
        &lambda_v,
        AnyScalar::new_real(-1.0),
    )?;
    residual.norm()
}

fn finalize_hermitian_lanczos_result<T, F>(
    apply_a: &F,
    basis: &[T],
    ritz: &HermitianRitzState,
    options: &HermitianLanczosOptions,
) -> Result<HermitianLanczosResult<T>>
where
    T: TensorVectorSpace,
    F: Fn(&T) -> Result<T>,
{
    let eigenvector = combine_basis_with_coefficients(basis, &ritz.coefficients, options)?;
    let residual_norm = hermitian_true_residual_norm(apply_a, &eigenvector, ritz.eigenvalue)?;
    let threshold = hermitian_lanczos_threshold(ritz.eigenvalue, options);
    let converged = residual_norm <= threshold;

    if options.verbose {
        eprintln!(
            "Hermitian Lanczos final check iter {}: residual_estimate={:.6e} true_residual={:.6e} threshold={:.6e}",
            ritz.iterations, ritz.residual_estimate, residual_norm, threshold
        );
    }

    Ok(HermitianLanczosResult {
        eigenvalue: ritz.eigenvalue,
        eigenvector,
        iterations: ritz.iterations,
        residual_norm,
        converged,
    })
}

fn combine_basis_with_coefficients<T>(
    basis: &[T],
    coefficients: &[Complex64],
    options: &HermitianLanczosOptions,
) -> Result<T>
where
    T: TensorVectorSpace,
{
    anyhow::ensure!(
        !basis.is_empty(),
        "hermitian_lanczos_lowest_eigenpair: empty Krylov basis"
    );
    anyhow::ensure!(
        basis.len() == coefficients.len(),
        "hermitian_lanczos_lowest_eigenpair: coefficient length {} does not match basis length {}",
        coefficients.len(),
        basis.len()
    );

    let mut result = basis[0].scale(any_scalar_from_complex(
        coefficients[0],
        options.hermitian_tol,
    )?)?;
    for (basis_vector, coefficient) in basis.iter().zip(coefficients.iter()).skip(1) {
        result = result.axpby(
            AnyScalar::new_real(1.0),
            basis_vector,
            any_scalar_from_complex(*coefficient, options.hermitian_tol)?,
        )?;
    }
    Ok(result)
}

fn combine_basis_with_complex_coefficients<T>(
    basis: &[T],
    coefficients: &[Complex64],
    hermitian_tol: f64,
    context: &'static str,
) -> Result<T>
where
    T: TensorVectorSpace,
{
    anyhow::ensure!(!basis.is_empty(), "{context}: empty Krylov basis");
    anyhow::ensure!(
        basis.len() == coefficients.len(),
        "{context}: coefficient length {} does not match basis length {}",
        coefficients.len(),
        basis.len()
    );

    let mut result = basis[0].scale(any_scalar_from_complex(coefficients[0], hermitian_tol)?)?;
    for (basis_vector, coefficient) in basis.iter().zip(coefficients.iter()).skip(1) {
        result = result.axpby(
            AnyScalar::new_real(1.0),
            basis_vector,
            any_scalar_from_complex(*coefficient, hermitian_tol)?,
        )?;
    }
    Ok(result)
}

/// Compute Givens rotation coefficients to eliminate b in (a, b).
///
/// This function keeps computation in `AnyScalar` space to preserve AD metadata
/// as much as possible.
fn compute_givens_rotation(a: &AnyScalar, b: &AnyScalar) -> (AnyScalar, AnyScalar) {
    let a_abs = a.abs();
    let b_abs = b.abs();
    let r = (a_abs * a_abs + b_abs * b_abs).sqrt();
    if r < 1e-15 {
        (AnyScalar::new_real(1.0), AnyScalar::new_real(0.0))
    } else if a_abs < 1e-15 {
        (
            AnyScalar::new_real(0.0),
            b.clone().conj() / AnyScalar::new_real(r),
        )
    } else {
        let phase = a.clone() / AnyScalar::new_real(a_abs);
        (
            AnyScalar::new_real(a_abs / r),
            phase * b.clone().conj() / AnyScalar::new_real(r),
        )
    }
}

/// Apply Givens rotation: (c, s) @ (x, y) -> (c*x + s*y, -conj(s)*x + c*y) for complex
/// or (c*x + s*y, -s*x + c*y) for real.
///
/// This function keeps computation in `AnyScalar` space to preserve AD metadata
/// as much as possible.
fn apply_givens_rotation(
    c: &AnyScalar,
    s: &AnyScalar,
    x: &AnyScalar,
    y: &AnyScalar,
) -> (AnyScalar, AnyScalar) {
    let new_x = c.clone() * x.clone() + s.clone() * y.clone();
    let new_y = -(s.clone().conj() * x.clone()) + c.clone() * y.clone();
    (new_x, new_y)
}

/// Solve upper triangular system R y = g using back substitution.
fn solve_upper_triangular(h: &[Vec<AnyScalar>], g: &[AnyScalar]) -> Result<Vec<AnyScalar>> {
    let n = g.len();
    if n == 0 {
        return Ok(vec![]);
    }

    let mut y = vec![AnyScalar::new_real(0.0); n];

    for i in (0..n).rev() {
        let mut sum = g[i].clone();

        for j in (i + 1)..n {
            // sum = sum - h[j][i] * y[j]
            let prod = h[j][i].clone() * y[j].clone();
            sum = sum - prod;
        }

        let h_ii = &h[i][i];
        if h_ii.abs() < 1e-15 {
            return Err(anyhow::anyhow!(
                "Near-singular upper triangular matrix in GMRES"
            ));
        }

        y[i] = sum / h_ii.clone();
    }

    Ok(y)
}

/// Update solution: x_new = x + sum_i y_i * v_i
fn update_solution<T: TensorVectorSpace>(x: &T, v_basis: &[T], y: &[AnyScalar]) -> Result<T> {
    let mut result = x.clone();

    for (vi, yi) in v_basis.iter().zip(y.iter()) {
        let scaled_vi = vi.scale(yi.clone())?;
        // result = result + scaled_vi = 1.0 * result + 1.0 * scaled_vi
        result = result.axpby(
            AnyScalar::new_real(1.0),
            &scaled_vi,
            AnyScalar::new_real(1.0),
        )?;
    }

    Ok(result)
}

/// Update solution with truncation: x_new = truncate(x + sum_i y_i * v_i)
fn update_solution_truncated<T, Tr>(
    x: &T,
    v_basis: &[T],
    y: &[AnyScalar],
    truncate: &Tr,
) -> Result<T>
where
    T: TensorVectorSpace,
    Tr: Fn(&mut T) -> Result<()>,
{
    let mut result = x.clone();
    // Detect if x is effectively zero.
    // When x is created via scale(0.0), it preserves the original bond structure
    // (e.g., bond dim 4), causing axpby to double bond dimensions unnecessarily.
    // By detecting zero, we can use scaled_vi directly, avoiding the doubling.
    let mut result_is_zero = x.norm()? == 0.0;

    for (vi, yi) in v_basis.iter().zip(y.iter()) {
        let scaled_vi = vi.scale(yi.clone())?;
        if result_is_zero {
            result = scaled_vi;
            result_is_zero = false;
        } else {
            result = result.axpby(
                AnyScalar::new_real(1.0),
                &scaled_vi,
                AnyScalar::new_real(1.0),
            )?;
        }
        // Truncate after each addition to control bond dimension growth
        truncate(&mut result)?;
    }

    Ok(result)
}

#[cfg(test)]
mod tests;
