//! Common options for linsolve algorithms.

use crate::TruncationOptions;
use tensor4all_core::{AnyScalar, SvdTruncationPolicy};

/// Residual tolerance convention for local GMRES solves.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GmresToleranceMode {
    /// Stop when `||b - A*x|| / ||b|| < gmres_tol`.
    Relative,
    /// Stop when `||b - A*x|| < gmres_tol`.
    Absolute,
}

/// Options for the linsolve algorithm.
#[derive(Debug, Clone)]
pub struct LinsolveOptions {
    /// Number of full sweeps to perform.
    ///
    /// A full sweep visits each edge twice (forward and backward) using an Euler tour.
    pub nfullsweeps: usize,
    /// Truncation options for factorization.
    pub truncation: TruncationOptions,
    /// Tolerance for GMRES convergence.
    pub gmres_tol: f64,
    /// Residual tolerance convention for GMRES convergence.
    pub gmres_tolerance_mode: GmresToleranceMode,
    /// Maximum number of GMRES restart cycles per local solve.
    ///
    /// This matches KrylovKit's `maxiter` convention. The maximum number of
    /// operator expansion steps is roughly `gmres_max_restarts * gmres_restart_dim`.
    pub gmres_max_restarts: usize,
    /// GMRES restart cycle length.
    pub gmres_restart_dim: usize,
    /// Coefficient a₀ in (a₀ + a₁ * A) * x = b.
    pub a0: AnyScalar,
    /// Coefficient a₁ in (a₀ + a₁ * A) * x = b.
    pub a1: AnyScalar,
    /// Convergence tolerance for early termination.
    /// If Some(tol), stop when relative residual < tol.
    pub convergence_tol: Option<f64>,
    /// Whether to compute and return a final true residual after the sweep.
    ///
    /// Disabling this skips an extra operator application after the requested sweeps. A residual
    /// is still computed when `convergence_tol` is set because early stopping depends on it.
    pub check_residual: bool,
}

impl Default for LinsolveOptions {
    fn default() -> Self {
        Self {
            nfullsweeps: 5,
            truncation: TruncationOptions::default(),
            gmres_tol: 1e-10,
            gmres_tolerance_mode: GmresToleranceMode::Relative,
            gmres_max_restarts: 100,
            gmres_restart_dim: 30,
            a0: AnyScalar::new_real(0.0),
            a1: AnyScalar::new_real(1.0),
            convergence_tol: None,
            check_residual: true,
        }
    }
}

impl LinsolveOptions {
    /// Create new options with specified number of full sweeps.
    pub fn new(nfullsweeps: usize) -> Self {
        Self {
            nfullsweeps,
            ..Default::default()
        }
    }

    /// Set number of full sweeps.
    pub fn with_nfullsweeps(mut self, nfullsweeps: usize) -> Self {
        self.nfullsweeps = nfullsweeps;
        self
    }

    /// Set truncation options.
    pub fn with_truncation(mut self, truncation: TruncationOptions) -> Self {
        self.truncation = truncation;
        self
    }

    /// Set maximum bond dimension.
    pub fn with_max_bond_dim(mut self, max_bond_dim: usize) -> Self {
        self.truncation = self.truncation.with_max_bond_dim(max_bond_dim);
        self
    }

    /// Set the SVD truncation policy used by TreeTN truncation steps.
    pub fn with_svd_policy(mut self, policy: SvdTruncationPolicy) -> Self {
        self.truncation = self.truncation.with_svd_policy(policy);
        self
    }

    /// Set GMRES tolerance.
    pub fn with_gmres_tol(mut self, tol: f64) -> Self {
        self.gmres_tol = tol;
        self
    }

    /// Set the residual tolerance convention for GMRES.
    pub fn with_gmres_tolerance_mode(mut self, mode: GmresToleranceMode) -> Self {
        self.gmres_tolerance_mode = mode;
        self
    }

    /// Use relative residual convergence for GMRES.
    pub fn with_gmres_relative_tolerance(mut self) -> Self {
        self.gmres_tolerance_mode = GmresToleranceMode::Relative;
        self
    }

    /// Use absolute residual convergence for GMRES.
    pub fn with_gmres_absolute_tolerance(mut self) -> Self {
        self.gmres_tolerance_mode = GmresToleranceMode::Absolute;
        self
    }

    /// Set maximum number of GMRES restart cycles.
    ///
    /// This follows KrylovKit's `maxiter` convention.
    pub fn with_gmres_max_restarts(mut self, max_restarts: usize) -> Self {
        self.gmres_max_restarts = max_restarts;
        self
    }

    /// Set GMRES restart cycle length.
    pub fn with_gmres_restart_dim(mut self, dim: usize) -> Self {
        self.gmres_restart_dim = dim;
        self
    }

    /// Set coefficients a₀ and a₁.
    pub fn with_coefficients<A0, A1>(mut self, a0: A0, a1: A1) -> Self
    where
        A0: Into<AnyScalar>,
        A1: Into<AnyScalar>,
    {
        self.a0 = a0.into();
        self.a1 = a1.into();
        self
    }

    /// Set convergence tolerance for early termination.
    pub fn with_convergence_tol(mut self, tol: f64) -> Self {
        self.convergence_tol = Some(tol);
        self
    }

    /// Set whether `square_linsolve` computes a final true residual.
    ///
    /// Use `false` when the caller only needs the swept solution and wants to avoid the extra
    /// post-solve operator application. The residual is still evaluated if `convergence_tol` is
    /// enabled, because it is required for early stopping.
    pub fn with_residual_check(mut self, check_residual: bool) -> Self {
        self.check_residual = check_residual;
        self
    }
}

#[cfg(test)]
mod tests;
