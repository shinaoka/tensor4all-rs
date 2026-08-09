//! Minimal test: GMRES with identity operator on MPS with bond dimension > 1.
//!
//! Creates an MPS `b` with bond dim 2 or 4 (by adding product-state MPS),
//! then solves A*x = b where A = identity using restart_gmres_with_truncation.
//! The solution should be x = b.
//!
//! Observation: GMRES reports convergence (residual < tol) but the actual
//! solution error ||x - b|| / ||b|| is much larger. This happens because
//! the true residual check in gmres_with_truncation truncates r_check
//! before computing its norm, which can make the residual appear smaller
//! than it actually is.
//!
//! Run with:
//!   cargo run --release --example test_gmres_identity_high_bonddim -p tensor4all-itensorlike

use anyhow::Result;
use tensor4all_core::krylov::{restart_gmres_with_truncation, RestartGmresOptions};
use tensor4all_core::{AnyScalar, DynIndex, TensorDynLen};
use tensor4all_itensorlike::{TensorTrain, TruncateOptions};

/// Create a product-state MPS (bond dim 1) with given per-site values.
fn create_product_mps(
    site_indices: &[DynIndex],
    bond_indices: &[DynIndex],
    site_values: &[[f64; 2]],
) -> Result<TensorTrain> {
    let n = site_indices.len();
    assert_eq!(n, site_values.len());
    assert_eq!(bond_indices.len(), n.saturating_sub(1));

    let mut tensors = Vec::with_capacity(n);
    for i in 0..n {
        let mut indices = Vec::new();
        if i > 0 {
            indices.push(bond_indices[i - 1].clone());
        }
        indices.push(site_indices[i].clone());
        if i < n - 1 {
            indices.push(bond_indices[i].clone());
        }
        let data = site_values[i].to_vec();
        tensors.push(TensorDynLen::from_dense(indices, data).unwrap());
    }
    TensorTrain::new(tensors).map_err(|e| anyhow::anyhow!("{e}"))
}

/// Create an MPS with given bond dim by summing product states.
fn create_mps_with_bond_dim(n: usize, phys_dim: usize, num_states: usize) -> Result<TensorTrain> {
    assert_eq!(phys_dim, 2, "Only phys_dim=2 supported");
    let sites: Vec<DynIndex> = (0..n).map(|_| DynIndex::new_dyn(phys_dim)).collect();

    let patterns: Vec<[f64; 2]> = vec![[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, -1.0]];

    let mut b: Option<TensorTrain> = None;
    for i in 0..num_states {
        let bonds: Vec<DynIndex> = (0..n.saturating_sub(1))
            .map(|_| DynIndex::new_dyn(1))
            .collect();
        let values: Vec<[f64; 2]> = vec![patterns[i % patterns.len()]; n];
        let mps = create_product_mps(&sites, &bonds, &values)?;
        b = Some(match b {
            None => mps,
            Some(prev) => prev.axpby(AnyScalar::new_real(1.0), &mps, AnyScalar::new_real(1.0))?,
        });
    }
    Ok(b.unwrap())
}

/// Run GMRES with identity operator and report results.
/// Returns (converged, reported_residual, actual_error).
fn run_identity_gmres(b: &TensorTrain, max_outer_iters: usize) -> Result<(bool, f64, f64)> {
    let b_norm = b.norm()?;

    let apply_identity = |x: &TensorTrain| -> Result<TensorTrain> { Ok(x.clone()) };

    let truncate_opts = TruncateOptions::svd()
        .with_svd_policy(tensor4all_core::SvdTruncationPolicy::new(1e-10))
        .with_max_rank(100);
    let truncate_fn = |x: &mut TensorTrain| -> Result<()> {
        x.truncate(&truncate_opts)?;
        Ok(())
    };

    let options = RestartGmresOptions {
        max_outer_iters,
        rtol: 1e-8,
        inner_max_iter: 10,
        inner_max_restarts: 0,
        min_reduction: None,
        inner_rtol: Some(0.1),
        verbose: false,
    };

    let result = restart_gmres_with_truncation(apply_identity, b, None, &options, truncate_fn)?;

    // Independently check ||x - b|| / ||b||
    let diff = result
        .solution
        .axpby(AnyScalar::new_real(1.0), b, AnyScalar::new_real(-1.0))?;
    let actual_err = diff.norm()? / b_norm;

    Ok((result.converged, result.residual_norm, actual_err))
}

fn main() -> Result<()> {
    eprintln!("=== GMRES identity operator: bond dim > 1 ===\n");
    eprintln!("Solve A*x = b where A = identity. Expected: x = b.\n");
    eprintln!(
        "{:<20} {:<12} {:<12} {:<15} {:<15}",
        "Case", "Bond dims", "Converged", "GMRES residual", "Actual ||x-b||/||b||"
    );
    eprintln!("{}", "-".repeat(74));

    let mut any_failed = false;

    for (n, num_states) in [(4, 1), (4, 2), (8, 2), (4, 4), (8, 4)] {
        let b = create_mps_with_bond_dim(n, 2, num_states)?;
        let bond_dims = b.bond_dims();
        let max_bond = bond_dims.iter().copied().max().unwrap_or(0);
        let label = format!("n={n}, bd={max_bond}");

        let max_iters = if max_bond <= 1 { 30 } else { 100 };
        let (converged, residual, actual_err) = run_identity_gmres(&b, max_iters)?;

        let status = if converged && actual_err < 1e-4 {
            "OK"
        } else if converged && actual_err >= 1e-4 {
            any_failed = true;
            "WRONG" // converged but solution is wrong
        } else {
            any_failed = true;
            "FAIL"
        };

        eprintln!(
            "{:<20} {:?} {:<12} {:<15.2e} {:<15.2e} {}",
            label, bond_dims, converged, residual, actual_err, status
        );
    }

    eprintln!();
    if any_failed {
        eprintln!("ISSUE: GMRES reports convergence but actual solution error is large.");
        eprintln!("       For identity operator A=I, ||b - Ax|| = ||b - x||,");
        eprintln!("       so GMRES residual and actual error should agree.");
        eprintln!("       The discrepancy likely comes from truncating r_check");
        eprintln!("       before computing its norm in the true residual check.");
        std::process::exit(1);
    } else {
        eprintln!("All tests PASSED.");
    }

    Ok(())
}
