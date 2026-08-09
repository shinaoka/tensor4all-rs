#![allow(clippy::needless_borrows_for_generic_args)]
#![allow(clippy::identity_op)]
//! Test: Restart GMRES solver with MPS/MPO format
//!
//! Tests with Identity MPO and Pauli-X MPO first (N=2,3),
//! then proceeds to larger N if successful.
//!
//! Run:
//!   cargo run -p tensor4all-itensorlike --example test_restart_gmres_mps --release

use tensor4all_core::krylov::{restart_gmres_with_truncation, RestartGmresOptions};
use tensor4all_core::{AnyScalar, DynIndex, IndexLike, TensorDynLen, TensorIndex};
use tensor4all_itensorlike::{ContractOptions, TensorTrain, TruncateOptions};

/// Shared indices for all MPS/MPO operations.
struct SharedIndices {
    sites: Vec<DynIndex>,
    bonds: Vec<DynIndex>,
    mpo_outputs: Vec<DynIndex>,
}

impl SharedIndices {
    fn new(n: usize, phys_dim: usize) -> Self {
        let sites: Vec<DynIndex> = (0..n).map(|_| DynIndex::new_dyn(phys_dim)).collect();
        let bonds: Vec<DynIndex> = (0..n.saturating_sub(1))
            .map(|_| DynIndex::new_dyn(1))
            .collect();
        let mpo_outputs: Vec<DynIndex> = (0..n).map(|_| DynIndex::new_dyn(phys_dim)).collect();

        SharedIndices {
            sites,
            bonds,
            mpo_outputs,
        }
    }
}

fn main() -> anyhow::Result<()> {
    println!("========================================");
    println!("  Restart GMRES with MPS/MPO Tests");
    println!("========================================\n");

    // Test Identity MPO first (simplest case)
    println!("========================================");
    println!("  Identity MPO Tests (A = I)");
    println!("========================================\n");

    for n in [2, 3] {
        let result = test_restart_gmres(n, "identity")?;
        println!(
            "N={}: converged={}, outer_iters={}, true_residual={:.2e}\n",
            n, result.0, result.1, result.2
        );
        assert!(result.0, "Identity MPO should converge for N={}", n);
    }

    // If N=2,3 passed, test larger N
    println!("Identity N=2,3 passed! Testing N=5...\n");
    let result = test_restart_gmres(5, "identity")?;
    println!(
        "N=5: converged={}, outer_iters={}, true_residual={:.2e}\n",
        result.0, result.1, result.2
    );
    assert!(result.0, "Identity MPO should converge for N=5");

    // Test Pauli-X MPO (X^2 = I, condition number = 1)
    println!("\n========================================");
    println!("  Pauli-X MPO Tests (X^2 = I)");
    println!("========================================\n");

    for n in [2, 3] {
        let result = test_restart_gmres(n, "pauli_x")?;
        println!(
            "N={}: converged={}, outer_iters={}, true_residual={:.2e}\n",
            n, result.0, result.1, result.2
        );
        assert!(result.0, "Pauli-X MPO should converge for N={}", n);
    }

    // If N=2,3 passed, test larger N
    println!("Pauli-X N=2,3 passed! Testing N=5, 8...\n");
    for n in [5, 8] {
        let result = test_restart_gmres(n, "pauli_x")?;
        println!(
            "N={}: converged={}, outer_iters={}, true_residual={:.2e}\n",
            n, result.0, result.1, result.2
        );
    }

    // Test with harder problem: Diagonal MPO with aggressive truncation
    println!("\n========================================");
    println!("  Diagonal MPO Tests (harder problem)");
    println!("========================================\n");

    let result = test_restart_gmres_hard(3)?;
    println!(
        "N=3 Diagonal: converged={}, outer_iters={}, true_residual={:.2e}\n",
        result.0, result.1, result.2
    );

    // Test with complex problem: imaginary diagonal MPO * imaginary ones MPS
    println!("\n========================================");
    println!("  Imaginary Diagonal MPO Tests (A = i*diag, x = i*ones)");
    println!("========================================\n");

    let result = test_restart_gmres_imaginary_diagonal(3)?;
    println!(
        "N=3 Imaginary Diagonal: converged={}, outer_iters={}, true_residual={:.2e}\n",
        result.0, result.1, result.2
    );

    println!("\n========================================");
    println!("  All tests completed!");
    println!("========================================");

    Ok(())
}

/// Test restart GMRES with specified operator
/// Returns (converged, outer_iterations, true_residual)
fn test_restart_gmres(n: usize, operator: &str) -> anyhow::Result<(bool, usize, f64)> {
    let phys_dim = 2;
    println!("--- N={}, operator={} ---", n, operator);

    let indices = SharedIndices::new(n, phys_dim);

    // Create MPO
    let mpo = match operator {
        "identity" => create_identity_mpo(&indices)?,
        "pauli_x" => create_pauli_x_mpo(&indices)?,
        _ => anyhow::bail!("Unknown operator: {}", operator),
    };

    // Create x_true (all ones MPS)
    let x_true = create_ones_mps(&indices)?;
    println!("x_true norm: {:.6}", x_true.norm()?);

    // Compute b = A * x_true
    let b = apply_mpo(&mpo, &x_true, &indices)?;
    let b_norm = b.norm()?;
    println!("b = A * x_true, norm: {:.6}", b_norm);

    // Truncation options
    let truncate_opts = TruncateOptions::svd()
        .with_svd_policy(tensor4all_core::SvdTruncationPolicy::new(1e-10))
        .with_max_rank(30);
    let truncate_fn = |x: &mut TensorTrain| -> anyhow::Result<()> {
        x.truncate(&truncate_opts)?;
        Ok(())
    };

    let apply_a = |x: &TensorTrain| -> anyhow::Result<TensorTrain> { apply_mpo(&mpo, x, &indices) };

    // Compute initial residual (x0 = 0, so r0 = b - A*0 = b)
    let initial_residual = 1.0; // ||b|| / ||b|| = 1
    println!("Initial residual (x0=0): {:.6e}", initial_residual);

    // Restart GMRES options
    let options = RestartGmresOptions {
        max_outer_iters: 30,
        rtol: 1e-8,
        inner_max_iter: 10,
        inner_max_restarts: 0,
        min_reduction: None, // Don't stop on stagnation for these tests
        inner_rtol: Some(0.1),
        verbose: true,
    };

    let result = restart_gmres_with_truncation(&apply_a, &b, None, &options, &truncate_fn)?;

    // Compute true residual
    let ax = apply_a(&result.solution)?;
    let r = ax.axpby(AnyScalar::new_real(1.0), &b, AnyScalar::new_real(-1.0))?;
    let true_residual = r.norm()? / b_norm;

    println!("Converged: {}", result.converged);
    println!("Outer iterations: {}", result.outer_iterations);
    println!("Total inner iterations: {}", result.iterations);
    println!("True residual: {:.6e}", true_residual);
    println!("Bond dims: {:?}", result.solution.bond_dims());

    // Compute error ||x_sol - x_true||
    let diff =
        result
            .solution
            .axpby(AnyScalar::new_real(1.0), &x_true, AnyScalar::new_real(-1.0))?;
    println!("Error ||x - x_true||: {:.6e}", diff.norm()?);

    Ok((result.converged, result.outer_iterations, true_residual))
}

/// Create an identity MPO.
fn create_identity_mpo(indices: &SharedIndices) -> anyhow::Result<TensorTrain> {
    let n = indices.sites.len();
    let mut tensors = Vec::with_capacity(n);

    let mpo_bonds: Vec<DynIndex> = (0..n.saturating_sub(1))
        .map(|_| DynIndex::new_dyn(1))
        .collect();

    for i in 0..n {
        let site_dim = indices.sites[i].dim();
        let s_in = indices.sites[i].clone();
        let s_out = indices.mpo_outputs[i].clone();

        // Identity tensor: delta(s_in, s_out)
        let mut data = vec![0.0; site_dim * site_dim];
        for j in 0..site_dim {
            data[j * site_dim + j] = 1.0;
        }

        if i == 0 && n == 1 {
            let tensor = TensorDynLen::from_dense(vec![s_in, s_out], data).unwrap();
            tensors.push(tensor);
        } else if i == 0 {
            let right_bond = mpo_bonds[i].clone();
            let tensor = TensorDynLen::from_dense(vec![s_in, s_out, right_bond], data).unwrap();
            tensors.push(tensor);
        } else if i == n - 1 {
            let left_bond = mpo_bonds[i - 1].clone();
            let tensor = TensorDynLen::from_dense(vec![left_bond, s_in, s_out], data).unwrap();
            tensors.push(tensor);
        } else {
            let left_bond = mpo_bonds[i - 1].clone();
            let right_bond = mpo_bonds[i].clone();
            let tensor =
                TensorDynLen::from_dense(vec![left_bond, s_in, s_out, right_bond], data).unwrap();
            tensors.push(tensor);
        }
    }

    TensorTrain::new(tensors).map_err(|e| anyhow::anyhow!("{}", e))
}

/// Test restart GMRES with a harder problem (diagonal MPO + aggressive truncation)
/// Returns (converged, outer_iterations, true_residual)
fn test_restart_gmres_hard(n: usize) -> anyhow::Result<(bool, usize, f64)> {
    let phys_dim = 2;
    println!("--- N={}, operator=diagonal (harder problem) ---", n);

    let indices = SharedIndices::new(n, phys_dim);

    // Create diagonal MPO: diag(2, 3) at each site
    // Condition number = (3/2)^N
    let mpo = create_diagonal_mpo(&indices)?;
    let cond_number = (1.5_f64).powi(n as i32);
    println!(
        "Diagonal MPO created, condition number ≈ {:.2}",
        cond_number
    );

    // Create x_true (all ones MPS)
    let x_true = create_ones_mps(&indices)?;
    println!("x_true norm: {:.6}", x_true.norm()?);

    // Compute b = A * x_true
    let b = apply_mpo(&mpo, &x_true, &indices)?;
    let b_norm = b.norm()?;
    println!("b = A * x_true, norm: {:.6}", b_norm);

    // AGGRESSIVE truncation to make the problem harder
    let truncate_opts = TruncateOptions::svd()
        .with_svd_policy(tensor4all_core::SvdTruncationPolicy::new(1e-6))
        .with_max_rank(5);
    let truncate_fn = |x: &mut TensorTrain| -> anyhow::Result<()> {
        x.truncate(&truncate_opts)?;
        Ok(())
    };

    let apply_a = |x: &TensorTrain| -> anyhow::Result<TensorTrain> { apply_mpo(&mpo, x, &indices) };

    let initial_residual = 1.0;
    println!("Initial residual (x0=0): {:.6e}", initial_residual);
    println!(
        "Truncation: rtol={:.0e}, max_rank={}",
        truncate_opts
            .svd_policy()
            .map(|policy| policy.threshold)
            .unwrap_or(0.0),
        truncate_opts.max_rank().unwrap_or(0)
    );

    // Restart GMRES options - small inner iterations to force multiple outer iterations
    let options = RestartGmresOptions {
        max_outer_iters: 50,
        rtol: 1e-6,
        inner_max_iter: 3, // Small to force multiple outer iterations
        inner_max_restarts: 0,
        min_reduction: None,
        inner_rtol: Some(0.5), // Loose inner tolerance
        verbose: true,
    };

    let result = restart_gmres_with_truncation(&apply_a, &b, None, &options, &truncate_fn)?;

    // Compute true residual
    let ax = apply_a(&result.solution)?;
    let r = ax.axpby(AnyScalar::new_real(1.0), &b, AnyScalar::new_real(-1.0))?;
    let true_residual = r.norm()? / b_norm;

    println!("Converged: {}", result.converged);
    println!("Outer iterations: {}", result.outer_iterations);
    println!("Total inner iterations: {}", result.iterations);
    println!("True residual: {:.6e}", true_residual);
    println!("Bond dims: {:?}", result.solution.bond_dims());

    // Compute error ||x_sol - x_true||
    let diff =
        result
            .solution
            .axpby(AnyScalar::new_real(1.0), &x_true, AnyScalar::new_real(-1.0))?;
    println!("Error ||x - x_true||: {:.6e}", diff.norm()?);

    Ok((result.converged, result.outer_iterations, true_residual))
}

/// Test restart GMRES with imaginary diagonal MPO and imaginary ones MPS.
/// A = i * diag(2, 3), x_true = i * ones, b = A * x_true
/// Returns (converged, outer_iterations, true_residual)
fn test_restart_gmres_imaginary_diagonal(n: usize) -> anyhow::Result<(bool, usize, f64)> {
    let phys_dim = 2;
    println!("--- N={}, operator=i*diagonal, x_true=i*ones ---", n);

    let indices = SharedIndices::new(n, phys_dim);

    // Create A = i * diag(2, 3) MPO
    let mpo_real = create_diagonal_mpo(&indices)?;
    let mpo = mpo_real.scale(AnyScalar::new_complex(0.0, 1.0))?;
    let cond_number = (1.5_f64).powi(n as i32);
    println!(
        "Imaginary diagonal MPO created (i * diag(2,3)), condition number ≈ {:.2}",
        cond_number
    );

    // Create x_true = i * ones MPS
    let ones_real = create_ones_mps(&indices)?;
    let x_true = ones_real.scale(AnyScalar::new_complex(0.0, 1.0))?;
    println!("x_true (i * ones MPS) norm: {:.6}", x_true.norm()?);

    // Compute b = A * x_true
    let b = apply_mpo(&mpo, &x_true, &indices)?;
    let b_norm = b.norm()?;
    println!("b = A * x_true, norm: {:.6}", b_norm);

    // Aggressive truncation
    let truncate_opts = TruncateOptions::svd()
        .with_svd_policy(tensor4all_core::SvdTruncationPolicy::new(1e-6))
        .with_max_rank(5);
    let truncate_fn = |x: &mut TensorTrain| -> anyhow::Result<()> {
        x.truncate(&truncate_opts)?;
        Ok(())
    };

    let apply_a = |x: &TensorTrain| -> anyhow::Result<TensorTrain> { apply_mpo(&mpo, x, &indices) };

    println!(
        "Truncation: rtol={:.0e}, max_rank={}",
        truncate_opts
            .svd_policy()
            .map(|policy| policy.threshold)
            .unwrap_or(0.0),
        truncate_opts.max_rank().unwrap_or(0)
    );

    // Restart GMRES options - small inner iterations to force multiple outer iterations
    let options = RestartGmresOptions {
        max_outer_iters: 50,
        rtol: 1e-6,
        inner_max_iter: 3,
        inner_max_restarts: 0,
        min_reduction: None,
        inner_rtol: Some(0.5),
        verbose: true,
    };

    let result = restart_gmres_with_truncation(&apply_a, &b, None, &options, &truncate_fn)?;

    // Compute true residual
    let ax = apply_a(&result.solution)?;
    let r = ax.axpby(AnyScalar::new_real(1.0), &b, AnyScalar::new_real(-1.0))?;
    let true_residual = r.norm()? / b_norm;

    println!("Converged: {}", result.converged);
    println!("Outer iterations: {}", result.outer_iterations);
    println!("Total inner iterations: {}", result.iterations);
    println!("True residual: {:.6e}", true_residual);
    println!("Bond dims: {:?}", result.solution.bond_dims());

    // Compute error ||x_sol - x_true||
    // Note: solution may be F64 (real b after contraction) while x_true is C64
    match result
        .solution
        .axpby(AnyScalar::new_real(1.0), &x_true, AnyScalar::new_real(-1.0))
    {
        Ok(diff) => println!("Error ||x - x_true||: {:.6e}", diff.norm()?),
        Err(_) => {
            // F64/C64 storage mismatch: compute via norms
            let sol_norm = result.solution.norm()?;
            let xt_norm = x_true.norm()?;
            println!(
                "Error (norm comparison): ||x||={:.6e}, ||x_true||={:.6e}",
                sol_norm, xt_norm
            );
        }
    }

    Ok((result.converged, result.outer_iterations, true_residual))
}

/// Create a diagonal MPO: diag(2, 3) at each site
fn create_diagonal_mpo(indices: &SharedIndices) -> anyhow::Result<TensorTrain> {
    let n = indices.sites.len();
    let mut tensors = Vec::with_capacity(n);

    let mpo_bonds: Vec<DynIndex> = (0..n.saturating_sub(1))
        .map(|_| DynIndex::new_dyn(1))
        .collect();

    // Diagonal matrix: [[2, 0], [0, 3]]
    let diag_data = [2.0, 0.0, 0.0, 3.0];

    for i in 0..n {
        let s_in = indices.sites[i].clone();
        let s_out = indices.mpo_outputs[i].clone();

        if i == 0 && n == 1 {
            let tensor = TensorDynLen::from_dense(vec![s_in, s_out], diag_data.to_vec()).unwrap();
            tensors.push(tensor);
        } else if i == 0 {
            let right_bond = mpo_bonds[i].clone();
            let tensor =
                TensorDynLen::from_dense(vec![s_in, s_out, right_bond], diag_data.to_vec())
                    .unwrap();
            tensors.push(tensor);
        } else if i == n - 1 {
            let left_bond = mpo_bonds[i - 1].clone();
            let tensor =
                TensorDynLen::from_dense(vec![left_bond, s_in, s_out], diag_data.to_vec()).unwrap();
            tensors.push(tensor);
        } else {
            let left_bond = mpo_bonds[i - 1].clone();
            let right_bond = mpo_bonds[i].clone();
            let tensor = TensorDynLen::from_dense(
                vec![left_bond, s_in, s_out, right_bond],
                diag_data.to_vec(),
            )
            .unwrap();
            tensors.push(tensor);
        }
    }

    TensorTrain::new(tensors).map_err(|e| anyhow::anyhow!("{}", e))
}

/// Create a Pauli-X MPO operator.
/// Pauli-X matrix: [[0, 1], [1, 0]] (spin flip operator)
/// X^2 = I, so condition number = 1
fn create_pauli_x_mpo(indices: &SharedIndices) -> anyhow::Result<TensorTrain> {
    let n = indices.sites.len();
    let mut tensors = Vec::with_capacity(n);

    let mpo_bonds: Vec<DynIndex> = (0..n.saturating_sub(1))
        .map(|_| DynIndex::new_dyn(1))
        .collect();

    // Pauli-X matrix: [[0, 1], [1, 0]]
    let pauli_x = [0.0, 1.0, 1.0, 0.0];

    for i in 0..n {
        let s_in = indices.sites[i].clone();
        let s_out = indices.mpo_outputs[i].clone();

        if i == 0 && n == 1 {
            let tensor = TensorDynLen::from_dense(vec![s_in, s_out], pauli_x.to_vec()).unwrap();
            tensors.push(tensor);
        } else if i == 0 {
            let right_bond = mpo_bonds[i].clone();
            let tensor =
                TensorDynLen::from_dense(vec![s_in, s_out, right_bond], pauli_x.to_vec()).unwrap();
            tensors.push(tensor);
        } else if i == n - 1 {
            let left_bond = mpo_bonds[i - 1].clone();
            let tensor =
                TensorDynLen::from_dense(vec![left_bond, s_in, s_out], pauli_x.to_vec()).unwrap();
            tensors.push(tensor);
        } else {
            let left_bond = mpo_bonds[i - 1].clone();
            let right_bond = mpo_bonds[i].clone();
            let tensor = TensorDynLen::from_dense(
                vec![left_bond, s_in, s_out, right_bond],
                pauli_x.to_vec(),
            )
            .unwrap();
            tensors.push(tensor);
        }
    }

    TensorTrain::new(tensors).map_err(|e| anyhow::anyhow!("{}", e))
}

/// Create an MPS with all elements = 1.
fn create_ones_mps(indices: &SharedIndices) -> anyhow::Result<TensorTrain> {
    let n = indices.sites.len();
    let mut tensors = Vec::with_capacity(n);

    for i in 0..n {
        let site_dim = indices.sites[i].dim();
        let site_idx = indices.sites[i].clone();

        if i == 0 && n == 1 {
            let data = vec![1.0; site_dim];
            let tensor = TensorDynLen::from_dense(vec![site_idx], data).unwrap();
            tensors.push(tensor);
        } else if i == 0 {
            let right_bond = indices.bonds[i].clone();
            let data = vec![1.0; site_dim];
            let tensor = TensorDynLen::from_dense(vec![site_idx, right_bond], data).unwrap();
            tensors.push(tensor);
        } else if i == n - 1 {
            let left_bond = indices.bonds[i - 1].clone();
            let data = vec![1.0; site_dim];
            let tensor = TensorDynLen::from_dense(vec![left_bond, site_idx], data).unwrap();
            tensors.push(tensor);
        } else {
            let left_bond = indices.bonds[i - 1].clone();
            let right_bond = indices.bonds[i].clone();
            let data = vec![1.0; site_dim];
            let tensor =
                TensorDynLen::from_dense(vec![left_bond, site_idx, right_bond], data).unwrap();
            tensors.push(tensor);
        }
    }

    TensorTrain::new(tensors).map_err(|e| anyhow::anyhow!("{}", e))
}

/// Apply MPO to MPS and return result with same external indices as input MPS.
fn apply_mpo(
    mpo: &TensorTrain,
    mps: &TensorTrain,
    indices: &SharedIndices,
) -> anyhow::Result<TensorTrain> {
    let options = ContractOptions::fit()
        .with_nhalfsweeps(4)
        .with_svd_policy(tensor4all_core::SvdTruncationPolicy::new(1e-10))
        .with_max_rank(30);
    let result = mpo
        .contract(mps, &options)
        .map_err(|e| anyhow::anyhow!("{}", e))?;

    // Replace MPO output indices with site indices
    let result = result.replaceinds(&indices.mpo_outputs, &indices.sites)?;

    Ok(result)
}
