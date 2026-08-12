#![allow(clippy::needless_borrows_for_generic_args)]
#![allow(clippy::identity_op)]
#![allow(clippy::type_complexity)]
//! Test: Restart GMRES solver with MPO format for x and b
//!
//! Unlike test_restart_gmres_mps.rs where x is an MPS (vector),
//! here x and b are MPOs (operators) with all elements = 1.
//! A is a superoperator that acts on MPOs.
//!
//! Run:
//!   cargo run -p tensor4all-itensorlike --example test_restart_gmres_mpo --release

use tensor4all_core::krylov::{restart_gmres_with_truncation, RestartGmresOptions};
use tensor4all_core::{AnyScalar, DynIndex, IdxTensor, IndexLike, TensorIndex};
use tensor4all_itensorlike::{ContractOptions, TensorTrain, TruncateOptions};

/// Shared indices for all MPO operations.
/// For MPO, each site has input (row) and output (column) indices.
struct SharedIndices {
    /// Input (row) indices of the MPO
    inputs: Vec<DynIndex>,
    /// Output (column) indices of the MPO
    outputs: Vec<DynIndex>,
    /// Bond indices between MPO sites
    bonds: Vec<DynIndex>,
    /// Operator output indices (for the superoperator A)
    operator_outputs: Vec<DynIndex>,
}

impl SharedIndices {
    fn new(n: usize, phys_dim: usize) -> Self {
        let inputs: Vec<DynIndex> = (0..n).map(|_| DynIndex::new_dyn(phys_dim)).collect();
        let outputs: Vec<DynIndex> = (0..n).map(|_| DynIndex::new_dyn(phys_dim)).collect();
        let bonds: Vec<DynIndex> = (0..n.saturating_sub(1))
            .map(|_| DynIndex::new_dyn(1))
            .collect();
        let operator_outputs: Vec<DynIndex> = (0..n).map(|_| DynIndex::new_dyn(phys_dim)).collect();
        Self {
            inputs,
            outputs,
            bonds,
            operator_outputs,
        }
    }
}

fn main() -> anyhow::Result<()> {
    println!("========================================");
    println!("  Restart GMRES with MPO Tests (x = all-ones MPO)");
    println!("========================================\n");

    // Test 1: Identity superoperator A(X) = X, N=3
    println!("========================================");
    println!("  Identity Superoperator Tests (A(X) = X)");
    println!("========================================\n");

    let result = test_restart_gmres_mpo(3, "identity")?;
    println!(
        "N=3: converged={}, outer_iters={}, true_residual={:.2e}\n",
        result.0, result.1, result.2
    );
    assert!(result.0, "Identity superoperator should converge for N=3");

    // Test 2: Pauli-X superoperator A(X) = σ_x * X, N=3
    println!("========================================");
    println!("  Pauli-X Superoperator Tests (A(X) = σ_x * X)");
    println!("========================================\n");

    let result = test_restart_gmres_mpo(3, "pauli_x")?;
    println!(
        "N=3: converged={}, outer_iters={}, true_residual={:.2e}\n",
        result.0, result.1, result.2
    );
    assert!(result.0, "Pauli-X superoperator should converge for N=3");

    // Test 3: Diagonal superoperator with aggressive truncation (harder problem)
    println!("\n========================================");
    println!("  Diagonal Superoperator Tests (harder problem)");
    println!("========================================\n");

    let result = test_restart_gmres_mpo_hard(3)?;
    println!(
        "N=3 Diagonal: converged={}, outer_iters={}, true_residual={:.2e}\n",
        result.0, result.1, result.2
    );

    // Test 4: Imaginary diagonal superoperator with imaginary all-ones MPO
    println!("\n========================================");
    println!("  Imaginary Diagonal Superoperator Tests (A = i*diag, x = i*ones)");
    println!("========================================\n");

    let result = test_restart_gmres_mpo_imaginary_diagonal(3)?;
    println!(
        "N=3 Imaginary Diagonal: converged={}, outer_iters={}, true_residual={:.2e}\n",
        result.0, result.1, result.2
    );

    println!("\n========================================");
    println!("  All tests completed!");
    println!("========================================");

    Ok(())
}

/// Test restart GMRES with specified superoperator, x_true = all-ones MPO.
/// Returns (converged, outer_iterations, true_residual).
fn test_restart_gmres_mpo(n: usize, operator: &str) -> anyhow::Result<(bool, usize, f64)> {
    let phys_dim = 2;
    println!("--- N={}, operator={} ---", n, operator);

    let indices = SharedIndices::new(n, phys_dim);

    // Create x_true: all-ones MPO
    let x_true = create_ones_mpo(&indices)?;
    println!("x_true (all-ones MPO) norm: {:.6}", x_true.norm()?);

    // Define the superoperator and compute b = A(x_true)
    let (apply_a, b): (
        Box<dyn Fn(&TensorTrain) -> anyhow::Result<TensorTrain>>,
        TensorTrain,
    ) = match operator {
        "identity" => {
            // A(X) = X
            let b = x_true.clone();
            let apply_a =
                Box::new(|x: &TensorTrain| -> anyhow::Result<TensorTrain> { Ok(x.clone()) });
            (apply_a, b)
        }
        "pauli_x" => {
            // A(X) = σ_x * X
            let pauli_x_op = create_pauli_x_operator_mpo(&indices)?;
            let b = apply_operator_to_mpo(&pauli_x_op, &x_true, &indices)?;
            let apply_a = Box::new(move |x: &TensorTrain| -> anyhow::Result<TensorTrain> {
                apply_operator_to_mpo(&pauli_x_op, x, &indices)
            });
            (apply_a, b)
        }
        _ => anyhow::bail!("Unknown operator: {}", operator),
    };

    let b_norm = b.norm()?;
    println!("b = A(x_true), norm: {:.6}", b_norm);

    // Truncation options
    let truncate_opts = TruncateOptions::svd()
        .with_svd_policy(tensor4all_core::SvdTruncationPolicy::new(1e-10))
        .with_max_bond_dim(30);
    let truncate_fn = |x: &mut TensorTrain| -> anyhow::Result<()> {
        x.truncate(&truncate_opts)?;
        Ok(())
    };

    // Restart GMRES options
    let options = RestartGmresOptions {
        max_outer_iters: 30,
        rtol: 1e-8,
        inner_max_iter: 10,
        inner_max_restarts: 0,
        min_reduction: None,
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

/// Test restart GMRES with diagonal superoperator + aggressive truncation.
/// Forces multiple outer iterations by using small inner_max_iter.
/// Returns (converged, outer_iterations, true_residual).
fn test_restart_gmres_mpo_hard(n: usize) -> anyhow::Result<(bool, usize, f64)> {
    let phys_dim = 2;
    println!("--- N={}, operator=diagonal (harder problem) ---", n);

    let indices = SharedIndices::new(n, phys_dim);

    // Create diagonal operator: diag(2, 3) at each site
    // Condition number = (3/2)^N
    let diag_op = create_diagonal_operator_mpo(&indices)?;
    let cond_number = (1.5_f64).powi(n as i32);
    println!(
        "Diagonal operator MPO created, condition number ≈ {:.2}",
        cond_number
    );

    // Create x_true: all-ones MPO
    let x_true = create_ones_mpo(&indices)?;
    println!("x_true (all-ones MPO) norm: {:.6}", x_true.norm()?);

    // Compute b = A(x_true)
    let b = apply_operator_to_mpo(&diag_op, &x_true, &indices)?;
    let b_norm = b.norm()?;
    println!("b = A(x_true), norm: {:.6}", b_norm);

    // AGGRESSIVE truncation to make the problem harder
    let truncate_opts = TruncateOptions::svd()
        .with_svd_policy(tensor4all_core::SvdTruncationPolicy::new(1e-6))
        .with_max_bond_dim(5);
    let truncate_fn = |x: &mut TensorTrain| -> anyhow::Result<()> {
        x.truncate(&truncate_opts)?;
        Ok(())
    };

    println!(
        "Truncation: rtol={:.0e}, max_bond_dim={}",
        truncate_opts
            .svd_policy()
            .map(|policy| policy.threshold)
            .unwrap_or(0.0),
        truncate_opts.max_bond_dim().unwrap_or(0)
    );

    // Define apply_a closure
    let apply_a = |x: &TensorTrain| -> anyhow::Result<TensorTrain> {
        apply_operator_to_mpo(&diag_op, x, &indices)
    };

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
    let diff =
        result
            .solution
            .axpby(AnyScalar::new_real(1.0), &x_true, AnyScalar::new_real(-1.0))?;
    println!("Error ||x - x_true||: {:.6e}", diff.norm()?);

    Ok((result.converged, result.outer_iterations, true_residual))
}

/// Test restart GMRES with imaginary diagonal superoperator and imaginary all-ones MPO.
/// A = i * diag(2, 3) superoperator, x_true = i * all-ones MPO, b = A(x_true)
/// Returns (converged, outer_iterations, true_residual).
fn test_restart_gmres_mpo_imaginary_diagonal(n: usize) -> anyhow::Result<(bool, usize, f64)> {
    let phys_dim = 2;
    println!("--- N={}, operator=i*diagonal, x_true=i*ones MPO ---", n);

    let indices = SharedIndices::new(n, phys_dim);

    // Create A = i * diag(2, 3) superoperator
    let diag_op_real = create_diagonal_operator_mpo(&indices)?;
    let diag_op = diag_op_real.scale(AnyScalar::new_complex(0.0, 1.0))?;
    let cond_number = (1.5_f64).powi(n as i32);
    println!(
        "Imaginary diagonal operator created (i * diag(2,3)), condition number ≈ {:.2}",
        cond_number
    );

    // Create x_true = i * all-ones MPO
    let ones_real = create_ones_mpo(&indices)?;
    let x_true = ones_real.scale(AnyScalar::new_complex(0.0, 1.0))?;
    println!("x_true (i * all-ones MPO) norm: {:.6}", x_true.norm()?);

    // Compute b = A(x_true)
    let b = apply_operator_to_mpo(&diag_op, &x_true, &indices)?;
    let b_norm = b.norm()?;
    println!("b = A(x_true), norm: {:.6}", b_norm);

    // Aggressive truncation
    let truncate_opts = TruncateOptions::svd()
        .with_svd_policy(tensor4all_core::SvdTruncationPolicy::new(1e-6))
        .with_max_bond_dim(5);
    let truncate_fn = |x: &mut TensorTrain| -> anyhow::Result<()> {
        x.truncate(&truncate_opts)?;
        Ok(())
    };

    println!(
        "Truncation: rtol={:.0e}, max_bond_dim={}",
        truncate_opts
            .svd_policy()
            .map(|policy| policy.threshold)
            .unwrap_or(0.0),
        truncate_opts.max_bond_dim().unwrap_or(0)
    );

    // Define apply_a closure
    let apply_a = |x: &TensorTrain| -> anyhow::Result<TensorTrain> {
        apply_operator_to_mpo(&diag_op, x, &indices)
    };

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

// ============================================================================
// Helper functions for creating MPOs
// ============================================================================

/// Create an all-ones MPO: every element = 1.
/// Each site tensor has entries T[in, out] = 1 for all (in, out).
fn create_ones_mpo(indices: &SharedIndices) -> anyhow::Result<TensorTrain> {
    let n = indices.inputs.len();
    let mut tensors = Vec::with_capacity(n);

    for i in 0..n {
        let in_dim = indices.inputs[i].dim();
        let out_dim = indices.outputs[i].dim();
        let in_idx = indices.inputs[i].clone();
        let out_idx = indices.outputs[i].clone();

        let data = vec![1.0_f64; in_dim * out_dim];

        if i == 0 && n == 1 {
            let tensor = IdxTensor::from_dense(vec![in_idx, out_idx], data).unwrap();
            tensors.push(tensor);
        } else if i == 0 {
            let right_bond = indices.bonds[i].clone();
            let tensor = IdxTensor::from_dense(vec![in_idx, out_idx, right_bond], data).unwrap();
            tensors.push(tensor);
        } else if i == n - 1 {
            let left_bond = indices.bonds[i - 1].clone();
            let tensor = IdxTensor::from_dense(vec![left_bond, in_idx, out_idx], data).unwrap();
            tensors.push(tensor);
        } else {
            let left_bond = indices.bonds[i - 1].clone();
            let right_bond = indices.bonds[i].clone();
            let tensor =
                IdxTensor::from_dense(vec![left_bond, in_idx, out_idx, right_bond], data).unwrap();
            tensors.push(tensor);
        }
    }

    TensorTrain::new(tensors).map_err(|e| anyhow::anyhow!("{}", e))
}

/// Create a Pauli-X operator MPO.
/// Acts on the input indices of an MPO: σ_x[in, out']
fn create_pauli_x_operator_mpo(indices: &SharedIndices) -> anyhow::Result<TensorTrain> {
    let n = indices.inputs.len();
    let mut tensors = Vec::with_capacity(n);

    let op_bonds: Vec<DynIndex> = (0..n.saturating_sub(1))
        .map(|_| DynIndex::new_dyn(1))
        .collect();

    // Pauli-X: [[0, 1], [1, 0]]
    let pauli_x = [0.0, 1.0, 1.0, 0.0];

    for i in 0..n {
        let in_idx = indices.inputs[i].clone();
        let op_out_idx = indices.operator_outputs[i].clone();

        if i == 0 && n == 1 {
            let tensor = IdxTensor::from_dense(vec![in_idx, op_out_idx], pauli_x.to_vec()).unwrap();
            tensors.push(tensor);
        } else if i == 0 {
            let right_bond = op_bonds[i].clone();
            let tensor =
                IdxTensor::from_dense(vec![in_idx, op_out_idx, right_bond], pauli_x.to_vec())
                    .unwrap();
            tensors.push(tensor);
        } else if i == n - 1 {
            let left_bond = op_bonds[i - 1].clone();
            let tensor =
                IdxTensor::from_dense(vec![left_bond, in_idx, op_out_idx], pauli_x.to_vec())
                    .unwrap();
            tensors.push(tensor);
        } else {
            let left_bond = op_bonds[i - 1].clone();
            let right_bond = op_bonds[i].clone();
            let tensor = IdxTensor::from_dense(
                vec![left_bond, in_idx, op_out_idx, right_bond],
                pauli_x.to_vec(),
            )
            .unwrap();
            tensors.push(tensor);
        }
    }

    TensorTrain::new(tensors).map_err(|e| anyhow::anyhow!("{}", e))
}

/// Create a diagonal operator MPO: diag(2, 3) at each site.
/// Condition number = (3/2)^N.
fn create_diagonal_operator_mpo(indices: &SharedIndices) -> anyhow::Result<TensorTrain> {
    let n = indices.inputs.len();
    let mut tensors = Vec::with_capacity(n);

    let op_bonds: Vec<DynIndex> = (0..n.saturating_sub(1))
        .map(|_| DynIndex::new_dyn(1))
        .collect();

    // Diagonal matrix: [[2, 0], [0, 3]]
    let diag_data = [2.0, 0.0, 0.0, 3.0];

    for i in 0..n {
        let in_idx = indices.inputs[i].clone();
        let op_out_idx = indices.operator_outputs[i].clone();

        if i == 0 && n == 1 {
            let tensor =
                IdxTensor::from_dense(vec![in_idx, op_out_idx], diag_data.to_vec()).unwrap();
            tensors.push(tensor);
        } else if i == 0 {
            let right_bond = op_bonds[i].clone();
            let tensor =
                IdxTensor::from_dense(vec![in_idx, op_out_idx, right_bond], diag_data.to_vec())
                    .unwrap();
            tensors.push(tensor);
        } else if i == n - 1 {
            let left_bond = op_bonds[i - 1].clone();
            let tensor =
                IdxTensor::from_dense(vec![left_bond, in_idx, op_out_idx], diag_data.to_vec())
                    .unwrap();
            tensors.push(tensor);
        } else {
            let left_bond = op_bonds[i - 1].clone();
            let right_bond = op_bonds[i].clone();
            let tensor = IdxTensor::from_dense(
                vec![left_bond, in_idx, op_out_idx, right_bond],
                diag_data.to_vec(),
            )
            .unwrap();
            tensors.push(tensor);
        }
    }

    TensorTrain::new(tensors).map_err(|e| anyhow::anyhow!("{}", e))
}

/// Apply an operator MPO to an MPO state: result = O * X
/// O has indices [in, out'] and X has indices [in, out]
/// Result has indices [out', out] -> relabeled to [in, out]
fn apply_operator_to_mpo(
    op: &TensorTrain,
    mpo: &TensorTrain,
    indices: &SharedIndices,
) -> anyhow::Result<TensorTrain> {
    let options = ContractOptions::fit()
        .with_nhalfsweeps(4)
        .with_svd_policy(tensor4all_core::SvdTruncationPolicy::new(1e-10))
        .with_max_bond_dim(30);

    let result = op
        .contract(mpo, &options)
        .map_err(|e| anyhow::anyhow!("{}", e))?;

    // Replace operator output indices with input indices
    let result = result.replace_indices(&indices.operator_outputs, &indices.inputs)?;

    Ok(result)
}
