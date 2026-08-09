#![allow(clippy::needless_borrows_for_generic_args)]
#![allow(clippy::identity_op)]
#![allow(clippy::type_complexity)]
//! Test GMRES solver with MPO (Matrix Product Operator) format for x and b.
//!
//! Unlike test_gmres_mps.rs where x and b are MPS (vectors), here x and b are MPOs (operators).
//! A is a superoperator that acts on MPOs: A(X) returns an MPO.
//!
//! Run:
//!   cargo run -p tensor4all-itensorlike --example test_gmres_mpo --release

use num_complex::Complex64;
use rand::{rngs::StdRng, Rng, SeedableRng};
use tensor4all_core::krylov::{gmres_with_truncation, GmresOptions};
use tensor4all_core::{AnyScalar, DynIndex, IndexLike, TensorDynLen, TensorIndex};
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
    // Test 1: Identity operator (A = I, where I(X) = X)
    println!("========================================");
    println!("  Identity Operator Tests (A = I)");
    println!("========================================\n");

    let mut identity_results = Vec::new();
    for n in [3, 5] {
        let result = test_gmres_mpo_identity(n)?;
        identity_results.push((n, result));
        println!();
    }

    // Summary for Identity
    println!("========================================");
    println!("  Identity Summary");
    println!("========================================");
    println!(
        "{:>4} | {:>12} | {:>12} | {:>12} | {:>6}",
        "N", "Initial Res", "Final Res", "Reduction", "Iters"
    );
    println!(
        "{:-<4}-+-{:-<12}-+-{:-<12}-+-{:-<12}-+-{:-<6}",
        "", "", "", "", ""
    );
    for (n, (init_res, final_res, iters)) in &identity_results {
        let reduction = init_res / final_res.max(1e-16);
        println!(
            "{:>4} | {:>12.2e} | {:>12.2e} | {:>12.2e} | {:>6}",
            n, init_res, final_res, reduction, iters
        );
    }

    // Test 2: Pauli-X operator
    println!("\n========================================");
    println!("  Pauli-X Operator Tests");
    println!("========================================\n");

    let mut pauli_results = Vec::new();
    for n in [3, 5] {
        let result = test_gmres_mpo_pauli(n)?;
        pauli_results.push((n, result));
        println!();
    }

    // Summary for Pauli-X
    println!("========================================");
    println!("  Pauli-X Summary");
    println!("========================================");
    println!(
        "{:>4} | {:>12} | {:>12} | {:>12} | {:>6}",
        "N", "Initial Res", "Final Res", "Reduction", "Iters"
    );
    println!(
        "{:-<4}-+-{:-<12}-+-{:-<12}-+-{:-<12}-+-{:-<6}",
        "", "", "", "", ""
    );
    for (n, (init_res, final_res, iters)) in &pauli_results {
        let reduction = init_res / final_res.max(1e-16);
        println!(
            "{:>4} | {:>12.2e} | {:>12.2e} | {:>12.2e} | {:>6}",
            n, init_res, final_res, reduction, iters
        );
    }

    // Test 3: Pure imaginary identity (A = i*I)
    println!("\n========================================");
    println!("  Pure Imaginary Identity Tests (A = i*I)");
    println!("========================================\n");

    let mut imaginary_results = Vec::new();
    for n in [3, 5] {
        let result = test_gmres_mpo_imaginary(n)?;
        imaginary_results.push((n, result));
        println!();
    }

    // Summary for Pure Imaginary Identity
    println!("========================================");
    println!("  Pure Imaginary Identity Summary");
    println!("========================================");
    println!(
        "{:>4} | {:>12} | {:>12} | {:>12} | {:>6}",
        "N", "Initial Res", "Final Res", "Reduction", "Iters"
    );
    println!(
        "{:-<4}-+-{:-<12}-+-{:-<12}-+-{:-<12}-+-{:-<6}",
        "", "", "", "", ""
    );
    for (n, (init_res, final_res, iters)) in &imaginary_results {
        let reduction = init_res / final_res.max(1e-16);
        println!(
            "{:>4} | {:>12.2e} | {:>12.2e} | {:>12.2e} | {:>6}",
            n, init_res, final_res, reduction, iters
        );
    }

    // Test 4: Random superoperator
    println!("\n========================================");
    println!("  Random Superoperator Tests");
    println!("========================================\n");

    let mut random_results = Vec::new();
    for n in [3, 5] {
        let result = test_gmres_mpo_random(n, 20)?;
        random_results.push((n, result));
        println!();
    }

    // Summary for Random
    println!("========================================");
    println!("  Random Superoperator Summary");
    println!("========================================");
    println!(
        "{:>4} | {:>12} | {:>12} | {:>12} | {:>6}",
        "N", "Initial Res", "Final Res", "Reduction", "Iters"
    );
    println!(
        "{:-<4}-+-{:-<12}-+-{:-<12}-+-{:-<12}-+-{:-<6}",
        "", "", "", "", ""
    );
    for (n, (init_res, final_res, iters)) in &random_results {
        let reduction = init_res / final_res.max(1e-16);
        println!(
            "{:>4} | {:>12.2e} | {:>12.2e} | {:>12.2e} | {:>6}",
            n, init_res, final_res, reduction, iters
        );
    }

    Ok(())
}

/// Test GMRES with identity superoperator: A(X) = X
/// This is the simplest case where A just returns its input.
fn test_gmres_mpo_identity(n: usize) -> anyhow::Result<(f64, f64, usize)> {
    let phys_dim = 2;

    println!("=== Test: GMRES with MPO (Identity Superoperator) ===");
    println!("N = {}, phys_dim = {}", n, phys_dim);
    println!("A(X) = X (identity superoperator)");

    let indices = SharedIndices::new(n, phys_dim);

    // Create x_true = identity MPO
    let x_true = create_identity_mpo(&indices)?;
    println!(
        "x_true (identity MPO) created with {} sites, norm: {:.6}",
        x_true.len(),
        x_true.norm()?
    );

    // For identity superoperator, b = A(x_true) = x_true
    let b = x_true.clone();
    println!("b = A(x_true) = x_true, norm: {:.6}", b.norm()?);

    // Initial guess: 0.5 * b
    let x0 = b.scale(AnyScalar::new_real(0.5))?;
    println!(
        "x0 (initial guess = 0.5*b) created, norm: {:.6}",
        x0.norm()?
    );

    // Define apply_a closure: A(X) = X (identity)
    let apply_a = |x: &TensorTrain| -> anyhow::Result<TensorTrain> { Ok(x.clone()) };

    // Compute initial residual
    let b_norm = b.norm()?;
    let ax0 = apply_a(&x0)?;
    let r0 = ax0.axpby(AnyScalar::new_real(1.0), &b, AnyScalar::new_real(-1.0))?;
    let initial_residual = r0.norm()? / b_norm;
    println!("\n=== Initial Residual ===");
    println!("|Ax0 - b| / |b| = {:.6e}", initial_residual);

    // Solve with GMRES
    let options = GmresOptions {
        max_iter: 10,
        rtol: 1e-8,
        max_restarts: 1,
        verbose: true,
        check_true_residual: false,
    };

    let truncate_opts = TruncateOptions::svd()
        .with_svd_policy(tensor4all_core::SvdTruncationPolicy::new(1e-8))
        .with_max_rank(20);
    let truncate_fn = |x: &mut TensorTrain| -> anyhow::Result<()> {
        x.truncate(&truncate_opts)?;
        Ok(())
    };

    println!(
        "\n=== Running GMRES with truncation (max_iter={}) ===",
        options.max_iter
    );
    println!("rtol = {:.2e}", options.rtol);
    let result = gmres_with_truncation(&apply_a, &b, &x0, &options, truncate_fn)?;

    // Compute final residual
    let ax_sol = apply_a(&result.solution)?;
    let r_final = ax_sol.axpby(AnyScalar::new_real(1.0), &b, AnyScalar::new_real(-1.0))?;
    let final_residual = r_final.norm()? / b_norm;

    println!("\n=== Results ===");
    println!("Converged: {}", result.converged);
    println!("Iterations: {}", result.iterations);
    println!("GMRES reported residual: {:.6e}", result.residual_norm);
    println!("Final |Ax - b| / |b|:    {:.6e}", final_residual);

    // Compute error
    let diff =
        result
            .solution
            .axpby(AnyScalar::new_real(1.0), &x_true, AnyScalar::new_real(-1.0))?;
    let error = diff.norm()?;
    println!("Error ||x_sol - x_true||: {:.6e}", error);
    println!("Solution bond dims: {:?}", result.solution.bond_dims());

    println!("\n=== Done ===");
    Ok((initial_residual, final_residual, result.iterations))
}

/// Test GMRES with Pauli-X superoperator: A(X) = σ_x * X
/// Applies Pauli-X to the input (row) indices of X.
fn test_gmres_mpo_pauli(n: usize) -> anyhow::Result<(f64, f64, usize)> {
    let phys_dim = 2;

    println!("=== Test: GMRES with MPO (Pauli-X Superoperator) ===");
    println!("N = {}, phys_dim = {}", n, phys_dim);
    println!("A(X) = σ_x * X (left multiplication by Pauli-X)");

    let indices = SharedIndices::new(n, phys_dim);

    // Create Pauli-X operator (as TensorTrain MPO)
    let pauli_x_op = create_pauli_x_operator_mpo(&indices)?;
    println!(
        "Pauli-X operator MPO created with {} sites",
        pauli_x_op.len()
    );

    // Create x_true = identity MPO
    let x_true = create_identity_mpo(&indices)?;
    println!(
        "x_true (identity MPO) created with {} sites, norm: {:.6}",
        x_true.len(),
        x_true.norm()?
    );

    // b = A(x_true) = σ_x * I = σ_x
    let b = apply_operator_to_mpo(&pauli_x_op, &x_true, &indices)?;
    println!("b = A(x_true) computed, norm: {:.6}", b.norm()?);

    // Initial guess: 0.5 * b
    let x0 = b.scale(AnyScalar::new_real(0.5))?;
    println!(
        "x0 (initial guess = 0.5*b) created, norm: {:.6}",
        x0.norm()?
    );

    // Define apply_a closure
    let apply_a = |x: &TensorTrain| -> anyhow::Result<TensorTrain> {
        apply_operator_to_mpo(&pauli_x_op, x, &indices)
    };

    // Compute initial residual
    let b_norm = b.norm()?;
    let ax0 = apply_a(&x0)?;
    let r0 = ax0.axpby(AnyScalar::new_real(1.0), &b, AnyScalar::new_real(-1.0))?;
    let initial_residual = r0.norm()? / b_norm;
    println!("\n=== Initial Residual ===");
    println!("|Ax0 - b| / |b| = {:.6e}", initial_residual);

    // Solve with GMRES
    let options = GmresOptions {
        max_iter: 10,
        rtol: 1e-8,
        max_restarts: 1,
        verbose: true,
        check_true_residual: false,
    };

    let truncate_opts = TruncateOptions::svd()
        .with_svd_policy(tensor4all_core::SvdTruncationPolicy::new(1e-8))
        .with_max_rank(20);
    let truncate_fn = |x: &mut TensorTrain| -> anyhow::Result<()> {
        x.truncate(&truncate_opts)?;
        Ok(())
    };

    println!(
        "\n=== Running GMRES with truncation (max_iter={}) ===",
        options.max_iter
    );
    println!("rtol = {:.2e}", options.rtol);
    let result = gmres_with_truncation(&apply_a, &b, &x0, &options, truncate_fn)?;

    // Compute final residual
    let ax_sol = apply_a(&result.solution)?;
    let r_final = ax_sol.axpby(AnyScalar::new_real(1.0), &b, AnyScalar::new_real(-1.0))?;
    let final_residual = r_final.norm()? / b_norm;

    println!("\n=== Results ===");
    println!("Converged: {}", result.converged);
    println!("Iterations: {}", result.iterations);
    println!("GMRES reported residual: {:.6e}", result.residual_norm);
    println!("Final |Ax - b| / |b|:    {:.6e}", final_residual);

    // Compute error
    let diff =
        result
            .solution
            .axpby(AnyScalar::new_real(1.0), &x_true, AnyScalar::new_real(-1.0))?;
    let error = diff.norm()?;
    println!("Error ||x_sol - x_true||: {:.6e}", error);
    println!("Solution bond dims: {:?}", result.solution.bond_dims());

    println!("\n=== Done ===");
    Ok((initial_residual, final_residual, result.iterations))
}

/// Test GMRES with pure imaginary identity superoperator: A(X) = i * X
fn test_gmres_mpo_imaginary(n: usize) -> anyhow::Result<(f64, f64, usize)> {
    let phys_dim = 2;

    println!("=== Test: GMRES with MPO (Pure Imaginary Identity) ===");
    println!("N = {}, phys_dim = {}", n, phys_dim);
    println!("A(X) = i * X");

    let indices = SharedIndices::new(n, phys_dim);

    // Create x_true = i * identity MPO
    let x_true = create_imaginary_identity_mpo(&indices)?;
    println!(
        "x_true (i * identity MPO) created with {} sites, norm: {:.6}",
        x_true.len(),
        x_true.norm()?
    );

    // b = A(x_true) = i * (i * I) = -I
    // Apply i to x_true
    let b = x_true.scale(AnyScalar::from(Complex64::new(0.0, 1.0)))?;
    println!("b = i * x_true computed, norm: {:.6}", b.norm()?);

    // Initial guess: b
    let x0 = b.clone();
    println!("x0 (initial guess = b) created, norm: {:.6}", x0.norm()?);

    // Define apply_a closure: A(X) = i * X
    let apply_a = |x: &TensorTrain| -> anyhow::Result<TensorTrain> {
        Ok(x.scale(AnyScalar::from(Complex64::new(0.0, 1.0)))?)
    };

    // Compute initial residual
    let b_norm = b.norm()?;
    let ax0 = apply_a(&x0)?;
    let r0 = ax0.axpby(AnyScalar::new_real(1.0), &b, AnyScalar::new_real(-1.0))?;
    let initial_residual = r0.norm()? / b_norm;
    println!("\n=== Initial Residual ===");
    println!("|Ax0 - b| / |b| = {:.6e}", initial_residual);

    // Solve with GMRES
    let options = GmresOptions {
        max_iter: 5,
        rtol: 1e-8,
        max_restarts: 1,
        verbose: true,
        check_true_residual: false,
    };

    let truncate_opts = TruncateOptions::svd()
        .with_svd_policy(tensor4all_core::SvdTruncationPolicy::new(1e-8))
        .with_max_rank(20);
    let truncate_fn = |x: &mut TensorTrain| -> anyhow::Result<()> {
        x.truncate(&truncate_opts)?;
        Ok(())
    };

    println!(
        "\n=== Running GMRES with truncation (max_iter={}) ===",
        options.max_iter
    );
    println!("rtol = {:.2e}", options.rtol);
    let result = gmres_with_truncation(&apply_a, &b, &x0, &options, truncate_fn)?;

    // Compute final residual
    let ax_sol = apply_a(&result.solution)?;
    let r_final = ax_sol.axpby(AnyScalar::new_real(1.0), &b, AnyScalar::new_real(-1.0))?;
    let final_residual = r_final.norm()? / b_norm;

    println!("\n=== Results ===");
    println!("Converged: {}", result.converged);
    println!("Iterations: {}", result.iterations);
    println!("GMRES reported residual: {:.6e}", result.residual_norm);
    println!("Final |Ax - b| / |b|:    {:.6e}", final_residual);

    // Compute error
    let diff =
        result
            .solution
            .axpby(AnyScalar::new_real(1.0), &x_true, AnyScalar::new_real(-1.0))?;
    let error = diff.norm()?;
    println!("Error ||x_sol - x_true||: {:.6e}", error);
    println!("Solution bond dims: {:?}", result.solution.bond_dims());

    println!("\n=== Done ===");
    Ok((initial_residual, final_residual, result.iterations))
}

/// Test GMRES with random superoperator.
/// A(X) = O * X where O is a random diagonally dominant MPO.
fn test_gmres_mpo_random(n: usize, max_iter: usize) -> anyhow::Result<(f64, f64, usize)> {
    let phys_dim = 2;

    println!("=== Test: GMRES with MPO (Random Superoperator) ===");
    println!(
        "N = {}, phys_dim = {}, max_iter = {}",
        n, phys_dim, max_iter
    );

    let indices = SharedIndices::new(n, phys_dim);

    // Create random operator MPO
    let seed = 42u64;
    let random_op = create_random_operator_mpo(&indices, seed)?;
    println!("Random operator MPO created with {} sites", random_op.len());

    // Create x_true = identity MPO
    let x_true = create_identity_mpo(&indices)?;
    println!(
        "x_true (identity MPO) created with {} sites, norm: {:.6}",
        x_true.len(),
        x_true.norm()?
    );

    // b = A(x_true) = O * I
    let b = apply_operator_to_mpo(&random_op, &x_true, &indices)?;
    println!("b = A(x_true) computed, norm: {:.6}", b.norm()?);

    // Initial guess: b
    let x0 = b.clone();
    println!("x0 (initial guess = b) created, norm: {:.6}", x0.norm()?);

    // Define apply_a closure
    let apply_a = |x: &TensorTrain| -> anyhow::Result<TensorTrain> {
        apply_operator_to_mpo(&random_op, x, &indices)
    };

    // Compute initial residual
    let b_norm = b.norm()?;
    let ax0 = apply_a(&x0)?;
    let r0 = ax0.axpby(AnyScalar::new_real(1.0), &b, AnyScalar::new_real(-1.0))?;
    let initial_residual = r0.norm()? / b_norm;
    println!("\n=== Initial Residual ===");
    println!("|Ax0 - b| / |b| = {:.6e}", initial_residual);

    // Solve with GMRES
    let options = GmresOptions {
        max_iter,
        rtol: 1e-8,
        max_restarts: 5,
        verbose: true,
        check_true_residual: true,
    };

    let truncate_opts = TruncateOptions::svd()
        .with_svd_policy(tensor4all_core::SvdTruncationPolicy::new(1e-8))
        .with_max_rank(50);
    let truncate_fn = |x: &mut TensorTrain| -> anyhow::Result<()> {
        x.truncate(&truncate_opts)?;
        Ok(())
    };

    println!(
        "\n=== Running GMRES with truncation (max_iter={}) ===",
        options.max_iter
    );
    println!("rtol = {:.2e}", options.rtol);
    let result = gmres_with_truncation(&apply_a, &b, &x0, &options, truncate_fn)?;

    // Compute final residual
    let ax_sol = apply_a(&result.solution)?;
    let r_final = ax_sol.axpby(AnyScalar::new_real(1.0), &b, AnyScalar::new_real(-1.0))?;
    let final_residual = r_final.norm()? / b_norm;

    println!("\n=== Results ===");
    println!("Converged: {}", result.converged);
    println!("Iterations: {}", result.iterations);
    println!("GMRES reported residual: {:.6e}", result.residual_norm);
    println!("Final |Ax - b| / |b|:    {:.6e}", final_residual);

    // Compute error
    let diff =
        result
            .solution
            .axpby(AnyScalar::new_real(1.0), &x_true, AnyScalar::new_real(-1.0))?;
    let error = diff.norm()?;
    println!("Error ||x_sol - x_true||: {:.6e}", error);
    println!("Solution bond dims: {:?}", result.solution.bond_dims());

    println!("\n=== Done ===");
    Ok((initial_residual, final_residual, result.iterations))
}

// ============================================================================
// Helper functions for creating MPOs
// ============================================================================

/// Create an identity MPO: I[in, out] = δ(in, out)
fn create_identity_mpo(indices: &SharedIndices) -> anyhow::Result<TensorTrain> {
    let n = indices.inputs.len();
    let mut tensors = Vec::with_capacity(n);

    for i in 0..n {
        let in_dim = indices.inputs[i].dim();
        let out_dim = indices.outputs[i].dim();
        let in_idx = indices.inputs[i].clone();
        let out_idx = indices.outputs[i].clone();

        // Identity: δ(in, out)
        let mut data = vec![0.0_f64; in_dim * out_dim];
        for j in 0..in_dim.min(out_dim) {
            data[j * out_dim + j] = 1.0;
        }

        if i == 0 && n == 1 {
            let tensor = TensorDynLen::from_dense(vec![in_idx, out_idx], data).unwrap();
            tensors.push(tensor);
        } else if i == 0 {
            let right_bond = indices.bonds[i].clone();
            let tensor = TensorDynLen::from_dense(vec![in_idx, out_idx, right_bond], data).unwrap();
            tensors.push(tensor);
        } else if i == n - 1 {
            let left_bond = indices.bonds[i - 1].clone();
            let tensor = TensorDynLen::from_dense(vec![left_bond, in_idx, out_idx], data).unwrap();
            tensors.push(tensor);
        } else {
            let left_bond = indices.bonds[i - 1].clone();
            let right_bond = indices.bonds[i].clone();
            let tensor =
                TensorDynLen::from_dense(vec![left_bond, in_idx, out_idx, right_bond], data)
                    .unwrap();
            tensors.push(tensor);
        }
    }

    TensorTrain::new(tensors).map_err(|e| anyhow::anyhow!("{}", e))
}

/// Create a pure imaginary identity MPO: i * I
fn create_imaginary_identity_mpo(indices: &SharedIndices) -> anyhow::Result<TensorTrain> {
    let n = indices.inputs.len();
    let mut tensors = Vec::with_capacity(n);

    let i_unit = Complex64::new(0.0, 1.0);
    let one = Complex64::new(1.0, 0.0);

    for i in 0..n {
        let in_dim = indices.inputs[i].dim();
        let out_dim = indices.outputs[i].dim();
        let in_idx = indices.inputs[i].clone();
        let out_idx = indices.outputs[i].clone();

        // Factor i on first site only
        let factor = if i == 0 { i_unit } else { one };

        // Identity: δ(in, out) * factor
        let mut data = vec![Complex64::new(0.0, 0.0); in_dim * out_dim];
        for j in 0..in_dim.min(out_dim) {
            data[j * out_dim + j] = factor;
        }

        if i == 0 && n == 1 {
            let tensor = TensorDynLen::from_dense(vec![in_idx, out_idx], data).unwrap();
            tensors.push(tensor);
        } else if i == 0 {
            let right_bond = indices.bonds[i].clone();
            let tensor = TensorDynLen::from_dense(vec![in_idx, out_idx, right_bond], data).unwrap();
            tensors.push(tensor);
        } else if i == n - 1 {
            let left_bond = indices.bonds[i - 1].clone();
            let tensor = TensorDynLen::from_dense(vec![left_bond, in_idx, out_idx], data).unwrap();
            tensors.push(tensor);
        } else {
            let left_bond = indices.bonds[i - 1].clone();
            let right_bond = indices.bonds[i].clone();
            let tensor =
                TensorDynLen::from_dense(vec![left_bond, in_idx, out_idx, right_bond], data)
                    .unwrap();
            tensors.push(tensor);
        }
    }

    TensorTrain::new(tensors).map_err(|e| anyhow::anyhow!("{}", e))
}

/// Create a Pauli-X operator MPO.
/// This acts on the input indices of an MPO: σ_x[in', in]
fn create_pauli_x_operator_mpo(indices: &SharedIndices) -> anyhow::Result<TensorTrain> {
    let n = indices.inputs.len();
    let mut tensors = Vec::with_capacity(n);

    // Operator bonds (separate from MPO bonds)
    let op_bonds: Vec<DynIndex> = (0..n.saturating_sub(1))
        .map(|_| DynIndex::new_dyn(1))
        .collect();

    // Pauli-X: [[0, 1], [1, 0]]
    let pauli_x = [0.0, 1.0, 1.0, 0.0];

    for i in 0..n {
        let in_idx = indices.inputs[i].clone();
        let op_out_idx = indices.operator_outputs[i].clone();

        if i == 0 && n == 1 {
            let tensor =
                TensorDynLen::from_dense(vec![in_idx, op_out_idx], pauli_x.to_vec()).unwrap();
            tensors.push(tensor);
        } else if i == 0 {
            let right_bond = op_bonds[i].clone();
            let tensor =
                TensorDynLen::from_dense(vec![in_idx, op_out_idx, right_bond], pauli_x.to_vec())
                    .unwrap();
            tensors.push(tensor);
        } else if i == n - 1 {
            let left_bond = op_bonds[i - 1].clone();
            let tensor =
                TensorDynLen::from_dense(vec![left_bond, in_idx, op_out_idx], pauli_x.to_vec())
                    .unwrap();
            tensors.push(tensor);
        } else {
            let left_bond = op_bonds[i - 1].clone();
            let right_bond = op_bonds[i].clone();
            let tensor = TensorDynLen::from_dense(
                vec![left_bond, in_idx, op_out_idx, right_bond],
                pauli_x.to_vec(),
            )
            .unwrap();
            tensors.push(tensor);
        }
    }

    TensorTrain::new(tensors).map_err(|e| anyhow::anyhow!("{}", e))
}

/// Create a random diagonally dominant operator MPO.
fn create_random_operator_mpo(indices: &SharedIndices, seed: u64) -> anyhow::Result<TensorTrain> {
    let n = indices.inputs.len();
    let mut tensors = Vec::with_capacity(n);
    let mut rng = StdRng::seed_from_u64(seed);

    // Operator bonds (separate from MPO bonds)
    let op_bonds: Vec<DynIndex> = (0..n.saturating_sub(1))
        .map(|_| DynIndex::new_dyn(1))
        .collect();

    for i in 0..n {
        let in_dim = indices.inputs[i].dim();
        let in_idx = indices.inputs[i].clone();
        let op_out_idx = indices.operator_outputs[i].clone();

        // Diagonally dominant random matrix
        let mut data: Vec<f64> = Vec::with_capacity(in_dim * in_dim);
        for j in 0..in_dim {
            for k in 0..in_dim {
                if j == k {
                    data.push(2.0 + rng.random::<f64>());
                } else {
                    data.push(0.1 * (rng.random::<f64>() - 0.5));
                }
            }
        }

        if i == 0 && n == 1 {
            let tensor = TensorDynLen::from_dense(vec![in_idx, op_out_idx], data).unwrap();
            tensors.push(tensor);
        } else if i == 0 {
            let right_bond = op_bonds[i].clone();
            let tensor =
                TensorDynLen::from_dense(vec![in_idx, op_out_idx, right_bond], data).unwrap();
            tensors.push(tensor);
        } else if i == n - 1 {
            let left_bond = op_bonds[i - 1].clone();
            let tensor =
                TensorDynLen::from_dense(vec![left_bond, in_idx, op_out_idx], data).unwrap();
            tensors.push(tensor);
        } else {
            let left_bond = op_bonds[i - 1].clone();
            let right_bond = op_bonds[i].clone();
            let tensor =
                TensorDynLen::from_dense(vec![left_bond, in_idx, op_out_idx, right_bond], data)
                    .unwrap();
            tensors.push(tensor);
        }
    }

    TensorTrain::new(tensors).map_err(|e| anyhow::anyhow!("{}", e))
}

/// Apply an operator MPO to an MPO state: result = O * X
/// O has indices [in, out'] and X has indices [in, out]
/// Result has indices [out', out] (O's output replaces X's input)
fn apply_operator_to_mpo(
    op: &TensorTrain,
    mpo: &TensorTrain,
    indices: &SharedIndices,
) -> anyhow::Result<TensorTrain> {
    // Contract operator with MPO using fit method
    let options = ContractOptions::fit()
        .with_nhalfsweeps(4)
        .with_svd_policy(tensor4all_core::SvdTruncationPolicy::new(1e-10))
        .with_max_rank(50);

    let result = op
        .contract(mpo, &options)
        .map_err(|e| anyhow::anyhow!("{}", e))?;

    // Replace operator output indices with input indices
    let result = result.replaceinds(&indices.operator_outputs, &indices.inputs)?;

    Ok(result)
}
