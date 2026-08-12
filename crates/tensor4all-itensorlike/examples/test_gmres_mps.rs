#![allow(clippy::needless_borrows_for_generic_args)]
#![allow(clippy::identity_op)]
//! Test: GMRES solver with MPS/MPO format
//!
//! A = Pauli-X MPO (spin flip operator at each site)
//! x_true = MPS with all elements = 1
//! b = A * x_true
//!
//! Run:
//!   cargo run -p tensor4all-itensorlike --example test_gmres_mps --release

use num_complex::Complex64;
use rand::{rngs::StdRng, SeedableRng};
use tensor4all_core::krylov::{gmres_with_truncation, GmresOptions};
use tensor4all_core::{AnyScalar, DynIndex, IdxTensor, IndexLike, TensorIndex};
use tensor4all_itensorlike::{ContractOptions, TensorTrain, TruncateOptions};

/// Shared indices for all MPS/MPO operations.
/// This ensures all tensors can be added and have inner products computed.
struct SharedIndices {
    /// Physical (site) indices
    sites: Vec<DynIndex>,
    /// Bond indices between sites
    bonds: Vec<DynIndex>,
    /// MPO output indices (separate from input = site indices)
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
    println!("  Pauli-X Operator Tests");
    println!("========================================\n");

    // Test Pauli-X with varying N
    let mut pauli_results = Vec::new();
    for n in [3, 5] {
        let result = test_gmres_mps(n, "pauli_x")?;
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

    println!("\n========================================");
    println!("  Identity Operator Tests (A = I)");
    println!("========================================\n");

    // Test Identity with varying N
    let mut identity_results = Vec::new();
    for n in [3, 5] {
        let result = test_gmres_mps(n, "identity")?;
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

    // Test: Pure imaginary identity (A = i*I, x_true = i*ones)
    println!("\n========================================");
    println!("  Pure Imaginary Identity Tests (A = i*I)");
    println!("========================================\n");

    let mut imaginary_results = Vec::new();
    for n in [3, 5] {
        let result = test_gmres_mps_imaginary(n, 5)?;
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

    // Test: Random MPO
    println!("\n========================================");
    println!("  Random MPO Tests");
    println!("========================================\n");

    let mut random_results = Vec::new();
    for n in [3, 5] {
        let result = test_gmres_mps_random(n, 20)?;
        random_results.push((n, result));
        println!();
    }

    // Summary for Random MPO
    println!("========================================");
    println!("  Random MPO Summary");
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

/// Returns (initial_residual, final_residual, iterations)
fn test_gmres_mps(n: usize, operator: &str) -> anyhow::Result<(f64, f64, usize)> {
    let phys_dim = 2;

    println!("=== Test: GMRES with MPS ===");
    println!(
        "N = {}, phys_dim = {}, operator = {}",
        n, phys_dim, operator
    );

    // Create shared indices for all operations
    let indices = SharedIndices::new(n, phys_dim);

    // Create MPO based on operator type
    let (mpo, op_name) = match operator {
        "pauli_x" => (create_pauli_x_mpo(&indices)?, "Pauli-X"),
        "diagonal" => (create_diagonal_mpo(&indices)?, "Diagonal(2,3)"),
        "identity" => (create_identity_mpo(&indices)?, "Identity"),
        _ => anyhow::bail!("Unknown operator: {}", operator),
    };
    println!("{} MPO created with {} sites", op_name, mpo.len());

    // Create x_true (all ones MPS)
    let x_true = create_ones_mps(&indices)?;
    println!(
        "x_true MPS created with {} sites, norm: {:.6}",
        x_true.len(),
        x_true.norm()?
    );

    // Compute b = A * x_true
    let b = apply_mpo(&mpo, &x_true, &indices)?;
    println!("b = A * x_true computed, norm: {:.6}", b.norm()?);

    // Use 0.5*b as initial guess (x0 = 0.5*b, so initial residual is 0.5)
    // Note: x0=b gives tiny residual (~1e-8 from MPS contraction errors),
    // causing numerical instability in GMRES Krylov vectors.
    let x0 = b.scale(AnyScalar::new_real(0.5))?;
    println!(
        "x0 (initial guess = 0.5*b) created, norm: {:.6}",
        x0.norm()?
    );

    // Define apply_a closure
    let apply_a = |x: &TensorTrain| -> anyhow::Result<TensorTrain> { apply_mpo(&mpo, x, &indices) };

    // Compute initial residual: |Ax0 - b| / |b|
    let b_norm = b.norm()?;
    let ax0 = apply_a(&x0)?;
    let r0 = ax0.axpby(AnyScalar::new_real(1.0), &b, AnyScalar::new_real(-1.0))?;
    let initial_residual = r0.norm()? / b_norm;
    println!("\n=== Initial Residual ===");
    println!("|Ax0 - b| / |b| = {:.6e}", initial_residual);

    // Solve with GMRES with truncation
    let options = GmresOptions {
        max_iter: 10,
        rtol: 1e-8, // Strict tolerance to force multiple iterations
        max_restarts: 1,
        verbose: true,
        check_true_residual: false,
    };

    // Truncation options: control bond dimension growth
    let truncate_opts = TruncateOptions::svd()
        .with_svd_policy(tensor4all_core::SvdTruncationPolicy::new(1e-8))
        .with_max_bond_dim(20);
    let truncate_fn = |x: &mut TensorTrain| -> anyhow::Result<()> {
        x.truncate(&truncate_opts)?;
        Ok(())
    };

    println!(
        "\n=== Running GMRES with truncation (max_iter={}) ===",
        options.max_iter
    );
    println!("rtol = {:.2e}", options.rtol);
    println!(
        "truncation: rtol={:.2e}, max_bond_dim={}",
        truncate_opts
            .svd_policy()
            .map(|policy| policy.threshold)
            .unwrap_or(0.0),
        truncate_opts.max_bond_dim().unwrap_or(0)
    );
    let result = gmres_with_truncation(&apply_a, &b, &x0, &options, truncate_fn)?;

    // Compute final residual: |Ax_sol - b| / |b|
    let ax_sol = apply_a(&result.solution)?;
    let r_final = ax_sol.axpby(AnyScalar::new_real(1.0), &b, AnyScalar::new_real(-1.0))?;
    let final_residual = r_final.norm()? / b_norm;

    // Results
    println!("\n=== Results ===");
    println!("Converged: {}", result.converged);
    println!("Iterations: {}", result.iterations);
    println!("GMRES reported residual: {:.6e}", result.residual_norm);
    println!("Final |Ax - b| / |b|:    {:.6e}", final_residual);

    // Compute error ||x_sol - x_true||
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

/// Create an identity MPO using shared indices.
fn create_identity_mpo(indices: &SharedIndices) -> anyhow::Result<TensorTrain> {
    let n = indices.sites.len();
    let mut tensors = Vec::with_capacity(n);

    // MPO bond indices (separate from MPS bonds)
    let mpo_bonds: Vec<DynIndex> = (0..n.saturating_sub(1))
        .map(|_| DynIndex::new_dyn(1))
        .collect();

    for i in 0..n {
        let site_dim = indices.sites[i].dim();
        let s_in = indices.sites[i].clone();
        let s_out = indices.mpo_outputs[i].clone();

        // Identity tensor: delta(s_in, s_out)
        if i == 0 && n == 1 {
            // Single site: [s_in, s_out]
            let mut data = vec![0.0; site_dim * site_dim];
            for j in 0..site_dim {
                data[j * site_dim + j] = 1.0;
            }
            let tensor = IdxTensor::from_dense(vec![s_in, s_out], data).unwrap();
            tensors.push(tensor);
        } else if i == 0 {
            // First site: [s_in, s_out, right]
            let right_bond = mpo_bonds[i].clone();
            let mut data = vec![0.0; site_dim * site_dim * 1];
            for j in 0..site_dim {
                data[j * site_dim + j] = 1.0;
            }
            let tensor = IdxTensor::from_dense(vec![s_in, s_out, right_bond], data).unwrap();
            tensors.push(tensor);
        } else if i == n - 1 {
            // Last site: [left, s_in, s_out]
            let left_bond = mpo_bonds[i - 1].clone();
            let mut data = vec![0.0; 1 * site_dim * site_dim];
            for j in 0..site_dim {
                data[j * site_dim + j] = 1.0;
            }
            let tensor = IdxTensor::from_dense(vec![left_bond, s_in, s_out], data).unwrap();
            tensors.push(tensor);
        } else {
            // Middle site: [left, s_in, s_out, right]
            let left_bond = mpo_bonds[i - 1].clone();
            let right_bond = mpo_bonds[i].clone();
            let mut data = vec![0.0; 1 * site_dim * site_dim * 1];
            for j in 0..site_dim {
                data[j * site_dim + j] = 1.0;
            }
            let tensor =
                IdxTensor::from_dense(vec![left_bond, s_in, s_out, right_bond], data).unwrap();
            tensors.push(tensor);
        }
    }

    TensorTrain::new(tensors).map_err(|e| anyhow::anyhow!("{}", e))
}

/// Create an MPS with all elements = 1, using shared indices.
fn create_ones_mps(indices: &SharedIndices) -> anyhow::Result<TensorTrain> {
    let n = indices.sites.len();
    let mut tensors = Vec::with_capacity(n);

    for i in 0..n {
        let site_dim = indices.sites[i].dim();
        let site_idx = indices.sites[i].clone();

        if i == 0 && n == 1 {
            let data = vec![1.0; site_dim];
            let tensor = IdxTensor::from_dense(vec![site_idx], data).unwrap();
            tensors.push(tensor);
        } else if i == 0 {
            let right_bond = indices.bonds[i].clone();
            let data = vec![1.0; site_dim * 1];
            let tensor = IdxTensor::from_dense(vec![site_idx, right_bond], data).unwrap();
            tensors.push(tensor);
        } else if i == n - 1 {
            let left_bond = indices.bonds[i - 1].clone();
            let data = vec![1.0; 1 * site_dim];
            let tensor = IdxTensor::from_dense(vec![left_bond, site_idx], data).unwrap();
            tensors.push(tensor);
        } else {
            let left_bond = indices.bonds[i - 1].clone();
            let right_bond = indices.bonds[i].clone();
            let data = vec![1.0; 1 * site_dim * 1];
            let tensor =
                IdxTensor::from_dense(vec![left_bond, site_idx, right_bond], data).unwrap();
            tensors.push(tensor);
        }
    }

    TensorTrain::new(tensors).map_err(|e| anyhow::anyhow!("{}", e))
}

/// Apply MPO to MPS and return result with same external indices as input MPS.
///
/// After contraction, replaces MPO output indices with site indices,
/// ensuring the result is in the same vector space as the input.
fn apply_mpo(
    mpo: &TensorTrain,
    mps: &TensorTrain,
    indices: &SharedIndices,
) -> anyhow::Result<TensorTrain> {
    // Contract MPO with MPS using fit method (more stable)
    // Use consistent truncation: rtol=1e-10, max_bond_dim=20
    let options = ContractOptions::fit()
        .with_nhalfsweeps(4)
        .with_svd_policy(tensor4all_core::SvdTruncationPolicy::new(1e-8))
        .with_max_bond_dim(20);
    let result = mpo
        .contract(mps, &options)
        .map_err(|e| anyhow::anyhow!("{}", e))?;

    // Replace MPO output indices with site indices
    let result = result.replace_indices(&indices.mpo_outputs, &indices.sites)?;

    Ok(result)
}

/// Create a Pauli-X MPO operator.
/// Pauli-X matrix: [[0, 1], [1, 0]] (spin flip operator)
/// X^2 = I, so applying X twice gives identity.
fn create_pauli_x_mpo(indices: &SharedIndices) -> anyhow::Result<TensorTrain> {
    let n = indices.sites.len();
    let mut tensors = Vec::with_capacity(n);

    // MPO bond indices (separate from MPS bonds)
    let mpo_bonds: Vec<DynIndex> = (0..n.saturating_sub(1))
        .map(|_| DynIndex::new_dyn(1))
        .collect();

    // Pauli-X matrix: [[0, 1], [1, 0]]
    // As tensor [s_in, s_out]: X[in, out] where X|0>=|1>, X|1>=|0>
    // X[0,0]=0, X[0,1]=1, X[1,0]=1, X[1,1]=0
    let pauli_x = [0.0, 1.0, 1.0, 0.0];

    for i in 0..n {
        let s_in = indices.sites[i].clone();
        let s_out = indices.mpo_outputs[i].clone();

        if i == 0 && n == 1 {
            let tensor = IdxTensor::from_dense(vec![s_in, s_out], pauli_x.to_vec()).unwrap();
            tensors.push(tensor);
        } else if i == 0 {
            let right_bond = mpo_bonds[i].clone();
            let tensor =
                IdxTensor::from_dense(vec![s_in, s_out, right_bond], pauli_x.to_vec()).unwrap();
            tensors.push(tensor);
        } else if i == n - 1 {
            let left_bond = mpo_bonds[i - 1].clone();
            let tensor =
                IdxTensor::from_dense(vec![left_bond, s_in, s_out], pauli_x.to_vec()).unwrap();
            tensors.push(tensor);
        } else {
            let left_bond = mpo_bonds[i - 1].clone();
            let right_bond = mpo_bonds[i].clone();
            let tensor =
                IdxTensor::from_dense(vec![left_bond, s_in, s_out, right_bond], pauli_x.to_vec())
                    .unwrap();
            tensors.push(tensor);
        }
    }

    TensorTrain::new(tensors).map_err(|e| anyhow::anyhow!("{}", e))
}

/// Test GMRES with pure imaginary identity operator and MPS.
/// A = i * I (identity operator scaled by pure imaginary i)
/// x_true = i * (all-ones MPS)
/// b = A * x_true = i * i * (all-ones) = -(all-ones)
/// x_init = b
fn test_gmres_mps_imaginary(n: usize, max_iter: usize) -> anyhow::Result<(f64, f64, usize)> {
    let phys_dim = 2;

    println!("=== Test: GMRES with Pure Imaginary MPS ===");
    println!(
        "N = {}, phys_dim = {}, max_iter = {}",
        n, phys_dim, max_iter
    );
    println!("A = i * Identity, x_true = i * ones");

    // Create shared indices for all operations
    let indices = SharedIndices::new(n, phys_dim);

    // Create A = i * Identity MPO
    let mpo = create_imaginary_identity_mpo(&indices)?;
    println!("i*I MPO created with {} sites", mpo.len());

    // Create x_true = i * (all ones MPS)
    let x_true = create_imaginary_ones_mps(&indices)?;
    println!(
        "x_true (i*ones) MPS created with {} sites, norm: {:.6}",
        x_true.len(),
        x_true.norm()?
    );

    // Compute b = A * x_true = i*I * i*ones = -ones
    let b = apply_mpo(&mpo, &x_true, &indices)?;
    println!("b = A * x_true computed, norm: {:.6}", b.norm()?);

    // Use x0 = b as initial guess
    let x0 = b.clone();
    println!("x0 (initial guess = b) created, norm: {:.6}", x0.norm()?);

    // Define apply_a closure
    let apply_a = |x: &TensorTrain| -> anyhow::Result<TensorTrain> { apply_mpo(&mpo, x, &indices) };

    // Compute initial residual: |Ax0 - b| / |b|
    let b_norm = b.norm()?;
    let ax0 = apply_a(&x0)?;
    let r0 = ax0.axpby(AnyScalar::new_real(1.0), &b, AnyScalar::new_real(-1.0))?;
    let initial_residual = r0.norm()? / b_norm;
    println!("\n=== Initial Residual ===");
    println!("|Ax0 - b| / |b| = {:.6e}", initial_residual);

    // Solve with GMRES with truncation
    let options = GmresOptions {
        max_iter,
        rtol: 1e-8,
        max_restarts: 1,
        verbose: true,
        check_true_residual: false,
    };

    // Truncation options: control bond dimension growth
    let truncate_opts = TruncateOptions::svd()
        .with_svd_policy(tensor4all_core::SvdTruncationPolicy::new(1e-8))
        .with_max_bond_dim(20);
    let truncate_fn = |x: &mut TensorTrain| -> anyhow::Result<()> {
        x.truncate(&truncate_opts)?;
        Ok(())
    };

    println!(
        "\n=== Running GMRES with truncation (max_iter={}) ===",
        options.max_iter
    );
    println!("rtol = {:.2e}", options.rtol);
    println!(
        "truncation: rtol={:.2e}, max_bond_dim={}",
        truncate_opts
            .svd_policy()
            .map(|policy| policy.threshold)
            .unwrap_or(0.0),
        truncate_opts.max_bond_dim().unwrap_or(0)
    );
    let result = gmres_with_truncation(&apply_a, &b, &x0, &options, truncate_fn)?;

    // Compute final residual: |Ax_sol - b| / |b|
    let ax_sol = apply_a(&result.solution)?;
    let r_final = ax_sol.axpby(AnyScalar::new_real(1.0), &b, AnyScalar::new_real(-1.0))?;
    let final_residual = r_final.norm()? / b_norm;

    // Results
    println!("\n=== Results ===");
    println!("Converged: {}", result.converged);
    println!("Iterations: {}", result.iterations);
    println!("GMRES reported residual: {:.6e}", result.residual_norm);
    println!("Final |Ax - b| / |b|:    {:.6e}", final_residual);

    // Compute error ||x_sol - x_true||
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

/// Create an identity MPO scaled by pure imaginary i.
/// A = i * I
fn create_imaginary_identity_mpo(indices: &SharedIndices) -> anyhow::Result<TensorTrain> {
    let n = indices.sites.len();
    let mut tensors = Vec::with_capacity(n);

    // MPO bond indices (separate from MPS bonds)
    let mpo_bonds: Vec<DynIndex> = (0..n.saturating_sub(1))
        .map(|_| DynIndex::new_dyn(1))
        .collect();

    // Pure imaginary identity: i * delta(s_in, s_out)
    // We put the factor i on the first site only
    let i_unit = Complex64::new(0.0, 1.0);

    for i in 0..n {
        let site_dim = indices.sites[i].dim();
        let s_in = indices.sites[i].clone();
        let s_out = indices.mpo_outputs[i].clone();

        // Identity with i factor on first site
        let factor = if i == 0 {
            i_unit
        } else {
            Complex64::new(1.0, 0.0)
        };

        if i == 0 && n == 1 {
            // Single site: [s_in, s_out]
            let mut data = vec![Complex64::new(0.0, 0.0); site_dim * site_dim];
            for j in 0..site_dim {
                data[j * site_dim + j] = factor;
            }
            let tensor = IdxTensor::from_dense(vec![s_in, s_out], data).unwrap();
            tensors.push(tensor);
        } else if i == 0 {
            // First site: [s_in, s_out, right]
            let right_bond = mpo_bonds[i].clone();
            let mut data = vec![Complex64::new(0.0, 0.0); site_dim * site_dim * 1];
            for j in 0..site_dim {
                data[j * site_dim + j] = factor;
            }
            let tensor = IdxTensor::from_dense(vec![s_in, s_out, right_bond], data).unwrap();
            tensors.push(tensor);
        } else if i == n - 1 {
            // Last site: [left, s_in, s_out]
            let left_bond = mpo_bonds[i - 1].clone();
            let mut data = vec![Complex64::new(0.0, 0.0); 1 * site_dim * site_dim];
            for j in 0..site_dim {
                data[j * site_dim + j] = factor;
            }
            let tensor = IdxTensor::from_dense(vec![left_bond, s_in, s_out], data).unwrap();
            tensors.push(tensor);
        } else {
            // Middle site: [left, s_in, s_out, right]
            let left_bond = mpo_bonds[i - 1].clone();
            let right_bond = mpo_bonds[i].clone();
            let mut data = vec![Complex64::new(0.0, 0.0); 1 * site_dim * site_dim * 1];
            for j in 0..site_dim {
                data[j * site_dim + j] = factor;
            }
            let tensor =
                IdxTensor::from_dense(vec![left_bond, s_in, s_out, right_bond], data).unwrap();
            tensors.push(tensor);
        }
    }

    TensorTrain::new(tensors).map_err(|e| anyhow::anyhow!("{}", e))
}

/// Create an MPS with all elements = i (pure imaginary ones).
fn create_imaginary_ones_mps(indices: &SharedIndices) -> anyhow::Result<TensorTrain> {
    let n = indices.sites.len();
    let mut tensors = Vec::with_capacity(n);

    // Pure imaginary i on the first site only
    let i_unit = Complex64::new(0.0, 1.0);
    let one = Complex64::new(1.0, 0.0);

    for i in 0..n {
        let site_dim = indices.sites[i].dim();
        let site_idx = indices.sites[i].clone();

        let factor = if i == 0 { i_unit } else { one };

        if i == 0 && n == 1 {
            let data = vec![factor; site_dim];
            let tensor = IdxTensor::from_dense(vec![site_idx], data).unwrap();
            tensors.push(tensor);
        } else if i == 0 {
            let right_bond = indices.bonds[i].clone();
            let data = vec![factor; site_dim];
            let tensor = IdxTensor::from_dense(vec![site_idx, right_bond], data).unwrap();
            tensors.push(tensor);
        } else if i == n - 1 {
            let left_bond = indices.bonds[i - 1].clone();
            let data = vec![factor; site_dim];
            let tensor = IdxTensor::from_dense(vec![left_bond, site_idx], data).unwrap();
            tensors.push(tensor);
        } else {
            let left_bond = indices.bonds[i - 1].clone();
            let right_bond = indices.bonds[i].clone();
            let data = vec![factor; site_dim];
            let tensor =
                IdxTensor::from_dense(vec![left_bond, site_idx, right_bond], data).unwrap();
            tensors.push(tensor);
        }
    }

    TensorTrain::new(tensors).map_err(|e| anyhow::anyhow!("{}", e))
}

/// Create a diagonal MPO operator.
/// Diagonal matrix: diag(d0, d1) where d0=2.0, d1=3.0
/// This is a more challenging test case than Pauli-X since it's not self-inverse.
fn create_diagonal_mpo(indices: &SharedIndices) -> anyhow::Result<TensorTrain> {
    let n = indices.sites.len();
    let mut tensors = Vec::with_capacity(n);

    // MPO bond indices (separate from MPS bonds)
    let mpo_bonds: Vec<DynIndex> = (0..n.saturating_sub(1))
        .map(|_| DynIndex::new_dyn(1))
        .collect();

    // Diagonal matrix: [[2, 0], [0, 3]]
    // As tensor [s_in, s_out]: D[in, out]
    // D[0,0]=2, D[0,1]=0, D[1,0]=0, D[1,1]=3
    let diag_mat = [2.0, 0.0, 0.0, 3.0];

    for i in 0..n {
        let s_in = indices.sites[i].clone();
        let s_out = indices.mpo_outputs[i].clone();

        if i == 0 && n == 1 {
            let tensor = IdxTensor::from_dense(vec![s_in, s_out], diag_mat.to_vec()).unwrap();
            tensors.push(tensor);
        } else if i == 0 {
            let right_bond = mpo_bonds[i].clone();
            let tensor =
                IdxTensor::from_dense(vec![s_in, s_out, right_bond], diag_mat.to_vec()).unwrap();
            tensors.push(tensor);
        } else if i == n - 1 {
            let left_bond = mpo_bonds[i - 1].clone();
            let tensor =
                IdxTensor::from_dense(vec![left_bond, s_in, s_out], diag_mat.to_vec()).unwrap();
            tensors.push(tensor);
        } else {
            let left_bond = mpo_bonds[i - 1].clone();
            let right_bond = mpo_bonds[i].clone();
            let tensor =
                IdxTensor::from_dense(vec![left_bond, s_in, s_out, right_bond], diag_mat.to_vec())
                    .unwrap();
            tensors.push(tensor);
        }
    }

    TensorTrain::new(tensors).map_err(|e| anyhow::anyhow!("{}", e))
}

/// Test GMRES with a random MPO operator.
/// A = random MPO (with diagonal dominance for stability)
/// x_true = all-ones MPS
/// b = A * x_true
/// x_init = b
fn test_gmres_mps_random(n: usize, max_iter: usize) -> anyhow::Result<(f64, f64, usize)> {
    let phys_dim = 2;

    println!("=== Test: GMRES with Random MPO ===");
    println!(
        "N = {}, phys_dim = {}, max_iter = {}",
        n, phys_dim, max_iter
    );

    // Create shared indices for all operations
    let indices = SharedIndices::new(n, phys_dim);

    // Create random MPO with fixed seed for reproducibility
    let seed = 42u64;
    let mpo = create_random_mpo(&indices, seed)?;
    println!("Random MPO created with {} sites", mpo.len());

    // Create x_true (all ones MPS)
    let x_true = create_ones_mps(&indices)?;
    println!(
        "x_true MPS created with {} sites, norm: {:.6}",
        x_true.len(),
        x_true.norm()?
    );

    // Compute b = A * x_true
    let b = apply_mpo(&mpo, &x_true, &indices)?;
    println!("b = A * x_true computed, norm: {:.6}", b.norm()?);

    // Use x0 = b as initial guess
    let x0 = b.clone();
    println!("x0 (initial guess = b) created, norm: {:.6}", x0.norm()?);

    // Define apply_a closure
    let apply_a = |x: &TensorTrain| -> anyhow::Result<TensorTrain> { apply_mpo(&mpo, x, &indices) };

    // Compute initial residual: |Ax0 - b| / |b|
    let b_norm = b.norm()?;
    let ax0 = apply_a(&x0)?;
    let r0 = ax0.axpby(AnyScalar::new_real(1.0), &b, AnyScalar::new_real(-1.0))?;
    let initial_residual = r0.norm()? / b_norm;
    println!("\n=== Initial Residual ===");
    println!("|Ax0 - b| / |b| = {:.6e}", initial_residual);

    // Solve with GMRES with truncation
    let options = GmresOptions {
        max_iter,
        rtol: 1e-8,
        max_restarts: 5,
        verbose: true,
        check_true_residual: true,
    };

    // Truncation options: control bond dimension growth
    let truncate_opts = TruncateOptions::svd()
        .with_svd_policy(tensor4all_core::SvdTruncationPolicy::new(1e-8))
        .with_max_bond_dim(50);
    let truncate_fn = |x: &mut TensorTrain| -> anyhow::Result<()> {
        x.truncate(&truncate_opts)?;
        Ok(())
    };

    println!(
        "\n=== Running GMRES with truncation (max_iter={}) ===",
        options.max_iter
    );
    println!("rtol = {:.2e}", options.rtol);
    println!(
        "truncation: rtol={:.2e}, max_bond_dim={}",
        truncate_opts
            .svd_policy()
            .map(|policy| policy.threshold)
            .unwrap_or(0.0),
        truncate_opts.max_bond_dim().unwrap_or(0)
    );
    let result = gmres_with_truncation(&apply_a, &b, &x0, &options, truncate_fn)?;

    // Compute final residual: |Ax_sol - b| / |b|
    let ax_sol = apply_a(&result.solution)?;
    let r_final = ax_sol.axpby(AnyScalar::new_real(1.0), &b, AnyScalar::new_real(-1.0))?;
    let final_residual = r_final.norm()? / b_norm;

    // Results
    println!("\n=== Results ===");
    println!("Converged: {}", result.converged);
    println!("Iterations: {}", result.iterations);
    println!("GMRES reported residual: {:.6e}", result.residual_norm);
    println!("Final |Ax - b| / |b|:    {:.6e}", final_residual);

    // Compute error ||x_sol - x_true||
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

/// Create a random MPO with diagonal dominance for numerical stability.
/// The diagonal elements are larger than off-diagonal to ensure the operator is invertible.
fn create_random_mpo(indices: &SharedIndices, seed: u64) -> anyhow::Result<TensorTrain> {
    use rand::Rng;

    let n = indices.sites.len();
    let mut tensors = Vec::with_capacity(n);
    let mut rng = StdRng::seed_from_u64(seed);

    // MPO bond indices (separate from MPS bonds)
    let mpo_bonds: Vec<DynIndex> = (0..n.saturating_sub(1))
        .map(|_| DynIndex::new_dyn(1))
        .collect();

    for i in 0..n {
        let site_dim = indices.sites[i].dim();
        let s_in = indices.sites[i].clone();
        let s_out = indices.mpo_outputs[i].clone();

        // Create a diagonally dominant random matrix
        // M[j,k] = delta[j,k] * (2.0 + random) + (1 - delta[j,k]) * 0.1 * random
        let mut data: Vec<f64> = Vec::with_capacity(site_dim * site_dim);
        for j in 0..site_dim {
            for k in 0..site_dim {
                if j == k {
                    // Diagonal: 2.0 + random in [0, 1)
                    data.push(2.0 + rng.random::<f64>());
                } else {
                    // Off-diagonal: small random value
                    data.push(0.1 * (rng.random::<f64>() - 0.5));
                }
            }
        }

        if i == 0 && n == 1 {
            let tensor = IdxTensor::from_dense(vec![s_in, s_out], data).unwrap();
            tensors.push(tensor);
        } else if i == 0 {
            let right_bond = mpo_bonds[i].clone();
            let tensor = IdxTensor::from_dense(vec![s_in, s_out, right_bond], data).unwrap();
            tensors.push(tensor);
        } else if i == n - 1 {
            let left_bond = mpo_bonds[i - 1].clone();
            let tensor = IdxTensor::from_dense(vec![left_bond, s_in, s_out], data).unwrap();
            tensors.push(tensor);
        } else {
            let left_bond = mpo_bonds[i - 1].clone();
            let right_bond = mpo_bonds[i].clone();
            let tensor =
                IdxTensor::from_dense(vec![left_bond, s_in, s_out, right_bond], data).unwrap();
            tensors.push(tensor);
        }
    }

    TensorTrain::new(tensors).map_err(|e| anyhow::anyhow!("{}", e))
}
