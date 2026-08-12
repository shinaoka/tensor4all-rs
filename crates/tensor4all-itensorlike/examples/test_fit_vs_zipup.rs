#![allow(clippy::needless_borrows_for_generic_args)]
#![allow(clippy::println_empty_string)]
#![allow(unused_variables)]
//! Minimal test to compare ContractOptions::fit() vs zipup() for MPO×MPO contraction.
//!
//! Run:
//!   cargo run -p tensor4all-itensorlike --example test_fit_vs_zipup --release

use tensor4all_core::krylov::{gmres, gmres_with_truncation, GmresOptions};
use tensor4all_core::{AnyScalar, DynIndex, IdxTensor, IndexLike, TensorIndex};
use tensor4all_itensorlike::{ContractOptions, TensorTrain, TruncateOptions};

/// Shared indices for MPO operations.
struct SharedIndices {
    inputs: Vec<DynIndex>,
    outputs: Vec<DynIndex>,
    bonds: Vec<DynIndex>,
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
    println!("=== Comparing fit() vs zipup() for MPO×MPO contraction ===\n");

    // First, minimal test to find the exact failure point
    println!("========================================");
    println!("  MINIMAL N=3 ARNOLDI TRACE");
    println!("========================================\n");
    trace_arnoldi_n3()?;

    // Test truncation effect on orthogonality
    test_truncation_effect()?;

    // Debug solution update to find why zipup fails
    debug_solution_update()?;

    // Debug GMRES internal computations
    debug_gmres_internal()?;

    // Detailed debug for N=3 only
    println!("========================================");
    println!("  DETAILED DEBUG FOR N=3");
    println!("========================================\n");
    detailed_debug_n3()?;

    for n in [3, 5, 10] {
        println!("----------------------------------------");
        println!("N = {}", n);
        println!("----------------------------------------");

        let indices = SharedIndices::new(n, 2);

        // Create identity MPO and Pauli-X operator
        let identity = create_identity_mpo(&indices)?;
        let pauli_x = create_pauli_x_operator(&indices)?;

        println!("Identity MPO norm: {:.6}", identity.norm()?);
        println!("Pauli-X operator norm: {:.6}", pauli_x.norm()?);

        // Apply Pauli-X to identity with zipup
        let result_zipup = apply_with_zipup(&pauli_x, &identity, &indices)?;
        println!("\n[zipup] Result norm: {:.6}", result_zipup.norm()?);

        // Apply Pauli-X to identity with fit
        let result_fit = apply_with_fit(&pauli_x, &identity, &indices)?;
        println!("[fit]   Result norm: {:.6}", result_fit.norm()?);

        // Compare results
        let diff = result_zipup.axpby(
            AnyScalar::new_real(1.0),
            &result_fit,
            AnyScalar::new_real(-1.0),
        )?;
        println!("\n||zipup - fit|| = {:.6e}", diff.norm()?);

        // Apply Pauli-X twice: σ_x² = I
        println!("\n--- Applying Pauli-X twice (should give identity) ---");

        let twice_zipup = apply_with_zipup(&pauli_x, &result_zipup, &indices)?;
        let twice_fit = apply_with_fit(&pauli_x, &result_fit, &indices)?;

        let diff_from_identity_zipup = twice_zipup.axpby(
            AnyScalar::new_real(1.0),
            &identity,
            AnyScalar::new_real(-1.0),
        )?;
        let diff_from_identity_fit = twice_fit.axpby(
            AnyScalar::new_real(1.0),
            &identity,
            AnyScalar::new_real(-1.0),
        )?;

        println!(
            "[zipup] ||σ_x²(I) - I|| = {:.6e}",
            diff_from_identity_zipup.norm()?
        );
        println!(
            "[fit]   ||σ_x²(I) - I|| = {:.6e}",
            diff_from_identity_fit.norm()?
        );

        // Inner products with identity
        let inner_zipup = result_zipup.inner(&result_zipup)?;
        let inner_fit = result_fit.inner(&result_fit)?;
        println!("\n[zipup] <result, result> = {:?}", inner_zipup);
        println!("[fit]   <result, result> = {:?}", inner_fit);

        // Repeated application test
        println!("\n--- Repeated application (10 times) ---");
        let mut current_zipup = identity.clone();
        let mut current_fit = identity.clone();

        for iter in 1..=10 {
            current_zipup = apply_with_zipup(&pauli_x, &current_zipup, &indices)?;
            current_fit = apply_with_fit(&pauli_x, &current_fit, &indices)?;

            // σ_x^(2k) = I, σ_x^(2k+1) = σ_x
            // So after even iterations, we should get identity back
            if iter % 2 == 0 {
                let diff_zipup = current_zipup.axpby(
                    AnyScalar::new_real(1.0),
                    &identity,
                    AnyScalar::new_real(-1.0),
                )?;
                let diff_fit = current_fit.axpby(
                    AnyScalar::new_real(1.0),
                    &identity,
                    AnyScalar::new_real(-1.0),
                )?;
                println!(
                    "After {} iters: [zipup] ||result - I|| = {:.6e}, [fit] ||result - I|| = {:.6e}",
                    iter,
                    diff_zipup.norm()?,
                    diff_fit.norm()?
                );
            }
        }

        // Final norms
        println!(
            "\nFinal norms after 10 iters: [zipup] {:.6}, [fit] {:.6}",
            current_zipup.norm()?,
            current_fit.norm()?
        );
        println!("Expected norm (σ_x^10 = I): {:.6}", identity.norm()?);

        // GMRES-like operations test
        println!("\n--- GMRES-like operations test ---");
        test_gmres_like_operations(&pauli_x, &identity, &indices)?;

        // GMRES test
        println!("\n--- GMRES test with fit vs zipup ---");
        test_gmres_with_both(&pauli_x, &identity, &indices)?;

        println!();
    }

    Ok(())
}

/// Trace Arnoldi process for N=3 to find exact failure point
fn trace_arnoldi_n3() -> anyhow::Result<()> {
    let n = 3;
    let indices = SharedIndices::new(n, 2);
    let identity = create_identity_mpo(&indices)?;
    let pauli_x = create_pauli_x_operator(&indices)?;

    // x_true = I, b = σ_x * I = σ_x
    let x_true = identity.clone();
    let b = apply_with_zipup(&pauli_x, &x_true, &indices)?;
    let x0 = b.scale(AnyScalar::new_real(0.5))?;

    // For σ_x * x = σ_x, solution is x = I
    // Initial residual: r0 = b - A(x0) = σ_x - σ_x*(0.5*σ_x) = σ_x - 0.5*I

    println!("=== Setup ===");
    println!("||b|| = {:.6}", b.norm()?);
    println!("||x0|| = {:.6}", x0.norm()?);
    println!("||x_true|| = {:.6}", x_true.norm()?);

    // Compute A(x0)
    let ax0 = apply_with_zipup(&pauli_x, &x0, &indices)?;
    println!("||A(x0)|| = {:.6}", ax0.norm()?);

    // r0 = b - A(x0)
    let r0 = b.axpby(AnyScalar::new_real(1.0), &ax0, AnyScalar::new_real(-1.0))?;
    let beta = r0.norm()?;
    println!("||r0|| = beta = {:.6}", beta);

    // v1 = r0 / ||r0||
    let v1 = r0.scale(AnyScalar::new_real(1.0 / beta))?;
    println!("||v1|| = {:.6}", v1.norm()?);

    // Check: <v1, v1> should be 1
    let v1v1 = v1.inner(&v1)?;
    println!("<v1, v1> = {:?}", v1v1);

    // w = A(v1)
    let w = apply_with_zipup(&pauli_x, &v1, &indices)?;
    println!("||w|| = ||A(v1)|| = {:.6}", w.norm()?);

    // h11 = <w, v1>
    let h11 = w.inner(&v1)?;
    println!("h11 = <A(v1), v1> = {:?}", h11);
    let h11_val = h11.real();

    // v2_tilde = w - h11 * v1
    let v2_tilde = w.axpby(AnyScalar::new_real(1.0), &v1, AnyScalar::new_real(-h11_val))?;
    let h21 = v2_tilde.norm()?;
    println!("h21 = ||w - h11*v1|| = {:.6}", h21);

    // v2 = v2_tilde / h21
    let v2 = v2_tilde.scale(AnyScalar::new_real(1.0 / h21))?;
    println!("||v2|| = {:.6}", v2.norm()?);

    // Check orthogonality
    let v1v2 = v1.inner(&v2)?;
    println!("<v1, v2> = {:?}", v1v2);

    // w2 = A(v2)
    let w2 = apply_with_zipup(&pauli_x, &v2, &indices)?;
    println!("||w2|| = ||A(v2)|| = {:.6}", w2.norm()?);

    // h12 = <w2, v1>, h22 = <w2, v2>
    let h12 = w2.inner(&v1)?;
    let h22 = w2.inner(&v2)?;
    println!("h12 = <A(v2), v1> = {:?}", h12);
    println!("h22 = <A(v2), v2> = {:?}", h22);
    let h12_val = h12.real();
    let h22_val = h22.real();

    // v3_tilde = w2 - h12*v1 - h22*v2
    let v3_tilde_temp = w2.axpby(AnyScalar::new_real(1.0), &v1, AnyScalar::new_real(-h12_val))?;
    let v3_tilde =
        v3_tilde_temp.axpby(AnyScalar::new_real(1.0), &v2, AnyScalar::new_real(-h22_val))?;
    let h32 = v3_tilde.norm()?;
    println!("h32 = ||w2 - h12*v1 - h22*v2|| = {:.6}", h32);

    println!("\n=== Hessenberg Matrix ===");
    println!("H = [[{:.6}, {:.6}],", h11_val, h12_val);
    println!("     [{:.6}, {:.6}],", h21, h22_val);
    println!("     [0.0,     {:.6}]]", h32);

    // Solve the least squares problem: min ||beta*e1 - H*y||
    // For 2 iterations with H being 3x2
    println!("\n=== Solve least squares ===");
    // Simple case: after 2 iterations
    // H = [[h11, h12], [h21, h22], [0, h32]]
    // We want to minimize ||beta*[1,0,0]^T - H*y||

    // Let's compute what GMRES would compute
    // Using QR factorization of H via Givens rotations

    // First rotation: eliminate h21
    let r = (h11_val * h11_val + h21 * h21).sqrt();
    let c1 = h11_val / r;
    let s1 = h21 / r;
    println!("Givens 1: c1={:.6}, s1={:.6}, r={:.6}", c1, s1, r);

    // Apply to first column: [h11, h21, 0] -> [r, 0, 0]
    // Apply to second column: [h12, h22, h32]
    let h12_new = c1 * h12_val + s1 * h22_val;
    let h22_new = -s1 * h12_val + c1 * h22_val;
    println!("After G1: R[0,1]={:.6}, R[1,1]={:.6}", h12_new, h22_new);

    // Apply to RHS: beta*e1 = [beta, 0, 0] -> [c1*beta, -s1*beta, 0]
    let g1 = c1 * beta;
    let g2 = -s1 * beta;
    println!("After G1: g=[{:.6}, {:.6}, 0]", g1, g2);

    // Second rotation: eliminate h32 using h22_new
    let r2 = (h22_new * h22_new + h32 * h32).sqrt();
    let c2 = h22_new / r2;
    let s2 = h32 / r2;
    println!("Givens 2: c2={:.6}, s2={:.6}, r2={:.6}", c2, s2, r2);

    // Apply to second column: [h12_new, h22_new, h32] -> [h12_new, r2, 0]
    // Apply to RHS: [g1, g2, 0] -> [g1, c2*g2, -s2*g2]
    let g2_new = c2 * g2;
    let g3 = -s2 * g2;
    println!("After G2: g=[{:.6}, {:.6}, {:.6}]", g1, g2_new, g3);

    // GMRES residual estimate = |g3| / ||b||
    let gmres_residual = g3.abs() / b.norm()?;
    println!(
        "GMRES residual estimate = |g3|/||b|| = {:.6e}",
        gmres_residual
    );

    // Solve R*y = g (upper triangular)
    // R = [[r, h12_new], [0, r2]]
    // g = [g1, g2_new]
    let y2 = g2_new / r2;
    let y1 = (g1 - h12_new * y2) / r;
    println!("y = [{:.6}, {:.6}]", y1, y2);

    // x_sol = x0 + V*y = x0 + y1*v1 + y2*v2
    let temp = x0.axpby(AnyScalar::new_real(1.0), &v1, AnyScalar::new_real(y1))?;
    let x_sol = temp.axpby(AnyScalar::new_real(1.0), &v2, AnyScalar::new_real(y2))?;
    println!("||x_sol|| = {:.6}", x_sol.norm()?);

    // Actual residual
    let ax_sol = apply_with_zipup(&pauli_x, &x_sol, &indices)?;
    let actual_r = ax_sol.axpby(AnyScalar::new_real(1.0), &b, AnyScalar::new_real(-1.0))?;
    let actual_residual = actual_r.norm()? / b.norm()?;
    println!("\n=== Compare Residuals ===");
    println!("GMRES estimated residual: {:.6e}", gmres_residual);
    println!("Actual residual:          {:.6e}", actual_residual);

    // Check x_sol vs x_true
    let sol_error = x_sol.axpby(AnyScalar::new_real(1.0), &x_true, AnyScalar::new_real(-1.0))?;
    println!("||x_sol - x_true|| = {:.6e}", sol_error.norm()?);

    // Check inner product with x_true
    let inner_sol_true = x_sol.inner(&x_true)?;
    let inner_true_true = x_true.inner(&x_true)?;
    println!("<x_sol, x_true> = {:?}", inner_sol_true);
    println!("<x_true, x_true> = {:?}", inner_true_true);

    // Additional check: verify A(x_true) = b
    let a_x_true = apply_with_zipup(&pauli_x, &x_true, &indices)?;
    let check_diff = a_x_true.axpby(AnyScalar::new_real(1.0), &b, AnyScalar::new_real(-1.0))?;
    println!("\n=== Verify A(x_true) = b ===");
    println!("||A(x_true) - b|| = {:.6e}", check_diff.norm()?);

    // Check: does A(V) span the right space?
    // V = [v1, v2] forms the Krylov subspace
    // A*V should be close to V*H (up to the residual)
    println!("\n=== Arnoldi Relation Check ===");
    // A*V = V*H + h32*v3*e2^T
    // So A*v1 should = h11*v1 + h21*v2
    let av1_computed = v1.axpby(AnyScalar::new_real(h11_val), &v2, AnyScalar::new_real(h21))?;
    let av1_diff = w.axpby(
        AnyScalar::new_real(1.0),
        &av1_computed,
        AnyScalar::new_real(-1.0),
    )?;
    println!("||A*v1 - (h11*v1 + h21*v2)|| = {:.6e}", av1_diff.norm()?);

    // A*v2 should = h12*v1 + h22*v2 + h32*v3
    let v3 = if h32 > 1e-14 {
        v3_tilde.scale(AnyScalar::new_real(1.0 / h32))?
    } else {
        v3_tilde.clone()
    };
    let temp1 = v1.axpby(
        AnyScalar::new_real(h12_val),
        &v2,
        AnyScalar::new_real(h22_val),
    )?;
    let av2_computed = temp1.axpby(AnyScalar::new_real(1.0), &v3, AnyScalar::new_real(h32))?;
    let av2_diff = w2.axpby(
        AnyScalar::new_real(1.0),
        &av2_computed,
        AnyScalar::new_real(-1.0),
    )?;
    println!(
        "||A*v2 - (h12*v1 + h22*v2 + h32*v3)|| = {:.6e}",
        av2_diff.norm()?
    );

    Ok(())
}

/// Debug GMRES internal computations to find the discrepancy
fn debug_gmres_internal() -> anyhow::Result<()> {
    println!("\n========================================");
    println!("  DEBUG GMRES INTERNAL (N=3)");
    println!("========================================\n");

    let n = 3;
    let indices = SharedIndices::new(n, 2);
    let identity = create_identity_mpo(&indices)?;
    let pauli_x = create_pauli_x_operator(&indices)?;

    let x_true = identity.clone();
    let b = apply_with_zipup(&pauli_x, &x_true, &indices)?;
    let x0 = b.scale(AnyScalar::new_real(0.5))?;

    let b_norm = b.norm()?;

    println!("=== Setup ===");
    println!("||b|| = {:.6}", b_norm);
    println!("||x0|| = {:.6}", x0.norm()?);

    // Manually run GMRES algorithm step by step with detailed debug
    let apply_a = |x: &TensorTrain| -> anyhow::Result<TensorTrain> {
        apply_with_zipup(&pauli_x, x, &indices)
    };

    // r = b - A*x0
    let ax0 = apply_a(&x0)?;
    let r = b.axpby(AnyScalar::new_real(1.0), &ax0, AnyScalar::new_real(-1.0))?;
    let r_norm = r.norm()?;

    println!("\n=== Initial residual ===");
    println!("||r0|| = {:.6}", r_norm);
    println!("r0 bond dims: {:?}", r.bond_dims());

    // v0 = r / ||r||
    let v0 = r.scale(AnyScalar::new_real(1.0 / r_norm))?;
    println!("\n=== Arnoldi iteration 1 ===");
    println!("||v0|| = {:.6}", v0.norm()?);

    // w = A*v0
    let w = apply_a(&v0)?;
    println!("||A*v0|| = {:.6}", w.norm()?);

    // h00 = <v0, w>
    let h00 = v0.inner(&w)?;
    println!("h00 = <v0, A*v0> = {:?}", h00);
    let h00_val = h00.real();

    // w_orth = w - h00*v0
    let w_orth = w.axpby(AnyScalar::new_real(1.0), &v0, AnyScalar::new_real(-h00_val))?;
    let h10 = w_orth.norm()?;
    println!("h10 = ||w - h00*v0|| = {:.6}", h10);

    // v1 = w_orth / h10
    let v1 = w_orth.scale(AnyScalar::new_real(1.0 / h10))?;
    println!("||v1|| = {:.6}", v1.norm()?);
    println!("<v0, v1> = {:?}", v0.inner(&v1)?);

    // Arnoldi iteration 2
    println!("\n=== Arnoldi iteration 2 ===");
    let w2 = apply_a(&v1)?;
    println!("||A*v1|| = {:.6}", w2.norm()?);

    let h01 = v0.inner(&w2)?;
    let h11 = v1.inner(&w2)?;
    println!("h01 = <v0, A*v1> = {:?}", h01);
    println!("h11 = <v1, A*v1> = {:?}", h11);
    let h01_val = h01.real();
    let h11_val = h11.real();

    // w2_orth = w2 - h01*v0 - h11*v1
    let temp = w2.axpby(AnyScalar::new_real(1.0), &v0, AnyScalar::new_real(-h01_val))?;
    let w2_orth = temp.axpby(AnyScalar::new_real(1.0), &v1, AnyScalar::new_real(-h11_val))?;
    let h21 = w2_orth.norm()?;
    println!("h21 = ||w2 - h01*v0 - h11*v1|| = {:.6}", h21);

    // Hessenberg matrix
    println!("\n=== Hessenberg matrix ===");
    println!("H = [[{:.6}, {:.6}],", h00_val, h01_val);
    println!("     [{:.6}, {:.6}],", h10, h11_val);
    println!("     [0.0,     {:.6}]]", h21);

    // Solve least squares using Givens rotations
    println!("\n=== Solve least squares ===");
    let r1 = (h00_val * h00_val + h10 * h10).sqrt();
    let c1 = h00_val / r1;
    let s1 = h10 / r1;
    println!("Givens 1: c1={:.6}, s1={:.6}, r1={:.6}", c1, s1, r1);

    // Apply to column 1
    let h01_rot = c1 * h01_val + s1 * h11_val;
    let h11_rot = -s1 * h01_val + c1 * h11_val;
    println!("After G1: H[0,1]={:.6}, H[1,1]={:.6}", h01_rot, h11_rot);

    // Apply to RHS: [r_norm, 0, 0]
    let g0 = c1 * r_norm;
    let g1 = -s1 * r_norm;
    println!("After G1: g=[{:.6}, {:.6}, 0]", g0, g1);

    // Second Givens to eliminate h21
    let r2 = (h11_rot * h11_rot + h21 * h21).sqrt();
    let c2 = h11_rot / r2;
    let s2 = h21 / r2;
    println!("Givens 2: c2={:.6}, s2={:.6}, r2={:.6}", c2, s2, r2);

    let g1_rot = c2 * g1;
    let g2 = -s2 * g1;
    println!("After G2: g=[{:.6}, {:.6}, {:.6}]", g0, g1_rot, g2);

    let gmres_residual = g2.abs() / b_norm;
    println!("GMRES residual = |g2|/||b|| = {:.6e}", gmres_residual);

    // Back substitution
    // R = [[r1, h01_rot], [0, r2]]
    // Ry = [g0, g1_rot]
    let y1 = g1_rot / r2;
    let y0 = (g0 - h01_rot * y1) / r1;
    println!("\n=== Solution coefficients ===");
    println!("y = [{:.6}, {:.6}]", y0, y1);

    // Compute solution: x = x0 + y0*v0 + y1*v1
    println!("\n=== Solution update ===");
    println!("Computing: x_sol = x0 + y0*v0 + y1*v1");

    // Method 1: Direct (like my manual trace)
    let scaled_v0 = v0.scale(AnyScalar::new_real(y0))?;
    let scaled_v1 = v1.scale(AnyScalar::new_real(y1))?;
    let temp1 = x0.axpby(
        AnyScalar::new_real(1.0),
        &scaled_v0,
        AnyScalar::new_real(1.0),
    )?;
    let x_sol_direct = temp1.axpby(
        AnyScalar::new_real(1.0),
        &scaled_v1,
        AnyScalar::new_real(1.0),
    )?;
    println!("Direct method: ||x_sol|| = {:.6}", x_sol_direct.norm()?);

    // Method 2: Like GMRES code
    let mut x_sol_gmres = x0.clone();
    let scaled_v0_2 = v0.scale(AnyScalar::new_real(y0))?;
    x_sol_gmres = x_sol_gmres.axpby(
        AnyScalar::new_real(1.0),
        &scaled_v0_2,
        AnyScalar::new_real(1.0),
    )?;
    println!("After adding y0*v0: ||x_sol|| = {:.6}", x_sol_gmres.norm()?);
    let scaled_v1_2 = v1.scale(AnyScalar::new_real(y1))?;
    x_sol_gmres = x_sol_gmres.axpby(
        AnyScalar::new_real(1.0),
        &scaled_v1_2,
        AnyScalar::new_real(1.0),
    )?;
    println!("After adding y1*v1: ||x_sol|| = {:.6}", x_sol_gmres.norm()?);

    // Check difference between methods
    let diff = x_sol_direct.axpby(
        AnyScalar::new_real(1.0),
        &x_sol_gmres,
        AnyScalar::new_real(-1.0),
    )?;
    println!("||direct - gmres|| = {:.6e}", diff.norm()?);

    // Check actual residual
    let ax_sol = apply_a(&x_sol_gmres)?;
    let r_sol = ax_sol.axpby(AnyScalar::new_real(1.0), &b, AnyScalar::new_real(-1.0))?;
    let actual_residual = r_sol.norm()? / b_norm;
    println!("\n=== Verify ===");
    println!("||x_sol|| = {:.6}", x_sol_gmres.norm()?);
    println!("||x_true|| = {:.6}", x_true.norm()?);
    println!("GMRES residual: {:.6e}", gmres_residual);
    println!("Actual residual: {:.6e}", actual_residual);

    // Check solution error
    let sol_error =
        x_sol_gmres.axpby(AnyScalar::new_real(1.0), &x_true, AnyScalar::new_real(-1.0))?;
    println!("||x_sol - x_true|| = {:.6e}", sol_error.norm()?);

    // Check intermediate values
    println!("\n=== Debug intermediates ===");
    println!("y0*v0 norm: {:.6}", scaled_v0.norm()?);
    println!("y1*v1 norm: {:.6}", scaled_v1.norm()?);
    println!("x0 + y0*v0 bond dims: {:?}", temp1.bond_dims());
    println!("x_sol bond dims: {:?}", x_sol_gmres.bond_dims());

    // Now test the actual gmres function (no truncation)
    println!("\n=== Test actual gmres function (no truncation) ===");
    let apply_a_clone = |x: &TensorTrain| -> anyhow::Result<TensorTrain> {
        apply_with_zipup(&pauli_x, x, &indices)
    };
    let options = GmresOptions {
        max_iter: 10,
        rtol: 1e-8,
        max_restarts: 1,
        verbose: true,
        check_true_residual: false,
    };
    let x0_for_gmres = b.scale(AnyScalar::new_real(0.5))?;
    let gmres_result = gmres(&apply_a_clone, &b, &x0_for_gmres, &options)?;
    println!("GMRES converged: {}", gmres_result.converged);
    println!("GMRES iters: {}", gmres_result.iterations);
    println!(
        "GMRES reported residual: {:.6e}",
        gmres_result.residual_norm
    );
    println!("||x_sol|| from gmres: {:.6}", gmres_result.solution.norm()?);

    // Check actual residual
    let ax_gmres = apply_a(&gmres_result.solution)?;
    let r_gmres = ax_gmres.axpby(AnyScalar::new_real(1.0), &b, AnyScalar::new_real(-1.0))?;
    println!(
        "Actual residual from gmres: {:.6e}",
        r_gmres.norm()? / b_norm
    );

    // Test gmres_with_truncation with no-op truncation
    println!("\n=== Test gmres_with_truncation with no-op ===");
    let no_truncate = |_x: &mut TensorTrain| -> anyhow::Result<()> { Ok(()) };
    let x0_for_gmres2 = b.scale(AnyScalar::new_real(0.5))?;
    let gmres_trunc_result =
        gmres_with_truncation(&apply_a_clone, &b, &x0_for_gmres2, &options, no_truncate)?;
    println!("GMRES_trunc converged: {}", gmres_trunc_result.converged);
    println!("GMRES_trunc iters: {}", gmres_trunc_result.iterations);
    println!(
        "GMRES_trunc reported residual: {:.6e}",
        gmres_trunc_result.residual_norm
    );
    println!(
        "||x_sol|| from gmres_trunc: {:.6}",
        gmres_trunc_result.solution.norm()?
    );

    // Check actual residual
    let ax_gmres_trunc = apply_a(&gmres_trunc_result.solution)?;
    let r_gmres_trunc =
        ax_gmres_trunc.axpby(AnyScalar::new_real(1.0), &b, AnyScalar::new_real(-1.0))?;
    println!(
        "Actual residual from gmres_trunc: {:.6e}",
        r_gmres_trunc.norm()? / b_norm
    );

    Ok(())
}

/// Debug the solution update to find why zipup fails for N=3
fn debug_solution_update() -> anyhow::Result<()> {
    println!("\n========================================");
    println!("  DEBUG SOLUTION UPDATE (N=3)");
    println!("========================================\n");

    let n = 3;
    let indices = SharedIndices::new(n, 2);
    let identity = create_identity_mpo(&indices)?;
    let pauli_x = create_pauli_x_operator(&indices)?;

    let x_true = identity.clone();
    let b_zipup = apply_with_zipup(&pauli_x, &x_true, &indices)?;
    let b_fit = apply_with_fit(&pauli_x, &x_true, &indices)?;

    println!("=== Initial setup ===");
    println!("||b_zipup|| = {:.6}", b_zipup.norm()?);
    println!("||b_fit||   = {:.6}", b_fit.norm()?);

    // Initial guess: x0 = 0.5 * b
    let x0_zipup = b_zipup.scale(AnyScalar::new_real(0.5))?;
    let x0_fit = b_fit.scale(AnyScalar::new_real(0.5))?;

    println!("||x0_zipup|| = {:.6}", x0_zipup.norm()?);
    println!("||x0_fit||   = {:.6}", x0_fit.norm()?);

    // Check index structure of x0
    println!("\n=== Index structure ===");
    for site in 0..n {
        let t_zipup = x0_zipup.tensor(site)?;
        let t_fit = x0_fit.tensor(site)?;
        println!(
            "Site {} x0_zipup indices: {:?}",
            site,
            t_zipup
                .indices()
                .iter()
                .map(|i| (i.id(), i.dim()))
                .collect::<Vec<_>>()
        );
        println!(
            "Site {} x0_fit indices:   {:?}",
            site,
            t_fit
                .indices()
                .iter()
                .map(|i| (i.id(), i.dim()))
                .collect::<Vec<_>>()
        );
    }

    // Now manually do one Arnoldi step and solution update
    println!("\n=== Manual Arnoldi (zipup) ===");
    let ax0_zipup = apply_with_zipup(&pauli_x, &x0_zipup, &indices)?;
    let r0_zipup = b_zipup.axpby(
        AnyScalar::new_real(1.0),
        &ax0_zipup,
        AnyScalar::new_real(-1.0),
    )?;
    let beta_zipup = r0_zipup.norm()?;
    let v0_zipup = r0_zipup.scale(AnyScalar::new_real(1.0 / beta_zipup))?;

    println!("r0_zipup bond dims: {:?}", r0_zipup.bond_dims());
    println!("v0_zipup bond dims: {:?}", v0_zipup.bond_dims());

    // Check if v0 indices match x0 indices (important for axpby!)
    println!("\n=== Check index compatibility ===");
    println!(
        "v0_zipup Site 0 indices: {:?}",
        v0_zipup
            .tensor(0)?
            .indices()
            .iter()
            .map(|i| i.id())
            .collect::<Vec<_>>()
    );
    println!(
        "x0_zipup Site 0 indices: {:?}",
        x0_zipup
            .tensor(0)?
            .indices()
            .iter()
            .map(|i| i.id())
            .collect::<Vec<_>>()
    );

    // Check if they share physical indices (required for axpby)
    let v0_phys_ids: std::collections::HashSet<_> = v0_zipup
        .tensor(0)?
        .indices()
        .iter()
        .filter(|i| i.dim() == 2)
        .map(|i| i.id())
        .collect();
    let x0_phys_ids: std::collections::HashSet<_> = x0_zipup
        .tensor(0)?
        .indices()
        .iter()
        .filter(|i| i.dim() == 2)
        .map(|i| i.id())
        .collect();
    println!("v0 physical index IDs: {:?}", v0_phys_ids);
    println!("x0 physical index IDs: {:?}", x0_phys_ids);
    println!(
        "Shared physical indices: {:?}",
        v0_phys_ids.intersection(&x0_phys_ids).collect::<Vec<_>>()
    );

    // Manual solution update: x_new = x0 + y1 * v0
    // Using y1 = something (we'll use 1.0 for testing)
    println!("\n=== Manual solution update ===");
    let scaled_v0 = v0_zipup.scale(AnyScalar::new_real(1.0))?;
    println!("||scaled_v0|| = {:.6}", scaled_v0.norm()?);

    // Try axpby: x_new = 1.0 * x0 + 1.0 * scaled_v0
    let x_new_zipup = x0_zipup.axpby(
        AnyScalar::new_real(1.0),
        &scaled_v0,
        AnyScalar::new_real(1.0),
    )?;
    println!("||x_new|| after axpby = {:.6}", x_new_zipup.norm()?);

    // Check if x_new has expected indices
    println!(
        "x_new Site 0 indices: {:?}",
        x_new_zipup
            .tensor(0)?
            .indices()
            .iter()
            .map(|i| (i.id(), i.dim()))
            .collect::<Vec<_>>()
    );
    println!("x_new bond dims: {:?}", x_new_zipup.bond_dims());

    // Compare: what if we use fit instead?
    println!("\n=== Same with fit ===");
    let ax0_fit = apply_with_fit(&pauli_x, &x0_fit, &indices)?;
    let r0_fit = b_fit.axpby(
        AnyScalar::new_real(1.0),
        &ax0_fit,
        AnyScalar::new_real(-1.0),
    )?;
    let beta_fit = r0_fit.norm()?;
    let v0_fit = r0_fit.scale(AnyScalar::new_real(1.0 / beta_fit))?;

    println!("r0_fit bond dims: {:?}", r0_fit.bond_dims());
    println!("v0_fit bond dims: {:?}", v0_fit.bond_dims());

    let v0_fit_phys_ids: std::collections::HashSet<_> = v0_fit
        .tensor(0)?
        .indices()
        .iter()
        .filter(|i| i.dim() == 2)
        .map(|i| i.id())
        .collect();
    let x0_fit_phys_ids: std::collections::HashSet<_> = x0_fit
        .tensor(0)?
        .indices()
        .iter()
        .filter(|i| i.dim() == 2)
        .map(|i| i.id())
        .collect();
    println!("v0_fit physical index IDs: {:?}", v0_fit_phys_ids);
    println!("x0_fit physical index IDs: {:?}", x0_fit_phys_ids);
    println!(
        "Shared physical indices: {:?}",
        v0_fit_phys_ids
            .intersection(&x0_fit_phys_ids)
            .collect::<Vec<_>>()
    );

    let scaled_v0_fit = v0_fit.scale(AnyScalar::new_real(1.0))?;
    let x_new_fit = x0_fit.axpby(
        AnyScalar::new_real(1.0),
        &scaled_v0_fit,
        AnyScalar::new_real(1.0),
    )?;
    println!("||x_new_fit|| after axpby = {:.6}", x_new_fit.norm()?);

    // Key check: do the resulting MPOs have the same structure?
    println!("\n=== Result comparison ===");
    println!("x_new_zipup bond dims: {:?}", x_new_zipup.bond_dims());
    println!("x_new_fit bond dims:   {:?}", x_new_fit.bond_dims());

    // Check actual values
    println!("\nSite 0 data comparison:");
    println!("x_new_zipup: {:?}", x_new_zipup.tensor(0)?.to_vec::<f64>()?);
    println!("x_new_fit:   {:?}", x_new_fit.tensor(0)?.to_vec::<f64>()?);

    Ok(())
}

/// Test effect of truncation on Arnoldi orthogonality
fn test_truncation_effect() -> anyhow::Result<()> {
    println!("\n========================================");
    println!("  TRUNCATION EFFECT ON ARNOLDI");
    println!("========================================\n");

    let n = 3;
    let indices = SharedIndices::new(n, 2);
    let identity = create_identity_mpo(&indices)?;
    let pauli_x = create_pauli_x_operator(&indices)?;

    let x_true = identity.clone();
    let b = apply_with_zipup(&pauli_x, &x_true, &indices)?;
    let x0 = b.scale(AnyScalar::new_real(0.5))?;

    // Compute A(x0)
    let ax0 = apply_with_zipup(&pauli_x, &x0, &indices)?;

    // r0 = b - A(x0)
    let mut r0 = b.axpby(AnyScalar::new_real(1.0), &ax0, AnyScalar::new_real(-1.0))?;

    println!("=== Before truncation ===");
    println!("r0 norm: {:.6}", r0.norm()?);
    println!("r0 bond dims: {:?}", r0.bond_dims());

    // Apply truncation
    let truncate_opts = TruncateOptions::svd()
        .with_svd_policy(tensor4all_core::SvdTruncationPolicy::new(1e-8))
        .with_max_bond_dim(20);
    r0.truncate(&truncate_opts)?;

    println!("\n=== After truncation ===");
    println!("r0 norm: {:.6}", r0.norm()?);
    println!("r0 bond dims: {:?}", r0.bond_dims());

    let r_norm = r0.norm()?;
    let mut v0 = r0.scale(AnyScalar::new_real(1.0 / r_norm))?;
    v0.truncate(&truncate_opts)?;
    let v0_norm = v0.norm()?;
    v0 = v0.scale(AnyScalar::new_real(1.0 / v0_norm))?;

    println!("\n=== v0 after truncation and renormalization ===");
    println!("v0 norm: {:.6}", v0.norm()?);
    println!("v0 bond dims: {:?}", v0.bond_dims());

    // w = A(v0)
    let w = apply_with_zipup(&pauli_x, &v0, &indices)?;
    println!("\n=== w = A(v0) ===");
    println!("w norm: {:.6}", w.norm()?);

    // h10 = <v0, w>
    let h10 = v0.inner(&w)?;
    println!("h10 = <v0, w>: {:?}", h10);
    let h10_val = h10.real();

    // w_orth = w - h10 * v0
    let mut w_orth = w.axpby(AnyScalar::new_real(1.0), &v0, AnyScalar::new_real(-h10_val))?;
    println!("\n=== w_orth before truncation ===");
    println!("w_orth norm: {:.6}", w_orth.norm()?);
    println!("<v0, w_orth>: {:?}", v0.inner(&w_orth)?);

    // Truncate w_orth
    w_orth.truncate(&truncate_opts)?;
    println!("\n=== w_orth after truncation ===");
    println!("w_orth norm: {:.6}", w_orth.norm()?);
    println!("<v0, w_orth>: {:?}", v0.inner(&w_orth)?); // Should still be ~0 if truncation preserves orthogonality

    let h20 = w_orth.norm()?;
    let mut v1 = w_orth.scale(AnyScalar::new_real(1.0 / h20))?;
    v1.truncate(&truncate_opts)?;
    let v1_norm = v1.norm()?;
    v1 = v1.scale(AnyScalar::new_real(1.0 / v1_norm))?;

    println!("\n=== v1 after truncation and renormalization ===");
    println!("v1 norm: {:.6}", v1.norm()?);
    println!("<v0, v1>: {:?}", v0.inner(&v1)?); // This should be ~0, but may not be!

    // Now check if Arnoldi relation holds
    // A*v0 should = h10*v0 + h20*v1
    let av0_computed = v0.axpby(AnyScalar::new_real(h10_val), &v1, AnyScalar::new_real(h20))?;
    let av0_actual = apply_with_zipup(&pauli_x, &v0, &indices)?;
    let av0_diff = av0_actual.axpby(
        AnyScalar::new_real(1.0),
        &av0_computed,
        AnyScalar::new_real(-1.0),
    )?;

    println!("\n=== Arnoldi relation check ===");
    println!("||A*v0 - (h10*v0 + h20*v1)||: {:.6e}", av0_diff.norm()?);

    // Second iteration
    let w2 = apply_with_zipup(&pauli_x, &v1, &indices)?;
    let h11 = v0.inner(&w2)?;
    let h21 = v1.inner(&w2)?;
    println!("\n=== Second iteration ===");
    println!("h11 = <v0, A*v1>: {:?}", h11);
    println!("h21 = <v1, A*v1>: {:?}", h21);
    let h11_val = h11.real();
    let h21_val = h21.real();

    // w2_orth = w2 - h11*v0 - h21*v1
    let temp = w2.axpby(AnyScalar::new_real(1.0), &v0, AnyScalar::new_real(-h11_val))?;
    let mut w2_orth = temp.axpby(AnyScalar::new_real(1.0), &v1, AnyScalar::new_real(-h21_val))?;

    println!("w2_orth norm before trunc: {:.6}", w2_orth.norm()?);
    w2_orth.truncate(&truncate_opts)?;
    println!("w2_orth norm after trunc:  {:.6}", w2_orth.norm()?);

    // Check orthogonality after truncation
    println!("<v0, w2_orth>: {:?}", v0.inner(&w2_orth)?);
    println!("<v1, w2_orth>: {:?}", v1.inner(&w2_orth)?);

    Ok(())
}

/// Detailed debug for N=3 to find root cause
fn detailed_debug_n3() -> anyhow::Result<()> {
    let n = 3;
    let indices = SharedIndices::new(n, 2);

    let identity = create_identity_mpo(&indices)?;
    let pauli_x = create_pauli_x_operator(&indices)?;

    println!("=== Step 1: Check basic contraction ===");
    let result_zipup = apply_with_zipup(&pauli_x, &identity, &indices)?;
    let result_fit = apply_with_fit(&pauli_x, &identity, &indices)?;

    println!("zipup bond dims: {:?}", result_zipup.bond_dims());
    println!("fit bond dims:   {:?}", result_fit.bond_dims());

    // Print tensor data at each site
    for site in 0..n {
        println!("\n--- Site {} tensor data ---", site);
        let t_zipup = result_zipup.tensor(site)?;
        let t_fit = result_fit.tensor(site)?;

        println!(
            "zipup shape: {:?}",
            t_zipup
                .indices()
                .iter()
                .map(|i| i.dim())
                .collect::<Vec<_>>()
        );
        println!(
            "fit shape:   {:?}",
            t_fit.indices().iter().map(|i| i.dim()).collect::<Vec<_>>()
        );

        // Print index IDs to compare
        println!(
            "zipup index IDs: {:?}",
            t_zipup.indices().iter().map(|i| i.id()).collect::<Vec<_>>()
        );
        println!(
            "fit index IDs:   {:?}",
            t_fit.indices().iter().map(|i| i.id()).collect::<Vec<_>>()
        );

        // Compare dense data
        let data_zipup = t_zipup.to_vec::<f64>()?;
        let data_fit = t_fit.to_vec::<f64>()?;

        println!("zipup data: {:?}", data_zipup);
        println!("fit data:   {:?}", data_fit);

        let diff: f64 = data_zipup
            .iter()
            .zip(data_fit.iter())
            .map(|(a, b): (&f64, &f64)| (a - b).abs())
            .sum();
        println!("Sum of absolute differences: {:.6e}", diff);
    }

    // Check if indices are in same order
    println!("\n=== Check index ordering ===");
    println!(
        "Expected input indices:  {:?}",
        indices.inputs.iter().map(|i| i.id()).collect::<Vec<_>>()
    );
    println!(
        "Expected output indices: {:?}",
        indices.outputs.iter().map(|i| i.id()).collect::<Vec<_>>()
    );
    println!(
        "Expected bond indices:   {:?}",
        indices.bonds.iter().map(|i| i.id()).collect::<Vec<_>>()
    );

    // Debug contract result
    debug_contract_result(&pauli_x, &identity, &indices)?;

    println!("\n=== Step 2: Check inner product computation ===");
    // Test inner product between zipup and fit results
    let inner_zz = result_zipup.inner(&result_zipup)?;
    let inner_ff = result_fit.inner(&result_fit)?;
    let inner_zf = result_zipup.inner(&result_fit)?;
    let inner_fz = result_fit.inner(&result_zipup)?;

    println!("<zipup, zipup> = {:?}", inner_zz);
    println!("<fit, fit>     = {:?}", inner_ff);
    println!("<zipup, fit>   = {:?}", inner_zf);
    println!("<fit, zipup>   = {:?}", inner_fz);

    println!("\n=== Step 3: Simulate GMRES step by step ===");
    // b = σ_x (using zipup as ground truth)
    let b = apply_with_zipup(&pauli_x, &identity, &indices)?;
    let x_true = identity.clone();

    // x0 = 0.5 * b
    let x0 = b.scale(AnyScalar::new_real(0.5))?;
    println!("x0 norm: {:.6}", x0.norm()?);

    // r0 = b - A(x0) for both methods
    let ax0_zipup = apply_with_zipup(&pauli_x, &x0, &indices)?;
    let ax0_fit = apply_with_fit(&pauli_x, &x0, &indices)?;

    println!("\nA(x0) with zipup norm: {:.6}", ax0_zipup.norm()?);
    println!("A(x0) with fit norm:   {:.6}", ax0_fit.norm()?);

    // Check if A(x0) results are the same
    let ax0_diff = ax0_zipup.axpby(
        AnyScalar::new_real(1.0),
        &ax0_fit,
        AnyScalar::new_real(-1.0),
    )?;
    println!("||A(x0)_zipup - A(x0)_fit||: {:.6e}", ax0_diff.norm()?);

    // r0 = b - A(x0)
    let r0_zipup = b.axpby(
        AnyScalar::new_real(1.0),
        &ax0_zipup,
        AnyScalar::new_real(-1.0),
    )?;
    let r0_fit = b.axpby(
        AnyScalar::new_real(1.0),
        &ax0_fit,
        AnyScalar::new_real(-1.0),
    )?;

    println!("\nr0_zipup norm: {:.6}", r0_zipup.norm()?);
    println!("r0_fit norm:   {:.6}", r0_fit.norm()?);

    // Check inner products - this is crucial for GMRES
    println!("\n=== Step 4: Check Arnoldi process inner products ===");

    // v1 = r0 / ||r0|| (normalized)
    let beta_zipup = r0_zipup.norm()?;
    let beta_fit = r0_fit.norm()?;
    let v1_zipup = r0_zipup.scale(AnyScalar::new_real(1.0 / beta_zipup))?;
    let v1_fit = r0_fit.scale(AnyScalar::new_real(1.0 / beta_fit))?;

    println!("v1_zipup norm: {:.6}", v1_zipup.norm()?);
    println!("v1_fit norm:   {:.6}", v1_fit.norm()?);

    // w = A(v1)
    let w_zipup = apply_with_zipup(&pauli_x, &v1_zipup, &indices)?;
    let w_fit = apply_with_fit(&pauli_x, &v1_fit, &indices)?;

    println!("\nw_zipup = A(v1_zipup) norm: {:.6}", w_zipup.norm()?);
    println!("w_fit = A(v1_fit) norm:     {:.6}", w_fit.norm()?);

    // h11 = <w, v1> - this is where the issue might be
    let h11_zipup = w_zipup.inner(&v1_zipup)?;
    let h11_fit = w_fit.inner(&v1_fit)?;

    println!("\nh11_zipup = <w, v1>: {:?}", h11_zipup);
    println!("h11_fit = <w, v1>:   {:?}", h11_fit);

    // Check cross inner products
    let cross1 = w_zipup.inner(&v1_fit)?;
    let cross2 = w_fit.inner(&v1_zipup)?;
    println!("<w_zipup, v1_fit>: {:?}", cross1);
    println!("<w_fit, v1_zipup>: {:?}", cross2);

    // v2_tilde = w - h11 * v1
    let h11_val_zipup = h11_zipup.real();
    let h11_val_fit = h11_fit.real();

    let v2_tilde_zipup = w_zipup.axpby(
        AnyScalar::new_real(1.0),
        &v1_zipup,
        AnyScalar::new_real(-h11_val_zipup),
    )?;
    let v2_tilde_fit = w_fit.axpby(
        AnyScalar::new_real(1.0),
        &v1_fit,
        AnyScalar::new_real(-h11_val_fit),
    )?;

    let h21_zipup = v2_tilde_zipup.norm()?;
    let h21_fit = v2_tilde_fit.norm()?;

    println!("\nh21_zipup = ||v2_tilde||: {:.6}", h21_zipup);
    println!("h21_fit = ||v2_tilde||:   {:.6}", h21_fit);

    // Normalize v2
    let v2_zipup = v2_tilde_zipup.scale(AnyScalar::new_real(1.0 / h21_zipup))?;
    let v2_fit = v2_tilde_fit.scale(AnyScalar::new_real(1.0 / h21_fit))?;

    println!("\nv2_zipup norm: {:.6}", v2_zipup.norm()?);
    println!("v2_fit norm:   {:.6}", v2_fit.norm()?);

    // Check orthogonality <v1, v2>
    let v1v2_zipup = v1_zipup.inner(&v2_zipup)?;
    let v1v2_fit = v1_fit.inner(&v2_fit)?;
    println!("\n<v1, v2> zipup: {:?}", v1v2_zipup);
    println!("<v1, v2> fit:   {:?}", v1v2_fit);

    println!("\n=== Step 5: Check second Arnoldi iteration ===");

    // w2 = A(v2)
    let w2_zipup = apply_with_zipup(&pauli_x, &v2_zipup, &indices)?;
    let w2_fit = apply_with_fit(&pauli_x, &v2_fit, &indices)?;

    println!("w2_zipup = A(v2) norm: {:.6}", w2_zipup.norm()?);
    println!("w2_fit = A(v2) norm:   {:.6}", w2_fit.norm()?);

    // h12 = <w2, v1>, h22 = <w2, v2>
    let h12_zipup = w2_zipup.inner(&v1_zipup)?;
    let h12_fit = w2_fit.inner(&v1_fit)?;
    let h22_zipup = w2_zipup.inner(&v2_zipup)?;
    let h22_fit = w2_fit.inner(&v2_fit)?;

    println!("\nh12_zipup = <w2, v1>: {:?}", h12_zipup);
    println!("h12_fit = <w2, v1>:   {:?}", h12_fit);
    println!("h22_zipup = <w2, v2>: {:?}", h22_zipup);
    println!("h22_fit = <w2, v2>:   {:?}", h22_fit);

    println!("\n=== Step 6: Verify solution directly ===");
    // For Pauli-X: A(x) = σ_x * x
    // Solve σ_x * x = σ_x
    // Solution: x = I

    // After 2 GMRES iterations, compute the minimizer
    // Hessenberg matrix H = [[h11, h12], [h21, h22], [0, h32]]
    // We minimize ||beta*e1 - H*y||

    println!("Hessenberg matrix (zipup):");
    println!("  H = [[{:.6}, {:.6}],", h11_val_zipup, h12_zipup.real());
    println!("       [{:.6}, {:.6}]]", h21_zipup, h22_zipup.real());

    println!("Hessenberg matrix (fit):");
    println!("  H = [[{:.6}, {:.6}],", h11_val_fit, h12_fit.real());
    println!("       [{:.6}, {:.6}]]", h21_fit, h22_fit.real());

    println!("\nbeta (zipup): {:.6}", beta_zipup);
    println!("beta (fit):   {:.6}", beta_fit);

    Ok(())
}

/// Test GMRES with both fit and zipup
fn test_gmres_with_both(
    op: &TensorTrain,
    x_true: &TensorTrain,
    indices: &SharedIndices,
) -> anyhow::Result<()> {
    // b = A(x_true) = σ_x * I (using zipup for reference)
    let b = apply_with_zipup(op, x_true, indices)?;
    println!("b norm: {:.6}", b.norm()?);
    println!("x_true norm: {:.6}", x_true.norm()?);

    // Initial guess: 0.5 * b
    let x0 = b.scale(AnyScalar::new_real(0.5))?;

    let options = GmresOptions {
        max_iter: 10,
        rtol: 1e-8,
        max_restarts: 1,
        verbose: true, // Enable verbose to see iteration details
        check_true_residual: false,
    };

    // Test WITHOUT truncation first
    let no_truncate_fn = |_x: &mut TensorTrain| -> anyhow::Result<()> { Ok(()) };

    println!("\n--- GMRES WITHOUT truncation ---");

    // GMRES with zipup, no truncation
    let apply_a_zipup =
        |x: &TensorTrain| -> anyhow::Result<TensorTrain> { apply_with_zipup(op, x, indices) };
    let result_zipup_notrunc =
        gmres_with_truncation(&apply_a_zipup, &b, &x0, &options, no_truncate_fn)?;

    // GMRES with fit, no truncation
    let apply_a_fit =
        |x: &TensorTrain| -> anyhow::Result<TensorTrain> { apply_with_fit(op, x, indices) };
    let result_fit_notrunc =
        gmres_with_truncation(&apply_a_fit, &b, &x0, &options, no_truncate_fn)?;

    // Check no-truncation results
    let ax_zipup_notrunc = apply_with_zipup(op, &result_zipup_notrunc.solution, indices)?;
    let ax_fit_notrunc = apply_with_zipup(op, &result_fit_notrunc.solution, indices)?;
    let r_zipup_notrunc =
        ax_zipup_notrunc.axpby(AnyScalar::new_real(1.0), &b, AnyScalar::new_real(-1.0))?;
    let r_fit_notrunc =
        ax_fit_notrunc.axpby(AnyScalar::new_real(1.0), &b, AnyScalar::new_real(-1.0))?;
    let b_norm = b.norm()?;

    println!(
        "[zipup no-trunc] Actual residual: {:.6e}, ||x_sol||: {:.6}",
        r_zipup_notrunc.norm()? / b_norm,
        result_zipup_notrunc.solution.norm()?
    );
    println!(
        "[fit no-trunc]   Actual residual: {:.6e}, ||x_sol||: {:.6}",
        r_fit_notrunc.norm()? / b_norm,
        result_fit_notrunc.solution.norm()?
    );

    println!("\n--- GMRES WITH truncation ---");

    let truncate_opts = TruncateOptions::svd()
        .with_svd_policy(tensor4all_core::SvdTruncationPolicy::new(1e-8))
        .with_max_bond_dim(20);
    let truncate_fn = |x: &mut TensorTrain| -> anyhow::Result<()> {
        x.truncate(&truncate_opts)?;
        Ok(())
    };

    // GMRES with zipup
    let result_zipup = gmres_with_truncation(&apply_a_zipup, &b, &x0, &options, truncate_fn)?;

    // GMRES with fit
    let result_fit = gmres_with_truncation(&apply_a_fit, &b, &x0, &options, truncate_fn)?;

    // Check results
    println!(
        "[zipup] Converged: {}, Iters: {}, Reported residual: {:.6e}",
        result_zipup.converged, result_zipup.iterations, result_zipup.residual_norm
    );
    println!(
        "[fit]   Converged: {}, Iters: {}, Reported residual: {:.6e}",
        result_fit.converged, result_fit.iterations, result_fit.residual_norm
    );

    // Compute actual residuals
    let ax_zipup = apply_with_zipup(op, &result_zipup.solution, indices)?;
    let ax_fit = apply_with_zipup(op, &result_fit.solution, indices)?; // Use zipup for fair comparison

    let r_zipup = ax_zipup.axpby(AnyScalar::new_real(1.0), &b, AnyScalar::new_real(-1.0))?;
    let r_fit = ax_fit.axpby(AnyScalar::new_real(1.0), &b, AnyScalar::new_real(-1.0))?;

    let b_norm = b.norm()?;
    println!(
        "[zipup] Actual residual |Ax-b|/|b|: {:.6e}",
        r_zipup.norm()? / b_norm
    );
    println!(
        "[fit]   Actual residual |Ax-b|/|b|: {:.6e}",
        r_fit.norm()? / b_norm
    );

    // Check inner products with x_true
    let inner_sol_true_zipup = result_zipup.solution.inner(x_true)?;
    let inner_sol_true_fit = result_fit.solution.inner(x_true)?;
    let expected_inner = x_true.inner(x_true)?;
    println!(
        "[zipup] <x_sol, x_true>: {:?} (expected: {:?})",
        inner_sol_true_zipup, expected_inner
    );
    println!(
        "[fit]   <x_sol, x_true>: {:?} (expected: {:?})",
        inner_sol_true_fit, expected_inner
    );

    // Check solution norms
    println!("[zipup] ||x_sol||: {:.6}", result_zipup.solution.norm()?);
    println!("[fit]   ||x_sol||: {:.6}", result_fit.solution.norm()?);
    println!("Expected ||x_true||: {:.6}", x_true.norm()?);

    Ok(())
}

/// Simulate GMRES-like operations with fit vs zipup
fn test_gmres_like_operations(
    op: &TensorTrain,
    b: &TensorTrain,
    indices: &SharedIndices,
) -> anyhow::Result<()> {
    // x_true = I (identity)
    // A(x) = σ_x * x
    // So A(I) = σ_x, meaning b should be σ_x
    // And we want to solve A(x) = b, i.e., find x such that σ_x * x = σ_x
    // The solution is x = I

    let x_true = b.clone(); // x_true = I

    // b = A(x_true) = σ_x * I = σ_x
    let b_zipup = apply_with_zipup(op, &x_true, indices)?;
    let b_fit = apply_with_fit(op, &x_true, indices)?;

    println!("b_zipup norm: {:.6}", b_zipup.norm()?);
    println!("b_fit norm: {:.6}", b_fit.norm()?);

    // Initial guess: x0 = 0.5 * b
    let x0_zipup = b_zipup.scale(AnyScalar::new_real(0.5))?;
    let x0_fit = b_fit.scale(AnyScalar::new_real(0.5))?;

    // Compute A(x0)
    let ax0_zipup = apply_with_zipup(op, &x0_zipup, indices)?;
    let ax0_fit = apply_with_fit(op, &x0_fit, indices)?;

    println!("A(x0) zipup norm: {:.6}", ax0_zipup.norm()?);
    println!("A(x0) fit norm: {:.6}", ax0_fit.norm()?);

    // Compute residual r0 = A(x0) - b
    let r0_zipup = ax0_zipup.axpby(
        AnyScalar::new_real(1.0),
        &b_zipup,
        AnyScalar::new_real(-1.0),
    )?;
    let r0_fit = ax0_fit.axpby(AnyScalar::new_real(1.0), &b_fit, AnyScalar::new_real(-1.0))?;

    println!("||r0|| zipup: {:.6e}", r0_zipup.norm()?);
    println!("||r0|| fit: {:.6e}", r0_fit.norm()?);

    // Compute inner products
    let inner_r0_r0_zipup = r0_zipup.inner(&r0_zipup)?;
    let inner_r0_r0_fit = r0_fit.inner(&r0_fit)?;
    println!("<r0, r0> zipup: {:?}", inner_r0_r0_zipup);
    println!("<r0, r0> fit: {:?}", inner_r0_r0_fit);

    // Simulate one GMRES iteration: v1 = A(r0) / ||A(r0)||
    let ar0_zipup = apply_with_zipup(op, &r0_zipup, indices)?;
    let ar0_fit = apply_with_fit(op, &r0_fit, indices)?;

    println!("||A(r0)|| zipup: {:.6e}", ar0_zipup.norm()?);
    println!("||A(r0)|| fit: {:.6e}", ar0_fit.norm()?);

    // Check <A(r0), r0>
    let inner_ar0_r0_zipup = ar0_zipup.inner(&r0_zipup)?;
    let inner_ar0_r0_fit = ar0_fit.inner(&r0_fit)?;
    println!("<A(r0), r0> zipup: {:?}", inner_ar0_r0_zipup);
    println!("<A(r0), r0> fit: {:?}", inner_ar0_r0_fit);

    // Check index structure
    println!("\n--- Index structure check ---");
    println!(
        "b_zipup indices at site 0: {:?}",
        b_zipup.tensor(0)?.indices()
    );
    println!("b_fit indices at site 0: {:?}", b_fit.tensor(0)?.indices());

    Ok(())
}

/// Create identity MPO
fn create_identity_mpo(indices: &SharedIndices) -> anyhow::Result<TensorTrain> {
    let n = indices.inputs.len();
    let mut tensors = Vec::with_capacity(n);

    for i in 0..n {
        let in_dim = indices.inputs[i].dim();
        let out_dim = indices.outputs[i].dim();
        let in_idx = indices.inputs[i].clone();
        let out_idx = indices.outputs[i].clone();

        let mut data = vec![0.0_f64; in_dim * out_dim];
        for j in 0..in_dim.min(out_dim) {
            data[j * out_dim + j] = 1.0;
        }

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

/// Create Pauli-X operator MPO
fn create_pauli_x_operator(indices: &SharedIndices) -> anyhow::Result<TensorTrain> {
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

/// Apply operator using zipup
fn apply_with_zipup(
    op: &TensorTrain,
    mpo: &TensorTrain,
    indices: &SharedIndices,
) -> anyhow::Result<TensorTrain> {
    let options = ContractOptions::zipup()
        .with_svd_policy(tensor4all_core::SvdTruncationPolicy::new(1e-10))
        .with_max_bond_dim(50);

    let result = op
        .contract(mpo, &options)
        .map_err(|e| anyhow::anyhow!("{}", e))?;

    let result = result.replace_indices(&indices.operator_outputs, &indices.inputs)?;
    Ok(result)
}

/// Apply operator using fit
fn apply_with_fit(
    op: &TensorTrain,
    mpo: &TensorTrain,
    indices: &SharedIndices,
) -> anyhow::Result<TensorTrain> {
    let options = ContractOptions::fit()
        .with_nhalfsweeps(4)
        .with_svd_policy(tensor4all_core::SvdTruncationPolicy::new(1e-10))
        .with_max_bond_dim(50);

    let result = op
        .contract(mpo, &options)
        .map_err(|e| anyhow::anyhow!("{}", e))?;

    let result = result.replace_indices(&indices.operator_outputs, &indices.inputs)?;
    Ok(result)
}

/// Debug: Check contract result before and after replace_indices
fn debug_contract_result(
    op: &TensorTrain,
    mpo: &TensorTrain,
    indices: &SharedIndices,
) -> anyhow::Result<()> {
    println!("\n=== DEBUG: Contract result before replace_indices ===");

    let options_zipup = ContractOptions::zipup()
        .with_svd_policy(tensor4all_core::SvdTruncationPolicy::new(1e-10))
        .with_max_bond_dim(50);
    let options_fit = ContractOptions::fit()
        .with_nhalfsweeps(4)
        .with_svd_policy(tensor4all_core::SvdTruncationPolicy::new(1e-10))
        .with_max_bond_dim(50);

    let raw_zipup = op
        .contract(mpo, &options_zipup)
        .map_err(|e| anyhow::anyhow!("{}", e))?;
    let raw_fit = op
        .contract(mpo, &options_fit)
        .map_err(|e| anyhow::anyhow!("{}", e))?;

    println!(
        "operator_outputs IDs: {:?}",
        indices
            .operator_outputs
            .iter()
            .map(|i| i.id())
            .collect::<Vec<_>>()
    );

    for site in 0..3 {
        println!("\n--- Raw result Site {} ---", site);
        let t_zipup = raw_zipup.tensor(site)?;
        let t_fit = raw_fit.tensor(site)?;

        println!(
            "zipup indices: {:?}",
            t_zipup
                .indices()
                .iter()
                .map(|i| (i.id(), i.dim()))
                .collect::<Vec<_>>()
        );
        println!(
            "fit indices:   {:?}",
            t_fit
                .indices()
                .iter()
                .map(|i| (i.id(), i.dim()))
                .collect::<Vec<_>>()
        );
    }

    // Check if zipup bonds are same as fit bonds
    println!("\n--- Bond index comparison ---");
    println!("zipup link indices:");
    for site in 0..2 {
        let t = raw_zipup.tensor(site)?;
        let link_idx = t.indices().iter().find(|i| {
            i.dim() == 1 && site == 0 || t.indices().last().map(|x| x.id()) == Some(i.id())
        });
        if let Some(idx) = link_idx {
            println!("  Site {} right bond: {:?}", site, idx.id());
        }
    }

    println!("fit link indices:");
    for site in 0..2 {
        let t = raw_fit.tensor(site)?;
        for idx in t.indices() {
            if idx.dim() == 1 {
                println!("  Site {} bond dim=1: {:?}", site, idx.id());
            }
        }
    }

    // KEY INSIGHT: Check index ordering convention
    println!("\n=== KEY: Index ordering analysis ===");
    println!("TensorTrain expects: [left_bond, physical..., right_bond]");
    println!("");
    for site in 0..3 {
        let t_zipup = raw_zipup.tensor(site)?;
        let t_fit = raw_fit.tensor(site)?;

        let zipup_dims: Vec<_> = t_zipup.indices().iter().map(|i| i.dim()).collect();
        let fit_dims: Vec<_> = t_fit.indices().iter().map(|i| i.dim()).collect();

        let zipup_order = classify_indices(&zipup_dims);
        let fit_order = classify_indices(&fit_dims);

        println!(
            "Site {}: zipup [{}] vs fit [{}]",
            site, zipup_order, fit_order
        );
    }

    // Check link indices (common between adjacent tensors)
    println!("\n=== Link indices check (common between adjacent) ===");
    println!(
        "zipup link_indices: {:?}",
        raw_zipup
            .link_indices()
            .iter()
            .map(|i| i.id())
            .collect::<Vec<_>>()
    );
    println!(
        "fit link_indices:   {:?}",
        raw_fit
            .link_indices()
            .iter()
            .map(|i| i.id())
            .collect::<Vec<_>>()
    );

    // Check if Site 1 has duplicate bond IDs
    println!("\n=== Site 1 bond ID analysis ===");
    let t1_fit = raw_fit.tensor(1)?;
    let bond_ids: Vec<_> = t1_fit
        .indices()
        .iter()
        .filter(|i| i.dim() == 1)
        .map(|i| i.id())
        .collect();
    println!("fit Site 1 bond IDs: {:?}", bond_ids);
    let unique_bonds: std::collections::HashSet<_> = bond_ids.iter().collect();
    println!(
        "Unique bond IDs count: {} (should be 2)",
        unique_bonds.len()
    );
    if bond_ids.len() != unique_bonds.len() {
        println!("WARNING: Duplicate bond IDs detected!");
    }

    // Check truncation effect
    println!("\n=== Truncation effect on index ordering ===");
    let result_zipup = apply_with_zipup(op, mpo, indices)?;
    let result_fit = apply_with_fit(op, mpo, indices)?;

    let mut result_zipup_trunc = result_zipup.clone();
    let mut result_fit_trunc = result_fit.clone();

    let trunc_opts = TruncateOptions::svd()
        .with_svd_policy(tensor4all_core::SvdTruncationPolicy::new(1e-8))
        .with_max_bond_dim(20);
    result_zipup_trunc.truncate(&trunc_opts)?;
    result_fit_trunc.truncate(&trunc_opts)?;

    println!("Before truncation:");
    println!(
        "  zipup Site 1: {:?}",
        result_zipup
            .tensor(1)?
            .indices()
            .iter()
            .map(|i| i.dim())
            .collect::<Vec<_>>()
    );
    println!(
        "  fit Site 1:   {:?}",
        result_fit
            .tensor(1)?
            .indices()
            .iter()
            .map(|i| i.dim())
            .collect::<Vec<_>>()
    );

    println!("After truncation:");
    println!(
        "  zipup Site 1: {:?}",
        result_zipup_trunc
            .tensor(1)?
            .indices()
            .iter()
            .map(|i| i.dim())
            .collect::<Vec<_>>()
    );
    println!(
        "  fit Site 1:   {:?}",
        result_fit_trunc
            .tensor(1)?
            .indices()
            .iter()
            .map(|i| i.dim())
            .collect::<Vec<_>>()
    );

    // Check if norms are preserved
    println!("\nNorms:");
    println!(
        "  zipup before: {:.6}, after: {:.6}",
        result_zipup.norm()?,
        result_zipup_trunc.norm()?
    );
    println!(
        "  fit before:   {:.6}, after: {:.6}",
        result_fit.norm()?,
        result_fit_trunc.norm()?
    );

    Ok(())
}

fn classify_indices(dims: &[usize]) -> String {
    dims.iter()
        .map(|&d| if d == 1 { "B" } else { "P" }) // B=bond, P=physical
        .collect::<Vec<_>>()
        .join("-")
}
