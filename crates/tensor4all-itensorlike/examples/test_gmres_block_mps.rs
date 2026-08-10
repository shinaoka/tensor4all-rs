#![allow(clippy::needless_borrows_for_generic_args)]
#![allow(clippy::identity_op)]
//! Test: GMRES solver with BlockTensor<TensorTrain> (block MPS)
//!
//! Tests GMRES with block vectors where each block is an MPS.
//!
//! Run:
//!   cargo run -p tensor4all-itensorlike --example test_gmres_block_mps --release

use std::time::Instant;

use num_complex::Complex64;
use tensor4all_core::block_tensor::BlockTensor;
use tensor4all_core::krylov::{
    gmres_with_truncation, restart_gmres_with_truncation, GmresOptions, RestartGmresOptions,
};
use tensor4all_core::TensorVectorSpace;
use tensor4all_core::{AnyScalar, DynIndex, IndexLike, TensorDynLen, TensorIndex};
use tensor4all_itensorlike::{ContractOptions, TensorTrain, TruncateOptions};

// ============================================================================
// BlockSharedIndices: Index management for block MPS
// ============================================================================

/// Shared indices for block MPS.
///
/// All blocks share the same site and MPO output indices (same DynId).
/// Bond indices remain independent per block.
#[allow(dead_code)]
struct BlockSharedIndices {
    /// Number of blocks
    num_blocks: usize,
    /// Number of sites per block
    n_sites: usize,
    /// Physical dimension (stored for potential future use)
    phys_dim: usize,
    /// Site indices for each block: sites[block_idx][site_idx]
    /// All blocks share the same DynIndex IDs (cloned from one set).
    sites: Vec<Vec<DynIndex>>,
    /// Bond indices for each block: bonds[block_idx][bond_idx]
    bonds: Vec<Vec<DynIndex>>,
    /// MPO output indices for each block: mpo_outputs[block_idx][site_idx]
    /// All blocks share the same DynIndex IDs (cloned from one set).
    mpo_outputs: Vec<Vec<DynIndex>>,
}

impl BlockSharedIndices {
    fn new(num_blocks: usize, n_sites: usize, phys_dim: usize) -> Self {
        // Shared site indices: all blocks share the same physical indices
        let shared_sites: Vec<DynIndex> =
            (0..n_sites).map(|_| DynIndex::new_dyn(phys_dim)).collect();

        // Shared MPO output indices
        let shared_mpo_outputs: Vec<DynIndex> =
            (0..n_sites).map(|_| DynIndex::new_dyn(phys_dim)).collect();

        // Build per-block arrays by cloning shared index sets
        let mut sites = Vec::with_capacity(num_blocks);
        let mut bonds = Vec::with_capacity(num_blocks);
        let mut mpo_outputs = Vec::with_capacity(num_blocks);

        for _ in 0..num_blocks {
            sites.push(shared_sites.clone());
            bonds.push(
                (0..n_sites.saturating_sub(1))
                    .map(|_| DynIndex::new_dyn(1))
                    .collect(),
            );
            mpo_outputs.push(shared_mpo_outputs.clone());
        }

        BlockSharedIndices {
            num_blocks,
            n_sites,
            phys_dim,
            sites,
            bonds,
            mpo_outputs,
        }
    }
}

// ============================================================================
// Helper functions: MPS creation
// ============================================================================

/// Create an MPS with all elements = 1 for a specific block.
fn create_ones_mps_for_block(
    indices: &BlockSharedIndices,
    block_idx: usize,
) -> anyhow::Result<TensorTrain> {
    let n = indices.n_sites;
    let mut tensors = Vec::with_capacity(n);

    for i in 0..n {
        let site_dim = indices.sites[block_idx][i].dim();
        let site_idx = indices.sites[block_idx][i].clone();

        if i == 0 && n == 1 {
            let data = vec![1.0; site_dim];
            let tensor = TensorDynLen::from_dense(vec![site_idx], data).unwrap();
            tensors.push(tensor);
        } else if i == 0 {
            let right_bond = indices.bonds[block_idx][i].clone();
            let data = vec![1.0; site_dim];
            let tensor = TensorDynLen::from_dense(vec![site_idx, right_bond], data).unwrap();
            tensors.push(tensor);
        } else if i == n - 1 {
            let left_bond = indices.bonds[block_idx][i - 1].clone();
            let data = vec![1.0; site_dim];
            let tensor = TensorDynLen::from_dense(vec![left_bond, site_idx], data).unwrap();
            tensors.push(tensor);
        } else {
            let left_bond = indices.bonds[block_idx][i - 1].clone();
            let right_bond = indices.bonds[block_idx][i].clone();
            let data = vec![1.0; site_dim];
            let tensor =
                TensorDynLen::from_dense(vec![left_bond, site_idx, right_bond], data).unwrap();
            tensors.push(tensor);
        }
    }

    TensorTrain::new(tensors).map_err(|e| anyhow::anyhow!("{}", e))
}

/// Create a BlockTensor with all-ones MPS for each block.
fn create_block_ones_mps(indices: &BlockSharedIndices) -> anyhow::Result<BlockTensor<TensorTrain>> {
    let mut blocks = Vec::with_capacity(indices.num_blocks);
    for block_idx in 0..indices.num_blocks {
        blocks.push(create_ones_mps_for_block(indices, block_idx)?);
    }
    BlockTensor::try_new(blocks, (indices.num_blocks, 1)).map_err(anyhow::Error::from)
}

/// Create an MPS with specified constant value for a specific block.
fn create_const_mps_for_block(
    indices: &BlockSharedIndices,
    block_idx: usize,
    value: f64,
) -> anyhow::Result<TensorTrain> {
    let n = indices.n_sites;
    let mut tensors = Vec::with_capacity(n);

    for i in 0..n {
        let site_dim = indices.sites[block_idx][i].dim();
        let site_idx = indices.sites[block_idx][i].clone();

        // For constant value MPS: first site has the value, rest are 1.0
        let fill_value = if i == 0 { value } else { 1.0 };

        if i == 0 && n == 1 {
            let data = vec![fill_value; site_dim];
            let tensor = TensorDynLen::from_dense(vec![site_idx], data).unwrap();
            tensors.push(tensor);
        } else if i == 0 {
            let right_bond = indices.bonds[block_idx][i].clone();
            let data = vec![fill_value; site_dim];
            let tensor = TensorDynLen::from_dense(vec![site_idx, right_bond], data).unwrap();
            tensors.push(tensor);
        } else if i == n - 1 {
            let left_bond = indices.bonds[block_idx][i - 1].clone();
            let data = vec![fill_value; site_dim];
            let tensor = TensorDynLen::from_dense(vec![left_bond, site_idx], data).unwrap();
            tensors.push(tensor);
        } else {
            let left_bond = indices.bonds[block_idx][i - 1].clone();
            let right_bond = indices.bonds[block_idx][i].clone();
            let data = vec![fill_value; site_dim];
            let tensor =
                TensorDynLen::from_dense(vec![left_bond, site_idx, right_bond], data).unwrap();
            tensors.push(tensor);
        }
    }

    TensorTrain::new(tensors).map_err(|e| anyhow::anyhow!("{}", e))
}

// ============================================================================
// Helper functions: MPO creation
// ============================================================================

/// Create an identity MPO for a specific block.
fn create_identity_mpo_for_block(
    indices: &BlockSharedIndices,
    block_idx: usize,
) -> anyhow::Result<TensorTrain> {
    let n = indices.n_sites;
    let mut tensors = Vec::with_capacity(n);

    // MPO bond indices (separate from MPS bonds)
    let mpo_bonds: Vec<DynIndex> = (0..n.saturating_sub(1))
        .map(|_| DynIndex::new_dyn(1))
        .collect();

    for i in 0..n {
        let site_dim = indices.sites[block_idx][i].dim();
        let s_in = indices.sites[block_idx][i].clone();
        let s_out = indices.mpo_outputs[block_idx][i].clone();

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

/// Create a diagonal MPO: diag(2, 3) at each site for a specific block.
fn create_diagonal_mpo_for_block(
    indices: &BlockSharedIndices,
    block_idx: usize,
) -> anyhow::Result<TensorTrain> {
    let n = indices.n_sites;
    let mut tensors = Vec::with_capacity(n);

    let mpo_bonds: Vec<DynIndex> = (0..n.saturating_sub(1))
        .map(|_| DynIndex::new_dyn(1))
        .collect();

    // Diagonal matrix: [[2, 0], [0, 3]]
    let diag_data = [2.0, 0.0, 0.0, 3.0];

    for i in 0..n {
        let s_in = indices.sites[block_idx][i].clone();
        let s_out = indices.mpo_outputs[block_idx][i].clone();

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

/// Create an MPS with all elements = i (pure imaginary ones) for a specific block.
/// All cores use DenseC64 storage.
fn create_complex_ones_mps_for_block(
    indices: &BlockSharedIndices,
    block_idx: usize,
    value: Complex64,
) -> anyhow::Result<TensorTrain> {
    let n = indices.n_sites;
    let mut tensors = Vec::with_capacity(n);

    // Factor goes on first site only, rest get 1+0i
    let one = Complex64::new(1.0, 0.0);

    for i in 0..n {
        let site_dim = indices.sites[block_idx][i].dim();
        let site_idx = indices.sites[block_idx][i].clone();
        let fill = if i == 0 { value } else { one };

        if i == 0 && n == 1 {
            let data = vec![fill; site_dim];
            let tensor = TensorDynLen::from_dense(vec![site_idx], data).unwrap();
            tensors.push(tensor);
        } else if i == 0 {
            let right_bond = indices.bonds[block_idx][i].clone();
            let data = vec![fill; site_dim];
            let tensor = TensorDynLen::from_dense(vec![site_idx, right_bond], data).unwrap();
            tensors.push(tensor);
        } else if i == n - 1 {
            let left_bond = indices.bonds[block_idx][i - 1].clone();
            let data = vec![fill; site_dim];
            let tensor = TensorDynLen::from_dense(vec![left_bond, site_idx], data).unwrap();
            tensors.push(tensor);
        } else {
            let left_bond = indices.bonds[block_idx][i - 1].clone();
            let right_bond = indices.bonds[block_idx][i].clone();
            let data = vec![fill; site_dim];
            let tensor =
                TensorDynLen::from_dense(vec![left_bond, site_idx, right_bond], data).unwrap();
            tensors.push(tensor);
        }
    }

    TensorTrain::new(tensors).map_err(|e| anyhow::anyhow!("{}", e))
}

/// Create an i*Pauli-X MPO that maps from block src_idx to block dst_idx (all cores DenseC64).
/// Input indices: sites[src_idx], Output indices: mpo_outputs[dst_idx]
fn create_i_pauli_x_cross_block_mpo(
    indices: &BlockSharedIndices,
    src_idx: usize,
    dst_idx: usize,
) -> anyhow::Result<TensorTrain> {
    let n = indices.n_sites;
    let mut tensors = Vec::with_capacity(n);

    let mpo_bonds: Vec<DynIndex> = (0..n.saturating_sub(1))
        .map(|_| DynIndex::new_dyn(1))
        .collect();

    let i_unit = Complex64::new(0.0, 1.0);
    let one = Complex64::new(1.0, 0.0);
    let zero = Complex64::new(0.0, 0.0);

    for i in 0..n {
        let s_in = indices.sites[src_idx][i].clone();
        let s_out = indices.mpo_outputs[dst_idx][i].clone();

        // i*σ_x on first site, σ_x (stored as complex) on remaining sites
        let factor = if i == 0 { i_unit } else { one };
        let data = vec![zero, factor, factor, zero];

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

/// Create an identity MPO that maps from block src_idx to block dst_idx.
/// Input indices: sites[src_idx], Output indices: mpo_outputs[dst_idx]
/// (after replaceinds, output becomes sites[dst_idx])
fn create_cross_block_identity_mpo(
    indices: &BlockSharedIndices,
    src_idx: usize,
    dst_idx: usize,
) -> anyhow::Result<TensorTrain> {
    let n = indices.n_sites;
    let mut tensors = Vec::with_capacity(n);

    let mpo_bonds: Vec<DynIndex> = (0..n.saturating_sub(1))
        .map(|_| DynIndex::new_dyn(1))
        .collect();

    for i in 0..n {
        let site_dim = indices.sites[src_idx][i].dim();
        let s_in = indices.sites[src_idx][i].clone();
        let s_out = indices.mpo_outputs[dst_idx][i].clone();

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

// ============================================================================
// Helper functions: MPO application
// ============================================================================

/// Apply MPO to MPS for a specific block and return result with site indices.
fn apply_mpo_for_block(
    mpo: &TensorTrain,
    mps: &TensorTrain,
    indices: &BlockSharedIndices,
    block_idx: usize,
) -> anyhow::Result<TensorTrain> {
    let options = ContractOptions::fit()
        .with_nhalfsweeps(4)
        .with_svd_policy(tensor4all_core::SvdTruncationPolicy::new(1e-10))
        .with_max_rank(30);
    let result = mpo
        .contract(mps, &options)
        .map_err(|e| anyhow::anyhow!("{}", e))?;

    // Replace MPO output indices with site indices
    let result = result.replaceinds(&indices.mpo_outputs[block_idx], &indices.sites[block_idx])?;

    Ok(result)
}

/// Apply cross-block MPO (from src_idx to dst_idx) and return result with dst site indices.
fn apply_cross_block_mpo(
    mpo: &TensorTrain,
    mps: &TensorTrain,
    indices: &BlockSharedIndices,
    dst_idx: usize,
) -> anyhow::Result<TensorTrain> {
    let options = ContractOptions::fit()
        .with_nhalfsweeps(4)
        .with_svd_policy(tensor4all_core::SvdTruncationPolicy::new(1e-10))
        .with_max_rank(30);
    let result = mpo
        .contract(mps, &options)
        .map_err(|e| anyhow::anyhow!("{}", e))?;

    // Replace MPO output indices with destination site indices
    let result = result.replaceinds(&indices.mpo_outputs[dst_idx], &indices.sites[dst_idx])?;

    Ok(result)
}

fn block_entry(
    x: &BlockTensor<TensorTrain>,
    row: usize,
    col: usize,
) -> anyhow::Result<&TensorTrain> {
    x.get(row, col)
        .ok_or_else(|| anyhow::anyhow!("missing block ({row}, {col})"))
}

// ============================================================================
// Truncation helper for BlockTensor<TensorTrain>
// ============================================================================

/// Truncate each block in a BlockTensor.
fn truncate_block_tensor(
    x: &mut BlockTensor<TensorTrain>,
    opts: &TruncateOptions,
) -> anyhow::Result<()> {
    for block in x.blocks_mut() {
        block.truncate(opts)?;
    }
    Ok(())
}

// ============================================================================
// Test 1: Block diagonal identity operator
// ============================================================================

fn test_block_diagonal_identity() -> anyhow::Result<()> {
    println!("=== Test 1: Block Diagonal Identity ===");

    let num_blocks = 2;
    let n_sites = 2;
    let phys_dim = 2;

    println!(
        "Block structure: {}x1, each block: N={} sites, phys_dim={}",
        num_blocks, n_sites, phys_dim
    );

    let indices = BlockSharedIndices::new(num_blocks, n_sites, phys_dim);

    // Create b = [ones, ones]^T
    let b = create_block_ones_mps(&indices)?;
    let b_norm = b.norm()?;
    println!("b (all-ones block MPS) created, norm: {:.6}", b_norm);

    // Create identity MPOs for each block
    let identity_mpos: Vec<TensorTrain> = (0..num_blocks)
        .map(|i| create_identity_mpo_for_block(&indices, i))
        .collect::<anyhow::Result<Vec<_>>>()?;

    // Define block diagonal identity operator: A = [[I, 0], [0, I]]
    let apply_a = |x: &BlockTensor<TensorTrain>| -> anyhow::Result<BlockTensor<TensorTrain>> {
        let mut result_blocks = Vec::with_capacity(num_blocks);
        for (block_idx, mpo) in identity_mpos.iter().enumerate() {
            let block = block_entry(x, block_idx, 0)?;
            let result = apply_mpo_for_block(mpo, block, &indices, block_idx)?;
            result_blocks.push(result);
        }
        BlockTensor::try_new(result_blocks, (num_blocks, 1)).map_err(anyhow::Error::from)
    };

    // Initial guess: x0 = 0 (use zero-scaled b)
    let x0 = b.scale(AnyScalar::new_real(0.0))?;

    // Compute initial residual
    let ax0 = apply_a(&x0)?;
    let r0 = ax0.axpby(AnyScalar::new_real(1.0), &b, AnyScalar::new_real(-1.0))?;
    let initial_residual = r0.norm()? / b_norm;
    println!("Initial residual: {:.6e}", initial_residual);

    // GMRES options
    let options = GmresOptions {
        max_iter: 10,
        rtol: 1e-8,
        max_restarts: 1,
        verbose: true,
        check_true_residual: false,
    };

    // Truncation
    let truncate_opts = TruncateOptions::svd()
        .with_svd_policy(tensor4all_core::SvdTruncationPolicy::new(1e-10))
        .with_max_rank(30);
    let truncate_fn = |x: &mut BlockTensor<TensorTrain>| -> anyhow::Result<()> {
        truncate_block_tensor(x, &truncate_opts)
    };

    println!("\nRunning GMRES...");
    let result = gmres_with_truncation(&apply_a, &b, &x0, &options, truncate_fn)?;

    // Compute true residual
    let ax_sol = apply_a(&result.solution)?;
    let r_final = ax_sol.axpby(AnyScalar::new_real(1.0), &b, AnyScalar::new_real(-1.0))?;
    let final_residual = r_final.norm()? / b_norm;

    println!("\nResults:");
    println!("Converged: {}", result.converged);
    println!("Iterations: {}", result.iterations);
    println!("GMRES residual: {:.6e}", result.residual_norm);
    println!("True residual:  {:.6e}", final_residual);

    // Verify convergence
    assert!(
        result.converged,
        "GMRES should converge for identity operator"
    );
    assert!(
        final_residual < 1e-6,
        "True residual should be small: {}",
        final_residual
    );

    println!("Test 1 PASSED\n");
    Ok(())
}

// ============================================================================
// Test 2: Block diagonal non-trivial operator (diagonal MPO)
// ============================================================================

fn test_block_diagonal_diagonal_mpo() -> anyhow::Result<()> {
    println!("=== Test 2: Block Diagonal diag(2,3) MPO ===");

    let num_blocks = 2;
    let n_sites = 2;
    let phys_dim = 2;

    println!(
        "Block structure: {}x1, each block: N={} sites",
        num_blocks, n_sites
    );

    let indices = BlockSharedIndices::new(num_blocks, n_sites, phys_dim);

    // Create x_true = [ones, ones]^T
    let x_true = create_block_ones_mps(&indices)?;
    println!("x_true (all-ones) created, norm: {:.6}", x_true.norm()?);

    // Create diagonal MPOs for each block: diag(2, 3)
    let diagonal_mpos: Vec<TensorTrain> = (0..num_blocks)
        .map(|i| create_diagonal_mpo_for_block(&indices, i))
        .collect::<anyhow::Result<Vec<_>>>()?;

    // Define block diagonal operator: A = [[D, 0], [0, D]]
    let apply_a = |x: &BlockTensor<TensorTrain>| -> anyhow::Result<BlockTensor<TensorTrain>> {
        let mut result_blocks = Vec::with_capacity(num_blocks);
        for (block_idx, mpo) in diagonal_mpos.iter().enumerate() {
            let block = block_entry(x, block_idx, 0)?;
            let result = apply_mpo_for_block(mpo, block, &indices, block_idx)?;
            result_blocks.push(result);
        }
        BlockTensor::try_new(result_blocks, (num_blocks, 1)).map_err(anyhow::Error::from)
    };

    // Compute b = A * x_true
    let b = apply_a(&x_true)?;
    let b_norm = b.norm()?;
    println!("b = A * x_true computed, norm: {:.6}", b_norm);

    // Initial guess: x0 = 0.5 * b
    let x0 = b.scale(AnyScalar::new_real(0.5))?;
    println!("x0 (0.5 * b) created, norm: {:.6}", x0.norm()?);

    // Compute initial residual
    let ax0 = apply_a(&x0)?;
    let r0 = ax0.axpby(AnyScalar::new_real(1.0), &b, AnyScalar::new_real(-1.0))?;
    let initial_residual = r0.norm()? / b_norm;
    println!("Initial residual: {:.6e}", initial_residual);

    // GMRES options
    let options = GmresOptions {
        max_iter: 20,
        rtol: 1e-8,
        max_restarts: 1,
        verbose: true,
        check_true_residual: false,
    };

    // Truncation
    let truncate_opts = TruncateOptions::svd()
        .with_svd_policy(tensor4all_core::SvdTruncationPolicy::new(1e-10))
        .with_max_rank(30);
    let truncate_fn = |x: &mut BlockTensor<TensorTrain>| -> anyhow::Result<()> {
        truncate_block_tensor(x, &truncate_opts)
    };

    println!("\nRunning GMRES...");
    let result = gmres_with_truncation(&apply_a, &b, &x0, &options, truncate_fn)?;

    // Compute true residual
    let ax_sol = apply_a(&result.solution)?;
    let r_final = ax_sol.axpby(AnyScalar::new_real(1.0), &b, AnyScalar::new_real(-1.0))?;
    let final_residual = r_final.norm()? / b_norm;

    // Compute error ||x_sol - x_true||
    let diff =
        result
            .solution
            .axpby(AnyScalar::new_real(1.0), &x_true, AnyScalar::new_real(-1.0))?;
    let error = diff.norm()?;

    println!("\nResults:");
    println!("Converged: {}", result.converged);
    println!("Iterations: {}", result.iterations);
    println!("GMRES residual: {:.6e}", result.residual_norm);
    println!("True residual:  {:.6e}", final_residual);
    println!("Error ||x - x_true||: {:.6e}", error);

    // Verify convergence
    assert!(
        result.converged,
        "GMRES should converge for diagonal operator"
    );
    assert!(
        final_residual < 1e-6,
        "True residual should be small: {}",
        final_residual
    );

    println!("Test 2 PASSED\n");
    Ok(())
}

// ============================================================================
// Test 3: Block upper triangular operator
// ============================================================================

fn test_block_upper_triangular() -> anyhow::Result<()> {
    println!("=== Test 3: Block Upper Triangular ===");

    let num_blocks = 2;
    let n_sites = 2;
    let phys_dim = 2;

    println!(
        "Block structure: {}x1, each block: N={} sites",
        num_blocks, n_sites
    );
    println!("A = [[I, B], [0, I]] where B is identity mapping from block 1 to block 0");

    let indices = BlockSharedIndices::new(num_blocks, n_sites, phys_dim);

    // Create identity MPOs for diagonal blocks
    let identity_mpo_0 = create_identity_mpo_for_block(&indices, 0)?;
    let identity_mpo_1 = create_identity_mpo_for_block(&indices, 1)?;

    // Create cross-block identity MPO: B maps from block 1 -> block 0
    let cross_block_mpo = create_cross_block_identity_mpo(&indices, 1, 0)?;

    // Define upper triangular operator: A = [[I, B], [0, I]]
    // Ax = [I*x1 + B*x2, I*x2]^T
    let apply_a = |x: &BlockTensor<TensorTrain>| -> anyhow::Result<BlockTensor<TensorTrain>> {
        let x1 = block_entry(x, 0, 0)?;
        let x2 = block_entry(x, 1, 0)?;

        // y2 = I * x2
        let y2 = apply_mpo_for_block(&identity_mpo_1, x2, &indices, 1)?;

        // y1 = I * x1 + B * x2
        let i_x1 = apply_mpo_for_block(&identity_mpo_0, x1, &indices, 0)?;
        let b_x2 = apply_cross_block_mpo(&cross_block_mpo, x2, &indices, 0)?;
        let y1 = i_x1.axpby(AnyScalar::new_real(1.0), &b_x2, AnyScalar::new_real(1.0))?;

        BlockTensor::try_new(vec![y1, y2], (2, 1)).map_err(anyhow::Error::from)
    };

    // x_true = [2*ones, ones]^T
    // Then Ax_true = [2*ones + ones, ones]^T = [3*ones, ones]^T = b
    let x1_true = create_const_mps_for_block(&indices, 0, 2.0)?;
    let x2_true = create_ones_mps_for_block(&indices, 1)?;
    let x_true = BlockTensor::try_new(vec![x1_true, x2_true], (2, 1))?;
    println!(
        "x_true created: x1=2*ones, x2=ones, norm: {:.6}",
        x_true.norm()?
    );

    // Compute b = A * x_true
    let b = apply_a(&x_true)?;
    let b_norm = b.norm()?;
    println!("b = A * x_true computed, norm: {:.6}", b_norm);
    println!("b should be [3*ones, ones]^T");

    // Initial guess: x0 = 0
    let x0 = x_true.scale(AnyScalar::new_real(0.0))?;

    // Compute initial residual
    let ax0 = apply_a(&x0)?;
    let r0 = ax0.axpby(AnyScalar::new_real(1.0), &b, AnyScalar::new_real(-1.0))?;
    let initial_residual = r0.norm()? / b_norm;
    println!("Initial residual: {:.6e}", initial_residual);

    // GMRES options
    let options = GmresOptions {
        max_iter: 20,
        rtol: 1e-8,
        max_restarts: 1,
        verbose: true,
        check_true_residual: false,
    };

    // Truncation
    let truncate_opts = TruncateOptions::svd()
        .with_svd_policy(tensor4all_core::SvdTruncationPolicy::new(1e-10))
        .with_max_rank(30);
    let truncate_fn = |x: &mut BlockTensor<TensorTrain>| -> anyhow::Result<()> {
        truncate_block_tensor(x, &truncate_opts)
    };

    println!("\nRunning GMRES...");
    let result = gmres_with_truncation(&apply_a, &b, &x0, &options, truncate_fn)?;

    // Compute true residual
    let ax_sol = apply_a(&result.solution)?;
    let r_final = ax_sol.axpby(AnyScalar::new_real(1.0), &b, AnyScalar::new_real(-1.0))?;
    let final_residual = r_final.norm()? / b_norm;

    // Compute error ||x_sol - x_true||
    let diff =
        result
            .solution
            .axpby(AnyScalar::new_real(1.0), &x_true, AnyScalar::new_real(-1.0))?;
    let error = diff.norm()?;

    println!("\nResults:");
    println!("Converged: {}", result.converged);
    println!("Iterations: {}", result.iterations);
    println!("GMRES residual: {:.6e}", result.residual_norm);
    println!("True residual:  {:.6e}", final_residual);
    println!("Error ||x - x_true||: {:.6e}", error);

    // Verify convergence
    assert!(
        result.converged,
        "GMRES should converge for upper triangular operator"
    );
    assert!(
        final_residual < 1e-6,
        "True residual should be small: {}",
        final_residual
    );

    println!("Test 3 PASSED\n");
    Ok(())
}

// ============================================================================
// Test 4: Restart GMRES with block MPS
// ============================================================================

fn test_restart_gmres_block_mps() -> anyhow::Result<()> {
    println!("=== Test 4: Restart GMRES with Block MPS ===");

    let num_blocks = 2;
    let n_sites = 2;
    let phys_dim = 2;

    println!(
        "Block structure: {}x1, each block: N={} sites",
        num_blocks, n_sites
    );
    println!("A = [[D, 0], [0, D]] where D = diag(2, 3)");

    let indices = BlockSharedIndices::new(num_blocks, n_sites, phys_dim);

    // Create x_true = [ones, ones]^T
    let x_true = create_block_ones_mps(&indices)?;
    println!("x_true (all-ones) created, norm: {:.6}", x_true.norm()?);

    // Create diagonal MPOs for each block
    let diagonal_mpos: Vec<TensorTrain> = (0..num_blocks)
        .map(|i| create_diagonal_mpo_for_block(&indices, i))
        .collect::<anyhow::Result<Vec<_>>>()?;

    // Define block diagonal operator
    let apply_a = |x: &BlockTensor<TensorTrain>| -> anyhow::Result<BlockTensor<TensorTrain>> {
        let mut result_blocks = Vec::with_capacity(num_blocks);
        for (block_idx, mpo) in diagonal_mpos.iter().enumerate() {
            let block = block_entry(x, block_idx, 0)?;
            let result = apply_mpo_for_block(mpo, block, &indices, block_idx)?;
            result_blocks.push(result);
        }
        BlockTensor::try_new(result_blocks, (num_blocks, 1)).map_err(anyhow::Error::from)
    };

    // Compute b = A * x_true
    let b = apply_a(&x_true)?;
    let b_norm = b.norm()?;
    println!("b = A * x_true computed, norm: {:.6}", b_norm);

    // Truncation with aggressive settings
    let truncate_opts = TruncateOptions::svd()
        .with_svd_policy(tensor4all_core::SvdTruncationPolicy::new(1e-6))
        .with_max_rank(10);
    let truncate_fn = |x: &mut BlockTensor<TensorTrain>| -> anyhow::Result<()> {
        truncate_block_tensor(x, &truncate_opts)
    };

    // Restart GMRES options
    let options = RestartGmresOptions {
        max_outer_iters: 30,
        rtol: 1e-6,
        inner_max_iter: 5,
        inner_max_restarts: 0,
        min_reduction: None,
        inner_rtol: Some(0.1),
        verbose: true,
    };

    println!("\nRunning Restart GMRES...");
    println!(
        "Truncation: rtol={:.0e}, max_rank={}",
        truncate_opts
            .svd_policy()
            .map(|policy| policy.threshold)
            .unwrap_or(0.0),
        truncate_opts.max_rank().unwrap_or(0)
    );

    let result = restart_gmres_with_truncation(&apply_a, &b, None, &options, &truncate_fn)?;

    // Compute true residual
    let ax_sol = apply_a(&result.solution)?;
    let r_final = ax_sol.axpby(AnyScalar::new_real(1.0), &b, AnyScalar::new_real(-1.0))?;
    let final_residual = r_final.norm()? / b_norm;

    // Compute error ||x_sol - x_true||
    let diff =
        result
            .solution
            .axpby(AnyScalar::new_real(1.0), &x_true, AnyScalar::new_real(-1.0))?;
    let error = diff.norm()?;

    println!("\nResults:");
    println!("Converged: {}", result.converged);
    println!("Outer iterations: {}", result.outer_iterations);
    println!("Total inner iterations: {}", result.iterations);
    println!("GMRES residual: {:.6e}", result.residual_norm);
    println!("True residual:  {:.6e}", final_residual);
    println!("Error ||x - x_true||: {:.6e}", error);

    // Verify convergence
    assert!(
        result.converged,
        "Restart GMRES should converge for diagonal operator"
    );
    assert!(
        final_residual < 1e-4,
        "True residual should be reasonably small: {}",
        final_residual
    );

    println!("Test 4 PASSED\n");
    Ok(())
}

// ============================================================================
// Test 5: Block diagonal complex operator (i * Pauli-X)
// ============================================================================

fn test_block_offdiagonal_complex_pauli_x() -> anyhow::Result<()> {
    println!("=== Test 5: Block Off-Diagonal Complex i*σ_x (MPS) ===");

    let num_blocks = 2;
    let n_sites = 2;
    let phys_dim = 2;

    println!(
        "Block structure: {}x1, each block: N={} sites, phys_dim={}",
        num_blocks, n_sites, phys_dim
    );
    println!("A = [[0, i*σ_x], [i*σ_x, 0]]");
    println!("x_true = [i*ones, 2i*ones]^T");
    println!("b = Ax = [i*σ_x * 2i*ones, i*σ_x * i*ones]^T = [-2*ones, -ones]^T");

    let indices = BlockSharedIndices::new(num_blocks, n_sites, phys_dim);

    // Create cross-block i*σ_x MPOs (all cores DenseC64):
    // mpo_1to0: acts on block 1, produces result in block 0's index space
    // mpo_0to1: acts on block 0, produces result in block 1's index space
    let mpo_1to0 = create_i_pauli_x_cross_block_mpo(&indices, 1, 0)?;
    let mpo_0to1 = create_i_pauli_x_cross_block_mpo(&indices, 0, 1)?;

    // x_true = [i*ones, 2i*ones]^T (all cores DenseC64)
    // σ_x * ones = ones (uniform superposition is invariant under spin flip)
    // A * x_true = [i*σ_x * 2i*ones, i*σ_x * i*ones]^T = [-2*ones, -ones]^T
    let x1_true = create_complex_ones_mps_for_block(&indices, 0, Complex64::new(0.0, 1.0))?;
    let x2_true = create_complex_ones_mps_for_block(&indices, 1, Complex64::new(0.0, 2.0))?;
    let x_true = BlockTensor::new(vec![x1_true, x2_true], (num_blocks, 1))?;
    println!("x_true created, norm: {:.6}", x_true.norm()?);

    // Define off-diagonal block operator: A = [[0, i*σ_x], [i*σ_x, 0]]
    // y1 = i*σ_x * x2, y2 = i*σ_x * x1
    let apply_a = |x: &BlockTensor<TensorTrain>| -> anyhow::Result<BlockTensor<TensorTrain>> {
        let x1 = block_entry(x, 0, 0)?;
        let x2 = block_entry(x, 1, 0)?;

        // y1 = i*σ_x * x2 (cross-block: src=1 -> dst=0)
        let y1 = apply_cross_block_mpo(&mpo_1to0, x2, &indices, 0)?;
        // y2 = i*σ_x * x1 (cross-block: src=0 -> dst=1)
        let y2 = apply_cross_block_mpo(&mpo_0to1, x1, &indices, 1)?;

        BlockTensor::try_new(vec![y1, y2], (num_blocks, 1)).map_err(anyhow::Error::from)
    };

    // Compute b = A * x_true
    let b = apply_a(&x_true)?;
    let b_norm = b.norm()?;
    println!("b = A * x_true computed, norm: {:.6}", b_norm);

    // Initial guess: x0 = complex zero (all cores DenseC64)
    let x0_blocks: Vec<TensorTrain> = (0..num_blocks)
        .map(|block_idx| {
            create_complex_ones_mps_for_block(&indices, block_idx, Complex64::new(0.0, 0.0))
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    let x0 = BlockTensor::new(x0_blocks, (num_blocks, 1))?;
    println!("x0 (complex zero) created, norm: {:.6}", x0.norm()?);

    // GMRES options
    let options = GmresOptions {
        max_iter: 20,
        rtol: 1e-8,
        max_restarts: 1,
        verbose: true,
        check_true_residual: false,
    };

    // Truncation
    let truncate_opts = TruncateOptions::svd()
        .with_svd_policy(tensor4all_core::SvdTruncationPolicy::new(1e-10))
        .with_max_rank(30);
    let truncate_fn = |x: &mut BlockTensor<TensorTrain>| -> anyhow::Result<()> {
        truncate_block_tensor(x, &truncate_opts)
    };

    println!("\nRunning GMRES...");
    let result = gmres_with_truncation(&apply_a, &b, &x0, &options, truncate_fn)?;

    // Compute true residual
    let ax_sol = apply_a(&result.solution)?;
    let r_final = ax_sol.axpby(AnyScalar::new_real(1.0), &b, AnyScalar::new_real(-1.0))?;
    let final_residual = r_final.norm()? / b_norm;

    // Compute error ||x_sol - x_true||
    let diff =
        result
            .solution
            .axpby(AnyScalar::new_real(1.0), &x_true, AnyScalar::new_real(-1.0))?;
    let error = diff.norm()?;

    println!("\nResults:");
    println!("Converged: {}", result.converged);
    println!("Iterations: {}", result.iterations);
    println!("GMRES residual: {:.6e}", result.residual_norm);
    println!("True residual:  {:.6e}", final_residual);
    println!("Error ||x - x_true||: {:.6e}", error);

    // Verify convergence
    assert!(
        result.converged,
        "GMRES should converge for off-diagonal complex i*σ_x operator"
    );
    assert!(
        final_residual < 1e-6,
        "True residual should be small: {}",
        final_residual
    );

    println!("Test 5 PASSED\n");
    Ok(())
}

// ============================================================================
// Test 6: 3x3 block anti-diagonal complex operator (i * Pauli-X)
// ============================================================================

fn test_3x3_block_antidiagonal_complex_pauli_x() -> anyhow::Result<()> {
    println!("=== Test 6: 3x3 Block Anti-Diagonal Complex i*σ_x (MPS) ===");

    let num_blocks = 3;
    let n_sites = 2;
    let phys_dim = 2;

    println!(
        "Block structure: {}x1, each block: N={} sites, phys_dim={}",
        num_blocks, n_sites, phys_dim
    );
    println!("A = [[0, 0, i*σ_x], [0, i*σ_x, 0], [i*σ_x, 0, 0]]");
    println!("x_true = [i*ones, 2i*ones, 3i*ones]^T");
    println!("b = Ax = [-3*ones, -2*ones, -ones]^T");

    let indices = BlockSharedIndices::new(num_blocks, n_sites, phys_dim);

    // Create cross-block i*σ_x MPOs (all cores DenseC64):
    // A[0,2]: src=2 -> dst=0
    let mpo_2to0 = create_i_pauli_x_cross_block_mpo(&indices, 2, 0)?;
    // A[1,1]: src=1 -> dst=1 (diagonal block)
    let mpo_1to1 = create_i_pauli_x_cross_block_mpo(&indices, 1, 1)?;
    // A[2,0]: src=0 -> dst=2
    let mpo_0to2 = create_i_pauli_x_cross_block_mpo(&indices, 0, 2)?;

    // x_true = [i*ones, 2i*ones, 3i*ones]^T (all cores DenseC64)
    let x0_true = create_complex_ones_mps_for_block(&indices, 0, Complex64::new(0.0, 1.0))?;
    let x1_true = create_complex_ones_mps_for_block(&indices, 1, Complex64::new(0.0, 2.0))?;
    let x2_true = create_complex_ones_mps_for_block(&indices, 2, Complex64::new(0.0, 3.0))?;
    let x_true = BlockTensor::new(vec![x0_true, x1_true, x2_true], (num_blocks, 1))?;
    println!("x_true created, norm: {:.6}", x_true.norm()?);

    // Define anti-diagonal block operator:
    // y0 = i*σ_x * x2, y1 = i*σ_x * x1, y2 = i*σ_x * x0
    let apply_a = |x: &BlockTensor<TensorTrain>| -> anyhow::Result<BlockTensor<TensorTrain>> {
        let x0 = block_entry(x, 0, 0)?;
        let x1 = block_entry(x, 1, 0)?;
        let x2 = block_entry(x, 2, 0)?;

        let y0 = apply_cross_block_mpo(&mpo_2to0, x2, &indices, 0)?;
        let y1 = apply_cross_block_mpo(&mpo_1to1, x1, &indices, 1)?;
        let y2 = apply_cross_block_mpo(&mpo_0to2, x0, &indices, 2)?;

        BlockTensor::try_new(vec![y0, y1, y2], (num_blocks, 1)).map_err(anyhow::Error::from)
    };

    // Compute b = A * x_true
    let b = apply_a(&x_true)?;
    let b_norm = b.norm()?;
    println!("b = A * x_true computed, norm: {:.6}", b_norm);

    // Initial guess: x0 = complex zero (all cores DenseC64)
    let x0_blocks: Vec<TensorTrain> = (0..num_blocks)
        .map(|block_idx| {
            create_complex_ones_mps_for_block(&indices, block_idx, Complex64::new(0.0, 0.0))
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    let x0 = BlockTensor::new(x0_blocks, (num_blocks, 1))?;
    println!("x0 (complex zero) created, norm: {:.6}", x0.norm()?);

    // GMRES options
    let options = GmresOptions {
        max_iter: 30,
        rtol: 1e-8,
        max_restarts: 1,
        verbose: true,
        check_true_residual: false,
    };

    // Truncation
    let truncate_opts = TruncateOptions::svd()
        .with_svd_policy(tensor4all_core::SvdTruncationPolicy::new(1e-10))
        .with_max_rank(30);
    let truncate_fn = |x: &mut BlockTensor<TensorTrain>| -> anyhow::Result<()> {
        truncate_block_tensor(x, &truncate_opts)
    };

    println!("\nRunning GMRES...");
    let result = gmres_with_truncation(&apply_a, &b, &x0, &options, truncate_fn)?;

    // Compute true residual
    let ax_sol = apply_a(&result.solution)?;
    let r_final = ax_sol.axpby(AnyScalar::new_real(1.0), &b, AnyScalar::new_real(-1.0))?;
    let final_residual = r_final.norm()? / b_norm;

    // Compute error ||x_sol - x_true||
    let diff =
        result
            .solution
            .axpby(AnyScalar::new_real(1.0), &x_true, AnyScalar::new_real(-1.0))?;
    let error = diff.norm()?;

    println!("\nResults:");
    println!("Converged: {}", result.converged);
    println!("Iterations: {}", result.iterations);
    println!("GMRES residual: {:.6e}", result.residual_norm);
    println!("True residual:  {:.6e}", final_residual);
    println!("Error ||x - x_true||: {:.6e}", error);

    // Verify convergence
    assert!(
        result.converged,
        "GMRES should converge for 3x3 anti-diagonal complex i*σ_x operator"
    );
    assert!(
        final_residual < 1e-6,
        "True residual should be small: {}",
        final_residual
    );

    println!("Test 6 PASSED\n");
    Ok(())
}

// ============================================================================
// Scaling study: vary n_sites for 2x2 off-diagonal i*σ_x
// ============================================================================

fn scaling_offdiagonal_complex_pauli_x(n_sites: usize) -> anyhow::Result<()> {
    let num_blocks = 2;
    let phys_dim = 2;
    let block_dim = (phys_dim as u64).pow(n_sites as u32);

    println!(
        "  N={:>2}: block_dim=2^{}={}, total_dim={}",
        n_sites,
        n_sites,
        block_dim,
        block_dim * num_blocks as u64
    );

    let indices = BlockSharedIndices::new(num_blocks, n_sites, phys_dim);

    // Cross-block i*σ_x MPOs
    let mpo_1to0 = create_i_pauli_x_cross_block_mpo(&indices, 1, 0)?;
    let mpo_0to1 = create_i_pauli_x_cross_block_mpo(&indices, 0, 1)?;

    // x_true = [i*ones, 2i*ones]^T
    let x1_true = create_complex_ones_mps_for_block(&indices, 0, Complex64::new(0.0, 1.0))?;
    let x2_true = create_complex_ones_mps_for_block(&indices, 1, Complex64::new(0.0, 2.0))?;
    let x_true = BlockTensor::new(vec![x1_true, x2_true], (num_blocks, 1))?;

    // A = [[0, i*σ_x], [i*σ_x, 0]]
    let apply_a = |x: &BlockTensor<TensorTrain>| -> anyhow::Result<BlockTensor<TensorTrain>> {
        let x1 = block_entry(x, 0, 0)?;
        let x2 = block_entry(x, 1, 0)?;
        let y1 = apply_cross_block_mpo(&mpo_1to0, x2, &indices, 0)?;
        let y2 = apply_cross_block_mpo(&mpo_0to1, x1, &indices, 1)?;
        BlockTensor::try_new(vec![y1, y2], (num_blocks, 1)).map_err(anyhow::Error::from)
    };

    let b = apply_a(&x_true)?;
    let b_norm = b.norm()?;

    // x0 = complex zero
    let x0_blocks: Vec<TensorTrain> = (0..num_blocks)
        .map(|block_idx| {
            create_complex_ones_mps_for_block(&indices, block_idx, Complex64::new(0.0, 0.0))
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    let x0 = BlockTensor::new(x0_blocks, (num_blocks, 1))?;

    let options = GmresOptions {
        max_iter: 30,
        rtol: 1e-8,
        max_restarts: 1,
        verbose: false,
        check_true_residual: false,
    };

    let truncate_opts = TruncateOptions::svd()
        .with_svd_policy(tensor4all_core::SvdTruncationPolicy::new(1e-10))
        .with_max_rank(30);
    let truncate_fn = |x: &mut BlockTensor<TensorTrain>| -> anyhow::Result<()> {
        truncate_block_tensor(x, &truncate_opts)
    };

    let start = Instant::now();
    let result = gmres_with_truncation(&apply_a, &b, &x0, &options, truncate_fn)?;
    let elapsed = start.elapsed();

    // True residual
    let ax_sol = apply_a(&result.solution)?;
    let r_final = ax_sol.axpby(AnyScalar::new_real(1.0), &b, AnyScalar::new_real(-1.0))?;
    let final_residual = r_final.norm()? / b_norm;

    // Error
    let diff =
        result
            .solution
            .axpby(AnyScalar::new_real(1.0), &x_true, AnyScalar::new_real(-1.0))?;
    let error = diff.norm()?;

    println!(
        "         iters={}, residual={:.2e}, error={:.2e}, time={:.3}s, converged={}",
        result.iterations,
        final_residual,
        error,
        elapsed.as_secs_f64(),
        result.converged
    );

    assert!(result.converged, "GMRES should converge for N={}", n_sites);
    assert!(
        final_residual < 1e-6,
        "True residual too large for N={}: {}",
        n_sites,
        final_residual
    );

    Ok(())
}

fn scaling_3x3_antidiagonal_complex_pauli_x(n_sites: usize) -> anyhow::Result<()> {
    let num_blocks = 3;
    let phys_dim = 2;
    let block_dim = (phys_dim as u64).pow(n_sites as u32);

    println!(
        "  N={:>2}: block_dim=2^{}={}, total_dim={}",
        n_sites,
        n_sites,
        block_dim,
        block_dim * num_blocks as u64
    );

    let indices = BlockSharedIndices::new(num_blocks, n_sites, phys_dim);

    // A[0,2]: src=2 -> dst=0
    let mpo_2to0 = create_i_pauli_x_cross_block_mpo(&indices, 2, 0)?;
    // A[1,1]: src=1 -> dst=1
    let mpo_1to1 = create_i_pauli_x_cross_block_mpo(&indices, 1, 1)?;
    // A[2,0]: src=0 -> dst=2
    let mpo_0to2 = create_i_pauli_x_cross_block_mpo(&indices, 0, 2)?;

    // x_true = [i*ones, 2i*ones, 3i*ones]^T
    let x0_true = create_complex_ones_mps_for_block(&indices, 0, Complex64::new(0.0, 1.0))?;
    let x1_true = create_complex_ones_mps_for_block(&indices, 1, Complex64::new(0.0, 2.0))?;
    let x2_true = create_complex_ones_mps_for_block(&indices, 2, Complex64::new(0.0, 3.0))?;
    let x_true = BlockTensor::new(vec![x0_true, x1_true, x2_true], (num_blocks, 1))?;

    // A = [[0, 0, i*σ_x], [0, i*σ_x, 0], [i*σ_x, 0, 0]]
    let apply_a = |x: &BlockTensor<TensorTrain>| -> anyhow::Result<BlockTensor<TensorTrain>> {
        let x0 = block_entry(x, 0, 0)?;
        let x1 = block_entry(x, 1, 0)?;
        let x2 = block_entry(x, 2, 0)?;
        let y0 = apply_cross_block_mpo(&mpo_2to0, x2, &indices, 0)?;
        let y1 = apply_cross_block_mpo(&mpo_1to1, x1, &indices, 1)?;
        let y2 = apply_cross_block_mpo(&mpo_0to2, x0, &indices, 2)?;
        BlockTensor::try_new(vec![y0, y1, y2], (num_blocks, 1)).map_err(anyhow::Error::from)
    };

    let b = apply_a(&x_true)?;
    let b_norm = b.norm()?;

    // x0 = complex zero
    let x0_blocks: Vec<TensorTrain> = (0..num_blocks)
        .map(|block_idx| {
            create_complex_ones_mps_for_block(&indices, block_idx, Complex64::new(0.0, 0.0))
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    let x0 = BlockTensor::new(x0_blocks, (num_blocks, 1))?;

    let options = GmresOptions {
        max_iter: 30,
        rtol: 1e-8,
        max_restarts: 1,
        verbose: false,
        check_true_residual: false,
    };

    let truncate_opts = TruncateOptions::svd()
        .with_svd_policy(tensor4all_core::SvdTruncationPolicy::new(1e-10))
        .with_max_rank(30);
    let truncate_fn = |x: &mut BlockTensor<TensorTrain>| -> anyhow::Result<()> {
        truncate_block_tensor(x, &truncate_opts)
    };

    let start = Instant::now();
    let result = gmres_with_truncation(&apply_a, &b, &x0, &options, truncate_fn)?;
    let elapsed = start.elapsed();

    // True residual
    let ax_sol = apply_a(&result.solution)?;
    let r_final = ax_sol.axpby(AnyScalar::new_real(1.0), &b, AnyScalar::new_real(-1.0))?;
    let final_residual = r_final.norm()? / b_norm;

    // Error
    let diff =
        result
            .solution
            .axpby(AnyScalar::new_real(1.0), &x_true, AnyScalar::new_real(-1.0))?;
    let error = diff.norm()?;

    println!(
        "         iters={}, residual={:.2e}, error={:.2e}, time={:.3}s, converged={}",
        result.iterations,
        final_residual,
        error,
        elapsed.as_secs_f64(),
        result.converged
    );

    assert!(
        result.converged,
        "GMRES should converge for 3x3 N={}",
        n_sites
    );
    assert!(
        final_residual < 1e-6,
        "True residual too large for 3x3 N={}: {}",
        n_sites,
        final_residual
    );

    Ok(())
}

fn test_scaling_study() -> anyhow::Result<()> {
    println!("=== Scaling Study: 2x2 Off-Diagonal i*σ_x vs n_sites ===");
    println!("A = [[0, i*σ_x], [i*σ_x, 0]], phys_dim=2\n");

    for &n_sites in &[2, 4, 6, 8, 10, 14, 20] {
        scaling_offdiagonal_complex_pauli_x(n_sites)?;
    }

    println!("\n=== Scaling Study: 3x3 Anti-Diagonal i*σ_x vs n_sites ===");
    println!("A = [[0,0,i*σ_x],[0,i*σ_x,0],[i*σ_x,0,0]], phys_dim=2\n");

    for &n_sites in &[2, 4, 6, 8, 10, 14, 20] {
        scaling_3x3_antidiagonal_complex_pauli_x(n_sites)?;
    }

    println!("\nScaling study PASSED\n");
    Ok(())
}

// ============================================================================
// Main
// ============================================================================

fn main() -> anyhow::Result<()> {
    println!("========================================");
    println!("  Block MPS GMRES Tests");
    println!("========================================\n");

    // Test 1: Block diagonal identity
    test_block_diagonal_identity()?;

    // Test 2: Block diagonal non-trivial operator (diagonal MPO)
    test_block_diagonal_diagonal_mpo()?;

    // Test 3: Block upper triangular
    test_block_upper_triangular()?;

    // Test 4: Restart GMRES with block MPS
    test_restart_gmres_block_mps()?;

    // Test 5: Off-diagonal complex operator (i * Pauli-X)
    test_block_offdiagonal_complex_pauli_x()?;

    // Test 6: 3x3 block anti-diagonal complex operator (i * Pauli-X)
    test_3x3_block_antidiagonal_complex_pauli_x()?;

    // Scaling study: vary n_sites
    test_scaling_study()?;

    println!("========================================");
    println!("  All tests completed!");
    println!("========================================");

    Ok(())
}
