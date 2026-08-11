#![allow(clippy::needless_borrows_for_generic_args)]
#![allow(clippy::identity_op)]
#![allow(clippy::type_complexity)]
//! Test: GMRES solver with BlockTensor<TensorTrain> (block MPO)
//!
//! An N=3 MPO (phys_dim=2) can be written as a 2x2 block matrix of four N=2 MPOs,
//! by splitting the first site's input/output indices.
//! This tests GMRES with such block-structured MPOs.
//!
//! Run:
//!   cargo run -p tensor4all-itensorlike --example test_gmres_block_mpo --release

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
// BlockMpoSharedIndices: Index management for block MPOs
// ============================================================================

/// Shared indices for block MPOs.
///
/// Blocks in the same column share input indices (same DynId).
/// Blocks in the same row share output indices (same DynId).
/// Operator output indices are shared across all blocks.
/// Bond indices remain independent per block.
/// Blocks are indexed in row-major order: block_idx = row * num_cols + col.
#[allow(dead_code)]
struct BlockMpoSharedIndices {
    /// Total number of blocks
    num_blocks: usize,
    /// Block structure (rows, cols)
    block_shape: (usize, usize),
    /// Number of sites per block MPO
    n_sites: usize,
    /// Physical dimension
    phys_dim: usize,
    /// MPO input indices for each block: inputs[block_idx][site_idx]
    /// Blocks in the same column share the same DynIndex IDs.
    inputs: Vec<Vec<DynIndex>>,
    /// MPO output indices for each block: outputs[block_idx][site_idx]
    /// Blocks in the same row share the same DynIndex IDs.
    outputs: Vec<Vec<DynIndex>>,
    /// MPO bond indices for each block: bonds[block_idx][bond_idx]
    bonds: Vec<Vec<DynIndex>>,
    /// Superoperator output indices for each block: operator_outputs[block_idx][site_idx]
    /// All blocks share the same DynIndex IDs.
    operator_outputs: Vec<Vec<DynIndex>>,
}

impl BlockMpoSharedIndices {
    fn new(block_rows: usize, block_cols: usize, n_sites: usize, phys_dim: usize) -> Self {
        let num_blocks = block_rows * block_cols;

        // Shared column input indices: all blocks in the same column share inputs
        let col_inputs: Vec<Vec<DynIndex>> = (0..block_cols)
            .map(|_| (0..n_sites).map(|_| DynIndex::new_dyn(phys_dim)).collect())
            .collect();

        // Shared row output indices: all blocks in the same row share outputs
        let row_outputs: Vec<Vec<DynIndex>> = (0..block_rows)
            .map(|_| (0..n_sites).map(|_| DynIndex::new_dyn(phys_dim)).collect())
            .collect();

        // Shared operator output indices (same for all blocks)
        let shared_op_outputs: Vec<DynIndex> =
            (0..n_sites).map(|_| DynIndex::new_dyn(phys_dim)).collect();

        // Build per-block index arrays by cloning from shared sets
        let mut inputs = Vec::with_capacity(num_blocks);
        let mut outputs = Vec::with_capacity(num_blocks);
        let mut bonds = Vec::with_capacity(num_blocks);
        let mut operator_outputs = Vec::with_capacity(num_blocks);

        for row_output in &row_outputs {
            for col_input in &col_inputs {
                inputs.push(col_input.clone());
                outputs.push(row_output.clone());
                bonds.push(
                    (0..n_sites.saturating_sub(1))
                        .map(|_| DynIndex::new_dyn(1))
                        .collect(),
                );
                operator_outputs.push(shared_op_outputs.clone());
            }
        }

        BlockMpoSharedIndices {
            num_blocks,
            block_shape: (block_rows, block_cols),
            n_sites,
            phys_dim,
            inputs,
            outputs,
            bonds,
            operator_outputs,
        }
    }

    /// Flat block index from (row, col).
    fn flat_idx(&self, row: usize, col: usize) -> usize {
        row * self.block_shape.1 + col
    }
}

// ============================================================================
// Helper functions: MPO creation for blocks
// ============================================================================

/// Create an all-ones MPO for a specific block.
fn create_ones_mpo_for_block(
    indices: &BlockMpoSharedIndices,
    block_idx: usize,
) -> anyhow::Result<TensorTrain> {
    let n = indices.n_sites;
    let mut tensors = Vec::with_capacity(n);

    for i in 0..n {
        let in_dim = indices.inputs[block_idx][i].dim();
        let out_dim = indices.outputs[block_idx][i].dim();
        let in_idx = indices.inputs[block_idx][i].clone();
        let out_idx = indices.outputs[block_idx][i].clone();

        let data = vec![1.0_f64; in_dim * out_dim];

        if i == 0 && n == 1 {
            let tensor = TensorDynLen::from_dense(vec![in_idx, out_idx], data).unwrap();
            tensors.push(tensor);
        } else if i == 0 {
            let right_bond = indices.bonds[block_idx][i].clone();
            let tensor = TensorDynLen::from_dense(vec![in_idx, out_idx, right_bond], data).unwrap();
            tensors.push(tensor);
        } else if i == n - 1 {
            let left_bond = indices.bonds[block_idx][i - 1].clone();
            let tensor = TensorDynLen::from_dense(vec![left_bond, in_idx, out_idx], data).unwrap();
            tensors.push(tensor);
        } else {
            let left_bond = indices.bonds[block_idx][i - 1].clone();
            let right_bond = indices.bonds[block_idx][i].clone();
            let tensor =
                TensorDynLen::from_dense(vec![left_bond, in_idx, out_idx, right_bond], data)
                    .unwrap();
            tensors.push(tensor);
        }
    }

    TensorTrain::new(tensors).map_err(|e| anyhow::anyhow!("{}", e))
}

/// Create a BlockTensor with all-ones MPO for each block.
fn create_block_ones_mpo(
    indices: &BlockMpoSharedIndices,
) -> anyhow::Result<BlockTensor<TensorTrain>> {
    let mut blocks = Vec::with_capacity(indices.num_blocks);
    for block_idx in 0..indices.num_blocks {
        blocks.push(create_ones_mpo_for_block(indices, block_idx)?);
    }
    BlockTensor::try_new(blocks, indices.block_shape).map_err(anyhow::Error::from)
}

/// Create an MPO with specified constant value for a specific block.
fn create_const_mpo_for_block(
    indices: &BlockMpoSharedIndices,
    block_idx: usize,
    value: f64,
) -> anyhow::Result<TensorTrain> {
    let n = indices.n_sites;
    let mut tensors = Vec::with_capacity(n);

    for i in 0..n {
        let in_dim = indices.inputs[block_idx][i].dim();
        let out_dim = indices.outputs[block_idx][i].dim();
        let in_idx = indices.inputs[block_idx][i].clone();
        let out_idx = indices.outputs[block_idx][i].clone();

        // For constant value MPO: first site has the value, rest are 1.0
        let fill_value = if i == 0 { value } else { 1.0 };
        let data = vec![fill_value; in_dim * out_dim];

        if i == 0 && n == 1 {
            let tensor = TensorDynLen::from_dense(vec![in_idx, out_idx], data).unwrap();
            tensors.push(tensor);
        } else if i == 0 {
            let right_bond = indices.bonds[block_idx][i].clone();
            let tensor = TensorDynLen::from_dense(vec![in_idx, out_idx, right_bond], data).unwrap();
            tensors.push(tensor);
        } else if i == n - 1 {
            let left_bond = indices.bonds[block_idx][i - 1].clone();
            let tensor = TensorDynLen::from_dense(vec![left_bond, in_idx, out_idx], data).unwrap();
            tensors.push(tensor);
        } else {
            let left_bond = indices.bonds[block_idx][i - 1].clone();
            let right_bond = indices.bonds[block_idx][i].clone();
            let tensor =
                TensorDynLen::from_dense(vec![left_bond, in_idx, out_idx, right_bond], data)
                    .unwrap();
            tensors.push(tensor);
        }
    }

    TensorTrain::new(tensors).map_err(|e| anyhow::anyhow!("{}", e))
}

// ============================================================================
// Helper functions: Superoperator MPO creation
// ============================================================================

/// Create an identity superoperator for a specific block.
/// Acts on the MPO's input indices: O[in, op_out] = delta(in, op_out).
fn create_identity_operator_for_block(
    indices: &BlockMpoSharedIndices,
    block_idx: usize,
) -> anyhow::Result<TensorTrain> {
    let n = indices.n_sites;
    let mut tensors = Vec::with_capacity(n);

    let op_bonds: Vec<DynIndex> = (0..n.saturating_sub(1))
        .map(|_| DynIndex::new_dyn(1))
        .collect();

    for i in 0..n {
        let in_dim = indices.inputs[block_idx][i].dim();
        let in_idx = indices.inputs[block_idx][i].clone();
        let op_out_idx = indices.operator_outputs[block_idx][i].clone();

        // Identity: delta(in, op_out)
        let mut data = vec![0.0; in_dim * in_dim];
        for j in 0..in_dim {
            data[j * in_dim + j] = 1.0;
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

/// Create a diagonal superoperator for a specific block: diag(2, 3) at each site.
fn create_diagonal_operator_for_block(
    indices: &BlockMpoSharedIndices,
    block_idx: usize,
) -> anyhow::Result<TensorTrain> {
    let n = indices.n_sites;
    let mut tensors = Vec::with_capacity(n);

    let op_bonds: Vec<DynIndex> = (0..n.saturating_sub(1))
        .map(|_| DynIndex::new_dyn(1))
        .collect();

    // Diagonal: [[2, 0], [0, 3]]
    let diag_data = [2.0, 0.0, 0.0, 3.0];

    for i in 0..n {
        let in_idx = indices.inputs[block_idx][i].clone();
        let op_out_idx = indices.operator_outputs[block_idx][i].clone();

        if i == 0 && n == 1 {
            let tensor =
                TensorDynLen::from_dense(vec![in_idx, op_out_idx], diag_data.to_vec()).unwrap();
            tensors.push(tensor);
        } else if i == 0 {
            let right_bond = op_bonds[i].clone();
            let tensor =
                TensorDynLen::from_dense(vec![in_idx, op_out_idx, right_bond], diag_data.to_vec())
                    .unwrap();
            tensors.push(tensor);
        } else if i == n - 1 {
            let left_bond = op_bonds[i - 1].clone();
            let tensor =
                TensorDynLen::from_dense(vec![left_bond, in_idx, op_out_idx], diag_data.to_vec())
                    .unwrap();
            tensors.push(tensor);
        } else {
            let left_bond = op_bonds[i - 1].clone();
            let right_bond = op_bonds[i].clone();
            let tensor = TensorDynLen::from_dense(
                vec![left_bond, in_idx, op_out_idx, right_bond],
                diag_data.to_vec(),
            )
            .unwrap();
            tensors.push(tensor);
        }
    }

    TensorTrain::new(tensors).map_err(|e| anyhow::anyhow!("{}", e))
}

/// Create a cross-block identity superoperator mapping from src block to dst block.
/// Input indices: inputs[src_idx], Output indices: operator_outputs[dst_idx].
fn create_cross_block_identity_operator(
    indices: &BlockMpoSharedIndices,
    src_idx: usize,
    dst_idx: usize,
) -> anyhow::Result<TensorTrain> {
    let n = indices.n_sites;
    let mut tensors = Vec::with_capacity(n);

    let op_bonds: Vec<DynIndex> = (0..n.saturating_sub(1))
        .map(|_| DynIndex::new_dyn(1))
        .collect();

    for i in 0..n {
        let in_dim = indices.inputs[src_idx][i].dim();
        let in_idx = indices.inputs[src_idx][i].clone();
        let op_out_idx = indices.operator_outputs[dst_idx][i].clone();

        // Identity: delta(in, op_out)
        let mut data = vec![0.0; in_dim * in_dim];
        for j in 0..in_dim {
            data[j * in_dim + j] = 1.0;
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

/// Create an i*Pauli-X cross-block superoperator (all cores DenseC64).
/// Maps from src block's input indices to dst block's operator_outputs.
/// i*σ_x = [[0, i], [i, 0]] at each site (bond dim 1).
fn create_i_pauli_x_cross_block_operator(
    indices: &BlockMpoSharedIndices,
    src_idx: usize,
    dst_idx: usize,
) -> anyhow::Result<TensorTrain> {
    let n = indices.n_sites;
    let mut tensors = Vec::with_capacity(n);

    let op_bonds: Vec<DynIndex> = (0..n.saturating_sub(1))
        .map(|_| DynIndex::new_dyn(1))
        .collect();

    let i_unit = Complex64::new(0.0, 1.0);
    let one = Complex64::new(1.0, 0.0);
    let zero = Complex64::new(0.0, 0.0);

    for i in 0..n {
        let in_idx = indices.inputs[src_idx][i].clone();
        let op_out_idx = indices.operator_outputs[dst_idx][i].clone();

        // i*σ_x on first site, σ_x (as Complex64) on remaining sites
        let factor = if i == 0 { i_unit } else { one };
        let data = vec![zero, factor, factor, zero];

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

/// Create an MPO with complex constant value for a specific block (all cores DenseC64).
fn create_complex_const_mpo_for_block(
    indices: &BlockMpoSharedIndices,
    block_idx: usize,
    value: Complex64,
) -> anyhow::Result<TensorTrain> {
    let n = indices.n_sites;
    let mut tensors = Vec::with_capacity(n);

    let one = Complex64::new(1.0, 0.0);

    for i in 0..n {
        let in_dim = indices.inputs[block_idx][i].dim();
        let out_dim = indices.outputs[block_idx][i].dim();
        let in_idx = indices.inputs[block_idx][i].clone();
        let out_idx = indices.outputs[block_idx][i].clone();

        let fill = if i == 0 { value } else { one };
        let data = vec![fill; in_dim * out_dim];

        if i == 0 && n == 1 {
            let tensor = TensorDynLen::from_dense(vec![in_idx, out_idx], data).unwrap();
            tensors.push(tensor);
        } else if i == 0 {
            let right_bond = indices.bonds[block_idx][i].clone();
            let tensor = TensorDynLen::from_dense(vec![in_idx, out_idx, right_bond], data).unwrap();
            tensors.push(tensor);
        } else if i == n - 1 {
            let left_bond = indices.bonds[block_idx][i - 1].clone();
            let tensor = TensorDynLen::from_dense(vec![left_bond, in_idx, out_idx], data).unwrap();
            tensors.push(tensor);
        } else {
            let left_bond = indices.bonds[block_idx][i - 1].clone();
            let right_bond = indices.bonds[block_idx][i].clone();
            let tensor =
                TensorDynLen::from_dense(vec![left_bond, in_idx, out_idx, right_bond], data)
                    .unwrap();
            tensors.push(tensor);
        }
    }

    TensorTrain::new(tensors).map_err(|e| anyhow::anyhow!("{}", e))
}

// ============================================================================
// Helper functions: Superoperator application
// ============================================================================

/// Apply superoperator to MPO for a specific block.
/// Contracts operator with MPO and replaces operator_outputs with input indices.
fn apply_operator_for_block(
    op: &TensorTrain,
    mpo: &TensorTrain,
    indices: &BlockMpoSharedIndices,
    block_idx: usize,
) -> anyhow::Result<TensorTrain> {
    let options = ContractOptions::fit()
        .with_nhalfsweeps(4)
        .with_svd_policy(tensor4all_core::SvdTruncationPolicy::new(1e-10))
        .with_max_bond_dim(30);
    let result = op
        .contract(mpo, &options)
        .map_err(|e| anyhow::anyhow!("{}", e))?;

    // Replace operator output indices with input indices
    let result = result.replace_indices(
        &indices.operator_outputs[block_idx],
        &indices.inputs[block_idx],
    )?;

    Ok(result)
}

/// Apply cross-block superoperator and return result with destination input indices.
fn apply_cross_block_operator(
    op: &TensorTrain,
    mpo: &TensorTrain,
    indices: &BlockMpoSharedIndices,
    dst_idx: usize,
) -> anyhow::Result<TensorTrain> {
    let options = ContractOptions::fit()
        .with_nhalfsweeps(4)
        .with_svd_policy(tensor4all_core::SvdTruncationPolicy::new(1e-10))
        .with_max_bond_dim(30);
    let result = op
        .contract(mpo, &options)
        .map_err(|e| anyhow::anyhow!("{}", e))?;

    // Replace operator output indices with destination input indices
    let result =
        result.replace_indices(&indices.operator_outputs[dst_idx], &indices.inputs[dst_idx])?;

    Ok(result)
}

/// Apply cross-block superoperator, remapping both input AND output indices to destination.
/// This ensures the result block has consistent indices (inputs[dst] and outputs[dst]),
/// which is necessary for GMRES inner products when the operator is purely off-diagonal.
fn apply_cross_block_operator_full(
    op: &TensorTrain,
    mpo: &TensorTrain,
    indices: &BlockMpoSharedIndices,
    src_idx: usize,
    dst_idx: usize,
) -> anyhow::Result<TensorTrain> {
    let options = ContractOptions::fit()
        .with_nhalfsweeps(4)
        .with_svd_policy(tensor4all_core::SvdTruncationPolicy::new(1e-10))
        .with_max_bond_dim(30);
    let result = op
        .contract(mpo, &options)
        .map_err(|e| anyhow::anyhow!("{}", e))?;

    // Replace operator output indices with destination input indices
    let result =
        result.replace_indices(&indices.operator_outputs[dst_idx], &indices.inputs[dst_idx])?;

    // Replace source output indices with destination output indices
    let result = result.replace_indices(&indices.outputs[src_idx], &indices.outputs[dst_idx])?;

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
// Truncation helper
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
    println!("=== Test 1: Block Diagonal Identity (MPO) ===");

    let block_rows = 2;
    let block_cols = 2;
    let n_sites = 2;
    let phys_dim = 2;

    println!(
        "Block structure: {}x{}, each block: N={} sites MPO, phys_dim={}",
        block_rows, block_cols, n_sites, phys_dim
    );

    let indices = BlockMpoSharedIndices::new(block_rows, block_cols, n_sites, phys_dim);

    // Create b = block of all-ones MPOs
    let b = create_block_ones_mpo(&indices)?;
    let b_norm = b.norm()?;
    println!("b (all-ones block MPO) created, norm: {:.6}", b_norm);

    // Create identity superoperators for each block
    let identity_ops: Vec<TensorTrain> = (0..indices.num_blocks)
        .map(|i| create_identity_operator_for_block(&indices, i))
        .collect::<anyhow::Result<Vec<_>>>()?;

    // Define block diagonal identity operator
    let apply_a = |x: &BlockTensor<TensorTrain>| -> anyhow::Result<BlockTensor<TensorTrain>> {
        let mut result_blocks = Vec::with_capacity(indices.num_blocks);
        for (block_idx, op) in identity_ops.iter().enumerate() {
            let block = block_entry(x, block_idx / block_cols, block_idx % block_cols)?;
            let result = apply_operator_for_block(op, block, &indices, block_idx)?;
            result_blocks.push(result);
        }
        BlockTensor::try_new(result_blocks, indices.block_shape).map_err(anyhow::Error::from)
    };

    // Initial guess: x0 = 0
    let x0 = b.scale(AnyScalar::new_real(0.0))?;

    // Compute initial residual
    let ax0 = apply_a(&x0)?;
    let r0 = ax0.axpby(AnyScalar::new_real(1.0), &b, AnyScalar::new_real(-1.0))?;
    let initial_residual = r0.norm()? / b_norm;
    println!("Initial residual: {:.6e}", initial_residual);

    let options = GmresOptions {
        max_iter: 10,
        rtol: 1e-8,
        max_restarts: 1,
        verbose: true,
        check_true_residual: false,
    };

    let truncate_opts = TruncateOptions::svd()
        .with_svd_policy(tensor4all_core::SvdTruncationPolicy::new(1e-10))
        .with_max_bond_dim(30);
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
// Test 2: Block diagonal non-trivial operator (diagonal superoperator)
// ============================================================================

fn test_block_diagonal_diagonal_operator() -> anyhow::Result<()> {
    println!("=== Test 2: Block Diagonal diag(2,3) Superoperator (MPO) ===");

    let block_rows = 2;
    let block_cols = 2;
    let n_sites = 2;
    let phys_dim = 2;

    println!(
        "Block structure: {}x{}, each block: N={} sites MPO",
        block_rows, block_cols, n_sites
    );

    let indices = BlockMpoSharedIndices::new(block_rows, block_cols, n_sites, phys_dim);

    // Create x_true = all-ones block MPO
    let x_true = create_block_ones_mpo(&indices)?;
    println!("x_true (all-ones) created, norm: {:.6}", x_true.norm()?);

    // Create diagonal superoperators for each block
    let diagonal_ops: Vec<TensorTrain> = (0..indices.num_blocks)
        .map(|i| create_diagonal_operator_for_block(&indices, i))
        .collect::<anyhow::Result<Vec<_>>>()?;

    // Define block diagonal operator
    let apply_a = |x: &BlockTensor<TensorTrain>| -> anyhow::Result<BlockTensor<TensorTrain>> {
        let mut result_blocks = Vec::with_capacity(indices.num_blocks);
        for (block_idx, op) in diagonal_ops.iter().enumerate() {
            let block = block_entry(x, block_idx / block_cols, block_idx % block_cols)?;
            let result = apply_operator_for_block(op, block, &indices, block_idx)?;
            result_blocks.push(result);
        }
        BlockTensor::try_new(result_blocks, indices.block_shape).map_err(anyhow::Error::from)
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

    let options = GmresOptions {
        max_iter: 20,
        rtol: 1e-8,
        max_restarts: 1,
        verbose: true,
        check_true_residual: false,
    };

    let truncate_opts = TruncateOptions::svd()
        .with_svd_policy(tensor4all_core::SvdTruncationPolicy::new(1e-10))
        .with_max_bond_dim(30);
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
    println!("=== Test 3: Block Upper Triangular (MPO) ===");

    let block_rows = 2;
    let block_cols = 2;
    let n_sites = 2;
    let phys_dim = 2;

    println!(
        "Block structure: {}x{}, each block: N={} sites MPO",
        block_rows, block_cols, n_sites
    );
    println!("A = [[I, B], [0, I]] acting on block rows");
    println!("y_{{0,j}} = X_{{0,j}} + B * X_{{1,j}}, y_{{1,j}} = X_{{1,j}}");

    let indices = BlockMpoSharedIndices::new(block_rows, block_cols, n_sites, phys_dim);

    // Create identity superoperators for diagonal blocks
    let identity_ops: Vec<TensorTrain> = (0..indices.num_blocks)
        .map(|i| create_identity_operator_for_block(&indices, i))
        .collect::<anyhow::Result<Vec<_>>>()?;

    // Create cross-block identity operators: B maps row 1 -> row 0
    // For column j: B maps block (1,j) -> block (0,j)
    let cross_block_ops: Vec<TensorTrain> = (0..block_cols)
        .map(|j| {
            let src = indices.flat_idx(1, j);
            let dst = indices.flat_idx(0, j);
            create_cross_block_identity_operator(&indices, src, dst)
        })
        .collect::<anyhow::Result<Vec<_>>>()?;

    // Define upper triangular operator: A = [[I, B], [0, I]] on block rows
    let apply_a = |x: &BlockTensor<TensorTrain>| -> anyhow::Result<BlockTensor<TensorTrain>> {
        let mut result_blocks = Vec::with_capacity(indices.num_blocks);

        for row in 0..block_rows {
            for (col, cross_op) in cross_block_ops.iter().enumerate() {
                let flat = indices.flat_idx(row, col);
                let x_rc = block_entry(x, row, col)?;

                if row == 0 {
                    // y_{0,j} = I * x_{0,j} + B * x_{1,j}
                    let i_x = apply_operator_for_block(&identity_ops[flat], x_rc, &indices, flat)?;
                    let x_1j = block_entry(x, 1, col)?;
                    let b_x = apply_cross_block_operator(cross_op, x_1j, &indices, flat)?;
                    let y = i_x.axpby(AnyScalar::new_real(1.0), &b_x, AnyScalar::new_real(1.0))?;
                    result_blocks.push(y);
                } else {
                    // y_{1,j} = I * x_{1,j}
                    let y = apply_operator_for_block(&identity_ops[flat], x_rc, &indices, flat)?;
                    result_blocks.push(y);
                }
            }
        }

        BlockTensor::try_new(result_blocks, indices.block_shape).map_err(anyhow::Error::from)
    };

    // x_true: row 0 blocks = 2*ones, row 1 blocks = ones
    // Ax_true = [2*ones + ones, ones]^T = [3*ones, ones]^T per column
    let mut x_true_blocks = Vec::with_capacity(indices.num_blocks);
    for row in 0..block_rows {
        for col in 0..block_cols {
            let flat = indices.flat_idx(row, col);
            let value = if row == 0 { 2.0 } else { 1.0 };
            x_true_blocks.push(create_const_mpo_for_block(&indices, flat, value)?);
        }
    }
    let x_true = BlockTensor::try_new(x_true_blocks, indices.block_shape)?;
    println!("x_true created, norm: {:.6}", x_true.norm()?);

    // Compute b = A * x_true
    let b = apply_a(&x_true)?;
    let b_norm = b.norm()?;
    println!("b = A * x_true computed, norm: {:.6}", b_norm);

    // Initial guess: x0 = 0
    let x0 = x_true.scale(AnyScalar::new_real(0.0))?;

    // Compute initial residual
    let ax0 = apply_a(&x0)?;
    let r0 = ax0.axpby(AnyScalar::new_real(1.0), &b, AnyScalar::new_real(-1.0))?;
    let initial_residual = r0.norm()? / b_norm;
    println!("Initial residual: {:.6e}", initial_residual);

    let options = GmresOptions {
        max_iter: 20,
        rtol: 1e-8,
        max_restarts: 3,
        verbose: true,
        check_true_residual: false,
    };

    let truncate_opts = TruncateOptions::svd()
        .with_svd_policy(tensor4all_core::SvdTruncationPolicy::new(1e-10))
        .with_max_bond_dim(30);
    let truncate_fn = |x: &mut BlockTensor<TensorTrain>| -> anyhow::Result<()> {
        truncate_block_tensor(x, &truncate_opts)
    };

    println!("\nRunning GMRES...");
    let result = gmres_with_truncation(&apply_a, &b, &x0, &options, truncate_fn)?;

    // Compute true residual
    let ax_sol = apply_a(&result.solution)?;
    let r_final = ax_sol.axpby(AnyScalar::new_real(1.0), &b, AnyScalar::new_real(-1.0))?;
    let final_residual = r_final.norm()? / b_norm;

    // Compute error
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
// Test 4: Restart GMRES with block MPO
// ============================================================================

fn test_restart_gmres_block_mpo() -> anyhow::Result<()> {
    println!("=== Test 4: Restart GMRES with Block MPO ===");

    let block_rows = 2;
    let block_cols = 2;
    let n_sites = 2;
    let phys_dim = 2;

    println!(
        "Block structure: {}x{}, each block: N={} sites MPO",
        block_rows, block_cols, n_sites
    );
    println!("A = [[D, 0], [0, D]] where D = diag(2, 3) superoperator");

    let indices = BlockMpoSharedIndices::new(block_rows, block_cols, n_sites, phys_dim);

    // Create x_true = all-ones block MPO
    let x_true = create_block_ones_mpo(&indices)?;
    println!("x_true (all-ones) created, norm: {:.6}", x_true.norm()?);

    // Create diagonal superoperators for each block
    let diagonal_ops: Vec<TensorTrain> = (0..indices.num_blocks)
        .map(|i| create_diagonal_operator_for_block(&indices, i))
        .collect::<anyhow::Result<Vec<_>>>()?;

    // Define block diagonal operator
    let apply_a = |x: &BlockTensor<TensorTrain>| -> anyhow::Result<BlockTensor<TensorTrain>> {
        let mut result_blocks = Vec::with_capacity(indices.num_blocks);
        for (block_idx, op) in diagonal_ops.iter().enumerate() {
            let block = block_entry(x, block_idx / block_cols, block_idx % block_cols)?;
            let result = apply_operator_for_block(op, block, &indices, block_idx)?;
            result_blocks.push(result);
        }
        BlockTensor::try_new(result_blocks, indices.block_shape).map_err(anyhow::Error::from)
    };

    // Compute b = A * x_true
    let b = apply_a(&x_true)?;
    let b_norm = b.norm()?;
    println!("b = A * x_true computed, norm: {:.6}", b_norm);

    // Truncation with aggressive settings
    let truncate_opts = TruncateOptions::svd()
        .with_svd_policy(tensor4all_core::SvdTruncationPolicy::new(1e-6))
        .with_max_bond_dim(10);
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
        "Truncation: rtol={:.0e}, max_bond_dim={}",
        truncate_opts
            .svd_policy()
            .map(|policy| policy.threshold)
            .unwrap_or(0.0),
        truncate_opts.max_bond_dim().unwrap_or(0)
    );

    let result = restart_gmres_with_truncation(&apply_a, &b, None, &options, &truncate_fn)?;

    // Compute true residual
    let ax_sol = apply_a(&result.solution)?;
    let r_final = ax_sol.axpby(AnyScalar::new_real(1.0), &b, AnyScalar::new_real(-1.0))?;
    let final_residual = r_final.norm()? / b_norm;

    // Compute error
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
// Test 5: 2x2 block off-diagonal complex operator (i * Pauli-X) for MPO
// ============================================================================

fn test_block_offdiagonal_complex_pauli_x_mpo() -> anyhow::Result<()> {
    println!("=== Test 5: 2x2 Block Off-Diagonal Complex i*σ_x (MPO) ===");

    let block_rows = 2;
    let block_cols = 2;
    let n_sites = 2;
    let phys_dim = 2;

    println!(
        "Block structure: {}x{}, each block: N={} sites MPO, phys_dim={}",
        block_rows, block_cols, n_sites, phys_dim
    );
    println!("A = [[0, i*σ_x], [i*σ_x, 0]] acting on block rows");

    let indices = BlockMpoSharedIndices::new(block_rows, block_cols, n_sites, phys_dim);

    // Create i*σ_x cross-block operators for each column:
    // For column j: op_1to0[j] maps block (1,j) -> (0,j), op_0to1[j] maps block (0,j) -> (1,j)
    let ops_1to0: Vec<TensorTrain> = (0..block_cols)
        .map(|j| {
            let src = indices.flat_idx(1, j);
            let dst = indices.flat_idx(0, j);
            create_i_pauli_x_cross_block_operator(&indices, src, dst)
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    let ops_0to1: Vec<TensorTrain> = (0..block_cols)
        .map(|j| {
            let src = indices.flat_idx(0, j);
            let dst = indices.flat_idx(1, j);
            create_i_pauli_x_cross_block_operator(&indices, src, dst)
        })
        .collect::<anyhow::Result<Vec<_>>>()?;

    // x_true: row 0 = i*ones, row 1 = 2i*ones
    let mut x_true_blocks = Vec::with_capacity(indices.num_blocks);
    for row in 0..block_rows {
        let value = if row == 0 {
            Complex64::new(0.0, 1.0)
        } else {
            Complex64::new(0.0, 2.0)
        };
        for col in 0..block_cols {
            let flat = indices.flat_idx(row, col);
            x_true_blocks.push(create_complex_const_mpo_for_block(&indices, flat, value)?);
        }
    }
    let x_true = BlockTensor::try_new(x_true_blocks, indices.block_shape)?;
    println!("x_true created, norm: {:.6}", x_true.norm()?);

    // A = [[0, i*σ_x], [i*σ_x, 0]] acting on block rows
    // y_{0,j} = i*σ_x * x_{1,j}, y_{1,j} = i*σ_x * x_{0,j}
    let apply_a = |x: &BlockTensor<TensorTrain>| -> anyhow::Result<BlockTensor<TensorTrain>> {
        let mut result_blocks = Vec::with_capacity(indices.num_blocks);

        for (col, op) in ops_1to0.iter().enumerate() {
            // y_{0,j} = i*σ_x * x_{1,j}
            let x_1j = block_entry(x, 1, col)?;
            let src = indices.flat_idx(1, col);
            let dst = indices.flat_idx(0, col);
            let y = apply_cross_block_operator_full(op, x_1j, &indices, src, dst)?;
            result_blocks.push(y);
        }
        for (col, op) in ops_0to1.iter().enumerate() {
            // y_{1,j} = i*σ_x * x_{0,j}
            let x_0j = block_entry(x, 0, col)?;
            let src = indices.flat_idx(0, col);
            let dst = indices.flat_idx(1, col);
            let y = apply_cross_block_operator_full(op, x_0j, &indices, src, dst)?;
            result_blocks.push(y);
        }

        BlockTensor::try_new(result_blocks, indices.block_shape).map_err(anyhow::Error::from)
    };

    // Compute b = A * x_true
    let b = apply_a(&x_true)?;
    let b_norm = b.norm()?;
    println!("b = A * x_true computed, norm: {:.6}", b_norm);

    // Initial guess: complex zero
    let x0_blocks: Vec<TensorTrain> = (0..indices.num_blocks)
        .map(|block_idx| {
            create_complex_const_mpo_for_block(&indices, block_idx, Complex64::new(0.0, 0.0))
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    let x0 = BlockTensor::try_new(x0_blocks, indices.block_shape)?;
    println!("x0 (complex zero) created, norm: {:.6}", x0.norm()?);

    let options = GmresOptions {
        max_iter: 20,
        rtol: 1e-8,
        max_restarts: 1,
        verbose: true,
        check_true_residual: false,
    };

    let truncate_opts = TruncateOptions::svd()
        .with_svd_policy(tensor4all_core::SvdTruncationPolicy::new(1e-10))
        .with_max_bond_dim(30);
    let truncate_fn = |x: &mut BlockTensor<TensorTrain>| -> anyhow::Result<()> {
        truncate_block_tensor(x, &truncate_opts)
    };

    println!("\nRunning GMRES...");
    let result = gmres_with_truncation(&apply_a, &b, &x0, &options, truncate_fn)?;

    // Compute true residual
    let ax_sol = apply_a(&result.solution)?;
    let r_final = ax_sol.axpby(AnyScalar::new_real(1.0), &b, AnyScalar::new_real(-1.0))?;
    let final_residual = r_final.norm()? / b_norm;

    // Compute error
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

    assert!(
        result.converged,
        "GMRES should converge for off-diagonal complex i*σ_x operator (MPO)"
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
// Test 6: 3x3 block anti-diagonal complex operator (i * Pauli-X) for MPO
// ============================================================================

fn test_3x3_block_antidiagonal_complex_pauli_x_mpo() -> anyhow::Result<()> {
    println!("=== Test 6: 3x3 Block Anti-Diagonal Complex i*σ_x (MPO) ===");

    let block_rows = 3;
    let block_cols = 3;
    let n_sites = 2;
    let phys_dim = 2;

    println!(
        "Block structure: {}x{}, each block: N={} sites MPO, phys_dim={}",
        block_rows, block_cols, n_sites, phys_dim
    );
    println!("A = [[0, 0, i*σ_x], [0, i*σ_x, 0], [i*σ_x, 0, 0]] acting on block rows");

    let indices = BlockMpoSharedIndices::new(block_rows, block_cols, n_sites, phys_dim);

    // Create i*σ_x cross-block operators for each column:
    // A[0,2]: row 2 -> row 0, A[1,1]: row 1 -> row 1, A[2,0]: row 0 -> row 2
    let ops_2to0: Vec<TensorTrain> = (0..block_cols)
        .map(|j| {
            let src = indices.flat_idx(2, j);
            let dst = indices.flat_idx(0, j);
            create_i_pauli_x_cross_block_operator(&indices, src, dst)
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    let ops_1to1: Vec<TensorTrain> = (0..block_cols)
        .map(|j| {
            let src = indices.flat_idx(1, j);
            let dst = indices.flat_idx(1, j);
            create_i_pauli_x_cross_block_operator(&indices, src, dst)
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    let ops_0to2: Vec<TensorTrain> = (0..block_cols)
        .map(|j| {
            let src = indices.flat_idx(0, j);
            let dst = indices.flat_idx(2, j);
            create_i_pauli_x_cross_block_operator(&indices, src, dst)
        })
        .collect::<anyhow::Result<Vec<_>>>()?;

    // x_true: row 0 = i*ones, row 1 = 2i*ones, row 2 = 3i*ones
    let mut x_true_blocks = Vec::with_capacity(indices.num_blocks);
    for row in 0..block_rows {
        let value = match row {
            0 => Complex64::new(0.0, 1.0),
            1 => Complex64::new(0.0, 2.0),
            _ => Complex64::new(0.0, 3.0),
        };
        for col in 0..block_cols {
            let flat = indices.flat_idx(row, col);
            x_true_blocks.push(create_complex_const_mpo_for_block(&indices, flat, value)?);
        }
    }
    let x_true = BlockTensor::try_new(x_true_blocks, indices.block_shape)?;
    println!("x_true created, norm: {:.6}", x_true.norm()?);

    // A = [[0, 0, i*σ_x], [0, i*σ_x, 0], [i*σ_x, 0, 0]] acting on block rows
    // y_{0,j} = i*σ_x * x_{2,j}, y_{1,j} = i*σ_x * x_{1,j}, y_{2,j} = i*σ_x * x_{0,j}
    let apply_a = |x: &BlockTensor<TensorTrain>| -> anyhow::Result<BlockTensor<TensorTrain>> {
        let mut result_blocks = Vec::with_capacity(indices.num_blocks);

        // Row 0: y_{0,j} = i*σ_x * x_{2,j}
        for (col, op) in ops_2to0.iter().enumerate() {
            let x_2j = block_entry(x, 2, col)?;
            let src = indices.flat_idx(2, col);
            let dst = indices.flat_idx(0, col);
            let y = apply_cross_block_operator_full(op, x_2j, &indices, src, dst)?;
            result_blocks.push(y);
        }
        // Row 1: y_{1,j} = i*σ_x * x_{1,j}
        for (col, op) in ops_1to1.iter().enumerate() {
            let x_1j = block_entry(x, 1, col)?;
            let src = indices.flat_idx(1, col);
            let dst = indices.flat_idx(1, col);
            let y = apply_cross_block_operator_full(op, x_1j, &indices, src, dst)?;
            result_blocks.push(y);
        }
        // Row 2: y_{2,j} = i*σ_x * x_{0,j}
        for (col, op) in ops_0to2.iter().enumerate() {
            let x_0j = block_entry(x, 0, col)?;
            let src = indices.flat_idx(0, col);
            let dst = indices.flat_idx(2, col);
            let y = apply_cross_block_operator_full(op, x_0j, &indices, src, dst)?;
            result_blocks.push(y);
        }

        BlockTensor::try_new(result_blocks, indices.block_shape).map_err(anyhow::Error::from)
    };

    // Compute b = A * x_true
    let b = apply_a(&x_true)?;
    let b_norm = b.norm()?;
    println!("b = A * x_true computed, norm: {:.6}", b_norm);

    // Initial guess: complex zero
    let x0_blocks: Vec<TensorTrain> = (0..indices.num_blocks)
        .map(|block_idx| {
            create_complex_const_mpo_for_block(&indices, block_idx, Complex64::new(0.0, 0.0))
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    let x0 = BlockTensor::try_new(x0_blocks, indices.block_shape)?;
    println!("x0 (complex zero) created, norm: {:.6}", x0.norm()?);

    let options = GmresOptions {
        max_iter: 30,
        rtol: 1e-8,
        max_restarts: 1,
        verbose: true,
        check_true_residual: false,
    };

    let truncate_opts = TruncateOptions::svd()
        .with_svd_policy(tensor4all_core::SvdTruncationPolicy::new(1e-10))
        .with_max_bond_dim(30);
    let truncate_fn = |x: &mut BlockTensor<TensorTrain>| -> anyhow::Result<()> {
        truncate_block_tensor(x, &truncate_opts)
    };

    println!("\nRunning GMRES...");
    let result = gmres_with_truncation(&apply_a, &b, &x0, &options, truncate_fn)?;

    // Compute true residual
    let ax_sol = apply_a(&result.solution)?;
    let r_final = ax_sol.axpby(AnyScalar::new_real(1.0), &b, AnyScalar::new_real(-1.0))?;
    let final_residual = r_final.norm()? / b_norm;

    // Compute error
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

    assert!(
        result.converged,
        "GMRES should converge for 3x3 anti-diagonal complex i*σ_x operator (MPO)"
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
// Scaling study: 2x2 off-diagonal i*σ_x (MPO) with varying n_sites
// ============================================================================

fn scaling_2x2_offdiagonal_mpo(n_sites: usize) -> anyhow::Result<()> {
    let block_rows = 2;
    let block_cols = 2;
    let phys_dim = 2;

    let indices = BlockMpoSharedIndices::new(block_rows, block_cols, n_sites, phys_dim);

    // Create i*σ_x cross-block operators
    let ops_1to0: Vec<TensorTrain> = (0..block_cols)
        .map(|j| {
            let src = indices.flat_idx(1, j);
            let dst = indices.flat_idx(0, j);
            create_i_pauli_x_cross_block_operator(&indices, src, dst)
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    let ops_0to1: Vec<TensorTrain> = (0..block_cols)
        .map(|j| {
            let src = indices.flat_idx(0, j);
            let dst = indices.flat_idx(1, j);
            create_i_pauli_x_cross_block_operator(&indices, src, dst)
        })
        .collect::<anyhow::Result<Vec<_>>>()?;

    // x_true: row 0 = i*ones, row 1 = 2i*ones
    let mut x_true_blocks = Vec::with_capacity(indices.num_blocks);
    for row in 0..block_rows {
        let value = if row == 0 {
            Complex64::new(0.0, 1.0)
        } else {
            Complex64::new(0.0, 2.0)
        };
        for col in 0..block_cols {
            let flat = indices.flat_idx(row, col);
            x_true_blocks.push(create_complex_const_mpo_for_block(&indices, flat, value)?);
        }
    }
    let x_true = BlockTensor::try_new(x_true_blocks, indices.block_shape)?;

    let apply_a = |x: &BlockTensor<TensorTrain>| -> anyhow::Result<BlockTensor<TensorTrain>> {
        let mut result_blocks = Vec::with_capacity(indices.num_blocks);

        for (col, op) in ops_1to0.iter().enumerate() {
            let x_1j = block_entry(x, 1, col)?;
            let src = indices.flat_idx(1, col);
            let dst = indices.flat_idx(0, col);
            let y = apply_cross_block_operator_full(op, x_1j, &indices, src, dst)?;
            result_blocks.push(y);
        }
        for (col, op) in ops_0to1.iter().enumerate() {
            let x_0j = block_entry(x, 0, col)?;
            let src = indices.flat_idx(0, col);
            let dst = indices.flat_idx(1, col);
            let y = apply_cross_block_operator_full(op, x_0j, &indices, src, dst)?;
            result_blocks.push(y);
        }

        BlockTensor::try_new(result_blocks, indices.block_shape).map_err(anyhow::Error::from)
    };

    let b = apply_a(&x_true)?;
    let b_norm = b.norm()?;

    let x0_blocks: Vec<TensorTrain> = (0..indices.num_blocks)
        .map(|block_idx| {
            create_complex_const_mpo_for_block(&indices, block_idx, Complex64::new(0.0, 0.0))
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    let x0 = BlockTensor::try_new(x0_blocks, indices.block_shape)?;

    let options = GmresOptions {
        max_iter: 20,
        rtol: 1e-8,
        max_restarts: 1,
        verbose: false,
        check_true_residual: false,
    };

    let truncate_opts = TruncateOptions::svd()
        .with_svd_policy(tensor4all_core::SvdTruncationPolicy::new(1e-10))
        .with_max_bond_dim(30);
    let truncate_fn = |x: &mut BlockTensor<TensorTrain>| -> anyhow::Result<()> {
        truncate_block_tensor(x, &truncate_opts)
    };

    let start = Instant::now();
    let result = gmres_with_truncation(&apply_a, &b, &x0, &options, truncate_fn)?;
    let elapsed = start.elapsed();

    let ax_sol = apply_a(&result.solution)?;
    let r_final = ax_sol.axpby(AnyScalar::new_real(1.0), &b, AnyScalar::new_real(-1.0))?;
    let final_residual = r_final.norm()? / b_norm;

    let block_dim: usize = (0..n_sites).map(|_| phys_dim).product::<usize>().pow(2);
    let total_dim = block_dim * block_rows * block_cols;

    println!(
        "  N={:>2}: block_dim={}x{}, total={}x{}, iters={}, residual={:.2e}, time={:.3}s",
        n_sites,
        block_dim,
        block_dim,
        total_dim,
        total_dim,
        result.iterations,
        final_residual,
        elapsed.as_secs_f64()
    );

    assert!(
        result.converged,
        "2x2 MPO scaling: GMRES should converge for N={}",
        n_sites
    );
    assert!(
        final_residual < 1e-6,
        "2x2 MPO scaling: residual too large for N={}: {}",
        n_sites,
        final_residual
    );

    Ok(())
}

// ============================================================================
// Scaling study: 3x3 anti-diagonal i*σ_x (MPO) with varying n_sites
// ============================================================================

fn scaling_3x3_antidiagonal_mpo(n_sites: usize) -> anyhow::Result<()> {
    let block_rows = 3;
    let block_cols = 3;
    let phys_dim = 2;

    let indices = BlockMpoSharedIndices::new(block_rows, block_cols, n_sites, phys_dim);

    // Create i*σ_x cross-block operators
    let ops_2to0: Vec<TensorTrain> = (0..block_cols)
        .map(|j| {
            let src = indices.flat_idx(2, j);
            let dst = indices.flat_idx(0, j);
            create_i_pauli_x_cross_block_operator(&indices, src, dst)
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    let ops_1to1: Vec<TensorTrain> = (0..block_cols)
        .map(|j| {
            let src = indices.flat_idx(1, j);
            let dst = indices.flat_idx(1, j);
            create_i_pauli_x_cross_block_operator(&indices, src, dst)
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    let ops_0to2: Vec<TensorTrain> = (0..block_cols)
        .map(|j| {
            let src = indices.flat_idx(0, j);
            let dst = indices.flat_idx(2, j);
            create_i_pauli_x_cross_block_operator(&indices, src, dst)
        })
        .collect::<anyhow::Result<Vec<_>>>()?;

    // x_true: row 0 = i*ones, row 1 = 2i*ones, row 2 = 3i*ones
    let mut x_true_blocks = Vec::with_capacity(indices.num_blocks);
    for row in 0..block_rows {
        let value = match row {
            0 => Complex64::new(0.0, 1.0),
            1 => Complex64::new(0.0, 2.0),
            _ => Complex64::new(0.0, 3.0),
        };
        for col in 0..block_cols {
            let flat = indices.flat_idx(row, col);
            x_true_blocks.push(create_complex_const_mpo_for_block(&indices, flat, value)?);
        }
    }
    let x_true = BlockTensor::try_new(x_true_blocks, indices.block_shape)?;

    let apply_a = |x: &BlockTensor<TensorTrain>| -> anyhow::Result<BlockTensor<TensorTrain>> {
        let mut result_blocks = Vec::with_capacity(indices.num_blocks);

        for (col, op) in ops_2to0.iter().enumerate() {
            let x_2j = block_entry(x, 2, col)?;
            let src = indices.flat_idx(2, col);
            let dst = indices.flat_idx(0, col);
            let y = apply_cross_block_operator_full(op, x_2j, &indices, src, dst)?;
            result_blocks.push(y);
        }
        for (col, op) in ops_1to1.iter().enumerate() {
            let x_1j = block_entry(x, 1, col)?;
            let src = indices.flat_idx(1, col);
            let dst = indices.flat_idx(1, col);
            let y = apply_cross_block_operator_full(op, x_1j, &indices, src, dst)?;
            result_blocks.push(y);
        }
        for (col, op) in ops_0to2.iter().enumerate() {
            let x_0j = block_entry(x, 0, col)?;
            let src = indices.flat_idx(0, col);
            let dst = indices.flat_idx(2, col);
            let y = apply_cross_block_operator_full(op, x_0j, &indices, src, dst)?;
            result_blocks.push(y);
        }

        BlockTensor::try_new(result_blocks, indices.block_shape).map_err(anyhow::Error::from)
    };

    let b = apply_a(&x_true)?;
    let b_norm = b.norm()?;

    let x0_blocks: Vec<TensorTrain> = (0..indices.num_blocks)
        .map(|block_idx| {
            create_complex_const_mpo_for_block(&indices, block_idx, Complex64::new(0.0, 0.0))
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    let x0 = BlockTensor::try_new(x0_blocks, indices.block_shape)?;

    let options = GmresOptions {
        max_iter: 30,
        rtol: 1e-8,
        max_restarts: 1,
        verbose: false,
        check_true_residual: false,
    };

    let truncate_opts = TruncateOptions::svd()
        .with_svd_policy(tensor4all_core::SvdTruncationPolicy::new(1e-10))
        .with_max_bond_dim(30);
    let truncate_fn = |x: &mut BlockTensor<TensorTrain>| -> anyhow::Result<()> {
        truncate_block_tensor(x, &truncate_opts)
    };

    let start = Instant::now();
    let result = gmres_with_truncation(&apply_a, &b, &x0, &options, truncate_fn)?;
    let elapsed = start.elapsed();

    let ax_sol = apply_a(&result.solution)?;
    let r_final = ax_sol.axpby(AnyScalar::new_real(1.0), &b, AnyScalar::new_real(-1.0))?;
    let final_residual = r_final.norm()? / b_norm;

    let block_dim: usize = (0..n_sites).map(|_| phys_dim).product::<usize>().pow(2);
    let total_dim = block_dim * block_rows * block_cols;

    println!(
        "  N={:>2}: block_dim={}x{}, total={}x{}, iters={}, residual={:.2e}, time={:.3}s",
        n_sites,
        block_dim,
        block_dim,
        total_dim,
        total_dim,
        result.iterations,
        final_residual,
        elapsed.as_secs_f64()
    );

    assert!(
        result.converged,
        "3x3 MPO scaling: GMRES should converge for N={}",
        n_sites
    );
    assert!(
        final_residual < 1e-6,
        "3x3 MPO scaling: residual too large for N={}: {}",
        n_sites,
        final_residual
    );

    Ok(())
}

// ============================================================================
// Scaling study runner
// ============================================================================

fn test_scaling_study_mpo() -> anyhow::Result<()> {
    println!("=== Scaling Study: MPO Block GMRES with varying n_sites ===\n");

    let n_sites_list = [2, 4, 6, 8, 10, 14, 20];

    println!("--- 2x2 off-diagonal i*σ_x (MPO) ---");
    for &n in &n_sites_list {
        scaling_2x2_offdiagonal_mpo(n)?;
    }

    println!("\n--- 3x3 anti-diagonal i*σ_x (MPO) ---");
    for &n in &n_sites_list {
        scaling_3x3_antidiagonal_mpo(n)?;
    }

    println!("\nScaling study PASSED\n");
    Ok(())
}

// ============================================================================
// Main
// ============================================================================

fn main() -> anyhow::Result<()> {
    println!("========================================");
    println!("  Block MPO GMRES Tests");
    println!("========================================\n");

    // Test 1: Block diagonal identity
    test_block_diagonal_identity()?;

    // Test 2: Block diagonal non-trivial operator (diagonal superoperator)
    test_block_diagonal_diagonal_operator()?;

    // Test 3: Block upper triangular
    test_block_upper_triangular()?;

    // Test 4: Restart GMRES with block MPO
    test_restart_gmres_block_mpo()?;

    // Test 5: 2x2 off-diagonal complex operator (i * Pauli-X)
    test_block_offdiagonal_complex_pauli_x_mpo()?;

    // Test 6: 3x3 block anti-diagonal complex operator (i * Pauli-X)
    test_3x3_block_antidiagonal_complex_pauli_x_mpo()?;

    // Scaling study: MPO block GMRES with varying n_sites
    test_scaling_study_mpo()?;

    println!("========================================");
    println!("  All tests completed!");
    println!("========================================");

    Ok(())
}
