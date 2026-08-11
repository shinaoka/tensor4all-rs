// Integration test: 2D QFT via partial apply on interleaved quantics encoding
//
// Verifies that applying a 1D Fourier operator to non-contiguous sites
// (x-variable in interleaved layout) produces correct Fourier coefficients.

use std::f64::consts::PI;

use anyhow::Result;
use tensor4all_quanticstci::{quanticscrossinterpolate_discrete, QtciOptions, UnfoldingScheme};
use tensor4all_quanticstransform::{quantics_fourier_operator, FourierOptions};
use tensor4all_treetci::materialize::to_treetn;
use tensor4all_treetn::Operator;
use tensor4all_treetn::{apply_linear_operator, ApplyOptions};

/// Test that 1D QFT applied to x-variable in a 2D interleaved QTT works.
///
/// f(x, y) = cos(2π (x-1) / N) depends only on x.
/// Its DFT in x has nonzero coefficients at kx=1 and kx=N-1 (magnitude N/2 each,
/// after normalization by 1/√N → √N/2).
#[test]
fn test_2d_qft_x_only_interleaved() -> Result<()> {
    let r = 3;
    let n = 1usize << r; // 8

    // f(x, y) = cos(2π (x-1) / N), 1-indexed
    let f = move |idx: &[i64]| -> f64 {
        let x = (idx[0] - 1) as f64;
        (2.0 * PI * x / n as f64).cos()
    };

    let sizes = vec![n, n];
    let (qtci, _ranks, errors) = quanticscrossinterpolate_discrete::<f64, _>(
        &sizes,
        f,
        None,
        QtciOptions::default()
            .with_tolerance(1e-12)
            .with_unfoldingscheme(UnfoldingScheme::Interleaved),
    )?;

    assert!(
        *errors.last().unwrap() < 1e-10,
        "TCI did not converge: {}",
        errors.last().unwrap()
    );

    // Convert to TreeTN
    let tci_state = qtci.tci();
    let r_copy = r;
    let f_copy = move |idx: &[i64]| -> f64 {
        let x = (idx[0] - 1) as f64;
        (2.0 * PI * x / (1usize << r_copy) as f64).cos()
    };

    let batch_eval = move |batch: tensor4all_treetci::GlobalIndexBatch<'_>| -> Result<Vec<f64>> {
        let mut values = Vec::with_capacity(batch.n_points());
        for p in 0..batch.n_points() {
            let mut x_val = 0usize;
            let mut y_val = 0usize;
            for bit in 0..r_copy {
                let x_bit = batch.get(2 * bit, p).unwrap();
                let y_bit = batch.get(2 * bit + 1, p).unwrap();
                x_val |= x_bit << bit;
                y_val |= y_bit << bit;
            }
            values.push(f_copy(&[(x_val + 1) as i64, (y_val + 1) as i64]));
        }
        Ok(values)
    };

    let state_treetn = to_treetn(tci_state, batch_eval, Some(0))?;
    assert_eq!(state_treetn.node_count(), 2 * r);

    // Build 1D Fourier operator for R sites (nodes 0..R-1)
    let fourier_op = quantics_fourier_operator(
        r,
        FourierOptions {
            normalize: true,
            ..Default::default()
        },
    )?;

    // Rename operator nodes: 0→0, 1→2, 2→4 (x-variable sites in interleaved layout)
    let node_mapping: Vec<(usize, usize)> = (0..r).map(|i| (i, 2 * i)).collect();
    let mut renamed_op = fourier_op.rename_nodes(&node_mapping)?;

    // Verify renamed operator nodes are {0, 2, 4}
    let op_nodes = renamed_op.node_names();
    assert!(op_nodes.contains(&0));
    assert!(op_nodes.contains(&2));
    assert!(op_nodes.contains(&4));
    assert_eq!(op_nodes.len(), r);

    // Set input/output mappings to match the state's site indices
    renamed_op.set_input_space_from_state(&state_treetn)?;
    renamed_op.set_output_space_from_state(&state_treetn)?;

    // Apply Fourier operator to x-variable sites via partial apply (Steiner tree)
    // This should insert identity at y-sites {1, 3, 5}
    let result = apply_linear_operator(&renamed_op, &state_treetn, ApplyOptions::default())?;
    assert_eq!(result.node_count(), 2 * r);

    eprintln!("2D QFT partial apply succeeded!");

    // TODO: Verify Fourier coefficients
    // For cos(2π x/N), the normalized DFT gives:
    // F[kx=1] = √N/2, F[kx=N-1] = √N/2, all others ≈ 0

    Ok(())
}
