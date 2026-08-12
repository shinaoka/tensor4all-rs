//! Detailed benchmark with step-by-step timing and comparison with Julia.

use rand::rng;
use std::time::Instant;

use anyhow::Result;
use tensor4all_core::{DynIndex, IdxTensor};
use tensor4all_itensorlike::{CanonicalForm, ContractOptions, TensorTrain};

/// Create a random MPO (Matrix Product Operator).
fn create_random_mpo(
    length: usize,
    _phys_dim: usize,
    _bond_dim: usize,
    input_indices: &[DynIndex],
    output_indices: &[DynIndex],
    link_indices: &[DynIndex],
) -> Result<TensorTrain> {
    let mut rng = rng();
    let mut tensors = Vec::with_capacity(length);

    for i in 0..length {
        let mut indices = vec![input_indices[i].clone(), output_indices[i].clone()];

        if i > 0 {
            indices.push(link_indices[i - 1].clone());
        }

        if i < length - 1 {
            indices.push(link_indices[i].clone());
        }

        let tensor = IdxTensor::random::<f64, _>(&mut rng, indices)?;
        tensors.push(tensor);
    }

    Ok(TensorTrain::new(tensors)?)
}

fn main() -> Result<()> {
    let length = 10;
    let phys_dim = 2;
    let bond_dim = 50;
    let max_bond_dim = 50;
    let n_runs = 5;

    println!("=== Detailed MPO Contraction Benchmark (Rust) ===");
    println!("Length: {} sites", length);
    println!("Bond dimension: {}", bond_dim);
    println!("Max rank: {}", max_bond_dim);
    println!();

    // Create fixed MPOs
    let input_indices_a: Vec<_> = (0..length).map(|_| DynIndex::new_dyn(phys_dim)).collect();
    let output_indices_shared: Vec<_> = (0..length).map(|_| DynIndex::new_dyn(phys_dim)).collect();
    let num_links = if length > 0 { length - 1 } else { 0 };
    let link_indices_a: Vec<_> = (0..num_links)
        .map(|_| DynIndex::new_dyn(bond_dim))
        .collect();

    let mpo_a_original = create_random_mpo(
        length,
        phys_dim,
        bond_dim,
        &input_indices_a,
        &output_indices_shared,
        &link_indices_a,
    )?;

    let output_indices_b: Vec<_> = (0..length).map(|_| DynIndex::new_dyn(phys_dim)).collect();
    let link_indices_b: Vec<_> = (0..num_links)
        .map(|_| DynIndex::new_dyn(bond_dim))
        .collect();

    let mpo_b_original = create_random_mpo(
        length,
        phys_dim,
        bond_dim,
        &output_indices_shared,
        &output_indices_b,
        &link_indices_b,
    )?;

    println!("MPO A: max bond dim = {}", mpo_a_original.max_bond_dim());
    println!("MPO B: max bond dim = {}", mpo_b_original.max_bond_dim());
    println!("Expected number of edges to process: {}", length - 1);
    println!();

    // Enable profiling
    std::env::set_var("T4A_PROFILE_CONTRACTION", "1");

    let options = ContractOptions::zipup().with_max_bond_dim(max_bond_dim);

    println!("Running {} iterations...", n_runs);
    let mut times = Vec::new();

    for run in 1..=n_runs {
        println!("\n--- Run {} ---", run);
        let start = Instant::now();

        let mut mpo_a = mpo_a_original.clone();
        let mut mpo_b = mpo_b_original.clone();

        // Orthogonalize
        let t_ortho = Instant::now();
        mpo_a.orthogonalize_with(length - 1, CanonicalForm::Unitary)?;
        mpo_b.orthogonalize_with(length - 1, CanonicalForm::Unitary)?;
        let ortho_time = t_ortho.elapsed();
        println!("Orthogonalization: {:?}", ortho_time);

        // Contract
        let t_contract = Instant::now();
        let result = mpo_a.contract(&mpo_b, &options)?;
        let contract_time = t_contract.elapsed();
        println!("Contraction: {:?}", contract_time);

        let total_time = start.elapsed();
        times.push(total_time);
        println!(
            "Total: {:?} (max bond dim: {})",
            total_time,
            result.max_bond_dim()
        );
    }

    let avg_time: std::time::Duration =
        times.iter().sum::<std::time::Duration>() / times.len() as u32;
    println!("\n=== Summary ===");
    println!("Average time: {:?}", avg_time);
    println!("Min time: {:?}", times.iter().min().unwrap());
    println!("Max time: {:?}", times.iter().max().unwrap());

    Ok(())
}
