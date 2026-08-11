//! Profiled version of benchmark_contract with detailed timing.

use rand::rng;
use std::time::Instant;

use anyhow::Result;
use tensor4all_core::{DynIndex, TensorDynLen};
use tensor4all_itensorlike::{ContractOptions, TensorTrain};

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

        let tensor = TensorDynLen::random::<f64, _>(&mut rng, indices)?;
        tensors.push(tensor);
    }

    Ok(TensorTrain::new(tensors)?)
}

fn main() -> Result<()> {
    let length = 10;
    let phys_dim = 2;
    let bond_dim = 20;

    println!("=== Profiled MPO Contraction Benchmark ===");
    println!("Length: {} sites", length);
    println!("Physical dimension: {}", phys_dim);
    println!("Bond dimension: {}", bond_dim);
    println!();

    // Create MPO A
    let t0 = Instant::now();
    let input_indices_a: Vec<_> = (0..length).map(|_| DynIndex::new_dyn(phys_dim)).collect();
    let output_indices_shared: Vec<_> = (0..length).map(|_| DynIndex::new_dyn(phys_dim)).collect();
    let num_links = if length > 0 { length - 1 } else { 0 };
    let link_indices_a: Vec<_> = (0..num_links)
        .map(|_| DynIndex::new_dyn(bond_dim))
        .collect();
    let mpo_a = create_random_mpo(
        length,
        phys_dim,
        bond_dim,
        &input_indices_a,
        &output_indices_shared,
        &link_indices_a,
    )?;
    let t_create_a = t0.elapsed();
    println!("Creating MPO A: {:?}", t_create_a);

    // Create MPO B
    let t0 = Instant::now();
    let output_indices_b: Vec<_> = (0..length).map(|_| DynIndex::new_dyn(phys_dim)).collect();
    let link_indices_b: Vec<_> = (0..num_links)
        .map(|_| DynIndex::new_dyn(bond_dim))
        .collect();
    let mpo_b = create_random_mpo(
        length,
        phys_dim,
        bond_dim,
        &output_indices_shared,
        &output_indices_b,
        &link_indices_b,
    )?;
    let t_create_b = t0.elapsed();
    println!("Creating MPO B: {:?}", t_create_b);
    println!();

    // Contract with profiling
    let max_rank = 20;
    let options = ContractOptions::zipup().with_max_rank(max_rank);

    println!("Contracting MPOs using zip-up method...");
    println!(
        "Options: method=Zipup, max_rank={}, rtol={:?}",
        max_rank,
        options.svd_policy().map(|policy| policy.threshold)
    );
    println!();

    let total_start = Instant::now();
    let result = mpo_a.contract(&mpo_b, &options)?;
    let total_duration = total_start.elapsed();

    println!("=== Timing Results ===");
    println!("MPO creation (A): {:?}", t_create_a);
    println!("MPO creation (B): {:?}", t_create_b);
    println!("Contraction: {:?}", total_duration);
    println!("Total: {:?}", t_create_a + t_create_b + total_duration);
    println!();
    println!("=== Results ===");
    println!(
        "Resulting MPO max bond dimension: {}",
        result.max_bond_dim()
    );
    println!("Resulting MPO bond dimensions: {:?}", result.bond_dims());

    Ok(())
}
