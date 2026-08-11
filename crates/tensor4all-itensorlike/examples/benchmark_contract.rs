//! Benchmark: Contract two random MPOs with bond dimension 20 using zip-up method.
//!
//! This example creates two random MPOs (Matrix Product Operators) with:
//! - Length: 10 sites
//! - Physical dimension: 2 per site (input and output)
//! - Bond dimension: 20
//!
//! Then contracts them using zip-up method with cutoff=0 (no truncation).

use rand::rng;
use std::time::Instant;

use anyhow::Result;
use tensor4all_core::{DynIndex, TensorDynLen};
use tensor4all_itensorlike::{CanonicalForm, ContractOptions, TensorTrain};

/// Create a random MPO (Matrix Product Operator).
///
/// An MPO has two physical indices per site: one for input and one for output.
///
/// # Arguments
/// * `length` - Number of sites
/// * `phys_dim` - Physical dimension (same for input and output)
/// * `bond_dim` - Bond dimension between sites
/// * `input_indices` - Input physical indices (one per site)
/// * `output_indices` - Output physical indices (one per site)
/// * `link_indices` - Link indices (one per bond, length should be length-1)
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

    // Create tensors for each site
    for i in 0..length {
        let mut indices = vec![input_indices[i].clone(), output_indices[i].clone()];

        // Add left link (if not first site)
        if i > 0 {
            indices.push(link_indices[i - 1].clone());
        }

        // Add right link (if not last site)
        if i < length - 1 {
            indices.push(link_indices[i].clone());
        }

        // Create random tensor
        let tensor = TensorDynLen::random::<f64, _>(&mut rng, indices)?;
        tensors.push(tensor);
    }

    Ok(TensorTrain::new(tensors)?)
}

fn main() -> Result<()> {
    // Parameters
    let length = 10;
    let phys_dim = 2;
    let bond_dim = 50;
    let max_bond_dim = 50;
    let n_runs = 10; // Number of runs for averaging

    println!("=== Random MPO Contraction Benchmark (Rust/tensor4all-rs) ===");
    println!("Length: {} sites", length);
    println!("Physical dimension: {}", phys_dim);
    println!("Bond dimension: {}", bond_dim);
    println!("Max rank: {}", max_bond_dim);
    println!(
        "Number of runs: {} (excluding first compilation run)",
        n_runs
    );
    println!();

    // Create indices for MPO A: A[s_i, s'_i]
    // Input indices (s_i) - unique for MPO A
    let input_indices_a: Vec<_> = (0..length).map(|_| DynIndex::new_dyn(phys_dim)).collect();

    // Output indices (s'_i) - shared between MPO A and MPO B for contraction
    let output_indices_shared: Vec<_> = (0..length).map(|_| DynIndex::new_dyn(phys_dim)).collect();

    // Link indices for MPO A
    let num_links = if length > 0 { length - 1 } else { 0 };
    let link_indices_a: Vec<_> = (0..num_links)
        .map(|_| DynIndex::new_dyn(bond_dim))
        .collect();

    // Create first MPO: A[s_i, s'_i] (input s, output s')
    // Create once and keep fixed for all measurements
    println!("Creating first MPO (A) [fixed for all runs]...");
    let mpo_a_original = create_random_mpo(
        length,
        phys_dim,
        bond_dim,
        &input_indices_a,
        &output_indices_shared,
        &link_indices_a,
    )?;
    println!(
        "MPO A created. Max bond dim: {}",
        mpo_a_original.max_bond_dim()
    );
    println!();

    // Create indices for MPO B: B[s'_i, s''_i]
    // Input indices (s'_i) - same as output_indices_shared (for contraction)
    // Output indices (s''_i) - unique for MPO B
    let output_indices_b: Vec<_> = (0..length).map(|_| DynIndex::new_dyn(phys_dim)).collect();

    // Link indices for MPO B
    let num_links = if length > 0 { length - 1 } else { 0 };
    let link_indices_b: Vec<_> = (0..num_links)
        .map(|_| DynIndex::new_dyn(bond_dim))
        .collect();

    // Create second MPO: B[s'_i, s''_i] (input s', output s'')
    // Create once and keep fixed for all measurements
    println!("Creating second MPO (B) [fixed for all runs]...");
    let mpo_b_original = create_random_mpo(
        length,
        phys_dim,
        bond_dim,
        &output_indices_shared,
        &output_indices_b,
        &link_indices_b,
    )?;
    println!(
        "MPO B created. Max bond dim: {}",
        mpo_b_original.max_bond_dim()
    );
    println!();

    // Contract options: zip-up with max_bond_dim=50
    let options = ContractOptions::zipup().with_max_bond_dim(max_bond_dim);

    println!("Contracting MPOs using zip-up method...");
    println!(
        "Options: method=Zipup, max_bond_dim={}, rtol={:?}",
        max_bond_dim,
        options.svd_policy().map(|policy| policy.threshold)
    );
    println!("Note: Each run copies MPOs and includes orthogonalization time");
    println!();

    // First run (compilation/warmup - excluded from average)
    println!("Warmup run (excluded from average)...");
    let start_warmup = Instant::now();
    let mut mpo_a_warmup = mpo_a_original.clone();
    let mut mpo_b_warmup = mpo_b_original.clone();
    // Orthogonalize (included in timing)
    // Match ITensorMPS.jl's zip-up implementation which orthogonalizes to the left edge.
    // Note: site is 0-indexed.
    mpo_a_warmup.orthogonalize_with(0, CanonicalForm::Unitary)?;
    mpo_b_warmup.orthogonalize_with(0, CanonicalForm::Unitary)?;
    // Contract
    let result_warmup = mpo_a_warmup.contract(&mpo_b_warmup, &options)?;
    let duration_warmup = start_warmup.elapsed();
    println!(
        "Warmup completed in: {:?}. Result max bond dim: {}",
        duration_warmup,
        result_warmup.max_bond_dim()
    );
    println!();

    // Multiple runs for averaging
    println!("Running {} iterations for averaging...", n_runs);
    let mut times = Vec::new();
    let mut result_final = None;

    for run in 1..=n_runs {
        let start = Instant::now();
        // Copy MPOs for this run
        let mut mpo_a = mpo_a_original.clone();
        let mut mpo_b = mpo_b_original.clone();
        // Orthogonalize (included in timing)
        // Match ITensorMPS.jl's zip-up implementation which orthogonalizes to the left edge.
        // Note: site is 0-indexed.
        mpo_a.orthogonalize_with(0, CanonicalForm::Unitary)?;
        mpo_b.orthogonalize_with(0, CanonicalForm::Unitary)?;
        // Contract
        let result = mpo_a.contract(&mpo_b, &options)?;
        let duration = start.elapsed();
        times.push(duration);
        result_final = Some(result.clone());
        println!(
            "  Run {}: {:?} (max bond dim: {})",
            run,
            duration,
            result.max_bond_dim()
        );
    }

    // Calculate statistics
    let total: std::time::Duration = times.iter().sum();
    let avg_time = total / times.len() as u32;
    let min_time = times.iter().min().unwrap();
    let max_time = times.iter().max().unwrap();

    // Calculate standard deviation
    let avg_secs = avg_time.as_secs_f64();
    let variance: f64 = times
        .iter()
        .map(|&d| {
            let diff = d.as_secs_f64() - avg_secs;
            diff * diff
        })
        .sum::<f64>()
        / times.len() as f64;
    let std_time = std::time::Duration::from_secs_f64(variance.sqrt());

    println!();
    println!("=== Results ===");
    println!("Average time: {:?}", avg_time);
    println!("Min time: {:?}", min_time);
    println!("Max time: {:?}", max_time);
    println!("Std dev: {:?}", std_time);
    if let Some(ref result) = result_final {
        println!("Final result max bond dimension: {}", result.max_bond_dim());
        println!("Final result bond dimensions: {:?}", result.bond_dims());
    }

    Ok(())
}
