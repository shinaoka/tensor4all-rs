use rand::rng;
use std::time::Instant;

use anyhow::Result;
use tensor4all_core::{DynIndex, TensorDynLen};
use tensor4all_itensorlike::{CanonicalForm, ContractOptions, TensorTrain};

fn create_random_mpo(
    length: usize,
    phys_dim: usize,
    bond_dim: usize,
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
        if i + 1 < length {
            indices.push(link_indices[i].clone());
        }
        let tensor = TensorDynLen::random::<f64, _>(&mut rng, indices)?;
        tensors.push(tensor);
    }

    let _ = phys_dim;
    let _ = bond_dim;
    Ok(TensorTrain::new(tensors)?)
}

fn run_zipup(length: usize, phys_dim: usize, bond_dim: usize, max_bond_dim: usize) -> Result<()> {
    let input_indices_a: Vec<_> = (0..length).map(|_| DynIndex::new_dyn(phys_dim)).collect();
    let output_indices_shared: Vec<_> = (0..length).map(|_| DynIndex::new_dyn(phys_dim)).collect();
    let output_indices_b: Vec<_> = (0..length).map(|_| DynIndex::new_dyn(phys_dim)).collect();
    let link_indices_a: Vec<_> = (0..length.saturating_sub(1))
        .map(|_| DynIndex::new_dyn(bond_dim))
        .collect();
    let link_indices_b: Vec<_> = (0..length.saturating_sub(1))
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
    let mpo_b = create_random_mpo(
        length,
        phys_dim,
        bond_dim,
        &output_indices_shared,
        &output_indices_b,
        &link_indices_b,
    )?;

    let options = ContractOptions::zipup().with_max_bond_dim(max_bond_dim);

    let start = Instant::now();
    let mut mpo_a = mpo_a;
    let mut mpo_b = mpo_b;
    mpo_a.orthogonalize_with(length - 1, CanonicalForm::Unitary)?;
    mpo_b.orthogonalize_with(length - 1, CanonicalForm::Unitary)?;
    let result = mpo_a.contract(&mpo_b, &options)?;
    let elapsed = start.elapsed();

    println!(
        "zipup one-shot: {:?} (max bond dim: {})",
        elapsed,
        result.max_bond_dim()
    );
    Ok(())
}

fn run_fit(
    length: usize,
    phys_dim: usize,
    bond_dim: usize,
    max_bond_dim: usize,
    n_half_sweeps: usize,
) -> Result<()> {
    let input_indices_a: Vec<_> = (0..length).map(|_| DynIndex::new_dyn(phys_dim)).collect();
    let output_indices_shared: Vec<_> = (0..length).map(|_| DynIndex::new_dyn(phys_dim)).collect();
    let output_indices_b: Vec<_> = (0..length).map(|_| DynIndex::new_dyn(phys_dim)).collect();
    let link_indices_a: Vec<_> = (0..length.saturating_sub(1))
        .map(|_| DynIndex::new_dyn(bond_dim))
        .collect();
    let link_indices_b: Vec<_> = (0..length.saturating_sub(1))
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
    let mpo_b = create_random_mpo(
        length,
        phys_dim,
        bond_dim,
        &output_indices_shared,
        &output_indices_b,
        &link_indices_b,
    )?;

    let options = ContractOptions::fit()
        .with_max_bond_dim(max_bond_dim)
        .with_nhalfsweeps(n_half_sweeps);

    let start = Instant::now();
    let mut mpo_a = mpo_a;
    let mut mpo_b = mpo_b;
    mpo_a.orthogonalize_with(0, CanonicalForm::Unitary)?;
    mpo_b.orthogonalize_with(0, CanonicalForm::Unitary)?;
    let result = mpo_a.contract(&mpo_b, &options)?;
    let elapsed = start.elapsed();

    println!(
        "fit one-shot: {:?} (max bond dim: {})",
        elapsed,
        result.max_bond_dim()
    );
    Ok(())
}

fn main() -> Result<()> {
    let length = 10;
    let phys_dim = 2;
    let bond_dim = 80;
    let max_bond_dim = 80;
    let n_half_sweeps = 10;

    println!(
        "one-shot benchmark: length={}, phys_dim={}, bond_dim={}, max_bond_dim={}, n_half_sweeps={}",
        length, phys_dim, bond_dim, max_bond_dim, n_half_sweeps
    );

    run_zipup(length, phys_dim, bond_dim, max_bond_dim)?;
    run_fit(length, phys_dim, bond_dim, max_bond_dim, n_half_sweeps)?;
    Ok(())
}
