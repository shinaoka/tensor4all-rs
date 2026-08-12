// Inspect Julia-dumped local linsolve inputs stored in ITensorMPS-compatible HDF5.
//
// Run:
//   cargo run -p tensor4all-hdf5 --example inspect_mps_inputs --release -- benchmarks/results/local_linsolve_inputs_N8_b4_o4.h5

use std::env;

use tensor4all_core::IdxTensor;
use tensor4all_hdf5::{load_itensor, load_mps};
use tensor4all_itensorlike::TensorTrain;

fn summarize(name: &str, tt: &TensorTrain) {
    let tensors = tt.tensors();
    let site_indices = tt.site_indices();
    let tensor_dims: Vec<_> = tensors.iter().map(|tensor| tensor.dims()).collect();
    let tensor_index_counts: Vec<_> = tensors.iter().map(|tensor| tensor.indices().len()).collect();
    let site_index_counts: Vec<_> = site_indices.iter().map(Vec::len).collect();

    println!("{name}.length = {}", tt.len());
    println!("{name}.llim = {}", tt.llim());
    println!("{name}.rlim = {}", tt.rlim());
    println!("{name}.bond_dims = {:?}", tt.bond_dims());
    println!("{name}.max_bond_dim = {}", tt.max_bond_dim());
    println!("{name}.tensor_dims = {:?}", tensor_dims);
    println!("{name}.tensor_index_counts = {:?}", tensor_index_counts);
    println!("{name}.site_index_counts = {:?}", site_index_counts);

    if let Some(first) = tensors.first() {
        println!("{name}.first_tensor_indices = {:?}", first.indices());
    }
    if let Some(last) = tensors.last() {
        println!("{name}.last_tensor_indices = {:?}", last.indices());
    }
}

fn summarize_raw_tensor(path: &str, name: &str) -> anyhow::Result<()> {
    let tensor: IdxTensor = load_itensor(path, name)?;
    println!("{name}.raw_dims = {:?}", tensor.dims());
    println!("{name}.raw_indices = {:?}", tensor.indices());
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let path = env::args()
        .nth(1)
        .unwrap_or_else(|| "benchmarks/results/local_linsolve_inputs_N8_b4_o4.h5".to_string());

    println!("=== Inspect HDF5 MPS inputs (Rust/tensor4all-hdf5) ===");
    println!("path = {path}");

    let operator_as_mps = load_mps(&path, "operator_as_mps")?;
    let rhs = load_mps(&path, "rhs")?;
    let init = load_mps(&path, "init")?;

    summarize("operator_as_mps", &operator_as_mps);
    summarize("rhs", &rhs);
    summarize("init", &init);

    println!("--- Raw HDF5 site tensors ---");
    let last_operator_site = format!("operator_as_mps/MPS[{}]", operator_as_mps.len());
    summarize_raw_tensor(&path, "operator_as_mps/MPS[1]")?;
    summarize_raw_tensor(&path, &last_operator_site)?;
    summarize_raw_tensor(&path, "rhs/MPS[1]")?;
    summarize_raw_tensor(&path, "init/MPS[1]")?;

    Ok(())
}
