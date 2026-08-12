// Benchmark isolated `ProjectedOperator::apply` calls for mapped MPO local solves.
//
// Run:
//   cargo run -p tensor4all-treetn --example benchmark_projected_apply --release
//
// Optional args:
//   cargo run -p tensor4all-treetn --example benchmark_projected_apply --release -- <N> <state_bond_dim> <operator_bond_dim> <repeats> <step_index>

use std::collections::{HashMap, HashSet};
use std::hint::black_box;
use std::time::{Duration, Instant};

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use tensor4all_core::{
    index::{DynId, Index},
    print_and_reset_contract_profile, reset_contract_profile,
    DynIndex, IdxTensor, TensorIndex,
};
use tensor4all_treetn::{IndexMapping, LocalUpdateSweepPlan, ProjectedOperator, TreeTN};

fn make_node_name(i: usize) -> String {
    format!("site{i}")
}

fn unique_dyn_index(used: &mut HashSet<DynId>, dim: usize, rng: &mut StdRng) -> DynIndex {
    loop {
        let id = DynId(rng.random());
        if used.insert(id) {
            return Index::new(id, dim);
        }
    }
}

fn chain_node_indices(n: usize, i: usize, bonds: &[DynIndex], sites: &[DynIndex]) -> Vec<DynIndex> {
    if n == 1 {
        vec![sites[i].clone()]
    } else if i == 0 {
        vec![sites[i].clone(), bonds[i].clone()]
    } else if i + 1 == n {
        vec![bonds[i - 1].clone(), sites[i].clone()]
    } else {
        vec![bonds[i - 1].clone(), sites[i].clone(), bonds[i].clone()]
    }
}

fn create_state_chain(
    n: usize,
    state_bond_dim: usize,
    acted_sites: &[DynIndex],
    spectator_sites: &[DynIndex],
    used_ids: &mut HashSet<DynId>,
    rng: &mut StdRng,
) -> anyhow::Result<TreeTN<IdxTensor, String>> {
    let mut tree = TreeTN::<IdxTensor, String>::new();
    let bonds: Vec<_> = (0..n.saturating_sub(1))
        .map(|_| unique_dyn_index(used_ids, state_bond_dim, rng))
        .collect();

    let mut nodes = Vec::with_capacity(n);
    for (i, spectator_site) in spectator_sites.iter().enumerate().take(n) {
        let mut indices = chain_node_indices(n, i, &bonds, acted_sites);
        indices.insert(0, spectator_site.clone());
        let tensor = IdxTensor::random::<f64, _>(rng, indices)?;
        let node = tree.add_tensor(make_node_name(i), tensor)?;
        nodes.push(node);
    }

    for i in 0..n.saturating_sub(1) {
        tree.connect(nodes[i], &bonds[i], nodes[i + 1], &bonds[i])?;
    }

    Ok(tree)
}

#[allow(clippy::type_complexity)]
fn create_operator_chain(
    n: usize,
    phys_dim: usize,
    operator_bond_dim: usize,
    acted_sites: &[DynIndex],
    used_ids: &mut HashSet<DynId>,
    rng: &mut StdRng,
) -> anyhow::Result<(
    TreeTN<IdxTensor, String>,
    HashMap<String, IndexMapping<DynIndex>>,
    HashMap<String, IndexMapping<DynIndex>>,
)> {
    let mut tree = TreeTN::<IdxTensor, String>::new();
    let bonds: Vec<_> = (0..n.saturating_sub(1))
        .map(|_| unique_dyn_index(used_ids, operator_bond_dim, rng))
        .collect();
    let s_in: Vec<_> = (0..n)
        .map(|_| unique_dyn_index(used_ids, phys_dim, rng))
        .collect();
    let s_out: Vec<_> = (0..n)
        .map(|_| unique_dyn_index(used_ids, phys_dim, rng))
        .collect();

    let mut input_mapping = HashMap::new();
    let mut output_mapping = HashMap::new();
    let mut nodes = Vec::with_capacity(n);
    for i in 0..n {
        let sites = vec![s_out[i].clone(), s_in[i].clone()];
        let mut indices = if n == 1 {
            sites
        } else if i == 0 {
            vec![s_out[i].clone(), s_in[i].clone(), bonds[i].clone()]
        } else if i + 1 == n {
            vec![bonds[i - 1].clone(), s_out[i].clone(), s_in[i].clone()]
        } else {
            vec![
                bonds[i - 1].clone(),
                s_out[i].clone(),
                s_in[i].clone(),
                bonds[i].clone(),
            ]
        };
        indices.shrink_to_fit();
        let tensor = IdxTensor::random::<f64, _>(rng, indices)?;
        let name = make_node_name(i);
        let node = tree.add_tensor(name.clone(), tensor)?;
        nodes.push(node);

        input_mapping.insert(
            name.clone(),
            IndexMapping {
                true_index: acted_sites[i].clone(),
                internal_index: s_in[i].clone(),
            },
        );
        output_mapping.insert(
            name,
            IndexMapping {
                true_index: acted_sites[i].clone(),
                internal_index: s_out[i].clone(),
            },
        );
    }

    for i in 0..n.saturating_sub(1) {
        tree.connect(nodes[i], &bonds[i], nodes[i + 1], &bonds[i])?;
    }

    Ok((tree, input_mapping, output_mapping))
}

fn summarize(label: &str, times: &[Duration]) {
    let secs: Vec<f64> = times.iter().map(Duration::as_secs_f64).collect();
    let mean = secs.iter().sum::<f64>() / secs.len() as f64;
    let min = secs.iter().copied().fold(f64::INFINITY, f64::min);
    let max = secs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    println!(
        "{label}: mean={:.3} ms min={:.3} ms max={:.3} ms n={}",
        mean * 1000.0,
        min * 1000.0,
        max * 1000.0,
        times.len()
    );
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let n_sites: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(38);
    let state_bond_dim: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(8);
    let operator_bond_dim: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(8);
    let repeats: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(20);
    let step_index: usize = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(0);

    anyhow::ensure!(
        n_sites >= 2,
        "N must be at least 2 for a two-site local step"
    );
    anyhow::ensure!(repeats > 0, "repeats must be greater than zero");

    let phys_dim = 2usize;
    let seed = 20260518_u64;
    let mut used_ids = HashSet::<DynId>::new();
    let mut rng = StdRng::seed_from_u64(seed);

    let acted_sites: Vec<_> = (0..n_sites)
        .map(|_| unique_dyn_index(&mut used_ids, phys_dim, &mut rng))
        .collect();
    let spectator_sites: Vec<_> = (0..n_sites)
        .map(|_| unique_dyn_index(&mut used_ids, phys_dim, &mut rng))
        .collect();

    let state = create_state_chain(
        n_sites,
        state_bond_dim,
        &acted_sites,
        &spectator_sites,
        &mut used_ids,
        &mut rng,
    )?;
    let reference_state = state.sim_link_indices()?;
    let (operator, input_mapping, output_mapping) = create_operator_chain(
        n_sites,
        phys_dim,
        operator_bond_dim,
        &acted_sites,
        &mut used_ids,
        &mut rng,
    )?;

    let center = make_node_name(n_sites / 2);
    let plan = LocalUpdateSweepPlan::from_treetn(&state, &center, 2)
        .ok_or_else(|| anyhow::anyhow!("failed to build two-site sweep plan"))?;
    let step = plan
        .steps
        .get(step_index % plan.steps.len())
        .ok_or_else(|| anyhow::anyhow!("empty sweep plan"))?;
    let local_tensor = state.extract_subtree(&step.nodes)?.contract_to_tensor()?;

    println!("=== ProjectedOperator::apply benchmark ===");
    println!("N = {n_sites}");
    println!("phys_dim = {phys_dim}");
    println!("state_bond_dim = {state_bond_dim}");
    println!("operator_bond_dim = {operator_bond_dim}");
    println!("repeats = {repeats}");
    println!("center = {center}");
    println!("step_index = {}", step_index % plan.steps.len());
    println!("step_nodes = {:?}", step.nodes);
    println!("local_dims = {:?}", local_tensor.dims());
    println!();

    reset_contract_profile();
    let mut projected_cold = ProjectedOperator::with_index_mappings(
        operator.clone(),
        input_mapping.clone(),
        output_mapping.clone(),
    );
    let cold_start = Instant::now();
    let cold_result = projected_cold.apply(
        black_box(&local_tensor),
        &step.nodes,
        &state,
        &reference_state,
        state.site_index_network(),
    )?;
    let cold = cold_start.elapsed();
    println!(
        "cold apply (environment build + one apply): {:.3} ms, output_rank={}",
        cold.as_secs_f64() * 1000.0,
        cold_result.external_indices().len()
    );

    let mut warm_times = Vec::with_capacity(repeats);
    for _ in 0..repeats {
        let start = Instant::now();
        let out = projected_cold.apply(
            black_box(&local_tensor),
            &step.nodes,
            &state,
            &reference_state,
            state.site_index_network(),
        )?;
        black_box(out);
        warm_times.push(start.elapsed());
    }
    summarize("warm apply (environment cache hot)", &warm_times);

    let mut cold_times = Vec::with_capacity(repeats);
    for _ in 0..repeats {
        let mut projected = ProjectedOperator::with_index_mappings(
            operator.clone(),
            input_mapping.clone(),
            output_mapping.clone(),
        );
        let start = Instant::now();
        let out = projected.apply(
            black_box(&local_tensor),
            &step.nodes,
            &state,
            &reference_state,
            state.site_index_network(),
        )?;
        black_box(out);
        cold_times.push(start.elapsed());
    }
    summarize("cold apply repeated (fresh environment cache)", &cold_times);
    print_and_reset_contract_profile();

    Ok(())
}
