// Two-site TreeTN DMRG benchmark.
//
// Run:
//   RAYON_NUM_THREADS=1 cargo run -p tensor4all-treetn --example benchmark_dmrg --release
//
// Optional args:
//   cargo run -p tensor4all-treetn --example benchmark_dmrg --release -- <n_sites> <nsweeps> <repeats>

use std::collections::HashMap;
use std::hint::black_box;
use std::time::{Duration, Instant};

use tensor4all_core::krylov::HermitianLanczosOptions;
use tensor4all_core::{DynIndex, IndexLike, SvdTruncationPolicy, TensorDynLen};
use tensor4all_tensorbackend::{lowest_hermitian_eigenpair, Matrix};
use tensor4all_treetn::{
    compose_exclusive_linear_operators, dmrg, DmrgOptions, IndexMapping, LinearOperator, TreeTN,
    TreeTopology, TruncationOptions,
};

const ITENSOR_CUTOFF: f64 = 1.0e-12;

fn itensor_cutoff_policy() -> SvdTruncationPolicy {
    SvdTruncationPolicy::new(ITENSOR_CUTOFF)
        .with_squared_values()
        .with_discarded_tail_sum()
}

#[derive(Debug, Clone, Copy)]
enum Topology {
    Chain,
    Star,
}

impl Topology {
    fn label(self) -> &'static str {
        match self {
            Self::Chain => "chain",
            Self::Star => "star",
        }
    }
}

fn node_name(i: usize) -> String {
    format!("site{i}")
}

fn dmrg_root_name(topology: Topology) -> String {
    match topology {
        Topology::Chain => node_name(0),
        Topology::Star => node_name(1),
    }
}

fn benchmark_dmrg_options(nsweeps: usize) -> DmrgOptions {
    DmrgOptions::default()
        .with_nsweeps(nsweeps)
        .with_max_bond_dim(32)
        .with_svd_policy(itensor_cutoff_policy())
        .with_lanczos_options(HermitianLanczosOptions {
            max_iter: 16,
            rtol: 1.0e-12,
            ..HermitianLanczosOptions::default()
        })
}

fn col_major_offset(coords: &[usize], dims: &[usize]) -> usize {
    let mut stride = 1usize;
    let mut offset = 0usize;
    for (&coord, &dim) in coords.iter().zip(dims.iter()) {
        offset += coord * stride;
        stride *= dim;
    }
    offset
}

fn edges_for(topology: Topology, n_sites: usize) -> Vec<(usize, usize)> {
    match topology {
        Topology::Chain => (0..n_sites.saturating_sub(1)).map(|i| (i, i + 1)).collect(),
        Topology::Star => (1..n_sites).map(|i| (0, i)).collect(),
    }
}

fn make_initial_state(
    topology: Topology,
    n_sites: usize,
) -> anyhow::Result<(TreeTN<TensorDynLen, String>, Vec<DynIndex>)> {
    let edges = edges_for(topology, n_sites);
    let sites: Vec<_> = (0..n_sites).map(|_| DynIndex::new_dyn(2)).collect();
    let bonds: Vec<_> = (0..edges.len()).map(|_| DynIndex::new_dyn(1)).collect();
    let mut incident: Vec<Vec<(usize, DynIndex)>> = vec![Vec::new(); n_sites];
    for (edge_id, &(a, b)) in edges.iter().enumerate() {
        incident[a].push((b, bonds[edge_id].clone()));
        incident[b].push((a, bonds[edge_id].clone()));
    }

    let mut state = TreeTN::<TensorDynLen, String>::new();
    let mut graph_nodes = Vec::with_capacity(n_sites);
    for i in 0..n_sites {
        let mut indices = Vec::with_capacity(1 + incident[i].len());
        for (_, bond) in &incident[i] {
            indices.push(bond.clone());
        }
        indices.push(sites[i].clone());
        let site_one_value = if i % 2 == 0 {
            0.31 + 0.07 * i as f64
        } else {
            -0.43 + 0.05 * i as f64
        };
        let mut data = vec![0.0; indices.iter().map(DynIndex::dim).product()];
        data[0] = 1.0;
        data[1] = site_one_value;
        let tensor = TensorDynLen::from_dense(indices, data)?;
        graph_nodes.push(state.add_tensor(node_name(i), tensor)?);
    }
    for (edge_id, &(a, b)) in edges.iter().enumerate() {
        state.connect(graph_nodes[a], &bonds[edge_id], graph_nodes[b], &bonds[edge_id])?;
    }
    Ok((state, sites))
}

fn local_heisenberg_tensor(
    out_left: DynIndex,
    in_left: DynIndex,
    out_right: DynIndex,
    in_right: DynIndex,
) -> anyhow::Result<TensorDynLen> {
    let dims = [2, 2, 2, 2];
    let mut data = vec![0.0; dims.iter().product()];

    for left in 0..2 {
        for right in 0..2 {
            let z_left = if left == 0 { 1.0 } else { -1.0 };
            let z_right = if right == 0 { 1.0 } else { -1.0 };
            data[col_major_offset(&[left, left, right, right], &dims)] += z_left * z_right;

            let yy_coeff = if left == right { -1.0 } else { 1.0 };
            let flip_coeff = 1.0 + yy_coeff;
            if flip_coeff != 0.0 {
                data[col_major_offset(&[left ^ 1, left, right ^ 1, right], &dims)] += flip_coeff;
            }
        }
    }

    TensorDynLen::from_dense(vec![out_left, in_left, out_right, in_right], data)
        .map_err(anyhow::Error::from)
}

fn make_edge_heisenberg_operator(
    left: usize,
    right: usize,
    state_sites: &[DynIndex],
    op_inputs: &[DynIndex],
    op_outputs: &[DynIndex],
) -> anyhow::Result<LinearOperator<TensorDynLen, String>> {
    let left_name = node_name(left);
    let right_name = node_name(right);
    let local = local_heisenberg_tensor(
        op_outputs[left].clone(),
        op_inputs[left].clone(),
        op_outputs[right].clone(),
        op_inputs[right].clone(),
    )?;
    let topology = TreeTopology::new(
        HashMap::from([
            (
                left_name.clone(),
                vec![op_outputs[left].clone(), op_inputs[left].clone()],
            ),
            (
                right_name.clone(),
                vec![op_outputs[right].clone(), op_inputs[right].clone()],
            ),
        ]),
        vec![(left_name.clone(), right_name.clone())],
    );
    let mpo =
        tensor4all_treetn::factorize_tensor_to_treetn_with(&local, &topology, Default::default(), &left_name)?;
    let input_mapping = HashMap::from([
        (
            left_name.clone(),
            IndexMapping {
                true_index: state_sites[left].clone(),
                internal_index: op_inputs[left].clone(),
            },
        ),
        (
            right_name.clone(),
            IndexMapping {
                true_index: state_sites[right].clone(),
                internal_index: op_inputs[right].clone(),
            },
        ),
    ]);
    let output_mapping = HashMap::from([
        (
            left_name,
            IndexMapping {
                true_index: state_sites[left].clone(),
                internal_index: op_outputs[left].clone(),
            },
        ),
        (
            right_name,
            IndexMapping {
                true_index: state_sites[right].clone(),
                internal_index: op_outputs[right].clone(),
            },
        ),
    ]);
    Ok(LinearOperator::new(mpo, input_mapping, output_mapping))
}

fn make_heisenberg_operator(
    topology: Topology,
    state: &TreeTN<TensorDynLen, String>,
    state_sites: &[DynIndex],
) -> anyhow::Result<LinearOperator<TensorDynLen, String>> {
    let n_sites = state_sites.len();
    let edges = edges_for(topology, n_sites);
    let op_inputs: Vec<_> = (0..n_sites).map(|_| DynIndex::new_dyn(2)).collect();
    let op_outputs: Vec<_> = (0..n_sites).map(|_| DynIndex::new_dyn(2)).collect();

    let gap_site_indices: HashMap<_, _> = (0..n_sites)
        .map(|i| (node_name(i), vec![(op_inputs[i].clone(), op_outputs[i].clone())]))
        .collect();
    let mut term_mpos = Vec::with_capacity(edges.len());
    for &(left, right) in &edges {
        let edge_op =
            make_edge_heisenberg_operator(left, right, state_sites, &op_inputs, &op_outputs)?;
        let composed = compose_exclusive_linear_operators(
            state.site_index_network(),
            &[&edge_op],
            &gap_site_indices,
        )?;
        term_mpos.push(composed.into_mpo());
    }

    let mpo = term_mpos
        .into_iter()
        .reduce(|acc, term| acc.add(&term).expect("matching Heisenberg MPO term topology"))
        .ok_or_else(|| anyhow::anyhow!("Heisenberg operator requires at least one edge"))?;
    let mpo = mpo.truncate(
        [node_name(0)],
        TruncationOptions::default().with_svd_policy(itensor_cutoff_policy()),
    )?;

    let input_mapping: HashMap<_, _> = (0..n_sites)
        .map(|i| {
            (
                node_name(i),
                IndexMapping {
                    true_index: state_sites[i].clone(),
                    internal_index: op_inputs[i].clone(),
                },
            )
        })
        .collect();
    let output_mapping: HashMap<_, _> = (0..n_sites)
        .map(|i| {
            (
                node_name(i),
                IndexMapping {
                    true_index: state_sites[i].clone(),
                    internal_index: op_outputs[i].clone(),
                },
            )
        })
        .collect();
    mpo.verify_internal_consistency()?;

    Ok(LinearOperator::new(mpo, input_mapping, output_mapping))
}

fn dense_heisenberg_exact(topology: Topology, n_sites: usize) -> anyhow::Result<f64> {
    anyhow::ensure!(
        n_sites <= 10,
        "dense exact benchmark reference is capped at n_sites <= 10"
    );
    let edges = edges_for(topology, n_sites);
    let dim = 1usize << n_sites;
    let mut data = vec![0.0; dim * dim];
    for input_state in 0..dim {
        let bits: Vec<_> = (0..n_sites)
            .map(|site| (input_state >> site) & 1)
            .collect();
        for &(left, right) in &edges {
            let z_left = if bits[left] == 0 { 1.0 } else { -1.0 };
            let z_right = if bits[right] == 0 { 1.0 } else { -1.0 };
            data[input_state + dim * input_state] += z_left * z_right;

            let flipped = input_state ^ (1usize << left) ^ (1usize << right);
            let yy_coeff = if bits[left] == bits[right] {
                -1.0
            } else {
                1.0
            };
            data[flipped + dim * input_state] += 1.0 + yy_coeff;
        }
    }
    let matrix = Matrix::from_col_major_vec(dim, dim, data);
    Ok(lowest_hermitian_eigenpair(&matrix, 1.0e-10)?.eigenvalue)
}

fn summarize(times: &[Duration]) -> (f64, f64, f64) {
    let seconds: Vec<_> = times.iter().map(Duration::as_secs_f64).collect();
    let mean = seconds.iter().sum::<f64>() / seconds.len() as f64;
    let min = seconds.iter().copied().fold(f64::INFINITY, f64::min);
    let max = seconds.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    (mean, min, max)
}

fn run_case(topology: Topology, n_sites: usize, nsweeps: usize, repeats: usize) -> anyhow::Result<()> {
    let (state, sites) = make_initial_state(topology, n_sites)?;
    let operator = make_heisenberg_operator(topology, &state, &sites)?;
    let exact_energy = dense_heisenberg_exact(topology, n_sites)?;
    let root = dmrg_root_name(topology);
    let options = benchmark_dmrg_options(nsweeps);

    let warmup = dmrg(&operator, black_box(state.clone()), &root, options.clone())?;
    black_box(&warmup.state);

    let mut times = Vec::with_capacity(repeats);
    let mut energy = f64::NAN;
    let mut residual = f64::NAN;
    let mut sweeps_completed = 0usize;
    let mut local_updates = 0usize;
    let mut converged = false;
    for _ in 0..repeats {
        let init = black_box(state.clone());
        let start = Instant::now();
        let result = dmrg(&operator, init, &root, options.clone())?;
        times.push(start.elapsed());
        energy = result.energy;
        residual = result.max_residual_norm;
        sweeps_completed = result.sweeps_completed;
        local_updates = result.local_updates;
        converged = result.converged;
        black_box(&result.state);
    }

    let (mean, min, max) = summarize(&times);
    println!(
        "case={} n={} sweeps={} sweeps_completed={} local_updates={} converged={} warmups=1 repeats={} energy={:.12} exact={:.12} abs_error={:.3e} max_residual={:.3e} mean_ms={:.3} min_ms={:.3} max_ms={:.3}",
        topology.label(),
        n_sites,
        nsweeps,
        sweeps_completed,
        local_updates,
        converged,
        repeats,
        energy,
        exact_energy,
        (energy - exact_energy).abs(),
        residual,
        mean * 1000.0,
        min * 1000.0,
        max * 1000.0,
    );
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let n_sites = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(8);
    let nsweeps = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(4);
    let repeats = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(3);
    anyhow::ensure!(n_sites >= 2, "n_sites must be at least 2");
    anyhow::ensure!(repeats >= 1, "repeats must be at least 1");

    run_case(Topology::Chain, n_sites, nsweeps, repeats)?;
    run_case(Topology::Star, n_sites, nsweeps, repeats)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tensor4all_core::{SingularValueMeasure, ThresholdScale, TruncationRule};

    #[test]
    fn dmrg_benchmark_uses_itensors_cutoff_strategy() {
        let policy = itensor_cutoff_policy();
        assert_eq!(policy.threshold, ITENSOR_CUTOFF);
        assert_eq!(policy.scale, ThresholdScale::Relative);
        assert_eq!(policy.measure, SingularValueMeasure::SquaredValue);
        assert_eq!(policy.rule, TruncationRule::DiscardedTailSum);
    }

    #[test]
    fn star_benchmark_uses_itensornetworks_leaf_root() {
        assert_eq!(dmrg_root_name(Topology::Chain), node_name(0));
        assert_eq!(dmrg_root_name(Topology::Star), node_name(1));
    }

    #[test]
    fn dmrg_benchmark_runs_all_requested_sweeps() {
        let options = benchmark_dmrg_options(4);
        assert_eq!(options.nsweeps, 4);
        assert_eq!(options.energy_tol, None);
    }
}
