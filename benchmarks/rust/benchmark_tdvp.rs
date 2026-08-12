// Two-site TreeTN TDVP benchmark.
//
// Run:
//   RAYON_NUM_THREADS=1 BLAS_NUM_THREADS=1 cargo run -p tensor4all-treetn --example benchmark_tdvp --release
//
// Optional args:
//   cargo run -p tensor4all-treetn --example benchmark_tdvp --release -- <n_sites> <time_steps> <repeats> <dt> [all|chain|star]

use std::collections::HashMap;
use std::hint::black_box;
use std::time::{Duration, Instant};

use num_complex::Complex64;
use tensor4all_core::krylov::HermitianKrylovExpmOptions;
use tensor4all_core::{
    DynIndex, FactorizeOptions, IndexLike, SvdTruncationPolicy, TensorContractionLike,
    IdxTensor,
};
use tensor4all_tensorbackend::{hermitian_eigendecomposition, Matrix};
use tensor4all_treetn::{
    compose_exclusive_linear_operators, tdvp, IndexMapping, LinearOperator, TdvpOptions, TreeTN,
    TreeTopology, TruncationOptions,
};

const ITENSOR_CUTOFF: f64 = 1.0e-12;

type BenchmarkState = (TreeTN<IdxTensor, String>, Vec<DynIndex>, Vec<Complex64>);

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

#[derive(Debug, Clone, Copy)]
enum TopologySelection {
    All,
    One(Topology),
}

fn parse_topology_selection(value: Option<&str>) -> anyhow::Result<TopologySelection> {
    match value.unwrap_or("all") {
        "all" => Ok(TopologySelection::All),
        "chain" => Ok(TopologySelection::One(Topology::Chain)),
        "star" => Ok(TopologySelection::One(Topology::Star)),
        other => anyhow::bail!("unknown topology selection {other:?}; use all, chain, or star"),
    }
}

fn node_name(i: usize) -> String {
    format!("site{i}")
}

fn tdvp_root_name(topology: Topology) -> String {
    match topology {
        Topology::Chain => node_name(0),
        Topology::Star => node_name(1),
    }
}

fn benchmark_tdvp_options(time_steps: usize, dt: f64) -> TdvpOptions {
    TdvpOptions::default()
        .with_nsite(2)
        .with_order(2)
        .with_nsweeps(time_steps)
        .with_exponent_step(Complex64::new(0.0, -dt))
        .with_max_bond_dim(32)
        .with_svd_policy(itensor_cutoff_policy())
        .with_krylov_options(HermitianKrylovExpmOptions {
            max_iter: 30,
            max_time_splits: 100,
            tol: 1.0e-12,
            ..HermitianKrylovExpmOptions::default()
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

fn initial_bit(site: usize) -> usize {
    site % 2
}

fn make_initial_state(
    topology: Topology,
    n_sites: usize,
) -> anyhow::Result<BenchmarkState> {
    let edges = edges_for(topology, n_sites);
    let sites: Vec<_> = (0..n_sites).map(|_| DynIndex::new_dyn(2)).collect();
    let bonds: Vec<_> = (0..edges.len()).map(|_| DynIndex::new_dyn(1)).collect();
    let mut incident: Vec<Vec<(usize, DynIndex)>> = vec![Vec::new(); n_sites];
    for (edge_id, &(a, b)) in edges.iter().enumerate() {
        incident[a].push((b, bonds[edge_id].clone()));
        incident[b].push((a, bonds[edge_id].clone()));
    }

    let mut state = TreeTN::<IdxTensor, String>::new();
    let mut graph_nodes = Vec::with_capacity(n_sites);
    for i in 0..n_sites {
        let mut indices = Vec::with_capacity(1 + incident[i].len());
        for (_, bond) in &incident[i] {
            indices.push(bond.clone());
        }
        indices.push(sites[i].clone());
        let mut data = vec![Complex64::new(0.0, 0.0); indices.iter().map(DynIndex::dim).product()];
        data[initial_bit(i)] = Complex64::new(1.0, 0.0);
        let tensor = IdxTensor::from_dense(indices, data)?;
        graph_nodes.push(state.add_tensor(node_name(i), tensor)?);
    }
    for (edge_id, &(a, b)) in edges.iter().enumerate() {
        state.connect(graph_nodes[a], &bonds[edge_id], graph_nodes[b], &bonds[edge_id])?;
    }

    let dim = 1usize << n_sites;
    let mut vector = vec![Complex64::new(0.0, 0.0); dim];
    let mut basis = 0usize;
    for i in 0..n_sites {
        basis |= initial_bit(i) << i;
    }
    vector[basis] = Complex64::new(1.0, 0.0);
    Ok((state, sites, vector))
}

fn local_heisenberg_tensor(
    out_left: DynIndex,
    in_left: DynIndex,
    out_right: DynIndex,
    in_right: DynIndex,
) -> anyhow::Result<IdxTensor> {
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

    IdxTensor::from_dense(vec![out_left, in_left, out_right, in_right], data)
        .map_err(anyhow::Error::from)
}

fn make_edge_heisenberg_operator(
    left: usize,
    right: usize,
    state_sites: &[DynIndex],
    op_inputs: &[DynIndex],
    op_outputs: &[DynIndex],
) -> anyhow::Result<LinearOperator<IdxTensor, String>> {
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
    let mpo = tensor4all_treetn::factorize_tensor_to_treetn_with(
        &local,
        &topology,
        FactorizeOptions::svd(),
        &left_name,
    )?;
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
    state: &TreeTN<IdxTensor, String>,
    state_sites: &[DynIndex],
) -> anyhow::Result<LinearOperator<IdxTensor, String>> {
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
        .reduce(|acc, term| {
            acc.add(&term)
                .expect("matching Heisenberg MPO term topology")
        })
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

fn dense_heisenberg_matrix(topology: Topology, n_sites: usize) -> anyhow::Result<Matrix<Complex64>> {
    anyhow::ensure!(
        n_sites <= 10,
        "dense exact benchmark reference is capped at n_sites <= 10"
    );
    let edges = edges_for(topology, n_sites);
    let dim = 1usize << n_sites;
    let mut data = vec![Complex64::new(0.0, 0.0); dim * dim];
    for input_state in 0..dim {
        let bits: Vec<_> = (0..n_sites)
            .map(|site| (input_state >> site) & 1)
            .collect();
        for &(left, right) in &edges {
            let z_left = if bits[left] == 0 { 1.0 } else { -1.0 };
            let z_right = if bits[right] == 0 { 1.0 } else { -1.0 };
            data[input_state + dim * input_state] += Complex64::new(z_left * z_right, 0.0);

            let flipped = input_state ^ (1usize << left) ^ (1usize << right);
            let yy_coeff = if bits[left] == bits[right] {
                -1.0
            } else {
                1.0
            };
            data[flipped + dim * input_state] += Complex64::new(1.0 + yy_coeff, 0.0);
        }
    }
    Ok(Matrix::from_col_major_vec(dim, dim, data))
}

fn exact_evolve(
    hamiltonian: &Matrix<Complex64>,
    initial: &[Complex64],
    total_time: f64,
) -> anyhow::Result<Vec<Complex64>> {
    let n = hamiltonian.nrows();
    anyhow::ensure!(hamiltonian.ncols() == n, "Hamiltonian must be square");
    anyhow::ensure!(initial.len() == n, "initial vector length mismatch");

    let decomp = hermitian_eigendecomposition(hamiltonian, 1.0e-10)?;
    let vectors = decomp.eigenvectors.as_col_major_slice();
    let mut coefficients = vec![Complex64::new(0.0, 0.0); n];
    for col in 0..n {
        let mut coeff = Complex64::new(0.0, 0.0);
        for row in 0..n {
            coeff += vectors[row + col * n].conj() * initial[row];
        }
        coefficients[col] = coeff;
    }

    let mut result = vec![Complex64::new(0.0, 0.0); n];
    for col in 0..n {
        let phase = (Complex64::new(0.0, -total_time) * decomp.eigenvalues[col]).exp();
        let coeff = phase * coefficients[col];
        for row in 0..n {
            result[row] += vectors[row + col * n] * coeff;
        }
    }
    Ok(result)
}

fn state_vector(
    state: &TreeTN<IdxTensor, String>,
    sites: &[DynIndex],
) -> anyhow::Result<Vec<Complex64>> {
    let tensor = state.contract_to_tensor()?;
    let aligned = tensor.permuteinds(sites)?;
    aligned.to_vec::<Complex64>().map_err(anyhow::Error::from)
}

fn vector_norm(vector: &[Complex64]) -> f64 {
    vector.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt()
}

fn l2_error(actual: &[Complex64], expected: &[Complex64]) -> f64 {
    actual
        .iter()
        .zip(expected)
        .map(|(a, e)| (*a - *e).norm_sqr())
        .sum::<f64>()
        .sqrt()
}

fn summarize(times: &[Duration]) -> (f64, f64, f64) {
    let seconds: Vec<_> = times.iter().map(Duration::as_secs_f64).collect();
    let mean = seconds.iter().sum::<f64>() / seconds.len() as f64;
    let min = seconds.iter().copied().fold(f64::INFINITY, f64::min);
    let max = seconds.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    (mean, min, max)
}

fn run_case(
    topology: Topology,
    n_sites: usize,
    time_steps: usize,
    repeats: usize,
    dt: f64,
) -> anyhow::Result<()> {
    let (state, sites, initial_vector) = make_initial_state(topology, n_sites)?;
    let operator = make_heisenberg_operator(topology, &state, &sites)?;
    let total_time = dt * time_steps as f64;
    let exact = exact_evolve(
        &dense_heisenberg_matrix(topology, n_sites)?,
        &initial_vector,
        total_time,
    )?;
    let exact_norm = vector_norm(&exact);
    let root = tdvp_root_name(topology);
    let options = benchmark_tdvp_options(time_steps, dt);

    let warmup = tdvp(&operator, black_box(state.clone()), &root, options.clone())?;
    black_box(&warmup.state);

    let mut times = Vec::with_capacity(repeats);
    let mut norm = f64::NAN;
    let mut error = f64::NAN;
    let mut sweeps_completed = 0usize;
    let mut local_updates = 0usize;
    let mut max_krylov_error = f64::NAN;
    let mut max_krylov_iterations = 0usize;
    for _ in 0..repeats {
        let init = black_box(state.clone());
        let start = Instant::now();
        let result = tdvp(&operator, init, &root, options.clone())?;
        times.push(start.elapsed());
        let actual = state_vector(&result.state, &sites)?;
        norm = vector_norm(&actual);
        error = l2_error(&actual, &exact);
        sweeps_completed = result.sweeps_completed;
        local_updates = result.local_updates;
        max_krylov_error = result.max_error_estimate;
        max_krylov_iterations = result.max_krylov_iterations;
        black_box(&result.state);
    }

    let (mean, min, max) = summarize(&times);
    println!(
        "case={} n={} time_steps={} dt={:.12} time={:.12} sweeps_completed={} local_updates={} warmups=1 repeats={} norm={:.12} exact_norm={:.12} l2_error={:.3e} rel_l2_error={:.3e} max_krylov_error={:.3e} max_krylov_iterations={} mean_ms={:.3} min_ms={:.3} max_ms={:.3}",
        topology.label(),
        n_sites,
        time_steps,
        dt,
        total_time,
        sweeps_completed,
        local_updates,
        repeats,
        norm,
        exact_norm,
        error,
        error / exact_norm.max(f64::MIN_POSITIVE),
        max_krylov_error,
        max_krylov_iterations,
        mean * 1000.0,
        min * 1000.0,
        max * 1000.0,
    );
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let n_sites = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(8);
    let time_steps = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(4);
    let repeats = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(3);
    let dt: f64 = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(0.02);
    let topology_selection = parse_topology_selection(args.get(5).map(String::as_str))?;
    anyhow::ensure!(n_sites >= 2, "n_sites must be at least 2");
    anyhow::ensure!(time_steps >= 1, "time_steps must be at least 1");
    anyhow::ensure!(repeats >= 1, "repeats must be at least 1");
    anyhow::ensure!(dt.is_finite() && dt > 0.0, "dt must be finite and positive");

    match topology_selection {
        TopologySelection::All => {
            run_case(Topology::Chain, n_sites, time_steps, repeats, dt)?;
            run_case(Topology::Star, n_sites, time_steps, repeats, dt)?;
        }
        TopologySelection::One(topology) => {
            run_case(topology, n_sites, time_steps, repeats, dt)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tensor4all_core::{SingularValueMeasure, ThresholdScale, TruncationRule};

    #[test]
    fn tdvp_benchmark_uses_itensors_cutoff_strategy() {
        let policy = itensor_cutoff_policy();
        assert_eq!(policy.threshold, ITENSOR_CUTOFF);
        assert_eq!(policy.scale, ThresholdScale::Relative);
        assert_eq!(policy.measure, SingularValueMeasure::SquaredValue);
        assert_eq!(policy.rule, TruncationRule::DiscardedTailSum);
    }

    #[test]
    fn star_benchmark_uses_itensornetworks_leaf_root() {
        assert_eq!(tdvp_root_name(Topology::Chain), node_name(0));
        assert_eq!(tdvp_root_name(Topology::Star), node_name(1));
    }

    #[test]
    fn topology_selection_parses_all_chain_and_star() {
        assert!(matches!(
            parse_topology_selection(None).unwrap(),
            TopologySelection::All
        ));
        assert!(matches!(
            parse_topology_selection(Some("chain")).unwrap(),
            TopologySelection::One(Topology::Chain)
        ));
        assert!(matches!(
            parse_topology_selection(Some("star")).unwrap(),
            TopologySelection::One(Topology::Star)
        ));
        assert!(parse_topology_selection(Some("line")).is_err());
    }

    #[test]
    fn tdvp_benchmark_options_set_requested_solver_caps() {
        let options = benchmark_tdvp_options(4, 0.02);
        assert_eq!(options.nsite, 2);
        assert_eq!(options.order, 2);
        assert_eq!(options.nsweeps, 4);
        assert_eq!(options.max_bond_dim, Some(32));
        assert_eq!(options.krylov.max_iter, 30);
        assert_eq!(options.krylov.max_time_splits, 100);
        assert_eq!(options.krylov.tol, 1.0e-12);
    }
}
