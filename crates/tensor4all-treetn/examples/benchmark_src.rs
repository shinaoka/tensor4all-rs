//! Small, deterministic SRC-vs-zip-up contraction benchmark.
//!
//! Run with `RAYON_NUM_THREADS=1` for reproducible CPU measurements. Set
//! `T4A_PROFILE_CONTRACT=1` to print the aggregated dense contraction
//! signatures after each case. For MPO--MPS scaling studies, the positional
//! bond-dimension argument controls the MPO bond dimension and
//! `T4A_BENCH_MPS_BOND_DIM` optionally sets a different MPS bond dimension.

use std::mem::size_of;
use std::time::Instant;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use tensor4all_core::{DynIndex, IdxTensor, IndexLike};
use tensor4all_treetn::contraction::{contract, ContractionOptions, SrcOptions};
use tensor4all_treetn::TreeTN;

type Network = TreeTN<IdxTensor, String>;

const DEFAULT_MAX_INPUT_BYTES: usize = 512 << 20;
const DEFAULT_MAX_DENSE_BYTES: usize = 256 << 20;
const BUILD_GIT_COMMIT: &str = match option_env!("T4A_BENCH_GIT_COMMIT") {
    Some(commit) => commit,
    None => "unknown",
};

#[derive(Debug, Clone, Copy)]
struct MemoryEstimate {
    total_input_bytes: usize,
    largest_input_tensor_bytes: usize,
    dense_output_bytes: usize,
    max_degree: usize,
}

fn checked_mul(lhs: usize, rhs: usize, label: &str) -> usize {
    lhs.checked_mul(rhs)
        .unwrap_or_else(|| panic!("{label} overflow"))
}

fn checked_pow(mut base: usize, mut exponent: usize, label: &str) -> usize {
    let mut result = 1usize;
    while exponent > 0 {
        if exponent % 2 == 1 {
            result = checked_mul(result, base, label);
        }
        exponent /= 2;
        if exponent > 0 {
            base = checked_mul(base, base, label);
        }
    }
    result
}

fn bytes(elements: usize, label: &str) -> usize {
    checked_mul(elements, size_of::<f64>(), label)
}

fn env_bytes(name: &str, default: usize) -> usize {
    std::env::var(name).map_or(default, |value| {
        let parsed = value
            .parse()
            .unwrap_or_else(|_| panic!("{name} must be a positive byte count"));
        assert!(parsed > 0, "{name} must be a positive byte count");
        parsed
    })
}

fn chain_elements(n_sites: usize, local_dim: usize, bond_dim: usize) -> (usize, usize) {
    let endpoint = checked_mul(local_dim, bond_dim, "chain endpoint elements");
    let interior = checked_mul(endpoint, bond_dim, "chain interior elements");
    let total = checked_mul(2, endpoint, "chain endpoint total")
        .checked_add(checked_mul(
            n_sites.saturating_sub(2),
            interior,
            "chain interior total",
        ))
        .expect("chain input element count overflow");
    (total, endpoint.max(interior))
}

fn binary_tree_shape_counts(n_nodes: usize) -> (usize, usize, usize, usize) {
    assert!(n_nodes >= 3, "binary tree requires at least three nodes");
    let two_child_nodes = (n_nodes - 1) / 2;
    let nonroot_two_child_nodes = two_child_nodes - 1;
    let one_child_nodes = usize::from(n_nodes.is_multiple_of(2));
    let leaves = n_nodes - n_nodes / 2;
    let max_degree = if nonroot_two_child_nodes > 0 { 3 } else { 2 };
    (leaves, one_child_nodes, nonroot_two_child_nodes, max_degree)
}

fn estimate_memory(
    mode: &str,
    n_sites: usize,
    physical_dim: usize,
    bond_dim: usize,
    mps_bond_dim: usize,
) -> MemoryEstimate {
    let physical_squared = checked_mul(physical_dim, physical_dim, "physical dimension");
    let (input_elements, largest_input_elements, output_local_dim, max_degree) = match mode {
        "mpo-mps" => {
            let (mpo_total, mpo_largest) = chain_elements(n_sites, physical_squared, bond_dim);
            let (mps_total, mps_largest) = chain_elements(n_sites, physical_dim, mps_bond_dim);
            (
                mpo_total
                    .checked_add(mps_total)
                    .expect("MPO-MPS input element count overflow"),
                mpo_largest.max(mps_largest),
                physical_dim,
                2,
            )
        }
        "mpo-mpo" => {
            let (one_total, one_largest) = chain_elements(n_sites, physical_squared, bond_dim);
            (
                checked_mul(2, one_total, "MPO-MPO input elements"),
                one_largest,
                physical_squared,
                2,
            )
        }
        "tree" => {
            let (leaves, one_child, nonroot_two_child, max_degree) =
                binary_tree_shape_counts(n_sites);
            let degree_one = checked_mul(physical_squared, bond_dim, "tree leaf elements");
            let degree_two = checked_mul(
                physical_squared,
                checked_pow(bond_dim, 2, "tree degree-two bond product"),
                "tree degree-two elements",
            );
            let degree_three = checked_mul(
                physical_squared,
                checked_pow(bond_dim, 3, "tree degree-three bond product"),
                "tree degree-three elements",
            );
            let one_total = checked_mul(leaves, degree_one, "tree leaf total")
                .checked_add(checked_mul(
                    one_child + 1,
                    degree_two,
                    "tree degree-two total",
                ))
                .and_then(|total| {
                    total.checked_add(checked_mul(
                        nonroot_two_child,
                        degree_three,
                        "tree degree-three total",
                    ))
                })
                .expect("tree input element count overflow");
            let one_largest = if max_degree == 3 {
                degree_three
            } else {
                degree_two
            };
            (
                checked_mul(2, one_total, "tree pair input elements"),
                one_largest,
                physical_squared,
                max_degree,
            )
        }
        _ => panic!("mode must be mpo-mps, mpo-mpo, both, or tree"),
    };
    MemoryEstimate {
        total_input_bytes: bytes(input_elements, "total input bytes"),
        largest_input_tensor_bytes: bytes(largest_input_elements, "largest input tensor bytes"),
        dense_output_bytes: bytes(
            checked_pow(output_local_dim, n_sites, "dense output elements"),
            "dense output bytes",
        ),
        max_degree,
    }
}

fn validate_memory(mode: &str, estimate: MemoryEstimate, exact: bool, network_seed: u64) {
    let max_input_bytes = env_bytes("T4A_BENCH_MAX_INPUT_BYTES", DEFAULT_MAX_INPUT_BYTES);
    let max_dense_bytes = env_bytes("T4A_BENCH_MAX_DENSE_BYTES", DEFAULT_MAX_DENSE_BYTES);
    assert!(
        estimate.total_input_bytes <= max_input_bytes,
        "estimated input {} bytes exceeds T4A_BENCH_MAX_INPUT_BYTES={max_input_bytes}",
        estimate.total_input_bytes
    );
    assert!(
        !exact || estimate.dense_output_bytes <= max_dense_bytes,
        "estimated dense oracle {} bytes exceeds T4A_BENCH_MAX_DENSE_BYTES={max_dense_bytes}",
        estimate.dense_output_bytes
    );
    println!(
        "record=preflight mode={mode} network_seed={network_seed} total_input_bytes={} largest_input_tensor_bytes={} dense_output_bytes={} max_degree={} max_input_bytes={max_input_bytes} max_dense_bytes={max_dense_bytes}",
        estimate.total_input_bytes,
        estimate.largest_input_tensor_bytes,
        estimate.dense_output_bytes,
        estimate.max_degree,
    );
}

fn random_tensor(indices: Vec<DynIndex>, rng: &mut StdRng) -> IdxTensor {
    let elements = indices
        .iter()
        .map(IndexLike::dim)
        .try_fold(1usize, |size, dim| size.checked_mul(dim))
        .expect("tensor element count was checked by benchmark preflight");
    let data = (0..elements)
        .map(|_| rng.random_range(-1.0_f64..1.0_f64))
        .collect();
    IdxTensor::from_dense(indices, data).expect("valid benchmark tensor")
}

fn make_mpo_mps(
    n_sites: usize,
    physical_dim: usize,
    mpo_bond_dim: usize,
    mps_bond_dim: usize,
    seed: u64,
) -> (Network, Network) {
    assert!(n_sites >= 2);
    let mut rng = StdRng::seed_from_u64(seed);
    let inputs = (0..n_sites)
        .map(|_| DynIndex::new_dyn(physical_dim))
        .collect::<Vec<_>>();
    let outputs = (0..n_sites)
        .map(|_| DynIndex::new_dyn(physical_dim))
        .collect::<Vec<_>>();
    let operator_bonds = (0..n_sites - 1)
        .map(|_| DynIndex::new_dyn(mpo_bond_dim))
        .collect::<Vec<_>>();
    let state_bonds = (0..n_sites - 1)
        .map(|_| DynIndex::new_dyn(mps_bond_dim))
        .collect::<Vec<_>>();

    let mut operator_tensors = Vec::with_capacity(n_sites);
    let mut state_tensors = Vec::with_capacity(n_sites);
    for site in 0..n_sites {
        let mut operator_indices = Vec::with_capacity(4);
        if site > 0 {
            operator_indices.push(operator_bonds[site - 1].clone());
        }
        operator_indices.push(inputs[site].clone());
        operator_indices.push(outputs[site].clone());
        if site + 1 < n_sites {
            operator_indices.push(operator_bonds[site].clone());
        }
        operator_tensors.push(random_tensor(operator_indices, &mut rng));

        let mut state_indices = Vec::with_capacity(3);
        if site > 0 {
            state_indices.push(state_bonds[site - 1].clone());
        }
        state_indices.push(inputs[site].clone());
        if site + 1 < n_sites {
            state_indices.push(state_bonds[site].clone());
        }
        state_tensors.push(random_tensor(state_indices, &mut rng));
    }

    let names = (0..n_sites)
        .map(|site| format!("S{site}"))
        .collect::<Vec<_>>();
    (
        Network::from_tensors(operator_tensors, names.clone()).expect("operator topology"),
        Network::from_tensors(state_tensors, names).expect("state topology"),
    )
}

fn make_mpo_mpo(
    n_sites: usize,
    physical_dim: usize,
    bond_dim: usize,
    seed: u64,
) -> (Network, Network) {
    assert!(n_sites >= 2);
    let mut rng = StdRng::seed_from_u64(seed);
    let inputs = (0..n_sites)
        .map(|_| DynIndex::new_dyn(physical_dim))
        .collect::<Vec<_>>();
    let outputs_a = (0..n_sites)
        .map(|_| DynIndex::new_dyn(physical_dim))
        .collect::<Vec<_>>();
    let outputs_b = (0..n_sites)
        .map(|_| DynIndex::new_dyn(physical_dim))
        .collect::<Vec<_>>();
    let bonds_a = (0..n_sites - 1)
        .map(|_| DynIndex::new_dyn(bond_dim))
        .collect::<Vec<_>>();
    let bonds_b = (0..n_sites - 1)
        .map(|_| DynIndex::new_dyn(bond_dim))
        .collect::<Vec<_>>();

    let mut tensors_a = Vec::with_capacity(n_sites);
    let mut tensors_b = Vec::with_capacity(n_sites);
    for site in 0..n_sites {
        let mut indices_a = Vec::with_capacity(5);
        let mut indices_b = Vec::with_capacity(5);
        if site > 0 {
            indices_a.push(bonds_a[site - 1].clone());
            indices_b.push(bonds_b[site - 1].clone());
        }
        indices_a.push(inputs[site].clone());
        indices_b.push(inputs[site].clone());
        indices_a.push(outputs_a[site].clone());
        indices_b.push(outputs_b[site].clone());
        if site + 1 < n_sites {
            indices_a.push(bonds_a[site].clone());
            indices_b.push(bonds_b[site].clone());
        }
        tensors_a.push(random_tensor(indices_a, &mut rng));
        tensors_b.push(random_tensor(indices_b, &mut rng));
    }

    let names = (0..n_sites)
        .map(|site| format!("S{site}"))
        .collect::<Vec<_>>();
    (
        Network::from_tensors(tensors_a, names.clone()).expect("first operator topology"),
        Network::from_tensors(tensors_b, names).expect("second operator topology"),
    )
}

fn make_binary_tree_pair(
    n_nodes: usize,
    physical_dim: usize,
    bond_dim: usize,
    seed: u64,
) -> (Network, Network) {
    assert!(n_nodes >= 3);
    let mut rng = StdRng::seed_from_u64(seed);
    let shared = (0..n_nodes)
        .map(|_| DynIndex::new_dyn(physical_dim))
        .collect::<Vec<_>>();
    let outputs_a = (0..n_nodes)
        .map(|_| DynIndex::new_dyn(physical_dim))
        .collect::<Vec<_>>();
    let outputs_b = (0..n_nodes)
        .map(|_| DynIndex::new_dyn(physical_dim))
        .collect::<Vec<_>>();
    let bonds_a = (1..n_nodes)
        .map(|_| DynIndex::new_dyn(bond_dim))
        .collect::<Vec<_>>();
    let bonds_b = (1..n_nodes)
        .map(|_| DynIndex::new_dyn(bond_dim))
        .collect::<Vec<_>>();
    let names = (0..n_nodes)
        .map(|node| format!("N{node:04}"))
        .collect::<Vec<_>>();

    let build_indices = |node: usize, outputs: &[DynIndex], bonds: &[DynIndex]| {
        let mut indices = vec![shared[node].clone(), outputs[node].clone()];
        if node > 0 {
            indices.push(bonds[node - 1].clone());
        }
        for child in [2 * node + 1, 2 * node + 2] {
            if child < n_nodes {
                indices.push(bonds[child - 1].clone());
            }
        }
        indices
    };
    let tensors_a = (0..n_nodes)
        .map(|node| random_tensor(build_indices(node, &outputs_a, &bonds_a), &mut rng))
        .collect();
    let tensors_b = (0..n_nodes)
        .map(|node| random_tensor(build_indices(node, &outputs_b, &bonds_b), &mut rng))
        .collect();
    (
        Network::from_tensors(tensors_a, names.clone()).expect("first binary-tree topology"),
        Network::from_tensors(tensors_b, names).expect("second binary-tree topology"),
    )
}

fn max_bond(network: &Network) -> usize {
    network
        .node_indices()
        .into_iter()
        .flat_map(|node| network.edges_for_node(node))
        .filter_map(|(edge, _)| network.bond_index(edge).map(IndexLike::dim))
        .max()
        .unwrap_or(1)
}

fn run_case(
    label: &str,
    left: &Network,
    right: &Network,
    options: ContractionOptions,
    requested_max_rank: usize,
    reps: usize,
    exact: Option<&IdxTensor>,
) {
    tensor4all_core::reset_contract_profile();
    tensor4all_core::reset_native_einsum_profile();
    tensor4all_core::reset_pairwise_contract_profile();
    let center = std::env::var("T4A_BENCH_CENTER").unwrap_or_else(|_| {
        left.node_names()
            .into_iter()
            .min()
            .expect("non-empty benchmark network")
    });
    let warmup = contract(left, right, &center, options.clone()).expect("benchmark warm-up");
    std::hint::black_box(warmup);
    let start = Instant::now();
    let mut result = None;
    for _ in 0..reps {
        result =
            Some(contract(left, right, &center, options.clone()).expect("benchmark contraction"));
    }
    let elapsed = start.elapsed();
    let result = result.expect("at least one benchmark repetition");
    let rel_error = exact.map(|exact_dense| {
        let dense = result.to_dense().expect("benchmark dense materialization");
        let dist = dense.distance(exact_dense).expect("benchmark distance");
        let denom = exact_dense.norm().expect("benchmark norm");
        if denom > 0.0 {
            dist / denom
        } else {
            dist
        }
    });
    println!(
        "record=case name={label} reps={reps} elapsed_seconds={:.6} per_run_seconds={:.6} nodes={} edges={} requested_max_rank={requested_max_rank} effective_max_bond={} src_seed={} center={center} relative_error={}",
        elapsed.as_secs_f64(),
        elapsed.as_secs_f64() / reps as f64,
        result.node_count(),
        result.edge_count(),
        max_bond(&result),
        options.src_options.seed,
        rel_error.map_or_else(|| "n/a".to_string(), |e| format!("{e:.3e}")),
    );
    tensor4all_core::print_and_reset_contract_profile();
    tensor4all_core::print_and_reset_native_einsum_profile();
    tensor4all_core::print_and_reset_pairwise_contract_profile();
}

fn enabled_features() -> String {
    [
        ("tenferro-cpu-faer", cfg!(feature = "tenferro-cpu-faer")),
        (
            "tenferro-provider-inject",
            cfg!(feature = "tenferro-provider-inject"),
        ),
        ("tenferro-cuda", cfg!(feature = "tenferro-cuda")),
    ]
    .into_iter()
    .filter_map(|(name, enabled)| enabled.then_some(name))
    .collect::<Vec<_>>()
    .join(",")
}

fn main() {
    let mut args = std::env::args().skip(1);
    let n_sites = args
        .next()
        .map_or(10, |value| value.parse().expect("n_sites"));
    let bond_dim: usize = args
        .next()
        .map_or(4, |value| value.parse().expect("bond_dim"));
    let reps = args.next().map_or(1, |value| value.parse().expect("reps"));
    let mode = args.next().unwrap_or_else(|| "both".to_string());
    let rank_increment = args
        .next()
        .map_or(3, |value| value.parse().expect("rank_increment"));
    let final_svd = args
        .next()
        .is_some_and(|value| value.parse().expect("final_svd"));
    let physical_dim = 2;
    let mps_bond_dim: usize = std::env::var("T4A_BENCH_MPS_BOND_DIM")
        .map_or(bond_dim, |value| value.parse().expect("MPS bond dimension"));
    let mpo_mps_product_bond = bond_dim
        .checked_mul(mps_bond_dim)
        .expect("MPO-MPS product bond dimension");
    let equal_bond_product = bond_dim
        .checked_mul(bond_dim)
        .expect("product bond dimension");
    let default_max_rank = match mode.as_str() {
        "mpo-mps" => mpo_mps_product_bond,
        "both" => mpo_mps_product_bond.max(equal_bond_product),
        _ => equal_bond_product,
    };
    let max_rank = args.next().map_or(default_max_rank, |value| {
        value.parse().expect("target_rank")
    });
    let algorithm = std::env::var("T4A_BENCH_ALGORITHM").unwrap_or_else(|_| "all".to_string());
    assert!(n_sites >= 2, "n_sites must be at least 2");
    assert!(reps >= 1, "reps must be at least 1");
    assert!(
        mode != "tree" || n_sites >= 3,
        "tree mode requires at least 3 nodes"
    );
    assert!(
        matches!(mode.as_str(), "mpo-mps" | "mpo-mpo" | "both" | "tree"),
        "mode must be mpo-mps, mpo-mpo, both, or tree"
    );
    assert!(
        matches!(
            algorithm.as_str(),
            "all" | "zipup" | "src-fixed" | "src-adaptive"
        ),
        "T4A_BENCH_ALGORITHM must be all, zipup, src-fixed, or src-adaptive"
    );

    let skip_exact = std::env::var("T4A_BENCH_SKIP_EXACT").is_ok();
    let profile = if cfg!(debug_assertions) {
        "debug"
    } else {
        "release"
    };
    println!(
        "record=build git_commit={BUILD_GIT_COMMIT} profile={profile} backend=tenferro features={}",
        enabled_features()
    );
    println!(
        "record=config n_sites={n_sites} physical_dim={physical_dim} mpo_bond={bond_dim} mps_bond={mps_bond_dim} requested_max_rank={max_rank} reps={reps} mode={mode} algorithm={algorithm} rank_increment={rank_increment} final_svd={final_svd} src_seed=1234 adaptive_rtol=1e-4 adaptive_atol=0 adaptive_min_rank=2"
    );
    let run = |label: &str, left: Network, right: Network| {
        let exact = if skip_exact {
            None
        } else {
            let start = Instant::now();
            let dense = left
                .contract_naive(&right)
                .expect("benchmark naive exact contraction");
            println!(
                "record=reference name={label}/naive-exact elapsed_seconds={:.6}",
                start.elapsed().as_secs_f64()
            );
            Some(dense)
        };
        let exact_ref = exact.as_ref();
        if algorithm == "all" || algorithm == "zipup" {
            run_case(
                &format!("{label}/zipup"),
                &left,
                &right,
                ContractionOptions::zipup().with_max_bond_dim(max_rank),
                max_rank,
                reps,
                exact_ref,
            );
        }
        if algorithm == "all" || algorithm == "src-fixed" {
            run_case(
                &format!("{label}/src-fixed"),
                &left,
                &right,
                ContractionOptions::src()
                    .with_max_bond_dim(max_rank)
                    .with_src_options(
                        SrcOptions::fixed()
                            .with_seed(1234)
                            .with_final_svd(final_svd),
                    ),
                max_rank,
                reps,
                exact_ref,
            );
        }
        if algorithm == "all" || algorithm == "src-adaptive" {
            run_case(
                &format!("{label}/src-adaptive"),
                &left,
                &right,
                ContractionOptions::src()
                    .with_max_bond_dim(max_rank)
                    .with_src_options(
                        SrcOptions::adaptive(1.0e-4, max_rank)
                            .with_min_rank(2)
                            .with_rank_increment(rank_increment)
                            .with_seed(1234)
                            .with_final_svd(final_svd),
                    ),
                max_rank,
                reps,
                exact_ref,
            );
        }
    };

    if mode == "mpo-mps" || mode == "both" {
        validate_memory(
            "mpo-mps",
            estimate_memory("mpo-mps", n_sites, physical_dim, bond_dim, mps_bond_dim),
            !skip_exact,
            7,
        );
        let (operator, state) = make_mpo_mps(n_sites, physical_dim, bond_dim, mps_bond_dim, 7);
        run("mpo-mps", operator, state);
    }
    if mode == "mpo-mpo" || mode == "both" {
        validate_memory(
            "mpo-mpo",
            estimate_memory("mpo-mpo", n_sites, physical_dim, bond_dim, mps_bond_dim),
            !skip_exact,
            11,
        );
        let (left, right) = make_mpo_mpo(n_sites, physical_dim, bond_dim, 11);
        run("mpo-mpo", left, right);
    }
    if mode == "tree" {
        validate_memory(
            "tree",
            estimate_memory("tree", n_sites, physical_dim, bond_dim, mps_bond_dim),
            !skip_exact,
            13,
        );
        let (left, right) = make_binary_tree_pair(n_sites, physical_dim, bond_dim, 13);
        run("tree-mpo-mpo", left, right);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chain_memory_estimate_is_checked_and_exact() {
        let estimate = estimate_memory("mpo-mps", 3, 2, 2, 2);
        assert_eq!(estimate.total_input_bytes, 384);
        assert_eq!(estimate.largest_input_tensor_bytes, 128);
        assert_eq!(estimate.dense_output_bytes, 64);
        assert_eq!(estimate.max_degree, 2);
    }

    #[test]
    fn binary_tree_estimate_has_bounded_degree_and_core_size() {
        let estimate = estimate_memory("tree", 7, 2, 4, 4);
        assert_eq!(estimate.total_input_bytes, 10_240);
        assert_eq!(estimate.largest_input_tensor_bytes, 2_048);
        assert_eq!(estimate.dense_output_bytes, 131_072);
        assert_eq!(estimate.max_degree, 3);
        assert_eq!(binary_tree_shape_counts(15).3, 3);
    }

    #[test]
    #[should_panic(expected = "dense output elements overflow")]
    fn memory_estimate_rejects_dense_output_overflow() {
        let _ = estimate_memory("mpo-mpo", usize::BITS as usize, 2, 2, 2);
    }
}
