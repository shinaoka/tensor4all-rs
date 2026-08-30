//! Small, deterministic SRC-vs-zip-up contraction benchmark.
//!
//! Run with `RAYON_NUM_THREADS=1` for reproducible CPU measurements. Set
//! `T4A_PROFILE_CONTRACT=1` to print the aggregated dense contraction
//! signatures after each case.

use std::time::Instant;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use tensor4all_core::{DynIndex, IdxTensor, IndexLike};
use tensor4all_treetn::contraction::{contract, ContractionOptions, SrcOptions};
use tensor4all_treetn::TreeTN;

type Network = TreeTN<IdxTensor, String>;

fn random_tensor(indices: Vec<DynIndex>, rng: &mut StdRng) -> IdxTensor {
    let elements = indices.iter().map(IndexLike::dim).product();
    let data = (0..elements)
        .map(|_| rng.random_range(-1.0_f64..1.0_f64))
        .collect();
    IdxTensor::from_dense(indices, data).expect("valid benchmark tensor")
}

fn make_mpo_mps(
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
    let outputs = (0..n_sites)
        .map(|_| DynIndex::new_dyn(physical_dim))
        .collect::<Vec<_>>();
    let operator_bonds = (0..n_sites - 1)
        .map(|_| DynIndex::new_dyn(bond_dim))
        .collect::<Vec<_>>();
    let state_bonds = (0..n_sites - 1)
        .map(|_| DynIndex::new_dyn(bond_dim))
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

fn make_star_pair(
    n_leaves: usize,
    physical_dim: usize,
    bond_dim: usize,
    seed: u64,
) -> (Network, Network) {
    assert!(n_leaves >= 2);
    let mut rng = StdRng::seed_from_u64(seed);
    let shared = (0..n_leaves + 1)
        .map(|_| DynIndex::new_dyn(physical_dim))
        .collect::<Vec<_>>();
    let outputs_a = (0..n_leaves + 1)
        .map(|_| DynIndex::new_dyn(physical_dim))
        .collect::<Vec<_>>();
    let outputs_b = (0..n_leaves + 1)
        .map(|_| DynIndex::new_dyn(physical_dim))
        .collect::<Vec<_>>();
    let bonds_a = (0..n_leaves)
        .map(|_| DynIndex::new_dyn(bond_dim))
        .collect::<Vec<_>>();
    let bonds_b = (0..n_leaves)
        .map(|_| DynIndex::new_dyn(bond_dim))
        .collect::<Vec<_>>();
    let names = std::iter::once("C".to_string())
        .chain((0..n_leaves).map(|leaf| format!("L{leaf}")))
        .collect::<Vec<_>>();

    let mut center_a = vec![shared[0].clone(), outputs_a[0].clone()];
    center_a.extend(bonds_a.iter().cloned());
    let mut center_b = vec![shared[0].clone(), outputs_b[0].clone()];
    center_b.extend(bonds_b.iter().cloned());
    let mut tensors_a = vec![random_tensor(center_a, &mut rng)];
    let mut tensors_b = vec![random_tensor(center_b, &mut rng)];
    for leaf in 0..n_leaves {
        tensors_a.push(random_tensor(
            vec![
                shared[leaf + 1].clone(),
                outputs_a[leaf + 1].clone(),
                bonds_a[leaf].clone(),
            ],
            &mut rng,
        ));
        tensors_b.push(random_tensor(
            vec![
                shared[leaf + 1].clone(),
                outputs_b[leaf + 1].clone(),
                bonds_b[leaf].clone(),
            ],
            &mut rng,
        ));
    }
    (
        Network::from_tensors(tensors_a, names.clone()).expect("first star topology"),
        Network::from_tensors(tensors_b, names).expect("second star topology"),
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
        "case={label} reps={reps} elapsed={:.6}s per_run={:.6}s nodes={} edges={} max_bond={} relative_error={}",
        elapsed.as_secs_f64(),
        elapsed.as_secs_f64() / reps as f64,
        result.node_count(),
        result.edge_count(),
        max_bond(&result),
        rel_error.map_or_else(|| "n/a".to_string(), |e| format!("{e:.3e}")),
    );
    tensor4all_core::print_and_reset_contract_profile();
    tensor4all_core::print_and_reset_native_einsum_profile();
    tensor4all_core::print_and_reset_pairwise_contract_profile();
}

fn main() {
    let mut args = std::env::args().skip(1);
    let n_sites = args
        .next()
        .map_or(10, |value| value.parse().expect("n_sites"));
    let bond_dim = args
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
    let max_rank = args.next().map_or(bond_dim * bond_dim, |value| {
        value.parse().expect("target_rank")
    });

    println!(
        "config=n_sites:{n_sites} physical_dim:{physical_dim} input_bond:{bond_dim} max_rank:{max_rank} reps:{reps} mode:{mode} rank_increment:{rank_increment} final_svd:{final_svd}"
    );

    let skip_exact = std::env::var("T4A_BENCH_SKIP_EXACT").is_ok();
    let run = |label: &str, left: Network, right: Network| {
        let exact = if skip_exact {
            None
        } else {
            let start = Instant::now();
            let dense = left
                .contract_naive(&right)
                .expect("benchmark naive exact contraction");
            println!(
                "case={label}/naive-exact elapsed={:.6}s (reference only)",
                start.elapsed().as_secs_f64()
            );
            Some(dense)
        };
        let exact_ref = exact.as_ref();
        run_case(
            &format!("{label}/zipup"),
            &left,
            &right,
            ContractionOptions::zipup().with_max_bond_dim(max_rank),
            reps,
            exact_ref,
        );
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
            reps,
            exact_ref,
        );
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
            reps,
            exact_ref,
        );
    };

    if mode == "mpo-mps" || mode == "both" {
        let (operator, state) = make_mpo_mps(n_sites, physical_dim, bond_dim, 7);
        run("mpo-mps", operator, state);
    }
    if mode == "mpo-mpo" || mode == "both" {
        let (left, right) = make_mpo_mpo(n_sites, physical_dim, bond_dim, 11);
        run("mpo-mpo", left, right);
    }
    if mode == "tree" {
        let (left, right) = make_star_pair(n_sites - 1, physical_dim, bond_dim, 13);
        run("tree-mpo-mpo", left, right);
    }
}
