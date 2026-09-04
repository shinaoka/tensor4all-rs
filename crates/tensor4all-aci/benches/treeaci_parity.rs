//! Paired comparison of `tensor4all-aci` and `tensor4all-treeaci` on a chain.
//!
//! The question is whether the two implementations perform the same asymptotic
//! work per evaluation, not how fast either is: a chain is where they are
//! directly comparable, and a difference in fitted scaling exponent against bond
//! dimension would be an algorithmic defect that no amount of interface work
//! would fix. At small bond dimension almost all of the cost is a per-call
//! constant, so such a defect is invisible there; this runs at 16 through 256.
//!
//! Both arms start from the same first input, converted to their native network
//! type. This keeps initialization, numerical state, and canonicalization work
//! matched rather than comparing two unrelated random pivot trajectories.
//!
//! It lives here rather than beside `tensor4all-treeaci` because that crate's
//! own test asserts its manifest does not mention `tensor4all-aci`. The
//! dependency direction is one-way by design, so the comparison belongs on this
//! side.
//!
//! # #718 complete-ACI parity report
//!
//! This is the parity gate #718 requires to be preserved, including its
//! bond-256 reference point. Each case reports, before its timed loop:
//! callback invocations and evaluated points for both arms, sweeps, maximum
//! output rank, the last pivot error, the dense residual against the exact
//! elementwise product, the tree arm's retained frame/sample/candidate cache
//! accounting, the configured cache and working-byte budgets, and the fact
//! that the run completed without a `ResourceLimit` refusal. Set
//! `T4A_TREEACI_PARITY_ENABLE_GUARD=1` for the default-Guard comparison.
//!
//! Two of the counters #718 lists are **not** observable from this level and
//! are deliberately not invented here: the cached evaluator's message-cache
//! hit/miss/eviction counts and a process peak-byte figure. `TreeACI`'s public
//! diagnostics expose retained records and logical retained bytes, not cache
//! hit rates; the hit/miss/eviction properties are gated by the crate-internal
//! counter tests in `tensor4all-treetn` and `tensor4all-treeaci` instead.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use tensor4all_aci::{elementwise_batched, AciOptions, ElementwiseBatch};
use tensor4all_core::{DynIndex, IdxTensor};
use tensor4all_simplett::{tensor3_from_data, SimpleTensorTrain};
use tensor4all_treeaci::{tree_elementwise_batched, TreeAciOptions, TreeElementwiseBatch};
use tensor4all_treetn::{tensor_train_to_treetn_with_names_and_site_indices, TreeTN};

const N_SITES: usize = 16;
const LOCAL_DIM: usize = 2;
const N_INPUTS: usize = 2;
const TOLERANCE: f64 = 1e-8;
const MAX_BOND_DIM: usize = 4096;
const MAX_SWEEPS: usize = 20;
const MIN_SWEEPS: usize = 2;
const CHI_VALUES: [usize; 5] = [16, 32, 64, 128, 256];
const MAX_PARITY_ERROR_FACTOR: f64 = 10.0;

fn link_dims(n_sites: usize, local_dim: usize, chi: usize) -> Vec<usize> {
    (0..n_sites.saturating_sub(1))
        .map(|bond| {
            let left_sites = bond + 1;
            let right_sites = n_sites - left_sites;
            let max_exact_rank = local_dim.pow(left_sites.min(right_sites) as u32);
            chi.min(max_exact_rank).max(1)
        })
        .collect()
}

/// Deterministic, well-conditioned cores, verbatim from the neighbouring
/// `elementwise_scaling` benchmark so the two are comparable.
///
/// The full mixing keeps the high-rank cases non-degenerate, so the benchmark
/// measures interpolation work instead of accidental rank collapse.
#[allow(clippy::too_many_arguments)]
fn core_value(
    input_index: usize,
    site: usize,
    physical: usize,
    left: usize,
    right: usize,
    left_dim: usize,
    right_dim: usize,
) -> f64 {
    let input = input_index as f64 + 1.0;
    let site = site as f64 + 1.0;
    let physical = physical as f64 + 1.0;
    let left = left as f64 + 1.0;
    let right = right as f64 + 1.0;
    let left_coord = left / (left_dim as f64 + 1.0);
    let right_coord = right / (right_dim as f64 + 1.0);
    let phase = 0.173 * input * site
        + 0.193 * physical
        + 0.071 * left * right
        + 0.109 * input * left
        + 0.131 * site * right;
    let bond_mix = 0.29 * phase.sin()
        + 0.23 * (0.157 * input * physical * right + 0.211 * site * left).cos()
        + 0.17 * (left_coord - right_coord) * physical;
    let site_value = 0.31 + bond_mix;
    let scale = ((left_dim * right_dim) as f64).powf(0.25);
    site_value / scale
}

fn deterministic_tt(input: usize, chi: usize) -> SimpleTensorTrain<f64> {
    let links = link_dims(N_SITES, LOCAL_DIM, chi);
    let cores = (0..N_SITES)
        .map(|site| {
            let left_dim = if site == 0 { 1 } else { links[site - 1] };
            let right_dim = links.get(site).copied().unwrap_or(1);
            let mut data = vec![0.0; left_dim * LOCAL_DIM * right_dim];
            for right in 0..right_dim {
                for physical in 0..LOCAL_DIM {
                    for left in 0..left_dim {
                        data[left + left_dim * (physical + LOCAL_DIM * right)] =
                            core_value(input, site, physical, left, right, left_dim, right_dim);
                    }
                }
            }
            tensor3_from_data(data, left_dim, LOCAL_DIM, right_dim).unwrap()
        })
        .collect::<Vec<_>>();
    SimpleTensorTrain::new(cores).unwrap()
}

fn multiply_train(batch: ElementwiseBatch<'_, f64>, out: &mut [f64]) -> tensor4all_aci::Result<()> {
    for (point, value) in out.iter_mut().enumerate().take(batch.n_points()) {
        let mut product = 1.0;
        for input in 0..batch.n_inputs() {
            product *= batch.get(input, point)?;
        }
        *value = product;
    }
    Ok(())
}

fn multiply_tree(
    batch: TreeElementwiseBatch<'_, f64>,
    out: &mut [f64],
) -> tensor4all_treeaci::Result<()> {
    for (point, value) in out.iter_mut().enumerate() {
        let mut product = 1.0;
        for input in 0..N_INPUTS {
            product *= batch.get(input, point)?;
        }
        *value = product;
    }
    Ok(())
}

struct Case {
    trains: Vec<SimpleTensorTrain<f64>>,
    trees: Vec<TreeTN<IdxTensor, usize>>,
    sites: Vec<DynIndex>,
}

fn build(chi: usize) -> Case {
    let trains: Vec<_> = (0..N_INPUTS).map(|i| deterministic_tt(i, chi)).collect();

    // One shared site-index set, so the tree arm's inputs and initial guess are
    // index-compatible; converting each independently mints fresh indices and
    // `tree_elementwise` rejects the mismatch.
    let sites: Vec<DynIndex> = (0..N_SITES).map(|_| DynIndex::new_dyn(LOCAL_DIM)).collect();
    let names: Vec<usize> = (0..N_SITES).collect();
    let convert = |tt: &SimpleTensorTrain<f64>| {
        tensor_train_to_treetn_with_names_and_site_indices::<f64, usize>(
            tt,
            names.clone(),
            sites.clone(),
        )
        .unwrap()
    };
    Case {
        trees: trains.iter().map(convert).collect(),
        trains,
        sites,
    }
}

#[derive(Clone, Copy, Debug)]
struct AccuracyOracle {
    exact_scale: f64,
    train_maxabs: f64,
    tree_maxabs: f64,
}

impl AccuracyOracle {
    fn train_relative(self) -> f64 {
        self.train_maxabs / self.exact_scale
    }

    fn tree_relative(self) -> f64 {
        self.tree_maxabs / self.exact_scale
    }
}

fn maxabs_difference(actual: &[f64], expected: &[f64]) -> f64 {
    actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0, f64::max)
}

fn accuracy_oracle(
    case: &Case,
    train: &tensor4all_aci::AciResult<f64>,
    tree: &tensor4all_treeaci::TreeAciResult<usize>,
) -> AccuracyOracle {
    let dense_inputs = case
        .trains
        .iter()
        .map(|input| input.full_tensor().unwrap().0)
        .collect::<Vec<_>>();
    let mut exact = vec![1.0; dense_inputs[0].len()];
    for input in dense_inputs {
        for (product, value) in exact.iter_mut().zip(input) {
            *product *= value;
        }
    }
    let train_dense = train.tensor_train.full_tensor().unwrap().0;
    let tree_dense = tree
        .tree
        .to_dense()
        .unwrap()
        .permute_indices(&case.sites)
        .unwrap()
        .to_vec::<f64>()
        .unwrap();
    AccuracyOracle {
        exact_scale: exact.iter().map(|value| value.abs()).fold(0.0, f64::max),
        train_maxabs: maxabs_difference(&train_dense, &exact),
        tree_maxabs: maxabs_difference(&tree_dense, &exact),
    }
}

fn train_options(
    guess: Option<SimpleTensorTrain<f64>>,
    enable_global_guard: bool,
) -> AciOptions<f64> {
    AciOptions {
        tolerance: TOLERANCE,
        max_bond_dim: Some(MAX_BOND_DIM),
        max_iters: MAX_SWEEPS,
        min_iters: MIN_SWEEPS,
        initial_guess: guess,
        enable_global_guard,
        ..AciOptions::default()
    }
}

fn tree_options(
    guess: Option<TreeTN<IdxTensor, usize>>,
    enable_global_guard: bool,
) -> TreeAciOptions<usize> {
    TreeAciOptions {
        tolerance: TOLERANCE,
        max_bond_dim: Some(MAX_BOND_DIM),
        max_sweeps: MAX_SWEEPS,
        min_sweeps: MIN_SWEEPS,
        initial_guess: guess,
        enable_global_guard,
        ..TreeAciOptions::default()
    }
}

/// Prints the reproducibility metadata #718 requires on every report.
fn print_provenance_header(enable_global_guard: bool) {
    let environment = |key: &str| std::env::var(key).unwrap_or_else(|_| "unset".to_owned());
    let cpu_model = std::fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|info| {
            info.lines()
                .find(|line| line.starts_with("model name"))
                .and_then(|line| line.split_once(':'))
                .map(|(_, value)| value.trim().to_owned())
        })
        .unwrap_or_else(|| "unknown".to_owned());
    let affinity = std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|status| {
            status
                .lines()
                .find(|line| line.starts_with("Cpus_allowed_list"))
                .and_then(|line| line.split_once(':'))
                .map(|(_, value)| value.trim().to_owned())
        })
        .unwrap_or_else(|| "unknown".to_owned());
    let options = tree_options(None, enable_global_guard);
    println!("=== #718 complete-ACI parity gate: reproducibility metadata ===");
    println!(
        "baseline_commit={} candidate_commit={}",
        environment("T4A_ACI_SCALING_BASELINE"),
        environment("T4A_ACI_SCALING_CANDIDATE")
    );
    println!("hardware={cpu_model} | cpu_affinity={affinity}");
    println!(
        "provider/threads: RAYON_NUM_THREADS={} OMP_NUM_THREADS={} TENFERRO_NUM_THREADS={}",
        environment("RAYON_NUM_THREADS"),
        environment("OMP_NUM_THREADS"),
        environment("TENFERRO_NUM_THREADS")
    );
    println!(
        "build_profile={} | global_guard={enable_global_guard} | rng_seed={} | \
         tolerance={TOLERANCE:e} | min_sweeps={MIN_SWEEPS} max_sweeps={MAX_SWEEPS} | \
         max_bond_dim={MAX_BOND_DIM} | sites={N_SITES} local_dim={LOCAL_DIM} inputs={N_INPUTS}",
        if cfg!(debug_assertions) {
            "debug-or-unoptimized"
        } else {
            "release/bench"
        },
        options.rng_seed
    );
    println!(
        "budgets: message_cache_max_bytes={} max_frame_bytes={} max_sample_arena_bytes={} \
         max_working_bytes={} max_candidate_rows={} max_candidate_cols={}",
        options.message_cache_max_bytes,
        options.max_frame_bytes,
        options.max_sample_arena_bytes,
        options.max_working_bytes,
        options.max_candidate_rows,
        options.max_candidate_cols
    );
    println!(
        "statistic=Criterion median with a 95% bootstrap confidence interval | samples=10 | \
         noise gate = run-to-run spread of per-run medians over at least three whole-binary \
         repetitions; predeclared MDE = above that spread"
    );
    println!("chi sweep = {CHI_VALUES:?} (the bond-256 rung is the #699 reference point)");
    println!("===============================================================");
}

fn bench_parity(c: &mut Criterion) {
    // [AI Supplied] Keep the established no-Guard sweep comparison as the
    // default, while allowing an explicit default-path run that includes the
    // global guard without maintaining a second copy of this fixture.
    let enable_global_guard = std::env::var_os("T4A_TREEACI_PARITY_ENABLE_GUARD").is_some();
    let group_name = if enable_global_guard {
        "aci_vs_treeaci_chain_guard"
    } else {
        "aci_vs_treeaci_chain"
    };
    print_provenance_header(enable_global_guard);
    let mut group = c.benchmark_group(group_name);
    group.sample_size(10);

    for chi in CHI_VALUES {
        let case = build(chi);

        // Report rank and termination once per case, outside the timing loop, so
        // a case where an arm failed to converge is visible rather than merely
        // fast.
        let mut train_evaluated_points = 0u64;
        let mut train_callbacks = 0u64;
        let train = elementwise_batched(
            |batch, output| {
                train_callbacks = train_callbacks.saturating_add(1);
                train_evaluated_points = train_evaluated_points
                    .saturating_add(u64::try_from(batch.n_points()).unwrap_or(u64::MAX));
                multiply_train(batch, output)
            },
            &case.trains,
            &train_options(Some(case.trains[0].clone()), enable_global_guard),
        )
        .unwrap();
        let mut tree_callbacks = 0u64;
        let mut tree_callback_points = 0u64;
        let tree = tree_elementwise_batched::<f64, _, _>(
            |batch, output| {
                tree_callbacks = tree_callbacks.saturating_add(1);
                tree_callback_points = tree_callback_points
                    .saturating_add(u64::try_from(batch.n_points()).unwrap_or(u64::MAX));
                multiply_tree(batch, output)
            },
            &case.trees,
            &tree_options(Some(case.trees[0].clone()), enable_global_guard),
        )
        .unwrap();
        let accuracy = accuracy_oracle(&case, &train, &tree);
        println!(
            "chi={chi:<4} train: rank {:>5} err {:.3e} sweeps {:>2} | tree: rank {:>5} err {:.3e} sweeps {:>2} ({:?})",
            train.ranks.iter().copied().max().unwrap_or(0),
            train.errors.last().copied().unwrap_or(f64::NAN),
            train.ranks.len(),
            tree.max_ranks.iter().copied().max().unwrap_or(0),
            tree.max_errors.last().copied().unwrap_or(f64::NAN),
            tree.max_ranks.len(),
            tree.termination,
        );
        println!(
            "         callbacks: train {train_callbacks} | tree {tree_callbacks} | \
             evaluated_points: train {} | tree {} (callback-observed {tree_callback_points})",
            train_evaluated_points, tree.diagnostics.evaluated_points,
        );
        let tree_candidate_entries: usize = tree
            .diagnostics
            .candidate_set_sizes
            .iter()
            .map(|(_, _, len)| *len)
            .sum();
        println!(
            "         tree retained: frame_records {} frame_bytes {} sample_records {} \
             sample_bytes {} candidate_entries {} saturated_edges {} global_pivots {:?}",
            tree.diagnostics.frame_records,
            tree.diagnostics.frame_retained_bytes,
            tree.diagnostics.sample_arena_records,
            tree.diagnostics.sample_arena_retained_bytes,
            tree_candidate_entries,
            tree.diagnostics.saturated_edges.len(),
            tree.global_pivots_found,
        );
        // Working-limit compliance: the run above returned `Ok`, so every
        // prepared allocation stayed inside the configured budgets. The
        // complementary refusal path is exercised by the treeaci scaling
        // benchmark's working-budget ladder and by the crate's unit tests.
        let budgets = tree_options(None, enable_global_guard);
        assert!(
            tree.diagnostics.frame_retained_bytes <= budgets.max_frame_bytes,
            "chi={chi}: retained frame bytes {} exceed max_frame_bytes {}",
            tree.diagnostics.frame_retained_bytes,
            budgets.max_frame_bytes
        );
        assert!(
            tree.diagnostics.sample_arena_retained_bytes <= budgets.max_sample_arena_bytes,
            "chi={chi}: retained sample bytes {} exceed max_sample_arena_bytes {}",
            tree.diagnostics.sample_arena_retained_bytes,
            budgets.max_sample_arena_bytes
        );
        println!(
            "         exact scale {:.3e} | train maxabs {:.3e} rel {:.3e} | tree maxabs {:.3e} rel {:.3e}",
            accuracy.exact_scale,
            accuracy.train_maxabs,
            accuracy.train_relative(),
            accuracy.tree_maxabs,
            accuracy.tree_relative(),
        );
        assert!(
            accuracy.tree_relative()
                <= MAX_PARITY_ERROR_FACTOR
                    * accuracy.train_relative().max(TOLERANCE),
            "chi={chi}: TreeACI relative error {:.3e} exceeds the chain parity bound {:.3e} (train {:.3e})",
            accuracy.tree_relative(),
            MAX_PARITY_ERROR_FACTOR * accuracy.train_relative().max(TOLERANCE),
            accuracy.train_relative(),
        );

        group.bench_with_input(BenchmarkId::new("train", chi), &case, |b, case| {
            b.iter(|| {
                elementwise_batched(
                    multiply_train,
                    black_box(&case.trains),
                    &train_options(Some(case.trains[0].clone()), enable_global_guard),
                )
                .unwrap()
            })
        });
        group.bench_with_input(BenchmarkId::new("tree", chi), &case, |b, case| {
            b.iter(|| {
                tree_elementwise_batched::<f64, _, _>(
                    multiply_tree,
                    black_box(&case.trees),
                    &tree_options(Some(case.trees[0].clone()), enable_global_guard),
                )
                .unwrap()
            })
        });
    }
    group.finish();
}

criterion_group!(benches, bench_parity);
criterion_main!(benches);
