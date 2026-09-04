//! Independent scaling gates for TreeACI (issue #718).
//!
//! Every earlier subissue of the #699 umbrella measured one mechanism at one
//! or two fixed sizes. A fixed size cannot separate a constant-factor win from
//! an exponent change, and a single favourable run cannot separate either from
//! noise. This benchmark therefore varies **one** dimension at a time, holding
//! all others fixed, over the six dimensions #718 names:
//!
//! 1. chain length `N`;
//! 2. fixed input bond dimension `chi`;
//! 3. active candidate/output rank (`max_bond_dim`);
//! 4. coordination number `z`;
//! 5. unequal incident bonds at a fixed bond product;
//! 6. evaluator batch size, cold and warm.
//!
//! Plus a working-memory case: the same arbitrary-degree fixture under a
//! generous and a tight `max_working_bytes`, and a budget below the prepared
//! minimum, which must be refused rather than silently exceeded.
//!
//! # Terminology
//!
//! `z` here is always the **tree coordination number** of a node: the number
//! of incident bonds. The candidate-frame kernels are selected by the number
//! of *incoming components* of a directed edge, which is `z - 1` because an
//! outward arc excludes its own target. So `z = 3` (a Y junction or a comb
//! branch point) has two incoming components and uses the exactly-two-incoming
//! kernel, and the arbitrary-degree route added for #713 is first reached at
//! `z = 4`. Every case below states both numbers.
//!
//! # Correctness before timing
//!
//! Each case materializes the interpolated tree once, compares it against one
//! dense elementwise reference, and prints the residual, sweeps, maximum
//! output rank, evaluated points, and retained cache bytes before its timed
//! loop starts. A case whose residual is not within its bound aborts the
//! benchmark: a timing number from a wrong result is not evidence.
//!
//! # Running
//!
//! ```text
//! taskset -c 0 cargo bench -p tensor4all-treeaci --bench aci_scaling -- --noplot
//! ```
//!
//! Set `T4A_ACI_SCALING_BASELINE` / `T4A_ACI_SCALING_CANDIDATE` to the two
//! commits being compared so the printed provenance header identifies the
//! pair.

use std::time::Duration;

use criterion::{black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use tensor4all_core::{ColMajorArrayRef, DynIndex, IdxTensor};
use tensor4all_treeaci::{
    tree_elementwise_batched, TreeAciDiagnostics, TreeAciError, TreeAciOptions, TreeAciResult,
    TreeElementwiseBatch,
};
use tensor4all_treetn::{CachedEvaluatorOptions, EvaluationHint, TreeTN, TreeTNCachedEvaluator};

/// Physical dimension of every site in every fixture.
const LOCAL_DIM: usize = 2;
/// Number of inputs the elementwise operator consumes.
const N_INPUTS: usize = 2;
/// Interpolation tolerance shared by every timed case.
const TOLERANCE: f64 = 1.0e-8;
/// Sweep bounds shared by every timed case.
const MAX_SWEEPS: usize = 12;
const MIN_SWEEPS: usize = 2;
/// Criterion samples per timed case.
const SAMPLE_SIZE: usize = 10;
/// Criterion warm-up and measurement windows, in seconds.
const WARM_UP_SECONDS: f64 = 0.5;
const MEASUREMENT_SECONDS: f64 = 2.0;
/// Residual bound applied to every case, relative to the reference scale.
///
/// Ten times the interpolation tolerance, matching the chain parity
/// benchmark's factor; it is a correctness gate on the fixture, not a
/// relaxation of any test tolerance.
const RESIDUAL_FACTOR: f64 = 10.0;
/// Deterministic fixture seed. The cores are an analytic function of
/// `(input, node, physical, bond coordinates)`, so there is no RNG state and
/// this constant is the only source of run-to-run variation, which is none.
const FIXTURE_SEED: u64 = 0x7718;

// ---------------------------------------------------------------------------
// Reproducibility metadata
// ---------------------------------------------------------------------------

fn environment_value(key: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| "unset".to_owned())
}

fn cpu_model() -> String {
    std::fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|info| {
            info.lines()
                .find(|line| line.starts_with("model name"))
                .and_then(|line| line.split_once(':'))
                .map(|(_, value)| value.trim().to_owned())
        })
        .unwrap_or_else(|| "unknown".to_owned())
}

fn cpu_affinity() -> String {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|status| {
            status
                .lines()
                .find(|line| line.starts_with("Cpus_allowed_list"))
                .and_then(|line| line.split_once(':'))
                .map(|(_, value)| value.trim().to_owned())
        })
        .unwrap_or_else(|| "unknown".to_owned())
}

/// Prints every field #718 requires a benchmark report to carry.
///
/// The commits are supplied by the runbook rather than inferred, because a
/// benchmark binary cannot know which pair of revisions the operator intends
/// to compare.
fn print_provenance_header() {
    println!("=== #718 TreeACI scaling gate: reproducibility metadata ===");
    println!(
        "baseline_commit={} candidate_commit={}",
        environment_value("T4A_ACI_SCALING_BASELINE"),
        environment_value("T4A_ACI_SCALING_CANDIDATE")
    );
    println!(
        "hardware={} | logical_cpus={} | cpu_affinity={}",
        cpu_model(),
        std::thread::available_parallelism()
            .map(|count| count.get().to_string())
            .unwrap_or_else(|_| "unknown".to_owned()),
        cpu_affinity()
    );
    println!(
        "provider/threads: RAYON_NUM_THREADS={} OMP_NUM_THREADS={} \
         TENFERRO_NUM_THREADS={} T4A_TREEACI_DISABLE_CANDIDATE_CACHE={}",
        environment_value("RAYON_NUM_THREADS"),
        environment_value("OMP_NUM_THREADS"),
        environment_value("TENFERRO_NUM_THREADS"),
        environment_value("T4A_TREEACI_DISABLE_CANDIDATE_CACHE")
    );
    println!(
        "build_profile={} | debug_assertions={} | fixture_seed=0x{FIXTURE_SEED:x} \
         | tolerance={TOLERANCE:e} | min_sweeps={MIN_SWEEPS} max_sweeps={MAX_SWEEPS}",
        if cfg!(debug_assertions) {
            "debug-or-unoptimized"
        } else {
            "release/bench"
        },
        cfg!(debug_assertions)
    );
    println!(
        "statistic=Criterion median with a 95% bootstrap confidence interval | \
         warm_up={WARM_UP_SECONDS}s | measurement_window={MEASUREMENT_SECONDS}s | \
         samples_per_case={SAMPLE_SIZE}"
    );
    println!(
        "noise gate: the run-to-run spread of per-run medians over at least three \
         repetitions of the whole binary; predeclared MDE for every claim is \
         \"above that observed spread\", or an exact counter delta. Counters, not \
         wall clock, carry every exponent claim (see the crate's complexity tests)."
    );
    println!(
        "residual bound per case = {RESIDUAL_FACTOR} x max(reference tolerance, {TOLERANCE:e})"
    );
    println!("===========================================================");
}

// ---------------------------------------------------------------------------
// Deterministic fixtures
// ---------------------------------------------------------------------------

/// Analytic, well-conditioned core entries with genuine bond entanglement.
///
/// Deterministic in every argument, so two runs of this benchmark build
/// bit-identical inputs and any timing difference is machine state, never
/// fixture state. The pairwise `coord_a * coord_b` term is what makes the
/// core non-separable across its bonds: without it the interpolated output
/// collapses to a near-rank-one tree and the fixture would measure almost no
/// candidate work at any requested bond dimension.
fn core_value(input: usize, node: usize, flat: usize, dims: &[usize]) -> f64 {
    let seed = (FIXTURE_SEED % 997) as f64 / 997.0;
    let mut phase = 0.173 * (input as f64 + 1.0) * (node as f64 + 1.3) + 0.31 * seed;
    let mut coordinates = Vec::with_capacity(dims.len());
    let mut rest = flat;
    for &dim in dims {
        // Raw, not normalized: the phase must vary fast across a bond for the
        // fixture to have a genuine rank. Dividing by `dim` here makes every
        // core a smooth function of its bond coordinates, the product of two
        // such trees stays near rank two, and the rank sweep degenerates.
        coordinates.push((rest % dim) as f64);
        rest /= dim;
    }
    for (axis, &coordinate) in coordinates.iter().enumerate() {
        phase += (0.191 + 0.037 * axis as f64) * coordinate * (axis as f64 + 1.7);
    }
    let mut cross = 0.0;
    for (axis, &left) in coordinates.iter().enumerate() {
        for (other, &right) in coordinates.iter().enumerate().skip(axis + 1) {
            cross += 0.071 * left * right * (axis + other + 2) as f64;
        }
    }
    let value = 0.31
        + 0.29 * phase.sin()
        + 0.23 * (2.7 * cross + 0.11 * phase).cos()
        + 0.17 * (cross - phase).sin();
    let bond_product: usize = dims.iter().product();
    value / (bond_product as f64).powf(0.25)
}

fn dense_tensor(indices: Vec<DynIndex>, input: usize, node: usize) -> IdxTensor {
    let dims: Vec<usize> = indices
        .iter()
        .map(tensor4all_core::IndexLike::dim)
        .collect();
    let size: usize = dims.iter().product();
    IdxTensor::from_dense(
        indices,
        (0..size)
            .map(|flat| core_value(input, node, flat, &dims))
            .collect::<Vec<f64>>(),
    )
    .expect("fixture tensor")
}

/// Rescales one node so the whole tree's dense maximum modulus is exactly 1.
///
/// Every case then shares the same reference scale, so one residual bound is
/// comparable across chain lengths, bond dimensions, and topologies, and no
/// case can pass merely because its values underflowed towards the tolerance.
fn normalize(tree: TreeTN<IdxTensor, usize>, first_node: usize) -> TreeTN<IdxTensor, usize> {
    let scale = tree
        .to_dense()
        .expect("dense materialization")
        .maxabs()
        .expect("dense scale");
    assert!(
        scale.is_finite() && scale > 0.0,
        "fixture has a degenerate dense scale {scale:e}"
    );
    let names: Vec<usize> = (0..tree.node_count()).collect();
    let tensors = names
        .iter()
        .map(|name| {
            let index = tree.node_index(name).expect("fixture node index");
            let tensor = tree.tensor(index).expect("fixture node tensor");
            if *name == first_node {
                let indices = tensor.indices().to_vec();
                let values = tensor
                    .to_vec::<f64>()
                    .expect("fixture node values")
                    .into_iter()
                    .map(|value| value / scale)
                    .collect::<Vec<f64>>();
                IdxTensor::from_dense(indices, values).expect("rescaled fixture node")
            } else {
                tensor.clone()
            }
        })
        .collect::<Vec<_>>();
    TreeTN::from_tensors(tensors, names).expect("normalized fixture")
}

/// A chain of `n_sites` nodes with a single fixed bond dimension.
///
/// Bond dimensions are capped by the exact algebraic rank of their cut, so a
/// large requested `chi` never inflates a boundary bond past what the chain
/// can represent.
fn chain_tree(
    n_sites: usize,
    chi: usize,
    input: usize,
    sites: &[DynIndex],
) -> TreeTN<IdxTensor, usize> {
    let links: Vec<DynIndex> = (0..n_sites - 1)
        .map(|bond| {
            let left = bond + 1;
            let right = n_sites - left;
            let exact = LOCAL_DIM.saturating_pow(left.min(right) as u32);
            DynIndex::new_dyn(chi.min(exact).max(1))
        })
        .collect();
    let tensors = (0..n_sites)
        .map(|node| {
            let mut indices = vec![sites[node].clone()];
            if node > 0 {
                indices.push(links[node - 1].clone());
            }
            if node + 1 < n_sites {
                indices.push(links[node].clone());
            }
            dense_tensor(indices, input, node)
        })
        .collect::<Vec<_>>();
    normalize(
        TreeTN::from_tensors(tensors, (0..n_sites).collect()).expect("chain fixture"),
        0,
    )
}

/// A spider: one hub of tree coordination number `arm_bonds.len()`, each arm a
/// chain of `arm_length` sites.
///
/// A one-site arm would bound every hub cut at the leaf's algebraic rank
/// `LOCAL_DIM`, so a plain star can never produce an output rank above 2 no
/// matter how large its bonds are and would measure almost no candidate work.
/// Multi-site arms give each hub cut algebraic room, which is what makes the
/// coordination sweep a measurement rather than a formality.
///
/// The hub carries `arm_bonds.len() - 1` incoming components on each outward
/// directed edge, so `arm_bonds.len() >= 4` is what reaches the #713
/// arbitrary-degree candidate-frame route. Node `0` is the hub; arm `a`'s
/// sites are `1 + a * arm_length ..`, ordered outwards.
fn spider_tree(
    arm_bonds: &[usize],
    arm_length: usize,
    chi: usize,
    input: usize,
    sites: &[DynIndex],
) -> TreeTN<IdxTensor, usize> {
    let arms = arm_bonds.len();
    let n_sites = 1 + arms * arm_length;
    assert!(arm_length >= 1);
    assert_eq!(sites.len(), n_sites);

    // Cap every bond at the exact algebraic rank of its own cut, so a request
    // that the topology cannot represent is reported as the dimension the
    // fixture actually built rather than silently padded.
    let exact_cut_rank = |outer_sites: usize| {
        let inner = n_sites - outer_sites;
        LOCAL_DIM
            .saturating_pow(outer_sites.min(inner) as u32)
            .max(1)
    };
    let hub_bonds: Vec<DynIndex> = arm_bonds
        .iter()
        .map(|&dim| DynIndex::new_dyn(dim.min(exact_cut_rank(arm_length)).max(1)))
        .collect();
    let arm_bond_indices: Vec<Vec<DynIndex>> = (0..arms)
        .map(|_| {
            (1..arm_length)
                .map(|position| {
                    let outer = arm_length - position;
                    DynIndex::new_dyn(chi.min(exact_cut_rank(outer)).max(1))
                })
                .collect()
        })
        .collect();

    let mut hub_indices = vec![sites[0].clone()];
    hub_indices.extend(hub_bonds.iter().cloned());
    let mut tensors = vec![dense_tensor(hub_indices, input, 0)];
    for arm in 0..arms {
        for position in 0..arm_length {
            let node = 1 + arm * arm_length + position;
            let mut indices = vec![sites[node].clone()];
            if position == 0 {
                indices.push(hub_bonds[arm].clone());
            } else {
                indices.push(arm_bond_indices[arm][position - 1].clone());
            }
            if position + 1 < arm_length {
                indices.push(arm_bond_indices[arm][position].clone());
            }
            tensors.push(dense_tensor(indices, input, node));
        }
    }
    normalize(
        TreeTN::from_tensors(tensors, (0..n_sites).collect()).expect("spider fixture"),
        0,
    )
}

fn site_indices(count: usize) -> Vec<DynIndex> {
    (0..count).map(|_| DynIndex::new_dyn(LOCAL_DIM)).collect()
}

// ---------------------------------------------------------------------------
// Operator, reference, and reporting
// ---------------------------------------------------------------------------

fn multiply(
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

fn dense_in_site_order(tree: &TreeTN<IdxTensor, usize>, sites: &[DynIndex]) -> Vec<f64> {
    tree.to_dense()
        .expect("dense materialization")
        .permute_indices(sites)
        .expect("canonical site order")
        .to_vec::<f64>()
        .expect("dense values")
}

/// The exact elementwise product of the inputs, in canonical site order.
fn dense_reference(inputs: &[TreeTN<IdxTensor, usize>], sites: &[DynIndex]) -> Vec<f64> {
    let mut product = vec![1.0; LOCAL_DIM.pow(sites.len() as u32)];
    for input in inputs {
        for (slot, value) in product.iter_mut().zip(dense_in_site_order(input, sites)) {
            *slot *= value;
        }
    }
    product
}

fn default_options() -> TreeAciOptions<usize> {
    TreeAciOptions {
        tolerance: TOLERANCE,
        max_sweeps: MAX_SWEEPS,
        min_sweeps: MIN_SWEEPS,
        ..TreeAciOptions::default()
    }
}

fn report(case: &str, result: &TreeAciResult<usize>, residual: f64, scale: f64) {
    let diagnostics: &TreeAciDiagnostics<usize> = &result.diagnostics;
    let max_rank = diagnostics
        .edge_ranks
        .iter()
        .map(|(_, _, rank)| *rank)
        .max()
        .unwrap_or(0);
    let candidate_total: usize = diagnostics
        .candidate_set_sizes
        .iter()
        .map(|(_, _, len)| *len)
        .sum();
    println!(
        "case={case:<34} sweeps={:<2} max_output_rank={:<4} last_pivot_error={:.3e} \
         residual={residual:.3e} scale={scale:.3e} evaluated_points={:<9} \
         global_pivots={:?} termination={:?}",
        result.max_ranks.len(),
        max_rank,
        result.max_errors.last().copied().unwrap_or(f64::NAN),
        diagnostics.evaluated_points,
        result.global_pivots_found,
        result.termination,
    );
    println!(
        "     retained: frame_records={} frame_bytes={} sample_records={} sample_bytes={} \
         candidate_entries={} saturated_edges={}",
        diagnostics.frame_records,
        diagnostics.frame_retained_bytes,
        diagnostics.sample_arena_records,
        diagnostics.sample_arena_retained_bytes,
        candidate_total,
        diagnostics.saturated_edges.len(),
    );
}

/// Runs one case once outside the timing loop, checks the whole result against
/// the dense reference, and prints its diagnostics.
///
/// # Panics
///
/// Panics when the residual exceeds the case bound, which aborts the benchmark
/// rather than reporting a time for a wrong answer.
fn check_and_report(
    case: &str,
    inputs: &[TreeTN<IdxTensor, usize>],
    sites: &[DynIndex],
    options: &TreeAciOptions<usize>,
) -> TreeAciResult<usize> {
    let reference = dense_reference(inputs, sites);
    let scale = reference
        .iter()
        .copied()
        .fold(0.0f64, |a, b| a.max(b.abs()));
    let result =
        tree_elementwise_batched::<f64, _, _>(multiply, inputs, options).expect("tree ACI run");
    let actual = dense_in_site_order(&result.tree, sites);
    let residual = actual
        .iter()
        .zip(&reference)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0f64, f64::max);
    report(case, &result, residual, scale);
    assert!(
        scale > 1.0e-6,
        "{case}: degenerate fixture, reference scale {scale:.3e}"
    );
    assert!(
        residual <= RESIDUAL_FACTOR * TOLERANCE * scale,
        "{case}: residual {residual:.3e} exceeds {:.3e}",
        RESIDUAL_FACTOR * TOLERANCE * scale
    );
    result
}

fn configure(group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>) {
    group.sample_size(SAMPLE_SIZE);
    group.warm_up_time(Duration::from_secs_f64(WARM_UP_SECONDS));
    group.measurement_time(Duration::from_secs_f64(MEASUREMENT_SECONDS));
}

// ---------------------------------------------------------------------------
// Dimension 1: chain length, at fixed bond dimension and fixed rank cap
// ---------------------------------------------------------------------------

fn bench_chain_length(c: &mut Criterion) {
    const CHI: usize = 16;
    const LENGTHS: [usize; 3] = [4, 8, 16];

    let mut group = c.benchmark_group("treeaci_scale_chain_length");
    configure(&mut group);
    for n_sites in LENGTHS {
        let sites = site_indices(n_sites);
        let inputs: Vec<_> = (0..N_INPUTS)
            .map(|input| chain_tree(n_sites, CHI, input, &sites))
            .collect();
        let options = default_options();
        check_and_report(
            &format!("chain_length/N={n_sites}"),
            &inputs,
            &sites,
            &options,
        );
        group.bench_with_input(
            BenchmarkId::from_parameter(n_sites),
            &inputs,
            |b, inputs| {
                b.iter(|| {
                    tree_elementwise_batched::<f64, _, _>(multiply, black_box(inputs), &options)
                        .expect("tree ACI run")
                })
            },
        );
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Dimension 2: input bond dimension, at fixed chain length and rank cap
// ---------------------------------------------------------------------------

fn bench_input_bond(c: &mut Criterion) {
    const N_SITES: usize = 12;
    const BONDS: [usize; 3] = [8, 16, 32];

    let mut group = c.benchmark_group("treeaci_scale_input_bond");
    configure(&mut group);
    for chi in BONDS {
        let sites = site_indices(N_SITES);
        let inputs: Vec<_> = (0..N_INPUTS)
            .map(|input| chain_tree(N_SITES, chi, input, &sites))
            .collect();
        let options = default_options();
        check_and_report(&format!("input_bond/chi={chi}"), &inputs, &sites, &options);
        group.bench_with_input(BenchmarkId::from_parameter(chi), &inputs, |b, inputs| {
            b.iter(|| {
                tree_elementwise_batched::<f64, _, _>(multiply, black_box(inputs), &options)
                    .expect("tree ACI run")
            })
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Dimension 3: active candidate/output rank, at fixed length and input bond
// ---------------------------------------------------------------------------

fn bench_output_rank(c: &mut Criterion) {
    const N_SITES: usize = 12;
    const CHI: usize = 16;
    const RANK_CAPS: [usize; 3] = [4, 8, 16];

    let mut group = c.benchmark_group("treeaci_scale_output_rank");
    configure(&mut group);
    for cap in RANK_CAPS {
        let sites = site_indices(N_SITES);
        let inputs: Vec<_> = (0..N_INPUTS)
            .map(|input| chain_tree(N_SITES, CHI, input, &sites))
            .collect();
        // A rank cap below the exact product rank stops the run at
        // `RankLimited` with a finite residual, so this case reports the
        // truncated residual instead of asserting the exact bound.
        let options = TreeAciOptions {
            max_bond_dim: Some(cap),
            ..default_options()
        };
        let reference = dense_reference(&inputs, &sites);
        let scale = reference
            .iter()
            .copied()
            .fold(0.0f64, |a, b| a.max(b.abs()));
        let result = tree_elementwise_batched::<f64, _, _>(multiply, &inputs, &options)
            .expect("tree ACI run");
        let actual = dense_in_site_order(&result.tree, &sites);
        let residual = actual
            .iter()
            .zip(&reference)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0f64, f64::max);
        report(&format!("output_rank/cap={cap}"), &result, residual, scale);
        let observed_rank = result
            .diagnostics
            .edge_ranks
            .iter()
            .map(|(_, _, rank)| *rank)
            .max()
            .unwrap_or(0);
        assert!(
            observed_rank <= cap,
            "output_rank/cap={cap}: observed rank {observed_rank} exceeds the cap"
        );
        assert!(
            residual.is_finite(),
            "output_rank/cap={cap}: non-finite truncated residual"
        );
        group.bench_with_input(BenchmarkId::from_parameter(cap), &inputs, |b, inputs| {
            b.iter(|| {
                tree_elementwise_batched::<f64, _, _>(multiply, black_box(inputs), &options)
                    .expect("tree ACI run")
            })
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Dimension 4: coordination number
// ---------------------------------------------------------------------------

/// Coordination `z` means `z` incident bonds, hence `z - 1` incoming
/// components per outward directed edge:
///
/// | `z` | incoming | candidate-frame route |
/// |-----|----------|-----------------------|
/// | 2   | 1        | single-incoming kernel (this spider is a plain chain) |
/// | 3   | 2        | exactly-two-incoming kernel (Y junction, comb branch point) |
/// | 4   | 3        | #713 arbitrary-degree kernel |
/// | 6   | 5        | #713 arbitrary-degree kernel |
///
/// The site count is held at 13 in all four cases, so the only thing that
/// varies is how those sites are arranged around the hub. A sweep that let the
/// node count grow with `z` would confound coordination with system size.
fn bench_coordination_number(c: &mut Criterion) {
    const HUB_BOND: usize = 4;
    const ARM_CHI: usize = 4;
    const N_SITES: usize = 13;
    // (coordination, arm length): arms * arm_length + 1 == N_SITES in each.
    const LAYOUTS: [(usize, usize); 4] = [(2, 6), (3, 4), (4, 3), (6, 2)];

    let mut group = c.benchmark_group("treeaci_scale_coordination");
    configure(&mut group);
    for (z, arm_length) in LAYOUTS {
        assert_eq!(z * arm_length + 1, N_SITES);
        let arm_bonds = vec![HUB_BOND; z];
        let sites = site_indices(N_SITES);
        let inputs: Vec<_> = (0..N_INPUTS)
            .map(|input| spider_tree(&arm_bonds, arm_length, ARM_CHI, input, &sites))
            .collect();
        let options = default_options();
        let case = format!("coordination/z={z}_incoming={}_arm_len={arm_length}", z - 1);
        check_and_report(&case, &inputs, &sites, &options);
        group.bench_with_input(BenchmarkId::from_parameter(z), &inputs, |b, inputs| {
            b.iter(|| {
                tree_elementwise_batched::<f64, _, _>(multiply, black_box(inputs), &options)
                    .expect("tree ACI run")
            })
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Dimension 5: unequal incident bonds at a fixed bond product
// ---------------------------------------------------------------------------

/// The topology-required work of a hub is `d * product(incident bonds)`, so
/// two layouts with the same bond product must cost the same if the
/// implementation really uses the actual dimensions rather than `max(chi)^z`.
///
/// All three layouts are coordination 4 (three incoming components per
/// outward edge, i.e. the #713 route) on the same 13-site spider, with hub
/// bond product 256 and every bond inside the arms' algebraic cut rank.
fn bench_unequal_incident_bonds(c: &mut Criterion) {
    const ARM_LENGTH: usize = 3;
    const ARM_CHI: usize = 4;
    const N_SITES: usize = 13;
    let layouts: [(&str, [usize; 4]); 3] = [
        ("equal_4x4x4x4", [4, 4, 4, 4]),
        ("unequal_2x4x8x4", [2, 4, 8, 4]),
        ("unequal_8x4x2x4", [8, 4, 2, 4]),
    ];

    let mut group = c.benchmark_group("treeaci_scale_unequal_bonds");
    configure(&mut group);
    for (name, bonds) in layouts {
        assert_eq!(bonds.iter().product::<usize>(), 256);
        assert_eq!(bonds.len() * ARM_LENGTH + 1, N_SITES);
        let sites = site_indices(N_SITES);
        let inputs: Vec<_> = (0..N_INPUTS)
            .map(|input| spider_tree(&bonds, ARM_LENGTH, ARM_CHI, input, &sites))
            .collect();
        let options = default_options();
        check_and_report(&format!("unequal_bonds/{name}"), &inputs, &sites, &options);
        group.bench_with_input(BenchmarkId::from_parameter(name), &inputs, |b, inputs| {
            b.iter(|| {
                tree_elementwise_batched::<f64, _, _>(multiply, black_box(inputs), &options)
                    .expect("tree ACI run")
            })
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Dimension 6: evaluator batch size, cold and warm
// ---------------------------------------------------------------------------

/// Points that all vary around the centre site, which is the shape the Guard
/// and the local update actually request.
fn scan_points(n_sites: usize, center: usize, n_points: usize) -> Vec<usize> {
    let mut values = Vec::with_capacity(n_sites * n_points);
    for point in 0..n_points {
        for site in 0..n_sites {
            values.push(if site == center {
                point % LOCAL_DIM
            } else {
                (site + point / LOCAL_DIM) % LOCAL_DIM
            });
        }
    }
    values
}

fn bench_evaluator_batch_size(c: &mut Criterion) {
    const N_SITES: usize = 16;
    const CHI: usize = 32;
    const CENTER: usize = 8;
    const BATCHES: [usize; 3] = [4, 16, 64];

    let sites = site_indices(N_SITES);
    let tree = chain_tree(N_SITES, CHI, 0, &sites);

    let mut group = c.benchmark_group("treeaci_scale_evaluator_batch");
    configure(&mut group);
    for n_points in BATCHES {
        let values = scan_points(N_SITES, CENTER, n_points);
        let shape = [N_SITES, n_points];
        let points = ColMajorArrayRef::new(&values, &shape).expect("point batch");
        let expected = tree.evaluate(&sites, points).expect("ordinary evaluation");

        // One cold and one warm correctness check per size, before timing.
        let mut evaluator = TreeTNCachedEvaluator::new(
            &tree,
            &sites,
            CachedEvaluatorOptions {
                center: Some(CENTER),
                ..Default::default()
            },
        )
        .expect("evaluator");
        for pass in ["cold", "warm"] {
            let observed = evaluator
                .evaluate_batched_typed::<f64>(points, EvaluationHint::around(CENTER))
                .expect("typed batch evaluation");
            let residual = observed
                .iter()
                .zip(&expected)
                .map(|(observed, expected)| (observed - expected.real()).abs())
                .fold(0.0f64, f64::max);
            let scale = expected
                .iter()
                .map(|value| value.real().abs())
                .fold(0.0f64, f64::max);
            println!(
                "case=evaluator/{pass}/points={n_points:<3} residual={residual:.3e} scale={scale:.3e}"
            );
            assert!(
                residual <= 1.0e-10 * scale.max(1.0),
                "evaluator/{pass}/points={n_points}: residual {residual:.3e}"
            );
        }

        group.bench_with_input(
            BenchmarkId::new("cold", n_points),
            &n_points,
            |b, &n_points| {
                let values = scan_points(N_SITES, CENTER, n_points);
                let shape = [N_SITES, n_points];
                b.iter_batched(
                    || {
                        TreeTNCachedEvaluator::new(
                            &tree,
                            &sites,
                            CachedEvaluatorOptions {
                                center: Some(CENTER),
                                ..Default::default()
                            },
                        )
                        .expect("evaluator")
                    },
                    |mut evaluator| {
                        let points = ColMajorArrayRef::new(&values, &shape).expect("points");
                        evaluator
                            .evaluate_batched_typed::<f64>(points, EvaluationHint::around(CENTER))
                            .expect("typed batch evaluation")
                    },
                    BatchSize::SmallInput,
                )
            },
        );
        group.bench_with_input(
            BenchmarkId::new("warm", n_points),
            &n_points,
            |b, &n_points| {
                let values = scan_points(N_SITES, CENTER, n_points);
                let shape = [N_SITES, n_points];
                let mut evaluator = TreeTNCachedEvaluator::new(
                    &tree,
                    &sites,
                    CachedEvaluatorOptions {
                        center: Some(CENTER),
                        ..Default::default()
                    },
                )
                .expect("evaluator");
                let points = ColMajorArrayRef::new(&values, &shape).expect("points");
                evaluator
                    .evaluate_batched_typed::<f64>(points, EvaluationHint::around(CENTER))
                    .expect("warm-up call");
                b.iter(|| {
                    let points = ColMajorArrayRef::new(&values, &shape).expect("points");
                    evaluator
                        .evaluate_batched_typed::<f64>(points, EvaluationHint::around(CENTER))
                        .expect("typed batch evaluation")
                })
            },
        );
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Working-memory limit: generous budget, tight budget, refused budget
// ---------------------------------------------------------------------------

/// `max_working_bytes` is a hard limit, not a hint. Three points are covered:
///
/// * the default generous budget;
/// * the smallest budget on a descending power-of-two ladder at which this
///   arbitrary-degree fixture still completes, which must produce the same
///   ranks and the same dense residual as the generous run -- a smaller
///   budget may change which candidate route is taken, but it may not change
///   the answer;
/// * the next rung down, plus a one-byte budget, both of which must be
///   refused with `TreeAciError::ResourceLimit` rather than silently
///   exceeded.
///
/// The tight rung is found by measurement rather than hard-coded, because the
/// charge depends on the candidate counts the run itself discovers.
fn bench_working_memory_budget(c: &mut Criterion) {
    // Coordination 4: three incoming components, so the #713 route applies.
    const BONDS: [usize; 4] = [8, 8, 8, 8];
    const ARM_LENGTH: usize = 3;
    const ARM_CHI: usize = 4;
    /// Descending working-byte budgets, in bytes.
    const LADDER: [usize; 7] = [
        512 << 20,
        16 << 20,
        4 << 20,
        1 << 20,
        512 << 10,
        256 << 10,
        64 << 10,
    ];

    let sites = site_indices(BONDS.len() * ARM_LENGTH + 1);
    let inputs: Vec<_> = (0..N_INPUTS)
        .map(|input| spider_tree(&BONDS, ARM_LENGTH, ARM_CHI, input, &sites))
        .collect();

    let generous = default_options();
    let generous_result = check_and_report("working_budget/generous", &inputs, &sites, &generous);
    let generous_rank = generous_result
        .diagnostics
        .edge_ranks
        .iter()
        .map(|(_, _, rank)| *rank)
        .max();

    let mut feasible: Option<usize> = None;
    for budget in LADDER {
        let options = TreeAciOptions {
            max_working_bytes: budget,
            ..default_options()
        };
        match tree_elementwise_batched::<f64, _, _>(multiply, &inputs, &options) {
            Ok(result) => {
                let rank = result
                    .diagnostics
                    .edge_ranks
                    .iter()
                    .map(|(_, _, rank)| *rank)
                    .max();
                assert_eq!(
                    rank, generous_rank,
                    "max_working_bytes={budget} changed the interpolated rank"
                );
                feasible = Some(budget);
                println!("case=working_budget/ladder            max_working_bytes={budget} status=ok max_rank={rank:?}");
            }
            Err(TreeAciError::ResourceLimit {
                resource,
                requested,
                limit,
            }) => {
                println!(
                    "case=working_budget/ladder            max_working_bytes={budget}                      status=refused resource={resource} requested={requested} limit={limit}"
                );
                break;
            }
            Err(other) => panic!("unexpected working-budget error: {other:?}"),
        }
    }
    let tight = feasible.expect("at least the generous rung must be feasible");
    let tight_options = TreeAciOptions {
        max_working_bytes: tight,
        ..default_options()
    };
    check_and_report(
        &format!("working_budget/tight_{tight}B"),
        &inputs,
        &sites,
        &tight_options,
    );

    // Below the prepared minimum local matrix: refused, not exceeded.
    let refused = TreeAciOptions {
        max_working_bytes: 1,
        ..default_options()
    };
    match tree_elementwise_batched::<f64, _, _>(multiply, &inputs, &refused) {
        Err(TreeAciError::ResourceLimit {
            resource,
            requested,
            limit,
        }) => println!(
            "case=working_budget/refused           resource={resource} requested={requested} limit={limit}"
        ),
        other => panic!("a 1-byte working budget must be refused, got {other:?}"),
    }

    let mut group = c.benchmark_group("treeaci_working_budget");
    configure(&mut group);
    group.bench_with_input(
        BenchmarkId::from_parameter("generous_512MiB"),
        &inputs,
        |b, inputs| {
            b.iter(|| {
                tree_elementwise_batched::<f64, _, _>(multiply, black_box(inputs), &generous)
                    .expect("tree ACI run")
            })
        },
    );
    group.bench_with_input(
        BenchmarkId::from_parameter(format!("tight_{tight}B")),
        &inputs,
        |b, inputs| {
            b.iter(|| {
                tree_elementwise_batched::<f64, _, _>(multiply, black_box(inputs), &tight_options)
                    .expect("tree ACI run")
            })
        },
    );
    group.finish();
}

fn bench_all(c: &mut Criterion) {
    print_provenance_header();
    bench_chain_length(c);
    bench_input_bond(c);
    bench_output_rank(c);
    bench_coordination_number(c);
    bench_unequal_incident_bonds(c);
    bench_evaluator_batch_size(c);
    bench_working_memory_budget(c);
}

criterion_group!(benches, bench_all);
criterion_main!(benches);
