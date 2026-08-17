//! Paired comparison of `tensor4all-aci` and `tensor4all-treeaci` on a chain.
//!
//! The question is whether the two implementations perform the same asymptotic
//! work per evaluation, not how fast either is: a chain is where they are
//! directly comparable, and a difference in fitted scaling exponent against bond
//! dimension would be an algorithmic defect that no amount of interface work
//! would fix. At small bond dimension almost all of the cost is a per-call
//! constant, so such a defect is invisible there; this runs at 16 through 128.
//!
//! Neither arm is given an initial guess. The registered design seeded both from
//! the same converted guess, but `tree_elementwise` rejects a guess whose rank
//! matches the inputs' with `canonicalize: factorization failed` (a singular
//! fixed-pivot solve), while `tensor4all-aci` accepts the same object. Guesses of
//! rank 1 and 2 are accepted, so the failure is rank-dependent. Running both arms
//! unseeded keeps the comparison paired; the rejection is reported separately as
//! a defect rather than worked around silently.
//!
//! It lives here rather than beside `tensor4all-treeaci` because that crate's
//! own test asserts its manifest does not mention `tensor4all-aci`. The
//! dependency direction is one-way by design, so the comparison belongs on this
//! side.

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
const CHI_VALUES: [usize; 4] = [16, 32, 64, 128];

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
/// The full mixing matters: a shortened version of this made the bond matrices
/// rank-deficient at chi = 128 and the tree arm's canonicalization failed with a
/// singular solve.
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
    }
}

fn train_options(guess: Option<SimpleTensorTrain<f64>>) -> AciOptions<f64> {
    AciOptions {
        tolerance: TOLERANCE,
        max_bond_dim: Some(MAX_BOND_DIM),
        max_iters: MAX_SWEEPS,
        min_iters: MIN_SWEEPS,
        initial_guess: guess,
        enable_global_guard: false,
        ..AciOptions::default()
    }
}

fn tree_options(guess: Option<TreeTN<IdxTensor, usize>>) -> TreeAciOptions<usize> {
    TreeAciOptions {
        tolerance: TOLERANCE,
        max_bond_dim: Some(MAX_BOND_DIM),
        max_sweeps: MAX_SWEEPS,
        min_sweeps: MIN_SWEEPS,
        initial_guess: guess,
        enable_global_guard: false,
        ..TreeAciOptions::default()
    }
}

fn bench_parity(c: &mut Criterion) {
    let mut group = c.benchmark_group("aci_vs_treeaci_chain");
    group.sample_size(10);

    for chi in CHI_VALUES {
        let case = build(chi);

        // Report rank and termination once per case, outside the timing loop, so
        // a case where an arm failed to converge is visible rather than merely
        // fast.
        let train =
            elementwise_batched(multiply_train, &case.trains, &train_options(None)).unwrap();
        let tree =
            tree_elementwise_batched::<f64, _, _>(multiply_tree, &case.trees, &tree_options(None))
                .unwrap();
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
            "         tree: evaluated_points {} | frame records {} | frame bytes {}",
            tree.diagnostics.evaluated_points,
            tree.diagnostics.frame_records,
            tree.diagnostics.frame_retained_bytes
        );

        group.bench_with_input(BenchmarkId::new("train", chi), &case, |b, case| {
            b.iter(|| {
                elementwise_batched(
                    multiply_train,
                    black_box(&case.trains),
                    &train_options(None),
                )
                .unwrap()
            })
        });
        group.bench_with_input(BenchmarkId::new("tree", chi), &case, |b, case| {
            b.iter(|| {
                tree_elementwise_batched::<f64, _, _>(
                    multiply_tree,
                    black_box(&case.trees),
                    &tree_options(None),
                )
                .unwrap()
            })
        });
    }
    group.finish();
}

criterion_group!(benches, bench_parity);
criterion_main!(benches);
