use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use tensor4all_aci::{elementwise_batched, AciOptions, AciResult, ElementwiseBatch};
use tensor4all_simplett::{tensor3_from_data, AbstractTensorTrain, SimpleTensorTrain};

const N_SITES: usize = 12;
const LOCAL_DIM: usize = 2;
const N_INPUTS: usize = 2;
const TOLERANCE: f64 = 1e-10;
const MAX_ITERS: usize = 20;
const SAMPLE_POINTS: usize = 64;
const DEFAULT_CHI_VALUES: [usize; 4] = [2, 4, 8, 16];
const OPTIONAL_CHI: usize = 32;

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

fn col_major_index3(
    left: usize,
    physical: usize,
    right: usize,
    left_dim: usize,
    site_dim: usize,
) -> usize {
    left + left_dim * (physical + site_dim * right)
}

fn deterministic_tt(
    input_index: usize,
    n_sites: usize,
    local_dim: usize,
    chi: usize,
) -> SimpleTensorTrain<f64> {
    let links = link_dims(n_sites, local_dim, chi);
    let cores = (0..n_sites)
        .map(|site| {
            let left_dim = if site == 0 { 1 } else { links[site - 1] };
            let right_dim = links.get(site).copied().unwrap_or(1);
            let mut data = vec![0.0; left_dim * local_dim * right_dim];
            for right in 0..right_dim {
                for physical in 0..local_dim {
                    for left in 0..left_dim {
                        data[col_major_index3(left, physical, right, left_dim, local_dim)] =
                            core_value(
                                input_index,
                                site,
                                physical,
                                left,
                                right,
                                left_dim,
                                right_dim,
                            );
                    }
                }
            }
            tensor3_from_data(data, left_dim, local_dim, right_dim)
        })
        .collect::<Result<Vec<_>, _>>()
        .expect("deterministic benchmark cores should have valid dimensions");
    SimpleTensorTrain::new(cores).expect("deterministic benchmark tensor train should be valid")
}

fn deterministic_inputs(chi: usize) -> Vec<SimpleTensorTrain<f64>> {
    (0..N_INPUTS)
        .map(|input_index| deterministic_tt(input_index, N_SITES, LOCAL_DIM, chi))
        .collect()
}

fn deterministic_initial_guess(chi: usize) -> SimpleTensorTrain<f64> {
    deterministic_tt(N_INPUTS, N_SITES, LOCAL_DIM, chi)
}

fn multiply_batch(
    batch: ElementwiseBatch<'_, f64>,
    output: &mut [f64],
) -> tensor4all_aci::Result<()> {
    for (point, value) in output.iter_mut().enumerate().take(batch.n_points()) {
        let mut product = 1.0;
        for input in 0..batch.n_inputs() {
            product *= batch.get(input, point)?;
        }
        *value = product;
    }
    Ok(())
}

fn sample_index(sample: usize, chi: usize) -> Vec<usize> {
    let mut state = (sample as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ (chi as u64 + 17).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    (0..N_SITES)
        .map(|site| {
            state ^= state >> 30;
            state = state.wrapping_mul(0xBF58_476D_1CE4_E5B9);
            state ^= state >> 27;
            state = state.wrapping_mul(0x94D0_49BB_1331_11EB);
            ((state ^ (site as u64)) as usize) % LOCAL_DIM
        })
        .collect()
}

fn sampled_max_abs_error(
    inputs: &[SimpleTensorTrain<f64>],
    output: &SimpleTensorTrain<f64>,
    chi: usize,
) -> f64 {
    (0..SAMPLE_POINTS)
        .map(|sample| {
            let index = sample_index(sample, chi);
            let expected = inputs
                .iter()
                .map(|input| input.evaluate(&index).expect("sample input evaluation"))
                .product::<f64>();
            let actual = output
                .evaluate(&index)
                .expect("sample output evaluation should succeed");
            (actual - expected).abs()
        })
        .fold(0.0, f64::max)
}

fn run_aci(
    inputs: &[SimpleTensorTrain<f64>],
    initial_guess: &SimpleTensorTrain<f64>,
) -> AciResult<f64> {
    let options = AciOptions {
        max_iters: MAX_ITERS,
        tolerance: TOLERANCE,
        initial_guess: Some(initial_guess.clone()),
        ..AciOptions::default()
    };
    elementwise_batched(multiply_batch, inputs, &options)
        .expect("ACI elementwise multiplication benchmark should converge")
}

fn assert_nontrivial_output_rank(chi: usize, output_max_chi: usize) {
    if chi <= 2 {
        return;
    }

    assert!(
        output_max_chi > 1,
        "deterministic chi-scaling fixture collapsed to rank one for chi={chi}"
    );
}

fn bench_aci_elementwise_chi_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("aci_elementwise_chi_scaling");
    for chi in DEFAULT_CHI_VALUES {
        let inputs = deterministic_inputs(chi);
        let initial_guess = deterministic_initial_guess(chi);
        let checked = run_aci(&inputs, &initial_guess);
        let sampled_error = sampled_max_abs_error(&inputs, &checked.tensor_train, chi);
        assert!(
            sampled_error < 1e-8,
            "sampled max abs error for chi={chi} was {sampled_error:e}"
        );

        let output_max_chi = checked.tensor_train.rank();
        assert_nontrivial_output_rank(chi, output_max_chi);
        let n_sweeps = checked.ranks.len();
        let final_error = checked.errors.last().copied().unwrap_or(0.0);
        println!(
            "aci_elementwise_metadata,impl=rust,n_sites={N_SITES},local_dim={LOCAL_DIM},\
             n_inputs={N_INPUTS},chi={chi},tolerance={TOLERANCE:.1e},\
             output_max_chi={output_max_chi},n_sweeps={n_sweeps},\
             final_error={final_error:.6e},sampled_max_abs_error={sampled_error:.6e}"
        );

        group.bench_with_input(BenchmarkId::from_parameter(chi), &chi, |b, _| {
            b.iter(|| {
                black_box(run_aci(black_box(&inputs), black_box(&initial_guess)));
            });
        });
    }
    group.finish();
}

fn command_line_requests_long_case() -> bool {
    std::env::args()
        .any(|arg| arg.contains("aci_elementwise_chi_scaling_long") || arg.contains("chi32"))
}

fn bench_aci_elementwise_chi_scaling_long(c: &mut Criterion) {
    if !command_line_requests_long_case() {
        return;
    }

    let mut group = c.benchmark_group("aci_elementwise_chi_scaling_long");
    let chi = OPTIONAL_CHI;
    let inputs = deterministic_inputs(chi);
    let initial_guess = deterministic_initial_guess(chi);
    let checked = run_aci(&inputs, &initial_guess);
    let sampled_error = sampled_max_abs_error(&inputs, &checked.tensor_train, chi);
    assert!(
        sampled_error < 1e-8,
        "sampled max abs error for chi={chi} was {sampled_error:e}"
    );

    let output_max_chi = checked.tensor_train.rank();
    assert_nontrivial_output_rank(chi, output_max_chi);
    let n_sweeps = checked.ranks.len();
    let final_error = checked.errors.last().copied().unwrap_or(0.0);
    println!(
        "aci_elementwise_metadata,impl=rust,n_sites={N_SITES},local_dim={LOCAL_DIM},\
         n_inputs={N_INPUTS},chi={chi},tolerance={TOLERANCE:.1e},\
         output_max_chi={output_max_chi},n_sweeps={n_sweeps},\
         final_error={final_error:.6e},sampled_max_abs_error={sampled_error:.6e}"
    );

    group.bench_with_input(BenchmarkId::from_parameter(chi), &chi, |b, _| {
        b.iter(|| {
            black_box(run_aci(black_box(&inputs), black_box(&initial_guess)));
        });
    });
    group.finish();
}

criterion_group!(
    aci_elementwise_chi_scaling,
    bench_aci_elementwise_chi_scaling,
    bench_aci_elementwise_chi_scaling_long
);
criterion_main!(aci_elementwise_chi_scaling);
