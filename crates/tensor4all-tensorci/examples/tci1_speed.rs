use std::time::{Duration, Instant};

use tensor4all_core::MultiIndex;
use tensor4all_tensorci::{crossinterpolate1, TCI1Options};

#[derive(Debug, Clone)]
struct Case {
    name: &'static str,
    local_dims: Vec<usize>,
    first_pivot: MultiIndex,
    tolerance: f64,
    max_iter: usize,
}

fn main() {
    let repeats = parse_repeats();
    let cases = [Case {
        name: "lorentz5d_d10_tol1e-8_maxiter8",
        local_dims: vec![10; 5],
        first_pivot: vec![0; 5],
        tolerance: 1e-8,
        max_iter: 8,
    }];

    for case in &cases {
        // Match the Julia benchmark: warm up lazy backend state before timing.
        let _ = run_case(case);

        let mut durations = Vec::with_capacity(repeats);
        let mut rank = 0;
        let mut last_error = 0.0;

        for _ in 0..repeats {
            let start = Instant::now();
            let (tci, _ranks, errors) = run_case(case);
            durations.push(start.elapsed());
            rank = tci.rank();
            last_error = errors.last().copied().unwrap_or(0.0);
        }

        durations.sort_unstable();
        let median = durations[durations.len() / 2];
        let best = durations[0];
        println!(
            "impl=rust case={} repeats={} median_seconds={:.9} best_seconds={:.9} rank={} last_error={:.6e}",
            case.name,
            repeats,
            seconds(median),
            seconds(best),
            rank,
            last_error
        );
    }
}

fn parse_repeats() -> usize {
    let mut repeats = 5;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--repeats" {
            let value = args
                .next()
                .unwrap_or_else(|| panic!("--repeats requires a positive integer"));
            repeats = value
                .parse::<usize>()
                .unwrap_or_else(|_| panic!("invalid --repeats value: {value}"));
        }
    }
    assert!(repeats > 0, "--repeats must be positive");
    repeats
}

fn run_case(case: &Case) -> (tensor4all_tensorci::TensorCI1<f64>, Vec<usize>, Vec<f64>) {
    crossinterpolate1::<f64, _>(
        lorentz,
        case.local_dims.clone(),
        case.first_pivot.clone(),
        TCI1Options {
            tolerance: case.tolerance,
            max_iter: case.max_iter,
            ..TCI1Options::default()
        },
    )
    .unwrap()
}

fn lorentz(idx: &MultiIndex) -> f64 {
    let denom = idx
        .iter()
        .map(|&i| {
            let x = (i + 1) as f64;
            x * x
        })
        .sum::<f64>()
        + 1.0;
    1.0 / denom
}

fn seconds(duration: Duration) -> f64 {
    duration.as_secs_f64()
}
