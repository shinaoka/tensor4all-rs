use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use tensor4all_tcicore::MultiIndex;
use tensor4all_tensorci::{crossinterpolate2, PivotSearchStrategy, TCI2Options};

fn chain_value(idx: &MultiIndex) -> f64 {
    let local = idx
        .iter()
        .enumerate()
        .map(|(site, &value)| ((site + 1) as f64) * ((value + 1) as f64))
        .sum::<f64>();
    let pairwise = idx
        .windows(2)
        .map(|window| ((window[0] * window[1] + 1) as f64) * 0.05)
        .sum::<f64>();
    local + pairwise + 1.0
}

fn chain_values(points: &[MultiIndex]) -> Vec<f64> {
    points.iter().map(chain_value).collect()
}

fn bench_chain_tci(c: &mut Criterion) {
    let mut group = c.benchmark_group("end_to_end_chain_tci");

    for &n_sites in &[4usize, 6] {
        let local_dims = vec![4; n_sites];
        let initial_pivots = vec![vec![0; n_sites]];

        let full_options = TCI2Options {
            tolerance: 1e-8,
            max_iter: 8,
            max_bond_dim: 16,
            pivot_search: PivotSearchStrategy::Full,
            ..Default::default()
        };
        group.bench_with_input(BenchmarkId::new("full", n_sites), &n_sites, |b, _| {
            b.iter(|| {
                let tensor4all_tensorci::TCI2OptimizationResult {
                    tci, ranks, errors, ..
                } = crossinterpolate2::<f64, _, fn(&[MultiIndex]) -> Vec<f64>>(
                    chain_value,
                    Some(chain_values),
                    local_dims.clone(),
                    initial_pivots.clone(),
                    full_options.clone(),
                )
                .unwrap();
                black_box((
                    tci.rank(),
                    ranks.len(),
                    errors.last().copied().unwrap_or(0.0),
                ));
            });
        });

        let rook_options = TCI2Options {
            pivot_search: PivotSearchStrategy::Rook,
            ..full_options.clone()
        };
        group.bench_with_input(BenchmarkId::new("rook", n_sites), &n_sites, |b, _| {
            b.iter(|| {
                let tensor4all_tensorci::TCI2OptimizationResult {
                    tci, ranks, errors, ..
                } = crossinterpolate2::<f64, _, fn(&[MultiIndex]) -> Vec<f64>>(
                    chain_value,
                    Some(chain_values),
                    local_dims.clone(),
                    initial_pivots.clone(),
                    rook_options.clone(),
                )
                .unwrap();
                black_box((
                    tci.rank(),
                    ranks.len(),
                    errors.last().copied().unwrap_or(0.0),
                ));
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_chain_tci);
criterion_main!(benches);
