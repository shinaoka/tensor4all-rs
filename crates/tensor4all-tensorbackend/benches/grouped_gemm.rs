//! Complete paired benchmark for the shared-buffer grouped-GEMM facade.
//!
//! Run on a pinned CPU with the declared ten-sample matrix:
//! `taskset -c 0 cargo bench --release -p tensor4all-tensorbackend --bench grouped_gemm`.

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use tenferro_cpu::CpuBackend;
use tensor4all_tensorbackend::{
    grouped_mat_mul_shared_with_backend, GroupedGemmJob, GroupedGemmOptions,
};

const ROWS: usize = 16;
const CONTRACTED: usize = 16;
const COLS: usize = 16;
const JOB_COUNTS: [usize; 4] = [1, 2, 8, 32];

#[derive(Clone, Copy, Debug)]
enum SharedOperand {
    Lhs,
    Rhs,
}

impl SharedOperand {
    fn label(self) -> &'static str {
        match self {
            Self::Lhs => "shared_lhs",
            Self::Rhs => "shared_rhs",
        }
    }
}

struct Fixture {
    lhs: Vec<f64>,
    rhs: Vec<f64>,
    jobs: Vec<GroupedGemmJob>,
    lhs_block: usize,
    rhs_block: usize,
    output_len: usize,
    shared: SharedOperand,
}

fn data(len: usize, seed: usize) -> Vec<f64> {
    (0..len)
        .map(|index| ((index * 17 + seed * 31 + 7) % 101) as f64 / 101.0 - 0.5)
        .collect()
}

fn fixture(job_count: usize, shared: SharedOperand) -> Fixture {
    let lhs_block = ROWS * CONTRACTED;
    let rhs_block = CONTRACTED * COLS;
    let output_block = ROWS * COLS;
    let (lhs_blocks, rhs_blocks) = match shared {
        SharedOperand::Lhs => (1, job_count),
        SharedOperand::Rhs => (job_count, 1),
    };
    let jobs = (0..job_count)
        .map(|job| {
            GroupedGemmJob::new(
                job * output_block,
                match shared {
                    SharedOperand::Lhs => 0,
                    SharedOperand::Rhs => job * lhs_block,
                },
                match shared {
                    SharedOperand::Lhs => job * rhs_block,
                    SharedOperand::Rhs => 0,
                },
                ROWS,
                CONTRACTED,
                COLS,
            )
        })
        .collect();
    Fixture {
        lhs: data(lhs_blocks * lhs_block, 1),
        rhs: data(rhs_blocks * rhs_block, 2),
        jobs,
        lhs_block,
        rhs_block,
        output_len: job_count * output_block,
        shared,
    }
}

fn run_shared(fixture: &Fixture, backend: &mut CpuBackend) -> Vec<f64> {
    let mut output = vec![0.0; fixture.output_len];
    grouped_mat_mul_shared_with_backend(
        backend,
        &fixture.lhs,
        &fixture.rhs,
        &mut output,
        &fixture.jobs,
        GroupedGemmOptions::default(),
    )
    .unwrap();
    output
}

fn run_duplicated(fixture: &Fixture, backend: &mut CpuBackend) -> Vec<f64> {
    let mut lhs = fixture.lhs.clone();
    let mut rhs = fixture.rhs.clone();
    let mut jobs = Vec::with_capacity(fixture.jobs.len());
    for (job_index, job) in fixture.jobs.iter().enumerate() {
        let (lhs_offset, rhs_offset) = match fixture.shared {
            SharedOperand::Lhs => {
                lhs.extend_from_slice(&fixture.lhs);
                (job_index * fixture.lhs_block, job.rhs_offset())
            }
            SharedOperand::Rhs => {
                rhs.extend_from_slice(&fixture.rhs);
                (job.lhs_offset(), job_index * fixture.rhs_block)
            }
        };
        jobs.push(GroupedGemmJob::new(
            job.out_offset(),
            lhs_offset,
            rhs_offset,
            job.rows(),
            job.contracted(),
            job.cols(),
        ));
    }
    let mut output = vec![0.0; fixture.output_len];
    grouped_mat_mul_shared_with_backend(
        backend,
        &lhs,
        &rhs,
        &mut output,
        &jobs,
        GroupedGemmOptions::default(),
    )
    .unwrap();
    output
}

fn assert_close(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert!((actual - expected).abs() < 1.0e-10);
    }
}

fn bench_shared_vs_duplicated(c: &mut Criterion) {
    let mut group = c.benchmark_group("grouped_gemm_shared_operand");
    group.sample_size(10);
    for job_count in JOB_COUNTS {
        for shared in [SharedOperand::Lhs, SharedOperand::Rhs] {
            let fixture = fixture(job_count, shared);
            let mut check_backend = CpuBackend::with_threads(1).unwrap();
            let shared_result = run_shared(&fixture, &mut check_backend);
            let duplicated_result = run_duplicated(&fixture, &mut check_backend);
            assert_close(&shared_result, &duplicated_result);
            let copied_elements = match shared {
                SharedOperand::Lhs => fixture.lhs.len() * job_count,
                SharedOperand::Rhs => fixture.rhs.len() * job_count,
            };
            eprintln!(
                "case={} jobs={} duplicated_input_bytes={} oracle=pass",
                shared.label(),
                job_count,
                copied_elements * std::mem::size_of::<f64>()
            );

            group.bench_with_input(
                BenchmarkId::new(format!("{}/shared", shared.label()), job_count),
                &fixture,
                |b, fixture| {
                    let mut backend = CpuBackend::with_threads(1).unwrap();
                    b.iter(|| black_box(run_shared(black_box(fixture), &mut backend)));
                },
            );
            group.bench_with_input(
                BenchmarkId::new(format!("{}/duplicated", shared.label()), job_count),
                &fixture,
                |b, fixture| {
                    let mut backend = CpuBackend::with_threads(1).unwrap();
                    b.iter(|| black_box(run_duplicated(black_box(fixture), &mut backend)));
                },
            );
        }
    }
    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default();
    targets = bench_shared_vs_duplicated
}
criterion_main!(benches);
