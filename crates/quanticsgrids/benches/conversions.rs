//! Benchmarks for quanticsgrids conversion functions

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use quanticsgrids::{DiscretizedGrid, InherentDiscreteGrid, UnfoldingScheme};

fn bench_grididx_to_quantics(c: &mut Criterion) {
    let mut group = c.benchmark_group("grididx_to_quantics");

    for r in [4, 8, 12, 16] {
        let grid = InherentDiscreteGrid::builder(&[r])
            .build()
            .unwrap();
        let max_idx = 2i64.pow(r as u32);
        let grididx = vec![max_idx / 2];

        group.bench_with_input(BenchmarkId::new("1D_base2", r), &r, |b, _| {
            b.iter(|| grid.grididx_to_quantics(black_box(&grididx)))
        });
    }

    // 2D grid
    for r in [4, 8, 12] {
        let grid = InherentDiscreteGrid::builder(&[r, r])
            .build()
            .unwrap();
        let max_idx = 2i64.pow(r as u32);
        let grididx = vec![max_idx / 2, max_idx / 2];

        group.bench_with_input(BenchmarkId::new("2D_base2", r), &r, |b, _| {
            b.iter(|| grid.grididx_to_quantics(black_box(&grididx)))
        });
    }

    // Base 3
    let grid = InherentDiscreteGrid::builder(&[8])
        .with_base(3)
        .build()
        .unwrap();
    let grididx = vec![3i64.pow(4)];
    group.bench_function("1D_base3_R8", |b| {
        b.iter(|| grid.grididx_to_quantics(black_box(&grididx)))
    });

    group.finish();
}

fn bench_quantics_to_grididx(c: &mut Criterion) {
    let mut group = c.benchmark_group("quantics_to_grididx");

    for r in [4, 8, 12, 16] {
        let grid = InherentDiscreteGrid::builder(&[r])
            .build()
            .unwrap();
        let quantics: Vec<i64> = vec![1; r];

        group.bench_with_input(BenchmarkId::new("1D_base2", r), &r, |b, _| {
            b.iter(|| grid.quantics_to_grididx(black_box(&quantics)))
        });
    }

    // 2D grid
    for r in [4, 8, 12] {
        let grid = InherentDiscreteGrid::builder(&[r, r])
            .with_unfolding_scheme(UnfoldingScheme::Fused)
            .build()
            .unwrap();
        let num_sites = grid.len();
        let quantics: Vec<i64> = vec![1; num_sites];

        group.bench_with_input(BenchmarkId::new("2D_fused_base2", r), &r, |b, _| {
            b.iter(|| grid.quantics_to_grididx(black_box(&quantics)))
        });
    }

    group.finish();
}

fn bench_origcoord_conversions(c: &mut Criterion) {
    let mut group = c.benchmark_group("origcoord_conversions");

    // DiscretizedGrid conversions
    for r in [8, 12, 16] {
        let grid = DiscretizedGrid::builder(&[r])
            .build()
            .unwrap();
        let coord = vec![0.5];
        let grididx = vec![2i64.pow(r as u32 - 1)];

        group.bench_with_input(BenchmarkId::new("origcoord_to_grididx_1D", r), &r, |b, _| {
            b.iter(|| grid.origcoord_to_grididx(black_box(&coord)))
        });

        group.bench_with_input(BenchmarkId::new("grididx_to_origcoord_1D", r), &r, |b, _| {
            b.iter(|| grid.grididx_to_origcoord(black_box(&grididx)))
        });
    }

    // 2D
    let grid = DiscretizedGrid::builder(&[10, 10])
        .build()
        .unwrap();
    let coord = vec![0.5, 0.5];
    group.bench_function("origcoord_to_quantics_2D_R10", |b| {
        b.iter(|| grid.origcoord_to_quantics(black_box(&coord)))
    });

    group.finish();
}

fn bench_roundtrip(c: &mut Criterion) {
    let mut group = c.benchmark_group("roundtrip");

    let grid = DiscretizedGrid::builder(&[10, 10])
        .with_variable_names(&["x", "y"])
        .build()
        .unwrap();

    let coord = vec![0.5, 0.25];

    group.bench_function("origcoord_roundtrip_2D", |b| {
        b.iter(|| {
            let quantics = grid.origcoord_to_quantics(black_box(&coord)).unwrap();
            let grididx = grid.quantics_to_grididx(&quantics).unwrap();
            grid.grididx_to_origcoord(&grididx)
        })
    });

    group.finish();
}

fn bench_unfolding_schemes(c: &mut Criterion) {
    let mut group = c.benchmark_group("unfolding_schemes");

    let r = 8;

    // Fused scheme
    let grid_fused = InherentDiscreteGrid::builder(&[r, r])
        .with_unfolding_scheme(UnfoldingScheme::Fused)
        .build()
        .unwrap();
    let grididx = vec![128, 128];

    group.bench_function("fused_grididx_to_quantics", |b| {
        b.iter(|| grid_fused.grididx_to_quantics(black_box(&grididx)))
    });

    // Interleaved scheme
    let grid_interleaved = InherentDiscreteGrid::builder(&[r, r])
        .with_unfolding_scheme(UnfoldingScheme::Interleaved)
        .build()
        .unwrap();

    group.bench_function("interleaved_grididx_to_quantics", |b| {
        b.iter(|| grid_interleaved.grididx_to_quantics(black_box(&grididx)))
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_grididx_to_quantics,
    bench_quantics_to_grididx,
    bench_origcoord_conversions,
    bench_roundtrip,
    bench_unfolding_schemes,
);
criterion_main!(benches);
