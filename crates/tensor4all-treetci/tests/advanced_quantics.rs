mod common;

use anyhow::Result;
use common::assert_real_samples_close;
use quanticsgrids::{DiscretizedGrid, UnfoldingScheme};
use tensor4all_core::ColMajorArrayRef;
use tensor4all_treetci::{
    crossinterpolate2, GlobalIndexBatch, SimpleProposer, TreeTciEdge, TreeTciGraph, TreeTciOptions,
};

fn branching_tree(n_sites: usize) -> TreeTciGraph {
    match n_sites {
        4 => TreeTciGraph::new(
            4,
            &[
                TreeTciEdge::new(0, 1),
                TreeTciEdge::new(1, 2),
                TreeTciEdge::new(1, 3),
            ],
        )
        .unwrap(),
        _ => panic!("unexpected site count {n_sites}"),
    }
}

fn evaluate_treetn(
    tn: &tensor4all_treetn::TreeTN<tensor4all_core::TensorDynLen, usize>,
    point: &[usize],
) -> f64 {
    let (indices, _vertices) = tn.all_site_indices().unwrap();
    // Build position map: vertex -> position in indices
    let mut data = vec![0usize; indices.len()];
    for (site, &value) in point.iter().enumerate() {
        let site_index = tn.site_space(&site).unwrap().iter().next().unwrap();
        let pos = indices
            .iter()
            .position(|index| index == site_index)
            .unwrap();
        data[pos] = value;
    }
    let shape = [indices.len(), 1];
    let values = ColMajorArrayRef::new(&data, &shape).unwrap();
    tn.evaluate(&indices, values).unwrap()[0].real()
}

fn batch_eval_from_point<T: Clone>(
    point_eval: impl Fn(&[usize]) -> T,
) -> impl Fn(GlobalIndexBatch<'_>) -> Result<Vec<T>> {
    move |batch: GlobalIndexBatch<'_>| {
        let mut values = Vec::with_capacity(batch.n_points());
        let mut point = vec![0usize; batch.n_sites()];
        for p in 0..batch.n_points() {
            for (s, slot) in point.iter_mut().enumerate() {
                *slot = batch.get(s, p).unwrap();
            }
            values.push(point_eval(&point));
        }
        Ok(values)
    }
}

#[test]
fn quantics_grid_polynomial_matches_all_points_on_branching_tree() {
    let grid = DiscretizedGrid::builder(&[2, 2])
        .with_lower_bound(&[-3.0, -17.0])
        .with_upper_bound(&[2.0, 12.0])
        .include_endpoint(true)
        .with_unfolding_scheme(UnfoldingScheme::Interleaved)
        .build()
        .unwrap();

    let f = |coords: &[f64]| {
        0.1 * coords[0] * coords[0] + 0.01 * coords[1].powi(3)
            - std::f64::consts::PI * coords[0] * coords[1]
            + 5.0
    };
    let qf = |point: &[usize]| -> f64 {
        let quantics = point
            .iter()
            .map(|&value| value as i64 + 1)
            .collect::<Vec<_>>();
        let coords = grid.quantics_to_origcoord(&quantics).unwrap();
        f(&coords)
    };

    let (tn, _ranks, errors) = crossinterpolate2(
        batch_eval_from_point(qf),
        grid.local_dimensions(),
        branching_tree(grid.len()),
        vec![vec![0; grid.len()]],
        TreeTciOptions {
            tolerance: 1e-10,
            max_iter: 12,
            max_bond_dim: Some(8),
            normalize_error: true,
        },
        Some(1),
        &SimpleProposer::seeded(0),
    )
    .unwrap();

    assert!(errors.last().copied().unwrap_or(f64::INFINITY) < 1e-10);

    let mut samples = Vec::new();
    for i in 1..=4 {
        for j in 1..=4 {
            let quantics = grid.grididx_to_quantics(&[i, j]).unwrap();
            let point = quantics
                .iter()
                .map(|&value| (value - 1) as usize)
                .collect::<Vec<_>>();
            let got = evaluate_treetn(&tn, &point);
            samples.push((point, got));
        }
    }
    assert_real_samples_close(
        &samples,
        &|point| {
            let quantics = point
                .iter()
                .map(|&value| value as i64 + 1)
                .collect::<Vec<_>>();
            let coords = grid.quantics_to_origcoord(&quantics).unwrap();
            f(&coords)
        },
        1e-8,
    );
}

#[test]
fn quantics_grid_batch_evaluator_matches_point_evaluator() {
    let grid = DiscretizedGrid::builder(&[2, 2])
        .with_lower_bound(&[-3.0, -17.0])
        .with_upper_bound(&[2.0, 12.0])
        .include_endpoint(true)
        .with_unfolding_scheme(UnfoldingScheme::Interleaved)
        .build()
        .unwrap();

    let f = |coords: &[f64]| {
        0.1 * coords[0] * coords[0] + 0.01 * coords[1].powi(3)
            - std::f64::consts::PI * coords[0] * coords[1]
            + 5.0
    };
    let qf = |point: &[usize]| -> f64 {
        let quantics = point
            .iter()
            .map(|&value| value as i64 + 1)
            .collect::<Vec<_>>();
        let coords = grid.quantics_to_origcoord(&quantics).unwrap();
        f(&coords)
    };

    let options = TreeTciOptions {
        tolerance: 1e-10,
        max_iter: 12,
        max_bond_dim: Some(8),
        normalize_error: true,
    };

    // Run with point-eval-based batch closure
    let (tn_point, _, _) = crossinterpolate2(
        batch_eval_from_point(qf),
        grid.local_dimensions(),
        branching_tree(grid.len()),
        vec![vec![0; grid.len()]],
        options.clone(),
        Some(1),
        &SimpleProposer::seeded(0),
    )
    .unwrap();

    // Run with explicit batch evaluator
    let grid_clone = grid.clone();
    let batch_eval = |batch: GlobalIndexBatch<'_>| -> Result<Vec<f64>> {
        let mut values = Vec::with_capacity(batch.n_points());
        for point_idx in 0..batch.n_points() {
            let quantics = (0..batch.n_sites())
                .map(|site| batch.get(site, point_idx).unwrap() as i64 + 1)
                .collect::<Vec<_>>();
            let coords = grid_clone.quantics_to_origcoord(&quantics).unwrap();
            values.push(f(&coords));
        }
        Ok(values)
    };

    let (tn_batch, _, _) = crossinterpolate2(
        batch_eval,
        grid.local_dimensions(),
        branching_tree(grid.len()),
        vec![vec![0; grid.len()]],
        options,
        Some(1),
        &SimpleProposer::seeded(0),
    )
    .unwrap();

    let mut samples = Vec::new();
    for i in 1..=4 {
        for j in 1..=4 {
            let quantics = grid.grididx_to_quantics(&[i, j]).unwrap();
            let point = quantics
                .iter()
                .map(|&value| (value - 1) as usize)
                .collect::<Vec<_>>();
            let batch_only = evaluate_treetn(&tn_batch, &point);
            samples.push((point, batch_only));
        }
    }
    assert_real_samples_close(&samples, &|point| evaluate_treetn(&tn_point, point), 1e-10);
}
