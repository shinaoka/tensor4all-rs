mod common;

use anyhow::Result;
use common::{assert_complex_samples_close, assert_real_samples_close};
use num_complex::Complex64;
use tensor4all_core::ColMajorArrayRef;
use tensor4all_treetci::{
    crossinterpolate2, DefaultProposer, GlobalIndexBatch, SimpleProposer, TreeTciEdge,
    TreeTciGraph, TreeTciOptions,
};

fn sample_graph() -> TreeTciGraph {
    TreeTciGraph::new(
        7,
        &[
            TreeTciEdge::new(0, 1),
            TreeTciEdge::new(1, 2),
            TreeTciEdge::new(1, 3),
            TreeTciEdge::new(3, 4),
            TreeTciEdge::new(4, 5),
            TreeTciEdge::new(4, 6),
        ],
    )
    .unwrap()
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
fn simple_tree_parity_matches_reference_points() {
    let f = |idx: &[usize]| {
        let norm_sq = idx.iter().map(|&x| (x + 1) * (x + 1)).sum::<usize>() as f64;
        1.0 / (1.0 + norm_sq)
    };

    let (tn, _ranks, _errors) = crossinterpolate2(
        batch_eval_from_point(f),
        vec![2; 7],
        sample_graph(),
        vec![vec![0; 7]],
        TreeTciOptions {
            tolerance: 1e-8,
            max_iter: 10,
            max_bond_dim: Some(5),
            normalize_error: true,
            ..Default::default()
        },
        Some(0),
        &SimpleProposer::seeded(0),
    )
    .unwrap();

    let (indices, _vertices) = tn.all_site_indices().unwrap();
    // Build position map: vertex -> position in indices
    let pos: Vec<usize> = (0..7)
        .map(|v| {
            let site_index = tn.site_space(&v).unwrap().iter().next().unwrap();
            indices
                .iter()
                .position(|index| index == site_index)
                .unwrap()
        })
        .collect();

    let eval = |point: [usize; 7]| -> f64 {
        let mut data = vec![0usize; indices.len()];
        for (v, &val) in point.iter().enumerate() {
            data[pos[v]] = val;
        }
        let shape = [indices.len(), 1];
        let values = ColMajorArrayRef::new(&data, &shape).unwrap();
        tn.evaluate(&indices, values).unwrap()[0].real()
    };

    let got = [
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1],
        [1, 1, 1, 1, 0, 0, 0],
        [0, 1, 0, 1, 0, 1, 0],
        [1, 0, 1, 0, 1, 0, 1],
        [1, 1, 1, 1, 1, 1, 1],
    ]
    .into_iter()
    .map(|point| (point, eval(point)))
    .collect::<Vec<_>>();
    assert_real_samples_close(&got, &f, 1e-8);
}

#[test]
fn simple_tree_product_function_is_exact_on_branching_tree() {
    let f = |idx: &[usize]| idx.iter().fold(1.0, |acc, &x| acc * (x as f64 + 1.0));

    let (tn, _ranks, _errors) = crossinterpolate2(
        batch_eval_from_point(f),
        vec![2; 7],
        sample_graph(),
        vec![vec![0; 7]],
        TreeTciOptions {
            tolerance: 1e-12,
            max_iter: 8,
            max_bond_dim: Some(2),
            normalize_error: true,
            ..Default::default()
        },
        Some(0),
        &DefaultProposer,
    )
    .unwrap();

    let (indices, _vertices) = tn.all_site_indices().unwrap();
    let pos: Vec<usize> = (0..7)
        .map(|v| {
            let site_index = tn.site_space(&v).unwrap().iter().next().unwrap();
            indices
                .iter()
                .position(|index| index == site_index)
                .unwrap()
        })
        .collect();

    let eval = |point: [usize; 7]| -> f64 {
        let mut data = vec![0usize; indices.len()];
        for (v, &val) in point.iter().enumerate() {
            data[pos[v]] = val;
        }
        let shape = [indices.len(), 1];
        let values = ColMajorArrayRef::new(&data, &shape).unwrap();
        tn.evaluate(&indices, values).unwrap()[0].real()
    };

    let got = [
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1],
        [1, 1, 1, 1, 0, 0, 0],
        [0, 1, 0, 1, 0, 1, 0],
        [1, 0, 1, 0, 1, 0, 1],
        [1, 1, 1, 1, 1, 1, 1],
    ]
    .into_iter()
    .map(|point| (point, eval(point)))
    .collect::<Vec<_>>();
    assert_real_samples_close(&got, &f, 1e-12);
}

#[test]
fn simple_tree_complex_product_function_is_exact_on_branching_tree() {
    let f = |idx: &[usize]| {
        idx.iter().fold(Complex64::new(1.0, 0.0), |acc, &x| {
            acc * Complex64::new(x as f64 + 1.0, 2.0 * x as f64 + 1.0)
        })
    };

    let (tn, _ranks, _errors) = crossinterpolate2(
        batch_eval_from_point(f),
        vec![2; 7],
        sample_graph(),
        vec![vec![0; 7]],
        TreeTciOptions {
            tolerance: 1e-12,
            max_iter: 8,
            max_bond_dim: Some(2),
            normalize_error: true,
            ..Default::default()
        },
        Some(0),
        &DefaultProposer,
    )
    .unwrap();

    let (indices, _vertices) = tn.all_site_indices().unwrap();
    let pos: Vec<usize> = (0..7)
        .map(|v| {
            let site_index = tn.site_space(&v).unwrap().iter().next().unwrap();
            indices
                .iter()
                .position(|index| index == site_index)
                .unwrap()
        })
        .collect();

    let eval = |point: [usize; 7]| {
        let mut data = vec![0usize; indices.len()];
        for (v, &val) in point.iter().enumerate() {
            data[pos[v]] = val;
        }
        let shape = [indices.len(), 1];
        let values = ColMajorArrayRef::new(&data, &shape).unwrap();
        tn.evaluate(&indices, values)
            .unwrap()
            .into_iter()
            .next()
            .unwrap()
    };

    let got = [
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1],
        [1, 1, 1, 1, 0, 0, 0],
        [0, 1, 0, 1, 0, 1, 0],
        [1, 0, 1, 0, 1, 0, 1],
        [1, 1, 1, 1, 1, 1, 1],
    ]
    .into_iter()
    .map(|point| {
        let val = eval(point);
        (point, Complex64::new(val.real(), val.imag()))
    })
    .collect::<Vec<_>>();
    assert_complex_samples_close(&got, &f, 1e-12);
}

#[test]
fn simple_tree_complex_product_function_is_exact_on_two_site_tree() {
    let graph = TreeTciGraph::new(2, &[TreeTciEdge::new(0, 1)]).unwrap();
    let f = |idx: &[usize]| {
        idx.iter().fold(Complex64::new(1.0, 0.0), |acc, &x| {
            acc * Complex64::new(x as f64 + 1.0, 2.0 * x as f64 + 1.0)
        })
    };

    let (tn, _ranks, _errors) = crossinterpolate2(
        batch_eval_from_point(f),
        vec![2; 2],
        graph,
        vec![vec![0; 2]],
        TreeTciOptions {
            tolerance: 1e-12,
            max_iter: 8,
            max_bond_dim: Some(2),
            normalize_error: true,
            ..Default::default()
        },
        Some(0),
        &DefaultProposer,
    )
    .unwrap();

    let (indices, _vertices) = tn.all_site_indices().unwrap();
    let pos: Vec<usize> = (0..2)
        .map(|v| {
            let site_index = tn.site_space(&v).unwrap().iter().next().unwrap();
            indices
                .iter()
                .position(|index| index == site_index)
                .unwrap()
        })
        .collect();

    let got = [[0, 0], [0, 1], [1, 0], [1, 1]]
        .into_iter()
        .map(|point| {
            let mut data = vec![0usize; indices.len()];
            for (v, &val) in point.iter().enumerate() {
                data[pos[v]] = val;
            }
            let shape = [indices.len(), 1];
            let values = ColMajorArrayRef::new(&data, &shape).unwrap();
            let value = tn
                .evaluate(&indices, values)
                .unwrap()
                .into_iter()
                .next()
                .unwrap();
            (point, Complex64::new(value.real(), value.imag()))
        })
        .collect::<Vec<_>>();
    assert_complex_samples_close(&got, &f, 1e-12);
}
