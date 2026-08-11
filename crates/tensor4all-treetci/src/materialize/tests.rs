use super::{cartesian_entries, to_treetn, FullPivLuScalar};
use crate::test_support::{assert_complex_slice_close, assert_scalar_close};
use crate::{
    optimize_default, GlobalIndexBatch, SubtreeKey, TreeTCI2, TreeTciEdge, TreeTciGraph,
    TreeTciOptions,
};
use anyhow::Result;
use num_complex::Complex64;
use std::collections::HashMap as StdHashMap;
use tensor4all_core::{ColMajorArray, ColMajorArrayRef};

fn two_site_graph() -> TreeTciGraph {
    TreeTciGraph::new(2, &[TreeTciEdge::new(0, 1)]).unwrap()
}

#[test]
fn to_treetn_preserves_two_site_identity_evaluations() {
    let mut tci = TreeTCI2::<f64>::new(vec![2, 2], two_site_graph()).unwrap();
    tci.add_global_pivots(&[vec![0, 0]]).unwrap();

    let batch_eval = |batch: GlobalIndexBatch<'_>| -> Result<Vec<f64>> {
        let mut values = Vec::with_capacity(batch.n_points());
        for point in 0..batch.n_points() {
            let i = batch.get(0, point).unwrap();
            let j = batch.get(1, point).unwrap();
            values.push(if i == j { 1.0 } else { 0.0 });
        }
        Ok(values)
    };

    optimize_default(
        &mut tci,
        batch_eval,
        &TreeTciOptions {
            tolerance: 1e-12,
            max_iter: 4,
            max_bond_dim: None,
            normalize_error: true,
            ..Default::default()
        },
    )
    .unwrap();

    let tn = to_treetn(&tci, batch_eval, Some(0)).unwrap();

    let (indices, _vertices) = tn.all_site_indices().unwrap();
    let pos0 = {
        let site_index = tn.site_space(&0usize).unwrap().iter().next().unwrap();
        indices
            .iter()
            .position(|index| index == site_index)
            .unwrap()
    };
    let pos1 = {
        let site_index = tn.site_space(&1usize).unwrap().iter().next().unwrap();
        indices
            .iter()
            .position(|index| index == site_index)
            .unwrap()
    };

    let eval = |i: usize, j: usize| -> f64 {
        let mut data = vec![0usize; indices.len()];
        data[pos0] = i;
        data[pos1] = j;
        let shape = [indices.len(), 1];
        let values = ColMajorArrayRef::new(&data, &shape).unwrap();
        tn.evaluate(&indices, values).unwrap()[0].real()
    };

    assert_scalar_close(eval(0, 0), 1.0, 1.0, 1e-12);
    assert_scalar_close(eval(0, 1), 0.0, 1.0, 1e-12);
    assert_scalar_close(eval(1, 0), 0.0, 1.0, 1e-12);
    assert_scalar_close(eval(1, 1), 1.0, 1.0, 1e-12);
}

#[test]
fn cartesian_entries_matches_julia_product_order() {
    let key_a = SubtreeKey::new(vec![0]);
    let key_b = SubtreeKey::new(vec![1]);
    // key_a: shape [1, 2], columns [0] and [1]
    // key_b: shape [1, 3], columns [10], [11], [12]
    let ijset = StdHashMap::from([
        (
            key_a.clone(),
            ColMajorArray::new(vec![0, 1], vec![1, 2]).unwrap(),
        ),
        (
            key_b.clone(),
            ColMajorArray::new(vec![10, 11, 12], vec![1, 3]).unwrap(),
        ),
    ]);

    let combos = cartesian_entries(&ijset, &[key_a, key_b]).unwrap();

    assert_eq!(
        combos,
        vec![
            vec![vec![0], vec![10]],
            vec![vec![1], vec![10]],
            vec![vec![0], vec![11]],
            vec![vec![1], vec![11]],
            vec![vec![0], vec![12]],
            vec![vec![1], vec![12]],
        ]
    );
}

#[test]
fn solve_right_full_piv_lu_preserves_complex_entries_for_identity_pivot() {
    let lhs = vec![
        Complex64::new(1.0, 2.0),
        Complex64::new(-3.0, 4.0),
        Complex64::new(5.0, -6.0),
        Complex64::new(-7.0, -8.0),
    ];
    let pivot = vec![
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(1.0, 0.0),
    ];

    let solved = Complex64::solve_right_full_piv_lu(&lhs, 2, 2, &pivot, 2, 2).unwrap();

    let max_sample = lhs.iter().map(|value| value.norm()).fold(0.0_f64, f64::max);
    assert_complex_slice_close(&solved, &lhs, max_sample, 1e-12);
}

#[test]
fn solve_right_full_piv_lu_recovers_complex_rhs_for_nontrivial_pivot() {
    let target = vec![
        Complex64::new(1.0, 2.0),
        Complex64::new(-3.0, 4.0),
        Complex64::new(5.0, -6.0),
        Complex64::new(-7.0, -8.0),
    ];
    let pivot = vec![
        Complex64::new(2.0, 1.0),
        Complex64::new(-1.0, 3.0),
        Complex64::new(4.0, -2.0),
        Complex64::new(3.0, 5.0),
    ];
    let pi1 = vec![
        target[0] * pivot[0] + target[2] * pivot[1],
        target[1] * pivot[0] + target[3] * pivot[1],
        target[0] * pivot[2] + target[2] * pivot[3],
        target[1] * pivot[2] + target[3] * pivot[3],
    ];

    let solved = Complex64::solve_right_full_piv_lu(&pi1, 2, 2, &pivot, 2, 2).unwrap();

    let max_sample = target
        .iter()
        .map(|value| value.norm())
        .fold(0.0_f64, f64::max);
    assert_complex_slice_close(&solved, &target, max_sample, 1e-12);
}

#[test]
fn to_treetn_emits_zero_core_for_zero_pivot_matrix() {
    // A pivot whose sampled function values are all exactly zero produces a
    // singular pivot matrix in `site_tensor_with_parent`; materialization
    // must emit a zero core instead of failing the solve.
    let mut tci = TreeTCI2::<f64>::new(vec![2, 2], two_site_graph()).unwrap();
    tci.add_global_pivots(&[vec![0, 0]]).unwrap();

    // Evaluator that is exactly zero at every pivot cross-section.
    let batch_eval =
        |batch: GlobalIndexBatch<'_>| -> Result<Vec<f64>> { Ok(vec![0.0; batch.n_points()]) };

    let tn = to_treetn(&tci, batch_eval, Some(0)).unwrap();
    let dense = tn.to_dense().unwrap();
    let values = dense.to_vec::<f64>().unwrap();
    assert_eq!(values, vec![0.0; 4]);
}
