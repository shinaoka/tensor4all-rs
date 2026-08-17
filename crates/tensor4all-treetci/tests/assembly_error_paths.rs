//! Integration tests for treetci point assembly error paths and success cases.

use tensor4all_treetci::{
    assemble_global_point, assemble_points_column_major, crossinterpolate2, DefaultProposer,
    SubtreeKey, TreeTciGraph, TreeTciOptions,
};

#[test]
fn assemble_global_point_builds_valid_point() {
    let key = SubtreeKey::new(vec![0, 1]);
    let values = vec![3, 4];
    let point = assemble_global_point(3, &[(&key, &values)], &[(2, 7)]).unwrap();
    assert_eq!(point, vec![3, 4, 7]);
}

#[test]
fn assemble_global_point_rejects_key_length_mismatch() {
    let key = SubtreeKey::new(vec![0, 1, 2]);
    let values = vec![3, 4];
    let err = assemble_global_point(3, &[(&key, &values)], &[]).unwrap_err();
    assert!(err.to_string().contains("cannot be filled"));
}

#[test]
fn assemble_global_point_rejects_out_of_bounds_site() {
    let key = SubtreeKey::new(vec![5]);
    let values = vec![1];
    let err = assemble_global_point(3, &[(&key, &values)], &[]).unwrap_err();
    assert!(err.to_string().contains("out of bounds"));
}

#[test]
fn assemble_global_point_rejects_duplicate_assignment() {
    let key = SubtreeKey::new(vec![0]);
    let values = vec![1];
    let err = assemble_global_point(3, &[(&key, &values), (&key, &values)], &[]).unwrap_err();
    assert!(err.to_string().contains("assigned more than once"));
}

#[test]
fn assemble_global_point_rejects_central_out_of_bounds() {
    let err = assemble_global_point(2, &[], &[(7, 1)]).unwrap_err();
    assert!(err.to_string().contains("out of bounds"));
}

#[test]
fn assemble_global_point_rejects_central_duplicate() {
    let key = SubtreeKey::new(vec![0]);
    let values = vec![1];
    let err = assemble_global_point(2, &[(&key, &values)], &[(0, 2)]).unwrap_err();
    assert!(err.to_string().contains("assigned more than once"));
}

#[test]
fn crossinterpolate2_propagates_initial_callback_error() {
    let graph = TreeTciGraph::linear_chain(2).unwrap();
    let error = crossinterpolate2::<f64, _, _>(
        |_| Err(anyhow::anyhow!("callback failed")),
        vec![2, 2],
        graph,
        vec![vec![0, 0]],
        TreeTciOptions::default(),
        None,
        &DefaultProposer,
    )
    .unwrap_err();
    assert!(error.to_string().contains("callback failed"));
}

#[test]
fn crossinterpolate2_accepts_empty_initial_pivots() {
    let graph = TreeTciGraph::linear_chain(2).unwrap();
    let result = crossinterpolate2::<f64, _, _>(
        |batch| Ok(vec![1.0; batch.n_points()]),
        vec![2, 2],
        graph,
        vec![],
        TreeTciOptions {
            max_iter: 1,
            max_bond_dim: Some(2),
            ..TreeTciOptions::default()
        },
        None,
        &DefaultProposer,
    );
    assert!(
        result.is_ok(),
        "empty initial pivots should be accepted: {:?}",
        result.as_ref().err()
    );
}

#[test]
fn crossinterpolate2_rejects_local_dimension_mismatch() {
    let graph = TreeTciGraph::linear_chain(2).unwrap();
    let error = crossinterpolate2::<f64, _, _>(
        |_| Ok(vec![1.0]),
        vec![2],
        graph,
        vec![vec![0, 0]],
        TreeTciOptions::default(),
        None,
        &DefaultProposer,
    )
    .unwrap_err();
    assert!(error.to_string().contains("local_dims length"));
}

#[test]
fn crossinterpolate2_rejects_initial_callback_length_mismatch() {
    let graph = TreeTciGraph::linear_chain(2).unwrap();
    let error = crossinterpolate2::<f64, _, _>(
        |_| Ok(Vec::new()),
        vec![2, 2],
        graph,
        vec![vec![0, 0]],
        TreeTciOptions::default(),
        None,
        &DefaultProposer,
    )
    .unwrap_err();
    assert!(error.to_string().contains("initial evaluator returned"));
}

#[test]
fn crossinterpolate2_rejects_all_zero_initial_pivots() {
    let graph = TreeTciGraph::linear_chain(2).unwrap();
    let error = crossinterpolate2::<f64, _, _>(
        |_| Ok(vec![0.0]),
        vec![2, 2],
        graph,
        vec![vec![0, 0]],
        TreeTciOptions::default(),
        None,
        &DefaultProposer,
    )
    .unwrap_err();
    assert!(error.to_string().contains("must not all evaluate to zero"));
}

#[test]
fn assemble_points_column_major_roundtrips_batch() {
    let points = vec![vec![0, 1, 1], vec![1, 0, 0]];
    let batch = assemble_points_column_major(&points).unwrap();
    let view = batch.as_view();
    assert_eq!(view.n_points(), 2);
    assert_eq!(view.n_sites(), 3);
    assert_eq!(view.get(0, 0), Some(0));
    assert_eq!(view.get(1, 1), Some(0));
}

#[test]
fn assemble_points_column_major_rejects_empty_points() {
    let err = assemble_points_column_major(&[] as &[Vec<usize>]).unwrap_err();
    assert!(err.to_string().contains("at least one point"));
}

#[test]
fn assemble_points_column_major_rejects_inconsistent_site_count() {
    let points = vec![vec![0, 1], vec![0]];
    let err = assemble_points_column_major(&points).unwrap_err();
    assert!(err.to_string().contains("same site count"));
}
