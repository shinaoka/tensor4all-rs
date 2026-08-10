//! Integration tests for treetci point assembly error paths and success cases.

use tensor4all_treetci::{assemble_global_point, assemble_points_column_major, SubtreeKey};

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
