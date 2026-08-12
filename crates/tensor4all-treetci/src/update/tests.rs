use super::{evaluate_candidate_matrix, update_edge_default};
use crate::test_support::assert_scalar_close;
use crate::{GlobalIndexBatch, SubtreeKey, TreeTCI2, TreeTciEdge, TreeTciGraph};
use anyhow::Result;
use tensor4all_core::ColMajorArray;
use tensor4all_tcicore::RrLUOptions;

fn no_truncation_options() -> RrLUOptions {
    RrLUOptions {
        max_bond_dim: usize::MAX,
        rel_tol: 0.0,
        abs_tol: 0.0,
        left_orthogonal: true,
    }
}

fn two_site_graph() -> TreeTciGraph {
    TreeTciGraph::new(2, &[TreeTciEdge::new(0, 1)]).unwrap()
}

#[test]
fn update_edge_selects_identity_pivots_on_two_site_tree() {
    let mut tci = TreeTCI2::<f64>::new(vec![2, 2], two_site_graph()).unwrap();
    tci.add_global_pivots(&[vec![0, 0]]).unwrap();
    tci.flush_pivot_errors();

    let batch_eval = |batch: GlobalIndexBatch<'_>| -> Result<Vec<f64>> {
        let mut values = Vec::with_capacity(batch.n_points());
        for point in 0..batch.n_points() {
            let i = batch.get(0, point).unwrap();
            let j = batch.get(1, point).unwrap();
            values.push(if i == j { 1.0 } else { 0.0 });
        }
        Ok(values)
    };

    let selection = update_edge_default(
        &mut tci,
        TreeTciEdge::new(0, 1),
        batch_eval,
        &no_truncation_options(),
    )
    .unwrap();

    assert_eq!(selection.rank, 2);
    // [1, 2]: Column 0 = [0], Column 1 = [1]
    assert_eq!(
        tci.ijset[&crate::SubtreeKey::new(vec![0])],
        ColMajorArray::new(vec![0, 1], vec![1, 2]).unwrap()
    );
    assert_eq!(
        tci.ijset[&crate::SubtreeKey::new(vec![1])],
        ColMajorArray::new(vec![0, 1], vec![1, 2]).unwrap()
    );
    assert_scalar_close(tci.max_sample_value, 1.0, 1.0, 1e-12);
    assert_scalar_close(tci.max_bond_error(), 0.0, tci.max_sample_value, 1e-12);
    assert_scalar_close(
        tci.pivot_errors.last().copied().unwrap_or(f64::NAN),
        0.0,
        tci.max_sample_value,
        1e-12,
    );
}

#[test]
fn update_edge_rejects_bad_batch_length() {
    let mut tci = TreeTCI2::<f64>::new(vec![2, 2], two_site_graph()).unwrap();
    tci.add_global_pivots(&[vec![0, 0]]).unwrap();

    let bad_eval = |_batch: GlobalIndexBatch<'_>| -> Result<Vec<f64>> { Ok(vec![1.0]) };
    let result = update_edge_default(
        &mut tci,
        TreeTciEdge::new(0, 1),
        bad_eval,
        &no_truncation_options(),
    );

    assert!(result.is_err());
}

#[test]
fn evaluate_candidate_matrix_writes_points_in_column_major_order() {
    // Unequal candidate counts, so a transposed or mis-ordered assembly shows up.
    let n_sites = 4;
    // Local dims large enough for the probe values (2/3 on sites 2/3).
    let local_dims = vec![4, 4, 4, 4];
    let left_key = SubtreeKey::new(vec![0, 1]);
    let left_candidates = vec![vec![0, 0], vec![1, 1], vec![0, 1]];
    let right_key = SubtreeKey::new(vec![2, 3]);
    let right_candidates = vec![vec![2, 3], vec![1, 0]];

    // Encode each point's site values into a scalar so order and values are
    // both asserted in one shot: right candidates vary slowest (column-major).
    let evaluate = |batch: GlobalIndexBatch<'_>| -> Result<Vec<f64>> {
        let mut out = Vec::with_capacity(batch.n_points());
        for p in 0..batch.n_points() {
            let code = (0..n_sites)
                .map(|s| batch.get(s, p).unwrap())
                .fold(0f64, |acc, v| acc * 10.0 + v as f64);
            out.push(code);
        }
        Ok(out)
    };

    let values = evaluate_candidate_matrix(
        n_sites,
        &left_key,
        &left_candidates,
        &right_key,
        &right_candidates,
        &local_dims,
        evaluate,
    )
    .unwrap();

    // right=[2,3] x left=[0,0],[1,1],[0,1], then right=[1,0] x same lefts.
    assert_eq!(values, vec![23.0, 1123.0, 123.0, 10.0, 1110.0, 110.0]);
}

#[test]
fn evaluate_candidate_matrix_rejects_malformed_candidate_length() {
    // Unequal partitions: left subtree has 2 sites, right has 3. A left
    // candidate whose length coincides with the *right* key's length is the
    // case the previous length-based side inference got wrong: it passed
    // validation and silently dropped the excess entry, leaving part of its
    // point at zero.
    let n_sites = 5;
    let local_dims = vec![2, 2, 2, 2, 2];
    let left_key = SubtreeKey::new(vec![0, 1]);
    let right_key = SubtreeKey::new(vec![2, 3, 4]);
    let right_candidates = vec![vec![1, 0, 1]];

    let malformed_left = vec![vec![0, 0, 1]]; // len 3 == right key len
    let evaluate = |_batch: GlobalIndexBatch<'_>| -> Result<Vec<f64>> { Ok(vec![1.0]) };
    let result = evaluate_candidate_matrix(
        n_sites,
        &left_key,
        &malformed_left,
        &right_key,
        &right_candidates,
        &local_dims,
        evaluate,
    );
    assert!(result.is_err());

    // A well-formed candidate still assembles correctly.
    let ok_left = vec![vec![0, 1]];
    let result = evaluate_candidate_matrix(
        n_sites,
        &left_key,
        &ok_left,
        &right_key,
        &right_candidates,
        &local_dims,
        |batch: GlobalIndexBatch<'_>| -> Result<Vec<f64>> {
            Ok(vec![
                (batch.get(0, 0).unwrap() + batch.get(4, 0).unwrap()) as f64,
            ])
        },
    );
    assert_eq!(result.unwrap(), vec![1.0]);
}

#[test]
fn evaluate_candidate_matrix_rejects_out_of_range_coordinates() {
    let n_sites = 4;
    let local_dims = vec![2, 2, 2, 2];
    let left_key = SubtreeKey::new(vec![0, 1]);
    let right_key = SubtreeKey::new(vec![2, 3]);
    let right_candidates = vec![vec![0, 1]];
    let evaluate = |_batch: GlobalIndexBatch<'_>| -> Result<Vec<f64>> { Ok(vec![1.0]) };

    // Value 2 is out of range for a 2-state site.
    let bad_left = vec![vec![0, 2]];
    let result = evaluate_candidate_matrix(
        n_sites,
        &left_key,
        &bad_left,
        &right_key,
        &right_candidates,
        &local_dims,
        evaluate,
    );
    assert!(result.is_err());
}
