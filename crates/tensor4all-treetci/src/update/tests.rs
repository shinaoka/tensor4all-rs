use super::update_edge_default;
use crate::test_support::assert_scalar_close;
use crate::{GlobalIndexBatch, TreeTCI2, TreeTciEdge, TreeTciGraph};
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
