//! `max_working_bytes` is the governing memory budget of a TreeACI run.
//!
//! Issue #729: the element ceilings used to be constants that stayed where
//! they were when a caller raised the byte budget, so a run could be refused
//! by a limit two orders of magnitude below the budget it had been granted.
//! These tests pin the coupling from the public surface: an unset ceiling
//! moves with the budget in both directions, and an explicitly set one does
//! not move at all.

use tensor4all_core::{DynIndex, IdxTensor};
use tensor4all_treeaci::{tree_elementwise_batched, Result, TreeAciError, TreeAciOptions};
use tensor4all_treetn::TreeTN;

/// Two shared sites of physical dimension two joined by a bond of dimension one.
///
/// Each core holds two elements and the only edge's smallest possible local
/// matrix holds `2 * 2 = 4`, so a ceiling of three refuses the edge while the
/// cores still fit -- which is what makes the reported `requested` value below
/// unambiguous.
fn two_site_tree(
    sites: &[DynIndex; 2],
    first: [f64; 2],
    second: [f64; 2],
) -> TreeTN<IdxTensor, usize> {
    let bond = DynIndex::new_dyn(1);
    let tensors = vec![
        IdxTensor::from_dense(vec![sites[0].clone(), bond.clone()], first.to_vec())
            .expect("first fixture core"),
        IdxTensor::from_dense(vec![sites[1].clone(), bond], second.to_vec())
            .expect("second fixture core"),
    ];
    TreeTN::from_tensors(tensors, vec![0usize, 1usize]).expect("two-site fixture tree")
}

fn multiply(
    batch: tensor4all_treeaci::TreeElementwiseBatch<'_, f64>,
    out: &mut [f64],
) -> Result<()> {
    for (point, value) in out.iter_mut().enumerate() {
        *value = batch.get(0, point)? * batch.get(1, point)?;
    }
    Ok(())
}

/// The budget a run is granted is the budget its element ceilings enforce.
#[test]
fn unset_element_ceilings_follow_the_working_budget() {
    let sites = [DynIndex::new_dyn(2), DynIndex::new_dyn(2)];
    let inputs = vec![
        two_site_tree(&sites, [1.0, 2.0], [3.0, 4.0]),
        two_site_tree(&sites, [5.0, 6.0], [7.0, 8.0]),
    ];

    // Ninety-six bytes is twelve f64, so one object may claim three of them.
    // Before #729 this same configuration ran with a ceiling of 2^24.
    let starved = TreeAciOptions::<usize> {
        max_working_bytes: 96,
        ..TreeAciOptions::default()
    };
    let error = tree_elementwise_batched::<f64, _, _>(multiply, &inputs, &starved)
        .expect_err("a twelve-element budget cannot hold a four-element local matrix");
    assert!(
        matches!(
            error,
            TreeAciError::ResourceLimit {
                resource: "local matrix elements",
                requested: 4,
                limit: 3,
            }
        ),
        "unexpected error: {error}"
    );

    // The same run at the default budget succeeds and is exact: the product of
    // two rank-one trees is the rank-one tree of the products.
    let generous = TreeAciOptions::<usize>::default();
    let result = tree_elementwise_batched::<f64, _, _>(multiply, &inputs, &generous)
        .expect("the default budget admits this run");
    let expected = two_site_tree(&sites, [5.0, 12.0], [21.0, 32.0])
        .to_dense()
        .expect("dense reference");
    assert!(result
        .tree
        .to_dense()
        .expect("dense interpolation")
        .isapprox(&expected, 1.0e-12, 1.0e-14)
        .expect("comparable dense results"));
}

/// An explicitly set ceiling is a pin: the budget no longer moves it, so the
/// starved run is refused by the budget itself rather than by a ceiling.
#[test]
fn explicit_element_ceilings_do_not_follow_the_working_budget() {
    let sites = [DynIndex::new_dyn(2), DynIndex::new_dyn(2)];
    let inputs = vec![
        two_site_tree(&sites, [1.0, 2.0], [3.0, 4.0]),
        two_site_tree(&sites, [5.0, 6.0], [7.0, 8.0]),
    ];
    let pinned = TreeAciOptions::<usize> {
        max_working_bytes: 96,
        max_local_matrix_elements: Some(1 << 24),
        max_core_elements: Some(1 << 24),
        max_frame_elements: Some(1 << 24),
        ..TreeAciOptions::default()
    };

    let error = tree_elementwise_batched::<f64, _, _>(multiply, &inputs, &pinned)
        .expect_err("ninety-six bytes cannot hold this run's live buffers");
    assert!(
        matches!(
            error,
            TreeAciError::ResourceLimit {
                resource: "working bytes",
                ..
            }
        ),
        "unexpected error: {error}"
    );
}

/// A ceiling set to zero is refused as an invalid option, while leaving it
/// unset is not: an unset ceiling is derived from a budget that is itself
/// validated.
#[test]
fn zero_is_rejected_only_when_a_ceiling_is_set_explicitly() {
    let sites = [DynIndex::new_dyn(2), DynIndex::new_dyn(2)];
    let inputs = vec![
        two_site_tree(&sites, [1.0, 2.0], [3.0, 4.0]),
        two_site_tree(&sites, [5.0, 6.0], [7.0, 8.0]),
    ];
    let zeroed = TreeAciOptions::<usize> {
        max_core_elements: Some(0),
        ..TreeAciOptions::default()
    };

    let error = tree_elementwise_batched::<f64, _, _>(multiply, &inputs, &zeroed)
        .expect_err("an explicit zero ceiling admits nothing");
    assert!(
        matches!(
            error,
            TreeAciError::InvalidOption {
                option: "max_core_elements",
                ..
            }
        ),
        "unexpected error: {error}"
    );
}

/// A budget too small for the topology itself is reported against the budget,
/// not against a ceiling derived from it.
///
/// Downstream callers (the `sgw` stage harness among them) assert exactly this
/// message for a deliberately impossible budget, and the derived ceilings must
/// not take that report away from them.
#[test]
fn an_impossible_budget_is_reported_as_working_bytes() {
    let sites = [DynIndex::new_dyn(2), DynIndex::new_dyn(2)];
    let inputs = vec![
        two_site_tree(&sites, [1.0, 2.0], [3.0, 4.0]),
        two_site_tree(&sites, [5.0, 6.0], [7.0, 8.0]),
    ];
    let impossible = TreeAciOptions::<usize> {
        max_working_bytes: 1,
        ..TreeAciOptions::default()
    };

    let error = tree_elementwise_batched::<f64, _, _>(multiply, &inputs, &impossible)
        .expect_err("a one-byte budget admits nothing");
    assert!(
        matches!(
            error,
            TreeAciError::ResourceLimit {
                resource: "working bytes",
                ..
            }
        ),
        "unexpected error: {error}"
    );
}
