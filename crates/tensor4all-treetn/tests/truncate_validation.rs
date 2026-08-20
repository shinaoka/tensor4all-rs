//! Regression tests for SVD truncation/contraction option validation before
//! empty-center, single-node, zero-sweep, and method-dispatch shortcuts.
//!
//! These cover the issue #655 contract requirement that invalid thresholds and
//! `max_bond_dim == 0` are rejected on every path, including the paths that
//! previously returned `Ok` without performing any factorization.

use tensor4all_core::{DynIndex, IdxTensor, SvdTruncationPolicy};
use tensor4all_treetn::{contraction::ContractionOptions, TreeTN, TruncationOptions};

fn one_site(name: usize, dim: usize) -> TreeTN<IdxTensor, usize> {
    let site = DynIndex::new_dyn(dim);
    TreeTN::from_tensors(
        vec![IdxTensor::from_dense(vec![site], vec![1.0_f64; dim]).unwrap()],
        vec![name],
    )
    .unwrap()
}

#[test]
fn truncate_rejects_invalid_options_before_the_empty_center_shortcut() {
    // A NaN/infinite/negative policy or `max_bond_dim == 0` must be rejected
    // even when the empty center would otherwise be a no-op.
    let tree = one_site(0usize, 2);
    for threshold in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -1.0] {
        let options = TruncationOptions::new().with_svd_policy(SvdTruncationPolicy::new(threshold));
        assert!(
            tree.clone().truncate([], options).is_err(),
            "empty-center truncate must reject threshold {threshold}"
        );
        assert!(
            tree.clone().truncate([0usize], options).is_err(),
            "ordinary single-node truncate must reject threshold {threshold}"
        );
    }

    let zero_cap = TruncationOptions::default().with_max_bond_dim(0);
    assert!(tree.clone().truncate([], zero_cap).is_err());
    assert!(tree.clone().truncate([0usize], zero_cap).is_err());
}

#[test]
fn contract_dispatch_rejects_invalid_options_before_method_shortcuts() {
    // Both Zipup and Naive dispatch must reject the invalid policy before any
    // single-node / dense shortcut runs.
    let left = one_site(0usize, 2);
    let right = one_site(1usize, 3);
    for method in [
        tensor4all_treetn::contraction::ContractionMethod::Zipup,
        tensor4all_treetn::contraction::ContractionMethod::Naive,
    ] {
        for threshold in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -1.0] {
            let options = ContractionOptions::new(method)
                .with_svd_policy(SvdTruncationPolicy::new(threshold));
            assert!(
                tensor4all_treetn::contraction::contract(&left, &right, &0, options).is_err(),
                "contract {method:?} must reject threshold {threshold}"
            );
        }
        let zero_cap = ContractionOptions::new(method).with_max_bond_dim(0);
        assert!(
            tensor4all_treetn::contraction::contract(&left, &right, &0, zero_cap).is_err(),
            "contract {method:?} must reject max_bond_dim == 0"
        );
    }
}
