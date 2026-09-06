#![cfg(feature = "diagnostics")]

use tensor4all_core::{ColMajorArrayRef, DynIndex, IdxTensor};
use tensor4all_treeaci::{branch_diagnostics as diag, tree_elementwise, TreeAciOptions};
use tensor4all_treetn::{CachedEvaluatorOptions, EvaluationHint, TreeTN, TreeTNCachedEvaluator};

fn star(sites: &[DynIndex], shift: usize) -> TreeTN<IdxTensor, usize> {
    let bonds = [
        DynIndex::new_dyn(2),
        DynIndex::new_dyn(3),
        DynIndex::new_dyn(2),
    ];
    let mut indices = vec![sites[0].clone()];
    indices.extend(bonds.iter().cloned());
    let values = |n, offset| {
        (0..n)
            .map(|i| ((i + shift + offset) as f64 * 0.7).sin())
            .collect::<Vec<_>>()
    };
    let mut tensors = vec![IdxTensor::from_dense(indices, values(24, 0)).unwrap()];
    for (arm, bond) in bonds.into_iter().enumerate() {
        use tensor4all_core::IndexLike;
        let n = 2 * bond.dim();
        tensors.push(
            IdxTensor::from_dense(vec![bond, sites[arm + 1].clone()], values(n, arm + 3)).unwrap(),
        );
    }
    TreeTN::from_tensors(tensors, vec![0, 1, 2, 3]).unwrap()
}

#[test]
fn public_aci_records_both_operands_output_and_frames_with_actual_dimensions() {
    let sites: Vec<_> = (0..4).map(|_| DynIndex::new_dyn(2)).collect();
    let inputs = [star(&sites, 0), star(&sites, 3)];
    // A rank-one target leaves the cuts below their algebraic rank caps, so
    // the Guard must actually query both nonconstant inputs and the output.
    let expected = IdxTensor::from_dense(sites.clone(), vec![1.0_f64; 16]).unwrap();
    diag::reset();
    let result = tree_elementwise::<f64, _, _>(
        |_v| 1.0,
        &inputs,
        &TreeAciOptions {
            tolerance: 1e-12,
            max_sweeps: 12,
            ..Default::default()
        },
    )
    .unwrap();
    let residual = result
        .tree
        .to_dense()
        .unwrap()
        .sub(&expected)
        .unwrap()
        .maxabs()
        .unwrap();
    assert!(residual < 1e-10, "residual {residual}");
    let rows = diag::snapshot();
    for operand in ["input:0", "input:1", "output"] {
        assert!(
            rows.iter()
                .any(|r| r.node.starts_with(operand) && r.guard_batches.points > 0),
            "missing {operand}"
        );
    }
    for operand in ["input:0:0", "input:1:0"] {
        let row = rows.iter().find(|r| r.node == operand).unwrap();
        assert_eq!(row.bond_dims, vec![2, 2, 3]);
        assert_eq!(row.physical_dim, 2);
        assert_eq!(row.bond_product, Some(12));
        assert_eq!(row.local_elements, Some(24));
        assert_eq!(
            row.frame_batches.points,
            row.frame_cache_hits + row.frame_cache_misses
        );
        assert!(row.frame_batches.points > 0);
        assert!(row.frame_kernel.matmul_calls > 0);
    }
}

#[test]
fn warm_queries_report_hits_and_exclusive_message_times_without_double_counting() {
    let sites: Vec<_> = (0..4).map(|_| DynIndex::new_dyn(2)).collect();
    let tree = star(&sites, 0);
    let expected = tree
        .to_dense()
        .unwrap()
        .permute_indices(&sites)
        .unwrap()
        .to_vec::<f64>()
        .unwrap();
    let coords: Vec<_> = (0..16usize)
        .flat_map(|p| (0..4).map(move |i| (p >> i) & 1))
        .collect();
    let points = ColMajorArrayRef::new(&coords, &[4, 16]).unwrap();
    let mut evaluator = TreeTNCachedEvaluator::new(
        &tree,
        &sites,
        CachedEvaluatorOptions {
            diagnostic_namespace: "test".into(),
            center: Some(1),
            message_cache_max_bytes: 65536,
            branch_slice_cache_max_bytes: 65536,
            ..Default::default()
        },
    )
    .unwrap();
    for warm in [false, true] {
        diag::reset();
        let actual = evaluator
            .evaluate_batched_typed::<f64>(points, EvaluationHint::around(1))
            .unwrap();
        let residual = actual
            .iter()
            .zip(&expected)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(residual < 1e-12, "residual {residual}");
        let rows = diag::snapshot();
        let query_ns: u64 = rows.iter().map(|r| r.query_ns).sum();
        let messages_ns: u64 = rows.iter().map(|r| r.guard_ns).sum();
        assert!(
            messages_ns <= query_ns,
            "exclusive messages {messages_ns} exceed query {query_ns}"
        );
        assert_eq!(rows.iter().map(|r| r.query_batches.points).sum::<u64>(), 16);
        assert_eq!(rows.iter().map(|r| r.query_batches.calls).sum::<u64>(), 1);
        let cache = rows
            .iter()
            .find(|r| r.query_batches.calls == 1)
            .unwrap()
            .query_cache;
        assert!(cache.message_entries > 0);
        assert!(cache.message_payload_bytes <= 65536 * 6);
        assert!(cache.message_owned_bytes >= cache.message_payload_bytes);
        assert!(cache.prepared_payload_bytes <= 65536);
        assert!(cache.prepared_owned_bytes >= cache.prepared_payload_bytes);
        if warm {
            assert_eq!(rows.iter().map(|r| r.guard_cache_misses).sum::<u64>(), 0);
            assert_eq!(
                rows.iter()
                    .map(|r| r.guard_kernel.matmul_calls)
                    .sum::<u64>(),
                0
            );
        } else {
            let hub = rows.iter().find(|r| r.node == "test:0").unwrap();
            assert_eq!(hub.local_elements, Some(24));
            assert!(hub.guard_cache_misses > 0);
        }
    }
    diag::reset();
    assert!(evaluator
        .evaluate_batched_typed::<f64>(points, EvaluationHint::around(99))
        .is_err());
    assert!(diag::snapshot().is_empty());
    let empty = ColMajorArrayRef::new(&[], &[4, 0]).unwrap();
    assert!(evaluator
        .evaluate_batched_typed::<f64>(empty, EvaluationHint::default())
        .unwrap()
        .is_empty());
    assert!(diag::snapshot().is_empty());
}
