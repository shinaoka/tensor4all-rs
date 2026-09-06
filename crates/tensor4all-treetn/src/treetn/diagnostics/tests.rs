use super::*;

#[test]
fn shapes_and_operands_are_separate_and_snapshots_are_sorted() {
    reset();
    let sample = PhaseMeasurement {
        elapsed: Duration::from_nanos(7),
        hits: 2,
        misses: 1,
        kernel: KernelDiagnostics {
            matmul_calls: 1,
            ..Default::default()
        },
    };
    for (node, d, bonds) in [
        ("output:0", 2, vec![4, 3]),
        ("input:1:0", 2, vec![4, 3]),
        ("input:0:0", 2, vec![3, 4]),
        ("output:0", 3, vec![3, 4]),
        ("output:0", 2, vec![2, 4]),
        ("input:0:0", 2, vec![4, 3]),
    ] {
        record_guard(
            node,
            NodeShape {
                physical_dim: d,
                bond_dims: bonds,
            },
            sample,
        );
    }
    let rows = snapshot();
    assert_eq!(rows.len(), 5);
    assert_eq!(rows[0].node, "input:0:0");
    assert_eq!(rows[0].local_elements, Some(24));
    assert_eq!(rows[0].guard_ns, 14);
    assert_eq!(
        rows[0].guard_batches,
        BatchDiagnostics {
            calls: 2,
            points: 6,
            min: 3,
            max: 3
        }
    );
    assert_eq!(rows[0].guard_kernel.matmul_calls, 2);
    assert_eq!(rows[2].bond_dims, vec![2, 4]);
    assert_eq!(rows[4].physical_dim, 3);
}

#[test]
fn empty_batches_and_overflow_are_explicit() {
    reset();
    record_frame(
        "hub",
        NodeShape {
            physical_dim: 2,
            bond_dims: vec![usize::MAX, 2],
        },
        PhaseMeasurement::default(),
    );
    let row = &snapshot()[0];
    assert_eq!(row.bond_product, None);
    assert_eq!(row.local_elements, None);
    assert_eq!(row.frame_batches, BatchDiagnostics::default());
    assert_eq!(
        NodeShape {
            physical_dim: 3,
            bond_dims: vec![]
        }
        .local_elements(),
        Some(3)
    );
    assert_eq!(nanos(Duration::MAX), u64::MAX);
}

#[test]
fn kernel_deltas_and_reset_are_thread_local() {
    reset();
    record_kernel(KernelDiagnostics {
        setup_ns: 4,
        matmul_ns: 7,
        prepared_hits: 2,
        ..Default::default()
    });
    let before = kernel_snapshot();
    std::thread::spawn(|| {
        reset();
        record_kernel(KernelDiagnostics {
            matmul_ns: 900,
            ..Default::default()
        });
        assert_eq!(kernel_snapshot().matmul_ns, 900);
    })
    .join()
    .unwrap();
    assert_eq!(kernel_snapshot(), before);
    record_kernel(KernelDiagnostics {
        setup_ns: 3,
        accumulate_ns: 5,
        ..Default::default()
    });
    assert_eq!(
        kernel_snapshot().since(before),
        KernelDiagnostics {
            setup_ns: 3,
            accumulate_ns: 5,
            ..Default::default()
        }
    );
    reset();
    assert_eq!(kernel_snapshot(), KernelDiagnostics::default());
}

#[test]
fn reset_then_snapshot_is_empty() {
    record_guard(
        "a",
        NodeShape {
            physical_dim: 2,
            bond_dims: vec![3, 3],
        },
        PhaseMeasurement {
            elapsed: Duration::from_nanos(10),
            hits: 1,
            misses: 0,
            ..Default::default()
        },
    );
    reset();
    assert!(snapshot().is_empty());
}

#[test]
fn record_guard_and_record_frame_accumulate_into_one_entry_per_node() {
    reset();
    record_guard(
        "hub",
        NodeShape {
            physical_dim: 2,
            bond_dims: vec![4, 4, 4],
        },
        PhaseMeasurement {
            elapsed: Duration::from_nanos(100),
            hits: 2,
            misses: 1,
            ..Default::default()
        },
    );
    record_guard(
        "hub",
        NodeShape {
            physical_dim: 2,
            bond_dims: vec![4, 4, 4],
        },
        PhaseMeasurement {
            elapsed: Duration::from_nanos(50),
            hits: 0,
            misses: 1,
            ..Default::default()
        },
    );
    record_frame(
        "hub",
        NodeShape {
            physical_dim: 2,
            bond_dims: vec![4, 4, 4],
        },
        PhaseMeasurement {
            elapsed: Duration::from_nanos(30),
            hits: 5,
            misses: 2,
            ..Default::default()
        },
    );

    let snap = snapshot();
    assert_eq!(snap.len(), 1);
    let hub = &snap[0];
    assert_eq!(hub.node, "hub");
    assert_eq!(hub.coordination_number, 3);
    assert_eq!(hub.bond_dims, vec![4, 4, 4]);
    assert_eq!(hub.guard_ns, 150);
    assert_eq!(hub.guard_cache_hits, 2);
    assert_eq!(hub.guard_cache_misses, 2);
    assert_eq!(hub.frame_ns, 30);
    assert_eq!(hub.frame_cache_hits, 5);
    assert_eq!(hub.frame_cache_misses, 2);
}

#[test]
fn distinct_nodes_get_distinct_entries() {
    reset();
    record_guard(
        "a",
        NodeShape {
            physical_dim: 2,
            bond_dims: vec![3, 3],
        },
        PhaseMeasurement {
            elapsed: Duration::from_nanos(10),
            hits: 1,
            misses: 0,
            ..Default::default()
        },
    );
    record_guard(
        "b",
        NodeShape {
            physical_dim: 2,
            bond_dims: vec![5, 5, 5],
        },
        PhaseMeasurement {
            elapsed: Duration::from_nanos(20),
            hits: 1,
            misses: 0,
            ..Default::default()
        },
    );
    let mut nodes: Vec<String> = snapshot().into_iter().map(|d| d.node).collect();
    nodes.sort();
    assert_eq!(nodes, vec!["a".to_string(), "b".to_string()]);
}
