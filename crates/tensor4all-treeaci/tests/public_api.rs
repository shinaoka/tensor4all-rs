use std::{fs, path::Path};

use tensor4all_treeaci::{
    TreeAciDiagnostics, TreeAciError, TreeAciNode, TreeAciOptions, TreeAciResult, TreeAciScalar,
    TreeAciTermination, TreeAciTraversalStrategy, TreeElementwiseBatch,
};

fn require_scalar<T: TreeAciScalar>() {}

fn require_node<V: TreeAciNode>() {}

#[test]
fn scalar_and_node_contracts_cover_supported_basics() {
    require_scalar::<f64>();
    require_scalar::<num_complex::Complex64>();
    require_node::<usize>();
    require_node::<String>();
}

#[test]
fn options_defaults_are_bounded_and_consistent() {
    let options = TreeAciOptions::<usize>::default();

    assert!(options.max_sweeps >= options.min_sweeps);
    assert!(options.min_sweeps > 0);
    assert!(options.tolerance.is_finite() && options.tolerance > 0.0);
    assert!(options.scale_tolerance);
    assert!(options.initial_guess.is_none());
    assert!(options.root.is_none());
    assert!(options.max_candidate_rows > 0);
    assert!(options.max_candidate_cols > 0);
    assert!(options.max_local_matrix_elements > 0);
    assert!(options.max_core_elements > 0);
    assert!(options.max_frame_elements > 0);
    assert!(options.max_sample_arena_bytes > 0);
    assert!(options.max_working_bytes > 0);
    assert_eq!(
        options.traversal_strategy,
        TreeAciTraversalStrategy::MinimumRetracingWalk
    );
}

#[test]
fn result_value_types_preserve_history_and_diagnostics() {
    let diagnostics = TreeAciDiagnostics {
        edge_ranks: vec![(0usize, 1usize, 3)],
        saturated_edges: vec![(0, 1)],
        evaluated_points: 21,
        sample_arena_records: 8,
        sample_arena_retained_bytes: 1024,
        candidate_set_sizes: vec![(0usize, 1usize, 3), (1usize, 0usize, 3)],
    };
    let result = TreeAciResult {
        tree: tensor4all_treetn::TreeTN::new(),
        max_ranks: vec![2, 3],
        max_errors: vec![1.0e-2, 1.0e-6],
        global_pivots_found: vec![1, 0],
        termination: TreeAciTermination::Converged,
        diagnostics,
    };

    assert_eq!(result.max_ranks, vec![2, 3]);
    assert_eq!(result.diagnostics.evaluated_points, 21);
    assert_eq!(result.diagnostics.sample_arena_records, 8);
    assert_eq!(
        result.diagnostics.candidate_set_sizes,
        vec![(0usize, 1usize, 3), (1usize, 0usize, 3)]
    );
    assert_eq!(result.termination, TreeAciTermination::Converged);
}

#[test]
fn traversal_strategy_has_a_stable_default() {
    assert_eq!(
        TreeAciTraversalStrategy::default(),
        TreeAciTraversalStrategy::MinimumRetracingWalk
    );
}

#[test]
fn batch_uses_column_major_input_point_layout() {
    let values = [10, 20, 11, 21, 12, 22];
    let batch = TreeElementwiseBatch::new(&values, 2, 3).unwrap();

    assert_eq!(batch.n_inputs(), 2);
    assert_eq!(batch.n_points(), 3);
    assert_eq!(batch.get(0, 0).unwrap(), 10);
    assert_eq!(batch.get(1, 0).unwrap(), 20);
    assert_eq!(batch.get(0, 2).unwrap(), 12);
    assert_eq!(batch.get(1, 2).unwrap(), 22);
    assert_eq!(batch.as_col_major_slice(), values.as_slice());
}

#[test]
fn batch_rejects_invalid_shapes_and_indices() {
    assert!(matches!(
        TreeElementwiseBatch::<f64>::new(&[], 0, 1),
        Err(TreeAciError::EmptyBatchAxis { axis: "input" })
    ));
    assert!(matches!(
        TreeElementwiseBatch::<f64>::new(&[], 1, 0),
        Err(TreeAciError::EmptyBatchAxis { axis: "point" })
    ));
    assert!(matches!(
        TreeElementwiseBatch::<f64>::new(&[], usize::MAX, 2),
        Err(TreeAciError::SizeOverflow { context: "batch" })
    ));
    assert!(matches!(
        TreeElementwiseBatch::new(&[1, 2, 3], 2, 2),
        Err(TreeAciError::LengthMismatch {
            expected: 4,
            actual: 3
        })
    ));

    let values = [10, 20, 11, 21];
    let batch = TreeElementwiseBatch::new(&values, 2, 2).unwrap();
    assert!(matches!(
        batch.get(2, 0),
        Err(TreeAciError::BatchIndexOutOfBounds {
            axis: "input",
            index: 2,
            len: 2
        })
    ));
    assert!(matches!(
        batch.get(0, 2),
        Err(TreeAciError::BatchIndexOutOfBounds {
            axis: "point",
            index: 2,
            len: 2
        })
    ));
}

#[test]
fn crate_does_not_reference_train_specific_crates_or_types() {
    let crate_root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let manifest = fs::read_to_string(crate_root.join("Cargo.toml")).unwrap();
    let forbidden_dependencies = [
        "tensor4all-simplett",
        "tensor4all-aci",
        "tensor4all-treetci",
    ];
    for dependency in forbidden_dependencies {
        assert!(
            !manifest.contains(dependency),
            "forbidden dependency {dependency} found in treeaci manifest"
        );
    }
    assert!(manifest.contains(
        "tensor4all-treetn = { path = \"../tensor4all-treetn\", default-features = false }"
    ));

    let tree_manifest =
        fs::read_to_string(crate_root.join("../tensor4all-treetn/Cargo.toml")).unwrap();
    assert!(tree_manifest.contains(
        "tensor4all-simplett = { path = \"../tensor4all-simplett\", default-features = false, optional = true }"
    ));

    let mut pending = vec![crate_root.join("src")];
    let forbidden_symbols = ["SimpleTensorTrain", "TTCache"];
    while let Some(path) = pending.pop() {
        for entry in fs::read_dir(path).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                pending.push(path);
            } else if path.extension().is_some_and(|extension| extension == "rs") {
                let source = fs::read_to_string(&path).unwrap();
                for symbol in forbidden_symbols {
                    assert!(
                        !source.contains(symbol),
                        "forbidden symbol {symbol} found in {}",
                        path.display()
                    );
                }
            }
        }
    }
}
