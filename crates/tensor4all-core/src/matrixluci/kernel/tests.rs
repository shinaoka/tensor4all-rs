use crate::matrixluci::{PivotKernelOptions, PivotSelectionCore};

#[test]
fn pivot_selection_core_stores_rank_and_indices() {
    let selection = PivotSelectionCore {
        row_indices: vec![0, 2],
        col_indices: vec![1, 3],
        pivot_errors: vec![1e-3, 1e-6],
        rank: 2,
    };
    assert_eq!(selection.rank, 2);
    assert_eq!(selection.row_indices, vec![0, 2]);
}

#[test]
fn pivot_kernel_options_no_truncation_uses_canonical_values() {
    let opts = PivotKernelOptions::no_truncation();
    assert_eq!(opts.rel_tol, 0.0);
    assert_eq!(opts.abs_tol, 0.0);
    assert_eq!(opts.max_bond_dim, usize::MAX);
    assert!(opts.left_orthogonal);
}
