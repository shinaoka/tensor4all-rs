use crate::matrixluci::{
    DenseLuKernel, DenseMatrixSource, LazyBlockRookKernel, LazyMatrixSource, PivotKernel,
    PivotKernelOptions,
};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

type LazyDenseSource = LazyMatrixSource<f64, Box<dyn Fn(&[usize], &[usize], &mut [f64])>>;

fn unique_pivot_test_matrix() -> Vec<f64> {
    vec![
        9.0, 0.2, 0.3, 0.4, //
        0.1, 8.0, 0.2, 0.3, //
        0.2, 0.1, 7.0, 0.2, //
        0.3, 0.2, 0.1, 6.0, //
    ]
}

fn dense_to_lazy(data: Vec<f64>, nrows: usize, ncols: usize) -> LazyDenseSource {
    LazyMatrixSource::new(
        nrows,
        ncols,
        Box::new(move |rows, cols, out: &mut [f64]| {
            for (j, &col) in cols.iter().enumerate() {
                for (i, &row) in rows.iter().enumerate() {
                    out[i + rows.len() * j] = data[row + nrows * col];
                }
            }
        }),
    )
}

#[test]
fn lazy_block_rook_kernel_matches_dense_kernel_on_unique_pivot_matrix() {
    let data = unique_pivot_test_matrix();
    let dense = DenseMatrixSource::from_column_major(&data, 4, 4);
    let lazy = dense_to_lazy(data.clone(), 4, 4);

    let dense_out = DenseLuKernel
        .factorize(&dense, &PivotKernelOptions::no_truncation())
        .unwrap();
    let lazy_out = LazyBlockRookKernel
        .factorize(&lazy, &PivotKernelOptions::no_truncation())
        .unwrap();

    assert_eq!(lazy_out.row_indices, dense_out.row_indices);
    assert_eq!(lazy_out.col_indices, dense_out.col_indices);
    assert_eq!(lazy_out.rank, dense_out.rank);
    assert_eq!(lazy_out.pivot_errors, dense_out.pivot_errors);
}

#[test]
fn lazy_block_rook_kernel_matches_dense_kernel_for_abs_tol_stop() {
    let data = unique_pivot_test_matrix();
    let dense = DenseMatrixSource::from_column_major(&data, 4, 4);
    let lazy = dense_to_lazy(data.clone(), 4, 4);
    let options = PivotKernelOptions {
        abs_tol: 6.5,
        ..PivotKernelOptions::default()
    };

    let dense_out = DenseLuKernel.factorize(&dense, &options).unwrap();
    let lazy_out = LazyBlockRookKernel.factorize(&lazy, &options).unwrap();

    assert_eq!(lazy_out.row_indices, dense_out.row_indices);
    assert_eq!(lazy_out.col_indices, dense_out.col_indices);
    assert_eq!(lazy_out.rank, dense_out.rank);
    assert_eq!(lazy_out.pivot_errors, dense_out.pivot_errors);
}

#[test]
fn lazy_block_rook_kernel_avoids_full_matrix_request() {
    let data = unique_pivot_test_matrix();
    let max_requested = Arc::new(AtomicUsize::new(0));
    let lazy = LazyMatrixSource::new(4, 4, {
        let max_requested = max_requested.clone();
        move |rows, cols, out: &mut [f64]| {
            max_requested.fetch_max(rows.len() * cols.len(), Ordering::SeqCst);
            for (j, &col) in cols.iter().enumerate() {
                for (i, &row) in rows.iter().enumerate() {
                    out[i + rows.len() * j] = data[row + 4 * col];
                }
            }
        }
    });

    let out = LazyBlockRookKernel
        .factorize(&lazy, &PivotKernelOptions::no_truncation())
        .unwrap();

    assert_eq!(out.rank, 4);
    assert!(max_requested.load(Ordering::SeqCst) < 16);
}
