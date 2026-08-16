use crate::matrixluci::{CandidateMatrixSource, DenseMatrixSource, LazyMatrixSource};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

#[test]
fn dense_source_get_block_is_column_major() {
    let src = DenseMatrixSource::from_column_major(&[1.0, 3.0, 2.0, 4.0], 2, 2);
    let mut out = [0.0; 4];
    src.get_block(&[0, 1], &[0, 1], &mut out);
    assert_eq!(out, [1.0, 3.0, 2.0, 4.0]);
}

#[test]
fn dense_source_get_block_uses_cross_product_for_noncontiguous_indices() {
    let src = DenseMatrixSource::from_column_major(&[1.0, 4.0, 2.0, 5.0, 3.0, 6.0], 2, 3);
    let mut out = [0.0; 4];
    src.get_block(&[1, 0], &[2, 0], &mut out);
    assert_eq!(out, [6.0, 3.0, 4.0, 1.0]);
}

#[test]
fn scalar_get_delegates_to_get_block() {
    let src = DenseMatrixSource::from_column_major(&[1.0, 3.0, 2.0, 4.0], 2, 2);
    assert_eq!(src.get(1, 0), 3.0);
}

#[test]
fn lazy_source_get_block_batches_requests() {
    let calls = Arc::new(AtomicUsize::new(0));
    let src = LazyMatrixSource::new(4, 4, {
        let calls = calls.clone();
        move |rows, cols, out: &mut [f64]| {
            calls.fetch_add(1, Ordering::SeqCst);
            for (j, &col) in cols.iter().enumerate() {
                for (i, &row) in rows.iter().enumerate() {
                    out[i + rows.len() * j] = (10 * row + col) as f64;
                }
            }
        }
    });
    let mut out = [0.0; 4];
    src.get_block(&[0, 2], &[1, 3], &mut out);
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert_eq!(out, [1.0, 21.0, 3.0, 23.0]);
}

#[test]
fn lazy_source_scalar_get_delegates_to_block_callback() {
    let calls = Arc::new(AtomicUsize::new(0));
    let src = LazyMatrixSource::new(3, 3, {
        let calls = calls.clone();
        move |rows, cols, out: &mut [f64]| {
            calls.fetch_add(1, Ordering::SeqCst);
            for (j, &col) in cols.iter().enumerate() {
                for (i, &row) in rows.iter().enumerate() {
                    out[i + rows.len() * j] = (row + col) as f64;
                }
            }
        }
    });

    assert_eq!(src.get(2, 1), 3.0);
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}
