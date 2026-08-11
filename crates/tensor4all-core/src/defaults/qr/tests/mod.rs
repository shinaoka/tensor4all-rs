use super::*;
use crate::index::DefaultIndex as Index;
use crate::unfold_split;
use num_complex::Complex64;
use tensor4all_tensorbackend::native_tensor_primal_to_dense_col_major;

#[test]
fn compute_retained_rank_qr_from_dense_truncates_and_keeps_one() {
    let retained =
        compute_retained_rank_qr_from_dense(&[3.0, 0.0, 1.0, 1.0e-14], 2, 2, 1.0e-10).unwrap();
    assert_eq!(retained, 1);

    let retained_zero =
        compute_retained_rank_qr_from_dense(&[0.0, 0.0, 0.0, 0.0], 2, 2, 1.0).unwrap();
    assert_eq!(retained_zero, 1);
}

#[test]
fn compute_retained_rank_qr_from_dense_handles_empty_and_complex_dense() {
    assert_eq!(
        compute_retained_rank_qr_from_dense::<Complex64>(&[], 0, 2, 1.0e-12).unwrap(),
        1
    );

    assert_eq!(
        compute_retained_rank_qr_from_dense(
            &[
                Complex64::new(2.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0e-14, 0.0),
            ],
            2,
            2,
            1.0e-10,
        )
        .unwrap(),
        1
    );
}

#[test]
fn unfold_split_preserves_column_major_linearization_with_unit_dims() {
    let i1 = Index::new_dyn(1);
    let i2 = Index::new_dyn(2);
    let i3 = Index::new_dyn(2);
    let tensor = TensorDynLen::from_dense(
        vec![i1.clone(), i2.clone(), i3.clone()],
        vec![1.0, 2.0, 3.0, 4.0],
    )
    .unwrap();

    let (matrix, _, m, n, _, _) = unfold_split(&tensor, &[i2, i3]).unwrap();
    assert_eq!((m, n), (4, 1));
    assert_eq!(
        native_tensor_primal_to_dense_col_major::<f64>(&matrix).unwrap(),
        vec![1.0, 2.0, 3.0, 4.0]
    );
}

#[test]
fn set_default_qr_rtol_rejects_invalid_values() {
    let original = default_qr_rtol();
    assert!(set_default_qr_rtol(f64::NAN).is_err());
    assert!(set_default_qr_rtol(-1.0).is_err());
    set_default_qr_rtol(original).unwrap();
}

#[test]
fn qr_options_report_rtol_and_default_roundtrips() {
    let original = default_qr_rtol();
    let options = QrOptions::new().with_rtol(1.0e-7);
    assert_eq!(options.rtol, Some(1.0e-7));
    assert_eq!(QrOptions::new().rtol, None);

    set_default_qr_rtol(1.0e-9).unwrap();
    assert_eq!(default_qr_rtol(), 1.0e-9);
    set_default_qr_rtol(original).unwrap();
}

#[test]
fn qr_with_invalid_rtol_is_rejected_before_linalg() {
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(2);
    let tensor = TensorDynLen::from_dense(vec![i.clone(), j.clone()], vec![0.0; 4]).unwrap();

    let nan = qr_with::<f64>(
        &tensor,
        std::slice::from_ref(&i),
        &QrOptions::new().with_rtol(f64::NAN),
    );
    assert!(matches!(nan, Err(QrError::InvalidRtol(v)) if v.is_nan()));

    let negative = qr_with::<f64>(
        &tensor,
        std::slice::from_ref(&i),
        &QrOptions::new().with_rtol(-1.0),
    );
    assert!(matches!(negative, Err(QrError::InvalidRtol(v)) if v == -1.0));
}

#[test]
fn qr_with_native_truncation_reduces_bond_dimension() {
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(2);
    let mut data = vec![0.0; 4];
    data[0] = 1.0;
    data[3] = 1.0e-14;
    let tensor = TensorDynLen::from_dense(vec![i.clone(), j.clone()], data).unwrap();

    let (q, r) = qr_with::<f64>(
        &tensor,
        std::slice::from_ref(&i),
        &QrOptions::new().with_rtol(1.0e-10),
    )
    .unwrap();
    assert_eq!(q.dims(), vec![2, 1]);
    assert_eq!(r.dims(), vec![1, 2]);
}

#[test]
fn qr_with_complex_fallback_truncation_reduces_bond_dimension() {
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(2);
    let mut data = vec![Complex64::new(0.0, 0.0); 4];
    data[0] = Complex64::new(1.0, 0.0);
    data[3] = Complex64::new(1.0e-14, 0.0);
    let tensor = TensorDynLen::from_dense(vec![i.clone(), j.clone()], data).unwrap();

    let (q, r) = qr_with::<Complex64>(
        &tensor,
        std::slice::from_ref(&i),
        &QrOptions::new().with_rtol(1.0e-10),
    )
    .unwrap();
    assert_eq!(q.dims(), vec![2, 1]);
    assert_eq!(r.dims(), vec![1, 2]);
}

/// Helper: build an upper-triangular dense buffer for low-level QR tests.
fn make_upper_triangular(nrows: usize, ncols: usize, entries: &[(usize, usize, f64)]) -> Vec<f64> {
    let mut data = vec![0.0; nrows * ncols];
    for &(i, j, value) in entries {
        data[i + j * nrows] = value;
    }
    data
}

fn retained_rank_from_f64(data: Vec<f64>, k: usize, n: usize, rtol: f64) -> usize {
    compute_retained_rank_qr_from_dense(&data, k, n, rtol)
        .expect("row-norm retained-rank helper should accept dense column-major f64 buffers")
}

fn retained_rank_from_c64(data: Vec<Complex64>, k: usize, n: usize, rtol: f64) -> usize {
    compute_retained_rank_qr_from_dense(&data, k, n, rtol)
        .expect("row-norm retained-rank helper should accept dense column-major c64 buffers")
}

#[test]
fn test_retained_rank_zero_diagonal_nonzero_offdiag() {
    // 3×4 R with zero diagonal at row 1 but nonzero off-diag
    // R = [[10, 1, 1, 1],
    //      [ 0, 0, 5, 5],   ← diagonal=0, but row norm = sqrt(50) ≈ 7.07
    //      [ 0, 0, 0, 1]]
    let r = make_upper_triangular(
        3,
        4,
        &[
            (0, 0, 10.0),
            (0, 1, 1.0),
            (0, 2, 1.0),
            (0, 3, 1.0),
            (1, 2, 5.0),
            (1, 3, 5.0),
            (2, 3, 1.0),
        ],
    );
    // rtol=1e-15: all rows should be retained
    assert_eq!(retained_rank_from_f64(r, 3, 4, 1e-15), 3);
}

#[test]
fn test_retained_rank_all_zero_rows() {
    // R with only row 0 non-zero
    let r = make_upper_triangular(3, 3, &[(0, 0, 5.0), (0, 1, 3.0), (0, 2, 1.0)]);
    assert_eq!(retained_rank_from_f64(r, 3, 3, 1e-15), 1);
}

#[test]
fn test_retained_rank_full_rank() {
    // Fully non-degenerate upper triangular
    let r = make_upper_triangular(
        3,
        3,
        &[
            (0, 0, 10.0),
            (0, 1, 1.0),
            (0, 2, 1.0),
            (1, 1, 8.0),
            (1, 2, 1.0),
            (2, 2, 6.0),
        ],
    );
    assert_eq!(retained_rank_from_f64(r, 3, 3, 1e-15), 3);
}

#[test]
fn test_retained_rank_rtol_truncation() {
    let r = make_upper_triangular(
        3,
        3,
        &[
            (0, 0, 10.0),
            (0, 1, 0.5),
            (0, 2, 0.1),
            (1, 1, 0.01),
            (2, 2, 0.001),
        ],
    );
    assert_eq!(retained_rank_from_f64(r.clone(), 3, 3, 0.01), 1);
    assert_eq!(retained_rank_from_f64(r, 3, 3, 1e-4), 2);
}

#[test]
fn test_retained_rank_zero_matrix() {
    let r = vec![0.0; 9];
    assert_eq!(retained_rank_from_f64(r, 3, 3, 1e-15), 1);
}

#[test]
fn test_retained_rank_complex() {
    use num_complex::Complex64;
    let r = vec![
        Complex64::new(5.0, 3.0),
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 1.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(3.0, 4.0),
    ];
    // Row 1 has zero diagonal but norm = 5.0, should NOT be truncated
    assert_eq!(retained_rank_from_c64(r, 2, 3, 1e-15), 2);
}
