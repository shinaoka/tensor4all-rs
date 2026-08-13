use super::*;
use crate::einsum_helper::{
    einsum_tensors, matrix_times_col_vector, row_vector_times_matrix, tensor_to_col_major_vec,
    typed_tensor_from_col_major_slice, typed_tensor_reshape, EinsumScalar,
};
use crate::types::{tensor3_zeros, Tensor3};

#[test]
fn test_dot_constant() {
    let tt1 = SimpleTensorTrain::<f64>::constant(&[2, 3], 2.0);
    let tt2 = SimpleTensorTrain::<f64>::constant(&[2, 3], 3.0);

    let result = tt1.inner_product(&tt2).unwrap();

    // Each element product is 2.0 * 3.0 = 6.0
    // Sum over 2*3=6 elements: 6.0 * 6 = 36.0
    assert!((result - 36.0).abs() < 1e-10);
}

#[test]
fn test_dot_different_tensors() {
    let mut t0_a: Tensor3<f64> = tensor3_zeros(1, 3, 2);
    for s in 0..3 {
        for r in 0..2 {
            t0_a.set3(0, s, r, (s + r + 1) as f64);
        }
    }

    let mut t1_a: Tensor3<f64> = tensor3_zeros(2, 2, 1);
    for l in 0..2 {
        for s in 0..2 {
            t1_a.set3(l, s, 0, (l + s + 1) as f64);
        }
    }

    let tt_a = SimpleTensorTrain::new(vec![t0_a.clone(), t1_a.clone()]).unwrap();

    let mut t0_b: Tensor3<f64> = tensor3_zeros(1, 3, 1);
    for s in 0..3 {
        t0_b.set3(0, s, 0, (s + 1) as f64 * 0.5);
    }

    let mut t1_b: Tensor3<f64> = tensor3_zeros(1, 2, 1);
    for s in 0..2 {
        t1_b.set3(0, s, 0, (s + 2) as f64 * 0.3);
    }

    let tt_b = SimpleTensorTrain::new(vec![t0_b, t1_b]).unwrap();

    let dot_result = tt_a.inner_product(&tt_b).unwrap();

    // Compute expected value by brute force
    let mut expected = 0.0;
    for i0 in 0..3 {
        for i1 in 0..2 {
            let val_a = tt_a.evaluate(&[i0, i1]).unwrap();
            let val_b = tt_b.evaluate(&[i0, i1]).unwrap();
            expected += val_a * val_b;
        }
    }

    assert!(
        (dot_result - expected).abs() < 1e-10,
        "inner_product = {}, expected = {}",
        dot_result,
        expected
    );
}

#[test]
fn test_dot_length_mismatch() {
    let tt1 = SimpleTensorTrain::<f64>::constant(&[2, 3], 1.0);
    let tt2 = SimpleTensorTrain::<f64>::constant(&[2, 3, 2], 1.0);
    let err = tt1.inner_product(&tt2).unwrap_err();
    assert!(err.to_string().contains("different lengths"));
    assert!(err.to_string().contains("2 vs 3"));
}

#[test]
fn test_dot_site_dim_mismatch() {
    let tt1 = SimpleTensorTrain::<f64>::constant(&[2, 3], 1.0);
    let tt2 = SimpleTensorTrain::<f64>::constant(&[2, 4], 1.0);
    let err = tt1.inner_product(&tt2).unwrap_err();
    assert!(err
        .to_string()
        .contains("Site dimensions mismatch at site 1"));
    assert!(err.to_string().contains("3 vs 4"));
}

#[test]
fn test_dot_site_dim_mismatch_middle() {
    // Mismatch at site 1 (not site 0) to cover the inner loop error path
    let mut t0: Tensor3<f64> = tensor3_zeros(1, 2, 1);
    t0.set3(0, 0, 0, 1.0);
    t0.set3(0, 1, 0, 1.0);

    let mut t1_a: Tensor3<f64> = tensor3_zeros(1, 3, 1);
    for s in 0..3 {
        t1_a.set3(0, s, 0, 1.0);
    }
    let mut t1_b: Tensor3<f64> = tensor3_zeros(1, 4, 1);
    for s in 0..4 {
        t1_b.set3(0, s, 0, 1.0);
    }

    let tt_a = SimpleTensorTrain::new(vec![t0.clone(), t1_a]).unwrap();
    let tt_b = SimpleTensorTrain::new(vec![t0, t1_b]).unwrap();
    let err = tt_a.inner_product(&tt_b).unwrap_err();
    assert!(err
        .to_string()
        .contains("Site dimensions mismatch at site 1"));
    assert!(err.to_string().contains("3 vs 4"));
}

#[test]
fn test_dot_empty() {
    let tt1 = SimpleTensorTrain::<f64>::from_tensors_unchecked(Vec::new());
    let tt2 = SimpleTensorTrain::<f64>::from_tensors_unchecked(Vec::new());
    let result = tt1.inner_product(&tt2).unwrap();
    assert!((result - 0.0).abs() < 1e-15);
}

fn test_dot_generic<T: TTScalar + tensor4all_tcicore::Scalar + Default + EinsumScalar>() {
    let tt1 = SimpleTensorTrain::<T>::constant(&[2, 3], T::from_f64(2.0));
    let tt2 = SimpleTensorTrain::<T>::constant(&[2, 3], T::from_f64(3.0));

    let result = tt1.inner_product(&tt2).unwrap();
    // Each element: 2*3 = 6, sum over 2*3=6 elements = 36
    assert!(
        (result - T::from_f64(36.0)).abs_sq().sqrt() < 1e-10,
        "inner_product product mismatch"
    );
}

#[test]
fn test_dot_f64() {
    test_dot_generic::<f64>();
}

#[test]
fn test_dot_c64() {
    test_dot_generic::<num_complex::Complex64>();
}

#[test]
fn test_dot_convenience_function() {
    let tt1 = SimpleTensorTrain::<f64>::constant(&[2, 3], 2.0);
    let tt2 = SimpleTensorTrain::<f64>::constant(&[2, 3], 3.0);
    let result = inner_product(&tt1, &tt2).unwrap();
    assert!((result - 36.0).abs() < 1e-10);
}

#[test]
fn test_contraction_options_default_values() {
    let opts = ContractionOptions::default();
    assert!((opts.tolerance - 1e-12).abs() < 1e-15);
    assert_eq!(opts.max_bond_dim, None);
    assert!(matches!(opts.method, CompressionMethod::LU));
}

#[test]
fn test_contraction_helper_error_formats_context() {
    let err = contraction_helper_error("while contracting", "backend rejected labels");
    let message = err.to_string();
    assert!(message.contains("while contracting"));
    assert!(message.contains("backend rejected labels"));
}

#[test]
fn test_dot_three_sites() {
    // Test inner_product product with 3 sites to exercise the inner loop more fully
    let tt1 = SimpleTensorTrain::<f64>::constant(&[2, 3, 2], 1.0);
    let tt2 = SimpleTensorTrain::<f64>::constant(&[2, 3, 2], 2.0);
    let result = tt1.inner_product(&tt2).unwrap();
    // Each element: 1*2 = 2, total elements = 2*3*2 = 12, sum = 24
    assert!((result - 24.0).abs() < 1e-10);
}

#[test]
fn test_einsum_tensors_matmul() {
    let a = typed_tensor_from_col_major_slice(&[1.0, 3.0, 2.0, 4.0], &[2, 2]).unwrap();
    let b = typed_tensor_from_col_major_slice(&[5.0, 7.0, 6.0, 8.0], &[2, 2]).unwrap();

    let c = einsum_tensors("ij,jk->ik", &[&a, &b]).unwrap();
    assert_eq!(tensor_to_col_major_vec(&c), &[19.0, 43.0, 22.0, 50.0]);
}

#[test]
fn test_einsum_tensors_reports_backend_error() {
    let a = typed_tensor_from_col_major_slice(&[1.0, 3.0, 2.0, 4.0], &[2, 2]).unwrap();

    let err = einsum_tensors("ij,jk->ik", &[&a]).unwrap_err();
    assert!(err.to_string().contains("einsum failed"));
    assert!(err.to_string().contains("ij,jk->ik"));
}

#[test]
fn test_row_vector_times_matrix_matches_mat_vec() {
    // Column-major (2 x 3): M[r, c] at r + 2*c.
    let v = [1.0, -2.0];
    let m = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let out = row_vector_times_matrix::<f64>(&v, &m, 2, 3).unwrap();
    // out[c] = v0*m[0,c] + v1*m[1,c]
    assert_eq!(
        out,
        vec![
            1.0 * 1.0 + (-2.0) * 2.0,
            1.0 * 3.0 + (-2.0) * 4.0,
            1.0 * 5.0 + (-2.0) * 6.0
        ]
    );
}

#[test]
fn test_matrix_times_col_vector_matches_mat_vec() {
    // Column-major (2 x 3): M[r, c] at r + 2*c.
    let m = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let v = [1.0, -2.0, 3.0];
    let out = matrix_times_col_vector::<f64>(&m, 2, 3, &v).unwrap();
    // out[r] = sum_c M[r,c]*v[c]
    assert_eq!(
        out,
        vec![
            1.0 * 1.0 + 3.0 * (-2.0) + 5.0 * 3.0,
            2.0 * 1.0 + 4.0 * (-2.0) + 6.0 * 3.0
        ]
    );
}

#[test]
fn test_row_vector_times_matrix_reports_shape_error() {
    let err = row_vector_times_matrix::<f64>(&[1.0], &[1.0, 2.0, 3.0, 4.0], 2, 2).unwrap_err();

    assert!(err.to_string().contains("row vector length mismatch"));
    assert!(err.to_string().contains("expected 2"));
    assert!(err.to_string().contains("got 1"));
}

#[test]
fn test_einsum_helper_error_paths_report_context() {
    let err = typed_tensor_from_col_major_slice::<f64>(&[], &[usize::MAX, 2]).unwrap_err();
    assert!(err.to_string().contains("shape element count overflow"));

    let err = typed_tensor_from_col_major_slice::<f64>(&[1.0, 2.0], &[3]).unwrap_err();
    assert!(err.to_string().contains("tensor data length mismatch"));
    assert!(err.to_string().contains("expected 3"));
    assert!(err.to_string().contains("got 2"));

    let tensor = typed_tensor_from_col_major_slice::<f64>(&[1.0, 2.0], &[2]).unwrap();
    let err = typed_tensor_reshape(&tensor, &[3]).unwrap_err();
    assert!(err
        .to_string()
        .contains("tensor reshape element count mismatch"));

    let err = row_vector_times_matrix::<f64>(&[1.0, 2.0], &[1.0, 2.0, 3.0], 2, 2).unwrap_err();
    assert!(err
        .to_string()
        .contains("row vector times matrix length mismatch"));

    let err = matrix_times_col_vector::<f64>(&[1.0, 2.0, 3.0], 2, 2, &[1.0, 2.0]).unwrap_err();
    assert!(err
        .to_string()
        .contains("matrix times column vector length mismatch"));

    let err = matrix_times_col_vector::<f64>(&[1.0, 2.0, 3.0, 4.0], 2, 2, &[1.0]).unwrap_err();
    assert!(err.to_string().contains("column vector length mismatch"));
}
