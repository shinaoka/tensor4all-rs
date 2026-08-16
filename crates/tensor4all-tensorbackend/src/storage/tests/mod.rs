use super::*;

/// Helper to extract f64 data from storage
fn extract_f64(storage: &Storage) -> Vec<f64> {
    match storage.repr() {
        StorageRepr::F64(ds) => ds.data().to_vec(),
        _ => panic!("Expected f64 storage"),
    }
}

/// Helper to extract Complex64 data from storage
fn extract_c64(storage: &Storage) -> Vec<Complex64> {
    match storage.repr() {
        StorageRepr::C64(ds) => ds.data().to_vec(),
        _ => panic!("Expected c64 storage"),
    }
}

#[test]
fn referenced_strided_entries_ignore_unreferenced_stride_gaps() {
    let storage = Storage::new_structured(
        vec![1.0_f64, 2.0, f64::NAN, 3.0, 4.0],
        vec![2, 2],
        vec![3, 1],
        vec![0, 1],
    )
    .unwrap();

    assert_eq!(storage.payload_nonfinite_flags(), (false, false));
    assert_eq!(storage.sum::<f64>(), 10.0);
    assert_eq!(storage.max_abs(), 4.0);
    assert_eq!(storage.scalar_at(&[1, 0]).unwrap().real(), 3.0);
    assert_eq!(storage.scalar_at(&[1, 1]).unwrap().real(), 4.0);

    let complex_storage = Storage::new_structured(
        vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(f64::NAN, 0.0),
            Complex64::new(3.0, 0.0),
            Complex64::new(4.0, 0.0),
        ],
        vec![2, 2],
        vec![3, 1],
        vec![0, 1],
    )
    .unwrap();
    assert_eq!(complex_storage.payload_nonfinite_flags(), (false, false));
    assert_eq!(
        complex_storage.sum::<Complex64>(),
        Complex64::new(10.0, 0.0)
    );
    assert_eq!(complex_storage.max_abs(), 4.0);
}

#[test]
fn scalar_at_returns_the_stored_dynamic_scalar_kind() {
    let real = Storage::from_dense_col_major(vec![1.0_f64, 2.0], &[2]).unwrap();
    let complex = Storage::from_dense_col_major(
        vec![Complex64::new(1.0, -2.0), Complex64::new(3.0, 4.0)],
        &[2],
    )
    .unwrap();

    assert_eq!(real.scalar_at(&[1]).unwrap().real(), 2.0);
    assert_eq!(
        complex.scalar_at(&[0]).unwrap().as_c64(),
        Some(Complex64::new(1.0, -2.0))
    );
}

#[test]
fn structured_storage_rejects_payload_length_product_overflow() {
    let error =
        StructuredStorage::<u8>::new(Vec::new(), vec![usize::MAX, 2], vec![0, 0], vec![0, 1])
            .unwrap_err();
    assert!(error.to_string().contains("payload length overflow"));

    let dense_error =
        Storage::from_dense_col_major(Vec::<f64>::new(), &[usize::MAX, 2]).unwrap_err();
    assert!(dense_error.to_string().contains("overflow"));
}

#[test]
fn to_dense_rejects_logical_product_overflow_fail_closed() {
    // axis_classes [0,0,0,0] make the logical dims [n,n,n,n]; n = 2^17 gives
    // a 1 MiB payload while n^4 overflows usize, so dense materialization
    // must fail closed instead of returning an empty buffer.
    let n = 1 << 17;
    let structured =
        StructuredStorage::<f64>::new(vec![1.0; n], vec![n], vec![1], vec![0, 0, 0, 0]).unwrap();
    let storage = Storage::from_repr(StorageRepr::F64(structured));
    let error = storage
        .to_dense_f64_col_major_vec(&[n, n, n, n])
        .unwrap_err();
    assert!(error.to_string().contains("overflow"));

    let structured = StructuredStorage::<Complex64>::new(
        vec![Complex64::new(1.0, 0.0); n],
        vec![n],
        vec![1],
        vec![0, 0, 0, 0],
    )
    .unwrap();
    let storage = Storage::from_repr(StorageRepr::C64(structured));
    assert!(storage.to_dense_c64_col_major_vec(&[n, n, n, n]).is_err());
}

#[test]
fn scalar_at_reads_compact_payload_coordinates_directly() {
    let storage = Storage::new_structured(
        vec![10.0_f64, 20.0, -1.0, 30.0, 40.0],
        vec![2, 2],
        vec![1, 3],
        vec![0, 1],
    )
    .unwrap();
    assert_eq!(storage.scalar_at(&[0, 0]).unwrap().real(), 10.0);
    assert_eq!(storage.scalar_at(&[0, 1]).unwrap().real(), 30.0);
    assert_eq!(storage.scalar_at(&[1, 0]).unwrap().real(), 20.0);
    assert_eq!(storage.scalar_at(&[1, 1]).unwrap().real(), 40.0);
    assert!(storage.scalar_at(&[2, 0]).is_err());
}

// ===== Type inspection tests =====

#[test]
fn test_is_f64() {
    let dense_f64 = Storage::from_dense_col_major(vec![1.0], &[1]).unwrap();
    let dense_c64 = Storage::from_dense_col_major(vec![Complex64::new(1.0, 0.0)], &[1]).unwrap();
    let diag_f64 = Storage::from_diag_col_major(vec![1.0], 2).unwrap();
    let diag_c64 = Storage::from_diag_col_major(vec![Complex64::new(1.0, 0.0)], 2).unwrap();

    assert!(dense_f64.is_f64());
    assert!(!dense_c64.is_f64());
    assert!(diag_f64.is_f64());
    assert!(!diag_c64.is_f64());
}

#[test]
fn test_is_c64() {
    let dense_f64 = Storage::from_dense_col_major(vec![1.0], &[1]).unwrap();
    let dense_c64 = Storage::from_dense_col_major(vec![Complex64::new(1.0, 0.0)], &[1]).unwrap();
    let diag_f64 = Storage::from_diag_col_major(vec![1.0], 2).unwrap();
    let diag_c64 = Storage::from_diag_col_major(vec![Complex64::new(1.0, 0.0)], 2).unwrap();

    assert!(!dense_f64.is_c64());
    assert!(dense_c64.is_c64());
    assert!(!diag_f64.is_c64());
    assert!(diag_c64.is_c64());
}

#[test]
fn test_is_complex() {
    let dense_f64 = Storage::from_dense_col_major(vec![1.0], &[1]).unwrap();
    let dense_c64 = Storage::from_dense_col_major(vec![Complex64::new(1.0, 0.0)], &[1]).unwrap();
    let diag_f64 = Storage::from_diag_col_major(vec![1.0], 2).unwrap();
    let diag_c64 = Storage::from_diag_col_major(vec![Complex64::new(1.0, 0.0)], 2).unwrap();

    // is_complex is an alias for is_c64
    assert!(!dense_f64.is_complex());
    assert!(dense_c64.is_complex());
    assert!(!diag_f64.is_complex());
    assert!(diag_c64.is_complex());
}

// ===== Storage-level tests for is_diag =====

#[test]
fn test_storage_is_diag() {
    let dense = Storage::from_dense_col_major(vec![1.0], &[1]).unwrap();
    let diag = Storage::from_diag_col_major(vec![1.0], 2).unwrap();
    assert!(!dense.is_diag());
    assert!(diag.is_diag());
}

#[test]
fn storage_kind_and_metadata_accessors_cover_dense_diag_and_structured() {
    let dense = Storage::from_dense_col_major(vec![1.0_f64, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    assert_eq!(dense.storage_kind(), StorageKind::Dense);
    assert_eq!(dense.logical_dims(), vec![2, 2]);
    assert_eq!(dense.logical_rank(), 2);
    assert_eq!(dense.payload_dims(), &[2, 2]);
    assert_eq!(dense.payload_strides(), &[1, 2]);
    assert_eq!(dense.axis_classes(), &[0, 1]);
    assert_eq!(dense.payload_len(), 4);
    assert_eq!(
        dense.payload_f64_col_major_vec().unwrap(),
        vec![1.0, 2.0, 3.0, 4.0]
    );

    let diag = Storage::from_diag_col_major(vec![10.0_f64, 20.0], 2).unwrap();
    assert_eq!(diag.storage_kind(), StorageKind::Diagonal);
    assert_eq!(diag.logical_dims(), vec![2, 2]);
    assert_eq!(diag.payload_dims(), &[2]);
    assert_eq!(diag.axis_classes(), &[0, 0]);
    assert_eq!(diag.payload_f64_col_major_vec().unwrap(), vec![10.0, 20.0]);

    let structured = Storage::new_structured(
        vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![2, 3],
        vec![1, 2],
        vec![0, 1, 0],
    )
    .unwrap();
    assert_eq!(structured.storage_kind(), StorageKind::Structured);
    assert_eq!(structured.logical_dims(), vec![2, 3, 2]);
    assert_eq!(structured.payload_dims(), &[2, 3]);
    assert_eq!(structured.payload_strides(), &[1, 2]);
    assert_eq!(structured.axis_classes(), &[0, 1, 0]);
}

#[test]
fn storage_payload_c64_readback_is_interpreted_as_payload_not_logical_dense() {
    let data = vec![Complex64::new(1.0, -1.0), Complex64::new(2.0, -2.0)];
    let storage = Storage::from_diag_col_major(data.clone(), 2).unwrap();
    assert_eq!(storage.storage_kind(), StorageKind::Diagonal);
    assert_eq!(storage.payload_c64_col_major_vec().unwrap(), data);
    match storage.payload_f64_col_major_vec().unwrap_err() {
        StorageError::ScalarKindMismatch {
            expected, actual, ..
        } => {
            assert_eq!(expected, "f64");
            assert_eq!(actual, "Complex64");
        }
        err => panic!("unexpected error: {err}"),
    }
}

#[test]
fn payload_f64_reports_scalar_kind_mismatch() {
    let storage = Storage::from_dense_col_major(vec![Complex64::new(1.0, 0.0)], &[1]).unwrap();

    match storage.payload_f64_col_major_vec().unwrap_err() {
        StorageError::ScalarKindMismatch {
            expected, actual, ..
        } => {
            assert_eq!(expected, "f64");
            assert_eq!(actual, "Complex64");
        }
        err => panic!("unexpected error: {err}"),
    }
}

#[test]
fn try_add_reports_length_mismatch() {
    let a = Storage::from_dense_col_major(vec![1.0_f64, 2.0], &[2]).unwrap();
    let b = Storage::from_dense_col_major(vec![3.0_f64], &[1]).unwrap();

    match a.try_add(&b).unwrap_err() {
        StorageError::LengthMismatch {
            operation,
            left,
            right,
        } => {
            assert_eq!(operation, "addition");
            assert_eq!(left, 2);
            assert_eq!(right, 1);
        }
        err => panic!("unexpected error: {err}"),
    }
}

#[test]
fn dense_storage_materialization_reports_dimension_mismatch() {
    let storage = Storage::from_diag_col_major(vec![1.0_f64, 2.0], 2).unwrap();

    let err = storage.to_dense_storage(&[2, 3]).unwrap_err();

    assert!(matches!(err, StorageError::InvalidStructuredStorage(_)));
    assert!(err.to_string().contains("logical dims"));
}

#[test]
fn combine_to_complex_reports_invalid_inputs() {
    let real = Storage::from_dense_col_major(vec![1.0_f64, 2.0], &[2]).unwrap();
    let short_imag = Storage::from_dense_col_major(vec![3.0_f64], &[1]).unwrap();
    let complex_imag =
        Storage::from_dense_col_major(vec![Complex64::new(0.0, 1.0); 2], &[2]).unwrap();

    assert!(matches!(
        Storage::combine_to_complex(&real, &short_imag).unwrap_err(),
        StorageError::LengthMismatch {
            operation: "combine_to_complex",
            left: 2,
            right: 1,
        }
    ));
    assert!(matches!(
        Storage::combine_to_complex(&real, &complex_imag).unwrap_err(),
        StorageError::OperationNotSupported {
            operation: "combine_to_complex",
            left: "f64",
            right: "Complex64",
        }
    ));
}

#[test]
fn contract_storage_reports_invalid_axes_and_dimensions() {
    let a = Storage::from_dense_col_major(vec![1.0_f64, 2.0], &[2]).unwrap();
    let b = Storage::from_dense_col_major(vec![3.0_f64, 4.0, 5.0], &[3]).unwrap();

    let dim_err = contract_storage(&a, &[2], &[0], &b, &[3], &[0], &[]).unwrap_err();
    assert!(matches!(dim_err, StorageError::InvalidStructuredStorage(_)));
    assert!(dim_err.to_string().contains("contracted dimensions"));

    let axis_err = contract_storage(&a, &[2], &[1], &b, &[3], &[0], &[]).unwrap_err();
    assert!(matches!(
        axis_err,
        StorageError::InvalidStructuredStorage(_)
    ));
    assert!(axis_err.to_string().contains("axis"));
}

#[test]
fn structured_storage_rejects_noncanonical_axis_classes() {
    let err = StructuredStorage::<f64>::new(
        vec![1.0, 2.0, 3.0, 4.0],
        vec![2, 2],
        vec![1, 2],
        vec![1, 0, 0],
    )
    .unwrap_err();

    assert!(err.to_string().contains("canonical"));
}

#[test]
fn structured_storage_column_major_helpers_cover_contiguous_padded_and_empty_payloads() {
    let dense =
        StructuredStorage::from_dense_col_major(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])
            .unwrap();
    assert_eq!(dense.logical_dims(), vec![2, 3]);
    assert!(dense.is_dense());
    assert!(!dense.is_diag());
    assert_eq!(
        dense.dense_col_major_view_if_contiguous().unwrap(),
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    );
    assert_eq!(
        dense.payload_col_major_vec(),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    );

    let padded = StructuredStorage::new(
        vec![10.0, 20.0, -1.0, 30.0, 40.0, -1.0, 50.0, 60.0],
        vec![2, 3],
        vec![1, 3],
        vec![0, 1],
    )
    .unwrap();
    assert!(padded.dense_col_major_view_if_contiguous().is_none());
    assert_eq!(
        padded.payload_col_major_vec(),
        vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
    );

    let empty = StructuredStorage::from_dense_col_major(Vec::<f64>::new(), &[0, 3]).unwrap();
    assert!(empty.is_empty());
    assert_eq!(empty.payload_col_major_vec(), Vec::<f64>::new());
}

#[test]
fn structured_storage_permute_and_map_copy_preserve_metadata() {
    let storage = StructuredStorage::new(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![2, 3],
        vec![1, 2],
        vec![0, 1, 0],
    )
    .unwrap();
    assert_eq!(storage.logical_dims(), vec![2, 3, 2]);

    let permuted = storage.permute_logical_axes(&[0, 2, 1]).unwrap();
    assert_eq!(permuted.axis_classes(), &[0, 0, 1]);
    assert_eq!(permuted.logical_dims(), vec![2, 2, 3]);

    let mapped = permuted.map_copy(|x| x * 10.0);
    assert_eq!(mapped.data(), &[10.0, 20.0, 30.0, 40.0, 50.0, 60.0]);
    assert_eq!(mapped.payload_dims(), &[2, 3]);
    assert_eq!(mapped.strides(), &[1, 2]);
    assert_eq!(mapped.axis_classes(), &[0, 0, 1]);
}

#[test]
fn structured_storage_validates_payload_rank_and_required_len() {
    let rank_err =
        StructuredStorage::<f64>::new(vec![1.0, 2.0], vec![2], vec![1], vec![0, 1]).unwrap_err();
    assert!(rank_err.to_string().contains("payload rank"));

    let len_err = StructuredStorage::<f64>::new(vec![1.0, 2.0], vec![2, 2], vec![1, 3], vec![0, 1])
        .unwrap_err();
    assert!(len_err.to_string().contains("required len"));

    let scalar_diag = StructuredStorage::from_diag_col_major(vec![42.0], 0).unwrap();
    assert_eq!(scalar_diag.payload_dims(), &[] as &[usize]);
    assert_eq!(scalar_diag.logical_rank(), 0);
    assert!(scalar_diag.is_dense());
    assert!(!scalar_diag.is_diag());
    assert_eq!(scalar_diag.payload_col_major_vec(), vec![42.0]);
}

// ===== Storage len / is_empty =====

#[test]
fn test_storage_len_is_empty() {
    let dense = Storage::from_dense_col_major(vec![1.0, 2.0], &[2]).unwrap();
    assert_eq!(dense.len(), 2);
    assert!(!dense.is_empty());

    let diag = Storage::from_diag_col_major(vec![0.0f64; 0], 2).unwrap();
    assert_eq!(diag.len(), 0);
    assert!(diag.is_empty());
}

// ===== Storage zero-constructor tests =====

#[test]
fn test_storage_new_dense_f64() {
    let s = Storage::new_dense::<f64>(3).unwrap();
    assert_eq!(s.len(), 3);
    assert!(s.is_f64());
    let data = extract_f64(&s);
    assert_eq!(data, vec![0.0, 0.0, 0.0]);
}

#[test]
fn test_storage_new_dense_c64() {
    let s = Storage::new_dense::<Complex64>(2).unwrap();
    assert_eq!(s.len(), 2);
    assert!(s.is_c64());
}

#[test]
fn test_storage_from_dense_col_major_f64_zeros_with_shape() {
    let s = Storage::from_dense_col_major(vec![0.0; 6], &[2, 3]).unwrap();
    assert_eq!(s.len(), 6);
}

#[test]
fn test_storage_from_dense_col_major_c64_zeros_with_shape() {
    let s = Storage::from_dense_col_major(vec![Complex64::new(0.0, 0.0); 6], &[3, 2]).unwrap();
    assert_eq!(s.len(), 6);
}

// ===== SumFromStorage tests =====

#[test]
fn test_sum_from_storage_f64() {
    let s = Storage::from_dense_col_major(vec![1.0, 2.0, 3.0], &[3]).unwrap();
    let sum: f64 = f64::sum_from_storage(&s);
    assert!((sum - 6.0).abs() < 1e-10);
}

#[test]
fn test_sum_from_storage_diag_f64() {
    let s = Storage::from_diag_col_major(vec![10.0, 20.0], 2).unwrap();
    let sum: f64 = f64::sum_from_storage(&s);
    assert!((sum - 30.0).abs() < 1e-10);
}

#[test]
fn test_storage_sum_f64_method() {
    let s = Storage::from_dense_col_major(vec![1.0, 2.0, 3.0], &[3]).unwrap();
    assert!((s.sum::<f64>() - 6.0).abs() < 1e-10);
}

#[test]
fn test_storage_sum_c64_method() {
    let s = Storage::from_dense_col_major(
        vec![Complex64::new(1.0, 2.0), Complex64::new(3.0, 4.0)],
        &[2],
    )
    .unwrap();
    let sum = s.sum::<Complex64>();
    assert!((sum - Complex64::new(4.0, 6.0)).norm() < 1e-10);
}

#[test]
fn test_storage_max_abs_and_to_dense_storage_cover_complex_and_diag() {
    let dense_c64 = Storage::from_dense_col_major(
        vec![Complex64::new(3.0, 4.0), Complex64::new(1.0, -1.0)],
        &[2],
    )
    .unwrap();
    assert!((dense_c64.max_abs() - 5.0).abs() < 1e-10);
    match dense_c64.to_dense_storage(&[2]).unwrap().repr() {
        StorageRepr::C64(ds) => assert_eq!(
            ds.payload_col_major_vec().as_slice(),
            &[Complex64::new(3.0, 4.0), Complex64::new(1.0, -1.0)]
        ),
        other => panic!("expected C64, got {other:?}"),
    }

    let diag_c64 =
        Storage::from_diag_col_major(vec![Complex64::new(0.0, 2.0), Complex64::new(3.0, 4.0)], 2)
            .unwrap();
    assert!((diag_c64.max_abs() - 5.0).abs() < 1e-10);
    match diag_c64.to_dense_storage(&[2, 2]).unwrap().repr() {
        StorageRepr::C64(ds) => {
            assert_eq!(
                ds.payload_col_major_vec().as_slice(),
                &[
                    Complex64::new(0.0, 2.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(3.0, 4.0),
                ]
            );
        }
        other => panic!("expected C64, got {other:?}"),
    }
}

#[test]
fn test_storage_projection_promotion_and_conjugation_helpers() {
    let dense_c64 = Storage::from_dense_col_major(
        vec![Complex64::new(1.0, -2.0), Complex64::new(3.0, 4.0)],
        &[2],
    )
    .unwrap();
    match dense_c64.extract_real_part().repr() {
        StorageRepr::F64(ds) => {
            assert_eq!(ds.payload_col_major_vec().as_slice(), &[1.0, 3.0])
        }
        other => panic!("expected F64, got {other:?}"),
    }
    match dense_c64.extract_imag_part(&[2]).repr() {
        StorageRepr::F64(ds) => {
            assert_eq!(ds.payload_col_major_vec().as_slice(), &[-2.0, 4.0])
        }
        other => panic!("expected F64, got {other:?}"),
    }
    match dense_c64.conj().repr() {
        StorageRepr::C64(ds) => {
            assert_eq!(
                ds.payload_col_major_vec().as_slice(),
                &[Complex64::new(1.0, 2.0), Complex64::new(3.0, -4.0)]
            )
        }
        other => panic!("expected C64, got {other:?}"),
    }

    let diag_f64 = Storage::from_diag_col_major(vec![2.0, -1.0], 2).unwrap();
    match diag_f64.extract_imag_part(&[2, 2]).repr() {
        StorageRepr::F64(ds) => {
            assert_eq!(ds.payload_col_major_vec().as_slice(), &[0.0, 0.0])
        }
        other => panic!("expected F64, got {other:?}"),
    }
    match diag_f64.to_complex_storage().repr() {
        StorageRepr::C64(ds) => {
            assert_eq!(
                ds.payload_col_major_vec().as_slice(),
                &[Complex64::new(2.0, 0.0), Complex64::new(-1.0, 0.0)]
            )
        }
        other => panic!("expected C64, got {other:?}"),
    }
    let real = Storage::from_dense_col_major(vec![1.0, 2.0], &[2]).unwrap();
    let imag = Storage::from_dense_col_major(vec![0.5, -1.5], &[2]).unwrap();
    match Storage::combine_to_complex(&real, &imag).unwrap().repr() {
        StorageRepr::C64(ds) => {
            assert_eq!(
                ds.payload_col_major_vec().as_slice(),
                &[Complex64::new(1.0, 0.5), Complex64::new(2.0, -1.5)]
            )
        }
        other => panic!("expected C64, got {other:?}"),
    }
}

#[test]
fn test_storage_try_add_and_try_sub_cover_all_variants_and_errors() {
    let dense_f64_a = Storage::from_dense_col_major(vec![1.0, 2.0], &[2]).unwrap();
    let dense_f64_b = Storage::from_dense_col_major(vec![3.0, -1.0], &[2]).unwrap();
    match dense_f64_a.try_add(&dense_f64_b).unwrap().repr() {
        StorageRepr::F64(ds) => assert_eq!(ds.data(), &[4.0, 1.0]),
        other => panic!("expected F64, got {other:?}"),
    }
    match dense_f64_a.try_sub(&dense_f64_b).unwrap().repr() {
        StorageRepr::F64(ds) => assert_eq!(ds.data(), &[-2.0, 3.0]),
        other => panic!("expected F64, got {other:?}"),
    }

    let dense_c64_a = Storage::from_dense_col_major(
        vec![Complex64::new(1.0, 1.0), Complex64::new(0.0, -2.0)],
        &[2],
    )
    .unwrap();
    let dense_c64_b = Storage::from_dense_col_major(
        vec![Complex64::new(-1.0, 0.5), Complex64::new(3.0, 1.0)],
        &[2],
    )
    .unwrap();
    assert!(matches!(
        dense_c64_a.try_add(&dense_c64_b).unwrap().repr(),
        StorageRepr::C64(_)
    ));
    assert!(matches!(
        dense_c64_a.try_sub(&dense_c64_b).unwrap().repr(),
        StorageRepr::C64(_)
    ));

    let diag_f64_a = Storage::from_diag_col_major(vec![1.0, 2.0], 2).unwrap();
    let diag_f64_b = Storage::from_diag_col_major(vec![0.5, -3.0], 2).unwrap();
    assert!(matches!(
        diag_f64_a.try_add(&diag_f64_b).unwrap().repr(),
        StorageRepr::F64(_)
    ));
    assert!(matches!(
        diag_f64_a.try_sub(&diag_f64_b).unwrap().repr(),
        StorageRepr::F64(_)
    ));

    let diag_c64_a =
        Storage::from_diag_col_major(vec![Complex64::new(1.0, -1.0), Complex64::new(0.0, 2.0)], 2)
            .unwrap();
    let diag_c64_b =
        Storage::from_diag_col_major(vec![Complex64::new(0.5, 0.5), Complex64::new(-3.0, 1.0)], 2)
            .unwrap();
    assert!(matches!(
        diag_c64_a.try_add(&diag_c64_b).unwrap().repr(),
        StorageRepr::C64(_)
    ));
    assert!(matches!(
        diag_c64_a.try_sub(&diag_c64_b).unwrap().repr(),
        StorageRepr::C64(_)
    ));

    let mismatched_len = Storage::from_dense_col_major(vec![1.0], &[1]).unwrap();
    assert!(matches!(
        dense_f64_a.try_add(&mismatched_len).unwrap_err(),
        StorageError::LengthMismatch {
            operation: "addition",
            left: 2,
            right: 1
        }
    ));
    assert!(matches!(
        dense_f64_a.try_sub(&mismatched_len).unwrap_err(),
        StorageError::LengthMismatch {
            operation: "subtraction",
            left: 2,
            right: 1
        }
    ));

    let mismatched_type = Storage::from_dense_col_major(
        vec![Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)],
        &[2],
    )
    .unwrap();
    assert!(matches!(
        dense_f64_a.try_add(&mismatched_type).unwrap_err(),
        StorageError::OperationNotSupported {
            operation: "addition",
            left: "f64",
            right: "Complex64"
        }
    ));
    assert!(matches!(
        dense_f64_a.try_sub(&mismatched_type).unwrap_err(),
        StorageError::OperationNotSupported {
            operation: "subtraction",
            left: "f64",
            right: "Complex64"
        }
    ));
}

// ===== StructuredStorage permute_logical_axes assertion path =====

#[test]
fn test_structured_storage_permute_logical_axes_basic() {
    let storage = StructuredStorage::new(vec![1.0, 2.0], vec![2], vec![1], vec![0, 0]).unwrap();
    let permuted = storage.permute_logical_axes(&[1, 0]).unwrap();
    // Permuting a diagonal storage: axis_classes should be swapped but are the same
    assert_eq!(permuted.axis_classes(), &[0, 0]);
}

// ===== Storage::scale and Storage::axpby coverage =====

#[test]
fn test_storage_scale_f64() {
    let s = Storage::from_dense_col_major(vec![1.0, 2.0], &[2]).unwrap();
    let scaled = s.scale(&crate::AnyScalar::new_real(3.0));
    let data = extract_f64(&scaled);
    assert!((data[0] - 3.0).abs() < 1e-10);
    assert!((data[1] - 6.0).abs() < 1e-10);
}

#[test]
fn test_storage_axpby_dense_f64() {
    let a = Storage::from_dense_col_major(vec![1.0, 2.0], &[2]).unwrap();
    let b = Storage::from_dense_col_major(vec![3.0, 4.0], &[2]).unwrap();
    let result = a
        .axpby(
            &crate::AnyScalar::new_real(2.0),
            &b,
            &crate::AnyScalar::new_real(3.0),
        )
        .unwrap();
    let data = extract_f64(&result);
    // 2*1 + 3*3 = 11, 2*2 + 3*4 = 16
    assert!((data[0] - 11.0).abs() < 1e-10);
    assert!((data[1] - 16.0).abs() < 1e-10);
}

#[test]
fn test_storage_axpby_diag_f64() {
    let a = Storage::from_diag_col_major(vec![1.0, 2.0], 2).unwrap();
    let b = Storage::from_diag_col_major(vec![3.0, 4.0], 2).unwrap();
    let result = a
        .axpby(
            &crate::AnyScalar::new_real(2.0),
            &b,
            &crate::AnyScalar::new_real(3.0),
        )
        .unwrap();
    match result.repr() {
        StorageRepr::F64(d) => {
            assert!((d.data()[0] - 11.0).abs() < 1e-10);
            assert!((d.data()[1] - 16.0).abs() < 1e-10);
        }
        _ => panic!("Expected F64"),
    }
}

#[test]
fn test_storage_axpby_complex_promotion() {
    let a = Storage::from_dense_col_major(vec![1.0, 2.0], &[2]).unwrap();
    let b = Storage::from_dense_col_major(vec![3.0, 4.0], &[2]).unwrap();
    let result = a
        .axpby(
            &crate::AnyScalar::new_complex(1.0, 1.0),
            &b,
            &crate::AnyScalar::new_real(1.0),
        )
        .unwrap();
    let data = extract_c64(&result);
    // (1+i)*1 + 1*3 = 4+i, (1+i)*2 + 1*4 = 6+2i
    assert!((data[0] - Complex64::new(4.0, 1.0)).norm() < 1e-10);
    assert!((data[1] - Complex64::new(6.0, 2.0)).norm() < 1e-10);
}

// ===== Storage::new_diag constructors =====

#[test]
fn test_storage_new_diag_f64() {
    let s = Storage::new_diag::<f64>(vec![1.0, 2.0, 3.0]).unwrap();
    assert_eq!(s.len(), 3);
    assert!(s.is_f64());
    assert!(s.is_diag());
}

#[test]
fn test_storage_new_diag_c64() {
    let s =
        Storage::new_diag::<Complex64>(vec![Complex64::new(1.0, 2.0), Complex64::new(3.0, 4.0)])
            .unwrap();
    assert_eq!(s.len(), 2);
    assert!(s.is_c64());
    assert!(s.is_diag());
}

// ===== Storage is_dense / is_diag for structured variants =====

#[test]
fn test_storage_is_dense_structured() {
    let dense = Storage::from_dense_col_major(vec![1.0, 2.0], &[2]).unwrap();
    assert!(dense.is_dense());
    assert!(!dense.is_diag());

    let diag = Storage::from_diag_col_major(vec![1.0, 2.0], 2).unwrap();
    assert!(!diag.is_dense());
    assert!(diag.is_diag());
}

#[test]
fn test_storage_is_dense_structured_c64() {
    let dense = Storage::from_dense_col_major(vec![Complex64::new(1.0, 0.0)], &[1]).unwrap();
    assert!(dense.is_dense());
    assert!(!dense.is_diag());

    let diag = Storage::from_diag_col_major(vec![Complex64::new(1.0, 0.0)], 2).unwrap();
    assert!(!diag.is_dense());
    assert!(diag.is_diag());
}

// ===== Storage from_diag col_major constructors =====

#[test]
fn test_storage_from_diag_f64_col_major() {
    let s = Storage::from_diag_col_major(vec![5.0, 10.0], 3).unwrap();
    assert!(s.is_diag());
    assert!(s.is_f64());
}

#[test]
fn test_storage_from_diag_c64_col_major() {
    let s =
        Storage::from_diag_col_major(vec![Complex64::new(5.0, 1.0), Complex64::new(10.0, 2.0)], 3)
            .unwrap();
    assert!(s.is_diag());
    assert!(s.is_c64());
}

// ===== Storage max_abs for structured variants =====

#[test]
fn test_storage_max_abs_structured() {
    let s = Storage::from_dense_col_major(vec![1.0, -5.0, 3.0], &[3]).unwrap();
    assert!((s.max_abs() - 5.0).abs() < 1e-10);

    let s_c64 = Storage::from_dense_col_major(
        vec![Complex64::new(3.0, 4.0), Complex64::new(1.0, 0.0)],
        &[2],
    )
    .unwrap();
    assert!((s_c64.max_abs() - 5.0).abs() < 1e-10);
}

#[test]
fn storage_max_abs_propagates_nan_and_infinity() {
    let nan = Storage::from_dense_col_major(vec![1.0, f64::NAN], &[2]).unwrap();
    assert!(nan.max_abs().is_nan());

    let nan_complex = Storage::from_dense_col_major(
        vec![Complex64::new(1.0, 0.0), Complex64::new(f64::NAN, 0.0)],
        &[2],
    )
    .unwrap();
    assert!(nan_complex.max_abs().is_nan());

    for value in [
        Complex64::new(f64::INFINITY, f64::NAN),
        Complex64::new(f64::NAN, f64::INFINITY),
    ] {
        let mixed = Storage::from_dense_col_major(vec![value], &[1]).unwrap();
        assert!(mixed.max_abs().is_nan());
    }

    let infinity = Storage::from_dense_col_major(vec![1.0, f64::INFINITY], &[2]).unwrap();
    assert!(infinity.max_abs().is_infinite());
}

// ===== Storage permute_storage for structured variants =====

#[test]
fn test_storage_permute_storage_structured_diag() {
    // Diagonal structured storage: permuting is trivial (data doesn't change).
    let s = Storage::from_diag_col_major(vec![1.0, 2.0, 3.0], 2).unwrap();
    let permuted = s.permute_storage(&[3, 3], &[1, 0]).unwrap();
    assert!(permuted.is_f64());
    assert!(permuted.is_diag());
}

#[test]
fn test_storage_permute_storage_structured_diag_c64() {
    let s =
        Storage::from_diag_col_major(vec![Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)], 2)
            .unwrap();
    let permuted = s.permute_storage(&[2, 2], &[1, 0]).unwrap();
    assert!(permuted.is_c64());
    assert!(permuted.is_diag());
}

// ===== Storage conj for structured variants =====

#[test]
fn test_storage_conj_structured_f64() {
    let s = Storage::from_dense_col_major(vec![1.0, -2.0], &[2]).unwrap();
    let conj = s.conj();
    assert!(conj.is_f64());
}

#[test]
fn test_storage_conj_structured_c64() {
    let s = Storage::from_dense_col_major(
        vec![Complex64::new(1.0, 2.0), Complex64::new(3.0, -4.0)],
        &[2],
    )
    .unwrap();
    let conj = s.conj();
    assert!(conj.is_c64());
}

// ===== Storage extract_real/imag and to_complex for structured variants =====

#[test]
fn test_storage_extract_real_structured() {
    let s = Storage::from_dense_col_major(
        vec![Complex64::new(1.0, 2.0), Complex64::new(3.0, 4.0)],
        &[2],
    )
    .unwrap();
    let real = s.extract_real_part();
    assert!(real.is_f64());

    let f64_s = Storage::from_dense_col_major(vec![1.0, 2.0], &[2]).unwrap();
    let real2 = f64_s.extract_real_part();
    assert!(real2.is_f64());
}

#[test]
fn test_storage_extract_imag_structured() {
    let s = Storage::from_dense_col_major(
        vec![Complex64::new(1.0, 2.0), Complex64::new(3.0, 4.0)],
        &[2],
    )
    .unwrap();
    let imag = s.extract_imag_part(&[2]);
    assert!(imag.is_f64());

    let f64_s = Storage::from_dense_col_major(vec![1.0, 2.0], &[2]).unwrap();
    let imag2 = f64_s.extract_imag_part(&[2]);
    assert!(imag2.is_f64());
}

#[test]
fn test_storage_to_complex_structured() {
    let s = Storage::from_dense_col_major(vec![1.0, 2.0], &[2]).unwrap();
    let c = s.to_complex_storage();
    assert!(c.is_c64());

    let already_complex =
        Storage::from_dense_col_major(vec![Complex64::new(1.0, 2.0)], &[1]).unwrap();
    let c2 = already_complex.to_complex_storage();
    assert!(c2.is_c64());
}

// ===== Storage to_dense_storage for structured variants =====

#[test]
fn test_storage_to_dense_storage_structured_f64() {
    let s = Storage::from_diag_col_major(vec![1.0, 2.0], 2).unwrap();
    let dense = s.to_dense_storage(&[2, 2]).unwrap();
    assert!(dense.is_dense());
    assert!(dense.is_f64());
}

#[test]
fn test_storage_to_dense_storage_structured_c64() {
    let s =
        Storage::from_diag_col_major(vec![Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)], 2)
            .unwrap();
    let dense = s.to_dense_storage(&[2, 2]).unwrap();
    assert!(dense.is_dense());
    assert!(dense.is_c64());
}

// ===== Storage to_dense_*_col_major_vec for structured non-diag =====

#[test]
fn test_storage_to_dense_f64_col_major_vec_structured_non_contiguous() {
    // Create a structured storage that is not dense - a diagonal
    let s = Storage::from_diag_col_major(vec![3.0, 7.0], 2).unwrap();
    let values = s.to_dense_f64_col_major_vec(&[2, 2]).unwrap();
    // Col-major for [[3,0],[0,7]]: [3, 0, 0, 7]
    assert_eq!(values.len(), 4);
    assert!((values[0] - 3.0).abs() < 1e-10);
    assert!((values[3] - 7.0).abs() < 1e-10);
}

#[test]
fn test_storage_to_dense_c64_col_major_vec_structured_non_contiguous() {
    let s =
        Storage::from_diag_col_major(vec![Complex64::new(3.0, 1.0), Complex64::new(7.0, 2.0)], 2)
            .unwrap();
    let values = s.to_dense_c64_col_major_vec(&[2, 2]).unwrap();
    assert_eq!(values.len(), 4);
    assert!((values[0] - Complex64::new(3.0, 1.0)).norm() < 1e-10);
    assert!((values[3] - Complex64::new(7.0, 2.0)).norm() < 1e-10);
}

// ===== new_structured constructors =====

#[test]
fn test_storage_new_structured_f64() {
    let s = Storage::new_structured::<f64>(vec![1.0, 2.0], vec![2], vec![1], vec![0, 0]).unwrap();
    assert!(s.is_f64());
    assert!(s.is_diag());
}

#[test]
fn test_storage_new_structured_c64() {
    let s = Storage::new_structured::<Complex64>(
        vec![Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)],
        vec![2],
        vec![1],
        vec![0, 0],
    )
    .unwrap();
    assert!(s.is_c64());
    assert!(s.is_diag());
}

// ===== make_mut_storage =====

#[test]
fn test_make_mut_storage() {
    let storage = Storage::from_dense_col_major(vec![1.0], &[1]).unwrap();
    let mut arc = Arc::new(storage);
    let _mutable = make_mut_storage(&mut arc);
    // Just verify it doesn't panic and returns a mutable ref
}

// ===== SumFromStorage coverage for Structured variants =====

#[test]
fn test_sum_from_storage_diag_c64() {
    let s =
        Storage::from_diag_col_major(vec![Complex64::new(1.0, 2.0), Complex64::new(3.0, 4.0)], 2)
            .unwrap();
    let sum_f64: f64 = f64::sum_from_storage(&s);
    // real part sum: 1.0 + 3.0 = 4.0
    assert!((sum_f64 - 4.0).abs() < 1e-10);

    let sum_c64: Complex64 = Complex64::sum_from_storage(&s);
    assert!((sum_c64 - Complex64::new(4.0, 6.0)).norm() < 1e-10);
}

#[test]
fn test_sum_from_storage_structured_f64() {
    let s = Storage::from_dense_col_major(vec![1.0, 2.0, 3.0], &[3]).unwrap();
    let sum_f64: f64 = f64::sum_from_storage(&s);
    assert!((sum_f64 - 6.0).abs() < 1e-10);

    let sum_c64: Complex64 = Complex64::sum_from_storage(&s);
    assert!((sum_c64 - Complex64::new(6.0, 0.0)).norm() < 1e-10);
}

#[test]
fn test_sum_from_storage_structured_c64() {
    let s = Storage::from_dense_col_major(
        vec![Complex64::new(1.0, 2.0), Complex64::new(3.0, 4.0)],
        &[2],
    )
    .unwrap();
    let sum_f64: f64 = f64::sum_from_storage(&s);
    // real part sum: 1.0 + 3.0 = 4.0
    assert!((sum_f64 - 4.0).abs() < 1e-10);

    let sum_c64: Complex64 = Complex64::sum_from_storage(&s);
    assert!((sum_c64 - Complex64::new(4.0, 6.0)).norm() < 1e-10);
}

#[test]
fn test_sum_from_storage_dense_c64_as_f64() {
    let s = Storage::from_dense_col_major(
        vec![Complex64::new(10.0, 5.0), Complex64::new(-3.0, 2.0)],
        &[2],
    )
    .unwrap();
    let sum_f64: f64 = f64::sum_from_storage(&s);
    // real part sum: 10.0 + (-3.0) = 7.0
    assert!((sum_f64 - 7.0).abs() < 1e-10);
}

#[test]
fn test_sum_from_storage_dense_f64_as_c64() {
    let s = Storage::from_dense_col_major(vec![2.0, 3.0], &[2]).unwrap();
    let sum_c64: Complex64 = Complex64::sum_from_storage(&s);
    assert!((sum_c64 - Complex64::new(5.0, 0.0)).norm() < 1e-10);
}

#[test]
fn test_sum_from_storage_diag_f64_as_c64() {
    let s = Storage::from_diag_col_major(vec![10.0, 20.0], 2).unwrap();
    let sum_c64: Complex64 = Complex64::sum_from_storage(&s);
    assert!((sum_c64 - Complex64::new(30.0, 0.0)).norm() < 1e-10);
}

// ===== Contraction via native path =====

#[test]
fn test_contract_storage_dense_f64_matmul() {
    // A = [[1,2],[3,4]] col-major: [1, 3, 2, 4]
    // B = [[5,6],[7,8]] col-major: [5, 7, 6, 8]
    // C = A @ B = [[19,22],[43,50]] col-major: [19, 43, 22, 50]
    let a = Storage::from_dense_col_major(vec![1.0, 3.0, 2.0, 4.0], &[2, 2]).unwrap();
    let b = Storage::from_dense_col_major(vec![5.0, 7.0, 6.0, 8.0], &[2, 2]).unwrap();

    let result = contract_storage(&a, &[2, 2], &[1], &b, &[2, 2], &[0], &[2, 2]).unwrap();

    let data = result
        .to_dense_f64_col_major_vec(&[2, 2])
        .expect("materialization failed");
    // col-major: [19, 43, 22, 50]
    assert!((data[0] - 19.0).abs() < 1e-10);
    assert!((data[1] - 43.0).abs() < 1e-10);
    assert!((data[2] - 22.0).abs() < 1e-10);
    assert!((data[3] - 50.0).abs() < 1e-10);
}

#[test]
fn test_contract_storage_diag_f64_partial() {
    // Diag [3, 3] with diag = [1, 2, 3]
    // Diag [3, 3] with diag = [4, 5, 6]
    // Contract axis 1 of first with axis 0 of second
    // Result should be element-wise product: [4, 10, 18] (diagonal)
    let diag1 = Storage::from_diag_col_major(vec![1.0, 2.0, 3.0], 2).unwrap();
    let diag2 = Storage::from_diag_col_major(vec![4.0, 5.0, 6.0], 2).unwrap();

    let result = contract_storage(&diag1, &[3, 3], &[1], &diag2, &[3, 3], &[0], &[3, 3]).unwrap();

    // Materialize as dense to verify
    let dense = result
        .to_dense_f64_col_major_vec(&[3, 3])
        .expect("materialization failed");
    // Col-major [3,3]: diag = [4, 10, 18], rest = 0
    assert!((dense[0] - 4.0).abs() < 1e-10); // (0,0)
    assert!((dense[4] - 10.0).abs() < 1e-10); // (1,1) = col-major offset 1 + 1*3 = 4
    assert!((dense[8] - 18.0).abs() < 1e-10); // (2,2) = col-major offset 2 + 2*3 = 8
}

#[test]
fn test_contract_storage_inner_product() {
    // [1, 2, 3] . [4, 5, 6] = 4 + 10 + 18 = 32
    let a = Storage::from_dense_col_major(vec![1.0, 2.0, 3.0], &[3]).unwrap();
    let b = Storage::from_dense_col_major(vec![4.0, 5.0, 6.0], &[3]).unwrap();

    let result = contract_storage(&a, &[3], &[0], &b, &[3], &[0], &[]).unwrap();

    let data = result
        .to_dense_f64_col_major_vec(&[])
        .expect("materialization failed");
    assert_eq!(data.len(), 1);
    assert!((data[0] - 32.0).abs() < 1e-10);
}
