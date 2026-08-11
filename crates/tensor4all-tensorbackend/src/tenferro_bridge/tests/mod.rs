use super::*;
use crate::storage::Storage;
use crate::tensor_element::TensorElement;
use num_complex::{Complex32, Complex64};

fn assert_storage_eq(lhs: &Storage, rhs: &Storage) {
    match (lhs.repr(), rhs.repr()) {
        (StorageRepr::F64(a), StorageRepr::F64(b)) => {
            assert_eq!(a.payload_dims(), b.payload_dims());
            assert_eq!(a.strides(), b.strides());
            assert_eq!(a.axis_classes(), b.axis_classes());
            assert_eq!(a.data(), b.data());
        }
        (StorageRepr::C64(a), StorageRepr::C64(b)) => {
            assert_eq!(a.payload_dims(), b.payload_dims());
            assert_eq!(a.strides(), b.strides());
            assert_eq!(a.axis_classes(), b.axis_classes());
            assert_eq!(a.data(), b.data());
        }
        _ => panic!(
            "storage mismatch: lhs variant {:?}, rhs variant {:?}",
            std::mem::discriminant(lhs.repr()),
            std::mem::discriminant(rhs.repr())
        ),
    }
}

fn recorded_native_einsum_call_count(path: NativeEinsumPath) -> usize {
    NATIVE_EINSUM_PROFILE_STATE.with(|state| {
        state
            .borrow()
            .iter()
            .filter(|(signature, _)| signature.path == path)
            .map(|(_, entry)| entry.calls)
            .sum()
    })
}

fn default_graph_runtime_has_einsum_extension_cache_entries() -> bool {
    crate::context::with_default_graph_runtime(|compiler, _| {
        compiler.cache_stats().extensions.entries > 0
    })
}

struct ProfileGuard;

impl ProfileGuard {
    fn enable() -> Self {
        reset_native_einsum_profile();
        set_native_einsum_profile_enabled_for_tests(true);
        Self
    }
}

impl Drop for ProfileGuard {
    fn drop(&mut self) {
        set_native_einsum_profile_enabled_for_tests(false);
        reset_native_einsum_profile();
    }
}

#[test]
fn storage_native_roundtrip_dense_f64() {
    let storage = Storage::from_dense_col_major(vec![1.0, 3.0, 2.0, 4.0], &[2, 2]).unwrap();
    let native = storage_to_native_tensor(&storage, &[2, 2]).unwrap();
    let roundtrip = native_tensor_primal_to_storage(&native).unwrap();
    assert_storage_eq(&roundtrip, &storage);
}

#[test]
fn storage_native_roundtrip_diag_densifies_at_public_bridge() {
    let storage = Storage::from_diag_col_major(vec![2.0, -1.0, 4.0], 2).unwrap();
    let native = storage_to_native_tensor(&storage, &[3, 3]).unwrap();
    let roundtrip = native_tensor_primal_to_storage(&native).unwrap();
    let expected = storage.to_dense_storage(&[3, 3]).unwrap();

    assert_eq!(native.shape(), &[3, 3]);
    assert_storage_eq(&roundtrip, &expected);
}

#[test]
fn storage_native_roundtrip_structured_c64_densifies_at_public_bridge() {
    let storage = Storage::new_structured(
        vec![Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)],
        vec![2],
        vec![1],
        vec![0, 0],
    )
    .unwrap();
    let native = storage_to_native_tensor(&storage, &[2, 2]).unwrap();
    let roundtrip = native_tensor_primal_to_storage(&native).unwrap();
    let expected = storage.to_dense_storage(&[2, 2]).unwrap();

    assert_eq!(native.shape(), &[2, 2]);
    assert_storage_eq(&roundtrip, &expected);
}

#[test]
fn dense_native_tensor_column_major_roundtrip_preserves_linearization() {
    let native =
        dense_native_tensor_from_col_major(&[1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
    let values = native_tensor_primal_to_dense_col_major::<f64>(&native).unwrap();
    assert_eq!(values, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn dense_native_tensor_from_col_major_c64_roundtrip() {
    let data = vec![Complex64::new(1.0, 2.0), Complex64::new(3.0, 4.0)];
    let native = dense_native_tensor_from_col_major(&data, &[2]).unwrap();
    let values = native_tensor_primal_to_dense_col_major::<Complex64>(&native).unwrap();
    assert_eq!(values, data);
}

#[test]
fn diag_native_tensor_from_col_major_f64_roundtrip() {
    let data = vec![1.0_f64, 2.0, 3.0];
    let native = diag_native_tensor_from_col_major(&data, 2).unwrap();
    let diag_values = native_tensor_primal_to_diag::<f64>(&native).unwrap();
    let dense = native_tensor_primal_to_dense_col_major::<f64>(&native).unwrap();

    assert_eq!(diag_values, data);
    assert_eq!(
        dense,
        vec![
            1.0, 0.0, 0.0, //
            0.0, 2.0, 0.0, //
            0.0, 0.0, 3.0,
        ]
    );
}

#[test]
fn dense_native_tensor_from_col_major_f32_roundtrip() {
    let native = dense_native_tensor_from_col_major(&[1.25_f32, -2.5_f32], &[2]).unwrap();
    // Strict materialization preserves the scalar kind: an f32 native tensor
    // must not be readable as f64.
    assert!(native_tensor_primal_to_dense_col_major::<f64>(&native).is_err());
    assert_eq!(
        native_tensor_primal_to_dense_col_major::<f32>(&native).unwrap(),
        vec![1.25_f32, -2.5_f32]
    );
    let snapshot = native_tensor_primal_to_storage(&native).unwrap();
    assert_storage_eq(
        &snapshot,
        &Storage::from_dense_col_major(vec![1.25, -2.5], &[2]).unwrap(),
    );
}

#[test]
fn diag_native_tensor_from_col_major_c32_promotes_to_c64_values() {
    let data = vec![Complex32::new(1.0, -0.5), Complex32::new(-2.0, 0.25)];
    let native = diag_native_tensor_from_col_major(&data, 2).unwrap();
    let diag_values = native_tensor_primal_to_diag::<Complex64>(&native).unwrap();
    // Strict dense materialization rejects the c32 source; only the
    // promoting diag extractor converts c32 -> c64.
    assert!(native_tensor_primal_to_dense_col_major::<Complex64>(&native).is_err());
    let snapshot = native_tensor_primal_to_storage(&native).unwrap();

    assert_eq!(
        diag_values,
        vec![Complex64::new(1.0, -0.5), Complex64::new(-2.0, 0.25)]
    );

    assert_storage_eq(
        &snapshot,
        &Storage::from_dense_col_major(
            vec![
                Complex64::new(1.0, -0.5),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(-2.0, 0.25),
            ],
            &[2, 2],
        )
        .unwrap(),
    );
}

#[test]
fn sum_native_tensor_returns_rank0_scalar() {
    let storage = Storage::from_dense_col_major(
        vec![Complex64::new(1.0, -1.0), Complex64::new(-0.5, 2.0)],
        &[2],
    )
    .unwrap();
    let native = storage_to_native_tensor(&storage, &[2]).unwrap();

    let sum = sum_native_tensor(&native).unwrap();

    assert!(sum.is_complex());
    assert_eq!(sum.as_c64(), Some(Complex64::new(0.5, 1.0)));
}

#[test]
fn sum_native_tensor_preserves_rank0_scalar() {
    let native = Complex64::scalar_native_tensor(Complex64::new(1.5, -0.25)).unwrap();
    let sum = sum_native_tensor(&native).unwrap();

    assert_eq!(sum.as_c64(), Some(Complex64::new(1.5, -0.25)));
}

#[test]
fn native_snapshot_materializes_lazy_conjugation() {
    let storage = Storage::from_dense_col_major(
        vec![
            Complex64::new(1.0, 2.0),
            Complex64::new(0.0, -1.0),
            Complex64::new(-3.0, 4.5),
            Complex64::new(2.5, 0.25),
        ],
        &[2, 2],
    )
    .unwrap();
    let native = storage_to_native_tensor(&storage, &[2, 2]).unwrap();
    let conjugated = conj_native_tensor(&native).unwrap();
    let snapshot = native_tensor_primal_to_storage(&conjugated).unwrap();

    let expected = Storage::from_dense_col_major(
        vec![
            Complex64::new(1.0, -2.0),
            Complex64::new(0.0, 1.0),
            Complex64::new(-3.0, -4.5),
            Complex64::new(2.5, -0.25),
        ],
        &[2, 2],
    )
    .unwrap();
    assert_storage_eq(&snapshot, &expected);
}

#[test]
fn native_einsum_accepts_unsorted_nonfirst_operand_labels() {
    let lhs = storage_to_native_tensor(
        &Storage::from_dense_col_major(vec![1.0, 3.0, 2.0, 4.0], &[2, 2]).unwrap(),
        &[2, 2],
    )
    .unwrap();
    let rhs = storage_to_native_tensor(
        &Storage::from_dense_col_major(vec![10.0, 30.0, 50.0, 20.0, 40.0, 60.0], &[3, 2]).unwrap(),
        &[3, 2],
    )
    .unwrap();

    let out = einsum_native_tensors(&[(&lhs, &[0, 1]), (&rhs, &[2, 1])], &[0, 2]).unwrap();
    let snapshot = native_tensor_primal_to_storage(&out).unwrap();

    let expected =
        Storage::from_dense_col_major(vec![50.0, 110.0, 110.0, 250.0, 170.0, 390.0], &[2, 3])
            .unwrap();
    assert_storage_eq(&snapshot, &expected);
}

#[test]
fn einsum_native_tensors_supports_retained_shared_nary_label() {
    let a = NativeTensor::from_vec_col_major(vec![2, 2], vec![5.0_f64, 7.0, 11.0, 13.0]);
    let b = NativeTensor::from_vec_col_major(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let c = NativeTensor::from_vec_col_major(vec![2, 2], vec![11.0_f64, 13.0, 17.0, 19.0]);

    let out = einsum_native_tensors(
        &[(&a, &[0, 1]), (&b, &[0, 2]), (&c, &[0, 3])],
        &[0, 1, 2, 3],
    )
    .unwrap();
    let values = native_tensor_primal_to_dense_col_major::<f64>(&out).unwrap();

    let mut expected = vec![0.0; 24];
    for b_idx in 0..2 {
        for i_idx in 0..2 {
            for j_idx in 0..3 {
                for k_idx in 0..2 {
                    let a_offset = b_idx + 2 * i_idx;
                    let b_offset = b_idx + 2 * j_idx;
                    let c_offset = b_idx + 2 * k_idx;
                    let out_offset = b_idx + 2 * (i_idx + 2 * (j_idx + 3 * k_idx));
                    expected[out_offset] = a.as_slice::<f64>().unwrap()[a_offset]
                        * b.as_slice::<f64>().unwrap()[b_offset]
                        * c.as_slice::<f64>().unwrap()[c_offset];
                }
            }
        }
    }

    assert_eq!(out.shape(), &[2, 2, 3, 2]);
    assert_eq!(values, expected);
    assert_eq!(values[0], 55.0);
}

#[test]
fn einsum_native_tensors_populates_process_global_path_cache() {
    let a = NativeTensor::from_vec_col_major(vec![2, 3, 4], vec![1.0_f64; 24]);
    let b = NativeTensor::from_vec_col_major(vec![4, 5], vec![2.0_f64; 20]);
    let c = NativeTensor::from_vec_col_major(vec![3, 2], vec![3.0_f64; 6]);

    let out =
        einsum_native_tensors(&[(&a, &[0, 1, 2]), (&b, &[2, 3]), (&c, &[1, 0])], &[3]).unwrap();

    assert_eq!(out.shape(), &[5]);
    assert_eq!(
        native_tensor_primal_to_dense_col_major::<f64>(&out).unwrap(),
        vec![144.0; 5]
    );
    assert!(default_graph_runtime_has_einsum_extension_cache_entries());
}

#[test]
fn einsum_native_tensors_mixed_dtype_records_borrowed_conversion_profile() {
    let _guard = ProfileGuard::enable();
    let lhs = NativeTensor::from_vec_col_major(vec![2, 2], vec![1.0_f32, 2.0, 3.0, 4.0]);
    let rhs = NativeTensor::from_vec_col_major(vec![2, 3], vec![5.0_f64, 6.0, 7.0, 8.0, 9.0, 10.0]);

    let owned = einsum_native_tensors_owned(
        vec![(lhs.clone(), vec![0, 1]), (rhs.clone(), vec![1, 2])],
        &[0, 2],
    )
    .unwrap();
    let borrowed = einsum_native_tensors(&[(&lhs, &[0, 1]), (&rhs, &[1, 2])], &[0, 2]).unwrap();

    assert_eq!(owned.shape(), &[2, 3]);
    assert_eq!(owned.dtype(), DType::F64);
    assert_eq!(
        native_tensor_primal_to_dense_col_major::<f64>(&owned).unwrap(),
        native_tensor_primal_to_dense_col_major::<f64>(&borrowed).unwrap()
    );
    assert_eq!(
        recorded_native_einsum_call_count(NativeEinsumPath::BorrowedWithConversions),
        1
    );
    assert_eq!(
        recorded_native_einsum_call_count(NativeEinsumPath::Borrowed),
        0
    );
}

#[test]
fn einsum_native_tensors_dense_binary_records_borrowed_profile() {
    let _guard = ProfileGuard::enable();

    let lhs = dense_native_tensor_from_col_major(&[1.0_f64, 3.0, 2.0, 4.0], &[2, 2]).unwrap();
    let rhs =
        dense_native_tensor_from_col_major(&[10.0_f64, 30.0, 50.0, 20.0, 40.0, 60.0], &[3, 2])
            .unwrap();
    let out = einsum_native_tensors(&[(&lhs, &[0, 1]), (&rhs, &[2, 1])], &[0, 2]).unwrap();
    let snapshot = native_tensor_primal_to_storage(&out).unwrap();
    let expected =
        Storage::from_dense_col_major(vec![50.0, 110.0, 110.0, 250.0, 170.0, 390.0], &[2, 3])
            .unwrap();

    assert_eq!(out.shape(), &[2, 3]);
    assert_storage_eq(&snapshot, &expected);
    assert_eq!(
        recorded_native_einsum_call_count(NativeEinsumPath::Borrowed),
        1
    );
    assert_eq!(
        recorded_native_einsum_call_count(NativeEinsumPath::BorrowedWithConversions),
        0
    );
}

#[test]
fn native_read_input_owned_and_plan_helpers_cover_debug_paths() {
    let tensor = NativeTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]);
    let input = NativeTensorReadInput::Owned(tensor.clone());
    assert_eq!(input.dtype(), DType::F64);
    assert_eq!(input.shape(), &[2]);
    assert_eq!(input.as_read().shape(), &[2]);

    assert_eq!(native_einsum_path_trace_min_bytes(), 0);
    assert_eq!(native_einsum_path_trace_max_signatures(), 64);
    assert_eq!(native_einsum_pool_trace_min_output_bytes(), 0);
    assert_eq!(native_einsum_pool_trace_min_retained_bytes(), 0);

    assert_eq!(dtype_size_bytes(DType::F32), 4);
    assert_eq!(dtype_size_bytes(DType::F64), 8);
    assert_eq!(dtype_size_bytes(DType::C32), 8);
    assert_eq!(dtype_size_bytes(DType::C64), 16);
    assert_eq!(dtype_size_bytes(DType::I64), 8);
    assert_eq!(native_tensor_bytes(&tensor), 16);
    assert_eq!(format_label('x' as u32), "x");
    assert_eq!(format_label(0x110000), "1114112");
    assert_eq!(format_labels(&[]), "scalar");
    assert_eq!(format_labels(&['a' as u32, 'b' as u32]), "ab");

    let subscripts = Subscripts {
        inputs: vec![vec![0, 1], vec![1, 2]],
        output: vec![0, 2],
    };
    let dims = label_dims(&subscripts, &[vec![2, 3], vec![3, 4]]).unwrap();
    assert_eq!(labels_size(&[0, 2], &dims), 8);
    assert_eq!(union_labels(&[0, 1], &[1, 2]), vec![0, 1, 2]);

    let bad_len = label_dims(&subscripts, &[vec![2], vec![3, 4]]).unwrap_err();
    assert!(bad_len.to_string().contains("do not match shape"));
    let bad_dim = label_dims(&subscripts, &[vec![2, 3], vec![4, 4]]).unwrap_err();
    assert!(bad_dim.to_string().contains("inconsistent dimension"));

    let signature = NativeEinsumSignature {
        path: NativeEinsumPath::Borrowed,
        operands: vec![
            NativeOperandSignature {
                shape: vec![2, 3],
                ids: vec![0, 1],
                dtype: DType::F64,
            },
            NativeOperandSignature {
                shape: vec![3, 4],
                ids: vec![1, 2],
                dtype: DType::F64,
            },
        ],
        output_ids: vec![0, 2],
    };
    let time_report = native_einsum_time_optimized_plan_report(&signature).unwrap();
    let balanced_report = native_einsum_balanced_plan_report(&signature).unwrap();
    assert!(time_report.peak_intermediate_bytes >= 8);
    assert!(!time_report.lines.is_empty());
    assert!(!balanced_report.lines.is_empty());
}

#[test]
fn native_einsum_profile_print_and_c32_arithmetic_paths() {
    let _guard = ProfileGuard::enable();

    let lhs = NativeTensor::from_vec_col_major(
        vec![2],
        vec![Complex32::new(1.0, 2.0), Complex32::new(-3.0, 0.5)],
    );
    let rhs = NativeTensor::from_vec_col_major(
        vec![2],
        vec![Complex32::new(0.5, -1.0), Complex32::new(4.0, 2.0)],
    );

    let scaled = scale_native_tensor(
        &lhs,
        &crate::AnyScalar::from_value(Complex32::new(2.0, -1.0)),
    )
    .unwrap();
    assert_eq!(scaled.dtype(), DType::C32);
    assert_eq!(
        scaled.as_slice::<Complex32>().unwrap(),
        &[Complex32::new(4.0, 3.0), Complex32::new(-5.5, 4.0)]
    );

    let combined = axpby_native_tensor(
        &lhs,
        &crate::AnyScalar::from_value(Complex32::new(1.0, 0.0)),
        &rhs,
        &crate::AnyScalar::from_value(Complex32::new(0.0, 1.0)),
    )
    .unwrap();
    assert_eq!(combined.dtype(), DType::C32);
    assert_eq!(
        combined.as_slice::<Complex32>().unwrap(),
        &[Complex32::new(2.0, 2.5), Complex32::new(-5.0, 4.5)]
    );

    let contraction = einsum_native_tensors(&[(&lhs, &[0]), (&rhs, &[0])], &[]).unwrap();
    assert_eq!(contraction.shape(), &[] as &[usize]);
    print_and_reset_native_einsum_profile();
    assert_eq!(
        recorded_native_einsum_call_count(NativeEinsumPath::Borrowed),
        0
    );
}

#[test]
fn contract_native_tensor_restores_rhs_free_axis_order() {
    let lhs = storage_to_native_tensor(
        &Storage::from_dense_col_major(vec![1.0, 3.0, 2.0, 4.0], &[2, 2]).unwrap(),
        &[2, 2],
    )
    .unwrap();
    let rhs = storage_to_native_tensor(
        &Storage::from_dense_col_major(vec![10.0, 30.0, 20.0, 40.0], &[2, 2]).unwrap(),
        &[2, 2],
    )
    .unwrap();

    let out = contract_native_tensor(&lhs, &[1], &rhs, &[1]).unwrap();
    let snapshot = native_tensor_primal_to_storage(&out).unwrap();
    let expected = Storage::from_dense_col_major(vec![50.0, 110.0, 110.0, 250.0], &[2, 2]).unwrap();

    assert_storage_eq(&snapshot, &expected);
}

#[test]
fn permute_storage_native_dense_matches_expected_data() {
    let storage =
        Storage::from_dense_col_major(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], &[2, 3]).unwrap();
    let permuted = permute_storage_native(&storage, &[2, 3], &[1, 0]).unwrap();
    let expected =
        Storage::from_dense_col_major(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]).unwrap();
    assert_storage_eq(&permuted, &expected);
}

#[test]
fn reshape_col_major_native_tensor_handles_permuted_input() {
    let native = dense_native_tensor_from_col_major(
        &(1..=24).map(|x| x as f64).collect::<Vec<_>>(),
        &[2, 3, 2, 2],
    )
    .unwrap();
    let permuted = permute_native_tensor(&native, &[0, 2, 1, 3]).unwrap();
    let permuted_values = native_tensor_primal_to_dense_col_major::<f64>(&permuted).unwrap();

    let reshaped = reshape_col_major_native_tensor(&permuted, &[4, 6]).unwrap();
    let reshaped_values = native_tensor_primal_to_dense_col_major::<f64>(&reshaped).unwrap();

    assert_eq!(reshaped_values, permuted_values);
}

#[test]
fn scale_native_tensor_promotes_real_scalar_for_complex_tensor() {
    let native = dense_native_tensor_from_col_major(
        &[Complex64::new(1.0, 2.0), Complex64::new(-0.5, 1.0)],
        &[2],
    )
    .unwrap();
    let scalar = crate::AnyScalar::new_real(2.0);
    let scaled = scale_native_tensor(&native, &scalar).unwrap();
    let values = native_tensor_primal_to_dense_col_major::<Complex64>(&scaled).unwrap();

    assert_eq!(
        values,
        vec![Complex64::new(2.0, 4.0), Complex64::new(-1.0, 2.0)]
    );
}

#[test]
fn scale_and_axpby_native_tensor_cover_f32_paths() {
    let lhs = dense_native_tensor_from_col_major(&[1.0_f32, -2.0_f32], &[2]).unwrap();
    let rhs = dense_native_tensor_from_col_major(&[0.5_f32, 4.0_f32], &[2]).unwrap();

    let scaled = scale_native_tensor(&lhs, &crate::AnyScalar::from_value(0.5_f32)).unwrap();
    let scaled_values = native_tensor_primal_to_dense_col_major::<f32>(&scaled).unwrap();
    assert_eq!(scaled_values, vec![0.5_f32, -1.0_f32]);

    let combined = axpby_native_tensor(
        &lhs,
        &crate::AnyScalar::from_value(2.0_f32),
        &rhs,
        &crate::AnyScalar::from_value(-1.0_f32),
    )
    .unwrap();
    let combined_values = native_tensor_primal_to_dense_col_major::<f32>(&combined).unwrap();
    assert_eq!(combined_values, vec![1.5_f32, -8.0_f32]);
}

#[test]
fn axpby_native_tensor_promotes_real_scalars_for_complex_tensors() {
    let lhs = dense_native_tensor_from_col_major(
        &[Complex64::new(1.0, 2.0), Complex64::new(-1.0, 0.5)],
        &[2],
    )
    .unwrap();
    let rhs = dense_native_tensor_from_col_major(
        &[Complex64::new(0.5, -1.0), Complex64::new(2.0, 1.0)],
        &[2],
    )
    .unwrap();
    let a = crate::AnyScalar::new_real(2.0);
    let b = crate::AnyScalar::new_real(-1.0);
    let combined = axpby_native_tensor(&lhs, &a, &rhs, &b).unwrap();
    let values = native_tensor_primal_to_dense_col_major::<Complex64>(&combined).unwrap();

    assert_eq!(
        values,
        vec![Complex64::new(1.5, 5.0), Complex64::new(-4.0, 0.0)]
    );
}

#[test]
fn scale_and_axpby_storage_native_roundtrip() {
    let storage = Storage::from_dense_col_major(vec![1.0, 2.0, 3.0], &[3]).unwrap();
    let scalar = crate::AnyScalar::new_real(2.0);
    let scaled = scale_storage_native(&storage, &[3], &scalar).unwrap();
    let expected_scaled = Storage::from_dense_col_major(vec![2.0, 4.0, 6.0], &[3]).unwrap();
    assert_storage_eq(&scaled, &expected_scaled);

    let rhs = Storage::from_dense_col_major(vec![3.0, 4.0, 5.0], &[3]).unwrap();
    let combined = axpby_storage_native(
        &storage,
        &[3],
        &crate::AnyScalar::new_real(2.0),
        &rhs,
        &[3],
        &crate::AnyScalar::new_real(3.0),
    )
    .unwrap();
    let expected_combined = Storage::from_dense_col_major(vec![11.0, 16.0, 21.0], &[3]).unwrap();
    assert_storage_eq(&combined, &expected_combined);
}

#[test]
fn outer_product_storage_native_produces_correct_shape() {
    let a = Storage::from_dense_col_major(vec![1.0, 2.0], &[2]).unwrap();
    let b = Storage::from_dense_col_major(vec![3.0, 4.0, 5.0], &[3]).unwrap();
    let result = outer_product_storage_native(&a, &[2], &b, &[3], &[2, 3]).unwrap();
    let expected =
        Storage::from_dense_col_major(vec![3.0, 6.0, 4.0, 8.0, 5.0, 10.0], &[2, 3]).unwrap();
    assert_storage_eq(&result, &expected);
}

#[test]
fn qr_and_svd_native_tensor_return_expected_shapes() {
    let native = dense_native_tensor_from_col_major(&[1.0_f64, 3.0, 2.0, 4.0], &[2, 2]).unwrap();

    let (q, r) = qr_native_tensor(&native).unwrap();
    assert_eq!(q.shape(), &[2, 2]);
    assert_eq!(r.shape(), &[2, 2]);

    let (u, s, vt) = svd_native_tensor(&native).unwrap();
    assert_eq!(u.shape(), &[2, 2]);
    assert_eq!(s.shape(), &[2]);
    assert_eq!(vt.shape(), &[2, 2]);
}

#[test]
fn tangent_native_tensor_returns_none_for_primal() {
    let native = dense_native_tensor_from_col_major(&[1.0_f64, 2.0], &[2]).unwrap();
    assert!(tangent_native_tensor(&native).is_none());
}

#[test]
fn dense_and_diag_extractors_reject_wrong_dtypes() {
    let real = dense_native_tensor_from_col_major(&[1.0_f64, 2.0], &[2]).unwrap();
    let complex = dense_native_tensor_from_col_major(&[Complex64::new(1.0, -1.0)], &[1]).unwrap();

    assert!(native_tensor_primal_to_dense_col_major::<f64>(&complex)
        .unwrap_err()
        .to_string()
        .contains("expected real native tensor"));
    assert!(native_tensor_primal_to_dense_col_major::<Complex64>(&real)
        .unwrap_err()
        .to_string()
        .contains("expected complex native tensor"));
    assert!(native_tensor_primal_to_diag::<f64>(&complex)
        .unwrap_err()
        .to_string()
        .contains("expected real native tensor"));
    assert!(native_tensor_primal_to_diag::<Complex64>(&real)
        .unwrap_err()
        .to_string()
        .contains("expected complex native tensor"));
}

#[test]
fn einsum_native_tensors_rejects_empty_operands() {
    let result = einsum_native_tensors(&[], &[]);
    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("at least one operand"));
}

#[test]
fn ids_to_subscript_and_build_einsum_subscripts_cover_helper_paths() {
    assert_eq!(ids_to_subscript(&[0, 25, 51]).unwrap(), "azZ");
    assert_eq!(
        build_einsum_subscripts(&[&[0, 1], &[2, 1]], &[0, 2]).unwrap(),
        "ab,cb->ac"
    );
    assert!(ids_to_subscript(&[52])
        .unwrap_err()
        .to_string()
        .contains("exceeds supported label range"));
}

#[test]
fn build_binary_einsum_ids_rejects_mismatched_axes_len() {
    let result = build_binary_einsum_ids(2, &[0], 2, &[0, 1]);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("length mismatch"));
}

#[test]
fn build_binary_einsum_ids_success_assigns_free_axes_after_contract_axes() {
    let (lhs_ids, rhs_ids, output_ids) = build_binary_einsum_ids(3, &[1], 2, &[0]).unwrap();

    assert_eq!(lhs_ids, vec![1, 0, 2]);
    assert_eq!(rhs_ids, vec![0, 3]);
    assert_eq!(output_ids, vec![1, 2, 3]);
}

#[test]
fn build_binary_einsum_ids_rejects_out_of_range_axis() {
    let result = build_binary_einsum_ids(2, &[5], 2, &[0]);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("out of range"));
}

#[test]
fn build_binary_einsum_ids_rejects_duplicate_axis() {
    let result = build_binary_einsum_ids(3, &[0, 0], 3, &[0, 1]);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("duplicate"));
}
