use super::*;
use crate::index::{t4a_index_new, t4a_index_release};
use crate::types::{
    t4a_singular_value_measure, t4a_storage_kind, t4a_threshold_scale, t4a_truncation_rule,
};
use num_complex::Complex64;
use std::ffi::CStr;
use std::sync::Arc;
use tensor4all_core::{AnyScalar, TensorContractionLike, TensorStorageError};

fn last_error() -> String {
    let mut len = 0usize;
    assert_eq!(
        crate::t4a_last_error_message(std::ptr::null_mut(), 0, &mut len),
        T4A_SUCCESS
    );
    let mut buf = vec![0u8; len];
    assert_eq!(
        crate::t4a_last_error_message(buf.as_mut_ptr(), buf.len(), &mut len),
        T4A_SUCCESS
    );
    CStr::from_bytes_until_nul(&buf)
        .unwrap()
        .to_str()
        .unwrap()
        .to_string()
}

fn new_index(dim: usize) -> *mut t4a_index {
    let mut out = std::ptr::null_mut();
    assert_eq!(
        t4a_index_new(dim, std::ptr::null(), 0, &mut out),
        T4A_SUCCESS
    );
    assert!(!out.is_null());
    out
}

fn new_tensor_f64(indices: &[*const t4a_index], data: &[f64]) -> *mut t4a_tensor {
    let mut out = std::ptr::null_mut();
    assert_eq!(
        t4a_tensor_new_dense_f64(
            indices.len(),
            indices.as_ptr(),
            data.as_ptr(),
            data.len(),
            &mut out
        ),
        T4A_SUCCESS
    );
    assert!(!out.is_null());
    out
}

fn new_tensor_c64(indices: &[*const t4a_index], data_interleaved: &[f64]) -> *mut t4a_tensor {
    let mut out = std::ptr::null_mut();
    assert_eq!(
        t4a_tensor_new_dense_c64(
            indices.len(),
            indices.as_ptr(),
            data_interleaved.as_ptr(),
            data_interleaved.len() / 2,
            &mut out
        ),
        T4A_SUCCESS
    );
    assert!(!out.is_null());
    out
}

fn read_dense_f64(tensor: *const t4a_tensor) -> Vec<f64> {
    let mut len = 0usize;
    assert_eq!(
        t4a_tensor_copy_dense_f64(tensor, std::ptr::null_mut(), 0, &mut len),
        T4A_SUCCESS
    );
    let mut data = vec![0.0; len];
    assert_eq!(
        t4a_tensor_copy_dense_f64(tensor, data.as_mut_ptr(), data.len(), &mut len),
        T4A_SUCCESS
    );
    data
}

#[test]
fn test_tensor_query_failures_set_last_error() {
    let mut rank = 0usize;
    assert_eq!(
        t4a_tensor_rank(std::ptr::null(), &mut rank),
        T4A_NULL_POINTER
    );
    assert_eq!(last_error(), "tensor is null");

    let i = new_index(2);
    let tensor = new_tensor_f64(&[i], &[1.0, 2.0]);
    let mut len = 0usize;
    let mut short = [0usize; 0];
    assert_eq!(
        t4a_tensor_dims(tensor, short.as_mut_ptr(), short.len(), &mut len),
        T4A_BUFFER_TOO_SMALL
    );
    assert_eq!(len, 1);
    assert!(last_error().contains("tensor dims buffer too small"));

    t4a_tensor_release(tensor);
    t4a_index_release(i);
}

fn read_payload_dims(tensor: *const t4a_tensor) -> Vec<usize> {
    let mut len = 0usize;
    assert_eq!(
        t4a_tensor_payload_dims(tensor, std::ptr::null_mut(), 0, &mut len),
        T4A_SUCCESS
    );
    let mut dims = vec![0usize; len];
    assert_eq!(
        t4a_tensor_payload_dims(tensor, dims.as_mut_ptr(), dims.len(), &mut len),
        T4A_SUCCESS
    );
    dims
}

fn read_payload_strides(tensor: *const t4a_tensor) -> Vec<isize> {
    let mut len = 0usize;
    assert_eq!(
        t4a_tensor_payload_strides(tensor, std::ptr::null_mut(), 0, &mut len),
        T4A_SUCCESS
    );
    let mut strides = vec![0isize; len];
    assert_eq!(
        t4a_tensor_payload_strides(tensor, strides.as_mut_ptr(), strides.len(), &mut len),
        T4A_SUCCESS
    );
    strides
}

fn read_axis_classes(tensor: *const t4a_tensor) -> Vec<usize> {
    let mut len = 0usize;
    assert_eq!(
        t4a_tensor_axis_classes(tensor, std::ptr::null_mut(), 0, &mut len),
        T4A_SUCCESS
    );
    let mut classes = vec![0usize; len];
    assert_eq!(
        t4a_tensor_axis_classes(tensor, classes.as_mut_ptr(), classes.len(), &mut len),
        T4A_SUCCESS
    );
    classes
}

fn read_payload_f64(tensor: *const t4a_tensor) -> Vec<f64> {
    let mut len = 0usize;
    assert_eq!(
        t4a_tensor_copy_payload_f64(tensor, std::ptr::null_mut(), 0, &mut len),
        T4A_SUCCESS
    );
    let mut data = vec![0.0; len];
    assert_eq!(
        t4a_tensor_copy_payload_f64(tensor, data.as_mut_ptr(), data.len(), &mut len),
        T4A_SUCCESS
    );
    data
}

fn read_payload_c64(tensor: *const t4a_tensor) -> Vec<f64> {
    let mut len = 0usize;
    assert_eq!(
        t4a_tensor_copy_payload_c64(tensor, std::ptr::null_mut(), 0, &mut len),
        T4A_SUCCESS
    );
    let mut data = vec![0.0; 2 * len];
    assert_eq!(
        t4a_tensor_copy_payload_c64(tensor, data.as_mut_ptr(), len, &mut len),
        T4A_SUCCESS
    );
    data
}

fn assert_tensors_close(actual: &InternalTensor, expected: &InternalTensor, tol: f64) {
    let neg_expected = expected.scale(AnyScalar::new_real(-1.0)).unwrap();
    let diff = actual.add(&neg_expected).unwrap();
    let maxabs = diff.maxabs();
    assert!(
        maxabs < tol,
        "tensor maxabs diff {maxabs} exceeded tolerance {tol}"
    );
}

fn reconstruct_svd(
    u: *const t4a_tensor,
    s: *const t4a_tensor,
    v: *const t4a_tensor,
) -> InternalTensor {
    let v = unsafe { (*v).inner() };
    let s = unsafe { (*s).inner() };
    let mut perm = vec![v.indices.len() - 1];
    perm.extend(0..(v.indices.len() - 1));
    let vh = v.conj().permute(&perm).unwrap();
    let svh = s.contract_pair(&vh).unwrap();
    let sim_bond = s.indices[1].clone();
    let bond = v.indices[v.indices.len() - 1].clone();
    let svh = svh.replaceind(&sim_bond, &bond).unwrap();
    unsafe { (*u).inner().contract_pair(&svh).unwrap() }
}

fn reconstruct_qr(q: *const t4a_tensor, r: *const t4a_tensor) -> InternalTensor {
    unsafe { (*q).inner().contract_pair((*r).inner()).unwrap() }
}

fn internal_tensor_f64(indices: &[*const t4a_index], data: &[f64]) -> InternalTensor {
    let indices = indices
        .iter()
        .map(|index| unsafe { (**index).inner().clone() })
        .collect();
    InternalTensor::from_dense(indices, data.to_vec()).unwrap()
}

fn internal_tensor_c64(indices: &[*const t4a_index], data: &[Complex64]) -> InternalTensor {
    let indices = indices
        .iter()
        .map(|index| unsafe { (**index).inner().clone() })
        .collect();
    InternalTensor::from_dense(indices, data.to_vec()).unwrap()
}

#[test]
fn test_tensor_dense_f64_query_then_fill() {
    let i = new_index(2);
    let j = new_index(3);
    let tensor = new_tensor_f64(&[i, j], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

    let mut rank = 0usize;
    assert_eq!(t4a_tensor_rank(tensor, &mut rank), T4A_SUCCESS);
    assert_eq!(rank, 2);

    let mut dims_len = 0usize;
    assert_eq!(
        t4a_tensor_dims(tensor, std::ptr::null_mut(), 0, &mut dims_len),
        T4A_SUCCESS
    );
    assert_eq!(dims_len, 2);
    let mut dims = vec![0usize; dims_len];
    assert_eq!(
        t4a_tensor_dims(tensor, dims.as_mut_ptr(), dims.len(), &mut dims_len),
        T4A_SUCCESS
    );
    assert_eq!(dims, vec![2, 3]);

    let mut indices_len = 0usize;
    assert_eq!(
        t4a_tensor_indices(tensor, std::ptr::null_mut(), 0, &mut indices_len),
        T4A_SUCCESS
    );
    let mut indices = vec![std::ptr::null_mut(); indices_len];
    assert_eq!(
        t4a_tensor_indices(
            tensor,
            indices.as_mut_ptr(),
            indices.len(),
            &mut indices_len
        ),
        T4A_SUCCESS
    );
    assert_eq!(indices_len, 2);

    let mut copied_dims = Vec::new();
    for index in &indices {
        let mut dim = 0usize;
        assert_eq!(crate::index::t4a_index_dim(*index, &mut dim), T4A_SUCCESS);
        copied_dims.push(dim);
        t4a_index_release(*index);
    }
    assert_eq!(copied_dims, vec![2, 3]);

    let mut kind = t4a_scalar_kind::C64;
    assert_eq!(t4a_tensor_scalar_kind(tensor, &mut kind), T4A_SUCCESS);
    assert_eq!(kind, t4a_scalar_kind::F64);
    assert_eq!(read_dense_f64(tensor), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

    t4a_tensor_release(tensor);
    t4a_index_release(i);
    t4a_index_release(j);
}

#[test]
fn test_tensor_dense_c64_roundtrip() {
    let i = new_index(2);
    let j = new_index(2);
    let mut tensor = std::ptr::null_mut();
    let interleaved = [1.0, 0.5, 2.0, -1.5, 3.0, 2.5, 4.0, -3.5];
    let index_ptrs = [i as *const t4a_index, j as *const t4a_index];
    assert_eq!(
        t4a_tensor_new_dense_c64(2, index_ptrs.as_ptr(), interleaved.as_ptr(), 4, &mut tensor),
        T4A_SUCCESS
    );
    assert!(!tensor.is_null());

    let mut kind = t4a_scalar_kind::F64;
    assert_eq!(t4a_tensor_scalar_kind(tensor, &mut kind), T4A_SUCCESS);
    assert_eq!(kind, t4a_scalar_kind::C64);

    let mut len = 0usize;
    assert_eq!(
        t4a_tensor_copy_dense_c64(tensor, std::ptr::null_mut(), 0, &mut len),
        T4A_SUCCESS
    );
    assert_eq!(len, 4);
    let mut out = vec![0.0; 2 * len];
    assert_eq!(
        t4a_tensor_copy_dense_c64(tensor, out.as_mut_ptr(), len, &mut len),
        T4A_SUCCESS
    );
    assert_eq!(out, interleaved);

    t4a_tensor_release(tensor);
    t4a_index_release(i);
    t4a_index_release(j);
}

#[test]
fn dense_constructor_rejects_dimension_product_overflow() {
    let huge = new_index(usize::MAX);
    let two = new_index(2);
    let indices = [huge as *const t4a_index, two as *const t4a_index];

    let mut real = std::ptr::null_mut();
    assert_eq!(
        t4a_tensor_new_dense_f64(
            indices.len(),
            indices.as_ptr(),
            std::ptr::null(),
            0,
            &mut real,
        ),
        T4A_INVALID_ARGUMENT
    );
    assert!(real.is_null());
    assert!(last_error().contains("dimension product"));

    let mut complex = std::ptr::null_mut();
    assert_eq!(
        t4a_tensor_new_dense_c64(
            indices.len(),
            indices.as_ptr(),
            std::ptr::null(),
            0,
            &mut complex,
        ),
        T4A_INVALID_ARGUMENT
    );
    assert!(complex.is_null());
    assert!(last_error().contains("dimension product"));

    t4a_index_release(two);
    t4a_index_release(huge);
}

#[test]
fn c64_reader_rejects_interleaved_length_overflow_before_dereference() {
    let error = read_c64_slice(
        "data_interleaved",
        std::ptr::NonNull::<f64>::dangling().as_ptr(),
        usize::MAX,
    )
    .unwrap_err();
    assert_eq!(error.0, T4A_INVALID_ARGUMENT);
    assert!(error.1.contains("overflows"));
}

#[test]
fn raw_slice_rejects_byte_length_overflow() {
    let len = isize::MAX as usize / std::mem::size_of::<f64>() + 1;
    let error =
        read_plain_slice("data", std::ptr::NonNull::<f64>::dangling().as_ptr(), len).unwrap_err();
    assert_eq!(error.0, T4A_INVALID_ARGUMENT);
    assert!(error.1.contains("byte length"));
}

#[test]
fn c64_reader_rejects_interleaved_byte_length_overflow() {
    let n_complex = isize::MAX as usize / (2 * std::mem::size_of::<f64>()) + 1;
    let error = read_c64_slice(
        "data_interleaved",
        std::ptr::NonNull::<f64>::dangling().as_ptr(),
        n_complex,
    )
    .unwrap_err();
    assert_eq!(error.0, T4A_INVALID_ARGUMENT);
    assert!(error.1.contains("byte length"));
}

#[test]
fn tensor_storage_materialization_failure_maps_to_internal_error_with_diagnostic() {
    let error = TensorStorageError::Materialization {
        source: Arc::new(std::io::Error::other(
            "forced backend materialization failure",
        )),
    };
    let (status, message) = tensor_storage_error(error);

    assert_eq!(status, T4A_INTERNAL_ERROR);
    assert!(message.contains("forced backend materialization failure"));
}

#[test]
fn exported_payload_function_preserves_status_and_last_error_message() {
    let index = new_index(2);
    let tensor = new_tensor_c64(&[index], &[1.0, 2.0, 3.0, 4.0]);
    let mut out_len = 0usize;

    assert_eq!(
        t4a_tensor_copy_payload_f64(tensor, std::ptr::null_mut(), 0, &mut out_len,),
        T4A_INVALID_ARGUMENT
    );
    assert!(last_error().contains("tensor storage payload operation failed"));
    assert!(last_error().contains("expected f64 storage"));

    t4a_tensor_release(tensor);
    t4a_index_release(index);
}

#[cfg(feature = "test-support")]
#[test]
fn exported_dense_function_preserves_real_deferred_backend_diagnostic() {
    let index = new_index(2);
    let core_index = unsafe { (*index).inner().clone() };
    let internal = InternalTensor::from_dense(vec![core_index], vec![1.0_f64, 2.0])
        .unwrap()
        .with_deferred_storage_error_for_testing(TensorStorageError::Materialization {
            source: Arc::new(std::io::Error::other(
                "forced deferred C API backend failure",
            )),
        });
    let tensor = Box::into_raw(Box::new(t4a_tensor::new(internal)));
    let mut out_len = 0usize;

    assert_eq!(
        t4a_tensor_copy_dense_f64(tensor, std::ptr::null_mut(), 0, &mut out_len),
        T4A_INTERNAL_ERROR
    );
    let error = last_error();
    assert!(error.contains("forced deferred C API backend failure"));
    assert!(error.contains("tensor dense f64 materialization failed"));

    t4a_tensor_release(tensor);
    t4a_index_release(index);
}

#[test]
fn raw_slice_null_pointer_precedes_byte_length_validation() {
    let error = read_plain_slice::<f64>("data", std::ptr::null(), 1).unwrap_err();
    assert_eq!(error.0, T4A_NULL_POINTER);
    assert!(error.1.contains("data is null"));
}

#[test]
fn raw_slice_accepts_positive_length_zst() {
    let len = 3;
    let values = read_plain_slice::<()>("data", std::ptr::NonNull::dangling().as_ptr(), len)
        .expect("ZST slices should not panic or fail validation");
    assert_eq!(values, vec![(); len]);
}

#[test]
fn test_tensor_select_indices_f64_drops_selected_axes() {
    let i = new_index(2);
    let j = new_index(3);
    let k = new_index(4);
    let data: Vec<f64> = (0..24).map(|value| value as f64).collect();
    let tensor = new_tensor_f64(&[i, j, k], &data);

    let selected = [i as *const t4a_index, k as *const t4a_index];
    let positions = [1usize, 2usize];
    let mut out = std::ptr::null_mut();
    assert_eq!(
        t4a_tensor_select_indices(tensor, 2, selected.as_ptr(), positions.as_ptr(), &mut out),
        T4A_SUCCESS
    );
    assert!(!out.is_null());

    let mut dims_len = 0usize;
    assert_eq!(
        t4a_tensor_dims(out, std::ptr::null_mut(), 0, &mut dims_len),
        T4A_SUCCESS
    );
    let mut dims = vec![0usize; dims_len];
    assert_eq!(
        t4a_tensor_dims(out, dims.as_mut_ptr(), dims.len(), &mut dims_len),
        T4A_SUCCESS
    );
    assert_eq!(dims, vec![3]);
    assert_eq!(read_dense_f64(out), vec![13.0, 15.0, 17.0]);

    t4a_tensor_release(out);
    t4a_tensor_release(tensor);
    t4a_index_release(i);
    t4a_index_release(j);
    t4a_index_release(k);
}

#[test]
fn test_tensor_select_indices_c64_drops_one_axis() {
    let i = new_index(2);
    let j = new_index(3);
    let data = [
        0.0, 10.0, 1.0, 11.0, 2.0, 12.0, 3.0, 13.0, 4.0, 14.0, 5.0, 15.0,
    ];
    let tensor = new_tensor_c64(&[i, j], &data);

    let selected = [j as *const t4a_index];
    let positions = [1usize];
    let mut out = std::ptr::null_mut();
    assert_eq!(
        t4a_tensor_select_indices(tensor, 1, selected.as_ptr(), positions.as_ptr(), &mut out),
        T4A_SUCCESS
    );
    assert!(!out.is_null());
    assert_eq!(read_payload_c64(out), vec![2.0, 12.0, 3.0, 13.0]);

    t4a_tensor_release(out);
    t4a_tensor_release(tensor);
    t4a_index_release(i);
    t4a_index_release(j);
}

#[test]
fn test_tensor_select_indices_rejects_invalid_inputs() {
    let i = new_index(2);
    let j = new_index(3);
    let k = new_index(4);
    let tensor = new_tensor_f64(&[i, j], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let mut out = std::ptr::null_mut();

    let selected = [k as *const t4a_index];
    let positions = [0usize];
    assert_eq!(
        t4a_tensor_select_indices(tensor, 1, selected.as_ptr(), positions.as_ptr(), &mut out),
        T4A_INVALID_ARGUMENT
    );
    assert!(out.is_null());

    let selected = [j as *const t4a_index];
    let positions = [3usize];
    assert_eq!(
        t4a_tensor_select_indices(tensor, 1, selected.as_ptr(), positions.as_ptr(), &mut out),
        T4A_INVALID_ARGUMENT
    );
    assert!(out.is_null());

    let selected = [i as *const t4a_index, i as *const t4a_index];
    let positions = [0usize, 1usize];
    assert_eq!(
        t4a_tensor_select_indices(tensor, 2, selected.as_ptr(), positions.as_ptr(), &mut out),
        T4A_INVALID_ARGUMENT
    );
    assert!(out.is_null());

    assert_eq!(
        t4a_tensor_select_indices(
            tensor,
            1,
            selected.as_ptr(),
            positions.as_ptr(),
            std::ptr::null_mut()
        ),
        T4A_NULL_POINTER
    );

    t4a_tensor_release(tensor);
    t4a_index_release(i);
    t4a_index_release(j);
    t4a_index_release(k);
}

#[test]
fn test_tensor_select_indices_accepts_diagonal_storage() {
    let i = new_index(3);
    let j = new_index(3);
    let diag = [1.0, 2.0, 4.0];
    let index_ptrs = [i as *const t4a_index, j as *const t4a_index];
    let mut tensor = std::ptr::null_mut();
    assert_eq!(
        t4a_tensor_new_diag_f64(
            2,
            index_ptrs.as_ptr(),
            diag.as_ptr(),
            diag.len(),
            &mut tensor
        ),
        T4A_SUCCESS
    );

    let selected = [i as *const t4a_index];
    let positions = [1usize];
    let mut out = std::ptr::null_mut();
    assert_eq!(
        t4a_tensor_select_indices(tensor, 1, selected.as_ptr(), positions.as_ptr(), &mut out),
        T4A_SUCCESS
    );
    assert_eq!(read_dense_f64(out), vec![0.0, 2.0, 0.0]);
    assert_eq!(read_payload_dims(out), vec![3]);
    assert_eq!(read_payload_f64(out), vec![0.0, 2.0, 0.0]);

    t4a_tensor_release(out);
    t4a_tensor_release(tensor);
    t4a_index_release(i);
    t4a_index_release(j);
}

#[test]
fn test_tensor_select_indices_accepts_structured_storage() {
    let i = new_index(2);
    let j = new_index(3);
    let k = new_index(2);
    let index_ptrs = [
        i as *const t4a_index,
        j as *const t4a_index,
        k as *const t4a_index,
    ];
    let payload_dims = [2usize, 3];
    let payload_strides = [1isize, 2];
    let axis_classes = [0usize, 1, 0];
    let payload = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let mut tensor = std::ptr::null_mut();
    assert_eq!(
        t4a_tensor_new_structured_f64(
            3,
            index_ptrs.as_ptr(),
            payload.as_ptr(),
            payload.len(),
            payload_dims.as_ptr(),
            payload_dims.len(),
            payload_strides.as_ptr(),
            payload_strides.len(),
            axis_classes.as_ptr(),
            axis_classes.len(),
            &mut tensor,
        ),
        T4A_SUCCESS
    );

    let selected = [j as *const t4a_index];
    let positions = [1usize];
    let mut out = std::ptr::null_mut();
    assert_eq!(
        t4a_tensor_select_indices(tensor, 1, selected.as_ptr(), positions.as_ptr(), &mut out),
        T4A_SUCCESS
    );
    assert!(!out.is_null());

    let mut kind = t4a_storage_kind::Dense;
    assert_eq!(t4a_tensor_storage_kind(out, &mut kind), T4A_SUCCESS);
    assert_eq!(kind, t4a_storage_kind::Diagonal);
    assert_eq!(read_payload_dims(out), vec![2]);
    assert_eq!(read_payload_strides(out), vec![1]);
    assert_eq!(read_axis_classes(out), vec![0, 0]);
    assert_eq!(read_payload_f64(out), vec![3.0, 4.0]);
    assert_eq!(read_dense_f64(out), vec![3.0, 0.0, 0.0, 4.0]);

    t4a_tensor_release(out);
    t4a_tensor_release(tensor);
    for index in [k, j, i] {
        t4a_index_release(index);
    }
}

#[test]
fn test_tensor_diag_f64_storage_payload_roundtrip() {
    let i = new_index(3);
    let j = new_index(3);
    let index_ptrs = [i as *const t4a_index, j as *const t4a_index];
    let diag = [1.0, 2.0, 4.0];
    let mut tensor = std::ptr::null_mut();
    assert_eq!(
        t4a_tensor_new_diag_f64(
            2,
            index_ptrs.as_ptr(),
            diag.as_ptr(),
            diag.len(),
            &mut tensor
        ),
        T4A_SUCCESS
    );
    assert!(!tensor.is_null());

    let mut kind = t4a_storage_kind::Dense;
    assert_eq!(t4a_tensor_storage_kind(tensor, &mut kind), T4A_SUCCESS);
    assert_eq!(kind, t4a_storage_kind::Diagonal);

    let mut payload_rank = 0usize;
    assert_eq!(
        t4a_tensor_payload_rank(tensor, &mut payload_rank),
        T4A_SUCCESS
    );
    assert_eq!(payload_rank, 1);

    let mut payload_len = 0usize;
    assert_eq!(
        t4a_tensor_payload_len(tensor, &mut payload_len),
        T4A_SUCCESS
    );
    assert_eq!(payload_len, 3);
    assert_eq!(read_payload_dims(tensor), vec![3]);
    assert_eq!(read_payload_strides(tensor), vec![1]);
    assert_eq!(read_axis_classes(tensor), vec![0, 0]);
    assert_eq!(read_payload_f64(tensor), diag);
    assert_eq!(
        read_dense_f64(tensor),
        vec![1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 4.0]
    );

    t4a_tensor_release(tensor);
    t4a_index_release(j);
    t4a_index_release(i);
}

#[test]
fn test_tensor_structured_f64_storage_payload_roundtrip() {
    let i = new_index(2);
    let j = new_index(3);
    let k = new_index(2);
    let index_ptrs = [
        i as *const t4a_index,
        j as *const t4a_index,
        k as *const t4a_index,
    ];
    let payload_dims = [2usize, 3];
    let payload_strides = [1isize, 2];
    let axis_classes = [0usize, 1, 0];
    let payload = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let mut tensor = std::ptr::null_mut();
    assert_eq!(
        t4a_tensor_new_structured_f64(
            3,
            index_ptrs.as_ptr(),
            payload.as_ptr(),
            payload.len(),
            payload_dims.as_ptr(),
            payload_dims.len(),
            payload_strides.as_ptr(),
            payload_strides.len(),
            axis_classes.as_ptr(),
            axis_classes.len(),
            &mut tensor,
        ),
        T4A_SUCCESS
    );
    assert!(!tensor.is_null());

    let mut kind = t4a_storage_kind::Dense;
    assert_eq!(t4a_tensor_storage_kind(tensor, &mut kind), T4A_SUCCESS);
    assert_eq!(kind, t4a_storage_kind::Structured);
    assert_eq!(read_payload_dims(tensor), payload_dims);
    assert_eq!(read_payload_strides(tensor), payload_strides);
    assert_eq!(read_axis_classes(tensor), axis_classes);
    assert_eq!(read_payload_f64(tensor), payload);
    assert_eq!(
        read_dense_f64(tensor),
        vec![1.0, 0.0, 3.0, 0.0, 5.0, 0.0, 0.0, 2.0, 0.0, 4.0, 0.0, 6.0]
    );

    t4a_tensor_release(tensor);
    for index in [k, j, i] {
        t4a_index_release(index);
    }
}

#[test]
fn test_tensor_structured_c64_storage_payload_roundtrip() {
    let i = new_index(2);
    let j = new_index(2);
    let index_ptrs = [i as *const t4a_index, j as *const t4a_index];
    let payload_dims = [2usize];
    let payload_strides = [1isize];
    let axis_classes = [0usize, 0];
    let payload = [1.0, 0.5, 2.0, -1.5];
    let mut tensor = std::ptr::null_mut();
    assert_eq!(
        t4a_tensor_new_structured_c64(
            2,
            index_ptrs.as_ptr(),
            payload.as_ptr(),
            payload.len() / 2,
            payload_dims.as_ptr(),
            payload_dims.len(),
            payload_strides.as_ptr(),
            payload_strides.len(),
            axis_classes.as_ptr(),
            axis_classes.len(),
            &mut tensor,
        ),
        T4A_SUCCESS
    );
    assert!(!tensor.is_null());

    let mut scalar_kind = t4a_scalar_kind::F64;
    assert_eq!(
        t4a_tensor_scalar_kind(tensor, &mut scalar_kind),
        T4A_SUCCESS
    );
    assert_eq!(scalar_kind, t4a_scalar_kind::C64);

    let mut storage_kind = t4a_storage_kind::Dense;
    assert_eq!(
        t4a_tensor_storage_kind(tensor, &mut storage_kind),
        T4A_SUCCESS
    );
    assert_eq!(storage_kind, t4a_storage_kind::Diagonal);
    assert_eq!(read_payload_c64(tensor), payload);

    t4a_tensor_release(tensor);
    t4a_index_release(j);
    t4a_index_release(i);
}

#[test]
fn test_tensor_diag_c64_storage_payload_roundtrip() {
    let i = new_index(2);
    let j = new_index(2);
    let index_ptrs = [i as *const t4a_index, j as *const t4a_index];
    let diag = [3.0, 0.25, 4.0, -0.5];
    let mut tensor = std::ptr::null_mut();
    assert_eq!(
        t4a_tensor_new_diag_c64(2, index_ptrs.as_ptr(), diag.as_ptr(), 2, &mut tensor),
        T4A_SUCCESS
    );
    assert!(!tensor.is_null());
    assert_eq!(read_payload_c64(tensor), diag);

    let mut storage_kind = t4a_storage_kind::Dense;
    assert_eq!(
        t4a_tensor_storage_kind(tensor, &mut storage_kind),
        T4A_SUCCESS
    );
    assert_eq!(storage_kind, t4a_storage_kind::Diagonal);

    t4a_tensor_release(tensor);
    t4a_index_release(j);
    t4a_index_release(i);
}

#[test]
fn test_tensor_structured_constructor_rejects_logical_dim_mismatch() {
    let i = new_index(2);
    let j = new_index(4);
    let k = new_index(2);
    let index_ptrs = [
        i as *const t4a_index,
        j as *const t4a_index,
        k as *const t4a_index,
    ];
    let payload_dims = [2usize, 3];
    let payload_strides = [1isize, 2];
    let axis_classes = [0usize, 1, 0];
    let payload = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let mut tensor = std::ptr::null_mut();

    assert_eq!(
        t4a_tensor_new_structured_f64(
            3,
            index_ptrs.as_ptr(),
            payload.as_ptr(),
            payload.len(),
            payload_dims.as_ptr(),
            payload_dims.len(),
            payload_strides.as_ptr(),
            payload_strides.len(),
            axis_classes.as_ptr(),
            axis_classes.len(),
            &mut tensor,
        ),
        T4A_INVALID_ARGUMENT
    );
    assert!(tensor.is_null());

    for index in [k, j, i] {
        t4a_index_release(index);
    }
}

#[test]
fn test_tensor_payload_readback_reports_buffer_and_type_errors() {
    let i = new_index(2);
    let tensor = new_tensor_f64(&[i as *const t4a_index], &[1.0, 2.0]);

    let mut len = 0usize;
    let mut dim = 0usize;
    assert_eq!(
        t4a_tensor_payload_dims(tensor, &mut dim, 0, &mut len),
        T4A_BUFFER_TOO_SMALL
    );
    assert_eq!(len, 1);

    let mut payload = [0.0_f64; 1];
    assert_eq!(
        t4a_tensor_copy_payload_f64(tensor, payload.as_mut_ptr(), payload.len(), &mut len),
        T4A_BUFFER_TOO_SMALL
    );
    assert_eq!(len, 2);

    let mut complex_len = 0usize;
    assert_eq!(
        t4a_tensor_copy_payload_c64(tensor, std::ptr::null_mut(), 0, &mut complex_len),
        T4A_INVALID_ARGUMENT
    );

    t4a_tensor_release(tensor);
    t4a_index_release(i);
}

#[test]
fn test_tensor_contract_matches_matrix_multiplication() {
    let i = new_index(2);
    let j = new_index(3);
    let k = new_index(2);

    let mut j_clone = std::ptr::null_mut();
    assert_eq!(crate::index::t4a_index_clone(j, &mut j_clone), T4A_SUCCESS);

    // A = [ [1, 3, 5], [2, 4, 6] ] in column-major [2, 3]
    let a = new_tensor_f64(&[i, j], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    // B = [ [7, 10], [8, 11], [9, 12] ] in column-major [3, 2]
    let mut b = std::ptr::null_mut();
    assert_eq!(
        t4a_tensor_new_dense_f64(
            2,
            [j_clone as *const t4a_index, k as *const t4a_index].as_ptr(),
            [7.0, 8.0, 9.0, 10.0, 11.0, 12.0].as_ptr(),
            6,
            &mut b
        ),
        T4A_SUCCESS
    );

    let mut c = std::ptr::null_mut();
    assert_eq!(t4a_tensor_contract(a, b, &mut c), T4A_SUCCESS);
    assert_eq!(read_dense_f64(c), vec![76.0, 100.0, 103.0, 136.0]);

    t4a_tensor_release(c);
    t4a_tensor_release(b);
    t4a_tensor_release(a);
    t4a_index_release(j_clone);
    t4a_index_release(k);
    t4a_index_release(j);
    t4a_index_release(i);
}

#[test]
fn test_tensor_contract_retain_preserves_shared_index() {
    let i = new_index(2);
    let j = new_index(3);
    let k = new_index(2);

    let mut j_clone = std::ptr::null_mut();
    assert_eq!(crate::index::t4a_index_clone(j, &mut j_clone), T4A_SUCCESS);

    let a = new_tensor_f64(&[i, j], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let b = new_tensor_f64(
        &[j_clone as *const t4a_index, k],
        &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
    );

    let retain = [j as *const t4a_index];
    let mut c = std::ptr::null_mut();
    assert_eq!(
        t4a_tensor_contract_retain(a, b, retain.as_ptr(), retain.len(), &mut c),
        T4A_SUCCESS
    );

    assert_eq!(unsafe { (*c).inner().dims() }, vec![2, 3, 2]);
    assert_eq!(
        read_dense_f64(c),
        vec![7.0, 14.0, 24.0, 32.0, 45.0, 54.0, 10.0, 20.0, 33.0, 44.0, 60.0, 72.0]
    );

    t4a_tensor_release(c);
    t4a_tensor_release(b);
    t4a_tensor_release(a);
    t4a_index_release(j_clone);
    t4a_index_release(k);
    t4a_index_release(j);
    t4a_index_release(i);
}

#[test]
fn test_tensor_contract_many_retain_preserves_batch_index() {
    let i = new_index(2);
    let j = new_index(3);
    let k = new_index(2);
    let l = new_index(2);

    let mut j_clone = std::ptr::null_mut();
    assert_eq!(crate::index::t4a_index_clone(j, &mut j_clone), T4A_SUCCESS);
    let mut k_clone = std::ptr::null_mut();
    assert_eq!(crate::index::t4a_index_clone(k, &mut k_clone), T4A_SUCCESS);

    let a = new_tensor_f64(&[i, j], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let b = new_tensor_f64(
        &[j_clone as *const t4a_index, k],
        &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
    );
    let c = new_tensor_f64(&[k_clone as *const t4a_index, l], &[2.0, 3.0, 4.0, 5.0]);

    let tensors = [
        a as *const t4a_tensor,
        b as *const t4a_tensor,
        c as *const t4a_tensor,
    ];
    let retain = [j as *const t4a_index];
    let mut out = std::ptr::null_mut();
    assert_eq!(
        t4a_tensor_contract_many_retain(
            tensors.as_ptr(),
            tensors.len(),
            retain.as_ptr(),
            retain.len(),
            &mut out
        ),
        T4A_SUCCESS
    );

    assert_eq!(unsafe { (*out).inner().dims() }, vec![2, 3, 2]);
    assert_eq!(
        read_dense_f64(out),
        vec![44.0, 88.0, 147.0, 196.0, 270.0, 324.0, 78.0, 156.0, 261.0, 348.0, 480.0, 576.0,]
    );

    t4a_tensor_release(out);
    t4a_tensor_release(c);
    t4a_tensor_release(b);
    t4a_tensor_release(a);
    t4a_index_release(k_clone);
    t4a_index_release(j_clone);
    t4a_index_release(l);
    t4a_index_release(k);
    t4a_index_release(j);
    t4a_index_release(i);
}

#[test]
fn test_tensor_svd_rank2() {
    let i = new_index(3);
    let j = new_index(4);
    let tensor = new_tensor_f64(
        &[i as *const t4a_index, j as *const t4a_index],
        &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0],
    );

    let mut u = std::ptr::null_mut();
    let mut s = std::ptr::null_mut();
    let mut v = std::ptr::null_mut();
    assert_eq!(
        t4a_tensor_svd(
            tensor,
            [i as *const t4a_index].as_ptr(),
            1,
            std::ptr::null(),
            0,
            &mut u,
            &mut s,
            &mut v
        ),
        T4A_SUCCESS
    );

    assert_eq!(unsafe { (*u).inner().dims() }, vec![3, 3]);
    assert_eq!(unsafe { (*s).inner().dims() }, vec![3, 3]);
    assert_eq!(unsafe { (*v).inner().dims() }, vec![4, 3]);

    let reconstructed = reconstruct_svd(u, s, v);
    assert_tensors_close(&reconstructed, unsafe { (*tensor).inner() }, 1.0e-10);

    for handle in [v, s, u, tensor] {
        t4a_tensor_release(handle);
    }
    for index in [j, i] {
        t4a_index_release(index);
    }
}

#[test]
fn test_tensor_svd_truncation() {
    let i = new_index(3);
    let j = new_index(3);
    let tensor = new_tensor_f64(
        &[i as *const t4a_index, j as *const t4a_index],
        &[4.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    );

    let mut u = std::ptr::null_mut();
    let mut s = std::ptr::null_mut();
    let mut v = std::ptr::null_mut();
    assert_eq!(
        t4a_tensor_svd(
            tensor,
            [i as *const t4a_index].as_ptr(),
            1,
            std::ptr::null(),
            1,
            &mut u,
            &mut s,
            &mut v
        ),
        T4A_SUCCESS
    );

    assert_eq!(unsafe { (*u).inner().dims() }, vec![3, 1]);
    assert_eq!(unsafe { (*s).inner().dims() }, vec![1, 1]);
    assert_eq!(unsafe { (*v).inner().dims() }, vec![3, 1]);

    let reconstructed = reconstruct_svd(u, s, v);
    let expected = internal_tensor_f64(
        &[i as *const t4a_index, j as *const t4a_index],
        &[4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    );
    assert_tensors_close(&reconstructed, &expected, 1.0e-10);

    for handle in [v, s, u, tensor] {
        t4a_tensor_release(handle);
    }
    for index in [j, i] {
        t4a_index_release(index);
    }
}

#[test]
fn test_tensor_svd_rank3() {
    let i = new_index(2);
    let j = new_index(3);
    let k = new_index(4);
    let data = vec![
        1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ];
    let tensor = new_tensor_f64(
        &[
            i as *const t4a_index,
            j as *const t4a_index,
            k as *const t4a_index,
        ],
        &data,
    );

    let mut u = std::ptr::null_mut();
    let mut s = std::ptr::null_mut();
    let mut v = std::ptr::null_mut();
    assert_eq!(
        t4a_tensor_svd(
            tensor,
            [i as *const t4a_index, j as *const t4a_index].as_ptr(),
            2,
            std::ptr::null(),
            0,
            &mut u,
            &mut s,
            &mut v
        ),
        T4A_SUCCESS
    );

    assert_eq!(unsafe { (*u).inner().dims() }, vec![2, 3, 4]);
    assert_eq!(unsafe { (*s).inner().dims() }, vec![4, 4]);
    assert_eq!(unsafe { (*v).inner().dims() }, vec![4, 4]);

    let reconstructed = reconstruct_svd(u, s, v);
    assert_tensors_close(&reconstructed, unsafe { (*tensor).inner() }, 1.0e-10);

    for handle in [v, s, u, tensor] {
        t4a_tensor_release(handle);
    }
    for index in [k, j, i] {
        t4a_index_release(index);
    }
}

#[test]
fn test_tensor_svd_explicit_policy() {
    let i = new_index(3);
    let j = new_index(3);
    let tensor = new_tensor_f64(
        &[i as *const t4a_index, j as *const t4a_index],
        &[4.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    );
    let policy = t4a_svd_truncation_policy {
        threshold: 0.3,
        scale: t4a_threshold_scale::Relative,
        measure: t4a_singular_value_measure::Value,
        rule: t4a_truncation_rule::PerValue,
    };

    let mut u = std::ptr::null_mut();
    let mut s = std::ptr::null_mut();
    let mut v = std::ptr::null_mut();
    assert_eq!(
        t4a_tensor_svd(
            tensor,
            [i as *const t4a_index].as_ptr(),
            1,
            &policy,
            0,
            &mut u,
            &mut s,
            &mut v
        ),
        T4A_SUCCESS
    );

    assert_eq!(unsafe { (*u).inner().dims() }, vec![3, 1]);
    assert_eq!(unsafe { (*s).inner().dims() }, vec![1, 1]);
    assert_eq!(unsafe { (*v).inner().dims() }, vec![3, 1]);

    for handle in [v, s, u, tensor] {
        t4a_tensor_release(handle);
    }
    for index in [j, i] {
        t4a_index_release(index);
    }
}

#[test]
fn test_tensor_qr_rank2() {
    let i = new_index(3);
    let j = new_index(4);
    let tensor = new_tensor_f64(
        &[i as *const t4a_index, j as *const t4a_index],
        &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0],
    );

    let mut q = std::ptr::null_mut();
    let mut r = std::ptr::null_mut();
    assert_eq!(
        t4a_tensor_qr(
            tensor,
            [i as *const t4a_index].as_ptr(),
            1,
            0.0,
            &mut q,
            &mut r
        ),
        T4A_SUCCESS
    );

    assert_eq!(unsafe { (*q).inner().dims() }, vec![3, 3]);
    assert_eq!(unsafe { (*r).inner().dims() }, vec![3, 4]);

    let reconstructed = reconstruct_qr(q, r);
    assert_tensors_close(&reconstructed, unsafe { (*tensor).inner() }, 1.0e-10);

    for handle in [r, q, tensor] {
        t4a_tensor_release(handle);
    }
    for index in [j, i] {
        t4a_index_release(index);
    }
}

#[test]
fn test_tensor_qr_rank3() {
    let i = new_index(2);
    let j = new_index(3);
    let k = new_index(4);
    let data = vec![
        1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ];
    let tensor = new_tensor_f64(
        &[
            i as *const t4a_index,
            j as *const t4a_index,
            k as *const t4a_index,
        ],
        &data,
    );

    let mut q = std::ptr::null_mut();
    let mut r = std::ptr::null_mut();
    assert_eq!(
        t4a_tensor_qr(
            tensor,
            [i as *const t4a_index, j as *const t4a_index].as_ptr(),
            2,
            0.0,
            &mut q,
            &mut r
        ),
        T4A_SUCCESS
    );

    assert_eq!(unsafe { (*q).inner().dims() }, vec![2, 3, 4]);
    assert_eq!(unsafe { (*r).inner().dims() }, vec![4, 4]);

    let reconstructed = reconstruct_qr(q, r);
    assert_tensors_close(&reconstructed, unsafe { (*tensor).inner() }, 1.0e-10);

    for handle in [r, q, tensor] {
        t4a_tensor_release(handle);
    }
    for index in [k, j, i] {
        t4a_index_release(index);
    }
}

#[test]
fn test_tensor_qr_rank2_c64() {
    let i = new_index(2);
    let j = new_index(2);
    let data = [
        Complex64::new(1.0, 1.0),
        Complex64::new(3.0, -0.5),
        Complex64::new(2.0, -2.0),
        Complex64::new(-1.0, 0.25),
    ];
    let tensor = new_tensor_c64(
        &[i as *const t4a_index, j as *const t4a_index],
        &[1.0, 1.0, 3.0, -0.5, 2.0, -2.0, -1.0, 0.25],
    );

    let mut q = std::ptr::null_mut();
    let mut r = std::ptr::null_mut();
    assert_eq!(
        t4a_tensor_qr(
            tensor,
            [i as *const t4a_index].as_ptr(),
            1,
            0.0,
            &mut q,
            &mut r
        ),
        T4A_SUCCESS
    );

    let reconstructed = reconstruct_qr(q, r);
    let expected = internal_tensor_c64(&[i as *const t4a_index, j as *const t4a_index], &data);
    assert_tensors_close(&reconstructed, &expected, 1.0e-10);

    for handle in [r, q, tensor] {
        t4a_tensor_release(handle);
    }
    for index in [j, i] {
        t4a_index_release(index);
    }
}

#[test]
fn test_tensor_qr_zero_rtol_disables_truncation() {
    let i = new_index(2);
    let j = new_index(2);
    let tensor = new_tensor_f64(
        &[i as *const t4a_index, j as *const t4a_index],
        &[1.0, 0.0, 0.0, 1.0e-16],
    );

    let mut q_exact = std::ptr::null_mut();
    let mut r_exact = std::ptr::null_mut();
    assert_eq!(
        t4a_tensor_qr(
            tensor,
            [i as *const t4a_index].as_ptr(),
            1,
            0.0,
            &mut q_exact,
            &mut r_exact
        ),
        T4A_SUCCESS
    );

    assert_eq!(unsafe { (*q_exact).inner().dims() }, vec![2, 2]);
    assert_eq!(unsafe { (*r_exact).inner().dims() }, vec![2, 2]);

    let mut q_truncated = std::ptr::null_mut();
    let mut r_truncated = std::ptr::null_mut();
    assert_eq!(
        t4a_tensor_qr(
            tensor,
            [i as *const t4a_index].as_ptr(),
            1,
            1.0e-12,
            &mut q_truncated,
            &mut r_truncated
        ),
        T4A_SUCCESS
    );

    assert_eq!(unsafe { (*q_truncated).inner().dims() }, vec![2, 1]);
    assert_eq!(unsafe { (*r_truncated).inner().dims() }, vec![1, 2]);

    for handle in [r_truncated, q_truncated, r_exact, q_exact, tensor] {
        t4a_tensor_release(handle);
    }
    for index in [j, i] {
        t4a_index_release(index);
    }
}
