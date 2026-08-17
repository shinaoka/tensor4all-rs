//! C API for dense tensor construction, inspection, and contraction.

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::Arc;

use num_complex::Complex64;
use tensor4all_core::{
    contract_pair, contract_pair_with_options, contract_with_options, qr_with, svd_with,
    ContractionOptions, QrOptions, SvdOptions, SvdTruncationPolicy, TensorStorageError,
};
use tensor4all_tensorbackend::{Storage, StorageError};

use crate::types::{
    t4a_index, t4a_scalar_kind, t4a_storage_kind, t4a_svd_truncation_policy, t4a_tensor,
    InternalIndex, InternalTensor,
};
use crate::{
    capi_error, clone_opaque, err_buffer_too_small, err_null_pointer, is_assigned_opaque,
    panic_message, release_opaque, run_catching, set_last_error, t4a_status_code, CapiResult,
    T4A_BUFFER_TOO_SMALL, T4A_INTERNAL_ERROR, T4A_INVALID_ARGUMENT, T4A_NULL_POINTER, T4A_SUCCESS,
};

/// Release a tensor handle.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_tensor_release(obj: *mut t4a_tensor) {
    release_opaque(obj);
}

/// Clone a tensor handle.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_tensor_clone(
    src: *const t4a_tensor,
    out: *mut *mut t4a_tensor,
) -> t4a_status_code {
    clone_opaque(src, out)
}

/// Select fixed coordinates for tensor indices and drop those axes.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_tensor_select_indices(
    tensor: *const t4a_tensor,
    n_select: usize,
    selected_indices: *const *const t4a_index,
    positions: *const usize,
    out: *mut *mut t4a_tensor,
) -> t4a_status_code {
    run_status(|| {
        let tensor = require_tensor(tensor)?;
        if out.is_null() {
            return Err(capi_error(T4A_NULL_POINTER, "out is null"));
        }
        unsafe {
            *out = std::ptr::null_mut();
        }

        let selected_indices = read_indices_from_ptrs(n_select, selected_indices)?;
        let positions = read_plain_slice("positions", positions, n_select)?;
        let result = tensor
            .inner()
            .select_indices(&selected_indices, &positions)
            .map_err(|e| capi_error(T4A_INVALID_ARGUMENT, format!("select_indices failed: {e}")))?;
        unsafe {
            *out = box_tensor_handle(result);
        }
        Ok(())
    })
}

/// Check whether a tensor handle is assigned.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_tensor_is_assigned(obj: *const t4a_tensor) -> i32 {
    is_assigned_opaque(obj)
}

fn read_indices_from_ptrs(
    rank: usize,
    index_ptrs: *const *const t4a_index,
) -> Result<Vec<InternalIndex>, (t4a_status_code, String)> {
    if rank == 0 {
        return Ok(Vec::new());
    }
    if index_ptrs.is_null() {
        return Err(capi_error(T4A_NULL_POINTER, "index_ptrs is null"));
    }
    // Reject an impossible pointer-array byte length before any allocation
    // or pointer arithmetic, matching the scalar raw-slice contract.
    validate_raw_slice_len::<*const t4a_index>("index_ptrs", rank)?;

    let mut indices = Vec::with_capacity(rank);
    for i in 0..rank {
        let idx_ptr = unsafe { *index_ptrs.add(i) };
        if idx_ptr.is_null() {
            return Err(capi_error(
                T4A_NULL_POINTER,
                format!("index_ptrs[{i}] is null"),
            ));
        }
        indices.push(unsafe { (&*idx_ptr).inner().clone() });
    }
    Ok(indices)
}

fn dims_from_indices(indices: &[InternalIndex]) -> Vec<usize> {
    indices.iter().map(|index| index.size()).collect()
}

fn tensor_storage_error(error: TensorStorageError) -> (t4a_status_code, String) {
    capi_error(
        T4A_INTERNAL_ERROR,
        format!("tensor storage materialization failed: {error}"),
    )
}

fn storage_payload_error(error: StorageError) -> (t4a_status_code, String) {
    let status = match error {
        StorageError::ScalarKindMismatch { .. } => T4A_INVALID_ARGUMENT,
        _ => T4A_INTERNAL_ERROR,
    };
    capi_error(
        status,
        format!("tensor storage payload operation failed: {error}"),
    )
}

fn checked_dims_product(name: &str, dims: &[usize]) -> CapiResult<usize> {
    dims.iter().try_fold(1usize, |product, &dim| {
        product.checked_mul(dim).ok_or_else(|| {
            capi_error(
                T4A_INVALID_ARGUMENT,
                format!("{name} dimension product overflows usize for dims {dims:?}"),
            )
        })
    })
}

fn validate_raw_slice_len<T>(name: &str, len: usize) -> CapiResult<()> {
    let element_size = std::mem::size_of::<T>();
    if element_size != 0 && len > isize::MAX as usize / element_size {
        return Err(capi_error(
            T4A_INVALID_ARGUMENT,
            format!("{name} byte length overflows isize::MAX"),
        ));
    }
    Ok(())
}

fn read_plain_slice<T: Copy>(
    name: &str,
    ptr: *const T,
    len: usize,
) -> Result<Vec<T>, (t4a_status_code, String)> {
    if len == 0 {
        return Ok(Vec::new());
    }
    if ptr.is_null() {
        return Err(capi_error(T4A_NULL_POINTER, format!("{name} is null")));
    }
    validate_raw_slice_len::<T>(name, len)?;
    Ok(unsafe { std::slice::from_raw_parts(ptr, len) }.to_vec())
}

fn read_c64_slice(
    name: &str,
    ptr: *const f64,
    n_complex: usize,
) -> Result<Vec<Complex64>, (t4a_status_code, String)> {
    if n_complex == 0 {
        return Ok(Vec::new());
    }
    if ptr.is_null() {
        return Err(capi_error(T4A_NULL_POINTER, format!("{name} is null")));
    }
    let raw_len = n_complex.checked_mul(2).ok_or_else(|| {
        capi_error(
            T4A_INVALID_ARGUMENT,
            format!("{name} length overflows usize"),
        )
    })?;
    validate_raw_slice_len::<f64>(name, raw_len)?;
    let raw = unsafe { std::slice::from_raw_parts(ptr, raw_len) };
    Ok((0..n_complex)
        .map(|i| Complex64::new(raw[2 * i], raw[2 * i + 1]))
        .collect())
}

fn copy_plain_slice<T: Copy>(
    name: &str,
    values: &[T],
    buf: *mut T,
    buf_len: usize,
    out_len: *mut usize,
) -> CapiResult<()> {
    if out_len.is_null() {
        return Err(capi_error(
            T4A_NULL_POINTER,
            format!("{name} out_len is null"),
        ));
    }
    unsafe {
        *out_len = values.len();
    }

    if buf.is_null() {
        return Ok(());
    }
    validate_raw_slice_len::<T>(name, values.len())?;
    if buf_len < values.len() {
        return Err(capi_error(
            T4A_BUFFER_TOO_SMALL,
            format!(
                "{name} buffer too small: required {}, got {buf_len}",
                values.len()
            ),
        ));
    }
    unsafe {
        std::ptr::copy_nonoverlapping(values.as_ptr(), buf, values.len());
    }
    Ok(())
}

fn copy_c64_interleaved(
    name: &str,
    values: &[Complex64],
    buf_interleaved: *mut f64,
    n_complex: usize,
    out_len: *mut usize,
) -> CapiResult<()> {
    if out_len.is_null() {
        return Err(capi_error(
            T4A_NULL_POINTER,
            format!("{name} out_len is null"),
        ));
    }
    unsafe {
        *out_len = values.len();
    }

    if buf_interleaved.is_null() {
        return Ok(());
    }
    let raw_capacity = n_complex.checked_mul(2).ok_or_else(|| {
        capi_error(
            T4A_INVALID_ARGUMENT,
            format!("{name} output byte length overflows usize"),
        )
    })?;
    validate_raw_slice_len::<f64>(name, raw_capacity)?;
    if n_complex < values.len() {
        return Err(capi_error(
            T4A_BUFFER_TOO_SMALL,
            format!(
                "{name} buffer too small: required {}, got {n_complex}",
                values.len()
            ),
        ));
    }

    for (i, value) in values.iter().enumerate() {
        unsafe {
            *buf_interleaved.add(2 * i) = value.re;
            *buf_interleaved.add(2 * i + 1) = value.im;
        }
    }
    Ok(())
}

fn run_status<F>(f: F) -> t4a_status_code
where
    F: FnOnce() -> CapiResult<()>,
{
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(())) => T4A_SUCCESS,
        Ok(Err((code, msg))) => {
            set_last_error(&msg);
            code
        }
        Err(panic) => {
            let msg = panic_message(&*panic);
            set_last_error(&msg);
            T4A_INTERNAL_ERROR
        }
    }
}

fn require_tensor<'a>(ptr: *const t4a_tensor) -> Result<&'a t4a_tensor, (t4a_status_code, String)> {
    if ptr.is_null() {
        return Err(capi_error(T4A_NULL_POINTER, "tensor is null"));
    }
    Ok(unsafe { &*ptr })
}

fn read_tensor_refs<'a>(
    tensors: *const *const t4a_tensor,
    n_tensors: usize,
) -> Result<Vec<&'a InternalTensor>, (t4a_status_code, String)> {
    if n_tensors == 0 {
        return Ok(Vec::new());
    }
    if tensors.is_null() {
        return Err(capi_error(T4A_NULL_POINTER, "tensors is null"));
    }

    validate_raw_slice_len::<*const t4a_tensor>("tensor pointer array", n_tensors)?;
    let mut refs = Vec::with_capacity(n_tensors);
    for i in 0..n_tensors {
        let tensor = unsafe { *tensors.add(i) };
        if tensor.is_null() {
            return Err(capi_error(
                T4A_NULL_POINTER,
                format!("tensors[{i}] is null"),
            ));
        }
        refs.push(unsafe { (&*tensor).inner() });
    }
    Ok(refs)
}

fn build_svd_options(policy: *const t4a_svd_truncation_policy, maxdim: usize) -> SvdOptions {
    let mut options = SvdOptions::new();
    if !policy.is_null() {
        let policy = unsafe { *policy };
        options = options.with_policy(SvdTruncationPolicy::from(policy));
    }
    if maxdim > 0 {
        options = options.with_max_bond_dim(maxdim);
    }
    options
}

fn build_qr_options(rtol: f64) -> QrOptions {
    // `0.0` is the C-API sentinel for exact QR without truncation.
    QrOptions::new().with_rtol(rtol)
}

fn box_tensor_handle(tensor: InternalTensor) -> *mut t4a_tensor {
    Box::into_raw(Box::new(t4a_tensor::new(tensor)))
}

/// Get the rank of a tensor.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_tensor_rank(ptr: *const t4a_tensor, out_rank: *mut usize) -> t4a_status_code {
    if ptr.is_null() {
        return err_null_pointer("tensor");
    }
    if out_rank.is_null() {
        return err_null_pointer("out_rank");
    }

    let result = catch_unwind(AssertUnwindSafe(|| unsafe {
        *out_rank = (*ptr).inner().indices().len();
        T4A_SUCCESS
    }));

    crate::unwrap_catch(result)
}

/// Copy tensor dimensions in index order.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_tensor_dims(
    ptr: *const t4a_tensor,
    buf: *mut usize,
    buf_len: usize,
    out_len: *mut usize,
) -> t4a_status_code {
    if ptr.is_null() {
        return err_null_pointer("tensor");
    }
    if out_len.is_null() {
        return err_null_pointer("out_len");
    }

    let result = catch_unwind(AssertUnwindSafe(|| unsafe {
        let dims = (*ptr).inner().dims();
        *out_len = dims.len();

        if buf.is_null() {
            return T4A_SUCCESS;
        }
        if buf_len < dims.len() {
            return err_buffer_too_small("tensor dims", dims.len(), buf_len);
        }

        if let Err((code, message)) = validate_raw_slice_len::<usize>("tensor dims", dims.len()) {
            set_last_error(&message);
            return code;
        }
        std::ptr::copy_nonoverlapping(dims.as_ptr(), buf, dims.len());
        T4A_SUCCESS
    }));

    crate::unwrap_catch(result)
}

/// Copy cloned index handles describing the tensor axes.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_tensor_indices(
    ptr: *const t4a_tensor,
    buf: *mut *mut t4a_index,
    buf_len: usize,
    out_len: *mut usize,
) -> t4a_status_code {
    if ptr.is_null() {
        return err_null_pointer("tensor");
    }
    if out_len.is_null() {
        return err_null_pointer("out_len");
    }

    let result = catch_unwind(AssertUnwindSafe(|| unsafe {
        let indices = (*ptr).inner().indices();
        *out_len = indices.len();

        if buf.is_null() {
            return T4A_SUCCESS;
        }
        if buf_len < indices.len() {
            return err_buffer_too_small("tensor indices", indices.len(), buf_len);
        }

        if let Err((code, message)) =
            validate_raw_slice_len::<*mut t4a_index>("tensor indices", indices.len())
        {
            set_last_error(&message);
            return code;
        }
        for (i, index) in indices.iter().enumerate() {
            *buf.add(i) = Box::into_raw(Box::new(t4a_index::new(index.clone())));
        }
        T4A_SUCCESS
    }));

    crate::unwrap_catch(result)
}

/// Get the scalar kind of a tensor.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_tensor_scalar_kind(
    ptr: *const t4a_tensor,
    out_kind: *mut t4a_scalar_kind,
) -> t4a_status_code {
    if ptr.is_null() {
        return err_null_pointer("tensor");
    }
    if out_kind.is_null() {
        return err_null_pointer("out_kind");
    }

    let result = catch_unwind(AssertUnwindSafe(|| unsafe {
        *out_kind = t4a_scalar_kind::from_tensor((*ptr).inner());
        T4A_SUCCESS
    }));

    crate::unwrap_catch(result)
}

/// Get the storage layout kind of a tensor.
///
/// # Returns
///
/// Returns `T4A_INTERNAL_ERROR` when backend storage materialization fails;
/// retrieve the diagnostic with `t4a_last_error_message`.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_tensor_storage_kind(
    ptr: *const t4a_tensor,
    out_kind: *mut t4a_storage_kind,
) -> t4a_status_code {
    run_status(|| {
        let tensor = require_tensor(ptr)?;
        if out_kind.is_null() {
            return Err(capi_error(T4A_NULL_POINTER, "out_kind is null"));
        }
        unsafe {
            let storage = tensor.inner().storage().map_err(tensor_storage_error)?;
            *out_kind = t4a_storage_kind::from(storage.storage_kind());
        }
        Ok(())
    })
}

/// Get the rank of the compact payload tensor.
///
/// # Returns
///
/// Returns `T4A_INTERNAL_ERROR` when backend storage materialization fails;
/// retrieve the diagnostic with `t4a_last_error_message`.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_tensor_payload_rank(
    ptr: *const t4a_tensor,
    out_rank: *mut usize,
) -> t4a_status_code {
    run_status(|| {
        let tensor = require_tensor(ptr)?;
        if out_rank.is_null() {
            return Err(capi_error(T4A_NULL_POINTER, "out_rank is null"));
        }
        unsafe {
            let storage = tensor.inner().storage().map_err(tensor_storage_error)?;
            *out_rank = storage.payload_dims().len();
        }
        Ok(())
    })
}

/// Get the compact payload length in scalar elements.
///
/// # Returns
///
/// Returns `T4A_INTERNAL_ERROR` when backend storage materialization fails;
/// retrieve the diagnostic with `t4a_last_error_message`.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_tensor_payload_len(
    ptr: *const t4a_tensor,
    out_len: *mut usize,
) -> t4a_status_code {
    run_status(|| {
        let tensor = require_tensor(ptr)?;
        if out_len.is_null() {
            return Err(capi_error(T4A_NULL_POINTER, "out_len is null"));
        }
        unsafe {
            let storage = tensor.inner().storage().map_err(tensor_storage_error)?;
            *out_len = storage.payload_len();
        }
        Ok(())
    })
}

/// Copy compact payload dimensions.
///
/// # Returns
///
/// Returns `T4A_INTERNAL_ERROR` when backend storage materialization fails;
/// retrieve the diagnostic with `t4a_last_error_message`.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_tensor_payload_dims(
    ptr: *const t4a_tensor,
    buf: *mut usize,
    buf_len: usize,
    out_len: *mut usize,
) -> t4a_status_code {
    run_status(|| {
        let tensor = require_tensor(ptr)?;
        let storage = tensor.inner().storage().map_err(tensor_storage_error)?;
        copy_plain_slice(
            "payload_dims",
            storage.payload_dims(),
            buf,
            buf_len,
            out_len,
        )
    })
}

/// Copy compact payload strides in scalar elements.
///
/// # Returns
///
/// Returns `T4A_INTERNAL_ERROR` when backend storage materialization fails;
/// retrieve the diagnostic with `t4a_last_error_message`.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_tensor_payload_strides(
    ptr: *const t4a_tensor,
    buf: *mut isize,
    buf_len: usize,
    out_len: *mut usize,
) -> t4a_status_code {
    run_status(|| {
        let tensor = require_tensor(ptr)?;
        let storage = tensor.inner().storage().map_err(tensor_storage_error)?;
        copy_plain_slice(
            "payload_strides",
            storage.payload_strides(),
            buf,
            buf_len,
            out_len,
        )
    })
}

/// Copy logical-axis to payload-axis class mapping.
///
/// # Returns
///
/// Returns `T4A_INTERNAL_ERROR` when backend storage materialization fails;
/// retrieve the diagnostic with `t4a_last_error_message`.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_tensor_axis_classes(
    ptr: *const t4a_tensor,
    buf: *mut usize,
    buf_len: usize,
    out_len: *mut usize,
) -> t4a_status_code {
    run_status(|| {
        let tensor = require_tensor(ptr)?;
        let storage = tensor.inner().storage().map_err(tensor_storage_error)?;
        copy_plain_slice(
            "axis_classes",
            storage.axis_classes(),
            buf,
            buf_len,
            out_len,
        )
    })
}

/// Copy dense `f64` data in column-major order.
///
/// # Returns
///
/// Returns `T4A_INTERNAL_ERROR` when backend/materialization fails; retrieve
/// the diagnostic with `t4a_last_error_message`.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_tensor_copy_dense_f64(
    ptr: *const t4a_tensor,
    buf: *mut f64,
    buf_len: usize,
    out_len: *mut usize,
) -> t4a_status_code {
    if ptr.is_null() {
        return err_null_pointer("tensor");
    }
    if out_len.is_null() {
        return err_null_pointer("out_len");
    }
    #[cfg(test)]
    if let Some(error) = unsafe { (*ptr).test_storage_error() } {
        return crate::err_status(
            format!("tensor dense f64 materialization failed: {error}"),
            T4A_INTERNAL_ERROR,
        );
    }
    if unsafe { !(*ptr).inner().is_f64() } {
        return crate::err_status(
            "tensor dense f64 copy requires an f64 tensor",
            T4A_INVALID_ARGUMENT,
        );
    }

    let result = catch_unwind(AssertUnwindSafe(|| unsafe {
        let data = match (*ptr).inner().to_vec::<f64>() {
            Ok(data) => data,
            Err(e) => {
                return crate::err_status(
                    format!("tensor dense f64 materialization failed: {e}"),
                    T4A_INTERNAL_ERROR,
                )
            }
        };
        match copy_plain_slice("tensor dense f64", &data, buf, buf_len, out_len) {
            Ok(()) => T4A_SUCCESS,
            Err((code, message)) => {
                set_last_error(&message);
                code
            }
        }
    }));

    crate::unwrap_catch(result)
}

/// Copy dense `Complex64` data as interleaved doubles in column-major order.
///
/// # Returns
///
/// Returns `T4A_INTERNAL_ERROR` when backend/materialization fails; retrieve
/// the diagnostic with `t4a_last_error_message`.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_tensor_copy_dense_c64(
    ptr: *const t4a_tensor,
    buf_interleaved: *mut f64,
    n_complex: usize,
    out_len: *mut usize,
) -> t4a_status_code {
    if ptr.is_null() {
        return err_null_pointer("tensor");
    }
    if out_len.is_null() {
        return err_null_pointer("out_len");
    }
    if unsafe { !(*ptr).inner().is_complex() } {
        return crate::err_status(
            "tensor dense c64 copy requires a complex tensor",
            T4A_INVALID_ARGUMENT,
        );
    }

    let result = catch_unwind(AssertUnwindSafe(|| unsafe {
        let data = match (*ptr).inner().to_vec::<Complex64>() {
            Ok(data) => data,
            Err(e) => {
                return crate::err_status(
                    format!("tensor dense c64 materialization failed: {e}"),
                    T4A_INTERNAL_ERROR,
                )
            }
        };
        match copy_c64_interleaved(
            "tensor dense c64",
            &data,
            buf_interleaved,
            n_complex,
            out_len,
        ) {
            Ok(()) => T4A_SUCCESS,
            Err((code, message)) => {
                set_last_error(&message);
                code
            }
        }
    }));

    crate::unwrap_catch(result)
}

/// Copy compact payload `f64` data in payload column-major order.
///
/// # Returns
///
/// Backend/materialization failures return `T4A_INTERNAL_ERROR` with their
/// diagnostic preserved in `t4a_last_error_message`.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_tensor_copy_payload_f64(
    ptr: *const t4a_tensor,
    buf: *mut f64,
    buf_len: usize,
    out_len: *mut usize,
) -> t4a_status_code {
    run_status(|| {
        let tensor = require_tensor(ptr)?;
        let data = tensor
            .inner()
            .storage()
            .map_err(tensor_storage_error)?
            .payload_f64_col_major_vec()
            .map_err(storage_payload_error)?;
        copy_plain_slice("payload_f64", &data, buf, buf_len, out_len)
    })
}

/// Copy compact payload `Complex64` data as interleaved doubles.
///
/// # Returns
///
/// Backend/materialization failures return `T4A_INTERNAL_ERROR` with their
/// diagnostic preserved in `t4a_last_error_message`.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_tensor_copy_payload_c64(
    ptr: *const t4a_tensor,
    buf_interleaved: *mut f64,
    n_complex: usize,
    out_len: *mut usize,
) -> t4a_status_code {
    run_status(|| {
        let tensor = require_tensor(ptr)?;
        let data = tensor
            .inner()
            .storage()
            .map_err(tensor_storage_error)?
            .payload_c64_col_major_vec()
            .map_err(storage_payload_error)?;
        copy_c64_interleaved("payload_c64", &data, buf_interleaved, n_complex, out_len)
    })
}

/// Contract two tensors by matching common indices.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_tensor_contract(
    a: *const t4a_tensor,
    b: *const t4a_tensor,
    out: *mut *mut t4a_tensor,
) -> t4a_status_code {
    if a.is_null() {
        return err_null_pointer("a");
    }
    if b.is_null() {
        return err_null_pointer("b");
    }

    run_catching(out, || unsafe {
        let tensor = contract_pair((*a).inner(), (*b).inner())
            .map_err(|err| capi_error(T4A_INVALID_ARGUMENT, err))?;
        Ok(t4a_tensor::new(tensor))
    })
}

/// Contract two tensors while retaining selected shared indices as output legs.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_tensor_contract_retain(
    a: *const t4a_tensor,
    b: *const t4a_tensor,
    retain_indices: *const *const t4a_index,
    n_retain: usize,
    out: *mut *mut t4a_tensor,
) -> t4a_status_code {
    run_catching(out, || {
        let a = require_tensor(a)?;
        let b = require_tensor(b)?;
        let retain_indices = read_indices_from_ptrs(n_retain, retain_indices)?;
        let options = ContractionOptions::new().with_retain_indices(&retain_indices);
        let result = contract_pair_with_options(a.inner(), b.inner(), options)
            .map_err(|err| capi_error(T4A_INVALID_ARGUMENT, err))?;
        Ok(t4a_tensor::new(result))
    })
}

/// Contract multiple tensors while retaining selected shared indices as output legs.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_tensor_contract_many_retain(
    tensors: *const *const t4a_tensor,
    n_tensors: usize,
    retain_indices: *const *const t4a_index,
    n_retain: usize,
    out: *mut *mut t4a_tensor,
) -> t4a_status_code {
    run_catching(out, || {
        let tensors = read_tensor_refs(tensors, n_tensors)?;
        let retain_indices = read_indices_from_ptrs(n_retain, retain_indices)?;
        let options = ContractionOptions::new().with_retain_indices(&retain_indices);
        let result = contract_with_options(&tensors, options)
            .map_err(|err| capi_error(T4A_INVALID_ARGUMENT, err))?;
        Ok(t4a_tensor::new(result))
    })
}

/// Compute the SVD of a tensor split by the requested left indices.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_tensor_svd(
    tensor: *const t4a_tensor,
    left_inds: *const *const t4a_index,
    n_left: usize,
    policy: *const t4a_svd_truncation_policy,
    maxdim: usize,
    out_u: *mut *mut t4a_tensor,
    out_s: *mut *mut t4a_tensor,
    out_v: *mut *mut t4a_tensor,
) -> t4a_status_code {
    run_status(|| {
        let tensor = require_tensor(tensor)?;
        if out_u.is_null() || out_s.is_null() || out_v.is_null() {
            return Err(capi_error(
                T4A_NULL_POINTER,
                "out_u, out_s, or out_v is null",
            ));
        }

        let left_inds = read_indices_from_ptrs(n_left, left_inds)?;
        let options = build_svd_options(policy, maxdim);
        let (u, s, v) = match t4a_scalar_kind::from_tensor(tensor.inner()) {
            t4a_scalar_kind::F64 => svd_with::<f64>(tensor.inner(), &left_inds, &options),
            t4a_scalar_kind::C64 => svd_with::<Complex64>(tensor.inner(), &left_inds, &options),
        }
        .map_err(|err| capi_error(T4A_INVALID_ARGUMENT, err))?;

        unsafe {
            *out_u = box_tensor_handle(u);
            *out_s = box_tensor_handle(s);
            *out_v = box_tensor_handle(v);
        }
        Ok(())
    })
}

/// Compute the QR decomposition of a tensor split by the requested left indices.
///
/// The tensor is unfolded into a matrix by treating `left_inds[..n_left]` as row
/// indices and all remaining tensor indices as columns, then factorized as
/// `tensor = Q * R`.
///
/// # Arguments
/// * `tensor` - Input tensor handle to factorize.
/// * `left_inds` - Pointers to the indices that should appear on the left side
///
///   of the factorization.
/// * `n_left` - Number of entries in `left_inds`.
/// * `rtol` - Relative truncation tolerance for QR row-norm truncation. Pass
///
///   `0.0` to disable truncation and keep the exact QR rank. Pass any other
///   finite, non-negative value to request truncation for this call.
/// * `out_q` - Output slot that receives a newly allocated `Q` tensor handle on
///
///   success.
/// * `out_r` - Output slot that receives a newly allocated `R` tensor handle on
///
///   success.
///
/// # Returns
/// Returns `T4A_SUCCESS` on success and writes owned tensor handles to
/// `out_q` and `out_r`. The caller must release both handles with
/// [`t4a_tensor_release`].
///
/// # Errors
/// Returns:
/// - `T4A_NULL_POINTER` if `tensor`, any required index pointer, `out_q`, or
///
///   `out_r` is null.
/// - `T4A_INVALID_ARGUMENT` if the index split is invalid, QR factorization
///
///   fails, or `rtol` is negative or non-finite (for example `NaN` or
///   `+/-inf`).
/// - `T4A_INTERNAL_ERROR` if the Rust implementation panics while processing
///
///   the request.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_tensor_qr(
    tensor: *const t4a_tensor,
    left_inds: *const *const t4a_index,
    n_left: usize,
    rtol: f64,
    out_q: *mut *mut t4a_tensor,
    out_r: *mut *mut t4a_tensor,
) -> t4a_status_code {
    run_status(|| {
        let tensor = require_tensor(tensor)?;
        if out_q.is_null() || out_r.is_null() {
            return Err(capi_error(T4A_NULL_POINTER, "out_q or out_r is null"));
        }

        let left_inds = read_indices_from_ptrs(n_left, left_inds)?;
        let options = build_qr_options(rtol);
        let (q, r) = match t4a_scalar_kind::from_tensor(tensor.inner()) {
            t4a_scalar_kind::F64 => qr_with::<f64>(tensor.inner(), &left_inds, &options),
            t4a_scalar_kind::C64 => qr_with::<Complex64>(tensor.inner(), &left_inds, &options),
        }
        .map_err(|err| capi_error(T4A_INVALID_ARGUMENT, err))?;

        unsafe {
            *out_q = box_tensor_handle(q);
            *out_r = box_tensor_handle(r);
        }
        Ok(())
    })
}

/// Create a dense real tensor from column-major data.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_tensor_new_dense_f64(
    rank: usize,
    index_ptrs: *const *const t4a_index,
    data: *const f64,
    data_len: usize,
    out: *mut *mut t4a_tensor,
) -> t4a_status_code {
    run_catching(out, || {
        let indices = read_indices_from_ptrs(rank, index_ptrs)?;
        let dims = dims_from_indices(&indices);
        let expected_len = checked_dims_product("dense tensor", &dims)?;
        if expected_len > 0 && data.is_null() {
            return Err(capi_error(T4A_NULL_POINTER, "data is null"));
        }
        if data_len != expected_len {
            return Err(capi_error(
                T4A_INVALID_ARGUMENT,
                format!("data_len mismatch: expected {expected_len}, got {data_len}"),
            ));
        }

        let values = read_plain_slice("data", data, data_len)?;
        let tensor = InternalTensor::from_dense(indices, values)
            .map_err(|e| capi_error(T4A_INVALID_ARGUMENT, e))?;
        Ok(t4a_tensor::new(tensor))
    })
}

/// Create a dense complex tensor from interleaved column-major data.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_tensor_new_dense_c64(
    rank: usize,
    index_ptrs: *const *const t4a_index,
    data_interleaved: *const f64,
    n_complex: usize,
    out: *mut *mut t4a_tensor,
) -> t4a_status_code {
    run_catching(out, || {
        let indices = read_indices_from_ptrs(rank, index_ptrs)?;
        let dims = dims_from_indices(&indices);
        let expected_len = checked_dims_product("dense tensor", &dims)?;
        if expected_len > 0 && data_interleaved.is_null() {
            return Err(capi_error(T4A_NULL_POINTER, "data_interleaved is null"));
        }
        if n_complex != expected_len {
            return Err(capi_error(
                T4A_INVALID_ARGUMENT,
                format!("n_complex mismatch: expected {expected_len}, got {n_complex}"),
            ));
        }

        let values = read_c64_slice("data_interleaved", data_interleaved, n_complex)?;
        let tensor = InternalTensor::from_dense(indices, values)
            .map_err(|e| capi_error(T4A_INVALID_ARGUMENT, e))?;
        Ok(t4a_tensor::new(tensor))
    })
}

/// Create a real tensor from explicit compact structured storage.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_tensor_new_structured_f64(
    rank: usize,
    index_ptrs: *const *const t4a_index,
    data: *const f64,
    data_len: usize,
    payload_dims: *const usize,
    payload_rank: usize,
    payload_strides: *const isize,
    payload_strides_len: usize,
    axis_classes: *const usize,
    axis_classes_len: usize,
    out: *mut *mut t4a_tensor,
) -> t4a_status_code {
    run_catching(out, || {
        let indices = read_indices_from_ptrs(rank, index_ptrs)?;
        // Validate compact metadata and the exact required payload length
        // BEFORE copying the payload, so an inconsistent or attacker-sized
        // length is rejected without a large allocation.
        let payload_dims = read_plain_slice("payload_dims", payload_dims, payload_rank)?;
        let payload_strides =
            read_plain_slice("payload_strides", payload_strides, payload_strides_len)?;
        let axis_classes = read_plain_slice("axis_classes", axis_classes, axis_classes_len)?;
        let required_len =
            Storage::validate_structured_metadata(&payload_dims, &payload_strides, &axis_classes)
                .map_err(|err| {
                capi_error(
                    T4A_INVALID_ARGUMENT,
                    format!("structured payload metadata invalid: {err}"),
                )
            })?;
        if data_len != required_len {
            return Err(capi_error(
                T4A_INVALID_ARGUMENT,
                format!(
                    "data length {data_len} does not match structured payload length {required_len}"
                ),
            ));
        }
        let values = read_plain_slice("data", data, data_len)?;
        let storage = Storage::new_structured(values, payload_dims, payload_strides, axis_classes)
            .map_err(|err| capi_error(T4A_INVALID_ARGUMENT, err))?;
        let tensor = InternalTensor::from_structured_storage(indices, Arc::new(storage))
            .map_err(|err| capi_error(T4A_INVALID_ARGUMENT, err))?;
        Ok(t4a_tensor::new(tensor))
    })
}

/// Create a complex tensor from explicit compact structured storage.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_tensor_new_structured_c64(
    rank: usize,
    index_ptrs: *const *const t4a_index,
    data_interleaved: *const f64,
    n_complex: usize,
    payload_dims: *const usize,
    payload_rank: usize,
    payload_strides: *const isize,
    payload_strides_len: usize,
    axis_classes: *const usize,
    axis_classes_len: usize,
    out: *mut *mut t4a_tensor,
) -> t4a_status_code {
    run_catching(out, || {
        let indices = read_indices_from_ptrs(rank, index_ptrs)?;
        let payload_dims = read_plain_slice("payload_dims", payload_dims, payload_rank)?;
        let payload_strides =
            read_plain_slice("payload_strides", payload_strides, payload_strides_len)?;
        let axis_classes = read_plain_slice("axis_classes", axis_classes, axis_classes_len)?;
        let required_len =
            Storage::validate_structured_metadata(&payload_dims, &payload_strides, &axis_classes)
                .map_err(|err| {
                capi_error(
                    T4A_INVALID_ARGUMENT,
                    format!("structured payload metadata invalid: {err}"),
                )
            })?;
        if n_complex != required_len {
            return Err(capi_error(
                T4A_INVALID_ARGUMENT,
                format!(
                    "data length {n_complex} does not match structured payload length {required_len}"
                ),
            ));
        }
        let values = read_c64_slice("data_interleaved", data_interleaved, n_complex)?;
        let storage = Storage::new_structured(values, payload_dims, payload_strides, axis_classes)
            .map_err(|err| capi_error(T4A_INVALID_ARGUMENT, err))?;
        let tensor = InternalTensor::from_structured_storage(indices, Arc::new(storage))
            .map_err(|err| capi_error(T4A_INVALID_ARGUMENT, err))?;
        Ok(t4a_tensor::new(tensor))
    })
}

/// Create a real diagonal tensor from compact diagonal payload data.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_tensor_new_diag_f64(
    rank: usize,
    index_ptrs: *const *const t4a_index,
    diag_data: *const f64,
    diag_len: usize,
    out: *mut *mut t4a_tensor,
) -> t4a_status_code {
    run_catching(out, || {
        let indices = read_indices_from_ptrs(rank, index_ptrs)?;
        // The diagonal payload must match the declared index dimension; reject
        // a mismatched length BEFORE copying the payload. The diagonal
        // constructor remains the authority on index-dimension equality.
        let expected = indices.first().map(|index| index.size()).unwrap_or(0);
        if diag_len != expected {
            return Err(capi_error(
                T4A_INVALID_ARGUMENT,
                format!(
                    "diagonal data length {diag_len} does not match index dimension {expected}"
                ),
            ));
        }
        let values = read_plain_slice("diag_data", diag_data, diag_len)?;
        let tensor = InternalTensor::from_diag(indices, values)
            .map_err(|err| capi_error(T4A_INVALID_ARGUMENT, err))?;
        Ok(t4a_tensor::new(tensor))
    })
}

/// Create a complex diagonal tensor from compact diagonal payload data.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_tensor_new_diag_c64(
    rank: usize,
    index_ptrs: *const *const t4a_index,
    diag_data_interleaved: *const f64,
    n_complex: usize,
    out: *mut *mut t4a_tensor,
) -> t4a_status_code {
    run_catching(out, || {
        let indices = read_indices_from_ptrs(rank, index_ptrs)?;
        let expected = indices.first().map(|index| index.size()).unwrap_or(0);
        if n_complex != expected {
            return Err(capi_error(
                T4A_INVALID_ARGUMENT,
                format!(
                    "diagonal data length {n_complex} does not match index dimension {expected}"
                ),
            ));
        }
        let values = read_c64_slice("diag_data_interleaved", diag_data_interleaved, n_complex)?;
        let tensor = InternalTensor::from_diag(indices, values)
            .map_err(|err| capi_error(T4A_INVALID_ARGUMENT, err))?;
        Ok(t4a_tensor::new(tensor))
    })
}

#[cfg(test)]
mod tests;
