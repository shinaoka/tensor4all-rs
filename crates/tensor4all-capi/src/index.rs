//! C API for immutable `Index` handles.

use std::collections::hash_map::DefaultHasher;
use std::ffi::{c_char, CStr};
use std::hash::{Hash, Hasher};
use std::panic::{catch_unwind, AssertUnwindSafe};

use crate::types::{t4a_index, InternalIndex};
use crate::{
    capi_error, clone_opaque, err_buffer_too_small, err_null_pointer, is_assigned_opaque,
    panic_message, release_opaque, run_catching, set_last_error, t4a_status_code, CapiResult,
    T4A_INTERNAL_ERROR, T4A_INVALID_ARGUMENT, T4A_NULL_POINTER, T4A_SUCCESS,
};
use tensor4all_core::index::{DynId, Index, TagSet};
use tensor4all_core::{IndexLike, TagSetLike};

/// Release an index handle.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_index_release(obj: *mut t4a_index) {
    release_opaque(obj);
}

/// Clone an index handle.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_index_clone(
    src: *const t4a_index,
    out: *mut *mut t4a_index,
) -> t4a_status_code {
    clone_opaque(src, out)
}

/// Check whether an index handle is assigned.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_index_is_assigned(obj: *const t4a_index) -> i32 {
    is_assigned_opaque(obj)
}

fn read_optional_tags_csv(
    tags_csv: *const c_char,
) -> Result<Option<String>, (t4a_status_code, String)> {
    if tags_csv.is_null() {
        return Ok(None);
    }

    let c_str = unsafe { CStr::from_ptr(tags_csv) };
    let tags = c_str
        .to_str()
        .map_err(|e| capi_error(T4A_INVALID_ARGUMENT, e))?;
    Ok(Some(tags.to_string()))
}

fn build_index(
    dim: usize,
    tags_csv: *const c_char,
    plev: i64,
) -> Result<InternalIndex, (t4a_status_code, String)> {
    if dim == 0 {
        return Err(capi_error(
            T4A_INVALID_ARGUMENT,
            "dim must be greater than zero",
        ));
    }
    if plev < 0 {
        return Err(capi_error(
            T4A_INVALID_ARGUMENT,
            "plev must be greater than or equal to zero",
        ));
    }

    let tags = read_optional_tags_csv(tags_csv)?;
    let index = match tags {
        Some(tags) => {
            Index::new_dyn_with_tag(dim, &tags).map_err(|e| capi_error(T4A_INVALID_ARGUMENT, e))?
        }
        None => Index::new_dyn(dim),
    };

    Ok(index.set_plev(plev))
}

fn build_index_with_id(
    dim: usize,
    id: u64,
    tags_csv: *const c_char,
    plev: i64,
) -> Result<InternalIndex, (t4a_status_code, String)> {
    if dim == 0 {
        return Err(capi_error(
            T4A_INVALID_ARGUMENT,
            "dim must be greater than zero",
        ));
    }
    if plev < 0 {
        return Err(capi_error(
            T4A_INVALID_ARGUMENT,
            "plev must be greater than or equal to zero",
        ));
    }

    let id = DynId(id);
    let tags = read_optional_tags_csv(tags_csv)?;
    let index = match tags {
        Some(tags) => {
            let tags = TagSet::from_str(&tags).map_err(|e| capi_error(T4A_INVALID_ARGUMENT, e))?;
            Index::new_with_tags(id, dim, tags)
        }
        None => Index::new(id, dim),
    };

    Ok(index.set_plev(plev))
}

fn require_index<'a>(ptr: *const t4a_index, what: &str) -> CapiResult<&'a InternalIndex> {
    if ptr.is_null() {
        return Err(capi_error(T4A_NULL_POINTER, format!("{what} is null")));
    }
    Ok(unsafe { &*ptr }.inner())
}

fn run_value<T: Copy, F>(out: *mut T, f: F) -> t4a_status_code
where
    F: FnOnce() -> CapiResult<T>,
{
    if out.is_null() {
        return err_null_pointer("out");
    }

    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(value)) => {
            unsafe { *out = value };
            T4A_SUCCESS
        }
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

/// Create a new index with explicit tags and prime level.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_index_new(
    dim: usize,
    tags_csv: *const c_char,
    plev: i64,
    out: *mut *mut t4a_index,
) -> t4a_status_code {
    run_catching(out, || {
        Ok(t4a_index::new(build_index(dim, tags_csv, plev)?))
    })
}

/// Create a new index with an explicit identity, tags, and prime level.
///
/// **Serialization semantics:** index IDs are internal identities used to
/// distinguish indices with identical tags and prime levels. Constructing an
/// index with an explicit ID is intended for round-tripping indices through a
/// serialized representation (file, stream, or foreign buffer) where the
/// original ID must be preserved. Callers must not fabricate IDs for
/// application logic; two indices with the same ID, tags, and prime level are
/// treated as the same index by contraction and index-matching operations.
///
/// # Safety
///
/// `tags_csv` must be a valid NUL-terminated UTF-8 string. `out` must be a
/// valid, non-null pointer to a `t4a_index*` location.
///
/// # Errors
///
/// Returns [`T4A_INVALID_ARGUMENT`] when `dim` is zero, `plev` is negative, or
/// the tag string is malformed.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_index_new_with_id(
    dim: usize,
    id: u64,
    tags_csv: *const c_char,
    plev: i64,
    out: *mut *mut t4a_index,
) -> t4a_status_code {
    run_catching(out, || {
        Ok(t4a_index::new(build_index_with_id(
            dim, id, tags_csv, plev,
        )?))
    })
}

/// Get the dimension of an index.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_index_dim(ptr: *const t4a_index, out_dim: *mut usize) -> t4a_status_code {
    if ptr.is_null() {
        return err_null_pointer("index");
    }
    if out_dim.is_null() {
        return err_null_pointer("out_dim");
    }

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
        *out_dim = (*ptr).inner().size();
        T4A_SUCCESS
    }));

    crate::unwrap_catch(result)
}

/// Get the explicit identity of an index.
///
/// **Serialization semantics:** the returned ID identifies the index for
/// round-tripping through a serialized representation. It is not a stable
/// user-facing label; callers should rely on tags and prime levels for
/// application-level identity.
///
/// # Safety
///
/// `ptr` must point to a valid `t4a_index` created by this library.
/// `out_id` must be a valid, non-null pointer to a `u64`.
///
/// # Errors
///
/// Returns a null-pointer error when `ptr` or `out_id` is null.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_index_id(ptr: *const t4a_index, out_id: *mut u64) -> t4a_status_code {
    run_value(out_id, || {
        let index = require_index(ptr, "index")?;
        Ok(index.id().value())
    })
}

/// Compare two full index handles for equality.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_index_equal(
    lhs: *const t4a_index,
    rhs: *const t4a_index,
    out_equal: *mut i32,
) -> t4a_status_code {
    run_value(out_equal, || {
        let lhs = require_index(lhs, "lhs")?;
        let rhs = require_index(rhs, "rhs")?;
        Ok((lhs == rhs) as i32)
    })
}

/// Hash the full index value for process-local hash tables.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_index_hash(ptr: *const t4a_index, out_hash: *mut u64) -> t4a_status_code {
    run_value(out_hash, || {
        let index = require_index(ptr, "index")?;
        let mut hasher = DefaultHasher::new();
        index.hash(&mut hasher);
        Ok(hasher.finish())
    })
}

/// Get the prime level of an index.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_index_plev(ptr: *const t4a_index, out_plev: *mut i64) -> t4a_status_code {
    if ptr.is_null() {
        return err_null_pointer("index");
    }
    if out_plev.is_null() {
        return err_null_pointer("out_plev");
    }

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
        *out_plev = (*ptr).inner().plev();
        T4A_SUCCESS
    }));

    crate::unwrap_catch(result)
}

/// Return a new index handle with prime level incremented by one.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_index_prime(
    ptr: *const t4a_index,
    out: *mut *mut t4a_index,
) -> t4a_status_code {
    run_catching(out, || {
        let index = require_index(ptr, "index")?;
        Ok(t4a_index::new(index.prime()))
    })
}

/// Return a new index handle with prime level reset to zero.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_index_noprime(
    ptr: *const t4a_index,
    out: *mut *mut t4a_index,
) -> t4a_status_code {
    run_catching(out, || {
        let index = require_index(ptr, "index")?;
        Ok(t4a_index::new(index.noprime()))
    })
}

/// Return a new index handle with an explicit prime level.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_index_set_plev(
    ptr: *const t4a_index,
    plev: i64,
    out: *mut *mut t4a_index,
) -> t4a_status_code {
    run_catching(out, || {
        if plev < 0 {
            return Err(capi_error(
                T4A_INVALID_ARGUMENT,
                "plev must be greater than or equal to zero",
            ));
        }
        let index = require_index(ptr, "index")?;
        Ok(t4a_index::new(index.set_plev(plev)))
    })
}

/// Copy tags as a comma-separated UTF-8 string.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_index_tags(
    ptr: *const t4a_index,
    buf: *mut u8,
    buf_len: usize,
    out_len: *mut usize,
) -> t4a_status_code {
    if ptr.is_null() {
        return err_null_pointer("index");
    }
    if out_len.is_null() {
        return err_null_pointer("out_len");
    }

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
        let tags = (*ptr)
            .inner()
            .tags()
            .iter()
            .map(|tag| tag.to_string())
            .collect::<Vec<_>>()
            .join(",");
        let required_len = match tags.len().checked_add(1) {
            Some(length) => length,
            None => {
                let message = "index tags output length overflowed usize";
                set_last_error(message);
                return T4A_INVALID_ARGUMENT;
            }
        };
        *out_len = required_len;

        if buf.is_null() {
            return T4A_SUCCESS;
        }
        if buf_len < required_len {
            return err_buffer_too_small("index tags", required_len, buf_len);
        }

        std::ptr::copy_nonoverlapping(tags.as_ptr(), buf, tags.len());
        *buf.add(tags.len()) = 0;
        T4A_SUCCESS
    }));

    crate::unwrap_catch(result)
}

/// Query whether an index has the provided tag.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_index_has_tag(
    ptr: *const t4a_index,
    tag: *const c_char,
    out_has_tag: *mut i32,
) -> t4a_status_code {
    if ptr.is_null() {
        return err_null_pointer("index");
    }
    if tag.is_null() {
        return err_null_pointer("tag");
    }
    if out_has_tag.is_null() {
        return err_null_pointer("out_has_tag");
    }

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
        let c_str = CStr::from_ptr(tag);
        let tag = c_str
            .to_str()
            .map_err(|e| crate::err_status(e, T4A_INVALID_ARGUMENT))?;
        *out_has_tag = if (*ptr).inner().tags().has_tag(tag) {
            1
        } else {
            0
        };
        Ok::<t4a_status_code, t4a_status_code>(T4A_SUCCESS)
    }));

    match result {
        Ok(Ok(code)) => code,
        Ok(Err(code)) => code,
        Err(panic) => crate::unwrap_catch(Err(panic)),
    }
}

#[cfg(test)]
mod tests;
