//! Opaque types for C API
//!
//! All Rust objects are wrapped in opaque pointers to hide implementation
//! details from C code.

use std::ffi::c_void;
use tensor4all_core_common::index::{DefaultIndex, DynId, NoSymmSpace};

/// The internal index type we're wrapping
pub(crate) type InternalIndex = DefaultIndex<DynId, NoSymmSpace>;

/// Opaque index type for C API
///
/// Wraps `DefaultIndex<DynId, NoSymmSpace>` which corresponds to ITensors.jl's `Index{Int}`.
///
/// The internal structure is hidden using a void pointer.
#[repr(C)]
pub struct t4a_index {
    pub(crate) _private: *const c_void,
}

impl t4a_index {
    /// Create a new t4a_index from an InternalIndex
    pub(crate) fn new(index: InternalIndex) -> Self {
        Self {
            _private: Box::into_raw(Box::new(index)) as *const c_void,
        }
    }

    /// Get a reference to the inner InternalIndex
    pub(crate) fn inner(&self) -> &InternalIndex {
        unsafe { &*(self._private as *const InternalIndex) }
    }

    /// Get a mutable reference to the inner InternalIndex
    pub(crate) fn inner_mut(&mut self) -> &mut InternalIndex {
        unsafe { &mut *(self._private as *mut InternalIndex) }
    }
}

impl Clone for t4a_index {
    fn clone(&self) -> Self {
        let inner = self.inner().clone();
        Self::new(inner)
    }
}

impl Drop for t4a_index {
    fn drop(&mut self) {
        unsafe {
            if !self._private.is_null() {
                let _ = Box::from_raw(self._private as *mut InternalIndex);
            }
        }
    }
}

// Safety: t4a_index is Send + Sync because InternalIndex is Send + Sync
unsafe impl Send for t4a_index {}
unsafe impl Sync for t4a_index {}
