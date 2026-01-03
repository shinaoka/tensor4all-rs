//! C API for tensor4all Rust implementation
//!
//! This crate provides a C-compatible interface to the tensor4all library,
//! enabling usage from languages like Julia, Python, and C++.
//!
//! ## Milestone 1: Index types
//!
//! The initial implementation provides wrappers for:
//! - `Index<DynId, NoSymmSpace, DefaultTagSet>` (the default index type)
//!
//! ## Design patterns
//!
//! Following the patterns from `sparse-ir-capi`:
//! - Opaque pointers with `_private: *const c_void`
//! - Explicit lifecycle functions: `*_release`, `*_clone`, `*_is_assigned`
//! - Status codes for error handling
//! - `catch_unwind` to prevent Rust panics from crossing FFI boundary

// C API requires unsafe operations with raw pointers
#![allow(clippy::not_unsafe_ptr_arg_deref)]

#[macro_use]
mod macros;

mod index;
mod types;

pub use index::*;
pub use types::*;

/// Status code type for C API
pub type StatusCode = libc::c_int;

// Status codes
pub const T4A_SUCCESS: StatusCode = 0;
pub const T4A_NULL_POINTER: StatusCode = -1;
pub const T4A_INVALID_ARGUMENT: StatusCode = -2;
pub const T4A_TAG_OVERFLOW: StatusCode = -3;
pub const T4A_TAG_TOO_LONG: StatusCode = -4;
pub const T4A_BUFFER_TOO_SMALL: StatusCode = -5;
pub const T4A_INTERNAL_ERROR: StatusCode = -6;
