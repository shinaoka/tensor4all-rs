//! HDF5 backend abstraction layer.
//!
//! Selects between link-time (`hdf5-metno`) and runtime-loading (`hdf5-rt`) backends
//! based on Cargo feature flags. All other modules import HDF5 types through this
//! module, so the backend choice is transparent to the rest of the crate.
//!
//! When both features are active (due to Cargo feature unification),
//! `runtime-loading` takes priority.

use anyhow::Context;
use std::ffi::CString;

// When both features are active (due to Cargo feature unification),
// runtime-loading takes priority.
#[cfg(all(feature = "link", not(feature = "runtime-loading")))]
pub use hdf5_metno::{types, Attribute, Dataset, File, Group, LinkType, Result};

#[cfg(feature = "runtime-loading")]
pub use hdf5_rt::{types, Attribute, Dataset, File, Group, LinkType, Result};

// H5Aexists checks one known name without enumerating or allocating all attributes.
#[cfg(all(feature = "link", not(feature = "runtime-loading")))]
pub(crate) fn attribute_exists(group: &Group, name: &str) -> anyhow::Result<bool> {
    let name = CString::new(name).context("HDF5 attribute name contains NUL")?;
    hdf5_metno::sync::sync(|| {
        let result = unsafe { hdf5_metno_sys::h5a::H5Aexists(group.id(), name.as_ptr()) };
        hdf5_metno::h5check(result)
            .map(|result| result > 0)
            .context("H5Aexists failed")
    })
}

#[cfg(feature = "runtime-loading")]
pub(crate) fn attribute_exists(group: &Group, name: &str) -> anyhow::Result<bool> {
    let name = CString::new(name).context("HDF5 attribute name contains NUL")?;
    hdf5_rt::sync::sync(|| {
        let result = unsafe { hdf5_rt::sys::h5a::H5Aexists(group.id(), name.as_ptr()) };
        hdf5_rt::h5check(result)
            .map(|result| result > 0)
            .context("H5Aexists failed")
    })
}

/// Disable HDF5's OS file locking once, before any HDF5 call the crate makes,
/// unless the caller already set `HDF5_USE_FILE_LOCKING` explicitly.
///
/// The HDF5 C library is not thread-safe and its per-file OS locks have a
/// release window that outlives `H5Fclose`, so even fully serialized calls can
/// hit `errno = 35` (EAGAIN) when the same path is reopened immediately. With
/// file locking disabled, a process-wide call lock (see [`hdf5_sync`]) is
/// sufficient. The environment variable is process-global: it also affects any
/// other HDF5 usage in the process. Callers who want locking keep control by
/// setting `HDF5_USE_FILE_LOCKING` themselves (their value wins).
static FILE_LOCKING_INIT: std::sync::Once = std::sync::Once::new();

fn disable_file_locking_once() {
    FILE_LOCKING_INIT.call_once(|| {
        if std::env::var_os("HDF5_USE_FILE_LOCKING").is_none() {
            // Safe on this crate's edition (2021). Runs before the first HDF5
            // init/open the crate triggers, so it takes effect regardless of
            // when the C library parses the variable.
            std::env::set_var("HDF5_USE_FILE_LOCKING", "FALSE");
        }
    });
}

/// Run one whole public HDF5 operation under the backend's process-wide
/// reentrant lock.
///
/// This is the crate's single thread-safety choke point: every public
/// `save_*`/`load_*`/`append_*` entry point wraps its body in [`hdf5_sync`]
/// and no internal helper locks on its own. The backend lock is reentrant, so
/// nested HDF5 calls inside the operation are safe, and it also serializes
/// against direct hdf5-metno/hdf5-rt usage elsewhere in the process that goes
/// through the same lock.
pub(crate) fn hdf5_sync<T>(f: impl FnOnce() -> T) -> T {
    disable_file_locking_once();
    #[cfg(all(feature = "link", not(feature = "runtime-loading")))]
    {
        hdf5_metno::sync::sync(f)
    }
    #[cfg(feature = "runtime-loading")]
    {
        hdf5_rt::sync::sync(f)
    }
}
