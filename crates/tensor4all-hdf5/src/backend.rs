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
