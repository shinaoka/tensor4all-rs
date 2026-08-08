//! HDF5 backend abstraction layer.
//!
//! Selects between link-time (`hdf5-metno`) and runtime-loading (`hdf5-rt`) backends
//! based on Cargo feature flags. All other modules import HDF5 types through this
//! module, so the backend choice is transparent to the rest of the crate.
//!
//! When both features are active (due to Cargo feature unification),
//! `runtime-loading` takes priority.

// When both features are active (due to Cargo feature unification),
// runtime-loading takes priority.
#[cfg(all(feature = "link", not(feature = "runtime-loading")))]
pub use hdf5_metno::{types, Attribute, Dataset, File, Group, LinkType, Result};

#[cfg(feature = "runtime-loading")]
pub use hdf5_rt::{types, Attribute, Dataset, File, Group, LinkType, Result};
