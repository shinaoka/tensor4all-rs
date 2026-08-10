//! HDF5 serialization for tensor4all-rs (ITensors.jl compatible format).
//!
//! This crate provides read/write functionality for tensor4all-rs data structures
//! using the HDF5 format compatible with ITensors.jl / ITensorMPS.jl. Files
//! written by this crate can be read by ITensors.jl and vice versa.
//!
//! # Supported types
//!
//! | Rust type | HDF5 schema | Julia equivalent |
//! |-----------|-------------|------------------|
//! | [`TensorDynLen`] | `ITensor` | `ITensors.ITensor` |
//! | [`TensorTrain`] | `MPS` | `ITensorMPS.MPS` |
//!
//! Both `f64` and `Complex64` element types are supported for dense storage.
//!
//! # Data layout
//!
//! tensor4all-rs and ITensors.jl both use column-major dense linearization.
//! This crate therefore preserves dense flat buffers as-is when serializing and
//! deserializing ITensors.jl-compatible payloads.
//!
//! # Backend selection
//!
//! - `link` feature (default): uses `hdf5-metno` with compile-time linking
//! - `runtime-loading` feature: uses `hdf5-rt` with dlopen (for Julia/Python FFI)
//!
//! # Quick start
//!
//! ```
//! use tensor4all_hdf5::{save_itensor, load_itensor, save_mps, load_mps};
//! use tensor4all_core::{Index, TensorDynLen};
//! use tensor4all_itensorlike::TensorTrain;
//!
//! # fn main() -> anyhow::Result<()> {
//! // Save and load a single tensor
//! let i = Index::new_dyn(2);
//! let j = Index::new_dyn(3);
//! let tensor = TensorDynLen::from_dense(
//!     vec![i, j],
//!     vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
//! )?;
//!
//! let dir = tempfile::tempdir()?;
//! let path = dir.path().join("example.h5");
//! let path = path.to_str().unwrap();
//!
//! save_itensor(path, "my_tensor", &tensor)?;
//! let loaded = load_itensor(path, "my_tensor")?;
//! assert_eq!(loaded.dims(), vec![2, 3]);
//! assert_eq!(loaded.to_vec::<f64>()?, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
//!
//! // Save and load an MPS (TensorTrain)
//! let s0 = Index::new_dyn(2);
//! let bond = Index::new_dyn(1);
//! let s1 = Index::new_dyn(2);
//! let t0 = TensorDynLen::from_dense(vec![s0, bond.clone()], vec![1.0, 0.0])?;
//! let t1 = TensorDynLen::from_dense(vec![bond, s1], vec![1.0, 0.0])?;
//! let tt = TensorTrain::new(vec![t0, t1])?;
//!
//! let mps_path = dir.path().join("mps.h5");
//! let mps_path = mps_path.to_str().unwrap();
//! save_mps(mps_path, "my_mps", &tt)?;
//! let loaded_mps = load_mps(mps_path, "my_mps")?;
//! assert_eq!(loaded_mps.len(), 2);
//! # Ok(())
//! # }
//! ```

pub(crate) mod backend;
mod compat;
mod index;
mod itensor;
mod mps;
mod schema;
/// Error returned by tensor4all-hdf5 save/load operations.
///
/// The full original diagnostic is preserved in [`Hdf5Error::source`].
///
/// # Examples
///
/// ```
/// use tensor4all_hdf5::Hdf5Error;
///
/// let err = Hdf5Error::from(anyhow::anyhow!("file open failed"));
/// assert!(err.to_string().contains("file open failed"));
/// ```
#[derive(Debug, thiserror::Error)]
#[error("HDF5 tensor operation failed: {source}")]
pub struct Hdf5Error {
    /// Original HDF5 or tensor diagnostic, preserving the full source chain.
    #[source]
    pub source: anyhow::Error,
}

#[cfg(all(feature = "link", not(feature = "runtime-loading")))]
impl From<hdf5_metno::Error> for Hdf5Error {
    fn from(source: hdf5_metno::Error) -> Self {
        Self {
            source: anyhow::Error::new(source),
        }
    }
}

#[cfg(feature = "runtime-loading")]
impl From<hdf5_rt::Error> for Hdf5Error {
    fn from(source: hdf5_rt::Error) -> Self {
        Self {
            source: anyhow::Error::new(source),
        }
    }
}

impl From<anyhow::Error> for Hdf5Error {
    fn from(source: anyhow::Error) -> Self {
        Self { source }
    }
}

use backend::File;
use tensor4all_core::TensorDynLen;
use tensor4all_itensorlike::TensorTrain;

// Re-export the HDF5 initialization functions (runtime-loading mode only)
#[cfg(feature = "runtime-loading")]
pub use hdf5_rt::sys::{
    init as hdf5_init, is_initialized as hdf5_is_initialized, library_path as hdf5_library_path,
};

/// Save a [`TensorDynLen`] as an ITensors.jl-compatible `ITensor` in an HDF5 file.
///
/// Creates the file if it does not exist, or overwrites an existing file.
/// The tensor is stored under a group named `name` within the file.
///
/// Both `f64` and `Complex64` storage types are supported. Index metadata
/// (id, dimension, prime level, tags) is preserved in the HDF5 schema.
///
/// # Errors
///
/// Returns an error if the file cannot be created, or if the tensor uses an
/// unsupported storage type (only `f64` and `Complex64` are supported).
///
/// # Examples
///
/// Save and reload an f64 tensor:
///
/// ```
/// use tensor4all_hdf5::{save_itensor, load_itensor};
/// use tensor4all_core::{Index, TensorDynLen};
///
/// # fn main() -> anyhow::Result<()> {
/// let i = Index::new_dyn(2);
/// let j = Index::new_dyn(3);
/// let tensor = TensorDynLen::from_dense(
///     vec![i.clone(), j.clone()],
///     vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
/// )?;
///
/// let dir = tempfile::tempdir()?;
/// let path = dir.path().join("save_itensor.h5");
/// let path = path.to_str().unwrap();
///
/// save_itensor(path, "my_tensor", &tensor)?;
/// let loaded = load_itensor(path, "my_tensor")?;
/// assert_eq!(loaded.dims(), vec![2, 3]);
/// assert_eq!(loaded.to_vec::<f64>()?, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
/// # Ok(())
/// # }
/// ```
///
/// Save a complex tensor:
///
/// ```
/// use tensor4all_hdf5::{save_itensor, load_itensor};
/// use tensor4all_core::{Index, TensorDynLen};
/// use num_complex::Complex64;
///
/// # fn main() -> anyhow::Result<()> {
/// let i = Index::new_dyn(2);
/// let data = vec![Complex64::new(1.0, 0.5), Complex64::new(2.0, -0.5)];
/// let tensor = TensorDynLen::from_dense(vec![i], data.clone())?;
///
/// let dir = tempfile::tempdir()?;
/// let path = dir.path().join("save_itensor_c64.h5");
/// let path = path.to_str().unwrap();
///
/// save_itensor(path, "z_tensor", &tensor)?;
/// let loaded = load_itensor(path, "z_tensor")?;
/// assert_eq!(loaded.to_vec::<Complex64>()?, data);
/// # Ok(())
/// # }
/// ```
pub fn save_itensor(
    filepath: &str,
    name: &str,
    tensor: &TensorDynLen,
) -> std::result::Result<(), Hdf5Error> {
    let file = File::create(filepath)?;
    let group = file.create_group(name)?;
    itensor::write_itensor(&group, tensor).map_err(Hdf5Error::from)
}

/// Append a [`TensorDynLen`] as an ITensors.jl-compatible `ITensor` to an HDF5 file.
///
/// Opens `filepath` read/write if it exists, or creates it otherwise, then
/// writes the tensor under `name`. This is useful for files containing multiple
/// tensor objects. The target group must not already exist.
///
/// # Errors
///
/// Returns an error when the operation fails (a shape or index mismatch, or
/// /// a backend failure).
///
/// # Examples
///
/// ```
/// use tensor4all_core::{DynIndex, TensorDynLen};
/// use tensor4all_hdf5::{append_itensor, load_itensor};
///
/// # fn main() -> anyhow::Result<()> {
/// let dir = tempfile::tempdir()?;
/// let path = dir.path().join("append_itensor.h5");
/// let path = path.to_str().unwrap();
/// let a = TensorDynLen::from_dense(vec![DynIndex::new_dyn(2)], vec![1.0, 2.0])?;
/// let b = TensorDynLen::from_dense(vec![DynIndex::new_dyn(2)], vec![3.0, 4.0])?;
///
/// append_itensor(path, "a", &a)?;
/// append_itensor(path, "b", &b)?;
/// assert_eq!(load_itensor(path, "a")?.to_vec::<f64>()?, vec![1.0, 2.0]);
/// assert_eq!(load_itensor(path, "b")?.to_vec::<f64>()?, vec![3.0, 4.0]);
/// # Ok(())
/// # }
/// ```
pub fn append_itensor(
    filepath: &str,
    name: &str,
    tensor: &TensorDynLen,
) -> std::result::Result<(), Hdf5Error> {
    let file = File::append(filepath)?;
    let group = file.create_group(name)?;
    itensor::write_itensor(&group, tensor).map_err(Hdf5Error::from)
}

/// Load a [`TensorDynLen`] from an ITensors.jl-compatible `ITensor` in an HDF5 file.
///
/// Opens the file in read-only mode and reads the tensor from the group named
/// `name`. Index metadata (id, dimension, prime level, tags) is restored from
/// the HDF5 schema.
///
/// This function can read files written by both this crate and ITensors.jl,
/// since the HDF5 schema is compatible.
///
/// # Errors
///
/// Returns an error if:
/// - The file does not exist or cannot be opened
/// - The named group is missing or has an incompatible schema
/// - The storage type is not `Dense{Float64}` or `Dense{ComplexF64}`
///
/// # Examples
///
/// Round-trip save and load, verifying data and index preservation:
///
/// ```
/// use tensor4all_hdf5::{save_itensor, load_itensor};
/// use tensor4all_core::{Index, TensorDynLen};
///
/// # fn main() -> anyhow::Result<()> {
/// let i = Index::new_dyn(2);
/// let j = Index::new_dyn(3);
/// let tensor = TensorDynLen::from_dense(
///     vec![i.clone(), j.clone()],
///     vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
/// )?;
///
/// let dir = tempfile::tempdir()?;
/// let path = dir.path().join("load_itensor.h5");
/// let path = path.to_str().unwrap();
///
/// save_itensor(path, "tensor", &tensor)?;
/// let loaded = load_itensor(path, "tensor")?;
///
/// // Data is preserved exactly
/// assert_eq!(loaded.to_vec::<f64>()?, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
///
/// // Index dimensions are preserved
/// assert_eq!(loaded.dims(), vec![2, 3]);
///
/// // Index identity and metadata are preserved
/// assert_eq!(loaded.indices(), tensor.indices());
/// # Ok(())
/// # }
/// ```
pub fn load_itensor(filepath: &str, name: &str) -> std::result::Result<TensorDynLen, Hdf5Error> {
    let file = File::open(filepath)?;
    let group = file.group(name)?;
    itensor::read_itensor(&group).map_err(Hdf5Error::from)
}

/// Save a [`TensorTrain`] as an ITensorMPS.jl-compatible `MPS` in an HDF5 file.
///
/// Creates the file if it does not exist, or overwrites an existing file.
/// The MPS is stored under a group named `name`, with each site tensor
/// written as a 1-indexed subgroup (`MPS[1]`, `MPS[2]`, ...).
///
/// Metadata preserved:
/// - `length`: number of sites
/// - `llim`, `rlim`: orthogonality center bounds
/// - `canonical_form` (if set): tensor4all-rs extension attribute
/// - Per-site tensor data and index metadata
///
/// # Errors
///
/// Returns an error if the file cannot be created, or if any site tensor
/// uses an unsupported storage type.
///
/// # Examples
///
/// Save a 2-site MPS and reload it:
///
/// ```
/// use tensor4all_hdf5::{save_mps, load_mps};
/// use tensor4all_core::{Index, TensorDynLen};
/// use tensor4all_itensorlike::TensorTrain;
///
/// # fn main() -> anyhow::Result<()> {
/// let s0 = Index::new_dyn(2);
/// let bond = Index::new_dyn(1);
/// let s1 = Index::new_dyn(2);
/// let t0 = TensorDynLen::from_dense(vec![s0, bond.clone()], vec![1.0, 0.0])?;
/// let t1 = TensorDynLen::from_dense(vec![bond, s1], vec![1.0, 0.0])?;
/// let tt = TensorTrain::new(vec![t0, t1])?;
///
/// let dir = tempfile::tempdir()?;
/// let path = dir.path().join("save_mps.h5");
/// let path = path.to_str().unwrap();
///
/// save_mps(path, "my_mps", &tt)?;
/// let loaded = load_mps(path, "my_mps")?;
/// assert_eq!(loaded.len(), 2);
///
/// // Site tensor data is preserved
/// let orig_data = tt.tensors()[0].to_vec::<f64>()?;
/// let loaded_data = loaded.tensors()[0].to_vec::<f64>()?;
/// assert_eq!(orig_data, loaded_data);
/// # Ok(())
/// # }
/// ```
pub fn save_mps(
    filepath: &str,
    name: &str,
    tt: &TensorTrain,
) -> std::result::Result<(), Hdf5Error> {
    let file = File::create(filepath)?;
    let group = file.create_group(name)?;
    mps::write_mps(&group, tt).map_err(Hdf5Error::from)
}

/// Append a [`TensorTrain`] as an ITensorMPS.jl-compatible `MPS` to an HDF5 file.
///
/// Opens `filepath` read/write if it exists, or creates it otherwise, then
/// writes the MPS under `name`. This keeps the same `MPS` v1 schema as
/// [`save_mps`] while allowing multiple named MPS objects in a single file.
/// The target group must not already exist.
///
/// # Errors
///
/// Returns an error when the operation fails (a shape or index mismatch, or
/// /// a backend failure).
///
/// # Examples
///
/// ```
/// use tensor4all_core::{DynIndex, TensorDynLen};
/// use tensor4all_hdf5::{append_mps, load_mps};
/// use tensor4all_itensorlike::TensorTrain;
///
/// # fn main() -> anyhow::Result<()> {
/// let dir = tempfile::tempdir()?;
/// let path = dir.path().join("append_mps.h5");
/// let path = path.to_str().unwrap();
/// let s0 = DynIndex::new_dyn(2);
/// let s1 = DynIndex::new_dyn(2);
/// let a = TensorTrain::new(vec![TensorDynLen::from_dense(vec![s0], vec![1.0, 2.0])?])?;
/// let b = TensorTrain::new(vec![TensorDynLen::from_dense(vec![s1], vec![3.0, 4.0])?])?;
///
/// append_mps(path, "a", &a)?;
/// append_mps(path, "b", &b)?;
/// assert_eq!(load_mps(path, "a")?.len(), 1);
/// assert_eq!(load_mps(path, "b")?.siteinds()[0][0].size(), 2);
/// # Ok(())
/// # }
/// ```
pub fn append_mps(
    filepath: &str,
    name: &str,
    tt: &TensorTrain,
) -> std::result::Result<(), Hdf5Error> {
    let file = File::append(filepath)?;
    let group = file.create_group(name)?;
    mps::write_mps(&group, tt).map_err(Hdf5Error::from)
}

/// Load a [`TensorTrain`] from an ITensorMPS.jl-compatible `MPS` in an HDF5 file.
///
/// Opens the file in read-only mode and reads the MPS from the group named
/// `name`. Site tensors, bond structure, orthogonality limits, and canonical
/// form (if present) are all restored.
///
/// This function can read files written by both this crate and ITensorMPS.jl.
///
/// # Errors
///
/// Returns an error if:
/// - The file does not exist or cannot be opened
/// - The named group is missing or has an incompatible schema
/// - The type/version metadata does not match `MPS` v1
///
/// # Examples
///
/// Round-trip an MPS, verifying structure and orthogonality limits:
///
/// ```
/// use tensor4all_hdf5::{save_mps, load_mps};
/// use tensor4all_core::{Index, TensorDynLen};
/// use tensor4all_itensorlike::TensorTrain;
///
/// # fn main() -> anyhow::Result<()> {
/// let s0 = Index::new_dyn(2);
/// let bond = Index::new_dyn(1);
/// let s1 = Index::new_dyn(2);
/// let t0 = TensorDynLen::from_dense(vec![s0, bond.clone()], vec![1.0, 0.0])?;
/// let t1 = TensorDynLen::from_dense(vec![bond, s1], vec![1.0, 0.0])?;
/// let tt = TensorTrain::new(vec![t0, t1])?;
///
/// let dir = tempfile::tempdir()?;
/// let path = dir.path().join("load_mps.h5");
/// let path = path.to_str().unwrap();
///
/// save_mps(path, "my_mps", &tt)?;
/// let loaded = load_mps(path, "my_mps")?;
///
/// assert_eq!(loaded.len(), 2);
/// assert_eq!(loaded.llim(), tt.llim());
/// assert_eq!(loaded.rlim(), tt.rlim());
///
/// // Each site tensor's dimensions are preserved
/// for (orig, loaded_t) in tt.tensors().iter().zip(loaded.tensors().iter()) {
///     assert_eq!(orig.dims(), loaded_t.dims());
/// }
/// # Ok(())
/// # }
/// ```
pub fn load_mps(filepath: &str, name: &str) -> std::result::Result<TensorTrain, Hdf5Error> {
    let file = File::open(filepath)?;
    let group = file.group(name)?;
    mps::read_mps(&group).map_err(Hdf5Error::from)
}
