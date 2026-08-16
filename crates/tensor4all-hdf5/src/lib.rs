//! HDF5 serialization for tensor4all-rs (ITensors.jl compatible format).
//!
//! This crate provides read/write functionality for tensor4all-rs data structures
//! using the HDF5 format compatible with ITensors.jl / ITensorMPS.jl. Files
//! written by this crate can be read by ITensors.jl and vice versa.
//!
//! # Thread safety
//!
//! The HDF5 C library is not thread-safe. Every public
//! [`save_*`](save_itensor) / [`append_*`](append_itensor) / [`load_*`](load_itensor)
//! call is serialized through one process-wide lock (the hdf5 binding's
//! reentrant mutex), so the crate is **safe to call concurrently by
//! construction** — even on distinct files. The lock covers the whole
//! operation (open, read/write, close), not individual HDF5 calls.
//!
//! The crate also disables HDF5's OS file locking by setting
//! `HDF5_USE_FILE_LOCKING=FALSE` once, before the first HDF5 call, unless the
//! caller already set that variable (their value wins). This closes the
//! same-path lock-release window (a writer's OS lock can outlive `H5Fclose`,
//! so a serialized reopen of the same path could otherwise fail with
//! `errno = 35`). The variable is process-global and affects any other HDF5
//! usage in the process; callers who need cross-process write protection
//! should set it themselves.
//!
//! New public functions that touch HDF5 MUST wrap their body in
//! [`backend::hdf5_sync`] — the public boundary is the crate's single
//! thread-safety choke point; internal helpers never lock on their own, they
//! run under the caller's lock. Direct use of the re-exported low-level
//! hdf5-rt passthroughs (`hdf5_init`, ...) bypasses the lock and is outside
//! this guarantee.
//!
//! # Supported types
//!
//! | Rust type | HDF5 schema | Julia equivalent |
//! |-----------|-------------|------------------|
//! | [`IdxTensor`] | `ITensor` | `ITensors.ITensor` |
//! | [`TensorTrain`] | `MPS` | `ITensorMPS.MPS` |
//! | [`TreeTN`](tensor4all_treetn::TreeTN) | `TreeTN` (tensor4all-rs schema) | — (no ITensorNetworks.jl equivalent) |
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
//! use tensor4all_core::{Index, IdxTensor};
//! use tensor4all_itensorlike::TensorTrain;
//!
//! # fn main() -> anyhow::Result<()> {
//! // Save and load a single tensor
//! let i = Index::new_dyn(2);
//! let j = Index::new_dyn(3);
//! let tensor = IdxTensor::from_dense(
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
//! let t0 = IdxTensor::from_dense(vec![s0, bond.clone()], vec![1.0, 0.0])?;
//! let t1 = IdxTensor::from_dense(vec![bond, s1], vec![1.0, 0.0])?;
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
mod treetn;
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
/// # Remedies
/// - File open/read failures: check the file path exists and is a valid HDF5
///   file, and that the requested dataset name is present.
/// - Shape/dtype mismatches: confirm the stored tensor metadata matches the
///   requested element type and dimensions.
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
use std::fmt::Debug;
use std::hash::Hash;
use std::str::FromStr;
use tensor4all_core::IdxTensor;
use tensor4all_itensorlike::TensorTrain;
use tensor4all_treetn::TreeTN;

// Re-export the HDF5 initialization functions (runtime-loading mode only)
#[cfg(feature = "runtime-loading")]
pub use hdf5_rt::sys::{
    init as hdf5_init, is_initialized as hdf5_is_initialized, library_path as hdf5_library_path,
};

/// Save a [`IdxTensor`] as an ITensors.jl-compatible `ITensor` in an HDF5 file.
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
/// use tensor4all_core::{Index, IdxTensor};
///
/// # fn main() -> anyhow::Result<()> {
/// let i = Index::new_dyn(2);
/// let j = Index::new_dyn(3);
/// let tensor = IdxTensor::from_dense(
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
/// use tensor4all_core::{Index, IdxTensor};
/// use num_complex::Complex64;
///
/// # fn main() -> anyhow::Result<()> {
/// let i = Index::new_dyn(2);
/// let data = vec![Complex64::new(1.0, 0.5), Complex64::new(2.0, -0.5)];
/// let tensor = IdxTensor::from_dense(vec![i], data.clone())?;
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
    tensor: &IdxTensor,
) -> std::result::Result<(), Hdf5Error> {
    backend::hdf5_sync(|| {
        let file = File::create(filepath)?;
        let group = file.create_group(name)?;
        itensor::write_itensor(&group, tensor).map_err(Hdf5Error::from)
    })
}

/// Append a [`IdxTensor`] as an ITensors.jl-compatible `ITensor` to an HDF5 file.
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
/// use tensor4all_core::{DynIndex, IdxTensor};
/// use tensor4all_hdf5::{append_itensor, load_itensor};
///
/// # fn main() -> anyhow::Result<()> {
/// let dir = tempfile::tempdir()?;
/// let path = dir.path().join("append_itensor.h5");
/// let path = path.to_str().unwrap();
/// let a = IdxTensor::from_dense(vec![DynIndex::new_dyn(2)], vec![1.0, 2.0])?;
/// let b = IdxTensor::from_dense(vec![DynIndex::new_dyn(2)], vec![3.0, 4.0])?;
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
    tensor: &IdxTensor,
) -> std::result::Result<(), Hdf5Error> {
    backend::hdf5_sync(|| {
        let file = File::append(filepath)?;
        let group = file.create_group(name)?;
        itensor::write_itensor(&group, tensor).map_err(Hdf5Error::from)
    })
}

/// Load a [`IdxTensor`] from an ITensors.jl-compatible `ITensor` in an HDF5 file.
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
/// use tensor4all_core::{Index, IdxTensor};
///
/// # fn main() -> anyhow::Result<()> {
/// let i = Index::new_dyn(2);
/// let j = Index::new_dyn(3);
/// let tensor = IdxTensor::from_dense(
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
pub fn load_itensor(filepath: &str, name: &str) -> std::result::Result<IdxTensor, Hdf5Error> {
    backend::hdf5_sync(|| {
        let file = File::open(filepath)?;
        let group = file.group(name)?;
        itensor::read_itensor(&group).map_err(Hdf5Error::from)
    })
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
/// use tensor4all_core::{Index, IdxTensor};
/// use tensor4all_itensorlike::TensorTrain;
///
/// # fn main() -> anyhow::Result<()> {
/// let s0 = Index::new_dyn(2);
/// let bond = Index::new_dyn(1);
/// let s1 = Index::new_dyn(2);
/// let t0 = IdxTensor::from_dense(vec![s0, bond.clone()], vec![1.0, 0.0])?;
/// let t1 = IdxTensor::from_dense(vec![bond, s1], vec![1.0, 0.0])?;
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
    backend::hdf5_sync(|| {
        let file = File::create(filepath)?;
        let group = file.create_group(name)?;
        mps::write_mps(&group, tt).map_err(Hdf5Error::from)
    })
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
/// use tensor4all_core::{DynIndex, IdxTensor};
/// use tensor4all_hdf5::{append_mps, load_mps};
/// use tensor4all_itensorlike::TensorTrain;
///
/// # fn main() -> anyhow::Result<()> {
/// let dir = tempfile::tempdir()?;
/// let path = dir.path().join("append_mps.h5");
/// let path = path.to_str().unwrap();
/// let s0 = DynIndex::new_dyn(2);
/// let s1 = DynIndex::new_dyn(2);
/// let a = TensorTrain::new(vec![IdxTensor::from_dense(vec![s0], vec![1.0, 2.0])?])?;
/// let b = TensorTrain::new(vec![IdxTensor::from_dense(vec![s1], vec![3.0, 4.0])?])?;
///
/// append_mps(path, "a", &a)?;
/// append_mps(path, "b", &b)?;
/// assert_eq!(load_mps(path, "a")?.len(), 1);
/// assert_eq!(load_mps(path, "b")?.site_indices()[0][0].size(), 2);
/// # Ok(())
/// # }
/// ```
pub fn append_mps(
    filepath: &str,
    name: &str,
    tt: &TensorTrain,
) -> std::result::Result<(), Hdf5Error> {
    backend::hdf5_sync(|| {
        let file = File::append(filepath)?;
        let group = file.create_group(name)?;
        mps::write_mps(&group, tt).map_err(Hdf5Error::from)
    })
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
/// use tensor4all_core::{Index, IdxTensor};
/// use tensor4all_itensorlike::TensorTrain;
///
/// # fn main() -> anyhow::Result<()> {
/// let s0 = Index::new_dyn(2);
/// let bond = Index::new_dyn(1);
/// let s1 = Index::new_dyn(2);
/// let t0 = IdxTensor::from_dense(vec![s0, bond.clone()], vec![1.0, 0.0])?;
/// let t1 = IdxTensor::from_dense(vec![bond, s1], vec![1.0, 0.0])?;
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
    backend::hdf5_sync(|| {
        let file = File::open(filepath)?;
        let group = file.group(name)?;
        mps::read_mps(&group).map_err(Hdf5Error::from)
    })
}

/// Save a [`TreeTN`] as a tensor4all-rs `TreeTN` in an HDF5 file.
///
/// Creates the file if it does not exist, or overwrites an existing file.
/// The tree is stored under a group named `name`, with each node tensor
/// written as a 1-indexed subgroup (`node_1/`, `node_2/`, ...) using the
/// ITensor schema, plus a `node_name` attribute per node. Topology is not
/// stored explicitly; on load it is recovered from shared index identity via
/// [`TreeTN::from_tensors`].
///
/// The node name type `V` is serialized via [`ToString`] and restored via
/// [`FromStr`]; any type supporting both round-trips exactly (e.g. `String`
/// or `usize`) can be stored.
///
/// # Errors
///
/// Returns an error if the file cannot be created, if any node tensor uses
/// an unsupported storage type (only `f64` and `Complex64` are supported),
/// or if the tree is internally inconsistent.
///
/// # Examples
///
/// Round-trip a small tree with `String` node names:
///
/// ```
/// use tensor4all_hdf5::{load_treetn, save_treetn};
/// use tensor4all_core::{DynIndex, IdxTensor};
/// use tensor4all_treetn::TreeTN;
///
/// # fn main() -> anyhow::Result<()> {
/// let s0 = DynIndex::new_dyn(2);
/// let s1 = DynIndex::new_dyn(2);
/// let b01 = DynIndex::new_dyn(4);
/// let t0 = IdxTensor::from_dense(vec![s0, b01.clone()], vec![1.0; 8])?;
/// let t1 = IdxTensor::from_dense(vec![b01, s1], vec![1.0; 8])?;
/// let tn = TreeTN::<IdxTensor, String>::from_tensors(
///     vec![t0, t1],
///     vec!["left".to_string(), "right".to_string()],
/// )?;
///
/// let dir = tempfile::tempdir()?;
/// let path = dir.path().join("treetn.h5");
/// let path = path.to_str().unwrap();
///
/// save_treetn(path, "tn", &tn)?;
/// let loaded = load_treetn::<String>(path, "tn")?;
///
/// assert_eq!(loaded.node_count(), 2);
/// assert_eq!(loaded.edge_count(), 1);
/// assert_eq!(loaded.node_names(), vec!["left".to_string(), "right".to_string()]);
/// # Ok(())
/// # }
/// ```
pub fn save_treetn<V>(
    filepath: &str,
    name: &str,
    tn: &TreeTN<IdxTensor, V>,
) -> std::result::Result<(), Hdf5Error>
where
    V: ToString + Clone + Hash + Eq + Send + Sync + Debug,
{
    backend::hdf5_sync(|| {
        let file = File::create(filepath)?;
        let group = file.create_group(name)?;
        treetn::write_treetn(&group, tn).map_err(Hdf5Error::from)
    })
}

/// Append a [`TreeTN`] to an HDF5 file.
///
/// Opens `filepath` read/write if it exists, or creates it otherwise, then
/// writes the tree under `name`. This keeps the same `TreeTN` v1 schema as
/// [`save_treetn`] while allowing multiple named trees in a single file.
/// The target group must not already exist.
///
/// # Errors
///
/// Returns an error when the operation fails (a shape or index mismatch, or
/// a backend failure).
pub fn append_treetn<V>(
    filepath: &str,
    name: &str,
    tn: &TreeTN<IdxTensor, V>,
) -> std::result::Result<(), Hdf5Error>
where
    V: ToString + Clone + Hash + Eq + Send + Sync + Debug,
{
    backend::hdf5_sync(|| {
        let file = File::append(filepath)?;
        let group = file.create_group(name)?;
        treetn::write_treetn(&group, tn).map_err(Hdf5Error::from)
    })
}

/// Load a [`TreeTN`] from a tensor4all-rs `TreeTN` in an HDF5 file.
///
/// Opens the file in read-only mode and reads the tree from the group named
/// `name`. Node tensors, node names, and the tree topology (recovered from
/// shared index identity) are all restored.
///
/// # Errors
///
/// Returns an error if:
/// - The file does not exist or cannot be opened
/// - The named group is missing or has an incompatible schema
/// - The type/version metadata does not match `TreeTN` v1
/// - A `node_name` attribute fails to parse as `V`
/// - The stored tensors do not form a consistent tree (e.g. a bond index
///   shared by more than two nodes)
pub fn load_treetn<V>(
    filepath: &str,
    name: &str,
) -> std::result::Result<TreeTN<IdxTensor, V>, Hdf5Error>
where
    V: FromStr + Ord + Clone + Hash + Eq + Send + Sync + Debug,
    V::Err: std::fmt::Display,
{
    backend::hdf5_sync(|| {
        let file = File::open(filepath)?;
        let group = file.group(name)?;
        treetn::read_treetn(&group).map_err(Hdf5Error::from)
    })
}
