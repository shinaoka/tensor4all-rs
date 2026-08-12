//! Layer 1: ITensor (IdxTensor) HDF5 read/write (ITensors.jl compatible).

use crate::backend::Group;
use anyhow::{bail, Context, Result};
use num_complex::Complex64;
use tensor4all_core::IdxTensor;

use crate::index;
use crate::schema;

/// Write a [`IdxTensor`] as an ITensors.jl `ITensor` to an HDF5 group.
///
/// The tensor data is stored in column-major order (matching ITensors.jl convention).
/// Both `f64` and `Complex64` storage types are supported.
///
/// # HDF5 Schema
///
/// ```text
/// <group>/
///   @type = "ITensor"
///   @version = 1
///   inds/          (IndexSet — see write_index_set)
///   storage/       (Dense{Float64} or Dense{ComplexF64})
///     @type = "Dense{Float64}" | "Dense{ComplexF64}"
///     @version = 1
///     data: [N]    (flat column-major array)
/// ```
pub(crate) fn write_itensor(group: &Group, tensor: &IdxTensor) -> Result<()> {
    schema::write_type_version(group, "ITensor", 1)?;

    // Write indices
    let inds_group = group.create_group("inds")?;
    index::write_index_set(&inds_group, tensor.indices())?;

    // Write storage
    let storage_group = group.create_group("storage")?;
    if tensor.is_f64() {
        let data = tensor
            .to_vec::<f64>()
            .context("Failed to extract f64 data")?;

        schema::write_type_version(&storage_group, "Dense{Float64}", 1)?;

        let data_ds = storage_group
            .new_dataset::<f64>()
            .shape([data.len()])
            .create("data")?;
        data_ds.as_writer().write(&data)?;
    } else if tensor.is_complex() {
        let data = tensor
            .to_vec::<Complex64>()
            .context("Failed to extract c64 data")?;

        schema::write_type_version(&storage_group, "Dense{ComplexF64}", 1)?;

        // Store as native HDF5 compound type (compatible with ITensors.jl)
        let data_ds = storage_group
            .new_dataset::<Complex64>()
            .shape([data.len()])
            .create("data")?;
        data_ds.as_writer().write(&data)?;
    } else {
        bail!("Unsupported storage type for HDF5 serialization");
    }

    Ok(())
}

/// Read a [`IdxTensor`] from an ITensors.jl `ITensor` in an HDF5 group.
///
/// Validates the `@type` and `@version` attributes before reading. Supports
/// both `Dense{Float64}` and `Dense{ComplexF64}` storage types.
///
/// String attributes are read using [`crate::compat`] helpers, which handle
/// both variable-length Unicode (our format) and fixed-length Unicode
/// (ITensors.jl format).
pub(crate) fn read_itensor(group: &Group) -> Result<IdxTensor> {
    schema::require_type_version(group, "ITensor", 1)?;

    // Read indices
    let inds_group = group.group("inds")?;
    let indices = index::read_index_set(&inds_group)?;
    // Read storage
    let storage_group = group.group("storage")?;
    let storage_type_str = crate::compat::read_string_attr_by_name(&storage_group, "type")?;

    if storage_type_str.contains("Dense{Float64}") {
        let data_ds = storage_group.dataset("data")?;
        let col_major_data: Vec<f64> = data_ds
            .as_reader()
            .read_1d()
            .context("Failed to read f64 data")?
            .to_vec();
        IdxTensor::from_dense(indices, col_major_data)
            .context("Failed to build f64 IdxTensor from HDF5 data")
    } else if storage_type_str.contains("Dense{ComplexF64}") {
        let data_ds = storage_group.dataset("data")?;
        // Read as native HDF5 compound type (Complex64)
        let col_major_data: Vec<Complex64> = data_ds
            .as_reader()
            .read_1d()
            .context("Failed to read complex data")?
            .to_vec();
        IdxTensor::from_dense(indices, col_major_data)
            .context("Failed to build complex IdxTensor from HDF5 data")
    } else {
        bail!(
            "Unsupported storage type: {}. Only Dense{{Float64}} and Dense{{ComplexF64}} are supported.",
            storage_type_str
        );
    }
}
