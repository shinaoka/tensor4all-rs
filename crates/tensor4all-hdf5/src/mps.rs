//! Layer 2: MPS (TensorTrain) HDF5 read/write (ITensorMPS.jl compatible).
//!
//! MPS is simply metadata (length, llim, rlim) + a sequence of ITensors,
//! so this module is a thin wrapper around [`crate::itensor`].

use crate::backend::Group;
use anyhow::{Context, Result};
use tensor4all_itensorlike::{CanonicalForm, TensorTrain};

use crate::index;
use crate::itensor;
use crate::schema;

fn read_i32(name: &'static str, value: i64, length: usize) -> Result<i32> {
    i32::try_from(value).with_context(|| {
        format!("HDF5 dataset {name} does not fit in i32, got {value} for MPS length {length}")
    })
}

fn validate_ortho_limits(llim: i32, rlim: i32, length: usize) -> Result<()> {
    let length_i32 = i32::try_from(length).map_err(|_| {
        anyhow::anyhow!(
            "invalid HDF5 MPS orthogonality limits: llim={llim}, rlim={rlim}, length={length}; length does not fit in i32"
        )
    })?;
    let default_rlim = length_i32.checked_add(1).ok_or_else(|| {
        anyhow::anyhow!(
            "invalid HDF5 MPS orthogonality limits: llim={llim}, rlim={rlim}, length={length}; length + 1 overflows i32"
        )
    })?;
    let center_in_bounds = llim
        .checked_add(1)
        .map(|center| center < length_i32)
        .unwrap_or(false);
    let single_center = llim >= -1 && center_in_bounds && llim.checked_add(2) == Some(rlim);
    let no_center = llim == -1 && rlim == default_rlim;
    if !single_center && !no_center {
        anyhow::bail!(
            "invalid HDF5 MPS orthogonality limits: llim={llim}, rlim={rlim}, length={length}; expected llim=-1 and rlim={default_rlim}, or llim + 2 = rlim with a center in 0..{length}"
        );
    }
    Ok(())
}

const CANONICAL_FORM_ATTR: &str = "canonical_form";

fn write_canonical_form(group: &Group, tt: &TensorTrain) -> Result<()> {
    if let Some(form) = tt.canonical_form() {
        let canonical_form_attr = group
            .new_attr::<i32>()
            .shape(())
            .create(CANONICAL_FORM_ATTR)?;
        canonical_form_attr
            .as_writer()
            .write_scalar(&form.to_i32())?;
    }

    Ok(())
}

fn read_canonical_form(group: &Group) -> Result<Option<CanonicalForm>> {
    if !group
        .attr_names()?
        .iter()
        .any(|name| name == CANONICAL_FORM_ATTR)
    {
        return Ok(None);
    }

    let value: i32 = group
        .attr(CANONICAL_FORM_ATTR)?
        .as_reader()
        .read_scalar()
        .context("Failed to read MPS canonical_form")?;

    CanonicalForm::from_i32(value)
        .ok_or_else(|| anyhow::anyhow!("Invalid MPS canonical_form value: {}", value))
        .map(Some)
}

/// Write a [`TensorTrain`] as an ITensorMPS.jl `MPS` to an HDF5 group.
///
/// Site tensors are stored as 1-indexed subgroups (`MPS[1]`, `MPS[2]`, ...),
/// following the Julia convention. Each site tensor is written using
/// [`crate::itensor::write_itensor`].
///
/// The `canonical_form` attribute is a tensor4all-rs extension not present in
/// the ITensorMPS.jl schema. It is silently ignored when reading files that
/// lack it.
///
/// # HDF5 Schema
///
/// ```text
/// <group>/
///   @type = "MPS"
///   @version = 1
///   length: Int64
///   llim: Int64
///   rlim: Int64
///   @canonical_form: Int32?  (tensor4all-rs extension; absent for non-canonical MPS)
///   MPS[1]/ ...   (ITensor, 1-indexed)
///   MPS[2]/ ...
/// ```
pub(crate) fn write_mps(group: &Group, tt: &TensorTrain) -> Result<()> {
    schema::write_type_version(group, "MPS", 1)?;

    // Metadata datasets
    let length = tt.len() as i64;
    let length_ds = group.new_dataset::<i64>().shape(()).create("length")?;
    length_ds.as_writer().write_scalar(&length)?;

    let llim_ds = group.new_dataset::<i64>().shape(()).create("llim")?;
    llim_ds.as_writer().write_scalar(&(tt.llim() as i64))?;

    let rlim_ds = group.new_dataset::<i64>().shape(()).create("rlim")?;
    rlim_ds.as_writer().write_scalar(&(tt.rlim() as i64))?;

    write_canonical_form(group, tt)?;

    // Write each tensor (1-indexed, Julia convention)
    let tensors = tt.tensors();
    for (i, tensor) in tensors.iter().enumerate() {
        let name = format!("MPS[{}]", i + 1);
        let tensor_group = group.create_group(&name)?;
        itensor::write_itensor(&tensor_group, tensor)?;
    }

    Ok(())
}

/// Read a [`TensorTrain`] from an ITensorMPS.jl `MPS` in an HDF5 group.
///
/// Validates the `@type` and `@version` attributes, then reads site tensors
/// from 1-indexed subgroups. The `canonical_form` attribute is read if present
/// (tensor4all-rs extension), otherwise defaults to `None`.
pub(crate) fn read_mps(group: &Group) -> Result<TensorTrain> {
    schema::require_type_version(group, "MPS", 1)?;

    let length: i64 = group
        .dataset("length")?
        .as_reader()
        .read_scalar()
        .context("Failed to read MPS length")?;
    let length = index::read_nonnegative_usize("length", length)?;

    let llim: i64 = group
        .dataset("llim")?
        .as_reader()
        .read_scalar()
        .context("Failed to read MPS llim")?;
    let llim = read_i32("llim", llim, length)?;

    let rlim: i64 = group
        .dataset("rlim")?
        .as_reader()
        .read_scalar()
        .context("Failed to read MPS rlim")?;
    let rlim = read_i32("rlim", rlim, length)?;
    validate_ortho_limits(llim, rlim, length)?;
    let canonical_form = read_canonical_form(group)?;

    index::validate_expected_child_groups(
        group,
        length,
        "MPS",
        "MPS[",
        Some(']'),
        &["length", "llim", "rlim"],
    )?;
    let mut tensors = Vec::with_capacity(length);
    for ordinal in 1..=length {
        let name = index::expected_child_name("MPS[", ordinal, Some(']'));
        let tensor_group = group
            .group(&name)
            .with_context(|| format!("Failed to open HDF5 MPS child group {name}"))?;
        tensors.push(itensor::read_itensor(&tensor_group)?);
    }

    Ok(TensorTrain::with_ortho(
        tensors,
        llim,
        rlim,
        canonical_form,
    )?)
}
