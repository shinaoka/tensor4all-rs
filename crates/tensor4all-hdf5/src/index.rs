//! Layer 0: Index / TagSet HDF5 read/write (ITensors.jl compatible).

use crate::backend::types::VarLenUnicode;
use crate::backend::Group;
use anyhow::{bail, Context, Result};
use std::str::FromStr;
use tensor4all_core::index::{DynId, DynIndex, Index, TagSet};
use tensor4all_core::tagset::TagSetLike;
use tensor4all_core::IndexLike;

use crate::schema;

pub(crate) fn read_nonnegative_usize(name: &'static str, value: i64) -> Result<usize> {
    usize::try_from(value)
        .with_context(|| format!("HDF5 dataset {name} must be non-negative, got {value}"))
}

/// Convert a [`TagSet`] to a comma-separated string (ITensors.jl format).
fn tagset_to_string(tags: &TagSet) -> String {
    let tag_strs: Vec<String> = TagSetLike::iter(tags).map(|s| s.to_string()).collect();
    tag_strs.join(",")
}

/// Write a [`TagSet`] to an HDF5 group (ITensors.jl compatible).
///
/// Tags are stored as a single comma-separated string, matching the
/// ITensors.jl convention.
///
/// # HDF5 Schema
///
/// ```text
/// <group>/
///   @type = "TagSet"
///   @version = 1
///   tags: String  (comma-separated, e.g. "Site,n=1")
/// ```
pub(crate) fn write_tagset(group: &Group, tags: &TagSet) -> Result<()> {
    schema::write_type_version(group, "TagSet", 1)?;

    let tag_string = tagset_to_string(tags);
    let ds = group
        .new_dataset::<VarLenUnicode>()
        .shape(())
        .create("tags")?;
    ds.as_writer()
        .write_scalar(&VarLenUnicode::from_str(&tag_string)?)?;

    Ok(())
}

/// Read a [`TagSet`] from an HDF5 group.
///
/// Handles both variable-length Unicode (our format) and fixed-length Unicode
/// (ITensors.jl format) via [`crate::compat::read_string_dataset`].
pub(crate) fn read_tagset(group: &Group) -> Result<TagSet> {
    schema::require_type_version(group, "TagSet", 1)?;

    let ds = group.dataset("tags")?;
    let s = crate::compat::read_string_dataset(&ds)?;
    if s.is_empty() {
        Ok(TagSet::new())
    } else {
        TagSet::from_str(&s)
            .map_err(|e| anyhow::anyhow!("Failed to parse TagSet from HDF5: {:?}", e))
    }
}

/// Write a [`DynIndex`] to an HDF5 group (ITensors.jl compatible).
///
/// All index metadata is preserved: unique id, dimension, prime level, and tags.
/// The `dir` field is always written as 0 (direction is unused in tensor4all-rs
/// but required by the ITensors.jl schema).
///
/// # HDF5 Schema
///
/// ```text
/// <group>/
///   @type = "Index"
///   @version = 1
///   @space_type = "Int"
///   id: UInt64
///   dim: Int64
///   dir: Int64       (always 0 -- direction is unused in tensor4all-rs)
///   plev: Int64
///   tags/            (TagSet group)
/// ```
pub(crate) fn write_index(group: &Group, index: &DynIndex) -> Result<()> {
    schema::write_type_version(group, "Index", 1)?;

    let space_type_attr = group
        .new_attr::<VarLenUnicode>()
        .shape(())
        .create("space_type")?;
    space_type_attr
        .as_writer()
        .write_scalar(&VarLenUnicode::from_str("Int")?)?;

    // Datasets
    let id_ds = group.new_dataset::<u64>().shape(()).create("id")?;
    id_ds.as_writer().write_scalar(&index.id().value())?;

    let dim_ds = group.new_dataset::<i64>().shape(()).create("dim")?;
    dim_ds.as_writer().write_scalar(&(index.dim() as i64))?;

    // dir: always 0 (direction is unused in tensor4all-rs)
    let dir_ds = group.new_dataset::<i64>().shape(()).create("dir")?;
    dir_ds.as_writer().write_scalar(&0i64)?;

    let plev_ds = group.new_dataset::<i64>().shape(()).create("plev")?;
    plev_ds.as_writer().write_scalar(&index.plev())?;

    // Tags subgroup
    let tags_group = group.create_group("tags")?;
    write_tagset(&tags_group, index.tags())?;

    Ok(())
}

/// Read a [`DynIndex`] from an HDF5 group.
///
/// Restores all metadata: id, dimension, prime level, and tags. The `dir`
/// field is read for schema compatibility but ignored (always unused in
/// tensor4all-rs).
pub(crate) fn read_index(group: &Group) -> Result<DynIndex> {
    schema::require_type_version(group, "Index", 1)?;

    let id: u64 = group
        .dataset("id")?
        .as_reader()
        .read_scalar()
        .context("Failed to read index id")?;

    let dim: i64 = group
        .dataset("dim")?
        .as_reader()
        .read_scalar()
        .context("Failed to read index dim")?;
    let dim = read_nonnegative_usize("dim", dim)?;

    // dir is read for schema compatibility but ignored
    let _dir: i64 = group
        .dataset("dir")?
        .as_reader()
        .read_scalar()
        .context("Failed to read index dir")?;

    let plev: i64 = group
        .dataset("plev")?
        .as_reader()
        .read_scalar()
        .context("Failed to read index plev")?;

    let tags_group = group.group("tags")?;
    let tags = read_tagset(&tags_group)?;

    let idx = Index::new_with_tags(DynId(id), dim, tags).set_plev(plev);
    Ok(idx)
}

/// Write an IndexSet (slice of [`DynIndex`]) to an HDF5 group (ITensors.jl compatible).
///
/// Indices are stored as 1-indexed subgroups (`index_1`, `index_2`, ...),
/// following the Julia convention.
///
/// # HDF5 Schema
///
/// ```text
/// <group>/
///   @type = "IndexSet"
///   @version = 1
///   length: Int64
///   index_1/ ...   (Index group)
///   index_2/ ...
/// ```
pub(crate) fn write_index_set(group: &Group, indices: &[DynIndex]) -> Result<()> {
    schema::write_type_version(group, "IndexSet", 1)?;

    let length_ds = group.new_dataset::<i64>().shape(()).create("length")?;
    length_ds
        .as_writer()
        .write_scalar(&(indices.len() as i64))?;

    for (i, index) in indices.iter().enumerate() {
        let name = format!("index_{}", i + 1); // 1-indexed
        let index_group = group.create_group(&name)?;
        write_index(&index_group, index)?;
    }

    Ok(())
}

/// Read an IndexSet from an HDF5 group.
///
/// Returns a `Vec<DynIndex>` with indices read from 1-indexed subgroups.
/// The number of indices is determined by the `length` dataset.
pub(crate) fn read_index_set(group: &Group) -> Result<Vec<DynIndex>> {
    schema::require_type_version(group, "IndexSet", 1)?;

    let length: i64 = group
        .dataset("length")?
        .as_reader()
        .read_scalar()
        .context("Failed to read IndexSet length")?;
    let length = read_nonnegative_usize("length", length)?;

    let member_names = group
        .member_names()
        .context("Failed to enumerate IndexSet child groups")?;
    let child_group_count = member_names
        .iter()
        .filter(|name| name.starts_with("index_"))
        .count();
    if length > child_group_count {
        bail!(
            "HDF5 dataset length declares {length} indices, but found only {child_group_count} index_N child groups"
        );
    }
    for i in 0..length {
        let name = format!("index_{}", i + 1);
        if !group.link_exists(&name) {
            bail!(
                "HDF5 dataset length declares {length} indices, but child group {name} is missing"
            );
        }
    }

    let mut indices = Vec::with_capacity(length);
    for i in 0..length {
        let name = format!("index_{}", i + 1);
        let index_group = group.group(&name)?;
        indices.push(read_index(&index_group)?);
    }

    Ok(indices)
}

#[cfg(test)]
mod tests;
