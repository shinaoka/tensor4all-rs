//! Layer 0: Index / TagSet HDF5 read/write (ITensors.jl compatible).

use crate::backend::types::VarLenUnicode;
use crate::backend::{Group, LinkType};
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

pub(crate) fn expected_child_name(prefix: &str, ordinal: usize, closing: Option<char>) -> String {
    match closing {
        Some(closing) => format!("{prefix}{ordinal}{closing}"),
        None => format!("{prefix}{ordinal}"),
    }
}

fn canonical_child_ordinal(name: &str, prefix: &str, closing: Option<char>) -> Option<usize> {
    let suffix = name.strip_prefix(prefix)?;
    let suffix = match closing {
        Some(closing) => suffix.strip_suffix(closing)?,
        None => suffix,
    };
    if suffix.is_empty()
        || (suffix.len() > 1 && suffix.as_bytes().first() == Some(&b'0'))
        || !suffix.bytes().all(|byte| byte.is_ascii_digit())
    {
        return None;
    }
    suffix.parse().ok()
}

fn is_expected_member(
    name: &str,
    length: usize,
    prefix: &str,
    closing: Option<char>,
    metadata_names: &[&str],
) -> bool {
    metadata_names.contains(&name)
        || canonical_child_ordinal(name, prefix, closing)
            .is_some_and(|ordinal| ordinal != 0 && ordinal <= length)
}

fn first_unexpected_member(
    group: &Group,
    length: usize,
    prefix: &str,
    closing: Option<char>,
    metadata_names: &[&str],
) -> Result<Option<String>> {
    group
        .iter_visit_default(None, |_, name, _, unexpected| {
            if is_expected_member(name, length, prefix, closing, metadata_names) {
                true
            } else {
                *unexpected = Some(name.to_owned());
                false
            }
        })
        .with_context(|| "Failed to enumerate HDF5 child members")
}

/// Validate the exact child-link schema without retaining attacker-controlled members.
///
/// The declared child count is checked against the parent link count before any
/// caller allocates its result. Each expected child is then opened as a group and
/// dropped immediately. Callers reopen the validated groups only while reading them.
pub(crate) fn validate_expected_child_groups(
    group: &Group,
    length: usize,
    object_kind: &str,
    prefix: &str,
    closing: Option<char>,
    metadata_names: &[&str],
) -> Result<()> {
    let expected_member_count = u64::try_from(length)
        .context("HDF5 child length does not fit in a member count")?
        .checked_add(
            u64::try_from(metadata_names.len())
                .context("HDF5 metadata member count does not fit in a member count")?,
        )
        .context("HDF5 child length overflows the expected member count")?;
    let member_count = group.len();

    if member_count != expected_member_count {
        if let Some(name) = first_unexpected_member(group, length, prefix, closing, metadata_names)?
        {
            bail!(
                "HDF5 {object_kind} child member {name} is outside the exact declared range 1..={length} for declared length {length}"
            );
        }
        bail!(
            "HDF5 dataset length declares {length} {object_kind} children, but found {member_count} parent members instead of {expected_member_count}; expected child groups do not match"
        );
    }
    if let Some(name) = first_unexpected_member(group, length, prefix, closing, metadata_names)? {
        bail!(
            "HDF5 {object_kind} child member {name} is outside the exact declared range 1..={length} for declared length {length}"
        );
    }

    let non_hard_child = group
        .iter_visit_default(None, |_, name, info, non_hard_child| {
            if canonical_child_ordinal(name, prefix, closing)
                .is_some_and(|ordinal| ordinal != 0 && ordinal <= length)
                && info.link_type != LinkType::Hard
            {
                *non_hard_child = Some(name.to_owned());
                false
            } else {
                true
            }
        })
        .with_context(|| format!("Failed to inspect {object_kind} child links"))?;
    if let Some(name) = non_hard_child {
        bail!("HDF5 {object_kind} expected child {name} must be a hard-linked group");
    }

    for ordinal in 1..=length {
        let name = expected_child_name(prefix, ordinal, closing);
        let _child_group = group
            .group(&name)
            .with_context(|| format!("Failed to open HDF5 {object_kind} child group {name}"))?;
    }

    Ok(())
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

    validate_expected_child_groups(group, length, "IndexSet", "index_", None, &["length"])?;
    let mut indices = Vec::with_capacity(length);
    for ordinal in 1..=length {
        let name = expected_child_name("index_", ordinal, None);
        let index_group = group
            .group(&name)
            .with_context(|| format!("Failed to open HDF5 IndexSet child group {name}"))?;
        indices.push(read_index(&index_group)?);
    }

    Ok(indices)
}

#[cfg(test)]
mod tests;
