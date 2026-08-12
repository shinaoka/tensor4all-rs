//! Layer 3: TreeTN HDF5 read/write (tensor4all-rs TreeTN schema v1).
//!
//! A [`TreeTN`] is stored as metadata (`node_count`) plus one ITensor
//! subgroup per node (`node_1/` ... `node_N/`, 1-indexed like the MPS
//! schema), each carrying the node name as a string attribute. Topology is
//! not stored explicitly: bond connections are recovered on load from shared
//! [`Index`](tensor4all_core::Index) identity via
//! [`TreeTN::from_tensors`], exactly as the tree was assembled originally.
//! This works because the per-node ITensor schema already preserves full
//! index identity (id + prime level + tags), and every bond index appears in
//! exactly the two nodes it connects.

use crate::backend::types::VarLenUnicode;
use crate::backend::Group;
use anyhow::{Context, Result};
use std::fmt::Debug;
use std::fmt::Display;
use std::hash::Hash;
use std::str::FromStr;
use tensor4all_core::IdxTensor;
use tensor4all_treetn::TreeTN;

use crate::index;
use crate::itensor;
use crate::schema;

const NODE_NAME_ATTR: &str = "node_name";

/// Write a [`TreeTN`] to an HDF5 group using the tensor4all-rs TreeTN schema
/// v1.
///
/// Node tensors are stored as 1-indexed subgroups (`node_1/`, `node_2/`,
/// ...), each written with [`crate::itensor::write_itensor`] and annotated
/// with the node name as a `node_name` string attribute. The `node_count`
/// dataset records the number of nodes.
///
/// # HDF5 Schema
///
/// ```text
/// <group>/
///   @type = "TreeTN"
///   @version = 1
///   node_count: Int64
///   node_1/
///     @node_name: VarLenUnicode
///     (ITensor — see crate::itensor)
///   node_2/
///     ...
/// ```
pub(crate) fn write_treetn<V>(group: &Group, tn: &TreeTN<IdxTensor, V>) -> Result<()>
where
    V: ToString + Clone + Hash + Eq + Send + Sync + Debug,
{
    schema::write_type_version(group, "TreeTN", 1)?;

    let node_count = tn.node_count() as i64;
    let count_ds = group.new_dataset::<i64>().shape(()).create("node_count")?;
    count_ds.as_writer().write_scalar(&node_count)?;

    let names = tn.node_names();
    let indices = tn.node_indices();
    if names.len() != indices.len() {
        anyhow::bail!(
            "TreeTN has {} node names but {} node indices; refusing to save an inconsistent tree",
            names.len(),
            indices.len()
        );
    }

    for (i, (idx, name)) in indices.iter().zip(&names).enumerate() {
        let tensor = tn
            .tensor(*idx)
            .with_context(|| format!("TreeTN node {name:?} has no tensor"))?;
        let node_group = group.create_group(&format!("node_{}", i + 1))?;

        let name_attr = node_group
            .new_attr::<VarLenUnicode>()
            .shape(())
            .create(NODE_NAME_ATTR)?;
        name_attr
            .as_writer()
            .write_scalar(&VarLenUnicode::from_str(&name.to_string())?)?;

        itensor::write_itensor(&node_group, tensor)?;
    }

    Ok(())
}

/// Read a [`TreeTN`] from an HDF5 group using the tensor4all-rs TreeTN schema
/// v1.
///
/// Validates the `@type` and `@version` attributes, then reads each node's
/// tensor and `node_name` attribute from 1-indexed subgroups. The tree is
/// reconstructed with [`TreeTN::from_tensors`], which reconnects bond indices
/// by shared [`Index`](tensor4all_core::Index) identity and validates that
/// the result is a consistent tree.
pub(crate) fn read_treetn<V>(group: &Group) -> Result<TreeTN<IdxTensor, V>>
where
    V: FromStr + Ord + Clone + Hash + Eq + Send + Sync + Debug,
    V::Err: Display,
{
    schema::require_type_version(group, "TreeTN", 1)?;

    let node_count: i64 = group
        .dataset("node_count")?
        .as_reader()
        .read_scalar()
        .context("Failed to read TreeTN node_count")?;
    let node_count = index::read_nonnegative_usize("node_count", node_count)?;

    index::validate_expected_child_groups(
        group,
        node_count,
        "TreeTN",
        "node_",
        None,
        &["node_count"],
    )?;

    let mut names = Vec::with_capacity(node_count);
    let mut tensors = Vec::with_capacity(node_count);
    for ordinal in 1..=node_count {
        let name = format!("node_{ordinal}");
        let node_group = group
            .group(&name)
            .with_context(|| format!("Failed to open HDF5 TreeTN child group {name}"))?;

        let name_str = crate::compat::read_string_attr_by_name(&node_group, NODE_NAME_ATTR)?;
        let node_name = name_str
            .parse::<V>()
            .map_err(|e| anyhow::anyhow!("Failed to parse TreeTN node name {name_str:?}: {e}"))?;
        names.push(node_name);
        tensors.push(itensor::read_itensor(&node_group)?);
    }

    TreeTN::from_tensors(tensors, names).context("Failed to reconstruct TreeTN from HDF5 tensors")
}
