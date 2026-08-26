//! TreeTN extension of successive randomized compression.

use anyhow::Result;
use std::hash::Hash;
use tensor4all_core::{IndexLike, SvdTruncationPolicy, TensorLike};

use super::{SrcOptions, TreeTN};

/// Temporary implementation seam for the paper-faithful SRC rewrite.
pub(super) fn contract<T, V>(
    tn_a: &TreeTN<T, V>,
    tn_b: &TreeTN<T, V>,
    center: &V,
    svd_policy: Option<SvdTruncationPolicy>,
    max_bond_dim: usize,
    src_options: &SrcOptions,
) -> Result<TreeTN<T, V>>
where
    T: TensorLike,
    <T::Index as IndexLike>::Id: Clone + Hash + Eq + Ord + std::fmt::Debug + Send + Sync,
    V: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug,
{
    let _ = (tn_a, tn_b, center, svd_policy, max_bond_dim, src_options);
    anyhow::bail!("contract_src: paper-faithful implementation pending")
}
