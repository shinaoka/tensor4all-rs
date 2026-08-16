//! Commonly used types for chain-facing tensor train operations backed by
//! [`TreeTN`](crate::tensor4all_treetn::TreeTN).
//!
//! ```rust
//! # fn main() -> anyhow::Result<()> {
//! use tensor4all_itensorlike::prelude::*;
//!
//! let s0 = DynIndex::new_dyn(2);
//! let s1 = DynIndex::new_dyn(2);
//! let b01 = DynIndex::new_bond(2)?;
//! let t0 = IdxTensor::from_dense(vec![s0, b01.clone()], vec![1.0, 0.0, 0.0, 1.0])?;
//! let t1 = IdxTensor::from_dense(vec![b01, s1], vec![1.0, 0.0, 0.0, 1.0])?;
//! let mut tt = TensorTrain::new(vec![t0, t1])?;
//! tt.orthogonalize(0)?;
//! tt.truncate(&TruncateOptions::svd()
//!     .with_svd_policy(SvdTruncationPolicy::new(1e-10))
//!     .with_max_bond_dim(2))?;
//! assert!(tt.is_ortho());
//! # Ok(())
//! # }
//! ```

pub use crate::{
    CanonicalForm, ContractMethod, ContractOptions, LinsolveOptions, TensorTrain, TruncateOptions,
};
pub use tensor4all_core::{DynIndex, IdxTensor, IndexLike, SvdTruncationPolicy};
