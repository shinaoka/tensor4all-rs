//! Commonly used traits and types for tree tensor network construction,
//! canonicalization, truncation, and application.
//!
//! ```rust
//! use tensor4all_core::{DynIndex, TensorDynLen};
//! use tensor4all_treetn::prelude::*;
//!
//! let s0 = DynIndex::new_dyn(2);
//! let s1 = DynIndex::new_dyn(2);
//! let b01 = DynIndex::new_dyn(4);
//! let t0 = TensorDynLen::from_dense(vec![s0, b01.clone()], vec![1.0; 8]).unwrap();
//! let t1 = TensorDynLen::from_dense(vec![b01, s1], vec![1.0; 8]).unwrap();
//! let ttn = TreeTN::<TensorDynLen, usize>::from_tensors(vec![t0, t1], vec![0, 1]).unwrap();
//! assert_eq!(ttn.node_count(), 2);
//! assert_eq!(ttn.edge_count(), 1);
//! ```

pub use crate::{
    apply_linear_operator, dmrg, dmrg_with_treetn_operator, random_treetn, tdvp,
    tdvp_with_treetn_operator, tensor_train_to_treetn, ApplyOptions, BoundaryEdge, CanonicalForm,
    CanonicalizationOptions, CompressionAlgorithm, ContractionAlgorithm, DmrgOptions, DmrgResult,
    LinkIndexNetwork, LinkSpace, NamedGraph, NodeNameNetwork, Operator, RestructureOptions,
    SiteIndexNetwork, SplitOptions, SwapOptions, TdvpOptions, TdvpResult, TreeTN, TreeTopology,
    TruncationOptions,
};
