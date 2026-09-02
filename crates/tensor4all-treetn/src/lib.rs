#![warn(missing_docs)]
//! Tree Tensor Network (TreeTN) library for tensor4all.
//!
//! This crate provides data structures and algorithms for working with tree tensor networks,
//! a generalization of matrix product states (MPS) to tree-shaped graphs. Tree tensor networks
//! are useful for representing quantum states and operators on systems with tree-like connectivity.
//!
//! # Key Types
//!
//! - [`TreeTN`]: The main tree tensor network type, parameterized by tensor and node name types.
//! - [`DefaultTreeTN`]: A convenient alias for `TreeTN<IdxTensor, NodeIndex>`.
//! - [`NamedGraph`]: A graph wrapper that maps node names to internal graph indices.
//!
//! # Features
//!
//! - **Canonicalization**: Transform networks into canonical forms (unitary, LU, CI).
//! - **Truncation**: Compress networks using SVD-based truncation.
//! - **Contraction**: Contract two networks using zip-up, fit, or SRC algorithms.
//! - **Linear operators**: Apply and compose linear operators on tree tensor networks.
//! - **Linear solvers**: Solve linear systems involving tree tensor network operators.

#[cfg(doctest)]
#[doc = include_str!("../README.md")]
pub struct ReadmeDoctests;

pub mod prelude;

mod algorithm;
#[cfg(feature = "tenferro-cuda")]
mod cuda;
// dyn_treetn.rs has been removed.
// TreeTN uses the `T: TensorLike` pattern, making a separate dyn wrapper unnecessary.
mod dmrg;
mod error;
mod gse;
mod link_index_network;
mod linsolve;
mod local_update_support;
mod named_graph;
mod node_name_network;
mod operator;
mod options;
mod random;
#[cfg(feature = "simplett-bridge")]
mod simplett_bridge;
mod site_index_network;
mod tdvp;
mod treetn;

pub use algorithm::{CanonicalForm, CompressionAlgorithm, ContractionAlgorithm};
#[cfg(feature = "tenferro-cuda")]
pub use cuda::CudaTreeTNError;
pub use dmrg::{dmrg, dmrg_with_treetn_operator, DmrgError, DmrgOptions, DmrgResult};
pub use error::{
    LinearOperatorIndexApplyError, LinearOperatorIndexBindingError, LinearOperatorTaggedApplyError,
    NumberedTagSelectionError, SelectedIndexContractionError, TreeTNOperationError,
};
pub use gse::{
    global_subspace_expand, global_subspace_expand_with_references, gse_tdvp, GseError, GseOptions,
    GseResult, GseTdvpOptions, GseTdvpResult,
};
pub use treetn::contraction;
pub use treetn::contraction::{contract_src_with_rng, SrcOptions};

// dyn_treetn exports removed - use TreeTN<IdxTensor, V> directly
pub use link_index_network::LinkIndexNetwork;
pub use named_graph::NamedGraph;
pub use node_name_network::{CanonicalizeEdges, NodeNameNetwork};
pub use operator::{
    apply_linear_operator, apply_linear_operator_to_indices,
    apply_linear_operator_to_numbered_tags, are_exclusive_operators, bind_linear_operator_indices,
    build_identity_operator_tensor, compose_exclusive_linear_operators, ApplyOptions,
    ArcLinearOperator, IndexMapping, LinearOperator, Operator,
};
pub use options::{CanonicalizationOptions, RestructureOptions, SplitOptions, TruncationOptions};
pub use random::{random_treetn, LinkSpace};
#[cfg(feature = "simplett-bridge")]
pub use simplett_bridge::{
    fix_and_remove_site_from_treetn_chain, insert_onehot_site_in_treetn_chain,
    tensor_train_to_treetn, tensor_train_to_treetn_with_names,
    tensor_train_to_treetn_with_names_and_site_indices, treetn_to_tensor_train,
    weighted_remove_site_from_treetn_chain,
};
pub use site_index_network::SiteIndexNetwork;
pub use tdvp::{tdvp, tdvp_with_treetn_operator, TdvpError, TdvpOptions, TdvpResult};
#[cfg(feature = "diagnostics")]
pub use treetn::diagnostics;
pub use treetn::{
    // Local update
    apply_local_update_sweep,
    // Decomposition
    factorize_tensor_to_treetn,
    factorize_tensor_to_treetn_with,
    fit_sum,
    get_boundary_edges,
    hadamard,
    partial_contract,
    partial_contract_to_site_network,
    sum_over_indices,
    weighted_sum_over_index_pairs,
    BoundaryEdge,
    CachedEvaluatorOptions,
    CenterSearchResult,
    EvaluationHint,
    FitOptions,
    // Core type
    GreedyCenterSearch,
    LocalUpdateStep,
    LocalUpdateSweepPlan,
    LocalUpdater,
    PartialContractionSpec,
    // Swap
    ScheduledSwapStep,
    SwapOptions,
    SwapSchedule,
    TreeTN,
    TreeTNCachedEvaluator,
    TreeTNEvaluator,
    TreeTopology,
    TruncateUpdater,
};

// Re-export linsolve types from new location
pub use linsolve::{
    relative_linear_system_residual, square_linsolve, EnvironmentCache, LinsolveOptions,
    LinsolveVerifyReport, NetworkTopology, NodeVerifyDetail, ProjectedOperator, ProjectedState,
    SquareLinsolveResult, SquareLinsolveUpdater,
};

use petgraph::graph::NodeIndex;
use tensor4all_core::IdxTensor;

/// Default TreeTN type using IdxTensor as the tensor type.
///
/// This is the most common configuration for TreeTN, equivalent to:
/// ```
/// use tensor4all_treetn::DefaultTreeTN;
///
/// let tree: DefaultTreeTN = DefaultTreeTN::new();
/// assert_eq!(tree.node_count(), 0);
/// ```
///
/// Use this when you don't need custom tensor types.
pub type DefaultTreeTN<V = NodeIndex> = TreeTN<IdxTensor, V>;
