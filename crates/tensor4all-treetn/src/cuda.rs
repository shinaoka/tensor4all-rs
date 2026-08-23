//! Explicit CUDA transfer and pairwise TreeTN contraction for `IdxTensor`.

use std::collections::HashMap;
use std::fmt::Debug;
use std::hash::Hash;

use petgraph::stable_graph::NodeIndex;
use tensor4all_core::{
    IdxTensor, IdxTensorCudaError, IdxTensorError, TensorContractionLike, TensorIndex,
};
use tensor4all_tensorbackend::CudaExecutionContext;

use crate::error::TreeTNOperationError;
use crate::TreeTN;

/// Error returned by the explicit CUDA TreeTN transfer and contraction methods.
///
/// Node-local failures retain the failing [`NodeIndex`] and the typed source
/// error. The methods never transfer implicitly and never fall back to the CPU
/// contraction runtime.
///
/// # Examples
///
/// ```
/// use petgraph::stable_graph::NodeIndex;
/// use tensor4all_treetn::CudaTreeTNError;
///
/// let error = CudaTreeTNError::MissingTensor {
///     node: NodeIndex::new(0),
/// };
/// assert!(error.to_string().contains("node tensor"));
/// ```
#[derive(Debug, thiserror::Error)]
pub enum CudaTreeTNError {
    /// The network has no nodes to transfer or contract.
    #[error("CUDA TreeTN received an empty TreeTN; add at least one node")]
    EmptyNetwork,
    /// The network is not a connected tree.
    #[error("CUDA TreeTN topology validation failed; provide one connected tree: {source}")]
    Topology {
        /// Original topology diagnostic.
        #[source]
        source: TreeTNOperationError,
    },
    /// The selected root name had no graph node.
    #[error("CUDA TreeTN root node is missing; repair the network topology")]
    MissingRoot,
    /// A node index had no tensor payload.
    #[error(
        "CUDA TreeTN node tensor is missing; repair the network before transfer or contraction"
    )]
    MissingTensor {
        /// Node whose tensor was missing.
        node: NodeIndex,
    },
    /// An explicit upload failed for one node.
    #[error("CUDA TreeTN upload failed for a node; keep the source unchanged and inspect explicit upload: {source}")]
    Upload {
        /// Node whose tensor failed to upload.
        node: NodeIndex,
        /// Original typed tensor transfer diagnostic.
        #[source]
        source: IdxTensorCudaError,
    },
    /// An explicit download failed for one node.
    #[error("CUDA TreeTN download failed for a node; keep the source unchanged and use the same CUDA context: {source}")]
    Download {
        /// Node whose tensor failed to download.
        node: NodeIndex,
        /// Original typed tensor transfer diagnostic.
        #[source]
        source: IdxTensorCudaError,
    },
    /// A node or intermediate was not resident on the supplied CUDA context.
    #[error("CUDA TreeTN node residency validation failed; upload every node with this context and keep output transfer explicit: {source}")]
    Residency {
        /// Node whose residency failed validation.
        node: NodeIndex,
        /// Original typed residency diagnostic.
        #[source]
        source: IdxTensorCudaError,
    },
    /// A node dtype could not be read from metadata.
    #[error(
        "CUDA TreeTN node dtype validation failed; provide one supported dense dtype: {source}"
    )]
    Dtype {
        /// Node whose dtype failed validation.
        node: NodeIndex,
        /// Original typed dtype diagnostic.
        #[source]
        source: IdxTensorError,
    },
    /// Node tensors do not all have one exact dtype.
    #[error("CUDA TreeTN nodes have mixed dtypes; upload one exact dtype for every node before contraction")]
    MixedDtype {
        /// Node whose dtype differs from the first validated node.
        node: NodeIndex,
    },
    /// A pairwise edge contraction failed.
    #[error("CUDA TreeTN pairwise contraction failed; use dense same-dtype CUDA nodes: {source}")]
    Pairwise {
        /// Parent node receiving the pairwise contraction result.
        node: NodeIndex,
        /// Child node removed by this edge step.
        other_node: NodeIndex,
        /// Original typed pairwise contraction diagnostic.
        #[source]
        source: IdxTensorError,
    },
    /// The final result could not be permuted to canonical site-index order.
    #[error("CUDA TreeTN final index permutation failed; preserve the network site-index metadata: {source}")]
    Permutation {
        /// Root node holding the final result.
        node: NodeIndex,
        /// Original typed permutation diagnostic.
        #[source]
        source: IdxTensorError,
    },
    /// A transferred tensor could not replace its cloned source node.
    #[error("CUDA TreeTN node replacement failed; preserve the source network and repair its metadata: {source}")]
    Replacement {
        /// Node whose cloned network replacement failed.
        node: NodeIndex,
        /// Original typed TreeTN replacement diagnostic.
        #[source]
        source: TreeTNOperationError,
    },
    /// The pairwise result did not contain the network's complete site-index set.
    #[error("CUDA TreeTN result site indices are inconsistent; preserve the network site-index metadata")]
    OutputIndices {
        /// Root node holding the inconsistent result.
        node: NodeIndex,
    },
}

fn validate_cuda_tensor(
    node: NodeIndex,
    tensor: &IdxTensor,
    context: &CudaExecutionContext,
) -> Result<(), CudaTreeTNError> {
    tensor
        .validate_cuda_residency(context)
        .map_err(|source| CudaTreeTNError::Residency { node, source })
}

impl<V> TreeTN<IdxTensor, V>
where
    V: Clone + Hash + Eq + Send + Sync + Debug,
{
    /// Upload every node into one caller-owned CUDA context.
    ///
    /// The source TreeTN and all of its metadata remain unchanged. The returned
    /// network is a cloned metadata graph whose node tensors are CUDA-resident.
    /// All transfers are staged before the cloned graph is changed.
    ///
    /// # Errors
    ///
    /// Returns [`CudaTreeTNError`] when a node cannot be transferred or the
    /// cloned metadata graph rejects a replacement.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::{CudaExecutionContext, IdxTensor};
    /// use tensor4all_treetn::{CudaTreeTNError, TreeTN};
    ///
    /// let upload: fn(
    ///     &TreeTN<IdxTensor, usize>,
    ///     &CudaExecutionContext,
    /// ) -> Result<TreeTN<IdxTensor, usize>, CudaTreeTNError> = TreeTN::upload_cuda;
    /// assert_eq!(
    ///     std::mem::size_of_val(&upload),
    ///     std::mem::size_of::<fn(
    ///         &TreeTN<IdxTensor, usize>,
    ///         &CudaExecutionContext,
    ///     ) -> Result<TreeTN<IdxTensor, usize>, CudaTreeTNError>>(),
    /// );
    /// ```
    pub fn upload_cuda(&self, context: &CudaExecutionContext) -> Result<Self, CudaTreeTNError> {
        let nodes = self.node_indices();
        let mut replacements = Vec::with_capacity(nodes.len());
        for node in nodes {
            let tensor = self
                .tensor(node)
                .ok_or(CudaTreeTNError::MissingTensor { node })?;
            let uploaded = tensor
                .upload_cuda(context)
                .map_err(|source| CudaTreeTNError::Upload { node, source })?;
            replacements.push((node, uploaded));
        }

        let mut result = self.clone();
        for (node, tensor) in replacements {
            result
                .replace_tensor(node, tensor)
                .map_err(|source| CudaTreeTNError::Replacement { node, source })?;
        }
        Ok(result)
    }

    /// Download every node into host storage using one caller-owned CUDA context.
    ///
    /// The source TreeTN and all of its metadata remain unchanged. The returned
    /// network is a cloned metadata graph with host-backed node tensors.
    /// All downloads are staged before the cloned graph is changed.
    ///
    /// # Errors
    ///
    /// Returns [`CudaTreeTNError`] when a node is not resident on `context`, an
    /// explicit download fails, or a cloned metadata replacement fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::{CudaExecutionContext, IdxTensor};
    /// use tensor4all_treetn::{CudaTreeTNError, TreeTN};
    ///
    /// let download: fn(
    ///     &TreeTN<IdxTensor, usize>,
    ///     &CudaExecutionContext,
    /// ) -> Result<TreeTN<IdxTensor, usize>, CudaTreeTNError> = TreeTN::download;
    /// assert_eq!(
    ///     std::mem::size_of_val(&download),
    ///     std::mem::size_of::<fn(
    ///         &TreeTN<IdxTensor, usize>,
    ///         &CudaExecutionContext,
    ///     ) -> Result<TreeTN<IdxTensor, usize>, CudaTreeTNError>>(),
    /// );
    /// ```
    pub fn download(&self, context: &CudaExecutionContext) -> Result<Self, CudaTreeTNError> {
        let nodes = self.node_indices();
        let mut replacements = Vec::with_capacity(nodes.len());
        for node in nodes {
            let tensor = self
                .tensor(node)
                .ok_or(CudaTreeTNError::MissingTensor { node })?;
            let downloaded = tensor
                .download(context)
                .map_err(|source| CudaTreeTNError::Download { node, source })?;
            replacements.push((node, downloaded));
        }

        let mut result = self.clone();
        for (node, tensor) in replacements {
            result
                .replace_tensor(node, tensor)
                .map_err(|source| CudaTreeTNError::Replacement { node, source })?;
        }
        Ok(result)
    }

    /// Contract a dense CUDA-resident TreeTN to one CUDA-resident tensor.
    ///
    /// This is an explicitly dense full-network contraction. It validates every
    /// input before the first edge, walks the existing post-order topology, and
    /// calls only pairwise [`TensorContractionLike::contract_pair`] operations.
    /// The source network is never modified and the returned tensor remains on
    /// the supplied CUDA context until [`IdxTensor::download`] is called.
    ///
    /// # Errors
    ///
    /// Returns [`CudaTreeTNError`] for an empty or invalid tree, host/foreign
    /// placement, structured or tracked tensors, mixed dtypes, pairwise
    /// contraction failures, or final index-order failures. No CPU fallback is
    /// attempted.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::{CudaExecutionContext, IdxTensor};
    /// use tensor4all_treetn::{CudaTreeTNError, TreeTN};
    ///
    /// let contract: fn(
    ///     &TreeTN<IdxTensor, usize>,
    ///     &CudaExecutionContext,
    /// ) -> Result<IdxTensor, CudaTreeTNError> = TreeTN::contract_to_tensor_cuda;
    /// assert_eq!(
    ///     std::mem::size_of_val(&contract),
    ///     std::mem::size_of::<fn(
    ///         &TreeTN<IdxTensor, usize>,
    ///         &CudaExecutionContext,
    ///     ) -> Result<IdxTensor, CudaTreeTNError>>(),
    /// );
    /// ```
    pub fn contract_to_tensor_cuda(
        &self,
        context: &CudaExecutionContext,
    ) -> Result<IdxTensor, CudaTreeTNError>
    where
        V: Ord,
    {
        if self.node_count() == 0 {
            return Err(CudaTreeTNError::EmptyNetwork);
        }
        self.validate_tree()
            .map_err(|source| CudaTreeTNError::Topology { source })?;

        let mut node_names = self.node_names();
        node_names.sort();
        let root_name = node_names.first().ok_or(CudaTreeTNError::EmptyNetwork)?;
        let root = self
            .node_index(root_name)
            .ok_or(CudaTreeTNError::MissingRoot)?;

        let nodes = self.node_indices();
        let mut tensors = HashMap::with_capacity(nodes.len());
        let mut common_dtype = None;
        for node in nodes {
            let tensor = self
                .tensor(node)
                .cloned()
                .ok_or(CudaTreeTNError::MissingTensor { node })?;
            validate_cuda_tensor(node, &tensor, context)?;
            let dtype = tensor
                .cuda_dtype()
                .map_err(|source| CudaTreeTNError::Dtype { node, source })?;
            if let Some(expected) = common_dtype {
                if dtype != expected {
                    return Err(CudaTreeTNError::MixedDtype { node });
                }
            } else {
                common_dtype = Some(dtype);
            }
            tensors.insert(node, tensor);
        }

        if self.node_count() == 1 {
            return tensors
                .remove(&root)
                .ok_or(CudaTreeTNError::MissingTensor { node: root });
        }

        let edges = self.site_index_network().edges_to_canonicalize(None, root);
        for (from, to) in edges {
            let from_tensor = tensors
                .remove(&from)
                .ok_or(CudaTreeTNError::MissingTensor { node: from })?;
            let to_tensor = tensors
                .remove(&to)
                .ok_or(CudaTreeTNError::MissingTensor { node: to })?;
            let contracted = to_tensor.contract_pair(&from_tensor).map_err(|source| {
                CudaTreeTNError::Pairwise {
                    node: to,
                    other_node: from,
                    source,
                }
            })?;
            validate_cuda_tensor(to, &contracted, context)?;
            let dtype = contracted
                .cuda_dtype()
                .map_err(|source| CudaTreeTNError::Dtype { node: to, source })?;
            if common_dtype != Some(dtype) {
                return Err(CudaTreeTNError::MixedDtype { node: to });
            }
            tensors.insert(to, contracted);
        }

        let mut result = tensors
            .remove(&root)
            .ok_or(CudaTreeTNError::MissingTensor { node: root })?;
        let mut expected_indices = Vec::new();
        let mut expected_node_names = self.node_names();
        expected_node_names.sort();
        for node_name in expected_node_names {
            if let Some(site_space) = self.site_space(&node_name) {
                expected_indices.extend(site_space.iter().cloned());
            }
        }
        let current_indices = result.external_indices();
        if current_indices.len() != expected_indices.len() {
            return Err(CudaTreeTNError::OutputIndices { node: root });
        }
        if current_indices != expected_indices {
            result = result
                .permute_indices(&expected_indices)
                .map_err(|source| CudaTreeTNError::Permutation { node: root, source })?;
        }
        validate_cuda_tensor(root, &result, context)?;
        let dtype = result
            .cuda_dtype()
            .map_err(|source| CudaTreeTNError::Dtype { node: root, source })?;
        if common_dtype != Some(dtype) {
            return Err(CudaTreeTNError::MixedDtype { node: root });
        }
        Ok(result)
    }
}
