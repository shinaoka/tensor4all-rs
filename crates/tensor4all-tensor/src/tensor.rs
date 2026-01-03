use std::sync::Arc;
use std::collections::HashSet;
use num_complex::Complex64;
use tensor4all_core::index::{Index, NoSymmSpace, Symmetry};
use tensor4all_core::index_ops::{common_inds, check_unique_indices};
use crate::storage::{AnyScalar, Storage, StorageScalar, SumFromStorage, contract_storage, storage_to_dtensor};
use anyhow::Result;
use mdarray::DTensor;

/// Compute the permutation array from original indices to new indices.
///
/// This function finds the mapping from new indices to original indices by
/// matching index IDs. The result is a permutation array `perm` such that
/// `new_indices[i]` corresponds to `original_indices[perm[i]]`.
///
/// # Arguments
/// * `original_indices` - The original indices in their current order
/// * `new_indices` - The desired new indices order (must be a permutation of original_indices)
///
/// # Returns
/// A `Vec<usize>` representing the permutation: `perm[i]` is the position in
/// `original_indices` of the index that should be at position `i` in `new_indices`.
///
/// # Panics
/// Panics if any index ID in `new_indices` doesn't match an index in `original_indices`,
/// or if there are duplicate indices in `new_indices`.
///
/// # Example
/// ```
/// use tensor4all_tensor::tensor::compute_permutation_from_indices;
/// use tensor4all_core::index::{DefaultIndex as Index, DynId};
///
/// let i = Index::new_dyn(2);
/// let j = Index::new_dyn(3);
/// let original = vec![i.clone(), j.clone()];
/// let new_order = vec![j.clone(), i.clone()];
///
/// let perm = compute_permutation_from_indices(&original, &new_order);
/// assert_eq!(perm, vec![1, 0]);  // j is at position 1, i is at position 0
/// ```
pub fn compute_permutation_from_indices<Id, Symm>(
    original_indices: &[Index<Id, Symm>],
    new_indices: &[Index<Id, Symm>],
) -> Vec<usize>
where
    Id: std::hash::Hash + Eq,
{
    assert_eq!(
        new_indices.len(),
        original_indices.len(),
        "new_indices length must match original_indices length"
    );

    let mut perm = Vec::with_capacity(new_indices.len());
    let mut used = std::collections::HashSet::new();

    for new_idx in new_indices {
        // Find the position of this index in the original indices
        let pos = original_indices
            .iter()
            .position(|old_idx| old_idx.id == new_idx.id)
            .expect("new_indices must be a permutation of original_indices");
        
        if used.contains(&pos) {
            panic!("duplicate index in new_indices");
        }
        used.insert(pos);
        perm.push(pos);
    }

    perm
}

/// Trait for extracting type parameters from tensor types.
///
/// This allows extracting `Id` and `Symm` from tensor types like `TensorDynLen<Id, Symm>`
/// without exposing them directly in the type signature.
pub trait TensorType {
    /// Index ID type
    type Id: Clone + std::hash::Hash + Eq;
    /// Symmetry type
    type Symm: Clone + Symmetry;
}

/// Trait for accessing tensor fields.
///
/// This trait provides access to the internal fields of tensor types,
/// allowing generic code to work with tensors without knowing the exact type.
pub trait TensorAccess: TensorType {
    /// Get a reference to the indices.
    fn indices(&self) -> &[Index<Self::Id, Self::Symm>];
    
    /// Get a reference to the storage.
    fn storage(&self) -> &Storage;
}

/// Tensor with dynamic rank (number of indices) and dynamic scalar type.
pub struct TensorDynLen<Id, Symm = NoSymmSpace> {
    pub indices: Vec<Index<Id, Symm>>,
    pub dims: Vec<usize>,
    pub storage: Arc<Storage>,
}

impl<Id, Symm> TensorType for TensorDynLen<Id, Symm>
where
    Id: Clone + std::hash::Hash + Eq,
    Symm: Clone + Symmetry,
{
    type Id = Id;
    type Symm = Symm;
}

impl<Id, Symm> TensorAccess for TensorDynLen<Id, Symm>
where
    Id: Clone + std::hash::Hash + Eq,
    Symm: Clone + Symmetry,
{
    fn indices(&self) -> &[Index<Self::Id, Self::Symm>] {
        &self.indices
    }
    
    fn storage(&self) -> &Storage {
        &self.storage
    }
}

impl<Id, Symm> TensorDynLen<Id, Symm> {
    /// Create a new tensor with dynamic rank.
    ///
    /// Dimensions are automatically computed from the indices using `Index::size()`.
    ///
    /// # Panics
    /// Panics if the storage is Diag and not all indices have the same dimension.
    /// Panics if there are duplicate indices (indices with the same ID).
    pub fn new(indices: Vec<Index<Id, Symm>>, dims: Vec<usize>, storage: Arc<Storage>) -> Self
    where
        Id: std::hash::Hash + Eq,
    {
        assert_eq!(
            indices.len(),
            dims.len(),
            "indices and dims must have the same length"
        );
        
        // Check for duplicate indices
        check_unique_indices(&indices).expect("Tensor indices must all be unique (no duplicate IDs)");
        
        // Validate DiagTensor: all indices must have the same dimension
        if storage.as_ref().is_diag() {
            let first_dim = dims[0];
            for (i, &dim) in dims.iter().enumerate() {
                assert_eq!(
                    dim, first_dim,
                    "DiagTensor requires all indices to have the same dimension, but dims[{}] = {} != dims[0] = {}",
                    i, dim, first_dim
                );
            }
        }
        
        Self {
            indices,
            dims,
            storage,
        }
    }

    /// Create a new tensor with dynamic rank, automatically computing dimensions from indices.
    ///
    /// This is a convenience constructor that extracts dimensions from indices using `Index::size()`.
    ///
    /// # Panics
    /// Panics if the storage is Diag and not all indices have the same dimension.
    /// Panics if there are duplicate indices (indices with the same ID).
    pub fn from_indices(indices: Vec<Index<Id, Symm>>, storage: Arc<Storage>) -> Self
    where
        Symm: Symmetry,
        Id: std::hash::Hash + Eq,
    {
        let dims: Vec<usize> = indices.iter().map(|idx| idx.size()).collect();
        Self::new(indices, dims, storage)
    }

    /// Get a mutable reference to storage (COW: clones if shared).
    pub fn storage_mut(&mut self) -> &mut Storage {
        Arc::make_mut(&mut self.storage)
    }

    /// Sum all elements, returning `AnyScalar`.
    pub fn sum(&self) -> AnyScalar {
        AnyScalar::sum_from_storage(&self.storage)
    }

    /// Sum all elements as f64.
    pub fn sum_f64(&self) -> f64 {
        f64::sum_from_storage(&self.storage)
    }

    /// Permute the tensor dimensions using the given new indices order.
    ///
    /// This is the main permutation method that takes the desired new indices
    /// and automatically computes the corresponding permutation of dimensions
    /// and data. The new indices must be a permutation of the original indices
    /// (matched by ID).
    ///
    /// # Arguments
    /// * `new_indices` - The desired new indices order. Must be a permutation
    ///   of `self.indices` (matched by ID).
    ///
    /// # Panics
    /// Panics if `new_indices.len() != self.indices.len()`, if any index ID
    /// doesn't match, or if there are duplicate indices.
    ///
    /// # Example
    /// ```
    /// use tensor4all_tensor::TensorDynLen;
    /// use tensor4all_core::index::{DefaultIndex as Index, DynId};
    /// use tensor4all_tensor::Storage;
    /// use tensor4all_tensor::storage::DenseStorageF64;
    /// use std::sync::Arc;
    ///
    /// // Create a 2×3 tensor
    /// let i = Index::new_dyn(2);
    /// let j = Index::new_dyn(3);
    /// let indices = vec![i.clone(), j.clone()];
    /// let dims = vec![2, 3];
    /// let storage = Arc::new(Storage::DenseF64(DenseStorageF64::from_vec(vec![0.0; 6])));
    /// let tensor: TensorDynLen<DynId> = TensorDynLen::new(indices, dims, storage);
    ///
    /// // Permute to 3×2: swap the two dimensions by providing new indices order
    /// let permuted = tensor.permute_indices(&[j, i]);
    /// assert_eq!(permuted.dims, vec![3, 2]);
    /// ```
    pub fn permute_indices(&self, new_indices: &[Index<Id, Symm>]) -> Self
    where
        Id: Clone + std::hash::Hash + Eq,
        Symm: Clone + Symmetry,
    {
        // Compute permutation by matching IDs
        let perm = compute_permutation_from_indices(&self.indices, new_indices);

        // Compute new dims from new indices
        let new_dims: Vec<usize> = new_indices
            .iter()
            .map(|idx| idx.size())
            .collect();

        // Permute storage data using the computed permutation
        let new_storage = Arc::new(self.storage.permute_storage(&self.dims, &perm));

        Self {
            indices: new_indices.to_vec(),
            dims: new_dims,
            storage: new_storage,
        }
    }

    /// Permute the tensor dimensions, returning a new tensor.
    ///
    /// This method reorders the indices, dimensions, and data according to the
    /// given permutation. The permutation specifies which old axis each new
    /// axis corresponds to: `new_axis[i] = old_axis[perm[i]]`.
    ///
    /// # Arguments
    /// * `perm` - The permutation: `perm[i]` is the old axis index for new axis `i`
    ///
    /// # Panics
    /// Panics if `perm.len() != self.dims.len()` or if the permutation is invalid.
    ///
    /// # Example
    /// ```
    /// use tensor4all_tensor::TensorDynLen;
    /// use tensor4all_core::index::{DefaultIndex as Index, DynId};
    /// use tensor4all_tensor::Storage;
    /// use tensor4all_tensor::storage::DenseStorageF64;
    /// use std::sync::Arc;
    ///
    /// // Create a 2×3 tensor
    /// let indices = vec![
    ///     Index::new_dyn(2),
    ///     Index::new_dyn(3),
    /// ];
    /// let dims = vec![2, 3];
    /// let storage = Arc::new(Storage::DenseF64(DenseStorageF64::from_vec(vec![0.0; 6])));
    /// let tensor: TensorDynLen<DynId> = TensorDynLen::new(indices, dims, storage);
    ///
    /// // Permute to 3×2: swap the two dimensions
    /// let permuted = tensor.permute(&[1, 0]);
    /// assert_eq!(permuted.dims, vec![3, 2]);
    /// ```
    pub fn permute(&self, perm: &[usize]) -> Self
    where
        Id: Clone,
        Symm: Clone,
    {
        assert_eq!(
            perm.len(),
            self.dims.len(),
            "permutation length must match tensor rank"
        );

        // Permute indices and dims
        let new_indices: Vec<Index<Id, Symm>> = perm
            .iter()
            .map(|&i| self.indices[i].clone())
            .collect();
        let new_dims: Vec<usize> = perm
            .iter()
            .map(|&i| self.dims[i])
            .collect();

        // Permute storage data
        let new_storage = Arc::new(self.storage.permute_storage(&self.dims, perm));

        Self {
            indices: new_indices,
            dims: new_dims,
            storage: new_storage,
        }
    }

    /// Contract this tensor with another tensor along common indices.
    ///
    /// This method finds common indices between `self` and `other`, then contracts
    /// along those indices. The result tensor contains all non-contracted indices
    /// from both tensors, with indices from `self` appearing first, followed by
    /// indices from `other` that are not common.
    ///
    /// # Arguments
    /// * `other` - The tensor to contract with
    ///
    /// # Returns
    /// A new tensor resulting from the contraction.
    ///
    /// # Panics
    /// Panics if there are no common indices, if common indices have mismatched
    /// dimensions, or if storage types don't match.
    ///
    /// # Example
    /// ```
    /// use tensor4all_tensor::TensorDynLen;
    /// use tensor4all_core::index::{DefaultIndex as Index, DynId};
    /// use tensor4all_tensor::Storage;
    /// use tensor4all_tensor::storage::DenseStorageF64;
    /// use std::sync::Arc;
    ///
    /// // Create two tensors: A[i, j] and B[j, k]
    /// let i = Index::new_dyn(2);
    /// let j = Index::new_dyn(3);
    /// let k = Index::new_dyn(4);
    ///
    /// let indices_a = vec![i.clone(), j.clone()];
    /// let dims_a = vec![2, 3];
    /// let storage_a = Arc::new(Storage::DenseF64(DenseStorageF64::from_vec(vec![0.0; 6])));
    /// let tensor_a: TensorDynLen<DynId> = TensorDynLen::new(indices_a, dims_a, storage_a);
    ///
    /// let indices_b = vec![j.clone(), k.clone()];
    /// let dims_b = vec![3, 4];
    /// let storage_b = Arc::new(Storage::DenseF64(DenseStorageF64::from_vec(vec![0.0; 12])));
    /// let tensor_b: TensorDynLen<DynId> = TensorDynLen::new(indices_b, dims_b, storage_b);
    ///
    /// // Contract along j: result is C[i, k]
    /// let result = tensor_a.contract(&tensor_b);
    /// assert_eq!(result.dims, vec![2, 4]);
    /// ```
    pub fn contract(&self, other: &Self) -> Self
    where
        Id: Clone + std::hash::Hash + Eq,
        Symm: Clone + Symmetry,
    {
        // Find common indices
        let common = common_inds(&self.indices, &other.indices);
        if common.is_empty() {
            panic!("No common indices found for contraction");
        }

        // Find positions of common indices in both tensors
        let mut axes_a = Vec::new();
        let mut axes_b = Vec::new();

        for common_idx in &common {
            // Find position in self
            let pos_a = self.indices
                .iter()
                .position(|idx| idx.id == common_idx.id)
                .expect("common index must be in self");
            axes_a.push(pos_a);

            // Find position in other
            let pos_b = other.indices
                .iter()
                .position(|idx| idx.id == common_idx.id)
                .expect("common index must be in other");
            axes_b.push(pos_b);

            // Verify dimensions match
            assert_eq!(
                self.dims[pos_a],
                other.dims[pos_b],
                "Common index dimension mismatch: {} != {}",
                self.dims[pos_a],
                other.dims[pos_b]
            );
        }

        // Get non-contracted indices
        let mut result_indices = Vec::new();
        let mut result_dims = Vec::new();

        // Add non-contracted indices from self
        for (i, idx) in self.indices.iter().enumerate() {
            if !axes_a.contains(&i) {
                result_indices.push(idx.clone());
                result_dims.push(self.dims[i]);
            }
        }

        // Add non-contracted indices from other
        for (i, idx) in other.indices.iter().enumerate() {
            if !axes_b.contains(&i) {
                result_indices.push(idx.clone());
                result_dims.push(other.dims[i]);
            }
        }

        // Perform contraction
        let result_storage = Arc::new(contract_storage(
            &self.storage,
            &self.dims,
            &axes_a,
            &other.storage,
            &other.dims,
            &axes_b,
            &result_dims,
        ));

        Self {
            indices: result_indices,
            dims: result_dims,
            storage: result_storage,
        }
    }

    /// Contract this tensor with another tensor along explicitly specified index pairs.
    ///
    /// This method contracts along the specified pairs of indices, where each pair
    /// consists of an index from `self` and an index from `other`. The indices
    /// do not need to have matching IDs - only matching dimensions are required.
    ///
    /// # Arguments
    /// * `other` - The tensor to contract with
    /// * `pairs` - Pairs of indices to contract: `(index_from_self, index_from_other)`
    ///
    /// # Returns
    /// A new tensor resulting from the contraction, or an error if:
    /// - Any specified index is not found in the respective tensor
    /// - Dimensions don't match for any pair
    /// - The same axis is specified multiple times in `self` or `other`
    ///
    /// # Example
    /// ```
    /// use tensor4all_tensor::TensorDynLen;
    /// use tensor4all_core::index::{DefaultIndex as Index, DynId};
    /// use tensor4all_tensor::Storage;
    /// use tensor4all_tensor::storage::DenseStorageF64;
    /// use std::sync::Arc;
    ///
    /// // Create two tensors: A[i, j] and B[k, l] where j and k have same dimension but different IDs
    /// let i = Index::new_dyn(2);
    /// let j = Index::new_dyn(3);
    /// let k = Index::new_dyn(3);  // Same dimension as j, but different ID
    /// let l = Index::new_dyn(4);
    ///
    /// let indices_a = vec![i.clone(), j.clone()];
    /// let dims_a = vec![2, 3];
    /// let storage_a = Arc::new(Storage::DenseF64(DenseStorageF64::from_vec(vec![0.0; 6])));
    /// let tensor_a: TensorDynLen<DynId> = TensorDynLen::new(indices_a, dims_a, storage_a);
    ///
    /// let indices_b = vec![k.clone(), l.clone()];
    /// let dims_b = vec![3, 4];
    /// let storage_b = Arc::new(Storage::DenseF64(DenseStorageF64::from_vec(vec![0.0; 12])));
    /// let tensor_b: TensorDynLen<DynId> = TensorDynLen::new(indices_b, dims_b, storage_b);
    ///
    /// // Contract j (from A) with k (from B): result is C[i, l]
    /// let result = tensor_a.contract_pairs(&tensor_b, &[(j.clone(), k.clone())]).unwrap();
    /// assert_eq!(result.dims, vec![2, 4]);
    /// ```
    pub fn contract_pairs(
        &self,
        other: &Self,
        pairs: &[(Index<Id, Symm>, Index<Id, Symm>)],
    ) -> Result<Self>
    where
        Id: Clone + std::hash::Hash + Eq,
        Symm: Clone + Symmetry,
    {
        use anyhow::Context;

        if pairs.is_empty() {
            return Err(anyhow::anyhow!("No pairs specified for contraction"))
                .context("contract_pairs: at least one pair must be specified");
        }

        // Find positions of indices in both tensors and validate
        let mut axes_a = Vec::new();
        let mut axes_b = Vec::new();

        for (idx_a, idx_b) in pairs {
            // Find position in self
            let pos_a = self.indices
                .iter()
                .position(|idx| idx.id == idx_a.id)
                .ok_or_else(|| anyhow::anyhow!("Index with id matching specified index not found in self tensor"))
                .context("contract_pairs: index from self not found")?;

            // Find position in other
            let pos_b = other.indices
                .iter()
                .position(|idx| idx.id == idx_b.id)
                .ok_or_else(|| anyhow::anyhow!("Index with id matching specified index not found in other tensor"))
                .context("contract_pairs: index from other not found")?;

            // Verify dimensions match
            if self.dims[pos_a] != other.dims[pos_b] {
                return Err(anyhow::anyhow!(
                    "Dimension mismatch for pair: self[{}] = {} != other[{}] = {}",
                    pos_a,
                    self.dims[pos_a],
                    pos_b,
                    other.dims[pos_b]
                ))
                .context("contract_pairs: dimensions must match for each pair");
            }

            // Check for duplicate axes in self
            if axes_a.contains(&pos_a) {
                return Err(anyhow::anyhow!(
                    "Duplicate axis {} in self tensor",
                    pos_a
                ))
                .context("contract_pairs: each axis can only be contracted once");
            }

            // Check for duplicate axes in other
            if axes_b.contains(&pos_b) {
                return Err(anyhow::anyhow!(
                    "Duplicate axis {} in other tensor",
                    pos_b
                ))
                .context("contract_pairs: each axis can only be contracted once");
            }

            axes_a.push(pos_a);
            axes_b.push(pos_b);
        }

        // Get non-contracted indices
        let mut result_indices = Vec::new();
        let mut result_dims = Vec::new();

        // Add non-contracted indices from self
        for (i, idx) in self.indices.iter().enumerate() {
            if !axes_a.contains(&i) {
                result_indices.push(idx.clone());
                result_dims.push(self.dims[i]);
            }
        }

        // Add non-contracted indices from other
        for (i, idx) in other.indices.iter().enumerate() {
            if !axes_b.contains(&i) {
                result_indices.push(idx.clone());
                result_dims.push(other.dims[i]);
            }
        }

        // Perform contraction
        let result_storage = Arc::new(contract_storage(
            &self.storage,
            &self.dims,
            &axes_a,
            &other.storage,
            &other.dims,
            &axes_b,
            &result_dims,
        ));

        Ok(Self {
            indices: result_indices,
            dims: result_dims,
            storage: result_storage,
        })
    }
}


/// Check if a tensor is a DiagTensor (has Diag storage).
pub fn is_diag_tensor<Id, Symm>(tensor: &TensorDynLen<Id, Symm>) -> bool {
    tensor.storage.as_ref().is_diag()
}


/// Create a DiagTensor with dynamic rank from diagonal data.
///
/// # Arguments
/// * `indices` - The indices for the tensor (all must have the same dimension)
/// * `diag_data` - The diagonal elements (length must equal the dimension of indices)
///
/// # Panics
/// Panics if indices have different dimensions, or if diag_data length doesn't match.
pub fn diag_tensor_dyn_len<Id, Symm>(
    indices: Vec<Index<Id, Symm>>,
    diag_data: Vec<f64>,
) -> TensorDynLen<Id, Symm>
where
    Id: Clone + std::hash::Hash + Eq,
    Symm: Clone + Symmetry,
{
    let dims: Vec<usize> = indices.iter().map(|idx| idx.size()).collect();
    let first_dim = dims[0];
    
    // Validate all indices have same dimension
    for (i, &dim) in dims.iter().enumerate() {
        assert_eq!(
            dim, first_dim,
            "DiagTensor requires all indices to have the same dimension, but dims[{}] = {} != dims[0] = {}",
            i, dim, first_dim
        );
    }
    
    assert_eq!(
        diag_data.len(),
        first_dim,
        "diag_data length ({}) must equal index dimension ({})",
        diag_data.len(),
        first_dim
    );
    
    let storage = Arc::new(Storage::new_diag_f64(diag_data));
    TensorDynLen::new(indices, dims, storage)
}

/// Create a DiagTensor with dynamic rank from complex diagonal data.
pub fn diag_tensor_dyn_len_c64<Id, Symm>(
    indices: Vec<Index<Id, Symm>>,
    diag_data: Vec<Complex64>,
) -> TensorDynLen<Id, Symm>
where
    Id: Clone + std::hash::Hash + Eq,
    Symm: Clone + Symmetry,
{
    let dims: Vec<usize> = indices.iter().map(|idx| idx.size()).collect();
    let first_dim = dims[0];
    
    // Validate all indices have same dimension
    for (i, &dim) in dims.iter().enumerate() {
        assert_eq!(
            dim, first_dim,
            "DiagTensor requires all indices to have the same dimension, but dims[{}] = {} != dims[0] = {}",
            i, dim, first_dim
        );
    }
    
    assert_eq!(
        diag_data.len(),
        first_dim,
        "diag_data length ({}) must equal index dimension ({})",
        diag_data.len(),
        first_dim
    );
    
    let storage = Arc::new(Storage::new_diag_c64(diag_data));
    TensorDynLen::new(indices, dims, storage)
}


/// Unfold a tensor into a matrix by splitting indices into left and right groups.
///
/// This function validates the split, permutes the tensor so that left indices come first,
/// and returns a 2D matrix tensor (`DTensor<T, 2>`) along with metadata.
///
/// # Arguments
/// * `t` - Input tensor
/// * `left_inds` - Indices to place on the left (row) side of the matrix
///
/// # Returns
/// A tuple `(matrix_tensor, left_len, m, n, left_indices, right_indices)` where:
/// - `matrix_tensor` is a `DTensor<T, 2>` with shape `[m, n]` containing the unfolded data
/// - `left_len` is the number of left indices
/// - `m` is the product of left index dimensions
/// - `n` is the product of right index dimensions
/// - `left_indices` is the vector of left indices (cloned)
/// - `right_indices` is the vector of right indices (cloned)
///
/// # Errors
/// Returns an error if:
/// - The tensor rank is < 2
/// - `left_inds` is empty or contains all indices
/// - `left_inds` contains indices not in the tensor or duplicates
/// - Storage type is not supported (must be DenseF64 or DenseC64)
pub fn unfold_split<Id, T, Symm>(
    t: &TensorDynLen<Id, Symm>,
    left_inds: &[Index<Id, Symm>],
) -> Result<(
    DTensor<T, 2>,
    usize,
    usize,
    usize,
    Vec<Index<Id, Symm>>,
    Vec<Index<Id, Symm>>,
)>
where
    Id: Clone + std::hash::Hash + Eq,
    Symm: Clone + Symmetry,
    T: StorageScalar,
{
    let rank = t.dims.len();
    
    // Validate rank
    anyhow::ensure!(rank >= 2, "Tensor must have rank >= 2, got rank {}", rank);

    let left_len = left_inds.len();
    
    // Validate split: must be a proper subset
    anyhow::ensure!(
        left_len > 0 && left_len < rank,
        "Left indices must be a non-empty proper subset of tensor indices (0 < left_len < rank), got left_len={}, rank={}",
        left_len,
        rank
    );

    // Validate that all left_inds are in the tensor and there are no duplicates
    let tensor_id_set: HashSet<_> = t.indices.iter().map(|idx| &idx.id).collect();
    let mut left_id_set = HashSet::new();
    
    for left_idx in left_inds {
        anyhow::ensure!(
            tensor_id_set.contains(&left_idx.id),
            "Index in left_inds not found in tensor"
        );
        anyhow::ensure!(
            left_id_set.insert(&left_idx.id),
            "Duplicate index in left_inds"
        );
    }

    // Build right_inds: all indices not in left_inds, in original order
    let mut right_inds = Vec::new();
    for idx in &t.indices {
        if !left_id_set.contains(&idx.id) {
            right_inds.push(idx.clone());
        }
    }

    // Build new_indices: left_inds first, then right_inds
    let mut new_indices = Vec::with_capacity(rank);
    new_indices.extend_from_slice(left_inds);
    new_indices.extend_from_slice(&right_inds);

    // Permute tensor to have left indices first, then right indices
    let unfolded = t.permute_indices(&new_indices);

    // Compute matrix dimensions
    let m: usize = unfolded.dims[..left_len].iter().product();
    let n: usize = unfolded.dims[left_len..].iter().product();

    // Create DTensor directly from storage
    let matrix_tensor = storage_to_dtensor::<T>(unfolded.storage.as_ref(), [m, n])
        .map_err(|e| anyhow::anyhow!("Failed to create DTensor: {}", e))?;

    Ok((
        matrix_tensor,
        left_len,
        m,
        n,
        left_inds.to_vec(),
        right_inds,
    ))
}

