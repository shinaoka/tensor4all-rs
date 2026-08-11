//! Block tensor type for GMRES with block matrices.
//!
//! This module provides [`BlockTensor`], a collection of tensors organized
//! in a block structure. It implements [`TensorLike`] for the vector space
//! operations required by GMRES, allowing block matrix linear equations
//! `Ax = b` to be solved using the existing GMRES implementation.
//!
//! # Example
//!
//! ```
//! use tensor4all_core::block_tensor::BlockTensor;
//! use tensor4all_core::krylov::{gmres, GmresOptions};
//! use tensor4all_core::{DynIndex, TensorDynLen};
//!
//! # fn main() -> anyhow::Result<()> {
//! let i = DynIndex::new_dyn(2);
//! let b_block = TensorDynLen::from_dense(vec![i.clone()], vec![1.0, 2.0])?;
//! let zero_block = TensorDynLen::from_dense(vec![i.clone()], vec![0.0, 0.0])?;
//!
//! let b = BlockTensor::new(vec![b_block], (1, 1))?;
//! let x0 = BlockTensor::new(vec![zero_block], (1, 1))?;
//!
//! let apply_a = |x: &BlockTensor<TensorDynLen>| Ok(x.clone());
//! let result = gmres(apply_a, &b, &x0, &GmresOptions::default())?;
//!
//! assert!(result.converged);
//! assert_eq!(result.solution.shape(), (1, 1));
//! # Ok(())
//! # }
//! ```

use std::collections::HashSet;

use crate::any_scalar::AnyScalar;
use crate::tensor_index::TensorIndex;
use crate::tensor_like::{
    DirectSumResult, FactorizeError, FactorizeOptions, FactorizeResult, LinearizationOrder,
    TensorConstructionLike, TensorContractionLike, TensorFactorizationLike, TensorLike,
    TensorVectorSpace, TensorVectorSpaceError,
};

/// A collection of tensors organized in a block structure.
///
/// Each block is a tensor of type `T` implementing [`TensorLike`].
/// The flattened block list is ordered row-by-row:
/// `(0, 0), (0, 1), ..., (1, 0), (1, 1), ...`.
///
/// # Type Parameters
///
/// * `T` - The tensor type for each block, must implement `TensorLike`
#[derive(Debug, Clone)]
pub struct BlockTensor<T: TensorLike> {
    /// Blocks flattened row-by-row in block-matrix order
    blocks: Vec<T>,
    /// Block structure (rows, cols)
    shape: (usize, usize),
}

impl<T: TensorLike> BlockTensor<T> {
    /// Create a new block tensor with validation.
    ///
    /// # Arguments
    ///
    /// * `blocks` - Vector of blocks flattened row-by-row
    /// * `shape` - Block structure as (rows, cols)
    ///
    /// # Errors
    ///
    /// Returns an error when the block count does not match the shape (a shape
    /// /// mismatch).
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::block_tensor::BlockTensor;
    /// use tensor4all_core::{DynIndex, TensorDynLen};
    ///
    /// let i = DynIndex::new_dyn(2);
    /// let t1 = TensorDynLen::from_dense(vec![i.clone()], vec![1.0, 2.0]).unwrap();
    /// let t2 = TensorDynLen::from_dense(vec![i.clone()], vec![3.0, 4.0]).unwrap();
    ///
    /// let bt = BlockTensor::try_new(vec![t1, t2], (2, 1)).unwrap();
    /// assert_eq!(bt.shape(), (2, 1));
    ///
    /// // Wrong number of blocks returns an error
    /// let t3 = TensorDynLen::from_dense(vec![i], vec![5.0, 6.0]).unwrap();
    /// assert!(BlockTensor::try_new(vec![t3], (2, 1)).is_err());
    /// ```
    pub fn try_new(
        blocks: Vec<T>,
        shape: (usize, usize),
    ) -> std::result::Result<Self, TensorVectorSpaceError> {
        let (rows, cols) = shape;
        if rows * cols != blocks.len() {
            return Err(TensorVectorSpaceError::from(anyhow::anyhow!(
                "Block count mismatch: shape ({}, {}) requires {} blocks, but got {}",
                rows,
                cols,
                rows * cols,
                blocks.len()
            )));
        }
        Ok(Self { blocks, shape })
    }

    /// Create a new block tensor.
    ///
    /// # Arguments
    ///
    /// * `blocks` - Vector of blocks flattened row-by-row
    /// * `shape` - Block structure as (rows, cols)
    ///
    /// # Errors
    ///
    /// Returns an error when the block count does not match the shape (a shape
    /// /// mismatch).
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::block_tensor::BlockTensor;
    /// use tensor4all_core::{DynIndex, TensorDynLen};
    ///
    /// let i = DynIndex::new_dyn(2);
    /// let t = TensorDynLen::from_dense(vec![i], vec![1.0, 2.0]).unwrap();
    /// let bt = BlockTensor::new(vec![t], (1, 1)).unwrap();
    /// assert_eq!(bt.shape(), (1, 1));
    /// ```
    pub fn new(
        blocks: Vec<T>,
        shape: (usize, usize),
    ) -> std::result::Result<Self, TensorVectorSpaceError> {
        Self::try_new(blocks, shape)
    }

    /// Get the block structure (rows, cols).
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::block_tensor::BlockTensor;
    /// use tensor4all_core::{DynIndex, TensorDynLen};
    ///
    /// let i = DynIndex::new_dyn(2);
    /// let blocks: Vec<TensorDynLen> = (0..6)
    ///     .map(|_| TensorDynLen::from_dense(vec![i.clone()], vec![0.0, 0.0]).unwrap())
    ///     .collect();
    /// let bt = BlockTensor::new(blocks, (2, 3)).unwrap();
    /// assert_eq!(bt.shape(), (2, 3));
    /// ```
    pub fn shape(&self) -> (usize, usize) {
        self.shape
    }

    /// Get the total number of blocks.
    pub fn num_blocks(&self) -> usize {
        self.blocks.len()
    }

    /// Get a reference to the block at (row, col).
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::block_tensor::BlockTensor;
    /// use tensor4all_core::{DynIndex, TensorDynLen};
    ///
    /// let i = DynIndex::new_dyn(2);
    /// let t1 = TensorDynLen::from_dense(vec![i.clone()], vec![1.0, 2.0]).unwrap();
    /// let t2 = TensorDynLen::from_dense(vec![i], vec![3.0, 4.0]).unwrap();
    /// let bt = BlockTensor::new(vec![t1, t2], (2, 1)).unwrap();
    /// assert_eq!(bt.get(0, 0).unwrap().dims(), vec![2]);
    /// assert_eq!(bt.get(1, 0).unwrap().dims(), vec![2]);
    /// ```
    pub fn get(&self, row: usize, col: usize) -> Option<&T> {
        let (rows, cols) = self.shape;
        if row >= rows || col >= cols {
            return None;
        }
        self.blocks.get(row * cols + col)
    }

    /// Get a mutable reference to the block at (row, col).
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::block_tensor::BlockTensor;
    /// use tensor4all_core::{DynIndex, TensorDynLen};
    ///
    /// let i = DynIndex::new_dyn(2);
    /// let t = TensorDynLen::from_dense(vec![i.clone()], vec![1.0, 2.0]).unwrap();
    /// let mut bt = BlockTensor::new(vec![t], (1, 1)).unwrap();
    /// let block = bt.get_mut(0, 0).unwrap();
    /// assert_eq!(block.dims(), vec![2]);
    /// ```
    pub fn get_mut(&mut self, row: usize, col: usize) -> Option<&mut T> {
        let (rows, cols) = self.shape;
        if row >= rows || col >= cols {
            return None;
        }
        self.blocks.get_mut(row * cols + col)
    }

    /// Get all blocks as a slice.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::block_tensor::BlockTensor;
    /// use tensor4all_core::{DynIndex, TensorDynLen};
    ///
    /// let i = DynIndex::new_dyn(2);
    /// let t1 = TensorDynLen::from_dense(vec![i.clone()], vec![1.0, 2.0]).unwrap();
    /// let t2 = TensorDynLen::from_dense(vec![i], vec![3.0, 4.0]).unwrap();
    /// let bt = BlockTensor::new(vec![t1, t2], (1, 2)).unwrap();
    /// assert_eq!(bt.blocks().len(), 2);
    /// ```
    pub fn blocks(&self) -> &[T] {
        &self.blocks
    }

    /// Get all blocks as a mutable slice.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::block_tensor::BlockTensor;
    /// use tensor4all_core::{DynIndex, TensorDynLen};
    ///
    /// let i = DynIndex::new_dyn(2);
    /// let t = TensorDynLen::from_dense(vec![i], vec![1.0, 2.0]).unwrap();
    /// let mut bt = BlockTensor::new(vec![t], (1, 1)).unwrap();
    /// assert_eq!(bt.blocks_mut().len(), 1);
    /// ```
    pub fn blocks_mut(&mut self) -> &mut [T] {
        &mut self.blocks
    }

    /// Consume self and return the blocks.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::block_tensor::BlockTensor;
    /// use tensor4all_core::{DynIndex, TensorDynLen};
    ///
    /// let i = DynIndex::new_dyn(2);
    /// let t = TensorDynLen::from_dense(vec![i], vec![1.0, 2.0]).unwrap();
    /// let bt = BlockTensor::new(vec![t], (1, 1)).unwrap();
    /// let blocks = bt.into_blocks();
    /// assert_eq!(blocks.len(), 1);
    /// assert_eq!(blocks[0].dims(), vec![2]);
    /// ```
    pub fn into_blocks(self) -> Vec<T> {
        self.blocks
    }

    /// Validate that blocks share external indices consistently.
    ///
    /// For column vectors (cols=1), no index sharing is required between
    /// blocks. Different rows can have independent physical indices
    /// (the operator determines their relationship).
    ///
    /// For matrices (rows x cols), checks that:
    /// - All blocks have the same number of external indices.
    /// - Blocks in the same row share some common index IDs (output indices).
    /// - Blocks in the same column share some common index IDs (input indices).
    ///
    /// # Errors
    ///
    /// Returns an error when the blocks have inconsistent indices (a shape or
    /// /// index mismatch).
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::block_tensor::BlockTensor;
    /// use tensor4all_core::{DynIndex, TensorDynLen};
    ///
    /// // Column vector: always valid regardless of index sharing
    /// let i = DynIndex::new_dyn(2);
    /// let j = DynIndex::new_dyn(2);
    /// let t1 = TensorDynLen::from_dense(vec![i], vec![1.0, 2.0]).unwrap();
    /// let t2 = TensorDynLen::from_dense(vec![j], vec![3.0, 4.0]).unwrap();
    /// let bt = BlockTensor::new(vec![t1, t2], (2, 1)).unwrap();
    /// assert!(bt.validate_indices().is_ok());
    /// ```
    pub fn validate_indices(&self) -> std::result::Result<(), TensorVectorSpaceError> {
        let (rows, cols) = self.shape;

        if cols <= 1 {
            // Column vector: blocks in different rows can have independent indices.
            // The operator determines the relationship between blocks.
            return Ok(());
        }

        // Matrix: check all blocks have the same number of external indices
        let first_count = self.blocks[0].num_external_indices();
        for (i, block) in self.blocks.iter().enumerate().skip(1) {
            let n = block.num_external_indices();
            if n != first_count {
                return Err(TensorVectorSpaceError::from(anyhow::anyhow!(
                    "Block {} has {} external indices, but block 0 has {}",
                    i,
                    n,
                    first_count
                )));
            }
        }

        // Same row: blocks should share at least one full output index.
        for row in 0..rows {
            let ref_indices: HashSet<_> = self
                .get(row, 0)
                .ok_or_else(|| anyhow::anyhow!("block index ({row}, 0) is out of bounds"))?
                .external_indices()
                .iter()
                .cloned()
                .collect();
            for col in 1..cols {
                let indices: HashSet<_> = self
                    .get(row, col)
                    .ok_or_else(|| anyhow::anyhow!("block index ({row}, {col}) is out of bounds"))?
                    .external_indices()
                    .iter()
                    .cloned()
                    .collect();
                let common_count = ref_indices.intersection(&indices).count();
                if common_count == 0 {
                    return Err(TensorVectorSpaceError::from(anyhow::anyhow!(
                        "Matrix row {}: blocks ({},{}) and ({},{}) share no common indices",
                        row,
                        row,
                        0,
                        row,
                        col
                    )));
                }
            }
        }

        // Same column: blocks should share at least one full input index.
        for col in 0..cols {
            let ref_indices: HashSet<_> = self
                .get(0, col)
                .ok_or_else(|| anyhow::anyhow!("block index (0, {col}) is out of bounds"))?
                .external_indices()
                .iter()
                .cloned()
                .collect();
            for row in 1..rows {
                let indices: HashSet<_> = self
                    .get(row, col)
                    .ok_or_else(|| anyhow::anyhow!("block index ({row}, {col}) is out of bounds"))?
                    .external_indices()
                    .iter()
                    .cloned()
                    .collect();
                let common_count = ref_indices.intersection(&indices).count();
                if common_count == 0 {
                    return Err(TensorVectorSpaceError::from(anyhow::anyhow!(
                        "Matrix col {}: blocks ({},{}) and ({},{}) share no common indices",
                        col,
                        0,
                        col,
                        row,
                        col
                    )));
                }
            }
        }

        Ok(())
    }
}

// ============================================================================
// TensorIndex implementation
// ============================================================================

impl<T: TensorLike> TensorIndex for BlockTensor<T> {
    type Index = T::Index;
    type Error = TensorVectorSpaceError;

    fn external_indices(&self) -> Vec<Self::Index> {
        // Collect unique external indices across all blocks (deduplicated by full index).
        let mut seen = HashSet::new();
        let mut result = Vec::new();
        for block in &self.blocks {
            for idx in block.external_indices() {
                if seen.insert(idx.clone()) {
                    result.push(idx);
                }
            }
        }
        result
    }

    fn replaceind(
        &self,
        old_index: &Self::Index,
        new_index: &Self::Index,
    ) -> std::result::Result<Self, Self::Error> {
        let replaced: std::result::Result<Vec<T>, Self::Error> = self
            .blocks
            .iter()
            .map(|b| {
                b.replaceind(old_index, new_index)
                    .map_err(|error| TensorVectorSpaceError::from(anyhow::Error::new(error)))
            })
            .collect();
        Ok(Self {
            blocks: replaced?,
            shape: self.shape,
        })
    }

    fn replace_indices(
        &self,
        old_indices: &[Self::Index],
        new_indices: &[Self::Index],
    ) -> std::result::Result<Self, Self::Error> {
        let replaced: std::result::Result<Vec<T>, Self::Error> = self
            .blocks
            .iter()
            .map(|b| {
                b.replace_indices(old_indices, new_indices)
                    .map_err(|error| TensorVectorSpaceError::from(anyhow::Error::new(error)))
            })
            .collect();
        Ok(Self {
            blocks: replaced?,
            shape: self.shape,
        })
    }
}

// ============================================================================
// TensorLike implementation
// ============================================================================

impl<T: TensorLike> TensorVectorSpace for BlockTensor<T> {
    // ------------------------------------------------------------------------
    // Vector space operations (required for GMRES)
    // ------------------------------------------------------------------------

    fn norm_squared(&self) -> std::result::Result<f64, Self::Error> {
        self.blocks.iter().try_fold(0.0, |sum, block| {
            let value = block
                .norm_squared()
                .map_err(|error| TensorVectorSpaceError::from(anyhow::Error::new(error)))?;
            Ok(sum + value)
        })
    }

    fn maxabs(&self) -> std::result::Result<f64, Self::Error> {
        self.blocks.iter().try_fold(0.0_f64, |acc, block| {
            let value = block
                .maxabs()
                .map_err(|error| TensorVectorSpaceError::from(anyhow::Error::new(error)))?;
            Ok(acc.max(value))
        })
    }

    fn scale(&self, scalar: AnyScalar) -> std::result::Result<Self, Self::Error> {
        let scaled: std::result::Result<Vec<T>, Self::Error> = self
            .blocks
            .iter()
            .map(|b| {
                b.scale(scalar.clone())
                    .map_err(|error| TensorVectorSpaceError::from(anyhow::Error::new(error)))
            })
            .collect();
        Ok(Self {
            blocks: scaled?,
            shape: self.shape,
        })
    }

    fn axpby(
        &self,
        a: AnyScalar,
        other: &Self,
        b: AnyScalar,
    ) -> std::result::Result<Self, Self::Error> {
        if self.shape != other.shape {
            return Err(TensorVectorSpaceError::from(anyhow::anyhow!(
                "Block shapes must match: {:?} vs {:?}",
                self.shape,
                other.shape
            )));
        }
        let result: std::result::Result<Vec<T>, Self::Error> = self
            .blocks
            .iter()
            .zip(other.blocks.iter())
            .map(|(s, o)| {
                s.axpby(a.clone(), o, b.clone())
                    .map_err(|error| TensorVectorSpaceError::from(anyhow::Error::new(error)))
            })
            .collect();
        Ok(Self {
            blocks: result?,
            shape: self.shape,
        })
    }

    fn inner_product(&self, other: &Self) -> std::result::Result<AnyScalar, Self::Error> {
        if self.shape != other.shape {
            return Err(TensorVectorSpaceError::from(anyhow::anyhow!(
                "Block shapes must match for inner product: {:?} vs {:?}",
                self.shape,
                other.shape
            )));
        }
        let mut sum = AnyScalar::new_real(0.0);
        for (s, o) in self.blocks.iter().zip(other.blocks.iter()) {
            let value = s
                .inner_product(o)
                .map_err(|error| TensorVectorSpaceError::from(anyhow::Error::new(error)))?;
            sum = sum + value;
        }
        Ok(sum)
    }

    fn validate(&self) -> std::result::Result<(), Self::Error> {
        self.validate_indices()
    }
}

impl<T: TensorLike> TensorContractionLike for BlockTensor<T> {
    // ------------------------------------------------------------------------
    // Tensor network operations
    // ------------------------------------------------------------------------

    fn conj(&self) -> Self {
        let conjugated: Vec<T> = self.blocks.iter().map(|b| b.conj()).collect();
        Self {
            blocks: conjugated,
            shape: self.shape,
        }
    }

    fn direct_sum(
        &self,
        _other: &Self,
        _pairs: &[(<Self as TensorIndex>::Index, <Self as TensorIndex>::Index)],
    ) -> std::result::Result<DirectSumResult<Self>, Self::Error> {
        Err(TensorVectorSpaceError::from(anyhow::anyhow!(
            "BlockTensor does not support direct_sum"
        )))
    }

    fn outer_product(&self, _other: &Self) -> std::result::Result<Self, Self::Error> {
        Err(TensorVectorSpaceError::from(anyhow::anyhow!(
            "BlockTensor does not support outer_product"
        )))
    }

    fn permuteinds(
        &self,
        _new_order: &[<Self as TensorIndex>::Index],
    ) -> std::result::Result<Self, Self::Error> {
        Err(TensorVectorSpaceError::from(anyhow::anyhow!(
            "BlockTensor does not support permuteinds"
        )))
    }

    fn fuse_indices(
        &self,
        old_indices: &[Self::Index],
        new_index: Self::Index,
        order: LinearizationOrder,
    ) -> std::result::Result<Self, Self::Error> {
        let blocks: std::result::Result<Vec<T>, Self::Error> = self
            .blocks
            .iter()
            .map(|block| {
                block
                    .fuse_indices(old_indices, new_index.clone(), order)
                    .map_err(|error| TensorVectorSpaceError::from(anyhow::Error::new(error)))
            })
            .collect();
        Ok(Self {
            blocks: blocks?,
            shape: self.shape,
        })
    }

    fn contract(_tensors: &[&Self]) -> std::result::Result<Self, Self::Error> {
        Err(TensorVectorSpaceError::from(anyhow::anyhow!(
            "BlockTensor does not support contract"
        )))
    }
}

impl<T: TensorLike> TensorFactorizationLike for BlockTensor<T> {
    fn factorize(
        &self,
        _left_inds: &[<Self as TensorIndex>::Index],
        _options: &FactorizeOptions,
    ) -> std::result::Result<FactorizeResult<Self>, FactorizeError> {
        Err(FactorizeError::ComputationError(anyhow::anyhow!(
            "BlockTensor does not support factorize"
        )))
    }

    fn factorize_full_rank(
        &self,
        _left_inds: &[<Self as TensorIndex>::Index],
        _alg: crate::FactorizeAlg,
        _canonical: crate::Canonical,
    ) -> std::result::Result<FactorizeResult<Self>, FactorizeError> {
        Err(FactorizeError::ComputationError(anyhow::anyhow!(
            "BlockTensor does not support factorize_full_rank"
        )))
    }
}

impl<T: TensorLike> TensorConstructionLike for BlockTensor<T> {
    fn diagonal(
        _input_index: &<Self as TensorIndex>::Index,
        _output_index: &<Self as TensorIndex>::Index,
    ) -> std::result::Result<Self, Self::Error> {
        Err(TensorVectorSpaceError::from(anyhow::anyhow!(
            "BlockTensor does not support diagonal"
        )))
    }

    fn scalar_one() -> std::result::Result<Self, Self::Error> {
        Err(TensorVectorSpaceError::from(anyhow::anyhow!(
            "BlockTensor does not support scalar_one"
        )))
    }

    fn ones(_indices: &[<Self as TensorIndex>::Index]) -> std::result::Result<Self, Self::Error> {
        Err(TensorVectorSpaceError::from(anyhow::anyhow!(
            "BlockTensor does not support ones"
        )))
    }

    fn onehot(
        _index_vals: &[(<Self as TensorIndex>::Index, usize)],
    ) -> std::result::Result<Self, Self::Error> {
        Err(TensorVectorSpaceError::from(anyhow::anyhow!(
            "BlockTensor does not support onehot"
        )))
    }
}

#[cfg(test)]
mod tests;
