//! Contraction operations for tensor trains.
//!
//! This module provides both a free function [`contract`] and an impl method
//! [`TensorTrain::contract`] for contracting two tensor trains.

use tensor4all_treetn::treetn::contraction::{
    contract as treetn_contract, ContractionMethod, ContractionOptions as TreeTNContractionOptions,
};
use tensor4all_treetn::CanonicalForm;

use crate::error::{Result, TensorTrainError};
use crate::options::{validate_svd_truncation_options, ContractMethod, ContractOptions};
use crate::tensortrain::TensorTrain;

/// Contract two tensor trains, returning a new tensor train.
///
/// This performs element-wise contraction of corresponding sites,
/// similar to MPO-MPO contraction in ITensor.
///
/// # Arguments
/// * `a` - The first tensor train
/// * `b` - The second tensor train
/// * `options` - Contraction options (method, max_rank, rtol, nhalfsweeps)
///
/// # Returns
/// A new tensor train resulting from the contraction.
///
/// # Errors
///
/// Returns an error when the contraction or operation fails (a shape or
/// /// index mismatch, or a backend failure).
///
pub fn contract(
    a: &TensorTrain,
    b: &TensorTrain,
    options: &ContractOptions,
) -> Result<TensorTrain> {
    if a.is_empty() || b.is_empty() {
        return Err(TensorTrainError::InvalidStructure {
            message: "Cannot contract empty tensor trains".to_string(),
        });
    }

    if a.len() != b.len() {
        return Err(TensorTrainError::InvalidStructure {
            message: format!(
                "Tensor trains must have the same length for contraction: {} vs {}",
                a.len(),
                b.len()
            ),
        });
    }

    validate_svd_truncation_options(options.max_rank(), options.svd_policy())?;

    if matches!(options.method(), ContractMethod::Fit) && !options.nhalfsweeps().is_multiple_of(2) {
        return Err(TensorTrainError::OperationError {
            message: format!(
                "nhalfsweeps must be a multiple of 2 for Fit method, got {}",
                options.nhalfsweeps()
            ),
        });
    }

    // Convert ContractOptions to TreeTN ContractionOptions
    let treetn_method = match options.method() {
        ContractMethod::Zipup => ContractionMethod::Zipup,
        ContractMethod::Fit => ContractionMethod::Fit,
        ContractMethod::Naive => ContractionMethod::Naive,
    };

    // Convert nhalfsweeps to nfullsweeps (nhalfsweeps / 2)
    let nfullsweeps = options.nhalfsweeps() / 2;
    let treetn_options = TreeTNContractionOptions::new(treetn_method).with_nfullsweeps(nfullsweeps);

    let treetn_options = if let Some(max_rank) = options.max_rank() {
        treetn_options.with_max_rank(max_rank)
    } else {
        treetn_options
    };

    let treetn_options = if let Some(policy) = options.svd_policy() {
        treetn_options.with_svd_policy(policy)
    } else {
        treetn_options
    };
    let treetn_options = if let Some(limit) = options.dense_reference_limit() {
        treetn_options.with_dense_reference_limit(limit)
    } else {
        treetn_options
    };

    // Use the last site as the canonical center (consistent with existing behavior)
    let center = a.len() - 1;

    let result_inner = if matches!(options.method(), ContractMethod::Zipup) {
        a.as_treetn()
            .contract_zipup_with(
                b.as_treetn(),
                &center,
                CanonicalForm::Unitary,
                options.svd_policy(),
                options.max_rank(),
            )
            .map_err(|e| TensorTrainError::InvalidStructure {
                message: format!("Zip-up contraction failed: {}", e),
            })?
    } else {
        treetn_contract(a.as_treetn(), b.as_treetn(), &center, treetn_options).map_err(|e| {
            TensorTrainError::InvalidStructure {
                message: format!("TreeTN contraction failed: {}", e),
            }
        })?
    };

    TensorTrain::from_inner(result_inner, Some(CanonicalForm::Unitary))
}

impl TensorTrain {
    /// Contract two tensor trains, returning a new tensor train.
    ///
    /// This performs element-wise contraction of corresponding sites,
    /// similar to MPO-MPO contraction in ITensor.
    ///
    /// # Arguments
    /// * `other` - The other tensor train to contract with
    /// * `options` - Contraction options (method, max_rank, rtol, nhalfsweeps)
    ///
    /// # Returns
    /// A new tensor train resulting from the contraction.
    ///
    /// # Errors
    ///
    /// Returns an error when the contraction or operation fails (a shape or
    /// /// index mismatch, or a backend failure).
    ///
    pub fn contract(&self, other: &Self, options: &ContractOptions) -> Result<Self> {
        contract(self, other, options)
    }

    /// Contract two tensor trains with explicit contraction options.
    ///
    /// This is an alias for [`TensorTrain::contract`]. It exists for callers
    /// that use `contract_pair` to mean pairwise tensor-train contraction with
    /// compression, rather than the dense tensor-level pair contraction.
    ///
    /// # Errors
    ///
    /// Returns an error when the pair is incompatible (a shape or index mismatch)
    /// /// or the contraction fails (a backend failure).
    ///
    pub fn contract_pair(&self, other: &Self, options: &ContractOptions) -> Result<Self> {
        self.contract(other, options)
    }
}

#[cfg(test)]
mod tests;
