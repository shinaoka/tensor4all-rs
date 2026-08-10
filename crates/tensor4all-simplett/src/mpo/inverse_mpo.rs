//! InverseMPO - Inverse form of MPO
//!
//! An InverseMPO stores inverse singular values for efficient local updates.

use super::error::{MPOError, Result};
use super::mpo::MPO;
use super::types::{Tensor4, Tensor4Ops};
use crate::traits::TTScalar;

/// Inverse form of MPO
///
/// The inverse form is similar to Vidal form but stores inverse singular values,
/// which is efficient for local update operations.
#[derive(Debug, Clone)]
pub struct InverseMPO<T: TTScalar> {
    /// The tensors (4D tensors with normalized bond indices)
    tensors: Vec<Tensor4<T>>,
    /// The inverse singular values on each bond
    inv_lambdas: Vec<Vec<f64>>,
}

impl<T: TTScalar> InverseMPO<T> {
    /// Create an InverseMPO from an MPO
    /// # Errors
    ///
    /// Returns an error when the construction or conversion fails (a shape or
    /// /// index mismatch, or a backend failure).
    ///
    pub fn from_mpo(_mpo: MPO<T>) -> Result<Self> {
        // TODO: Implement conversion to inverse form
        Err(MPOError::InvalidOperation {
            message: "InverseMPO::from_mpo not yet implemented".to_string(),
        })
    }

    /// Create an InverseMPO from parts without validation
    #[allow(dead_code)]
    pub(crate) fn from_parts_unchecked(
        tensors: Vec<Tensor4<T>>,
        inv_lambdas: Vec<Vec<f64>>,
    ) -> Self {
        Self {
            tensors,
            inv_lambdas,
        }
    }

    /// Get the number of sites
    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    /// Get the tensor at position i
    pub fn site_tensor(&self, i: usize) -> &Tensor4<T> {
        &self.tensors[i]
    }

    /// Get mutable reference to the tensor at position i
    pub fn site_tensor_mut(&mut self, i: usize) -> &mut Tensor4<T> {
        &mut self.tensors[i]
    }

    /// Get all tensors
    pub fn site_tensors(&self) -> &[Tensor4<T>] {
        &self.tensors
    }

    /// Get the inverse Lambda vector at bond i (between sites i and i+1)
    pub fn inv_lambda(&self, i: usize) -> &[f64] {
        &self.inv_lambdas[i]
    }

    /// Get all inverse Lambda vectors
    pub fn inv_lambdas(&self) -> &[Vec<f64>] {
        &self.inv_lambdas
    }

    /// Bond dimensions
    pub fn link_dims(&self) -> Vec<usize> {
        self.inv_lambdas.iter().map(|l| l.len()).collect()
    }

    /// Site dimensions
    pub fn site_dims(&self) -> Vec<(usize, usize)> {
        self.tensors
            .iter()
            .map(|t| (t.site_dim_1(), t.site_dim_2()))
            .collect()
    }

    /// Maximum bond dimension
    pub fn rank(&self) -> usize {
        let lds = self.link_dims();
        if lds.is_empty() {
            1
        } else {
            *lds.iter().max().unwrap_or(&1)
        }
    }

    /// Convert to basic MPO
    /// # Errors
    ///
    /// Returns an error when the construction or conversion fails (a shape or
    /// /// index mismatch, or a backend failure).
    ///
    pub fn into_mpo(self) -> Result<MPO<T>> {
        // TODO: Implement conversion back to MPO
        Err(MPOError::InvalidOperation {
            message: "InverseMPO::into_mpo not yet implemented".to_string(),
        })
    }
}

#[cfg(test)]
mod tests;
