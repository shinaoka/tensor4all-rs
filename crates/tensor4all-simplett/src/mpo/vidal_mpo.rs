//! VidalMPO - Vidal canonical form of MPO
//!
//! A VidalMPO stores tensors and singular values separately,
//! which is useful for TEBD-like algorithms.

use super::error::{MPOError, Result};
use super::mpo::MPO;
use super::types::{Tensor4, Tensor4Ops};
use crate::traits::TTScalar;

/// Vidal canonical form of MPO
///
/// The Vidal form represents the MPO as:
/// O = Gamma\[1\] * Lambda\[1\] * Gamma\[2\] * Lambda\[2\] * ... * Gamma\[L\]
///
/// where Gamma\[i\] are tensors and Lambda\[i\] are diagonal matrices of singular values.
#[derive(Debug, Clone)]
pub struct VidalMPO<T: TTScalar> {
    /// The Gamma tensors (4D tensors with normalized bond indices)
    gammas: Vec<Tensor4<T>>,
    /// The Lambda vectors (singular values on each bond)
    lambdas: Vec<Vec<f64>>,
}

impl<T: TTScalar> VidalMPO<T> {
    /// Create a VidalMPO from an MPO
    /// # Errors
    ///
    /// Returns an error when the MPO cannot be converted (a shape or index
    /// /// mismatch, or a backend failure).
    ///
    pub fn from_mpo(_mpo: MPO<T>) -> Result<Self> {
        // TODO: Implement SVD-based conversion to Vidal form
        Err(MPOError::InvalidOperation {
            message: "VidalMPO::from_mpo not yet implemented".to_string(),
        })
    }

    /// Create a VidalMPO from tensors and lambdas without validation
    #[allow(dead_code)]
    pub(crate) fn from_parts_unchecked(gammas: Vec<Tensor4<T>>, lambdas: Vec<Vec<f64>>) -> Self {
        Self { gammas, lambdas }
    }

    /// Get the number of sites
    pub fn len(&self) -> usize {
        self.gammas.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.gammas.is_empty()
    }

    /// Get the Gamma tensor at position i
    pub fn gamma(&self, i: usize) -> &Tensor4<T> {
        &self.gammas[i]
    }

    /// Get mutable reference to the Gamma tensor at position i
    pub fn gamma_mut(&mut self, i: usize) -> &mut Tensor4<T> {
        &mut self.gammas[i]
    }

    /// Get all Gamma tensors
    pub fn gammas(&self) -> &[Tensor4<T>] {
        &self.gammas
    }

    /// Get the Lambda vector at bond i (between sites i and i+1)
    pub fn lambda(&self, i: usize) -> &[f64] {
        &self.lambdas[i]
    }

    /// Get all Lambda vectors
    pub fn lambdas(&self) -> &[Vec<f64>] {
        &self.lambdas
    }

    /// Bond dimensions
    pub fn link_dims(&self) -> Vec<usize> {
        self.lambdas.iter().map(|l| l.len()).collect()
    }

    /// Site dimensions
    pub fn site_dims(&self) -> Vec<(usize, usize)> {
        self.gammas
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
    /// Returns an error when the conversion fails (a shape or index mismatch,
    /// /// or a backend failure).
    ///
    pub fn into_mpo(self) -> Result<MPO<T>> {
        // TODO: Implement conversion back to MPO
        Err(MPOError::InvalidOperation {
            message: "VidalMPO::into_mpo not yet implemented".to_string(),
        })
    }
}

#[cfg(test)]
mod tests;
