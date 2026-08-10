//! SiteMPO - Center-canonical form of MPO
//!
//! A SiteMPO maintains an orthogonality center that can be moved
//! through the tensor train for efficient local updates.

use super::error::{MPOError, Result};
use super::mpo::MPO;
use super::types::{Tensor4, Tensor4Ops};
use crate::traits::TTScalar;

/// Center-canonical form of MPO with orthogonality center
///
/// The tensors to the left of the center are left-orthogonal (left-isometric),
/// and the tensors to the right are right-orthogonal (right-isometric).
#[derive(Debug, Clone)]
pub struct SiteMPO<T: TTScalar> {
    /// The underlying MPO tensors
    tensors: Vec<Tensor4<T>>,
    /// Current orthogonality center position (0-indexed)
    center: usize,
}

impl<T: TTScalar> SiteMPO<T> {
    /// Create a SiteMPO from an MPO, placing the center at the given position
    /// # Errors
    ///
    /// Returns an error when the MPO cannot be converted (a shape or index
    /// /// mismatch, or a backend failure).
    ///
    pub fn from_mpo(mpo: MPO<T>, center: usize) -> Result<Self> {
        if mpo.is_empty() {
            return Err(MPOError::Empty);
        }
        if center >= mpo.len() {
            return Err(MPOError::InvalidCenter {
                center,
                max: mpo.len(),
            });
        }

        let tensors = mpo.site_tensors().to_vec();
        let mut result = Self { tensors, center: 0 };

        // Move center to the desired position
        result.set_center(center)?;

        Ok(result)
    }

    /// Create a SiteMPO from tensors without validation
    #[allow(dead_code)]
    pub(crate) fn from_tensors_unchecked(tensors: Vec<Tensor4<T>>, center: usize) -> Self {
        Self { tensors, center }
    }

    /// Get the current orthogonality center position
    pub fn center(&self) -> usize {
        self.center
    }

    /// Get the number of sites
    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    /// Get the site tensor at position i
    pub fn site_tensor(&self, i: usize) -> &Tensor4<T> {
        &self.tensors[i]
    }

    /// Get mutable reference to the site tensor at position i
    pub fn site_tensor_mut(&mut self, i: usize) -> &mut Tensor4<T> {
        &mut self.tensors[i]
    }

    /// Get all site tensors
    pub fn site_tensors(&self) -> &[Tensor4<T>] {
        &self.tensors
    }

    /// Get mutable access to site tensors
    pub fn site_tensors_mut(&mut self) -> &mut [Tensor4<T>] {
        &mut self.tensors
    }

    /// Bond dimensions
    pub fn link_dims(&self) -> Vec<usize> {
        if self.len() <= 1 {
            return Vec::new();
        }
        (1..self.len())
            .map(|i| self.tensors[i].left_dim())
            .collect()
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

    /// Move the orthogonality center one position to the left
    /// # Errors
    ///
    /// Returns an error when the center cannot be moved (a shape or index
    /// /// mismatch, or a backend failure).
    ///
    pub fn move_center_left(&mut self) -> Result<()> {
        if self.center == 0 {
            return Err(MPOError::InvalidOperation {
                message: "Cannot move center left from position 0".to_string(),
            });
        }

        // TODO: Implement QR decomposition to move center left
        // For now, just update the center position
        self.center -= 1;
        Ok(())
    }

    /// Move the orthogonality center one position to the right
    /// # Errors
    ///
    /// Returns an error when the center cannot be moved (a shape or index
    /// /// mismatch, or a backend failure).
    ///
    pub fn move_center_right(&mut self) -> Result<()> {
        if self.center >= self.len() - 1 {
            return Err(MPOError::InvalidOperation {
                message: "Cannot move center right from last position".to_string(),
            });
        }

        // TODO: Implement QR decomposition to move center right
        // For now, just update the center position
        self.center += 1;
        Ok(())
    }

    /// Move the orthogonality center to the specified position
    /// # Errors
    ///
    /// Returns an error when the site is out of bounds or the center cannot be
    /// /// set (a shape or index mismatch failure).
    ///
    pub fn set_center(&mut self, target: usize) -> Result<()> {
        if target >= self.len() {
            return Err(MPOError::InvalidCenter {
                center: target,
                max: self.len(),
            });
        }

        while self.center < target {
            self.move_center_right()?;
        }
        while self.center > target {
            self.move_center_left()?;
        }

        Ok(())
    }

    /// Convert to basic MPO
    pub fn into_mpo(self) -> MPO<T> {
        MPO::from_tensors_unchecked(self.tensors)
    }
}

#[cfg(test)]
mod tests;
