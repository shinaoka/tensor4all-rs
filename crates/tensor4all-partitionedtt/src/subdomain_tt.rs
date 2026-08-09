//! SubDomainTT: A tensor train with an associated projector
//!
//! A `SubDomainTT` represents a tensor train whose values are only valid
//! within a specific subdomain defined by a projector.

use std::collections::HashSet;

use crate::error::{PartitionedTTError, Result};
use crate::projector::Projector;
use tensor4all_core::{AnyScalar, DynIndex, TensorDynLen, TensorStorageError};
use tensor4all_itensorlike::{ContractOptions, TensorTrain, TruncateOptions};

/// A tensor train with an associated projector defining its subdomain.
///
/// The projector specifies which indices are fixed to specific values.
/// The tensor train values are only valid within this projected subdomain.
///
/// # Examples
///
/// ```
/// use tensor4all_partitionedtt::{DynIndex, Projector, SubDomainTT, TensorDynLen, TensorTrain};
///
/// let site0 = DynIndex::new_dyn(2);
/// let bond = DynIndex::new_dyn(1);
/// let site1 = DynIndex::new_dyn(2);
///
/// let t0 = TensorDynLen::from_dense(vec![site0.clone(), bond.clone()], vec![1.0, 2.0]).unwrap();
/// let t1 = TensorDynLen::from_dense(vec![bond.clone(), site1.clone()], vec![3.0, 4.0]).unwrap();
/// let tt = TensorTrain::new(vec![t0, t1]).unwrap();
///
/// let projector = Projector::from_pairs([(site0.clone(), 1)]);
/// let subdomain_tt = SubDomainTT::new(tt, projector);
///
/// assert_eq!(subdomain_tt.len(), 2);
/// assert_eq!(subdomain_tt.projector().get(&site0), Some(1));
/// assert_eq!(subdomain_tt.projector().get(&site1), None);
/// ```
#[derive(Debug, Clone)]
pub struct SubDomainTT {
    /// The underlying tensor train
    data: TensorTrain,
    /// The projector defining the subdomain
    projector: Projector,
    /// Absolute squared truncation budget assigned by adaptive patching.
    budget_squared: Option<f64>,
}

impl SubDomainTT {
    /// Create a new SubDomainTT from a tensor train and projector.
    ///
    /// The projector is trimmed to only include indices that exist in the tensor train.
    pub fn new(data: TensorTrain, projector: Projector) -> Self {
        // Trim projector to only include valid indices
        let all_indices = Self::collect_all_indices(&data);
        let trimmed_projector = projector.filter_indices(&all_indices);
        Self {
            data,
            projector: trimmed_projector,
            budget_squared: None,
        }
    }

    /// Create a SubDomainTT from a tensor train with an empty projector.
    pub fn from_tt(data: TensorTrain) -> Self {
        Self {
            data,
            projector: Projector::new(),
            budget_squared: None,
        }
    }

    /// Collect all site indices from the tensor train.
    fn collect_all_indices(tt: &TensorTrain) -> Vec<DynIndex> {
        tt.siteinds().into_iter().flatten().collect()
    }

    /// Get all site indices (flattened).
    pub fn all_indices(&self) -> Vec<DynIndex> {
        Self::collect_all_indices(&self.data)
    }

    /// Get a reference to the underlying tensor train.
    pub fn data(&self) -> &TensorTrain {
        &self.data
    }

    /// Get a mutable reference to the underlying tensor train.
    pub fn data_mut(&mut self) -> &mut TensorTrain {
        &mut self.data
    }

    /// Get a reference to the projector.
    pub fn projector(&self) -> &Projector {
        &self.projector
    }

    pub(crate) fn budget_squared(&self) -> Option<f64> {
        self.budget_squared
    }

    pub(crate) fn with_budget_squared(mut self, budget_squared: f64) -> Self {
        self.budget_squared = Some(budget_squared);
        self
    }

    /// Convert to the underlying tensor train, consuming self.
    pub fn into_data(self) -> TensorTrain {
        self.data
    }

    /// Get the number of sites.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if the tensor train is empty.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Get the maximum bond dimension.
    pub fn max_bond_dim(&self) -> usize {
        self.data.maxbonddim()
    }

    /// Get the site indices (nested per site).
    pub fn siteinds(&self) -> Vec<Vec<DynIndex>> {
        self.data.siteinds()
    }

    /// Check if an index is projected.
    pub fn is_projected_at(&self, index: &DynIndex) -> bool {
        self.projector.is_projected_at(index)
    }

    /// Project to a more restrictive projector.
    ///
    /// Returns `Ok(None)` if the projectors are incompatible (conflicting
    /// values). The resulting SubDomainTT has tensor values zeroed out where
    /// the projection does not match, while retaining tensor indices and
    /// backend autodiff metadata.
    ///
    /// # Errors
    ///
    /// Returns [`PartitionedTTError::ProjectorIndexNotFound`] for a projector
    /// index absent from the tensor train,
    /// [`PartitionedTTError::ProjectorCoordinateOutOfBounds`] for an invalid
    /// coordinate, [`PartitionedTTError::TensorStorage`] for deferred or
    /// failed storage materialization, [`PartitionedTTError::TensorConstruction`]
    /// for a backend construction failure, or
    /// [`PartitionedTTError::TensorTrain`] for invalid tensor-train structure.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_partitionedtt::{DynIndex, Projector, SubDomainTT, TensorDynLen, TensorTrain};
    ///
    /// let site = DynIndex::new_dyn(2);
    /// let tensor = TensorDynLen::from_dense(vec![site.clone()], vec![3.0_f64, 4.0]).unwrap();
    /// let subdomain = SubDomainTT::from_tt(TensorTrain::new(vec![tensor]).unwrap());
    /// let projected = subdomain
    ///     .project(&Projector::from_pairs([(site.clone(), 1)]))
    ///     .unwrap()
    ///     .unwrap();
    ///
    /// assert_eq!(projected.projector().get(&site), Some(1));
    /// assert_eq!(projected.data().tensor(0).unwrap().to_vec::<f64>().unwrap(), vec![0.0, 4.0]);
    /// ```
    pub fn project(&self, projector: &Projector) -> Result<Option<Self>> {
        self.validate_projector(projector)?;

        // Check if projectors are compatible.
        if !self.projector.is_compatible_with(projector) {
            return Ok(None);
        }

        let merged_projector = self
            .projector
            .intersection(projector)
            .ok_or(PartitionedTTError::ProjectorConflict)?;
        let projected_data = self.project_tensor_data(projector)?;

        Ok(Some(Self {
            data: projected_data,
            projector: merged_projector,
            budget_squared: self.budget_squared,
        }))
    }

    fn validate_projector(&self, projector: &Projector) -> Result<()> {
        let all_indices: HashSet<_> = self.all_indices().into_iter().collect();
        for (index, &value) in projector.iter() {
            if !all_indices.contains(index) {
                return Err(PartitionedTTError::ProjectorIndexNotFound {
                    index: index.clone(),
                });
            }
            if value >= index.dim {
                return Err(PartitionedTTError::ProjectorCoordinateOutOfBounds {
                    index: index.clone(),
                    value,
                    dim: index.dim,
                });
            }
        }
        Ok(())
    }

    /// Project the tensor data by differentiably masking each selected index.
    fn project_tensor_data(&self, projector: &Projector) -> Result<TensorTrain> {
        let siteinds = self.data.siteinds();
        let mut new_tensors = Vec::with_capacity(self.data.len());

        for (site, site_indices) in siteinds.iter().enumerate() {
            let tensor = self
                .data
                .tensor(site)
                .map_err(|source| PartitionedTTError::TensorTrain { source })?;
            let mut projected_tensor = tensor.clone();
            for index in site_indices {
                if let Some(value) = projector.get(index) {
                    projected_tensor =
                        Self::project_tensor_at_index(&projected_tensor, index, value)?;
                }
            }
            new_tensors.push(projected_tensor);
        }

        TensorTrain::new(new_tensors).map_err(|source| PartitionedTTError::TensorTrain { source })
    }

    fn tensor_operation_error(error: anyhow::Error) -> PartitionedTTError {
        match error.downcast::<TensorStorageError>() {
            Ok(source) => PartitionedTTError::TensorStorage { source },
            Err(source) => PartitionedTTError::TensorConstruction { source },
        }
    }

    /// Project a single tensor by applying a backend-level one-hot mask.
    fn project_tensor_at_index(
        tensor: &TensorDynLen,
        index: &DynIndex,
        projected_value: usize,
    ) -> Result<TensorDynLen> {
        if !tensor.indices().iter().any(|candidate| candidate == index) {
            return Err(PartitionedTTError::ProjectorIndexNotFound {
                index: index.clone(),
            });
        }
        if projected_value >= index.dim {
            return Err(PartitionedTTError::ProjectorCoordinateOutOfBounds {
                index: index.clone(),
                value: projected_value,
                dim: index.dim,
            });
        }
        tensor
            .mask_index(index, projected_value)
            .map_err(Self::tensor_operation_error)
    }

    /// Compute the Frobenius norm.
    ///
    /// # Errors
    /// Propagates tensor-train storage or contraction failures.
    pub fn norm(&self) -> Result<f64> {
        self.data
            .norm()
            .map_err(|source| PartitionedTTError::TensorTrain { source })
    }

    /// Compute the squared Frobenius norm.
    ///
    /// # Errors
    /// Propagates tensor-train storage or contraction failures.
    pub fn norm_squared(&self) -> Result<f64> {
        self.data
            .norm_squared()
            .map_err(|source| PartitionedTTError::TensorTrain { source })
    }

    /// Truncate the tensor train.
    pub fn truncate(&mut self, options: &TruncateOptions) -> Result<()> {
        self.data.truncate(options).map_err(|e| {
            PartitionedTTError::tensor_train_operation(format!("Truncation failed: {}", e))
        })
    }

    /// Contract with another SubDomainTT.
    ///
    /// Returns `Ok(None)` if the projectors are incompatible. Before
    /// contraction, both inputs are projected to their subdomains (values
    /// outside the subdomain are zeroed out).
    ///
    /// # Errors
    ///
    /// Returns [`PartitionedTTError::ProjectorIndexNotFound`] or
    /// [`PartitionedTTError::ProjectorCoordinateOutOfBounds`] for invalid
    /// projectors, [`PartitionedTTError::TensorStorage`] for storage failures,
    /// [`PartitionedTTError::TensorConstruction`] for backend failures, or
    /// [`PartitionedTTError::TensorTrain`] for tensor-train validation or
    /// contraction failures.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_partitionedtt::{DynIndex, SubDomainTT, TensorDynLen, TensorTrain};
    /// use tensor4all_itensorlike::ContractOptions;
    ///
    /// let left_index = DynIndex::new_dyn(2);
    /// let right_index = DynIndex::new_dyn(2);
    /// let left = SubDomainTT::from_tt(TensorTrain::new(vec![
    ///     TensorDynLen::from_dense(vec![left_index], vec![1.0_f64, 2.0]).unwrap(),
    /// ]).unwrap());
    /// let right = SubDomainTT::from_tt(TensorTrain::new(vec![
    ///     TensorDynLen::from_dense(vec![right_index], vec![3.0_f64, 4.0]).unwrap(),
    /// ]).unwrap());
    ///
    /// let result = left.contract(&right, &ContractOptions::default()).unwrap();
    /// assert!(result.is_some());
    /// ```
    pub fn contract(&self, other: &Self, options: &ContractOptions) -> Result<Option<Self>> {
        // Check if projectors are compatible
        if !self.projector.is_compatible_with(other.projector()) {
            return Ok(None);
        }

        // Compute the projector after contraction (external indices only)
        let (proj_after, _external_indices) = Self::projector_after_contract(self, other)?;

        // Project both inputs to their subdomains before contraction
        // This ensures values outside the subdomain are zeroed out
        let self_projected = self.apply_projection()?;
        let other_projected = other.apply_projection()?;

        let contracted_data = self_projected
            .contract(&other_projected, options)
            .map_err(|e| {
                PartitionedTTError::tensor_train_operation(format!("Contraction failed: {}", e))
            })?;

        // Create result with the new projector
        let result = Self::new(contracted_data, proj_after);

        Ok(Some(result))
    }

    /// Apply the projector to the tensor data, zeroing out values outside the subdomain.
    ///
    /// Returns the TensorTrain with projection applied.
    fn apply_projection(&self) -> Result<TensorTrain> {
        if self.projector.is_empty() {
            return Ok(self.data.clone());
        }

        self.project_tensor_data(&self.projector)
    }

    /// Compute the projector after contracting two SubDomainTTs.
    ///
    /// Returns (projector, external_indices) where:
    /// - projector contains only projections for external indices
    /// - external_indices are indices that are not contracted away
    fn projector_after_contract(m1: &Self, m2: &Self) -> Result<(Projector, HashSet<DynIndex>)> {
        let indices1: HashSet<_> = m1.all_indices().into_iter().collect();
        let indices2: HashSet<_> = m2.all_indices().into_iter().collect();

        // External indices = (indices1 ∪ indices2) - (indices1 ∩ indices2)
        let common: HashSet<_> = indices1.intersection(&indices2).cloned().collect();
        let all: HashSet<_> = indices1.union(&indices2).cloned().collect();
        let external: HashSet<_> = all.difference(&common).cloned().collect();

        // Build projector for external indices only
        let mut proj_data = Vec::new();
        for idx in &external {
            if let Some(val) = m1.projector.get(idx) {
                proj_data.push((idx.clone(), val));
            } else if let Some(val) = m2.projector.get(idx) {
                proj_data.push((idx.clone(), val));
            }
        }

        Ok((Projector::from_pairs(proj_data), external))
    }

    /// Inner product with another SubDomainTT.
    pub fn inner(&self, other: &Self) -> Result<AnyScalar> {
        self.data
            .inner(other.data())
            .map_err(|err| PartitionedTTError::tensor_train_operation(err.to_string()))
    }
}

// Conversion from TensorTrain
impl From<TensorTrain> for SubDomainTT {
    fn from(tt: TensorTrain) -> Self {
        Self::from_tt(tt)
    }
}

// Conversion to TensorTrain
impl From<SubDomainTT> for TensorTrain {
    fn from(subdomain: SubDomainTT) -> Self {
        subdomain.into_data()
    }
}

#[cfg(test)]
mod tests;
