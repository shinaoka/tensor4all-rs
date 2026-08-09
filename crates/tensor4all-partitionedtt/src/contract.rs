//! Contraction operations for SubDomainTT and PartitionedTT
//!
//! This module provides helper functions for contracting tensor trains within
//! partitioned structures.
//!
//! The main contraction functionality is implemented in `SubDomainTT::contract`
//! and `PartitionedTT::contract`. This module provides additional utilities.

use crate::error::Result;
use crate::projector::Projector;
use crate::subdomain_tt::SubDomainTT;
use tensor4all_itensorlike::ContractOptions;

/// Contract two [`SubDomainTT`] values.
///
/// The contraction is only non-vanishing if the projectors are compatible.
/// Returns `Ok(None)` if the projectors are incompatible.
///
/// # Errors
///
/// Propagates projector validation, typed tensor storage/construction, and
/// tensor-train errors from [`SubDomainTT::contract`].
///
/// # Examples
///
/// ```
/// use tensor4all_partitionedtt::{contract, DynIndex, SubDomainTT, TensorDynLen, TensorTrain};
/// use tensor4all_itensorlike::ContractOptions;
///
/// let i = DynIndex::new_dyn(2);
/// let j = DynIndex::new_dyn(2);
/// let left = SubDomainTT::from_tt(TensorTrain::new(vec![
///     TensorDynLen::from_dense(vec![i], vec![1.0_f64, 2.0]).unwrap(),
/// ]).unwrap());
/// let right = SubDomainTT::from_tt(TensorTrain::new(vec![
///     TensorDynLen::from_dense(vec![j], vec![3.0_f64, 4.0]).unwrap(),
/// ]).unwrap());
///
/// assert!(contract(&left, &right, &ContractOptions::default()).unwrap().is_some());
/// ```
pub fn contract(
    m1: &SubDomainTT,
    m2: &SubDomainTT,
    options: &ContractOptions,
) -> Result<Option<SubDomainTT>> {
    m1.contract(m2, options)
}

/// Project two [`SubDomainTT`] values to a projector before contracting them.
///
/// Returns `Ok(None)` when either projection or the final contraction has no
/// compatible subdomain.
///
/// # Errors
///
/// Propagates projector validation, typed tensor storage/construction, and
/// tensor-train errors from projection and contraction.
///
/// # Examples
///
/// ```
/// use tensor4all_partitionedtt::{proj_contract, DynIndex, Projector, SubDomainTT, TensorDynLen, TensorTrain};
/// use tensor4all_itensorlike::ContractOptions;
///
/// let i = DynIndex::new_dyn(2);
/// let left = SubDomainTT::from_tt(TensorTrain::new(vec![
///     TensorDynLen::from_dense(vec![i.clone()], vec![1.0_f64, 2.0]).unwrap(),
/// ]).unwrap());
/// let right = SubDomainTT::from_tt(TensorTrain::new(vec![
///     TensorDynLen::from_dense(vec![i], vec![3.0_f64, 4.0]).unwrap(),
/// ]).unwrap());
/// let projector = Projector::from_pairs([(left.all_indices()[0].clone(), 0)]);
///
/// assert!(proj_contract(&left, &right, &projector, &ContractOptions::default())
///     .unwrap()
///     .is_some());
/// ```
pub fn proj_contract(
    m1: &SubDomainTT,
    m2: &SubDomainTT,
    proj: &Projector,
    options: &ContractOptions,
) -> Result<Option<SubDomainTT>> {
    // A shared projector may mention an index belonging to only one input;
    // filter it at this two-input seam before each strict single-TT project.
    let m1_proj = match m1.project(&proj.filter_indices(&m1.all_indices()))? {
        Some(m) => m,
        None => return Ok(None),
    };
    let m2_proj = match m2.project(&proj.filter_indices(&m2.all_indices()))? {
        Some(m) => m,
        None => return Ok(None),
    };

    // Contract the projected tensor trains
    m1_proj.contract(&m2_proj, options)
}

#[cfg(test)]
mod tests;
