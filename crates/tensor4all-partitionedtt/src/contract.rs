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

/// Contract two SubDomainTTs.
///
/// The contraction is only non-vanishing if the projectors are compatible.
/// Returns `None` if the projectors are incompatible.
pub fn contract(
    m1: &SubDomainTT,
    m2: &SubDomainTT,
    options: &ContractOptions,
) -> Result<Option<SubDomainTT>> {
    m1.contract(m2, options)
}

/// Project two SubDomainTTs to a projector before contracting them.
pub fn proj_contract(
    m1: &SubDomainTT,
    m2: &SubDomainTT,
    proj: &Projector,
    options: &ContractOptions,
) -> Result<Option<SubDomainTT>> {
    // Project both inputs
    let m1_proj = match m1.project(proj)? {
        Some(m) => m,
        None => return Ok(None),
    };
    let m2_proj = match m2.project(proj)? {
        Some(m) => m,
        None => return Ok(None),
    };

    // Contract the projected tensor trains
    m1_proj.contract(&m2_proj, options)
}

#[cfg(test)]
mod tests;
