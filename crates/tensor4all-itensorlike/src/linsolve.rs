//! Linear equation solver for tensor trains.
//!
//! This module provides the [`linsolve`] function for solving linear systems
//! of the form `(a₀ + a₁ * A) * x = b` where `A` is an MPO (TensorTrain),
//! and `x`, `b` are MPS (TensorTrain).
//!
//! Internally delegates to [`tensor4all_treetn::square_linsolve`].

use std::collections::HashMap;

use tensor4all_core::DynIndex;
use tensor4all_treetn::{square_linsolve, IndexMapping, TruncationOptions};

use crate::error::{Result, TensorTrainError};
use crate::options::{validate_svd_truncation_options, CanonicalForm, LinsolveOptions};
use crate::tensortrain::TensorTrain;

/// Solve `(a₀ + a₁ * A) * x = b` for `x`.
///
/// # Arguments
/// * `operator` - The operator `A` (MPO as TensorTrain)
/// * `rhs` - The right-hand side `b` (MPS as TensorTrain)
/// * `init` - Initial guess for `x` (MPS as TensorTrain, consumed)
/// * `options` - Solver options
///
/// # Returns
/// The solution `x` as a TensorTrain.
///
/// # Errors
///
/// Returns an error when the solve or sweep fails (a shape or index
/// /// mismatch, a non-convergence failure, or a backend failure).
///
pub fn linsolve(
    operator: &TensorTrain,
    rhs: &TensorTrain,
    init: TensorTrain,
    options: &LinsolveOptions,
) -> Result<TensorTrain> {
    if operator.is_empty() || rhs.is_empty() || init.is_empty() {
        return Err(TensorTrainError::InvalidStructure {
            message: "Cannot linsolve with empty tensor trains".to_string(),
        });
    }
    if operator.len() != rhs.len() || operator.len() != init.len() {
        return Err(TensorTrainError::InvalidStructure {
            message: format!(
                "operator, rhs, and init must have equal lengths; got {}, {}, {}",
                operator.len(),
                rhs.len(),
                init.len()
            ),
        });
    }

    if !options.nhalfsweeps().is_multiple_of(2) {
        return Err(TensorTrainError::OperationError {
            message: format!(
                "nhalfsweeps must be a multiple of 2, got {}",
                options.nhalfsweeps()
            ),
        });
    }

    validate_svd_truncation_options(options.max_bond_dim(), options.svd_policy())?;

    if !options.gmres_tol().is_finite() || options.gmres_tol() <= 0.0 {
        return Err(TensorTrainError::OperationError {
            message: format!(
                "gmres_tol must be finite and > 0, got {}",
                options.gmres_tol()
            ),
        });
    }

    if options.gmres_max_restarts() == 0 {
        return Err(TensorTrainError::OperationError {
            message: "gmres_max_restarts must be >= 1".to_string(),
        });
    }

    if options.gmres_restart_dim() == 0 {
        return Err(TensorTrainError::OperationError {
            message: "gmres_restart_dim must be >= 1".to_string(),
        });
    }

    if let Some(tol) = options.convergence_tol() {
        if !tol.is_finite() || tol < 0.0 {
            return Err(TensorTrainError::OperationError {
                message: format!("convergence_tol must be finite and >= 0, got {}", tol),
            });
        }
    }

    // Convert LinsolveOptions → treetn::LinsolveOptions
    let nfullsweeps = options.nhalfsweeps() / 2;

    let (a0, a1) = options.coefficients();
    let treetn_options = tensor4all_treetn::LinsolveOptions::new(nfullsweeps)
        .with_truncation(TruncationOptions::new())
        .with_gmres_tol(options.gmres_tol())
        .with_gmres_max_restarts(options.gmres_max_restarts())
        .with_gmres_restart_dim(options.gmres_restart_dim())
        .with_coefficients(a0, a1);

    let treetn_options = if let Some(policy) = options.svd_policy() {
        treetn_options.with_svd_policy(policy)
    } else {
        treetn_options
    };

    let treetn_options = if let Some(max_bond_dim) = options.max_bond_dim() {
        treetn_options.with_max_bond_dim(max_bond_dim)
    } else {
        treetn_options
    };

    let treetn_options = if let Some(tol) = options.convergence_tol() {
        treetn_options.with_convergence_tol(tol)
    } else {
        treetn_options
    };
    let treetn_options = treetn_options.with_residual_check(options.check_residual());

    // Use the last site as the sweep center
    let center = init.len() - 1;

    // Auto-infer index mappings when the operator (MPO) has distinct input/output
    // site indices. For each site, the MPO has 2 site indices while the MPS has 1.
    // We find the MPO index that shares an ID with init's site index (input) and
    // the remaining one (output).
    let (input_mapping, output_mapping) = infer_index_mappings(operator, &init)?;

    let result = square_linsolve(
        operator.as_treetn(),
        rhs.as_treetn(),
        init.treetn,
        &center,
        treetn_options,
        input_mapping,
        output_mapping,
    )
    .map_err(|e| TensorTrainError::OperationError {
        message: format!("Linsolve failed: {}", e),
    })?;

    TensorTrain::from_inner(result.solution, Some(CanonicalForm::Unitary))
}

type SiteMappings = (
    Option<HashMap<usize, IndexMapping<DynIndex>>>,
    Option<HashMap<usize, IndexMapping<DynIndex>>>,
);

/// Infer index mappings from the MPO/MPS structure.
///
/// For each site where the operator has exactly 2 site indices and init has 1,
/// finds the operator index exactly matching init's index (input) and the
/// remaining full index (output). Returns `(None, None)` when no mappings are
/// needed (operator and init share all site indices).
fn infer_index_mappings(operator: &TensorTrain, init: &TensorTrain) -> Result<SiteMappings> {
    let op_treetn = operator.as_treetn();
    let init_treetn = init.as_treetn();
    let nsites = init.len();

    let mut needs_mapping = false;

    // First pass: check if any site needs mappings
    for site in 0..nsites {
        let op_site = op_treetn.site_space(&site);
        let init_site = init_treetn.site_space(&site);

        if let (Some(op_indices), Some(init_indices)) = (op_site, init_site) {
            if op_indices.len() == 2 && init_indices.len() == 1 {
                // MPO-like site: exact input match plus one distinct output leg.
                let init_idx =
                    init_indices
                        .iter()
                        .next()
                        .ok_or_else(|| TensorTrainError::OperationError {
                            message: format!("Site {site}: init site space is empty"),
                        })?;
                let op_input = op_indices
                    .iter()
                    .find(|idx| *idx == init_idx)
                    .ok_or_else(|| TensorTrainError::OperationError {
                        message: format!(
                            "Site {}: operator has 2 site indices but none exactly match init's \
                             site index {:?}. Cannot auto-infer index mappings. \
                             Use the treetn-level API with explicit IndexMapping.",
                            site, init_idx
                        ),
                    })?;
                op_indices
                    .iter()
                    .find(|idx| *idx != op_input)
                    .ok_or_else(|| TensorTrainError::OperationError {
                        message: format!(
                            "Site {site}: operator has no output index distinct from init index {:?}",
                            init_idx
                        ),
                    })?;
                needs_mapping = true;
            }
        }
    }

    if !needs_mapping {
        return Ok((None, None));
    }

    // Second pass: build mappings
    let mut input_mapping = HashMap::new();
    let mut output_mapping = HashMap::new();

    for site in 0..nsites {
        let op_site = op_treetn.site_space(&site);
        let init_site = init_treetn.site_space(&site);

        if let (Some(op_indices), Some(init_indices)) = (op_site, init_site) {
            if op_indices.len() == 2 && init_indices.len() == 1 {
                let init_idx =
                    init_indices
                        .iter()
                        .next()
                        .ok_or_else(|| TensorTrainError::OperationError {
                            message: format!("Site {site}: init site space is empty"),
                        })?;

                let op_input = op_indices
                    .iter()
                    .find(|idx| *idx == init_idx)
                    .ok_or_else(|| TensorTrainError::OperationError {
                        message: format!(
                            "Site {site}: operator has no input index exactly matching init index {:?}",
                            init_idx
                        ),
                    })?;
                let op_output = op_indices
                    .iter()
                    .find(|idx| *idx != op_input)
                    .ok_or_else(|| TensorTrainError::OperationError {
                        message: format!(
                            "Site {site}: operator has no output index distinct from init index {:?}",
                            init_idx
                        ),
                    })?;

                input_mapping.insert(
                    site,
                    IndexMapping {
                        true_index: init_idx.clone(),
                        internal_index: op_input.clone(),
                    },
                );
                output_mapping.insert(
                    site,
                    IndexMapping {
                        true_index: init_idx.clone(),
                        internal_index: op_output.clone(),
                    },
                );
            }
        }
    }

    Ok((Some(input_mapping), Some(output_mapping)))
}

impl TensorTrain {
    /// Solve `(a₀ + a₁ * A) * x = b` for `x`.
    ///
    /// `self` is the operator `A`, `rhs` is `b`, `init` is the initial guess.
    ///
    /// See [`linsolve`] for details.
    /// # Errors
    ///
    /// Returns an error when the solve or sweep fails (a shape or index
    /// /// mismatch, a non-convergence failure, or a backend failure).
    ///
    pub fn linsolve(&self, rhs: &Self, init: Self, options: &LinsolveOptions) -> Result<Self> {
        linsolve(self, rhs, init, options)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tensor4all_core::{DynIndex, IdxTensor};

    fn make_tensor(indices: Vec<DynIndex>, data: Vec<f64>) -> IdxTensor {
        IdxTensor::from_dense(indices, data).unwrap()
    }

    #[test]
    fn infer_index_mappings_returns_typed_error() {
        let op_in = DynIndex::new_dyn(2);
        let op_out = DynIndex::new_dyn(2);
        let init_site = DynIndex::new_dyn(2);

        let operator = TensorTrain::new(vec![make_tensor(
            vec![op_in, op_out],
            vec![1.0, 0.0, 0.0, 1.0],
        )])
        .unwrap();
        let init = TensorTrain::new(vec![make_tensor(vec![init_site], vec![1.0, 2.0])]).unwrap();

        let result: crate::error::Result<_> = infer_index_mappings(&operator, &init);
        let err = result.unwrap_err();

        assert!(matches!(err, TensorTrainError::OperationError { .. }));
        assert!(err.to_string().contains("Cannot auto-infer index mappings"));
    }

    #[test]
    fn infer_index_mappings_handles_same_id_primed_output() {
        let init_site = DynIndex::new_dyn(2);
        let output_site = init_site.prime();

        let operator = TensorTrain::new(vec![make_tensor(
            vec![output_site.clone(), init_site.clone()],
            vec![1.0, 0.0, 0.0, 1.0],
        )])
        .unwrap();
        let init =
            TensorTrain::new(vec![make_tensor(vec![init_site.clone()], vec![1.0, 2.0])]).unwrap();

        let (input_mapping, output_mapping) = infer_index_mappings(&operator, &init).unwrap();
        let input_mapping = input_mapping.unwrap();
        let output_mapping = output_mapping.unwrap();

        assert_eq!(input_mapping[&0].true_index, init_site);
        assert_eq!(input_mapping[&0].internal_index, init_site);
        assert_eq!(output_mapping[&0].true_index, init_site);
        assert_eq!(output_mapping[&0].internal_index, output_site);
    }
}
