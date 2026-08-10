//! Linear equation solver for Tree Tensor Networks (V_in = V_out case).
//!
//! This module provides the `square_linsolve` function for solving linear systems
//! of the form `(a₀ + a₁ * A) * x = b` where A is a TTN operator and x, b are TTN states,
//! and the input and output spaces are the same (V_in = V_out).
//!
//! # Algorithm
//!
//! The algorithm uses alternating updates (sweeping) similar to DMRG:
//! 1. Position environments to expose a local region
//! 2. Solve the local linear problem using GMRES (through `tensor4all_core::krylov::gmres`)
//! 3. Factorize the result and move the orthogonality center
//! 4. Update environment caches
//!
//! # Inspired By
//!
//! This implementation is inspired by:
//! - [ITensorMPS.jl](https://github.com/ITensor/ITensorMPS.jl) - Core algorithm structure
//! - [KrylovKit.jl](https://github.com/Jutho/KrylovKit.jl) - Krylov solver integration pattern
//!
//! # References
//!
//! - Phys. Rev. B 72, 180403 (2005) - Noise term technique (not implemented in initial version)

pub(crate) mod local_linop;
mod projected_state;
mod updater;

pub use projected_state::ProjectedState;
pub use updater::{LinsolveVerifyReport, NodeVerifyDetail, SquareLinsolveUpdater};

use std::collections::HashMap;
use std::hash::Hash;

use anyhow::{bail, Result};

use tensor4all_core::{AnyScalar, IndexLike, TensorLike};

use crate::linsolve::common::LinsolveOptions;
use crate::operator::{apply_linear_operator, ApplyOptions, IndexMapping, LinearOperator};
use crate::{apply_local_update_sweep, CanonicalizationOptions, LocalUpdateSweepPlan, TreeTN};

/// Result of square_linsolve operation.
#[derive(Debug, Clone)]
pub struct SquareLinsolveResult<T, V>
where
    T: TensorLike,
    V: Clone + Hash + Eq + Send + Sync + std::fmt::Debug,
{
    /// The solution TreeTN
    pub solution: TreeTN<T, V>,
    /// Number of sweeps performed
    pub sweeps: usize,
    /// Final relative residual norm `||(a0 I + a1 A)x - b|| / ||b||`.
    pub residual: Option<f64>,
    /// Converged flag
    pub converged: bool,
}

/// Validate that operator, rhs, and init have compatible structures for linsolve.
///
/// Checks:
/// 1. Operator can act on init (same topology)
/// 2. Result of operator action has compatible site dimensions with rhs
fn validate_linsolve_inputs<T, V>(
    operator: &TreeTN<T, V>,
    rhs: &TreeTN<T, V>,
    init: &TreeTN<T, V>,
) -> Result<()>
where
    T: TensorLike,
    <T::Index as tensor4all_core::IndexLike>::Id:
        Clone + std::hash::Hash + Eq + Ord + std::fmt::Debug + Send + Sync,
    V: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug,
{
    let init_network = init.site_index_network();
    let op_network = operator.site_index_network();
    let rhs_network = rhs.site_index_network();

    // The sweep planner builds two-site updates only; a one-node network yields
    // an empty plan and the solver would silently return the initial guess.
    if rhs_network.node_count() < 2 {
        return Err(anyhow::anyhow!(
            "square_linsolve requires at least two sites; the one-site local \
             solve is not implemented"
        ));
    }

    // Check 1: Operator can act on init
    let result_network = init_network
        .apply_operator_topology(op_network)
        .map_err(|e| anyhow::anyhow!("Operator cannot act on init: {}", e))?;

    // Check 2: Result has compatible dimensions with rhs
    if !result_network.compatible_site_dimensions(rhs_network) {
        return Err(anyhow::anyhow!(
            "Result of operator action is not compatible with RHS"
        ));
    }

    Ok(())
}

/// Solve the linear system `(a₀ + a₁ * H) |x⟩ = |b⟩` for TreeTN.
///
/// This solver is for the square case where V_in = V_out (input and output spaces are the same).
///
/// # Arguments
///
/// * `operator` - The operator H as a TreeTN (must have compatible structure with `rhs`)
/// * `rhs` - The right-hand side |b⟩ as a TreeTN
/// * `init` - Initial guess for |x⟩
/// * `center` - Node to use as sweep center
/// * `options` - Solver options
/// * `input_mapping` - Optional per-node mapping from state site index to operator input index.
///
///   Required when the operator (MPO) uses internal indices distinct from the state's site indices.
/// * `output_mapping` - Optional per-node mapping from state site index to operator output index.
///
///   Required when the operator (MPO) uses internal indices distinct from the state's site indices.
///
/// # Returns
///
/// The solution TreeTN, or an error if solving fails.
///
/// # Errors
///
/// Returns an error when the solve or sweep fails (a shape or index
/// /// mismatch, a non-convergence failure, or a backend failure).
///
/// # Example
///
/// ```
/// use std::collections::HashMap;
/// use tensor4all_core::{DynIndex, TensorDynLen};
/// use tensor4all_treetn::{square_linsolve, IndexMapping, LinsolveOptions, TreeTN};
///
/// # fn main() -> anyhow::Result<()> {
/// let phys_dim = 2usize;
/// let s0 = DynIndex::new_dyn(phys_dim);
/// let s1 = DynIndex::new_dyn(phys_dim);
/// let b01 = DynIndex::new_dyn(phys_dim);
///
/// // Right-hand side |b>: a two-site state with bond dimension two.
/// let mut rhs = TreeTN::<TensorDynLen, usize>::new();
/// let a = TensorDynLen::from_dense(
///     vec![s0.clone(), b01.clone()],
///     vec![1.0_f64, 0.0, 0.0, 2.0],
/// )?;
/// let b = TensorDynLen::from_dense(
///     vec![b01.clone(), s1.clone()],
///     vec![3.0_f64, 0.0, 0.0, 4.0],
/// )?;
/// let n0 = rhs.add_tensor(0, a)?;
/// let n1 = rhs.add_tensor(1, b)?;
/// rhs.connect(n0, &b01, n1, &b01)?;
///
/// // Identity MPO with internal input/output indices and a dim-1 bond.
/// let s0_in = DynIndex::new_dyn(phys_dim);
/// let s1_in = DynIndex::new_dyn(phys_dim);
/// let s0_out = DynIndex::new_dyn(phys_dim);
/// let s1_out = DynIndex::new_dyn(phys_dim);
/// let m_bond = DynIndex::new_dyn(1);
/// let mut id = vec![0.0_f64; phys_dim * phys_dim];
/// for i in 0..phys_dim {
///     id[i * phys_dim + i] = 1.0;
/// }
/// let mut operator = TreeTN::<TensorDynLen, usize>::new();
/// let t0 = TensorDynLen::from_dense(
///     vec![s0_out.clone(), s0_in.clone(), m_bond.clone()],
///     id.clone(),
/// )?;
/// let t1 = TensorDynLen::from_dense(
///     vec![m_bond.clone(), s1_out.clone(), s1_in.clone()],
///     id.clone(),
/// )?;
/// let m0 = operator.add_tensor(0, t0)?;
/// let m1 = operator.add_tensor(1, t1)?;
/// operator.connect(m0, &m_bond, m1, &m_bond)?;
///
/// // Map each state site to the operator's internal in/out indices.
/// let mut input_mapping = HashMap::new();
/// input_mapping.insert(0usize, IndexMapping { true_index: s0.clone(), internal_index: s0_in });
/// input_mapping.insert(1usize, IndexMapping { true_index: s1.clone(), internal_index: s1_in });
/// let mut output_mapping = HashMap::new();
///
/// // Zero initial guess: not the solution, so the assertions below prove the
/// // solver actually solves the system.
/// let mut init = TreeTN::<TensorDynLen, usize>::new();
/// let iz0 = TensorDynLen::from_dense(
///     vec![s0.clone(), b01.clone()],
///     vec![0.0_f64, 0.0, 0.0, 0.0],
/// )?;
/// let iz1 = TensorDynLen::from_dense(
///     vec![b01.clone(), s1.clone()],
///     vec![0.0_f64, 0.0, 0.0, 0.0],
/// )?;
/// let in0 = init.add_tensor(0, iz0)?;
/// let in1 = init.add_tensor(1, iz1)?;
/// init.connect(in0, &b01, in1, &b01)?;
///
/// output_mapping.insert(0usize, IndexMapping { true_index: s0, internal_index: s0_out });
/// output_mapping.insert(1usize, IndexMapping { true_index: s1, internal_index: s1_out });
/// let options = LinsolveOptions::default()
///     .with_nfullsweeps(5)
///     .with_gmres_tol(1.0e-10)
///     .with_gmres_restart_dim(10)
///     .with_gmres_max_restarts(30)
///     .with_max_rank(4)
///     .with_convergence_tol(1.0e-8);
/// let result = square_linsolve(
///     &operator,
///     &rhs,
///     init,
///     &0usize,
///     options,
///     Some(input_mapping),
///     Some(output_mapping),
/// )?;
/// assert!(result.residual.is_some_and(|residual| residual < 1.0e-8));
/// let expected = rhs.contract_to_tensor()?.to_vec::<f64>()?;
/// let actual = result.solution.contract_to_tensor()?.to_vec::<f64>()?;
/// let diff: f64 = expected
///     .iter()
///     .zip(actual.iter())
///     .map(|(e, a)| (e - a).powi(2))
///     .sum();
/// assert!(diff.sqrt() < 1.0e-6);
/// # Ok(())
/// # }
/// ```
pub fn square_linsolve<T, V>(
    operator: &TreeTN<T, V>,
    rhs: &TreeTN<T, V>,
    init: TreeTN<T, V>,
    center: &V,
    options: LinsolveOptions,
    input_mapping: Option<HashMap<V, IndexMapping<T::Index>>>,
    output_mapping: Option<HashMap<V, IndexMapping<T::Index>>>,
) -> Result<SquareLinsolveResult<T, V>>
where
    T: TensorLike + 'static,
    <T::Index as IndexLike>::Id:
        Clone + std::hash::Hash + Eq + Ord + std::fmt::Debug + Send + Sync + 'static,
    V: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug + 'static,
{
    // Validate inputs before proceeding
    validate_linsolve_inputs(operator, rhs, &init)?;

    if options.a1.is_zero() || operator_is_zero(operator)? {
        return solve_identity_term_only(
            operator,
            rhs,
            &init,
            options,
            input_mapping,
            output_mapping,
        );
    }

    // Canonicalize initial guess towards center
    let mut x = init.canonicalize([center.clone()], CanonicalizationOptions::default())?;

    let needs_residual_operator = options.check_residual || options.convergence_tol.is_some();
    let residual_operator = if needs_residual_operator {
        Some(linear_operator_for_residual(
            operator,
            &x,
            input_mapping.clone(),
            output_mapping.clone(),
        )?)
    } else {
        None
    };

    // Create SquareLinsolveUpdater with or without index mappings
    let mut updater = match (input_mapping, output_mapping) {
        (Some(input), Some(output)) => SquareLinsolveUpdater::with_index_mappings(
            operator.clone(),
            input,
            output,
            rhs.clone(),
            options.clone(),
        ),
        (None, None) => SquareLinsolveUpdater::new(operator.clone(), rhs.clone(), options.clone()),
        _ => {
            return Err(anyhow::anyhow!(
                "input_mapping and output_mapping must both be Some or both be None"
            ));
        }
    };

    // Create sweep plan (nsite=2 for 2-site updates)
    let plan = LocalUpdateSweepPlan::from_treetn(&x, center, 2)
        .ok_or_else(|| anyhow::anyhow!("Failed to create sweep plan"))?;

    let mut final_sweeps = 0;

    let mut residual = None;
    let mut converged = false;

    // Perform sweeps
    for sweep in 0..options.nfullsweeps {
        final_sweeps = sweep + 1;
        apply_local_update_sweep(&mut x, &plan, &mut updater)?;
        if let Some(tol) = options.convergence_tol {
            let current_residual = relative_linear_system_residual(
                residual_operator
                    .as_ref()
                    .ok_or_else(|| anyhow::anyhow!("missing residual operator"))?,
                &x,
                rhs,
                options.a0.clone(),
                options.a1.clone(),
                ApplyOptions::naive(),
            )?;
            residual = Some(current_residual);
            if current_residual < tol {
                converged = true;
                break;
            }
        }
    }

    if residual.is_none() && options.check_residual {
        let final_residual = relative_linear_system_residual(
            residual_operator
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("missing residual operator"))?,
            &x,
            rhs,
            options.a0.clone(),
            options.a1.clone(),
            ApplyOptions::naive(),
        )?;
        converged = options
            .convergence_tol
            .is_some_and(|tol| final_residual < tol);
        residual = Some(final_residual);
    }

    Ok(SquareLinsolveResult {
        solution: x,
        sweeps: final_sweeps,
        residual,
        converged,
    })
}

fn operator_is_zero<T, V>(operator: &TreeTN<T, V>) -> Result<bool>
where
    T: TensorLike,
    V: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug,
{
    let mut operator = operator.clone();
    Ok(operator.norm()? <= 1.0e-15)
}

fn solve_identity_term_only<T, V>(
    operator: &TreeTN<T, V>,
    rhs: &TreeTN<T, V>,
    init: &TreeTN<T, V>,
    options: LinsolveOptions,
    input_mapping: Option<HashMap<V, IndexMapping<T::Index>>>,
    output_mapping: Option<HashMap<V, IndexMapping<T::Index>>>,
) -> Result<SquareLinsolveResult<T, V>>
where
    T: TensorLike,
    T::Index: IndexLike + Clone + Hash + Eq + std::fmt::Debug,
    <T::Index as IndexLike>::Id: Clone + std::hash::Hash + Eq + Ord + std::fmt::Debug + Send + Sync,
    V: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug,
{
    if options.a0.is_zero() {
        bail!("square_linsolve: a0 and effective operator term are both zero");
    }

    let mut solution = rhs.clone();
    solution.scale(AnyScalar::new_real(1.0) / options.a0.clone())?;
    let residual = if options.check_residual || options.convergence_tol.is_some() {
        let residual_operator =
            linear_operator_for_residual(operator, init, input_mapping, output_mapping)?;
        Some(relative_linear_system_residual(
            &residual_operator,
            &solution,
            rhs,
            options.a0.clone(),
            options.a1.clone(),
            ApplyOptions::naive(),
        )?)
    } else {
        None
    };
    let converged = options
        .convergence_tol
        .zip(residual)
        .is_some_and(|(tol, residual)| residual < tol);

    Ok(SquareLinsolveResult {
        solution,
        sweeps: 0,
        residual,
        converged,
    })
}

fn linear_operator_for_residual<T, V>(
    operator: &TreeTN<T, V>,
    state: &TreeTN<T, V>,
    input_mapping: Option<HashMap<V, IndexMapping<T::Index>>>,
    output_mapping: Option<HashMap<V, IndexMapping<T::Index>>>,
) -> Result<LinearOperator<T, V>>
where
    T: TensorLike,
    T::Index: IndexLike + Clone + Hash + Eq + std::fmt::Debug,
    <T::Index as IndexLike>::Id: Clone + std::hash::Hash + Eq + Ord + std::fmt::Debug + Send + Sync,
    V: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug,
{
    match (input_mapping, output_mapping) {
        (Some(input), Some(output)) => Ok(LinearOperator::new(operator.clone(), input, output)),
        (None, None) => LinearOperator::from_mpo_and_state(operator.clone(), state),
        _ => Err(anyhow::anyhow!(
            "input_mapping and output_mapping must both be Some or both be None"
        )),
    }
}

/// Compute the true relative residual of a TreeTN linear system.
///
/// This evaluates `||(a0 I + a1 A)x - b|| / ||b||` by applying `operator` to
/// `solution`, combining the identity and operator terms, and measuring the
/// TreeTN norm of the resulting residual. When `||b||` is numerically zero, the
/// absolute residual norm is returned to avoid division by zero.
///
/// # Arguments
/// * `operator` - Linear operator `A` including any input/output index mappings.
/// * `solution` - Candidate solution `x`.
/// * `rhs` - Right-hand side `b`.
/// * `a0` - Identity coefficient.
/// * `a1` - Operator coefficient.
/// * `apply_options` - Options for applying `A`; use [`ApplyOptions::naive`] for
///
///   an exact residual of the represented TreeTN.
///
/// # Returns
/// The relative residual norm, or the absolute residual norm for zero RHS.
///
/// # Errors
///
/// Returns an error when the solve or sweep fails (a shape or index
/// /// mismatch, a non-convergence failure, or a backend failure).
///
/// # Examples
/// ```
/// use std::collections::HashMap;
/// use tensor4all_core::{DynIndex, TensorDynLen};
/// use tensor4all_treetn::{
///     relative_linear_system_residual, ApplyOptions, IndexMapping, LinearOperator, TreeTN,
/// };
///
/// let site = DynIndex::new_dyn(2);
/// let s_in = DynIndex::new_dyn(2);
/// let s_out = DynIndex::new_dyn(2);
/// let state_tensor = TensorDynLen::from_dense(vec![site.clone()], vec![3.0_f64, 5.0]).unwrap();
/// let state = TreeTN::<TensorDynLen, usize>::from_tensors(vec![state_tensor], vec![0]).unwrap();
/// let mpo_tensor = TensorDynLen::from_dense(
///     vec![s_out.clone(), s_in.clone()],
///     vec![1.0_f64, 0.0, 0.0, 1.0],
/// ).unwrap();
/// let mpo = TreeTN::<TensorDynLen, usize>::from_tensors(vec![mpo_tensor], vec![0]).unwrap();
/// let mut input_mapping = HashMap::new();
/// input_mapping.insert(0usize, IndexMapping { true_index: site.clone(), internal_index: s_in });
/// let mut output_mapping = HashMap::new();
/// output_mapping.insert(0usize, IndexMapping { true_index: site, internal_index: s_out });
/// let operator = LinearOperator::new(mpo, input_mapping, output_mapping);
///
/// let residual = relative_linear_system_residual(
///     &operator,
///     &state,
///     &state,
///     0.0,
///     1.0,
///     ApplyOptions::naive(),
/// ).unwrap();
/// assert!(residual < 1.0e-12);
/// ```
pub fn relative_linear_system_residual<T, V, A0, A1>(
    operator: &LinearOperator<T, V>,
    solution: &TreeTN<T, V>,
    rhs: &TreeTN<T, V>,
    a0: A0,
    a1: A1,
    apply_options: ApplyOptions,
) -> Result<f64>
where
    T: TensorLike,
    T::Index: IndexLike + Clone + Hash + Eq + std::fmt::Debug,
    <T::Index as IndexLike>::Id: Clone + std::hash::Hash + Eq + Ord + std::fmt::Debug + Send + Sync,
    V: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug,
    A0: Into<AnyScalar>,
    A1: Into<AnyScalar>,
{
    let a0 = a0.into();
    let a1 = a1.into();
    let mut lhs = solution.clone();
    lhs.scale(a0)?;

    if !a1.is_zero() {
        let mut ax = apply_linear_operator(operator, solution, apply_options)?;
        ax.scale(a1)?;
        lhs = lhs.add_reindexed_like_self(&ax)?;
    }

    let mut negative_rhs = rhs.clone();
    negative_rhs.scale(AnyScalar::new_real(-1.0))?;
    let mut residual = lhs.add_reindexed_like_self(&negative_rhs)?;

    let rhs_norm = rhs.clone().norm()?;
    if rhs_norm <= 1.0e-15 {
        return residual.norm().map_err(anyhow::Error::from);
    }
    Ok(residual.norm()? / rhs_norm)
}

#[cfg(test)]
mod tests;
