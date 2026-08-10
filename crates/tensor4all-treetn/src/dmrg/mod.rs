//! Two-site DMRG for TreeTN states.
//!
//! This module implements the first TreeTN DMRG path: two-site sweeps with a
//! Hermitian projected local eigensolver. The production algorithm contracts
//! only local regions and their environments; full TreeTN materialization is
//! reserved for small tests and external benchmark validation.

use std::collections::HashMap;
use std::hash::Hash;
use std::sync::{Arc, RwLock};

use tensor4all_core::krylov::{hermitian_lanczos_lowest_eigenpair, HermitianLanczosOptions};
use tensor4all_core::{AnyScalar, FactorizeOptions, IndexLike, SvdTruncationPolicy, TensorLike};
use thiserror::Error;

use crate::error::TreeTNOperationError;
use crate::linsolve::common::ProjectedOperator;
use crate::linsolve::square::local_linop::LocalLinOp;
use crate::local_update_support::{
    build_subtree_topology, contract_region, copy_decomposed_to_subtree,
    initialize_reference_state_if_empty, single_site_square_mappings, sync_reference_state_region,
    SquareSiteMappingError,
};
use crate::operator::{IndexMapping, LinearOperator};
use crate::{
    apply_local_update_sweep, factorize_tensor_to_treetn_with, CanonicalizationOptions,
    LocalUpdateStep, LocalUpdateSweepPlan, LocalUpdater, TreeTN,
};

/// Errors reported by TreeTN DMRG.
///
/// This error type covers unsupported v1 API shapes, invalid options, and
/// numerical failures from local Hermitian eigensolves.
#[derive(Debug, Error)]
pub enum DmrgError {
    /// The requested local update size is not implemented.
    #[error("DMRG supports only nsite={supported} in this version, got nsite={requested}")]
    UnsupportedNsite {
        /// Requested number of sites per local update.
        requested: usize,
        /// Supported number of sites per local update.
        supported: usize,
    },

    /// A numeric or algorithm option is invalid.
    #[error("invalid DMRG option {option}: {reason}")]
    InvalidOption {
        /// Option name.
        option: &'static str,
        /// Why the value is invalid.
        reason: String,
    },

    /// The requested sweep center does not exist in the state.
    #[error("DMRG center {center} is not a state node")]
    MissingCenter {
        /// Debug representation of the missing center.
        center: String,
    },

    /// Two-site DMRG cannot run because there is no update edge.
    #[error("two-site DMRG requires at least one state edge")]
    EmptyTwoSiteSweep,

    /// The operator and state do not share the same tree topology.
    #[error("DMRG requires the operator and state to have the same tree topology")]
    TopologyMismatch,

    /// The initial v1 implementation supports one physical site index per node.
    #[error("DMRG v1 requires exactly one state site index at node {node}, found {count}")]
    UnsupportedStateSiteCount {
        /// Debug representation of the node.
        node: String,
        /// Number of state site indices found.
        count: usize,
    },

    /// The initial v1 implementation supports one input/output mapping per node.
    #[error("DMRG v1 requires exactly one {role} mapping at node {node}, found {count}")]
    UnsupportedMultipleSiteMappings {
        /// Debug representation of the node.
        node: String,
        /// Mapping role, either input or output.
        role: &'static str,
        /// Number of mappings found.
        count: usize,
    },

    /// A required input/output mapping is missing.
    #[error("DMRG missing {role} mapping at node {node}")]
    MissingMapping {
        /// Debug representation of the node.
        node: String,
        /// Mapping role, either input or output.
        role: &'static str,
    },

    /// A mapping does not describe the same state vector space.
    #[error("invalid DMRG mapping at node {node}: {reason}")]
    InvalidMapping {
        /// Debug representation of the node.
        node: String,
        /// Why the mapping is invalid.
        reason: String,
    },

    /// A scalar expected to be real has a significant imaginary part.
    #[error("{context} is not real within tolerance {tolerance}: real={real}, imag={imag}")]
    NonRealScalar {
        /// Computation that produced the scalar.
        context: &'static str,
        /// Real part.
        real: f64,
        /// Imaginary part.
        imag: f64,
        /// Absolute tolerance used for the check.
        tolerance: f64,
    },

    /// A Rayleigh quotient denominator was numerically zero.
    #[error("DMRG Rayleigh quotient denominator is zero")]
    ZeroNormState,

    /// A lower-level TreeTN or Krylov operation failed.
    #[error("{context}: {source}")]
    Algorithm {
        /// Context for the failing operation.
        context: &'static str,
        /// Source error.
        #[source]
        source: anyhow::Error,
    },
}

impl From<SquareSiteMappingError> for DmrgError {
    fn from(error: SquareSiteMappingError) -> Self {
        match error {
            SquareSiteMappingError::UnsupportedStateSiteCount { node, count } => {
                Self::UnsupportedStateSiteCount { node, count }
            }
            SquareSiteMappingError::UnsupportedMultipleSiteMappings { node, role, count } => {
                Self::UnsupportedMultipleSiteMappings { node, role, count }
            }
            SquareSiteMappingError::MissingMapping { node, role } => {
                Self::MissingMapping { node, role }
            }
            SquareSiteMappingError::InvalidMapping { node, reason } => {
                Self::InvalidMapping { node, reason }
            }
        }
    }
}

/// Options for two-site TreeTN DMRG.
///
/// `DmrgOptions::default()` runs two-site sweeps with exact SVD splitting
/// unless `max_bond_dim` or `svd_policy` are set. Use `with_nsweeps` to control
/// the number of full sweeps and `with_lanczos_options` to tune the local
/// Hermitian eigensolver.
///
/// # Examples
/// ```
/// use tensor4all_treetn::DmrgOptions;
///
/// let options = DmrgOptions::default()
///     .with_nsweeps(4)
///     .with_max_bond_dim(32)
///     .with_energy_tol(1e-10);
///
/// assert_eq!(options.nsite, 2);
/// assert_eq!(options.nsweeps, 4);
/// assert_eq!(options.max_bond_dim, Some(32));
/// ```
#[derive(Debug, Clone)]
pub struct DmrgOptions {
    /// Number of sites optimized in each local update.
    ///
    /// The v1 implementation supports only `2`. `1` is rejected explicitly so
    /// callers do not accidentally run a different algorithm.
    pub nsite: usize,
    /// Number of full sweeps over the tree.
    ///
    /// Typical small validation runs use `1` to `4`; larger Hamiltonians often
    /// need more sweeps depending on the initial state.
    pub nsweeps: usize,
    /// Optional maximum bond dimension after each two-site split.
    ///
    /// `None` keeps all singular vectors allowed by the SVD truncation policy.
    pub max_bond_dim: Option<usize>,
    /// Optional SVD truncation policy applied after each local optimization.
    ///
    /// Use `SvdTruncationPolicy::new(rtol)` for relative singular-value
    /// truncation. `None` uses exact splitting except for `max_bond_dim`.
    pub svd_policy: Option<SvdTruncationPolicy>,
    /// Local Hermitian Lanczos eigensolver options.
    ///
    /// `hermitian_tol` controls projected Hermitian checks and the accepted
    /// imaginary part of projected eigenvalues.
    pub lanczos: HermitianLanczosOptions,
    /// Optional convergence tolerance for sweep-to-sweep energy changes.
    ///
    /// When `None`, all requested sweeps are run.
    pub energy_tol: Option<f64>,
    /// Absolute tolerance for checking scalar quantities that should be real.
    ///
    /// This is used for Rayleigh quotient numerator and denominator checks.
    pub real_scalar_tol: f64,
    /// Whether to print per-sweep diagnostic lines to stderr.
    pub verbose: bool,
}

impl Default for DmrgOptions {
    fn default() -> Self {
        Self {
            nsite: 2,
            nsweeps: 5,
            max_bond_dim: None,
            svd_policy: None,
            lanczos: HermitianLanczosOptions::default(),
            energy_tol: None,
            real_scalar_tol: 1e-10,
            verbose: false,
        }
    }
}

impl DmrgOptions {
    /// Set the local update size.
    ///
    /// # Examples
    /// ```
    /// use tensor4all_treetn::DmrgOptions;
    ///
    /// let options = DmrgOptions::default().with_nsite(2);
    /// assert_eq!(options.nsite, 2);
    /// ```
    pub fn with_nsite(mut self, nsite: usize) -> Self {
        self.nsite = nsite;
        self
    }

    /// Set the number of full sweeps.
    ///
    /// # Examples
    /// ```
    /// use tensor4all_treetn::DmrgOptions;
    ///
    /// let options = DmrgOptions::default().with_nsweeps(3);
    /// assert_eq!(options.nsweeps, 3);
    /// ```
    pub fn with_nsweeps(mut self, nsweeps: usize) -> Self {
        self.nsweeps = nsweeps;
        self
    }

    /// Set the maximum bond dimension after local SVD splitting.
    ///
    /// # Examples
    /// ```
    /// use tensor4all_treetn::DmrgOptions;
    ///
    /// let options = DmrgOptions::default().with_max_bond_dim(16);
    /// assert_eq!(options.max_bond_dim, Some(16));
    /// ```
    pub fn with_max_bond_dim(mut self, max_bond_dim: usize) -> Self {
        self.max_bond_dim = Some(max_bond_dim);
        self
    }

    /// Set the SVD truncation policy after local optimization.
    ///
    /// # Examples
    /// ```
    /// use tensor4all_core::SvdTruncationPolicy;
    /// use tensor4all_treetn::DmrgOptions;
    ///
    /// let policy = SvdTruncationPolicy::new(1e-9);
    /// let options = DmrgOptions::default().with_svd_policy(policy);
    /// assert_eq!(options.svd_policy, Some(policy));
    /// ```
    pub fn with_svd_policy(mut self, policy: SvdTruncationPolicy) -> Self {
        self.svd_policy = Some(policy);
        self
    }

    /// Set local Hermitian Lanczos options.
    ///
    /// # Examples
    /// ```
    /// use tensor4all_core::krylov::HermitianLanczosOptions;
    /// use tensor4all_treetn::DmrgOptions;
    ///
    /// let lanczos = HermitianLanczosOptions { max_iter: 8, ..Default::default() };
    /// let options = DmrgOptions::default().with_lanczos_options(lanczos.clone());
    /// assert_eq!(options.lanczos.max_iter, 8);
    /// ```
    pub fn with_lanczos_options(mut self, lanczos: HermitianLanczosOptions) -> Self {
        self.lanczos = lanczos;
        self
    }

    /// Set the sweep energy convergence tolerance.
    ///
    /// # Examples
    /// ```
    /// use tensor4all_treetn::DmrgOptions;
    ///
    /// let options = DmrgOptions::default().with_energy_tol(1e-8);
    /// assert_eq!(options.energy_tol, Some(1e-8));
    /// ```
    pub fn with_energy_tol(mut self, energy_tol: f64) -> Self {
        self.energy_tol = Some(energy_tol);
        self
    }

    /// Set the tolerance for real scalar checks.
    ///
    /// # Examples
    /// ```
    /// use tensor4all_treetn::DmrgOptions;
    ///
    /// let options = DmrgOptions::default().with_real_scalar_tol(1e-12);
    /// assert_eq!(options.real_scalar_tol, 1e-12);
    /// ```
    pub fn with_real_scalar_tol(mut self, tolerance: f64) -> Self {
        self.real_scalar_tol = tolerance;
        self
    }
}

/// Result returned by two-site TreeTN DMRG.
///
/// The `energy` is evaluated from the final state by a local Rayleigh quotient
/// through the projected operator, not by materializing the full tensor network.
#[derive(Debug, Clone)]
pub struct DmrgResult<T, V>
where
    T: TensorLike,
    V: Clone + Hash + Eq + Send + Sync + std::fmt::Debug,
{
    /// Optimized TreeTN state.
    pub state: TreeTN<T, V>,
    /// Final Rayleigh quotient energy.
    pub energy: f64,
    /// Number of full sweeps completed.
    pub sweeps_completed: usize,
    /// Number of local two-site updates performed.
    pub local_updates: usize,
    /// Whether the sweep-to-sweep energy tolerance was reached.
    pub converged: bool,
    /// Maximum true local residual norm reported by the Hermitian eigensolver.
    pub max_residual_norm: f64,
}

struct DmrgUpdater<T, V>
where
    T: TensorLike,
    V: Clone + Hash + Eq + Send + Sync + std::fmt::Debug,
{
    projected_operator: Arc<RwLock<ProjectedOperator<T, V>>>,
    options: DmrgOptions,
    reference_state: TreeTN<T, V>,
    local_updates: usize,
    last_energy: Option<f64>,
    max_residual_norm: f64,
}

impl<T, V> DmrgUpdater<T, V>
where
    T: TensorLike + 'static,
    T::Index: IndexLike,
    <T::Index as IndexLike>::Id:
        Clone + std::hash::Hash + Eq + Ord + std::fmt::Debug + Send + Sync + 'static,
    V: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug + 'static,
{
    fn new(
        operator: TreeTN<T, V>,
        input_mapping: HashMap<V, IndexMapping<T::Index>>,
        output_mapping: HashMap<V, IndexMapping<T::Index>>,
        options: DmrgOptions,
    ) -> Self {
        Self {
            projected_operator: Arc::new(RwLock::new(ProjectedOperator::with_index_mappings(
                operator,
                input_mapping,
                output_mapping,
            ))),
            options,
            reference_state: TreeTN::new(),
            local_updates: 0,
            last_energy: None,
            max_residual_norm: 0.0,
        }
    }

    fn solve_local(
        &mut self,
        region: &[V],
        init: &T,
        state: &TreeTN<T, V>,
    ) -> Result<T, DmrgError> {
        let linop = LocalLinOp::with_expected_input_indices(
            Arc::clone(&self.projected_operator),
            region.to_vec(),
            state,
            &self.reference_state,
            init.external_indices(),
        );

        let result = hermitian_lanczos_lowest_eigenpair(
            |x: &T| linop.apply_projected(x).map_err(anyhow::Error::from),
            init,
            &self.options.lanczos,
        )
        .map_err(|source| DmrgError::Algorithm {
            context: "DMRG local Hermitian eigensolve failed",
            source: anyhow::Error::new(source),
        })?;

        self.last_energy = Some(result.eigenvalue);
        self.max_residual_norm = self.max_residual_norm.max(result.residual_norm);
        Ok(result.eigenvector)
    }

    fn rayleigh_quotient(&self, state: &TreeTN<T, V>, region: &[V]) -> Result<f64, DmrgError> {
        let subtree = state
            .extract_subtree(region)
            .map_err(|source| DmrgError::Algorithm {
                context: "DMRG failed to extract final Rayleigh region",
                source: anyhow::Error::new(source),
            })?;
        let local = contract_region(&subtree, region).map_err(|source| DmrgError::Algorithm {
            context: "DMRG failed to contract final Rayleigh region",
            source,
        })?;
        let linop = LocalLinOp::with_expected_input_indices(
            Arc::clone(&self.projected_operator),
            region.to_vec(),
            state,
            &self.reference_state,
            local.external_indices(),
        );
        let h_local = linop
            .apply_projected(&local)
            .map_err(|source| DmrgError::Algorithm {
                context: "DMRG failed to apply final projected operator",
                source: anyhow::Error::new(source),
            })?;
        let numerator = local
            .inner_product(&h_local)
            .map_err(|source| DmrgError::Algorithm {
                context: "DMRG failed to compute Rayleigh numerator",
                source: anyhow::Error::new(source),
            })?;
        let denominator = local
            .inner_product(&local)
            .map_err(|source| DmrgError::Algorithm {
                context: "DMRG failed to compute Rayleigh denominator",
                source: anyhow::Error::new(source),
            })?;

        let numerator = checked_real_scalar(
            &numerator,
            self.options.real_scalar_tol,
            "Rayleigh numerator",
        )?;
        let denominator = checked_real_scalar(
            &denominator,
            self.options.real_scalar_tol,
            "Rayleigh denominator",
        )?;
        if denominator.abs() <= self.options.real_scalar_tol {
            return Err(DmrgError::ZeroNormState);
        }
        Ok(numerator / denominator)
    }
}

impl<T, V> LocalUpdater<T, V> for DmrgUpdater<T, V>
where
    T: TensorLike + 'static,
    T::Index: IndexLike,
    <T::Index as IndexLike>::Id:
        Clone + std::hash::Hash + Eq + Ord + std::fmt::Debug + Send + Sync + 'static,
    V: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug + 'static,
{
    fn before_step(
        &mut self,
        step: &LocalUpdateStep<V>,
        full_treetn_before: &TreeTN<T, V>,
    ) -> std::result::Result<(), TreeTNOperationError> {
        if step.nodes.len() != 2 {
            return Err(TreeTNOperationError::from(anyhow::Error::new(
                DmrgError::UnsupportedNsite {
                    requested: step.nodes.len(),
                    supported: 2,
                },
            )));
        }
        initialize_reference_state_if_empty(&mut self.reference_state, full_treetn_before)?;
        Ok(())
    }

    fn update(
        &mut self,
        mut subtree: TreeTN<T, V>,
        step: &LocalUpdateStep<V>,
        full_treetn: &TreeTN<T, V>,
    ) -> std::result::Result<TreeTN<T, V>, TreeTNOperationError> {
        let init_local = contract_region(&subtree, &step.nodes)?;
        let solved_local = self
            .solve_local(&step.nodes, &init_local, full_treetn)
            .map_err(anyhow::Error::new)?;

        let topology = build_subtree_topology(&solved_local, &step.nodes, full_treetn)?;
        let mut factorize_options = FactorizeOptions::svd();
        if let Some(max_rank) = self.options.max_bond_dim {
            factorize_options = factorize_options.with_max_rank(max_rank);
        }
        if let Some(policy) = self.options.svd_policy {
            factorize_options = factorize_options.with_svd_policy(policy);
        }

        let decomposed = factorize_tensor_to_treetn_with(
            &solved_local,
            &topology,
            factorize_options,
            &step.new_center,
        )?;
        copy_decomposed_to_subtree(&mut subtree, &decomposed, &step.nodes, full_treetn)?;

        subtree.set_canonical_region([step.new_center.clone()])?;
        if let Some(edges) = subtree.edges_to_canonicalize_by_names(&step.new_center) {
            for (from, to) in edges {
                if let Some(edge) = subtree.edge_between(&from, &to) {
                    subtree.set_edge_ortho_towards(edge, Some(to))?;
                }
            }
        }

        self.local_updates += 1;
        Ok(subtree)
    }

    fn after_step(
        &mut self,
        step: &LocalUpdateStep<V>,
        full_treetn_after: &TreeTN<T, V>,
    ) -> std::result::Result<(), TreeTNOperationError> {
        sync_reference_state_region(&mut self.reference_state, None, step, full_treetn_after)?;
        let topology = full_treetn_after.site_index_network();
        let mut projected_operator = self
            .projected_operator
            .write()
            .map_err(|err| anyhow::anyhow!("DMRG projected operator lock poisoned: {err}"))?;
        projected_operator.invalidate(&step.nodes, topology);
        Ok(())
    }
}

/// Run two-site DMRG for a TreeTN state and a mapped TreeTN linear operator.
///
/// # Arguments
///
/// * `operator` - Hamiltonian as a [`LinearOperator`]. The v1 implementation
///
///   supports exactly one input and one output site mapping per node.
/// * `init` - Initial TreeTN state. It is canonicalized at `center` before
///
///   sweeping.
/// * `center` - Initial sweep center node.
/// * `options` - Sweep, truncation, and local eigensolver options.
///
/// # Returns
///
/// A [`DmrgResult`] containing the optimized state and final Rayleigh energy.
///
/// # Errors
///
/// Returns [`DmrgError`] for unsupported options, incompatible mappings, missing
/// center nodes, non-real Rayleigh quotients, and local projected eigensolve
/// failures.
///
/// # Examples
/// ```
/// use std::collections::HashMap;
/// use tensor4all_core::{DynIndex, TensorDynLen};
/// use tensor4all_treetn::{dmrg, DmrgOptions, IndexMapping, LinearOperator, TreeTN};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let site0 = DynIndex::new_dyn(2);
/// let site1 = DynIndex::new_dyn(2);
/// let bond = DynIndex::new_dyn(1);
/// let left = TensorDynLen::from_dense(vec![site0.clone(), bond.clone()], vec![1.0, 1.0])?;
/// let right = TensorDynLen::from_dense(vec![bond.clone(), site1.clone()], vec![1.0, 1.0])?;
/// let mut state = TreeTN::<TensorDynLen, usize>::new();
/// let n0 = state.add_tensor(0, left)?;
/// let n1 = state.add_tensor(1, right)?;
/// state.connect(n0, &bond, n1, &bond)?;
///
/// let in0 = DynIndex::new_dyn(2);
/// let out0 = DynIndex::new_dyn(2);
/// let in1 = DynIndex::new_dyn(2);
/// let out1 = DynIndex::new_dyn(2);
/// let op_bond = DynIndex::new_dyn(1);
/// let mut id = vec![0.0; 4];
/// id[0] = 1.0;
/// id[3] = 1.0;
/// let op0 = TensorDynLen::from_dense(vec![out0.clone(), in0.clone(), op_bond.clone()], id.clone())?;
/// let op1 = TensorDynLen::from_dense(vec![op_bond.clone(), out1.clone(), in1.clone()], id)?;
/// let mut mpo = TreeTN::<TensorDynLen, usize>::new();
/// let m0 = mpo.add_tensor(0, op0)?;
/// let m1 = mpo.add_tensor(1, op1)?;
/// mpo.connect(m0, &op_bond, m1, &op_bond)?;
///
/// let input = HashMap::from([
///     (0, IndexMapping { true_index: site0.clone(), internal_index: in0 }),
///     (1, IndexMapping { true_index: site1.clone(), internal_index: in1 }),
/// ]);
/// let output = HashMap::from([
///     (0, IndexMapping { true_index: site0, internal_index: out0 }),
///     (1, IndexMapping { true_index: site1, internal_index: out1 }),
/// ]);
/// let operator = LinearOperator::new(mpo, input, output);
///
/// let result = dmrg(&operator, state, &0, DmrgOptions::default().with_nsweeps(1))?;
/// assert!((result.energy - 1.0).abs() < 1e-10);
/// # Ok(())
/// # }
/// ```
pub fn dmrg<T, V>(
    operator: &LinearOperator<T, V>,
    init: TreeTN<T, V>,
    center: &V,
    options: DmrgOptions,
) -> std::result::Result<DmrgResult<T, V>, DmrgError>
where
    T: TensorLike + 'static,
    T::Index: IndexLike,
    <T::Index as IndexLike>::Id:
        Clone + std::hash::Hash + Eq + Ord + std::fmt::Debug + Send + Sync + 'static,
    V: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug + 'static,
{
    validate_options(&options)?;
    if init.node_index(center).is_none() {
        return Err(DmrgError::MissingCenter {
            center: format!("{center:?}"),
        });
    }
    if !init.same_topology(operator.mpo()) {
        return Err(DmrgError::TopologyMismatch);
    }

    let (input_mapping, output_mapping) =
        single_site_square_mappings(operator, &init).map_err(DmrgError::from)?;
    let mut state = init
        .canonicalize([center.clone()], CanonicalizationOptions::default())
        .map_err(|source| DmrgError::Algorithm {
            context: "DMRG failed to canonicalize initial state",
            source: anyhow::Error::new(source),
        })?;

    let plan =
        LocalUpdateSweepPlan::from_treetn(&state, center, options.nsite).ok_or_else(|| {
            DmrgError::MissingCenter {
                center: format!("{center:?}"),
            }
        })?;
    if plan.is_empty() {
        return Err(DmrgError::EmptyTwoSiteSweep);
    }

    let mut updater = DmrgUpdater::new(
        operator.mpo().clone(),
        input_mapping,
        output_mapping,
        options.clone(),
    );
    let mut previous_energy: Option<f64> = None;
    let mut converged = false;
    let mut sweeps_completed = 0usize;

    for sweep in 0..options.nsweeps {
        apply_local_update_sweep(&mut state, &plan, &mut updater).map_err(|source| {
            DmrgError::Algorithm {
                context: "DMRG sweep failed",
                source: anyhow::Error::new(source),
            }
        })?;
        sweeps_completed = sweep + 1;

        if let Some(energy_tol) = options.energy_tol {
            let energy = updater.last_energy.ok_or_else(|| DmrgError::Algorithm {
                context: "DMRG sweep produced no local energy",
                source: anyhow::anyhow!("no local DMRG update was completed"),
            })?;
            if let Some(previous) = previous_energy {
                if (energy - previous).abs() <= energy_tol {
                    converged = true;
                    break;
                }
            }
            previous_energy = Some(energy);
        }

        if options.verbose {
            eprintln!(
                "DMRG sweep {}: local_energy={:?} max_residual={:.6e}",
                sweeps_completed, updater.last_energy, updater.max_residual_norm
            );
        }
    }

    let final_region = rayleigh_region(&state, center)?;
    let energy = updater.rayleigh_quotient(&state, &final_region)?;

    Ok(DmrgResult {
        state,
        energy,
        sweeps_completed,
        local_updates: updater.local_updates,
        converged,
        max_residual_norm: updater.max_residual_norm,
    })
}

/// Build a mapped operator from a TreeTN MPO and run two-site DMRG.
///
/// This is a convenience wrapper over [`LinearOperator::from_mpo_and_state`] and
/// [`dmrg`]. Prefer explicit [`LinearOperator`] mappings when multiple same-dim
/// operator site indices make input/output conventions ambiguous.
///
/// # Errors
///
/// Returns an error when the solve or sweep fails (a shape or index
/// /// mismatch, a non-convergence failure, or a backend failure).
///
/// # Examples
/// ```
/// use tensor4all_core::{DynIndex, TensorDynLen};
/// use tensor4all_treetn::{dmrg_with_treetn_operator, DmrgOptions, TreeTN};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let site = DynIndex::new_dyn(2);
/// let state_tensor = TensorDynLen::from_dense(vec![site.clone()], vec![1.0, 0.0])?;
/// let state = TreeTN::<TensorDynLen, usize>::from_tensors(vec![state_tensor], vec![0])?;
///
/// let op_in = DynIndex::new_dyn(2);
/// let op_out = DynIndex::new_dyn(2);
/// let op_tensor = TensorDynLen::from_dense(
///     vec![op_out, op_in],
///     vec![1.0, 0.0, 0.0, 2.0],
/// )?;
/// let operator = TreeTN::<TensorDynLen, usize>::from_tensors(vec![op_tensor], vec![0])?;
///
/// let err = dmrg_with_treetn_operator(&operator, state, &0, DmrgOptions::default()).unwrap_err();
/// assert!(matches!(err, tensor4all_treetn::DmrgError::EmptyTwoSiteSweep));
/// # Ok(())
/// # }
/// ```
pub fn dmrg_with_treetn_operator<T, V>(
    operator: &TreeTN<T, V>,
    init: TreeTN<T, V>,
    center: &V,
    options: DmrgOptions,
) -> std::result::Result<DmrgResult<T, V>, DmrgError>
where
    T: TensorLike + 'static,
    T::Index: IndexLike,
    <T::Index as IndexLike>::Id:
        Clone + std::hash::Hash + Eq + Ord + std::fmt::Debug + Send + Sync + 'static,
    V: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug + 'static,
{
    let linear_operator =
        LinearOperator::from_mpo_and_state(operator.clone(), &init).map_err(|source| {
            DmrgError::Algorithm {
                context: "DMRG failed to build LinearOperator from TreeTN operator",
                source: anyhow::Error::new(source),
            }
        })?;
    dmrg(&linear_operator, init, center, options)
}

fn validate_options(options: &DmrgOptions) -> Result<(), DmrgError> {
    if options.nsite != 2 {
        return Err(DmrgError::UnsupportedNsite {
            requested: options.nsite,
            supported: 2,
        });
    }
    if options.nsweeps == 0 {
        return Err(DmrgError::InvalidOption {
            option: "nsweeps",
            reason: "must be greater than zero".to_string(),
        });
    }
    if let Some(max_bond_dim) = options.max_bond_dim {
        if max_bond_dim == 0 {
            return Err(DmrgError::InvalidOption {
                option: "max_bond_dim",
                reason: "must be greater than zero when set".to_string(),
            });
        }
    }
    if let Some(energy_tol) = options.energy_tol {
        if !energy_tol.is_finite() || energy_tol < 0.0 {
            return Err(DmrgError::InvalidOption {
                option: "energy_tol",
                reason: "must be finite and non-negative".to_string(),
            });
        }
    }
    if !options.real_scalar_tol.is_finite() || options.real_scalar_tol < 0.0 {
        return Err(DmrgError::InvalidOption {
            option: "real_scalar_tol",
            reason: "must be finite and non-negative".to_string(),
        });
    }
    Ok(())
}

fn checked_real_scalar(
    value: &AnyScalar,
    tolerance: f64,
    context: &'static str,
) -> Result<f64, DmrgError> {
    let imag = value.imag();
    if imag.abs() > tolerance {
        return Err(DmrgError::NonRealScalar {
            context,
            real: value.real(),
            imag,
            tolerance,
        });
    }
    Ok(value.real())
}

fn rayleigh_region<T, V>(state: &TreeTN<T, V>, center: &V) -> Result<Vec<V>, DmrgError>
where
    T: TensorLike,
    V: Clone + Hash + Eq + Send + Sync + std::fmt::Debug,
{
    let neighbor = state
        .site_index_network()
        .neighbors(center)
        .next()
        .ok_or(DmrgError::EmptyTwoSiteSweep)?;
    Ok(vec![center.clone(), neighbor])
}
