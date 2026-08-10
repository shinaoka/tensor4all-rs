//! Time-dependent variational principle sweeps for TreeTN states.
//!
//! This module implements TreeTN TDVP with ITensorNetworks-compatible
//! `applyexp` region plans. Production updates use projected local operators
//! and Krylov exponentials; dense full-network materialization is reserved for
//! tests and external benchmark validation.

mod plan;

use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt::Debug;
use std::hash::Hash;
use std::sync::{Arc, OnceLock, RwLock};
use std::time::{Duration, Instant};

use num_complex::Complex64;
use tensor4all_core::{
    krylov::{hermitian_krylov_expm_multiply, HermitianKrylovExpmOptions},
    print_and_reset_contract_profile, print_and_reset_pairwise_contract_profile,
    reset_contract_profile, reset_pairwise_contract_profile, Canonical, FactorizeAlg,
    FactorizeOptions, IndexLike, SvdTruncationPolicy, TensorLike,
};
use thiserror::Error;

use self::plan::{TdvpRegionKind, TdvpRegionPlan, TdvpRegionStep};
use crate::linsolve::common::{CachedTopology, NetworkTopology, ProjectedOperator};
use crate::linsolve::square::local_linop::LocalLinOp;
use crate::local_update_support::{
    build_subtree_topology, contract_region, copy_decomposed_to_subtree,
    initialize_reference_state_if_empty, single_site_square_mappings, sync_reference_state_region,
    SquareSiteMappingError,
};
use crate::operator::{IndexMapping, LinearOperator};
use crate::{factorize_tensor_to_treetn_with, CanonicalizationOptions, LocalUpdateStep, TreeTN};

/// Errors reported by TreeTN TDVP.
///
/// This error type covers unsupported v1 API shapes, invalid options, topology
/// mismatches, and numerical failures from local Hermitian Krylov exponentials.
#[derive(Debug, Error)]
pub enum TdvpError {
    /// The requested local update size is not implemented.
    #[error("TDVP supports nsite=1 or nsite=2 in this version, got nsite={requested}")]
    UnsupportedNsite {
        /// Requested number of sites per local update.
        requested: usize,
    },

    /// The requested Suzuki-Trotter order is not implemented.
    #[error("TDVP applyexp order {requested} is not supported; supported orders are 1, 2, and 4")]
    UnsupportedOrder {
        /// Requested applyexp order.
        requested: usize,
    },

    /// A numeric or algorithm option is invalid.
    #[error("invalid TDVP option {option}: {reason}")]
    InvalidOption {
        /// Option name.
        option: &'static str,
        /// Why the value is invalid.
        reason: String,
    },

    /// The requested sweep center does not exist in the state.
    #[error("TDVP center {center} is not a state node")]
    MissingCenter {
        /// Debug representation of the missing center.
        center: String,
    },

    /// Two-site TDVP cannot run because there is no update edge.
    #[error("two-site TDVP requires at least one state edge")]
    EmptyTwoSiteSweep,

    /// The operator and state do not share the same tree topology.
    #[error("TDVP requires the operator and state to have the same tree topology")]
    TopologyMismatch,

    /// The initial v1 implementation supports one physical site index per node.
    #[error("TDVP v1 requires exactly one state site index at node {node}, found {count}")]
    UnsupportedStateSiteCount {
        /// Debug representation of the node.
        node: String,
        /// Number of state site indices found.
        count: usize,
    },

    /// The initial v1 implementation supports one input/output mapping per node.
    #[error("TDVP v1 requires exactly one {role} mapping at node {node}, found {count}")]
    UnsupportedMultipleSiteMappings {
        /// Debug representation of the node.
        node: String,
        /// Mapping role, either input or output.
        role: &'static str,
        /// Number of mappings found.
        count: usize,
    },

    /// A required input/output mapping is missing.
    #[error("TDVP missing {role} mapping at node {node}")]
    MissingMapping {
        /// Debug representation of the node.
        node: String,
        /// Mapping role, either input or output.
        role: &'static str,
    },

    /// A mapping does not describe the same state vector space.
    #[error("invalid TDVP mapping at node {node}: {reason}")]
    InvalidMapping {
        /// Debug representation of the node.
        node: String,
        /// Why the mapping is invalid.
        reason: String,
    },

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

impl From<SquareSiteMappingError> for TdvpError {
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

#[derive(Debug, Default, Clone)]
struct TdvpProfile {
    total: Duration,
    initial_mapping: Duration,
    initial_canonicalize: Duration,
    plan: Duration,
    move_center: Duration,
    reference_init: Duration,
    reference_move: Duration,
    two_site_total: Duration,
    one_site_total: Duration,
    extract_region: Duration,
    contract_region: Duration,
    evolve_local: Duration,
    projected_apply: Duration,
    build_topology: Duration,
    factorize: Duration,
    copy_decomposed: Duration,
    subtree_center: Duration,
    subtree_ortho: Duration,
    replace_subtree: Duration,
    state_center: Duration,
    sync_reference: Duration,
    invalidate: Duration,
    steps: usize,
    two_site_steps: usize,
    one_site_steps: usize,
    projected_apply_calls: usize,
}

thread_local! {
    static TDVP_PROFILE: RefCell<Option<TdvpProfile>> = const { RefCell::new(None) };
}

fn tdvp_profile_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("T4A_PROFILE_TDVP").is_some())
}

fn tdvp_profile_reset() {
    if tdvp_profile_enabled() {
        TDVP_PROFILE.with(|profile| *profile.borrow_mut() = Some(TdvpProfile::default()));
    }
}

fn with_tdvp_profile(f: impl FnOnce(&mut TdvpProfile)) {
    if tdvp_profile_enabled() {
        TDVP_PROFILE.with(|profile| {
            if let Some(profile) = profile.borrow_mut().as_mut() {
                f(profile);
            }
        });
    }
}

fn take_tdvp_profile() -> Option<TdvpProfile> {
    if !tdvp_profile_enabled() {
        return None;
    }
    TDVP_PROFILE.with(|profile| profile.borrow_mut().take())
}

fn record_tdvp_profile(elapsed: Option<Instant>, f: impl FnOnce(&mut TdvpProfile, Duration)) {
    if let Some(started) = elapsed {
        let elapsed = started.elapsed();
        with_tdvp_profile(|profile| f(profile, elapsed));
    }
}

fn print_tdvp_profile(profile: &TdvpProfile) {
    eprintln!("=== TreeTN TDVP profile ===");
    eprintln!("total:              {:?}", profile.total);
    eprintln!(
        "steps:              {} two_site={} one_site/site_correction={} projected_apply_calls={}",
        profile.steps,
        profile.two_site_steps,
        profile.one_site_steps,
        profile.projected_apply_calls
    );
    eprintln!("initial mapping:    {:?}", profile.initial_mapping);
    eprintln!("initial canon:      {:?}", profile.initial_canonicalize);
    eprintln!("plan:               {:?}", profile.plan);
    eprintln!("move center:        {:?}", profile.move_center);
    eprintln!("reference init:     {:?}", profile.reference_init);
    eprintln!("reference move:     {:?}", profile.reference_move);
    eprintln!("two-site total:     {:?}", profile.two_site_total);
    eprintln!("one-site total:     {:?}", profile.one_site_total);
    eprintln!("extract region:     {:?}", profile.extract_region);
    eprintln!("contract region:    {:?}", profile.contract_region);
    eprintln!("evolve local:       {:?}", profile.evolve_local);
    eprintln!("projected apply:    {:?}", profile.projected_apply);
    eprintln!("build topology:     {:?}", profile.build_topology);
    eprintln!("factorize:          {:?}", profile.factorize);
    eprintln!("copy decomposed:    {:?}", profile.copy_decomposed);
    eprintln!("subtree center:     {:?}", profile.subtree_center);
    eprintln!("subtree ortho:      {:?}", profile.subtree_ortho);
    eprintln!("replace subtree:    {:?}", profile.replace_subtree);
    eprintln!("state center:       {:?}", profile.state_center);
    eprintln!("sync reference:     {:?}", profile.sync_reference);
    eprintln!("invalidate:         {:?}", profile.invalidate);
}

/// Options for TreeTN TDVP.
///
/// `TdvpOptions::default()` runs one second-order two-site TDVP sweep with a
/// Krylov exponential. The `exponent_step` is the coefficient multiplying the
/// Hamiltonian in `exp(exponent_step * H)`, so real-time evolution for a time
/// step `dt` uses `Complex64::new(0.0, -dt)`.
///
/// # Examples
/// ```
/// use num_complex::Complex64;
/// use tensor4all_treetn::TdvpOptions;
///
/// let options = TdvpOptions::default()
///     .with_nsite(2)
///     .with_order(4)
///     .with_exponent_step(Complex64::new(0.0, -0.05));
///
/// assert_eq!(options.nsite, 2);
/// assert_eq!(options.order, 4);
/// assert_eq!(options.exponent_step, Complex64::new(0.0, -0.05));
/// ```
#[derive(Debug, Clone)]
pub struct TdvpOptions {
    /// Number of sites in each projected local update.
    ///
    /// Use `2` to allow bond-dimension changes through two-site factorization.
    /// Use `1` for fixed-rank one-site TDVP.
    pub nsite: usize,
    /// Number of full applyexp sweeps.
    ///
    /// Each sweep applies `exponent_step` once, split according to `order`.
    pub nsweeps: usize,
    /// Suzuki-Trotter/applyexp composition order.
    ///
    /// Supported values match ITensorNetworks.jl: `1`, `2`, and `4`.
    pub order: usize,
    /// Coefficient multiplying the Hamiltonian in the exponential.
    ///
    /// For real time `dt`, use `-i dt`; for imaginary time `tau`, use `-tau`.
    pub exponent_step: Complex64,
    /// Optional maximum bond dimension after each two-site split.
    ///
    /// `None` keeps all singular vectors allowed by the SVD truncation policy.
    /// This option is valid only for `nsite = 2`; one-site TDVP has fixed
    /// ranks and rejects truncation options.
    pub max_bond_dim: Option<usize>,
    /// Optional SVD truncation policy for two-site splits.
    ///
    /// For ITensors cutoff parity, use `SvdTruncationPolicy::new(cutoff)` with
    /// `.with_squared_values()` and `.with_discarded_tail_sum()`. This option
    /// is valid only for `nsite = 2`; one-site TDVP has fixed ranks and
    /// rejects truncation options.
    pub svd_policy: Option<SvdTruncationPolicy>,
    /// Local Hermitian Krylov exponential options.
    pub krylov: HermitianKrylovExpmOptions,
    /// Whether to print per-sweep progress to stderr.
    pub verbose: bool,
}

impl Default for TdvpOptions {
    fn default() -> Self {
        Self {
            nsite: 2,
            nsweeps: 1,
            order: 2,
            exponent_step: Complex64::new(0.0, -0.1),
            max_bond_dim: None,
            svd_policy: None,
            krylov: HermitianKrylovExpmOptions::default(),
            verbose: false,
        }
    }
}

impl TdvpOptions {
    /// Set the number of sites in each projected update.
    ///
    /// # Examples
    /// ```
    /// use tensor4all_treetn::TdvpOptions;
    ///
    /// let options = TdvpOptions::default().with_nsite(1);
    /// assert_eq!(options.nsite, 1);
    /// ```
    pub fn with_nsite(mut self, nsite: usize) -> Self {
        self.nsite = nsite;
        self
    }

    /// Set the number of full TDVP sweeps.
    ///
    /// # Examples
    /// ```
    /// use tensor4all_treetn::TdvpOptions;
    ///
    /// let options = TdvpOptions::default().with_nsweeps(3);
    /// assert_eq!(options.nsweeps, 3);
    /// ```
    pub fn with_nsweeps(mut self, nsweeps: usize) -> Self {
        self.nsweeps = nsweeps;
        self
    }

    /// Set the applyexp composition order.
    ///
    /// # Examples
    /// ```
    /// use tensor4all_treetn::TdvpOptions;
    ///
    /// let options = TdvpOptions::default().with_order(4);
    /// assert_eq!(options.order, 4);
    /// ```
    pub fn with_order(mut self, order: usize) -> Self {
        self.order = order;
        self
    }

    /// Set the exponential step coefficient.
    ///
    /// # Examples
    /// ```
    /// use num_complex::Complex64;
    /// use tensor4all_treetn::TdvpOptions;
    ///
    /// let options = TdvpOptions::default().with_exponent_step(Complex64::new(0.0, -0.25));
    /// assert_eq!(options.exponent_step, Complex64::new(0.0, -0.25));
    /// ```
    pub fn with_exponent_step(mut self, exponent_step: Complex64) -> Self {
        self.exponent_step = exponent_step;
        self
    }

    /// Set the maximum retained bond dimension for two-site splits.
    ///
    /// # Examples
    /// ```
    /// use tensor4all_treetn::TdvpOptions;
    ///
    /// let options = TdvpOptions::default().with_max_bond_dim(16);
    /// assert_eq!(options.max_bond_dim, Some(16));
    /// ```
    pub fn with_max_bond_dim(mut self, max_bond_dim: usize) -> Self {
        self.max_bond_dim = Some(max_bond_dim);
        self
    }

    /// Set the SVD truncation policy for two-site splits.
    ///
    /// # Examples
    /// ```
    /// use tensor4all_core::SvdTruncationPolicy;
    /// use tensor4all_treetn::TdvpOptions;
    ///
    /// let options = TdvpOptions::default().with_svd_policy(SvdTruncationPolicy::new(1e-8));
    /// assert!(options.svd_policy.is_some());
    /// ```
    pub fn with_svd_policy(mut self, policy: SvdTruncationPolicy) -> Self {
        self.svd_policy = Some(policy);
        self
    }

    /// Set local Hermitian Krylov exponential options.
    ///
    /// # Examples
    /// ```
    /// use tensor4all_core::krylov::HermitianKrylovExpmOptions;
    /// use tensor4all_treetn::TdvpOptions;
    ///
    /// let krylov = HermitianKrylovExpmOptions { max_iter: 10, ..Default::default() };
    /// let options = TdvpOptions::default().with_krylov_options(krylov);
    /// assert_eq!(options.krylov.max_iter, 10);
    /// ```
    pub fn with_krylov_options(mut self, krylov: HermitianKrylovExpmOptions) -> Self {
        self.krylov = krylov;
        self
    }
}

/// Result returned by TreeTN TDVP.
///
/// The evolved state remains a TreeTN and is never produced by dense
/// full-network materialization inside the production algorithm.
#[derive(Debug, Clone)]
pub struct TdvpResult<T, V>
where
    T: TensorLike,
    V: Clone + Hash + Eq + Send + Sync + Debug,
{
    /// Evolved TreeTN state.
    pub state: TreeTN<T, V>,
    /// Number of full sweeps completed.
    pub sweeps_completed: usize,
    /// Number of projected local exponential updates performed.
    pub local_updates: usize,
    /// Largest Krylov residual estimate reported by local updates.
    pub max_error_estimate: f64,
    /// Largest number of Krylov iterations used by a local update.
    pub max_krylov_iterations: usize,
}

struct TdvpUpdater<T, V>
where
    T: TensorLike,
    V: Clone + Hash + Eq + Send + Sync + Debug,
{
    projected_operator: Arc<RwLock<ProjectedOperator<T, V>>>,
    topology: CachedTopology<V>,
    options: TdvpOptions,
    reference_state: TreeTN<T, V>,
    local_updates: usize,
    max_error_estimate: f64,
    max_krylov_iterations: usize,
}

impl<T, V> TdvpUpdater<T, V>
where
    T: TensorLike + 'static,
    T::Index: IndexLike,
    <T::Index as IndexLike>::Id: Clone + std::hash::Hash + Eq + Ord + Debug + Send + Sync + 'static,
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug + 'static,
{
    fn new(
        operator: TreeTN<T, V>,
        input_mapping: HashMap<V, IndexMapping<T::Index>>,
        output_mapping: HashMap<V, IndexMapping<T::Index>>,
        topology: CachedTopology<V>,
        options: TdvpOptions,
    ) -> Self {
        Self {
            projected_operator: Arc::new(RwLock::new(ProjectedOperator::with_index_mappings(
                operator,
                input_mapping,
                output_mapping,
            ))),
            topology,
            options,
            reference_state: TreeTN::new(),
            local_updates: 0,
            max_error_estimate: 0.0,
            max_krylov_iterations: 0,
        }
    }

    fn evolve_local(
        &mut self,
        region: &[V],
        init: &T,
        state: &TreeTN<T, V>,
        exponent_step: Complex64,
        context: &'static str,
    ) -> Result<T, TdvpError> {
        let linop = LocalLinOp::with_expected_input_indices(
            Arc::clone(&self.projected_operator),
            region.to_vec(),
            state,
            &self.reference_state,
            init.external_indices(),
        );

        let mut projected_operator =
            self.projected_operator
                .write()
                .map_err(|err| TdvpError::Algorithm {
                    context: "TDVP failed to lock projected operator",
                    source: anyhow::anyhow!("{err}"),
                })?;
        let evolve_started = tdvp_profile_enabled().then(Instant::now);
        let result = hermitian_krylov_expm_multiply(
            |x: &T| {
                let apply_started = tdvp_profile_enabled().then(Instant::now);
                let result =
                    linop.apply_projected_with_operator(x, &mut projected_operator, &self.topology);
                record_tdvp_profile(apply_started, |profile, elapsed| {
                    profile.projected_apply += elapsed;
                    profile.projected_apply_calls += 1;
                });
                result
            },
            exponent_step,
            init,
            &self.options.krylov,
        )
        .map_err(|source| TdvpError::Algorithm {
            context,
            source: anyhow::Error::new(source),
        })?;
        record_tdvp_profile(evolve_started, |profile, elapsed| {
            profile.evolve_local += elapsed;
        });

        self.max_error_estimate = self.max_error_estimate.max(result.error_estimate);
        self.max_krylov_iterations = self.max_krylov_iterations.max(result.iterations);
        self.local_updates += 1;
        Ok(result.output)
    }

    fn update_step(
        &mut self,
        state: &mut TreeTN<T, V>,
        step: &TdvpRegionStep<V>,
        next_step: Option<&TdvpRegionStep<V>>,
    ) -> Result<(), TdvpError> {
        with_tdvp_profile(|profile| {
            profile.steps += 1;
            match step.kind {
                TdvpRegionKind::TwoSite => profile.two_site_steps += 1,
                TdvpRegionKind::OneSite | TdvpRegionKind::SiteCorrection => {
                    profile.one_site_steps += 1;
                }
            }
        });
        let reference_init_started = tdvp_profile_enabled().then(Instant::now);
        initialize_reference_state_if_empty(&mut self.reference_state, state).map_err(
            |source| TdvpError::Algorithm {
                context: "TDVP failed to initialize reference state",
                source,
            },
        )?;
        record_tdvp_profile(reference_init_started, |profile, elapsed| {
            profile.reference_init += elapsed;
        });
        let reference_move_started = tdvp_profile_enabled().then(Instant::now);
        move_center_to_region_full_rank(&mut self.reference_state, &step.nodes).map_err(
            |source| TdvpError::Algorithm {
                context: "TDVP failed to move reference center to local region",
                source,
            },
        )?;
        record_tdvp_profile(reference_move_started, |profile, elapsed| {
            profile.reference_move += elapsed;
        });

        let update_started = tdvp_profile_enabled().then(Instant::now);
        match step.kind {
            TdvpRegionKind::TwoSite => self.update_two_site(state, step),
            TdvpRegionKind::SiteCorrection => self.update_one_site_tensor(
                state,
                step,
                None,
                "TDVP single-site correction Krylov exponential failed",
            ),
            TdvpRegionKind::OneSite => self.update_one_site_tensor(
                state,
                step,
                next_step,
                "TDVP one-site Krylov exponential failed",
            ),
        }?;
        record_tdvp_profile(update_started, |profile, elapsed| match step.kind {
            TdvpRegionKind::TwoSite => profile.two_site_total += elapsed,
            TdvpRegionKind::OneSite | TdvpRegionKind::SiteCorrection => {
                profile.one_site_total += elapsed;
            }
        });

        let sync_started = tdvp_profile_enabled().then(Instant::now);
        sync_reference_state_region(
            &mut self.reference_state,
            None,
            &LocalUpdateStep {
                nodes: step.nodes.clone(),
                new_center: step.new_center.clone(),
            },
            state,
        )
        .map_err(|source| TdvpError::Algorithm {
            context: "TDVP failed to sync reference state",
            source,
        })?;
        record_tdvp_profile(sync_started, |profile, elapsed| {
            profile.sync_reference += elapsed;
        });

        let invalidate_started = tdvp_profile_enabled().then(Instant::now);
        let mut projected_operator =
            self.projected_operator
                .write()
                .map_err(|err| TdvpError::Algorithm {
                    context: "TDVP projected operator lock poisoned",
                    source: anyhow::anyhow!("{err}"),
                })?;
        projected_operator.invalidate(&step.nodes, &self.topology);
        record_tdvp_profile(invalidate_started, |profile, elapsed| {
            profile.invalidate += elapsed;
        });
        Ok(())
    }

    fn update_one_site_tensor(
        &mut self,
        state: &mut TreeTN<T, V>,
        step: &TdvpRegionStep<V>,
        next_step: Option<&TdvpRegionStep<V>>,
        context: &'static str,
    ) -> Result<(), TdvpError> {
        let extract_started = tdvp_profile_enabled().then(Instant::now);
        let subtree =
            state
                .extract_subtree(&step.nodes)
                .map_err(|source| TdvpError::Algorithm {
                    context: "TDVP failed to extract one-site region",
                    source,
                })?;
        record_tdvp_profile(extract_started, |profile, elapsed| {
            profile.extract_region += elapsed;
        });
        let contract_started = tdvp_profile_enabled().then(Instant::now);
        let init_local =
            contract_region(&subtree, &step.nodes).map_err(|source| TdvpError::Algorithm {
                context: "TDVP failed to contract one-site region",
                source,
            })?;
        record_tdvp_profile(contract_started, |profile, elapsed| {
            profile.contract_region += elapsed;
        });
        let evolved =
            self.evolve_local(&step.nodes, &init_local, state, step.exponent_step, context)?;
        let evolved = if step.kind == TdvpRegionKind::OneSite {
            self.apply_one_site_bond_correction(state, step, next_step, evolved)?
        } else {
            evolved
        };
        let node_idx =
            state
                .node_index(&step.nodes[0])
                .ok_or_else(|| TdvpError::MissingCenter {
                    center: format!("{:?}", step.nodes[0]),
                })?;
        let replace_started = tdvp_profile_enabled().then(Instant::now);
        state
            .replace_tensor(node_idx, evolved)
            .map_err(|source| TdvpError::Algorithm {
                context: "TDVP failed to replace one-site tensor",
                source,
            })?;
        record_tdvp_profile(replace_started, |profile, elapsed| {
            profile.replace_subtree += elapsed;
        });
        let center_started = tdvp_profile_enabled().then(Instant::now);
        state
            .set_canonical_region([step.new_center.clone()])
            .map_err(|source| TdvpError::Algorithm {
                context: "TDVP failed to set one-site canonical center",
                source,
            })?;
        record_tdvp_profile(center_started, |profile, elapsed| {
            profile.state_center += elapsed;
        });
        Ok(())
    }

    fn apply_one_site_bond_correction(
        &mut self,
        state: &TreeTN<T, V>,
        step: &TdvpRegionStep<V>,
        next_step: Option<&TdvpRegionStep<V>>,
        evolved: T,
    ) -> Result<T, TdvpError> {
        let Some(next_step) = next_step else {
            return Ok(evolved);
        };
        if next_step.kind != TdvpRegionKind::OneSite || next_step.nodes == step.nodes {
            return Ok(evolved);
        }
        let current = step.nodes.first().ok_or(TdvpError::UnsupportedNsite {
            requested: step.nodes.len(),
        })?;
        let target = next_step.nodes.first().ok_or(TdvpError::UnsupportedNsite {
            requested: next_step.nodes.len(),
        })?;
        let Some(next_neighbor) =
            first_path_neighbor(state, current, target).map_err(|source| TdvpError::Algorithm {
                context: "TDVP failed to find one-site correction edge",
                source,
            })?
        else {
            return Ok(evolved);
        };

        let edge =
            state
                .edge_between(current, &next_neighbor)
                .ok_or_else(|| TdvpError::Algorithm {
                    context: "TDVP one-site correction edge is missing",
                    source: anyhow::anyhow!(
                        "missing edge between {:?} and {:?}",
                        current,
                        next_neighbor
                    ),
                })?;
        let ket_edge_bond =
            state
                .bond_index(edge)
                .cloned()
                .ok_or_else(|| TdvpError::Algorithm {
                    context: "TDVP one-site correction edge has no bond",
                    source: anyhow::anyhow!(
                        "missing bond between {:?} and {:?}",
                        current,
                        next_neighbor
                    ),
                })?;
        let left_inds = evolved
            .external_indices()
            .into_iter()
            .filter(|idx| idx != &ket_edge_bond)
            .collect::<Vec<_>>();
        let factorized = evolved
            .factorize_full_rank(&left_inds, FactorizeAlg::QR, Canonical::Left)
            .map_err(|source| TdvpError::Algorithm {
                context: "TDVP one-site QR factorization failed",
                source: source.into(),
            })?;
        let q = factorized.left;
        let r = factorized.right;
        let ket_center_bond = factorized.bond_index;
        let ref_center_bond = ket_center_bond.sim();
        let mut q_ref = q
            .replaceind(&ket_center_bond, &ref_center_bond)
            .map_err(|source| TdvpError::Algorithm {
                context: "TDVP failed to relabel one-site Q reference bond",
                source: anyhow::Error::new(source),
            })?;

        for neighbor in self.topology.neighbors(current) {
            let Some(ket_edge) = state.edge_between(current, &neighbor) else {
                continue;
            };
            let Some(ket_bond) = state.bond_index(ket_edge) else {
                continue;
            };
            if !q_ref.external_indices().iter().any(|idx| idx == ket_bond) {
                continue;
            }
            let Some(ref_edge) = self.reference_state.edge_between(current, &neighbor) else {
                continue;
            };
            let Some(ref_bond) = self.reference_state.bond_index(ref_edge) else {
                continue;
            };
            q_ref =
                q_ref
                    .replaceind(ket_bond, ref_bond)
                    .map_err(|source| TdvpError::Algorithm {
                        context: "TDVP failed to relabel one-site Q boundary bond",
                        source: anyhow::Error::new(source),
                    })?;
        }

        let ref_edge = self
            .reference_state
            .edge_between(current, &next_neighbor)
            .ok_or_else(|| TdvpError::Algorithm {
                context: "TDVP reference state is missing one-site correction edge",
                source: anyhow::anyhow!(
                    "missing reference edge between {:?} and {:?}",
                    current,
                    next_neighbor
                ),
            })?;
        let ref_edge_bond = self
            .reference_state
            .bond_index(ref_edge)
            .cloned()
            .ok_or_else(|| TdvpError::Algorithm {
                context: "TDVP reference one-site correction edge has no bond",
                source: anyhow::anyhow!(
                    "missing reference bond between {:?} and {:?}",
                    current,
                    next_neighbor
                ),
            })?;
        let bra_to_ket = vec![
            (ref_center_bond, ket_center_bond),
            (ref_edge_bond, ket_edge_bond),
        ];
        let r_evolved = self.evolve_edge_center(
            state,
            current,
            &next_neighbor,
            &q,
            &q_ref,
            &bra_to_ket,
            &r,
            -step.exponent_step,
        )?;
        let mut local = T::contract(&[&q, &r_evolved]).map_err(|source| TdvpError::Algorithm {
            context: "TDVP failed to contract corrected one-site tensor",
            source: anyhow::Error::new(source),
        })?;
        let target_inds = evolved.external_indices();
        let local_inds = local.external_indices();
        let target_keys: std::collections::HashSet<_> = target_inds
            .iter()
            .map(|idx| (idx.clone(), idx.dim()))
            .collect();
        let local_keys: std::collections::HashSet<_> = local_inds
            .iter()
            .map(|idx| (idx.clone(), idx.dim()))
            .collect();
        if target_keys == local_keys && target_inds.len() == local_inds.len() {
            local = local
                .permuteinds(&target_inds)
                .map_err(|source| TdvpError::Algorithm {
                    context: "TDVP failed to align corrected one-site tensor indices",
                    source: anyhow::Error::new(source),
                })?;
        }
        Ok(local)
    }

    #[allow(clippy::too_many_arguments)]
    fn evolve_edge_center(
        &mut self,
        state: &TreeTN<T, V>,
        left: &V,
        right: &V,
        left_ket_tensor: &T,
        left_bra_tensor: &T,
        bra_to_ket_indices: &[(T::Index, T::Index)],
        init: &T,
        exponent_step: Complex64,
    ) -> Result<T, TdvpError> {
        let mut proj_op = self
            .projected_operator
            .write()
            .map_err(|err| TdvpError::Algorithm {
                context: "TDVP failed to lock projected operator",
                source: anyhow::anyhow!("{err}"),
            })?;
        let evolve_started = tdvp_profile_enabled().then(Instant::now);
        let result = hermitian_krylov_expm_multiply(
            |x: &T| {
                let apply_started = tdvp_profile_enabled().then(Instant::now);
                let result = proj_op.apply_edge_center(
                    x,
                    (left, right),
                    (left_ket_tensor, left_bra_tensor),
                    bra_to_ket_indices,
                    (state, &self.reference_state),
                    &self.topology,
                );
                record_tdvp_profile(apply_started, |profile, elapsed| {
                    profile.projected_apply += elapsed;
                    profile.projected_apply_calls += 1;
                });
                result
            },
            exponent_step,
            init,
            &self.options.krylov,
        )
        .map_err(|source| TdvpError::Algorithm {
            context: "TDVP one-site bond-center Krylov exponential failed",
            source: anyhow::Error::new(source),
        })?;
        record_tdvp_profile(evolve_started, |profile, elapsed| {
            profile.evolve_local += elapsed;
        });
        self.max_error_estimate = self.max_error_estimate.max(result.error_estimate);
        self.max_krylov_iterations = self.max_krylov_iterations.max(result.iterations);
        self.local_updates += 1;
        Ok(result.output)
    }

    fn update_two_site(
        &mut self,
        state: &mut TreeTN<T, V>,
        step: &TdvpRegionStep<V>,
    ) -> Result<(), TdvpError> {
        if step.nodes.len() != 2 {
            return Err(TdvpError::UnsupportedNsite {
                requested: step.nodes.len(),
            });
        }
        let extract_started = tdvp_profile_enabled().then(Instant::now);
        let mut subtree =
            state
                .extract_subtree(&step.nodes)
                .map_err(|source| TdvpError::Algorithm {
                    context: "TDVP failed to extract two-site region",
                    source,
                })?;
        record_tdvp_profile(extract_started, |profile, elapsed| {
            profile.extract_region += elapsed;
        });
        let contract_started = tdvp_profile_enabled().then(Instant::now);
        let init_local =
            contract_region(&subtree, &step.nodes).map_err(|source| TdvpError::Algorithm {
                context: "TDVP failed to contract two-site region",
                source,
            })?;
        record_tdvp_profile(contract_started, |profile, elapsed| {
            profile.contract_region += elapsed;
        });
        let evolved = self.evolve_local(
            &step.nodes,
            &init_local,
            state,
            step.exponent_step,
            "TDVP two-site Krylov exponential failed",
        )?;

        let topology_started = tdvp_profile_enabled().then(Instant::now);
        let topology = build_subtree_topology(&evolved, &step.nodes, state).map_err(|source| {
            TdvpError::Algorithm {
                context: "TDVP failed to build two-site subtree topology",
                source,
            }
        })?;
        record_tdvp_profile(topology_started, |profile, elapsed| {
            profile.build_topology += elapsed;
        });
        let mut factorize_options = FactorizeOptions::svd();
        if let Some(max_rank) = self.options.max_bond_dim {
            factorize_options = factorize_options.with_max_rank(max_rank);
        }
        if let Some(policy) = self.options.svd_policy {
            factorize_options = factorize_options.with_svd_policy(policy);
        }
        let factorize_started = tdvp_profile_enabled().then(Instant::now);
        let decomposed = factorize_tensor_to_treetn_with(
            &evolved,
            &topology,
            factorize_options,
            &step.new_center,
        )
        .map_err(|source| TdvpError::Algorithm {
            context: "TDVP failed to split evolved two-site tensor",
            source,
        })?;
        record_tdvp_profile(factorize_started, |profile, elapsed| {
            profile.factorize += elapsed;
        });
        let copy_started = tdvp_profile_enabled().then(Instant::now);
        copy_decomposed_to_subtree(&mut subtree, &decomposed, &step.nodes, state).map_err(
            |source| TdvpError::Algorithm {
                context: "TDVP failed to copy split two-site tensors",
                source,
            },
        )?;
        record_tdvp_profile(copy_started, |profile, elapsed| {
            profile.copy_decomposed += elapsed;
        });
        let subtree_center_started = tdvp_profile_enabled().then(Instant::now);
        subtree
            .set_canonical_region([step.new_center.clone()])
            .map_err(|source| TdvpError::Algorithm {
                context: "TDVP failed to set two-site subtree center",
                source,
            })?;
        record_tdvp_profile(subtree_center_started, |profile, elapsed| {
            profile.subtree_center += elapsed;
        });
        let subtree_ortho_started = tdvp_profile_enabled().then(Instant::now);
        if let Some(edges) = subtree.edges_to_canonicalize_by_names(&step.new_center) {
            for (from, to) in edges {
                if let Some(edge) = subtree.edge_between(&from, &to) {
                    subtree
                        .set_edge_ortho_towards(edge, Some(to))
                        .map_err(|source| TdvpError::Algorithm {
                            context: "TDVP failed to set subtree orthogonality direction",
                            source,
                        })?;
                }
            }
        }
        record_tdvp_profile(subtree_ortho_started, |profile, elapsed| {
            profile.subtree_ortho += elapsed;
        });
        let replace_started = tdvp_profile_enabled().then(Instant::now);
        state
            .replace_subtree(&step.nodes, &subtree)
            .map_err(|source| TdvpError::Algorithm {
                context: "TDVP failed to replace two-site subtree",
                source,
            })?;
        record_tdvp_profile(replace_started, |profile, elapsed| {
            profile.replace_subtree += elapsed;
        });
        let state_center_started = tdvp_profile_enabled().then(Instant::now);
        state
            .set_canonical_region([step.new_center.clone()])
            .map_err(|source| TdvpError::Algorithm {
                context: "TDVP failed to set two-site canonical center",
                source,
            })?;
        record_tdvp_profile(state_center_started, |profile, elapsed| {
            profile.state_center += elapsed;
        });
        Ok(())
    }
}

/// Run TreeTN TDVP for a mapped TreeTN linear operator.
///
/// # Arguments
///
/// * `operator` - Hamiltonian as a [`LinearOperator`]. The v1 implementation
///
///   supports exactly one input and one output site mapping per node.
/// * `init` - Initial TreeTN state. It is canonicalized at `center` before
///
///   sweeping.
/// * `center` - Initial sweep root used for ITensorNetworks-compatible
///
///   post-order region plans.
/// * `options` - Sweep, truncation, and Krylov exponential options.
///
/// # Returns
///
/// A [`TdvpResult`] containing the evolved state and local Krylov diagnostics.
///
/// # Errors
///
/// Returns [`TdvpError`] for unsupported options, incompatible mappings,
/// missing centers, topology mismatches, and local projected exponential
/// failures.
///
/// # Examples
/// ```
/// use std::collections::HashMap;
/// use num_complex::Complex64;
/// use tensor4all_core::{DynIndex, TensorDynLen};
/// use tensor4all_treetn::{tdvp, IndexMapping, LinearOperator, TdvpOptions, TreeTN};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let site = DynIndex::new_dyn(2);
/// let state_tensor = TensorDynLen::from_dense(vec![site.clone()], vec![1.0, 0.0])?;
/// let state = TreeTN::<TensorDynLen, usize>::from_tensors(vec![state_tensor], vec![0])?;
///
/// let input = DynIndex::new_dyn(2);
/// let output = DynIndex::new_dyn(2);
/// let op_tensor = TensorDynLen::from_dense(
///     vec![output.clone(), input.clone()],
///     vec![1.0, 0.0, 0.0, 1.0],
/// )?;
/// let mpo = TreeTN::<TensorDynLen, usize>::from_tensors(vec![op_tensor], vec![0])?;
/// let operator = LinearOperator::new(
///     mpo,
///     HashMap::from([(0, IndexMapping { true_index: site.clone(), internal_index: input })]),
///     HashMap::from([(0, IndexMapping { true_index: site, internal_index: output })]),
/// );
///
/// let result = tdvp(
///     &operator,
///     state,
///     &0,
///     TdvpOptions::default()
///         .with_nsite(1)
///         .with_exponent_step(Complex64::new(0.0, -0.1)),
/// )?;
/// assert_eq!(result.sweeps_completed, 1);
/// # Ok(())
/// # }
/// ```
pub fn tdvp<T, V>(
    operator: &LinearOperator<T, V>,
    init: TreeTN<T, V>,
    center: &V,
    options: TdvpOptions,
) -> Result<TdvpResult<T, V>, TdvpError>
where
    T: TensorLike + 'static,
    T::Index: IndexLike,
    <T::Index as IndexLike>::Id: Clone + std::hash::Hash + Eq + Ord + Debug + Send + Sync + 'static,
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug + 'static,
{
    let profile_enabled = tdvp_profile_enabled();
    if profile_enabled {
        tdvp_profile_reset();
        reset_contract_profile();
        reset_pairwise_contract_profile();
    }
    let total_started = profile_enabled.then(Instant::now);
    validate_options(&options)?;
    if init.node_index(center).is_none() {
        return Err(TdvpError::MissingCenter {
            center: format!("{center:?}"),
        });
    }
    if !init.same_topology(operator.mpo()) {
        return Err(TdvpError::TopologyMismatch);
    }

    let mapping_started = profile_enabled.then(Instant::now);
    let (input_mapping, output_mapping) =
        single_site_square_mappings(operator, &init).map_err(TdvpError::from)?;
    record_tdvp_profile(mapping_started, |profile, elapsed| {
        profile.initial_mapping += elapsed;
    });
    let canonicalize_started = profile_enabled.then(Instant::now);
    let mut state = init
        .canonicalize([center.clone()], CanonicalizationOptions::default())
        .map_err(|source| TdvpError::Algorithm {
            context: "TDVP failed to canonicalize initial state",
            source,
        })?;
    record_tdvp_profile(canonicalize_started, |profile, elapsed| {
        profile.initial_canonicalize += elapsed;
    });
    let plan_started = profile_enabled.then(Instant::now);
    let plan = TdvpRegionPlan::new(
        state.site_index_network().topology(),
        center,
        options.nsite,
        options.order,
        options.exponent_step,
    )
    .ok_or_else(|| TdvpError::MissingCenter {
        center: format!("{center:?}"),
    })?;
    record_tdvp_profile(plan_started, |profile, elapsed| {
        profile.plan += elapsed;
    });
    if options.nsite == 2 && plan.steps.is_empty() {
        return Err(TdvpError::EmptyTwoSiteSweep);
    }

    let topology = CachedTopology::from_site_index_network(state.site_index_network());
    let mut updater = TdvpUpdater::new(
        operator.mpo().clone(),
        input_mapping,
        output_mapping,
        topology,
        options.clone(),
    );
    let mut sweeps_completed = 0usize;
    for sweep in 0..options.nsweeps {
        for (step_index, step) in plan.steps.iter().enumerate() {
            let move_started = profile_enabled.then(Instant::now);
            move_center_to_region_full_rank(&mut state, &step.nodes).map_err(|source| {
                TdvpError::Algorithm {
                    context: "TDVP failed to move canonical center to local region",
                    source,
                }
            })?;
            record_tdvp_profile(move_started, |profile, elapsed| {
                profile.move_center += elapsed;
            });
            updater.update_step(&mut state, step, plan.steps.get(step_index + 1))?;
        }
        sweeps_completed = sweep + 1;
        if options.verbose {
            eprintln!(
                "TDVP sweep {}: updates={} max_krylov_error={:.6e}",
                sweeps_completed, updater.local_updates, updater.max_error_estimate
            );
        }
    }

    record_tdvp_profile(total_started, |profile, elapsed| {
        profile.total += elapsed;
    });
    if let Some(profile) = take_tdvp_profile() {
        print_tdvp_profile(&profile);
        print_and_reset_contract_profile();
        print_and_reset_pairwise_contract_profile();
    }
    Ok(TdvpResult {
        state,
        sweeps_completed,
        local_updates: updater.local_updates,
        max_error_estimate: updater.max_error_estimate,
        max_krylov_iterations: updater.max_krylov_iterations,
    })
}

/// Build a mapped operator from a TreeTN MPO and run TreeTN TDVP.
///
/// This is a convenience wrapper over [`LinearOperator::from_mpo_and_state`] and
/// [`tdvp`].
///
/// # Errors
///
/// Returns an error when the solve or sweep fails (a shape or index
/// /// mismatch, a non-convergence failure, or a backend failure).
///
/// # Examples
/// ```
/// use tensor4all_core::{DynIndex, TensorDynLen};
/// use tensor4all_treetn::{tdvp_with_treetn_operator, TdvpOptions, TreeTN};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let site = DynIndex::new_dyn(2);
/// let state_tensor = TensorDynLen::from_dense(vec![site.clone()], vec![1.0, 0.0])?;
/// let state = TreeTN::<TensorDynLen, usize>::from_tensors(vec![state_tensor], vec![0])?;
/// let input = DynIndex::new_dyn(2);
/// let output = DynIndex::new_dyn(2);
/// let op_tensor = TensorDynLen::from_dense(vec![output, input], vec![1.0, 0.0, 0.0, 1.0])?;
/// let operator = TreeTN::<TensorDynLen, usize>::from_tensors(vec![op_tensor], vec![0])?;
///
/// let result = tdvp_with_treetn_operator(&operator, state, &0, TdvpOptions::default().with_nsite(1))?;
/// assert_eq!(result.sweeps_completed, 1);
/// # Ok(())
/// # }
/// ```
pub fn tdvp_with_treetn_operator<T, V>(
    operator: &TreeTN<T, V>,
    init: TreeTN<T, V>,
    center: &V,
    options: TdvpOptions,
) -> Result<TdvpResult<T, V>, TdvpError>
where
    T: TensorLike + 'static,
    T::Index: IndexLike,
    <T::Index as IndexLike>::Id: Clone + std::hash::Hash + Eq + Ord + Debug + Send + Sync + 'static,
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug + 'static,
{
    let linear_operator =
        LinearOperator::from_mpo_and_state(operator.clone(), &init).map_err(|source| {
            TdvpError::Algorithm {
                context: "TDVP failed to build LinearOperator from TreeTN operator",
                source,
            }
        })?;
    tdvp(&linear_operator, init, center, options)
}

fn validate_options(options: &TdvpOptions) -> Result<(), TdvpError> {
    if options.nsite != 1 && options.nsite != 2 {
        return Err(TdvpError::UnsupportedNsite {
            requested: options.nsite,
        });
    }
    if options.nsweeps == 0 {
        return Err(TdvpError::InvalidOption {
            option: "nsweeps",
            reason: "must be greater than zero".to_string(),
        });
    }
    if !matches!(options.order, 1 | 2 | 4) {
        return Err(TdvpError::UnsupportedOrder {
            requested: options.order,
        });
    }
    if !options.exponent_step.re.is_finite() || !options.exponent_step.im.is_finite() {
        return Err(TdvpError::InvalidOption {
            option: "exponent_step",
            reason: "real and imaginary parts must be finite".to_string(),
        });
    }
    if let Some(max_bond_dim) = options.max_bond_dim {
        if max_bond_dim == 0 {
            return Err(TdvpError::InvalidOption {
                option: "max_bond_dim",
                reason: "must be greater than zero when set".to_string(),
            });
        }
    }
    if options.nsite == 1 {
        if options.max_bond_dim.is_some() {
            return Err(TdvpError::InvalidOption {
                option: "max_bond_dim",
                reason: "one-site TDVP has fixed ranks; use nsite=2 for truncation".to_string(),
            });
        }
        if options.svd_policy.is_some() {
            return Err(TdvpError::InvalidOption {
                option: "svd_policy",
                reason: "one-site TDVP has fixed ranks; use nsite=2 for truncation".to_string(),
            });
        }
    }
    Ok(())
}

fn move_center_to_region_full_rank<T, V>(
    state: &mut TreeTN<T, V>,
    region: &[V],
) -> anyhow::Result<()>
where
    T: TensorLike,
    T::Index: IndexLike,
    <T::Index as IndexLike>::Id: Clone + std::hash::Hash + Eq + Ord + Debug + Send + Sync + 'static,
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug,
{
    if region.is_empty() {
        return Err(anyhow::anyhow!(
            "TDVP cannot move center to an empty region"
        ));
    }
    let current = state.canonical_region();
    if current.is_empty() {
        return Err(anyhow::anyhow!("TDVP state is not canonicalized"));
    }
    if region.iter().any(|node| current.contains(node)) {
        return Ok(());
    }

    let (path, target) = {
        let topology = state.site_index_network().topology();
        let mut best_path = None;
        for src in current {
            let Some(src_idx) = topology.node_index(src) else {
                continue;
            };
            for dst in region {
                let Some(dst_idx) = topology.node_index(dst) else {
                    continue;
                };
                let Some(path) = topology.path_between(src_idx, dst_idx) else {
                    continue;
                };
                if best_path
                    .as_ref()
                    .is_none_or(|best: &Vec<_>| path.len() < best.len())
                {
                    best_path = Some(path);
                }
            }
        }
        let path = best_path.ok_or_else(|| {
            anyhow::anyhow!(
                "TDVP could not find a path from canonical region {:?} to {:?}",
                current,
                region
            )
        })?;
        let target = path
            .last()
            .and_then(|idx| topology.node_name(*idx))
            .ok_or_else(|| anyhow::anyhow!("TDVP center path ended at an unknown node"))?
            .clone();
        (path, target)
    };
    for pair in path.windows(2) {
        let src = pair[0];
        let dst = pair[1];
        state.sweep_edge_full_rank(
            src,
            dst,
            FactorizeAlg::QR,
            Canonical::Left,
            "TDVP path-only center move",
        )?;
    }
    state.set_canonical_region([target])?;
    Ok(())
}

fn first_path_neighbor<T, V>(state: &TreeTN<T, V>, from: &V, to: &V) -> anyhow::Result<Option<V>>
where
    T: TensorLike,
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug,
{
    if from == to {
        return Ok(None);
    }
    let topology = state.site_index_network().topology();
    let from_idx = topology
        .node_index(from)
        .ok_or_else(|| anyhow::anyhow!("source node {:?} is missing", from))?;
    let to_idx = topology
        .node_index(to)
        .ok_or_else(|| anyhow::anyhow!("target node {:?} is missing", to))?;
    let path = topology
        .path_between(from_idx, to_idx)
        .ok_or_else(|| anyhow::anyhow!("no path between {:?} and {:?}", from, to))?;
    let Some(next_idx) = path.get(1) else {
        return Ok(None);
    };
    let next = topology
        .node_name(*next_idx)
        .ok_or_else(|| anyhow::anyhow!("path neighbor {:?} has no node name", next_idx))?
        .clone();
    Ok(Some(next))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn square_site_mapping_errors_convert_to_tdvp_errors() {
        let err = TdvpError::from(SquareSiteMappingError::UnsupportedStateSiteCount {
            node: "site0".to_string(),
            count: 2,
        });
        assert!(matches!(
            err,
            TdvpError::UnsupportedStateSiteCount { node, count }
                if node == "site0" && count == 2
        ));

        let err = TdvpError::from(SquareSiteMappingError::UnsupportedMultipleSiteMappings {
            node: "site1".to_string(),
            role: "input",
            count: 3,
        });
        assert!(matches!(
            err,
            TdvpError::UnsupportedMultipleSiteMappings { node, role, count }
                if node == "site1" && role == "input" && count == 3
        ));

        let err = TdvpError::from(SquareSiteMappingError::MissingMapping {
            node: "site2".to_string(),
            role: "output",
        });
        assert!(matches!(
            err,
            TdvpError::MissingMapping { node, role }
                if node == "site2" && role == "output"
        ));

        let err = TdvpError::from(SquareSiteMappingError::InvalidMapping {
            node: "site3".to_string(),
            reason: "shape mismatch".to_string(),
        });
        assert!(matches!(
            err,
            TdvpError::InvalidMapping { node, reason }
                if node == "site3" && reason == "shape mismatch"
        ));
    }
}
