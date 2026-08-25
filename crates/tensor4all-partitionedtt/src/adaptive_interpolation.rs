//! Adaptive tensor cross interpolation over disjoint projected patches.
//!
//! The patch queue, convergence/splitting flow, and diagonal-pivot recycling are
//! derived from `adaptiveinterpolate`, `createpatch`, and `_globalpivots` in
//! TCIAlgorithms.jl at commit e501032278c9dd41b46c5851d8238169c8d178c5
//! (MIT license; Copyright 2023 Ritter.Marc and contributors).

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
#[cfg(feature = "adaptive-hataori-mpi")]
use std::num::NonZeroUsize;
#[cfg(feature = "adaptive-hataori-mpi")]
use std::panic::{catch_unwind, AssertUnwindSafe};

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use tensor4all_core::{DynIndex, IdxTensor, MatrixLuciScalar, MultiIndex, Scalar, TensorElement};
use tensor4all_itensorlike::TensorTrain;
#[cfg(feature = "adaptive-hataori-mpi")]
use tensor4all_simplett::Tensor3Ops;
use tensor4all_simplett::{tensor3_from_data, SimpleTensorTrain, TTScalar};
use tensor4all_tensorbackend::StorageScalar;
use tensor4all_tensorci::{
    crossinterpolate2, TCI2OptimizationResult, TCI2Options, TCI2Termination, TensorCI2,
};
use tensor4all_treetn::{tensor_train_to_treetn, TreeTN};

use crate::{PartitionedTT, PartitionedTTError, Projector, Result, SubDomainTT};

const ZERO_SAMPLE_THRESHOLD: f64 = 1.0e-30;

/// Options controlling adaptive interpolation and patch subdivision.
///
/// `AdaptiveInterpolateOptions` augments [`TCI2Options`] with a deterministic
/// patch order and initial-pivot policy. Use [`TCI2Options`] directly when no
/// domain subdivision is needed.
///
/// # Examples
///
/// ```
/// use tensor4all_partitionedtt::AdaptiveInterpolateOptions;
///
/// let options = AdaptiveInterpolateOptions::default();
/// assert_eq!(options.n_initial_pivots, 5);
/// assert!(!options.recycle_pivots);
/// assert!(options.patch_order.is_empty());
/// assert!((options.tci_options.tolerance - 1.0e-8).abs() < 1.0e-16);
/// ```
#[derive(Debug, Clone)]
pub struct AdaptiveInterpolateOptions {
    /// TCI2 sweep, tolerance, rank-cap, and random-search options.
    ///
    /// A patch is accepted when its final normalized or absolute bond error,
    /// according to `normalize_error`, is at most `tolerance`. Otherwise it is
    /// split. When in doubt, use [`TCI2Options::default`].
    pub tci_options: TCI2Options,

    /// Complete order in which site indices are fixed when patches split.
    ///
    /// An empty vector uses the order of `site_indices`. A nonempty vector must
    /// be an exact permutation of `site_indices`, including index identity,
    /// tags, prime level, and dimension.
    pub patch_order: Vec<DynIndex>,

    /// Target number of distinct initial pivot candidates per patch.
    ///
    /// Compatible user and recycled pivots are retained, then deterministic
    /// random candidates are added until this target is reached (or the patch
    /// contains fewer points). The recommended and default value is `5`.
    pub n_initial_pivots: usize,

    /// Whether a nonconverged parent TCI's diagonal pivots seed its child patches.
    ///
    /// Recycling is opt-in because it retains more pivot state. Incompatible
    /// pivots are discarded, and every child is replenished to
    /// `n_initial_pivots`; a child is never classified as zero merely because
    /// no recycled pivot is compatible.
    pub recycle_pivots: bool,
}

impl Default for AdaptiveInterpolateOptions {
    fn default() -> Self {
        Self {
            tci_options: TCI2Options::default(),
            patch_order: Vec::new(),
            n_initial_pivots: 5,
            recycle_pivots: false,
        }
    }
}

#[cfg_attr(
    feature = "adaptive-hataori-mpi",
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Debug, Clone)]
struct PatchCache<T> {
    active_dims: Vec<usize>,
    entries: HashMap<MultiIndex, T>,
}

impl<T> PatchCache<T> {
    fn new(active_dims: Vec<usize>) -> Self {
        Self {
            active_dims,
            entries: HashMap::new(),
        }
    }

    fn split(self, split_pos: usize, child_count: usize) -> Result<Vec<Self>> {
        if split_pos >= self.active_dims.len() || self.active_dims[split_pos] != child_count {
            return Err(PartitionedTTError::InvalidAdaptiveInterpolationInput(
                "cache split does not match the active patch dimensions".to_string(),
            ));
        }
        let mut child_dims = self.active_dims;
        child_dims.remove(split_pos);
        let mut children: Vec<_> = (0..child_count)
            .map(|_| Self::new(child_dims.clone()))
            .collect();
        for (mut key, value) in self.entries {
            if key.len() != child_dims.len() + 1 {
                return Err(PartitionedTTError::InvalidAdaptiveInterpolationInput(
                    "cached key rank does not match the parent patch".to_string(),
                ));
            }
            let child = key.remove(split_pos);
            let child_cache = children.get_mut(child).ok_or_else(|| {
                PartitionedTTError::InvalidAdaptiveInterpolationInput(
                    "cached split coordinate is outside the child range".to_string(),
                )
            })?;
            child_cache.entries.insert(key, value);
        }
        Ok(children)
    }
}

/// Samples retained for one accepted adaptive patch.
///
/// Keys passed to [`Self::get`] contain only the patch's active coordinates,
/// in the original site order. The projector records all fixed coordinates.
///
/// # Examples
///
/// ```
/// use tensor4all_partitionedtt::{adaptiveinterpolate, AdaptiveInterpolateOptions, DynIndex, MultiIndex};
/// let result = adaptiveinterpolate::<f64, _, fn(&[MultiIndex]) -> Vec<f64>>(
///     |index| (index[0] + 1) as f64,
///     None,
///     vec![DynIndex::new_dyn(2)],
///     Vec::new(),
///     AdaptiveInterpolateOptions::default(),
/// )?;
/// let cache = &result.patch_caches()[0];
/// assert_eq!(cache.get(&[1]), Some(&2.0));
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone)]
pub struct AcceptedPatchCache<T> {
    projector: Projector,
    active_positions: Vec<usize>,
    cache: PatchCache<T>,
}

impl<T> AcceptedPatchCache<T> {
    /// Return the projector identifying the accepted patch.
    ///
    /// The projector is empty when the root patch converges without splitting.
    pub fn projector(&self) -> &Projector {
        &self.projector
    }

    /// Return the full-domain positions represented by local cache keys.
    ///
    /// For an unsplit two-site patch this returns `[0, 1]`.
    pub fn active_positions(&self) -> &[usize] {
        &self.active_positions
    }

    /// Return a retained value by patch-local active coordinates.
    ///
    /// Returns `None` when TCI did not sample the requested point.
    pub fn get(&self, local_index: &[usize]) -> Option<&T> {
        self.cache.entries.get(local_index)
    }

    /// Return the number of retained sample entries.
    ///
    /// This is the cache's retained-entry statistic.
    pub fn len(&self) -> usize {
        self.cache.entries.len()
    }

    /// Return whether this patch retains no samples.
    pub fn is_empty(&self) -> bool {
        self.cache.entries.is_empty()
    }

    /// Drop all retained samples while preserving patch metadata.
    pub fn clear(&mut self) {
        self.cache.entries.clear();
    }
}

/// Adaptive interpolation output and its accepted per-patch sample caches.
///
/// The cache collection is paired with tensor-network patches by full
/// [`Projector`] equality, not by index ID.
///
/// # Examples
///
/// ```
/// use tensor4all_partitionedtt::{adaptiveinterpolate, AdaptiveInterpolateOptions, DynIndex, MultiIndex};
/// let result = adaptiveinterpolate::<f64, _, fn(&[MultiIndex]) -> Vec<f64>>(
///     |index| (index[0] + 1) as f64,
///     None,
///     vec![DynIndex::new_dyn(2)],
///     Vec::new(),
///     AdaptiveInterpolateOptions::default(),
/// )?;
/// assert_eq!(result.partitioned_tt().len(), 1);
/// assert_eq!(result.patch_caches().len(), 1);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone)]
pub struct AdaptiveInterpolationResult<T> {
    partitioned_tt: PartitionedTT,
    patch_caches: Vec<AcceptedPatchCache<T>>,
}

impl<T> AdaptiveInterpolationResult<T> {
    /// Return the interpolated partitioned tensor train.
    pub fn partitioned_tt(&self) -> &PartitionedTT {
        &self.partitioned_tt
    }

    /// Return caches paired with accepted patches by projector.
    pub fn patch_caches(&self) -> &[AcceptedPatchCache<T>] {
        &self.patch_caches
    }

    /// Consume the result without copying the tensor train or cache entries.
    pub fn into_parts(self) -> (PartitionedTT, Vec<AcceptedPatchCache<T>>) {
        (self.partitioned_tt, self.patch_caches)
    }
}

#[cfg_attr(
    feature = "adaptive-hataori-mpi",
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Debug)]
struct PendingPatch<T> {
    path: Vec<usize>,
    recycled_pivots: Vec<MultiIndex>,
    cache: PatchCache<T>,
}

#[derive(Debug)]
struct AcceptedPatch<T>
where
    T: TTScalar,
{
    path: Vec<usize>,
    data: AcceptedData<T>,
    cache: AcceptedPatchCache<T>,
}

#[derive(Debug)]
enum AcceptedData<T>
where
    T: TTScalar,
{
    Scalar(T),
    Active(SimpleTensorTrain<T>),
}

#[derive(Debug)]
enum PatchOutcome<T: TTScalar> {
    Accepted(AcceptedPatch<T>),
    Split(Vec<PendingPatch<T>>),
}

#[cfg(feature = "adaptive-hataori-mpi")]
#[derive(serde::Serialize, serde::Deserialize)]
struct WireCore<T> {
    dims: [usize; 3],
    data: Vec<T>,
}

#[cfg(feature = "adaptive-hataori-mpi")]
#[derive(serde::Serialize, serde::Deserialize)]
enum WireAcceptedData<T> {
    Scalar(T),
    Active(Vec<WireCore<T>>),
}

#[cfg(feature = "adaptive-hataori-mpi")]
#[derive(serde::Serialize, serde::Deserialize)]
struct WireAcceptedPatch<T> {
    path: Vec<usize>,
    active_positions: Vec<usize>,
    cache: PatchCache<T>,
    data: WireAcceptedData<T>,
}

#[cfg(feature = "adaptive-hataori-mpi")]
#[derive(serde::Serialize, serde::Deserialize)]
enum WirePatchOutcome<T> {
    Accepted(WireAcceptedPatch<T>),
    Split(Vec<PendingPatch<T>>),
}

#[cfg(feature = "adaptive-hataori-mpi")]
#[derive(serde::Serialize, serde::Deserialize)]
enum WaveControl {
    Continue,
    Stop,
    Fail(String),
}

struct PatchEvaluator<'a, T, F, B> {
    f: &'a F,
    batched_f: Option<&'a B>,
    active_positions: &'a [usize],
    projector: &'a Projector,
    site_indices: &'a [DynIndex],
    cache: RefCell<PatchCache<T>>,
    pending_error: RefCell<Option<PartitionedTTError>>,
}

impl<'a, T, F, B> PatchEvaluator<'a, T, F, B>
where
    T: Scalar + Copy,
    F: Fn(&MultiIndex) -> T,
    B: Fn(&[MultiIndex]) -> Vec<T>,
{
    fn new(
        f: &'a F,
        batched_f: Option<&'a B>,
        active_positions: &'a [usize],
        projector: &'a Projector,
        site_indices: &'a [DynIndex],
        cache: PatchCache<T>,
    ) -> Result<Self> {
        let expected_dims: Vec<_> = active_positions
            .iter()
            .map(|&position| site_indices[position].dim)
            .collect();
        if cache.active_dims != expected_dims {
            return Err(PartitionedTTError::InvalidAdaptiveInterpolationInput(
                "cache dimensions do not match the active patch".to_string(),
            ));
        }
        Ok(Self {
            f,
            batched_f,
            active_positions,
            projector,
            site_indices,
            cache: RefCell::new(cache),
            pending_error: RefCell::new(None),
        })
    }

    fn eval(&self, local_index: &MultiIndex) -> T {
        if let Some(value) = self.cache.borrow().entries.get(local_index).copied() {
            return value;
        }
        let full_index = expand_pivot(
            local_index,
            self.active_positions,
            self.projector,
            self.site_indices,
        );
        let value = (self.f)(&full_index);
        self.cache
            .borrow_mut()
            .entries
            .insert(local_index.clone(), value);
        value
    }

    fn eval_many(&self, local_indices: &[MultiIndex]) -> Vec<T> {
        let mut output = vec![None; local_indices.len()];
        let mut missing = Vec::new();
        let mut missing_positions = HashMap::<MultiIndex, Vec<usize>>::new();
        {
            let cache = self.cache.borrow();
            for (position, index) in local_indices.iter().enumerate() {
                if let Some(value) = cache.entries.get(index).copied() {
                    output[position] = Some(value);
                } else {
                    missing_positions
                        .entry(index.clone())
                        .or_insert_with(|| {
                            missing.push(index.clone());
                            Vec::new()
                        })
                        .push(position);
                }
            }
        }
        if !missing.is_empty() {
            let full_indices: Vec<_> = missing
                .iter()
                .map(|index| {
                    expand_pivot(
                        index,
                        self.active_positions,
                        self.projector,
                        self.site_indices,
                    )
                })
                .collect();
            let values = if let Some(batch) = self.batched_f {
                batch(&full_indices)
            } else {
                full_indices.iter().map(self.f).collect()
            };
            if values.len() != missing.len() {
                *self.pending_error.borrow_mut() = Some(
                    PartitionedTTError::InvalidAdaptiveInterpolationInput(format!(
                        "batch callback returned {} values for {} cache misses",
                        values.len(),
                        missing.len()
                    )),
                );
                return vec![T::zero(); local_indices.len()];
            }
            let mut cache = self.cache.borrow_mut();
            for (index, value) in missing.into_iter().zip(values) {
                cache.entries.insert(index.clone(), value);
                for &position in &missing_positions[&index] {
                    output[position] = Some(value);
                }
            }
        }
        output
            .into_iter()
            .map(|value| value.unwrap_or_else(T::zero))
            .collect()
    }

    fn take_error(&self) -> Option<PartitionedTTError> {
        self.pending_error.borrow_mut().take()
    }

    fn into_cache(self) -> PatchCache<T> {
        self.cache.into_inner()
    }
}

/// Adaptively interpolate a discrete function as a partitioned tensor train.
///
/// Each attempted patch runs TCI2 on the sites not fixed by its projector. A
/// converged patch is retained; a patch whose final error exceeds
/// `options.tci_options.tolerance` is split at the next index in
/// `options.patch_order`. Patches with zero or one active site are evaluated
/// exactly because TCI2 requires at least two sites.
///
/// Initial pivots use zero-based coordinates and span the full `site_indices`
/// domain. Compatible supplied and recycled pivots are supplemented with
/// seeded random pivots. If every candidate evaluates below `1e-30`, the patch
/// is represented as zero without further sampling; sparse functions should
/// therefore provide pivots in known nonzero regions.
///
/// # Arguments
///
/// - `f`: scalar evaluator receiving one full, zero-based multi-index.
/// - `batched_f`: optional batch evaluator receiving full multi-indices and
///   returning values in the same order. Use `None` when batching is unavailable.
/// - `site_indices`: one distinct [`DynIndex`] per TCI site, in evaluator order.
/// - `initial_pivots`: full-domain, zero-based pivots. Empty input is allowed.
/// - `options`: TCI2, patch-order, pivot-count, and recycling settings.
///
/// # Returns
///
/// An [`AdaptiveInterpolationResult`] containing mutually disjoint patches and
/// one physically separate sample cache for each accepted patch.
///
/// # Errors
///
/// Returns [`PartitionedTTError::InvalidAdaptiveInterpolationInput`] for empty,
/// duplicate, zero-dimensional, or inconsistently ordered site indices; invalid
/// pivots; a zero pivot target; or invalid TCI tolerances/rank limits. It also
/// forwards TCI2 and tensor-train construction failures.
///
/// # Examples
///
/// ```
/// use tensor4all_core::contract;
/// use tensor4all_partitionedtt::{
///     adaptiveinterpolate, AdaptiveInterpolateOptions, DynIndex, MultiIndex,
/// };
///
/// let sites = vec![DynIndex::new_dyn(2), DynIndex::new_dyn(2)];
/// let f = |idx: &MultiIndex| ((idx[0] + 1) * (idx[1] + 1)) as f64;
/// let result = adaptiveinterpolate::<f64, _, fn(&[MultiIndex]) -> Vec<f64>>(
///     f,
///     None,
///     sites,
///     vec![vec![1, 1]],
///     AdaptiveInterpolateOptions::default(),
/// )
/// .unwrap();
///
/// let tt = result.partitioned_tt().to_tensor_train().unwrap();
/// let dense = contract(&[tt.tensor(0).unwrap(), tt.tensor(1).unwrap()]).unwrap();
/// assert_eq!(dense.to_vec::<f64>().unwrap(), vec![1.0, 2.0, 2.0, 4.0]);
/// ```
pub fn adaptiveinterpolate<T, F, B>(
    f: F,
    batched_f: Option<B>,
    site_indices: Vec<DynIndex>,
    initial_pivots: Vec<MultiIndex>,
    options: AdaptiveInterpolateOptions,
) -> Result<AdaptiveInterpolationResult<T>>
where
    T: Scalar + TTScalar + MatrixLuciScalar + TensorElement + StorageScalar + Default + Copy,
    F: Fn(&MultiIndex) -> T,
    B: Fn(&[MultiIndex]) -> Vec<T>,
{
    let patch_order = validate_inputs(&site_indices, &initial_pivots, &options)?;
    let root_dims = site_indices.iter().map(|index| index.dim).collect();
    let mut wave = vec![PendingPatch {
        path: Vec::new(),
        recycled_pivots: Vec::new(),
        cache: PatchCache::new(root_dims),
    }];
    let mut accepted = Vec::new();

    while !wave.is_empty() {
        let mut next_wave = Vec::new();
        for patch in wave {
            match process_patch(
                patch,
                &f,
                batched_f.as_ref(),
                &site_indices,
                &initial_pivots,
                &patch_order,
                &options,
            )? {
                PatchOutcome::Accepted(patch) => accepted.push(patch),
                PatchOutcome::Split(children) => next_wave.extend(children),
            }
        }
        wave = next_wave;
    }

    assemble_result(accepted, &site_indices, &patch_order)
}

/// Adaptively interpolate patches in parallel on an explicit Hataori Rayon domain.
///
/// Different patches use [`hataori::LocalMode::Outer`]. Rayon work started by a
/// patch callback is nested in the same domain pool. The domain must be backed
/// by a Rayon pool, and the call must originate outside a foreign Rayon pool.
///
/// # Examples
///
/// ```
/// use std::sync::Arc;
/// use tensor4all_partitionedtt::{adaptiveinterpolate_in, AdaptiveInterpolateOptions, DynIndex, MultiIndex};
/// let pool = Arc::new(rayon::ThreadPoolBuilder::new().num_threads(1).build()?);
/// let domain = hataori::Domain::external(pool, vec![0], 1)?;
/// let result = adaptiveinterpolate_in::<f64, _, fn(&[MultiIndex]) -> Vec<f64>>(
///     &domain,
///     |index| (index[0] + 1) as f64,
///     None,
///     vec![DynIndex::new_dyn(2)],
///     Vec::new(),
///     AdaptiveInterpolateOptions::default(),
/// )?;
/// assert_eq!(result.patch_caches()[0].get(&[1]), Some(&2.0));
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// # Errors
///
/// Returns the same interpolation errors as [`adaptiveinterpolate`], or
/// [`PartitionedTTError::HataoriLocal`] when domain admission or scheduling
/// fails. No partial partition or cache is returned.
#[cfg(feature = "adaptive-hataori-rayon")]
pub fn adaptiveinterpolate_in<T, F, B>(
    domain: &hataori::Domain,
    f: F,
    batched_f: Option<B>,
    site_indices: Vec<DynIndex>,
    initial_pivots: Vec<MultiIndex>,
    options: AdaptiveInterpolateOptions,
) -> Result<AdaptiveInterpolationResult<T>>
where
    T: Scalar + TTScalar + MatrixLuciScalar + TensorElement + StorageScalar + Default + Copy + Send,
    F: Fn(&MultiIndex) -> T + Send + Sync,
    B: Fn(&[MultiIndex]) -> Vec<T> + Send + Sync,
{
    let patch_order = validate_inputs(&site_indices, &initial_pivots, &options)?;
    let root_dims = site_indices.iter().map(|index| index.dim).collect();
    let mut wave = vec![PendingPatch {
        path: Vec::new(),
        recycled_pivots: Vec::new(),
        cache: PatchCache::new(root_dims),
    }];
    let mut accepted = Vec::new();

    while !wave.is_empty() {
        let paths: Vec<_> = wave.iter().map(|patch| patch.path.clone()).collect();
        let outcomes = hataori::map_in(domain, hataori::LocalMode::Outer, wave, |patch| {
            process_patch(
                patch,
                &f,
                batched_f.as_ref(),
                &site_indices,
                &initial_pivots,
                &patch_order,
                &options,
            )
        })
        .map_err(|source| {
            let path = match &source {
                hataori::MapInError::Callback(error) => paths.get(error.index()).cloned(),
                _ => None,
            };
            PartitionedTTError::HataoriLocal { path, source }
        })?;
        let mut next_wave = Vec::new();
        for outcome in outcomes {
            match outcome {
                PatchOutcome::Accepted(patch) => accepted.push(patch),
                PatchOutcome::Split(children) => next_wave.extend(children),
            }
        }
        wave = next_wave;
    }

    assemble_result(accepted, &site_indices, &patch_order)
}

/// Collectively interpolate adaptive patches across MPI ranks and local Rayon domains.
///
/// Every rank must call this function from the MPI main thread with equivalent
/// callbacks and interpolation metadata. Each rank supplies a pool-backed,
/// id-zero Hataori domain; MPI must provide `MPI_THREAD_FUNNELED` or stronger.
/// Only `root` receives `Some(result)`.
///
/// # Examples
///
/// A runnable multi-rank example with numerical and cache assertions is provided
/// at `examples/adaptive_mpi_smoke.rs` and is executed with
/// `mpiexec -n 2 target/release/examples/adaptive_mpi_smoke`.
///
/// # Errors
///
/// Returns [`PartitionedTTError::HataoriMpiPlacement`] when collective
/// validation or `WaveControl` broadcast fails,
/// [`PartitionedTTError::HataoriMpiPmap`] when a patch callback, MPI scheduler,
/// or Hataori wire operation fails, and
/// [`PartitionedTTError::DistributedAdaptiveInterpolation`] when common input
/// is invalid or root-side cache/core reconstruction and final partition
/// validation fails. No rank receives a partial partition or cache.
#[cfg(feature = "adaptive-hataori-mpi")]
#[allow(clippy::too_many_arguments)]
pub fn adaptiveinterpolate_mpi<C, T, F, B>(
    world: &C,
    domain: &hataori::Domain,
    root: i32,
    f: F,
    batched_f: Option<B>,
    site_indices: Vec<DynIndex>,
    initial_pivots: Vec<MultiIndex>,
    options: AdaptiveInterpolateOptions,
) -> Result<Option<AdaptiveInterpolationResult<T>>>
where
    C: mpi::traits::Communicator,
    T: Scalar
        + TTScalar
        + MatrixLuciScalar
        + TensorElement
        + StorageScalar
        + Default
        + Copy
        + Send
        + Sync
        + serde::Serialize
        + serde::de::DeserializeOwned,
    F: Fn(&MultiIndex) -> T + Send + Sync,
    B: Fn(&[MultiIndex]) -> Vec<T> + Send + Sync,
{
    let rank = world.rank();
    let local_validation = validate_inputs(&site_indices, &initial_pivots, &options);
    let local_error = local_validation.as_ref().err().map(ToString::to_string);
    let gathered = hataori::gather(world, root, local_error)
        .map_err(|source| PartitionedTTError::HataoriMpiPlacement { source })?;
    let validation_control = if rank == root {
        let first_error = gathered
            .as_ref()
            .and_then(|errors| errors.iter().flatten().next())
            .cloned();
        Some(match first_error {
            Some(message) => WaveControl::Fail(message),
            None => WaveControl::Continue,
        })
    } else {
        None
    };
    match hataori::broadcast(world, root, validation_control)
        .map_err(|source| PartitionedTTError::HataoriMpiPlacement { source })?
    {
        WaveControl::Continue => {}
        WaveControl::Fail(message) => {
            return Err(PartitionedTTError::DistributedAdaptiveInterpolation(
                message,
            ));
        }
        WaveControl::Stop => {
            return Err(PartitionedTTError::DistributedAdaptiveInterpolation(
                "invalid stop control during MPI validation".to_string(),
            ));
        }
    }
    let patch_order = local_validation
        .map_err(|error| PartitionedTTError::DistributedAdaptiveInterpolation(error.to_string()))?;

    let root_dims = site_indices.iter().map(|index| index.dim).collect();
    let mut wave = if rank == root {
        vec![PendingPatch {
            path: Vec::new(),
            recycled_pivots: Vec::new(),
            cache: PatchCache::new(root_dims),
        }]
    } else {
        Vec::new()
    };
    let mut accepted = Vec::new();
    let pmap_options = hataori::PmapOptions {
        root,
        batch_size: NonZeroUsize::MIN,
        local_mode: hataori::LocalMode::Outer,
        prefetch: false,
    };

    loop {
        let root_items = (rank == root).then(|| std::mem::take(&mut wave));
        let wire_outcomes = hataori::pmap(world, domain, pmap_options, root_items, |patch| {
            process_patch(
                patch,
                &f,
                batched_f.as_ref(),
                &site_indices,
                &initial_pivots,
                &patch_order,
                &options,
            )
            .map(patch_outcome_to_wire)
        })
        .map_err(|source| PartitionedTTError::HataoriMpiPmap { source })?;

        let mut root_result = None;
        let root_control = if rank == root {
            let root_postprocess = catch_unwind(AssertUnwindSafe(|| -> Result<WaveControl> {
                let mut next_wave = Vec::new();
                for wire in wire_outcomes.ok_or_else(|| {
                    PartitionedTTError::DistributedAdaptiveInterpolation(
                        "MPI root did not receive patch outcomes".to_string(),
                    )
                })? {
                    match patch_outcome_from_wire(wire, &site_indices, &patch_order)? {
                        PatchOutcome::Accepted(patch) => accepted.push(patch),
                        PatchOutcome::Split(children) => next_wave.extend(children),
                    }
                }
                if next_wave.is_empty() {
                    root_result = Some(assemble_result(
                        std::mem::take(&mut accepted),
                        &site_indices,
                        &patch_order,
                    )?);
                    Ok(WaveControl::Stop)
                } else {
                    wave = next_wave;
                    Ok(WaveControl::Continue)
                }
            }));
            Some(match root_postprocess {
                Ok(Ok(control)) => control,
                Ok(Err(error)) => WaveControl::Fail(error.to_string()),
                Err(_) => world.abort(75),
            })
        } else {
            None
        };

        match hataori::broadcast(world, root, root_control)
            .map_err(|source| PartitionedTTError::HataoriMpiPlacement { source })?
        {
            WaveControl::Continue => continue,
            WaveControl::Stop if rank == root => {
                return Ok(Some(root_result.unwrap_or_else(|| world.abort(76))));
            }
            WaveControl::Stop => return Ok(None),
            WaveControl::Fail(message) => {
                return Err(PartitionedTTError::DistributedAdaptiveInterpolation(
                    message,
                ));
            }
        }
    }
}

fn assemble_result<T>(
    mut accepted: Vec<AcceptedPatch<T>>,
    site_indices: &[DynIndex],
    patch_order: &[DynIndex],
) -> Result<AdaptiveInterpolationResult<T>>
where
    T: Scalar + TTScalar + TensorElement + StorageScalar + Default + Copy,
{
    accepted.sort_by(|left, right| left.path.cmp(&right.path));
    let mut subdomains = Vec::with_capacity(accepted.len());
    let mut patch_caches = Vec::with_capacity(accepted.len());
    for patch in accepted {
        let projector = projector_from_path(patch_order, &patch.path)?;
        let tt = match patch.data {
            AcceptedData::Scalar(value) => rank_one_full_tt(site_indices, &projector, value)?,
            AcceptedData::Active(active_tt) => {
                let (active_tree, _) = tensor_train_to_treetn(&active_tt).map_err(|error| {
                    PartitionedTTError::tensor_train_operation(error.to_string())
                })?;
                embed_active_tt::<T>(
                    active_tree,
                    site_indices,
                    patch.cache.active_positions(),
                    &projector,
                )?
            }
        };
        subdomains.push(SubDomainTT::new(tt, projector)?);
        patch_caches.push(patch.cache);
    }
    let partitioned_tt = PartitionedTT::from_subdomains(subdomains)?;
    Ok(AdaptiveInterpolationResult {
        partitioned_tt,
        patch_caches,
    })
}

fn process_patch<T, F, B>(
    patch: PendingPatch<T>,
    f: &F,
    batched_f: Option<&B>,
    site_indices: &[DynIndex],
    initial_pivots: &[MultiIndex],
    patch_order: &[DynIndex],
    options: &AdaptiveInterpolateOptions,
) -> Result<PatchOutcome<T>>
where
    T: Scalar + TTScalar + MatrixLuciScalar + TensorElement + StorageScalar + Default + Copy,
    F: Fn(&MultiIndex) -> T,
    B: Fn(&[MultiIndex]) -> Vec<T>,
{
    let projector = projector_from_path(patch_order, &patch.path)?;
    let active_positions = active_positions(site_indices, &projector);
    let evaluator = PatchEvaluator::new(
        f,
        batched_f,
        &active_positions,
        &projector,
        site_indices,
        patch.cache,
    )?;

    if active_positions.is_empty() {
        let value = evaluator.eval(&Vec::new());
        let cache = evaluator.into_cache();
        return Ok(accepted_outcome(
            patch.path,
            projector,
            active_positions,
            AcceptedData::Scalar(value),
            cache,
        ));
    }

    if active_positions.len() == 1 {
        let dim = site_indices[active_positions[0]].dim;
        let data = (0..dim).map(|value| evaluator.eval(&vec![value])).collect();
        let core = tensor3_from_data(data, 1, dim, 1)
            .map_err(|error| PartitionedTTError::tensor_train_operation(error.to_string()))?;
        let exact = SimpleTensorTrain::new(vec![core])
            .map_err(|error| PartitionedTTError::tensor_train_operation(error.to_string()))?;
        let cache = evaluator.into_cache();
        return Ok(accepted_outcome(
            patch.path,
            projector,
            active_positions,
            AcceptedData::Active(exact),
            cache,
        ));
    }

    let seed = patch_seed(options.tci_options.seed.unwrap_or(0), &patch.path);
    let mut rng = StdRng::seed_from_u64(seed);
    let candidate_pivots = patch_candidates(
        site_indices,
        &active_positions,
        &projector,
        initial_pivots,
        &patch.recycled_pivots,
        options.n_initial_pivots,
        &mut rng,
    )?;
    let candidate_values = evaluator.eval_many(&candidate_pivots);
    if let Some(error) = evaluator.take_error() {
        return Err(error);
    }

    if candidate_values
        .iter()
        .all(|value| Scalar::abs_val(*value) < ZERO_SAMPLE_THRESHOLD)
    {
        let cache = evaluator.into_cache();
        return Ok(accepted_outcome(
            patch.path,
            projector,
            active_positions,
            AcceptedData::Scalar(T::zero()),
            cache,
        ));
    }

    let local_dims = active_positions
        .iter()
        .map(|&position| site_indices[position].dim)
        .collect();
    let local_f = |pivot: &MultiIndex| evaluator.eval(pivot);
    let local_batch = batched_f.map(|_| |pivots: &[MultiIndex]| evaluator.eval_many(pivots));
    let mut tci_options = options.tci_options.clone();
    tci_options.seed = Some(splitmix64(seed));
    let tci_result = crossinterpolate2(
        local_f,
        local_batch,
        local_dims,
        candidate_pivots,
        tci_options,
    );
    if let Some(error) = evaluator.take_error() {
        return Err(error);
    }
    let TCI2OptimizationResult {
        tci,
        errors,
        termination,
        ..
    } = tci_result?;
    let normalization = if options.tci_options.normalize_error && tci.max_sample_value() > 0.0 {
        tci.max_sample_value()
    } else {
        1.0
    };
    let final_error = errors
        .last()
        .copied()
        .unwrap_or_else(|| tci.max_bond_error() / normalization);

    if patch_is_accepted(termination, final_error, options.tci_options.tolerance) {
        let simple_tt = tci.to_tensor_train()?;
        let cache = evaluator.into_cache();
        return Ok(accepted_outcome(
            patch.path,
            projector,
            active_positions,
            AcceptedData::Active(simple_tt),
            cache,
        ));
    }

    let split_index = patch_order.get(patch.path.len()).ok_or_else(|| {
        PartitionedTTError::InvalidAdaptiveInterpolationInput(
            "a nonconverged patch has no remaining split index".to_string(),
        )
    })?;
    let split_site_position = site_indices
        .iter()
        .position(|index| index == split_index)
        .ok_or_else(|| {
            PartitionedTTError::InvalidAdaptiveInterpolationInput(
                "split index is absent from site_indices".to_string(),
            )
        })?;
    let split_active_position = active_positions
        .iter()
        .position(|&position| position == split_site_position)
        .ok_or_else(|| {
            PartitionedTTError::InvalidAdaptiveInterpolationInput(
                "split index is not active in its parent patch".to_string(),
            )
        })?;
    let recycled_pivots = if options.recycle_pivots {
        global_diagonal_pivots(&tci, &active_positions, &projector, site_indices)
    } else {
        Vec::new()
    };
    let child_caches = evaluator
        .into_cache()
        .split(split_active_position, split_index.dim)?;
    let mut children = Vec::with_capacity(split_index.dim);
    for (value, cache) in child_caches.into_iter().enumerate() {
        let mut path = patch.path.clone();
        path.push(value);
        children.push(PendingPatch {
            path,
            recycled_pivots: recycled_pivots.clone(),
            cache,
        });
    }
    Ok(PatchOutcome::Split(children))
}

fn accepted_outcome<T: TTScalar>(
    path: Vec<usize>,
    projector: Projector,
    active_positions: Vec<usize>,
    data: AcceptedData<T>,
    cache: PatchCache<T>,
) -> PatchOutcome<T> {
    PatchOutcome::Accepted(AcceptedPatch {
        path,
        data,
        cache: AcceptedPatchCache {
            projector,
            active_positions,
            cache,
        },
    })
}

#[cfg(feature = "adaptive-hataori-mpi")]
fn patch_outcome_to_wire<T>(outcome: PatchOutcome<T>) -> WirePatchOutcome<T>
where
    T: TTScalar + Copy,
{
    match outcome {
        PatchOutcome::Split(children) => WirePatchOutcome::Split(children),
        PatchOutcome::Accepted(patch) => {
            let data = match patch.data {
                AcceptedData::Scalar(value) => WireAcceptedData::Scalar(value),
                AcceptedData::Active(tt) => WireAcceptedData::Active(
                    tt.into_site_tensors()
                        .into_iter()
                        .map(|core| WireCore {
                            dims: [core.left_dim(), core.site_dim(), core.right_dim()],
                            data: core.to_col_major_vec(),
                        })
                        .collect(),
                ),
            };
            WirePatchOutcome::Accepted(WireAcceptedPatch {
                path: patch.path,
                active_positions: patch.cache.active_positions,
                cache: patch.cache.cache,
                data,
            })
        }
    }
}

#[cfg(feature = "adaptive-hataori-mpi")]
fn patch_outcome_from_wire<T>(
    outcome: WirePatchOutcome<T>,
    site_indices: &[DynIndex],
    patch_order: &[DynIndex],
) -> Result<PatchOutcome<T>>
where
    T: TTScalar + Copy,
{
    match outcome {
        WirePatchOutcome::Split(children) => Ok(PatchOutcome::Split(children)),
        WirePatchOutcome::Accepted(patch) => {
            let projector = projector_from_path(patch_order, &patch.path)?;
            let expected_active = active_positions(site_indices, &projector);
            let expected_dims: Vec<_> = expected_active
                .iter()
                .map(|&position| site_indices[position].dim)
                .collect();
            if patch.active_positions != expected_active || patch.cache.active_dims != expected_dims
            {
                return Err(PartitionedTTError::DistributedAdaptiveInterpolation(
                    "wire accepted cache metadata does not match its patch path".to_string(),
                ));
            }
            let data = match patch.data {
                WireAcceptedData::Scalar(value) => AcceptedData::Scalar(value),
                WireAcceptedData::Active(wire_cores) => {
                    if wire_cores.len() != expected_dims.len() {
                        return Err(PartitionedTTError::DistributedAdaptiveInterpolation(
                            format!(
                                "wire TT has {} cores for {} active sites",
                                wire_cores.len(),
                                expected_dims.len()
                            ),
                        ));
                    }
                    let mut cores = Vec::with_capacity(wire_cores.len());
                    for (site, core) in wire_cores.into_iter().enumerate() {
                        if core.dims[1] != expected_dims[site] {
                            return Err(PartitionedTTError::DistributedAdaptiveInterpolation(
                                format!(
                                    "wire TT core {site} has site dimension {} but patch requires {}",
                                    core.dims[1], expected_dims[site]
                                ),
                            ));
                        }
                        let expected = core.dims.iter().try_fold(1usize, |count, &dim| {
                            count.checked_mul(dim).ok_or_else(|| {
                                PartitionedTTError::DistributedAdaptiveInterpolation(
                                    "wire TT core shape product exceeds usize".to_string(),
                                )
                            })
                        })?;
                        if core.data.len() != expected {
                            return Err(
                                PartitionedTTError::DistributedAdaptiveInterpolation(format!(
                                    "wire TT core has {} values for shape {:?} requiring {expected}",
                                    core.data.len(),
                                    core.dims
                                )),
                            );
                        }
                        cores.push(
                            tensor3_from_data(core.data, core.dims[0], core.dims[1], core.dims[2])
                                .map_err(|error| {
                                    PartitionedTTError::tensor_train_operation(error.to_string())
                                })?,
                        );
                    }
                    AcceptedData::Active(SimpleTensorTrain::new(cores).map_err(|error| {
                        PartitionedTTError::tensor_train_operation(error.to_string())
                    })?)
                }
            };
            Ok(PatchOutcome::Accepted(AcceptedPatch {
                path: patch.path,
                data,
                cache: AcceptedPatchCache {
                    projector,
                    active_positions: patch.active_positions,
                    cache: patch.cache,
                },
            }))
        }
    }
}

fn projector_from_path(patch_order: &[DynIndex], path: &[usize]) -> Result<Projector> {
    if path.len() > patch_order.len() {
        return Err(PartitionedTTError::InvalidAdaptiveInterpolationInput(
            "patch path is deeper than patch_order".to_string(),
        ));
    }
    let mut projector = Projector::new();
    for (index, &value) in patch_order.iter().zip(path) {
        if value >= index.dim {
            return Err(PartitionedTTError::InvalidAdaptiveInterpolationInput(
                "patch path coordinate is outside its site dimension".to_string(),
            ));
        }
        projector.insert(index.clone(), value)?;
    }
    Ok(projector)
}

// Stable SplitMix64 derivation; changing these constants changes reproducible patch candidates.
fn patch_seed(root_seed: u64, path: &[usize]) -> u64 {
    path.iter().enumerate().fold(
        splitmix64(root_seed ^ 0x6a09_e667_f3bc_c909),
        |state, (depth, &value)| splitmix64(state ^ (depth as u64).rotate_left(32) ^ value as u64),
    )
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn validate_inputs(
    site_indices: &[DynIndex],
    initial_pivots: &[MultiIndex],
    options: &AdaptiveInterpolateOptions,
) -> Result<Vec<DynIndex>> {
    if site_indices.is_empty() {
        return Err(PartitionedTTError::InvalidAdaptiveInterpolationInput(
            "site_indices must not be empty".to_string(),
        ));
    }
    if site_indices.iter().any(|index| index.dim == 0) {
        return Err(PartitionedTTError::InvalidAdaptiveInterpolationInput(
            "site indices must have positive dimensions".to_string(),
        ));
    }
    let unique_sites: HashSet<_> = site_indices.iter().cloned().collect();
    if unique_sites.len() != site_indices.len() {
        return Err(PartitionedTTError::InvalidAdaptiveInterpolationInput(
            "site_indices contains duplicate indices".to_string(),
        ));
    }
    if options.n_initial_pivots == 0 {
        return Err(PartitionedTTError::InvalidAdaptiveInterpolationInput(
            "n_initial_pivots must be positive".to_string(),
        ));
    }
    if !options.tci_options.tolerance.is_finite() || options.tci_options.tolerance < 0.0 {
        return Err(PartitionedTTError::InvalidAdaptiveInterpolationInput(
            "TCI tolerance must be finite and nonnegative".to_string(),
        ));
    }
    if options.tci_options.max_iter == 0 {
        return Err(PartitionedTTError::InvalidAdaptiveInterpolationInput(
            "TCI max_iter must be positive".to_string(),
        ));
    }
    if options.tci_options.max_bond_dim == Some(0) {
        return Err(PartitionedTTError::InvalidAdaptiveInterpolationInput(
            "TCI max_bond_dim must be positive".to_string(),
        ));
    }
    if options.tci_options.ncheck_history == 0 {
        return Err(PartitionedTTError::InvalidAdaptiveInterpolationInput(
            "TCI ncheck_history must be positive".to_string(),
        ));
    }
    if !options.tci_options.tol_margin_global_search.is_finite()
        || options.tci_options.tol_margin_global_search < 0.0
    {
        return Err(PartitionedTTError::InvalidAdaptiveInterpolationInput(
            "TCI tol_margin_global_search must be finite and nonnegative".to_string(),
        ));
    }
    for pivot in initial_pivots {
        if pivot.len() != site_indices.len() {
            return Err(PartitionedTTError::InvalidAdaptiveInterpolationInput(
                "every initial pivot must have one coordinate per site".to_string(),
            ));
        }
        if pivot
            .iter()
            .zip(site_indices)
            .any(|(&value, index)| value >= index.dim)
        {
            return Err(PartitionedTTError::InvalidAdaptiveInterpolationInput(
                "an initial pivot coordinate is outside its site dimension".to_string(),
            ));
        }
    }

    let patch_order = if options.patch_order.is_empty() {
        site_indices.to_vec()
    } else {
        options.patch_order.clone()
    };
    let unique_order: HashSet<_> = patch_order.iter().cloned().collect();
    if patch_order.len() != site_indices.len()
        || unique_order.len() != patch_order.len()
        || unique_order != unique_sites
    {
        return Err(PartitionedTTError::InvalidAdaptiveInterpolationInput(
            "patch_order must be an exact permutation of site_indices".to_string(),
        ));
    }
    Ok(patch_order)
}

fn patch_is_accepted(termination: TCI2Termination, final_error: f64, tolerance: f64) -> bool {
    termination == TCI2Termination::Converged && final_error <= tolerance
}

fn active_positions(site_indices: &[DynIndex], projector: &Projector) -> Vec<usize> {
    site_indices
        .iter()
        .enumerate()
        .filter_map(|(position, index)| (!projector.is_projected_at(index)).then_some(position))
        .collect()
}

fn patch_candidates(
    site_indices: &[DynIndex],
    active_positions: &[usize],
    projector: &Projector,
    initial_pivots: &[MultiIndex],
    recycled_pivots: &[MultiIndex],
    target: usize,
    rng: &mut StdRng,
) -> Result<Vec<MultiIndex>> {
    let mut candidates = Vec::new();
    let mut seen = HashSet::new();
    for full_pivot in initial_pivots.iter().chain(recycled_pivots) {
        if is_compatible_pivot(full_pivot, site_indices, projector) {
            let local = active_positions
                .iter()
                .map(|&position| full_pivot[position])
                .collect::<Vec<_>>();
            if seen.insert(local.clone()) {
                candidates.push(local);
            }
        }
    }

    let local_dims: Vec<_> = active_positions
        .iter()
        .map(|&position| site_indices[position].dim)
        .collect();
    let point_count = local_dims.iter().try_fold(1usize, |count, &dim| {
        count.checked_mul(dim).ok_or_else(|| {
            PartitionedTTError::InvalidAdaptiveInterpolationInput(
                "active patch point count exceeds usize".to_string(),
            )
        })
    })?;
    let desired = target.max(candidates.len()).min(point_count);
    let random_attempts = desired
        .checked_mul(20)
        .and_then(|attempts| attempts.checked_add(100))
        .ok_or_else(|| {
            PartitionedTTError::InvalidAdaptiveInterpolationInput(
                "initial-pivot search attempt count exceeds usize".to_string(),
            )
        })?;
    for _ in 0..random_attempts {
        if candidates.len() >= desired {
            break;
        }
        let pivot: Vec<_> = local_dims
            .iter()
            .map(|&dim| rng.random_range(0..dim))
            .collect();
        if seen.insert(pivot.clone()) {
            candidates.push(pivot);
        }
    }
    for flat in 0..point_count {
        if candidates.len() >= desired {
            break;
        }
        let pivot = decode_col_major(flat, &local_dims);
        if seen.insert(pivot.clone()) {
            candidates.push(pivot);
        }
    }
    Ok(candidates)
}

fn is_compatible_pivot(
    pivot: &MultiIndex,
    site_indices: &[DynIndex],
    projector: &Projector,
) -> bool {
    pivot.len() == site_indices.len()
        && site_indices.iter().enumerate().all(|(position, index)| {
            projector
                .get(index)
                .is_none_or(|value| pivot[position] == value)
        })
}

fn decode_col_major(mut flat: usize, dims: &[usize]) -> MultiIndex {
    dims.iter()
        .map(|&dim| {
            let value = flat % dim;
            flat /= dim;
            value
        })
        .collect()
}

fn expand_pivot(
    local_pivot: &MultiIndex,
    active_positions: &[usize],
    projector: &Projector,
    site_indices: &[DynIndex],
) -> MultiIndex {
    let mut full = vec![0; site_indices.len()];
    for (&position, &value) in active_positions.iter().zip(local_pivot) {
        full[position] = value;
    }
    for (position, index) in site_indices.iter().enumerate() {
        if let Some(value) = projector.get(index) {
            full[position] = value;
        }
    }
    full
}

fn global_diagonal_pivots<T>(
    tci: &TensorCI2<T>,
    active_positions: &[usize],
    projector: &Projector,
    site_indices: &[DynIndex],
) -> Vec<MultiIndex>
where
    T: Scalar + TTScalar + MatrixLuciScalar + Default,
{
    let mut result = Vec::new();
    let mut seen = HashSet::new();
    for bond in 0..active_positions.len() - 1 {
        for (left, right) in tci.i_set(bond + 1).iter().zip(tci.j_set(bond)) {
            let mut local = left.clone();
            local.extend(right);
            if local.len() == active_positions.len() {
                let full = expand_pivot(&local, active_positions, projector, site_indices);
                if seen.insert(full.clone()) {
                    result.push(full);
                }
            }
        }
    }
    result
}

fn embed_active_tt<T>(
    active_tree: TreeTN<IdxTensor, usize>,
    site_indices: &[DynIndex],
    active_positions: &[usize],
    projector: &Projector,
) -> Result<TensorTrain>
where
    T: Scalar + TTScalar + TensorElement + StorageScalar + Default + Copy,
{
    let active_count = active_positions.len();
    let link_dims = active_tree.link_dims();
    let node_names = active_tree.node_names();
    let mut core_data: Vec<Vec<T>> = Vec::with_capacity(node_names.len());
    for name in &node_names {
        let node = active_tree.node_index(name).ok_or_else(|| {
            PartitionedTTError::tensor_train_operation(format!(
                "missing node {:?} in active tree",
                name
            ))
        })?;
        let tensor = active_tree.tensor(node).ok_or_else(|| {
            PartitionedTTError::tensor_train_operation(format!(
                "missing tensor for node {:?} in active tree",
                name
            ))
        })?;
        core_data.push(
            tensor
                .to_vec::<T>()
                .map_err(|error| PartitionedTTError::tensor_train_operation(error.to_string()))?,
        );
    }
    let mut edge_dims = Vec::with_capacity(site_indices.len().saturating_sub(1));
    for edge in 0..site_indices.len().saturating_sub(1) {
        let active_left = active_positions
            .iter()
            .filter(|&&position| position <= edge)
            .count();
        edge_dims.push(if active_left == 0 || active_left == active_count {
            1
        } else {
            link_dims[active_left - 1]
        });
    }
    let edge_indices: Vec<_> = edge_dims
        .iter()
        .map(|&dimension| DynIndex::new_dyn(dimension))
        .collect();

    let mut tensors = Vec::with_capacity(site_indices.len());
    let mut next_active = 0;
    for (position, site_index) in site_indices.iter().enumerate() {
        let left = position.checked_sub(1).map(|edge| &edge_indices[edge]);
        let right = edge_indices.get(position);
        if active_positions.get(next_active) == Some(&position) {
            let core = &core_data[next_active];
            let mut indices = Vec::with_capacity(3);
            if let Some(index) = left {
                indices.push(index.clone());
            }
            indices.push(site_index.clone());
            if let Some(index) = right {
                indices.push(index.clone());
            }
            tensors.push(
                IdxTensor::from_dense(indices, core.clone()).map_err(|error| {
                    PartitionedTTError::tensor_train_operation(error.to_string())
                })?,
            );
            next_active += 1;
        } else {
            let value = projector.get(site_index).ok_or_else(|| {
                PartitionedTTError::InvalidAdaptiveInterpolationInput(
                    "an embedded inactive site is missing from its projector".to_string(),
                )
            })?;
            tensors.push(projected_site_tensor::<T>(
                left,
                site_index,
                right,
                value,
                T::one(),
            )?);
        }
    }
    TensorTrain::new(tensors)
        .map_err(|error| PartitionedTTError::tensor_train_operation(error.to_string()))
}

fn projected_site_tensor<T>(
    left: Option<&DynIndex>,
    site: &DynIndex,
    right: Option<&DynIndex>,
    value: usize,
    scale: T,
) -> Result<IdxTensor>
where
    T: Scalar + TensorElement + StorageScalar + Default + Copy,
{
    match (left, right) {
        (Some(left), Some(right)) => {
            IdxTensor::from_copy_selector(left.clone(), site.clone(), right.clone(), value, scale)
                .map_err(|error| PartitionedTTError::tensor_train_operation(error.to_string()))
        }
        (None, Some(right)) => {
            if right.dim != 1 {
                return Err(PartitionedTTError::tensor_train_operation(format!(
                    "projected first site requires a unit right bond, got {}",
                    right.dim
                )));
            }
            let mut data = vec![T::zero(); site.dim];
            data[value] = scale;
            IdxTensor::from_dense(vec![site.clone(), right.clone()], data)
                .map_err(|error| PartitionedTTError::tensor_train_operation(error.to_string()))
        }
        (Some(left), None) => {
            if left.dim != 1 {
                return Err(PartitionedTTError::tensor_train_operation(format!(
                    "projected last site requires a unit left bond, got {}",
                    left.dim
                )));
            }
            let mut data = vec![T::zero(); site.dim];
            data[value] = scale;
            IdxTensor::from_dense(vec![left.clone(), site.clone()], data)
                .map_err(|error| PartitionedTTError::tensor_train_operation(error.to_string()))
        }
        (None, None) => {
            let mut data = vec![T::zero(); site.dim];
            data[value] = scale;
            IdxTensor::from_dense(vec![site.clone()], data)
                .map_err(|error| PartitionedTTError::tensor_train_operation(error.to_string()))
        }
    }
}

fn rank_one_full_tt<T>(
    site_indices: &[DynIndex],
    projector: &Projector,
    scale: T,
) -> Result<TensorTrain>
where
    T: Scalar + TensorElement + StorageScalar + Default + Copy,
{
    let edge_indices: Vec<_> = (0..site_indices.len().saturating_sub(1))
        .map(|_| DynIndex::new_dyn(1))
        .collect();
    let mut tensors = Vec::with_capacity(site_indices.len());
    for (position, site) in site_indices.iter().enumerate() {
        let left = position.checked_sub(1).map(|edge| &edge_indices[edge]);
        let right = edge_indices.get(position);
        let local_scale = if position == 0 { scale } else { T::one() };
        if let Some(value) = projector.get(site) {
            tensors.push(projected_site_tensor(
                left,
                site,
                right,
                value,
                local_scale,
            )?);
        } else {
            let mut indices = Vec::with_capacity(3);
            if let Some(index) = left {
                indices.push(index.clone());
            }
            indices.push(site.clone());
            if let Some(index) = right {
                indices.push(index.clone());
            }
            tensors.push(
                IdxTensor::from_dense(indices, vec![local_scale; site.dim]).map_err(|error| {
                    PartitionedTTError::tensor_train_operation(error.to_string())
                })?,
            );
        }
    }
    TensorTrain::new(tensors)
        .map_err(|error| PartitionedTTError::tensor_train_operation(error.to_string()))
}

#[cfg(test)]
mod tests;
